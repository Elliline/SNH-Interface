/**
 * MCP (Model Context Protocol) Client Manager
 * Maintains a registry of tool servers and provides methods to:
 * - Get all available tools in OpenAI function calling format
 * - Execute tool calls by routing to the correct tool implementation
 */

const WebSearchTool = require('./tools/web-search');
const WebFetchTool = require('./tools/web-fetch');
const CreateCronJobTool = require('./tools/create-cron-job');
const StartBackgroundJobTool = require('./tools/start-background-job');
const DispatchCodingJobTool = require('./tools/dispatch-coding-job');
const WriteMemoryTool = require('./tools/write-memory');
const {
  MemorySearchTool, MemoryListTool, MemoryCountTool, MemoryGetTool, MemoryCorrectionsTool
} = require('./tools/memory-inspect');
const {
  MergeFactsTool, ExpireFactTool, SupersedeFactTool
} = require('./tools/memory-correct');
const { MemoryJobsTool } = require('./tools/jobs-inspect');
// Read config THROUGH THE MODULE OBJECT rather than destructuring at load time —
// the rule db/agent-jobs.js and mcp/tools/web-search.js already follow. Two
// reasons, both live: the config seen here is always the one the process holds
// right now, and a test can substitute one without writing to the live
// data/config.json, which is deliberately NOT redirected by SNH_DATA_DIR.
function getConfig() { return require('../db/config').getConfig(); }
function getSearchConfig() { return require('../db/config').getSearchConfig(); }

/**
 * Tools a BACKGROUND step may be handed.
 *
 * The heartbeat had no tool access at all until 2026-08-03 — callLLM built its
 * request body as {messages, stream, max_tokens} with no tools key on either
 * provider branch, so every background role (cluster audit, reflection,
 * self-audit, initiative) was structurally incapable of calling anything. This
 * list is what a background step is ALLOWED to ask for; declaring the allowlist
 * is still per-step, and exactly one step declares one — the corrector.
 *
 * READ-ONLY BY DEFAULT, and the three exceptions are named rather than assumed.
 * A background agent that can write is a background agent that can change what
 * the entity believes about itself while nobody is in the room, so the only
 * writes on this list are the corrector's three narrow actions — merge, expire,
 * supersede — each of which goes through the fact-store funnel with the tier
 * table deciding autonomy. `write_memory` is deliberately NOT here: the general
 * power to write an arbitrary fact stays on the chat path, where a person is in
 * the room. Adding a fourth write tool is a decision, not a convenience.
 */
const BACKGROUND_TOOLS = [
  'memory_search', 'memory_list', 'memory_count', 'memory_get', 'memory_corrections',
  'memory_jobs',
  // 2026-08-18, for the agent-job queue: a handed-off job may look things up in
  // the world as well as in the record. Both are READS — they change nothing,
  // here or anywhere — and both remain per-step declarations, so nothing gains
  // them by being background: the corrector does not ask for them and therefore
  // does not have them. They ride on whether a search PROVIDER is available
  // (getSearchConfig().any — Exa with a key, or SearXNG enabled) like every other
  // appearance of the search stack, so a box with no provider simply has them
  // dropped by the registry intersection.
  'web_search', 'web_fetch',
  // Phase 2c: the corrector's write actions. Background-only — see
  // BACKGROUND_WRITE_TOOLS below and the backgroundOnly flag on each tool.
  'memory_merge_facts', 'memory_expire_fact', 'memory_supersede_fact'
];

/**
 * The subset of BACKGROUND_TOOLS that WRITE.
 *
 * Kept as its own list so "does this step change anything" is answerable without
 * reading three tool files. A step declaring any of these is a step that mutates
 * the corpus, and today exactly one does: the corrector.
 */
const BACKGROUND_WRITE_TOOLS = ['memory_merge_facts', 'memory_expire_fact', 'memory_supersede_fact'];

/**
 * THE CATALOGUE — every tool this system has, what gates it, and what its
 * settings are. One table, two readers: MCPClient.loadConfig() registers from it,
 * and describeCatalogue() renders the Tools tab from it.
 *
 * Adding a tool means adding a row here, which is not extra work — it is how a
 * tool gets registered at all. The point is that there is nowhere else to also
 * remember: the settings page has no list of its own, so it cannot fall behind
 * the way it did (three of fourteen tools shown, 2026-08-18).
 *
 * Per row:
 *   id/title    the function name, and a human name for the page
 *   Tool        the class. Constructed to read its name, description and tier —
 *               the page shows the model's own description, not a second copy.
 *   card        which section of the page it belongs to
 *   gate        ({cfg, registered}) => boolean. THE registration decision.
 *   gateWhy     the sentence shown when the gate is closed. Never "disabled" on
 *               its own — why it is off is the useful half.
 *   toggle      the config path the page's switch writes, or null when the row
 *               has no switch of its own (see toggleNote).
 *   fields      other settings, as dotted config paths the existing declarative
 *               binding already knows how to fold into a partial.
 */
const TOOL_CATALOGUE = [
  {
    id: 'web_search',
    title: 'Web search',
    Tool: WebSearchTool,
    card: 'search',
    // Composite by nature: the tool exists when ANY provider can actually be
    // called, so its switch is the per-provider switches below rather than one of
    // its own. Turning both providers off is how you turn search off.
    gate: () => getSearchConfig().any,
    gateWhy: () => {
      const chain = getSearchConfig();
      return chain.providers.length
        ? `no search provider is available (${chain.providers.map(p => `${p.name}: ${p.why}`).join('; ')})`
        : 'no search providers are configured';
    },
    toggle: null,
    toggleNote: 'Switched on by its providers — turn the providers below on or off.',
    note: ({ cfg }) => {
      const chain = getSearchConfig();
      return `providers in order: ${chain.providers.map(p => p.available ? p.name : `${p.name} (unavailable: ${p.why})`).join(' → ') || 'none'}`;
    }
  },
  {
    id: 'web_fetch',
    title: 'Read a web page',
    Tool: WebFetchTool,
    card: 'search',
    // Rides on SEARCH specifically, not on "any tool at all": fetching a page is
    // only useful when a search produced the URL, and action tools registering
    // must not drag it along.
    gate: ({ registered }) => registered.has('web_search'),
    gateWhy: () => 'web search is off, and a page fetch is only useful for a URL a search produced',
    toggle: null,
    toggleNote: 'Comes with web search.',
    fields: [
      { path: 'tools.webFetch.timeoutMs', label: 'Page fetch timeout (ms)', type: 'number', min: 1000, max: 60000,
        hint: 'How long one page may hang before it is given up on. A job reading whole pages spends most of its time here, and a page that never answers costs it a tool call it does not get back. Too low and slow-but-working sites read as broken.' }
    ]
  },
  {
    id: 'dispatch_coding_job',
    title: 'Send coding work to squatch-code',
    Tool: DispatchCodingJobTool,
    card: 'coding',
    // Two conditions, and the second is not a formality: the first real
    // dispatch failed with "spawn squatch-job ENOENT" because the service
    // PATH does not include the virtualenv. A tool that cannot possibly
    // work should not be offered to the model or listed as a capability.
    gate: ({ cfg }) => ((cfg.tools && cfg.tools.codingJobs) || {}).enabled === true
      && require('../db/coding-jobs').binaryStatus(
           (cfg.tools && cfg.tools.codingJobs) || {}).ok,
    gateWhy: ({ cfg }) => {
      const c = (cfg.tools && cfg.tools.codingJobs) || {};
      if (c.enabled !== true) return 'turned off here';
      return require('../db/coding-jobs').binaryStatus(c).why;
    },
    toggle: 'tools.codingJobs.enabled',
    toggleNote: 'An approved brief runs unattended and can edit files in the project. A git restore point is committed before it starts.',
    writes: true,
    fields: [
      { path: 'tools.codingJobs.projectsRoot', label: 'Projects directory', type: 'text',
        hint: 'Where projects live. squatch-code refuses any path outside it.' },
      { path: 'tools.codingJobs.timeoutMinutes', label: 'Timeout (minutes)', type: 'number', min: 1, max: 120,
        hint: 'How long one job may run before it is stopped. A stopped job keeps whatever it had already written; the restore point is how it is undone.' },
      { path: 'tools.codingJobs.maxPendingProposals', label: 'Briefs awaiting approval', type: 'number', min: 1, max: 20,
        hint: 'How many un-decided briefs may pile up before the tool refuses to write another.' }
    ]
  },
  {
    id: 'create_cron_job',
    title: 'Propose a scheduled job',
    Tool: CreateCronJobTool,
    card: 'cron',
    gate: ({ cfg }) => ((cfg.tools && cfg.tools.cron) || {}).enabled !== false,
    gateWhy: () => 'turned off here',
    toggle: 'tools.cron.enabled',
    fields: [
      { path: 'tools.cron.maxProposalsPerHour', label: 'Proposals per hour', type: 'number', min: 1, max: 50,
        hint: 'Proposals in any trailing hour, approved or not. Past this the tool refuses and says so.' },
      { path: 'tools.cron.maxKidCreatedJobs', label: 'Live jobs he may have created', type: 'number', min: 1, max: 100,
        hint: 'Hard ceiling on jobs of his own that exist at once.' }
    ]
  },
  {
    id: 'start_background_job',
    title: 'Hand work to a background agent',
    Tool: StartBackgroundJobTool,
    card: 'jobs',
    gate: ({ cfg }) => ((cfg.tools && cfg.tools.agentJobs) || {}).enabled !== false,
    gateWhy: () => 'turned off here',
    toggle: 'tools.agentJobs.enabled',
    fields: [
      { path: 'tools.agentJobs.dispatchBuildRequests', label: 'Dispatch "build me a thing" asks', type: 'toggle',
        hint: 'Tier 2 of the handoff triggers. Naming an agent yourself always dispatches, whatever this says.' },
      { path: 'agentJobs.maxStartsPerHour', label: 'Jobs he may start per hour', type: 'number', min: 1, max: 60 },
      { path: 'agentJobs.maxConcurrent', label: 'Jobs running at once', type: 'number', min: 1, max: 8 },
      { path: 'agentJobs.maxToolCallsPerJob', label: 'Tool calls per job', type: 'number', min: 1, max: 200,
        hint: 'Billed units: an error or an empty search costs a quarter of one.' },
      { path: 'agentJobs.maxRoundsPerJob', label: 'Tool rounds per job', type: 'number', min: 1, max: 40,
        hint: 'One round is one model turn, which may make several calls.' }
    ]
  },
  {
    id: 'write_memory',
    title: 'Write something to memory',
    Tool: WriteMemoryTool,
    card: 'memoryWrite',
    gate: ({ cfg }) => ((cfg.tools && cfg.tools.memoryWrite) || {}).enabled !== false,
    gateWhy: () => 'turned off here',
    toggle: 'tools.memoryWrite.enabled',
    writes: true,
    fields: [
      { path: 'tools.memoryWrite.maxWritesPerHour', label: 'Writes per hour', type: 'number', min: 1, max: 200 }
    ]
  },

  // The read set. One config flag, one rate cap, one backing module — registered
  // together on purpose, because a half-registered set would only ever be a bug.
  // Each still gets its own row on the page (they are separate tools to him), and
  // each row says plainly that the switch is shared.
  ...[
    ['memory_search', 'Search his own memory', MemorySearchTool],
    ['memory_list', 'List facts and clusters', MemoryListTool],
    ['memory_count', 'Count what matches', MemoryCountTool],
    ['memory_get', 'Open one fact in full', MemoryGetTool],
    ['memory_corrections', 'Read the corrections ledger', MemoryCorrectionsTool],
    ['memory_jobs', 'Look at his scheduled jobs', MemoryJobsTool]
  ].map(([id, title, Tool]) => ({
    id,
    title,
    Tool,
    card: 'memoryInspect',
    gate: ({ cfg }) => ((cfg.tools && cfg.tools.memoryInspect) || {}).enabled !== false,
    gateWhy: () => 'the memory-reading set is turned off here',
    toggle: 'tools.memoryInspect.enabled',
    toggleNote: 'These six share one switch and one rate cap — turning one off turns off the set.',
    fields: id === 'memory_search'
      ? [{ path: 'tools.memoryInspect.maxCallsPerHour', label: 'Lookups per hour (shared by all six)', type: 'number', min: 1, max: 500 }]
      : []
  })),

  // The corrector's three write actions. Registered UNCONDITIONALLY and marked
  // backgroundOnly, so they are structurally absent from every chat turn: a tool
  // that exists but is unreachable is easier to reason about than one that
  // vanishes from the registry and takes the manifest's drift-check with it.
  // Their real switch is corrector.enabled, which decides whether the step that
  // declares them ever runs — so that is what the page offers.
  ...[
    ['memory_merge_facts', 'Fold two duplicate facts together', MergeFactsTool],
    ['memory_expire_fact', 'Move a passing event out of memory', ExpireFactTool],
    ['memory_supersede_fact', 'Retire a fact a newer one replaces', SupersedeFactTool]
  ].map(([id, title, Tool]) => ({
    id,
    title,
    Tool,
    card: 'correctorWrites',
    gate: () => true,
    toggle: 'corrector.enabled',
    toggleNote: 'Always registered, and never offered in a conversation. This switch is the corrector itself — off means nothing ever calls these.',
    writes: true
  }))
];

let sharedInstance = null;

class MCPClient {
  constructor() {
    this.tools = new Map(); // tool name -> tool instance
  }

  /**
   * Register the enabled tools, FROM THE CATALOGUE.
   *
   * Every tool's on/off state and endpoint comes from db/config.js — the SINGLE
   * source of truth. There used to be a second one: mcp/tools.json carried its own
   * `enabled` flag that decided REGISTRATION while config decided ROUTING, and in
   * the disagreeing direction the tool registered, appeared in the model's tool
   * list, and was never routed to — available-looking and inert. One flag per
   * capability, in config.
   *
   * THIS USED TO BE A HAND-WRITTEN IF-CHAIN, and that is how the Tools tab came to
   * show three of fourteen tools (2026-08-18). The chain knew which config key
   * gated which tool; the settings page had that knowledge copied into it by hand,
   * so every tool shipped after the page was written was invisible in the UI. Two
   * copies of one fact, and only one of them maintained.
   *
   * So the gates are DATA now — TOOL_CATALOGUE — and both readers derive from it:
   * this method registers what its gates admit, and describeCatalogue() renders the
   * settings page from the same rows. A tool added tomorrow appears in the UI
   * because it had to be added here to exist at all.
   */
  loadConfig() {
    this.tools.clear();
    const cfg = getConfig();
    const registered = new Set();

    for (const entry of TOOL_CATALOGUE) {
      const admitted = entry.gate({ cfg, registered });
      if (!admitted) {
        console.log(`MCP: Skipping "${entry.id}" — ${entry.gateWhy ? entry.gateWhy({ cfg, registered }) : 'its gate is closed'}`);
        continue;
      }
      const tool = new entry.Tool();
      this.tools.set(tool.name, tool);
      registered.add(tool.name);
      console.log(`MCP: Registered "${tool.name}"${entry.note ? ` — ${entry.note({ cfg })}` : ''}`);
    }

    console.log(`MCP: ${this.tools.size} tool(s) ready: [${this.getToolNames().join(', ')}]`);
  }

  /**
   * THE SETTINGS PAGE'S DATA, derived from the same catalogue registration uses.
   *
   * Includes tools that are currently switched OFF, which is the whole point: a
   * page that listed only what is registered would lose the row for anything you
   * turned off, leaving no way to turn it back on. Each row says whether it is
   * registered right now and, when it is not, why.
   *
   * Instantiating a tool to read its name and description is safe — every
   * constructor here is pure — and it means the page shows the SAME description
   * the model is given, rather than a second one written for humans that can drift.
   */
  describeCatalogue() {
    const cfg = getConfig();
    const registered = new Set();
    // Replay the gates in order so `registered` is what it would be after a real
    // load — web_fetch's gate reads it.
    const rows = [];
    for (const entry of TOOL_CATALOGUE) {
      const admitted = !!entry.gate({ cfg, registered });
      if (admitted) registered.add(entry.id);
      const tool = new entry.Tool();
      const tier = typeof tool.getTierMetadata === 'function' ? tool.getTierMetadata() : null;
      rows.push({
        id: entry.id,
        title: entry.title,
        card: entry.card,
        description: tool.description || '',
        registered: admitted,
        why: admitted ? null : (entry.gateWhy ? entry.gateWhy({ cfg, registered }) : 'its gate is closed'),
        toggle: entry.toggle || null,
        toggleNote: entry.toggleNote || null,
        fields: entry.fields || [],
        backgroundOnly: !!tool.backgroundOnly,
        availableInChat: !tool.backgroundOnly,
        availableToBackground: BACKGROUND_TOOLS.includes(entry.id),
        writes: BACKGROUND_WRITE_TOOLS.includes(entry.id) || !!entry.writes,
        tier: tier ? tier.tier : (tool.backgroundOnly ? 'background' : 'read'),
        rateCaps: tier && tier.rateCaps ? tier.rateCaps : null
      });
    }
    return rows;
  }

  /**
   * OpenAI-format specs for a NAMED SUBSET of the registered tools.
   *
   * The background path needs this: a heartbeat step declares the tools it may
   * use, and handing it the whole registry instead would give a cluster-coherence
   * audit the ability to propose a cron job. Unknown or unregistered names are
   * dropped silently — the allowlist is a ceiling, not a request.
   */
  getToolsForOpenAISubset(names = []) {
    const wanted = new Set(names);
    const specs = [];
    for (const [name, tool] of this.tools) {
      if (wanted.has(name)) specs.push(tool.getOpenAIFunctionSpec());
    }
    return specs;
  }

  /** Which of `names` are both registered and allowed to background steps. */
  backgroundToolsAmong(names = []) {
    return names.filter(n => BACKGROUND_TOOLS.includes(n) && this.tools.has(n));
  }

  /**
   * Get all available tools formatted for OpenAI function calling
   * Returns array suitable for the "tools" parameter in chat completions
   */
  getToolsForOpenAI() {
    const specs = [];
    for (const tool of this.tools.values()) {
      // backgroundOnly tools are structurally unavailable to chat. This is the
      // one place that boundary is enforced, so it cannot be forgotten at a
      // call site: the corrector's write actions are simply not in the schema
      // any conversation is handed.
      if (tool.backgroundOnly) continue;
      specs.push(tool.getOpenAIFunctionSpec());
    }
    return specs;
  }

  /**
   * Execute a tool call by name
   * @param {string} toolName - The tool function name
   * @param {Object} args - The parsed arguments for the tool
   * @param {Object} context - Optional context, passed through UNCHANGED as
   *   every tool's second argument. web_search reads `caller` for its log and
   *   `searxngHost` (a string) as a SearXNG endpoint override; action tools read
   *   `conversationId`/`messageId`. Nothing is reshaped here.
   * @returns {Object} Tool execution result
   */
  async executeTool(toolName, args, context = {}) {
    const tool = this.tools.get(toolName);
    if (!tool) {
      return { error: `Unknown tool: ${toolName}` };
    }

    try {
      // EVERY TOOL HAS THE SAME SIGNATURE: (args, context). There is no special
      // case here any more, and the one that used to live here is the reason the
      // rule is stated this loudly.
      //
      // web_search was the odd one — `execute(args, endpointOverride)`, a
      // positional STRING. On 2026-08-18 the chat path passed { searxngHost } and
      // worked; the background tool loop passed { caller } and handed the CONTEXT
      // OBJECT to a parameter used as a URL base:
      //
      //   Search failed: Failed to parse URL from [object Object]/search?q=…
      //
      // Seven of those inside one job in 11 seconds, reported to Ellie as "an
      // issue with the search tool", which is all the model could see. The first
      // fix resolved the endpoint HERE, at this call site. That worked and was
      // still wrong in shape: a contract one tool alone breaks is a contract the
      // next call site forgets. web_search now takes (args, context) like the
      // rest and resolves its own providers, so this function no longer knows
      // which tool it is calling. Anything added here whose execute() is not
      // (args, context) is a bug in the tool, not a case for this switch.
      //
      // Action tools read the context (create_cron_job records which conversation
      // proposed the job); web_fetch takes only args and ignores it.
      return await tool.execute(args, context);
    } catch (error) {
      return { error: `Tool execution failed: ${error.message}` };
    }
  }

  /**
   * Check if any tools are registered
   */
  hasTools() {
    return this.tools.size > 0;
  }

  /** Is one specific tool registered? */
  hasTool(name) {
    return this.tools.has(name);
  }

  /**
   * Tier metadata for every registered tool that declares it. Nothing reads this
   * yet — the tool registry will. Tools without getTierMetadata() are omitted
   * rather than guessed at.
   */
  getTierMetadata() {
    const out = [];
    for (const tool of this.tools.values()) {
      if (typeof tool.getTierMetadata === 'function') out.push(tool.getTierMetadata());
    }
    return out;
  }

  /**
   * Get list of registered tool names
   */
  getToolNames() {
    return Array.from(this.tools.keys());
  }
}

/**
 * The one registry, shared by the chat path and the heartbeat.
 *
 * Two instances would mean two answers to "which tools exist", and the
 * capability drift-check reads that answer — a background registry that
 * registered a different set from the chat one would make the manifest true of
 * one of them and false of the other. `loadConfig()` on the shared instance is
 * what the settings route already calls, so a config change reaches both.
 */
MCPClient.shared = function shared() {
  if (!sharedInstance) {
    sharedInstance = new MCPClient();
    sharedInstance.loadConfig();
  }
  return sharedInstance;
};

MCPClient.TOOL_CATALOGUE = TOOL_CATALOGUE;
MCPClient.BACKGROUND_TOOLS = BACKGROUND_TOOLS;
MCPClient.BACKGROUND_WRITE_TOOLS = BACKGROUND_WRITE_TOOLS;

module.exports = MCPClient;
