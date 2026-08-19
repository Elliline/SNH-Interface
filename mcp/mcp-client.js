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
const WriteMemoryTool = require('./tools/write-memory');
const {
  MemorySearchTool, MemoryListTool, MemoryCountTool, MemoryGetTool, MemoryCorrectionsTool
} = require('./tools/memory-inspect');
const {
  MergeFactsTool, ExpireFactTool, SupersedeFactTool
} = require('./tools/memory-correct');
const { MemoryJobsTool } = require('./tools/jobs-inspect');
const { getConfig, getSearchConfig } = require('../db/config');

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

let sharedInstance = null;

class MCPClient {
  constructor() {
    this.tools = new Map(); // tool name -> tool instance
  }

  /**
   * Register the enabled tools. Every tool's on/off state and endpoint comes from
   * db/config.js — the SINGLE source of truth.
   *
   * There used to be a second one. mcp/tools.json carried its own `enabled` flag
   * that decided REGISTRATION, while config.tools.searxng.enabled decided ROUTING
   * (server.js) and whether the capability manifest claimed web search. Two
   * independent flags for one capability meant they could disagree, and in the
   * disagreeing direction the tool registered, appeared in the model's tool list,
   * and was then never routed to — available-looking and inert. tools.json also
   * carried a duplicate `endpoint`, already dead because the search tool resolves
   * its own providers from config on every call. The file is gone; one flag per
   * capability, in config, which is the pattern create_cron_job already used.
   */
  loadConfig() {
    this.tools.clear();

    // web_search — registered when ANY provider can actually be called, and the
    // chain behind it is resolved per call (mcp/tools/web-search.js). Note what
    // "available" means: not just an `enabled` flag, but the prerequisite too —
    // Exa with no EXA_API_KEY in the environment is not available, and on a box
    // with SearXNG off as well there is no search tool at all, which is the same
    // honest absence this has always had.
    const search = getSearchConfig();
    if (search.any) {
      const tool = new WebSearchTool();
      this.tools.set(tool.name, tool);
      const shape = search.providers
        .map(p => p.available ? p.name : `${p.name} (unavailable: ${p.why})`)
        .join(' → ');
      console.log(`MCP: Registered tool "${tool.name}" -> providers in order: ${shape}`);
    } else {
      const why = search.providers.map(p => `${p.name}: ${p.why}`).join('; ') || 'no providers configured';
      console.log(`MCP: Skipping "web_search" — no provider is available (${why})`);
    }

    // web_fetch rides along with SEARCH tools specifically — fetching a page is
    // only useful when there is a search that produced the URL. Note this checks
    // for a search tool rather than "any tool at all": action tools register
    // below and must not drag web_fetch on with them.
    if (this.tools.has('web_search') && !this.tools.has('web_fetch')) {
      const webFetch = new WebFetchTool();
      this.tools.set(webFetch.name, webFetch);
      console.log('MCP: Registered built-in tool "web_fetch"');
    }

    // Action tools register independently of the search stack. create_cron_job
    // is gated only on its own config flag — it must be available when SearXNG
    // is off, which is the default.
    const cronCfg = (getConfig().tools && getConfig().tools.cron) || {};
    if (cronCfg.enabled !== false) {
      const cronTool = new CreateCronJobTool();
      this.tools.set(cronTool.name, cronTool);
      console.log(`MCP: Registered action tool "${cronTool.name}" (tier=${cronTool.tier}, propose-only)`);
    }

    // start_background_job — the async handoff. Registered on its own flag beside
    // create_cron_job, for the same reason: it is an action tool and must be
    // available when the search stack is off. NOT backgroundOnly and NOT in
    // BACKGROUND_TOOLS — this one runs the other way round, chat-only, so a
    // background job cannot start a background job.
    const agentJobsCfg = (getConfig().tools && getConfig().tools.agentJobs) || {};
    if (agentJobsCfg.enabled !== false) {
      const jobTool = new StartBackgroundJobTool();
      this.tools.set(jobTool.name, jobTool);
      console.log(`MCP: Registered action tool "${jobTool.name}" (tier=${jobTool.tier}, starts work and returns immediately)`);
    }

    const memWriteCfg = (getConfig().tools && getConfig().tools.memoryWrite) || {};
    if (memWriteCfg.enabled !== false) {
      const writeTool = new WriteMemoryTool();
      this.tools.set(writeTool.name, writeTool);
      console.log(`MCP: Registered action tool "${writeTool.name}" (tier=${writeTool.tier}, direct-execute)`);
    }

    // Read tools. Registered as a set — they share one config flag, one rate cap
    // and one backing module, so a half-registered set would only ever be a bug.
    const memInspectCfg = (getConfig().tools && getConfig().tools.memoryInspect) || {};
    if (memInspectCfg.enabled !== false) {
      // memory_jobs rides with them: same tier, same rate cap, same "reads the
      // record, changes nothing" contract. It reads a different table, which is
      // why it has its own backing module, but it is the same capability from
      // his side — looking at what is already written down.
      for (const Tool of [MemorySearchTool, MemoryListTool, MemoryCountTool, MemoryGetTool, MemoryCorrectionsTool, MemoryJobsTool]) {
        const t = new Tool();
        this.tools.set(t.name, t);
      }
      console.log('MCP: Registered read tools [memory_search, memory_list, memory_count, memory_get, memory_corrections, memory_jobs] (tier=read, no writes)');
    }

    // Corrector write actions. Registered unconditionally so the corrector can
    // always reach them, and marked backgroundOnly so they never appear in a
    // chat turn's tool schema. Their gate is corrector.enabled, checked by the
    // heartbeat step that declares them — not by registration, because a tool
    // that exists but is unreachable is easier to reason about than one that
    // vanishes from the registry and takes the manifest's drift-check with it.
    for (const Tool of [MergeFactsTool, ExpireFactTool, SupersedeFactTool]) {
      const t = new Tool();
      this.tools.set(t.name, t);
    }
    console.log('MCP: Registered corrector write actions [memory_merge_facts, memory_expire_fact, memory_supersede_fact] (background only, never offered in chat)');

    console.log(`MCP: ${this.tools.size} tool(s) ready: [${this.getToolNames().join(', ')}]`);
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

MCPClient.BACKGROUND_TOOLS = BACKGROUND_TOOLS;
MCPClient.BACKGROUND_WRITE_TOOLS = BACKGROUND_WRITE_TOOLS;

module.exports = MCPClient;
