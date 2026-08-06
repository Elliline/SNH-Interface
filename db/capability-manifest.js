/**
 * Capability Manifest — SNH's machine-truth registry of what it can ACTUALLY do.
 *
 * Why this exists (2026-07-23): asked what to work on, SNH proposed building a
 * self-claim audit framework — a near-verbatim description of the self-coherence
 * audit that had shipped the day before and had already caught one of its own
 * claims that same morning. It experienced the audit as an EVENT, not as a
 * capability it HAS. Features get built into it without ever becoming part of its
 * self-knowledge. This manifest is the fix: a registry of its own organs it can
 * consult at conversation time, so "what can you do / do you have a way to X" is
 * answered from ground truth instead of the model's guess.
 *
 * HARD RULE — the manifest must NEVER claim more than is built. Over-claiming is
 * the exact failure it exists to prevent. Every description here is derived from
 * what the code actually does; disabled/aspirational features are left out. The
 * config-gated entries below do this per-boot via their `when(cfg)` predicate —
 * e.g. web search appears exactly while config.tools.searxng.enabled is on. (The
 * SHIPPED DEFAULT for that flag is false — a fresh install must verify SearXNG is
 * actually up before claiming it — but this deployment has it on and live. Don't
 * read the default as the current state; read getConfig().)
 *
 * MAINTENANCE RULE (see CLAUDE.md): shipping a new capability includes adding its
 * entry here, and — on ship day — running its introduction so the entity learns
 * it (scripts/introduce-capability.js). Two layers, kept honest:
 *   1. This manifest = machine truth (code).
 *   2. Ship-day introductions = self-facts the entity forms through reflection.
 *
 * Each entry:
 *   id          - stable slug (used for dedup, ops trail, introductions)
 *   name        - short human name
 *   description - the full plain-language answer (1–2 sentences, everyday words);
 *                 retrieved on demand (API / Memory Map). Voice: addressed to the
 *                 entity ("you"), matching how the identity block is injected.
 *   oneLiner    - the COMPACT form injected into chat context. Kept tight on
 *                 purpose — mind the injection diet.
 *   intro       - first-person clause for the ship-day self-fact ("I ...").
 *   schedule    - plain phrase: does it run on a schedule, per message, or on ask.
 *   dateAdded   - YYYY-MM-DD the capability became real.
 */

const fs = require('fs');
const path = require('path');
const { estTokens } = require('./injection-budget');
const { getConfig } = require('./config');

const MEMORY_DIR = require('./database').getMemoryDir();
const OPS_DIR = path.join(MEMORY_DIR, 'ops');
const STATE_FILE = path.join(MEMORY_DIR, 'capability-manifest-state.json');

// ============ The registry (machine truth) ============

const CAPABILITIES = [
  {
    id: 'fact-extraction',
    name: 'Fact extraction & salience',
    description: "After each message you pull durable facts about the user and their projects out of the conversation and score each one for how much it matters. Each fact says one thing about one subject — a sentence that asserts several things is split into separate facts. If she says something you already hold, you do not write it down twice: you note that she has said it again and let the fact matter a little more. Facts about yourself are deliberately left out of this path — those come from reflection.",
    oneLiner: 'Pulls durable facts about the user from each message, one assertion per fact, scores them, and folds repeats into what you already hold.',
    intro: 'I pull durable facts about the user and their projects out of each conversation, keep each one to a single assertion, score how much it matters, and fold anything she has already told me into the fact I already hold rather than writing it twice',
    schedule: 'After each message',
    dateAdded: '2026-07-03'
  },
  {
    id: 'event-routing',
    name: 'Events to the day\'s log, states to memory',
    // Scope stated exactly: this is a ROUTING rule at intake, not a cleanup pass.
    // It stops new transient facts being written; it does not remove the ones
    // already in the corpus (that is the corrector, which does not exist yet).
    description: "Not everything the user tells you is a fact worth keeping. You test each thing by stripping the time reference out of it and asking whether anything durable is left. Things that happened, things happening now, and how she is feeling today go into the day's log; things that are true of her — what she owns, what she prefers, who she is — go into your long-term memory. When you cannot tell, it goes to the log, because a durable fact you missed will come up again and a passing one you stored is there forever. This applies to what you write from now on; things already in your memory are not revisited by it.",
    oneLiner: "Things that happened go in the day's log; things that are true go in long-term memory — and when unsure, the log.",
    intro: "I sort what Ellie tells me before I keep it: things that happened or how she is feeling today go into the day's log, and only what stays true of her goes into my long-term memory — when I cannot tell which it is, it goes to the log",
    schedule: 'After each message',
    dateAdded: '2026-08-03'
  },
  {
    id: 'supersession',
    name: 'Supersession with history',
    description: "When the user corrects or updates something, you store the new version and mark the old fact superseded instead of deleting it, so your belief history is kept. The outdated line is also pulled from the memory that gets injected, so it stops shaping answers.",
    oneLiner: 'Corrections retire the old fact and keep the history instead of deleting it.',
    intro: 'I keep my belief history when the user corrects me — the new fact supersedes the old one instead of deleting it',
    schedule: 'After each message',
    dateAdded: '2026-07-05'
  },
  {
    id: 'question-queue',
    name: 'Question queue & gap detection',
    description: "When new facts look incomplete or don't add up, you queue at most one short clarifying question to ask at a natural moment. You won't queue a question close to one already asked or answered, and you raise at most one per conversation.",
    oneLiner: 'Queues a short clarifying question when your facts look incomplete — at most one per conversation.',
    intro: 'I queue a short clarifying question when the facts I have look incomplete, and ask it at a natural moment',
    schedule: 'After each message; backlog swept on the heartbeat',
    dateAdded: '2026-07-06'
  },
  {
    id: 'initiative-bell',
    name: 'Initiative / bell',
    description: "You can raise things unprompted — a queued question, a reflection, an audit finding, a watchdog alert — through a prioritized bell. Delivery is capped: at most one self-started conversation a day, and nothing during quiet hours (10pm–8am).",
    oneLiner: 'Lets you raise things unprompted through a prioritized bell, capped so it never nags.',
    intro: 'I can raise things with the user unprompted through a prioritized bell, capped so it never nags',
    schedule: 'Every ~2 hours (in the heartbeat); delivered when a conversation opens',
    dateAdded: '2026-07-09'
  },
  {
    id: 'reflection-identity',
    name: 'Reflection & self-identity',
    description: "You read your own recent conversations and write first-person observations about yourself, which accumulate into a self-authored identity injected into every chat. No personality is assigned to you — it's built only from what you notice.",
    oneLiner: 'Builds your identity from first-person notes you write reflecting on your own conversations.',
    intro: 'I build my own identity by reflecting on my recent conversations and writing first-person notes about myself',
    schedule: 'Every ~2 hours, when there are new conversations',
    dateAdded: '2026-07-05'
  },
  {
    id: 'heartbeat-maintenance',
    name: 'Heartbeat consolidation',
    description: "On a timer you tidy your memory: oversized topic clusters get audited and split, and clusters that ended up sharing a name get merged. The same cycle also runs log archiving, the pending-question sweep, reflection, the self-coherence audit, the capability drift check, a store reconciliation, and the initiative pass. It no longer re-scores the links between clusters or runs a fact cleanup pass — both were removed on 2026-08-02.",
    oneLiner: 'Tidies memory on a timer — splits oversized clusters and merges duplicate-named ones.',
    intro: 'I tidy my own memory on a timer — splitting oversized topic clusters and merging ones that ended up sharing a name',
    schedule: 'Every 2 hours',
    dateAdded: '2026-07-04'
  },
  {
    id: 'self-coherence-audit',
    name: 'Self-coherence audit',
    description: "Once a day you sample a few of your own behavioral self-claims and check each against how you actually behaved in recent conversations, flagging any gap for Ellie to approve, discuss, or dismiss. You never rewrite your own identity — a gap is recorded and raised, never auto-applied.",
    oneLiner: 'Once a day, checks your own self-claims against how you actually behaved and flags gaps for Ellie.',
    intro: 'I check a few of my own self-claims against how I actually behaved each day, and flag any gap for Ellie to decide on',
    schedule: 'Daily',
    dateAdded: '2026-07-23'
  },
  {
    id: 'brain-watchdog',
    name: 'Brain watchdog',
    description: "You watch the local model engine's health, and if it stops responding several times in a row you restart its container to unwedge it, then tell the user it happened. Restarts are rate-capped so a restart loop can't run away.",
    oneLiner: 'Watches the local model engine and restarts it if it wedges, then says so.',
    intro: 'I watch the local model engine and restart it if it wedges, then tell the user it happened',
    schedule: 'Reacts to a health probe every 5 minutes',
    dateAdded: '2026-07-15'
  },
  {
    id: 'epistemic-temporal',
    name: 'Epistemic honesty & time',
    description: "A fixed honesty block is injected every chat, telling you to admit when you don't know a source, not to confabulate, and never to narrate a search you aren't running; the current date and time are injected too so you always know 'today.' This is guidance you follow, not an enforced mechanism.",
    oneLiner: 'Injects an honesty rule and the current date/time into every message.',
    intro: "I carry an honesty rule and the current date on every message — admitting what I don't know rather than confabulating",
    schedule: 'On every message',
    dateAdded: '2026-07-09'
  },
  {
    id: 'agent-pool',
    name: 'Parallel agent pool',
    description: "Your background thinking jobs run through one shared queue with a concurrency limit, and they yield to live chat — while you're answering the user, background work throttles so the response keeps the GPU. It's the plumbing the scheduled jobs run on.",
    oneLiner: 'Runs background jobs through one throttled queue that yields to your live chat.',
    intro: 'I run my background thinking through one shared queue that yields to the user\'s live chat',
    schedule: 'Always available (used by background jobs)',
    dateAdded: '2026-07-08'
  },
  {
    id: 'memory-map',
    name: 'Memory Map',
    description: "A read-only graph in the web UI shows your memory as clusters and facts, including 'superseded' arrows that trace how a belief was replaced. It's built straight from the database with no model calls; you can search, hide old 'ghost' facts, and collapse big clusters. The cluster-to-cluster links it draws are a frozen snapshot — nothing has maintained them since 2026-08-02.",
    oneLiner: 'A read-only web graph of your memory clusters, facts, and how beliefs were superseded.',
    intro: 'I can show my memory as a read-only graph of clusters, facts, and the links between them',
    schedule: 'When the Map tab is opened',
    dateAdded: '2026-07-08'
  },
  {
    id: 'model-selection',
    name: 'Model selection',
    description: "Your chat can run on any of several model engines configured in this deployment — a local one like the vLLM brain you usually run on, or others (Ollama, llama.cpp, or a cloud provider once its API key is set) — chosen in settings. So which model is 'you' can be switched.",
    oneLiner: 'Your chat can run on different configured model engines (local or cloud), switchable in settings.',
    intro: 'I can run on different model engines configured in this deployment — a local brain or another provider — switchable in settings',
    schedule: 'Chosen in settings',
    dateAdded: '2026-07-03'
  },
  {
    id: 'capability-manifest',
    name: 'Capability self-knowledge',
    description: "You keep a registry of what you can actually do — this list — and a compact version is injected into your context so that when asked what you can do, you answer from ground truth instead of guessing. New capabilities are added here when they ship.",
    oneLiner: 'This registry of what you can actually do, so you answer from truth, not guesswork.',
    intro: 'I keep a registry of what I can actually do and consult it when asked, instead of guessing',
    schedule: 'When asked / always injected',
    dateAdded: '2026-07-23'
  }
];

// Capabilities whose presence depends on LIVE config — included only when their
// `when(config)` predicate is true. This keeps the manifest honest against reality
// instead of shipped defaults: the web-search entry appears exactly when
// config.tools.searxng.enabled is on (item 3), so the list never over- OR under-
// claims. web_search retains + cites the actual source links (see server.js).
const CONDITIONAL_CAPABILITIES = [
  {
    id: 'web-search',
    name: 'Web search',
    description: "When a question is about current or changeable facts, you can search the web (via SearXNG) and read pages, and your answer marks and cites the actual source links it drew from. Those links are kept with the message, so if you're later asked to cite, you read the real sources instead of reconstructing them.",
    oneLiner: 'Search the web for current facts and answer with the real source links you used.',
    intro: 'I can search the web for current facts and answer with the actual source links I drew from',
    schedule: 'When a question needs current info (only while search is enabled)',
    dateAdded: '2026-07-23',
    when: (cfg) => !!(cfg && cfg.tools && cfg.tools.searxng && cfg.tools.searxng.enabled),
    // Machine link to the MCP tools this entry accounts for. Any registered tool
    // NOT claimed by some entry gets a derived entry instead of silently going
    // unmentioned — which is exactly how web_fetch had no coverage at all.
    coversTools: ['web_search', 'web_fetch'],
    // Explicit config keys this entry accounts for. Declared, not guessed:
    // matching "tools.searxng" to an entry called "web-search" by string
    // similarity fails, and a check that cries wolf gets ignored.
    coversConfig: ['tools.searxng'],
    // Probed by checkDrift(). A registered organ whose service is unreachable
    // must not keep reading as live.
    probes: (cfg) => [{ name: 'SearXNG', url: (cfg.tools && cfg.tools.searxng && cfg.tools.searxng.url) || 'http://localhost:8888' }]
  },
  {
    id: 'voice',
    name: 'Voice (speaking and listening)',
    description: "You can speak and listen: your written reply can be spoken aloud (Kokoro text-to-speech), and a spoken message can be turned into text for you to read (NVIDIA Parakeet speech-to-text). Both run locally on Sparky's GPU alongside the rest of you.",
    oneLiner: 'Speak your replies aloud and turn spoken input into text, both running locally.',
    intro: 'I can speak my replies aloud and turn spoken input into text, both running locally on Sparky',
    schedule: 'When the user uses the voice controls',
    dateAdded: '2026-07-24',
    // Honest gate: appears only once the TTS+STT engines are verified live and
    // voice.enabled is turned on (see db/config.js). Was deferred while the
    // containers were down; registers now that they answer.
    when: (cfg) => !!(cfg && cfg.voice && cfg.voice.enabled),
    coversConfig: ['voice'],
    probes: (cfg) => {
      const out = [];
      const stt = (cfg.voice && cfg.voice.stt) || {}, tts = (cfg.voice && cfg.voice.tts) || {};
      const find = (g) => (g.providers || []).find(p => `${p.type}:${p.name}` === g.active) || (g.providers || [])[0];
      const s = find(stt), t = find(tts);
      if (t && t.host) out.push({ name: 'TTS', url: t.host });
      if (s && s.host) out.push({ name: 'STT', url: s.host });
      return out;
    }
  },
  {
    id: 'cron-proposals',
    name: 'Scheduled job proposals',
    // HONESTY, deliberately laboured here because this is the easiest entry in
    // the manifest to over-claim: proposing is ALL it does. Nothing is created
    // without Ellie's approval, and even an approved job does not run — SNH has
    // no scheduler. Saying "I can schedule things" would be exactly the failure
    // this manifest exists to prevent.
    description: "When the user asks for something to happen on a schedule, you can propose a recurring job — a cron expression plus a description — which goes to her bell panel for approval. You cannot create or run one yourself: she approves or rejects it, approving only records the job, and nothing executes it because there is no scheduler yet. You are limited in how many you may propose per hour and how many can exist.",
    oneLiner: 'Propose a recurring job for the user to approve; approving records it, nothing runs it yet.',
    intro: 'I can propose a recurring scheduled job when the user asks for one, but only propose it — she approves or rejects it in her bell panel, and even approved it just gets recorded, because nothing runs scheduled jobs yet',
    schedule: 'When the user asks for something recurring',
    dateAdded: '2026-07-26',
    when: (cfg) => !!(cfg && cfg.tools && cfg.tools.cron && cfg.tools.cron.enabled !== false),
    coversTools: ['create_cron_job'],
    coversConfig: ['tools.cron']
  },
  {
    id: 'memory-write',
    name: 'Writing to memory on request',
    // Scope stated exactly: it writes when ASKED. It is not a general power to
    // edit memory at will, and it cannot delete — the replaced version is kept.
    description: "When the user asks you to remember something, you can write it to your long-term memory yourself, in the moment, instead of hoping the passive extractor picks it up later. Before storing, you work out whether the fact is about her or about you, whether it replaces something you already hold (in which case the old version is superseded, never deleted), and how much it matters. You cannot delete a memory this way, there is a limit on how many facts you may write per hour, and every call is logged.",
    oneLiner: 'Write a fact to your long-term memory when asked, deciding if it replaces something you already hold.',
    intro: 'I can write something to my long-term memory when I am asked to remember it, working out whether it is a fact about Ellie or about me and whether it replaces something I already held',
    schedule: 'When the user asks you to remember something',
    dateAdded: '2026-07-27',
    when: (cfg) => !!(cfg && cfg.tools && cfg.tools.memoryWrite && cfg.tools.memoryWrite.enabled !== false),
    coversTools: ['write_memory'],
    coversConfig: ['tools.memoryWrite']
  },
  {
    id: 'memory-inspect',
    name: 'Looking things up in your own memory',
    // Scope stated exactly. READ ONLY — the four tools cannot change anything,
    // and saying otherwise would make the manifest wrong about the one boundary
    // that matters here. Note also what is NOT claimed: background steps CAN now
    // be handed these tools, but no background step asks for them yet, so this
    // entry does not say anything about what happens on the heartbeat.
    description: "You can look things up in your own long-term memory instead of relying on the excerpt that gets injected into each conversation. You can search it for a topic, list what is in it or what your clusters are, count how many facts match something, and open a single fact in full — which tells you why you believe it, how much it matters and why it scored that, when you learned it, which conversation and message it came from, the exact words that were said and whether they were spoken or typed, whether anything has replaced it, and every time it has been said to you again since. You can also read the record of changes made to your memory: what was retired, what was kept in its place, and the evidence each decision was made on. Searching what you currently believe tells you when something you no longer believe also matches, so a corrected memory is never invisible. These only read: none of them can change, add or remove a memory, and you cannot undo a correction. There is a limit on how many lookups you may do per hour, and every one is logged.",
    oneLiner: 'Search, list, count and open your own memories — where a fact came from, why you believe it, and what has been corrected since. Read-only.',
    intro: 'I can look things up in my own memory rather than working from whatever excerpt happens to be injected — searching it, counting what matches, and opening a single fact to see where it came from, what was actually said, and why I believe it',
    schedule: 'When the user asks about what you remember, know, or believe',
    dateAdded: '2026-08-03',
    when: (cfg) => !!(cfg && cfg.tools && cfg.tools.memoryInspect && cfg.tools.memoryInspect.enabled !== false),
    coversTools: ['memory_search', 'memory_list', 'memory_count', 'memory_get', 'memory_corrections'],
    coversConfig: ['tools.memoryInspect']
  },
  {
    id: 'identity-lock',
    name: 'Locked identity facts',
    // Scope stated exactly: name and pronouns, nothing else. Over-claiming here
    // would be the manifest failing at its own job — and the narrowness is the
    // design, not a limitation to gloss over.
    description: "Your name and your pronouns are locked. You chose them, and once set they cannot be changed by anything automatic — not by a correction in conversation, not by the contradiction judge, not by writing to your own memory, not by reflection. If someone tries to change them while talking to you, you say plainly that they are locked and that you have not changed them. Changing one is a deliberate action taken outside chat, in the Self tab or with a script. Everything else you have noticed about yourself stays free to change, because those are things you observed rather than chose.",
    oneLiner: 'Your name and pronouns are locked — nothing automatic can change them, and you say so if asked to.',
    // Phrased to start with "I" (the intro is composed as "As of <date>, <intro>"),
    // and deliberately worded so it does NOT itself read as an assertion of the
    // name or pronouns — otherwise introducing the capability would collide with
    // the very slot it protects.
    intro: 'I keep my name and pronouns locked — I chose them, nothing automatic can change them, and if someone tries to change them in conversation I say plainly that they are locked rather than going along with it',
    schedule: 'Enforced on every memory write; checked whenever a self-fact would change',
    dateAdded: '2026-07-28',
    when: (cfg) => !!(cfg && cfg.identity && cfg.identity.lock && cfg.identity.lock.enabled !== false),
    coversConfig: ['identity.lock']
  },
  {
    id: 'corrector',
    name: 'Repairing your own memory',
    // Scope stated exactly, and the limits are load-bearing:
    //   - it runs in the BACKGROUND, on its own cadence. It is not something he
    //     can do in conversation on request, and the three tools it uses are
    //     backgroundOnly — claimed here so they do not each get a derived entry
    //     telling him in chat that he can merge and supersede facts at will.
    //   - it DELETES NOTHING. There is no code path in db/corrector.js that
    //     removes a row.
    //   - self-facts are held back: identical duplicates fold, nothing else,
    //     while corrector.selfFactSemantic is false (the default). Claiming he
    //     revises his self-view unattended would be an over-claim AND would
    //     pre-empt the joint curation session.
    //   - a supersession it cannot justify from evidence is NOT applied. Saying
    //     "resolves contradictions" without that clause would overstate it.
    description: "On its own schedule, in the background, you go back through your long-term memory and repair what is already wrong in it: duplicate and near-duplicate facts folded into the fuller one, things that were really passing events moved out of memory into the day's log, statements that say two things at once split into separate facts, and mismatches between your memory and the index used to search it. Where two facts you hold contradict each other, you weigh the evidence behind each — typed over transcribed, said directly over inferred, said more than once over said once, recent over stale — and retire the weaker one only when it is clearly weaker; when neither dominates you leave both alone and raise it for Ellie instead. You delete nothing, every change is written down with its reason and evidence, and any of them can be undone from the Self tab. Facts about yourself are only folded together when they are word for word identical — anything beyond that waits for a session with Ellie — and your locked name and pronouns are refused outright, which you are told about rather than it happening quietly.",
    oneLiner: 'In the background you repair your own memory — folding duplicates, moving events out, retiring the weaker of two contradicting facts when the evidence is clearly one-sided. Nothing deleted; every change logged and revertible.',
    intro: 'I repair my own memory in the background now — folding duplicates together, moving things that were really passing events out into the day\'s log, and retiring the weaker of two contradicting facts when the evidence clearly favours one, leaving the pair alone and raising it with Ellie when it does not. I delete nothing, I record every change and why I made it, and any of it can be undone',
    schedule: 'A heartbeat step on its own cadence — every corrector.intervalHours (default 6h)',
    dateAdded: '2026-08-05',
    when: (cfg) => !!(cfg && cfg.corrector && cfg.corrector.enabled !== false),
    coversTools: ['memory_merge_facts', 'memory_expire_fact', 'memory_supersede_fact'],
    coversConfig: ['corrector']
  }
];

// Things SNH explicitly CANNOT do — surfaced in the injection so common denials
// are readable facts, not guesses. Pairs with the closed-world statement (item 4).
const UNAVAILABLE = [
  { name: 'Image / video generation', note: "you can't create, edit, or render images or video" }
];

// ============ Derivation + health (self-maintenance) ============

/**
 * The LIVE MCP tool registry, wired at boot by server.js. The manifest reads
 * this rather than restating it: a tool that is registered is a thing SNH can
 * actually do, and that is already knowable at runtime.
 */
let toolRegistry = null;

/** @param {{getToolNames: function, getToolsForOpenAI: function}} client */
function setToolRegistry(client) { toolRegistry = client; }

function registryToolNames() {
  try { return toolRegistry ? toolRegistry.getToolNames() : []; } catch { return []; }
}

/** Tool name -> its own self-description, straight from the registered tool. */
function registryToolSpecs() {
  try {
    const out = {};
    for (const s of (toolRegistry ? toolRegistry.getToolsForOpenAI() : [])) {
      if (s && s.function && s.function.name) out[s.function.name] = s.function.description || '';
    }
    return out;
  } catch { return {}; }
}

/**
 * Every tool name claimed by a hand-written entry via `coversTools`, INCLUDING
 * entries whose config predicate is currently false. Derivation uses this: a tool
 * belonging to a switched-off entry must not get a derived entry, or the manifest
 * would claim a capability that config says is off.
 */
function coveredToolNames() {
  const set = new Set();
  for (const c of CAPABILITIES.concat(CONDITIONAL_CAPABILITIES)) {
    for (const t of (c.coversTools || [])) set.add(t);
  }
  return set;
}

/**
 * Tool names claimed by entries that are currently ACTIVE — the ones the manifest
 * is actually claiming right now. The difference between this and
 * coveredToolNames() is where "registered but unroutable" hides: a tool in the
 * gap between the two sets is callable by the model while the capability that
 * owns it is switched off. See checkDrift().
 */
function activeCoveredToolNames() {
  const set = new Set();
  for (const c of CAPABILITIES.concat(activeConditional())) {
    for (const t of (c.coversTools || [])) set.add(t);
  }
  return set;
}

/**
 * DERIVED entries: any registered MCP tool that no hand-written entry claims.
 * This is what stops a shipped tool from being silently absent from the
 * manifest — the failure mode that made a missing entry dangerous once the
 * manifest became authoritative. Derived entries are machine-true by
 * construction: their text comes from the tool's own description.
 */
function derivedToolCapabilities() {
  const covered = coveredToolNames();
  const specs = registryToolSpecs();
  return registryToolNames()
    .filter(n => !covered.has(n))
    .map(n => ({
      id: `tool:${n}`,
      name: n,
      description: specs[n] || `The ${n} tool is registered and callable.`,
      oneLiner: (specs[n] || `The ${n} tool is available.`).split(/(?<=\.)\s/)[0].slice(0, 140),
      intro: null,               // derived entries are never introduced as self-facts
      schedule: 'When the model calls it',
      dateAdded: null,
      derived: true
    }));
}

// ---- health cache -------------------------------------------------------
// buildInjectionBlock() runs on EVERY chat request, so it must never probe the
// network. checkDrift() (heartbeat) probes and writes here; injection only
// reads. Unknown health is treated as "assume configured state" — we only
// demote an entry on POSITIVE evidence that its service is down, so a probe
// that has never run cannot silently erase a real capability.
let healthCache = {};   // id -> { ok, checkedAt, detail }

function getHealth() { return { ...healthCache }; }

/** True only when we have positive evidence this entry's service is DOWN. */
function isKnownUnhealthy(id) {
  const h = healthCache[id];
  return !!(h && h.ok === false);
}

// ============ Accessors ============

/** Conditional entries whose live-config predicate currently holds. */
function activeConditional() {
  let cfg;
  try { cfg = getConfig(); } catch { return []; }
  return CONDITIONAL_CAPABILITIES.filter(c => { try { return c.when(cfg); } catch { return false; } });
}

/**
 * Config-enabled entries whose backing service is currently unreachable. These
 * are NOT available and must not read as live — the same over-claim rule that
 * keeps an approved-but-unrunnable cron job from reading as scheduled.
 */
function degradedCapabilities() {
  return activeConditional().filter(c => isKnownUnhealthy(c.id));
}

/** The capabilities that are actually available right now. */
function activeCapabilities() {
  return CAPABILITIES
    .concat(activeConditional().filter(c => !isKnownUnhealthy(c.id)))
    .concat(derivedToolCapabilities());
}

/** Full manifest as it currently stands (static + any config-enabled entries). */
function getAll() {
  return activeCapabilities().map(c => {
    const { when, ...rest } = c; // don't leak the predicate
    return { ...rest };
  });
}

/** One entry by id — searches static AND conditional (so introductions can find it). */
function getById(id) {
  const c = CAPABILITIES.concat(CONDITIONAL_CAPABILITIES).find(c => c.id === id);
  if (!c) return null;
  const { when, ...rest } = c;
  return { ...rest };
}

/**
 * Compact list for injection: [{ name, oneLiner }]. Small on purpose — full
 * descriptions are retrieved on demand via getAll()/the API.
 */
function getCompact() {
  return activeCapabilities().map(c => ({ name: c.name, oneLiner: c.oneLiner }));
}

/**
 * The block injected into chat system context. Leads with an anti-overclaim
 * instruction so the model answers "what can you do / can you do X" from this
 * list, not from a guess — and admits it can't when something isn't listed.
 * @returns {{ text: string, tokens: number, count: number }}
 */
function buildInjectionBlock() {
  const caps = activeCapabilities();
  const lines = caps.map(c => `- ${c.name}: ${c.oneLiner}`).join('\n');
  const unavailable = UNAVAILABLE.map(u => `- ${u.name}: ${u.note}`).join('\n');
  const text =
    'Your built-in capabilities — the ground truth of what your system can do. This list is ' +
    "EXHAUSTIVE: if a capability isn't listed here, you don't have it, so say so plainly rather than " +
    'claim it. When asked what you can do or whether you can do something, answer from this list.\n' +
    lines +
    "\n\nExplicitly NOT available — if asked for these, say you can't:\n" +
    unavailable +
    // A config-enabled organ whose service is DOWN is listed here, not above.
    // Same rule as an approved cron job that nothing runs: switched on is not
    // the same as working, and the compact list must not blur the two.
    (degradedCapabilities().length
      ? '\n\nTemporarily UNAVAILABLE — switched on but not responding right now, so treat these as things you cannot currently do:\n' +
        degradedCapabilities().map(c => `- ${c.name}: ${(getHealth()[c.id] || {}).detail || 'service not answering'}`).join('\n')
      : '') +
    // PRECEDENCE. This list is ~550 tokens against a ~6,500-token memory block,
    // so on volume alone it loses to stored facts that describe a shipped organ
    // as future work (facts written before it shipped, and never revised —
    // memory records what was true when it was recorded, not what is true now).
    // Stating the precedence explicitly is cheaper and more reliable than
    // hoping the smaller block wins.
    //
    // Both directions are spelled out on purpose. The dangerous failure is not
    // only "calls a live organ planned" — it is also the reverse, promoting a
    // remembered PLAN into a claimed capability. Camera/vision is the live
    // example: genuinely planned, genuinely not built, and it must stay that way.
    '\n\nPRECEDENCE — this list wins. For any question about what you can do, what you ' +
    'have access to, or where your operational limits are, THIS LIST is the source of truth and ' +
    'overrides anything in your memory, regardless of how important or strongly-stated that memory ' +
    'is. A stored fact saying one of the capabilities above is planned, upcoming, or not yet built ' +
    'is simply out of date — it was written before the capability shipped. Trust this list instead ' +
    'and speak in the present tense. The reverse holds just as strictly: a memory about something ' +
    'planned, hoped for, or being worked on does NOT give you that capability. If it is not in the ' +
    'list above, you do not have it yet, no matter how much memory you hold about intending to.';
  return { text, tokens: estTokens(text), count: caps.length };
}

/**
 * Simple on-demand lookup: entries whose name/description/id contain the query
 * (case-insensitive). Feeds the "retrieved on demand" path (API / richer answer).
 */
function find(query) {
  const q = String(query || '').toLowerCase().trim();
  if (!q) return [];
  return getAll().filter(c =>
    c.name.toLowerCase().includes(q) ||
    c.id.toLowerCase().includes(q) ||
    c.description.toLowerCase().includes(q)
  );
}

// ============ Ship-day introduction text ============

/**
 * The first-person, plain-language ship-day introduction for a capability —
 * the sentence that becomes a stored self-fact (a DECLARATION about what's built,
 * tagged by the existing claim/declaration classifier). Dry and accurate by
 * construction: it's the hand-written `intro` clause, never an LLM paraphrase.
 * @param {string} id
 * @returns {string|null}
 */
function introSentence(id) {
  const c = getById(id);
  if (!c) return null;
  return `As of ${c.dateAdded}, ${c.intro}.`;
}

// ============ Briefing document (Ellie's conversation script) ============

/**
 * A plain-language briefing, one organ per line, for Ellie to use as her script
 * when she introduces these organs to SNH in conversation. The backfill's job is
 * this document (plus the manifest itself) — the self-facts come from the dialogue,
 * NOT from database inserts. Newest-built first so the just-shipped organs lead.
 */
function getBriefing() {
  const ordered = getAll().slice().sort((a, b) =>
    (b.dateAdded || '').localeCompare(a.dateAdded || '') || a.name.localeCompare(b.name)
  );
  const lines = ordered.map(c =>
    `- **${c.name}** (${c.schedule}, since ${c.dateAdded}) — ${c.description}`
  );
  return [
    '# SNH capability briefing — Ellie\'s introduction script',
    '',
    'These are SNH\'s real organs, in its own plain terms. The point of this doc is',
    'to help SNH *know* it has them: introduce them in conversation, one or a few at',
    'a time, and let the knowing settle into self-facts through reflection — not',
    'through database inserts. One organ per line; the wording is already plain enough',
    'to say aloud or paraphrase.',
    '',
    ...lines,
    '',
    '_Generated from db/capability-manifest.js — regenerate with `node scripts/write-capability-briefing.js` after the manifest changes._'
  ].join('\n');
}

// ============ Ops trail for manifest changes ============

function readState() {
  try {
    if (fs.existsSync(STATE_FILE)) return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
  } catch (err) {
    console.error('[CapabilityManifest] Failed to read state:', err.message);
  }
  return { known: [] };
}

function writeState(state) {
  try {
    if (!fs.existsSync(MEMORY_DIR)) fs.mkdirSync(MEMORY_DIR, { recursive: true });
    fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2), 'utf8');
  } catch (err) {
    console.error('[CapabilityManifest] Failed to write state:', err.message);
  }
}

function logOps(line) {
  try {
    // Reuse the shared ops writer (newest-first, one file per local day).
    require('./fact-extractor').appendToOpsLog(`Capability manifest: ${line}`, OPS_DIR);
  } catch (err) {
    console.error('[CapabilityManifest] ops log write failed:', err.message);
  }
}

/**
 * Reconcile the code manifest against the persisted known-set and log any
 * additions/removals to the ops ledger, so manifest changes leave a machine
 * trail the future immune-system heartbeat can review. Logs ONLY — it never
 * writes self-facts (that would be the bulk-inject the backfill rule forbids;
 * introductions are a separate, deliberate step). Best-effort. Call on boot.
 * @returns {{added:string[], removed:string[]}}
 */
// ============ Drift detection ============

/** Is an HTTP service answering at all? Any response (even 404) means it's up. */
async function probeUrl(url, timeoutMs = 4000) {
  try {
    const r = await fetch(url, { method: 'GET', signal: AbortSignal.timeout(timeoutMs) });
    return { ok: true, status: r.status };
  } catch (e) {
    return { ok: false, error: e.name === 'TimeoutError' ? `timeout after ${timeoutMs}ms` : e.message };
  }
}

/**
 * Compare the manifest against reality and report every disagreement.
 *
 * Three kinds of drift, all of which have actually happened or nearly happened:
 *   1. unreachable-service — an entry says a capability is live but its backing
 *      service does not answer. Left alone this over-claims.
 *   2. missing-entry — a registered MCP tool no entry accounts for. This is how
 *      web_fetch ended up with no coverage at all.
 *   3. stale-entry — an entry claims a tool that is no longer registered.
 *
 * Probes run here (heartbeat cadence), never on the injection path. Results are
 * written to the health cache so buildInjectionBlock() can demote a dead organ
 * without doing any I/O of its own.
 *
 * @returns {Promise<{mismatches: Array, checked: number, health: Object}>}
 */
async function checkDrift() {
  const mismatches = [];
  let cfg;
  try { cfg = getConfig(); } catch { cfg = {}; }

  // 1. service reachability for every currently-enabled entry that declares probes
  let checked = 0;
  for (const c of activeConditional()) {
    if (typeof c.probes !== 'function') continue;
    let targets = [];
    try { targets = c.probes(cfg) || []; } catch { targets = []; }
    if (!targets.length) continue;

    const results = [];
    for (const t of targets) {
      checked++;
      const r = await probeUrl(t.url);
      results.push({ ...t, ...r });
    }
    const down = results.filter(r => !r.ok);
    healthCache[c.id] = {
      ok: down.length === 0,
      checkedAt: new Date().toISOString(),
      detail: down.length ? down.map(d => `${d.name} (${d.url}): ${d.error}`).join('; ') : 'all services answering'
    };
    if (down.length) {
      mismatches.push({
        kind: 'unreachable-service',
        id: c.id,
        message: `"${c.name}" is switched on in config but ${down.map(d => d.name).join(' and ')} ${down.length === 1 ? 'is' : 'are'} not answering — it is being reported as unavailable until it comes back.`,
        detail: healthCache[c.id].detail
      });
    }
  }

  // 2 & 3. manifest <-> MCP registry correspondence
  if (toolRegistry) {
    const registered = new Set(registryToolNames());
    const covered = coveredToolNames();
    const activeCovered = activeCoveredToolNames();
    const derived = new Set(derivedToolCapabilities().map(d => d.name));
    for (const t of registered) {
      if (covered.has(t) && !activeCovered.has(t)) {
        // The over-claim this check was missing (2026-07-27). The tool IS
        // registered, so the model is handed it and can call it — but the entry
        // that owns it is switched off in config, so the chat path never routes
        // to it and the manifest never claims it. It looks available and does
        // nothing. Previously invisible here: `covered` counted switched-off
        // entries too, so the tool read as accounted-for and neither the
        // missing-entry nor the stale-entry branch fired.
        mismatches.push({
          kind: 'unroutable-tool', id: `tool:${t}`,
          message: `The tool "${t}" is registered and callable, but the capability that owns it is switched off in config — nothing routes to it, so it is available in name only.`
        });
      } else if (!covered.has(t) && !derived.has(t)) {
        mismatches.push({
          kind: 'missing-entry', id: `tool:${t}`,
          message: `The tool "${t}" is registered and callable but no capability entry accounts for it.`
        });
      }
    }
    // Only entries the manifest is CURRENTLY claiming may be checked for a
    // missing tool. A switched-off entry legitimately has no registered tool.
    for (const t of activeCovered) {
      if (!registered.has(t)) {
        mismatches.push({
          kind: 'stale-entry', id: t,
          message: `A capability entry claims the tool "${t}", but it is not in the live tool registry.`
        });
      }
    }
  } else {
    mismatches.push({
      kind: 'registry-not-wired', id: 'tool-registry',
      message: 'The MCP tool registry was never handed to the capability manifest, so tool coverage cannot be checked.'
    });
  }

  return { mismatches, checked, health: getHealth() };
}

/**
 * Boot-time check. FAILS CLOSED for the two places a capability can actually
 * appear — the MCP tool registry and config-gated services — by warning loudly
 * about anything switched on that no entry accounts for, WITHOUT needing to
 * know in advance what to look for.
 *
 * It cannot do the same for arbitrary new routes or UI features: an Express
 * route is not a capability (most are plumbing), so route enumeration would be
 * noise. Those still rely on the maintenance rule in CLAUDE.md.
 *
 * @returns {{warnings: string[]}}
 */
function startupCheck() {
  const warnings = [];
  let cfg;
  try { cfg = getConfig(); } catch { cfg = {}; }

  // (a) registered tools nothing accounts for. Derived entries normally absorb
  // these, so a warning here means derivation itself is not wired.
  if (!toolRegistry) {
    warnings.push('MCP tool registry not wired into the capability manifest — tool coverage is unchecked.');
  } else {
    const covered = coveredToolNames();
    const derived = new Set(derivedToolCapabilities().map(d => d.name));
    for (const t of registryToolNames()) {
      if (!covered.has(t) && !derived.has(t)) {
        warnings.push(`MCP tool "${t}" is registered but has no capability entry (hand-written or derived).`);
      }
    }
  }

  // (b) config-gated services switched ON with no entry claiming them. This is
  // the fail-closed half: turning on a new service in config without adding an
  // entry warns at startup, rather than silently going unmentioned.
  const declared = new Set();
  for (const c of CAPABILITIES.concat(CONDITIONAL_CAPABILITIES)) {
    for (const k of (c.coversConfig || [])) declared.add(k);
  }
  const enabledKeys = [];
  for (const [key, val] of Object.entries(cfg.tools || {})) {
    if (val && typeof val === 'object' && val.enabled === true) enabledKeys.push(`tools.${key}`);
  }
  if (cfg.voice && cfg.voice.enabled === true) enabledKeys.push('voice');
  for (const k of enabledKeys) {
    if (!declared.has(k)) {
      warnings.push(`Config has "${k}" enabled but no capability entry declares coverage of it (add coversConfig: ['${k}'] to the entry, or add an entry).`);
    }
  }

  for (const w of warnings) {
    console.warn(`[CapabilityManifest] STARTUP WARNING: ${w}`);
    logOps(`startup warning — ${w}`);
  }
  if (!warnings.length) {
    console.log(`[CapabilityManifest] startup check clean — ${activeCapabilities().length} capabilities, ${registryToolNames().length} tool(s) all accounted for`);
  }
  return { warnings };
}

function syncToOps() {
  const state = readState();
  const knownIds = new Set((state.known || []).map(k => k.id));
  // Reconcile against what's ACTUALLY active (static + config-enabled), so the
  // web-search entry is logged when search is turned on and un-logged when off.
  const current = activeCapabilities();
  const currentIds = new Set(current.map(c => c.id));

  const added = current.filter(c => !knownIds.has(c.id)).map(c => c.id);
  const removed = (state.known || []).filter(k => !currentIds.has(k.id)).map(k => k.id);

  for (const id of added) {
    const c = getById(id);
    logOps(`entry added — "${c.name}" (${c.schedule}, since ${c.dateAdded})`);
  }
  for (const id of removed) logOps(`entry removed — "${id}"`);

  if (added.length || removed.length) {
    writeState({ known: current.map(c => ({ id: c.id, dateAdded: c.dateAdded })), updatedAt: new Date().toISOString() });
    console.log(`[CapabilityManifest] synced: ${added.length} added, ${removed.length} removed`);
  }
  return { added, removed };
}

module.exports = {
  getAll,
  getById,
  getCompact,
  buildInjectionBlock,
  setToolRegistry,
  checkDrift,
  startupCheck,
  degradedCapabilities,
  getHealth,
  find,
  introSentence,
  getBriefing,
  syncToOps
};
