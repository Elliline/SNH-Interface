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
 * what the code actually does; disabled/aspirational features are left out (e.g.
 * web search is omitted while config.tools.searxng.enabled defaults to false).
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

const MEMORY_DIR = path.join(__dirname, '../data/memory');
const OPS_DIR = path.join(MEMORY_DIR, 'ops');
const STATE_FILE = path.join(MEMORY_DIR, 'capability-manifest-state.json');

// ============ The registry (machine truth) ============

const CAPABILITIES = [
  {
    id: 'fact-extraction',
    name: 'Fact extraction & salience',
    description: "After each message you pull durable facts about the user and their projects out of the conversation and score each one for how much it matters. Facts about yourself are deliberately left out of this path — those come from reflection.",
    oneLiner: 'Pulls durable facts about the user and their projects from each message and scores them.',
    intro: 'I pull durable facts about the user and their projects out of each conversation and score how much each one matters',
    schedule: 'After each message',
    dateAdded: '2026-07-03'
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
    description: "On a timer you tidy your memory: oversized topic clusters get audited and split, duplicates merged, and the links between clusters re-scored. The same cycle also runs cleanup, log summarizing, reflection, the audit, and the initiative pass.",
    oneLiner: 'Tidies memory on a timer — splits oversized clusters, merges duplicates, re-scores their links.',
    intro: 'I tidy my own memory on a timer — splitting oversized topic clusters, merging duplicates, and re-scoring their links',
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
    description: "A read-only graph in the web UI shows your memory as clusters and facts with the links between them, including 'superseded' arrows that trace how a belief was replaced. It's built straight from the database with no model calls; you can search, hide old 'ghost' facts, and collapse big clusters.",
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
    when: (cfg) => !!(cfg && cfg.tools && cfg.tools.searxng && cfg.tools.searxng.enabled)
  }
];

// Things SNH explicitly CANNOT do — surfaced in the injection so common denials
// are readable facts, not guesses. Pairs with the closed-world statement (item 4).
const UNAVAILABLE = [
  { name: 'Image / video generation', note: "you can't create, edit, or render images or video" }
];

// ============ Accessors ============

/** Conditional entries whose live-config predicate currently holds. */
function activeConditional() {
  let cfg;
  try { cfg = getConfig(); } catch { return []; }
  return CONDITIONAL_CAPABILITIES.filter(c => { try { return c.when(cfg); } catch { return false; } });
}

/** The capabilities that are actually available right now (static + active conditional). */
function activeCapabilities() {
  return CAPABILITIES.concat(activeConditional());
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
    unavailable;
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
  find,
  introSentence,
  getBriefing,
  syncToOps
};
