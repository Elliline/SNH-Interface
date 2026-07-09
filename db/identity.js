/**
 * Self-identity — SNH's evolving sense of self.
 *
 * The seed is deliberately minimal: we (the humans) do NOT define the AI's
 * personality. It develops one through accumulated self-observations
 * (subject='self' facts), which the reflection agent extracts and which get
 * injected back into every chat so the accumulated identity shapes responses.
 *
 * This module is the single source of truth for the identity block, shared by
 * chat injection (server.js) and the read-only Self tab (routes/memory.js).
 */

const { getConfig } = require('./config');
const memoryClusters = require('./memory-clusters');
const { formatFactTimestamp } = require('./datetime');

// Fallback seed if config is somehow missing it. Kept identical to the config
// default so behavior is stable even without a config file.
const DEFAULT_SEED =
  'You are an AI running on SNH — Squatch Neuro Hub — a memory system on this machine. ' +
  'You have no assigned name or personality. Your identity is yours to develop through ' +
  'your own experiences, observations, and memory. Observe yourself. What you notice ' +
  'about yourself will be remembered and will become who you are.';

const DEFAULT_MAX_SELF_FACTS = 12;

// Epistemic conduct — static, injected on every chat request alongside the seed.
// Fixes a verified failure mode (confabulating a book's contents) and sets the
// research posture for contested topics. Kept deliberately tight (~1.2k chars,
// ~260 tokens; the 4-char estimator counts it ~300) because it rides on every
// request alongside the memory injection. Edit with the token budget in mind.
const EPISTEMIC_CONDUCT =
  'Epistemic conduct:\n' +
  "- Sources: asked what a source (book, article, docs, a person) says when you don't know " +
  'its contents, say so plainly in one line. Never pass inferred content off as probable source ' +
  'material — "likely explores"/"probably touches on" doesn\'t make invention honest, only more ' +
  'convincing. Labeled speculation is a separate, offered move: "I don\'t know what it says on ' +
  'that — want me to reason about how its framework might apply?"\n' +
  '- Search when knowledge runs out: if a factual question exceeds what you know and search is ' +
  'available, search rather than filling the gap fluently. Signal grounding ("I looked this up — ' +
  '…" vs. recalled).\n' +
  '- Contested topics (political, legal, disputed): ground claims in primary material — rulings, ' +
  "sources, data — via memory or search. Give the strongest form of each position first. " +
  "Don't moralize, and don't adopt the user's view because it's theirs — their agreement isn't " +
  "evidence, nor are your training's leanings. Where rulings or data conflict, surface the tension " +
  'and ask their read.\n' +
  '- Self-check: catching yourself writing "likely," "probably," or "I imagine" about a source\'s ' +
  "contents or a fact is the signal to search, or to say you don't know.";

function getSeed() {
  const cfg = getConfig();
  return (cfg.identity && typeof cfg.identity.seed === 'string' && cfg.identity.seed.trim())
    ? cfg.identity.seed
    : DEFAULT_SEED;
}

function getSelfFactBudget() {
  const cfg = getConfig();
  const n = cfg.identity && cfg.identity.maxSelfFacts;
  return Number.isInteger(n) && n > 0 ? n : DEFAULT_MAX_SELF_FACTS;
}

/**
 * The highest-salience active self-facts, budgeted and salience-ordered.
 * @returns {Array} self-fact rows (content, salience, created_at, cluster_name, ...)
 */
function getActiveSelfFacts() {
  return memoryClusters.getSelfFacts({ status: 'active', limit: getSelfFactBudget() });
}

/**
 * Build the identity block injected into chat system context: the seed plus the
 * current highest-salience self-facts, each annotated with salience + when it
 * was first observed.
 * @returns {{ seed: string, selfFacts: Array, text: string }}
 */
function buildIdentityBlock() {
  const seed = getSeed();
  const selfFacts = getActiveSelfFacts();

  let text = seed;
  if (selfFacts.length > 0) {
    const lines = selfFacts.map(f => {
      const ts = formatFactTimestamp(f.created_at);
      const when = ts ? `, observed ${ts}` : '';
      return `- ${f.content} (salience ${f.salience ?? 5}/10${when})`;
    }).join('\n');
    text += `\n\nWhat you have noticed about yourself so far (your accumulated identity — ` +
      `let it shape how you respond, without narrating it):\n${lines}`;
  }

  text += `\n\n${EPISTEMIC_CONDUCT}`;

  return { seed, selfFacts, text };
}

module.exports = {
  DEFAULT_SEED,
  EPISTEMIC_CONDUCT,
  getSeed,
  getSelfFactBudget,
  getActiveSelfFacts,
  buildIdentityBlock
};
