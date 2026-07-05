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

  return { seed, selfFacts, text };
}

module.exports = {
  DEFAULT_SEED,
  getSeed,
  getSelfFactBudget,
  getActiveSelfFacts,
  buildIdentityBlock
};
