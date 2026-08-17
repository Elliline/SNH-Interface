#!/usr/bin/env node
/**
 * The corrector on the agent pool: does it fan out where it may, stay serial
 * where it must, and yield to chat?
 *
 * Runs against a throwaway SNH_DATA_DIR with the five judge functions stubbed
 * deterministically, because the real ones are model calls with no temperature
 * or seed — the corpus, the writes and the ledger are all real.
 *
 * Run: node scripts/test-corrector-pool.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'corrector-pool-'));
process.env.SNH_DATA_DIR = TMP;

let failures = 0;
const check = (cond, label) => {
  if (cond) console.log(`  ✅ ${label}`);
  else { console.log(`  ❌ ${label}`); failures++; }
};

const sleep = ms => new Promise(r => setTimeout(r, ms));
const JUDGE_MS = 150;

const db = require('../db/database');
const agentPool = require('../db/agent-pool');
const memoryClusters = require('../db/memory-clusters');
const factExtractor = require('../db/fact-extractor');
const corrector = require('../db/corrector');

// Deterministic stand-ins. Only the model calls are replaced.
factExtractor.judgeStripTheTimestamp = async (content) => {
  await sleep(JUDGE_MS);
  return { isEvent: /\b(yesterday|this morning|last night|on tuesday)\b/i.test(content), reasoning: 'stub' };
};
factExtractor.splitCompoundFact = async (text) => {
  await sleep(JUDGE_MS);
  const parts = String(text).split(/\s+and\s+/i).map(s => s.trim()).filter(Boolean);
  return parts.length >= 2 ? parts.map(p => (p.endsWith('.') ? p : p + '.')) : null;
};
factExtractor.judgeSameAssertion = async () => { await sleep(JUDGE_MS); return { same: false, reasoning: 'stub' }; };
factExtractor.judgeSubsumption = async () => { await sleep(JUDGE_MS); return { relation: 'neither', reasoning: 'stub' }; };
factExtractor.judgeWhichSurvives = async () => { await sleep(JUDGE_MS); return 'a'; };
factExtractor.judgeContradiction = async () => { await sleep(JUDGE_MS); return { verdict: 'no', reasoning: 'stub' }; };

const EVENTS = [
  'User let the dogs out this morning',
  'User had a good lunch yesterday',
  'User went to the range on Tuesday',
  'User watched a film last night with friends',
  'User met the new neighbour last night',
  'User cleaned the garage this morning too',
];

// Near-identical rows, so the merge phase actually has pairs to judge — without
// them its "stays serial" check passes vacuously at peak 0.
const DUPES = [
  "User's MSP is MettaSphere LLC",
  "User's MSP is MettaSphere LLC.",
  'User MSP is MettaSphere LLC',
];

/** Watch the pool while `work` runs; return the highest concurrency observed. */
async function peakDuring(work) {
  let peak = 0;
  let watching = true;
  const watcher = (async () => {
    while (watching) {
      peak = Math.max(peak, agentPool.stats().active);
      await sleep(10);
    }
  })();
  const value = await work();
  watching = false;
  await watcher;
  return { peak, value };
}

(async () => {
  db.initDatabase();
  await db.initVectorStore();

  for (const c of [...EVENTS, ...DUPES]) {
    await memoryClusters.assignToCluster(
      c, 'vllm', 'stub', '', 'http://localhost:7070', 'pool-test', 4, 'user', null,
      { modality: 'typed', directness: 'direct', mentions: 1 }
    );
  }

  console.log('\n── Independent judgements fan out ──');
  let pass = { passId: 'test-1', dryRun: true, session: null, startedMs: Date.now(),
    maxCalls: 999, maxWallMs: 600000, writes: 0, selfCorrections: 0, stopped: null, plan: [], unresolved: [] };
  let r = await peakDuring(() => corrector.expireDatedEvents(pass, {}));
  check(r.peak > 1, `expiry judgements ran concurrently (peak ${r.peak})`);
  check(r.peak <= 3, `and stayed within the configured width (peak ${r.peak} <= 3)`);

  console.log('\n── …and yield to chat ──');
  pass = { ...pass, passId: 'test-2', plan: [], unresolved: [] };
  agentPool.beginChat();
  try {
    r = await peakDuring(() => corrector.expireDatedEvents(pass, {}));
    check(r.peak === 1, `throttled to concurrency 1 while chat is in flight (peak ${r.peak})`);
  } finally {
    agentPool.endChat();
  }

  console.log('\n── Order is preserved regardless ──');
  // Two dry-run passes at different widths must plan the same actions in the
  // same order — the fan-out is in the judging, never in the acting.
  const planOf = p => p.plan.map(x => `${x.action}:${x.targetText}`).join('\n');
  const passA = { ...pass, passId: 'a', plan: [], unresolved: [] };
  await corrector.expireDatedEvents(passA, {});
  agentPool.beginChat();
  const passB = { ...pass, passId: 'b', plan: [], unresolved: [] };
  await corrector.expireDatedEvents(passB, {});
  agentPool.endChat();
  check(planOf(passA) === planOf(passB), 'same actions, same order, at width 3 and width 1');
  check(passA.plan.length === EVENTS.length,
    `every dated event planned for expiry (got ${passA.plan.length} of ${EVENTS.length})`);

  console.log('\n── Pair judgements stay serial (merge phase) ──');
  // mergeNearDuplicates must NOT fan out: its pair set depends on what it has
  // already folded, and its verdicts are memoised.
  const passC = { ...pass, passId: 'c', plan: [], unresolved: [] };
  r = await peakDuring(() => corrector.mergeNearDuplicates(passC, {}));
  check(r.value && r.value.pairsChecked > 0,
    `the merge phase actually judged pairs (${r.value && r.value.pairsChecked}) — otherwise the next check is vacuous`);
  check(r.peak <= 1, `and judged them one at a time (peak ${r.peak})`);

  fs.rmSync(TMP, { recursive: true, force: true });
  console.log(`\n${failures === 0 ? '✅ ALL PASSED' : `❌ ${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => {
  console.error('test crashed:', e);
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {}
  process.exit(1);
});
