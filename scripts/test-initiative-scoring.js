#!/usr/bin/env node
/**
 * The initiative prioritizer scores, and when it cannot it SAYS SO.
 *
 * This exists because of a two-month silent failure. `prioritize()` asked the
 * model for one integer with an answer budget of 8 tokens. On a reasoning model
 * callLLM sends max_tokens = maxTokens + backgroundThinkingTokens and
 * thinking_token_budget = backgroundThinkingTokens, and max_tokens is a HARD
 * TOTAL — so the answer's guaranteed floor was 8 no matter how the thinking
 * budget was set. A model whose reasoning channel had just been force-closed
 * wrote a short lead-in, the lead-in ate the 8, `match(/\d+/)` found nothing,
 * and the item silently kept the priority it already had. Most initiatives are
 * created at 6, so the visible symptom was a column of sixes and nothing else.
 *
 * Two rules are under test, and the second matters more than the first:
 *   1. the answer budget is big enough for an integer plus a lead-in
 *   2. an unparseable score is REPORTED — the fallback is correct behaviour and
 *      a silent fallback is what made this invisible
 *
 * Runs against a throwaway SNH_DATA_DIR and stubs the model on the
 * memory-manager module object, the way scripts/test-agent-jobs.js does.
 *
 * Usage: node scripts/test-initiative-scoring.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-initiative-scoring-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
const initiatives = require(path.join(ROOT, 'db/initiatives'));
const engine = require(path.join(ROOT, 'db/initiative-engine'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- the model stub -------------------------------------------------------
let mode = 'integer';
const seenOptions = [];
memoryManager.callLLM = async (systemPrompt, userPrompt, options) => {
  seenOptions.push(options || {});
  // What a working scorer returns.
  if (mode === 'integer') return { content: '9', provider: 'stub', truncated: false };
  // What a reasoning model returns when its answer budget is too small: a
  // lead-in, no digit, and finish_reason 'length'. This is the real failure.
  if (mode === 'truncated') {
    return { content: 'Based on the criteria above, I would', provider: 'stub', truncated: true };
  }
  // No digit, but the model finished of its own accord — a different fault with
  // the same consequence for her queue, and it must not be silent either.
  if (mode === 'nodigit') return { content: 'medium importance', provider: 'stub', truncated: false };
  return { content: '5', provider: 'stub', truncated: false };
};

// Capture the warnings the pass emits, so "it said so" is an assertion rather
// than something a person has to notice in a terminal.
const warnings = [];
const realWarn = console.warn;
console.warn = (...args) => { warnings.push(args.join(' ')); };

const opsFile = () => {
  const dir = path.join(TMP, 'memory', 'ops');
  if (!fs.existsSync(dir)) return '';
  return fs.readdirSync(dir).map(f => fs.readFileSync(path.join(dir, f), 'utf8')).join('\n');
};

async function seed(n, priority) {
  const ids = [];
  for (let i = 0; i < n; i++) {
    const it = await initiatives.addInitiative({
      type: 'observation',
      content: `Seeded item ${i} — ${Math.random().toString(36).slice(2)} something worth weighing.`,
      priority, dedupe: false
    });
    if (it) ids.push(it.id || it);
  }
  return ids;
}
const priorityOf = (id) => db.prepare('SELECT priority FROM initiatives WHERE id = ?').get(id)?.priority;

(async () => {
  console.log(`\nInitiative scoring tests (throwaway data dir: ${TMP})\n`);

  // =========================================================================
  console.log('── The answer budget leaves room for more than a bare digit ──');
  check('SCORE_ANSWER_TOKENS is 32, not the 8 that starved it',
    engine.SCORE_ANSWER_TOKENS === 32, String(engine.SCORE_ANSWER_TOKENS));

  mode = 'integer';
  await seed(3, 6);
  seenOptions.length = 0;
  const okRun = await engine.prioritize();
  check('every item was scored', okRun.unscored === 0, JSON.stringify(okRun));
  check('and the scoring call asked for the full answer budget',
    seenOptions.length > 0 && seenOptions.every(o => o.maxTokens === engine.SCORE_ANSWER_TOKENS),
    JSON.stringify(seenOptions.map(o => o.maxTokens)));
  check('the budget passed is the exported constant, so it cannot drift from the test',
    seenOptions[0].maxTokens === engine.SCORE_ANSWER_TOKENS);

  // =========================================================================
  console.log('\n── A truncated score is reported, never silently swallowed ──');
  // Fresh items at 6 — the birth priority that made the failure look like a
  // deliberate score of 6.
  db.prepare('DELETE FROM initiatives').run();
  const ids = await seed(4, 6);
  mode = 'truncated';
  warnings.length = 0;
  const truncRun = await engine.prioritize();

  check('the items keep the priority they already had',
    ids.every(id => priorityOf(id) === 6), JSON.stringify(ids.map(priorityOf)));
  check('the pass COUNTS what it could not score', truncRun.unscored === 4, JSON.stringify(truncRun));
  check('and does NOT report them as re-scored — a fallback writes what was already there',
    truncRun.rescored === 0, JSON.stringify(truncRun));
  check('and counts how many of those were truncated', truncRun.truncatedScores === 4, JSON.stringify(truncRun));
  check('`truncated` is read at the call site rather than destructured away',
    warnings.some(w => /TRUNCATED at 32 answer tokens/.test(w)),
    warnings.slice(0, 2).join(' | '));
  check('the per-item warning says what it kept instead',
    warnings.some(w => /keeping priority 6/.test(w)), warnings.slice(0, 2).join(' | '));
  // THE SUMMARY IS COMPARED TO THE ONE THE ENGINE BUILDS, not to a copy of its
  // wording kept here. It names the counts and the answer budget, and those are
  // what the assertions are for — but the sentence carrying them belongs to the
  // engine, so a reword moves both sides at once or neither.
  const expectedSummary = engine.unscoredSummaryLine(4, 4, 4);
  check('one summary line for the pass, not one per item',
    warnings.filter(w => w.includes(expectedSummary)).length === 1,
    warnings.filter(w => w.includes(expectedSummary)).length + ' of ' + warnings.length);
  check('the summary names the budget, which is the actionable half',
    expectedSummary.includes(`${engine.SCORE_ANSWER_TOKENS}-token answer budget`), expectedSummary);
  check('and it reaches the ops log, not just a console nobody is reading',
    opsFile().includes(expectedSummary), opsFile().slice(-240));

  // =========================================================================
  console.log('\n── An unparseable score that was NOT truncated is reported too ──');
  db.prepare('DELETE FROM initiatives').run();
  const ids2 = await seed(2, 6);
  mode = 'nodigit';
  warnings.length = 0;
  const nodigitRun = await engine.prioritize();
  check('it is counted as unscored', nodigitRun.unscored === 2, JSON.stringify(nodigitRun));
  check('but NOT counted as truncated — the two faults are different',
    nodigitRun.truncatedScores === 0, JSON.stringify(nodigitRun));
  check('the priority is still preserved', ids2.every(id => priorityOf(id) === 6));
  check('and it still says so out loud',
    warnings.some(w => w.includes(engine.unscoredSummaryLine(2, 2, 0))),
    warnings.join(' | '));
  check('without claiming a truncation that did not happen',
    !warnings.some(w => /TRUNCATED/.test(w)), warnings.join(' | '));

  // =========================================================================
  console.log('\n── A working pass says nothing, because nothing happened ──');
  db.prepare('DELETE FROM initiatives').run();
  await seed(2, 6);
  mode = 'integer';
  warnings.length = 0;
  const quiet = await engine.prioritize();
  check('no unscored items', quiet.unscored === 0);
  // A pass that scored everything says NOTHING — asserted as an absence of any
  // warning at all, which is both stronger than matching a phrase and immune to
  // the phrase changing.
  check('and no warning — telemetry reports CHANGE, not state',
    warnings.length === 0, warnings.join(' | '));

  console.warn = realWarn;
  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => {
  console.warn = realWarn;
  console.error('Test harness crashed:', err);
  process.exit(1);
});
