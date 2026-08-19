#!/usr/bin/env node
/**
 * "She named the agent, so it goes" — and what a tool call costs.
 *
 * Two rules, one file, because they were the same failure on 2026-08-18: a job
 * that should have existed and did not, and a job that existed and spent its
 * whole allowance learning nothing.
 *
 * WHAT THE LIVE FAILURE ACTUALLY WAS, since the diagnosis matters more than the
 * fix. "Use and agent and write me a python script for a calculator" was read
 * correctly at every layer: tier 1 fired, the guidance block fired, and
 * start_background_job was in the payload. The model returned `tool_calls: []`
 * and the sentence "I have started a background job to write a Python calculator
 * script." Nothing was gating it. The suggestion was simply declined — which is
 * why the fix is a forced call plus a backstop, and why the FIRST thing asserted
 * here is that the classifier was never the problem.
 *
 * Runs against a throwaway SNH_DATA_DIR. No model is called: the classifier and
 * the budget are pure, and the two server-side halves are asserted structurally
 * (the chat route is a request handler, and a source assertion that says WHICH
 * line is missing is worth more than no assertion at all).
 *
 * Usage: node scripts/test-handoff-dispatch.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-handoff-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();

const routing = require(path.join(ROOT, 'db/tool-routing'));
const mm = require(path.join(ROOT, 'db/memory-manager'));
const { getConfig } = require(path.join(ROOT, 'db/config'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

console.log(`\nHandoff dispatch + tool budget (throwaway data dir: ${TMP})\n`);

// =========================================================================
console.log('── TIER 1 IS NOT GATED, AND NEVER WAS ──');
// The build-ask flag is off by default and tier 1 is decided before it is read.
// Asserted with allowBuild FALSE on purpose: if this ever starts depending on
// the flag, the message that failed live starts failing again.
const LIVE = 'Use and agent and write me a python script for a calculator';
const t1 = routing.classifyHandoffSignal(LIVE, { allowBuild: false });
check('the message that failed live is tier 1 with the build flag OFF',
  t1.dispatch === true && t1.tier === 1, JSON.stringify(t1));
check('her spelling of "and agent" still matches — it is the only real instance in the corpus',
  routing.classifyHandoffSignal('use and agent and write me a write up on Paradox Interactive', { allowBuild: false }).tier === 1);
check('and so does the ordinary spelling',
  routing.classifyHandoffSignal('can you use an agent to look into vllm backends', { allowBuild: false }).tier === 1);
check('tier 1 outranks an immediacy negative — "use an agent, quick" is still "use an agent"',
  routing.classifyHandoffSignal('use an agent for this, quick', { allowBuild: false }).tier === 1);
check('a build ask WITHOUT the mechanism named is still gated, as configured',
  routing.classifyHandoffSignal('write me a python script for a calculator', { allowBuild: false }).dispatch === false);
check('…and dispatches as tier 2 when that flag is turned on',
  routing.classifyHandoffSignal('write me a python script for a calculator', { allowBuild: true }).tier === 2);
check('ordinary conversation still does not dispatch',
  routing.classifyHandoffSignal('what do you remember about my projects', { allowBuild: false }).dispatch === false);

// =========================================================================
console.log('\n── THE FORCED CALL AND THE BACKSTOP ──');
// Structural, and narrow enough to mean something: each check names the one
// property whose absence brings the live failure back.
const serverSrc = fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8');

check('forcing is decided from tier 1 alone',
  /const forceHandoffCall = needsHandoff && handoffSignal\.tier === 1/.test(serverSrc),
  'forceHandoffCall is not derived from tier 1');
check('and only on tier 1 — tiers 2-4 are inferences and must not force a dispatch',
  !/forceHandoffCall = needsHandoff(?!.*tier === 1)/.test(serverSrc));

const forcedRounds = serverSrc.match(/tool_choice: \{ type: 'function', function: \{ name: 'start_background_job' \} \}/g) || [];
check('both provider tool loops can pin the call', forcedRounds.length === 2, `${forcedRounds.length} of 2 branches`);
const firstRoundOnly = serverSrc.match(/const forceThisRound = forceHandoffCall && round === 0;/g) || [];
check('pinned on the FIRST round only — later rounds must stay free so it can still answer her',
  firstRoundOnly.length === 2, `${firstRoundOnly.length} of 2 branches`);
check('a refused tool_choice retries the round unforced instead of losing the turn',
  /retrying this round unforced/.test(serverSrc));

check('the backstop enqueues when tier 1 created no row',
  /if \(created\.length === 0 && forceHandoffCall\)/.test(serverSrc));
check('it is tagged as its own source, so the panel and the ops log can tell it apart',
  /source: 'tier1-backstop'/.test(serverSrc));
check('the task is her message verbatim, not a paraphrase of it',
  /task: asked,/.test(serverSrc) && /const asked = String\(userMessage\.content \|\| ''\)\.trim\(\)/.test(serverSrc));
check('and she is TOLD it was started — a silent backstop is the phantom bug wearing a hat',
  /You asked for an agent, so one is queued/.test(serverSrc));
check('a backstop that cannot queue says so, with the queue\'s own reason',
  /\*\*No agent was started\.\*\*/.test(serverSrc));
check('the phantom-dispatch correction still exists for the non-tier-1 case',
  /\*\*Correction — no job was actually started\.\*\*/.test(serverSrc));

// =========================================================================
console.log('\n── WHAT A CALL COSTS ──');
// The rule, as a pure function, one case at a time.
const cost = (name, result) => mm.toolCallCost(name, result, 0.25).cost;

check('a usable web search bills in full', cost('web_search', { results: [{ url: 'x' }] }) === 1);
check('an EMPTY web search bills a quarter — it is not progress',
  cost('web_search', { results: [] }) === 0.25);
check('an errored call bills a quarter, whatever the tool',
  cost('web_search', { error: 'boom' }) === 0.25 && cost('memory_get', { error: 'boom' }) === 0.25);
check('an empty memory_search bills a quarter',
  cost('memory_search', { results: [] }) === 0.25);
check('…UNLESS it found inactive facts, which is a real answer',
  cost('memory_search', { results: [], also_inactive: 3, note: 'you no longer hold this' }) === 1);
check('memory_count returning ZERO bills in full — zero is the answer, not a failure',
  cost('memory_count', { count: 0 }) === 1);
check('web_fetch of a page bills in full', cost('web_fetch', { url: 'x', content: 'y' }) === 1);

console.log('\n── AND HOW THE BUDGET SPENDS ──');
const dead = { error: 'Search failed: Failed to parse URL from [object Object]/search?q=…' };
const good = { results: [{ url: 'x' }] };

let s = mm.createToolSession('test', ['web_search'], { maxCalls: 40 });
check('the raw ceiling is 2× the budget by default', s.maxAttempts === 80, String(s.maxAttempts));
check('a fresh session is not spent', s.spent() === null);

for (let i = 0; i < 12; i++) s.charge('web_search', dead);
check('twelve dead searches do NOT exhaust a 40-call budget (they did on 2026-08-18)',
  s.spent() === null, s.spent());
check('they billed 3, not 12', s.billed === 3, String(s.billed));
check('and the raw count is still the truth of what happened', s.calls === 0 && s.failedCalls === 12,
  `calls=${s.calls} failed=${s.failedCalls}`);

for (let i = 0; i < 40; i++) s.charge('web_search', good);
check('forty real results DO exhaust it', /call budget spent/.test(s.spent() || ''), s.spent());
check('and the reason says how many were dead, so a thin result is explainable',
  /empty or failed/.test(s.spent() || ''), s.spent());

// The floor under the discount: everything failing must not run forever.
const s2 = mm.createToolSession('ceiling', ['web_search'], { maxCalls: 40 });
for (let i = 0; i < 80; i++) { s2.calls++; s2.charge('web_search', dead); }
check('the raw attempt ceiling stops an everything-fails loop',
  /attempt ceiling reached/.test(s2.spent() || ''), s2.spent());
check('and it says plainly that nothing was coming back',
  /nothing is coming back/.test(s2.spent() || ''), s2.spent());

// The discount is a config knob, and 1 means "as it was before".
const s3 = mm.createToolSession('nodiscount', ['web_search'], { maxCalls: 4, failedCallCost: 1 });
for (let i = 0; i < 4; i++) s3.charge('web_search', dead);
check('failedCallCost 1 restores the old behaviour exactly', /call budget spent/.test(s3.spent() || ''));

// =========================================================================
console.log('\n── THE NUMBERS THAT SHIPPED ──');
const aj = getConfig().agentJobs;
check('a job gets 40 tool calls', aj.maxToolCallsPerJob === 40, String(aj.maxToolCallsPerJob));
check('and 16 rounds — rounds were the limit that actually bound at 6',
  aj.maxRoundsPerJob === 16, String(aj.maxRoundsPerJob));
check('15 minutes of wall clock', aj.maxWallClockMs === 900000, String(aj.maxWallClockMs));
check('and 2000 output tokens, because 700 cannot hold a script',
  aj.maxOutputTokens === 2000, String(aj.maxOutputTokens));
check('rounds × 2-3 calls now reaches the call budget rather than starving it',
  aj.maxRoundsPerJob * 2 >= aj.maxToolCallsPerJob * 0.75,
  `${aj.maxRoundsPerJob} rounds vs ${aj.maxToolCallsPerJob} calls`);

console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
process.exit(fail ? 1 : 0);
