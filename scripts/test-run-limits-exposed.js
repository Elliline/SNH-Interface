#!/usr/bin/env node
/**
 * EVERY LIMIT THAT CAN STOP A RUN IS REACHABLE FROM SETTINGS.
 *
 * This is the anti-regression for a failure that was not a bug in any one place.
 * The per-run limits were chosen from a menu on 2026-08-18, written into
 * DEFAULTS, and then discovered one at a time by watching jobs die: raise the
 * output budget and rounds become binding, raise rounds and wall clock does,
 * raise wall clock and the LLM call timeout does. Each was individually
 * reasonable and the sequence cost three coding jobs and two evenings.
 *
 * So the rule is structural rather than remembered: a key in one of the blocks
 * below MUST have a row on the settings page, or this fails and names it.
 * Adding a limit and not exposing it is the thing being prevented, and the
 * exemption list is deliberately awkward — an entry there is a sentence someone
 * had to write about why a number is unreachable.
 *
 * The page is read as TEXT because the page IS that file: public/script.js
 * declares its fields as `key: 'dotted.path'` and there is no DOM here to
 * render. Same spirit as test-tools-settings.js asserting the Tools tab is
 * generated rather than hand-listed.
 *
 * Usage: node scripts/test-run-limits-exposed.js
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const { getConfig } = require(path.join(ROOT, 'db/config'));
const PAGE = fs.readFileSync(path.join(ROOT, 'public/script.js'), 'utf8');
const CATALOGUE = fs.readFileSync(path.join(ROOT, 'mcp/mcp-client.js'), 'utf8')
  + fs.readFileSync(path.join(ROOT, 'mcp/tools/search-providers.js'), 'utf8');

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

/**
 * A path is exposed if the Brain tab declares it as a field key, or the tool
 * catalogue declares it as a tool setting. Both are real UI; neither is better.
 */
function exposed(dotted) {
  return PAGE.includes(`key: '${dotted}'`) || CATALOGUE.includes(`path: '${dotted}'`);
}

/**
 * Blocks whose every leaf is a limit on a run, and the leaves that are
 * deliberately not on a screen. A reason is required, in prose, here.
 */
const EXEMPT = {
  'heartbeat.toolBudget.maxCallsPerStep':
    'the default for HEARTBEAT steps only — agent jobs and scheduled runs both pass their own, so it cannot bind on either path. Belongs to the heartbeat block, which has no UI yet.',
  'heartbeat.toolBudget.maxWallClockMsPerStep':
    'same: overridden by both job paths, binds only on heartbeat steps.',
  'heartbeat.toolBudget.maxRoundsPerCall':
    'same: overridden by both job paths, binds only on heartbeat steps.'
};

const BLOCKS = [
  ['agentJobs', () => getConfig().agentJobs],
  ['scheduler', () => getConfig().scheduler],
  ['agentPool', () => getConfig().agentPool],
  ['brainCircuit', () => getConfig().brainCircuit],
  ['heartbeat.toolBudget', () => getConfig().heartbeat.toolBudget]
];

console.log('\nRun-limit exposure\n');

console.log('── Every per-run limit has a settings row ──');
const missing = [];
for (const [prefix, read] of BLOCKS) {
  const block = read() || {};
  for (const leaf of Object.keys(block)) {
    if (typeof block[leaf] === 'object' && block[leaf] !== null) continue;  // nested blocks listed separately
    const dotted = `${prefix}.${leaf}`;
    if (EXEMPT[dotted]) continue;
    if (!exposed(dotted)) missing.push(dotted);
  }
}
check('no per-run limit is unreachable from the UI',
  missing.length === 0,
  missing.length ? `not on any settings page: ${missing.join(', ')}` : '');

// The generation budgets for the two job paths, plus the call timeout that
// binds when a budget is raised — the one that turns a raise into a dead job.
console.log('\n── The generation budgets and the call timeout ──');
for (const k of [
  'generation.agentJobThinkingTokens', 'generation.agentJobResponseTokens',
  'generation.scheduledJobThinkingTokens', 'generation.scheduledJobResponseTokens',
  'generation.llmTimeoutTokensPerSecond', 'generation.llmTimeoutFloorMs'
]) {
  check(`${k} has a row`, exposed(k));
}

console.log('\n── The HTTP timeouts a run can hang on ──');
for (const k of ['tools.exa.timeoutMs', 'tools.searxng.timeoutMs', 'tools.webFetch.timeoutMs']) {
  check(`${k} has a row`, exposed(k));
}

console.log('\n── Nothing is hardcoded where a config read belongs ──');
const mm = fs.readFileSync(path.join(ROOT, 'db/memory-manager.js'), 'utf8');
const aj = fs.readFileSync(path.join(ROOT, 'db/agent-jobs.js'), 'utf8');
const wf = fs.readFileSync(path.join(ROOT, 'mcp/tools/web-fetch.js'), 'utf8');
check('the LLM call timeout is not the old baked-in formula',
  !/\/ 45 \* 1000 \* 2/.test(mm), 'wireMaxTokens / 45 * 1000 * 2 is back in callLLM');
check('the circuit breaker threshold is not a module constant',
  !/CIRCUIT_TIMEOUT_THRESHOLD\s*=\s*\d/.test(mm));
check('the retry count is not the literal `< 2`',
  !/\(j\.attempts \|\| 0\) < 2/.test(aj));
check('web_fetch does not hardcode its timeout',
  !/AbortSignal\.timeout\(10000\)/.test(wf));

// Every exemption is a sentence, not a shrug.
console.log('\n── Every exemption carries a reason ──');
for (const [k, why] of Object.entries(EXEMPT)) {
  check(`${k} says why it is not exposed`, typeof why === 'string' && why.length > 40);
}

console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
process.exit(fail ? 1 : 0);
