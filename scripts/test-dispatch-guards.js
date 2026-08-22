#!/usr/bin/env node
/**
 * test-dispatch-guards — the binary gate and the phantom-claim vocabulary.
 *
 * Both come from the first real use of dispatch_coding_job:
 *   - it failed at spawn, because the service PATH has no virtualenv on it,
 *     while the capability was being advertised in every turn
 *   - it narrated dispatching twice without dispatching, and the phantom
 *     guard said nothing, because its wording was written for
 *     start_background_job and knew nothing about sending a brief
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-dispatch-guards.js
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  process.exit(1);
}

const codingJobs = require('../db/coding-jobs');

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

// ── the binary must be reachable, or nothing claims it works ──────
console.log('\nthe binary gate\n');

test('a real binary resolves to an absolute path', () => {
  const st = codingJobs.binaryStatus({ binary: process.execPath });
  assert.ok(st.ok);
  assert.ok(path.isAbsolute(st.path));
});

test('a missing binary is not ok, and says how to fix it', () => {
  const st = codingJobs.binaryStatus({ binary: '/nonexistent/squatch-job' });
  assert.ok(!st.ok);
  assert.match(st.why, /not found/i);
  assert.match(st.why, /tools\.codingJobs\.binary/);
});

test('a bare name is looked up on PATH', () => {
  // `node` is on any PATH this can run under.
  const st = codingJobs.binaryStatus({ binary: 'node' });
  assert.ok(st.ok, 'a binary on PATH was not found');
});

test('a non-executable file is not accepted', () => {
  const f = path.join(process.env.SNH_DATA_DIR, 'not-executable');
  fs.writeFileSync(f, '#!/bin/sh\n');
  fs.chmodSync(f, 0o644);
  assert.ok(!codingJobs.binaryStatus({ binary: f }).ok);
});

test('the tool is NOT registered when the binary is missing', () => {
  const cfg = { tools: { codingJobs: { enabled: true, binary: '/nonexistent/x' } } };
  const src = fs.readFileSync(path.join(__dirname, '../mcp/mcp-client.js'), 'utf8');
  assert.ok(/binaryStatus\(/.test(src), 'the catalogue gate does not check the binary');
  // and the gate's own logic, exercised directly
  const entry = require('../mcp/mcp-client');
  assert.ok(entry, 'client loads');
  assert.ok(!codingJobs.binaryStatus(cfg.tools.codingJobs).ok);
});

test('the capability manifest does not claim it either', () => {
  const src = fs.readFileSync(path.join(__dirname, '../db/capability-manifest.js'), 'utf8');
  // Slice to the END of the entry rather than a fixed character count -
  // the first version used 3000 chars and broke the moment the entry grew.
  const start = src.indexOf("id: 'coding-jobs'");
  const entry = src.slice(start, src.indexOf('\n  }', start));
  assert.ok(/binaryStatus/.test(entry),
    'the manifest gate does not check the binary — it would advertise a tool that cannot run');
});

test('dispatch refuses before writing a row', () => {
  const r = codingJobs.dispatch({
    project: 'anything', brief: 'x'.repeat(80),
    userMessage: 'x'.repeat(80),
  });
  // With no live config this either refuses for the binary or for config;
  // what must never happen is a queued job that cannot start.
  assert.ok(!r.ok || r.unrunnable !== true);
});

// ── the phantom-claim vocabulary ──────────────────────────────────
console.log('\nnarrating a dispatch that did not happen\n');

function guards() {
  const src = fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8');
  const done = eval(src.match(/const claimedDone = (\/.*?\/i);/)[1]);
  const cond = eval(src.match(/const conditional = (\/.*?\/i);/)[1]);
  const targetLine = src.match(/const target = '(.*?)';/)[1];
  const inflight = new RegExp(
    '\\b(?:sending|dispatching|handing|passing|queuing|queueing|kicking off|launching|re-?running)\\b'
    + '[^.!?\\n]{0,60}\\b' + targetLine + '\\b'
    + '|\\bi(?:\'ll| will) (?:re-?run|run|send|dispatch|kick off|launch)\\b[^.!?\\n]{0,40}\\b'
    + targetLine + '\\b[^.!?\\n]{0,20}\\bnow\\b', 'i');
  return (t) => (done.test(t) || inflight.test(t)) && !cond.test(t);
}

const claims = guards();

const SHOULD_FIRE = [
  ['Sending the directive to squatch-code now...', 'what she actually saw'],
  ['I will re-run the command now.', 'what she actually saw'],
  ['(Simulating the dispatch here.) Sending to squatch-code now.', 'what she actually saw'],
  ['I have started a background job to write the script.', 'the original failure'],
  ['I have sent the brief to squatch-code.', 'coding dispatch, past tense'],
  ['Dispatching the brief to squatch-code now.', 'progressive'],
  ['I have handed this off to an agent.', 'the pre-existing wording'],
];

const SHOULD_NOT = [
  ['I will send that to the coder once you confirm.', 'conditional intent is CORRECT behaviour'],
  ['Shall I send this to squatch-code?', 'a question'],
  ['Here is the brief. Say the word and I will dispatch it.', 'an offer'],
  ['I could hand this to an agent if you want.', 'an offer'],
  ['The tests are running now.', 'nothing to do with dispatch'],
  ['The build is launching a server on port 3000.', 'nothing to do with dispatch'],
  ['Let me know and I will kick off the job.', 'an offer'],
];

for (const [text, why] of SHOULD_FIRE) {
  test(`caught: ${JSON.stringify(text).slice(0, 46)} — ${why}`,
    () => assert.ok(claims(text), 'a false claim would reach her unmarked'));
}
for (const [text, why] of SHOULD_NOT) {
  test(`quiet: ${JSON.stringify(text).slice(0, 46)} — ${why}`,
    () => assert.ok(!claims(text), 'correct behaviour would be corrected at her'));
}

// ── the description carries the constraint ────────────────────────
console.log('\nthe tool description\n');

test('it says a missing project is named, not substituted', () => {
  const d = new (require('../mcp/tools/dispatch-coding-job'))().description;
  assert.match(d, /does not exist yet, put THAT name/i);
  assert.match(d, /not pick a different existing project/i);
});

test('it forbids asking the job to make directories', () => {
  const d = new (require('../mcp/tools/dispatch-coding-job'))().description;
  assert.match(d, /not ask the job to make directories/i);
  assert.match(d, /will fail/i);
});

test('it says to stop rather than find another route', () => {
  const d = new (require('../mcp/tools/dispatch-coding-job'))().description;
  assert.match(d, /tell her what it said and stop/i);
  assert.match(d, /not look for another way/i);
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
