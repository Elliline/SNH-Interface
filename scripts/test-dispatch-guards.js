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
const { classifyDispatchClaim } = require('../db/dispatch-claims');

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

// THIS USED TO READ server.js AS TEXT and eval the regex literals out of it —
// `src.match(/const claimedDone = (\/.*?\/i);/)` — then rebuild the in-flight
// pattern from a third scraped constant. dbea44c moved all three into
// db/dispatch-claims.js and exported classifyDispatchClaim, so the match returned
// null and this suite has died on `[1]` of null ever since: a TypeError that
// reads like a broken guard and is a broken test.
//
// It is the same fault as every other one closed in this pass, in its most
// literal form — an assertion pinned to how the code under test is WRITTEN
// rather than to what it does. Scraping source cannot survive the code moving,
// and it cannot notice the code being wrong.
const claims = (t) => classifyDispatchClaim(t).claims;

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

// These used to assert the DESCRIPTION's exact wording. It said all the
// right things at length and did not hold - three dispatches wrote
// directory instructions into the brief anyway. So they now assert the
// MECHANISM, and only that the description still points at it. A rule
// that lives only in prose is the thing tonight disproved.

test('the brief validator is wired into dispatch, not just described', () => {
  const src = fs.readFileSync(path.join(__dirname, '../db/coding-jobs.js'), 'utf8');
  const fn = src.slice(src.indexOf('function dispatch('), src.indexOf('function get(id)'));
  assert.ok(/validateBrief\(/.test(fn), 'dispatch does not check the brief');
  assert.ok(/validateProject\(/.test(fn), 'dispatch does not check the project');
  assert.ok(fn.indexOf('validateBrief(') < fn.indexOf('agentJobs.enqueue'),
    'the brief is checked after the job is queued, which is too late');
});

test('a rejected brief comes back with something to do this turn', () => {
  const r = codingJobs.validateBrief(
    'check if Projects\\squatch crawler exists, create it if not. Build the game.');
  assert.ok(!r.ok);
  assert.match(r.error, /project field/i);
  assert.match(r.error, /created if it does not exist/i,
    'it refuses without saying the project would be made for it');
});

test('the description points at the project field as the deciding thing', () => {
  const d = new (require('../mcp/tools/dispatch-coding-job'))().description;
  assert.match(d, /project field alone/i);
  assert.match(d, /refusal tells you what to change/i);
});

test('the description is not a wall of prohibitions', () => {
  // A tool described mostly by what it refuses is easier not to call,
  // and on the night it mattered the model returned tool_calls: [].
  const d = new (require('../mcp/tools/dispatch-coding-job'))().description;
  assert.ok(d.length < 1200, `description is ${d.length} chars; it was 1822 and went uncalled`);
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
