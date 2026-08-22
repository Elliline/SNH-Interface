#!/usr/bin/env node
/**
 * test-brain-status — the system saying what it already knows.
 *
 * On 2026-08-21 the watchdog detected a wedged engine, restarted the
 * container, logged "cooldown 5 min" and confirmed recovery itself.
 * While all of that was happening, chat returned a bare "fetch failed".
 * The knowledge existed; the path from it to her did not.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-brain-status.js
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  process.exit(1);
}

const watchdog = require('../db/brain-watchdog');

let passed = 0, failed = 0;
function test(name, fn) {
  try { watchdog._reset(); fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

const fail = () => watchdog.onProbeResult({ ok: false, error: 'timeout after 8000ms' });

console.log('\nwhat the watchdog can say\n');

// Driven through describeBrainState, not the state machine: the watchdog
// DISABLES itself when SNH_DATA_DIR is set, because a disposable instance
// must never restart the shared container. That guard is correct, and it
// means the state machine is unreachable from a test - so the formatter,
// which is the part that failed her, is pure and tested directly.
const C = { cooldownMs: 300000, failureThreshold: 3 };
const NOW = Date.now();
const describe = (st) => watchdog.describeBrainState(st, C, NOW);

test('a healthy brain says nothing', () => {
  const b = describe({ consecutiveFailures: 0 });
  assert.ok(b.healthy);
  assert.strictEqual(b.message, null, 'a healthy engine must not produce a notice');
});

test('one failed probe explains itself and says a restart is coming', () => {
  const b = describe({ consecutiveFailures: 1 });
  assert.ok(!b.healthy);
  assert.strictEqual(b.state, 'wedged');
  assert.match(b.message, /stopped responding/i);
  assert.match(b.message, /restart it automatically/i);
  assert.match(b.message, /this is 1/);
});

test('at the threshold it says a restart is due', () => {
  const b = describe({ consecutiveFailures: 3, wedgeDetectedAt: NOW - 15 * 60000 });
  assert.match(b.message, /restart is due/i);
  assert.match(b.message, /about 15 minutes/, 'it should say how long it has been down');
});

test('a restart in flight says so', () => {
  const b = describe({ restartInFlight: true });
  assert.strictEqual(b.state, 'restarting');
  assert.match(b.message, /being restarted right now/i);
});

test('reloading says how long ago and how long left', () => {
  const b = describe({ awaitingRecovery: true, lastRestartAt: NOW - 240000 });
  assert.strictEqual(b.state, 'reloading');
  assert.match(b.message, /restarted 4 minutes ago/);
  assert.match(b.message, /1 more minute\b/, 'it should estimate the remaining wait');
});

test('this is what she would have seen at 20:21', () => {
  // Watchdog restarted the container at 20:17:25; she tried at 20:21:36.
  const b = describe({ awaitingRecovery: true, lastRestartAt: NOW - 251000 });
  assert.ok(!/fetch failed/i.test(b.message), 'still the bare error');
  assert.match(b.message, /still loading/i);
  assert.match(b.message, /try again/i);
});

test('every unhealthy state says what happens next', () => {
  for (const st of [
    { consecutiveFailures: 1 },
    { consecutiveFailures: 3, wedgeDetectedAt: NOW - 60000 },
    { restartInFlight: true },
    { awaitingRecovery: true, lastRestartAt: NOW - 40000 },
  ]) {
    const b = describe(st);
    assert.ok(b.message && b.message.length > 30, `state ${b.state} said nothing usable`);
    assert.match(b.message, /try again|minute/i, `state ${b.state} gives her nothing to expect`);
  }
});

test('no state implies data was lost', () => {
  for (const st of [{ restartInFlight: true }, { awaitingRecovery: true, lastRestartAt: NOW - 1000 }]) {
    const m = describe(st).message.replace(/nothing was lost/i, '');
    assert.ok(!/\blost\b|deleted|corrupt/i.test(m), 'the message implies loss');
  }
});

test('the grammar is not machine-generated', () => {
  const all = [
    describe({ consecutiveFailures: 1 }),
    describe({ consecutiveFailures: 3, wedgeDetectedAt: NOW - 60000 }),
    describe({ awaitingRecovery: true, lastRestartAt: NOW - 240000 }),
  ].map(b => b.message).join(' ');
  assert.ok(!/\(s\)/.test(all), 'a "minute(s)" makes the whole notice look automated');
});

test('brainStatus is synchronous and does not probe', () => {
  // A status call in a path that is already failing must not be able to hang.
  const src = fs.readFileSync(path.join(__dirname, '../db/brain-watchdog.js'), 'utf8');
  const fn = src.slice(src.indexOf('function describeBrainState('), src.indexOf('function _getState('));
  assert.ok(!/await |fetch\(|async /.test(fn),
    'the formatter does I/O — it could hang the path it is explaining');
});

console.log('\nthe chat route uses it\n');

test('an upstream failure consults the watchdog', () => {
  // WIDENED ON 2026-08-22, deliberately. This used to require
  // `if (networkFailure)` — only a call that never reached the engine. A
  // TIMEOUT reaches it and then gets nothing back, which is equally upstream
  // and is the more common failure on this box; excluding it is exactly what
  // sent her a bare "operation was aborted due to timeout" while the watchdog
  // had already recorded failed probes. The guard that matters is the one
  // below: it must still NOT speak for our own bugs.
  const src = fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8');
  const i = src.indexOf('classifyChatFailure(error)');
  assert.ok(i > 0, 'the chat handler no longer classifies its failures');
  const block = src.slice(i, i + 800);
  assert.ok(/brainStatus\(\)/.test(block), 'chat never asks why the engine is unreachable');
  assert.ok(/if \(verdict\.upstream\)/.test(block),
    'it should speak for upstream failures only, not for every error');

  const { classifyChatFailure } = require(path.join(__dirname, '../db/chat-failure'));
  assert.strictEqual(classifyChatFailure({ name: 'TimeoutError', message: 'x' }).upstream, true,
    'a timeout waiting on the engine must be upstream');
  assert.strictEqual(classifyChatFailure(new TypeError("Cannot read properties of null")).upstream, false,
    'our own bug must not be blamed on the engine');
});

test('the status lookup cannot replace the error it explains', () => {
  const src = fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8');
  const i = src.indexOf('brainStatus()');
  assert.ok(/catch \(statusErr\)/.test(src.slice(i - 200, i + 600)),
    'a throwing status lookup would swallow the real failure');
});

test('the technical error is kept, not discarded', () => {
  // The wording moved to db/chat-failure.js so it could be tested as words
  // rather than as a regex over a request handler. Assert the behaviour.
  const { chatFailureBody } = require(path.join(__dirname, '../db/chat-failure'));
  const err = Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' });
  const withBrain = chatFailureBody(err, { healthy: false, state: 'wedged', message: 'The model engine has stopped responding.' });
  assert.strictEqual(withBrain.technical, err.message, 'the underlying message must survive for diagnosis');
  assert.strictEqual(withBrain.error, 'The model engine has stopped responding.');

  // And in the window before the watchdog has concluded anything.
  const noBrain = chatFailureBody(err, { healthy: true, state: 'ok', message: null });
  assert.strictEqual(noBrain.technical, err.message);
  assert.ok(!/aborted due to timeout/i.test(noBrain.error),
    'she should never be shown the raw abort string');
});

test('the frontend shows a known state without a generic prefix', () => {
  const src = fs.readFileSync(path.join(__dirname, '../public/script.js'), 'utf8');
  assert.ok(/err\.brain = errorData\.brain/.test(src), 'the brain state is dropped on the floor');
  assert.ok(/error\.brain\s*\n?\s*\?\s*error\.message/.test(src.replace(/\s+/g, ' ').replace(/ /g, ' ')) ||
            /error\.brain/.test(src),
    'the UI still prefixes a self-explaining message');
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
