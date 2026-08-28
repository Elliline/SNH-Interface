/**
 * Logic test for the brain watchdog state machine.
 * Mocks docker (child_process.execFile), config, initiatives, ops log, and the
 * clock so every guardrail is exercised deterministically — no real container,
 * no real time. Run: node scripts/test-brain-watchdog.js
 */

// ---- This suite must NOT be run with SNH_DATA_DIR ---------------------------
//
// Most suites in this repo need `SNH_DATA_DIR=$(mktemp -d)` to avoid the live
// store. This one is the exception, and it is the only one: the watchdog reads
// that variable at module load and disables itself when it is set, because a
// disposable instance may not restart the shared container. Scenarios A–D need
// it ENABLED, so the parent process must run without the variable; Scenario E
// sets it in a child on purpose to prove the guard holds.
//
// Run with it set and the watchdog never fires, `dockerCalls` stays empty, and
// the suite used to die on `dockerCalls[0].args` — a TypeError that reads like a
// broken watchdog rather than a broken invocation. It is safe without it: this
// file requires db/database only for a path join, and nothing here opens the DB.
if (process.env.SNH_DATA_DIR) {
  console.error(
    'test-brain-watchdog must run WITHOUT SNH_DATA_DIR — it is the one suite that does.\n' +
    `Got SNH_DATA_DIR=${process.env.SNH_DATA_DIR}, which disables the watchdog under test.\n` +
    'Run: node scripts/test-brain-watchdog.js'
  );
  process.exit(2);
}

// ---- Install mocks BEFORE requiring the module under test -------------------
// The module destructures execFile + getConfig at load, so patch first.
const path = require('path');
const cp = require('child_process');
const dockerCalls = [];
let dockerShouldFail = false;
cp.execFile = (cmd, args, opts, cb) => {
  dockerCalls.push({ cmd, args });
  // Simulate the async restart completing.
  setImmediate(() => dockerShouldFail
    ? cb(new Error('mock docker failure'), '', 'permission denied')
    : cb(null, args[1] + '\n', ''));
};

const config = require('../db/config');
let watchdogCfg = { enabled: true, container: 'test-brain', failureThreshold: 3, cooldownMinutes: 5, maxRestartsPerHour: 2 };
config.getConfig = () => ({ watchdog: watchdogCfg });

const initiatives = require('../db/initiatives');
const queuedInitiatives = [];
initiatives.addInitiative = async (init) => { queuedInitiatives.push(init); return 'mock-id'; };

const factExtractor = require('../db/fact-extractor');
const opsLines = [];
factExtractor.appendToOpsLog = (msg) => { opsLines.push(msg); };

// Controllable clock.
let NOW = 1_000_000_000_000;
const realNow = Date.now;
Date.now = () => NOW;
const MIN = 60 * 1000;

const wd = require('../db/brain-watchdog');

// ---- Tiny assert -----------------------------------------------------------
let failures = 0;
function check(cond, label) {
  if (cond) { console.log(`  ✅ ${label}`); }
  else { console.log(`  ❌ ${label}`); failures++; }
}
async function failProbe(err = 'timeout after 8000ms') { await wd.onProbeResult({ ok: false, ms: 8000, error: err }); }
async function okProbe(ms = 120) { await wd.onProbeResult({ ok: true, ms }); }
function advance(min) { NOW += min * MIN; }
function opsMatch(re) { return opsLines.some(l => re.test(l)); }
// ---- Ops lines are matched on their VALUES, never on their sentences --------
//
// The ops log is the watchdog's only observable for "it acted, and it said so",
// so these checks cannot be moved off the rendered text entirely. What they CAN
// stop doing is reading the prose. Every assertion below names the facts a line
// has to carry — the container this suite configured, the counters from its own
// config, the error text its own mock produced — and says nothing about the
// words joining them.
//
// This is not a style point. 2a47c71 reworded the fire line from `restarting
// <container>` to `restarting via \`<label>\`` so a box whose engine is a systemd
// unit reports the command it actually ran; every assertion pinned to the old
// sentence went red for days with nothing broken. And `docker restart
// <container>` is itself config-shaped — `watchdog.restartCommand` replaces it
// outright — so matching it literally pins the test to one deployment's config.
function opsCarrying(...values) {
  return opsLines.filter(l => values.every(v => l.includes(String(v))));
}
function opsAboutTarget() { return opsCarrying(watchdogCfg.container); }

(async () => {
  // ===== Scenario A: threshold → restart → cooldown → recovery → initiative ==
  console.log('\n── Scenario A: fire, cooldown, recover, alert ──');
  wd._reset(); dockerCalls.length = 0; queuedInitiatives.length = 0; opsLines.length = 0; dockerShouldFail = false;

  await failProbe();                         // failure 1
  check(wd._getState().consecutiveFailures === 1, 'failure 1 counted');
  check(dockerCalls.length === 0, 'no restart after 1 failure');
  advance(5); await failProbe();             // failure 2
  check(wd._getState().consecutiveFailures === 2, 'failure 2 counted');
  check(dockerCalls.length === 0, 'no restart after 2 failures');
  advance(5); await failProbe();             // failure 3 → restart
  check(dockerCalls.length === 1, 'restart fired at threshold (3)');
  check(dockerCalls[0].args[0] === 'restart' && dockerCalls[0].args[1] === 'test-brain', 'docker restart test-brain issued');
  check(wd._getState().consecutiveFailures === 0, 'counter reset after restart');
  check(wd._getState().awaitingRecovery === true, 'awaiting recovery');
  // The fire line has to carry three facts: which engine, what tripped it, and
  // which attempt this is against the cap. All three come from this suite's own
  // config, so a reword moves nothing and a wrong target fails immediately.
  check(opsCarrying(watchdogCfg.container, watchdogCfg.failureThreshold,
    `1/${watchdogCfg.maxRestartsPerHour}`).length === 1,
    'the fire names its target, what tripped it, and which attempt it is',
    opsAboutTarget().join(' || '));
  // The OUTCOME is logged, and logged truthfully. `completed` and `FAILED` are
  // wording; the error text is the fact — this mock succeeds, so no line about
  // this target may carry one. Scenario C asserts the mirror image.
  check(opsAboutTarget().length === 2, 'the restart and its outcome are both logged',
    `${opsAboutTarget().length} line(s)`);
  check(opsCarrying('permission denied').length === 0,
    'and a restart that worked is not reported as one that failed');

  advance(1); await failProbe();             // still reloading, inside 5min cooldown
  check(dockerCalls.length === 1, 'cooldown honored — no re-trigger during reload');

  advance(2); await okProbe(140);            // model back
  check(wd._getState().awaitingRecovery === false, 'recovery cleared awaiting flag');
  // Recovery is logged with the probe that proved it — 140ms is this suite's
  // number, so the assertion is on the evidence in the line, not on the line.
  check(opsCarrying('140ms').length === 1, 'recovery is logged with the probe that proved it',
    opsLines.slice(-2).join(' || '));
  check(queuedInitiatives.length === 1, 'exactly one recovery initiative queued');
  check(queuedInitiatives[0] && queuedInitiatives[0].type === 'alert', 'initiative type = alert');
  check(queuedInitiatives[0] && queuedInitiatives[0].priority === 7, 'initiative priority = 7 (surfaces in greeting)');
  check(/locked up/.test(queuedInitiatives[0] && queuedInitiatives[0].content || ''), 'initiative reports the seizure honestly');
  check(dockerCalls.length === 1, 'total 1 restart across scenario A');

  // ===== Scenario B: persistently dead → cap → CRITICAL, stop restarting =====
  console.log('\n── Scenario B: per-hour cap + CRITICAL ──');
  wd._reset(); dockerCalls.length = 0; queuedInitiatives.length = 0; opsLines.length = 0; dockerShouldFail = false;

  // Restart #1
  await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 1, 'restart #1 fired');
  // Cooldown then restart #2
  advance(6); await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 2, 'restart #2 fired after cooldown');
  // Cooldown then cap should block restart #3
  advance(6); await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 2, 'restart #3 BLOCKED by 2/hour cap');
  // The cap line is identified by the cap it is reporting — `cap 2/hr` here —
  // which comes from this suite's config and appears in no other line.
  const capMark = `${watchdogCfg.maxRestartsPerHour}/hr`;
  check(opsCarrying(capMark).length === 1, 'hitting the cap is logged, naming the cap',
    opsLines.join(' || ').slice(-200));
  const critCount = opsCarrying(capMark).length;
  // keep failing — CRITICAL must not re-log every probe
  advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 2, 'still no restart past cap');
  check(opsCarrying(capMark).length === critCount, 'the cap is announced once, not once per probe',
    `${opsCarrying(capMark).length} line(s)`);
  check(queuedInitiatives.length === 0, 'no recovery initiative while still dead');

  // Cap window rolls over (>1h since restart #1) → a restart is allowed again
  advance(50); await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 3, 'restart allowed again after 1h cap window rolls off');

  // ===== Scenario C: docker restart itself fails ============================
  console.log('\n── Scenario C: docker restart failure path ──');
  wd._reset(); dockerCalls.length = 0; queuedInitiatives.length = 0; opsLines.length = 0; dockerShouldFail = true;

  await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 1, 'restart attempted');
  check(wd._getState().awaitingRecovery === false, 'not awaiting recovery after failed restart');
  // The mirror of scenario A: the outcome line for this target must carry the
  // reason the mock gave, because a failed restart reported without its reason
  // is the failure this whole ops line exists to prevent.
  check(opsCarrying(watchdogCfg.container, 'permission denied').length === 1,
    'the failed restart is logged against its target, with the reason it failed',
    opsAboutTarget().join(' || '));
  check(wd._getState().restartTimes.length === 1, 'failed attempt counts toward cap (prevents infinite retry)');

  // ===== Scenario D: disabled =============================================
  console.log('\n── Scenario D: watchdog disabled ──');
  wd._reset(); dockerCalls.length = 0; opsLines.length = 0; dockerShouldFail = false;
  watchdogCfg = { ...watchdogCfg, enabled: false };
  await failProbe(); advance(5); await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 0, 'disabled watchdog never restarts');
  watchdogCfg = { ...watchdogCfg, enabled: true };

  // ===== Scenario E: disposable instance (SNH_DATA_DIR) ====================
  // The gate is read from the environment at module load, so it cannot be
  // toggled in-process the way watchdogCfg can — it is exercised in a child
  // with SNH_DATA_DIR set, running the same mocks. This is the case that cost a
  // day on 2026-08-16: a throwaway instance inheriting the live config's
  // watchdog and restarting the shared container.
  console.log('\n── Scenario E: SNH_DATA_DIR disables the watchdog ──');
  const child = `
    const cp = require('child_process');
    const calls = [];
    cp.execFile = (cmd, args, opts, cb) => { calls.push(args); setImmediate(() => cb(null, '', '')); };
    const config = require('${path.join(__dirname, '../db/config')}');
    config.getConfig = () => ({ watchdog: { enabled: true, container: 'test-brain', failureThreshold: 3, cooldownMinutes: 5, maxRestartsPerHour: 2 } });
    const fe = require('${path.join(__dirname, '../db/fact-extractor')}');
    const ops = []; fe.appendToOpsLog = (m) => ops.push(m);
    const wd = require('${path.join(__dirname, '../db/brain-watchdog')}');
    (async () => {
      for (let i = 0; i < 6; i++) await wd.onProbeResult({ ok: false, ms: 8000, error: 'timeout' });
      console.log(JSON.stringify({ restarts: calls.length, spoke: ops.some(o => /DISABLED for this process/.test(o)) }));
    })();
  `;
  let childOut = '';
  try {
    childOut = require('child_process').execFileSync(process.execPath, ['-e', child], {
      encoding: 'utf8',
      env: { ...process.env, SNH_DATA_DIR: '/tmp/throwaway-watchdog-test' },
      timeout: 30000
    });
  } catch (e) {
    childOut = `child failed: ${e.message}`;
  }
  const childResult = (() => {
    const m = childOut.match(/\{[\s\S]*\}/);
    try { return m ? JSON.parse(m[0]) : null; } catch { return null; }
  })();
  check(childResult !== null, `child produced a result (${childOut.trim().slice(0, 120)})`);
  check(childResult && childResult.restarts === 0, 'disposable instance never restarts the shared container');
  check(childResult && childResult.spoke === true, 'the refusal is SPOKEN to the ops log, not silent');

  Date.now = realNow;
  console.log(`\n${failures === 0 ? '✅ ALL PASSED' : `❌ ${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
})();
