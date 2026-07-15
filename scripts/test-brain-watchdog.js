/**
 * Logic test for the brain watchdog state machine.
 * Mocks docker (child_process.execFile), config, initiatives, ops log, and the
 * clock so every guardrail is exercised deterministically — no real container,
 * no real time. Run: node scripts/test-brain-watchdog.js
 */

// ---- Install mocks BEFORE requiring the module under test -------------------
// The module destructures execFile + getConfig at load, so patch first.
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
  check(opsMatch(/consecutive liveness failures.*restarting test-brain \(restart 1\/2/), 'fire logged to ops');
  check(opsMatch(/docker restart test-brain.* completed/), 'restart success logged to ops');

  advance(1); await failProbe();             // still reloading, inside 5min cooldown
  check(dockerCalls.length === 1, 'cooldown honored — no re-trigger during reload');

  advance(2); await okProbe(140);            // model back
  check(wd._getState().awaitingRecovery === false, 'recovery cleared awaiting flag');
  check(opsMatch(/Brain recovered after watchdog restart/), 'recovery logged to ops');
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
  check(opsMatch(/CRITICAL: Brain still wedged.*NOT restarting again/), 'CRITICAL logged when cap hit');
  const critCount = opsLines.filter(l => /CRITICAL/.test(l)).length;
  // keep failing — CRITICAL must not re-log every probe
  advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 2, 'still no restart past cap');
  check(opsLines.filter(l => /CRITICAL/.test(l)).length === critCount, 'CRITICAL logged once, not spamming');
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
  check(opsMatch(/docker restart test-brain.* FAILED/), 'restart failure logged to ops');
  check(wd._getState().restartTimes.length === 1, 'failed attempt counts toward cap (prevents infinite retry)');

  // ===== Scenario D: disabled =============================================
  console.log('\n── Scenario D: watchdog disabled ──');
  wd._reset(); dockerCalls.length = 0; opsLines.length = 0; dockerShouldFail = false;
  watchdogCfg = { ...watchdogCfg, enabled: false };
  await failProbe(); advance(5); await failProbe(); advance(5); await failProbe(); advance(5); await failProbe();
  check(dockerCalls.length === 0, 'disabled watchdog never restarts');
  watchdogCfg = { ...watchdogCfg, enabled: true };

  Date.now = realNow;
  console.log(`\n${failures === 0 ? '✅ ALL PASSED' : `❌ ${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
})();
