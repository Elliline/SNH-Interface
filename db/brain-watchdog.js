/**
 * Brain Watchdog — self-healing for the vLLM wedge failure mode.
 *
 * Failure mode (documented autopsy, 2026-07-10 17:44 PDT): the vLLM engine on
 * the GB10 wedges under sustained background load — generation throughput
 * collapses to 0.0 tokens/s with requests stuck in the Running state while the
 * HTTP server still answers /v1/models with 200. SNH's liveness probe detects
 * this perfectly (a tiny completion times out) but, before this module, nothing
 * acted: the circuit breaker degraded gracefully and waited forever for a human.
 *
 * This watchdog supplies the missing ACTION. It is fed every liveness-probe
 * result by memory-manager's probe loop and, after N consecutive failures,
 * runs `docker restart <container>` on the brain. Guardrails:
 *   - Cooldown: after any restart, a grace window (model reload takes ~3min) in
 *     which failures are observed but never trigger another restart.
 *   - Hard cap: at most maxRestartsPerHour. Past the cap, it STOPS restarting,
 *     logs CRITICAL once, and leaves the circuit-breaker degradation as the
 *     fallback — because if a restart-every-N-minutes loop isn't fixing it,
 *     something worse is wrong and thrashing the container won't help.
 *   - Loud logging: every decision goes to console AND the ops log.
 *   - Honest reporting: on recovery it queues an `alert` initiative so the user
 *     hears about the seizure conversationally ("my brain locked up and I
 *     restarted it myself").
 *
 * Config lives under `watchdog` in db/config.js (all knobs hot-read each probe).
 *
 * ROOT FIX APPLIED (2026-07-23): the sustained-load wedge was traced to the
 * upstream GB10 / SM 12.1 bug vllm-project/vllm#40969 — silent hang with the
 * default cudagraph_mode FULL_AND_PIECEWISE. sparky-brain now launches with
 * cudagraph_mode PIECEWISE and gpu-memory-utilization 0.80 (see
 * scripts/launch-brain.sh), which held clean over a 24-sequential + 24-concurrent
 * stress test. This watchdog is now the SMOKE ALARM, not the fix: with the root
 * cause addressed, a future trip should be treated as a NEW failure to diagnose,
 * not the known wedge.
 */

const { execFile } = require('child_process');
const path = require('path');
const { getConfig } = require('./config');

const OPS_DIR = path.join(__dirname, '../data/memory/ops');
const HOUR_MS = 60 * 60 * 1000;

// ---- State (module-local; single brain, single watchdog) --------------------
let consecutiveFailures = 0;   // consecutive liveness-probe failures
let lastRestartAt = 0;         // ms epoch of the last docker restart we issued
let restartTimes = [];         // ms epochs of restarts in the trailing hour (cap window)
let awaitingRecovery = false;  // a restart fired; watching for the probe to go green
let wedgeDetectedAt = 0;       // ms epoch of the first failure in the current streak
let wedgeDetectedAtBeforeRestart = 0; // wedge time preserved across the post-restart counter reset (for honest "down N min")
let restartInFlight = false;   // a docker restart is currently executing
let capCriticalLogged = false; // CRITICAL-once latch while blocked by the cap

/** Lazy requires to avoid a require cycle (initiatives → memory-manager → watchdog). */
function opsLog(msg) {
  try {
    require('./fact-extractor').appendToOpsLog(msg, OPS_DIR);
  } catch (e) { /* best-effort — never let logging break the probe loop */ }
}

function pruneWindow(now) {
  restartTimes = restartTimes.filter(t => now - t < HOUR_MS);
}

/** Read + normalize the watchdog config each probe so knobs take effect live. */
function cfg() {
  const w = (getConfig().watchdog) || {};
  return {
    enabled: w.enabled !== false,
    container: w.container || 'sparky-brain',
    failureThreshold: Math.max(1, w.failureThreshold || 3),
    cooldownMs: Math.max(0, (w.cooldownMinutes ?? 5) * 60 * 1000),
    maxRestartsPerHour: Math.max(1, w.maxRestartsPerHour || 2)
  };
}

/**
 * Run `docker restart <container>`. Resolves { ok, ms, error }.
 * execFile (no shell) — container name is config, not user input, but keep it clean.
 */
function dockerRestart(container) {
  const started = Date.now();
  return new Promise(resolve => {
    execFile('docker', ['restart', container], { timeout: 90000 }, (err, stdout, stderr) => {
      const ms = Date.now() - started;
      if (err) {
        resolve({ ok: false, ms, error: (stderr || err.message || '').toString().trim() });
      } else {
        resolve({ ok: true, ms });
      }
    });
  });
}

/**
 * Queue the honest "I had a seizure and fixed myself" alert so the user learns
 * about it conversationally on next contact. Best-effort; needs the brain (now
 * recovered) for greeting delivery, which is fine by the time this runs.
 */
async function queueRecoveryInitiative(wedgeAt, downMs) {
  try {
    const initiatives = require('./initiatives');
    const when = new Date(wedgeAt);
    const clock = when.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Los_Angeles' });
    const downMin = Math.max(1, Math.round(downMs / 60000));
    await initiatives.addInitiative({
      type: 'alert',
      content: `Heads up — my brain locked up around ${clock} (wedged engine, unresponsive for ~${downMin} min) and I restarted it myself. Everything's back to normal now; no action needed on your end.`,
      sourceKind: 'watchdog',
      sourceRef: `brain-restart:${new Date(wedgeAt).toISOString()}`,
      priority: 7
    });
  } catch (e) {
    console.error('[Watchdog] Failed to queue recovery initiative:', e.message);
  }
}

/**
 * Fed every liveness-probe result by memory-manager's probe loop.
 * @param {{ok: boolean, ms: number, error?: string}} probe
 */
async function onProbeResult(probe) {
  const c = cfg();
  if (!c.enabled) return;
  const now = Date.now();

  // ---- Brain is answering ---------------------------------------------------
  if (probe.ok) {
    consecutiveFailures = 0;
    wedgeDetectedAt = 0;
    capCriticalLogged = false;
    if (awaitingRecovery) {
      awaitingRecovery = false;
      const wedgeAt = wedgeDetectedAtBeforeRestart || lastRestartAt || now;
      const downMs = now - wedgeAt;
      const downMin = Math.max(1, Math.round(downMs / 60000));
      const msg = `✅ Brain recovered after watchdog restart — responded in ${probe.ms}ms (down ~${downMin} min from first failure). Engine healthy.`;
      console.log(`[Watchdog] ${msg}`);
      opsLog(msg);
      await queueRecoveryInitiative(wedgeAt, downMs);
    }
    return;
  }

  // ---- Brain is not answering ----------------------------------------------
  consecutiveFailures++;
  if (consecutiveFailures === 1) wedgeDetectedAt = now;

  // Don't stack restarts or act while one is executing.
  if (restartInFlight) return;

  // Cooldown grace: a restart just fired; the model is reloading and probes are
  // expected to fail. Observe but never re-trigger inside the window.
  if (lastRestartAt && (now - lastRestartAt) < c.cooldownMs) return;

  if (consecutiveFailures < c.failureThreshold) return;

  // Threshold reached. Enforce the per-hour cap BEFORE restarting.
  pruneWindow(now);
  if (restartTimes.length >= c.maxRestartsPerHour) {
    if (!capCriticalLogged) {
      capCriticalLogged = true;
      const msg = `🚨 CRITICAL: Brain still wedged after ${restartTimes.length} watchdog restart(s) in the last hour (cap ${c.maxRestartsPerHour}/hr) — NOT restarting again. Something worse than a routine wedge is wrong; manual intervention needed. Circuit breaker remains the degradation fallback.`;
      console.error(`[Watchdog] ${msg}`);
      opsLog(msg);
    }
    return;
  }

  // Fire the restart.
  restartInFlight = true;
  wedgeDetectedAtBeforeRestart = wedgeDetectedAt; // preserve original wedge time across the counter reset
  const attemptNum = restartTimes.length + 1;
  const fireMsg = `🔧 Brain watchdog: ${consecutiveFailures} consecutive liveness failures (last: ${probe.error || 'unknown'}) — restarting ${c.container} (restart ${attemptNum}/${c.maxRestartsPerHour} this hour).`;
  console.warn(`[Watchdog] ${fireMsg}`);
  opsLog(fireMsg);

  const result = await dockerRestart(c.container);
  restartInFlight = false;
  lastRestartAt = Date.now();
  restartTimes.push(lastRestartAt);
  consecutiveFailures = 0;      // fresh observation window after the action
  capCriticalLogged = false;

  if (result.ok) {
    awaitingRecovery = true;
    const okMsg = `Brain watchdog: \`docker restart ${c.container}\` completed in ${(result.ms / 1000).toFixed(1)}s — model reloading, cooldown ${Math.round(c.cooldownMs / 60000)} min before any re-trigger. Watching for recovery.`;
    console.log(`[Watchdog] ${okMsg}`);
    opsLog(okMsg);
  } else {
    const failMsg = `⚠️ Brain watchdog: \`docker restart ${c.container}\` FAILED: ${result.error}. Will retry next cycle (subject to cap). Check docker permissions / daemon.`;
    console.error(`[Watchdog] ${failMsg}`);
    opsLog(failMsg);
  }
}

/** Test/inspection helper: current internal state. */
function _getState() {
  return { consecutiveFailures, lastRestartAt, restartTimes: [...restartTimes], awaitingRecovery, restartInFlight, capCriticalLogged };
}

/** Test helper: reset all state. */
function _reset() {
  consecutiveFailures = 0;
  lastRestartAt = 0;
  restartTimes = [];
  awaitingRecovery = false;
  wedgeDetectedAt = 0;
  wedgeDetectedAtBeforeRestart = 0;
  restartInFlight = false;
  capCriticalLogged = false;
}

module.exports = { onProbeResult, _getState, _reset };
