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

const OPS_DIR = require('./database').getOpsDir();
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

// A DISPOSABLE INSTANCE DOES NOT GET TO RESTART SHARED INFRASTRUCTURE.
//
// SNH_DATA_DIR redirects SQLite and LanceDB so a throwaway process cannot touch
// the live corpus (see "The replay redirects a PROCESS, not a call"). It
// deliberately does NOT redirect data/config.json, which is the right call for
// the data — a throwaway wants the real configuration against a disposable
// store — and exactly the wrong thing here: the throwaway inherits the real
// `watchdog` block, container name and all, pointed at the shared engine.
//
// On 2026-08-16 two orphaned verification instances did precisely that. They had
// been given a capture proxy as their engine host; the proxy was shut down when
// the capture finished; their liveness probes then failed forever, hit the
// threshold, and ran `docker restart sparky-brain` against the LIVE engine five
// times across a day. Each restart cost ~3.5 minutes of model load during which
// the real assistant answered "fetch failed" — including on a message the user
// actually sent. None of it appeared in the live instance's logs or ops ledger,
// because the watchdog entries belonged to the throwaways, so the outages looked
// unattributable for a full day.
//
// The redirect is therefore the signal. A process pointed at a disposable store
// is by definition not the instance responsible for the shared container, so it
// observes and reports but never acts. There is no override: an instance that
// genuinely owns its engine is the one running against the live data directory.
const DISPOSABLE_INSTANCE = !!process.env.SNH_DATA_DIR;
let disposableNoticeLogged = false;

/** Read + normalize the watchdog config each probe so knobs take effect live. */
/** The whole config, for the probe cadence the alert quotes. */
function cfgAll() { return require('./config').getConfig(); }

function cfg() {
  const w = (getConfig().watchdog) || {};
  // Spoken, not silent — a guard that disables a self-healing action without
  // saying so is the same defect class as a refused write that reports success.
  if (DISPOSABLE_INSTANCE && w.enabled !== false && !disposableNoticeLogged) {
    disposableNoticeLogged = true;
    const msg = `Brain watchdog DISABLED for this process: SNH_DATA_DIR is set (${process.env.SNH_DATA_DIR}), so this is a disposable instance and must not restart the shared container "${w.container || 'sparky-brain'}". Liveness is still probed and reported.`;
    console.warn(`[Watchdog] ${msg}`);
    opsLog(msg);
  }
  return {
    enabled: w.enabled !== false && !DISPOSABLE_INSTANCE,
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
  // THE CARD SAYS WHICH POLICY FIRED. Reading "3 consecutive failures" tells
  // you nothing about how long that took unless you also know the interval —
  // and the interval is exactly what changed after the 15-minute detection on
  // 2026-08-22. Stating both makes a slow detection self-explaining.
  const everySec = Math.max(5, ((cfgAll().livenessProbe || {}).intervalSeconds) || 60);
  const policy = `${c.failureThreshold} failed checks ${everySec}s apart`;
  const fireMsg = `🔧 Brain watchdog: ${consecutiveFailures} consecutive liveness failures (last: ${probe.error || 'unknown'}) — restarting ${c.container} (restart ${attemptNum}/${c.maxRestartsPerHour} this hour). Policy: ${policy}.`;
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
/**
 * What the watchdog currently believes about the brain, in words a
 * person can use.
 *
 * WHY THIS EXISTS. On 2026-08-21 the watchdog detected a wedged engine,
 * restarted the container, logged "model reloading, cooldown 5 min", and
 * confirmed recovery itself - and while all that was happening Ellie's
 * chat returned a bare "fetch failed". The system knew exactly what was
 * wrong and how long it would last, and told her nothing. The knowledge
 * existed; only the path from it to her did not.
 *
 * Read-only, synchronous, and never probes: it reports the state the
 * probe loop has already established. A caller in a failing request path
 * cannot afford to wait, and a status call that could itself hang would
 * be a new way to fail.
 */
function brainStatus(now = Date.now()) {
  return describeBrainState({
    restartInFlight,
    awaitingRecovery,
    consecutiveFailures,
    lastRestartAt,
    wedgeDetectedAt,
  }, cfg(), now);
}

/**
 * The message for a given brain state. PURE - state in, words out.
 *
 * Separated from brainStatus() because the watchdog disables itself when
 * SNH_DATA_DIR is set (a disposable instance must not restart the shared
 * container), which means a test can never drive its state machine. That
 * guard is right and stays; extracting the formatter is how the words
 * get tested without it - and the words are the part that failed her.
 */
/** "1 minute" / "3 minutes" - a status line reading "1 minute(s)" is a
 *  small thing that makes the whole message look machine-generated. */
function plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

function describeBrainState(st, c = {}, now = Date.now()) {
  const cooldownMs = c.cooldownMs || 0;
  const threshold = c.failureThreshold ?? 3;

  if (st.restartInFlight) {
    return {
      healthy: false,
      state: 'restarting',
      message: 'The model engine is being restarted right now. It usually takes '
             + 'a minute or two to load, and nothing was lost — try again shortly.',
    };
  }

  if (st.awaitingRecovery) {
    const since = st.lastRestartAt ? Math.round((now - st.lastRestartAt) / 1000) : null;
    const left = st.lastRestartAt
      ? Math.max(0, Math.round((st.lastRestartAt + cooldownMs - now) / 1000))
      : 0;
    return {
      healthy: false,
      state: 'reloading',
      restartedSecondsAgo: since,
      message: 'The model engine was restarted'
             + (since !== null
                 ? ` ${since < 60 ? plural(since, 'second') : plural(Math.round(since / 60), 'minute')} ago`
                 : '')
             + ' and is still loading its model. Nothing was lost — '
             + (left
                 ? `give it about ${left < 60 ? plural(left, 'more second') : plural(Math.ceil(left / 60), 'more minute')} and try again.`
                 : 'try again in a minute.'),
    };
  }

  if (st.consecutiveFailures > 0) {
    const downFor = st.wedgeDetectedAt ? Math.round((now - st.wedgeDetectedAt) / 60000) : 0;
    return {
      healthy: false,
      state: 'wedged',
      consecutiveFailures: st.consecutiveFailures,
      message: `The model engine has stopped responding${downFor ? ` (about ${plural(downFor, 'minute')} now)` : ''}. `
             + (st.consecutiveFailures >= threshold
                 ? 'A restart is due on the next check — try again in a minute or two.'
                 : `I restart it automatically after ${threshold} failed checks; this is ${st.consecutiveFailures}. Try again in a minute.`),
    };
  }

  return { healthy: true, state: 'ok', message: null };
}

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

module.exports = { onProbeResult, brainStatus, describeBrainState, _getState, _reset };
