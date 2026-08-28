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
const { formatLocalTime } = require('./datetime');
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
// What the engine last told us about itself, so the alert can say what actually
// happened instead of assuming. Null when nothing has been adjudicated yet.
let lastEngineState = null;
let lastVerdict = null;
let restartIssuedAt = 0;       // ms epoch the restart command went out — the moment
                               // unavailability actually BEGINS, which is not the
                               // moment the first probe timed out

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
    // NOT EVERY BOX RUNS THE BRAIN IN A CONTAINER. `watchdog.restartCommand`,
    // when set, is an argv array run instead of `docker restart <container>`.
    // aiserver is why: the engine there is a systemd --user unit, the inherited
    // container name pointed at nothing, and every fire would have failed with
    // "no such container" while reporting a restart had been attempted.
    // Unset (the default) keeps the container path exactly as it was.
    restartArgv: Array.isArray(w.restartCommand) && w.restartCommand.length
      ? w.restartCommand.map(String)
      : null,
    failureThreshold: Math.max(1, w.failureThreshold || 3),
    cooldownMs: Math.max(0, (w.cooldownMinutes ?? 5) * 60 * 1000),
    maxRestartsPerHour: Math.max(1, w.maxRestartsPerHour || 2)
  };
}

/**
 * THE ONE NUMBER THAT SEPARATES A BUSY ENGINE FROM A STUCK ONE, in words.
 *
 * On 2026-08-27 the alert said "locked up ... wedged engine". The engine was
 * generating at 79.8 tok/s with 16 requests running and 10 queued. Nothing in
 * the alert carried the queue, so nothing in it could have said otherwise, and
 * the wrong word was reported onward in good faith and set the whole
 * investigation off in the wrong direction.
 *
 * Exported because both the probe loop and the alert have to say this, and a
 * second copy of the phrasing is how the two drift apart.
 */
function describeQueue(engine) {
  if (!engine || !engine.reachable) return 'the engine was not answering at all';
  const running = engine.running ?? 0;
  const waiting = engine.waiting ?? 0;
  const progress = engine.generating === true ? 'and still producing tokens'
    : engine.generating === false ? 'and producing nothing'
    : 'with progress unknown';
  return `${running} request(s) running, ${waiting} queued, ${progress}`;
}

/** How the restart reads in a log line — the real command, not an assumed one. */
function restartLabel(c) {
  return c.restartArgv ? c.restartArgv.join(' ') : `docker restart ${c.container}`;
}

/**
 * Restart the brain. Resolves { ok, ms, error }.
 * execFile (no shell) — argv comes from config, not user input, but keep it clean.
 */
function restartBrain(c) {
  const argv = c.restartArgv || ['docker', 'restart', c.container];
  const started = Date.now();
  return new Promise(resolve => {
    execFile(argv[0], argv.slice(1), { timeout: 90000 }, (err, stdout, stderr) => {
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
    // Was pinned to America/Los_Angeles. It reads the instance clock now, so a
    // box that is not in Oregon tells its own person the right time.
    const clock = formatLocalTime(wedgeAt, { style: 'time', fallback: 'an unclear time' });
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
      // Two different intervals, and conflating them is what made the 8/27 alert
      // wrong. `unavailableMs` is from the restart command to this answer — the
      // window nothing could be served. `wedgeAt` is when we first suspected,
      // which is when to go looking in the logs, and is reported as a time
      // rather than a duration.
      const unavailableMs = now - (restartIssuedAt || lastRestartAt || wedgeAt);
      const secs = Math.max(1, Math.round(unavailableMs / 1000));
      const msg = `✅ Brain recovered after watchdog restart — responded in ${probe.ms}ms `
        + `(unavailable ${secs}s from restart to first answer; first suspected at `
        + `${formatLocalTime(wedgeAt, { style: 'time', fallback: 'an unclear time' })}, `
        + `verdict then: ${lastVerdict || 'unclassified'}). Engine healthy.`;
      console.log(`[Watchdog] ${msg}`);
      opsLog(msg);
      await queueRecoveryInitiative(wedgeAt, unavailableMs);
    }
    return;
  }

  // ---- A SATURATED ENGINE IS NEVER RESTARTED --------------------------------
  //
  // The probe is an ordinary completion, so under load it queues and its latency
  // reports queue depth rather than health. `verdict === 'saturated'` means the
  // adjudicator went and asked the engine directly and found it holding work AND
  // making progress on it. That engine is working. Restarting it does not help;
  // it discards every request in flight — sixteen of them, on 2026-08-27.
  //
  // The counter is RESET rather than merely skipped. A saturated probe is
  // positive evidence that the engine is alive, which is exactly what a
  // consecutive-failure counter is trying to rule out, so letting an earlier
  // strike stand across it would let two unrelated busy minutes accumulate into
  // a restart.
  if (probe.verdict === 'saturated') {
    consecutiveFailures = 0;
    wedgeDetectedAt = 0;
    lastEngineState = probe.engine || null;
    return;
  }

  // ---- Brain is not answering ----------------------------------------------
  // Everything past here is 'unreachable' (nothing listening) or 'stalled'
  // (holding work, producing nothing) — the two conditions a restart can fix.
  // A probe with no verdict at all is treated as a strike: this runs on a path
  // where the engine already failed to answer, and an unadjudicated failure is
  // the old behaviour, which is the safe direction to fail in.
  consecutiveFailures++;
  lastEngineState = probe.engine || null;
  lastVerdict = probe.verdict || 'unclassified';
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
  // UNAVAILABILITY STARTS HERE, not at the first failed probe. On 2026-08-27 the
  // alert said "unresponsive for ~7 min" measured from the first timeout — but
  // the engine answered twice inside that window, at 4149ms and 5081ms. The only
  // interval it was genuinely unreachable was the one this restart opened.
  restartIssuedAt = Date.now();
  wedgeDetectedAtBeforeRestart = wedgeDetectedAt; // preserve original wedge time across the counter reset
  const attemptNum = restartTimes.length + 1;
  // THE CARD SAYS WHICH POLICY FIRED. Reading "3 consecutive failures" tells
  // you nothing about how long that took unless you also know the interval —
  // and the interval is exactly what changed after the 15-minute detection on
  // 2026-08-22. Stating both makes a slow detection self-explaining.
  const everySec = Math.max(5, ((cfgAll().livenessProbe || {}).intervalSeconds) || 60);
  const policy = `${c.failureThreshold} failed checks ${everySec}s apart`;
  const fireMsg = `🔧 Brain watchdog: ${consecutiveFailures} consecutive liveness failures (last: ${probe.error || 'unknown'}) — restarting via \`${restartLabel(c)}\` (restart ${attemptNum}/${c.maxRestartsPerHour} this hour). Policy: ${policy}.`;
  console.warn(`[Watchdog] ${fireMsg}`);
  opsLog(fireMsg);

  const result = await restartBrain(c);
  restartInFlight = false;
  lastRestartAt = Date.now();
  restartTimes.push(lastRestartAt);
  consecutiveFailures = 0;      // fresh observation window after the action
  capCriticalLogged = false;

  if (result.ok) {
    awaitingRecovery = true;
    const okMsg = `Brain watchdog: \`${restartLabel(c)}\` completed in ${(result.ms / 1000).toFixed(1)}s — model reloading, cooldown ${Math.round(c.cooldownMs / 60000)} min before any re-trigger. Watching for recovery.`;
    console.log(`[Watchdog] ${okMsg}`);
    opsLog(okMsg);
  } else {
    const failMsg = `⚠️ Brain watchdog: \`${restartLabel(c)}\` FAILED: ${result.error}. Will retry next cycle (subject to cap). Check the command's permissions and that the target exists.`;
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
  lastEngineState = null;
  lastVerdict = null;
  restartIssuedAt = 0;
}

module.exports = { onProbeResult, brainStatus, describeBrainState, describeQueue, _getState, _reset };
