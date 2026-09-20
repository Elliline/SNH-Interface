/**
 * What a failed chat turn WAS, and what to tell her about it.
 *
 * PURE — error in, verdict and words out. Extracted for the same reason
 * describeBrainState was: the failing part is the classification and the
 * wording, and neither could be tested while both lived inside an Express
 * handler that needs a live engine to reach.
 *
 * The incident this exists for, in full, because the shape recurs:
 * on 2026-08-22 the engine stopped generating at 07:03 with two requests in
 * flight. Her turn went out at 07:07 and our own 120s deadline killed it at
 * 07:09. The error was a TimeoutError, which has no `cause.code` and is not a
 * TypeError — so it matched neither arm of the upstream test, was classified
 * as "our own bug", answered 500, and skipped the block that would have told
 * her what the watchdog knew. A wedged engine is the single most likely reason
 * for a chat failure on this box, and it was the one case that got no
 * explanation.
 */

/**
 * Upstream (the engine) or ours (a bug in here)?
 *
 * Three ways a call can be upstream, and they arrive looking nothing alike:
 *   - `error.upstream` — set deliberately by a caller that already knows.
 *   - a network failure — never got a response at all. A bare TypeError with
 *     the reason on `cause`, carrying no flag of its own.
 *   - a timeout — got a connection and then nothing. `AbortSignal.timeout`
 *     raises TimeoutError; streamChat re-throws its own stall as one too,
 *     named deliberately so the circuit breaker keeps counting it.
 *
 * AbortError is included with the timeouts because an abort we initiated is a
 * timeout by another name. A CLIENT disconnect also aborts, but by then the
 * stream has started and the caller has already sent headers, so it never
 * reaches this decision.
 */
function classifyChatFailure(error = {}) {
  const networkFailure = !!error.cause?.code
    || (error.name === 'TypeError' && /fetch failed|network|socket/i.test(error.message || ''));
  const engineTimeout = error.name === 'TimeoutError' || error.name === 'AbortError';
  return {
    networkFailure,
    engineTimeout,
    upstream: !!error.upstream || networkFailure || engineTimeout,
    status: (!!error.upstream || networkFailure || engineTimeout) ? 502 : 500,
  };
}

/**
 * The body she actually receives.
 *
 * Preference order, and each step exists because the one below it failed her:
 *   1. The watchdog's own account, when it has one. It knows about restarts and
 *      cooldowns and can say when to try again.
 *   2. Our own deadline firing, when the watchdog has nothing yet. There is a
 *      real multi-minute window — it needs `failureThreshold` consecutive
 *      probes — where the engine is wedged and brainStatus() honestly reports
 *      healthy. Leaving the raw text there means she reads
 *      "stalled — no tokens for 61s".
 *   3. The raw message, for anything genuinely unclassified. Better than a
 *      reassuring lie about a failure nobody has diagnosed.
 *
 * `technical` always carries the original, so the real error is never lost —
 * it moves off the line she reads and into a field, which is where a stack
 * trace belongs.
 */
function chatFailureBody(error = {}, brain = null) {
  const { upstream, engineTimeout } = classifyChatFailure(error);
  const technical = error.message || null;

  if (upstream && brain && !brain.healthy && brain.message) {
    return { error: brain.message, brain: brain.state, technical };
  }

  if (upstream && engineTimeout) {
    return {
      error: 'The model engine accepted the request and then stopped '
           + 'responding, so I gave up waiting. Nothing was lost. It is '
           + 'checked automatically every few minutes and restarted if it '
           + 'stays down — try again shortly.',
      brain: 'unresponsive',
      technical,
    };
  }

  return { error: technical || 'Chat service unavailable' };
}

module.exports = { classifyChatFailure, chatFailureBody };
