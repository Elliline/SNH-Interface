/**
 * WHY A RUN STOPPED, AND WHICH SIDE STOPPED IT — decided in one place, in words.
 *
 * WHY THIS EXISTS. On 2026-09-09 a background job ran for 6m45s, made 27 tool
 * calls, and closed as `failed` with `error = "terminated"`. That word is what
 * Node's fetch throws when a response body is cut off mid-stream, and the job
 * runner wrote it onto the card verbatim. Nobody could tell from the card
 * whether SNH's own runner had killed the job (a clock, a budget, a restart) or
 * whether the thing it was waiting on had died under it — which is the one
 * question that decides where to go looking. It took the journal, the engine's
 * log and the systemd log together to establish that SNH's brain watchdog had
 * restarted vLLM underneath a healthy run.
 *
 * So the raw error text never reaches her as the reason again. Every way a run
 * can stop maps here to:
 *
 *   source   who stopped it —
 *              'runner'   SNH's job runner: one of its own clocks or budgets,
 *                         or SNH itself restarting
 *              'engine'   the model engine (vLLM/Ollama) died, dropped the
 *                         connection, or refused the request
 *              'dispatched' the thing the job was handed to (squatch-code)
 *              'user'     she cancelled it
 *              'unknown'  nothing above matched — and then the raw text is
 *                         carried in `technical`, never as the sentence
 *   kind     which clock, limit or event, as a short stable token the panel
 *            and the tests can key on
 *   plain    the sentence she reads, in plain language, with the numbers said
 *            in words ("the stall limit, 105 seconds")
 *
 * PURE. No database, no engine, no config read that is not handed in. The
 * caller passes what it knows (the budget summary, the round in progress, the
 * watchdog's last restart) and gets words back, so the mapping is testable
 * without a model in the loop — scripts/test-job-failure.js.
 */

const SOURCES = ['runner', 'engine', 'dispatched', 'user', 'unknown'];

/** "105 seconds" / "15 minutes" / "1 hour 30 minutes" — a duration said out loud. */
function sayDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return 'an unknown time';
  const s = Math.round(ms / 1000);
  const unit = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
  // Seconds stay seconds up to two minutes, so a 105-second stall limit and a
  // 106-second stall do not both come out as "2 minutes" — the whole point of
  // the sentence is that the numbers can be compared.
  if (s < 120) return unit(s, 'second');
  const m = Math.floor(s / 60), rs = s % 60;
  if (m < 60) return rs ? `${unit(m, 'minute')} ${unit(rs, 'second')}` : unit(m, 'minute');
  const h = Math.floor(m / 60), rm = m % 60;
  return rm ? `${unit(h, 'hour')} ${unit(rm, 'minute')}` : unit(h, 'hour');
}

/** Where a run was when it stopped, for the sentence. */
function whereItWas({ round = null, calls = null } = {}) {
  const bits = [];
  if (Number.isFinite(round) && round > 0) bits.push(`in tool round ${round}`);
  if (Number.isFinite(calls) && calls > 0) bits.push(`after ${calls} tool call${calls === 1 ? '' : 's'}`);
  return bits.length ? ` It was ${bits.join(', ')}.` : '';
}

/**
 * A budget summary that says it bound, turned into source + kind + words.
 * The `exhausted` string is what createToolSession.spent() wrote; this reads
 * its shape rather than re-deriving the numbers, so the two cannot disagree.
 */
function classifyBudgetStop(budget = {}, { answerTokens = null, maxRounds = null } = {}) {
  const ex = String(budget && budget.exhausted || '');
  if (/^call budget spent/.test(ex)) {
    return {
      source: 'runner', kind: 'call-budget',
      plain: `SNH's job runner stopped it at its call budget: ${ex.replace(/^call budget spent \(/, '').replace(/\)$/, '')}. ` +
        `That budget is "Tool calls per job" in Settings.`
    };
  }
  if (/^attempt ceiling reached/.test(ex)) {
    return {
      source: 'runner', kind: 'attempt-ceiling',
      plain: `SNH's job runner stopped it at the hard attempt ceiling: ${ex.replace(/^attempt ceiling reached \(/, '').replace(/\)$/, '')}. ` +
        `This is the runaway guard under the call budget — it binds when almost nothing is coming back.`
    };
  }
  if (/^time budget spent/.test(ex)) {
    const m = /\((\d+)s of (\d+)s\)/.exec(ex);
    const used = m ? sayDuration(Number(m[1]) * 1000) : 'its whole allowance';
    const cap = m ? sayDuration(Number(m[2]) * 1000) : 'the limit';
    return {
      source: 'runner', kind: 'wall-clock',
      plain: `SNH's job runner stopped it at its time limit: it had been running ${used}, and the limit for one job is ${cap} ("Time limit for one job" in Settings).`
    };
  }
  if (/^round budget spent/.test(ex)) {
    return {
      source: 'runner', kind: 'rounds',
      plain: `SNH's job runner stopped it at its round limit: it used all ${maxRounds ?? ''} tool rounds ("Tool rounds per job" in Settings) and was still asking for more.`
    };
  }
  if (/no allowed tools/.test(ex)) {
    return { source: 'runner', kind: 'no-tools', plain: 'It ran with no tools at all — none of the ones it wanted were registered on this box.' };
  }
  if (ex) return { source: 'runner', kind: 'budget', plain: `SNH's job runner stopped it: ${ex}.` };
  return null;
}

/**
 * The reasons a run can be CUT SHORT without throwing, in priority order —
 * truncation first because it is the only one that says WHERE the text stops.
 */
function classifyCutShort({ truncated = false, outOfRounds = false, budget = null, answerTokens = null, maxRounds = null } = {}) {
  if (truncated) {
    return {
      source: 'runner', kind: 'answer-budget',
      plain: `It hit the answer budget (${answerTokens ?? 'the'} tokens) and stopped mid-result — what is above is cut off, not finished. Raise "Answer budget, agent jobs" in Settings if this keeps happening.`
    };
  }
  const b = classifyBudgetStop(budget, { answerTokens, maxRounds });
  if (b) return b;
  if (outOfRounds) {
    return {
      source: 'runner', kind: 'rounds',
      plain: `SNH's job runner stopped it at its round limit (${maxRounds ?? 'all'} tool rounds, "Tool rounds per job" in Settings) before it was finished.`
    };
  }
  return null;
}

/**
 * A THROWN run, classified from the error the runner caught.
 *
 * @param {Error|string} err
 * @param {Object} [ctx]
 * @param {number} [ctx.round]         tool round in progress when it threw
 * @param {number} [ctx.calls]         tool calls completed before it threw
 * @param {number} [ctx.stallMs]       the stall limit that was in force
 * @param {number} [ctx.firstTokenMs]  the first-token limit that was in force
 * @param {{issuedAt:number, reason?:string}|null} [ctx.recentRestart]
 *        the brain watchdog's last restart, if it was recent enough to be the cause
 * @param {function} [ctx.formatTime]  ms → "9:11 PM"
 */
function classifyThrown(err, ctx = {}) {
  const msg = String(err && err.message || err || '').trim();
  const name = err && err.name;
  const status = err && err.status;
  const where = whereItWas(ctx);
  const raw = msg.slice(0, 300);

  // --- SNH's own clocks. streamChat names them in the message and types them
  //     TimeoutError so the circuit breaker still counts them.
  if (/stalled — no tokens for/.test(msg)) {
    const m = /no tokens for (\d+)s \(limit (\d+)s\)/.exec(msg);
    return {
      source: 'runner', kind: 'stall',
      plain: `SNH's job runner stopped it: the engine went quiet for ${m ? sayDuration(Number(m[1]) * 1000) : 'too long'} in the middle of an answer, ` +
        `and the stall limit is ${m ? sayDuration(Number(m[2]) * 1000) : sayDuration(ctx.stallMs)} ("Stall timeout" under Background engine limits in Settings).${where}`,
      technical: raw
    };
  }
  if (/timed out waiting for the first token/.test(msg)) {
    const m = /after (\d+)s \(limit (\d+)s\)/.exec(msg);
    return {
      source: 'runner', kind: 'first-token',
      plain: `SNH's job runner stopped it: the engine did not start answering within ${m ? sayDuration(Number(m[2]) * 1000) : sayDuration(ctx.firstTokenMs)} ` +
        `("First-token timeout" under Background engine limits in Settings). Nothing had come back for that whole wait.${where}`,
      technical: raw
    };
  }
  if (/Brain circuit open/.test(msg)) {
    return {
      source: 'runner', kind: 'circuit-open',
      plain: `SNH's job runner did not send the call: the engine had been marked wedged after repeated timeouts (the circuit breaker was open), so the request was refused before it left.${where}`,
      technical: raw
    };
  }

  // --- The window. Named before the generic refusal because the sentence
  //     already carries the numbers (db/job-window.refusalSentence) and the
  //     side is the runner's: it measured, compacted, refitted, and there was
  //     still no room for one more turn. That is a limit that binds loudly,
  //     not an engine that would not take a call.
  if (name === 'ContextWindowError' || (err && err.kind === 'context-window')) {
    return { source: 'runner', kind: 'context-window', plain: `${msg}${where}`, technical: raw };
  }

  // --- The engine answered and said no.
  if (Number.isFinite(status)) {
    return {
      source: 'engine', kind: 'refused',
      plain: `The engine refused the request (HTTP ${status}${err.body ? `: ${String(err.body).slice(0, 160)}` : ''}). SNH's runner did nothing wrong here — the engine would not take the call.${where}`,
      technical: raw
    };
  }

  // --- The connection to the engine went away mid-stream. `terminated` is
  //     what Node's fetch says when the body is cut off; the rest are the
  //     socket-level spellings of the same event.
  if (/^terminated$|other side closed|socket hang up|ECONNRESET|aborted|network error|Premature close/i.test(msg)
      || (err && err.cause && /ECONNRESET|other side closed|UND_ERR_SOCKET/.test(String(err.cause.code || err.cause.message || '')))) {
    let plain = `The engine's connection dropped in the middle of an answer — the model engine closed the stream while the job was waiting on it.`;
    let kind = 'connection-cut';
    if (ctx.recentRestart && Number.isFinite(ctx.recentRestart.issuedAt)) {
      const when = typeof ctx.formatTime === 'function' ? ctx.formatTime(ctx.recentRestart.issuedAt) : new Date(ctx.recentRestart.issuedAt).toISOString();
      plain = `SNH's own brain watchdog restarted the engine at ${when}` +
        `${ctx.recentRestart.reason ? ` (${ctx.recentRestart.reason})` : ''}, and this job's answer was cut off with it. ` +
        `The engine was NOT the problem for this job — the restart was.`;
      kind = 'watchdog-restart';
      return { source: 'runner', kind, plain: plain + where, technical: raw };
    }
    return {
      source: 'engine', kind,
      plain: plain + ` SNH's runner did not stop it; something restarted or killed the engine.${where}`,
      technical: raw
    };
  }
  if (/fetch failed|ECONNREFUSED|ENOTFOUND|EHOSTUNREACH|no response body/i.test(msg)
      || (err && err.cause && /ECONNREFUSED|ENOTFOUND|EHOSTUNREACH/.test(String(err.cause.code || '')))) {
    return {
      source: 'engine', kind: 'unreachable',
      plain: `Nothing was listening at the engine when the job tried to call it — the model engine was down or restarting.${where}`,
      technical: raw
    };
  }
  if (name === 'TimeoutError' || name === 'AbortError') {
    return {
      source: 'runner', kind: 'timeout',
      plain: `SNH's job runner stopped a call on a deadline (${raw || 'no detail'}).${where}`,
      technical: raw
    };
  }

  // --- Nothing matched. The raw text is carried, and the sentence says so.
  return {
    source: 'unknown', kind: 'unexpected',
    plain: `Something unexpected stopped it, and SNH could not tell which side it came from. The raw error was: "${raw || 'no message'}".${where}`,
    technical: raw
  };
}

/** A restart of SNH itself. */
function classifyRestart({ resumed = false } = {}) {
  return {
    source: 'runner', kind: 'service-restart',
    plain: resumed
      ? 'SNH itself restarted while this was running. The round in progress was lost; it picked up again from the last completed round.'
      : 'SNH itself restarted while this was running, and the run could not be resumed.'
  };
}

function classifyCancelled() {
  return { source: 'user', kind: 'cancelled', plain: 'You cancelled it before it started.' };
}

/**
 * The one-line label the card and the announcement share: "Stopped by SNH's
 * job runner (stall)" etc. Kept short because the plain sentence follows it.
 */
function sourceLabel(source) {
  switch (source) {
    case 'runner': return "stopped by SNH's job runner";
    case 'engine': return 'stopped on the engine side';
    case 'dispatched': return 'stopped by the agent it was handed to';
    case 'user': return 'stopped by you';
    default: return 'stopped for a reason SNH could not place';
  }
}

module.exports = {
  SOURCES,
  sayDuration,
  classifyThrown,
  classifyCutShort,
  classifyBudgetStop,
  classifyRestart,
  classifyCancelled,
  sourceLabel
};
