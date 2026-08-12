/**
 * The READ side of scheduled jobs — what he proposed, what Ellie decided, when
 * each one runs next, and what happened the last time it ran.
 *
 * WHY THIS EXISTS. He could propose a cron job and Ellie could approve it, and
 * then he had no way to look at either. Asked "which approved job never ran", he
 * reached for a tool that does not exist, the malformed call rendered into the
 * chat, and the answer he eventually gave was invented. Every part of that failure
 * is fixed somewhere else; this is the part that gives him something true to say.
 *
 * ⚠ THE ANSWER CHANGED ON 2026-08-12, AND THAT IS THE DANGEROUS PART.
 *
 * This file used to state, in every result, that there was no scheduler and that
 * every job had run zero times. That was true, and it was load-bearing: it was
 * the fix for him describing a job as though it had been running. There is a
 * scheduler now (db/scheduler.js), so repeating the old warning would make this
 * tool the thing that lies.
 *
 * What replaces it is NOT a cheerful "yes it runs". Both directions are wrong to
 * guess, and the new failure available to him is the opposite of the old one:
 * assuming a job ran because a scheduler exists and its hour has passed. So every
 * job carries REAL values — next_run and last_run computed from the job row and
 * the run log — and the top-level imperative now says: report these numbers,
 * never infer a run from the schedule.
 *
 * The top-level placement is unchanged and still not cosmetic. Measured in Phase
 * 2b: a warning nested inside an object reads as a field, gets skimmed, and he
 * answers around it — nested null provenance produced an invented verbatim quote.
 * Only a top-level imperative changed the behaviour. For the same reason a job
 * that has never run says so in words rather than by an absent last_run, because
 * an absence reads as a blank to be filled.
 *
 * READ-ONLY, and must stay that way. Proposing goes through create_cron_job;
 * approving is Ellie's, on the Self tab; running is the scheduler's. Nothing in
 * this file writes.
 */

const { getSqliteDb } = require('./database');
const { formatFactTimestamp } = require('./datetime');

/**
 * Times go to him in LOCAL time, the way every other timestamp in his context
 * does (formatFactTimestamp is what the identity block uses).
 *
 * Measured, on the first day this tool returned real times: handed the raw
 * "2026-08-12T16:01:00.174Z", he reported the run as "4:01 AM (UTC)" — the right
 * instant, converted wrong, and stated with the confidence of something read off
 * a record. A cron schedule here is local ("0 9 * * *" means 9am in this house),
 * so a UTC string is also the wrong unit for the question being asked. The ISO
 * value is kept alongside under an explicit name for anything that needs to
 * compute rather than say.
 */
function localTime(iso) {
  return iso ? formatFactTimestamp(iso) : null;
}

/** Only his. Jobs from any other source are not his to report on. */
const KID_SOURCE = 'kid-proposed';

const MAX_LIMIT = 25;
const DEFAULT_LIMIT = 10;

/**
 * The one true statement about execution, in his own register.
 *
 * Kept as a constant so the tool result, the manifest entry and the briefing
 * cannot drift into three different claims about the same thing.
 */
const SCHEDULER_NOTE =
  'THESE RUN NOW — but read the numbers, do not reason from the schedule. A scheduler ' +
  'checks every minute and runs a job that is approved, enabled and armed. So each job ' +
  'below carries real values: `times_run`, `last_run`, `last_status` and `next_run` come ' +
  'from the run log and the job row, not from the cron expression. Report those. NEVER ' +
  'infer that a job ran because its hour has passed, because Ellie approved it, or because ' +
  'a scheduler exists — a job can be unarmed, disabled after failing, or deferred. If ' +
  '`times_run` is 0, say it has never run. If `last_status` is "failed", say it failed and ' +
  'give the error. If `next_run` is null, say it is not scheduled to run and give the reason ' +
  'in `not_running_because`.';

/** "0 9 * * *" → "every day at 09:00", where it can be said simply. */
function describeSchedule(expr) {
  const raw = String(expr || '').trim();
  const f = raw.split(/\s+/);
  if (f.length !== 5) return { text: raw, plain: null };
  const [min, hour, dom, mon, dow] = f;
  const num = (s) => /^\d+$/.test(s);
  const at = num(min) && num(hour) ? `${hour.padStart(2, '0')}:${min.padStart(2, '0')}` : null;
  const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  let plain = null;
  if (at && dom === '*' && mon === '*' && dow === '*') plain = `every day at ${at}`;
  else if (at && dom === '*' && mon === '*' && num(dow)) plain = `every ${DAYS[Number(dow) % 7]} at ${at}`;
  else if (at && num(dom) && mon === '*' && dow === '*') plain = `on the ${dom}${ordinal(dom)} of each month at ${at}`;
  return { text: raw, plain };
}
function ordinal(n) {
  const v = Number(n) % 100;
  if (v >= 11 && v <= 13) return 'th';
  return { 1: 'st', 2: 'nd', 3: 'rd' }[Number(n) % 10] || 'th';
}

/** One job, shaped for him to read out. */
function shapeJob(row, db) {
  const sched = describeSchedule(row.schedule);

  // The bell item this proposal raised, and whether she ever saw it. A proposal
  // that was never delivered is a different situation from one she declined, and
  // he should be able to tell them apart.
  let initiative = null;
  if (row.initiative_id) {
    try {
      const i = db.prepare('SELECT id, status, content, delivered_at FROM initiatives WHERE id = ?').get(row.initiative_id);
      if (i) {
        initiative = {
          id: i.id,
          status: i.status,
          delivered_at: i.delivered_at || null,
          shown_to_her: i.status === 'delivered',
          content: i.content
        };
      }
    } catch { /* the initiative may have aged out; the job still stands */ }
  }

  // Execution state, read from the job row and the run log rather than derived
  // from the schedule. This is the half that used to be a constant "never".
  let rt = null;
  try { rt = require('./scheduler').runtimeState(row.id); } catch { /* the record still stands */ }

  // Why a job is not going to run, when it is not — stated rather than left as a
  // null for him to fill in. The four reasons are genuinely different and a
  // person asking "why didn't it run" wants exactly this sentence.
  let notRunningBecause = null;
  if (!rt || !rt.armed) {
    if (row.status === 'proposed') notRunningBecause = 'Ellie has not decided on it yet — a proposal is not scheduled.';
    else if (row.status === 'rejected') notRunningBecause = 'she rejected it.';
    else if (row.status === 'reverted') notRunningBecause = 'it was reverted after being approved.';
    else if (rt && rt.disabledReason) notRunningBecause = `it disabled itself: ${rt.disabledReason}`;
    else if (row.status === 'approved' && !row.enabled) notRunningBecause = 'it is approved but disabled, so the scheduler skips it.';
    else notRunningBecause = 'it is approved but not armed — its schedule could not be evaluated.';
  }

  const lastRun = rt && rt.lastRunAt;
  return {
    id: row.id,
    description: row.description,
    schedule: sched.text,
    schedule_in_words: sched.plain,
    status: row.status,
    // `enabled` is what HE asked for when proposing, and what the scheduler now
    // reads. Kept under the same name: it was never a statement that something
    // was running, and it still is not — `next_run` is.
    enabled_as_proposed: !!row.enabled,
    proposed_at: row.created_at || null,
    decided_at: row.decided_at || null,
    decision_note: row.decided_note || null,
    // The fields most likely to be answered from imagination. All measured.
    times_run: rt ? rt.timesRun : 0,
    last_run: localTime(lastRun) || 'never — it has not run yet',
    last_status: (rt && rt.lastStatus) || null,
    last_error: (rt && rt.lastError) || null,
    next_run: localTime(rt && rt.nextRunAt),
    // Local is what he says out loud; ISO is for anything that has to compute.
    times_are_local: 'All times here are local time. Say them as given — do not convert them or label them UTC.',
    last_run_iso: lastRun || null,
    next_run_iso: (rt && rt.nextRunAt) || null,
    not_running_because: notRunningBecause,
    consecutive_failures: rt ? rt.consecutiveFailures : 0,
    disabled_by_failures: !!(rt && rt.disabledReason),
    proposal_notice: initiative,
    conversation_id: row.conversation_id || null
  };
}

/**
 * List his scheduled jobs.
 *
 * @param {Object} args
 * @param {string} [args.status] - proposed | approved | rejected | reverted
 * @param {string} [args.id] - one job in full
 * @param {number} [args.limit]
 */
function jobs(args = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'The job store is not available right now.' };

  const wanted = String(args.id || '').trim();
  if (wanted) {
    const row = db.prepare(
      'SELECT * FROM cron_jobs WHERE source = ? AND (id = ? OR id LIKE ?)'
    ).get(KID_SOURCE, wanted, `${wanted}%`);
    if (!row) {
      return {
        scheduler_note: SCHEDULER_NOTE,
        not_found_warning:
          `There is no scheduled job with id "${wanted}". Say that you have no record of it ` +
          'rather than describing a job from memory — if it is not here, you did not propose it, ' +
          'or it was proposed by something other than you.',
        job: null
      };
    }
    return { scheduler_note: SCHEDULER_NOTE, job: shapeJob(row, db) };
  }

  const status = ['proposed', 'approved', 'rejected', 'reverted'].includes(args.status) ? args.status : null;
  const limit = Math.min(MAX_LIMIT, Math.max(1, parseInt(args.limit, 10) || DEFAULT_LIMIT));

  const rows = status
    ? db.prepare('SELECT * FROM cron_jobs WHERE source = ? AND status = ? ORDER BY datetime(created_at) DESC LIMIT ?').all(KID_SOURCE, status, limit)
    : db.prepare('SELECT * FROM cron_jobs WHERE source = ? ORDER BY datetime(created_at) DESC LIMIT ?').all(KID_SOURCE, limit);

  const counts = db.prepare(
    'SELECT status, COUNT(*) n FROM cron_jobs WHERE source = ? GROUP BY status'
  ).all(KID_SOURCE).reduce((a, r) => { a[r.status] = r.n; return a; }, {});

  const out = {
    scheduler_note: SCHEDULER_NOTE,
    total_by_status: counts,
    matched: rows.length,
    jobs: rows.map(r => shapeJob(r, db))
  };

  // An empty result must say what it means. "No record" reads as "nothing to
  // report" unless it is told otherwise — the same rule the search tool follows.
  if (!rows.length) {
    out.empty_warning = status
      ? `You have no jobs with status "${status}". Say exactly that. Do not describe jobs of another status as if they had this one.`
      : 'You have never proposed a scheduled job. Say exactly that rather than describing one from memory.';
  }
  return out;
}

/**
 * The single entry point, matching db/memory-inspect.js's `run` so the tool layer
 * treats every read tool the same way.
 *
 * EVERY CALL IS LOGGED, including the ones that fail, for the same reason the
 * inspect tools log theirs: "he said he checked" and "he checked" are
 * indistinguishable from the answer alone, and tool_call_log is the only thing
 * that can tell them apart afterwards. The first version of this file skipped the
 * log — the answer it produced was correct and there was no way to prove it had
 * not been guessed.
 *
 * It shares the memoryInspect rate cap rather than having its own, because it
 * shares the reason for one: his injection budget.
 */
async function run(tool, args = {}, context = {}) {
  const log = (outcome, detail) => {
    try {
      require('./cron-jobs').logToolCall({
        tool, args, outcome, detail,
        conversationId: context.conversationId || null
      });
    } catch (e) { console.error('[JobsInspect] logToolCall failed:', e.message); }
  };

  if (tool !== 'memory_jobs') {
    log('error', `unknown tool ${tool}`);
    return { error: `jobs-inspect does not serve "${tool}".` };
  }

  const memoryInspect = require('./memory-inspect');
  const cap = memoryInspect.checkCap();
  if (!cap.ok) {
    log('rejected-cap', cap.reason);
    return { error: `Not looked up — ${cap.reason}. Say plainly that you have hit your own limit for looking things up rather than answering as though you had checked.` };
  }

  let result;
  try {
    result = jobs(args);
  } catch (err) {
    console.error('[JobsInspect] error:', err.message);
    log('error', err.message);
    return { error: `Could not read the job store: ${err.message}` };
  }

  if (result && result.error) { log('error', String(result.error).slice(0, 200)); return result; }

  const detail = result.job !== undefined
    ? (result.job
        ? `job ${String(result.job.id).slice(0, 8)} (${result.job.status}, run ${result.job.times_run}×, next ${result.job.next_run || 'not scheduled'})`
        : 'job not found')
    : `${result.matched} job(s)${args.status ? ` status=${args.status}` : ''}`;
  log('read', detail);
  console.log(`[JobsInspect] memory_jobs: ${detail}`);
  return result;
}

module.exports = { run, jobs, describeSchedule, SCHEDULER_NOTE, KID_SOURCE };
