/**
 * The READ side of scheduled jobs — what he proposed, what Ellie decided, and
 * whether any of it has ever actually run.
 *
 * WHY THIS EXISTS. He could propose a cron job and Ellie could approve it, and
 * then he had no way to look at either. Asked "which approved job never ran", he
 * reached for a tool that does not exist, the malformed call rendered into the
 * chat, and the answer he eventually gave was invented. Every part of that failure
 * is fixed somewhere else; this is the part that gives him something true to say.
 *
 * ⚠ THE HONEST ANSWER ABOUT NEXT RUN IS "NEVER".
 *
 * There is no scheduler. Nothing in this codebase reads `cron_jobs` and executes
 * anything — no node-cron, no timer, no worker. `create_cron_job` writes a row,
 * the Self tab approves it, and the row sits there. An approved job is a RECORD
 * OF A DECISION, not a thing that happens.
 *
 * So every job this returns carries `next_run: "never"` and a `scheduler_warning`
 * at the TOP LEVEL of the result, phrased as an imperative. That placement is not
 * cosmetic and was measured in Phase 2b: a warning nested inside an object reads
 * as a field, gets skimmed, and he answers around it — nested null provenance
 * produced an invented verbatim quote. Only a top-level imperative changed the
 * behaviour. "Never" also has to be stated rather than implied by an absent
 * last_run, because an absence reads as a blank to be filled.
 *
 * READ-ONLY, and must stay that way. Proposing goes through create_cron_job;
 * approving is Ellie's, on the Self tab. Nothing in this file writes.
 */

const { getSqliteDb } = require('./database');

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
const NO_SCHEDULER =
  'NOTHING RUNS THESE. There is no scheduler in this system: no process reads the ' +
  'schedule and executes the job. An approved job is a record of a decision Ellie ' +
  'made, not a task that happens. If you are asked whether a job ran, or when it ' +
  'runs next, say plainly that it has never run and cannot until a scheduler is ' +
  'built — do NOT infer from the schedule that it has been running.';

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

  return {
    id: row.id,
    description: row.description,
    schedule: sched.text,
    schedule_in_words: sched.plain,
    status: row.status,
    // `enabled` is what HE asked for when proposing, not a statement about
    // whether anything is running. Named so it cannot be read as the latter.
    enabled_as_proposed: !!row.enabled,
    proposed_at: row.created_at || null,
    decided_at: row.decided_at || null,
    decision_note: row.decided_note || null,
    // The two fields most likely to be answered from imagination.
    last_run: 'never',
    next_run: 'never — no scheduler exists',
    times_run: 0,
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
        scheduler_warning: NO_SCHEDULER,
        not_found_warning:
          `There is no scheduled job with id "${wanted}". Say that you have no record of it ` +
          'rather than describing a job from memory — if it is not here, you did not propose it, ' +
          'or it was proposed by something other than you.',
        job: null
      };
    }
    return { scheduler_warning: NO_SCHEDULER, job: shapeJob(row, db) };
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
    scheduler_warning: NO_SCHEDULER,
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
    ? (result.job ? `job ${String(result.job.id).slice(0, 8)} (${result.job.status}, never run)` : 'job not found')
    : `${result.matched} job(s)${args.status ? ` status=${args.status}` : ''}`;
  log('read', detail);
  console.log(`[JobsInspect] memory_jobs: ${detail}`);
  return result;
}

module.exports = { run, jobs, describeSchedule, NO_SCHEDULER, KID_SOURCE };
