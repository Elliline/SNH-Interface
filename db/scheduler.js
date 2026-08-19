/**
 * The scheduler — the piece that was missing, and the first brick of the agent arc.
 *
 * WHY THIS EXISTS. On 2026-08-05 Aurelius proposed a daily digest of background
 * memory maintenance and Ellie approved it. Nothing happened, for six days,
 * because approving wrote a row and no process read it. Asked later which
 * approved job had never run, he had no tool that could answer and described the
 * job as though it had been running. Every part of that has been fixed except
 * the part that makes the answer worth having: something that actually runs it.
 *
 * ONE JOB TYPE. A job is an AGENT RUN. The description — which is prose, written
 * by him at propose time — becomes the task prompt for a background model call
 * with a READ-ONLY tool allowlist, and what it produces goes to Ellie's bell
 * panel and into the run log. There is no shell, no code execution, no arbitrary
 * side effect, and no per-job escape hatch to add one: a scheduler that can run
 * commands is a different security posture and a separate decision. The one
 * thing a job can do is look at the record and write a paragraph about it.
 *
 * BUILT LIKE THE CORRECTOR, because the same rules apply to anything that acts
 * while nobody is in the room:
 *
 *   GATED     — scheduler.enabled, and per job: approved AND enabled AND armed.
 *   BUDGETED  — the same two-limit tool session the corrector uses (calls and
 *               wall clock), plus a rounds cap and an output ceiling.
 *   SERIAL    — one run at a time, process-wide. A job that comes due while
 *               another is running waits for the next tick and SAYS SO in the
 *               run log rather than being quietly dropped.
 *   LEDGERED  — every attempt writes a job_runs row, including the ones that did
 *               not execute. A log of successes cannot answer "why didn't it run
 *               this morning", which is the only question anyone asks a
 *               scheduler.
 *   HONEST    — it never claims a run it did not make, and it never hides one it
 *               did. A run that produced nothing is a failure, not a silence.
 *
 * RE-ENTRANCY. A job never starts while its own previous run is unfinished —
 * checked in memory AND against the run log, because the in-memory flag dies
 * with the process and an interrupted run would otherwise block its job forever.
 * The startup sweep closes those out as `skipped` with the reason, which is both
 * true and unblocking.
 *
 * CATCH-UP. Exactly one missed run, and only if it is still worth having:
 * scheduler.catchupGraceMinutes (default 120). Past that the miss is recorded as
 * skipped, with the reason, and the job is re-armed forward. The arithmetic
 * cannot backfill more than one run because a job holds ONE next_run_at, and
 * re-arming computes from now rather than from the missed time.
 */

const { randomUUID } = require('crypto');
const path = require('path');
const { getSqliteDb, getDataDir } = require('./database');
const { getConfig } = require('./config');
const cronEval = require('./cron-eval');

/** Resolved per call from the PROCESS's data dir — never a module constant. */
function memoryDir() { return path.join(getDataDir(), 'memory'); }
function opsDir() { return path.join(memoryDir(), 'ops'); }
function dailyDir() { return path.join(memoryDir(), 'daily'); }

/** Lazy requires — memory-manager owns the timer that calls back into here. */
function factExtractor() { return require('./fact-extractor'); }
// The ONE remaining use of the bell from here: a job that has disabled itself
// after repeated failures. That is not a result — it is the entity telling her
// something has stopped and needs her — so it belongs in the channel that can
// reach her. Results go to the jobs panel; see the note where deliver() was.
function initiatives() { return require('./initiatives'); }

function opsLog(msg) {
  try { factExtractor().appendToOpsLog(msg, opsDir()); } catch { /* console is the floor */ }
}
function dailyLog(msg) {
  try { factExtractor().appendToDailyLog(msg, dailyDir()); } catch { /* console is the floor */ }
}

/**
 * The tools a scheduled job may use.
 *
 * Read-only, and NOT configurable per job — that is deliberate for the first
 * version. Every one of these answers "what does the record say"; none of them
 * changes anything. The corrector's three write actions are absent, web tools
 * are absent, and write_memory is absent for the reason it is absent from every
 * background path: the general power to write an arbitrary fact stays where a
 * person is in the room. Widening this list is a decision, not a config knob.
 */
const JOB_TOOLS = [
  'memory_search', 'memory_list', 'memory_count', 'memory_get',
  'memory_corrections', 'memory_jobs'
];

/**
 * The run statuses that ARE A RESULT — a thing she can open and read.
 *
 * `partial` joined them on 2026-08-19 and that is why this is a constant rather
 * than a literal. The status was filtered by `IN ('ok','failed')` in six places
 * across two files — the panel feed, the announcement queue, the unread badge,
 * the times-run count, lastExecutedRun, and the tick's counters — and every one
 * of them would have silently dropped a partial run on the floor while the row
 * sat in the database looking fine. A seventh place will be added one day; it
 * should not be possible to add a status and forget one of them.
 *
 * `deferred` and `skipped` are deliberately NOT here. They are records of a run
 * that did not happen, not results, and putting them in a feed of results is
 * exactly the noise the panel exists to avoid.
 */
const RESULT_STATUSES = ['ok', 'partial', 'failed'];

/**
 * A config key that MOVED must not go quiet where it used to live.
 * scheduler.maxOutputTokens became generation.scheduledJobResponseTokens on
 * 2026-08-19. Same warning, same reasoning, as the agent path's.
 */
let warnedDeadOutputKey = false;
function warnDeadOutputKey(value) {
  if (warnedDeadOutputKey) return;
  warnedDeadOutputKey = true;
  const line =
    `scheduler.maxOutputTokens (${value}) in data/config.json is NO LONGER READ — ` +
    `the scheduled-run answer budget moved to generation.scheduledJobResponseTokens ` +
    `(Settings -> Thinking and Answer Budgets). Delete the old key; it is doing nothing.`;
  console.warn(`[Scheduler] ${line}`);
  opsLog(line);
}

function cfg() {
  const all = getConfig();
  const c = all.scheduler || {};
  const gen = all.generation || {};
  if (c.maxOutputTokens !== undefined) warnDeadOutputKey(c.maxOutputTokens);
  return {
    enabled: c.enabled !== false,
    tickSeconds: Math.max(10, c.tickSeconds ?? 60),
    catchupGraceMinutes: Math.max(0, c.catchupGraceMinutes ?? 120),
    maxConsecutiveFailures: Math.max(1, c.maxConsecutiveFailures ?? 3),
    maxToolCallsPerRun: Math.max(1, c.maxToolCallsPerRun ?? 12),
    maxWallClockMsPerRun: Math.max(5000, c.maxWallClockMsPerRun ?? 180000),
    maxRoundsPerRun: Math.max(1, c.maxRoundsPerRun ?? 6),
    // Both halves of the run's generation budget, from `generation` so they are
    // read against the chat and agent-job rows rather than apart from them.
    answerTokens: Math.max(64, gen.scheduledJobResponseTokens ?? 4096),
    thinkingTokens: Number.isFinite(gen.scheduledJobThinkingTokens) ? gen.scheduledJobThinkingTokens : null
  };
}

/**
 * The one run in flight, process-wide. Serial by construction: a second job that
 * comes due while this is set is deferred, never run alongside.
 */
let runningJobId = null;

// ---------------------------------------------------------------------------
// Arming — when is this job next due
// ---------------------------------------------------------------------------

/** A job the scheduler will actually consider: approved, enabled, kid or not. */
function isArmable(job) {
  return !!job && job.status === 'approved' && !!job.enabled;
}

/**
 * Compute and store the next firing.
 *
 * `from` defaults to NOW rather than to the missed time, which is what keeps a
 * restart from backfilling a week of 9am digests: the schedule is a statement
 * about the future, and the past is the run log's business.
 *
 * @returns {string|null} the stored ISO time, or null if the job is not armable
 *   or its expression is unreadable
 */
function armJob(jobId, { from = new Date(), reason = null } = {}) {
  const db = getSqliteDb();
  if (!db) return null;
  const job = db.prepare('SELECT * FROM cron_jobs WHERE id = ?').get(jobId);
  if (!isArmable(job)) {
    // Disarm rather than leave a stale time on a row nothing will run: a
    // next_run_at on a rejected or disabled job is a claim that it is coming.
    db.prepare('UPDATE cron_jobs SET next_run_at = NULL WHERE id = ?').run(jobId);
    return null;
  }

  const next = cronEval.nextRunAfter(job.schedule, from);
  if (!next) {
    // An expression that validated at propose time but cannot be evaluated (a
    // 30th of February) is a job that will never run. Say so once, loudly, and
    // leave it disarmed rather than retrying the same impossible sum every tick.
    const line = `Scheduled job ${jobId.slice(0, 8)} has a schedule nothing can satisfy ("${job.schedule}") — it is not armed and will not run. Its description: "${job.description}"`;
    console.warn(`[Scheduler] ${line}`);
    opsLog(line);
    db.prepare('UPDATE cron_jobs SET next_run_at = NULL WHERE id = ?').run(jobId);
    return null;
  }

  const iso = next.toISOString();
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(iso, jobId);
  console.log(`[Scheduler] armed ${jobId.slice(0, 8)} → ${next.toLocaleString()}${reason ? ` (${reason})` : ''}`);
  return iso;
}

/**
 * Arm every approved+enabled job that is not armed, and disarm every row that
 * is armed but should not be. Called at startup and after any decision.
 *
 * The disarm half matters as much as the arm half: a job Ellie reverted, or one
 * the failure counter disabled, must stop advertising a next run.
 */
function armAll({ reason = 'startup' } = {}) {
  const db = getSqliteDb();
  if (!db) return { armed: 0, disarmed: 0 };
  let armed = 0, disarmed = 0;
  for (const job of db.prepare('SELECT * FROM cron_jobs').all()) {
    if (isArmable(job)) {
      if (!job.next_run_at) { if (armJob(job.id, { reason })) armed++; }
    } else if (job.next_run_at) {
      db.prepare('UPDATE cron_jobs SET next_run_at = NULL WHERE id = ?').run(job.id);
      disarmed++;
    }
  }
  return { armed, disarmed };
}

// ---------------------------------------------------------------------------
// The run log
// ---------------------------------------------------------------------------

/**
 * Write a run row. Used directly for rows that are born terminal — a deferral or
 * a skipped catch-up never had a middle.
 */
function recordRun(row) {
  const db = getSqliteDb();
  if (!db) return null;
  const id = row.id || randomUUID();
  db.prepare(`
    INSERT INTO job_runs
      (id, job_id, scheduled_for, started_at, finished_at, status, duration_ms,
       trigger, output_initiative_id, output_text, tool_calls, budget_json, error)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id, row.jobId, row.scheduledFor || null, row.startedAt || null, row.finishedAt || null,
    row.status, row.durationMs ?? null, row.trigger || null,
    row.outputInitiativeId || null, row.outputText || null,
    row.toolCalls ?? 0, row.budget ? JSON.stringify(row.budget) : null, row.error || null
  );
  return id;
}

/**
 * Open a run row BEFORE the work starts.
 *
 * Not an implementation detail: a row written only on completion means an
 * interrupted run leaves no trace at all, so a job killed by a restart is
 * indistinguishable from a job that never came due. Writing it first makes the
 * in-flight state visible on disk, which is also what gives the re-entrancy
 * check and the startup sweep something real to read.
 *
 * 'running' is the one transient status. It never survives a process: either
 * finishRun replaces it, or the next startup sweeps it to 'skipped'.
 */
function recordRunStart({ runId, jobId, scheduledFor, startedAt, trigger }) {
  return recordRun({
    id: runId, jobId, scheduledFor, startedAt, finishedAt: null,
    status: 'running', trigger
  });
}

/** Close an open run row with what actually happened. */
function finishRun(runId, { status, finishedAt, durationMs, outputInitiativeId, outputText, toolCalls, budget, error }) {
  const db = getSqliteDb();
  if (!db) return;
  db.prepare(`
    UPDATE job_runs
    SET status = ?, finished_at = ?, duration_ms = ?, output_initiative_id = ?,
        output_text = ?, tool_calls = ?, budget_json = ?, error = ?
    WHERE id = ?
  `).run(
    status, finishedAt, durationMs ?? null, outputInitiativeId || null,
    outputText || null, toolCalls ?? 0, budget ? JSON.stringify(budget) : null,
    error || null, runId
  );
}

/** Runs for a job (or all jobs), newest first. */
function listRuns({ jobId = null, limit = 20 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  const lim = Math.min(Math.max(1, limit), 200);
  return jobId
    ? db.prepare('SELECT * FROM job_runs WHERE job_id = ? ORDER BY datetime(started_at) DESC LIMIT ?').all(jobId, lim)
    : db.prepare('SELECT * FROM job_runs ORDER BY datetime(started_at) DESC LIMIT ?').all(lim);
}

/** The newest run that actually executed (ok or failed) — deferrals are not runs. */
function lastExecutedRun(jobId) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare(
    `SELECT * FROM job_runs WHERE job_id = ? AND status IN (${RESULT_STATUSES.map(() => '?').join(',')})
     ORDER BY datetime(started_at) DESC LIMIT 1`
  ).get(jobId, ...RESULT_STATUSES) || null;
}

/** Is a previous run of this job still open? The re-entrancy check's disk half. */
function hasUnfinishedRun(jobId) {
  const db = getSqliteDb();
  if (!db) return false;
  const row = db.prepare(
    "SELECT id FROM job_runs WHERE job_id = ? AND status = 'running' LIMIT 1"
  ).get(jobId);
  return !!row;
}

/**
 * Close out runs that a restart interrupted.
 *
 * Without this an interrupted run blocks its job forever: the row stays open,
 * the re-entrancy check sees it, and the job never fires again — a silent stop,
 * which is the failure mode this whole module is written against. Recorded as
 * `skipped` rather than `failed` because nothing about the job went wrong, and
 * because a failure here would count toward auto-disabling it.
 */
function sweepInterruptedRuns() {
  const db = getSqliteDb();
  if (!db) return 0;
  const open = db.prepare("SELECT id, job_id, started_at FROM job_runs WHERE status = 'running'").all();
  if (!open.length) return 0;
  const now = new Date().toISOString();
  const upd = db.prepare(
    "UPDATE job_runs SET status = 'skipped', finished_at = ?, error = ? WHERE id = ?"
  );
  for (const r of open) {
    upd.run(now, 'interrupted by a restart — the run never finished, so nothing was reported', r.id);
    const line = `Scheduled job ${String(r.job_id).slice(0, 8)}: a run started ${r.started_at} was interrupted by a restart. Recorded as skipped; it was not retried.`;
    console.warn(`[Scheduler] ${line}`);
    opsLog(line);
  }
  return open.length;
}

// ---------------------------------------------------------------------------
// The executor — a job is an agent run
// ---------------------------------------------------------------------------

function systemPrompt(job, tools, lastRunAt) {
  return (
    `You are Aurelius, running one of your own scheduled jobs. Nobody is in the room: this is not a ` +
    `conversation, it is a background run, and what you write goes straight to Ellie's notification panel.\n\n` +
    `THE JOB, as you proposed it and she approved it:\n"${job.description}"\n` +
    `It runs on the schedule "${job.schedule}".\n\n` +
    `You have these read-only tools: ${tools.join(', ')}. Use them. This job is worth nothing if you answer ` +
    `from impression — everything you report must come from a tool result.\n\n` +
    // Both failure directions, weighted the same, because the first live run of
    // this job got the SECOND one: it called memory_corrections, was handed four
    // merges from seven hours earlier, and reported that nothing had happened.
    // The instruction at the time pushed hard against inventing activity and
    // said nothing about missing it, so "nothing happened" read as the safe
    // answer. It is not safe — it is the same lie with the sign flipped, and it
    // is the more dangerous one here, because a digest that always says "nothing
    // to report" is indistinguishable from a job that is working.
    `THERE ARE TWO WAYS TO GET THIS WRONG AND THEY ARE EQUALLY BAD:\n` +
    `1. Reporting something the tools did not show you. Never state a number, a date or an event you did not ` +
    `read from a tool result.\n` +
    `2. Reporting nothing when the tools DID show you something. Before you conclude that a record falls ` +
    `outside the period, compare its timestamp with the current date and time above — do the arithmetic, do ` +
    `not eyeball it. "Nothing happened" is the right answer ONLY when you have looked and the records inside ` +
    `the period are genuinely empty.\n` +
    `If a tool result says it is capped, partial, or showing only the most recent few, say so rather than ` +
    `treating what you were given as all there is.\n\n` +
    `Cover the period since ${lastRunAt ? `your last run of this job (${new Date(lastRunAt).toLocaleString()})` : 'roughly the last 24 hours (this is the first time this job has ever run)'}.\n\n` +
    `Write for Ellie, in your own voice: a few plain sentences. No headings, no bullet lists unless they are ` +
    `genuinely the clearest form, no preamble like "here is your digest" — she knows what she is reading. ` +
    `If there is nothing worth reporting, one line saying so is the correct output.`
  );
}

/**
 * Run one job, now.
 *
 * Every exit from this function writes exactly one job_runs row. That is the
 * invariant the whole module rests on: if it was attempted, there is a record,
 * whatever happened.
 *
 * @param {Object} job - the cron_jobs row
 * @param {Object} opts
 * @param {'schedule'|'catchup'|'manual'} opts.trigger
 * @param {string} [opts.scheduledFor] - the firing this run is for
 * @returns {Promise<Object>} the run record
 */
async function runJob(job, { trigger = 'schedule', scheduledFor = null } = {}) {
  const db = getSqliteDb();
  if (!db) return { status: 'failed', error: 'database unavailable' };

  // Re-entrancy, both halves. In-memory covers the same process; the run log
  // covers a run that outlived one.
  if (runningJobId) {
    return recordDeferral(job, scheduledFor, `another job (${runningJobId.slice(0, 8)}) was still running`);
  }
  if (hasUnfinishedRun(job.id)) {
    return recordDeferral(job, scheduledFor, 'its own previous run has not finished');
  }

  const runId = randomUUID();
  const startedAt = new Date();
  runningJobId = job.id;
  recordRunStart({
    runId, jobId: job.id, scheduledFor,
    startedAt: startedAt.toISOString(), trigger
  });

  const memoryManager = require('./memory-manager');
  const MCPClient = require('../mcp/mcp-client');
  const allowed = MCPClient.shared().backgroundToolsAmong(JOB_TOOLS);
  const denied = JOB_TOOLS.filter(t => !allowed.includes(t));
  if (denied.length) {
    // Loud: a job that asked for a tool it cannot have produces a thinner report
    // than it should, and that must not read as "there was nothing to find".
    const line = `Scheduled job ${job.id.slice(0, 8)} could not be given tool(s): ${denied.join(', ')}. It ran with ${allowed.length} of ${JOB_TOOLS.length}.`;
    console.warn(`[Scheduler] ${line}`);
    opsLog(line);
  }

  const c = cfg();
  const lastRun = lastExecutedRun(job.id);
  const session = memoryManager.createToolSession(`job:${job.id.slice(0, 8)}`, allowed, {
    maxCalls: c.maxToolCallsPerRun,
    maxWallMs: c.maxWallClockMsPerRun,
    maxRounds: c.maxRoundsPerRun
  });

  console.log(`[Scheduler] === running job ${job.id.slice(0, 8)} (${trigger}): "${job.description}" ===`);

  let status = 'ok', error = null, output = '', budget = null, toolCalls = 0;
  try {
    const agentPool = require('./agent-pool');
    const res = await agentPool.schedule(
      () => memoryManager.callLLM(
        systemPrompt(job, allowed, lastRun && lastRun.started_at),
        job.description,
        { maxTokens: c.answerTokens, thinkingTokens: c.thinkingTokens, toolSession: session }
      ),
      `scheduled-job:${job.id.slice(0, 8)}`,
      'scheduled'
    );
    output = String(res && res.content || '').trim();
    budget = (res && res.budget) || session.summary();
    toolCalls = Array.isArray(res && res.toolCalls) ? res.toolCalls.length : 0;

    // A RUN THAT HIT ITS CEILING IS NOT A RUN THAT FINISHED. Same defect the
    // agent path had until 2026-08-19 and the same fix: callLLM has always
    // returned `truncated` off finish_reason 'length', and this function closed
    // every non-empty run as `ok` without ever looking at it. A cron digest cut
    // off mid-sentence arrived on the panel indistinguishable from a whole one,
    // every morning, for as long as the ceiling was too low.
    //
    // `partial` keeps the text — it is real work — and names the limit, because
    // "stopped early" is not something anyone can act on and "the answer budget
    // is 4096" is.
    if (output && (res.truncated || res.outOfRounds || (budget && budget.exhausted))) {
      status = 'partial';
      error = res.truncated
        ? `it hit the answer budget (${c.answerTokens} tokens) and stopped mid-result — what is above is cut off, not finished. Raise "Answer budget, scheduled jobs" in Settings if this keeps happening`
        : res.outOfRounds
          ? `it ran out of tool rounds (${c.maxRoundsPerRun}) before it was finished — what is above is what it had`
          : `it stopped early: ${budget.exhausted} — what is above is what it had`;
    }

    if (!output) {
      // An empty answer is a failure with a specific cause, not a quiet success.
      // Reported as one so the failure counter can eventually stop it.
      status = 'failed';
      error = toolCalls
        ? `the model made ${toolCalls} tool call(s) but returned no text`
        : 'the model returned no text';
    }
  } catch (err) {
    status = 'failed';
    error = err.message || String(err);
    budget = session.summary();
    console.error(`[Scheduler] job ${job.id.slice(0, 8)} failed:`, error);
  } finally {
    runningJobId = null;
  }

  const finishedAt = new Date();
  const durationMs = finishedAt.getTime() - startedAt.getTime();

  // DELIVERY IS NOW THE RUN ROW ITSELF (2026-08-18). See the note on the removed
  // deliver() below: the result goes to the jobs panel, which reads job_runs
  // directly, so writing the row IS delivering it. There is no second write that
  // can fail halfway and no second copy of the text to disagree with this one.
  const initiativeId = null;
  if (status === 'ok') {
    dailyLog(`My scheduled job ran — "${job.description}" — and the result is in Ellie's jobs panel: ${output}`);
  } else if (status === 'partial') {
    dailyLog(`My scheduled job ran — "${job.description}" — and stopped before it finished (${error}). What it did write is in Ellie's jobs panel: ${output}`);
  }

  finishRun(runId, {
    status, finishedAt: finishedAt.toISOString(), durationMs,
    outputInitiativeId: initiativeId, outputText: output || null,
    toolCalls, budget, error
  });

  applyOutcome(job, { status, error, at: finishedAt, durationMs, trigger, toolCalls });

  return { runId, status, error, output, initiativeId, durationMs, toolCalls, budget };
}

/**
 * A due job that could not start. Recorded once per due-time, not once per tick:
 * a job that waits twenty minutes behind a long-running one should leave one
 * line saying it waited, not twenty identical ones. Same reasoning as the
 * heartbeat's anomaly memo — a repeated unchanged condition is wallpaper.
 */
function recordDeferral(job, scheduledFor, why) {
  const db = getSqliteDb();
  if (db && scheduledFor) {
    const already = db.prepare(
      "SELECT id FROM job_runs WHERE job_id = ? AND status = 'deferred' AND scheduled_for = ? LIMIT 1"
    ).get(job.id, scheduledFor);
    if (already) return { status: 'deferred', error: why, runId: already.id };
  }
  const now = new Date().toISOString();
  const runId = recordRun({
    jobId: job.id, scheduledFor, startedAt: now, finishedAt: now,
    status: 'deferred', durationMs: 0, trigger: 'schedule',
    error: `deferred to the next tick — ${why}`
  });
  const line = `Scheduled job ${job.id.slice(0, 8)} came due at ${scheduledFor} but ${why}. Deferred to the next tick.`;
  console.log(`[Scheduler] ${line}`);
  opsLog(line);
  return { status: 'deferred', error: why, runId };
}

/**
 * WHERE A SCHEDULED RESULT GOES — and why it stopped going to the bell.
 *
 * There used to be a deliver() here that wrote the run's output into the
 * initiatives table as a `job-result` row. It was exempted, by name, from every
 * piece of the initiative pool's machinery — dedup, re-scoring, the stale sweep,
 * the cap — because none of that machinery makes sense for a record of something
 * that already happened. A queue whose every rule has to be switched off for one
 * of its types is telling you that type belongs somewhere else.
 *
 * It does now. Results go to the JOBS panel (db/agent-jobs.js `feed`), which
 * reads job_runs directly, and the channels are separated at the table:
 *
 *   ROBOT (jobs panel)  = results. NEVER opens a conversation.
 *   BELL (initiatives)  = things the entity wants to SAY. Can still open one.
 *
 * That separation is the point rather than a tidy-up. A job result arriving in
 * the queue that can start a conversation meant the most mechanical trigger in
 * the system — a timer fired and a job finished — sat in the same channel as
 * things he had decided were worth raising. If a finding IS worth saying
 * something about, he raises an ordinary initiative about it in an ordinary
 * turn, subject to the same judgement as anything else.
 *
 * Nothing replaced deliver(), because nothing needed to: the run row already
 * holds output_text, so writing it is delivering it. The one thing lost with the
 * bell item was a second copy of the text, which is not a loss.
 *
 * (`output_initiative_id` stays on job_runs, always null for new runs. The eight
 * rows written before this change still point at their bell items, and those
 * items still exist — relocated out of the pending pool, kept as history. See
 * scripts/migrate-job-results.js.)
 */

/**
 * Record the outcome on the job row, and stop a job that keeps failing.
 *
 * The failure counter is the thing that keeps this honest over weeks. A daily
 * job whose brain is down does not deserve to retry forever in a log nobody
 * reads; after scheduler.maxConsecutiveFailures it disables itself, disarms, and
 * raises an alert that NAMES THE ERROR — the disable is useless without the
 * reason, which is what would send someone reading server logs for an hour.
 */
function applyOutcome(job, { status, error, at, durationMs, trigger, toolCalls }) {
  const db = getSqliteDb();
  if (!db) return;

  // `partial` IS ARMED LIKE A SUCCESS, and getting this wrong would have been
  // the worst bug in the change. The branch below returns without arming for
  // anything that is not `ok` or `failed` — correct for deferred and skipped,
  // which are records of a run that did not happen — so a `partial` falling
  // through it would have left next_run_at NULL and the job would simply never
  // run again. No error, no disabled_reason, no alert: a cron job that quietly
  // stops, which is the exact class of silent loss this subsystem is written
  // against.
  //
  // consecutive_failures is RESET, not incremented. A run that produced a
  // truncated result did its work and hit a ceiling; that is a budget to raise,
  // not an engine that is failing, and letting it accumulate toward
  // maxConsecutiveFailures would disable a job for the crime of having more to
  // say than its budget allowed — throwing away results she can actually read.
  // The card and the ops line carry the signal instead.
  if (status === 'ok' || status === 'partial') {
    db.prepare(`
      UPDATE cron_jobs
      SET last_run_at = ?, last_status = ?, run_count = COALESCE(run_count, 0) + 1,
          consecutive_failures = 0
      WHERE id = ?
    `).run(at.toISOString(), status, job.id);
    armJob(job.id, {
      from: at,
      reason: status === 'ok' ? 'after a successful run' : 'after a run that stopped short'
    });
    opsLog(status === 'ok'
      ? `Scheduled job ran: "${job.description}" (${job.id.slice(0, 8)}, ${trigger}) — ok in ${(durationMs / 1000).toFixed(1)}s, ${toolCalls} tool call(s).`
      : `Scheduled job ran PARTIAL: "${job.description}" (${job.id.slice(0, 8)}, ${trigger}) — ${(durationMs / 1000).toFixed(1)}s, ${toolCalls} tool call(s). It wrote up what it had. Why it stopped: ${error}`);
    return;
  }

  if (status !== 'failed') return;   // deferred/skipped touch nothing on the job

  const fails = (db.prepare('SELECT consecutive_failures FROM cron_jobs WHERE id = ?').get(job.id)?.consecutive_failures || 0) + 1;
  db.prepare(`
    UPDATE cron_jobs
    SET last_run_at = ?, last_status = 'failed', run_count = COALESCE(run_count, 0) + 1,
        consecutive_failures = ?
    WHERE id = ?
  `).run(at.toISOString(), fails, job.id);

  const max = cfg().maxConsecutiveFailures;
  if (fails < max) {
    const line = `Scheduled job FAILED: "${job.description}" (${job.id.slice(0, 8)}) — ${error}. That is ${fails} in a row; it disables itself at ${max}.`;
    console.warn(`[Scheduler] ${line}`);
    opsLog(line);
    armJob(job.id, { from: at, reason: `after failure ${fails}/${max}` });
    return;
  }

  const reason = `disabled automatically after ${fails} consecutive failures — last error: ${error}`;
  db.prepare("UPDATE cron_jobs SET enabled = 0, next_run_at = NULL, disabled_reason = ? WHERE id = ?")
    .run(reason, job.id);
  const line = `Scheduled job DISABLED: "${job.description}" (${job.id.slice(0, 8)}) — ${reason}`;
  console.error(`[Scheduler] ${line}`);
  opsLog(line);
  dailyLog(`One of my scheduled jobs stopped itself: "${job.description}". It failed ${fails} times in a row and the last error was: ${error}. It will not run again until Ellie re-enables it.`);
  initiatives().addInitiative({
    type: 'alert',
    content:
      `A scheduled job of mine has disabled itself: "${job.description}" (${job.schedule}). ` +
      `It failed ${fails} times in a row and the last error was: ${error}. ` +
      `It will not run again until it is re-enabled.`,
    sourceKind: 'scheduled-job-disabled',
    sourceRef: job.id,
    priority: 8
  }).catch(err => console.error('[Scheduler] could not raise disable alert:', err.message));
}

// ---------------------------------------------------------------------------
// The tick
// ---------------------------------------------------------------------------

/** Approved, enabled, armed, and due. */
function dueJobs(now = new Date()) {
  const db = getSqliteDb();
  if (!db) return [];
  return db.prepare(`
    SELECT * FROM cron_jobs
    WHERE status = 'approved' AND enabled = 1 AND next_run_at IS NOT NULL
      AND datetime(next_run_at) <= datetime(?)
    ORDER BY datetime(next_run_at) ASC
  `).all(now.toISOString());
}

/**
 * One pass of the scheduler. Called every scheduler.tickSeconds.
 *
 * Serial: if a run is in flight, every due job is deferred and the tick returns.
 * Otherwise due jobs run one after another, awaited, in due-time order.
 */
async function tick({ now = new Date() } = {}) {
  const c = cfg();
  if (!c.enabled) return { skipped: 'scheduler disabled' };

  const due = dueJobs(now);
  if (!due.length) return { due: 0, ran: 0 };

  if (runningJobId) {
    for (const job of due) recordDeferral(job, job.next_run_at, `another job (${runningJobId.slice(0, 8)}) was still running`);
    return { due: due.length, ran: 0, deferred: due.length };
  }

  const out = { due: due.length, ran: 0, skipped: 0, failed: 0, deferred: 0, partial: 0 };
  for (const job of due) {
    const scheduledFor = job.next_run_at;
    const lateMs = now.getTime() - new Date(scheduledFor).getTime();
    const graceMs = c.catchupGraceMinutes * 60_000;

    // Too late to be worth running. Recorded, re-armed forward, not run — and
    // never accumulated: one missed firing, one skipped row, one re-arm.
    if (lateMs > graceMs) {
      const lateMin = Math.round(lateMs / 60_000);
      recordRun({
        jobId: job.id, scheduledFor,
        startedAt: now.toISOString(), finishedAt: now.toISOString(),
        status: 'skipped', durationMs: 0, trigger: 'catchup',
        error: `missed by ${lateMin} min, past the ${c.catchupGraceMinutes} min catch-up window — not run, and not backfilled`
      });
      const line = `Scheduled job "${job.description}" (${job.id.slice(0, 8)}) missed its ${new Date(scheduledFor).toLocaleString()} run by ${lateMin} min — past the ${c.catchupGraceMinutes} min catch-up window, so it was skipped rather than run late.`;
      console.warn(`[Scheduler] ${line}`);
      opsLog(line);
      armJob(job.id, { from: now, reason: 'after a skipped catch-up' });
      out.skipped++;
      continue;
    }

    // Late but inside the window: run it once, marked as the catch-up it is.
    const trigger = lateMs > c.tickSeconds * 1000 * 2 ? 'catchup' : 'schedule';
    const res = await runJob(job, { trigger, scheduledFor });
    if (res.status === 'ok') out.ran++;
    else if (res.status === 'partial') { out.ran++; out.partial++; }
    else if (res.status === 'failed') out.failed++;
    else if (res.status === 'deferred') out.deferred++;
  }
  return out;
}

/**
 * Force a run now, outside the schedule. The deliberate path: a person asked.
 *
 * Does not consume the pending firing. Re-arming computes from NOW, the same as
 * any other run, so forcing one at 08:33 re-arms to the 09:00 that was already
 * pending and it still fires. (Forcing one DURING the pending minute is the one
 * exception — the re-arm then lands on tomorrow, which is correct: it just ran.)
 *
 * Marked `manual` in the run log, so a forced run is never mistaken for evidence
 * that the timer works. That distinction is the whole reason the column exists.
 */
async function runNow(jobIdOrPrefix) {
  const db = getSqliteDb();
  if (!db) return { error: 'database unavailable' };
  const wanted = String(jobIdOrPrefix || '').trim();
  const job = db.prepare('SELECT * FROM cron_jobs WHERE id = ? OR id LIKE ?').get(wanted, `${wanted}%`);
  if (!job) return { error: `no job matching "${wanted}"` };
  if (job.status !== 'approved') return { error: `job is ${job.status}, not approved` };
  return runJob(job, { trigger: 'manual', scheduledFor: job.next_run_at || null });
}

/**
 * Everything the read surfaces need about one job's execution state.
 * One function so the tool, the API and the panel cannot disagree.
 */
function runtimeState(jobId) {
  const db = getSqliteDb();
  if (!db) return null;
  const job = db.prepare('SELECT * FROM cron_jobs WHERE id = ?').get(jobId);
  if (!job) return null;
  const last = lastExecutedRun(jobId);
  const runs = db.prepare(
    `SELECT COUNT(*) n FROM job_runs WHERE job_id = ? AND status IN (${RESULT_STATUSES.map(() => '?').join(',')})`
  ).get(jobId, ...RESULT_STATUSES).n;
  return {
    armed: !!job.next_run_at,
    nextRunAt: job.next_run_at || null,
    lastRunAt: job.last_run_at || null,
    lastStatus: job.last_status || null,
    // A partial run's reason is the thing worth surfacing here — it names a
    // budget she can change. Reporting only `failed` hid it.
    lastError: last && (last.status === 'failed' || last.status === 'partial') ? last.error : null,
    lastRunId: last ? last.id : null,
    timesRun: runs,
    consecutiveFailures: job.consecutive_failures || 0,
    disabledReason: job.disabled_reason || null,
    enabled: !!job.enabled,
    status: job.status
  };
}

/** Is the scheduler actually going to run anything? For the honest surfaces. */
function schedulerState() {
  const c = cfg();
  const db = getSqliteDb();
  const armedCount = db
    ? db.prepare("SELECT COUNT(*) n FROM cron_jobs WHERE status='approved' AND enabled=1 AND next_run_at IS NOT NULL").get().n
    : 0;
  return {
    enabled: c.enabled,
    running: !!runningJobId,
    runningJobId,
    tickSeconds: c.tickSeconds,
    catchupGraceMinutes: c.catchupGraceMinutes,
    maxConsecutiveFailures: c.maxConsecutiveFailures,
    armedJobs: armedCount,
    tools: JOB_TOOLS
  };
}

module.exports = {
  JOB_TOOLS,
  RESULT_STATUSES,
  tick,
  runJob,
  runNow,
  armJob,
  armAll,
  dueJobs,
  listRuns,
  lastExecutedRun,
  sweepInterruptedRuns,
  runtimeState,
  schedulerState
};
