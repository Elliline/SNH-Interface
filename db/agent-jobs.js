/**
 * The agent-job queue — the async handoff.
 *
 * WHY THIS EXISTS. Every tool the entity had blocked the turn. He could search,
 * or look through memory, or fetch a page, but only inside the seconds Ellie was
 * sitting there waiting for a reply — so the unit of work was the turn, and
 * anything bigger than a turn could not be started at all. This is the piece
 * that lets a chat turn START work and END: the tool call writes a row, returns
 * a job id, and the conversation carries on. "On it, I'll come back to you"
 * becomes a true sentence.
 *
 * ⚠ ROBOT, NOT BELL. This is the rule the whole design is bent around.
 *
 *   ROBOT (this queue) = RESULTS. A record of work that ran. It NEVER opens a
 *                        conversation. Ellie reads the jobs panel when she is
 *                        ready, and that is the only place a result goes.
 *   BELL (initiatives) = things the entity WANTS TO SAY, which may still open a
 *                        conversation, exactly as before.
 *
 * Nothing in this file requires db/initiatives.js. Not "must not" — DOES NOT,
 * and that absence is the enforcement. A job result can LEAD TO a conversation
 * by him deciding, in an ordinary turn, that a finding is worth raising, subject
 * to the same judgement as anything else he might say. Job completion is the
 * most mechanical trigger there is; a channel that let it speak on its own would
 * be a channel that routes around that judgement.
 *
 * A JOB IS AN AGENT RUN, the same one thing a scheduled job is: the task prose
 * becomes the prompt for one background model call with a READ-ONLY tool
 * allowlist. No shell, no code execution, no side effects, and no per-job
 * escape hatch to add one. `start_background_job` is deliberately NOT on the
 * allowlist, so a job cannot start a job.
 *
 * OUTSIDE THE REQUEST PATH. enqueue() writes the row synchronously and returns;
 * the run is handed to the agent pool and never awaited by the caller. The chat
 * response finishing, the stream closing, and the browser closing are all
 * unrelated events — nothing about a job runs client-side, and the promise is
 * owned by this module rather than by the request.
 *
 * WHAT A RESTART DOES, said plainly: it kills the run. An LLM call cannot be
 * resumed — but the transcript up to the last completed step can be, and since
 * 2026-09-10 it is written to disk as the job runs (see "The record on disk"
 * below). So the loss is made loud instead of silent — sweepInterrupted()
 * closes every in-flight row as `interrupted` WITH THE REASON and what it had,
 * or re-queues it once from its checkpoint if it is still young enough to be
 * worth finishing. A job never vanishes; that is the whole reason the row is
 * written before the work starts.
 *
 * WHY A JOB STOPPED IS SAID IN WORDS, WITH THE SIDE IT CAME FROM. The 2026-09-09
 * card said "terminated". See db/job-failure.js — every stop carries
 * stop_source (runner | engine | dispatched | user | unknown) and stop_kind, and
 * `error` is the sentence she reads.
 *
 * AND A JOB NEAR ITS BUDGET ASKS. The one time a job speaks, in
 * db/job-budget-ask.js: near a ceiling with work left it pauses, asks her in the
 * conversation that dispatched it, and waits. Not a result, so not a breach of
 * ROBOT-NOT-BELL: a decision she has to make, made where she makes them.
 *
 * CHAT IS STILL KING. Runs go through the agent pool, so they inherit its
 * throttle to concurrency 1 while a chat request is in flight. That gates
 * LAUNCH, never preemption, and the throttled width is 1 rather than 0 — a job
 * started mid-conversation starts immediately and keeps running while she types;
 * what waits is the second one.
 */

const { randomUUID } = require('crypto');
const path = require('path');
const { getSqliteDb, getDataDir } = require('./database');
const agentPool = require('./agent-pool');
const { formatLocalTime } = require('./datetime');

// Read through the module object rather than destructuring at load time, so the
// config seen here is always the one the process currently holds — and so a test
// can substitute one without writing to the live data/config.json, which is
// deliberately NOT redirected by SNH_DATA_DIR.
function getConfig() { return require('./config').getConfig(); }

/** Resolved per call from the PROCESS's data dir — never a module constant. */
function memoryDir() { return path.join(getDataDir(), 'memory'); }
function opsDir() { return path.join(memoryDir(), 'ops'); }

/** Lazy requires — these modules load us back. */
function factExtractor() { return require('./fact-extractor'); }
function memoryManager() { return require('./memory-manager'); }
/**
 * The scheduler owns job_runs and therefore owns which of its statuses count as
 * a result. Lazily required and read at call time rather than copied, because a
 * copy is how `partial` came to be missing from six filters at once.
 */
function runResultStatuses() { return require('./scheduler').RESULT_STATUSES; }

function opsLog(msg) {
  try { factExtractor().appendToOpsLog(msg, opsDir()); } catch { /* console is the floor */ }
}

/**
 * The tools a handed-off job may use.
 *
 * Read-only, and NOT configurable per job — same call as the scheduler's, for
 * the same reason. Every one of these answers "what does the record say" or
 * "what does the world say"; none of them changes anything. The corrector's
 * three write actions are absent, write_memory is absent (the general power to
 * write an arbitrary fact stays where a person is in the room), and
 * start_background_job is absent so a job cannot start a job. Widening this list
 * is a decision, not a config knob.
 *
 * web_search/web_fetch are intersected away automatically on a box where
 * config.tools.searxng.enabled is false — the registry simply does not hold
 * them, and backgroundToolsAmong drops what is not registered.
 */
const JOB_TOOLS = [
  'memory_search', 'memory_list', 'memory_count', 'memory_get',
  'memory_corrections', 'memory_jobs',
  'web_search', 'web_fetch'
];

/**
 * Statuses a job cannot leave.
 *
 * `partial` was added 2026-08-18 and it is the honest third answer. Before it
 * there were two: `ok`, which claims the job did what it set out to do, and
 * `failed`, which reads as "nothing came of this". A run that spent its budget on
 * dead searches and then wrote up the memory work it HAD finished is neither —
 * calling it ok over-claims, calling it failed throws away a real result, and
 * throwing away a real result is exactly what happened.
 *
 * It is terminal like the others, counted in the badge like the others, announced
 * like the others. The only thing that differs is what the panel says on the card.
 */
const TERMINAL = ['ok', 'partial', 'failed', 'interrupted', 'cancelled'];

/**
 * Sources whose rows are a RECORD, not a result for Ellie.
 *
 * ROBOT, NOT BELL is the rule this file is bent around, and this is the line
 * underneath it: the panel is for work the entity handed off and walked away
 * from, which she reads when she is ready. A history_search run is not that. It
 * happens inside a turn, its answer goes straight back into the conversation,
 * and by the time she could see a card the entity has already told her what it
 * found. A card per lookup would be up to forty a day of "here is something you
 * have already read", which is how a panel stops being worth opening.
 *
 * SO THE ROW STILL EXISTS — every one of them, with its digest, sweepable on
 * restart and queryable forever. What it does not do is claim her attention.
 * The lookup's Ellie-facing record is its tool_call_log entry, in the Thinking
 * tab, which is where tool calls belong.
 */
const IN_TURN_SOURCES = ['history-search'];
const inTurnSql = (col = 'source') =>
  `(${col} IS NULL OR ${col} NOT IN (${IN_TURN_SOURCES.map(x => `'${x}'`).join(',')}))`;

/**
 * A config key that MOVED must not go quiet where it used to live.
 *
 * agentJobs.maxOutputTokens became generation.agentJobResponseTokens on
 * 2026-08-19. A box that set the old one in its data/config.json would otherwise
 * keep it there, unread, looking exactly like the thing setting the job's answer
 * budget while something else set it — which is the two-sources-of-truth defect
 * with the sources a screen apart. Said once per process, not once per job.
 */
let warnedDeadConcurrencyKey = false;
function warnDeadConcurrencyKey(value) {
  if (warnedDeadConcurrencyKey) return;
  warnedDeadConcurrencyKey = true;
  const line =
    `agentJobs.maxConcurrent (${value}) in data/config.json is NO LONGER READ — how many jobs run at ` +
    `once is agentPool.lanes.agentJobs now (Settings -> Background lanes), so it cannot disagree with ` +
    `the lane that schedules them. Delete the old key; it is doing nothing.`;
  console.warn(`[AgentJobs] ${line}`);
  opsLog(line);
}

let warnedDeadOutputKey = false;
function warnDeadOutputKey(value) {
  if (warnedDeadOutputKey) return;
  warnedDeadOutputKey = true;
  const line =
    `agentJobs.maxOutputTokens (${value}) in data/config.json is NO LONGER READ — ` +
    `the job answer budget moved to generation.agentJobResponseTokens ` +
    `(Settings -> Thinking and Answer Budgets). Delete the old key; it is doing nothing.`;
  console.warn(`[AgentJobs] ${line}`);
  opsLog(line);
}

function cfg() {
  const all = getConfig();
  const c = all.agentJobs || {};
  const gen = all.generation || {};
  if (c.maxOutputTokens !== undefined) warnDeadOutputKey(c.maxOutputTokens);
  if (c.maxConcurrent !== undefined) warnDeadConcurrencyKey(c.maxConcurrent);
  return {
    enabled: c.enabled !== false,
    // ONE NUMBER, IN THE LANE. agentJobs.maxConcurrent was a second cap on the
    // same quantity — it gated here, before the pool ever saw the job, so the
    // lower of the two always won and the lane cap was decoration. It is read
    // from the pool now and enforced here, which keeps the surplus visible as
    // `queued` rows in her panel rather than hidden inside the pool's own queue.
    maxConcurrent: agentPool.laneCap('agentJobs'),
    maxQueued: Math.max(1, c.maxQueued ?? 10),
    maxStartsPerHour: Math.max(1, c.maxStartsPerHour ?? 6),
    maxToolCallsPerJob: Math.max(1, c.maxToolCallsPerJob ?? 12),
    maxWallClockMs: Math.max(5000, c.maxWallClockMs ?? 300000),
    maxRoundsPerJob: Math.max(1, c.maxRoundsPerJob ?? 6),
    // The two halves of the job's generation budget, both from `generation` so
    // they are read against the chat and background rows rather than apart from
    // them. The answer budget is always sent (it always has been); the thinking
    // budget is null-means-send-nothing like every other field in that section.
    answerTokens: Math.max(64, gen.agentJobResponseTokens ?? 8192),
    thinkingTokens: Number.isFinite(gen.agentJobThinkingTokens) ? gen.agentJobThinkingTokens : null,
    // Starts allowed per job, counting the first. Was the literal `< 2`.
    maxAttempts: Math.max(1, c.maxAttempts ?? 2),
    retryGraceMinutes: Math.max(0, c.retryGraceMinutes ?? 30),
    retentionDays: Math.max(1, c.retentionDays ?? 90),
    // The budget ask — see db/config.js. `askAtPercent` outside (0, 100) means
    // never ask, and so does the switch being off.
    askBeforeCeiling: c.askBeforeCeiling !== false,
    askAtPercent: Number.isFinite(c.askAtPercent) ? c.askAtPercent : 80,
    extensionPercent: Math.max(1, Number.isFinite(c.extensionPercent) ? c.extensionPercent : 50)
  };
}

// ---------------------------------------------------------------------------
// The record on disk — written as the job runs, not only at the end
// ---------------------------------------------------------------------------

/**
 * WHERE A RUN'S TRANSCRIPT LIVES WHILE IT RUNS: <data>/jobs/<id>.json.
 *
 * A crash cannot write anything after the fact. On 2026-09-09 a job made 27
 * tool calls across eight rounds, the engine was restarted underneath it, and
 * every one of those results went with the process — the card said "I have no
 * record of what it managed to look up first", which was true of the code and
 * false of the run. So the tool loop hands its transcript here after every tool
 * result and every round, and the three things that read it are:
 *
 *   - the failure path, so a killed run's card offers what it had (partial);
 *   - the pause, so a job waiting on her answer holds its place on disk, not
 *     in a lane;
 *   - the resume, so a yes (or a restart) continues from the last completed
 *     step rather than from zero.
 *
 * Kept after a non-ok finish until the row is pruned, because a retry's brief
 * reads it. Dropped on `ok`: a finished job's record is its result.
 */
function jobsDir() { return path.join(getDataDir(), 'jobs'); }
function checkpointPath(id) { return path.join(jobsDir(), `${id}.json`); }

function writeCheckpoint(id, state) {
  const fs = require('fs');
  try {
    fs.mkdirSync(jobsDir(), { recursive: true });
    const tmp = `${checkpointPath(id)}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify({ jobId: id, updatedAt: new Date().toISOString(), ...state }));
    fs.renameSync(tmp, checkpointPath(id));
    return true;
  } catch (err) {
    console.warn(`[AgentJobs] ${String(id).slice(0, 8)} could not write its checkpoint: ${err.message}`);
    return false;
  }
}

function readCheckpoint(id) {
  const fs = require('fs');
  try { return JSON.parse(fs.readFileSync(checkpointPath(id), 'utf8')); }
  catch { return null; }
}

function dropCheckpoint(id) {
  const fs = require('fs');
  try { fs.unlinkSync(checkpointPath(id)); } catch { /* none, or already gone */ }
}

/**
 * A transcript is only resumable from a COMPLETE round: an assistant turn that
 * asked for tools must be followed by every one of their results, or the next
 * request is malformed. A restart mid-round leaves a tail that is not — this
 * cuts it back to the last complete step and reports how much was dropped.
 */
function trimToCompleteRound(convo = [], toolCalls = []) {
  const msgs = [...convo];
  let dropped = 0;
  // Only an INCOMPLETE tail goes: trailing tool results whose assistant turn
  // asked for more than came back, or an assistant turn that asked and got
  // nothing. A round whose every call answered is complete and stays.
  let k = 0;
  while (k < msgs.length && msgs[msgs.length - 1 - k].role === 'tool') k++;
  const head = msgs[msgs.length - 1 - k];
  if (head && head.role === 'assistant' && Array.isArray(head.tool_calls) && head.tool_calls.length > k) {
    dropped = k + 1;
    msgs.splice(msgs.length - dropped, dropped);
  }
  // Which rounds survived: the tool records carry their round number, and a
  // record whose results were cut must not be carried as if they were there.
  let lastRound = 0;
  for (const m of msgs) if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) lastRound++;
  const calls = toolCalls.filter(k => !Number.isFinite(k.round) || k.round <= lastRound);
  return { convo: msgs, toolCalls: calls, dropped, lastRound };
}

function parseAsk(json) {
  try { return json ? JSON.parse(json) : null; } catch { return null; }
}

/**
 * Jobs this PROCESS currently has on the pool.
 *
 * The in-memory half of the re-entrancy check; the `running` status in the table
 * is the disk half. Both are needed for the same reason the scheduler needs
 * both: this set dies with the process, and a row that outlived one would
 * otherwise be launched a second time by the startup resume.
 */
const inFlight = new Set();

// ---------------------------------------------------------------------------
// Writing the record
// ---------------------------------------------------------------------------

function getJob(id) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id) || null;
}

/** Jobs queued or running right now, from the table (not from memory). */
function activeCount() {
  const db = getSqliteDb();
  if (!db) return 0;
  return db.prepare("SELECT COUNT(*) n FROM agent_jobs WHERE status IN ('queued','running')").get().n;
}

/** Starts in the trailing hour, counted from the table so a restart grants no fresh budget. */
function startsLastHour() {
  const db = getSqliteDb();
  if (!db) return 0;
  const since = new Date(Date.now() - 60 * 60 * 1000).toISOString();
  return db.prepare(
    `SELECT COUNT(*) n FROM agent_jobs WHERE datetime(created_at) > datetime(?) AND ${inTurnSql()}`
  ).get(since).n;
}

/**
 * Every call to start a job leaves a row in tool_call_log — started, refused or
 * errored.
 *
 * Added 2026-08-18, after an hour was spent establishing whether the tool had
 * been called at all. It had not (the turn never reached the tool loop), but
 * nothing in the data could say so: this tool logged nothing, ever, so "no entry"
 * meant "never called" and "called and refused" identically. create_cron_job has
 * logged its calls since the day it shipped; this is the same courtesy.
 */
function logToolCall({ outcome, detail, refId = null, conversationId = null, args = null }) {
  const db = getSqliteDb();
  if (!db) return;
  try {
    db.prepare(`
      INSERT INTO tool_call_log (id, created_at, tool, args_json, outcome, detail, ref_id, conversation_id)
      VALUES (?, ?, 'start_background_job', ?, ?, ?, ?, ?)
    `).run(randomUUID(), new Date().toISOString(), args ? JSON.stringify(args) : null,
      outcome, String(detail || '').slice(0, 300), refId, conversationId);
  } catch (err) {
    console.error('[AgentJobs] logToolCall failed:', err.message);
  }
}

/** A refusal, logged and returned in one move so neither can be forgotten. */
function refuse(outcome, error, { conversationId = null } = {}) {
  logToolCall({ outcome, detail: error, conversationId });
  return { ok: false, error };
}

/**
 * Put a job on the queue and start it.
 *
 * Returns synchronously-decided state: by the time this resolves the row exists
 * and the run has been handed to the pool, so "I started it" is checkable the
 * instant he says it. It does NOT wait for the run.
 *
 * A refusal is returned as a reason the model can read and repeat. That matters
 * more than it looks: a silent refusal here is the phantom-action bug — he says
 * he started something and nothing exists.
 *
 * @returns {{ok: true, id: string} | {ok: false, error: string}}
 */
function enqueue({ title, task, why = null, conversationId = null, messageId = null, source = 'chat-handoff',
                   countsAgainstStarts = true, retryOf = null } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'The job queue is unavailable (no database handle).' };

  const c = cfg();
  if (!c.enabled) return refuse('error', 'Background jobs are switched off in configuration, so nothing was started.', { conversationId });

  const t = String(title || '').trim();
  const k = String(task || '').trim();
  if (!t) return refuse('error', 'A job needs a short title — nothing was started.', { conversationId });
  if (!k) return refuse('error', 'A job needs a task describing what to do — nothing was started.', { conversationId });
  if (k.length > 4000) return refuse('error', 'That task is too long to hand off (4000 characters max) — nothing was started.', { conversationId });

  const active = activeCount();
  if (active >= c.maxQueued) {
    return refuse('refused-cap', `There are already ${active} jobs queued or running, which is the limit (${c.maxQueued}). Nothing was started — say so, and offer to do this once some of them finish.`, { conversationId });
  }
  // THE START BUDGET IS FOR HANDOFFS, and `countsAgainstStarts: false` is how a
  // caller says it is not one. It exists for exactly one case today
  // (history_search) and the case is not a loophole: that tool is a READ inside
  // a turn, it is already charged to the shared read budget the memory-inspect
  // tools spend, and charging it here as well would put two counters on one
  // action — which is how they come to disagree, and how the tighter one
  // silently becomes the only one that matters. The queue-depth cap above is
  // NOT waived: that one bounds the machine, not the entity's allowance.
  if (countsAgainstStarts) {
    const recent = startsLastHour();
    if (recent >= c.maxStartsPerHour) {
      return refuse('refused-cap', `You have already started ${recent} background jobs in the last hour, which is the limit (${c.maxStartsPerHour}). Nothing was started — say so plainly rather than implying it is running.`, { conversationId });
    }
  }

  const id = randomUUID();
  db.prepare(`
    INSERT INTO agent_jobs (id, title, task, why, status, source, conversation_id, message_id, created_at, retry_of)
    VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
  `).run(id, t.slice(0, 200), k, why ? String(why).trim().slice(0, 500) : null,
    source, conversationId, messageId, new Date().toISOString(), retryOf);

  console.log(`[AgentJobs] queued ${id.slice(0, 8)} (${source}): "${t}"`);
  opsLog(`Background job queued: "${t}" (${id.slice(0, 8)}, ${source}).`);
  logToolCall({ outcome: 'started', detail: `queued "${t}"`, refId: id, conversationId, args: { title: t } });

  launch(id);
  return { ok: true, id };
}

/**
 * Hand a queued job to the pool, without awaiting it.
 *
 * The deliberate un-awaited promise: the caller is a chat request that is about
 * to finish, and this run must outlive it. Rejections are impossible to lose
 * here because runJob() never throws — every exit writes a terminal row — but
 * the catch is kept anyway, because an unhandled rejection in a detached promise
 * is exactly the kind of silence this module exists to refuse.
 */
function launch(id) {
  const c = cfg();
  if (inFlight.size >= c.maxConcurrent) {
    // Not an error and not a drop: the row stays `queued` and the next
    // completion picks it up. Saying so is the point — a queue that silently
    // holds work looks identical to one that lost it.
    console.log(`[AgentJobs] ${id.slice(0, 8)} stays queued — ${inFlight.size} already in flight (max ${c.maxConcurrent})`);
    return false;
  }
  if (inFlight.has(id)) return false;
  inFlight.add(id);

  agentPool.schedule(() => runJob(id), `agent-job:${id.slice(0, 8)}`, 'agentJobs')
    .catch(err => {
      // runJob does not throw; if it somehow does, the row must still be closed.
      console.error(`[AgentJobs] ${id.slice(0, 8)} escaped its own error handling:`, err && err.message);
      try {
        finish(id, { status: 'failed', error: `the run failed in a way it was not built to handle: ${err && err.message}` });
      } catch { /* the console line above is the floor */ }
    })
    .finally(() => {
      inFlight.delete(id);
      drain();
    });
  return true;
}

/** Start whatever is queued and fits, oldest first. */
function drain() {
  const db = getSqliteDb();
  if (!db || !cfg().enabled) return 0;
  const waiting = db.prepare(
    "SELECT id FROM agent_jobs WHERE status = 'queued' ORDER BY datetime(created_at) ASC"
  ).all();
  let started = 0;
  for (const row of waiting) {
    if (!launch(row.id)) break;
    started++;
  }
  return started;
}

/** Close a job with what actually happened. The one write that ends a job. */
function finish(id, { status, resultText = null, error = null, toolCalls = 0, budget = null, stopSource = null, stopKind = null }) {
  const db = getSqliteDb();
  if (!db) return null;
  const job = getJob(id);
  if (!job) return null;
  const finishedAt = new Date();
  const startedAt = job.started_at ? new Date(job.started_at) : finishedAt;
  // A job that paused and resumed ran in pieces; the duration is the running
  // time, not the time she took to answer. The checkpointed elapsed figure is
  // the truth when there is one.
  const ranMs = budget && Number.isFinite(budget.elapsedMs) ? budget.elapsedMs : null;
  const durationMs = Math.max(0, ranMs ?? (finishedAt.getTime() - startedAt.getTime()));
  // THE CARD NEVER SHOWS A BARE WORD. `error` is the plain sentence; if a
  // caller closed a non-ok job without saying who stopped it, it is recorded as
  // unplaced rather than left null, so the panel can still say that much.
  const src = status === 'ok' ? null : (stopSource || (status === 'cancelled' ? 'user' : status === 'interrupted' ? 'runner' : 'unknown'));
  db.prepare(`
    UPDATE agent_jobs
    SET status = ?, finished_at = ?, duration_ms = ?, result_text = ?, error = ?,
        tool_calls = ?, budget_json = ?, stop_source = ?, stop_kind = ?, resume_mode = NULL
    WHERE id = ?
  `).run(status, finishedAt.toISOString(), durationMs, resultText, error,
    toolCalls ?? 0, budget ? JSON.stringify(budget) : null, src, status === 'ok' ? null : stopKind, id);
  if (status === 'ok') dropCheckpoint(id);
  return { ...getJob(id) };
}

/**
 * Give a finished job its file, if it should have one.
 *
 * SEPARATE FROM finish(), and after it, on purpose. finish() is the one write
 * that ENDS a job and the module's invariant is that every exit through it
 * writes exactly one terminal row; making a file is neither terminal nor
 * required, and folding it in would put "chromium would not start" on the same
 * line as "the job is over". So the status is settled first, and this is a
 * follow-up write that can only ever add columns.
 *
 * ⚠ IT CANNOT FAIL THE JOB. produce() is written not to throw and this catches
 * anyway: the result is already in the database, and a run that did real work
 * must never be turned into a failed one because a disk was full. The worst
 * outcome available here is a card with its text on it and a line saying why
 * there is no file — which is exactly the card she had before any of this.
 *
 * @returns {Promise<Object|null>} the updated row
 */
async function attachArtifact(id, { note = null } = {}) {
  const db = getSqliteDb();
  if (!db) return null;
  const job = getJob(id);
  if (!job) return null;
  // An in-turn read produces no document. A history search's result is a few
  // hundred tokens that have already been read in the conversation; filing one
  // in her documents folder per lookup would bury the reports she actually
  // asked for under forty transcripts of things she was told out loud.
  if (IN_TURN_SOURCES.includes(job.source)) return job;

  let made;
  try {
    made = await require('./job-artifacts').produce(job, {
      date: job.finished_at ? new Date(job.finished_at) : new Date(),
      note: note || job.error || ''
    });
  } catch (err) {
    // Belt and braces: produce() reports rather than throws, so reaching here
    // means something below it broke in a way it did not anticipate.
    const why = String(err && err.message || err).slice(0, 200);
    console.error(`[AgentJobs] ${id.slice(0, 8)} artifact step threw:`, why);
    opsLog(`Background job ${id.slice(0, 8)} produced a result but the file step threw: ${why}`);
    db.prepare('UPDATE agent_jobs SET artifact_error = ? WHERE id = ?')
      .run(`the file could not be written (${why}). The full result is above.`, id);
    return getJob(id);
  }

  db.prepare(`
    UPDATE agent_jobs
    SET artifact_kind = ?, artifact_path = ?, artifact_name = ?, artifact_bytes = ?,
        artifact_error = ?, summary_text = ?
    WHERE id = ?
  `).run(
    made.kind || null,
    made.path || null,
    made.name || null,
    Number.isFinite(made.bytes) ? made.bytes : null,
    made.error || null,
    // A summary only when there IS a file. For a result that stayed on the card
    // the text is the summary, and storing a second shorter copy beside it would
    // be two versions of one thing waiting to disagree.
    made.kind ? (made.summary || null) : null,
    id
  );

  if (made.kind) {
    opsLog(`Background job ${id.slice(0, 8)} saved as ${made.kind}: ${made.name} (${made.reason}).`
      + (made.error ? ` Note: ${made.error}` : ''));
  } else if (made.error) {
    opsLog(`Background job ${id.slice(0, 8)} produced no file: ${made.error}`);
  }
  return getJob(id);
}

// ---------------------------------------------------------------------------
// The executor
// ---------------------------------------------------------------------------

/**
 * WHAT A RETRY IS TOLD ABOUT THE ATTEMPT BEFORE IT. Bounded — a brief, not a
 * transcript — because the previous attempt's card is its honest summary and
 * a 60KB tool record would crowd out the task itself. The tool record comes
 * from the checkpoint when one survived, so the names and arguments of what
 * was already searched are there even when the result text is thin.
 */
const RETRY_BRIEF_CHARS = 6000;
function retryBrief(prev) {
  const lines = [];
  lines.push(`THIS IS A RETRY. Attempt ${prev.attemptNumber || 'before this one'} of this same job stopped: ${prev.error || 'reason unrecorded'}`);
  if (prev.resultText) {
    lines.push(`WHAT IT HAD WHEN IT STOPPED (its own words, possibly partial):\n"""\n${String(prev.resultText).slice(0, RETRY_BRIEF_CHARS)}\n"""`);
  }
  if (Array.isArray(prev.toolCalls) && prev.toolCalls.length) {
    const seen = prev.toolCalls.slice(0, 40).map((k, i) =>
      `${i + 1}. ${k.name}${k.args ? ' ' + JSON.stringify(k.args).slice(0, 120) : ''}${k.productive === false ? ' (came back empty or failed)' : ''}`);
    lines.push(`WHAT IT HAD ALREADY LOOKED UP (${prev.toolCalls.length} call(s)):\n${seen.join('\n')}`);
  }
  lines.push(`Build on this. Do not redo lookups that already came back with something unless you need to check them; ` +
    `pick up what was left unfinished. If the previous attempt found things, they count only if you can still ` +
    `stand behind them — say which parts are carried over from it.`);
  return lines.join('\n') + '\n';
}

function systemPrompt(job, tools, { previous = null } = {}) {
  const now = new Date();
  return (
    `You are Aurelius, running one of your own background jobs. Nobody is in the room. This is not a ` +
    `conversation — it is a job you started during one and then let go of, and what you write goes to ` +
    `Ellie's jobs panel, where she will read it when she is ready.\n\n` +
    `It is ${formatLocalTime(now, { style: 'full' })}.\n\n` +
    `THE JOB, as you set it when you handed it off:\n"${job.task}"\n` +
    (job.why ? `Why you handed it off: "${job.why}"\n` : '') +
    `\n` +
    // A RETRY DOES NOT START FROM ZERO. The previous attempt's reason for
    // stopping, what it had written, and what it had already looked up are all
    // in front of the model, so it builds on them rather than repeating them.
    (previous ? retryBrief(previous) + '\n' : '') +
    (tools.length
      ? `You have these read-only tools: ${tools.join(', ')}. Use them.\n\n` +
        // TWO KINDS OF JOB, AND THE OLD PROMPT ONLY ADMITTED ONE.
        //
        // It said "everything you report must come from a tool result", full
        // stop. That is exactly right for a job that reports on the world or on
        // the record, and it is wrong — flatly, self-defeatingly wrong — for a
        // job asked to PRODUCE something. Asked on 2026-08-18 to write a Python
        // calculator, a run under that instruction has no legal move: no tool
        // returns a calculator, so the only compliant answer is to report that
        // it found none. The rule is kept and scoped to what it was for: CLAIMS
        // ABOUT FACTS. What he makes, he makes out of what he knows.
        `TWO DIFFERENT THINGS, AND THEY HAVE DIFFERENT RULES:\n` +
        `- ANYTHING YOU ASSERT AS FACT — about the world, about your memory, about what happened — must ` +
        `come from a tool result in this run. Never state a number, a date or an event you did not read ` +
        `from one.\n` +
        `- ANYTHING YOU ARE ASKED TO PRODUCE — a script, a draft, a plan, a piece of writing — you write ` +
        `yourself, out of what you know. That is not answering from impression; that is the work. Use the ` +
        `tools to check facts it depends on, and say which parts you could not check.\n\n`
      : `You have NO tools in this run, which means you cannot look anything up. If this job needs facts, ` +
        `say plainly that you could not check them rather than answering from impression. If it asks you to ` +
        `WRITE something, write it — that needs no tools.\n\n`) +
    // The same two-sided instruction the scheduler carries, and for the same
    // measured reason: told only not to invent, the model discovers that
    // "nothing to report" is always safe, and a job that always reports nothing
    // is indistinguishable from a job that is broken.
    `THERE ARE TWO WAYS TO GET THIS WRONG AND THEY ARE EQUALLY BAD:\n` +
    `1. Reporting something the tools did not show you. Never state a number, a date or an event you did ` +
    `not read from a tool result.\n` +
    `2. Reporting nothing when the tools DID show you something. Do the arithmetic on timestamps rather ` +
    `than eyeballing them. "Nothing to report" is the right answer ONLY when you have looked and what you ` +
    `found is genuinely empty.\n` +
    `If a tool result says it is capped, partial, or showing only the most recent few, say so rather than ` +
    `treating what you were given as all there is.\n\n` +
    `WHAT TO WRITE. Plain writing to Ellie, in your own voice — what you found, and what it means if it ` +
    `means anything. No preamble like "here is the result", no restating the task back to her. If the ` +
    `honest answer is short, keep it short. If she asked for something built, the thing itself IS the ` +
    `result: put the script or the draft in the answer.\n\n` +
    // WHAT HE WRITES NOW BECOMES A FILE, so he is told the rule that decides
    // which one. The old prompt said "no headings" flatly — correct when every
    // result was three sentences on a card, and wrong the moment a long one
    // became a printed report, where the absence of headings is what makes it
    // unreadable. The line is not "use headings"; it is that the SHAPE should
    // follow the length, which is the same judgement the classifier makes.
    `WHAT HAPPENS TO IT. A short answer stays on her card. A long one is turned into a document and saved ` +
    `to her documents folder, and a single block of code becomes a source file with the right extension. ` +
    `You do not choose this and you must not announce it — write the thing, and it is filed by what it is. ` +
    `Two things follow. Keep a short answer free of headings and structure; it is a note, not a report. ` +
    `But if what you are writing IS long, give it the structure a document needs — headings, tables where ` +
    `the data is tabular, a short opening paragraph that says what you found, since that opening is what ` +
    `she reads on the card before deciding to open it.\n\n` +
    `CHARTS, when numbers are the point. A fenced block marked \`chart\` becomes a real figure in the ` +
    `document — a pie, a bar chart or a line chart drawn from the data you put in it:\n` +
    '```chart\n{"type":"pie","title":"Tickets by client","data":[{"label":"Acme","value":42},' +
    '{"label":"Beta","value":17}]}\n```\n' +
    `Use "pie" for shares of a whole, "bar" to compare amounts, "line" for change over time (that one takes ` +
    `"series":[{"name":…,"data":[…]}]). Only ever chart numbers you actually read from a tool result — a ` +
    `chart of invented figures is the most convincing way to be wrong. One or two per document; if there is ` +
    `nothing to compare, a sentence is better than a figure.\n\n` +
    // The empty-card rule, said to him as well as enforced in code below.
    `YOU MUST WRITE SOMETHING. Whatever state you are in when you stop — out of tool calls, out of time, ` +
    `every lookup failing — write up what you have and say where it stops and why. An empty result is the ` +
    `one outcome that tells her nothing at all, and it throws away whatever you did manage to do.\n\n` +
    `THIS IS NOT A MESSAGE TO HER. It lands in a panel; it does not open a conversation and it does not ` +
    `interrupt her. If what you find turns out to be worth actually SAYING to her, that is a separate ` +
    `decision you make in an ordinary conversation later — not something this run does.`
  );
}

/**
 * A one-line, human account of WHY a run stopped where it did.
 *
 * Written from the budget summary and the tool record rather than from a status
 * word, because "it did not produce a result" was all the panel could say and it
 * was the least useful true sentence available.
 */
function describeStop(calls = [], budget = null, c = cfg()) {
  const dead = calls.filter(k => k && k.productive === false).length;
  const bits = [];
  if (budget && budget.exhausted) bits.push(budget.exhausted);
  else if (calls.length) bits.push(`it made ${calls.length} tool call(s)`);
  if (dead) bits.push(`${dead} of its calls came back empty or failed`);
  if (!bits.length) bits.push('it stopped without writing an answer');
  return `${bits.join('; ')} — what is in the result is what it had when it stopped`;
}

/**
 * ONE MORE CALL, NO TOOLS: write up what you have.
 *
 * The first and better of the two salvage attempts. It is a fresh callLLM with no
 * toolSession — so it cannot look anything else up, cannot spend more budget, and
 * cannot loop — carrying a compact record of what this run actually did. What
 * comes back is a real partial answer in his own voice.
 *
 * Returns null rather than throwing: the caller has a deterministic fallback and
 * a salvage attempt that fails must not turn a partial job into a crashed one.
 */
async function salvageWriteup(job, calls = [], budget = null, c = cfg(), { convo = null, partialText = '', failure = null } = {}) {
  const mm = memoryManager();

  // THE TRANSCRIPT ITSELF, WHEN THERE IS ONE. The checkpointed conversation
  // holds every tool result the run received, which is a far better thing to
  // write up from than a list of what was called. A run that died on a cut
  // stream continues from its own transcript with one no-tools turn — the
  // same shape as the out-of-rounds writeup.
  if (Array.isArray(convo) && convo.length > 2) {
    try {
      const msgs = [...convo];
      const last = msgs[msgs.length - 1];
      // A dangling tool request (the round that was cut) gets results saying so,
      // or the transcript is malformed for the next request.
      if (last && last.role === 'assistant' && Array.isArray(last.tool_calls) && last.tool_calls.length) {
        for (const call of last.tool_calls) {
          msgs.push({ role: 'tool', tool_call_id: call.id, name: call.function && call.function.name,
            content: JSON.stringify({ error: 'Not run — the job stopped before this call could be made.' }) });
        }
      }
      msgs.push({
        role: 'user',
        content:
          `STOP — the job ended before you finished${failure ? `: ${failure}` : ''}. You have no tools for this turn. ` +
          `Write up what you have NOW, from the tool results above, for Ellie's jobs panel: what you found, what it ` +
          `means, and plainly where it stops. If a lookup failed, say it failed rather than reporting the gap as ` +
          `nothing to find. Do not invent anything and do not answer with nothing.` +
          (partialText ? `\n\nYou had already begun writing this before it was cut off — keep what is right in it:\n"""\n${partialText.slice(0, 6000)}\n"""` : '')
      });
      const res = await mm.callLLM(null, null, { continueMessages: msgs, maxTokens: c.answerTokens, thinkingTokens: c.thinkingTokens });
      const text = String(res && res.content || '').trim();
      if (text) return text;
    } catch (err) {
      console.warn(`[AgentJobs] salvage from transcript failed: ${err && err.message}`);
    }
  }

  const record = calls.length
    ? calls.map((k, i) => {
      const what = k && k.name ? k.name : 'a tool';
      const arg = k && k.args ? JSON.stringify(k.args).slice(0, 160) : '';
      const how = k && k.productive === false
        ? `nothing usable came back${k.note ? ` (${k.note})` : ''}`
        : 'it returned something usable';
      return `${i + 1}. ${what} ${arg} → ${how}`;
    }).join('\n')
    : '(no tool calls were made at all)';

  const system =
    `You are Aurelius, closing out one of your own background jobs that stopped before it wrote anything. ` +
    `You have no tools now and cannot look anything else up.\n\n` +
    `THE JOB was: "${job.task}"\n\n` +
    `WHAT THE RUN ACTUALLY DID:\n${record}\n` +
    (budget && budget.exhausted ? `\nWhy it stopped: ${budget.exhausted}\n` : '') +
    (partialText ? `\nWHAT YOU HAD ALREADY BEGUN WRITING before it was cut off (keep what is right in it):\n"""\n${partialText.slice(0, 6000)}\n"""\n` : '') +
    `\nWrite the result for Ellie's jobs panel now, in a few plain sentences in your own voice:\n` +
    `- What you managed to establish, if anything. Only what a tool result above actually supports.\n` +
    `- If the job asked you to WRITE or BUILD something, write it now from your own knowledge — that ` +
    `needs no tools, and it is the result.\n` +
    `- Where it stopped and why, plainly. If your lookups failed, say they failed — do NOT report that as ` +
    `having found nothing, because those are different things and only one of them is about the world.\n` +
    `Do not apologise, do not describe this as a salvage, and do not answer with nothing.`;

  try {
    const res = await mm.callLLM(system, job.task,
      { maxTokens: c.answerTokens, thinkingTokens: c.thinkingTokens });
    const text = String(res && res.content || '').trim();
    return text || null;
  } catch (err) {
    console.warn(`[AgentJobs] salvage writeup failed: ${err && err.message}`);
    return null;
  }
}

/**
 * THE FLOOR. No model, no network, cannot come back empty.
 *
 * Reached only when the run wrote nothing AND the salvage call could not either —
 * which in practice means the brain is unreachable. It is a thin card and it is
 * an honest one: what was asked, what ran, what came back, where it stopped. The
 * invariant it defends is simple and absolute: a job that started has a result
 * she can read.
 */
function mechanicalAccount(job, calls = [], budget = null, c = cfg(), thrownError = null, { partialText = '' } = {}) {
  const dead = calls.filter(k => k && k.productive === false);
  const lines = [];

  lines.push(`I could not write this up properly, so this is the plain account of what happened.`);
  lines.push('');
  lines.push(`What I set out to do: ${job.task}`);
  lines.push('');

  if (partialText) {
    // What it had streamed before the cut is the most valuable thing here and
    // it goes first, marked for what it is.
    lines.push(`What I had written before it stopped (cut off, not finished):`);
    lines.push('');
    lines.push(partialText.slice(0, 20000));
    lines.push('');
  }

  if (!calls.length) {
    // Careful with this sentence. When the run THREW and no checkpoint survived,
    // the tool record is gone — so "it looked nothing up" would be a claim, not
    // a fact. Say what is actually known: there is no record.
    lines.push(thrownError
      ? `The run failed and I have no record of what it managed to look up first. What stopped it: ${thrownError}`
      : `No tools were called at all, and no answer was produced.`);
  } else {
    lines.push(`I made ${calls.length} tool call(s): ${calls.map(k => k.name).join(', ')}.`);
    if (dead.length) {
      lines.push(`${dead.length} of them came back empty or failed${dead[0] && dead[0].note ? ` (${dead[0].note})` : ''}.`);
    }
    if (dead.length === calls.length) {
      lines.push(`Nothing usable came back from any of them, so there is nothing here I can tell you about ` +
        `the thing itself — that is a failure of my lookups, not a finding that there is nothing to find.`);
    } else {
      lines.push(`Some of them did return something, but I stopped before turning it into an answer.`);
    }
  }

  if (budget && budget.exhausted) lines.push(`Why it stopped: ${budget.exhausted}.`);
  if (thrownError && calls.length) lines.push(`What stopped it: ${thrownError}`);
  lines.push('');
  lines.push(`Retry it from the card and I will pick up from what is here.`);

  return lines.join('\n');
}

/**
 * Run one job, now.
 *
 * Every exit from this function writes exactly one terminal row. That is the
 * invariant: if it was started, there is a record, whatever happened.
 *
 * @returns {Promise<Object>} the finished agent_jobs row
 */
async function runJob(id) {
  const db = getSqliteDb();
  if (!db) return null;

  const job = getJob(id);
  if (!job) {
    console.warn(`[AgentJobs] ${String(id).slice(0, 8)} vanished before it could run`);
    return null;
  }
  if (job.status !== 'queued') {
    // Cancelled while it waited, or already running. Either way this launch is
    // not the one that should proceed.
    console.log(`[AgentJobs] ${id.slice(0, 8)} is "${job.status}", not queued — not running it`);
    return job;
  }

  const startedAt = new Date();
  // A resume after her yes or no is the SAME attempt continuing, not a new
  // start — only a fresh run or a restart's re-run counts. Otherwise one ask
  // would use up the one retry a later restart is allowed.
  const continuing = job.resume_mode === 'granted' || job.resume_mode === 'declined';
  db.prepare(
    `UPDATE agent_jobs SET status = 'running', started_at = ?, attempts = COALESCE(attempts, 0) + ${continuing ? 0 : 1} WHERE id = ?`
  ).run(startedAt.toISOString(), id);

  // A history search is an agent run, but not THIS one. It has its own system
  // prompt (search, read, quote, and say so when there is nothing), its own two
  // read tools, and — the part that makes it a separate runner rather than a
  // different task string — a verification pass over what the model returns
  // before any of it is allowed to reach the conversation. Same row, same pool,
  // same restart sweep; different contract. See db/history-search.js.
  if (job.source === require('./history-search').SOURCE) {
    let outcome;
    try {
      outcome = await require('./history-search').runDispatched(job);
    } catch (err) {
      // runDispatched is written not to throw. If it ever does, the waiting
      // chat turn must still be answered, and answered with nothing rather
      // than with silence it might fill in.
      outcome = {
        status: 'failed',
        resultText: `The history search failed before it could report: ${err.message}. You have nothing from your history — say so.`,
        error: err.message
      };
    }
    return finish(id, {
      status: outcome.status,
      resultText: outcome.resultText,
      error: outcome.error || null,
      toolCalls: outcome.toolCalls || 0,
      // budget_json carries the run's own timing/reasoning breakdown for this
      // source. It was empty for every v1 run, which is why the first slow one
      // had to be autopsied out of journalctl.
      budget: outcome.metrics || null
    });
  }

  // A dispatched coding job is not an agent run. It has no tool loop here,
  // no JOB_TOOLS, and no model call in this process: squatch-code has its
  // own agentic loop and its own model, and this hands it a brief rather
  // than driving it step by step. The row, the panel, the badge and the
  // announcement are shared; the execution is not.
  if (job.source === require('./coding-jobs').SOURCE) {
    let outcome;
    try {
      outcome = await require('./coding-jobs').runDispatched(job);
    } catch (err) {
      // runDispatched is written not to throw. If it ever does, the row
      // must still close with something readable rather than hanging.
      outcome = {
        status: 'failed',
        resultText: `The dispatch itself failed: ${err.message}. Check git status in the project before assuming nothing changed.`,
        error: `SNH's job runner failed while dispatching it: ${err.message}. Check git status in the project before assuming nothing changed.`,
        stopSource: 'runner', stopKind: 'dispatch'
      };
    }
    return finish(id, {
      status: outcome.status,
      resultText: outcome.resultText,
      error: outcome.error || null,
      toolCalls: outcome.toolCalls || 0,
      stopSource: outcome.stopSource || (outcome.status === 'ok' ? null : 'dispatched'),
      stopKind: outcome.stopKind || null
    });
  }

  return runChatHandoff(id, job);
}

/**
 * THE AGENT RUN ITSELF — a chat-handoff job, fresh or resumed.
 *
 * Every exit writes exactly one terminal row OR one paused row. The shape:
 *
 *   fresh run ──► tool loop ──► ok / partial (cut short) / failed (threw)
 *                     │
 *                     └─ near a ceiling with work left ──► ASK ──► paused
 *
 *   paused + she said yes  ──► resumed from the checkpoint with more budget
 *   paused + she said no   ──► one no-tools writeup ──► partial
 *   killed by a restart    ──► resumed from the last complete round (if young)
 *
 * The checkpoint on disk is what makes every one of those arrows possible; the
 * tool loop writes it after every step and this function reads it back.
 */
async function runChatHandoff(id, job) {
  const mm = memoryManager();
  const MCPClient = require('../mcp/mcp-client');
  const allowed = MCPClient.shared().backgroundToolsAmong(JOB_TOOLS);
  const denied = JOB_TOOLS.filter(t => !allowed.includes(t));
  if (denied.length) {
    // Loud: a job that ran without a tool it wanted produces a thinner answer
    // than it should, and that must never read as "there was nothing to find".
    const line = `Background job ${id.slice(0, 8)} could not be given tool(s): ${denied.join(', ')}. It ran with ${allowed.length} of ${JOB_TOOLS.length}.`;
    console.warn(`[AgentJobs] ${line}`);
    opsLog(line);
  }

  const c = cfg();
  const session = mm.createToolSession(`agent-job:${id.slice(0, 8)}`, allowed, {
    maxCalls: c.maxToolCallsPerJob,
    maxWallMs: c.maxWallClockMs,
    maxRounds: c.maxRoundsPerJob
  });

  // --- Where it picks up from, if anywhere ---------------------------------
  const mode = job.resume_mode || null;                       // granted | declined | restart | null
  const ask = parseAsk(job.ask_json);
  const ck = mode ? readCheckpoint(id) : null;
  let resume = null;
  let restartNote = null;
  if (mode && ck && Array.isArray(ck.convo) && ck.convo.length) {
    session.restore(ck.session);
    if (mode === 'granted') {
      const g = (ask && ask.grant) || {};
      session.extend({ calls: g.calls || 0, rounds: g.rounds || 0, wallMs: g.wallMs || 0 });
      resume = {
        convo: ck.convo, toolCalls: ck.toolCalls || [], pendingCalls: ck.pendingCalls || [],
        note: `Ellie said yes. You have ${g.calls || 0} more tool call(s), ${g.rounds || 0} more round(s) and ` +
          `${require('./job-failure').sayDuration(g.wallMs || 0)} more on the clock — ` +
          `${session.billed.toFixed(1)} of ${session.maxCalls} calls used so far. Carry on and finish the job; ` +
          `write the result when you have it.`
      };
      console.log(`[AgentJobs] ${id.slice(0, 8)} resuming with more budget: +${g.calls} calls, +${g.rounds} rounds, +${Math.round((g.wallMs || 0) / 60000)} min`);
    } else if (mode === 'restart') {
      const t = trimToCompleteRound(ck.convo, ck.toolCalls || []);
      session.roundsUsed = t.lastRound;
      resume = {
        convo: t.convo, toolCalls: t.toolCalls, pendingCalls: [],
        note: `SNH restarted while you were in the middle of a tool round, and that round was lost. ` +
          `Everything above it is intact${t.toolCalls.length ? ` (${t.toolCalls.length} tool call(s) so far)` : ''}. Continue from here.`
      };
      restartNote = t;
      console.log(`[AgentJobs] ${id.slice(0, 8)} resuming after a restart from round ${t.lastRound} (${t.dropped} message(s) of an unfinished round dropped)`);
    }
  } else if (mode && !ck) {
    console.warn(`[AgentJobs] ${id.slice(0, 8)} was to resume (${mode}) but has no checkpoint — running from the start`);
  }

  // --- She said NO: one writeup turn, no tools, and it is over ---------------
  if (mode === 'declined' && ck && Array.isArray(ck.convo) && ck.convo.length) {
    return wrapUpDeclined(id, job, ck, session, c);
  }

  // --- A retry carries the previous attempt ----------------------------------
  let previous = null;
  if (job.retry_of) {
    const prev = getJob(job.retry_of);
    if (prev) {
      const prevCk = readCheckpoint(prev.id);
      previous = {
        attemptNumber: prev.attempts || 1,
        error: prev.error,
        resultText: prev.result_text,
        toolCalls: prevCk && Array.isArray(prevCk.toolCalls) ? prevCk.toolCalls : []
      };
    }
  }

  const askPct = c.askBeforeCeiling ? c.askAtPercent : 0;
  const pauseWhen = (sess) => sess.nearing(askPct);
  const checkpoint = (state) => writeCheckpoint(id, { ...state, title: job.title, task: job.task });

  console.log(`[AgentJobs] === running ${id.slice(0, 8)}: "${job.title}"${mode ? ` (${mode})` : ''} ===`);

  let status = 'ok', error = null, output = '', budget = null, toolCalls = 0;
  let stopSource = null, stopKind = null;
  let calls = [];
  try {
    const res = await mm.callLLM(
      systemPrompt(job, allowed, { previous }),
      job.task,
      {
        maxTokens: c.answerTokens, thinkingTokens: c.thinkingTokens, toolSession: session,
        resume: resume ? { convo: resume.convo, toolCalls: resume.toolCalls, pendingCalls: resume.pendingCalls, note: resume.note } : null,
        checkpoint, pauseWhen
      }
    );
    calls = Array.isArray(res && res.toolCalls) ? res.toolCalls : [];
    toolCalls = calls.length;
    budget = (res && res.budget) || session.summary();

    // === NEAR A CEILING WITH WORK LEFT: ASK HER, THEN WAIT ===================
    if (res && res.paused) {
      return pauseAndAsk(id, job, res, session, c);
    }

    output = String(res && res.content || '').trim();

    // A run that was CUT SHORT but still wrote something is not "ok". The text is
    // kept in full and the card says which it was — see TERMINAL on `partial`.
    //
    // HITTING THE ANSWER BUDGET IS THE THIRD WAY TO BE CUT SHORT, and it was the
    // one nothing looked at. `runToolLoop` has always returned `truncated` off
    // finish_reason === 'length'; this function read `outOfRounds` and
    // `budget.exhausted` beside it and dropped `truncated` on the floor. So a job
    // that generated right up to max_tokens and stopped mid-token landed as `ok`
    // with a full-looking card. Measured cost (2026-08-18, aiserver): three
    // coding jobs cut off mid-function, all three presenting as finished.
    //
    // Truncation is checked FIRST because it is the most specific reason and the
    // only one that says WHERE the result stops — db/job-failure.js keeps that
    // order, and names the source and the limit for each.
    const cut = require('./job-failure').classifyCutShort({
      truncated: !!(res && res.truncated), outOfRounds: !!(res && res.outOfRounds), budget,
      answerTokens: c.answerTokens, maxRounds: session.maxRounds
    });
    if (output && cut) {
      status = 'partial';
      stopSource = cut.source; stopKind = cut.kind;
      error = `${cut.plain} What is above is what it had.`;
    }

    if (!output) {
      // === AN EMPTY RESULT CARD IS NOT ALLOWED TO HAPPEN ===
      //
      // 2026-08-18: a job spent all twelve of its tool calls on searches that
      // failed with the same broken-URL error, produced no text, and was closed
      // as `failed` with result_text NULL. The panel card was empty — and the
      // memory work it had already completed, before it ever reached a search,
      // went in the bin with it. The work was done. Only the writing-up was
      // missing, and nothing asked for it.
      //
      // Two attempts, in order of how much they can say:
      //   1. ASK IT TO WRITE UP WHAT IT HAS — one more call, no tools, with the
      //      run's own tool record in front of it. This is where a real partial
      //      answer comes from.
      //   2. FAILING THAT, WRITE THE ACCOUNT OURSELVES — deterministic, no model
      //      involved, so it cannot itself come back empty. It is a thinner
      //      thing than a writeup and it is still a hundred times better than a
      //      blank card: it says what ran, what came back, and where it stopped.
      const salvaged = await salvageWriteup(job, calls, budget, c, { convo: res && res.convo });
      const stop = cut || require('./job-failure').classifyBudgetStop(budget, { maxRounds: session.maxRounds }) || { source: 'runner', kind: 'no-answer', plain: 'It stopped without writing an answer.' };
      stopSource = stop.source; stopKind = stop.kind;
      if (salvaged) {
        output = salvaged;
        status = 'partial';
        error = `${stop.plain} ${describeStop(calls, null, c)}`;
        console.warn(`[AgentJobs] ${id.slice(0, 8)} produced no answer — salvaged a writeup from ${toolCalls} tool call(s)`);
      } else {
        output = mechanicalAccount(job, calls, budget, c);
        status = 'partial';
        error = `${stop.plain} ${describeStop(calls, null, c)}`;
        console.warn(`[AgentJobs] ${id.slice(0, 8)} produced no answer and no writeup — wrote the mechanical account instead`);
      }
    }
  } catch (err) {
    status = 'failed';
    budget = (err && err.budget) || session.summary();
    // THE RECORD SURVIVES THE THROW. The loop attaches its tool record and
    // transcript to the error, and the checkpoint on disk has the same; between
    // them the failure card can say what it had rather than "no record".
    calls = Array.isArray(err && err.toolCalls) ? err.toolCalls : [];
    if (!calls.length) { const saved = readCheckpoint(id); if (saved && Array.isArray(saved.toolCalls)) calls = saved.toolCalls; }
    toolCalls = calls.length;
    const partialText = String(err && err.partial && err.partial.content || '').trim();
    // WHO STOPPED IT. Never the raw error text as the sentence: the watchdog
    // is consulted so a restart it issued is named as the cause.
    const failure = require('./job-failure').classifyThrown(err, {
      round: err && err.round, calls: toolCalls,
      recentRestart: (() => { try { return require('./brain-watchdog').recentRestart(); } catch { return null; } })(),
      formatTime: (ms) => formatLocalTime(new Date(ms), { style: 'time', fallback: 'an unclear time' })
    });
    stopSource = failure.source; stopKind = failure.kind;
    error = failure.plain;
    console.error(`[AgentJobs] ${id.slice(0, 8)} failed (${failure.source}/${failure.kind}):`, failure.technical || error);
    // Even a thrown run writes what it has. First ask for a writeup from the
    // transcript — the engine may well be back by now — and fall to the
    // mechanical account if it is not. Either way the text it had streamed
    // before the cut is kept, labelled as partial.
    const convo = Array.isArray(err && err.convo) ? err.convo : ((readCheckpoint(id) || {}).convo || null);
    const salvaged = calls.length || partialText ? await salvageWriteup(job, calls, budget, c, { convo, partialText, failure: error }) : null;
    output = salvaged || mechanicalAccount(job, calls, budget, c, error, { partialText });
  }

  const done = finish(id, { status, resultText: output || null, error, toolCalls, budget, stopSource, stopKind });
  // The file comes after the status is settled — see attachArtifact.
  const withFile = done ? await attachArtifact(id, { note: error }) : null;
  const secs = done ? (done.duration_ms / 1000).toFixed(1) : '?';
  const line = status === 'ok'
    ? `Background job finished: "${job.title}" (${id.slice(0, 8)}) — ok in ${secs}s, ${toolCalls} tool call(s).`
    : status === 'partial'
      ? `Background job finished PARTIAL: "${job.title}" (${id.slice(0, 8)}) — ${secs}s, ${toolCalls} tool call(s). It wrote up what it had. Why it stopped: ${error}`
      : `Background job FAILED: "${job.title}" (${id.slice(0, 8)}) — ${error}`;
  console.log(`[AgentJobs] ${line}`);
  opsLog(line);
  return withFile || done;
}

/**
 * THE ASK. The loop stopped short of running the calls the model wanted; this
 * writes the message she will read, records what a yes grants, parks the row,
 * and hands the delivery to db/job-budget-ask.js — the one module that lets a
 * job speak, and only because a decision is needed from her.
 */
async function pauseAndAsk(id, job, res, session, c) {
  const mm = memoryManager();
  const { sayDuration } = require('./job-failure');
  const near = res.near || {};
  const pending = Array.isArray(res.pendingCalls) ? res.pendingCalls.length : 0;
  const st = session.state();
  const used = near.limit === 'calls'
    ? `${near.used} of ${near.max} tool calls`
    : near.limit === 'rounds'
      ? `${near.used} of ${near.max} tool rounds`
      : `${sayDuration(near.used)} of its ${sayDuration(near.max)} time limit`;

  // What a plain yes grants: a share of the ORIGINAL limits, so repeated
  // yeses do not compound.
  const base = { calls: c.maxToolCallsPerJob, rounds: c.maxRoundsPerJob, wallMs: c.maxWallClockMs };
  const f = c.extensionPercent / 100;
  const defaultGrant = {
    calls: Math.max(1, Math.round(base.calls * f)),
    rounds: Math.max(1, Math.round(base.rounds * f)),
    wallMs: Math.max(60000, Math.round(base.wallMs * f))
  };

  // The model writes the ask in its own voice, from its own transcript. A
  // throwaway copy of the transcript: the checkpoint the resume reads was
  // written by the loop before this turn and is not touched by it.
  let askBody = '';
  let wants = null;
  try {
    const convo = [...(res.convo || [])];
    // Well-formed transcript: the assistant turn asked for tools, so each gets
    // a result saying it was not run yet.
    for (const call of (res.pendingCalls || [])) {
      convo.push({ role: 'tool', tool_call_id: call.id, name: call.function && call.function.name,
        content: JSON.stringify({ paused: 'Not run yet — you are near your budget and Ellie is being asked whether to extend it.' }) });
    }
    convo.push({
      role: 'user',
      content:
        `PAUSE. You have used ${used}${near.limit === 'calls' ? ` (${st.calls} calls made, ${st.failedCalls} of them empty or failed)` : ''}, ` +
        `and you just asked for ${pending} more tool call(s), so there is work left. Nothing more runs until Ellie decides.\n\n` +
        `Write a short message TO ELLIE, in your own voice, that she will read in the conversation where she asked for this:\n` +
        `1. What you have established so far — the actual findings, briefly. Only what your tool results support.\n` +
        `2. What is still left to do.\n` +
        `3. How much more you need, and why.\n` +
        `End with one line on its own, exactly of the form:\nNEEDED: <whole number> more tool calls\n` +
        `Under 200 words. No headings, no preamble, no apology.`
    });
    const r = await mm.callLLM(null, null, {
      continueMessages: convo, maxTokens: Math.min(c.answerTokens, 1200), thinkingTokens: c.thinkingTokens
    });
    askBody = String(r && r.content || '').trim();
    const m = /NEEDED:\s*(\d{1,3})\s*more/i.exec(askBody);
    if (m) wants = Math.max(1, parseInt(m[1], 10));
    askBody = askBody.replace(/\n?\s*NEEDED:\s*\d{1,3}\s*more[^\n]*/i, '').trim();
  } catch (err) {
    console.warn(`[AgentJobs] ${id.slice(0, 8)} could not write its own ask (${err.message}) — using the plain account`);
  }
  if (!askBody) {
    const calls = Array.isArray(res.toolCalls) ? res.toolCalls : [];
    askBody =
      `I'm partway through "${job.title}" and I've used ${used}. I've made ${calls.length} lookup(s)` +
      `${calls.filter(k => k.productive === false).length ? `, ${calls.filter(k => k.productive === false).length} of which came back empty or failed` : ''}, ` +
      `and I had ${pending} more queued when I stopped to ask. I couldn't write up the findings in this pause, but they are saved and nothing is lost.`;
  }

  // A named number wins over the default, within reason: nobody gets to ask
  // for a thousand. The rounds and clock scale with it so a call grant is not
  // silently bound by the other two.
  const grant = { ...defaultGrant };
  if (wants) {
    const capped = Math.min(wants, base.calls * 3);
    grant.calls = capped;
    grant.rounds = Math.max(defaultGrant.rounds, Math.ceil(capped / 2));
  }

  const footer =
    `\n\nIf you say yes, I get ${grant.calls} more tool call${grant.calls === 1 ? '' : 's'}, ${grant.rounds} more round${grant.rounds === 1 ? '' : 's'} ` +
    `and ${sayDuration(grant.wallMs)} more on the clock (name a number if you want a different amount). ` +
    `If you say no, I'll write up what I have and stop there. I'll wait either way.`;
  const askText = askBody + footer;

  const askRecord = {
    askedAt: new Date().toISOString(),
    near, used: st, wants, grant, defaultGrant, pending,
    text: askText,
    answeredAt: null, decision: null
  };
  const db = getSqliteDb();
  db.prepare(`UPDATE agent_jobs SET status = 'paused', paused_at = ?, ask_json = ?, budget_json = ?, tool_calls = ? WHERE id = ?`)
    .run(askRecord.askedAt, JSON.stringify(askRecord), JSON.stringify(session.summary()), (res.toolCalls || []).length, id);

  // Delivery: the conversation that dispatched it, and the bell pointing there.
  let delivery = null;
  try {
    delivery = await require('./job-budget-ask').deliver(getJob(id), askText);
  } catch (err) {
    console.error(`[AgentJobs] ${id.slice(0, 8)} could not deliver its ask: ${err.message}`);
    delivery = { error: err.message };
  }
  askRecord.delivery = delivery;
  db.prepare('UPDATE agent_jobs SET ask_json = ? WHERE id = ?').run(JSON.stringify(askRecord), id);

  const line = `Background job PAUSED to ask for more: "${job.title}" (${id.slice(0, 8)}) — near its ${near.limit} limit (${used}) with ${pending} call(s) queued. ` +
    `A yes grants +${grant.calls} calls, +${grant.rounds} rounds, +${Math.round(grant.wallMs / 60000)} min.` +
    (delivery && delivery.conversationId ? ` Asked in conversation ${String(delivery.conversationId).slice(0, 8)}.` : ` The ask could not be delivered: ${delivery && delivery.error}`);
  console.log(`[AgentJobs] ${line}`);
  opsLog(line);
  return getJob(id);
}

/**
 * SHE SAID NO. One turn, no tools, from the saved transcript: write up what you
 * have. Then the row closes as partial, and the reason names her decision —
 * which is not a failure of anything.
 */
async function wrapUpDeclined(id, job, ck, session, c) {
  const mm = memoryManager();
  const { sayDuration } = require('./job-failure');
  const ask = parseAsk(job.ask_json) || {};
  const convo = [...ck.convo];
  for (const call of (ck.pendingCalls || [])) {
    convo.push({ role: 'tool', tool_call_id: call.id, name: call.function && call.function.name,
      content: JSON.stringify({ error: 'Not run — Ellie chose to stop here. Write up what you already have.' }) });
  }
  convo.push({
    role: 'user',
    content:
      'Ellie said no to more budget, so this is the end of the job. You have no tools for this turn. ' +
      'Write the result for her jobs panel now, from the tool results above: what you found, what it means, ' +
      'and plainly which part is unfinished. Do not invent anything to fill the gap and do not answer with nothing.'
  });
  const calls = Array.isArray(ck.toolCalls) ? ck.toolCalls : [];
  let output = '';
  let failedWriteup = null;
  try {
    const r = await mm.callLLM(null, null, { continueMessages: convo, maxTokens: c.answerTokens, thinkingTokens: c.thinkingTokens });
    output = String(r && r.content || '').trim();
  } catch (err) {
    failedWriteup = err.message;
    console.warn(`[AgentJobs] ${id.slice(0, 8)} could not write up after the decline: ${err.message}`);
  }
  const st = session.state();
  const usedLine = `${st.billed.toFixed(1)} of ${st.maxCalls} tool calls, ${st.roundsUsed} of ${st.maxRounds} rounds, ${sayDuration(st.elapsedMs)} of ${sayDuration(st.maxWallMs)}`;
  const error = `It stopped at your decision: you chose not to extend its budget (it had used ${usedLine}). What is above is what it had.`;
  if (!output) output = mechanicalAccount(job, calls, session.summary(), c, failedWriteup ? `the writeup call failed: ${failedWriteup}` : null);
  const done = finish(id, { status: 'partial', resultText: output, error, toolCalls: calls.length, budget: session.summary(), stopSource: 'user', stopKind: 'budget-declined' });
  const withFile = done ? await attachArtifact(id, { note: error }) : null;
  const line = `Background job finished PARTIAL at Ellie's decision: "${job.title}" (${id.slice(0, 8)}) — she declined more budget; it wrote up what it had (${calls.length} tool call(s)).`;
  console.log(`[AgentJobs] ${line}`);
  opsLog(line);
  return withFile || done;
}

// ---------------------------------------------------------------------------
// Restart recovery
// ---------------------------------------------------------------------------

/**
 * THE FOUR REASONS AN INTERRUPTED JOB IS NOT RUN AGAIN.
 *
 * Each is a different decision with a different consequence for her — one of
 * them ("it can write files") means work may be half-applied on disk right now —
 * so recording the wrong one is a real fault, not a cosmetic one. Named and
 * exported because the suite's job is to assert WHICH reason a row got, and a
 * substring copied into the test cannot tell two of these apart once either is
 * reworded.
 */
const NOT_RERUN = {
  ANSWERS_NOBODY:
    'interrupted by a restart. It was NOT run again: it was a lookup answering a conversation that the same restart ended, so there is nobody left to hand the answer to',
  WRITES_TO_DISK:
    'interrupted by a restart. It was NOT run again, because it can write files and re-running the brief could apply it twice. Anything it had already changed is still changed - check git status in the project; it commits a restore point before it starts',
  ALREADY_RETRIED:
    'interrupted by a restart, and it had already been retried once — it was not run again',
  tooOld: (graceMinutes) =>
    `interrupted by a restart, and by the time the server came back it was too old to be worth redoing (older than ${graceMinutes} minutes) — it was not run again`,
};

/**
 * Close out runs a restart interrupted, and redo the ones still worth redoing.
 *
 * A `running` row can never legitimately survive a process. Left alone it would
 * be a job that reads as in-flight forever — the silent-loss failure this whole
 * module is written against — so every one of them is closed with the reason
 * WRITTEN DOWN, and then judged:
 *
 *   young enough (agentJobs.retryGraceMinutes) and never retried → re-queued
 *   otherwise                                                    → stays interrupted
 *
 * Exactly one retry, ever, bounded by `attempts`. Safe to retry at all only
 * because every job in this phase is read-only; the day a job can write
 * something, this is the line that has to be revisited first.
 */
function sweepInterrupted({ now = new Date() } = {}) {
  const db = getSqliteDb();
  if (!db) return { closed: 0, requeued: 0 };
  const open = db.prepare("SELECT * FROM agent_jobs WHERE status = 'running'").all();
  if (!open.length) return { closed: 0, requeued: 0 };

  const c = cfg();
  const graceMs = c.retryGraceMinutes * 60 * 1000;
  let requeued = 0;

  for (const j of open) {
    const startedMs = j.started_at ? new Date(j.started_at).getTime() : 0;
    const age = now.getTime() - startedMs;
    // NEVER re-run a job that writes to disk. CLAUDE.md called this line out
    // in advance - "the retry is only safe because jobs are read-only, the day
    // one can write, that is the first line to revisit" - and a dispatched
    // coding job is that day. A killed run may have already edited files and
    // committed a restore point; running the brief again would apply it on top
    // of its own half-finished work.
    const writesToDisk = j.source === require('./coding-jobs').SOURCE;
    // AND NEVER RE-RUN A READ WHOSE READER HAS GONE. An in-turn lookup exists
    // to answer a conversation that was happening when it started; the restart
    // that killed it also killed the turn waiting on it, so a retry would spend
    // a model call producing a digest with nobody on the other end. It is
    // closed with the reason, like everything else here — a row that vanishes
    // is the failure this whole function exists to refuse — and the entity has
    // already been told, in the turn itself, that the lookup did not come back.
    const inTurn = IN_TURN_SOURCES.includes(j.source);
    const retryable = !writesToDisk && !inTurn && (j.attempts || 0) < c.maxAttempts && startedMs > 0 && age <= graceMs;
    // THE CHECKPOINT IS WHAT A RESTART COULD NOT TAKE. Written after every
    // tool result, so a run killed in round eight resumes at round eight —
    // or, if it is not run again, its card says what it had rather than
    // nothing.
    const ck = readCheckpoint(j.id);
    const hasRecord = !!(ck && Array.isArray(ck.convo) && ck.convo.length);

    if (retryable) {
      db.prepare("UPDATE agent_jobs SET status = 'queued', started_at = NULL, resume_mode = ? WHERE id = ?")
        .run(hasRecord ? 'restart' : null, j.id);
      requeued++;
      const line = `Background job ${j.id.slice(0, 8)} ("${j.title}") was interrupted by a restart ${Math.round(age / 60000)} minute(s) in. ` +
        (hasRecord ? `It picks up from its last completed step (${(ck.toolCalls || []).length} tool call(s) kept) — this is its last attempt.` : `It is being run again from the start — this is its last attempt.`);
      console.warn(`[AgentJobs] ${line}`);
      opsLog(line);
    } else {
      const why = inTurn
        ? NOT_RERUN.ANSWERS_NOBODY
        : writesToDisk
        ? NOT_RERUN.WRITES_TO_DISK
        : (j.attempts || 0) >= c.maxAttempts
        ? NOT_RERUN.ALREADY_RETRIED
        : NOT_RERUN.tooOld(c.retryGraceMinutes);
      // What it had goes on the card, labelled partial, from the record on
      // disk. No model call here: this runs at boot, before the engine may be
      // back, and the mechanical account cannot fail.
      const calls = hasRecord && Array.isArray(ck.toolCalls) ? ck.toolCalls : [];
      const partialText = hasRecord && calls.length ? mechanicalAccount(j, calls, ck.session ? { ...ck.session, exhausted: null } : null, c, 'SNH restarted') : null;
      finish(j.id, {
        status: 'interrupted', error: why, toolCalls: calls.length || j.tool_calls || 0,
        resultText: partialText, budget: hasRecord && ck.session ? { ...ck.session, exhausted: null } : null,
        stopSource: 'runner', stopKind: 'service-restart'
      });
      const line = `Background job ${j.id.slice(0, 8)} ("${j.title}"): ${why}.${calls.length ? ` Its ${calls.length} tool call(s) are on the card as partial output.` : ''}`;
      console.warn(`[AgentJobs] ${line}`);
      opsLog(line);
    }
  }
  return { closed: open.length, requeued };
}

/**
 * Drop terminal rows past the retention window.
 *
 * The panel is a panel, not an archive: what each run did is in the ops log, and
 * a list nobody can reach the bottom of is its own kind of hiding. Queued and
 * running rows are never touched, however old — an old queued row is a bug to
 * see, not a row to sweep.
 */
function prune({ now = new Date() } = {}) {
  const db = getSqliteDb();
  if (!db) return 0;
  const cutoff = new Date(now.getTime() - cfg().retentionDays * 24 * 60 * 60 * 1000).toISOString();
  const old = db.prepare(
    `SELECT id FROM agent_jobs WHERE status IN (${TERMINAL.map(() => '?').join(',')}) AND datetime(finished_at) < datetime(?)`
  ).all(...TERMINAL, cutoff);
  const res = db.prepare(
    `DELETE FROM agent_jobs WHERE status IN (${TERMINAL.map(() => '?').join(',')}) AND datetime(finished_at) < datetime(?)`
  ).run(...TERMINAL, cutoff);
  // The record on disk goes with the row: it was kept for a retry's brief, and
  // there is no row left to retry.
  for (const r of old) dropCheckpoint(r.id);
  if (res.changes) console.log(`[AgentJobs] pruned ${res.changes} job(s) finished before ${cutoff}`);
  return res.changes;
}

/**
 * Boot: close what the restart killed, restart what was waiting, prune the tail.
 *
 * Called once from startup, beside the scheduler's own sweep. Queued rows are
 * the easy half — they are just rows, so they survive a restart untouched and
 * only need launching again.
 */
function startup() {
  if (!getSqliteDb()) return { closed: 0, requeued: 0, resumed: 0, pruned: 0 };
  const swept = sweepInterrupted();
  const pruned = prune();
  const resumed = cfg().enabled ? drain() : 0;
  if (swept.closed || resumed) {
    console.log(`[AgentJobs] startup: ${swept.closed} interrupted (${swept.requeued} re-queued), ${resumed} launched`);
  }
  if (!cfg().enabled) console.log('[AgentJobs] startup: queue disabled in config — nothing launched');
  return { ...swept, resumed, pruned };
}

// ---------------------------------------------------------------------------
// Reading — the panel, and the entity's turn-start handoff
// ---------------------------------------------------------------------------

/**
 * Jobs this conversation created since `sinceIso` — the phantom-dispatch check.
 *
 * Scoped to BOTH the conversation and the turn window on purpose: a job started
 * ten minutes ago in another conversation must not vouch for a claim made in
 * this one.
 *
 * @returns {Array<{id: string, title: string, status: string}>}
 */
function jobsStartedInTurn(conversationId, sinceIso) {
  const db = getSqliteDb();
  if (!db || !conversationId || !sinceIso) return [];
  return db.prepare(
    'SELECT id, title, status FROM agent_jobs WHERE conversation_id = ? AND datetime(created_at) >= datetime(?)'
  ).all(conversationId, sinceIso);
}

/** Cancel a job. Queued only, and the refusal says why. */
function cancel(id) {
  const job = getJob(id);
  if (!job) return { ok: false, error: 'No such job.' };
  if (job.status === 'queued') {
    finish(id, { status: 'cancelled', error: 'cancelled before it started' });
    opsLog(`Background job cancelled before it started: "${job.title}" (${id.slice(0, 8)}).`);
    return { ok: true };
  }
  if (job.status === 'running') {
    // Honest refusal. An in-flight model call cannot be cleanly cancelled, and a
    // button that claimed otherwise would leave the run going while the panel
    // said it had stopped.
    return { ok: false, error: 'It is already running, and a run in progress cannot be stopped cleanly — it will finish and land in the panel.' };
  }
  if (job.status === 'paused') {
    // A paused job is holding nothing; "cancel" here means "no more" — it
    // writes up what it has, exactly as a no in the conversation would.
    return declineMore(id, { via: 'panel' });
  }
  return { ok: false, error: `It has already finished (${job.status}).` };
}

// ---------------------------------------------------------------------------
// The retry, and the two answers to a budget ask
// ---------------------------------------------------------------------------

/** Sources a person may retry from the card. A coding job re-runs from the conversation, where the restore-point rules live. */
const RETRYABLE_SOURCES = ['chat-handoff'];

/**
 * RETRY FROM THE CARD: the same task, a NEW row, and a brief that carries the
 * last attempt's reason and partial output so it does not start from zero.
 *
 * A new row on purpose — the old one holds the record of what happened and is
 * never rewritten. The two point at each other (`retry_of` / `retried_by`) so
 * the panel draws the chain. Her action, so it does not count against the
 * entity's starts-per-hour; the queue-depth cap still applies because that one
 * bounds the machine.
 */
function retry(id) {
  const job = getJob(id);
  if (!job) return { ok: false, error: 'No such job.' };
  if (!TERMINAL.includes(job.status)) return { ok: false, error: `It has not finished (${job.status}) — there is nothing to retry yet.` };
  if (!RETRYABLE_SOURCES.includes(job.source || 'chat-handoff')) {
    return { ok: false, error: job.source === require('./coding-jobs').SOURCE
      ? 'A coding job is re-run from the conversation, not from here — it edits files, and the restore-point rules live there.'
      : `A ${job.source} job cannot be retried from the panel.` };
  }
  if (job.status === 'ok') return { ok: false, error: 'It finished — there is nothing to retry. Ask for it again in conversation if you want it run afresh.' };
  if (job.retried_by) {
    const later = getJob(job.retried_by);
    return { ok: false, error: later && !TERMINAL.includes(later.status)
      ? `It is already being retried (${later.status}).`
      : 'It was already retried — retry the newer attempt instead, which carries this one.' };
  }
  const started = enqueue({
    title: job.title, task: job.task, why: job.why,
    conversationId: job.conversation_id, messageId: job.message_id,
    source: job.source || 'chat-handoff',
    countsAgainstStarts: false,
    retryOf: job.id
  });
  if (!started.ok) return started;
  const db = getSqliteDb();
  db.prepare('UPDATE agent_jobs SET retried_by = ? WHERE id = ?').run(started.id, job.id);
  opsLog(`Background job retried from the panel: "${job.title}" (${job.id.slice(0, 8)} → ${started.id.slice(0, 8)}), attempt ${(job.attempts || 1) + 1}. The new run carries the last one's reason and partial output.`);
  return { ok: true, id: started.id };
}

/**
 * SHE SAID YES. The grant is what the ask promised (or the number she named),
 * the row goes back on the queue in `granted` mode, and the resume reads the
 * checkpoint. Not a new attempt in her eyes — the same job, continuing.
 */
function grantMore(id, { calls = null, via = 'conversation' } = {}) {
  const job = getJob(id);
  if (!job) return { ok: false, error: 'No such job.' };
  if (job.status !== 'paused') return { ok: false, error: `It is not waiting on an answer (${job.status}).` };
  const ask = parseAsk(job.ask_json) || {};
  const grant = { ...(ask.grant || { calls: 10, rounds: 4, wallMs: 300000 }) };
  if (Number.isFinite(calls) && calls > 0) {
    grant.calls = Math.min(Math.floor(calls), cfg().maxToolCallsPerJob * 3);
    grant.rounds = Math.max(grant.rounds, Math.ceil(grant.calls / 2));
  }
  ask.answeredAt = new Date().toISOString();
  ask.decision = 'yes';
  ask.grant = grant;
  ask.answeredVia = via;
  const db = getSqliteDb();
  db.prepare("UPDATE agent_jobs SET status = 'queued', resume_mode = 'granted', ask_json = ?, started_at = NULL WHERE id = ?")
    .run(JSON.stringify(ask), id);
  settleAskDelivery(job, ask, 'yes');
  const line = `Background job "${job.title}" (${id.slice(0, 8)}): Ellie said YES to more budget (+${grant.calls} calls, +${grant.rounds} rounds, +${Math.round(grant.wallMs / 60000)} min). Resuming from where it paused.`;
  console.log(`[AgentJobs] ${line}`);
  opsLog(line);
  launch(id);
  return { ok: true, grant };
}

/** SHE SAID NO. One writeup turn from the checkpoint, then partial. */
function declineMore(id, { via = 'conversation' } = {}) {
  const job = getJob(id);
  if (!job) return { ok: false, error: 'No such job.' };
  if (job.status !== 'paused') return { ok: false, error: `It is not waiting on an answer (${job.status}).` };
  const ask = parseAsk(job.ask_json) || {};
  ask.answeredAt = new Date().toISOString();
  ask.decision = 'no';
  ask.answeredVia = via;
  const db = getSqliteDb();
  db.prepare("UPDATE agent_jobs SET status = 'queued', resume_mode = 'declined', ask_json = ?, started_at = NULL WHERE id = ?")
    .run(JSON.stringify(ask), id);
  settleAskDelivery(job, ask, 'no');
  const line = `Background job "${job.title}" (${id.slice(0, 8)}): Ellie said NO to more budget. It will write up what it has and finish as partial.`;
  console.log(`[AgentJobs] ${line}`);
  opsLog(line);
  launch(id);
  return { ok: true };
}

/** The bell item that pointed at the ask is decided, not left ringing. */
function settleAskDelivery(job, ask, decision) {
  try { require('./job-budget-ask').settle(job, ask, decision); }
  catch (err) { console.warn(`[AgentJobs] could not settle the ask's bell item: ${err.message}`); }
}

/**
 * The paused job (if any) whose ask is waiting on THIS conversation — what the
 * chat route checks before it decides whether her message is an answer.
 */
function pendingAsk(conversationId) {
  const db = getSqliteDb();
  if (!db || !conversationId) return null;
  const rows = db.prepare("SELECT * FROM agent_jobs WHERE status = 'paused' ORDER BY datetime(paused_at) ASC").all();
  for (const j of rows) {
    const ask = parseAsk(j.ask_json) || {};
    const convId = (ask.delivery && ask.delivery.conversationId) || j.conversation_id;
    if (convId === conversationId && !ask.decision) return { job: j, ask };
  }
  return null;
}

/** Every job waiting on an answer, for the panel and the live block. */
function pausedJobs() {
  const db = getSqliteDb();
  if (!db) return [];
  return db.prepare("SELECT * FROM agent_jobs WHERE status = 'paused' ORDER BY datetime(paused_at) ASC").all()
    .map(j => ({ ...j, ask: parseAsk(j.ask_json) }));
}

/** Mark a job read by Ellie in the panel. */
function markSeen(id) {
  const db = getSqliteDb();
  if (!db) return false;
  const res = db.prepare('UPDATE agent_jobs SET seen_at = ? WHERE id = ? AND seen_at IS NULL')
    .run(new Date().toISOString(), id);
  return res.changes > 0;
}

/** Mark a scheduled run read by Ellie in the panel. */
function markRunSeen(runId) {
  const db = getSqliteDb();
  if (!db) return false;
  const res = db.prepare('UPDATE job_runs SET seen_at = ? WHERE id = ? AND seen_at IS NULL')
    .run(new Date().toISOString(), runId);
  return res.changes > 0;
}

/**
 * The panel feed: handed-off jobs and scheduled-job runs, newest first.
 *
 * TWO SOURCES, NOT TWO COPIES. A scheduled run already holds its own output in
 * job_runs.output_text; copying that into this table on completion would make
 * the same result exist twice and let the copies disagree. The feed composes
 * instead, and each item says which kind it is.
 *
 * Scheduled runs that never executed — `deferred`, `skipped` — are not results
 * and are not in the feed. They remain fully visible where they belong, in the
 * run log the activity panel and memory_jobs read.
 */
function feed({ limit = 50 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  const lim = Math.min(Math.max(1, limit), 200);

  const jobs = db.prepare(
    `SELECT * FROM agent_jobs WHERE ${inTurnSql()} ORDER BY datetime(created_at) DESC LIMIT ?`
  ).all(lim).map(j => ({
    kind: 'handoff',
    id: j.id,
    title: j.title,
    task: j.task,
    why: j.why,
    status: j.status,
    created_at: j.created_at,
    started_at: j.started_at,
    finished_at: j.finished_at,
    duration_ms: j.duration_ms,
    result_text: j.result_text,
    error: j.error,
    tool_calls: j.tool_calls,
    attempts: j.attempts,
    seen_at: j.seen_at,
    conversation_id: j.conversation_id,
    cancellable: j.status === 'queued',
    // Why it stopped and who stopped it — see db/job-failure.js.
    stop_source: j.stop_source,
    stop_kind: j.stop_kind,
    // The attempt chain and the ask, for the card.
    retry_of: j.retry_of,
    retried_by: j.retried_by,
    retryable: TERMINAL.includes(j.status) && j.status !== 'ok' && RETRYABLE_SOURCES.includes(j.source || 'chat-handoff') && !j.retried_by,
    paused_at: j.paused_at,
    ask: (() => { const a = parseAsk(j.ask_json); return a ? { text: a.text, grant: a.grant, decision: a.decision, answeredAt: a.answeredAt, conversationId: a.delivery && a.delivery.conversationId, near: a.near } : null; })(),
    // A failed or interrupted job with text on it is offering PARTIAL output.
    partial_output: !!((j.status === 'failed' || j.status === 'interrupted') && (j.result_text || '').trim()),
    // What it produced as a file. `artifact_path` is deliberately NOT here: the
    // panel has no use for a server path it cannot open, and the download route
    // looks it up by job id rather than being handed one. `artifact_location` is
    // the folder, which IS worth showing — it is where the file will still be
    // when this row has been pruned.
    artifact_kind: j.artifact_kind,
    artifact_name: j.artifact_name,
    artifact_bytes: j.artifact_bytes,
    artifact_error: j.artifact_error,
    artifact_location: j.artifact_path ? path.dirname(j.artifact_path) : null,
    summary_text: j.summary_text
  }));

  const RS = runResultStatuses();
  const runs = db.prepare(`
    SELECT r.*, c.description AS job_description, c.schedule AS job_schedule
    FROM job_runs r LEFT JOIN cron_jobs c ON c.id = r.job_id
    WHERE r.status IN (${RS.map(() => '?').join(',')})
    ORDER BY datetime(r.started_at) DESC LIMIT ?
  `).all(...RS, lim).map(r => ({
    kind: 'scheduled',
    id: r.id,
    title: r.job_description || 'a scheduled job',
    task: r.job_description,
    why: r.job_schedule ? `runs on the schedule "${r.job_schedule}"` : null,
    status: r.status,
    created_at: r.started_at,
    started_at: r.started_at,
    finished_at: r.finished_at,
    duration_ms: r.duration_ms,
    result_text: r.output_text,
    error: r.error,
    tool_calls: r.tool_calls,
    attempts: null,
    seen_at: r.seen_at,
    conversation_id: null,
    cancellable: false,
    stop_source: r.status === 'ok' ? null : 'runner',
    stop_kind: null,
    retry_of: null, retried_by: null, retryable: false, paused_at: null, ask: null, partial_output: false,
    // A scheduled run produces no file. It is a digest that arrives on a
    // cadence, and one PDF per firing would silt up the documents folder with a
    // hundred near-identical reports nobody asked for. Named explicitly rather
    // than left undefined, so the card renders one way for both kinds.
    artifact_kind: null,
    artifact_name: null,
    artifact_bytes: null,
    artifact_error: null,
    artifact_location: null,
    summary_text: null
  }));

  return [...jobs, ...runs]
    .sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0))
    .slice(0, lim);
}

/** Counts for the panel badge: unread results, and work still going. */
function counts() {
  const db = getSqliteDb();
  if (!db) return { unseen: 0, active: 0, total: 0 };
  const unseenJobs = db.prepare(
    `SELECT COUNT(*) n FROM agent_jobs WHERE seen_at IS NULL AND status IN (${TERMINAL.map(() => '?').join(',')})
       AND ${inTurnSql()}`
  ).get(...TERMINAL).n;
  const RS3 = runResultStatuses();
  const unseenRuns = db.prepare(
    `SELECT COUNT(*) n FROM job_runs WHERE seen_at IS NULL AND status IN (${RS3.map(() => '?').join(',')})`
  ).get(...RS3).n;
  // Counted here rather than via activeCount(), which deliberately still counts
  // EVERYTHING: that one bounds the machine (the queue-depth cap), and an
  // in-turn read occupies the machine like anything else. This one feeds the
  // badge beside a panel that does not list them, and a badge promising a card
  // that is not there is worse than no badge.
  const active = db.prepare(
    `SELECT COUNT(*) n FROM agent_jobs WHERE status IN ('queued','running') AND ${inTurnSql()}`
  ).get().n;
  // Paused is its own number: not running, not a result — waiting on her, in
  // a conversation. The badge does not count it; the panel says it.
  const paused = db.prepare(`SELECT COUNT(*) n FROM agent_jobs WHERE status = 'paused' AND ${inTurnSql()}`).get().n;
  return { unseen: unseenJobs + unseenRuns, active, paused, total: unseenJobs + unseenRuns + active };
}

/**
 * What finished since he was last told — the chat-awareness half.
 *
 * Returns items and DOES NOT STAMP THEM. Stamping is markAnnounced(), called by
 * the chat route only once the block is in the message actually being sent.
 * That split is the correction-notice rule and it is load-bearing: a job stamped
 * as announced by a block that was then trimmed, or by a request that then
 * failed, is a result he is never told about again.
 *
 * Both sources, sharing one cap: a scheduled digest and a job he started are the
 * same kind of thing from his side — work of his that ran while he was not
 * looking. Newest first, so what he hears about is what just happened.
 */
function pendingAnnouncements({ limit = 3 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  const lim = Math.min(Math.max(1, limit), 10);

  const jobs = db.prepare(`
    SELECT id, title, status, finished_at, result_text, error, duration_ms,
           artifact_kind, artifact_name, summary_text
    FROM agent_jobs
    WHERE announced_at IS NULL AND status IN (${TERMINAL.map(() => '?').join(',')})
      AND ${inTurnSql()}
    ORDER BY datetime(finished_at) DESC LIMIT ?
  `).all(...TERMINAL, lim).map(j => ({
    kind: 'handoff', id: j.id, title: j.title, status: j.status,
    finished_at: j.finished_at, text: j.result_text, error: j.error, duration_ms: j.duration_ms,
    artifact_kind: j.artifact_kind, artifact_name: j.artifact_name
  }));

  // UNDELIVERED IN-TURN DIGESTS — the late-delivery path (v1.1).
  //
  // These are excluded from the block by `inTurnSql` above and re-admitted here
  // on one condition: the answer was never delivered. A history_search that came
  // back inside its wait stamped `delivered_at` and is done; one that outlived
  // the wait left a complete, verified digest on a row nobody can reach, while
  // the entity told Ellie she would report back. That happened on the first
  // live run, and the digest is still sitting there.
  //
  // They ride the ANNOUNCEMENT channel rather than getting one of their own,
  // and that is the deliberate choice among the three that were open. A
  // system-injected turn would have the queue speak into a conversation, which
  // is the exact boundary this file is built around (ROBOT, NOT BELL) — a
  // finished job does not get to open its mouth. A bespoke next-turn injection
  // would be a second mechanism doing what this one already does: hold a
  // finished result until the next time she speaks, hand it to him once, stamp
  // it, and never repeat it. This channel already has the stamping, the token
  // cap, the "she has not seen this" framing and the rule that he relays it in
  // his own words if it is worth saying. The only thing it needed was to know
  // this kind is different — see renderAnnouncementBlock.
  //
  // `announced_at IS NULL` still gates it, so a digest is handed over exactly
  // once whether or not he chooses to say it out loud.
  const lateDigests = db.prepare(`
    SELECT id, title, task, status, finished_at, result_text, error, duration_ms
    FROM agent_jobs
    WHERE announced_at IS NULL AND delivered_at IS NULL
      AND status IN (${TERMINAL.map(() => '?').join(',')})
      AND source IN (${IN_TURN_SOURCES.map(() => '?').join(',')})
    ORDER BY datetime(finished_at) DESC LIMIT ?
  `).all(...TERMINAL, ...IN_TURN_SOURCES, lim).map(j => ({
    kind: 'history', id: j.id, title: j.title, question: j.task, status: j.status,
    finished_at: j.finished_at, text: j.result_text, error: j.error, duration_ms: j.duration_ms
  }));

  const RS2 = runResultStatuses();
  const runs = db.prepare(`
    SELECT r.id, r.status, r.finished_at, r.output_text, r.error, r.duration_ms,
           c.description AS job_description
    FROM job_runs r LEFT JOIN cron_jobs c ON c.id = r.job_id
    WHERE r.announced_at IS NULL AND r.status IN (${RS2.map(() => '?').join(',')})
    ORDER BY datetime(r.finished_at) DESC LIMIT ?
  `).all(...RS2, lim).map(r => ({
    kind: 'scheduled', id: r.id, title: r.job_description || 'a scheduled job', status: r.status,
    finished_at: r.finished_at, text: r.output_text, error: r.error, duration_ms: r.duration_ms
  }));

  // Late digests first, unconditionally. They are an answer he PROMISED her and
  // could not give; a scheduled cluster audit can wait a turn.
  return [...lateDigests, ...[...jobs, ...runs]
    .sort((a, b) => new Date(b.finished_at || 0) - new Date(a.finished_at || 0))]
    .slice(0, lim);
}

/**
 * What is running or waiting RIGHT NOW — the live view he otherwise does not have.
 *
 * WHY THIS EXISTS. On 2026-08-18 Ellie asked "Are you still working on this?"
 * and got a detailed progress report: one job "slowed by a search connection
 * issue I am working through", another "scanning a large volume of memory and
 * logs". Both invented. Every job he had was already finished, and one of them
 * had never existed at all. He had no way to see the queue — the only chat-side
 * view is the announcement block, which by construction shows FINISHED work —
 * and nothing told him he could not see it, so he produced the plausible thing.
 *
 * This is the same failure as the capability manifest: not lying, exactly, but
 * having no ground truth and filling the gap. The fix is the same shape — give
 * him the true state, and say plainly what its absence means.
 *
 * A LINE, NOT A TOOL. A tool only helps on a turn that reaches the tool loop,
 * and "Are you still working on this?" trips no classifier; under the old gate
 * that turn was DIRECT, and a status tool would have been exactly as absent as
 * the handoff tool was. This renders into the per-turn context instead, so it is
 * there whether or not he thinks to ask for it.
 *
 * ZERO TOKENS WHEN NOTHING IS ACTIVE — returns null, and the absence is itself
 * the signal the standing instruction refers to.
 *
 * @returns {{text: string, running: number, queued: number}|null}
 */
function renderActiveJobsBlock() {
  const db = getSqliteDb();
  if (!db) return null;
  const rows = db.prepare(
    `SELECT id, title, status, created_at, started_at, paused_at, ask_json FROM agent_jobs
     WHERE status IN ('queued','running','paused') AND ${inTurnSql()} ORDER BY datetime(created_at) ASC`
  ).all();
  if (!rows.length) return null;

  const now = Date.now();
  const elapsed = (iso) => {
    if (!iso) return '';
    const ms = now - new Date(iso).getTime();
    if (ms < 0) return 'just now';
    const m = Math.floor(ms / 60000);
    return m < 1 ? `${Math.max(1, Math.round(ms / 1000))}s so far` : `${m} min so far`;
  };

  const running = rows.filter(r => r.status === 'running');
  const queued = rows.filter(r => r.status === 'queued');
  const paused = rows.filter(r => r.status === 'paused');
  const lines = rows.map(r => r.status === 'running'
    ? `- RUNNING: "${r.title}" (${elapsed(r.started_at)})`
    : r.status === 'paused'
      ? `- PAUSED, WAITING ON HER: "${r.title}" — it stopped near its budget with work left and asked her, in the conversation that started it, whether to give it more. It does not move until she answers yes or no there.`
      : `- WAITING TO START: "${r.title}"`);

  const text =
    '=== Your Background Jobs, Right Now ===\n' +
    `${running.length} running, ${queued.length} waiting to start${paused.length ? `, ${paused.length} paused waiting on her answer` : ''}.\n` +
    lines.join('\n') + '\n' +
    'This is the whole picture and it is live as of this message. You can see THAT they are ' +
    'running and for how long; you cannot see how far along one is, what it has found so far, or ' +
    'why it is taking the time it is. Do not describe progress you cannot see. When one finishes ' +
    'you are told what it found at the top of a reply — until then the honest answer about its ' +
    'contents is that you do not know yet.';

  return { text, running: running.length, queued: queued.length, paused: paused.length };
}

/**
 * THE SECTION HEADING EACH KIND RENDERS UNDER, derived from the text above
 * rather than retyped. This is what "was it really in the message" is answered
 * with, and retyping it is precisely how the answer went wrong: the caller held
 * its own copy of one heading and a whole section rendered under the other.
 */
// Computed on CALL, not at load: ANNOUNCEMENT is declared further down this
// file, and reading it from a top-level const here is a temporal-dead-zone
// crash on require. Lazy also means the marker can never drift from the text —
// it is re-derived from whatever the heading currently says.
function markers() {
  return {
    history: ANNOUNCEMENT.LATE_HEADER.split('\n')[0],
    default: ANNOUNCEMENT.HEADER.split('\n')[0]
  };
}
function markerFor(item) {
  const m = markers();
  return item && item.kind === 'history' ? m.history : m.default;
}

/**
 * Stamp announcements as delivered. Called only once the block is really in the
 * request — see confirmAnnounced, which is what callers should use.
 *
 * ALL OR NOTHING. One transaction over every item, so a stamp that throws
 * halfway does not leave some jobs marked and the rest not: either the whole
 * batch is recorded as handed over or none of it is, and a batch that failed
 * comes back around next turn intact. A partially-stamped batch is the one
 * outcome with no honest recovery — the unstamped half repeats and the stamped
 * half is lost, and nothing afterwards can tell which was which.
 *
 * `AND announced_at IS NULL` is the idempotency guard: stamping twice writes
 * nothing the second time and reports 0 changes, so a caller can never inflate
 * a re-delivery into a fresh one.
 *
 * @returns {number} rows actually stamped; 0 if the write failed
 */
function markAnnounced(items = []) {
  const db = getSqliteDb();
  if (!db || !items.length) return 0;
  const now = new Date().toISOString();
  const job = db.prepare('UPDATE agent_jobs SET announced_at = ? WHERE id = ? AND announced_at IS NULL');
  const run = db.prepare('UPDATE job_runs SET announced_at = ? WHERE id = ? AND announced_at IS NULL');
  try {
    return db.transaction(() => {
      let n = 0;
      for (const it of items) {
        n += (it.kind === 'scheduled' ? run : job).run(now, it.id).changes;
      }
      return n;
    })();
  } catch (err) {
    // The transaction rolled back, so nothing is stamped and every item will be
    // offered again. Reporting 0 rather than a partial count is the whole point:
    // a job whose mark did not land has not been delivered.
    console.error('[AgentJobs] markAnnounced ROLLED BACK — nothing stamped, all will be re-offered:', err.message);
    return 0;
  }
}

/**
 * Confirm the announcement block reached the assembled message, and stamp only
 * what actually did.
 *
 * WHY THIS EXISTS, AND WHY IT IS NOT A STRING SEARCH IN THE CALLER. The rule is
 * right and stays: stamp AFTER the injection ceiling, and only if the block
 * survived, because a job stamped by a block that was then trimmed is a result
 * he is never told about again. The caller implemented that rule by searching
 * the message for one hard-coded heading — and the late-digest section renders
 * under a different one, and deliberately suppresses the first when it is alone.
 * So for that whole class the answer was always "it did not survive", the stamp
 * never ran, and one Aug 27 lookup about Ellie's dogs was handed to him as news
 * on every turn for days, in five different conversations.
 *
 * The check now belongs to the module that owns the headings, and it is
 * PER ITEM: each is confirmed against the heading its own kind renders under.
 * A trimmed section skips only its own items instead of silencing or repeating
 * the rest.
 *
 * @param {Array} items - what renderAnnouncementBlock returned
 * @param {string} assembledText - the final system message, after the ceiling
 * @returns {{stamped:number, skipped:number, missing:string[]}}
 */
function confirmAnnounced(items = [], assembledText = '') {
  if (!items.length) return { stamped: 0, skipped: 0, missing: [] };
  const text = String(assembledText || '');
  const present = [], missing = [];
  for (const it of items) (text.includes(markerFor(it)) ? present : missing).push(it);
  return {
    stamped: present.length ? markAnnounced(present) : 0,
    skipped: missing.length,
    missing: missing.map(m => m.id)
  };
}

/**
 * THE WORDING OF THE ANNOUNCEMENT, IN ONE PLACE.
 *
 * These sentences are the whole mechanism of two rules: that he is told a job's
 * results are UNREAD, and that a job which stopped short is announced as having
 * stopped short rather than as having produced nothing. Both are properties of
 * the words, so the tests have to read the words — and a test that keeps its own
 * copy of them tests the copy. Rewording here changes the test in the same edit,
 * or changes neither.
 */
const ANNOUNCEMENT = {
  HEADER:
    '=== Background Work That Finished ===\n' +
    'These are your own jobs — work you handed off, or a scheduled job of yours — that finished since you ' +
    'last spoke with her. They landed in her jobs panel, which does not notify her, so assume she has NOT ' +
    'read them.\n',
  // THE LATE-DIGEST SECTION'S HEADER LIVES HERE TOO, and that is the fix for a
  // real bug rather than tidiness. It used to be a literal inside
  // renderAnnouncementBlock, and server.js confirmed delivery by searching the
  // assembled message for the ORDINARY header — which a late-digest-only block
  // deliberately never emits. So the confirmation never fired, the stamp was
  // never written, and one finished lookup was handed to him again on every
  // single turn. Both headers are here so `MARKER` below can be DERIVED from
  // the text that actually renders; a third section added later gets a marker
  // by construction instead of silently re-opening the same hole.
  LATE_HEADER:
    '=== The Lookup You Were Waiting On ===\n' +
    'You called history_search earlier, it did not come back inside the turn, and you said you would ' +
    'return to it. This is what it found. She has NOT seen any of this — it went nowhere except here.\n' +
    'Tell her now, in your own words, and say which question it answers, because the conversation may ' +
    'have moved on since you asked. The quoted lines are verified verbatim from her own messages and ' +
    'yours; everything else is the lookup\'s framing. If it says nothing was found, tell her that — ' +
    'do not fill it in.\n',
  FOOTER:
    '\nIf one of these is worth leading with, say it in your own words — what you found, not that a job ran. ' +
    'If none of it matters to what she just said, let it go; they are already recorded and you are not ' +
    'obliged to report them. Do not claim a result you cannot see here.',
  /** A run that finished with nothing to show. */
  noResult: (error) => `It did not produce a result. What went wrong: ${error || 'unrecorded'}.`,
  /** A run that produced something and then ran out — the text, and the fact it is partial. */
  stoppedShort: (text, error) =>
    `${text}\n  (This one stopped short of finishing: ${error || 'reason unrecorded'}. What is above is what it had.)`,
};

/**
 * Render the announcement block for injection.
 *
 * Says what happened and, explicitly, what it is NOT: a result is not a message
 * she has read, and it is not a thing he has to raise. Both halves have a
 * failure behind them — an entity that assumes she saw it says nothing, and an
 * entity that treats every finished job as news reports its own housekeeping at
 * her.
 *
 * @returns {{text: string, items: Array, tokens: number}|null}
 */
function renderAnnouncementBlock({ limit = 3, tokenCap = 400 } = {}) {
  const { estTokens } = require('./injection-budget');
  const items = pendingAnnouncements({ limit });
  if (!items.length) return null;

  const header = ANNOUNCEMENT.HEADER;
  const footer = ANNOUNCEMENT.FOOTER;

  // A LATE DIGEST IS NOT "BACKGROUND WORK THAT FINISHED", and telling him it is
  // would be wrong in both directions: the header above says it landed in her
  // jobs panel (this did not — it is not in her panel at all) and that she has
  // probably not read it (she has not, but more importantly she is WAITING for
  // it, because he said he would come back to her). So this kind gets its own
  // section, its own framing, and its own token allowance, and the digest goes
  // in WHOLE — it is already capped at digestChars, and half of a set of quotes
  // is not a smaller answer, it is a worse one.
  const late = items.filter(i => i.kind === 'history');
  const rest = items.filter(i => i.kind !== 'history');

  const lateLines = late.map(it => {
    const when = formatLocalTime(it.finished_at, { fallback: 'recently' });
    const body = String(it.text || '').trim() ||
      `It produced nothing. What went wrong: ${it.error || 'unrecorded'}.`;
    return `--- Your question, asked ${when}: "${it.question || it.title}"\n${body}`;
  });
  const lateBlock = late.length
    ? ANNOUNCEMENT.LATE_HEADER + '\n' + lateLines.join('\n\n')
    : '';

  const lines = [];
  const kept = [...late];
  let used = estTokens(header + footer + lateBlock);
  for (const it of rest) {
    const when = formatLocalTime(it.finished_at, { fallback: 'recently' });
    // WHAT HE IS TOLD MATCHES WHAT IS ON THE CARD. Before `partial` existed this
    // read "status is ok, or it produced nothing" — so a run that wrote up half
    // an answer was announced to him as having produced nothing, and he had no
    // way to mention to her a result that was sitting in her panel.
    const text = String(it.text || '').trim();
    const body = it.status === 'ok'
      ? text
      : (text ? ANNOUNCEMENT.stoppedShort(text, it.error) : ANNOUNCEMENT.noResult(it.error));
    const label = it.kind === 'scheduled' ? 'scheduled job' : 'job you started';
    // A result that became a file is announced AS a file. Without this he would
    // be told the text and, asked where it went, would have to guess — and the
    // guess would be wrong, because until this shipped there was nowhere for it
    // to go.
    const filed = it.artifact_name
      ? ` (saved as ${it.artifact_name}${it.artifact_kind === 'pdf' ? ', a PDF' : ''}; she has a link to it on the card, which does not mean she has opened it)`
      : '';
    const line = `- (${label}) "${it.title}" — finished ${when}${filed}: ${body}`;
    const t = estTokens(line);
    // Always deliver at least one, however long: a result too big for the batch
    // would otherwise never be delivered at all. The rest wait for the next turn
    // — nothing expires an unannounced job.
    if (kept.length > 0 && used + t > tokenCap) break;
    lines.push(line);
    kept.push(it);
    used += t;
  }

  // The ordinary block is only rendered when there is something ordinary to put
  // in it — a late digest alone must not drag in a header about her jobs panel.
  const ordinary = lines.length ? header + lines.join('\n') + footer : '';
  const text = [lateBlock, ordinary].filter(Boolean).join('\n\n');
  if (!text) return null;
  return { text, items: kept, tokens: estTokens(text) };
}

module.exports = {
  JOB_TOOLS,
  TERMINAL,
  IN_TURN_SOURCES,
  // Exported for test: the two halves of "a job always writes something" are
  // pure functions, and the floor especially must be provable without a model.
  describeStop,
  mechanicalAccount,
  enqueue,
  runJob,
  attachArtifact,
  drain,
  startup,
  sweepInterrupted,
  prune,
  cancel,
  getJob,
  feed,
  counts,
  markSeen,
  markRunSeen,
  pendingAnnouncements,
  markAnnounced,
  confirmAnnounced,
  markers,
  markerFor,
  renderActiveJobsBlock,
  jobsStartedInTurn,
  renderAnnouncementBlock,
  ANNOUNCEMENT,
  NOT_RERUN,
  activeCount,
  startsLastHour,
  // The retry and the budget ask.
  retry,
  grantMore,
  declineMore,
  pendingAsk,
  pausedJobs,
  RETRYABLE_SOURCES,
  // The record on disk, for tests and the ask module.
  readCheckpoint,
  writeCheckpoint,
  checkpointPath,
  trimToCompleteRound,
  _inFlight: inFlight
};
