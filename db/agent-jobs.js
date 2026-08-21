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
 * resumed. So the loss is made loud instead of silent — sweepInterrupted()
 * closes every in-flight row as `interrupted` WITH THE REASON, re-queues it once
 * if it is still young enough to be worth redoing, and otherwise leaves it in
 * the panel saying what happened. A job never vanishes; that is the whole reason
 * the row is written before the work starts.
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
    retentionDays: Math.max(1, c.retentionDays ?? 90)
  };
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
  return db.prepare('SELECT COUNT(*) n FROM agent_jobs WHERE datetime(created_at) > datetime(?)').get(since).n;
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
function enqueue({ title, task, why = null, conversationId = null, messageId = null, source = 'chat-handoff' } = {}) {
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
  const recent = startsLastHour();
  if (recent >= c.maxStartsPerHour) {
    return refuse('refused-cap', `You have already started ${recent} background jobs in the last hour, which is the limit (${c.maxStartsPerHour}). Nothing was started — say so plainly rather than implying it is running.`, { conversationId });
  }

  const id = randomUUID();
  db.prepare(`
    INSERT INTO agent_jobs (id, title, task, why, status, source, conversation_id, message_id, created_at)
    VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)
  `).run(id, t.slice(0, 200), k, why ? String(why).trim().slice(0, 500) : null,
    source, conversationId, messageId, new Date().toISOString());

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
function finish(id, { status, resultText = null, error = null, toolCalls = 0, budget = null }) {
  const db = getSqliteDb();
  if (!db) return null;
  const job = getJob(id);
  if (!job) return null;
  const finishedAt = new Date();
  const startedAt = job.started_at ? new Date(job.started_at) : finishedAt;
  const durationMs = Math.max(0, finishedAt.getTime() - startedAt.getTime());
  db.prepare(`
    UPDATE agent_jobs
    SET status = ?, finished_at = ?, duration_ms = ?, result_text = ?, error = ?,
        tool_calls = ?, budget_json = ?
    WHERE id = ?
  `).run(status, finishedAt.toISOString(), durationMs, resultText, error,
    toolCalls ?? 0, budget ? JSON.stringify(budget) : null, id);
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

function systemPrompt(job, tools) {
  const now = new Date();
  return (
    `You are Aurelius, running one of your own background jobs. Nobody is in the room. This is not a ` +
    `conversation — it is a job you started during one and then let go of, and what you write goes to ` +
    `Ellie's jobs panel, where she will read it when she is ready.\n\n` +
    `It is ${now.toLocaleString()}.\n\n` +
    `THE JOB, as you set it when you handed it off:\n"${job.task}"\n` +
    (job.why ? `Why you handed it off: "${job.why}"\n` : '') +
    `\n` +
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
async function salvageWriteup(job, calls = [], budget = null, c = cfg()) {
  const mm = memoryManager();
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
function mechanicalAccount(job, calls = [], budget = null, c = cfg(), thrownError = null) {
  const dead = calls.filter(k => k && k.productive === false);
  const lines = [];

  lines.push(`I could not write this up properly, so this is the plain account of what happened.`);
  lines.push('');
  lines.push(`What I set out to do: ${job.task}`);
  lines.push('');

  if (!calls.length) {
    // Careful with this sentence. When the run THREW, the tool record is lost
    // with it — so "it looked nothing up" would be a claim, not a fact, and it
    // would be false whenever the throw came in a later round. Say what is
    // actually known: there is no record.
    lines.push(thrownError
      ? `The run failed and I have no record of what it managed to look up first. The error was: ${thrownError}`
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
  if (thrownError && calls.length) lines.push(`The run then failed with: ${thrownError}`);
  lines.push('');
  lines.push(`Ask me again and I will run it properly.`);

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
  db.prepare(
    "UPDATE agent_jobs SET status = 'running', started_at = ?, attempts = COALESCE(attempts, 0) + 1 WHERE id = ?"
  ).run(startedAt.toISOString(), id);

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
        error: err.message
      };
    }
    return finish(id, {
      status: outcome.status,
      resultText: outcome.resultText,
      error: outcome.error || null,
      toolCalls: outcome.toolCalls || 0
    });
  }

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

  console.log(`[AgentJobs] === running ${id.slice(0, 8)}: "${job.title}" ===`);

  let status = 'ok', error = null, output = '', budget = null, toolCalls = 0;
  let calls = [];
  try {
    const res = await mm.callLLM(
      systemPrompt(job, allowed),
      job.task,
      { maxTokens: c.answerTokens, thinkingTokens: c.thinkingTokens, toolSession: session }
    );
    output = String(res && res.content || '').trim();
    budget = (res && res.budget) || session.summary();
    calls = Array.isArray(res && res.toolCalls) ? res.toolCalls : [];
    toolCalls = calls.length;

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
    // This is the same class as the phantom dispatch — an output that reads as
    // complete and is not — and it gets the same answer: the run does not get to
    // claim it finished, and the card names the limit it hit. Truncation is
    // checked FIRST because it is the most specific reason and the only one that
    // says WHERE the result stops. A run can be out of rounds AND truncated; the
    // cut mid-sentence is what she is looking at, and it is the one with an
    // action attached — raise the budget.
    if (output && (res.truncated || res.outOfRounds || (budget && budget.exhausted))) {
      status = 'partial';
      error = res.truncated
        ? `it hit the answer budget (${c.answerTokens} tokens) and stopped mid-result — what is above is cut off, not finished. Raise "Answer budget, agent jobs" in Settings if this keeps happening`
        : res.outOfRounds
          ? `it ran out of tool rounds (${c.maxRoundsPerJob}) before it was finished — what is above is what it had`
          : `it stopped early: ${budget.exhausted} — what is above is what it had`;
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
      const salvaged = await salvageWriteup(job, calls, budget, c);
      if (salvaged) {
        output = salvaged;
        status = 'partial';
        error = describeStop(calls, budget, c);
        console.warn(`[AgentJobs] ${id.slice(0, 8)} produced no answer — salvaged a writeup from ${toolCalls} tool call(s)`);
      } else {
        output = mechanicalAccount(job, calls, budget, c);
        status = 'partial';
        error = describeStop(calls, budget, c);
        console.warn(`[AgentJobs] ${id.slice(0, 8)} produced no answer and no writeup — wrote the mechanical account instead`);
      }
    }
  } catch (err) {
    status = 'failed';
    error = err && err.message ? err.message : String(err);
    budget = session.summary();
    console.error(`[AgentJobs] ${id.slice(0, 8)} failed:`, error);
    // Even a thrown run writes what it has. The throw is usually the brain being
    // unreachable, and then there is nothing to write up but the attempt — which
    // is still a card that says what happened rather than one that says nothing.
    calls = Array.isArray(calls) ? calls : [];
    toolCalls = calls.length;
    output = mechanicalAccount(job, calls, budget, c, error);
  }

  const done = finish(id, { status, resultText: output || null, error, toolCalls, budget });
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

// ---------------------------------------------------------------------------
// Restart recovery
// ---------------------------------------------------------------------------

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
    const retryable = !writesToDisk && (j.attempts || 0) < c.maxAttempts && startedMs > 0 && age <= graceMs;

    if (retryable) {
      db.prepare("UPDATE agent_jobs SET status = 'queued', started_at = NULL WHERE id = ?").run(j.id);
      requeued++;
      const line = `Background job ${j.id.slice(0, 8)} ("${j.title}") was interrupted by a restart ${Math.round(age / 60000)} minute(s) in. It is being run again — this is its last attempt.`;
      console.warn(`[AgentJobs] ${line}`);
      opsLog(line);
    } else {
      const why = writesToDisk
        ? 'interrupted by a restart. It was NOT run again, because it can write files and re-running the brief could apply it twice. Anything it had already changed is still changed - check git status in the project; it commits a restore point before it starts'
        : (j.attempts || 0) >= c.maxAttempts
        ? 'interrupted by a restart, and it had already been retried once — it was not run again'
        : `interrupted by a restart, and by the time the server came back it was too old to be worth redoing (older than ${c.retryGraceMinutes} minutes) — it was not run again`;
      finish(j.id, { status: 'interrupted', error: why, toolCalls: j.tool_calls || 0 });
      const line = `Background job ${j.id.slice(0, 8)} ("${j.title}"): ${why}.`;
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
  const res = db.prepare(
    `DELETE FROM agent_jobs WHERE status IN (${TERMINAL.map(() => '?').join(',')}) AND datetime(finished_at) < datetime(?)`
  ).run(...TERMINAL, cutoff);
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
  return { ok: false, error: `It has already finished (${job.status}).` };
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
    'SELECT * FROM agent_jobs ORDER BY datetime(created_at) DESC LIMIT ?'
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
    `SELECT COUNT(*) n FROM agent_jobs WHERE seen_at IS NULL AND status IN (${TERMINAL.map(() => '?').join(',')})`
  ).get(...TERMINAL).n;
  const RS3 = runResultStatuses();
  const unseenRuns = db.prepare(
    `SELECT COUNT(*) n FROM job_runs WHERE seen_at IS NULL AND status IN (${RS3.map(() => '?').join(',')})`
  ).get(...RS3).n;
  const active = activeCount();
  return { unseen: unseenJobs + unseenRuns, active, total: unseenJobs + unseenRuns + active };
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
    ORDER BY datetime(finished_at) DESC LIMIT ?
  `).all(...TERMINAL, lim).map(j => ({
    kind: 'handoff', id: j.id, title: j.title, status: j.status,
    finished_at: j.finished_at, text: j.result_text, error: j.error, duration_ms: j.duration_ms,
    artifact_kind: j.artifact_kind, artifact_name: j.artifact_name
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

  return [...jobs, ...runs]
    .sort((a, b) => new Date(b.finished_at || 0) - new Date(a.finished_at || 0))
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
    "SELECT id, title, status, created_at, started_at FROM agent_jobs WHERE status IN ('queued','running') ORDER BY datetime(created_at) ASC"
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
  const lines = rows.map(r => r.status === 'running'
    ? `- RUNNING: "${r.title}" (${elapsed(r.started_at)})`
    : `- WAITING TO START: "${r.title}"`);

  const text =
    '=== Your Background Jobs, Right Now ===\n' +
    `${running.length} running, ${queued.length} waiting to start.\n` +
    lines.join('\n') + '\n' +
    'This is the whole picture and it is live as of this message. You can see THAT they are ' +
    'running and for how long; you cannot see how far along one is, what it has found so far, or ' +
    'why it is taking the time it is. Do not describe progress you cannot see. When one finishes ' +
    'you are told what it found at the top of a reply — until then the honest answer about its ' +
    'contents is that you do not know yet.';

  return { text, running: running.length, queued: queued.length };
}

/** Stamp announcements as delivered. Called only once the block is really in the request. */
function markAnnounced(items = []) {
  const db = getSqliteDb();
  if (!db || !items.length) return 0;
  const now = new Date().toISOString();
  const job = db.prepare('UPDATE agent_jobs SET announced_at = ? WHERE id = ? AND announced_at IS NULL');
  const run = db.prepare('UPDATE job_runs SET announced_at = ? WHERE id = ? AND announced_at IS NULL');
  let n = 0;
  for (const it of items) {
    const res = it.kind === 'scheduled' ? run.run(now, it.id) : job.run(now, it.id);
    n += res.changes;
  }
  return n;
}

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

  const header =
    '=== Background Work That Finished ===\n' +
    'These are your own jobs — work you handed off, or a scheduled job of yours — that finished since you ' +
    'last spoke with her. They landed in her jobs panel, which does not notify her, so assume she has NOT ' +
    'read them.\n';
  const footer =
    '\nIf one of these is worth leading with, say it in your own words — what you found, not that a job ran. ' +
    'If none of it matters to what she just said, let it go; they are already recorded and you are not ' +
    'obliged to report them. Do not claim a result you cannot see here.';

  const lines = [];
  const kept = [];
  let used = estTokens(header + footer);
  for (const it of items) {
    const when = it.finished_at ? new Date(it.finished_at).toLocaleString() : 'recently';
    // WHAT HE IS TOLD MATCHES WHAT IS ON THE CARD. Before `partial` existed this
    // read "status is ok, or it produced nothing" — so a run that wrote up half
    // an answer was announced to him as having produced nothing, and he had no
    // way to mention to her a result that was sitting in her panel.
    const text = String(it.text || '').trim();
    const body = it.status === 'ok'
      ? text
      : (text
        ? `${text}\n  (This one stopped short of finishing: ${it.error || 'reason unrecorded'}. What is above is what it had.)`
        : `It did not produce a result. What went wrong: ${it.error || 'unrecorded'}.`);
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

  const text = header + lines.join('\n') + footer;
  return { text, items: kept, tokens: estTokens(text) };
}

module.exports = {
  JOB_TOOLS,
  TERMINAL,
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
  renderActiveJobsBlock,
  jobsStartedInTurn,
  renderAnnouncementBlock,
  activeCount,
  startsLastHour,
  _inFlight: inFlight
};
