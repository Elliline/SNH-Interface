/**
 * Agent Pool — bounded-concurrency task queue for background LLM work.
 *
 * SNH's background jobs (heartbeat cluster audits, fact salience scoring,
 * contradiction judging, gap detection) all hit the same local vLLM instance
 * (sparky-brain). vLLM serves concurrent requests natively via continuous
 * batching, so running these serially leaves throughput on the table. This
 * pool spools them through a shared promise queue with a concurrency cap.
 *
 * Priority rule — chat is king: while a chat request is in flight the pool
 * throttles to concurrency 1 so the user-facing response keeps the GPU. When
 * the last chat completes, the pool resumes full width and fills freed slots.
 * In-flight background tasks are never preempted (an LLM call can't be cleanly
 * cancelled) — throttling only chokes the *launch* of new tasks.
 *
 * Instrumentation: a "pass" groups the tasks of one logical batch (e.g. one
 * heartbeat cycle). startPass()/endPass() record task counts + wall-time and
 * append a one-line summary to the daily log.
 *
 * Concurrency safety: this module only schedules async *work functions*; it
 * does not touch the database. Callers keep their SQLite mutations sequential
 * (better-sqlite3 is synchronous anyway) and use the pool solely for the async
 * LLM phases, so there are no cross-task write races.
 */

const fs = require('fs');
const path = require('path');
const { getConfig } = require('./config');
const { getLocalDateStamp } = require('./datetime');

// Background-pass telemetry is operational, not cognitive, so it goes to the
// ops log (Thinking tab) rather than the daily log that gets injected into chat.
const OPS_DIR = require('./database').getOpsDir();
const LANE_DEFAULTS = { agentJobs: 8, scheduled: 2, background: 4 };
/** The lane anything unlabelled lands in. Every existing caller keeps working. */
const DEFAULT_LANE = 'background';
const LANES = Object.keys(LANE_DEFAULTS);

/**
 * agentPool.concurrency became agentPool.lanes.background. A box that set the
 * old key has it sitting in data/config.json reading like the thing that governs
 * background width, governing nothing. Said once per process.
 */
let warnedDeadConcurrency = false;
function warnDeadConcurrency(value) {
  if (warnedDeadConcurrency) return;
  warnedDeadConcurrency = true;
  console.warn(`[AgentPool] agentPool.concurrency (${value}) in data/config.json is NO LONGER READ — ` +
    `background width is agentPool.lanes.background now, and the other kinds of work have lanes of ` +
    `their own (Settings -> Background lanes). Delete the old key; it is doing nothing.`);
}

class AgentPool {
  constructor() {
    // One FIFO per lane. Separate queues rather than one queue with a priority
    // field, because the property wanted is "a swarm of agent jobs cannot make
    // the heartbeat wait", and that is a statement about queues, not ordering.
    this._lanes = {};
    for (const lane of LANES) this._lanes[lane] = { queue: [], active: 0 };
    this._rr = 0;              // round-robin cursor, so no lane starves another
    this._chatInFlight = 0;    // >0 → background drops to its reserved headroom
    this._pass = null;         // current instrumentation pass, or null
  }

  _poolCfg() {
    const c = (getConfig().agentPool) || {};
    if (c.concurrency !== undefined) warnDeadConcurrency(c.concurrency);
    return c;
  }

  /** This lane's own cap. */
  _laneCap(lane) {
    const c = this._poolCfg();
    const n = (c.lanes || {})[lane];
    return Math.max(1, Number.isInteger(n) ? n : LANE_DEFAULTS[lane] ?? LANE_DEFAULTS.background);
  }

  /**
   * The ceiling across every lane at once. While a reply is being written this
   * collapses to chat's reserved headroom — that is what chat has instead of a
   * lane of its own, because the user's turn must never queue behind anything.
   */
  _totalCap() {
    const c = this._poolCfg();
    if (this._chatInFlight > 0) {
      return Math.max(1, Number.isInteger(c.backgroundDuringChat) ? c.backgroundDuringChat : 2);
    }
    return Math.max(1, Number.isInteger(c.maxTotalBackground) ? c.maxTotalBackground : 12);
  }

  _totalActive() {
    return LANES.reduce((n, lane) => n + this._lanes[lane].active, 0);
  }

  _totalQueued() {
    return LANES.reduce((n, lane) => n + this._lanes[lane].queue.length, 0);
  }

  /** Back-compat for readers that asked the old question. */
  _effectiveConcurrency() {
    return this._totalCap();
  }

  isChatInFlight() {
    return this._chatInFlight > 0;
  }

  /**
   * A lane's cap, for a caller that gates work BEFORE handing it over.
   *
   * agent-jobs.js does: it holds surplus jobs at `queued` in the database rather
   * than inside this pool, so the panel can show "N running, N queued" and the
   * log can say why. That is worth keeping — but it must be the SAME number, or
   * the lower of the two silently wins and the other is decoration. It used to
   * be: agentJobs.maxConcurrent was 2 and would have capped a lane set to 8.
   */
  laneCap(lane) {
    return this._laneCap(LANES.includes(lane) ? lane : DEFAULT_LANE);
  }

  /** How many tasks are queued or running (for instrumentation/debug). */
  stats() {
    const lanes = {};
    for (const lane of LANES) {
      lanes[lane] = {
        active: this._lanes[lane].active,
        queued: this._lanes[lane].queue.length,
        cap: this._laneCap(lane)
      };
    }
    return {
      active: this._totalActive(),
      queued: this._totalQueued(),
      chatInFlight: this._chatInFlight,
      effectiveConcurrency: this._totalCap(),
      totalCap: this._totalCap(),
      lanes
    };
  }

  // ===== Chat priority =====

  /** Called by the chat route when a chat request starts. Throttles the pool to 1. */
  beginChat() {
    this._chatInFlight++;
    if (this._pass) this._pass.throttled = true;
    if (this._chatInFlight === 1) {
      console.log(`[AgentPool] Chat in flight — background held to its reserved headroom (${this._totalCap()})`);
    }
  }

  /** Called by the chat route when a chat request finishes (success or error). */
  endChat() {
    if (this._chatInFlight > 0) this._chatInFlight--;
    if (this._chatInFlight === 0) {
      console.log(`[AgentPool] Chat cleared — background lanes back to full width (${this._totalCap()})`);
      this._drain(); // fill slots freed up by the restored width
    }
  }

  // ===== Scheduling =====

  /**
   * Schedule a single async task. Returns a promise that resolves/rejects with
   * the task's result. Rejections propagate to the caller — use runBatch() for
   * error-isolated fan-out.
   * @param {() => Promise<any>} taskFn
   * @param {string} [label] - short label for instrumentation/logging
   * @returns {Promise<any>}
   */
  schedule(taskFn, label = 'task', lane = DEFAULT_LANE) {
    const key = LANES.includes(lane) ? lane : DEFAULT_LANE;
    return new Promise((resolve, reject) => {
      this._lanes[key].queue.push({ taskFn, resolve, reject, label });
      if (this._pass) this._pass.scheduled++;
      this._drain();
    });
  }

  /**
   * Launch queued work, round-robin across lanes.
   *
   * Two conditions, and both have to hold: the lane must be under ITS cap, and
   * the pool must be under the total. The round-robin cursor is what makes the
   * lanes fair — draining them in a fixed order would let a full agentJobs queue
   * take every freed slot and leave the heartbeat waiting behind eight jobs,
   * which is the starvation this exists to prevent, rebuilt one level down.
   */
  _drain() {
    for (;;) {
      if (this._totalActive() >= this._totalCap()) return;

      let picked = null;
      for (let i = 0; i < LANES.length; i++) {
        const lane = LANES[(this._rr + i) % LANES.length];
        const st = this._lanes[lane];
        if (st.queue.length > 0 && st.active < this._laneCap(lane)) {
          picked = lane;
          this._rr = (this._rr + i + 1) % LANES.length;
          break;
        }
      }
      if (!picked) return;   // everything queued is behind its own lane's cap

      const st = this._lanes[picked];
      const job = st.queue.shift();
      st.active++;
      if (this._pass) {
        this._pass.started++;
        const total = this._totalActive();
        if (total > this._pass.peakActive) this._pass.peakActive = total;
      }
      Promise.resolve()
        .then(() => job.taskFn())
        .then(
          (value) => { if (this._pass) this._pass.succeeded++; job.resolve(value); },
          (err) => { if (this._pass) this._pass.failed++; job.reject(err); }
        )
        .finally(() => {
          st.active--;
          this._drain();
        });
    }
  }

  /**
   * Run an array of task functions through the pool with error isolation.
   * One failed task never rejects the batch — it becomes a settled entry and
   * the rest continue. Mirrors Promise.allSettled shape.
   * @param {Array<() => Promise<any>>} taskFns
   * @param {string} [label]
   * @returns {Promise<Array<{status: 'fulfilled'|'rejected', value?: any, reason?: any}>>}
   */
  async runBatch(taskFns, label = 'batch', lane = DEFAULT_LANE) {
    return Promise.all(taskFns.map((fn, i) =>
      this.schedule(fn, `${label}[${i}]`, lane)
        .then(value => ({ status: 'fulfilled', value }))
        .catch(reason => {
          const msg = reason && reason.message ? reason.message : String(reason);
          console.error(`[AgentPool] Task ${label}[${i}] failed (isolated): ${msg}`);
          return { status: 'rejected', reason };
        })
    ));
  }

  // ===== Instrumentation =====

  /**
   * Begin an instrumentation pass. Subsequent schedule()/runBatch() tasks count
   * toward it until endPass() is called. Non-nesting: a new pass replaces any
   * current one.
   * @param {string} label
   */
  startPass(label) {
    this._pass = {
      label,
      startMs: Date.now(),
      scheduled: 0,
      started: 0,
      succeeded: 0,
      failed: 0,
      peakActive: 0,
      throttled: this._chatInFlight > 0
    };
    return this._pass;
  }

  /**
   * Close the current pass. Logs task counts + total wall-time to console and
   * (by default) the daily log.
   * @param {Object} [opts]
   * @param {boolean} [opts.toDailyLog=true]
   * @returns {Object|null} the pass stats (with wallMs), or null if no pass open
   */
  endPass({ toDailyLog = true } = {}) {
    if (!this._pass) return null;
    const p = this._pass;
    const wallMs = Date.now() - p.startMs;
    const summary =
      `Agent pool pass "${p.label}": ${p.started} task(s) ` +
      `(${p.succeeded} ok, ${p.failed} failed), ${wallMs}ms wall-time, ` +
      `peak concurrency ${p.peakActive}${p.throttled ? ', throttled by chat' : ''}`;
    console.log(`[AgentPool] ${summary}`);
    if (toDailyLog) this._appendOps(summary);
    this._pass = null;
    return { ...p, wallMs };
  }

  /**
   * Prepend a one-line entry at the top of today's OPS log (newest first,
   * under the H1 header). Best-effort. Kept self-contained (no cross-module
   * require) to avoid a dependency cycle with fact-extractor. Pass telemetry
   * is operational, so it goes to the ops log — never the injected daily log.
   */
  _appendOps(summary) {
    try {
      if (!fs.existsSync(OPS_DIR)) fs.mkdirSync(OPS_DIR, { recursive: true });
      const now = new Date();
      const date = getLocalDateStamp(now); // local Pacific date
      const time = now.toTimeString().slice(0, 5);
      const opsFile = path.join(OPS_DIR, `${date}.md`);
      const header = `# Ops Log - ${date}\n\n`;
      const entry = `### ${time}\n- ${summary}\n\n`;

      if (!fs.existsSync(opsFile)) {
        fs.writeFileSync(opsFile, header + entry, 'utf8');
        return;
      }
      const content = fs.readFileSync(opsFile, 'utf8');
      // Match only a level-1 "# " header, not a "## Heartbeat Report" block.
      const headerMatch = content.match(/^(# [^\n]*\r?\n(?:\r?\n)?)/);
      if (headerMatch) {
        const head = headerMatch[1];
        fs.writeFileSync(opsFile, head + entry + content.slice(head.length), 'utf8');
      } else {
        fs.writeFileSync(opsFile, header + entry + content, 'utf8');
      }
    } catch (err) {
      console.error('[AgentPool] Failed to write pass stats to ops log:', err.message);
    }
  }
}

// Single shared pool for the whole process.
const pool = new AgentPool();

module.exports = pool;
module.exports.AgentPool = AgentPool;
