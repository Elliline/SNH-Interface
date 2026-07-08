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
const OPS_DIR = path.join(__dirname, '../data/memory/ops');
const DEFAULT_CONCURRENCY = 3;

class AgentPool {
  constructor() {
    this._queue = [];          // [{ taskFn, resolve, reject, label }]
    this._active = 0;          // tasks currently running
    this._chatInFlight = 0;    // >0 → throttle to concurrency 1
    this._pass = null;         // current instrumentation pass, or null
  }

  /** Configured full-width concurrency (config.agentPool.concurrency, default 6). */
  _configuredConcurrency() {
    const cfg = getConfig();
    const n = cfg.agentPool && Number.isInteger(cfg.agentPool.concurrency)
      ? cfg.agentPool.concurrency
      : DEFAULT_CONCURRENCY;
    return Math.max(1, n);
  }

  /** Effective concurrency right now — 1 while chat is in flight, else full width. */
  _effectiveConcurrency() {
    return this._chatInFlight > 0 ? 1 : this._configuredConcurrency();
  }

  isChatInFlight() {
    return this._chatInFlight > 0;
  }

  /** How many tasks are queued or running (for instrumentation/debug). */
  stats() {
    return {
      active: this._active,
      queued: this._queue.length,
      chatInFlight: this._chatInFlight,
      effectiveConcurrency: this._effectiveConcurrency()
    };
  }

  // ===== Chat priority =====

  /** Called by the chat route when a chat request starts. Throttles the pool to 1. */
  beginChat() {
    this._chatInFlight++;
    if (this._pass) this._pass.throttled = true;
    if (this._chatInFlight === 1) {
      console.log('[AgentPool] Chat in flight — throttling background pool to concurrency 1');
    }
  }

  /** Called by the chat route when a chat request finishes (success or error). */
  endChat() {
    if (this._chatInFlight > 0) this._chatInFlight--;
    if (this._chatInFlight === 0) {
      console.log('[AgentPool] Chat cleared — background pool resuming full concurrency');
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
  schedule(taskFn, label = 'task') {
    return new Promise((resolve, reject) => {
      this._queue.push({ taskFn, resolve, reject, label });
      if (this._pass) this._pass.scheduled++;
      this._drain();
    });
  }

  /** Launch queued tasks up to the current effective concurrency. */
  _drain() {
    while (this._active < this._effectiveConcurrency() && this._queue.length > 0) {
      const job = this._queue.shift();
      this._active++;
      if (this._pass) {
        this._pass.started++;
        if (this._active > this._pass.peakActive) this._pass.peakActive = this._active;
      }
      Promise.resolve()
        .then(() => job.taskFn())
        .then(
          (value) => { if (this._pass) this._pass.succeeded++; job.resolve(value); },
          (err) => { if (this._pass) this._pass.failed++; job.reject(err); }
        )
        .finally(() => {
          this._active--;
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
  async runBatch(taskFns, label = 'batch') {
    return Promise.all(taskFns.map((fn, i) =>
      this.schedule(fn, `${label}[${i}]`)
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
