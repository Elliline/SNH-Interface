/**
 * Cron job proposals — the storage + decision layer behind the create_cron_job
 * tool. This is the FIRST action tool, and it is deliberately propose-only.
 *
 * Flow:
 *   1. The entity emits a create_cron_job tool call.
 *   2. propose() validates it, checks the tool's rate caps, writes a row with
 *      status='proposed' and provenance source='kid-proposed', and raises an
 *      initiative so it lands in the bell panel.
 *   3. Ellie approves or rejects there.
 *   4. approve() flips the row to 'approved'. reject() flips it to 'rejected'
 *      and writes a daily-log line so the entity learns the outcome the same
 *      way it learns anything else.
 *
 * WHAT APPROVING DOES NOT DO: it does not schedule anything. SNH has no
 * scheduler — approving records the job so a future scheduler can pick it up.
 * Said plainly here because the whole point of the capability manifest is that
 * this system does not claim more than it has built.
 *
 * Caps are on the tool itself (config.tools.cron), not on a trust level, and
 * are counted from the DB so a restart cannot reset them.
 */

const { randomUUID } = require('crypto');
const path = require('path');
const { getSqliteDb } = require('./database');
const { getConfig } = require('./config');

const MEMORY_DIR = require('./database').getMemoryDir();
const DAILY_DIR = path.join(MEMORY_DIR, 'daily');
const OPS_DIR = path.join(MEMORY_DIR, 'ops');

const HOUR_MS = 60 * 60 * 1000;

/** Provenance tag for everything this path creates (mirrors cluster_members.source). */
const KID_SOURCE = 'kid-proposed';

/** Lazy requires — avoid load-order cycles (initiatives → memory-clusters → …). */
function opsLog(msg) {
  try { require('./fact-extractor').appendToOpsLog(msg, OPS_DIR); } catch (e) { /* best effort */ }
}
function dailyLog(msg) {
  try { require('./fact-extractor').appendToDailyLog(msg, DAILY_DIR); } catch (e) { /* best effort */ }
}

/** Hot-read the tool's config each call so knobs take effect without a restart. */
function cfg() {
  const c = (getConfig().tools && getConfig().tools.cron) || {};
  return {
    enabled: c.enabled !== false,
    maxProposalsPerHour: Math.max(1, c.maxProposalsPerHour || 3),
    maxKidCreatedJobs: Math.max(1, c.maxKidCreatedJobs || 10)
  };
}

// ============ validation ============

const CRON_FIELD = [
  { name: 'minute', min: 0, max: 59 },
  { name: 'hour', min: 0, max: 23 },
  { name: 'day-of-month', min: 1, max: 31 },
  { name: 'month', min: 1, max: 12 },
  { name: 'day-of-week', min: 0, max: 7 }
];

/**
 * Validate a 5-field cron expression. Accepts *, ranges, steps and lists in each
 * field. Deliberately strict: this value is written by a language model at
 * temperature 1.0, and a malformed schedule that silently stores is worse than
 * a tool error the model can see and correct.
 * @returns {{ok: boolean, error?: string}}
 */
function validateSchedule(schedule) {
  if (typeof schedule !== 'string' || !schedule.trim()) {
    return { ok: false, error: 'schedule must be a non-empty string' };
  }
  const parts = schedule.trim().split(/\s+/);
  if (parts.length !== 5) {
    return { ok: false, error: `schedule must have exactly 5 fields (got ${parts.length}): "${schedule}"` };
  }
  for (let i = 0; i < 5; i++) {
    const spec = CRON_FIELD[i];
    const field = parts[i];
    // Each comma-separated term: * | N | N-M | */S | N-M/S | ?  (? only for day fields)
    for (const term of field.split(',')) {
      if (term === '*' || (term === '?' && (i === 2 || i === 4))) continue;
      const m = term.match(/^(\*|\d+(?:-\d+)?)(?:\/(\d+))?$/);
      if (!m) return { ok: false, error: `invalid ${spec.name} field "${field}"` };
      if (m[2] !== undefined && Number(m[2]) < 1) {
        return { ok: false, error: `invalid step in ${spec.name} field "${field}"` };
      }
      if (m[1] !== '*') {
        const bounds = m[1].split('-').map(Number);
        for (const v of bounds) {
          if (v < spec.min || v > spec.max) {
            return { ok: false, error: `${spec.name} value ${v} out of range ${spec.min}-${spec.max}` };
          }
        }
        if (bounds.length === 2 && bounds[0] > bounds[1]) {
          return { ok: false, error: `${spec.name} range "${m[1]}" is backwards` };
        }
      }
    }
  }
  return { ok: true };
}

// ============ tool-call log (Thinking tab) ============

/**
 * Record a tool call and what came of it. Every call is logged — including ones
 * refused by a cap — because a refused call is exactly what you want to be able
 * to look back at.
 */
function logToolCall({ tool, args, outcome, detail, refId = null, conversationId = null }) {
  try {
    const db = getSqliteDb();
    if (!db) return null;
    const id = randomUUID();
    db.prepare(`
      INSERT INTO tool_call_log (id, created_at, tool, args_json, outcome, detail, ref_id, conversation_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, new Date().toISOString(), tool, JSON.stringify(args || {}), outcome, detail || null, refId, conversationId);
    return id;
  } catch (err) {
    console.error('[CronJobs] logToolCall failed:', err.message);
    return null;
  }
}

/** Recent tool calls, newest first — read by the Thinking tab. */
function listToolCalls({ limit = 100 } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    return db.prepare(
      'SELECT * FROM tool_call_log ORDER BY datetime(created_at) DESC LIMIT ?'
    ).all(Math.min(Math.max(1, limit), 500));
  } catch (err) {
    console.error('[CronJobs] listToolCalls failed:', err.message);
    return [];
  }
}

// ============ rate caps ============

/**
 * Check the tool's caps. Same shape as the watchdog's restart cap: a trailing
 * one-hour window plus a hard ceiling. Counted from the DB, so restarting the
 * server does not hand the entity a fresh budget.
 * @returns {{ok: boolean, reason?: string}}
 */
function checkCaps() {
  const db = getSqliteDb();
  if (!db) return { ok: false, reason: 'database unavailable' };
  const { maxProposalsPerHour, maxKidCreatedJobs } = cfg();

  const sinceIso = new Date(Date.now() - HOUR_MS).toISOString();
  const recent = db.prepare(
    "SELECT COUNT(*) AS n FROM cron_jobs WHERE source = ? AND datetime(created_at) > datetime(?)"
  ).get(KID_SOURCE, sinceIso).n;
  if (recent >= maxProposalsPerHour) {
    return { ok: false, reason: `proposal rate cap reached (${recent}/${maxProposalsPerHour} in the last hour)` };
  }

  const live = db.prepare(
    "SELECT COUNT(*) AS n FROM cron_jobs WHERE source = ? AND status = 'approved'"
  ).get(KID_SOURCE).n;
  if (live >= maxKidCreatedJobs) {
    return { ok: false, reason: `total cap reached (${live}/${maxKidCreatedJobs} approved jobs already exist)` };
  }

  return { ok: true };
}

// ============ propose / decide ============

/**
 * Propose a cron job. Validates, checks caps, stores as 'proposed', and raises
 * an initiative so it reaches the bell panel.
 * @returns {Promise<{ok: boolean, id?: string, error?: string, status?: string}>}
 */
async function propose({ schedule, description, enabled, conversationId = null }) {
  const args = { schedule, description, enabled };
  const { enabled: toolOn } = cfg();

  if (!toolOn) {
    logToolCall({ tool: 'create_cron_job', args, outcome: 'error', detail: 'tool disabled in config', conversationId });
    return { ok: false, error: 'The create_cron_job tool is disabled.' };
  }

  const v = validateSchedule(schedule);
  if (!v.ok) {
    logToolCall({ tool: 'create_cron_job', args, outcome: 'error', detail: v.error, conversationId });
    return { ok: false, error: `Invalid schedule: ${v.error}. Use a 5-field cron expression, e.g. "0 6 * * *" for 6am daily.` };
  }
  if (typeof description !== 'string' || !description.trim()) {
    logToolCall({ tool: 'create_cron_job', args, outcome: 'error', detail: 'missing description', conversationId });
    return { ok: false, error: 'A description is required.' };
  }

  const caps = checkCaps();
  if (!caps.ok) {
    logToolCall({ tool: 'create_cron_job', args, outcome: 'rejected-cap', detail: caps.reason, conversationId });
    opsLog(`create_cron_job refused by cap: ${caps.reason} — "${String(description).slice(0, 80)}"`);
    return { ok: false, error: `Not proposed — ${caps.reason}. Tell the user plainly that you have hit your own limit for this, rather than trying again.` };
  }

  const db = getSqliteDb();
  const id = randomUUID();
  const desc = description.trim();
  const sched = schedule.trim();
  const enabledInt = enabled === true ? 1 : 0;

  try {
    db.prepare(`
      INSERT INTO cron_jobs (id, schedule, description, enabled, source, status, conversation_id, created_at)
      VALUES (?, ?, ?, ?, ?, 'proposed', ?, ?)
    `).run(id, sched, desc, enabledInt, KID_SOURCE, conversationId, new Date().toISOString());
  } catch (err) {
    logToolCall({ tool: 'create_cron_job', args, outcome: 'error', detail: err.message, conversationId });
    return { ok: false, error: `Could not record the proposal: ${err.message}` };
  }

  // Raise it for Ellie. source_kind/source_ref link the bell item back to the row
  // so the panel can offer Approve/Reject on exactly this proposal.
  let initiativeId = null;
  try {
    const initiatives = require('./initiatives');
    initiativeId = await initiatives.addInitiative({
      type: 'proposal',
      content: `I'd like to set up a scheduled job: ${desc} — running ${sched}${enabledInt ? '' : ' (starting disabled)'}. Approve it and I'll record it; reject it and I'll drop it.`,
      sourceKind: 'cron-proposal',
      sourceRef: id,
      priority: 6
    });
    if (initiativeId) {
      db.prepare('UPDATE cron_jobs SET initiative_id = ? WHERE id = ?').run(initiativeId, id);
    }
  } catch (err) {
    console.error('[CronJobs] could not raise initiative:', err.message);
  }

  logToolCall({
    tool: 'create_cron_job', args, outcome: 'proposed',
    detail: `"${desc}" @ ${sched}`, refId: id, conversationId
  });
  opsLog(`create_cron_job proposed: "${desc}" @ ${sched} (job ${id.slice(0, 8)}, awaiting approval)`);

  return {
    ok: true, id, status: 'proposed',
    error: undefined
  };
}

/** Fetch one job. */
function get(id) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare('SELECT * FROM cron_jobs WHERE id = ?').get(id) || null;
}

/**
 * Approve a proposal. Records the job. Does NOT schedule it — nothing in SNH
 * executes cron jobs yet.
 */
function approve(id, { note = null } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'database unavailable' };
  const job = get(id);
  if (!job) return { ok: false, error: 'no such proposal' };
  if (job.status !== 'proposed') return { ok: false, error: `proposal is already ${job.status}` };

  // Re-check the total cap at decision time: proposals can queue up, and the
  // ceiling is on approved jobs, not on pending asks.
  const { maxKidCreatedJobs } = cfg();
  const live = db.prepare("SELECT COUNT(*) AS n FROM cron_jobs WHERE source = ? AND status = 'approved'").get(KID_SOURCE).n;
  if (live >= maxKidCreatedJobs) {
    return { ok: false, error: `total cap reached (${live}/${maxKidCreatedJobs} approved kid-created jobs)` };
  }

  db.prepare("UPDATE cron_jobs SET status = 'approved', decided_at = ?, decided_note = ? WHERE id = ?")
    .run(new Date().toISOString(), note, id);

  // ARM IT. This is the line that used to be missing, and its absence was the
  // whole of the failure: approving wrote 'approved' and nothing computed a
  // first firing, so the row sat there and the honest answer to "when does it
  // run" was "never". A job that is approved and enabled now leaves this
  // function with a next_run_at on it.
  let armedFor = null;
  try {
    armedFor = require('./scheduler').armJob(id, { reason: 'approved' });
  } catch (err) {
    console.error('[CronJobs] could not arm approved job:', err.message);
  }

  try {
    const initiatives = require('./initiatives');
    if (job.initiative_id) initiatives.dismiss(job.initiative_id);
  } catch (e) { /* best effort */ }

  logToolCall({
    tool: 'create_cron_job', args: { schedule: job.schedule, description: job.description },
    outcome: 'approved', detail: `approved: "${job.description}" @ ${job.schedule}`,
    refId: id, conversationId: job.conversation_id
  });
  // Say which of the two things happened, rather than one sentence that is true
  // of both: an approved-but-disabled job is recorded and NOT armed, and that
  // distinction is exactly the sort of thing this system has been wrong about.
  const when = armedFor
    ? `first run ${new Date(armedFor).toLocaleString()}`
    : (job.enabled ? 'NOT armed — its schedule could not be evaluated' : 'not armed — it was proposed as disabled');
  opsLog(`cron proposal APPROVED: "${job.description}" @ ${job.schedule} (job ${id.slice(0, 8)}) — ${when}`);
  dailyLog(armedFor
    ? `Ellie approved my proposed scheduled job: ${job.description} (${job.schedule}). It is scheduled now — the first run is ${new Date(armedFor).toLocaleString()}.`
    : `Ellie approved my proposed scheduled job: ${job.description} (${job.schedule}), but it is not armed: ${when}.`);

  return { ok: true, job: get(id), nextRunAt: armedFor };
}

/** Reject a proposal, and tell the entity so it learns the outcome. */
function reject(id, { note = null } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'database unavailable' };
  const job = get(id);
  if (!job) return { ok: false, error: 'no such proposal' };
  if (job.status !== 'proposed') return { ok: false, error: `proposal is already ${job.status}` };

  db.prepare("UPDATE cron_jobs SET status = 'rejected', decided_at = ?, decided_note = ? WHERE id = ?")
    .run(new Date().toISOString(), note, id);

  try {
    const initiatives = require('./initiatives');
    if (job.initiative_id) initiatives.dismiss(job.initiative_id);
  } catch (e) { /* best effort */ }

  logToolCall({
    tool: 'create_cron_job', args: { schedule: job.schedule, description: job.description },
    outcome: 'rejected', detail: `rejected: "${job.description}" @ ${job.schedule}${note ? ` — ${note}` : ''}`,
    refId: id, conversationId: job.conversation_id
  });
  opsLog(`cron proposal REJECTED: "${job.description}" @ ${job.schedule} (job ${id.slice(0, 8)})${note ? ` — ${note}` : ''}`);
  // The daily log is how the entity finds out: it's cognitively meaningful, not
  // mere telemetry, so it goes here rather than only to ops.
  dailyLog(`Ellie rejected my proposed scheduled job: ${job.description} (${job.schedule})${note ? ` — her reason: ${note}` : ''}. Worth remembering what kind of thing she doesn't want scheduled.`);

  return { ok: true, job: get(id) };
}

// ============ provenance-driven listing + bulk revert ============

/** Everything the entity proposed, newest first. Optionally filtered by status. */
function listKidCreated({ status = null, limit = 200 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  const lim = Math.min(Math.max(1, limit), 500);
  return status
    ? db.prepare('SELECT * FROM cron_jobs WHERE source = ? AND status = ? ORDER BY datetime(created_at) DESC LIMIT ?').all(KID_SOURCE, status, lim)
    : db.prepare('SELECT * FROM cron_jobs WHERE source = ? ORDER BY datetime(created_at) DESC LIMIT ?').all(KID_SOURCE, lim);
}

/**
 * Bulk revert: flip every approved kid-created job to 'reverted'. This is the
 * "undo everything it did" lever the provenance tag exists for. Rows are kept —
 * supersede, never delete, same as the memory tools.
 */
function revertAllKidCreated({ note = 'bulk revert' } = {}) {
  const db = getSqliteDb();
  if (!db) return { reverted: 0 };
  const rows = db.prepare("SELECT id, description, schedule FROM cron_jobs WHERE source = ? AND status = 'approved'").all(KID_SOURCE);
  // Disarm as well as revert: a reverted job with a next_run_at still on it is a
  // row claiming a run that is not coming, and the scheduler selects on status
  // so it would never actually fire — the worst combination, a false promise
  // nothing corrects.
  const stmt = db.prepare("UPDATE cron_jobs SET status = 'reverted', next_run_at = NULL, decided_at = ?, decided_note = ? WHERE id = ?");
  const now = new Date().toISOString();
  const tx = db.transaction(list => { for (const r of list) stmt.run(now, note, r.id); });
  tx(rows);
  if (rows.length) {
    opsLog(`bulk-reverted ${rows.length} kid-created cron job(s): ${rows.map(r => `"${r.description}"`).join(', ')}`);
    dailyLog(`Ellie reverted ${rows.length} of the scheduled jobs I'd proposed and she'd approved.`);
  }
  return { reverted: rows.length };
}

/** Current cap usage, for the UI. */
function capStatus() {
  const db = getSqliteDb();
  const { maxProposalsPerHour, maxKidCreatedJobs } = cfg();
  if (!db) return { maxProposalsPerHour, maxKidCreatedJobs, lastHour: 0, approved: 0 };
  const sinceIso = new Date(Date.now() - HOUR_MS).toISOString();
  return {
    maxProposalsPerHour,
    maxKidCreatedJobs,
    lastHour: db.prepare("SELECT COUNT(*) AS n FROM cron_jobs WHERE source = ? AND datetime(created_at) > datetime(?)").get(KID_SOURCE, sinceIso).n,
    approved: db.prepare("SELECT COUNT(*) AS n FROM cron_jobs WHERE source = ? AND status = 'approved'").get(KID_SOURCE).n
  };
}

module.exports = {
  KID_SOURCE,
  validateSchedule,
  checkCaps,
  capStatus,
  propose,
  approve,
  reject,
  get,
  listKidCreated,
  revertAllKidCreated,
  logToolCall,
  listToolCalls
};
