/**
 * Initiative layer — things SNH notices and may raise unprompted.
 *
 * Heartbeat tasks write candidate initiatives (stale gap questions, reflection
 * insights worth sharing, blocking contradiction uncertainties, cluster-audit
 * findings that need the user). A prioritizer re-scores them, expires stale
 * ones, and caps the pool so it never becomes a nag queue. Delivery happens via
 * a conversation-open greeting or a rate-limited unprompted message.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');
const { getLocalDateStamp } = require('./datetime');

const VALID_TYPES = new Set(['question', 'observation', 'alert', 'reflection-insight']);

function clampPriority(p) {
  const n = Math.round(Number(p));
  if (!Number.isFinite(n)) return 5;
  return Math.max(1, Math.min(10, n));
}

/**
 * Add a candidate initiative. Deduped against existing pending initiatives with
 * the same (source_kind, source_ref) so the same underlying thing is never
 * queued twice.
 * @returns {string|null} id (existing or new), or null on failure
 */
function addInitiative({ type, content, sourceKind = null, sourceRef = null, priority = 5 }) {
  try {
    const db = getSqliteDb();
    if (!db || !content || !content.trim()) return null;
    const safeType = VALID_TYPES.has(type) ? type : 'observation';

    if (sourceRef) {
      const existing = db.prepare(
        "SELECT id FROM initiatives WHERE status = 'pending' AND source_kind IS ? AND source_ref IS ?"
      ).get(sourceKind, sourceRef);
      if (existing) return existing.id;
    }

    const id = randomUUID();
    db.prepare(`
      INSERT INTO initiatives (id, type, content, source_kind, source_ref, priority, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
    `).run(id, safeType, content.trim(), sourceKind, sourceRef, clampPriority(priority), new Date().toISOString());
    console.log(`[Initiatives] Queued ${safeType} (priority ${clampPriority(priority)}): "${content.trim().slice(0, 80)}"`);
    return id;
  } catch (error) {
    console.error('[Initiatives] addInitiative error:', error.message);
    return null;
  }
}

/** All pending initiatives, highest priority first. */
function listPending({ minPriority = 0, limit = 100 } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    return db.prepare(`
      SELECT * FROM initiatives
      WHERE status = 'pending' AND priority >= ?
      ORDER BY priority DESC, created_at ASC
      LIMIT ?
    `).all(minPriority, limit);
  } catch (error) {
    console.error('[Initiatives] listPending error:', error.message);
    return [];
  }
}

/** The single highest-priority pending initiative at or above minPriority. */
function getTopPending(minPriority = 0) {
  const list = listPending({ minPriority, limit: 1 });
  return list.length ? list[0] : null;
}

function get(id) {
  try {
    const db = getSqliteDb();
    if (!db) return null;
    return db.prepare('SELECT * FROM initiatives WHERE id = ?').get(id) || null;
  } catch (error) {
    console.error('[Initiatives] get error:', error.message);
    return null;
  }
}

function countPending() {
  try {
    const db = getSqliteDb();
    if (!db) return 0;
    return db.prepare("SELECT COUNT(*) AS c FROM initiatives WHERE status = 'pending'").get().c;
  } catch (error) {
    return 0;
  }
}

/**
 * Mark an initiative delivered.
 * @param {string} id
 * @param {Object} [opts]
 * @param {string} [opts.channel] - 'greeting' | 'unprompted' | 'panel'
 * @param {string} [opts.conversationId]
 * @returns {boolean}
 */
function markDelivered(id, { channel = null, conversationId = null } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare(`
      UPDATE initiatives
      SET status = 'delivered', delivered_at = ?, channel = ?, delivered_conversation_id = ?
      WHERE id = ? AND status = 'pending'
    `).run(new Date().toISOString(), channel, conversationId, id);
    return info.changes > 0;
  } catch (error) {
    console.error('[Initiatives] markDelivered error:', error.message);
    return false;
  }
}

function dismiss(id) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare(
      "UPDATE initiatives SET status = 'dismissed' WHERE id = ? AND status = 'pending'"
    ).run(id);
    return info.changes > 0;
  } catch (error) {
    console.error('[Initiatives] dismiss error:', error.message);
    return false;
  }
}

function expire(id) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare(
      "UPDATE initiatives SET status = 'expired' WHERE id = ? AND status = 'pending'"
    ).run(id);
    return info.changes > 0;
  } catch (error) {
    console.error('[Initiatives] expire error:', error.message);
    return false;
  }
}

function updatePriority(id, priority) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare("UPDATE initiatives SET priority = ? WHERE id = ? AND status = 'pending'")
      .run(clampPriority(priority), id);
    return info.changes > 0;
  } catch (error) {
    console.error('[Initiatives] updatePriority error:', error.message);
    return false;
  }
}

/**
 * How many unprompted initiatives were delivered "today" (local Pacific day).
 * Used to enforce the hard daily cap on SNH-initiated conversations.
 * @returns {number}
 */
function countUnpromptedDeliveredToday() {
  try {
    const db = getSqliteDb();
    if (!db) return 0;
    const rows = db.prepare(
      "SELECT delivered_at FROM initiatives WHERE channel = 'unprompted' AND delivered_at IS NOT NULL"
    ).all();
    const today = getLocalDateStamp();
    return rows.filter(r => getLocalDateStamp(new Date(r.delivered_at)) === today).length;
  } catch (error) {
    console.error('[Initiatives] countUnpromptedDeliveredToday error:', error.message);
    return 0;
  }
}

module.exports = {
  addInitiative,
  listPending,
  getTopPending,
  get,
  countPending,
  markDelivered,
  dismiss,
  expire,
  updatePriority,
  countUnpromptedDeliveredToday
};
