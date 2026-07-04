/**
 * Question queue.
 *
 * Gaps, oddities, and uncertain contradictions in the user's facts become
 * stored questions the model may ask at a natural moment. One question at most
 * surfaces per conversation, and never interrogation-style.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');

const VALID_REASONS = new Set(['gap', 'contradiction-uncertainty', 'staleness']);

/**
 * Queue a new pending question.
 * @param {Object} q
 * @param {string} q.question - The question text
 * @param {string} [q.reason] - gap | contradiction-uncertainty | staleness
 * @param {string} [q.clusterId] - Cluster the question is about
 * @param {string} [q.memberId] - Fact the question is about
 * @param {string} [q.conversationId] - Conversation it originated in
 * @returns {string|null} - The new question id, or null on failure
 */
function addQuestion({ question, reason = 'gap', clusterId = null, memberId = null, conversationId = null }) {
  try {
    const db = getSqliteDb();
    if (!db || !question || !question.trim()) return null;
    const safeReason = VALID_REASONS.has(reason) ? reason : 'gap';
    const id = randomUUID();
    db.prepare(`
      INSERT INTO questions (id, question, cluster_id, member_id, reason, status, created_at, origin_conversation_id)
      VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
    `).run(id, question.trim(), clusterId, memberId, safeReason, new Date().toISOString(), conversationId);
    console.log(`[Questions] Queued (${safeReason}): "${question.trim()}"`);
    return id;
  } catch (error) {
    console.error('[Questions] addQuestion error:', error.message);
    return null;
  }
}

/**
 * List pending questions (most recent first).
 * @param {number} [limit=50]
 * @returns {Array}
 */
function listPending(limit = 50) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    return db.prepare(`
      SELECT * FROM questions
      WHERE status = 'pending'
      ORDER BY created_at DESC
      LIMIT ?
    `).all(limit);
  } catch (error) {
    console.error('[Questions] listPending error:', error.message);
    return [];
  }
}

/**
 * Find one pending question whose cluster is among the given cluster ids.
 * Prefers contradiction-uncertainty over gaps, then oldest.
 * @param {string[]} clusterIds
 * @returns {Object|null}
 */
function getPendingForClusters(clusterIds) {
  try {
    const db = getSqliteDb();
    if (!db || !Array.isArray(clusterIds) || clusterIds.length === 0) return null;
    const placeholders = clusterIds.map(() => '?').join(',');
    return db.prepare(`
      SELECT * FROM questions
      WHERE status = 'pending' AND cluster_id IN (${placeholders})
      ORDER BY
        CASE reason WHEN 'contradiction-uncertainty' THEN 0 ELSE 1 END,
        created_at ASC
      LIMIT 1
    `).get(...clusterIds);
  } catch (error) {
    console.error('[Questions] getPendingForClusters error:', error.message);
    return null;
  }
}

/**
 * Whether a question has already been surfaced (asked) in a conversation.
 * Enforces "never more than one question per conversation".
 * @param {string} conversationId
 * @returns {boolean}
 */
function hasAskedInConversation(conversationId) {
  try {
    const db = getSqliteDb();
    if (!db || !conversationId) return false;
    const row = db.prepare(`
      SELECT 1 FROM questions
      WHERE asked_conversation_id = ? AND asked_at IS NOT NULL
      LIMIT 1
    `).get(conversationId);
    return !!row;
  } catch (error) {
    console.error('[Questions] hasAskedInConversation error:', error.message);
    return false;
  }
}

/**
 * Mark a question as surfaced/asked in a conversation.
 * @param {string} id
 * @param {string} conversationId
 * @returns {boolean}
 */
function markAsked(id, conversationId = null) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare(`
      UPDATE questions
      SET status = 'asked', asked_at = ?, asked_conversation_id = ?
      WHERE id = ? AND status = 'pending'
    `).run(new Date().toISOString(), conversationId, id);
    return info.changes > 0;
  } catch (error) {
    console.error('[Questions] markAsked error:', error.message);
    return false;
  }
}

/**
 * Questions that were asked in a conversation and are awaiting an answer.
 * @param {string} conversationId
 * @returns {Array}
 */
function getAskedForConversation(conversationId) {
  try {
    const db = getSqliteDb();
    if (!db || !conversationId) return [];
    return db.prepare(`
      SELECT * FROM questions
      WHERE status = 'asked' AND asked_conversation_id = ?
      ORDER BY asked_at DESC
    `).all(conversationId);
  } catch (error) {
    console.error('[Questions] getAskedForConversation error:', error.message);
    return [];
  }
}

/**
 * Flip a question to answered.
 * @param {string} id
 * @returns {boolean}
 */
function markAnswered(id) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare(
      "UPDATE questions SET status = 'answered' WHERE id = ? AND status IN ('asked','pending')"
    ).run(id);
    if (info.changes > 0) console.log(`[Questions] Answered: ${id}`);
    return info.changes > 0;
  } catch (error) {
    console.error('[Questions] markAnswered error:', error.message);
    return false;
  }
}

module.exports = {
  addQuestion,
  listPending,
  getPendingForClusters,
  hasAskedInConversation,
  markAsked,
  getAskedForConversation,
  markAnswered
};
