/**
 * Question queue.
 *
 * Gaps, oddities, and uncertain contradictions in the user's facts become
 * stored questions the model may ask at a natural moment. One question at most
 * surfaces per conversation, and never interrogation-style.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');
const { getConfig } = require('./config');

const VALID_REASONS = new Set(['gap', 'contradiction-uncertainty', 'staleness']);

/**
 * Cosine similarity that works for BOTH plain Arrays and Float32Arrays.
 * (memoryClusters.generateEmbedding returns a Float32Array, and the shared
 * memoryClusters.cosineSimilarity guards on Array.isArray and returns 0 for it —
 * this index-based version avoids that trap.)
 */
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den === 0 ? 0 : dot / den;
}

/** Cosine threshold above which two questions count as duplicates. */
function getDedupThreshold() {
  const cfg = getConfig();
  const n = cfg.questions && cfg.questions.dedupThreshold;
  return Number.isFinite(n) ? n : 0.85;
}

/** Parse a stored embedding (JSON array) back into a vector, or null. */
function parseEmbedding(json) {
  if (!json) return null;
  try {
    const arr = JSON.parse(json);
    return Array.isArray(arr) && arr.length ? arr : null;
  } catch { return null; }
}

/**
 * Embedding for a question row, using the cached column if present and lazily
 * backfilling (and persisting) it otherwise. Legacy rows have no embedding, so
 * the first dedup/answer pass over them computes and stores it once.
 * @returns {Promise<number[]|Float32Array|null>}
 */
async function embeddingForRow(row, memoryClusters) {
  const cached = parseEmbedding(row.embedding);
  if (cached) return cached;
  try {
    const emb = await memoryClusters.generateEmbedding(row.question);
    if (emb) {
      const db = getSqliteDb();
      if (db) db.prepare('UPDATE questions SET embedding = ? WHERE id = ?')
        .run(JSON.stringify(Array.from(emb)), row.id);
    }
    return emb || null;
  } catch { return null; }
}

/**
 * Queue a new pending question — unless a near-identical question already exists
 * (in ANY status: pending, asked, or answered). Semantic dedup mirrors the
 * initiative layer: embed the candidate, compare cosine against every existing
 * question, and skip if one is ≥ the dedup threshold. This is what stops the
 * queue re-asking something the user already answered days ago.
 * @param {Object} q
 * @param {string} q.question - The question text
 * @param {string} [q.reason] - gap | contradiction-uncertainty | staleness
 * @param {string} [q.clusterId] - Cluster the question is about
 * @param {string} [q.memberId] - Fact the question is about
 * @param {string} [q.conversationId] - Conversation it originated in
 * @param {boolean} [q.skipDedup] - Bypass the semantic dedup check (tests only)
 * @returns {Promise<string|null>} - The new question id, or null if skipped/failed
 */
async function addQuestion({ question, reason = 'gap', clusterId = null, memberId = null, conversationId = null, skipDedup = false }) {
  try {
    const db = getSqliteDb();
    if (!db || !question || !question.trim()) return null;
    const text = question.trim();
    const safeReason = VALID_REASONS.has(reason) ? reason : 'gap';

    // Semantic dedup against ALL existing questions, whatever their status.
    let newEmb = null;
    if (!skipDedup) {
      try {
        const memoryClusters = require('./memory-clusters');
        newEmb = await memoryClusters.generateEmbedding(text);
        if (newEmb) {
          const threshold = getDedupThreshold();
          const rows = db.prepare('SELECT id, question, status, embedding FROM questions').all();
          for (const r of rows) {
            const emb = await embeddingForRow(r, memoryClusters);
            if (!emb) continue;
            const sim = cosineSim(newEmb, emb);
            if (sim >= threshold) {
              console.log(`[Questions] Skipped near-duplicate (sim ${sim.toFixed(3)} ≥ ${threshold}) of ${r.status} question ${String(r.id).slice(0, 8)}: "${text.slice(0, 80)}"`);
              return null;
            }
          }
        }
      } catch (dedupErr) {
        // Embedding backend down → fall through and queue rather than lose it.
        console.error('[Questions] semantic dedup skipped (continuing):', dedupErr.message);
      }
    }

    const id = randomUUID();
    const embJson = newEmb ? JSON.stringify(Array.from(newEmb)) : null;
    db.prepare(`
      INSERT INTO questions (id, question, cluster_id, member_id, reason, status, created_at, origin_conversation_id, embedding)
      VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
    `).run(id, text, clusterId, memberId, safeReason, new Date().toISOString(), conversationId, embJson);
    console.log(`[Questions] Queued (${safeReason}): "${text}"`);
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
 * Every question still awaiting an answer — pending (never surfaced) OR asked
 * (surfaced in any conversation). Used by topic-matched answer detection so a
 * question can be retired when the user answers it anywhere, or before it was
 * ever asked. Includes the cached embedding column for cheap similarity gating.
 * @returns {Array}
 */
function getOutstanding() {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    return db.prepare(`
      SELECT id, question, status, cluster_id, member_id, embedding
      FROM questions
      WHERE status IN ('pending', 'asked')
      ORDER BY created_at ASC
    `).all();
  } catch (error) {
    console.error('[Questions] getOutstanding error:', error.message);
    return [];
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
      "UPDATE questions SET status = 'answered', answered_at = ? WHERE id = ? AND status IN ('asked','pending')"
    ).run(new Date().toISOString(), id);
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
  getOutstanding,
  markAnswered,
  cosineSim,
  parseEmbedding
};
