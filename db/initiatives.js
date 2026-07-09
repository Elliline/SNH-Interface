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
const { getConfig } = require('./config');

const VALID_TYPES = new Set(['question', 'observation', 'alert', 'reflection-insight', 'followup']);

function safeParse(json, fallback) {
  try { return JSON.parse(json); } catch { return fallback; }
}

function clampPriority(p) {
  const n = Math.round(Number(p));
  if (!Number.isFinite(n)) return 5;
  return Math.max(1, Math.min(10, n));
}

/** Cosine similarity between two equal-length vectors. */
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den === 0 ? 0 : dot / den;
}

/** Cosine threshold above which two same-type initiatives count as duplicates. */
function getDedupThreshold() {
  const cfg = getConfig();
  const n = cfg.initiative && cfg.initiative.dedupThreshold;
  return Number.isFinite(n) ? n : 0.85;
}

/**
 * Add a candidate initiative. Deduped two ways so the same thing is never queued
 * twice:
 *   1. Exact: an existing pending item with the same (source_kind, source_ref).
 *   2. Semantic: an existing pending item of the SAME type whose content is
 *      cosine-similar above initiative.dedupThreshold. Successive heartbeats emit
 *      near-identical observations that each carry a DIFFERENT source_ref (every
 *      reorganization mints a new cluster id), so only the semantic check catches
 *      them.
 * @returns {Promise<string|null>} id (existing match or new), or null on failure
 */
async function addInitiative({ type, content, sourceKind = null, sourceRef = null, priority = 5 }) {
  try {
    const db = getSqliteDb();
    if (!db || !content || !content.trim()) return null;
    const safeType = VALID_TYPES.has(type) ? type : 'observation';
    const text = content.trim();

    // 1. Exact dedup by source identity — against pending AND already-delivered
    // items. Checking 'delivered' too closes the re-mint loop that re-asked the
    // same question every heartbeat: the noticing pass re-proposes a still-open
    // source each cycle, and once the prior initiative had been delivered a
    // pending-only check no longer matched it, so a fresh duplicate was minted
    // and surfaced again. A source we've already surfaced once is not re-queued.
    if (sourceRef) {
      const existing = db.prepare(
        "SELECT id FROM initiatives WHERE status IN ('pending','delivered') AND source_kind IS ? AND source_ref IS ?"
      ).get(sourceKind, sourceRef);
      if (existing) return existing.id;
    }

    // 2. Semantic dedup against pending items of the same type.
    try {
      const pending = db.prepare(
        "SELECT id, content FROM initiatives WHERE status = 'pending' AND type = ?"
      ).all(safeType);
      if (pending.length > 0) {
        const memoryClusters = require('./memory-clusters');
        const newEmb = await memoryClusters.generateEmbedding(text);
        if (newEmb) {
          const threshold = getDedupThreshold();
          for (const p of pending) {
            const emb = await memoryClusters.generateEmbedding(p.content);
            if (!emb) continue;
            const sim = cosineSim(newEmb, emb);
            if (sim >= threshold) {
              console.log(`[Initiatives] Skipped near-duplicate ${safeType} (sim ${sim.toFixed(3)} ≥ ${threshold}) of pending ${p.id}: "${text.slice(0, 80)}"`);
              return p.id;
            }
          }
        }
      }
    } catch (dedupErr) {
      // Embedding backend down → fall through and queue rather than lose the item.
      console.error('[Initiatives] semantic dedup skipped (continuing):', dedupErr.message);
    }

    const id = randomUUID();
    db.prepare(`
      INSERT INTO initiatives (id, type, content, source_kind, source_ref, priority, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
    `).run(id, safeType, text, sourceKind, sourceRef, clampPriority(priority), new Date().toISOString());
    console.log(`[Initiatives] Queued ${safeType} (priority ${clampPriority(priority)}): "${text.slice(0, 80)}"`);
    return id;
  } catch (error) {
    console.error('[Initiatives] addInitiative error:', error.message);
    return null;
  }
}

/**
 * Clean the current pending pool: within each type, keep one representative per
 * near-duplicate group (highest priority, then newest) and expire the rest.
 * Used to retroactively collapse duplicates that were queued before semantic
 * dedup existed. Idempotent.
 * @param {Object} [opts]
 * @param {number} [opts.threshold] - cosine threshold (defaults to config)
 * @returns {Promise<{removed:number}>}
 */
async function dedupePending({ threshold } = {}) {
  const db = getSqliteDb();
  if (!db) return { removed: 0 };
  const thr = Number.isFinite(threshold) ? threshold : getDedupThreshold();
  const memoryClusters = require('./memory-clusters');
  let removed = 0;
  try {
    // Representative-first ordering: highest priority, then newest, so the item
    // we keep in each group is the strongest/most current one.
    const rows = db.prepare(
      "SELECT id, type, content, priority FROM initiatives WHERE status = 'pending' ORDER BY type, priority DESC, created_at DESC"
    ).all();
    const byType = new Map();
    for (const r of rows) {
      if (!byType.has(r.type)) byType.set(r.type, []);
      byType.get(r.type).push(r);
    }
    for (const [, items] of byType) {
      const kept = []; // { id, emb }
      for (const it of items) {
        const emb = await memoryClusters.generateEmbedding(it.content);
        let dupOf = null;
        if (emb) {
          for (const k of kept) {
            if (k.emb && cosineSim(emb, k.emb) >= thr) { dupOf = k; break; }
          }
        }
        if (dupOf) {
          if (expire(it.id)) {
            removed++;
            console.log(`[Initiatives] dedupePending: expired ${it.id} (duplicate of ${dupOf.id})`);
          }
        } else {
          kept.push({ id: it.id, emb });
        }
      }
    }
    console.log(`[Initiatives] dedupePending removed ${removed} duplicate(s)`);
    return { removed };
  } catch (error) {
    console.error('[Initiatives] dedupePending error:', error.message);
    return { removed };
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

/**
 * Every initiative ever minted, newest first — the full lifecycle
 * (pending/delivered/dismissed/expired) for the history view.
 * @param {Object} [opts]
 * @param {number} [opts.limit=200]
 * @returns {Array}
 */
function listAll({ limit = 200 } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    return db.prepare(`
      SELECT * FROM initiatives
      ORDER BY created_at DESC
      LIMIT ?
    `).all(limit);
  } catch (error) {
    console.error('[Initiatives] listAll error:', error.message);
    return [];
  }
}

/** The single highest-priority pending initiative at or above minPriority. */
function getTopPending(minPriority = 0) {
  const list = listPending({ minPriority, limit: 1 });
  return list.length ? list[0] : null;
}

/**
 * The single best pending initiative eligible for a conversation-open greeting,
 * applying a type-specific bar: most types must clear greetingThreshold, but
 * 'followup' items (conversation follow-ups SNH has been mulling over) surface at
 * the lower followupThreshold. Scans in priority order, so the most pressing
 * eligible item wins regardless of type.
 * @returns {Object|null}
 */
function getTopForGreeting({ greetingThreshold = 7, followupThreshold = 5 } = {}) {
  const pending = listPending({ limit: 100 }); // priority DESC
  for (const it of pending) {
    const bar = it.type === 'followup' ? followupThreshold : greetingThreshold;
    if (it.priority >= bar) return it;
  }
  return null;
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

/**
 * Dismiss every PENDING initiative that points at a given source (e.g. a
 * question that has since been answered). Delivered ones are left as history.
 * @returns {number} count dismissed
 */
function dismissForSource(sourceKind, sourceRef) {
  try {
    const db = getSqliteDb();
    if (!db || !sourceRef) return 0;
    const info = db.prepare(
      "UPDATE initiatives SET status = 'dismissed' WHERE status = 'pending' AND source_kind IS ? AND source_ref IS ?"
    ).run(sourceKind, sourceRef);
    return info.changes;
  } catch (error) {
    console.error('[Initiatives] dismissForSource error:', error.message);
    return 0;
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

/**
 * Persist a conversation-followup trace for the reflection cycle. Records what
 * was reviewed and reasoned about even when no follow-up is produced, so the
 * decision process is queryable (the UI reads these back).
 * @param {Object} trace
 * @returns {string|null} the trace id
 */
function recordFollowupTrace(trace = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return null;
    const id = randomUUID();
    const reviewed = Array.isArray(trace.conversationsReviewed) ? trace.conversationsReviewed : [];
    db.prepare(`
      INSERT INTO followup_traces
        (id, created_at, conversations_reviewed, message_count, reviewed_json,
         related_clusters_json, candidates_json, generated, skipped, reasoning, initiative_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      trace.at || new Date().toISOString(),
      reviewed.length,
      trace.messageCount || 0,
      JSON.stringify(reviewed),
      JSON.stringify(trace.relatedClusters || []),
      JSON.stringify(trace.candidates || []),
      trace.generated || null,
      trace.skipped ? 1 : 0,
      trace.reasoning || '',
      trace.initiativeId || null
    );
    return id;
  } catch (error) {
    console.error('[Initiatives] recordFollowupTrace error:', error.message);
    return null;
  }
}

/** Recent conversation-followup traces (newest first), rehydrated from JSON. */
function listFollowupTraces({ limit = 20 } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    const rows = db.prepare(
      'SELECT * FROM followup_traces ORDER BY created_at DESC LIMIT ?'
    ).all(limit);
    return rows.map(r => ({
      id: r.id,
      at: r.created_at,
      conversationsReviewed: safeParse(r.reviewed_json, []),
      messageCount: r.message_count,
      relatedClusters: safeParse(r.related_clusters_json, []),
      candidates: safeParse(r.candidates_json, []),
      generated: r.generated,
      skipped: !!r.skipped,
      reasoning: r.reasoning,
      initiativeId: r.initiative_id
    }));
  } catch (error) {
    console.error('[Initiatives] listFollowupTraces error:', error.message);
    return [];
  }
}

module.exports = {
  addInitiative,
  dedupePending,
  listPending,
  listAll,
  getTopPending,
  getTopForGreeting,
  get,
  countPending,
  markDelivered,
  dismiss,
  dismissForSource,
  expire,
  updatePriority,
  countUnpromptedDeliveredToday,
  recordFollowupTrace,
  listFollowupTraces
};
