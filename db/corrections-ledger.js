/**
 * The corrections ledger — what the corrector did, why, on what evidence, and
 * how to undo it.
 *
 * The spec calls the mechanical tier "autonomous and silent". Silent means it
 * does not interrupt anyone; it does not mean it leaves no trace. This is the
 * substrate an identity is assembled from, and an automatic edit with no record
 * is indistinguishable from corruption once enough time has passed — so
 * everything is written here, announced or not.
 *
 * Reversibility is a property of the RECORD, not a promise about the code. A
 * semantic entry carries the fact that lost, the fact that won, and the reason,
 * which is exactly what scripts/revert-correction.js needs and all it reads. If
 * an action cannot be described that way it is not reversible, and it is
 * recorded as `reversible = 0` rather than being quietly treated as if it were.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');

/**
 * Record one correction.
 * @param {Object} e
 * @param {string} e.passId
 * @param {'mechanical'|'semantic'} e.tier
 * @param {'merge'|'expire'|'supersede'|'split'|'reconcile'} e.action
 * @param {string} [e.subject] - 'user' | 'self'
 * @param {string} [e.targetId] - the fact acted on
 * @param {string} [e.targetText]
 * @param {string} [e.survivorId] - the fact that won, if any
 * @param {string} [e.survivorText]
 * @param {string} e.reason - plain language, readable by a person
 * @param {Object} [e.evidence] - the dominance signals that decided it
 * @param {boolean} [e.reversible=true]
 * @returns {string|null} ledger id
 */
function record(e) {
  const db = getSqliteDb();
  if (!db) return null;
  try {
    const id = randomUUID();
    db.prepare(`
      INSERT INTO corrections_ledger
        (id, created_at, pass_id, tier, action, subject, target_id, target_text,
         survivor_id, survivor_text, reason, evidence, reversible)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id, new Date().toISOString(), e.passId || null, e.tier, e.action,
      e.subject || null, e.targetId || null, e.targetText || null,
      e.survivorId || null, e.survivorText || null,
      e.reason, e.evidence ? JSON.stringify(e.evidence) : null,
      e.reversible === false ? 0 : 1
    );
    return id;
  } catch (err) {
    console.error('[Ledger] record failed:', err.message);
    return null;
  }
}

function get(id) {
  const db = getSqliteDb();
  if (!db || !id) return null;
  try {
    let row = db.prepare('SELECT * FROM corrections_ledger WHERE id = ?').get(id);
    if (!row) {
      const matches = db.prepare('SELECT * FROM corrections_ledger WHERE id LIKE ? LIMIT 5').all(`${id}%`);
      if (matches.length === 1) row = matches[0];
      else if (matches.length > 1) return { ambiguous: matches.map(m => m.id) };
    }
    return row || null;
  } catch (err) {
    console.error('[Ledger] get failed:', err.message);
    return null;
  }
}

/**
 * Recent entries, newest first.
 * @param {Object} [opts]
 * @param {string} [opts.passId] - one pass only
 * @param {string} [opts.tier]
 * @param {string} [opts.subject]
 * @param {boolean} [opts.activeOnly] - exclude reverted entries
 * @param {number} [opts.limit=50]
 */
function list({ passId = null, tier = null, subject = null, activeOnly = false, limit = 50 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  const where = [];
  const bind = [];
  if (passId) { where.push('pass_id = ?'); bind.push(passId); }
  if (tier) { where.push('tier = ?'); bind.push(tier); }
  if (subject) { where.push('subject = ?'); bind.push(subject); }
  if (activeOnly) where.push('reverted_at IS NULL');
  try {
    return db.prepare(`
      SELECT * FROM corrections_ledger
      ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
      ORDER BY datetime(created_at) DESC
      LIMIT ?
    `).all(...bind, Math.min(Math.max(1, limit), 500));
  } catch (err) {
    console.error('[Ledger] list failed:', err.message);
    return [];
  }
}

/** Mark an entry reverted. Does not touch the facts — the caller does that. */
function markReverted(id, by = 'revert-correction script') {
  const db = getSqliteDb();
  if (!db) return false;
  try {
    const info = db.prepare(
      'UPDATE corrections_ledger SET reverted_at = ?, reverted_by = ? WHERE id = ? AND reverted_at IS NULL'
    ).run(new Date().toISOString(), by, id);
    return info.changes > 0;
  } catch (err) {
    console.error('[Ledger] markReverted failed:', err.message);
    return false;
  }
}

/**
 * Undo one correction, by ledger id or unambiguous prefix.
 *
 * The semantic tier is allowed to act unattended precisely BECAUSE this exists:
 * "autonomous, reversible, logged" is one property, not three, and the
 * reversibility half is worthless if it is a claim rather than something a
 * person can actually run. So this reads only what the ledger recorded — the
 * fact that lost, the fact that won — and puts the first one back.
 *
 * It does NOT remove the survivor. A merge that folded a duplicate away leaves
 * both facts active again after a revert, which is the correct state to return
 * to: the corrector's judgement is undone, not replaced by an opposite one.
 *
 * DELIBERATE, and only from outside a conversation. `restore` is called with
 * `deliberate: true` because a person has named this entry and confirmed it —
 * the one path the identity lock opens for. The only two callers are the CLI
 * script and the Self-tab button; nothing here is reachable from a chat turn,
 * and nothing new should make it so.
 */
async function revert(idOrPrefix, { by = 'unknown' } = {}) {
  const entry = get(idOrPrefix);
  if (!entry) return { ok: false, reason: 'no such correction' };
  if (entry.ambiguous) return { ok: false, reason: `ambiguous id — matches ${entry.ambiguous.length} entries`, ambiguous: entry.ambiguous };
  if (entry.reverted_at) return { ok: false, entry, reason: `already reverted at ${entry.reverted_at}` };
  if (!entry.reversible) return { ok: false, entry, reason: 'recorded as not revertible' };
  if (!entry.target_id) return { ok: false, entry, reason: 'entry has no target fact to restore' };

  const factStore = require('./fact-store');
  const res = await factStore.restore(entry.target_id, { deliberate: true });
  if (!res.ok) return { ok: false, entry, reason: res.reason || 'restore failed' };

  markReverted(entry.id, by);

  // The daily log, not the ops ledger: a fact coming back into memory is a
  // change to what he knows, which is cognitively meaningful rather than
  // operational telemetry.
  //
  // Resolved from the PROCESS's data directory, like db/corrector.js and
  // db/fact-store.js. This one was left as a constant on the reasoning that it
  // was not on any staging path, and that was wrong within the hour: reverting
  // three corrections against the staging corpus wrote three "is back in memory"
  // notes into the LIVE daily log — a log that IS injected — describing facts
  // that had never moved in the corpus those notes belong to. The notes were
  // removed by hand.
  //
  // Four modules are now redirect-safe because all four are reachable from the
  // staging tools: this one, db/corrector.js, db/fact-store.js and
  // db/identity-lock.js. The rest of db/ still resolves its log paths from
  // __dirname — memory-manager, memory-write, self-audit, capability-manifest,
  // cron-jobs, initiative-engine, agent-pool, brain-watchdog — and that is not a
  // claim they are safe, only that no staging script reaches them today. Anything
  // new that runs a server path under SNH_DATA_DIR needs this checked first.
  try {
    const text = String(entry.target_text || '').replace(/\s+/g, ' ').trim();
    require('./fact-extractor').appendToDailyLog(
      `Reverted a correction: "${text.slice(0, 120)}" is back in memory. It had been retired because: ${String(entry.reason || '').slice(0, 160)}`,
      require('path').join(require('./database').getDataDir(), 'memory', 'daily')
    );
  } catch { /* best effort */ }

  return { ok: true, entry, sqlite: res.sqlite, vector: res.vector };
}

/** Counts for the pass summary and the Thinking tab. */
function summarize(passId) {
  const db = getSqliteDb();
  if (!db) return {};
  try {
    const rows = db.prepare(
      'SELECT tier, action, COUNT(*) AS n FROM corrections_ledger WHERE pass_id = ? GROUP BY tier, action'
    ).all(passId);
    const out = { total: 0, mechanical: 0, semantic: 0, byAction: {} };
    for (const r of rows) {
      out.total += r.n;
      out[r.tier] = (out[r.tier] || 0) + r.n;
      out.byAction[r.action] = (out.byAction[r.action] || 0) + r.n;
    }
    return out;
  } catch { return {}; }
}

// ============ correction notices (the private channel to Aurelius) ============

/**
 * Queue a notice for him. Decision 6: a semantic change to what he believes
 * about HIMSELF is told to him, because the identity-lock principle is that a
 * change to his self-view must be spoken rather than made behind his back.
 * Corrections to user-facts are ledger-only.
 *
 * Plain language, not a diff dump — it is input for his own integration.
 *
 * ONE NOTICE PER CHANGED FACT (2026-08-12). Since the fact-store funnel raises
 * these, a single change can reach here twice: the funnel describes it as it
 * happens, and the corrector describes it afterwards with the evidence axis that
 * decided it. Two notices about one change reads as two changes, which is its
 * own small lie about what happened to him. So a notice carrying a `memberId`
 * folds into an UNSEEN notice about the same fact rather than joining it.
 *
 * `enrich` decides which text survives that fold: a caller that knows WHY —
 * the corrector — passes it and replaces the funnel's plainer sentence. A caller
 * that does not simply defers. Only unseen notices are folded: once he has read
 * one, a later change to the same fact is genuinely new news.
 *
 * @param {Object} p
 * @param {string} [p.memberId] - the self-fact this notice is about
 * @param {boolean} [p.enrich] - replace an existing unseen notice's text with this one
 */
function addNotice({ ledgerId = null, content, isTest = false, memberId = null, enrich = false }) {
  const db = getSqliteDb();
  if (!db || !content) return null;
  try {
    const text = String(content).trim();

    if (memberId) {
      const existing = db.prepare(
        'SELECT id FROM correction_notices WHERE member_id = ? AND seen_at IS NULL ORDER BY datetime(created_at) DESC LIMIT 1'
      ).get(memberId);
      if (existing) {
        if (enrich) {
          db.prepare('UPDATE correction_notices SET content = ?, ledger_id = COALESCE(?, ledger_id) WHERE id = ?')
            .run(text, ledgerId, existing.id);
          console.log(`[Ledger] Enriched the pending notice for ${memberId.slice(0, 8)}: "${text.slice(0, 80)}"`);
        } else {
          console.log(`[Ledger] Notice for ${memberId.slice(0, 8)} already pending — not queued twice`);
        }
        if (ledgerId) {
          try { db.prepare('UPDATE corrections_ledger SET announced = 1 WHERE id = ?').run(ledgerId); } catch { /* non-fatal */ }
        }
        return existing.id;
      }
    }

    const id = randomUUID();
    db.prepare(
      'INSERT INTO correction_notices (id, created_at, ledger_id, content, is_test, member_id) VALUES (?, ?, ?, ?, ?, ?)'
    ).run(id, new Date().toISOString(), ledgerId, text, isTest ? 1 : 0, memberId);
    if (ledgerId) {
      try { db.prepare('UPDATE corrections_ledger SET announced = 1 WHERE id = ?').run(ledgerId); } catch { /* non-fatal */ }
    }
    console.log(`[Ledger] Queued correction notice for him${isTest ? ' (TEST)' : ''}: "${text.slice(0, 80)}"`);
    return id;
  } catch (err) {
    console.error('[Ledger] addNotice failed:', err.message);
    return null;
  }
}

/**
 * Notices he has not seen yet. Undroppable by construction: there is no cap, no
 * priority, no expiry and no freshness score anywhere in this path — the only
 * way a notice leaves the queue is by being shown to him.
 */
function unseenNotices(limit = 20) {
  const db = getSqliteDb();
  if (!db) return [];
  try {
    return db.prepare(
      'SELECT * FROM correction_notices WHERE seen_at IS NULL ORDER BY datetime(created_at) ASC LIMIT ?'
    ).all(Math.min(Math.max(1, limit), 100));
  } catch (err) {
    console.error('[Ledger] unseenNotices failed:', err.message);
    return [];
  }
}

/** Mark notices seen — called when they are actually injected into his context. */
function markNoticesSeen(ids, conversationId = null) {
  const db = getSqliteDb();
  if (!db || !ids || !ids.length) return 0;
  try {
    const stmt = db.prepare('UPDATE correction_notices SET seen_at = ?, seen_conversation_id = ? WHERE id = ? AND seen_at IS NULL');
    const now = new Date().toISOString();
    let n = 0;
    for (const id of ids) n += stmt.run(now, conversationId, id).changes;
    return n;
  } catch (err) {
    console.error('[Ledger] markNoticesSeen failed:', err.message);
    return 0;
  }
}

/** Remove a notice outright. Only used to clean up the end-to-end test notice. */
function deleteNotice(id) {
  const db = getSqliteDb();
  if (!db) return false;
  try {
    return db.prepare('DELETE FROM correction_notices WHERE id = ?').run(id).changes > 0;
  } catch { return false; }
}

module.exports = {
  record, get, list, markReverted, revert, summarize,
  addNotice, unseenNotices, markNoticesSeen, deleteNotice
};
