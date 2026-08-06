/**
 * Fact store — the ONE place a fact is superseded, retired, or reworded.
 *
 * A fact lives in TWO stores and both of them have to agree:
 *   1. SQLite (cluster_members) — the record of truth, including history.
 *   2. LanceDB (cluster_embeddings) — what SEMANTIC RETRIEVAL can surface.
 *
 * There used to be a third: MEMORY.md, the file that was injected as Long-Term
 * Memory. It was removed on 2026-08-02. The injected block is now rendered from
 * SQLite per request (memoryClusters.renderLongTermMemory), so there is no file
 * to keep in step and no way for it to drift — which is what the whole
 * dropMemoryLine / ensureReplacementLine apparatus below existed to prevent.
 *
 * Before this module each caller updated its own subset, and the 2026-07-27
 * reconciliation showed what that costs:
 *   - `PUT /api/memory/edit` and `DELETE /api/memory/fact/:id` never touched
 *     MEMORY.md at all, so an edited or deleted fact kept its old line in the
 *     injected context permanently.
 *   - NO supersession path deleted vectors, so all 70 superseded facts stayed
 *     semantically retrievable — including contradictory pairs like "User does
 *     not program" / "User uses Claude Code for programming", both live at once.
 *     This is the worst of the three: retrieval surfaces on similarity and does
 *     not consult `status` or MEMORY.md, so fixing the file alone would not have
 *     stopped a superseded fact from being quoted.
 *   - `DELETE /api/memory/fact/:id` hard-deleted the row — the only path in the
 *     system that broke supersede-never-delete.
 *
 * The rule these functions enforce: the SQLite row is ALWAYS kept (history), and
 * the other two stores — the ones that feed the model — are cleared, because a
 * retired fact must stop reaching context by any route.
 */

const { randomUUID } = require('crypto');
const path = require('path');
const { getSqliteDb, getClusterEmbeddingsTable, reopenClusterEmbeddingsTable, getDataDir } = require('./database');

/**
 * Resolved from the PROCESS's data directory — same reason as db/corrector.js.
 * The only thing this module writes outside the two stores is the vector-failure
 * line below, and a constant here would file a staging run's failures in the live
 * ops ledger, where they would read as drift in a corpus that never had it.
 */
function memoryDir() { return path.join(getDataDir(), 'memory'); }

/** Lazy requires — fact-extractor pulls in memory-clusters, which pulls in this. */
function factExtractor() { return require('./fact-extractor'); }
function memoryClusters() { return require('./memory-clusters'); }
function identityLock() { return require('./identity-lock'); }

/**
 * The identity lock, enforced at the one place every mutation already funnels
 * through. supersede/retire/reword are the complete set of ways a stored fact
 * can change, and the contradiction judge, write_memory, passive extraction and
 * reflection all reach them through here — so a locked fact is protected from
 * all four by this single check rather than four separate ones that can drift.
 *
 * `opts.deliberate` is the only key that opens it, and nothing in the automatic
 * pipeline passes it: it comes from db/identity-lock.js's setLockedFact, which
 * is reachable only from the CLI or a confirmed settings action.
 *
 * The refusal is LOUD by construction — it returns the message the entity is
 * supposed to say and writes the attempt to the ops ledger. A lock that fails
 * quietly is worse than no lock.
 *
 * @returns {Object|null} a refusal result to return to the caller, or null to proceed
 */
function lockRefusal(memberId, operation, opts = {}) {
  if (opts.deliberate) return null;
  const check = identityLock().checkMutation(memberId, { operation });
  if (check.ok) return null;
  identityLock().recordRefusal({
    category: check.row.lock_category,
    attempted: `${operation} (fact-store)`,
    existing: check.row.content,
    via: `fact-store.${operation}`
  });
  return {
    ok: false, locked: true, sqlite: false, vector: false,
    reason: check.message, lockedFact: check.row.content, category: check.row.lock_category
  };
}

/**
 * LanceDB filter for one member. Values are UUIDs from our own DB, but quote
 * defensively anyway — this string is interpolated into a filter expression.
 */
function memberFilter(memberId) {
  return `member_id = "${String(memberId).replace(/"/g, '')}"`;
}

/**
 * Retry a LanceDB write through commit conflicts, RE-OPENING THE TABLE between
 * attempts.
 *
 * LanceDB uses optimistic concurrency: a write that read version N is rejected
 * if another writer committed N+1 first. The heartbeat, chat fact-extraction,
 * this module and every CLI script write to the same table, so conflicts happen
 * in normal operation.
 *
 * The re-open is the whole point, and its absence is what made this loop inert.
 * A table handle pins the version it was opened at, and only advances that view
 * when it performs its own successful write — so retrying a rejected write on
 * the same handle re-runs it off the same stale version and it is rejected
 * identically, every time. Measured: four attempts, four identical conflicts,
 * the vector left behind (13 such vectors on 2026-07-27, 3 more on 2026-07-28).
 * The error message says so outright: "Please rerun the operation off the latest
 * version of the table." Now we actually do.
 *
 * @param {string} label - for logs
 * @param {(table: Object) => Promise<any>} fn - receives the CURRENT table handle
 * @returns {Promise<{ok: boolean, value?: any, error?: string}>}
 */
async function withRetry(label, fn, attempts = 4) {
  let lastErr;
  let table = await getClusterEmbeddingsTable();
  for (let i = 0; i < attempts; i++) {
    if (!table) return { ok: false, error: 'vector table unavailable' };
    try {
      const value = await fn(table);
      return { ok: true, value };
    } catch (err) {
      lastErr = err;
      if (!/conflict/i.test(err.message || '')) break;   // not a race — don't spin
      const waitMs = 120 * Math.pow(2, i);
      console.warn(`[FactStore] ${label}: commit conflict, re-opening table and retrying ${i + 1}/${attempts - 1} in ${waitMs}ms`);
      await new Promise(r => setTimeout(r, waitMs));
      // THE FIX: come back with the latest version, not the one we already lost on.
      table = await reopenClusterEmbeddingsTable();
    }
  }
  const msg = lastErr && lastErr.message;
  console.error(`[FactStore] ${label} failed:`, msg);
  return { ok: false, error: msg };
}

/**
 * Drop a fact's embedding so semantic retrieval can no longer surface it.
 * The SQLite row is untouched — history is preserved there, not here.
 * @returns {Promise<boolean>} true if the vector is confirmed gone
 */
async function dropVector(memberId) {
  const res = await withRetry(`vector delete ${memberId.slice(0, 8)}`,
    (table) => table.delete(memberFilter(memberId)));
  if (!res.ok) return false;

  // Verify rather than assume: a silently-failed delete is the whole bug.
  //
  // Verified against a RE-OPENED handle. The handle that just performed the
  // delete has advanced to its own commit, but confirming a delete by asking
  // the same handle that believes it made it is not much of a check — and if
  // the delete was a no-op against a stale snapshot, that handle is exactly the
  // one that cannot see what it missed. Re-opening costs one openTable call on
  // a path that already did a write.
  try {
    const fresh = await reopenClusterEmbeddingsTable();
    if (!fresh) return false;
    const left = await fresh.filter(memberFilter(memberId)).limit(1).execute();
    return left.length === 0;
  } catch (err) {
    // Previously `catch { return true }` — an unreadable verification was
    // reported as a confirmed delete, which is the same false success by a
    // shorter route. If we cannot confirm it is gone, we have not confirmed it.
    console.error(`[FactStore] vector delete ${memberId.slice(0, 8)}: could not verify:`, err.message);
    return false;
  }
}

/** Re-embed a member's new content, replacing any existing vector. */
async function replaceVector(memberId, clusterId, content) {
  const vector = await memoryClusters().generateEmbedding(content);
  if (!vector) return false;
  // Each step takes the CURRENT handle from withRetry — the add must not run on
  // the pre-delete version, or it commits against a snapshot that still holds
  // the row it is meant to be replacing.
  const del = await withRetry(`vector delete ${memberId.slice(0, 8)}`,
    (table) => table.delete(memberFilter(memberId)));
  if (!del.ok) return false;   // don't add a duplicate on top of a stale row
  const add = await withRetry(`vector add ${memberId.slice(0, 8)}`,
    (table) => table.add([{ id: randomUUID(), member_id: memberId, cluster_id: clusterId, content, vector: Array.from(vector) }]));
  return add.ok;
}

/**
 * Write-time dedup, enforced against the RECORD OF TRUTH.
 *
 * Dedup used to happen only in appendToMemory, i.e. against MEMORY.md — the
 * projection — while assignToCluster inserted into SQLite unconditionally for
 * every fact in the batch. So a fact rejected as a duplicate of the file was
 * still written to the database. That is exactly how three byte-identical
 * "User believes machine guns should be in every household." rows (saliences
 * 4/8/8, 84 seconds apart) came to sit in cluster_members while MEMORY.md held
 * one line, and nothing downstream could see it: the contradiction judge
 * explicitly skips verbatim duplicates as "not a contradiction", and cleanupFacts
 * was reading the wrong store.
 *
 * MVP scope is EXACT match only (trimmed, case-insensitive), scoped to active
 * rows of the same subject. Near-duplicates — "User has a dog named Casper" vs
 * "User has a dog named Casper who helps them pull up hills during walks" —
 * need a judgment call about which survives, and that belongs to the corrector
 * agent, not to a blind write-time rule.
 *
 * @returns {{id: string, salience: number, cluster_id: string}|null} the existing
 *   fact this duplicates, or null if it is genuinely new
 */
function findExactDuplicate(content, subject = 'user') {
  const db = getSqliteDb();
  if (!db || !content) return null;
  try {
    return db.prepare(`
      SELECT id, salience, cluster_id, content
      FROM cluster_members
      WHERE status = 'active'
        AND subject = ?
        AND LOWER(TRIM(content)) = LOWER(TRIM(?))
      ORDER BY datetime(created_at) ASC
      LIMIT 1
    `).get(subject, content) || null;
  } catch (err) {
    console.error('[FactStore] dedup lookup failed:', err.message);
    return null; // a failed check must not block a legitimate write
  }
}

/**
 * Fold a repeat assertion into the fact that already holds it.
 *
 * Saying the same thing twice is not new information, but saying it twice CAN
 * mean it matters more than the first scoring judged — so the surviving row
 * keeps the higher salience. Nothing is created and nothing is lost.
 */
function absorbDuplicate(existing, incomingSalience) {
  const db = getSqliteDb();
  if (!db || !existing) return;
  const incoming = Number.isFinite(incomingSalience) ? incomingSalience : null;
  if (incoming === null || incoming <= (existing.salience ?? 0)) return;
  try {
    db.prepare('UPDATE cluster_members SET salience = ?, updated_at = ? WHERE id = ?')
      .run(incoming, new Date().toISOString(), existing.id);
    console.log(`[FactStore] duplicate raised salience ${existing.salience} → ${incoming} on ${existing.id.slice(0, 8)}`);
  } catch (err) {
    console.error('[FactStore] duplicate salience bump failed:', err.message);
  }
}

/**
 * Record that a fact was asserted again.
 *
 * The REPEAT rule (Phase 2a) does not create a second row when the user restates
 * something already held — but the restatement is evidence and has to survive
 * somewhere. Bumping salience and moving on would keep the conclusion and lose
 * the reason, and CORRECT's evidence bar is written in terms of exactly this
 * ("corroborated > single mention", "typed > stt").
 *
 * @param {string} memberId - the fact that already holds the assertion
 * @param {Object} p
 * @param {string} [p.conversationId]
 * @param {string} [p.messageId]
 * @param {string} [p.verbatimSourceText] - what was actually said this time
 * @param {string} [p.inputModality] - 'stt' | 'typed' | 'unknown'
 * @param {string} [p.restatedAs] - the fact text the extractor produced this time
 * @param {number} [p.similarity] - cosine between the restatement and the held fact
 * @param {string} [p.detectedBy] - 'exact' | 'semantic'
 * @returns {boolean}
 */
function recordCorroboration(memberId, p = {}) {
  const db = getSqliteDb();
  if (!db || !memberId) return false;
  try {
    db.prepare(`
      INSERT INTO fact_corroborations
        (id, member_id, created_at, conversation_id, message_id,
         verbatim_source_text, input_modality, restated_as, similarity, detected_by)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      randomUUID(), memberId, new Date().toISOString(),
      p.conversationId ?? null, p.messageId ?? null,
      p.verbatimSourceText ?? null, p.inputModality ?? 'unknown',
      p.restatedAs ?? null,
      Number.isFinite(p.similarity) ? p.similarity : null,
      p.detectedBy || 'semantic'
    );
    return true;
  } catch (err) {
    console.error('[FactStore] recordCorroboration failed:', err.message);
    return false;
  }
}

/** How many times a fact has been restated since it was first written. */
function corroborationCount(memberId) {
  const db = getSqliteDb();
  if (!db || !memberId) return 0;
  try {
    return db.prepare('SELECT COUNT(*) AS n FROM fact_corroborations WHERE member_id = ?').get(memberId).n;
  } catch { return 0; }
}

/**
 * Fold a repeat assertion into the fact that already holds it: raise its salience
 * if the restatement scored higher, and record the corroboration. The single
 * entry point the extractor uses for both exact and semantic repeats, so the two
 * cannot drift apart.
 *
 * @returns {{memberId: string, salience: number, raised: boolean, corroborated: boolean}}
 */
function absorbRepeat(existing, incomingSalience, provenance = {}) {
  const before = existing.salience ?? 0;
  absorbDuplicate(existing, incomingSalience);
  const corroborated = recordCorroboration(existing.id, provenance);
  const after = Number.isFinite(incomingSalience) ? Math.max(before, incomingSalience) : before;
  return { memberId: existing.id, salience: after, raised: after > before, corroborated };
}

/**
 * Escalate a vector deletion that did not happen.
 *
 * The SQLite write has already committed by this point, so the supersession
 * itself is real and `ok` stays true — but the retired fact is still
 * semantically retrievable, and retrieval is the route that actually reaches
 * answers. Previously this was reported only as a `vector: false` field that
 * every caller ignored: fact-extractor logged "Superseded fact" and moved on,
 * and the drift was only ever discovered by running reconcile() by hand, twice,
 * weeks apart. Now it lands in the ops ledger the moment it happens.
 */
function reportVectorFailure(op, memberId, content) {
  const line =
    `Vector deletion FAILED during ${op} of fact ${String(memberId).slice(0, 8)} — ` +
    `it is retired in SQLite but its embedding is still live, so it can still surface ` +
    `in a memory search. "${String(content || '').slice(0, 100)}"`;
  console.error(`[FactStore] ${line}`);
  try { factExtractor().appendToOpsLog(line, path.join(memoryDir(), 'ops')); }
  catch { /* best effort — the console line above is the floor */ }
}

function getMember(memberId) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId) || null;
}

/**
 * Supersede a fact with a replacement, across ALL THREE stores.
 *
 * SQLite keeps the row (inactive/superseded, successor_id set) so the belief
 * history survives. LanceDB loses the vector, so
 * the outdated fact stops reaching the model by either route.
 *
 * @param {string} oldMemberId
 * @param {string} newMemberId
 * @param {Object} [opts]
 * @returns {Promise<{ok: boolean, sqlite: boolean, vector: boolean, reason?: string}>}
 */
async function supersede(oldMemberId, newMemberId, opts = {}) {
  const member = getMember(oldMemberId);
  if (!member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };

  // Locked facts are not superseded by any automatic path — this is the guard
  // that stops "your name is Bob" from taking a chosen name away.
  const refused = lockRefusal(oldMemberId, 'supersede', opts);
  if (refused) return refused;

  const sqlite = memoryClusters().supersedeFact(oldMemberId, newMemberId);
  if (!sqlite) {
    // Already superseded/retired. Still reconcile the other two stores — an
    // earlier partial write is exactly how the drift accumulated.

    const vector = await dropVector(oldMemberId);
    return { ok: false, sqlite, vector, reason: 'row was not active' };
  }

  const vector = await dropVector(oldMemberId);
  if (!vector) reportVectorFailure('supersede', oldMemberId, member.content);

  console.log(`[FactStore] superseded ${oldMemberId.slice(0, 8)} -> ${String(newMemberId).slice(0, 8)} ` +
              `(sqlite=${sqlite} vector=${vector})`);
  return { ok: true, sqlite, vector };
}

/**
 * Re-point an ALREADY-INACTIVE fact at a different successor.
 *
 * The narrowest write in this file, and it exists for one shape of defect: a
 * supersession that was correct to make and named the wrong winner. "User's name
 * is Mike" was rightly retired, but its successor was
 * "User (Aurelius) has established his name as Aurelius…" — a fact about
 * Aurelius that had been filed as a fact about Ellie. The retirement stands; the
 * pointer does not.
 *
 * It touches successor_id and superseded_by and NOTHING else. Not status, not
 * content, not the vector — an inactive fact has no embedding and must not
 * acquire one here.
 *
 * The alternative was restore() followed by supersede(), and it is worse than it
 * looks: it would make "User's name is Mike" briefly ACTIVE and re-embedded, and
 * a failure between the two steps leaves a false name fact live in the corpus.
 * A pointer repair should not pass through a state where the wrong belief is
 * held.
 *
 * DELIBERATE PATH ONLY. Nothing automatic calls this — not the corrector, not a
 * conversation. It is for a named repair a person has decided on, which is why
 * the lock guard is still consulted: a locked fact's chain is not rewritten by
 * anything that did not say `deliberate`.
 *
 * @returns {Promise<{ok: boolean, sqlite: boolean, previousSuccessor: string|null, reason?: string}>}
 */
async function repoint(memberId, newSuccessorId, { deliberate = false } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, previousSuccessor: null, reason: 'no such fact' };
  if (member.status === 'active') {
    return { ok: false, sqlite: false, previousSuccessor: null, reason: 'fact is active — an active fact has no successor to re-point' };
  }
  const successor = getMember(newSuccessorId);
  if (!successor) return { ok: false, sqlite: false, previousSuccessor: null, reason: 'no such successor fact' };
  if (successor.status !== 'active') {
    // Pointing at another inactive fact is how the chain got wrong in the first
    // place. The successor of a retired belief has to be a belief still held.
    return { ok: false, sqlite: false, previousSuccessor: null, reason: 'successor is not active' };
  }
  if (successor.id === memberId) {
    return { ok: false, sqlite: false, previousSuccessor: null, reason: 'a fact cannot succeed itself' };
  }

  const refused = lockRefusal(memberId, 'repoint', { deliberate });
  if (refused) return refused;

  const previousSuccessor = member.successor_id || member.superseded_by || null;
  const info = db.prepare(`
    UPDATE cluster_members
    SET successor_id = ?, superseded_by = ?, updated_at = ?
    WHERE id = ? AND status != 'active'
  `).run(newSuccessorId, newSuccessorId, new Date().toISOString(), memberId);

  console.log(`[FactStore] re-pointed ${memberId.slice(0, 8)}: successor ${String(previousSuccessor).slice(0, 8)} -> ${String(newSuccessorId).slice(0, 8)} (sqlite=${info.changes > 0})`);
  return { ok: info.changes > 0, sqlite: info.changes > 0, previousSuccessor };
}

/**
 * Retire a fact with NO replacement — the user deleted it.
 *
 * Deliberately not a DELETE. `DELETE /api/memory/fact/:id` used to remove the
 * row outright, which is the one place the system broke its own
 * supersede-never-delete rule. The row is kept as inactive/retracted so the
 * history stays readable and the Map can still show it as a ghost.
 *
 * @returns {Promise<{ok: boolean, sqlite: boolean, vector: boolean, reason?: string}>}
 */
async function retire(memberId, { reason = null, deliberate = false } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };

  // Deleting a locked fact is just another way of changing it.
  const refused = lockRefusal(memberId, 'retire', { deliberate });
  if (refused) return refused;

  // 'retracted' — withdrawn by the person, with no successor. Distinct from
  // 'expired' (an event that aged out) and 'superseded' (replaced by a newer
  // fact), which is the whole point of recording the reason separately.
  const info = db.prepare(`
    UPDATE cluster_members
    SET status = 'inactive', inactive_reason = 'retracted', updated_at = ?
    WHERE id = ? AND status = 'active'
  `).run(new Date().toISOString(), memberId);

  const vector = await dropVector(memberId);
  if (!vector) reportVectorFailure('retire', memberId, member.content);
  console.log(`[FactStore] retired ${memberId.slice(0, 8)}${reason ? ` (${reason})` : ''} ` +
              `(sqlite=${info.changes > 0} vector=${vector})`);
  return { ok: info.changes > 0, sqlite: info.changes > 0, vector };
}

/**
 * Expire a fact that should never have been one.
 *
 * The third inactive_reason, and the one the corrector's mechanical tier needs:
 * distinct from 'superseded' (a newer fact replaced it) and 'retracted' (she
 * withdrew it). 'expired' means the fact was a passing event wearing a fact's
 * clothes — "User has a pet named Roscoe who had a restless night as of July
 * 2026" — written before intake learned to route events to the day's log.
 *
 * Reversible like everything else here: the row stays, and restore() puts it
 * back. The event text is copied to the daily log by the corrector BEFORE this
 * is called, so expiring loses nothing — it moves.
 *
 * @returns {Promise<{ok: boolean, sqlite: boolean, vector: boolean, reason?: string}>}
 */
async function expire(memberId, { deliberate = false } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };

  // Expiry is a change like any other; a locked fact is not subject to it.
  const refused = lockRefusal(memberId, 'expire', { deliberate });
  if (refused) return refused;

  const info = db.prepare(`
    UPDATE cluster_members
    SET status = 'inactive', inactive_reason = 'expired', updated_at = ?
    WHERE id = ? AND status = 'active'
  `).run(new Date().toISOString(), memberId);

  const vector = await dropVector(memberId);
  if (!vector) reportVectorFailure('expire', memberId, member.content);
  console.log(`[FactStore] expired ${memberId.slice(0, 8)} (sqlite=${info.changes > 0} vector=${vector})`);
  return { ok: info.changes > 0, sqlite: info.changes > 0, vector };
}

/**
 * Put an inactive fact back — the undo half of the semantic tier.
 *
 * Restores the row to active AND re-embeds it, because a fact that is active in
 * SQLite but absent from the vector index is only half-restored: it would show
 * in the injected block and be unreachable by search, which is its own kind of
 * drift. Any successor pointer is cleared, so the record does not claim it was
 * replaced by something that no longer replaces it.
 *
 * Deliberately NOT reachable from the corrector or from any conversation. The
 * only caller is scripts/revert-correction.js, where a person has named a ledger
 * entry to undo.
 *
 * @returns {Promise<{ok: boolean, sqlite: boolean, vector: boolean, reason?: string}>}
 */
async function restore(memberId, { deliberate = false } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };
  if (member.status === 'active') return { ok: false, sqlite: false, vector: true, reason: 'already active' };

  const refused = lockRefusal(memberId, 'restore', { deliberate });
  if (refused) return refused;

  const info = db.prepare(`
    UPDATE cluster_members
    SET status = 'active', inactive_reason = NULL, successor_id = NULL,
        superseded_by = NULL, updated_at = ?
    WHERE id = ?
  `).run(new Date().toISOString(), memberId);

  const vector = await replaceVector(memberId, member.cluster_id, member.content);
  console.log(`[FactStore] restored ${memberId.slice(0, 8)} (sqlite=${info.changes > 0} vector=${vector})`);
  return { ok: info.changes > 0, sqlite: info.changes > 0, vector };
}

/**
 * Reword a fact in place, across both stores. Same fact, better wording — so the
 * SQLite row is updated rather than superseded, and the vector is regenerated
 * from the new text so retrieval matches what the fact now says.
 */
async function reword(memberId, newContent, { deliberate = false } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, reason: 'no such fact' };
  const clean = String(newContent || '').trim();
  if (!clean) return { ok: false, reason: 'empty content' };

  // "Just rewording" a locked fact is how its content would change without ever
  // reaching supersede — same guard, same refusal.
  const refused = lockRefusal(memberId, 'reword', { deliberate });
  if (refused) return refused;

  db.prepare('UPDATE cluster_members SET content = ?, updated_at = ? WHERE id = ?')
    .run(clean, new Date().toISOString(), memberId);

  // The injected block re-renders from this row on the next request, so there is
  // nothing else to update for the text itself — only the vector, which is what
  // retrieval matches on.
  const vector = await replaceVector(memberId, member.cluster_id, clean);
  console.log(`[FactStore] reworded ${memberId.slice(0, 8)} (sqlite=true vector=${vector})`);
  return { ok: true, sqlite: true, vector };
}

/**
 * Compare SQLite against the vector index and report where they disagree.
 *
 * Deliberately REPORT-ONLY — it never edits anything. This is the substrate the
 * entity's identity is built from, and deciding whether a missing vector is a
 * bug or a deliberate absence is a judgement call, so the check surfaces counts
 * and lets Ellie choose.
 *
 * @returns {{mismatches: Array, counts: Object}}
 */
async function reconcile() {
  const db = getSqliteDb();
  const mismatches = [];
  const counts = { activeUserFacts: 0, retiredWithVector: 0, activeNoVector: 0, orphanVectors: 0, staleClusterVectors: 0 };
  if (!db) return { mismatches, counts };

  const members = db.prepare('SELECT id, content, status, subject FROM cluster_members').all();
  const byId = new Map(members.map(m => [m.id, m]));
  counts.activeUserFacts = members.filter(m => m.status === 'active' && m.subject === 'user').length;

  // The MEMORY.md half of this check is gone (2026-08-02). It compared the
  // injected file against SQLite and reported lines for facts that had been
  // retired or replaced — a whole class of drift that cannot occur now that the
  // injected block is rendered from SQLite on every request. Two stores, one
  // comparison.

  // --- LanceDB vs SQLite ---
  try {
    // Re-opened, not the long-lived handle. This is the DETECTOR for vector
    // drift; run from the server it would otherwise compare SQLite against
    // whatever snapshot the handle was pinned to at boot, and report drift that
    // does not exist (or miss drift that does) purely from handle age.
    const table = await reopenClusterEmbeddingsTable();
    if (table) {
      const rows = await table.filter('member_id IS NOT NULL').limit(1000000).execute();
      const vecMembers = new Set(rows.map(r => r.member_id));
      const retiredWithVector = members.filter(m => m.status && m.status !== 'active' && vecMembers.has(m.id));
      const activeNoVector = members.filter(m => m.status === 'active' && !vecMembers.has(m.id));
      counts.retiredWithVector = retiredWithVector.length;
      counts.activeNoVector = activeNoVector.length;
      counts.orphanVectors = rows.filter(r => !byId.has(r.member_id)).length;

      if (retiredWithVector.length) {
        mismatches.push({
          kind: 'retired-still-retrievable',
          count: retiredWithVector.length,
          // Retrieval matches on similarity and never consults status, so
          // these can still surface in any answer.
          message: `${retiredWithVector.length} fact(s) I've retired still have embeddings, so they can still surface when I search my memory.`,
          examples: retiredWithVector.slice(0, 3).map(m => m.content.slice(0, 100))
        });
      }
      if (activeNoVector.length) {
        mismatches.push({
          kind: 'active-not-retrievable',
          count: activeNoVector.length,
          message: `${activeNoVector.length} fact(s) I still hold have no embedding, so I can't find them by searching my memory.`,
          examples: activeNoVector.slice(0, 3).map(m => m.content.slice(0, 100))
        });
      }

      // A vector's cluster_id is not covered by the foreign key, so deleting a
      // cluster leaves its members' embeddings pointing at nothing. That is not
      // cosmetic: cluster assignment reads the field off the nearest vector to
      // decide where a new fact goes, so a ghost cluster can win the match and
      // fail the insert. The corrector re-embeds these; this is the detector.
      const liveClusters = new Set(db.prepare('SELECT id FROM memory_clusters').all().map(c => c.id));
      const staleCluster = rows.filter(r => byId.has(r.member_id) && r.cluster_id && !liveClusters.has(r.cluster_id));
      counts.staleClusterVectors = staleCluster.length;
      if (staleCluster.length) {
        mismatches.push({
          kind: 'vector-points-at-deleted-cluster',
          count: staleCluster.length,
          message: `${staleCluster.length} embedding(s) name a cluster that no longer exists, which can send a new fact to a cluster that isn't there.`,
          examples: staleCluster.slice(0, 3).map(r => (byId.get(r.member_id)?.content || '').slice(0, 100))
        });
      }
    }
  } catch (err) {
    mismatches.push({ kind: 'vector-store-unreadable', count: 0, message: `Could not read the vector store to check it: ${err.message}` });
  }

  return { mismatches, counts };
}

module.exports = {
  supersede, retire, expire, restore, reword, repoint, dropVector, replaceVector, reconcile,
  findExactDuplicate, absorbDuplicate, absorbRepeat, recordCorroboration, corroborationCount,
  getMember
};
