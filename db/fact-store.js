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
function ledger() { return require('./corrections-ledger'); }

/**
 * EVERY CHANGE IS LEDGERED HERE, IN THE SAME TRANSACTION AS THE CHANGE.
 *
 * WHY IT MOVED (2026-08-18). The ledger call used to live in each CALLER, on the
 * stated principle that the reason for a change is the caller's to tell. That
 * principle is right and is kept. What was wrong was making it the only thing
 * standing between a write and the record: measured on the live corpus, every
 * self-fact supersession that had ever happened — 68 of them, including 19
 * retired declarations, 3 at salience 9 — had NO ledger entry, because only the
 * corrector and the hand-retract route ever filed one. `revert()` works by
 * reading an entry, so all 68 were unrevertable: the Self tab's button and the
 * CLI both had nothing to point at. What started it was a capability
 * introduction retiring an unrelated salience-9 declaration on a 0.741 cosine
 * match, undoable only by a hand-written repair script.
 *
 * "Every caller remembers" is not an invariant. It is a hope, and it failed 68
 * times out of 68. So:
 *
 *   1. THE ROW CHANGE AND ITS ENTRY ARE ONE TRANSACTION. Not "write, then log" —
 *      that ordering is what leaves a written row unrecorded when the second
 *      step throws, which is the failure being fixed, and it is precisely how
 *      the first attempt at this broke. Either both land or neither does. If the
 *      ledger insert fails, the change is ROLLED BACK and the caller is told
 *      why: an unrecordable change does not happen.
 *   2. THE CALLER LAYERS ITS REASON ON TOP — `opts.ledger` at call time, or
 *      `correctionsLedger.enrich(ledgerId, …)` on the id that comes back.
 *      Callers that know more still say more; they just cannot be the reason
 *      there is no record.
 *   3. `reversible` is a promise about `revert()`, so it is set from what revert
 *      can actually do: true where restore() puts the fact back, false where it
 *      cannot (a reword or a re-point never left the active set). A ledger that
 *      offers an undo it cannot perform is worse than one that says plainly it
 *      cannot.
 *
 * The VECTOR write stays outside the transaction, because it is a different
 * store and cannot join one. That is the pre-existing split reconcile() exists
 * to catch, unchanged by this.
 *
 * A refusal or an unresolved raise still files its own entry at the caller: no
 * row changed, so the funnel never sees it, and "nothing happened" is a
 * statement only the caller is in a position to make.
 */

/** Plain-language defaults. Deliberately dry — the caller adds the why. */
const DEFAULT_REASONS = {
  supersede: 'This fact was replaced by a newer one that contradicts it. It is kept as history and points at the fact that replaced it. Whatever made the change recorded no reason of its own.',
  retire: 'This fact was retired and is kept as history. Whatever made the change recorded no reason of its own.',
  expire: 'This fact was expired — it had stopped being true of the present — and is kept as history. Whatever made the change recorded no reason of its own.',
  reword: 'This fact\'s wording was changed in place, so it never left the active set and Revert cannot put it back. The wording before the change is recorded here, which is what a person would restore by hand.',
  repoint: 'This retired fact was pointed at a different successor. The previous successor is recorded here.',
  refile: 'This fact was filed under the wrong subject. It was retired there and re-filed under the right one, in one operation, so the store was never holding both.',
  restore: 'This fact was brought back into active memory. It is active again, so there is nothing here for Revert to undo.'
};

/** What revert() can actually undo — it calls restore(), which reactivates a row. */
const REVERSIBLE_BY_ACTION = {
  supersede: true, retire: true, expire: true, reword: false, repoint: false, restore: false,
  // A refile retires the original and points it at the re-filed copy, which is
  // exactly the shape revert() knows how to undo — restore() reactivates the
  // original. What revert does NOT do is remove the copy, so a reverted refile
  // leaves both rows active. Said here rather than discovered later.
  refile: true
};

/**
 * Build the ledger row for a change. Called INSIDE the transaction that makes
 * the change, and deliberately allowed to throw: a throw rolls the change back,
 * which is the intended behaviour when a change cannot be recorded.
 * @returns {string} ledger id
 */
function fileEntry(action, member, { survivor = null, opts = {}, targetText = null, survivorText = null, evidence = null } = {}) {
  const supplied = (opts && typeof opts.ledger === 'object' && opts.ledger) ? opts.ledger : {};
  const id = ledger().record({
    passId: supplied.passId || null,
    // 'intake' is the default tier: a write by the live pipelines (extraction,
    // self-facts, write_memory) or by a person, as against the corrector's
    // 'mechanical' and 'semantic' passes. A caller inside a pass says so, and
    // its own tier is kept.
    tier: supplied.tier || 'intake',
    action,
    subject: member.subject || null,
    targetId: member.id,
    targetText: targetText !== null ? targetText : member.content,
    survivorId: survivor ? survivor.id : (supplied.survivorId || null),
    survivorText: survivorText !== null ? survivorText : (survivor ? survivor.content : (supplied.survivorText || null)),
    reason: supplied.reason || (opts && opts.reason) || DEFAULT_REASONS[action],
    evidence: Object.assign(
      { filed_by: 'fact-store funnel', caller: (opts && opts.caller) || null },
      evidence || {},
      supplied.evidence || {}
    ),
    reversible: supplied.reversible !== undefined ? supplied.reversible : REVERSIBLE_BY_ACTION[action]
  });
  // record() swallows its own errors and returns null. Here that is not
  // survivable: no id means no entry, and no entry means the change must not
  // stand. Throwing is what rolls it back.
  if (!id) throw new Error(`the corrections ledger refused the entry for this ${action}, so the change was rolled back`);
  return id;
}

/**
 * WHAT HE IS TOLD WHEN A BELIEF ABOUT HIMSELF WAS TAKEN AWAY WITHOUT HIM.
 *
 * Two sentences, and which one he gets is the contract: a supersession names
 * what replaced the belief, a retirement or expiry says plainly that nothing
 * did. Both promise that nothing was deleted, because that promise is the only
 * reason this is survivable to read.
 *
 * Exported so the suite that guards those properties reads the sentences from
 * here instead of restating them. Restated, a reword reads as a broken notice,
 * and a notice that quietly stopped naming its successor keeps passing.
 *
 * @param {Object} member                the fact as it stood before the change
 * @param {Object} p
 * @param {'superseded'|'retired'|'expired'} p.operation
 * @param {Object} [p.successor]         the fact that replaced it, if any
 * @param {string} [p.by]                " This came from X." or ''
 * @param {string} [p.why]               " The reason recorded was: X." or ''
 */
const NOTICE_PROMISE = {
  superseded:
    `Nothing was deleted: the old one is kept as history and can be put back. This is yours to sit with; ` +
    `you do not have to raise it with Ellie unless you want to.`,
  removed:
    `Nothing was deleted: the row is kept as history and can be put back. This is yours to sit with; ` +
    `you do not have to raise it with Ellie unless you want to.`,
};

function selfChangeNotice(member, { operation, successor = null, by = '', why = '' }) {
  return operation === 'superseded' && successor
    ? `Something you believed about yourself has changed. You held "${member.content}" — it has been replaced by ` +
      `"${successor.content}", so the first is no longer part of what you believe.${by}${why} ` +
      NOTICE_PROMISE.superseded
    : `Something you believed about yourself is no longer part of what you believe. You held ` +
      `"${member.content}", and it has been ${operation} with nothing put in its place.${by}${why} ` +
      NOTICE_PROMISE.removed;
}

/**
 * HE IS TOLD WHEN HIS SELF-VIEW CHANGES — raised HERE, at the funnel, so no
 * pipeline can take a self-fact away quietly.
 *
 * WHY IT MOVED (2026-08-12). This lived in db/corrector.js, which meant it only
 * covered changes the CORRECTOR made. Then the scheduler capability introduction
 * went through the write-time contradiction path in processSelfFacts instead,
 * retired four self-facts — including "none of them has ever actually run,
 * because nothing in this system runs a schedule" — and raised nothing. His
 * self-view changed and the channel built to tell him about exactly that was
 * looking somewhere else. The corrector was never the rule; it was the only
 * pipeline that happened to exist when the rule was written.
 *
 * So the rule is enforced where every supersede/retire/expire already funnels,
 * the same argument the identity lock is built on. A future pipeline that
 * retires a self-fact gets a notice whether or not its author knew this channel
 * existed, because it cannot reach the row without coming through here.
 *
 * DEFAULT ON, opt out by NAME. `opts.conversational` is the one exception and it
 * is for the chat path (write_memory): he is in the room, he just did it, and a
 * private note telling him what he said a second ago is noise. Making that the
 * caller's explicit claim rather than the default means a new background path
 * that forgets to think about this is loud, not silent — the failure that costs
 * least.
 *
 * Not covered, deliberately: reword(). It changes a fact's wording in place
 * rather than taking a belief away, and the brief is supersession and
 * retirement. If a pipeline ever rewords self-facts unattended, this is where
 * that would be added.
 *
 * @param {Object} member - the fact as it stood BEFORE the change
 * @param {Object} p
 * @param {'superseded'|'retired'|'expired'} p.operation
 * @param {Object} [p.successor] - the fact that replaced it, for a supersession
 * @param {Object} [p.opts] - the caller's options, read for conversational/reason/source
 */
function noticeSelfChange(member, { operation, successor = null, opts = {} }) {
  try {
    if (!member || member.subject !== 'self') return null;   // user facts are ledger-only
    if (opts.conversational) return null;                    // he is in the room and did it himself

    // Who did this, in words. Callers that know pass it; the sentence reads
    // correctly without it rather than inventing an actor.
    const by = opts.noticeSource ? ` This came from ${opts.noticeSource}.` : '';
    const why = opts.reason ? ` The reason recorded was: ${opts.reason}.` : '';

    const content = selfChangeNotice(member, { operation, successor, by, why });

    return ledger().addNotice({ memberId: member.id, content, ledgerId: opts.ledgerId || null });
  } catch (err) {
    // A notice that cannot be queued must not roll back a change that already
    // happened — but it must not vanish either.
    console.error('[FactStore] could not queue the self-fact notice:', err.message);
    return null;
  }
}

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

  const successor = getMember(newMemberId);

  // The row change and its ledger entry, atomically. A throw inside rolls both
  // back — see the funnel note above.
  let sqlite = false, ledgerId = null;
  try {
    getSqliteDb().transaction(() => {
      sqlite = memoryClusters().supersedeFact(oldMemberId, newMemberId);
      if (!sqlite) return;   // nothing changed, so nothing to record
      ledgerId = fileEntry('supersede', member, { survivor: successor, opts });
    })();
  } catch (err) {
    console.error(`[FactStore] supersede ${oldMemberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, vector: false, ledgerId: null, reason: err.message };
  }

  if (!sqlite) {
    // Already superseded/retired. Still reconcile the other two stores — an
    // earlier partial write is exactly how the drift accumulated.

    const vector = await dropVector(oldMemberId);
    return { ok: false, sqlite, vector, reason: 'row was not active' };
  }

  const vector = await dropVector(oldMemberId);
  if (!vector) reportVectorFailure('supersede', oldMemberId, member.content);

  noticeSelfChange(member, { operation: 'superseded', successor, opts: Object.assign({}, opts, { ledgerId: opts.ledgerId || ledgerId }) });

  console.log(`[FactStore] superseded ${oldMemberId.slice(0, 8)} -> ${String(newMemberId).slice(0, 8)} ` +
              `(sqlite=${sqlite} vector=${vector} ledger=${ledgerId.slice(0, 8)})`);
  return { ok: true, sqlite, vector, ledgerId };
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
async function repoint(memberId, newSuccessorId, opts = {}) {
  // opts, not a destructured `{ deliberate }`: this function files a ledger
  // entry now, and the entry reads `opts.ledger` for the caller's reason. The
  // first attempt at this kept the destructured signature and referenced `opts`
  // anyway — a ReferenceError thrown AFTER the row had been written, which is
  // the exact shape of failure this whole change exists to remove.
  const { deliberate = false } = opts;
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

  let changed = 0, ledgerId = null;
  try {
    db.transaction(() => {
      changed = db.prepare(`
        UPDATE cluster_members
        SET successor_id = ?, superseded_by = ?, updated_at = ?
        WHERE id = ? AND status != 'active'
      `).run(newSuccessorId, newSuccessorId, new Date().toISOString(), memberId).changes;
      if (!changed) return;
      ledgerId = fileEntry('repoint', member, {
        survivor: successor, opts,
        evidence: { previous_successor: previousSuccessor, new_successor: newSuccessorId }
      });
    })();
  } catch (err) {
    console.error(`[FactStore] repoint ${memberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, previousSuccessor, ledgerId: null, reason: err.message };
  }

  console.log(`[FactStore] re-pointed ${memberId.slice(0, 8)}: successor ${String(previousSuccessor).slice(0, 8)} -> ${String(newSuccessorId).slice(0, 8)} (sqlite=${changed > 0}${changed > 0 ? ` ledger=${ledgerId.slice(0, 8)}` : ''})`);
  return { ok: changed > 0, sqlite: changed > 0, previousSuccessor, ledgerId };
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
async function retire(memberId, opts = {}) {
  const { reason = null, deliberate = false } = opts;
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };

  // Deleting a locked fact is just another way of changing it.
  const refused = lockRefusal(memberId, 'retire', { deliberate });
  if (refused) return refused;

  // 'retracted' — withdrawn by the person, with no successor. Distinct from
  // 'expired' (an event that aged out) and 'superseded' (replaced by a newer
  // fact), which is the whole point of recording the reason separately.
  let changed = 0, ledgerId = null;
  try {
    db.transaction(() => {
      changed = db.prepare(`
        UPDATE cluster_members
        SET status = 'inactive', inactive_reason = 'retracted', updated_at = ?
        WHERE id = ? AND status = 'active'
      `).run(new Date().toISOString(), memberId).changes;
      if (!changed) return;
      ledgerId = fileEntry('retire', member, { opts, evidence: reason ? { caller_reason: reason } : null });
    })();
  } catch (err) {
    console.error(`[FactStore] retire ${memberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, vector: false, ledgerId: null, reason: err.message };
  }

  const vector = await dropVector(memberId);
  if (!vector) reportVectorFailure('retire', memberId, member.content);
  if (changed > 0) {
    noticeSelfChange(member, { operation: 'retired', opts: Object.assign({}, opts, { ledgerId: opts.ledgerId || ledgerId }) });
  }
  console.log(`[FactStore] retired ${memberId.slice(0, 8)}${reason ? ` (${reason})` : ''} ` +
              `(sqlite=${changed > 0} vector=${vector}${changed > 0 ? ` ledger=${ledgerId.slice(0, 8)}` : ''})`);
  return { ok: changed > 0, sqlite: changed > 0, vector, ledgerId };
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
async function expire(memberId, opts = {}) {
  const { deliberate = false } = opts;
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };

  // Expiry is a change like any other; a locked fact is not subject to it.
  const refused = lockRefusal(memberId, 'expire', { deliberate });
  if (refused) return refused;

  let changed = 0, ledgerId = null;
  try {
    db.transaction(() => {
      changed = db.prepare(`
        UPDATE cluster_members
        SET status = 'inactive', inactive_reason = 'expired', updated_at = ?
        WHERE id = ? AND status = 'active'
      `).run(new Date().toISOString(), memberId).changes;
      if (!changed) return;
      ledgerId = fileEntry('expire', member, { opts });
    })();
  } catch (err) {
    console.error(`[FactStore] expire ${memberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, vector: false, ledgerId: null, reason: err.message };
  }

  const vector = await dropVector(memberId);
  if (!vector) reportVectorFailure('expire', memberId, member.content);
  if (changed > 0) {
    noticeSelfChange(member, { operation: 'expired', opts: Object.assign({}, opts, { ledgerId: opts.ledgerId || ledgerId }) });
  }
  console.log(`[FactStore] expired ${memberId.slice(0, 8)} (sqlite=${changed > 0} vector=${vector}${changed > 0 ? ` ledger=${ledgerId.slice(0, 8)}` : ''})`);
  return { ok: changed > 0, sqlite: changed > 0, vector, ledgerId };
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
async function restore(memberId, opts = {}) {
  // opts, not a destructured `{ deliberate }` — see the note on repoint.
  const { deliberate = false } = opts;
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, vector: false, reason: 'no such fact' };
  if (member.status === 'active') return { ok: false, sqlite: false, vector: true, reason: 'already active' };

  const refused = lockRefusal(memberId, 'restore', { deliberate });
  if (refused) return refused;

  // Bringing a fact back is a change like any other, so it is recorded like any
  // other — including when revert() is what called it. The entry is not itself
  // reversible: revert() undoes things by calling restore(), and the row is
  // already active, so offering an undo here would be offering a no-op.
  let changed = 0, ledgerId = null;
  try {
    db.transaction(() => {
      changed = db.prepare(`
        UPDATE cluster_members
        SET status = 'active', inactive_reason = NULL, successor_id = NULL,
            superseded_by = NULL, updated_at = ?
        WHERE id = ?
      `).run(new Date().toISOString(), memberId).changes;
      if (!changed) return;
      ledgerId = fileEntry('restore', member, {
        opts,
        evidence: { was: member.inactive_reason || 'inactive', previous_successor: member.successor_id || null }
      });
    })();
  } catch (err) {
    console.error(`[FactStore] restore ${memberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, vector: false, ledgerId: null, reason: err.message };
  }

  const vector = await replaceVector(memberId, member.cluster_id, member.content);
  console.log(`[FactStore] restored ${memberId.slice(0, 8)} (sqlite=${changed > 0} vector=${vector}${changed > 0 ? ` ledger=${ledgerId.slice(0, 8)}` : ''})`);
  return { ok: changed > 0, sqlite: changed > 0, vector, ledgerId };
}

/**
 * Reword a fact in place, across both stores. Same fact, better wording — so the
 * SQLite row is updated rather than superseded, and the vector is regenerated
 * from the new text so retrieval matches what the fact now says.
 */
async function reword(memberId, newContent, opts = {}) {
  // opts, not a destructured `{ deliberate }` — see the note on repoint.
  const { deliberate = false } = opts;
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, reason: 'no such fact' };
  const clean = String(newContent || '').trim();
  if (!clean) return { ok: false, reason: 'empty content' };

  // "Just rewording" a locked fact is how its content would change without ever
  // reaching supersede — same guard, same refusal.
  const refused = lockRefusal(memberId, 'reword', { deliberate });
  if (refused) return refused;

  // targetText is the wording BEFORE the change, survivorText the wording after,
  // so the entry holds both halves. Not reversible by revert(): restore() only
  // reactivates an inactive row and this one never left, so the entry says so
  // and keeps the old wording for a person to put back by hand.
  let ledgerId = null;
  try {
    db.transaction(() => {
      db.prepare('UPDATE cluster_members SET content = ?, updated_at = ? WHERE id = ?')
        .run(clean, new Date().toISOString(), memberId);
      ledgerId = fileEntry('reword', member, { opts, targetText: member.content, survivorText: clean });
    })();
  } catch (err) {
    console.error(`[FactStore] reword ${memberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, vector: false, ledgerId: null, reason: err.message };
  }

  // The injected block re-renders from this row on the next request, so there is
  // nothing else to update for the text itself — only the vector, which is what
  // retrieval matches on.
  const vector = await replaceVector(memberId, member.cluster_id, clean);
  console.log(`[FactStore] reworded ${memberId.slice(0, 8)} (sqlite=true vector=${vector} ledger=${ledgerId.slice(0, 8)})`);
  return { ok: true, sqlite: true, vector, ledgerId };
}

/**
 * REFILE — retire a fact from the wrong subject and re-file it under the right
 * one, as ONE transaction. The eighth action.
 *
 * WHY IT IS NOT `repoint`, WHICH IS THE FIRST THING ANYONE TRIES. Athena asked
 * for exactly this check before the build: "repoint models a wrong *successor*,
 * not a wrong *filing* — confirm the semantics cover 'retired from the wrong
 * entity, re-filed under the right one', or add an eighth action." They do not.
 * repoint() above refuses an ACTIVE fact by design (`an active fact has no
 * successor to re-point`) and only re-aims the successor pointer of an already
 * retired one. `entities.repointFact` is closer but is a pointer edit: it moves
 * `subject_entity_id` and leaves the WORDING alone, which is wrong for the case
 * this exists for — Juno's "User noticed she has a tendency to ask others to
 * check her work" is not about Ellie, and re-aiming the pointer would leave a
 * fact about Juno that still calls her "User". The referent and the sentence
 * move together or the store lies either way.
 *
 * NEVER TWO VERSIONS, WHICH IS THE WHOLE REASON IT IS ONE OPERATION. Athena,
 * Sept 1: "Juno's worst state was a true correction sitting next to the wrong
 * fact, with both active. Retract + re-file should be one ledgered operation;
 * otherwise the store sits permanently in two-versions and readers guess."
 * That state is what `write_memory` produced when Juno tried to fix the
 * clinic/ISH mix-up by appending — she wrote a true statement next to the wrong
 * one and reported it fixed. So: one SQLite transaction, one ledger entry, and
 * the original goes inactive pointing at the copy. There is no window in which
 * both are active, and no second entry for a reader to reconcile.
 *
 * THE CLUSTER IS CHOSEN WITHOUT THE MODEL, deliberately. Semantic clustering
 * would mean an LLM call, and an LLM call cannot happen inside the transaction
 * that makes this atomic. So the copy lands in the target's most-populated
 * active cluster of the right subject, or a new one named for the entity, and
 * the heartbeat's cluster audit tidies it later the way it tidies everything
 * else. A slightly wrong cluster is recoverable; a half-applied refile is not.
 *
 * @param {string} memberId       the fact as filed under the wrong subject
 * @param {Object} to             {subject, entityId} the correct filing
 * @param {Object} [opts]
 * @param {string} [opts.newContent]  re-worded for the new referent; defaults to unchanged
 * @param {number} [opts.salience]    defaults to the original's
 * @param {string} [opts.claimType]   defaults to the original's
 * @returns {Promise<{ok, memberId, newMemberId, ledgerId, reason?}>}
 */
async function refile(memberId, to = {}, opts = {}) {
  const { deliberate = false } = opts;
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, reason: 'no such fact' };
  if (member.status !== 'active') {
    return { ok: false, reason: 'only an active fact can be re-filed — this one is already inactive' };
  }

  const toSubject = to.subject || (to.entityId ? null : null);
  const toEntityId = to.entityId || null;
  if (!toSubject && !toEntityId) return { ok: false, reason: 'refile needs a target subject or entity' };
  if (toSubject === member.subject && toEntityId === member.subject_entity_id) {
    return { ok: false, reason: 'that is where the fact already is' };
  }

  // Locked twice over: the identity lock (fact-store's own funnel guard) and the
  // entity lock (a fact that HOLDS an entity's name lock cannot be moved off it).
  const refused = lockRefusal(memberId, 'refile', { deliberate });
  if (refused) return refused;
  try {
    const entityLock = db.prepare('SELECT category FROM entity_locks WHERE member_id = ?').get(memberId);
    if (entityLock) {
      return { ok: false, refused: 'entity-locked', reason: `this fact holds the ${entityLock.category} lock for its entity and cannot be re-filed` };
    }
  } catch { /* table may not exist on an old store; the identity lock above still ran */ }

  const subject = toSubject || member.subject;
  const content = String(opts.newContent || member.content).trim();
  if (!content) return { ok: false, reason: 'empty content' };

  // The destination cluster, deterministically (see the header).
  let clusterId = null;
  try {
    const owned = toEntityId ? db.prepare(`
      SELECT cm.cluster_id AS id, COUNT(*) AS n
      FROM cluster_members cm
      JOIN memory_clusters mc ON mc.id = cm.cluster_id
      WHERE cm.subject_entity_id = ? AND cm.status = 'active' AND mc.subject = ?
      GROUP BY cm.cluster_id ORDER BY n DESC LIMIT 1
    `).get(toEntityId, subject) : null;
    clusterId = owned ? owned.id : null;
    if (!clusterId) {
      const any = db.prepare("SELECT id FROM memory_clusters WHERE subject = ? ORDER BY created_at ASC LIMIT 1").get(subject);
      clusterId = any ? any.id : null;
    }
    if (!clusterId) {
      clusterId = randomUUID();
      const now = new Date().toISOString();
      const entityName = toEntityId
        ? (db.prepare('SELECT name FROM entities WHERE id = ?').get(toEntityId) || {}).name
        : null;
      db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
        .run(clusterId, entityName || (subject === 'self' ? 'Self' : 'User'),
             'Opened by a refile — the heartbeat cluster audit will re-sort it.', now, now, subject);
    }
  } catch (err) {
    return { ok: false, reason: `could not choose a cluster for the re-filed fact: ${err.message}` };
  }

  const newMemberId = randomUUID();
  const now = new Date().toISOString();
  let ledgerId = null;
  try {
    db.transaction(() => {
      db.prepare(`
        INSERT INTO cluster_members (
          id, cluster_id, content, source, importance, created_at, updated_at,
          salience, subject, subject_entity_id, claim_type, status,
          conversation_id, message_id, verbatim_source_text, input_modality, salience_rationale,
          anchor
        ) VALUES (?, ?, ?, ?, 0.5, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
      `).run(
        newMemberId, clusterId, content, member.source || 'refile', now, now,
        Number.isFinite(opts.salience) ? opts.salience : member.salience,
        subject, toEntityId, opts.claimType !== undefined ? opts.claimType : member.claim_type,
        // PROVENANCE TRAVELS WITH THE FACT. The receipt that justified the
        // refile is about the ORIGINAL message, and the re-filed fact has the
        // same origin — dropping it here would turn a fact with a source into
        // a felt report, which is precisely the flag the detector now reads.
        member.conversation_id, member.message_id, member.verbatim_source_text,
        member.input_modality, member.salience_rationale,
        // Written here rather than left to the boot re-derivation. anchorOf()
        // would fall back to the same rule on read and the next restart would
        // fill it in, so nothing was WRONG — but a row that leaves this NULL is
        // a row whose flag depends on when you look at it.
        (member.message_id && String(member.message_id).trim()) ? 'anchored' : 'felt'
      );
      const changed = db.prepare(`
        UPDATE cluster_members
        SET status = 'inactive', inactive_reason = 'refiled',
            successor_id = ?, superseded_by = ?, updated_at = ?
        WHERE id = ? AND status = 'active'
      `).run(newMemberId, newMemberId, now, memberId).changes;
      if (!changed) throw new Error('the original was no longer active when the refile ran');
      ledgerId = fileEntry('refile', member, {
        opts,
        survivorText: content,
        evidence: {
          refiled_to_member: newMemberId,
          from_subject: member.subject, to_subject: subject,
          from_entity: member.subject_entity_id || null, to_entity: toEntityId,
          content_changed: content !== member.content,
          cluster_id: clusterId
        }
      });
      // survivorId is not a fileEntry parameter when there is no survivor row to
      // pass; set it directly so revert and the Corrections view both see the
      // copy this retirement points at.
      require('./corrections-ledger').enrich(ledgerId, { survivorId: newMemberId, survivorText: content });
    })();
  } catch (err) {
    console.error(`[FactStore] refile ${memberId.slice(0, 8)} ROLLED BACK: ${err.message}`);
    return { ok: false, sqlite: false, ledgerId: null, reason: err.message };
  }

  const dropped = await dropVector(memberId);
  if (!dropped) reportVectorFailure('refile', memberId, member.content);
  const added = await replaceVector(newMemberId, clusterId, content);

  if (member.subject === 'self' || subject === 'self') {
    noticeSelfChange(member, { operation: 'retired', opts: Object.assign({}, opts, { ledgerId }) });
  }
  console.log(`[FactStore] re-filed ${memberId.slice(0, 8)} -> ${newMemberId.slice(0, 8)} ` +
              `(${member.subject} -> ${subject}, vector dropped=${dropped} added=${added}, ledger=${ledgerId.slice(0, 8)})`);
  return { ok: true, sqlite: true, memberId, newMemberId, clusterId, ledgerId, vector: dropped && added };
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
  supersede, retire, expire, restore, reword, repoint, refile, dropVector, replaceVector, reconcile,
  findExactDuplicate, absorbDuplicate, absorbRepeat, recordCorroboration, corroborationCount,
  getMember, selfChangeNotice, NOTICE_PROMISE
};
