/**
 * Fact store — the ONE place a fact is superseded, retired, or reworded.
 *
 * A fact lives in three stores and every one of them has to agree:
 *   1. SQLite (cluster_members) — the record of truth, including history.
 *   2. MEMORY.md               — what is actually INJECTED as Long-Term Memory.
 *   3. LanceDB (cluster_embeddings) — what SEMANTIC RETRIEVAL can surface.
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
const { getSqliteDb, getClusterEmbeddingsTable } = require('./database');

const MEMORY_DIR = path.join(__dirname, '../data/memory');
const MEMORY_FILE = path.join(MEMORY_DIR, 'MEMORY.md');

/** Lazy requires — fact-extractor pulls in memory-clusters, which pulls in this. */
function factExtractor() { return require('./fact-extractor'); }
function memoryClusters() { return require('./memory-clusters'); }

/**
 * LanceDB filter for one member. Values are UUIDs from our own DB, but quote
 * defensively anyway — this string is interpolated into a filter expression.
 */
function memberFilter(memberId) {
  return `member_id = "${String(memberId).replace(/"/g, '')}"`;
}

/**
 * Retry a LanceDB write through commit conflicts.
 *
 * LanceDB uses optimistic concurrency: a delete that reads version N fails if
 * another writer commits N+1 first. The heartbeat, chat fact-extraction and this
 * module all write to the same table, so conflicts happen in normal operation —
 * observed during testing while the server was live. Left unretried the delete
 * SILENTLY fails and the superseded fact keeps its vector, which is precisely
 * the bug this module exists to fix, so a lost race must not look like success.
 */
async function withRetry(label, fn, attempts = 4) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); } catch (err) {
      lastErr = err;
      if (!/conflict/i.test(err.message || '')) break;   // not a race — don't spin
      const waitMs = 120 * Math.pow(2, i);
      console.warn(`[FactStore] ${label}: commit conflict, retry ${i + 1}/${attempts - 1} in ${waitMs}ms`);
      await new Promise(r => setTimeout(r, waitMs));
    }
  }
  console.error(`[FactStore] ${label} failed:`, lastErr && lastErr.message);
  return null;
}

/**
 * Drop a fact's embedding so semantic retrieval can no longer surface it.
 * The SQLite row is untouched — history is preserved there, not here.
 * @returns {Promise<boolean>} true if the vector is confirmed gone
 */
async function dropVector(memberId) {
  const table = await getClusterEmbeddingsTable();
  if (!table) return false;
  const res = await withRetry(`vector delete ${memberId.slice(0, 8)}`,
    () => table.delete(memberFilter(memberId)));
  if (res === null) return false;
  // Verify rather than assume: a silently-failed delete is the whole bug.
  try {
    const left = await table.filter(memberFilter(memberId)).limit(1).execute();
    return left.length === 0;
  } catch { return true; }
}

/** Re-embed a member's new content, replacing any existing vector. */
async function replaceVector(memberId, clusterId, content) {
  const table = await getClusterEmbeddingsTable();
  if (!table) return false;
  const vector = await memoryClusters().generateEmbedding(content);
  if (!vector) return false;
  const del = await withRetry(`vector delete ${memberId.slice(0, 8)}`,
    () => table.delete(memberFilter(memberId)));
  if (del === null) return false;   // don't add a duplicate on top of a stale row
  const add = await withRetry(`vector add ${memberId.slice(0, 8)}`,
    () => table.add([{ id: randomUUID(), member_id: memberId, cluster_id: clusterId, content, vector: Array.from(vector) }]));
  return add !== null;
}

/**
 * Remove a fact's line from MEMORY.md. Self-facts are never written there (they
 * inject through the identity block), so this is a no-op for them by design.
 */
function dropMemoryLine(member, memoryFile = MEMORY_FILE) {
  if (!member || member.subject === 'self') return false;
  try {
    return factExtractor().removeFactLineFromMemory(member.content, memoryFile);
  } catch (err) {
    console.error('[FactStore] MEMORY.md line removal failed:', err.message);
    return false;
  }
}

function getMember(memberId) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId) || null;
}

/**
 * Supersede a fact with a replacement, across ALL THREE stores.
 *
 * SQLite keeps the row (status='superseded', superseded_by set) so the belief
 * history survives. MEMORY.md loses the line and LanceDB loses the vector, so
 * the outdated fact stops reaching the model by either route.
 *
 * @param {string} oldMemberId
 * @param {string} newMemberId
 * @param {Object} [opts]
 * @param {string} [opts.memoryFile]
 * @returns {Promise<{ok: boolean, sqlite: boolean, memoryLine: boolean, vector: boolean, reason?: string}>}
 */
async function supersede(oldMemberId, newMemberId, opts = {}) {
  const member = getMember(oldMemberId);
  if (!member) return { ok: false, sqlite: false, memoryLine: false, vector: false, reason: 'no such fact' };

  const sqlite = memoryClusters().supersedeFact(oldMemberId, newMemberId);
  if (!sqlite) {
    // Already superseded/retired. Still reconcile the other two stores — an
    // earlier partial write is exactly how the drift accumulated.
    const memoryLine = dropMemoryLine(member, opts.memoryFile);
    const vector = await dropVector(oldMemberId);
    return { ok: false, sqlite, memoryLine, vector, reason: 'row was not active' };
  }
  const memoryLine = dropMemoryLine(member, opts.memoryFile);
  const vector = await dropVector(oldMemberId);

  // NEVER leave a hole. appendToMemory does semantic dedup, so a replacement
  // that closely resembles the fact it replaces (the common case — "red car" ->
  // "blue car") gets skipped at write time. Dropping the old line then leaves
  // NOTHING in the injected file, which is how facts go missing from MEMORY.md
  // while the DB still believes it holds them.
  //
  // Order matters: the old line is already gone by this point, so the
  // near-duplicate it would have collided with is no longer there and the
  // replacement writes cleanly.
  const replacement = await ensureReplacementLine(newMemberId, opts.memoryFile);

  console.log(`[FactStore] superseded ${oldMemberId.slice(0, 8)} -> ${String(newMemberId).slice(0, 8)} ` +
              `(sqlite=${sqlite} memoryLine=${memoryLine} vector=${vector} replacement=${replacement})`);
  return { ok: true, sqlite, memoryLine, vector, replacement };
}

/**
 * Guarantee the replacing fact actually has a line in MEMORY.md.
 * @returns {Promise<'present'|'written'|'skipped-self'|'missing'>}
 */
async function ensureReplacementLine(newMemberId, memoryFile = MEMORY_FILE) {
  try {
    const nm = getMember(newMemberId);
    if (!nm) return 'missing';
    if (nm.subject === 'self') return 'skipped-self';   // self-facts never go in the file
    const fs = require('fs');
    if (fs.existsSync(memoryFile) && fs.readFileSync(memoryFile, 'utf8').includes(nm.content)) {
      return 'present';
    }
    await factExtractor().appendToMemory([nm.content], memoryFile);
    const ok = fs.readFileSync(memoryFile, 'utf8').includes(nm.content);
    if (!ok) console.warn(`[FactStore] replacement line still absent after append: "${nm.content.slice(0, 80)}"`);
    return ok ? 'written' : 'missing';
  } catch (err) {
    console.error('[FactStore] ensureReplacementLine failed:', err.message);
    return 'missing';
  }
}

/**
 * Retire a fact with NO replacement — the user deleted it.
 *
 * Deliberately not a DELETE. `DELETE /api/memory/fact/:id` used to remove the
 * row outright, which is the one place the system broke its own
 * supersede-never-delete rule. The row is kept with status='retired' so the
 * history stays readable and the Map can still show it as a ghost.
 *
 * @returns {Promise<{ok: boolean, sqlite: boolean, memoryLine: boolean, vector: boolean, reason?: string}>}
 */
async function retire(memberId, { reason = null, memoryFile } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, sqlite: false, memoryLine: false, vector: false, reason: 'no such fact' };

  const info = db.prepare(
    "UPDATE cluster_members SET status = 'retired', updated_at = ? WHERE id = ? AND status = 'active'"
  ).run(new Date().toISOString(), memberId);

  const memoryLine = dropMemoryLine(member, memoryFile);
  const vector = await dropVector(memberId);
  console.log(`[FactStore] retired ${memberId.slice(0, 8)}${reason ? ` (${reason})` : ''} ` +
              `(sqlite=${info.changes > 0} memoryLine=${memoryLine} vector=${vector})`);
  return { ok: info.changes > 0, sqlite: info.changes > 0, memoryLine, vector };
}

/**
 * Reword a fact in place, across all three stores. Same fact, better wording —
 * so the SQLite row is updated rather than superseded, the MEMORY.md line is
 * replaced rather than dropped, and the vector is regenerated from the new text.
 */
async function reword(memberId, newContent, { memoryFile = MEMORY_FILE } = {}) {
  const db = getSqliteDb();
  const member = getMember(memberId);
  if (!db || !member) return { ok: false, reason: 'no such fact' };
  const clean = String(newContent || '').trim();
  if (!clean) return { ok: false, reason: 'empty content' };

  db.prepare('UPDATE cluster_members SET content = ?, updated_at = ? WHERE id = ?')
    .run(clean, new Date().toISOString(), memberId);

  // MEMORY.md: swap the line in place, preserving its "(learned ...)" annotation
  // so rewording does not silently reset when a fact was first known.
  let memoryLine = false;
  if (member.subject !== 'self') {
    try {
      const fs = require('fs');
      if (fs.existsSync(memoryFile)) {
        const lines = fs.readFileSync(memoryFile, 'utf8').split('\n');
        const idx = lines.findIndex(l => l.startsWith('- ') && l.slice(2).replace(/\s*\(learned [^)]*\)\s*$/, '').trim() === member.content.trim());
        if (idx !== -1) {
          const learned = (lines[idx].match(/\s*(\(learned [^)]*\))\s*$/) || [])[1];
          lines[idx] = `- ${clean}${learned ? ' ' + learned : ''}`;
          fs.writeFileSync(memoryFile, lines.join('\n'), 'utf8');
          memoryLine = true;
        }
      }
    } catch (err) {
      console.error('[FactStore] MEMORY.md reword failed:', err.message);
    }
  }

  const vector = await replaceVector(memberId, member.cluster_id, clean);
  console.log(`[FactStore] reworded ${memberId.slice(0, 8)} (sqlite=true memoryLine=${memoryLine} vector=${vector})`);
  return { ok: true, sqlite: true, memoryLine, vector };
}

// ============ Reconciliation (drift detection) ============

const stripLearned = (s) => String(s || '').replace(/\s*\(learned [^)]*\)\s*$/, '').trim();
const normFact = (s) => stripLearned(s).toLowerCase().replace(/\s+/g, ' ').replace(/[.,;:]+$/, '').trim();

/**
 * Compare the three stores and report where they disagree.
 *
 * Deliberately REPORT-ONLY — it never edits anything. This is the substrate the
 * entity's identity is built from, and the 2026-07-27 audit showed the two
 * directions are not symmetric: some absences are by design (appendToMemory
 * skips semantic near-duplicates) while others are real loss. Deciding which is
 * a judgement call, so the check surfaces counts and lets Ellie choose.
 *
 * @returns {{mismatches: Array, counts: Object}}
 */
async function reconcile({ memoryFile = MEMORY_FILE } = {}) {
  const fs = require('fs');
  const db = getSqliteDb();
  const mismatches = [];
  const counts = { fileLines: 0, activeUserFacts: 0, staleInFile: 0, retiredWithVector: 0, activeNoVector: 0, orphanVectors: 0 };
  if (!db) return { mismatches, counts };

  const members = db.prepare('SELECT id, content, status, subject FROM cluster_members').all();
  const byId = new Map(members.map(m => [m.id, m]));

  // --- MEMORY.md vs SQLite ---
  let fileKeys = new Set();
  try {
    const lines = fs.existsSync(memoryFile) ? fs.readFileSync(memoryFile, 'utf8').split('\n') : [];
    const factLines = lines.filter(l => l.startsWith('- '));
    counts.fileLines = factLines.length;
    fileKeys = new Set(factLines.map(l => normFact(l.slice(2))));
  } catch (e) { /* counted as zero */ }

  // A line is only stale if NO active fact also backs it. Superseded and active
  // rows routinely share normalized text — "User does not program." superseded,
  // "User does not program" active — and flagging on the superseded row alone
  // raises an alert about a line that is correctly present and cannot be
  // actioned. A check that cannot be resolved is a check that gets ignored.
  const activeKeys = new Set(members.filter(m => !m.status || m.status === 'active').map(m => normFact(m.content)));
  const inactive = members.filter(m => m.status && m.status !== 'active' && m.subject !== 'self');
  const staleInFile = inactive.filter(m => fileKeys.has(normFact(m.content)) && !activeKeys.has(normFact(m.content)));
  counts.staleInFile = staleInFile.length;
  counts.activeUserFacts = members.filter(m => m.status === 'active' && m.subject === 'user').length;

  if (staleInFile.length) {
    mismatches.push({
      kind: 'stale-in-memory-file',
      count: staleInFile.length,
      message: `${staleInFile.length} fact(s) I've retired or replaced are still written in MEMORY.md, so I'm still reading them as current.`,
      examples: staleInFile.slice(0, 3).map(m => m.content.slice(0, 100))
    });
  }

  // --- LanceDB vs SQLite ---
  try {
    const table = await getClusterEmbeddingsTable();
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
          // The worst of the three: retrieval matches on similarity and never
          // consults status or MEMORY.md, so these can surface in any answer.
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
    }
  } catch (err) {
    mismatches.push({ kind: 'vector-store-unreadable', count: 0, message: `Could not read the vector store to check it: ${err.message}` });
  }

  return { mismatches, counts };
}

module.exports = { supersede, retire, reword, dropVector, replaceVector, reconcile, MEMORY_FILE };
