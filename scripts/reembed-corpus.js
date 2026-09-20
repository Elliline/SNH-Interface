#!/usr/bin/env node
/**
 * Rebuild the cluster vector index from SQLite.
 *
 * Run after the cutover, because the two stores no longer agree: SQLite now holds
 * a corpus of 485 user-fact rows that were written into a DIFFERENT database, and
 * LanceDB still holds embeddings for the 389 rows that were there before. Every
 * one of those old vectors points at a member_id that no longer exists, and
 * retrieval matches on similarity without ever consulting SQLite — so until this
 * runs, a search can surface a fact that is not in the corpus at all.
 *
 * ONLY ACTIVE FACTS GET A VECTOR. That is the rule the whole memory system rests
 * on: fact-store drops the embedding when a fact goes inactive, so retrieval
 * stops surfacing it, and reconcile() reports `retiredWithVector > 0` when
 * something has put one back. A rebuild that embedded everything would recreate
 * by hand the exact drift the funnel exists to prevent — 162 retired beliefs,
 * including superseded names and contradicted colours, live in search again.
 *
 * The table is DROPPED and recreated rather than emptied, so the index is built
 * fresh over the new corpus instead of inheriting one shaped by the old one.
 *
 * message_embeddings is untouched. It indexes conversation text, which the
 * cutover did not change.
 *
 * Usage:
 *   node scripts/reembed-corpus.js [--dry-run]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');

const DRY_RUN = process.argv.includes('--dry-run');

(async () => {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  const d = db.getSqliteDb();
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const { randomUUID } = require('crypto');

  const rows = d.prepare(`
    SELECT id, cluster_id, content, subject
    FROM cluster_members
    WHERE status = 'active'
    ORDER BY subject, datetime(created_at)
  `).all();

  const bySubject = rows.reduce((a, r) => { const k = r.subject || 'user'; a[k] = (a[k] || 0) + 1; return a; }, {});
  console.log(`[Reembed] ${rows.length} active fact(s) to embed: ${JSON.stringify(bySubject)}`);
  console.log(`[Reembed] data dir: ${db.getDataDir()}`);

  if (DRY_RUN) { console.log('[Reembed] --dry-run: nothing written'); process.exit(0); }

  const table = await db.resetClusterEmbeddingsTable();
  if (!table) { console.error('ABORT: could not recreate the cluster_embeddings table'); process.exit(1); }

  let ok = 0, failed = 0;
  const failures = [];
  // Batched: one add() per chunk rather than per fact. 700 individual commits
  // against LanceDB is minutes of optimistic-concurrency overhead for no gain,
  // and nothing else is writing while the server is down.
  const BATCH = 50;
  let batch = [];

  const flush = async () => {
    if (!batch.length) return;
    await table.add(batch);
    ok += batch.length;
    batch = [];
  };

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    try {
      const vector = await memoryClusters.generateEmbedding(r.content);
      if (!vector) { failed++; failures.push({ id: r.id, content: r.content, why: 'no embedding returned' }); continue; }
      batch.push({
        id: randomUUID(), member_id: r.id, cluster_id: r.cluster_id,
        content: r.content, vector: Array.from(vector)
      });
      if (batch.length >= BATCH) await flush();
    } catch (err) {
      failed++; failures.push({ id: r.id, content: r.content, why: err.message });
    }
    if ((i + 1) % 100 === 0) console.log(`[Reembed] ${i + 1}/${rows.length}`);
  }
  await flush();

  console.log(`\n[Reembed] embedded ${ok}, failed ${failed}`);
  for (const f of failures.slice(0, 10)) console.log(`   FAILED ${f.id.slice(0, 8)} (${f.why}): "${String(f.content).slice(0, 70)}"`);

  // The detector for exactly the drift this script exists to remove. Reported
  // here rather than assumed, because "I rebuilt the index" is a claim and this
  // is the check.
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const { counts, mismatches } = await factStore.reconcile();
  console.log('\n[Reembed] reconcile:', JSON.stringify(counts));
  for (const m of mismatches) console.log(`   ${m.kind}: ${m.message}`);

  const clean = counts.retiredWithVector === 0 && counts.activeNoVector === 0
    && counts.orphanVectors === 0 && counts.staleClusterVectors === 0;
  console.log(clean ? '\nAll four stores agree.' : '\nSTORES DISAGREE — see above.');
  process.exit(clean && failed === 0 ? 0 : 1);
})().catch(err => { console.error('reembed failed:', err); process.exit(1); });
