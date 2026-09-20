#!/usr/bin/env node

/**
 * Migration: pull self-observations out of user territory.
 *
 * Before the subject guard in assignToCluster (which now keeps self-facts in
 * self clusters), some subject:'self' observations were routed INTO user-fact
 * clusters, and the cross-cluster linker wired self clusters to user clusters.
 * The result: SNH's own self-facts leaked into the Facts/Clusters tabs, which
 * render the members of user clusters.
 *
 * This one-off migration:
 *   1. Finds every member whose subject differs from its cluster's subject
 *      (in practice: subject:'self' members sitting in subject:'user' clusters)
 *      and moves each to the best-matching cluster of its OWN subject, keeping
 *      SQLite and LanceDB in sync.
 *   2. Deletes every cluster_link that crosses subjects (self <-> user), since
 *      those only exist because of the leak and pollute user territory.
 *
 * Idempotent: re-running finds nothing to move and no cross-subject links.
 *
 * Usage:
 *   node scripts/migrate-self-facts.js --dry-run   # report only
 *   node scripts/migrate-self-facts.js             # apply
 */

require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const { initDatabase, initVectorStore, getSqliteDb, getClusterEmbeddingsTable } = require('../db/database');

const DRY_RUN = process.argv.includes('--dry-run');

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function safeId(id) {
  if (!UUID_RE.test(id)) throw new Error(`Invalid UUID for LanceDB filter: ${id}`);
  return id;
}
const shortId = id => (id ? id.substring(0, 8) : '????????');

/**
 * Pick the best home cluster of `subject` for a member, using its own stored
 * embedding: search the cluster-embeddings table, keep only rows in same-subject
 * clusters, and return the cluster with the highest similarity. Falls back to the
 * largest same-subject cluster (excluding the member's current one) if the vector
 * search is unavailable.
 */
async function pickTargetCluster(db, clusterTable, member, subject) {
  const subjById = new Map(
    db.prepare("SELECT id, COALESCE(subject,'user') AS subject FROM memory_clusters").all()
      .map(r => [r.id, r.subject])
  );

  // Try similarity via the member's existing vector.
  try {
    const rows = await clusterTable
      .filter(`member_id = "${safeId(member.id)}"`)
      .limit(1)
      .execute();
    const vector = rows && rows[0] && rows[0].vector;
    if (vector) {
      const results = await clusterTable
        .search(Array.from(vector))
        .metricType('cosine')
        .limit(25)
        .execute();
      let bestId = null;
      let bestSim = -Infinity;
      for (const r of results) {
        if (r.member_id === member.id) continue;              // skip self
        if (subjById.get(r.cluster_id) !== subject) continue; // same subject only
        if (r.cluster_id === member.cluster_id) continue;     // not the wrong home
        const sim = 1 - (r._distance || 0);
        if (sim > bestSim) { bestSim = sim; bestId = r.cluster_id; }
      }
      if (bestId) return { clusterId: bestId, via: `similarity ${bestSim.toFixed(3)}` };
    }
  } catch (e) {
    console.warn(`  [warn] vector lookup failed for ${shortId(member.id)}: ${e.message}`);
  }

  // Fallback: largest same-subject cluster that isn't the current (wrong) one.
  const fallback = db.prepare(`
    SELECT mc.id, COUNT(cm.id) AS n
    FROM memory_clusters mc
    LEFT JOIN cluster_members cm ON cm.cluster_id = mc.id
    WHERE COALESCE(mc.subject,'user') = ? AND mc.id != ?
    GROUP BY mc.id
    ORDER BY n DESC
    LIMIT 1
  `).get(subject, member.cluster_id);
  if (fallback) return { clusterId: fallback.id, via: 'largest same-subject cluster (fallback)' };
  return null;
}

async function main() {
  initDatabase();
  await initVectorStore();
  const db = getSqliteDb();
  const clusterTable = getClusterEmbeddingsTable();

  console.log(`\n=== migrate-self-facts ${DRY_RUN ? '(DRY RUN)' : '(APPLY)'} ===\n`);

  // 1. Members whose subject != their cluster's subject.
  const mismatched = db.prepare(`
    SELECT cm.id, cm.content, COALESCE(cm.subject,'user') AS subject,
           cm.cluster_id, mc.name AS cluster_name, COALESCE(mc.subject,'user') AS cluster_subject
    FROM cluster_members cm
    JOIN memory_clusters mc ON mc.id = cm.cluster_id
    WHERE COALESCE(cm.subject,'user') != COALESCE(mc.subject,'user')
  `).all();

  console.log(`Found ${mismatched.length} member(s) in the wrong subject's cluster:\n`);
  let moved = 0;
  for (const m of mismatched) {
    const target = await pickTargetCluster(db, clusterTable, m, m.subject);
    if (!target) {
      console.log(`  ! ${shortId(m.id)} [${m.subject}] no same-subject target found — leaving in place`);
      continue;
    }
    const targetName = db.prepare('SELECT name FROM memory_clusters WHERE id = ?').get(target.clusterId)?.name;
    console.log(`  • "${m.content.slice(0, 70)}${m.content.length > 70 ? '…' : ''}"`);
    console.log(`      [${m.subject}] "${m.cluster_name}" (${m.cluster_subject}) → "${targetName}" (${m.subject})  [${target.via}]`);

    if (!DRY_RUN) {
      const now = new Date().toISOString();
      db.prepare('UPDATE cluster_members SET cluster_id = ?, updated_at = ? WHERE id = ?')
        .run(target.clusterId, now, m.id);
      db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?').run(now, target.clusterId);
      try {
        await clusterTable.update({
          where: `member_id = "${safeId(m.id)}"`,
          valuesSql: { cluster_id: `'${safeId(target.clusterId)}'` }
        });
      } catch (e) {
        console.error(`      [warn] LanceDB cluster_id update failed: ${e.message}`);
      }
      moved++;
    }
  }

  // 2. Cross-subject cluster_links (self <-> user) created by the leak.
  const subjById = new Map(
    db.prepare("SELECT id, COALESCE(subject,'user') AS subject FROM memory_clusters").all()
      .map(r => [r.id, r.subject])
  );
  const links = db.prepare('SELECT id, cluster_a, cluster_b FROM cluster_links').all();
  const crossLinks = links.filter(l => subjById.get(l.cluster_a) !== subjById.get(l.cluster_b));

  console.log(`\nFound ${crossLinks.length} cross-subject cluster link(s) to remove:\n`);
  for (const l of crossLinks) {
    console.log(`  • ${shortId(l.cluster_a)} (${subjById.get(l.cluster_a)}) ✕ ${shortId(l.cluster_b)} (${subjById.get(l.cluster_b)})`);
    if (!DRY_RUN) {
      db.prepare('DELETE FROM cluster_links WHERE id = ?').run(l.id);
    }
  }

  console.log('\n=== summary ===');
  if (DRY_RUN) {
    console.log(`Would move ${mismatched.length} member(s) and delete ${crossLinks.length} cross-subject link(s).`);
    console.log('Re-run without --dry-run to apply.');
  } else {
    console.log(`Moved ${moved} member(s); deleted ${crossLinks.length} cross-subject link(s).`);
  }
  console.log('');
}

main().then(() => process.exit(0)).catch(err => {
  console.error('Migration failed:', err);
  process.exit(1);
});
