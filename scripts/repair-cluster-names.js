/**
 * One-time repair: regenerate every cluster's name (user AND self) from its
 * member facts using the shared LLM namer. Cluster IDs are NOT touched — only
 * the `name` column. Prints a before → after line for every cluster.
 *
 * Usage:  node scripts/repair-cluster-names.js
 */
const db = require('../db/database');
const memoryClusters = require('../db/memory-clusters');

async function main() {
  db.initDatabase();
  const sql = db.getSqliteDb();

  const clusters = sql.prepare(
    'SELECT id, name, subject FROM memory_clusters ORDER BY subject, name'
  ).all();

  console.log(`Repairing ${clusters.length} cluster name(s)...\n`);
  console.log('SUBJECT  BEFORE                              →  AFTER');
  console.log('─'.repeat(78));

  let changed = 0, kept = 0, empty = 0;
  for (const c of clusters) {
    const members = sql.prepare(
      "SELECT content FROM cluster_members WHERE cluster_id = ? AND (status = 'active' OR status IS NULL)"
    ).all(c.id);

    if (members.length === 0) {
      empty++;
      console.log(`${(c.subject || 'user').padEnd(7)}  ${String(c.name).padEnd(34).slice(0, 34)}  →  (skipped — no active facts)`);
      continue;
    }

    const newName = await memoryClusters.generateClusterNameLLM(members, { subject: c.subject });
    if (newName && newName !== c.name) {
      sql.prepare('UPDATE memory_clusters SET name = ? WHERE id = ?').run(newName, c.id);
      changed++;
      console.log(`${(c.subject || 'user').padEnd(7)}  ${String(c.name).padEnd(34).slice(0, 34)}  →  ${newName}`);
    } else {
      kept++;
      console.log(`${(c.subject || 'user').padEnd(7)}  ${String(c.name).padEnd(34).slice(0, 34)}  =  (kept: ${newName || 'LLM unavailable'})`);
    }
  }

  console.log('─'.repeat(78));
  console.log(`Done. ${changed} renamed, ${kept} kept, ${empty} empty-skipped, ${clusters.length} total.`);
  process.exit(0);
}

main().catch(e => { console.error('REPAIR ERROR:', e); process.exit(1); });
