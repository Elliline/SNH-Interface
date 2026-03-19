#!/usr/bin/env node

require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const path = require('path');
const fs = require('fs');
const { initDatabase, initVectorStore, getSqliteDb, getClusterEmbeddingsTable } = require('../db/database');
const { generateClusterNameFromMembers, matchCuratedCategory, isValidClusterName } = require('../db/memory-clusters');

const DB_PATH = path.join(__dirname, '../data/chat.db');
const DRY_RUN = process.argv.includes('--dry-run');

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function safeId(id) {
  if (!UUID_RE.test(id)) throw new Error(`Invalid UUID for LanceDB filter: ${id}`);
  return id;
}

function shortId(id) {
  return id ? id.substring(0, 8) : '????????';
}

function printSectionHeader(title) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`  ${title}`);
  console.log('='.repeat(60));
}

function getClusterMemberCounts(db) {
  return db.prepare(`
    SELECT mc.id, mc.name,
           COUNT(cm.id) AS member_count
    FROM memory_clusters mc
    LEFT JOIN cluster_members cm ON cm.cluster_id = mc.id
    GROUP BY mc.id
    ORDER BY mc.name ASC
  `).all();
}

function printClusterState(db, label) {
  const clusters = getClusterMemberCounts(db);
  console.log(`\n${label}:`);
  let totalMembers = 0;
  for (const c of clusters) {
    console.log(`  [${shortId(c.id)}] "${c.name}" — ${c.member_count} members`);
    totalMembers += c.member_count;
  }
  console.log(`\nTotal clusters: ${clusters.length} | Total members: ${totalMembers}`);
  return { clusterCount: clusters.length, memberCount: totalMembers, clusters };
}

// ---------------------------------------------------------------------------
// Phase 1
// ---------------------------------------------------------------------------

function phase1Backup(db) {
  printSectionHeader('Phase 1 — Backup & Before State');

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = `${DB_PATH}.backup-${timestamp}`;

  try {
    fs.copyFileSync(DB_PATH, backupPath);
    console.log(`Backup created: ${backupPath}`);
  } catch (err) {
    console.error(`Warning: Could not create backup — ${err.message}`);
  }

  const { clusterCount, memberCount } = printClusterState(db, 'Before State');
  return { beforeClusterCount: clusterCount, beforeMemberCount: memberCount };
}

// ---------------------------------------------------------------------------
// Phase 2
// ---------------------------------------------------------------------------

async function phase2RenameGarbage(db) {
  printSectionHeader('Phase 2 — Rename Garbage Clusters');

  const clusters = db.prepare('SELECT id, name FROM memory_clusters ORDER BY name ASC').all();

  // Resolve all new names first (async LLM calls happen here, outside any transaction)
  const pendingRenames = [];

  for (const cluster of clusters) {
    const members = db.prepare(
      'SELECT content FROM cluster_members WHERE cluster_id = ?'
    ).all(cluster.id);

    if (members.length === 0) continue;

    let newName = null;

    // For multi-member clusters, generateClusterNameFromMembers has better frequency-weighted
    // curated map matching than matchCuratedCategory on raw concatenated text
    if (members.length > 1) {
      try {
        newName = generateClusterNameFromMembers(members.map(m => ({ content: m.content })));
      } catch (err) {
        console.error(`  Error generating name for [${shortId(cluster.id)}]: ${err.message}`);
      }
    }

    // For single-member clusters, or if multi-member naming failed validation,
    // try matchCuratedCategory on concatenated text
    if (!newName || !isValidClusterName(newName)) {
      const concatenatedText = members.map(m => m.content).join(' ');
      const curatedName = matchCuratedCategory(concatenatedText);
      if (curatedName) newName = curatedName;
    }

    if (!isValidClusterName(newName)) {
      newName = 'General';
    }

    if (newName !== cluster.name) {
      pendingRenames.push({ id: cluster.id, oldName: cluster.name, newName });
    }
  }

  // Apply all renames synchronously
  if (!DRY_RUN && pendingRenames.length > 0) {
    const updateStmt = db.prepare(
      'UPDATE memory_clusters SET name = ?, updated_at = ? WHERE id = ?'
    );
    const applyRenames = db.transaction(() => {
      for (const { id, newName } of pendingRenames) {
        updateStmt.run(newName, new Date().toISOString(), id);
      }
    });
    applyRenames();
  }

  for (const { oldName, newName } of pendingRenames) {
    const prefix = DRY_RUN ? '[DRY RUN] Would rename' : 'Renamed';
    console.log(`  ${prefix}: "${oldName}" → "${newName}"`);
  }

  if (pendingRenames.length === 0) {
    console.log('  No clusters needed renaming.');
  }
}

// ---------------------------------------------------------------------------
// Phase 3 — synchronous, safe to run in a transaction
// ---------------------------------------------------------------------------

function phase3MergeByName(db) {
  const groups = db.prepare(`
    SELECT mc.id, mc.name,
           COUNT(cm.id) AS member_count
    FROM memory_clusters mc
    LEFT JOIN cluster_members cm ON cm.cluster_id = mc.id
    GROUP BY mc.id
    ORDER BY mc.name ASC, member_count DESC
  `).all();

  const byName = {};
  for (const row of groups) {
    if (!byName[row.name]) byName[row.name] = [];
    byName[row.name].push(row);
  }

  const movedMembers = []; // { memberId, newClusterId }
  let mergeCount = 0;

  for (const [name, clustersInGroup] of Object.entries(byName)) {
    if (clustersInGroup.length <= 1) continue;

    const [target, ...sources] = clustersInGroup; // sorted DESC by member_count

    for (const source of sources) {
      const sourceMembers = db.prepare(
        'SELECT id FROM cluster_members WHERE cluster_id = ?'
      ).all(source.id);

      console.log(
        `  Merging: "${name}" (id: ${shortId(source.id)}, ${source.member_count} members)` +
        ` → target (id: ${shortId(target.id)}, ${target.member_count} members)`
      );

      if (!DRY_RUN) {
        db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE cluster_id = ?')
          .run(target.id, source.id);

        const sourceLinks = db.prepare(
          'SELECT * FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?'
        ).all(source.id, source.id);

        for (const link of sourceLinks) {
          const newA = link.cluster_a === source.id ? target.id : link.cluster_a;
          const newB = link.cluster_b === source.id ? target.id : link.cluster_b;

          if (newA === newB) {
            db.prepare('DELETE FROM cluster_links WHERE id = ?').run(link.id);
            continue;
          }

          const pairA = newA < newB ? newA : newB;
          const pairB = newA < newB ? newB : newA;

          const duplicate = db.prepare(`
            SELECT id FROM cluster_links
            WHERE ((cluster_a = ? AND cluster_b = ?) OR (cluster_a = ? AND cluster_b = ?))
              AND id != ?
          `).get(pairA, pairB, pairB, pairA, link.id);

          if (duplicate) {
            db.prepare('DELETE FROM cluster_links WHERE id = ?').run(link.id);
          } else {
            db.prepare('UPDATE cluster_links SET cluster_a = ?, cluster_b = ? WHERE id = ?')
              .run(newA, newB, link.id);
          }
        }

        db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(source.id);
      }

      for (const m of sourceMembers) {
        movedMembers.push({ memberId: m.id, newClusterId: target.id });
      }

      mergeCount++;
    }
  }

  if (mergeCount === 0) {
    console.log('  No clusters needed merging.');
  }

  return { movedMembers, mergeCount };
}

// ---------------------------------------------------------------------------
// Phase 4
// ---------------------------------------------------------------------------

async function phase4SyncLanceDB(movedMembers) {
  printSectionHeader('Phase 4 — Re-sync LanceDB cluster_id Metadata');

  if (DRY_RUN) {
    console.log('  Skipped in dry-run mode.');
    return;
  }

  if (movedMembers.length === 0) {
    console.log('  No members were moved; nothing to sync.');
    return;
  }

  let clusterTable;
  try {
    clusterTable = getClusterEmbeddingsTable();
  } catch (err) {
    console.error(`  Could not get cluster embeddings table: ${err.message}`);
    return;
  }

  let updatedCount = 0;

  for (const { memberId, newClusterId } of movedMembers) {
    try {
      await clusterTable.update({
        where: `member_id = "${safeId(memberId)}"`,
        valuesSql: { cluster_id: `'${safeId(newClusterId)}'` }
      });
      updatedCount++;
    } catch (err) {
      console.error(`  Warning: Failed to update LanceDB for member ${memberId}: ${err.message}`);
    }
  }

  console.log(`  Updated ${updatedCount} LanceDB rows.`);
}

// ---------------------------------------------------------------------------
// Phase 5 — synchronous, safe to run in a transaction
// ---------------------------------------------------------------------------

function phase5DeleteEmpty(db) {
  const emptyClusters = db.prepare(`
    SELECT mc.id, mc.name
    FROM memory_clusters mc
    LEFT JOIN cluster_members cm ON cm.cluster_id = mc.id
    GROUP BY mc.id
    HAVING COUNT(cm.id) = 0
  `).all();

  if (emptyClusters.length === 0) {
    console.log('  No empty clusters found.');
    return;
  }

  for (const cluster of emptyClusters) {
    const prefix = DRY_RUN ? '[DRY RUN] Would delete empty cluster' : 'Deleted empty cluster';
    console.log(`  ${prefix}: "${cluster.name}" (id: ${shortId(cluster.id)})`);

    if (!DRY_RUN) {
      db.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?')
        .run(cluster.id, cluster.id);
      db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(cluster.id);
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 6
// ---------------------------------------------------------------------------

function phase6AfterState(db, beforeClusterCount) {
  printSectionHeader('Phase 6 — After State');

  if (DRY_RUN) {
    console.log('  Dry run — no changes made.');
    return;
  }

  const { clusterCount: afterClusterCount } = printClusterState(db, 'After State');
  const merged = beforeClusterCount - afterClusterCount;

  console.log(
    `\nBefore: ${beforeClusterCount} clusters → After: ${afterClusterCount} clusters` +
    ` (merged ${Math.max(0, merged)})`
  );

  const generalClusters = db.prepare(
    "SELECT id, name FROM memory_clusters WHERE name = 'General'"
  ).all();

  if (generalClusters.length > 0) {
    console.log('\n--- Clusters named "General" (manual review recommended) ---');
    for (const gc of generalClusters) {
      const members = db.prepare(
        'SELECT content FROM cluster_members WHERE cluster_id = ? ORDER BY created_at ASC'
      ).all(gc.id);
      console.log(`\n  [${shortId(gc.id)}] General (${members.length} members):`);
      for (const m of members) {
        const preview = m.content.length > 120
          ? m.content.substring(0, 117) + '...'
          : m.content;
        console.log(`    - ${preview}`);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  if (DRY_RUN) {
    console.log('*** DRY RUN MODE — no changes will be written ***');
  }

  try {
    await initDatabase();
    await initVectorStore();
  } catch (err) {
    console.error(`Failed to initialize database: ${err.message}`);
    process.exit(1);
  }

  const db = getSqliteDb();

  const { beforeClusterCount } = phase1Backup(db);

  // Phase 2: async name resolution, then sync SQLite writes internally
  await phase2RenameGarbage(db);

  // Phases 3 & 5: fully synchronous — wrap in a single transaction
  let movedMembers = [];
  let mergeCount = 0;

  if (!DRY_RUN) {
    printSectionHeader('Phase 3 — Merge Clusters by Exact Name');
    const runMergeAndCleanup = db.transaction(() => {
      const result = phase3MergeByName(db);
      movedMembers = result.movedMembers;
      mergeCount = result.mergeCount;

      printSectionHeader('Phase 5 — Delete Empty Clusters');
      phase5DeleteEmpty(db);
    });

    try {
      runMergeAndCleanup();
    } catch (err) {
      console.error(`\nTransaction failed, rolling back: ${err.message}`);
      process.exit(1);
    }
  } else {
    printSectionHeader('Phase 3 — Merge Clusters by Exact Name');
    const result = phase3MergeByName(db);
    movedMembers = result.movedMembers;
    mergeCount = result.mergeCount;

    printSectionHeader('Phase 5 — Delete Empty Clusters');
    phase5DeleteEmpty(db);
  }

  await phase4SyncLanceDB(movedMembers);

  phase6AfterState(db, beforeClusterCount);

  process.exit(0);
}

main().catch(err => {
  console.error(`Unhandled error: ${err.message}`);
  console.error(err.stack);
  process.exit(1);
});
