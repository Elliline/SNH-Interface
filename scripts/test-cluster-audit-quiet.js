#!/usr/bin/env node
/**
 * The cluster audit judges the LIVE corpus, and the ops log reports CHANGE.
 *
 * Both halves of one four-day bug. "Persona & Lifestyle Habits" and "Cello
 * Practice Habits" held thirteen and eleven facts, every one of them superseded.
 * Nothing in either cluster was believed any more, and:
 *
 *   1. the rotation picked them anyway, because `member_count` counts ghosts;
 *   2. the auditor was shown the ghosts, because getCluster returns them for the
 *      Memory Map, so it proposed splits made entirely of dead facts;
 *   3. executeSplits refused every one — correctly, the write guard is
 *      active-only — and filed each refusal as an anomaly;
 *   4. the same 29 anomalies were written to the ops log every two hours for
 *      four days, burying the corrector's actual entries under them, while the
 *      two dead clusters ate 47s of a 69s pass.
 *
 * So this asserts the two rules that end it: a decision about a cluster reads
 * active members only, and an anomaly is reported once and thereafter counted.
 *
 * PURE. Runs against a throwaway SNH_DATA_DIR — no live corpus, no model.
 *
 * Usage: node scripts/test-cluster-audit-quiet.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

// Redirect the PROCESS before anything resolves a store. Same mechanism the
// replay uses; see db/database.js.
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-audit-test-'));
process.env.SNH_DATA_DIR = TMP;

const database = require('../db/database');
database.initDatabase();
const db = database.getSqliteDb();

const memoryClusters = require('../db/memory-clusters');
const memoryManager = require('../db/memory-manager');

let failures = 0;
function check(name, cond, detail) {
  if (cond) {
    console.log(`  PASS  ${name}`);
  } else {
    failures++;
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function makeCluster(name, members) {
  const id = randomUUID();
  const now = new Date().toISOString();
  db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?, ?, ?, ?, ?, ?)')
    .run(id, name, '', now, now, 'user');
  for (const [content, status] of members) {
    db.prepare('INSERT INTO cluster_members (id, cluster_id, content, source, created_at, status, subject) VALUES (?, ?, ?, ?, ?, ?, ?)')
      .run(randomUUID(), id, content, 'test', now, status, 'user');
  }
  return id;
}

// A cluster like "Cello Practice Habits": every fact superseded, name kept.
const allGhosts = makeCluster('All Ghosts', Array.from({ length: 11 },
  (_, i) => [`Superseded fact ${i}`, 'inactive']));
// One live fact and a pile of ghosts — nothing to judge, still on the Map.
const oneLive = makeCluster('One Live', [
  ['The only live fact', 'active'],
  ...Array.from({ length: 12 }, (_, i) => [`Superseded fact ${i}`, 'inactive'])
]);
// A genuinely oversized cluster.
const reallyBig = makeCluster('Really Big', Array.from({ length: 12 },
  (_, i) => [`Live fact ${i}`, 'active']));

console.log('\n1. Counting: ghosts are drawn, not decided on');
const byId = Object.fromEntries(memoryClusters.getClusters().map(c => [c.id, c]));
check('all-ghost cluster still reports its full member_count for the Map',
  byId[allGhosts].member_count === 11, `got ${byId[allGhosts].member_count}`);
check('all-ghost cluster reports 0 active members',
  byId[allGhosts].active_member_count === 0, `got ${byId[allGhosts].active_member_count}`);
check('mixed cluster counts 1 active of 13',
  byId[oneLive].member_count === 13 && byId[oneLive].active_member_count === 1,
  `got ${byId[oneLive].member_count}/${byId[oneLive].active_member_count}`);

// The rotation rule from runMaintenance, with the default cap.
const maxFacts = 10;
const oversized = Object.values(byId).filter(c => c.active_member_count > maxFacts).map(c => c.name);
check('only the genuinely oversized cluster enters the rotation',
  oversized.length === 1 && oversized[0] === 'Really Big', `got [${oversized}]`);

console.log('\n2. The auditor is never shown a corpus nobody believes');
(async () => {
  const ghostResult = await memoryManager.auditClusterCoherence(byId[allGhosts]);
  check('all-ghost cluster is skipped without an LLM call',
    ghostResult.skipped === 'no active members — ghosts only', `got ${JSON.stringify(ghostResult.skipped)}`);
  check('a skipped cluster is reported coherent, never split',
    ghostResult.coherent === true && ghostResult.splits.length === 0);

  const oneResult = await memoryManager.auditClusterCoherence(byId[oneLive]);
  check('a single live fact cannot be incoherent with itself',
    oneResult.skipped === 'only one active member', `got ${JSON.stringify(oneResult.skipped)}`);

  // getCluster must STILL return the ghosts — the Map draws them.
  const detail = memoryClusters.getCluster(allGhosts);
  check('getCluster still returns inactive members for the Memory Map',
    detail.members.length === 11, `got ${detail.members.length}`);

  console.log('\n3. An anomaly is reported once, then counted');
  const warnings = [
    'Fact id "aaa" in cluster "All Ghosts" is inactive — not moved, and its embedding left alone',
    'No valid facts found for split "Trust Level Design" in cluster "All Ghosts"'
  ];

  const first = memoryManager.partitionAnomalies(warnings);
  check('first sighting reports both in full',
    first.fresh.length === 2 && first.suppressed === 0,
    `fresh=${first.fresh.length} suppressed=${first.suppressed}`);

  const second = memoryManager.partitionAnomalies(warnings);
  check('second pass reports nothing new and counts both',
    second.fresh.length === 0 && second.suppressed === 2,
    `fresh=${second.fresh.length} suppressed=${second.suppressed}`);
  check('the count carries when it was first seen',
    typeof second.oldestSuppressedAt === 'string' && second.oldestSuppressedAt.length > 0);

  const third = memoryManager.partitionAnomalies([...warnings, 'Audit error for "Really Big": brain unreachable']);
  check('a genuinely new anomaly still gets through the memo',
    third.fresh.length === 1 && third.fresh[0].startsWith('Audit error'),
    `fresh=${JSON.stringify(third.fresh)}`);
  check('...alongside the count of the unchanged ones',
    third.suppressed === 2, `got ${third.suppressed}`);

  const row = db.prepare('SELECT seen_count FROM heartbeat_anomaly_state WHERE anomaly_key = ?').get(warnings[0]);
  check('the memo counts sightings rather than rewriting itself',
    row && row.seen_count === 3, `got ${row && row.seen_count}`);

  const dupInOnePass = memoryManager.partitionAnomalies(['same warning twice', 'same warning twice']);
  check('one pass seeing the same condition twice reports it once',
    dupInOnePass.fresh.length === 1, `got ${dupInOnePass.fresh.length}`);

  console.log(failures === 0
    ? '\nAll checks passed.\n'
    : `\n${failures} check(s) FAILED.\n`);
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
  process.exit(failures === 0 ? 0 : 1);
})();
