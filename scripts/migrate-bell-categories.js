#!/usr/bin/env node
/**
 * Move the entity's THINKING out of the bell, and say where each item went.
 *
 * The bell was carrying four kinds of thing that were never notifications:
 * self-coherence audits, reflection insights, cluster housekeeping, and job
 * results. They rang as though the entity were addressing her, and they are the
 * reason the channel stopped being trusted — 223 of 390 items ever raised ended
 * as `expired`.
 *
 * NOTHING IS DELETED, and nothing is left without a home. Each pending item is
 * written to its destination FIRST and only then marked `relocated`, which is a
 * status this table already uses for the job results moved out on 2026-08-18.
 * Delivered and expired history is untouched: it is the record of what actually
 * happened, and rewriting it would be a lie about the past to tidy the present.
 *
 *   audit               -> corrections_ledger, as an unresolved raise
 *   reflection-insight  -> data/memory/reflections.jsonl
 *   observation         -> the ops log (a done thing is a log entry)
 *   job-result          -> already in the robot queue; only unrung here
 *
 * Usage:
 *   node scripts/migrate-bell-categories.js --dry-run
 *   node scripts/migrate-bell-categories.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const DRY = process.argv.includes('--dry-run');

const db = require(path.join(ROOT, 'db/database'));

(async () => {
  db.initDatabase();
  const sql = db.getSqliteDb();
  const initiatives = require(path.join(ROOT, 'db/initiatives'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const OPS_DIR = path.join(db.getMemoryDir(), 'ops');

  const before = sql.prepare(
    'SELECT type, status, COUNT(*) n FROM initiatives GROUP BY type, status ORDER BY type'
  ).all();
  console.log(`${DRY ? 'DRY RUN — nothing will be written\n' : ''}BEFORE`);
  for (const r of before) console.log(`  ${String(r.type).padEnd(20)} ${String(r.status).padEnd(11)} ${r.n}`);
  const bellPendingBefore = sql.prepare(
    "SELECT COUNT(*) n FROM initiatives WHERE status='pending'"
  ).get().n;
  console.log(`  pending total: ${bellPendingBefore}`);

  const DESTINATION = {
    'audit': 'corrections',
    'reflection-insight': 'reflections',
    'observation': 'ops-log',
    'job-result': 'robot-queue'
  };

  const pending = sql.prepare("SELECT * FROM initiatives WHERE status = 'pending'").all();
  const moved = { corrections: [], reflections: [], 'ops-log': [], 'robot-queue': [] };
  const kept = [];

  for (const it of pending) {
    if (initiatives.isBellType(it.type)) { kept.push(it); continue; }
    const dest = DESTINATION[it.type];
    if (!dest) {
      // `question` and anything else unclassified STAYS PENDING. It is not a
      // notification, so it no longer rings — but it is still a live candidate
      // for the greeting path, and relocating it would silently drop something
      // the entity still means to ask.
      kept.push(it);
      continue;
    }
    if (DRY) { moved[dest].push(it); continue; }

    try {
      if (dest === 'corrections') {
        ledger.record({
          tier: 'semantic', action: 'supersede', subject: 'self',
          targetText: it.content, reason: it.content,
          evidence: {
            unresolved: true, reason_code: 'self-audit-relocated',
            source_ref: it.source_ref, raised_by: 'self-coherence-audit',
            relocated_from: `initiative:${it.id}`, originally_raised: it.created_at
          },
          reversible: false
        });
      } else if (dest === 'reflections') {
        memoryManager.appendReflectionRecord({
          at: it.created_at, kind: 'insight', insight: it.content,
          source: 'reflection', relocatedFrom: `initiative:${it.id}`
        });
      } else if (dest === 'ops-log') {
        factExtractor.appendToOpsLog(
          `Housekeeping (relocated from the bell, raised ${it.created_at}): ${it.content}`, OPS_DIR);
      }
      // Only now is it taken off the bell. Destination first, always — the
      // other order loses the item if the write fails.
      sql.prepare("UPDATE initiatives SET status = 'relocated' WHERE id = ? AND status = 'pending'").run(it.id);
      moved[dest].push(it);
    } catch (err) {
      console.error(`  FAILED to relocate ${it.id.slice(0, 8)} (${it.type}): ${err.message} — left pending`);
    }
  }

  console.log('\nMOVED');
  for (const [dest, items] of Object.entries(moved)) {
    if (!items.length) continue;
    console.log(`  -> ${dest} (${items.length})`);
    for (const it of items) console.log(`       ${it.id.slice(0, 8)} ${String(it.type).padEnd(20)} "${String(it.content).slice(0, 70)}"`);
  }
  if (!Object.values(moved).some(a => a.length)) console.log('  (nothing to move)');

  console.log('\nSTILL IN THE BELL');
  for (const it of kept) {
    const rings = initiatives.isBellType(it.type);
    console.log(`  ${rings ? 'RINGS  ' : 'silent '} ${it.id.slice(0, 8)} ${String(it.type).padEnd(20)} "${String(it.content).slice(0, 60)}"`);
  }
  if (!kept.length) console.log('  (nothing)');

  if (!DRY) {
    const bellNow = initiatives.countPendingForBell();
    const pendingNow = sql.prepare("SELECT COUNT(*) n FROM initiatives WHERE status='pending'").get().n;
    console.log(`\nAFTER`);
    console.log(`  pending total : ${bellPendingBefore} -> ${pendingNow}`);
    console.log(`  RINGING       : ${bellNow}`);
    console.log(`  relocated     : ${sql.prepare("SELECT COUNT(*) n FROM initiatives WHERE status='relocated'").get().n}`);
    console.log(`  history intact: ${sql.prepare("SELECT COUNT(*) n FROM initiatives WHERE status IN ('delivered','expired','dismissed')").get().n} rows untouched`);
  }
  process.exit(0);
})().catch(e => { console.error('FAILED:', e.stack || e.message); process.exit(1); });
