#!/usr/bin/env node
/**
 * One-off data repair for the correction-supersession failure found 2026-07-16:
 * the 7/8 "Its Bernice, i was spelling it wrong" correction stored a new fact
 * but never superseded the misspelled original, so "Bernie" kept re-surfacing
 * in chat. Today's re-correction failed the same way and added a duplicate.
 *
 * Applies the supersessions the pipeline SHOULD have made — via the normal
 * supersedeFact path, so history is preserved (nothing is deleted):
 *   8b05824b (Bernie/manager)            → superseded by bd135dee (Bernice/Director of Rooms at ISH)
 *   c7a04ab6 (Bernice dupe, salience 10) → superseded by bd135dee (which inherits salience 10)
 *   71c821de (ISH-client dupe, sal. 1)   → superseded by b77ce55a (7/7 defining fact)
 * plus removal of the superseded facts' MEMORY.md bullets (SQLite keeps history;
 * only the injected markdown copies are pruned).
 *
 * Idempotent: supersedeFact only touches status='active' rows, so a re-run
 * finds nothing to do.
 *
 * Usage: node scripts/repair-bernice-supersession.js [--apply]
 *   (dry-run by default; pass --apply to actually supersede)
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

const REPAIRS = [
  {
    oldId: '8b05824b-e9dd-4f3e-b20f-92029d448ecf',
    newId: 'bd135dee-33d8-4798-abe6-1a02044acea1',
    why: 'user correction 7/8 + 7/16: her name is Bernice, not Bernie, and she is Director of Rooms'
  },
  {
    oldId: 'c7a04ab6-d456-468d-b440-97596bb31855',
    newId: 'bd135dee-33d8-4798-abe6-1a02044acea1',
    why: 'duplicate of the fuller 7/8 Bernice fact (re-extracted from today\'s re-correction)'
  },
  {
    oldId: '71c821de-e2d4-43fb-9984-0eb4358eea7c',
    newId: 'b77ce55a-58f4-4067-a2d2-7127645b1d32',
    why: 'duplicate of the 7/7 defining fact (salience scorer itself called it redundant)'
  }
];

(async () => {
  const apply = process.argv.includes('--apply');
  db.initDatabase();
  const sql = db.getSqliteDb();
  const mc = require(path.join(ROOT, 'db/memory-clusters'));
  const fx = require(path.join(ROOT, 'db/fact-extractor'));
  const memoryFile = path.join(fx.MEMORY_DIR, 'MEMORY.md');

  const getFact = id => sql.prepare(
    'SELECT id, content, status, superseded_by, salience FROM cluster_members WHERE id = ?'
  ).get(id);

  let applied = 0;
  for (const r of REPAIRS) {
    const oldFact = getFact(r.oldId);
    const newFact = getFact(r.newId);
    if (!oldFact || !newFact) {
      console.log(`SKIP ${r.oldId.slice(0, 8)}: fact row missing (old=${!!oldFact}, new=${!!newFact})`);
      continue;
    }
    if (oldFact.status !== 'active') {
      console.log(`SKIP ${r.oldId.slice(0, 8)}: already ${oldFact.status} (→ ${String(oldFact.superseded_by).slice(0, 8)})`);
      continue;
    }
    if (newFact.status !== 'active') {
      console.log(`SKIP ${r.oldId.slice(0, 8)}: superseding fact ${r.newId.slice(0, 8)} is not active (${newFact.status})`);
      continue;
    }

    console.log(`${apply ? 'SUPERSEDE' : 'WOULD SUPERSEDE'}: "${oldFact.content}"`);
    console.log(`  → replaced by "${newFact.content}"`);
    console.log(`  reason: ${r.why}`);

    // Same rule as the live pipeline: a superseding fact inherits at least the
    // salience of the fact it replaces.
    const inherit = oldFact.salience > newFact.salience ? oldFact.salience : null;
    if (inherit) console.log(`  salience: ${r.newId.slice(0, 8)} ${newFact.salience} → ${inherit} (inherited)`);

    if (!apply) continue;

    if (!mc.supersedeFact(r.oldId, r.newId)) {
      console.log(`  !! supersedeFact reported no change — skipping follow-ups`);
      continue;
    }
    applied++;
    if (inherit) mc.updateFactSalience(r.newId, inherit);
    const removed = fx.removeFactLineFromMemory(oldFact.content, memoryFile);
    console.log(`  MEMORY.md line ${removed ? 'removed' : 'not present (nothing to remove)'}`);
    fx.appendToDailyLog(
      `Superseded fact: "${oldFact.content}" → replaced by "${newFact.content}" (manual repair: correction-supersession)`,
      fx.DAILY_DIR
    );
  }

  console.log(`\n${apply ? `Applied ${applied} supersession(s).` : 'Dry run — pass --apply to make changes.'}`);
})();
