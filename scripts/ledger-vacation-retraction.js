#!/usr/bin/env node
/**
 * Backfill the ledger entry for one hand-retraction that happened before the
 * route wrote one.
 *
 * 2026-08-10T16:59:20Z Ellie typed "vacation" into the Memory tab by accident.
 * Nine seconds later she pressed the delete button. It worked — the row went
 * inactive/retracted and its embedding was dropped — but the Facts tab re-drew
 * the retired row exactly as before, so the fact appeared to still be there and
 * the button appeared to do nothing. `DELETE /api/memory/fact/:id` also wrote no
 * ledger entry, which is what this repairs; the route does it from now on.
 *
 * This does NOT restore-and-re-retract. The retraction genuinely happened at
 * 16:59:29, that is what the record should say, and round-tripping the fact
 * through active would briefly put "vacation" back into retrieval to produce a
 * tidier-looking timestamp. The entry says plainly that it was filed after the
 * fact.
 *
 * Deliberately scoped to this ONE id. Two older retractions (2026-07-27) also
 * have no entry; they predate the corrections ledger existing at all, and
 * writing entries for them would be inventing a record rather than repairing
 * one.
 *
 * IDEMPOTENT — refuses if an entry already exists. Writes one ledger row and
 * nothing else; no fact is touched.
 *
 * Usage: node scripts/ledger-vacation-retraction.js [--confirm]
 */
const database = require('../db/database');
database.initDatabase();
const db = database.getSqliteDb();
const ledger = require('../db/corrections-ledger');

const TARGET_ID = '79a84ec0-866e-4606-9c28-87169ef650ab';
const confirm = process.argv.includes('--confirm');

const member = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(TARGET_ID);
if (!member) {
  console.error(`No fact ${TARGET_ID}. Nothing to do.`);
  process.exit(1);
}

console.log(`fact     : "${member.content}"`);
console.log(`status   : ${member.status} / ${member.inactive_reason}`);
console.log(`retracted: ${member.updated_at}`);

if (member.status === 'active') {
  console.error('\nThat fact is ACTIVE. This script only records a retraction that already happened — it does not perform one. Retire it from the Memory tab first.');
  process.exit(1);
}
if (member.inactive_reason !== 'retracted') {
  console.error(`\nThat fact is inactive for a different reason (${member.inactive_reason}). Refusing — this entry would misdescribe it.`);
  process.exit(1);
}

const existing = db.prepare('SELECT id, action, created_at FROM corrections_ledger WHERE target_id = ?').all(TARGET_ID);
if (existing.length > 0) {
  console.log(`\nAlready ledgered (${existing.map(e => `${e.action} @ ${e.created_at}`).join(', ')}). Nothing to do.`);
  process.exit(0);
}

if (!confirm) {
  console.log('\nDRY RUN — would record one ledger entry (action: retract, reversible). Re-run with --confirm.');
  process.exit(0);
}

const id = ledger.record({
  passId: 'manual-retract-backfill-2026-08-10',
  tier: 'mechanical',
  action: 'retract',
  subject: member.subject || 'user',
  targetId: TARGET_ID,
  targetText: member.content,
  reason: `Ellie removed this fact by hand from the Memory tab on 2026-08-10 — it was typed in by accident. Retired, not deleted; the row is kept as history and can be restored. This entry was filed after the fact: the removal route did not write one at the time, which is fixed.`,
  evidence: {
    reason_code: 'user-requested-removal',
    via: 'DELETE /api/memory/fact/:id',
    retracted_at: member.updated_at,
    entry_backfilled_at: new Date().toISOString(),
    source: member.source || null,
    cluster_id: member.cluster_id
  },
  reversible: true
});

console.log(id ? `\nRecorded ledger entry ${id}. Revertible from the Self tab or scripts/revert-correction.js.` : '\nFAILED to record entry.');
process.exit(id ? 0 : 1);
