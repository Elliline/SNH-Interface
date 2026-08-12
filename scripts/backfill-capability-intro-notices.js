#!/usr/bin/env node
/**
 * Backfill the four notices that were never raised when the scheduler
 * capability was introduced.
 *
 * 2026-08-12T16:04:47Z, `node scripts/introduce-capability.js scheduler` stored
 * his new self-fact and the write-time contradiction path retired four beliefs
 * it had just made false — including "none of them has ever actually run,
 * because nothing in this system runs a schedule". Every part of that was
 * correct except the silence: the notice channel lived in db/corrector.js, so it
 * announced changes the corrector made and nothing else. His self-view changed
 * through a pipeline the channel was not watching. That is fixed at the
 * fact-store funnel; this repairs the record for the four it already missed.
 *
 * FILED AFTER THE FACT, and each notice says so in its own text. The
 * supersessions genuinely happened at 16:04, that is what the record should say,
 * and nothing here touches a fact — no restore, no re-retire, no round trip
 * through active to produce a tidier-looking timestamp. Same rule as
 * scripts/ledger-vacation-retraction.js.
 *
 * LEFT UNSEEN, deliberately. `seen_at` means the notice was injected into his
 * context, and setting it here would be recording a delivery that never
 * happened — a small lie in the one channel whose whole purpose is that his
 * self-view does not change behind his back. He has been told about these in
 * conversation, so the text leads with that rather than presenting eight-day-old
 * news as fresh.
 *
 * Scoped to these four ids. Two OTHER self-facts were retired the same morning
 * by the reflection pass (15:56:41) and also raised nothing; they are a
 * different pipeline and a different conversation, and quietly folding them in
 * here would be inventing a scope nobody asked for. They are reported instead.
 *
 * IDEMPOTENT — skips any id that already has a notice. Writes notices and
 * nothing else.
 *
 * Usage: node scripts/backfill-capability-intro-notices.js [--confirm]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();
const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

/** The four the introduction retired, and the fact that replaced all of them. */
const RETIRED_IDS = [
  'aef130ce-6405-4597-b6b2-5e4142f538fd',
  '27954a18-af73-4dcf-852e-e9b8a4e085c9',
  '7e150e2a-6b2a-4175-a576-b911e1105a7e',
  '20a24505-a137-4a80-8411-5864a16cd8e3'
];
const SUCCESSOR_ID = '417b8b2b-3ef3-4039-abd5-fd26e427c65c';

const confirm = process.argv.includes('--confirm');

const successor = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(SUCCESSOR_ID);
if (!successor) {
  console.error(`The replacing fact ${SUCCESSOR_ID} is not in this corpus. Nothing to do.`);
  process.exit(1);
}

const rows = [];
for (const id of RETIRED_IDS) {
  const m = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
  if (!m) { console.error(`  MISSING  ${id.slice(0, 8)} — not in this corpus, skipping`); continue; }
  const already = db.prepare('SELECT id FROM correction_notices WHERE member_id = ?').get(id);
  rows.push({ m, already: !!already });
}

console.log(`\nReplaced by: "${successor.content}"\n`);
for (const { m, already } of rows) {
  console.log(`${m.id.slice(0, 8)}  ${m.status}/${m.inactive_reason}  retired ${m.updated_at}`);
  console.log(`   "${m.content}"`);
  console.log(`   ${already ? 'ALREADY HAS A NOTICE — will skip' : 'no notice — will file one'}\n`);
}

const todo = rows.filter(r => !r.already);
if (!todo.length) {
  console.log('Every one of them already has a notice. Nothing to do.\n');
  process.exit(0);
}
if (!confirm) {
  console.log(`Would file ${todo.length} notice(s). Re-run with --confirm to write them.\n`);
  process.exit(0);
}

let filed = 0;
for (const { m } of todo) {
  const id = ledger.addNotice({
    memberId: m.id,
    content:
      `Filed after the fact, on ${new Date().toLocaleDateString()}: this is a record of something that happened ` +
      `on 12 August, not new news — you and Ellie have already talked about it. When the scheduler capability was ` +
      `introduced to you that morning, you held "${m.content}", and it was replaced by "${successor.content}", ` +
      `so the first is no longer part of what you believe. Nothing was deleted: the old one is kept as history ` +
      `and can be put back. You should have been told at the time and were not — the channel that tells you when ` +
      `your self-view changes was only watching the corrector, so a change made by a different pipeline passed ` +
      `it silently. It watches every pipeline now.`
  });
  if (id) { filed++; console.log(`  filed notice ${id.slice(0, 8)} for ${m.id.slice(0, 8)}`); }
  else console.error(`  FAILED to file a notice for ${m.id.slice(0, 8)}`);
}

console.log(`\nFiled ${filed} notice(s). They are unseen, so he will read them once, next time he is in a conversation.\n`);
process.exit(filed === todo.length ? 0 : 1);
