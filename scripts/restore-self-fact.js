#!/usr/bin/env node
/**
 * Put back a fact that was superseded off-ledger — a deliberate human path.
 *
 * WHY THIS EXISTS. `correctionsLedger.revert()` is the one shared undo, and both
 * the CLI and the Self tab's button go through it. It works by reading a LEDGER
 * ENTRY. So a supersession that never filed one cannot be undone by it: the
 * button has nothing to point at. On 2026-08-18 the capability introduction went
 * through `processSelfFacts`, which superseded an unrelated salience-9
 * declaration at cosine 0.741 and filed nothing, and there was no path back.
 *
 * This is the path back, and it is deliberately shaped like the identity-lock
 * setter rather than like a tool:
 *   - it takes a member id and requires --confirm
 *   - it is never reachable from a conversation, and there is no tool for it
 *   - the restore FILES A LEDGER ENTRY (the fact-store funnel does it, in the
 *     same transaction), which this script then enriches with the reason a
 *     person typed — an undo that leaves no trace is the defect it is undoing
 *   - it drops the correction notice that the bad supersession queued, because
 *     that notice tells him something about himself that is not true, and a
 *     notice is delivered exactly once — once he has read it, it has shaped him
 *
 * It is a REPAIR TOOL, not a routine one. If you find yourself reaching for it
 * regularly, the write path that keeps needing repair is the thing to fix.
 *
 * Usage:
 *   node scripts/restore-self-fact.js <member-id>              # report only
 *   node scripts/restore-self-fact.js <member-id> --confirm
 */

const path = require('path');
const ROOT = path.join(__dirname, '..');

const database = require(path.join(ROOT, 'db/database'));
const { getSqliteDb } = database;

(async () => {
  const id = process.argv[2];
  const confirm = process.argv.includes('--confirm');
  const why = (() => {
    const i = process.argv.indexOf('--reason');
    return i > -1 ? process.argv[i + 1] : null;
  })();

  if (!id) {
    console.error('Usage: node scripts/restore-self-fact.js <member-id> [--reason "..."] [--confirm]');
    process.exit(2);
  }

  database.initDatabase();
  // The vector store too, and not optionally: restore() re-adds the embedding,
  // and without a handle the fact comes back active but unfindable — the
  // 'active-not-retrievable' half of what reconcile() reports.
  await database.initVectorStore();

  const db = getSqliteDb();
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const correctionsLedger = require(path.join(ROOT, 'db/corrections-ledger'));

  const member = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
  if (!member) { console.error(`No fact with id ${id}`); process.exit(1); }

  const successor = member.successor_id
    ? db.prepare('SELECT id, content, salience FROM cluster_members WHERE id = ?').get(member.successor_id)
    : null;
  const notices = db.prepare(
    'SELECT id, content, seen_at FROM correction_notices WHERE member_id = ? AND seen_at IS NULL'
  ).all(id);

  console.log(`\nFact ${id.slice(0, 8)}`);
  console.log(`  status:     ${member.status}${member.inactive_reason ? ` (${member.inactive_reason})` : ''}`);
  console.log(`  claim_type: ${member.claim_type || '(none)'}   salience: ${member.salience}   subject: ${member.subject}`);
  console.log(`  source:     ${member.source}   created: ${member.created_at}`);
  console.log(`  text:       "${member.content}"`);
  if (successor) {
    console.log(`\n  It was superseded by ${successor.id.slice(0, 8)} (salience ${successor.salience}):`);
    console.log(`    "${successor.content.slice(0, 160)}${successor.content.length > 160 ? '…' : ''}"`);
  }
  console.log(`\n  Unseen notices about this change: ${notices.length}`);
  for (const n of notices) console.log(`    ${n.id.slice(0, 8)}: "${n.content.slice(0, 100)}…"`);

  if (member.status === 'active') {
    console.log('\nAlready active — nothing to restore.\n');
    process.exit(0);
  }
  if (!confirm) {
    console.log('\nReport only. Re-run with --confirm to restore it, file a ledger entry, and drop the notice(s).\n');
    process.exit(0);
  }

  // --- restore -------------------------------------------------------------
  // deliberate: true is what opens the identity lock. It is correct here and
  // only here: a person typed this command with --confirm.
  const res = await factStore.restore(id, { deliberate: true });
  if (!res.ok) { console.error(`\nRestore failed: ${res.reason || 'unknown'}\n`); process.exit(1); }
  console.log(`\nRestored: sqlite=${res.sqlite} vector=${res.vector}`);

  // --- ledger the restore --------------------------------------------------
  // reversible: 0 on purpose. The entry records an undo that has already been
  // applied; there is nothing further to revert, and a Revert button offering to
  // put the supersession BACK would be worse than no button at all.
  //
  // The restore itself filed the entry, inside its own transaction (see the
  // funnel note in db/fact-store.js). This script enriches that entry with the
  // reason a person typed, rather than filing a second one for one change.
  const ledgerId = res.ledgerId;
  const fields = {
    passId: `manual-restore-${new Date().toISOString().slice(0, 10)}`,
    tier: 'semantic',
    subject: member.subject || 'self',
    survivorId: successor ? successor.id : null,
    survivorText: successor ? successor.content : null,
    reason: why || 'Restored by hand: this fact was superseded by a write that filed no ledger entry, so it could not be reverted through the normal path. Nothing further to undo here.',
    evidence: { restoredBy: 'scripts/restore-self-fact.js', supersededBy: member.successor_id || null },
    reversible: false
  };
  const enriched = ledgerId && correctionsLedger.enrich(ledgerId, fields);
  console.log(`Ledger entry: ${ledgerId ? `${ledgerId.slice(0, 8)}${enriched ? ' (enriched with your reason)' : ' (could not be enriched)'}` : 'NONE — the restore filed no entry'}`);

  // --- drop the notice -----------------------------------------------------
  // A notice is delivered once. Leaving a false one queued means he is told, in
  // his own identity block, that he stopped believing something he still holds.
  let dropped = 0;
  for (const n of notices) if (correctionsLedger.deleteNotice(n.id)) dropped++;
  console.log(`Dropped ${dropped} unseen notice(s) about the change that has now been undone.`);

  try {
    const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
    factExtractor.appendToOpsLog(
      `Restored self-fact ${id.slice(0, 8)} by hand ("${member.content.slice(0, 80)}…"): it had been superseded off-ledger` +
      `${member.successor_id ? ` by ${member.successor_id.slice(0, 8)}` : ''}. ${dropped} unseen notice(s) about that change were dropped.`,
      path.join(database.getDataDir(), 'memory/ops')
    );
  } catch { /* console is the floor */ }

  const after = db.prepare('SELECT status, successor_id FROM cluster_members WHERE id = ?').get(id);
  console.log(`\nNow: status=${after.status}, successor=${after.successor_id || 'none'}\n`);
})().catch(err => {
  console.error('[restore-self-fact] error:', err);
  process.exit(1);
});
