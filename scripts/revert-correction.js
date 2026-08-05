#!/usr/bin/env node
/**
 * Undo one correction, by ledger entry id.
 *
 * The semantic tier is allowed to act unattended precisely BECAUSE this exists:
 * "autonomous, reversible, logged" is one property, not three, and the
 * reversibility half is worthless if it is a claim rather than a command anyone
 * can run. So this reads only what the ledger recorded — the fact that lost, the
 * fact that won — and puts the first one back.
 *
 * What it does:
 *   1. restores the superseded/expired fact to active, re-embedding it so it is
 *      searchable again and not merely present
 *   2. clears the successor pointer, so the record no longer claims it was
 *      replaced by something that no longer replaces it
 *   3. marks the ledger entry reverted, keeping the entry as history
 *
 * It does NOT remove the survivor. A merge that folded a duplicate away leaves
 * both facts active again after a revert, which is the correct state to return
 * to — the corrector's judgement is undone, not replaced by an opposite one.
 *
 * Usage:
 *   node scripts/revert-correction.js <ledgerId|prefix> [--confirm]
 *   node scripts/revert-correction.js --list [n]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

(async () => {
  const args = process.argv.slice(2);
  db.initDatabase();
  await db.initVectorStore();
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

  if (args[0] === '--list' || args.length === 0) {
    const n = parseInt(args[1], 10) || 20;
    const rows = ledger.list({ limit: n });
    if (!rows.length) { console.log('\nNo corrections recorded yet.\n'); process.exit(0); }
    console.log(`\nMost recent ${rows.length} correction(s):\n`);
    for (const r of rows) {
      const state = r.reverted_at ? 'REVERTED' : (r.reversible ? 'revertible' : 'not revertible');
      console.log(`  ${r.id.slice(0, 8)}  ${r.created_at.slice(0, 19)}  [${r.tier}/${r.action}]  ${state}`);
      console.log(`     ${trunc(r.target_text, 90)}`);
      if (r.survivor_text) console.log(`     → kept: ${trunc(r.survivor_text, 90)}`);
      console.log(`     ${trunc(r.reason, 140)}\n`);
    }
    process.exit(0);
  }

  const entry = ledger.get(args[0]);
  if (!entry) { console.error(`No ledger entry matching "${args[0]}".`); process.exit(2); }
  if (entry.ambiguous) {
    console.error(`"${args[0]}" matches ${entry.ambiguous.length} entries: ${entry.ambiguous.map(i => i.slice(0, 8)).join(', ')}`);
    process.exit(2);
  }
  if (entry.reverted_at) { console.error(`Already reverted at ${entry.reverted_at}.`); process.exit(2); }
  if (!entry.reversible) { console.error(`Entry ${entry.id.slice(0, 8)} is recorded as not revertible: ${entry.reason}`); process.exit(2); }
  if (!entry.target_id) { console.error('Entry has no target fact to restore.'); process.exit(2); }

  console.log(`\nCorrection ${entry.id.slice(0, 8)}  [${entry.tier}/${entry.action}]  ${entry.created_at}`);
  console.log(`  would restore: "${entry.target_text}"`);
  if (entry.survivor_text) console.log(`  kept in place: "${entry.survivor_text}"  (NOT removed)`);
  console.log(`  original reason: ${entry.reason}`);

  if (!args.includes('--confirm')) {
    console.log(`\nRe-run with --confirm to apply:\n  node scripts/revert-correction.js ${entry.id.slice(0, 8)} --confirm\n`);
    process.exit(0);
  }

  // One revert path, shared with the Self tab's button — a second copy of this
  // would be a second answer to "what does reverting actually do".
  const res = await ledger.revert(entry.id, { by: 'revert-correction script' });
  if (!res.ok) { console.error(`\nRestore failed: ${res.reason || 'unknown'}`); process.exit(1); }

  console.log(`\nRestored. sqlite=${res.sqlite} vector=${res.vector}`);
  console.log(`Ledger entry ${entry.id.slice(0, 8)} marked reverted.\n`);
  process.exit(0);
})().catch(err => { console.error('revert failed:', err); process.exit(1); });
