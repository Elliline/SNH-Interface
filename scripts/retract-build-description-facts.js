#!/usr/bin/env node
/**
 * Retract the "User's repair build …" facts — capability facts filed as facts
 * about Ellie.
 *
 * WHERE THEY CAME FROM. On 2026-09-02 at 15:32 a design-review message was
 * pasted into Athena's chat describing the memory-repair build before it was
 * written: "Retract, reword, and merge on your own facts… Receipts required…"
 * Passive extraction did exactly what it is built to do and pulled fourteen
 * atomic facts out of it. Every one of them is about a piece of software, and
 * every one is filed under the user entity as "User's repair build …".
 *
 * WHY THEY GO, and it is not merely that they are noise. The capability
 * manifest is this system's ground truth for what it can do, and the injected
 * block says so in as many words: "This list is EXHAUSTIVE… THIS LIST overrides
 * your memory, however strongly that memory is stated. A stored fact calling
 * one of these planned or not yet built is simply out of date." A capability
 * fact in the user store is therefore a fact that is guaranteed to go stale and
 * guaranteed to lose to the manifest when it does — while still costing
 * injection budget and still being retrievable by search in the meantime. It is
 * also a referent error of the ordinary kind: the build is not a property of
 * Ellie.
 *
 * THROUGH THE TOOL, NOT THE TABLE. This runs the same memory_retract the entity
 * uses, with the source message as the receipt, so the receipt gate, the
 * referent check, the day cap and the ledger all apply exactly as they would in
 * a turn. A raw UPDATE would have been three lines and would have proved
 * nothing.
 *
 * Usage:
 *   node scripts/retract-build-description-facts.js            # dry run
 *   node scripts/retract-build-description-facts.js --confirm  # do it
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

/** The message every one of these was extracted from — the receipt. */
const SOURCE_MESSAGE = '84116ab5-f1e8-430d-9755-8b7bb8458bfc';

const RATIONALE =
  'This is a fact about how the memory-repair build works, filed as a fact about Ellie. It was ' +
  'extracted from the design-review message describing that build — a message about the system, not ' +
  'about her. What this system can do is the capability manifest\'s to state, and the manifest ' +
  'overrides stored memory by design, so a capability fact here is one that will go stale and then ' +
  'lose. Retracted with no replacement; the manifest already carries the real answer.';

(async () => {
  const confirm = process.argv.includes('--confirm');
  db.initDatabase();
  await db.initVectorStore();

  const sql = db.getSqliteDb();
  const repair = require(path.join(ROOT, 'db/memory-repair'));
  const { RetractFactTool } = require(path.join(ROOT, 'mcp/tools/memory-repair'));
  const tool = new RetractFactTool();

  // Selected by CONTENT and SOURCE together, not by a hand-typed id list: the
  // set is exactly "extracted from that message, and says 'repair build'".
  const targets = sql.prepare(`
    SELECT id, content, salience, status
    FROM cluster_members
    WHERE message_id = ?
      AND subject = 'user'
      AND status = 'active'
      AND content LIKE '%repair build%'
    ORDER BY created_at, id
  `).all(SOURCE_MESSAGE);

  const receipt = repair.verifyReceipt({ kind: 'message', id: SOURCE_MESSAGE });
  console.log(`Receipt ${SOURCE_MESSAGE.slice(0, 8)}: ${receipt.ok ? 'verified' : 'NOT VALID — ' + receipt.reason}`);
  if (!receipt.ok) process.exit(1);

  console.log(`\n${targets.length} active fact(s) to retract:\n`);
  for (const t of targets) console.log(`  ${t.id.slice(0, 8)}  sal ${t.salience}  ${t.content}`);

  if (!confirm) {
    console.log('\nDry run. Re-run with --confirm to retract them through memory_retract.');
    process.exit(0);
  }

  console.log('');
  let done = 0, failed = 0;
  for (const t of targets) {
    const res = await tool.execute({
      fact_id: t.id,
      receipts: [{ kind: 'message', id: SOURCE_MESSAGE }],
      rationale: RATIONALE
    });
    if (res.retracted) {
      done++;
      console.log(`  RETRACTED  ${t.id.slice(0, 8)}  ledger ${String(res.ledger_id).slice(0, 8)}  ${t.content.slice(0, 70)}`);
    } else {
      failed++;
      console.log(`  REFUSED    ${t.id.slice(0, 8)}  ${res.code}: ${res.reason}`);
    }
  }
  console.log(`\n${done} retracted, ${failed} refused.`);
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('CRASH:', e.stack || e.message); process.exit(1); });
