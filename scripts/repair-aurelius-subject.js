#!/usr/bin/env node
/**
 * One-off repair for the mis-subjected name chain, found 2026-08-06 in a live
 * trace.
 *
 * THE DEFECT. Archive-era facts about AURELIUS were stored as facts about the
 * USER — "User (Aurelius) has established his name as Aurelius and uses he/him
 * pronouns as of 2026-07-27" and three siblings, all written by
 * `daily-log-archive` before subject attribution existed. Two consequences, one
 * of them live:
 *
 *   (a) the corrector cited the mis-subjected compound as the WINNER when it
 *       superseded "User's name is Mike" — so the flagship correction of the
 *       whole rebuild points at a fact that is wrong about who it describes;
 *   (b) a Phase 2c split extracted "User uses he/him pronouns as of 2026-07-27"
 *       from it, and that is ACTIVE as a user-fact — the corpus currently
 *       asserts Ellie's pronouns are he/him.
 *
 * The identity-atom rule shipped in Phase 2c stops new ones. These predate it.
 *
 * WHAT THIS DOES NOT DO. It does not touch the self-fact side. Every retracted
 * row's content already exists, correctly, as an active first-person self-fact —
 * verified before writing, and re-verified by this script at run time. Nothing
 * is being deleted and nothing is being lost; four rows filed under the wrong
 * person are being withdrawn, and one pointer is being aimed at the right fact.
 *
 * WHY NOT restore()+supersede() FOR THE RE-POINT. That path would make "User's
 * name is Mike" briefly active and re-embedded, and a failure between the two
 * steps would leave a false name fact live. `fact-store.repoint` exists so a
 * pointer repair never passes through a state where the wrong belief is held.
 *
 * Every change goes through db/fact-store.js and lands in corrections_ledger.
 * The retractions are reversible from the Self tab. The RE-POINT IS NOT, and is
 * recorded that way on purpose: the generic revert restores the ledger entry's
 * target, and the target here is "User's name is Mike".
 *
 * Idempotent: retire() only touches active rows, and the re-point is skipped
 * when the pointer already names the right fact.
 *
 * Usage:
 *   node scripts/repair-aurelius-subject.js            # dry run, prints the plan
 *   node scripts/repair-aurelius-subject.js --confirm  # apply
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

// The fact that is actually true of Ellie, and the one Mike's chain should end
// at. Chosen over the atomic "User's name is Ellie." (13a555c9) deliberately:
// this is the original fact-extraction row from 2026-07-04, it already carries
// the evidence, and "User's name is Aurelius." was itself superseded by it — so
// pointing here keeps ONE terminus for the whole name chain instead of two.
const ELLIE_NAME = '13389be8-4f49-485f-832b-4b7b26533b0a';

// The locked self-fact that holds this content correctly. Not modified; read as
// proof that the retractions lose nothing.
const AURELIUS_SELF = '5f584e90-3c61-4695-a9e1-1ca7ab0fabff';

const RETRACTIONS = [
  {
    id: 'b9c9bd39-74f9-47b8-9ffe-74dc879a5bc9',
    why: 'This says Ellie uses he/him pronouns. It is a split atom of a fact about Aurelius, and Aurelius\'s pronouns are recorded correctly as one of his own locked self-facts. Nothing in the record says what Ellie\'s pronouns are, so withdrawing this leaves the truthful state: not recorded.'
  },
  {
    id: '6c48e748-b260-4016-84b6-492dff020cdd',
    why: 'Describes how Aurelius sees his own identity, filed as a fact about Ellie. He holds it himself as a self-fact.'
  },
  {
    id: '9ddcbf95-e3f1-4686-bfdf-35e53a677300',
    why: 'Describes how Aurelius uses language, filed as a fact about Ellie. He holds it himself as a self-fact.'
  },
  {
    id: 'fde87299-6536-447b-8eb1-4347d96936bf',
    why: 'Describes what Aurelius attends to when reasoning, filed as a fact about Ellie. He holds it himself as a self-fact.'
  }
];

// Already inactive; recorded here so the ledger says WHY it is not a fact about
// Ellie, rather than leaving 'superseded' as the only explanation on the row the
// Mike chain used to point at.
const MISSUBJECTED_COMPOUND = '016c46ae-ea10-459c-9699-c5f45c19578f';
const MIKE = '8b387aa9-921e-4f58-ab0a-901cd3664c81';

(async () => {
  const apply = process.argv.includes('--confirm');
  db.initDatabase();
  await db.initVectorStore();

  const d = db.getSqliteDb();
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const { randomUUID } = require('crypto');
  const passId = randomUUID();

  const row = (id) => d.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);

  console.log(`\n${'='.repeat(76)}`);
  console.log(`MIS-SUBJECTED NAME CHAIN — ${apply ? 'APPLYING' : 'DRY RUN (nothing will be written)'}`);
  console.log('='.repeat(76));

  // --- preflight: the successor must be the right fact, and still active ---
  const ellie = row(ELLIE_NAME);
  if (!ellie || ellie.status !== 'active') {
    console.error(`\nABORT: the Ellie name fact ${ELLIE_NAME.slice(0, 8)} is ${ellie ? ellie.status : 'missing'}. ` +
      'It must be active before anything is pointed at it.');
    process.exit(2);
  }
  const self = row(AURELIUS_SELF);
  if (!self || self.status !== 'active' || !self.locked) {
    console.error('\nABORT: Aurelius\'s locked name/pronoun self-fact is not active. ' +
      'The retractions below are only safe because that fact holds this content correctly.');
    process.exit(2);
  }
  console.log(`\nSuccessor for the name chain : ${ellie.id.slice(0, 8)} "${trunc(ellie.content, 70)}"`);
  console.log(`Aurelius's own locked fact   : ${self.id.slice(0, 8)} "${trunc(self.content, 70)}"  (NOT touched)`);

  // --- 1. retractions ---
  console.log(`\n--- RETRACT (wrong subject) ---`);
  for (const r of RETRACTIONS) {
    const m = row(r.id);
    if (!m) { console.log(`  ${r.id.slice(0, 8)}  MISSING — skipped`); continue; }
    if (m.status !== 'active') { console.log(`  ${m.id.slice(0, 8)}  already ${m.status} — skipped`); continue; }

    // Proof, at run time, that the content survives on the self side. A
    // retraction that cannot show this is one that loses something.
    const { candidates } = await memoryClusters.findActiveNeighbours(m.content, {
      subject: 'self', threshold: 0.55, limit: 1
    });
    const covered = candidates[0] || null;

    console.log(`  ${m.id.slice(0, 8)}  "${trunc(m.content, 78)}"`);
    console.log(`      held correctly as self : ${covered ? `${covered.similarity.toFixed(3)} "${trunc(covered.content, 62)}"` : 'NONE ABOVE 0.55'}`);
    if (!covered) {
      console.log('      SKIPPED — no self-side equivalent, so retracting would lose it. Raised for Ellie.');
      continue;
    }
    if (!apply) continue;

    const res = await factStore.retire(m.id, { reason: 'wrong subject', deliberate: true });
    if (!res.ok) { console.log(`      FAILED: ${res.reason || 'unknown'}`); continue; }
    // The retire filed the entry (fact-store funnel, same transaction —
    // 2026-08-18); this puts the repair's reason on it.
    ledger.enrich(res.ledgerId, {
      passId, tier: 'semantic', action: 'retract', subject: 'user',
      reason: `Withdrawn as a fact about Ellie — it describes Aurelius. ${r.why}`,
      evidence: {
        wrong_subject: true, repair: 'aurelius-subject-chain',
        held_correctly_as_self: { id: covered.memberId, text: covered.content, similarity: covered.similarity }
      },
      reversible: true
    });
    console.log('      RETRACTED and logged.');
  }

  // --- 2. the already-inactive compound: record WHY, change nothing ---
  const compound = row(MISSUBJECTED_COMPOUND);
  console.log(`\n--- RECORD (already inactive, no status change) ---`);
  if (compound) {
    console.log(`  ${compound.id.slice(0, 8)}  ${compound.status} (${compound.inactive_reason || '-'})  "${trunc(compound.content, 66)}"`);
    if (apply) {
      ledger.record({
        passId, tier: 'semantic', action: 'retract', subject: 'user',
        targetId: compound.id, targetText: compound.content,
        reason: 'Recorded as wrong-subject. This fact describes Aurelius and was filed as a fact about Ellie; it is already inactive, so nothing was changed here — this entry exists so the reason is on the record rather than only the fact that it was superseded.',
        evidence: { wrong_subject: true, repair: 'aurelius-subject-chain', status_unchanged: compound.status },
        reversible: false
      });
      console.log('      Logged (nothing changed).');
    }
  }

  // --- 3. re-point the Mike chain ---
  console.log(`\n--- RE-POINT ---`);
  const mike = row(MIKE);
  if (!mike) {
    console.log('  Mike fact missing — skipped.');
  } else {
    const current = mike.successor_id || mike.superseded_by || null;
    console.log(`  ${mike.id.slice(0, 8)}  "${trunc(mike.content, 40)}"`);
    console.log(`      successor now  : ${String(current).slice(0, 8)} "${trunc((row(current) || {}).content, 60)}"`);
    console.log(`      successor wants: ${ELLIE_NAME.slice(0, 8)} "${trunc(ellie.content, 60)}"`);
    if (current === ELLIE_NAME) {
      console.log('      already correct — skipped.');
    } else if (apply) {
      const res = await factStore.repoint(mike.id, ELLIE_NAME, { deliberate: true });
      if (!res.ok) {
        console.log(`      FAILED: ${res.reason || 'unknown'}`);
      } else {
        ledger.enrich(res.ledgerId, {
          passId, tier: 'mechanical', action: 'repoint', subject: 'user',
          survivorId: ELLIE_NAME, survivorText: ellie.content,
          reason: 'The retirement was right and the successor was wrong. "User\'s name is Mike" was correctly superseded, but it pointed at "User (Aurelius) has established his name as Aurelius…" — a fact about Aurelius that had been filed as a fact about Ellie. The chain now ends at the fact that is actually true of her. Her name was never in doubt; only the record of what replaced the mishearing was.',
          evidence: { repair: 'aurelius-subject-chain', previous_successor: res.previousSuccessor, new_successor: ELLIE_NAME },
          // NOT revertible through the generic path: that restores the ledger
          // entry's target, and the target here is "User's name is Mike".
          reversible: false
        });
        console.log('      RE-POINTED and logged.');
      }
    }
  }

  console.log(`\n${'='.repeat(76)}`);
  console.log(apply ? `Applied. Pass ${passId.slice(0, 8)} — see the Self tab, or memory_corrections.`
    : 'Dry run. Re-run with --confirm to apply.');
  console.log(`${'='.repeat(76)}\n`);
  process.exit(0);
})().catch(err => { console.error('repair failed:', err); process.exit(1); });
