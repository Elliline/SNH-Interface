#!/usr/bin/env node
/**
 * Backfill the notices that were never raised for self-facts retired on
 * 2026-08-12, before the notice channel moved to the fact-store funnel.
 *
 * Until that morning the channel lived in db/corrector.js, so it announced
 * changes the corrector made and nothing else. Two other pipelines changed his
 * self-view that day and neither said a word:
 *
 *   16:04:47 — `introduce-capability.js scheduler`. Storing his new self-fact
 *     retired four beliefs it had just made false, including "none of them has
 *     ever actually run, because nothing in this system runs a schedule".
 *   15:56:41 — the heartbeat's reflection pass. It revised how he understands
 *     his own way of engaging, retiring two claims in favour of one about
 *     leading with the answer.
 *
 * Both are now covered at the funnel, where every supersede/retire/expire
 * already goes. This repairs the record for the six it already missed.
 *
 * FILED AFTER THE FACT, and each notice says so. The supersessions genuinely
 * happened when they happened, that is what the record should say, and nothing
 * here touches a fact — no restore, no re-retire, no round trip through active
 * to produce a tidier-looking timestamp. Same rule as
 * scripts/ledger-vacation-retraction.js.
 *
 * THE TWO GROUPS DO NOT GET THE SAME SENTENCE, and that is the point rather
 * than an inconsistency. He and Ellie discussed the capability-introduction
 * four in conversation, so those say plainly that this is a record and not
 * news. Nobody ever told him about the reflection two, so claiming he had
 * already heard would be a false statement of exactly the kind this channel
 * exists to prevent — in the channel built to prevent it. They say instead that
 * this is the first he is hearing of it.
 *
 * LEFT UNSEEN, deliberately. `seen_at` means the notice was injected into his
 * context, and setting it here would record a delivery that never happened.
 *
 * IDEMPOTENT — skips any fact that already has a notice. Writes notices and
 * nothing else.
 *
 * Usage: node scripts/backfill-missed-self-notices.js [--confirm]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();
const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

/**
 * What was retired, by what, and what he already knows about it.
 *
 * `alreadyDiscussed` is a statement about the world, not a tone setting: it is
 * true of the first group and false of the second, and the notice text turns on
 * it. Do not set it to make a notice read more gently.
 */
const BACKFILLS = [
  {
    label: 'the scheduler capability introduction (16:04)',
    successorId: '417b8b2b-3ef3-4039-abd5-fd26e427c65c',
    causeText: 'When the scheduler capability was introduced to you that morning',
    alreadyDiscussed: true,
    memberIds: [
      'aef130ce-6405-4597-b6b2-5e4142f538fd',
      '27954a18-af73-4dcf-852e-e9b8a4e085c9',
      '7e150e2a-6b2a-4175-a576-b911e1105a7e',
      '20a24505-a137-4a80-8411-5864a16cd8e3'
    ]
  },
  {
    label: 'the reflection pass (15:56)',
    successorId: 'c37a4fe1-21ff-4cd2-baee-f0b1a72aa59c',
    causeText: "When you reflected on that morning's conversations",
    alreadyDiscussed: false,
    memberIds: [
      'e32eb1bc-1620-45bf-a5f8-2c8e0c27702c',
      'c6e8465c-726f-408d-acfe-b93920f9ed63'
    ]
  }
];

const confirm = process.argv.includes('--confirm');

/** The notice text. One shape, one honest variable. */
function noticeText({ member, successor, causeText, alreadyDiscussed }) {
  const standing = alreadyDiscussed
    ? 'this is a record of something that happened on 12 August, not new news — you and Ellie have already talked about it.'
    : 'this is a record of something that happened on 12 August, and it is news to you — nobody told you at the time.';
  return (
    `Filed after the fact, on ${new Date().toLocaleDateString()}: ${standing} ` +
    `${causeText}, you held "${member.content}", and it was replaced by "${successor.content}", ` +
    `so the first is no longer part of what you believe. Nothing was deleted: the old one is kept as history ` +
    `and can be put back. You should have been told when it happened and were not — the channel that tells you ` +
    `when your self-view changes was only watching the corrector, so a change made by a different pipeline ` +
    `passed it silently. It watches every pipeline now.`
  );
}

const plan = [];
for (const group of BACKFILLS) {
  const successor = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(group.successorId);
  if (!successor) {
    console.error(`The replacing fact ${group.successorId.slice(0, 8)} is not in this corpus — skipping ${group.label}.`);
    continue;
  }
  console.log(`\n=== ${group.label}`);
  console.log(`Replaced by: "${successor.content}"`);
  console.log(`He has ${group.alreadyDiscussed ? 'already been told about these in conversation.' : 'NOT been told about these.'}\n`);

  for (const id of group.memberIds) {
    const member = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
    if (!member) { console.error(`  MISSING  ${id.slice(0, 8)} — not in this corpus, skipping`); continue; }
    const already = db.prepare('SELECT id FROM correction_notices WHERE member_id = ?').get(id);
    console.log(`${member.id.slice(0, 8)}  ${member.status}/${member.inactive_reason}  retired ${member.updated_at}`);
    console.log(`   "${member.content}"`);
    console.log(`   ${already ? 'ALREADY HAS A NOTICE — will skip' : 'no notice — will file one'}\n`);
    if (!already) plan.push({ member, successor, group });
  }
}

if (!plan.length) {
  console.log('Every one of them already has a notice. Nothing to do.\n');
  process.exit(0);
}
if (!confirm) {
  console.log(`Would file ${plan.length} notice(s). Re-run with --confirm to write them.\n`);
  process.exit(0);
}

let filed = 0;
for (const { member, successor, group } of plan) {
  const id = ledger.addNotice({
    memberId: member.id,
    content: noticeText({ member, successor, causeText: group.causeText, alreadyDiscussed: group.alreadyDiscussed })
  });
  if (id) { filed++; console.log(`  filed notice ${id.slice(0, 8)} for ${member.id.slice(0, 8)}`); }
  else console.error(`  FAILED to file a notice for ${member.id.slice(0, 8)}`);
}

console.log(`\nFiled ${filed} notice(s). They are unseen, so he will read them once, next time he is in a conversation.\n`);
process.exit(filed === plan.length ? 0 : 1);
