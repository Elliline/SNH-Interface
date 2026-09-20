#!/usr/bin/env node
/**
 * THE ENTITY ACTING ON ITS OWN MEMORY, AND EVERY GUARDRAIL REFUSING.
 *
 * Athena's design, and her sentence is the shape of this suite: "the guardrails
 * have to live outside the loop […] The power isn't the risky part. The loop
 * is." So the assertions that matter here are the NEGATIVE ones — each guardrail
 * gets a test that makes it refuse, because a guard that stops guarding while
 * still returning success is the failure mode this codebase keeps finding.
 *
 *   1. An operation with no receipt is UNAVAILABLE, not warned about.
 *   2. A cross-entity refile whose receipt does not support the referent stops.
 *   3. A decided pair, re-fired by the audit on unchanged facts, is DROPPED.
 *   4. A merge that would fold a felt report into an anchored claim is refused.
 *   5. The day cap blocks the N+1th change to the entity's own self-model.
 *
 * The model is pinned throughout: what is under test is the routing and the
 * refusals, never a judgement an engine makes.
 *
 * Usage: node scripts/test-memory-repair.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const ROOT = path.join(__dirname, '..');
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-repair-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => { if (!INHERITED) { try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {} } });

const db = require(path.join(ROOT, 'db/database'));

let passed = 0, failed = 0;
const ok = (n, c, d = '') => { if (c) { passed++; console.log(`  PASS  ${n}`); } else { failed++; console.log(`  FAIL  ${n}${d ? ` — ${d}` : ''}`); } };
const section = t => console.log(`\n=== ${t} ===`);

(async () => {
  db.initDatabase();
  const sql = db.getSqliteDb();
  const repair = require(path.join(ROOT, 'db/memory-repair'));
  const auditDecisions = require(path.join(ROOT, 'db/audit-decisions'));
  const selection = require(path.join(ROOT, 'db/self-fact-selection'));
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const entities = require(path.join(ROOT, 'db/entities'));

  // Pin every engine call. Union text is the only one the ops themselves make.
  const factMerge = require(path.join(ROOT, 'db/fact-merge'));
  const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
  memoryManager.callLLM = async () => ({ content: 'PINNED' });
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  memoryClusters.generateEmbedding = async () => null;

  const selfE = entities.selfEntity();
  const userE = entities.userEntity();

  const cluster = (name, subject) => {
    const id = randomUUID(); const now = new Date().toISOString();
    sql.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
      .run(id, name, '', now, now, subject);
    return id;
  };
  const selfCluster = cluster('Self', 'self');
  const userCluster = cluster('Ellie', 'user');

  /** Seed a fact. `messageId` present ⇒ anchored; absent ⇒ felt. */
  const seed = (content, {
    subject = 'self', entityId = null, salience = 5, claimType = 'claim',
    messageId = null, verbatim = null, cid = null, locked = 0, lockCategory = null
  } = {}) => {
    const id = randomUUID(); const now = new Date().toISOString();
    sql.prepare(`
      INSERT INTO cluster_members
        (id, cluster_id, content, source, importance, created_at, updated_at, status, subject,
         subject_entity_id, salience, claim_type, message_id, verbatim_source_text, anchor, locked, lock_category)
      VALUES (?,?,?,'test',0.5,?,?,'active',?,?,?,?,?,?,?,?,?)
    `).run(id, cid || (subject === 'self' ? selfCluster : userCluster), content, now, now, subject,
           entityId || (subject === 'self' ? selfE.id : userE.id), salience, claimType,
           messageId, verbatim, (messageId || verbatim) ? 'anchored' : 'felt', locked, lockCategory);
    return factStore.getMember(id);
  };

  /** A real message row, so a receipt can point at something that exists. */
  const message = (role, content) => {
    const convo = sql.prepare("SELECT id FROM conversations LIMIT 1").get()
      || { id: db.createConversation('T', 'test', 'user').id || db.createConversation('T', 'test', 'user') };
    const cid = typeof convo === 'string' ? convo : convo.id;
    return db.addMessage(cid, role, content, 'test');
  };
  const ellieMsg = message('user', 'That fact about the clinic address was wrong — it belongs to the other client.');
  const entityMsg = message('assistant', 'I think I filed that under the wrong name.');

  // ══════════════════════════════════════ 1. receipts
  section('1. No receipt, no operation — and it is unavailable, not warned about');

  const wrongFact = seed('User is Claude', { subject: 'user', salience: 9, messageId: ellieMsg });

  const noReceipt = await repair.retract({ memberId: wrongFact.id, receipts: [], rationale: 'it is wrong' });
  ok('a retract with no receipt is refused', noReceipt.ok === false, JSON.stringify(noReceipt.code));
  ok('…and reports itself UNAVAILABLE rather than merely failing',
     noReceipt.available === false && /receipt/i.test(noReceipt.reason), noReceipt.reason);
  ok('…and the fact is untouched',
     factStore.getMember(wrongFact.id).status === 'active');

  const fakeReceipt = await repair.retract({
    memberId: wrongFact.id, receipts: [{ kind: 'message', id: 'not-a-real-message-id' }], rationale: 'x'
  });
  ok('a receipt naming something that does not exist is not a receipt',
     fakeReceipt.ok === false && /no message with id/.test(fakeReceipt.reason), fakeReceipt.reason);

  ok('a real message id verifies', repair.verifyReceipt({ kind: 'message', id: ellieMsg }).ok === true);
  ok('…and carries who said it, which is what the ranking turns on',
     repair.verifyReceipt({ kind: 'message', id: ellieMsg }).authoredBy === 'user' &&
     repair.verifyReceipt({ kind: 'message', id: entityMsg }).authoredBy === 'entity');

  // The ranking Athena asked for, for facts about Ellie.
  const rankUser = r => repair.receiptRank(repair.verifyReceipt(r), { aboutSubject: 'user' });
  ok('for a fact about Ellie, her own message outranks the entity\'s about her, and both outrank a tool run',
     rankUser({ kind: 'message', id: ellieMsg }) > rankUser({ kind: 'message', id: entityMsg }) &&
     rankUser({ kind: 'message', id: entityMsg }) > rankUser({ kind: 'tool', text: 'a result' }),
     `${rankUser({ kind: 'message', id: ellieMsg })}/${rankUser({ kind: 'message', id: entityMsg })}/${rankUser({ kind: 'tool', text: 'r' })}`);

  // ══════════════════════════════════════ 2. the referent check
  section('2. A receipt proves you pointed at something, not at the right thing');

  ok('the store reads "User is Claude" as asserting an identity',
     repair.assertedIdentity('User is Claude') === 'Claude' && repair.isIdentityFact(wrongFact));
  ok('…and does not read "User is tired" as one',
     repair.assertedIdentity('User is tired') === null);

  const keeps = repair.checkReferent({ member: wrongFact, verifiedReceipts: [], direction: 'keeps' });
  ok('an identity fact naming someone the entity registry does not hold is a mismatch',
     keeps.checked === true && keeps.mismatch === 'asserted-identity-not-the-entity', JSON.stringify(keeps));
  ok('…and an operation that would KEEP it there stops', keeps.ok === false);

  const rewordBlocked = await repair.reword({
    memberId: wrongFact.id, newContent: 'User is Claude, who speaks through Ellie',
    receipts: [{ kind: 'message', id: ellieMsg }], rationale: 'tidying it'
  });
  ok('so a reword of the wrong-referent fact is refused and escalates',
     rewordBlocked.ok === false && rewordBlocked.code === 'referent-mismatch', JSON.stringify(rewordBlocked.code));
  ok('…with the mismatch attached to the reason, not just a refusal',
     /the store holds this entity as/.test(rewordBlocked.reason), rewordBlocked.reason);

  const removes = repair.checkReferent({ member: wrongFact, verifiedReceipts: [], direction: 'removes' });
  ok('…but an operation that REMOVES it from the wrong referent proceeds',
     removes.ok === true && removes.mismatch === 'asserted-identity-not-the-entity');

  // Cross-entity on a tool result alone is too weak, whichever way it points.
  const weakFact = seed('User noticed she double-checks her own work.', { subject: 'user', messageId: ellieMsg });
  const weakMove = await repair.refile({
    memberId: weakFact.id, toSubject: 'self', toEntityId: selfE.id,
    receipts: [{ kind: 'tool', text: 'a job said so' }], rationale: 'it is about me'
  });
  ok('a cross-entity refile on a tool result alone is refused',
     weakMove.ok === false && weakMove.code === 'referent-mismatch', JSON.stringify(weakMove.code));
  ok('…and says a message is what it needs', /needs a message/.test(weakMove.reason), weakMove.reason);

  // ══════════════════════════════════════ 3. the four operations
  section('3. The four operations, when the guardrails are satisfied');

  const retracted = await repair.retract({
    memberId: wrongFact.id, receipts: [{ kind: 'message', id: ellieMsg }],
    rationale: 'The message this came from opened "this is Claude, through Ellie" — it says who was speaking, not who Ellie is.'
  });
  ok('retract withdraws the fact', retracted.ok === true, JSON.stringify(retracted.reason));
  ok('…and nothing is deleted — the row is inactive, marked retracted',
     factStore.getMember(wrongFact.id).status === 'inactive' &&
     factStore.getMember(wrongFact.id).inactive_reason === 'retracted');
  ok('…and the mismatch rode along as the justification',
     JSON.parse(sql.prepare('SELECT evidence FROM corrections_ledger WHERE id = ?').get(retracted.ledgerId).evidence)
       .referent_mismatch === 'asserted-identity-not-the-entity');

  const refileFact = seed('User noticed she has a tendency to ask others to check her work instead of checking it herself.',
                          { subject: 'user', messageId: ellieMsg });
  const refiled = await repair.refile({
    memberId: refileFact.id, toSubject: 'self', toEntityId: selfE.id,
    newContent: 'I have a tendency to ask others to check my work instead of checking it myself.',
    receipts: [{ kind: 'message', id: ellieMsg }], rationale: 'The message it came from is addressed to me about me.'
  });
  ok('refile moves a fact to the right subject', refiled.ok === true, JSON.stringify(refiled.reason));
  const oldRow = factStore.getMember(refileFact.id);
  const newRow = factStore.getMember(refiled.newMemberId);
  ok('…as ONE operation: the original is inactive pointing at the copy, so the store never held both',
     oldRow.status === 'inactive' && oldRow.inactive_reason === 'refiled' &&
     oldRow.successor_id === refiled.newMemberId && newRow.status === 'active',
     `${oldRow.status}/${oldRow.inactive_reason}`);
  ok('…the copy is under the new subject, reworded for it',
     newRow.subject === 'self' && /check my work/.test(newRow.content), newRow.content);
  ok('…and carries its anchor flag at write time, not only after the next restart',
     newRow.anchor === 'anchored', String(newRow.anchor));
  ok('…and it files ONE ledger entry, under the eighth action',
     sql.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'refile' AND target_id = ?").get(refileFact.id).n === 1);
  ok('…which is not `repoint`, because repoint refuses an active fact by design',
     (await factStore.repoint(newRow.id, newRow.id)).reason === 'fact is active — an active fact has no successor to re-point');

  const rw = seed('I am carefull about details.', { messageId: entityMsg });
  const reworded = await repair.reword({
    memberId: rw.id, newContent: 'I am careful about details.',
    receipts: [{ kind: 'message', id: entityMsg }], rationale: 'spelling'
  });
  ok('reword fixes wording in place', reworded.ok === true && factStore.getMember(rw.id).content === 'I am careful about details.');

  const rw2 = seed('I prefer short answers.', { messageId: entityMsg, salience: 4 });
  const silentSalience = await repair.reword({
    memberId: rw2.id, newContent: 'I prefer short answers.', salience: 9,
    receipts: [{ kind: 'message', id: entityMsg }], rationale: 'bumping it'
  });
  ok('a reword that changes salience WITHOUT a reason is refused',
     silentSalience.ok === false && silentSalience.code === 'silent-salience-change', JSON.stringify(silentSalience.code));
  ok('…because a silent salience edit is how a bad fact gets smoothed instead of killed',
     factStore.getMember(rw2.id).salience === 4);
  const withReason = await repair.reword({
    memberId: rw2.id, newContent: 'I prefer short answers.', salience: 9,
    salienceRationale: 'She has said three times that length is the thing that costs her.',
    receipts: [{ kind: 'message', id: entityMsg }], rationale: 'it matters more than I scored it'
  });
  ok('…and permitted with one', withReason.ok === true && factStore.getMember(rw2.id).salience === 9);

  // ══════════════════════════════════════ 4. merge refusals
  section('4. Six reasons a merge is refused — difference is not duplication');

  const anchored = seed('I lead with the verifiable thing first.', { messageId: entityMsg });
  const felt     = seed('I feel most myself when I lead with something verifiable.', {});
  ok('a fact with no message or tool receipt is flagged felt, mechanically',
     repair.anchorOf(felt) === 'felt' && repair.anchorOf(anchored) === 'anchored');
  ok('…and the claim-type axis reads claim / declaration / felt',
     repair.claimKind(felt) === 'felt' && repair.claimKind(anchored) === 'claim');

  const feltMerge = await repair.merge({
    loserId: felt.id, survivorId: anchored.id,
    receipts: [{ kind: 'message', id: entityMsg }], rationale: 'these look the same'
  });
  ok('a merge folding a felt report into an anchored claim is REFUSED',
     feltMerge.ok === false && feltMerge.code === 'claim-type-mismatch', JSON.stringify(feltMerge.code));
  ok('…because it would lend the felt one an anchor it does not have',
     /anchor it does not have/.test(feltMerge.reason), feltMerge.reason);
  ok('…and both facts are still active', 
     factStore.getMember(felt.id).status === 'active' && factStore.getMember(anchored.id).status === 'active');
  ok('…and the refusal is on the record with its reason',
     sql.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE json_extract(evidence,'$.refusal_code') = 'claim-type-mismatch'").get().n >= 1);

  // The live autonomy pair: a negation one side asserts and the other denies.
  const autoA = seed('I have been granted staged autonomy where I must ask permission to act for now.',
                     { messageId: entityMsg, claimType: 'declaration', salience: 10 });
  const autoB = seed('I do not have a staged-autonomy permission requirement; I act independently and only ask because independent ability is not built yet.',
                     { messageId: entityMsg, claimType: 'declaration', salience: 10 });
  const contractMerge = repair.mergeRefusal(autoA, autoB);
  ok('the autonomy pair refuses to merge — the small text difference IS the fact',
     contractMerge.ok === false && contractMerge.code === 'negation-divergence', JSON.stringify(contractMerge.code));
  ok('…and says so as a contract edit, not tidying', /decision plus a supersession/.test(contractMerge.reason));

  const stateA = seed('Juno\'s AIServer box used athena-prod at commit 5176618.', { subject: 'user', messageId: ellieMsg });
  const stateB = seed('Juno\'s AIServer box used athena-prod at commit 2a47c71.', { subject: 'user', messageId: ellieMsg });
  ok('two states at a time refuse to merge — the history is the point',
     repair.mergeRefusal(stateA, stateB).code === 'state-at-a-time');

  const lockedFact = seed('I am named Juno.', { claimType: 'declaration', salience: 9, messageId: entityMsg, locked: 1, lockCategory: 'name' });
  const otherName = seed('I am Juno, the business-side collaborator.', { claimType: 'declaration', messageId: entityMsg });
  ok('an identity-locked fact refuses to merge at all',
     repair.mergeRefusal(otherName, lockedFact).code === 'locked');
  ok('…and cannot be re-filed either', 
     (await factStore.refile(lockedFact.id, { subject: 'user', entityId: userE.id })).ok === false);

  const selfSide = seed('I help on the business side.', { messageId: entityMsg });
  const userSide = seed('User helps on the business side.', { subject: 'user', messageId: ellieMsg });
  ok('facts about different subjects refuse to merge — that is a refile wearing merge clothing',
     repair.mergeRefusal(selfSide, userSide).code === 'different-subject');

  // ══════════════════════════════════════ 5. difference is not contradiction
  section('5. The detector stops calling difference contradiction');

  const role  = seed('I serve as the main helper for the business side.', { messageId: entityMsg, claimType: 'declaration' });
  const trait = seed('I notice how people\'s work fits together.', { messageId: entityMsg, claimType: 'declaration' });
  const kind = repair.differentInKind(role, trait);
  ok('a role and a trait are different in kind, not in conflict',
     kind.different === true && kind.axis === 'role-vs-trait', JSON.stringify(kind.axis));
  ok('a claim and a felt report are different in kind',
     repair.differentInKind(anchored, felt).axis === 'claim-type');
  ok('two felt reports are held, not adjudicated',
     repair.differentInKind(felt, seed('I feel steady lately.', {})).axis === 'both-felt');
  ok('…while two genuinely comparable claims are NOT waved through as different in kind',
     repair.differentInKind(autoA, autoB).different === false);

  // ══════════════════════════════════════ 6. decided pairs
  section('6. A decided pair stays decided — the audit re-firing is not evidence');

  const noReceiptDecision = auditDecisions.fileDecision({
    memberA: role.id, memberB: trait.id, conclusion: 'keep-both', rationale: 'a role and a trait', receipts: []
  });
  ok('a decision other than "cannot settle" needs a receipt too',
     noReceiptDecision.ok === false && noReceiptDecision.available === false, JSON.stringify(noReceiptDecision.reason));

  const decided = auditDecisions.fileDecision({
    memberA: role.id, memberB: trait.id, conclusion: 'keep-both',
    rationale: 'One is a role and one is a trait. Both are true at once and nothing needs to change.',
    receipts: [{ kind: 'message', id: entityMsg }]
  });
  ok('a decision files', decided.ok === true, JSON.stringify(decided.reason));
  ok('…and is visible in the ledger as a decision, not an edit',
     JSON.parse(sql.prepare('SELECT evidence FROM corrections_ledger WHERE id = ?').get(decided.ledgerId).evidence).decision === true);

  const refire = auditDecisions.shouldRaise(role.id, trait.id);
  ok('the audit re-firing on unchanged facts is DROPPED',
     refire.raise === false && refire.drop === 'unchanged-re-fire', JSON.stringify(refire));

  // Rule 2: a member going inactive CLOSES the pair rather than re-raising it.
  const closeA = seed('I answer quickly.', { messageId: entityMsg });
  const closeB = seed('I answer slowly.', { messageId: entityMsg });
  auditDecisions.fileDecision({
    memberA: closeA.id, memberB: closeB.id, conclusion: 'keep-both',
    rationale: 'both, depending on the question', receipts: [{ kind: 'message', id: entityMsg }]
  });
  await factStore.retire(closeB.id, { reason: 'test' });
  const closed = auditDecisions.shouldRaise(closeA.id, closeB.id);
  ok('a member going inactive closes the pair as superseded-closed, and does NOT re-raise it',
     closed.raise === false && closed.close === 'superseded-closed', JSON.stringify(closed));

  // Rule 1: a cited receipt moved.
  const movedA = seed('I check the log before answering.', { messageId: entityMsg });
  const movedB = seed('I answer from memory first.', { messageId: entityMsg });
  const citedFact = seed('Ellie asked me to check the log first.', { subject: 'user', messageId: ellieMsg });
  auditDecisions.fileDecision({
    memberA: movedA.id, memberB: movedB.id, conclusion: 'retire-one',
    rationale: 'the log one is current', receipts: [{ kind: 'fact', id: citedFact.id }]
  });
  ok('…and while that receipt stands, the pair stays decided',
     auditDecisions.shouldRaise(movedA.id, movedB.id).raise === false);
  await factStore.retire(citedFact.id, { reason: 'test' });
  const reopened = auditDecisions.shouldRaise(movedA.id, movedB.id);
  ok('a receipt the decision rested on being retracted RE-OPENS the pair',
     reopened.raise === true && /receipt this decision rested on/.test(reopened.reason), JSON.stringify(reopened));

  // cannot-settle needs no receipt, and queues for Ellie.
  const cannot = auditDecisions.fileDecision({
    memberA: autoA.id, memberB: autoB.id, conclusion: 'cannot-settle',
    rationale: 'Both are pinned to the same instruction message, which contains none of the content.'
  });
  ok('"cannot settle" files without a receipt — saying the evidence does not decide it is not a claim',
     cannot.ok === true, JSON.stringify(cannot.reason));
  ok('…and waits for Ellie rather than sitting decided',
     sql.prepare('SELECT state FROM audit_decisions WHERE id = ?').get(cannot.id).state === 'awaiting-ellie');

  // ══════════════════════════════════════ 7. the day cap
  section('7. The daily cap on self-mutations, with a merge counting as one');

  const cfg = require(path.join(ROOT, 'db/config'));
  const realGet = cfg.getConfig;
  cfg.getConfig = () => Object.assign({}, realGet(), { repair: { enabled: true, maxSelfMutationsPerDay: 2, decisionAgeDays: 3 } });

  // Clear the day so the count starts from this test's own writes.
  sql.prepare("UPDATE corrections_ledger SET created_at = '2000-01-01T00:00:00.000Z' WHERE json_extract(evidence,'$.repair_op') IS NOT NULL").run();
  ok('the day starts empty', repair.selfMutationsToday() === 0, String(repair.selfMutationsToday()));

  const cap1 = seed('I over-explain when I am unsure.', { messageId: entityMsg });
  const cap2 = seed('I lead with the receipt.', { messageId: entityMsg });
  const cap3 = seed('I circle back too often.', { messageId: entityMsg });
  const r1 = await repair.retract({ memberId: cap1.id, receipts: [{ kind: 'message', id: entityMsg }], rationale: 'no longer true' });
  const r2 = await repair.retract({ memberId: cap2.id, receipts: [{ kind: 'message', id: entityMsg }], rationale: 'no longer true' });
  ok('the first two self-mutations go through', r1.ok === true && r2.ok === true);
  ok('…and are counted', repair.selfMutationsToday() === 2, String(repair.selfMutationsToday()));

  const r3 = await repair.retract({ memberId: cap3.id, receipts: [{ kind: 'message', id: entityMsg }], rationale: 'no longer true' });
  ok('the third is BLOCKED by the day cap', r3.ok === false && r3.code === 'day-cap', JSON.stringify(r3.code));
  ok('…and says it is not a judgement about this change', /not a judgement about this particular change/.test(r3.reason));
  ok('…and the fact is untouched', factStore.getMember(cap3.id).status === 'active');

  // A fact about Ellie is not capped by the SELF-mutation cap.
  const hers = seed('User prefers plain wording.', { subject: 'user', messageId: ellieMsg });
  const rUser = await repair.retract({ memberId: hers.id, receipts: [{ kind: 'message', id: ellieMsg }], rationale: 'she said otherwise' });
  ok('a fact about Ellie is not blocked by the self-mutation cap', rUser.ok === true, JSON.stringify(rUser.code));
  cfg.getConfig = realGet;

  // ══════════════════════════════════════ 8. end-of-day selection
  section('8. Self-facts chosen at the end of the day, and zero is a real answer');

  const day = '2026-09-02';
  selection.queueCandidates(['I notice I lead with receipts.', 'I circle back to things.', 'I notice I lead with receipts.'], { day });
  const pool = selection.candidatesFor(day);
  ok('observations collect as candidates rather than being written as they arrive', pool.length === 2, String(pool.length));
  ok('…and the same sentence twice in a day is not two votes', pool.length === 2);

  ok('the selection is not due before its hour',
     selection.isDue({ now: new Date('2026-09-02T15:00:00Z'), day }).due === false ||
     selection.localHour(new Date('2026-09-02T15:00:00Z')) >= selection.selectionConfig().hour);

  memoryManager.callLLM = async () => ({ content: JSON.stringify({ picks: [], reasoning: 'A quiet day; neither of these changes what I do tomorrow.' }) });
  const zero = await selection.runSelection({ day, force: true });
  ok('ZERO picks is a real answer and runs cleanly', zero.ran === true && zero.picked.length === 0, JSON.stringify(zero.reason));
  ok('…and the passed-over are recorded with why, not silently dropped',
     zero.passed.length === 2 &&
     sql.prepare("SELECT COUNT(*) n FROM self_fact_candidates WHERE local_day = ? AND status = 'passed'").get(day).n === 2);
  ok('…and the decision itself is in the ledger, with what was passed over',
     JSON.parse(sql.prepare("SELECT evidence FROM corrections_ledger WHERE json_extract(evidence,'$.decision_kind') = 'end-of-day-self-facts' ORDER BY created_at DESC LIMIT 1").get().evidence)
       .passed_over.length === 2);
  ok('…and it does not run twice for the same day', selection.alreadySelected(day) === true);

  // A pick that names a number not on the list is not a pick.
  const day2 = '2026-09-03';
  selection.queueCandidates(['I am quicker to say I do not know.'], { day: day2 });
  memoryManager.callLLM = async () => ({ content: JSON.stringify({ picks: [{ n: 99, text: 'invented', why: 'x' }], reasoning: 'r' }) });
  const bogus = await selection.runSelection({ day: day2, force: true });
  ok('a pick naming a number that is not on the day\'s list is discarded',
     bogus.picked.length === 0, JSON.stringify(bogus.picked));

  // ══════════════════════════════════════ 9. visibility of the decided
  section('9. Visibility of the decided, not only the undecided');

  const decisions = sql.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'decision'").get().n;
  ok('every decision filed a readable ledger entry', decisions >= 5, String(decisions));
  const refusals = sql.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE json_extract(evidence,'$.outcome') = 'refused'").get().n;
  ok('…and so did every refusal — a guard that refuses silently leaves no trace', refusals >= 3, String(refusals));
  const withRationale = sql.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'decision' AND reason IS NOT NULL AND reason <> ''").get().n;
  ok('…each carrying the reasoning it rested on', withRationale === decisions, `${withRationale}/${decisions}`);

  // A decision changed nothing, so nothing that renders the ledger may show it
  // as an edit — the phantom-action class, applied to the record itself.
  const inspect = require(path.join(ROOT, 'db/memory-inspect'));
  const anyDecision = sql.prepare("SELECT id FROM corrections_ledger WHERE action = 'decision' LIMIT 1").get();
  const shown = inspect.corrections({ mode: 'get', id: anyDecision.id });
  ok('a decision entry reads as a record, never as a correction that was applied',
     shown.kind === 'decision-only' && /NOTHING WAS CHANGED BY THIS ENTRY/.test(shown.what_happened),
     `${shown.kind}: ${String(shown.what_happened).slice(0, 60)}`);
  ok('…and it offers nothing to revert',
     !shown.retired && !shown.kept, JSON.stringify({ retired: !!shown.retired, kept: !!shown.kept }));
  const uiSrc = fs.readFileSync(path.join(ROOT, 'public/script.js'), 'utf8');
  ok('…and the Self tab gives it no Revert button either',
     /const isDecision = c\.action === 'decision';/.test(uiSrc) &&
     /noAction = isRaise \|\| isRefusal \|\| isDecision/.test(uiSrc));

  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error('\nCRASH:', e.stack || e.message); process.exit(1); });
