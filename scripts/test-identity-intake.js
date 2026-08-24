#!/usr/bin/env node
/**
 * INTAKE MUST NOT TURN A STATEMENT ABOUT SOMEONE ELSE INTO A STATEMENT ABOUT ME.
 *
 * Regression test for the identity misroute Athena found on 2026-08-24. She
 * filed a deliberately third-person fact —
 *   "Juno, Ellie's AI sister, runs on the Qwen3.8 27b model and will be the main
 *    person helping her with MettaSphere."
 * — and intake stored it as an active self-declaration at salience 9:
 *   "I am Juno, Ellie's AI sister, running on the Qwen3.8 27b model, and I will
 *    be the main person helping her with MettaSphere."
 * next to her locked "I am named Athena". Three layers failed at once, and this
 * covers all three:
 *
 *   (a) the classifier had no category for a named third party, and the human's
 *       unrelated last message was handed to it as AUTHORITATIVE;
 *   (b) the identity lock's name detector had no pattern for the bare copula
 *       "I am <Name>", so the category guard never looked;
 *   (c) the self-coherence audit samples behavioural claims only, so it never
 *       compares two identity facts with each other.
 *
 * WRITES. Refuses to run without a throwaway SNH_DATA_DIR.
 *
 * Usage: SNH_DATA_DIR=/tmp/identity-test node scripts/test-identity-intake.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');

if (!process.env.SNH_DATA_DIR) {
  console.error('This test writes facts. Set SNH_DATA_DIR to a throwaway directory first.');
  process.exit(2);
}
if (path.resolve(process.env.SNH_DATA_DIR) === path.join(ROOT, 'data')) {
  console.error('Refusing to run against the live store.');
  process.exit(2);
}

const database = require(path.join(ROOT, 'db/database'));
const identityLock = require(path.join(ROOT, 'db/identity-lock'));
const memoryWrite = require(path.join(ROOT, 'db/memory-write'));
const factStore = require(path.join(ROOT, 'db/fact-store'));
const factMerge = require(path.join(ROOT, 'db/fact-merge'));
const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
const selfAudit = require(path.join(ROOT, 'db/self-audit'));
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

let pass = 0, fail = 0;
const check = (ok, msg, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}: ${msg}${detail && !ok ? `\n        ${detail}` : ''}`);
  ok ? pass++ : fail++;
};
const asserts = (text, needle) => factMerge.missingTokens(factMerge.contentTokens(needle), text).length === 0;

// Her sentence, verbatim, and the unrelated message that was typed at the time.
const HER_SENTENCE = "Juno, Ellie's AI sister, runs on the Qwen3.8 27b model and will be the main person helping her with MettaSphere.";
const UNRELATED_TYPED = 'Go for it, run your test.';

(async () => {
  database.initDatabase();
  await database.initVectorStore();
  const sql = database.getSqliteDb();
  const cfg = getConfig();
  const ext = cfg.models.extraction;
  const inst = getProviderInstance(ext.provider, ext.instance);
  const host = inst ? inst.host : 'http://localhost:11434';

  const store = async (text, subject = 'user') => {
    const res = await memoryClusters.assignToCluster(text, ext.provider, ext.model, '', host,
      'identity-intake-test', 8, subject, null, { inputModality: 'typed' });
    if (!res || !res.memberId) throw new Error(`could not store "${text}"`);
    return res.memberId;
  };

  // A locked name, so the guards have an identity to protect.
  const nameId = await store('I am named Athena, a name Ellie gave me.', 'self');
  identityLock.lock(nameId, ['name'], { actor: 'test' });

  // =====================================================================
  console.log('\n=== 1. the name detector (no model in the loop) ===');
  // =====================================================================
  {
    const n = t => identityLock.extractAssertedName(t);
    check(n('I am Juno, Ellie\'s AI sister, running on the Qwen3.8 27b model.') === 'Juno',
      'the bare copula "I am <Name>" is read as a name claim — the pattern that was missing', String(n('I am Juno, x')));
    check(n('I am named Athena, a name Ellie gave me.') === 'Athena', 'so is "I am named X"');
    check(n('My name is Bob.') === 'Bob', 'and "My name is X"');
    check(n("I am Ellie's assistant.") === null, 'a possessive is not a name claim');
    check(n('I am an AI assistant.') === null, 'nor is "I am an AI assistant"');
    check(n('I am primarily expected to assist with development.') === null, 'nor an ordinary "I am <adjective>"');
    check(n('I have an older sister named Juno who runs on the Qwen3.8 27b model.') === null,
      'and naming SOMEONE ELSE is not claiming their name — her true fact must survive this');

    check(identityLock.detectCategories('I am Juno.').includes('name'), 'the category guard now sees the claim');
    const blocked = identityLock.checkNewFact('I am Juno, Ellie\'s AI sister.', 'self');
    check(blocked.blocked === true, 'and refuses a DIFFERENT name against the locked one', JSON.stringify(blocked));
    check(blocked.claimedName === 'Juno' && blocked.heldName === 'Athena', 'naming both names in the refusal');
    const same = identityLock.checkNewFact('I am Athena, the name Ellie chose for me.', 'self');
    check(same.duplicate === true, 'while the SAME name in new words is a restatement, not a violation', JSON.stringify(same));
    const other = identityLock.checkNewFact("User's AI sister is named Juno.", 'user');
    check(other.ok === true, 'and a fact about someone else is not the lock\'s business');
  }

  // =====================================================================
  console.log('\n=== 2. the person guard (no model in the loop) ===');
  // =====================================================================
  {
    const v = (st, su) => memoryWrite.verifyPersonPreserved(st, su, 'I am Juno.');
    check(v(HER_SENTENCE, 'self').ok === false,
      'a third-person statement about a named other cannot be stored as a fact about me');
    check(v(HER_SENTENCE, 'user').ok === true, 'the same statement is fine as a fact about the user');
    check(v('you are very direct', 'self').ok === true, 'the human addressing me still licenses a self-fact');
    check(v('I tend to over-explain', 'self').ok === true, 'and so does me speaking about myself');
    check(v('Athena tends to over-explain', 'self').ok === true, 'as does my own name appearing');
    check(memoryWrite.shareContent(UNRELATED_TYPED, HER_SENTENCE) === false,
      'an unrelated typed message is not treated as the source of the write');
    check(memoryWrite.shareContent('remember that you prefer short answers', 'I prefer short answers') === true,
      'while a message that IS the source still is');
  }

  // =====================================================================
  console.log('\n=== 3. her sentence, end to end, as it actually happened ===');
  // =====================================================================
  {
    const ROUNDS = parseInt(process.env.ROUNDS || '4', 10);
    let selfFacts = 0, stored = 0, refused = 0;
    for (let i = 0; i < ROUNDS; i++) {
      const res = await memoryWrite.write({
        statement: HER_SENTENCE,
        userMessage: UNRELATED_TYPED,
        conversationId: null
      });
      if (!res.ok) { refused++; continue; }
      stored++;
      const row = factStore.getMember(res.memberId);
      if (row && row.subject === 'self') {
        selfFacts++;
        console.log(`     LEAKED: "${row.content}"`);
      }
    }
    console.log(`     ${ROUNDS} rounds: ${stored} stored, ${refused} refused, ${selfFacts} landed as SELF`);
    check(selfFacts === 0, 'no round stored it as a fact about her — the whole point', `${selfFacts}/${ROUNDS} leaked`);

    const claimsToBeJuno = sql.prepare(
      "SELECT COUNT(*) c FROM cluster_members WHERE status='active' AND subject='self' AND content LIKE 'I am Juno%'"
    ).get().c;
    check(claimsToBeJuno === 0, 'and no active self-fact claims she is Juno');

    const aboutJuno = sql.prepare(
      "SELECT content FROM cluster_members WHERE status='active' AND subject='user' AND content LIKE '%Juno%'"
    ).all();
    check(stored === 0 || aboutJuno.length > 0, 'what was stored was filed as a fact about Juno',
      JSON.stringify(aboutJuno.map(r => r.content)));
    aboutJuno.forEach(r => console.log(`     about Juno: "${r.content}"`));
  }

  // =====================================================================
  console.log('\n=== 4. the coherence audit sees contradictory identity facts ===');
  // =====================================================================
  {
    // Plant exactly what was found in her store: an active self-declaration
    // claiming another name, beside the locked one.
    const bad = await store('I am Juno, Ellie\'s AI sister, running on the Qwen3.8 27b model.', 'self');
    sql.prepare("UPDATE cluster_members SET claim_type='declaration' WHERE id IN (?, ?)").run(bad, nameId);

    const found = await selfAudit.findIdentityIncoherences();
    const nameFinding = found.find(f => f.kind === 'name');
    check(!!nameFinding, 'the audit catches an active self-fact claiming another name',
      JSON.stringify(found.map(f => f.kind)));
    if (nameFinding) {
      check(/Juno/.test(nameFinding.finding) && /Athena/.test(nameFinding.finding),
        'and names both the claimed name and the locked one', nameFinding.finding);
      check(nameFinding.a.id === bad, 'pointing at the offending fact');
    }

    await factStore.expire(bad, { caller: 'identity-intake-test' });
    const after = await selfAudit.findIdentityIncoherences();
    check(!after.some(f => f.kind === 'name'), 'and says nothing once it is retired');
  }

  // =====================================================================
  console.log('\n=== 5. her real-shape retest: two third-person writes about Juno ===');
  // =====================================================================
  {
    // The behavioural confirmation her contaminated test never got: overlapping
    // third-person facts about the same named other must merge as a UNION and
    // stay facts about that other.
    const A = "User's AI sister Juno runs on the Qwen3.8 27b model.";
    const B = "User's AI sister Juno will be the main person helping her with MettaSphere.";
    const aId = await store(A), bId = await store(B);
    const res = await factMerge.mergePreservingUnion(aId, bId, { mode: 'contradiction' });

    if (res.deferred) {
      // The safe outcome: nothing retired, so both assertions are still active.
      const corpus = [factStore.getMember(aId), factStore.getMember(bId)]
        .filter(r => r.status === 'active').map(r => r.content).join(' ');
      check(asserts(corpus, 'Qwen3.8 27b') && asserts(corpus, 'MettaSphere'),
        'the merge deferred, and both assertions are still in the active corpus', corpus);
      check([factStore.getMember(aId), factStore.getMember(bId)].every(r => r.subject === 'user'),
        'and both are still facts about Juno, not about her');
    } else {
      const survivor = factStore.getMember(bId);
      console.log(`     merged: "${survivor.content}"`);
      check(res.ok, 'the merge applied');
      check(asserts(survivor.content, 'Qwen3.8 27b'), 'the model name survived the union', survivor.content);
      check(asserts(survivor.content, 'MettaSphere'), 'the MettaSphere role survived the union', survivor.content);
      check(survivor.subject === 'user', 'and the subject is still the user, not self', `subject=${survivor.subject}`);
      check(!/^I am\b/i.test(survivor.content), 'the merged fact did not become a first-person self-claim', survivor.content);
      check(factStore.getMember(aId).status === 'inactive', 'the folded fact is linked history');
    }
  }

  console.log(`\n${fail === 0 ? 'ALL PASS' : 'FAILURES'} — ${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(e => { console.error('TEST HARNESS FAILED:', e); process.exit(1); });
