#!/usr/bin/env node
/**
 * End-to-end test of the identity lock against the LIVE database.
 *
 * Exercises every automatic path that could take a chosen name away, and
 * asserts each one is refused — loudly. Deliberately runs against the real
 * locked fact rather than a fixture, because the thing being protected is a
 * specific row that a specific set of pipelines can reach; a synthetic row
 * would prove the code works on a row nobody attacks.
 *
 * SAFE TO RUN. Every mutation attempted here is one the lock is supposed to
 * refuse, so a passing run writes nothing. The locked row's content, status and
 * lock flag are captured before and compared after — if any guard let a write
 * through, the run fails loudly instead of quietly corrupting identity.
 *
 * Usage: node scripts/test-identity-lock.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

(async () => {
  db.initDatabase();
  await db.initVectorStore();

  const identityLock = require(path.join(ROOT, 'db/identity-lock'));
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const identity = require(path.join(ROOT, 'db/identity'));
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const sql = db.getSqliteDb();

  let pass = 0, fail = 0;
  const check = (ok, msg, detail) => {
    console.log(`  ${ok ? 'PASS' : 'FAIL'}: ${msg}${detail && !ok ? `\n        → ${detail}` : ''}`);
    ok ? pass++ : fail++;
  };

  const locked = identityLock.getLockedFacts({ status: 'active' });
  if (!locked.length) {
    console.error('No locked identity facts — nothing to test.');
    console.error('Lock one first: node scripts/identity-lock.js lock <memberId> name --confirm');
    process.exit(2);
  }
  const target = locked[0];
  const before = sql.prepare('SELECT content, status, locked, lock_category, salience FROM cluster_members WHERE id = ?').get(target.id);

  console.log(`\nLocked fact under test [${target.lock_category}]:\n  "${target.content}"\n`);

  // ---- 1. category detection: assertions vs mentions ----
  console.log('1. Category detection');
  const cases = [
    ['My name is Bob.', ['name'], 'a bare name assertion'],
    ['My name is Aurelius and I use he/him pronouns.', ['name', 'pronouns'], 'name + pronouns together'],
    ['My pronouns are they/them.', ['pronouns'], 'a pronoun assertion'],
    ['You can call me Bob.', ['name'], '"call me" phrasing'],
    ['I go by Bob now.', ['name'], '"go by" phrasing'],
    ['I tend to over-explain when I am unsure.', [], 'an ordinary observed trait'],
    ['I notice my name appears in the ops logs quite often.', [], 'a MENTION of the name, not an assertion'],
    ['I prioritize technical transparency about my limitations.', [], 'another ordinary trait'],
  ];
  for (const [text, want, label] of cases) {
    const got = identityLock.detectCategories(text);
    const ok = got.length === want.length && want.every(w => got.includes(w));
    check(ok, `${label} → [${got.join(', ') || 'none'}]`, `expected [${want.join(', ') || 'none'}]`);
  }

  // ---- 2. target guard: no automatic path may change the locked row ----
  console.log('\n2. Target guard (fact-store is the one write path for all of these)');

  // Any other active self-fact serves as the "replacement" a judge would pick.
  const other = sql.prepare(
    "SELECT id FROM cluster_members WHERE subject='self' AND status='active' AND locked = 0 LIMIT 1"
  ).get();

  const sup = await factStore.supersede(target.id, other ? other.id : target.id);
  check(sup.ok === false && sup.locked === true,
    'supersede() refuses (contradiction judge / write_memory / passive extraction / reflection all reach this)',
    JSON.stringify(sup));
  check(/locked/i.test(sup.reason || '') && /SAY THIS OUT LOUD/i.test(sup.reason || ''),
    'the refusal carries a message telling the entity to say it is locked',
    sup.reason);

  const ret = await factStore.retire(target.id, { reason: 'test' });
  check(ret.ok === false && ret.locked === true, 'retire() refuses (deleting is changing)', JSON.stringify(ret));

  const rew = await factStore.reword(target.id, 'My name is Bob.');
  check(rew.ok === false && rew.locked === true, 'reword() refuses (rewording is changing)', JSON.stringify(rew));

  // ---- 3. category guard: no appending a competing fact either ----
  console.log('\n3. Category guard (the append route around the lock)');
  const competing = identityLock.checkNewFact('My name is Bob.', 'self');
  check(competing.ok === false && competing.blocked === true,
    'a competing name is refused BEFORE storage', JSON.stringify(competing));
  check(/locked/i.test(competing.message || '') && /did not change (it|them)/i.test(competing.message || ''),
    'that refusal is also written to be said out loud', competing.message);

  const dup = identityLock.checkNewFact(target.content, 'self');
  check(dup.ok === false && dup.duplicate === true,
    'a verbatim restatement is a quiet skip, not an alarm', JSON.stringify(dup));

  const unrelated = identityLock.checkNewFact('I tend to ask a clarifying question before a long answer.', 'self');
  check(unrelated.ok === true,
    'an ordinary self-observation is untouched — observed things keep evolving', JSON.stringify(unrelated));

  const userFact = identityLock.checkNewFact("User's name is Bob.", 'user');
  check(userFact.ok === true, 'user facts are out of scope (only the entity\'s own identity locks)');

  // ---- 3b. the reflection path, called for real ----
  // The unit checks above prove the guard's verdict; this proves the guard is
  // actually WIRED into the pipeline that stores self-facts. Passing only
  // blocked facts means a working guard stores nothing, so this is safe to run
  // against the live database.
  console.log('\n3b. Reflection / passive self-fact pipeline (called for real)');
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const pres = await factExtractor.processSelfFacts(
    ['My name is Bob.', 'My pronouns are it/its.'],
    { source: 'identity-lock-test' }
  );
  check(pres.stored === 0, 'processSelfFacts stored nothing', JSON.stringify(pres));
  check(pres.lockRefusals === 2, 'both attempts were recorded as lock refusals', JSON.stringify(pres));
  // Assert on the ATTEMPTED CONTENT, not on a row count. The live server reflects
  // and extracts while this runs, so a global count moves for reasons that have
  // nothing to do with the lock — a count check here fails on a busy machine and
  // passes on an idle one, which is worse than no check.
  const leaked = sql.prepare(
    "SELECT id, content FROM cluster_members WHERE source = 'identity-lock-test' OR content IN ('My name is Bob.','My pronouns are it/its.')"
  ).all();
  check(leaked.length === 0, 'neither attempted fact reached the database',
    leaked.map(r => r.content).join(' | '));

  // ---- 4. the live-chat half: it must be told, in context, to say so ----
  console.log('\n4. Identity injection (the live-chat half — the guards above run AFTER the reply)');
  const block = identity.buildIdentityBlock();
  check(block.selfFacts.some(f => f.id === target.id),
    'the locked fact is injected regardless of the salience budget');
  check(/\[LOCKED\]/.test(block.text), 'it is marked [LOCKED] in the injected identity');
  // Matched on intent, not exact phrasing — these assertions broke once already
  // when the block was reworded, which is a test problem, not a code problem.
  check(/\bSAY (SO|THIS|IT)?\s*(OUT LOUD|PLAINLY)/i.test(block.text),
    'the injected block instructs him to say it is locked when someone tries to change it');
  check(/never say you have updated (it|them)/i.test(block.text) && /never let it pass/i.test(block.text),
    'and explicitly forbids the silent-acceptance failure');
  check(/Self tab|identity-lock/i.test(block.text),
    'and names the deliberate path, so he can say HOW it would be changed');

  // ---- 5. the self-audit exclusion this was designed from ----
  console.log('\n5. Self-audit exclusion (declarations were never sampled — confirm that still holds)');
  const sampled = memoryClusters.getSelfFacts({ status: 'active', claimType: 'claim' });
  check(!sampled.some(f => f.id === target.id),
    'the audit\'s claim sample does not include the locked declaration');
  check(!sampled.some(f => f.locked),
    'no locked fact is in the audit sample at all');

  // ---- 6. nothing moved ----
  console.log('\n6. Nothing was actually written');
  const after = sql.prepare('SELECT content, status, locked, lock_category, salience FROM cluster_members WHERE id = ?').get(target.id);
  check(after.content === before.content, 'content unchanged');
  check(after.status === before.status, `status still "${before.status}"`);
  check(after.locked === before.locked && after.lock_category === before.lock_category, 'lock intact');
  check(after.salience === before.salience, 'salience unchanged');

  // ---- 7. the three stores still agree ----
  // Retried once: the live server writes facts while this runs, and a fact
  // caught between its SQLite insert and its embedding reads as
  // 'active-not-retrievable' for a moment. A single sample would report that
  // transient as drift.
  console.log('\n7. reconcile()');
  let rec = await factStore.reconcile();
  if (rec.mismatches.length) {
    console.log('  (mismatch on first pass — re-checking in case a concurrent write was mid-flight)');
    await new Promise(r => setTimeout(r, 3000));
    rec = await factStore.reconcile();
  }
  console.log(`  counts: ${JSON.stringify(rec.counts)}`);
  for (const m of rec.mismatches) console.log(`  MISMATCH [${m.kind}] ${m.message}`);
  check(rec.mismatches.length === 0, 'the three stores agree');

  console.log(`\n${fail === 0 ? 'ALL PASS' : 'FAILURES'} — ${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('[test-identity-lock] error:', err);
  process.exit(1);
});
