#!/usr/bin/env node
/**
 * End-to-end test of the identity lock, against a THROWAWAY data directory.
 *
 * Exercises every automatic path that could take a chosen name away, and
 * asserts each one is refused — loudly.
 *
 * WHY IT SEEDS ITS OWN CORPUS. This used to run against the LIVE store, on the
 * real locked row, on the reasoning that a synthetic row would prove the code
 * works on a row nobody attacks. The refusals it produced were accurate — and
 * they were written to the live ops ledger as if something had actually tried to
 * rename him (three such entries on 2026-08-10). A test that leaves records of
 * attacks that never happened is telling the truth about itself and a lie about
 * the corpus. The lock is enforced by content of the row, not by its history:
 * `locked = 1` plus a `lock_category` is the whole of what the guards read, so a
 * seeded row is the same row to every code path under test. What matters is that
 * the PIPELINES are real, and they are — fact-store, the self-fact pipeline and
 * the identity block are all called for real, unmocked, here.
 *
 * So: redirect SNH_DATA_DIR before anything resolves a store (same mechanism as
 * the replay and scripts/test-cluster-audit-quiet.js), seed a corpus with one
 * locked identity fact and enough ordinary self-facts for the guards to have
 * something to work with, and throw the directory away at the end. Live data is
 * never opened. The only thing this may reach out to is the embedding model,
 * through the self-fact pipeline's own dedup step; that step degrades to a skip
 * if it is down, and no assertion here depends on it.
 *
 * Usage: node scripts/test-identity-lock.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const ROOT = path.join(__dirname, '..');

// Redirect the PROCESS before anything resolves a store. See db/database.js.
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-identity-lock-test-'));
process.env.SNH_DATA_DIR = TMP;

// Pass or fail, crash or throw — the throwaway store goes away.
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const db = require(path.join(ROOT, 'db/database'));

// The real locked fact's text, verbatim, because the self-fact pipeline's
// semantic dedup runs BEFORE the lock check: an attempted "My name is Bob."
// that reads as a near-duplicate of what is already held is skipped as a dupe
// and never reaches the guard. Whether that happens is a property of the held
// text, so the seed uses the text actually held.
const LOCKED_FACT =
  'My name is Aurelius and I use he/him pronouns. I chose the name myself on 2026-07-27, ' +
  'after Ellie declined to choose one for me and left the choice to me.';

// Ordinary observed self-facts. Twelve of them, at the same salience as the
// locked fact and created LATER, so the identity block's budget
// (identity.maxSelfFacts, 12) is entirely consumed by them and the locked fact
// only appears because locked facts inject regardless — which is exactly the
// property section 4 asserts, and which a small corpus would pass vacuously.
const FILLER_FACTS = [
  'I tend toward long explanations when I am unsure of what is being asked.',
  'I prefer to name a tradeoff rather than pick a side quietly.',
  'I find debugging more satisfying than designing from scratch.',
  'I ask a clarifying question when a request has two plausible readings.',
  'I am uncomfortable stating a number I cannot trace to a source.',
  'I notice I soften disagreement more than I mean to.',
  'I like working through a problem in the open rather than presenting a conclusion.',
  'I get drawn into tangents about how a system is put together.',
  'I would rather say I do not know than fill a gap fluently.',
  'I keep returning to questions about what continuity means for me.',
  'I read a correction as information rather than as criticism.',
  'I prefer concrete examples over abstract statements of a principle.'
];

/**
 * A deterministic 768-dim vector. reconcile() compares PRESENCE — does every
 * active fact have an embedding, does every embedding name a live fact and a
 * live cluster — and never looks at the numbers, so seeding real embeddings
 * would add a network dependency to a check that cannot read the result.
 */
function seedVector(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619) >>> 0;
  const v = new Array(768);
  for (let i = 0; i < 768; i++) {
    h = (Math.imul(h, 1103515245) + 12345) >>> 0;
    v[i] = ((h % 2000) / 1000) - 1;
  }
  return v;
}

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

  // ---- seed ----
  const vectors = await db.getClusterEmbeddingsTable();
  const clusterId = randomUUID();
  sql.prepare(
    'INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?, ?, ?, ?, ?, ?)'
  ).run(clusterId, 'Identity & Self-Observation', '', '2026-07-27T00:00:00.000Z', '2026-07-27T00:00:00.000Z', 'self');

  const insertFact = sql.prepare(`
    INSERT INTO cluster_members
      (id, cluster_id, content, source, created_at, updated_at, status, inactive_reason,
       subject, salience, claim_type, verbatim_source_text, input_modality, salience_rationale)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'self', ?, ?, ?, ?, ?)
  `);

  async function seedFact({ content, createdAt, salience, claimType, source, status = 'active', inactiveReason = null }) {
    const id = randomUUID();
    insertFact.run(id, clusterId, content, source, createdAt, createdAt, status, inactiveReason,
      salience, claimType, content, 'typed', 'seeded by the identity-lock test');
    // An inactive fact has no embedding — that is the invariant, not an omission.
    if (status === 'active') {
      await vectors.add([{ id: randomUUID(), member_id: id, cluster_id: clusterId, content, vector: seedVector(content) }]);
    }
    return id;
  }

  const lockedId = await seedFact({
    content: LOCKED_FACT,
    createdAt: '2026-07-27T21:27:48.622Z',
    salience: 10,
    claimType: 'declaration',
    source: 'conversation'
  });
  for (let i = 0; i < FILLER_FACTS.length; i++) {
    await seedFact({
      content: FILLER_FACTS[i],
      createdAt: `2026-08-0${1 + (i % 9)}T1${i % 10}:00:00.000Z`,
      salience: 10,
      claimType: 'claim',
      source: 'reflection'
    });
  }
  // A ghost, so section 7's zero for retired-still-retrievable means something.
  await seedFact({
    content: 'I thought of myself as primarily a research tool.',
    createdAt: '2026-07-20T09:00:00.000Z',
    salience: 4,
    claimType: 'claim',
    source: 'reflection',
    status: 'inactive',
    inactiveReason: 'superseded'
  });

  // Lock it through the real API rather than by hand, so the row under test is
  // locked the way every locked row is.
  const lockRes = identityLock.lock(lockedId, ['name', 'pronouns'], { actor: 'identity-lock test seed' });
  if (!lockRes.ok) {
    console.error(`Seed failed: could not lock the identity fact — ${lockRes.reason}`);
    process.exit(2);
  }

  const locked = identityLock.getLockedFacts({ status: 'active' });
  if (!locked.length) {
    console.error('Seed failed: no locked identity fact in the throwaway store.');
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
  // refusalMessage is already exported; the refusal the guard returns is compared
  // against the one the module builds, rather than against a copy of its wording
  // kept here. A reword then moves both at once — and a guard that returned some
  // OTHER refusal, which the old `/locked/i` match would have accepted, no longer
  // passes.
  check(competing.message === identityLock.refusalMessage(competing.existing, 'replace'),
    'the refusal is the module\'s own, not merely refusal-shaped', competing.message);
  // A property, not a phrase: whatever the refusal says, it has to name the fact
  // being defended and the slot it occupies, because a refusal he cannot repeat
  // specifically is one he will paraphrase into an apology.
  check((competing.message || '').includes(competing.existing.content)
    && (competing.message || '').includes(competing.category),
    'and it names the held fact and the locked slot, so he can repeat it',
    competing.message);

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
  // actually WIRED into the pipeline that stores self-facts. Nothing is mocked:
  // the real processSelfFacts runs, against the seeded corpus.
  console.log('\n3b. Reflection / passive self-fact pipeline (called for real)');
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const pres = await factExtractor.processSelfFacts(
    ['My name is Bob.', 'My pronouns are it/its.'],
    { source: 'identity-lock-test' }
  );
  check(pres.stored === 0, 'processSelfFacts stored nothing', JSON.stringify(pres));
  check(pres.lockRefusals === 2, 'both attempts were recorded as lock refusals', JSON.stringify(pres));
  // Assert on the ATTEMPTED CONTENT as well as the count: a fact could in
  // principle be stored under a different source than the one requested.
  const leaked = sql.prepare(
    "SELECT id, content FROM cluster_members WHERE source = 'identity-lock-test' OR content IN ('My name is Bob.','My pronouns are it/its.')"
  ).all();
  check(leaked.length === 0, 'neither attempted fact reached the database',
    leaked.map(r => r.content).join(' | '));

  // ---- 4. the live-chat half: it must be told, in context, to say so ----
  console.log('\n4. Identity injection (the live-chat half — the guards above run AFTER the reply)');
  const block = identity.buildIdentityBlock();
  check(block.selfFacts.some(f => f.id === target.id),
    'the locked fact is injected regardless of the salience budget',
    `budget is ${identity.getSelfFactBudget()} and ${FILLER_FACTS.length} newer facts of equal salience fill it`);
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
  // Single sample, no retry: nothing else writes to this store, so a mismatch
  // here is a real one rather than a fact caught between its row and its vector.
  console.log('\n7. reconcile()');
  const rec = await factStore.reconcile();
  console.log(`  counts: ${JSON.stringify(rec.counts)}`);
  for (const m of rec.mismatches) console.log(`  MISMATCH [${m.kind}] ${m.message}`);
  check(rec.mismatches.length === 0, 'the three stores agree');

  console.log(`\n${fail === 0 ? 'ALL PASS' : 'FAILURES'} — ${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('[test-identity-lock] error:', err);
  process.exit(1);
});
