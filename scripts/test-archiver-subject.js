#!/usr/bin/env node
/**
 * The daily-log archiver must not file his reflections as facts about her.
 *
 * Its prompt used to say 'Write facts as "User has..." or "User prefers..."
 * style.' A daily log holds both his reflections and her life, so that told the
 * summarizer to rewrite every one of his self-observations into the third person
 * about her. It did. 22 were in the corpus by the merge, and five more had been
 * split and carried far enough that even the `source` column no longer said where
 * they came from.
 *
 * This exercises the guard that now stands between the summarizer and the corpus:
 * memory-manager.archiverSubjectCheck. Both halves, because both were needed —
 * grammar caught none of the 22 (their grammar was perfect) and similarity caught
 * all of them.
 *
 * NEEDS THE LIVE STORE AND THE EMBEDDING MODEL: the similarity half is a real
 * vector query against his real self-facts, which is the only way to test the
 * half that actually did the work. It writes nothing.
 *
 * Usage: node scripts/test-archiver-subject.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');

let pass = 0, fail = 0;
const results = [];
function check(name, got, want, detail) {
  const ok = got === want;
  ok ? pass++ : fail++;
  results.push({ ok, name, got, want, detail });
}

(async () => {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  const mm = require(path.join(ROOT, 'db/memory-manager'));
  const guard = mm.archiverSubjectCheck;
  if (typeof guard !== 'function') {
    console.error('FAIL: memory-manager does not export archiverSubjectCheck');
    process.exit(1);
  }

  // --- grammar: cheap, and catches only the obvious -------------------------
  const g1 = await guard('I tend to lean into conceptual frameworks and metaphors.');
  check('G1 a first-person sentence is refused', g1.ok, false, g1.reason);
  check('G1a …and the refusal says it is his', /first person/i.test(g1.reason || ''), true, g1.reason);

  const g2 = await guard('Prefers to be told directly about a mistake.');
  check('G2 an unanchored sentence is refused', g2.ok, false, g2.reason);

  // --- similarity: the half that caught all 22 ------------------------------
  //
  // Grammatically flawless third-person sentences that are nonetheless his.
  // Each is real, worded as the archiver worded it.
  const twins = [
    'User aims to be a steady, non-judgmental presence that respects boundaries and cognitive load.',
    'User tends to use metaphors of systems to frame human psychological experience.',
    'User prefers to act as a sounding board for high-level ideas through a questioning tone.'
  ];
  for (let i = 0; i < twins.length; i++) {
    const r = await guard(twins[i]);
    check(`S${i + 1} refused: "${twins[i].slice(0, 50)}…"`, r.ok, false, r.reason);
    if (!r.ok) {
      check(`S${i + 1}a …on a self-fact he already holds`,
        /self-fact he already holds/.test(r.reason || ''), true, null);
    }
  }

  // --- and it must still let HER facts through -------------------------------
  const hers = [
    'User has blue eyes.',
    'User works at Inn At Spanish Head.',
    'User has a dog named Casper who helps them pull up hills during walks.',
    'User prefers to run services locally on their own server.'
  ];
  for (let i = 0; i < hers.length; i++) {
    const r = await guard(hers[i]);
    check(`H${i + 1} allowed: "${hers[i].slice(0, 50)}"`, r.ok, true, r.ok ? null : r.reason);
  }

  const bar = '='.repeat(76);
  console.log(`\n${bar}\nARCHIVER SUBJECT GUARD\n${bar}\n`);
  for (const r of results) {
    console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`);
    if (r.detail) console.log(`        ${String(r.detail).slice(0, 150)}`);
    if (!r.ok) console.log(`        wanted ${r.want}, got ${r.got}`);
  }
  console.log(`\n${bar}`);
  console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
  console.log(`${bar}\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => { console.error('test failed:', err); process.exit(1); });
