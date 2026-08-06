#!/usr/bin/env node
/**
 * A compound must never lose a contradiction whole.
 *
 * "User (Ellie) has blue eyes and her favorite color is green (last updated
 * 2026-07-27)" was retired for "User's favorite color is blue". The colour was
 * genuinely disputed. The eye colour was not, and it went anyway.
 *
 * Three things had to be true for that to happen, and this tests the two that
 * are pure:
 *
 *   1. The archiver's "(Ellie)" annotation made the splitter emit "User's name
 *      is Ellie." as an atom, which made the corrector abandon the split — the
 *      identity guard doing its job on a sentence that should never have
 *      produced an identity atom. stripSubjectAnnotation removes the annotation
 *      before the split, so the split can proceed.
 *   2. looksCompound is what the corrector now checks before applying any
 *      supersession: a loser that still says several things is raised, not
 *      retired.
 *
 * The third — that the split PHASE completes before the semantic tier runs at
 * all — needs a corpus and a pass, and is verified by the staging run recorded
 * in docs/staging-gate4-report.txt: the compound split into "User has blue eyes."
 * and "User's favorite color is green (last updated 2026-07-27)", and only the
 * colour clause was then superseded.
 *
 * PURE. No database, no model.
 *
 * Usage: node scripts/test-compound-safety.js
 */
const path = require('path');
const rules = require(path.join(__dirname, '..', 'db', 'extraction-rules'));

let pass = 0, fail = 0;
const results = [];
const check = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  ok ? pass++ : fail++;
  results.push({ ok, name, got, want });
};

// --- 1. the annotation, removed -------------------------------------------
const strip = (t) => rules.stripSubjectAnnotation(t);

check('A1 the exact sentence that lost its eye colour',
  strip('User (Ellie) has blue eyes and her favorite color is green (last updated 2026-07-27).').text,
  'User has blue eyes and her favorite color is green (last updated 2026-07-27).');

check('A2 …and it reports what it removed',
  strip('User (Ellie) has blue eyes.').stripped, 'Ellie');

check('A3 the possessive form',
  strip("User (Ellie)'s system architecture contains a schema gap.").text,
  "User's system architecture contains a schema gap.");

check('A4 an unannotated fact is untouched',
  strip('User has blue eyes').text, 'User has blue eyes');

check('A5 …and reports nothing removed',
  strip('User has blue eyes').stripped, null);

// The guard that keeps this narrow: a parenthetical carrying SUBSTANCE is not an
// annotation and must survive. Removing it would lose part of the claim.
check('A6 a substantive parenthetical is not stripped',
  strip("User's system (which runs 24/7) is on Sparky").text,
  "User's system (which runs 24/7) is on Sparky");

check('A7 a parenthetical that is not right after the subject is not stripped',
  strip('User has a dog (Casper) who pulls them up hills').text,
  'User has a dog (Casper) who pulls them up hills');

check('A8 a multi-word parenthetical is not a bare name',
  strip('User (the primary account holder) has blue eyes').stripped, null);

// --- 2. the guard that stops a compound losing whole -----------------------
check('C1 the blue-eyes sentence reads as a compound',
  rules.looksCompound('User (Ellie) has blue eyes and her favorite color is green (last updated 2026-07-27).').compound,
  true);

check('C2 …and still does once the annotation is gone',
  rules.looksCompound(strip('User (Ellie) has blue eyes and her favorite color is green (last updated 2026-07-27).').text).compound,
  true);

check('C3 the clause on its own is NOT a compound, so it may be superseded',
  rules.looksCompound("User's favorite color is green (last updated 2026-07-27).").compound,
  false);

check('C4 the other clause is not a compound either',
  rules.looksCompound('User has blue eyes.').compound, false);

check('C5 the fact that won is not a compound',
  rules.looksCompound("User's favorite color is blue").compound, false);

// --- report ----------------------------------------------------------------
const line = '='.repeat(74);
console.log(`\n${line}\nA COMPOUND MUST NOT LOSE WHOLE — rule test\n${line}\n`);
for (const r of results) {
  console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`);
  if (!r.ok) console.log(`        wanted ${JSON.stringify(r.want)}\n        got    ${JSON.stringify(r.got)}`);
}
console.log(`\n${line}`);
console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
console.log(`${line}\n`);
process.exit(fail === 0 ? 0 : 1);
