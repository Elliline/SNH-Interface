#!/usr/bin/env node
/**
 * Household names are WORDS, not strings.
 *
 * The two cluster heuristics that know the household's first names —
 * isPersonFact (singleton merging) and the "People & Family" label — used to
 * find a name by substring and by prefix. So a name matched inside a longer
 * word, and a two-letter name could not be added at all. Family words like
 * "mom" and "stepfather" were not recognised by either.
 *
 * Every name here is MADE UP. The real household lives in data/config.json
 * and never in this tree.
 *
 * PURE: no model, no live store. Runs under SNH_DATA_DIR.
 * Usage: node scripts/test-household-names.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const clusters = require(path.join(ROOT, 'db/memory-clusters'));
const { isPersonFact, generateClusterNameFromMembers, matchCuratedCategory, wordMatcher } = clusters;

// Fake household: a two-letter person, an ordinary person, a one-word animal,
// a two-word animal.
clusters._overrideHouseholdNamesForTests({
  personNames: ['zo', 'brightwater'],
  animalNames: ['pumpernickel', 'moss boot'],
});

let pass = 0, fail = 0;
const results = [];
const check = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  ok ? pass++ : fail++;
  results.push({ ok, name, got, want });
};
const label = (...facts) => generateClusterNameFromMembers(facts.map(content => ({ content })));

// --- 1. a short name must not match inside longer words --------------------
check('W1 two-letter name inside "zone", "ozone", "zoo", "gonzo" is NOT a match',
  isPersonFact('The zone has an ozone monitor by the zoo, says Gonzo.'), false);
check('W2 the same two-letter name as a whole word IS a match',
  isPersonFact('Zo fixed the fence yesterday.'), true);
check('W3 a longer name inside a longer word is NOT a match',
  isPersonFact('The brightwaters reservoir is full; brightwaterfall is a typo.'), false);
check('W4 the longer name as a whole word IS a match',
  isPersonFact('Brightwater came for dinner.'), true);
check('W5 an animal name inside a longer word is NOT a match',
  isPersonFact('Pumpernickels were on sale.'), false);

// --- 2. possessives and case ---------------------------------------------
check('P1 a possessive matches', isPersonFact("Zo's truck needs tyres."), true);
check('P2 a curly-apostrophe possessive matches', isPersonFact('Brightwater’s house has bad wifi.'), true);
check('P3 case does not matter', isPersonFact('ZO and BRIGHTWATER argued.'), true);
check('P4 a possessive animal matches', isPersonFact("Pumpernickel's bed is by the door."), true);

// --- 3. two-word animal name, both spellings ------------------------------
check('T1 the two-word animal name matches as a phrase', isPersonFact('Moss Boot slept on the porch.'), true);
check('T2 the run-together spelling matches too', isPersonFact('Mossboot slept on the porch.'), true);
check('T3 one word of the pair alone does NOT match', isPersonFact('The moss on the boot was green.'), false);
check('T4 the pair split by another word does NOT match', isPersonFact('Moss, then boot.'), false);

// --- 4. family words, without any first name ------------------------------
for (const w of ['mom', 'dad', 'stepdad', 'stepfather', 'stepmom', 'stepmother', 'parent', 'parents', 'grandma', 'uncle']) {
  check(`F-${w} "my ${w}" is a person fact`, isPersonFact(`User visited my ${w} on Sunday.`), true);
}
check('F-mom-possessive "my mom\'s house" is a person fact', isPersonFact("User drove to my mom's house."), true);
check('F-moment "moment" is NOT a family word', isPersonFact('It was a proud moment for the team.'), false);
check('F-song "song" is NOT "son"', isPersonFact('That song is stuck in my head.'), false);
check('F-motherboard "motherboard" is NOT "mother"', isPersonFact('The motherboard has a dead slot.'), false);

// --- 5. the label heuristic -------------------------------------------------
check('L1 two person names label People & Family',
  label('Zo fixed the fence.', 'Brightwater came by for dinner.'), 'People & Family');
check('L2 "my mom" plus a name labels People & Family',
  label('My mom called about Sunday.', 'Zo will drive her.'), 'People & Family');
check('L3 "stepfather" plus "partner" labels People & Family',
  label('Her stepfather retired in May.', 'Her partner likes hiking.'), 'People & Family');
check('L4 a cluster of only animals is Pets & Animals, not people',
  label('Pumpernickel the dog needs shots.', 'Moss Boot the cat sleeps all day.', 'Pumpernickel is a good dog.'), 'Pets & Animals');
check('L5 a two-letter name inside "zone"/"ozone" does NOT label People & Family',
  label('The zone has an ozone monitor.', 'The ozone alarm is loud.') === 'People & Family', false);
check('L6 "moment" and "song" do NOT label People & Family',
  label('A proud moment for the team.', 'That song is stuck in my head.') === 'People & Family', false);
check('L7 matchCuratedCategory: a name with a family word',
  matchCuratedCategory("Zo is my mom's brother."), 'People & Family');
check('L8 matchCuratedCategory: a name inside a longer word does not count',
  matchCuratedCategory('The zone alarm is loud'), null);
check('L9 ordinary topic keys still match loosely (plural)',
  label('Both servers need more vram.', 'The gpus run hot.'), 'Hardware & Infrastructure');

// --- 6. the matcher itself ---------------------------------------------
const m = wordMatcher("Zo's zone, and Moss-Boot's Mossboot.");
check('M1 tokens are lowercased, possessive-stripped words',
  m.tokens, ['zo', 'zone', 'and', 'moss-boot', 'mossboot']);
check('M2 an empty name never matches', m.has(''), false);

clusters._overrideHouseholdNamesForTests(null);

const line = '='.repeat(72);
console.log(`\n${line}\nHOUSEHOLD NAMES ARE WORDS\n${line}`);
for (const r of results) {
  console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`);
  if (!r.ok) console.log(`        wanted ${JSON.stringify(r.want)}\n        got    ${JSON.stringify(r.got)}`);
}
console.log(`\n${line}`);
console.log(`${pass} passed, ${fail} failed`);
console.log(`${line}\n`);
process.exit(fail === 0 ? 0 : 1);
