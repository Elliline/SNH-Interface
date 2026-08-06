#!/usr/bin/env node
/**
 * The history-vs-current-state rule, tested.
 *
 * A past-tense fact and a present-tense one about the same subject matter are not
 * a contradiction — they are two true statements about two different times. This
 * rule exists because the corrector retired three facts on 2026-08-06 that it
 * should never have been given, and the worst of them took the reason a car went
 * with it, and that reason was a death.
 *
 * PURE. No database, no model, no data directory — extraction-rules is pure and
 * synchronous by design, which is what lets this run anywhere in a second.
 *
 * The first three cases are the observed failures, verbatim from the ledger.
 *
 * Usage: node scripts/test-history-coexists.js
 * Exit 0 if every case passes.
 */
const path = require('path');
const rules = require(path.join(__dirname, '..', 'db', 'extraction-rules'));

let pass = 0, fail = 0;
const results = [];

function check(name, got, want, detail = '') {
  const ok = got === want;
  ok ? pass++ : fail++;
  results.push({ ok, name, got, want, detail });
}

/** The pair must be exempted — these two do not compete. */
function exempt(name, a, b) {
  const r = rules.historyCoexists(a, b);
  check(name, r.exempt, true, r.reason || 'no reason given');
  // Order must not matter. A rule that only works one way round is half a rule,
  // and the enumeration hands pairs over in whichever order the vector search
  // returned them.
  const rev = rules.historyCoexists(b, a);
  check(`${name} (reversed)`, rev.exempt, true, rev.reason || 'no reason given');
}

/** The pair must still reach the judge — this rule must not eat real conflicts. */
function judged(name, a, b) {
  const r = rules.historyCoexists(a, b);
  check(name, r.exempt, false, r.reason || '');
}

// ---------------------------------------------------------------------------
// 1. The three the corrector actually got wrong, verbatim from the ledger
// ---------------------------------------------------------------------------
exempt('H1 highlander/rav4 — the trade vs what is owned now',
  'User kept the Highlander Limited for six months before trading it for a 2023 RAV4 Prime.',
  'User owns a Rav4 GR Sport');

exempt('H2 rav+tacoma/rav4 — the one that took a death with it',
  'User had to get rid of the RAV and the Tacoma due to painful memories associated with a death.',
  'User has a Rav4');

exempt('H3 amd/macbook — preferred vs prefers',
  'User preferred AMD',
  'User prefers their MacBook');

// ---------------------------------------------------------------------------
// 1b. The SECOND round of the same failure — the present-tense side written as
//     "User's <thing> is <state>" rather than "User has <thing>".
//
//     The first version of this rule matched only "User has…"/"User owns…" and
//     exempted H2 against "User has a Rav4" — then the next corrector pass
//     retired it again for "User's Rav4 is brand new", which is the same claim in
//     a different dress. The RAV/Tacoma fact was retired twice by two spellings
//     of one mistake, which is why these cases exist.
// ---------------------------------------------------------------------------
exempt('H4 rav+tacoma vs "User\'s Rav4 is brand new" — the second retirement',
  'User had to get rid of the RAV and the Tacoma due to painful memories associated with a death.',
  "User's Rav4 is brand new");

exempt('H5 truck trade vs "User\'s Tundra is brand new"',
  'User traded a truck in for a 2015 Tundra Limited.',
  "User's Tundra is brand new");

exempt('H6 rav4 trade vs "User\'s Rav4 is brand new"',
  'User traded a RAV4 in for a Highlander Limited.',
  "User's Rav4 is brand new");

check('P1 "User\'s Rav4 is brand new" reads as current state',
  rules.isCurrentState("User's Rav4 is brand new"), true);

check('P2 "User\'s gaming system has an RTX 5080" reads as current state',
  rules.isCurrentState("User's gaming system has an RTX 5080"), true);

check('P3 "User\'s old truck was totalled" reads as historical',
  rules.isHistorical("User's old truck was totalled").historical, true);

check('P4 the possessed-noun window does not reach across a whole sentence',
  rules.isCurrentState("User's plan to replace the truck before the winter storms arrive is settled"), false);

// The window must not eat a PRESENT-tense auxiliary on its way to a past
// participle. Caught firing live in a corrector pass: "User's RAV4 has not had
// wax applied" consumed "RAV4 has not" as the noun phrase, matched "had", and
// was filed as history — a statement about the car's condition right now.
check('P5 "User\'s RAV4 has not had wax applied" is current state, not history',
  rules.isHistorical("User's RAV4 has not had wax applied.").historical, false);

check('P6 …and reads as current state',
  rules.isCurrentState("User's RAV4 has not had wax applied."), true);

check('P7 "User\'s Tundra had 200k miles" is still history',
  rules.isHistorical("User's Tundra had 200k miles").historical, true);

judged('N2 two present-perfect claims about the same thing still compete',
  "User's RAV4 has not had wax applied.",
  "User's RAV4 has had a ceramic coating applied.");

// ---------------------------------------------------------------------------
// 2. Real contradictions, which must still be judged
// ---------------------------------------------------------------------------
judged('C1 two present-tense claims still compete',
  "User's machine has 32GB of RAM",
  "User's main computer is an M5 MacBook Pro with 48GB of RAM");

judged('C2 two past-tense claims still compete',
  'User owned a Tundra in 2019',
  'User had never owned a Tundra');

judged('C3 same tense, same slot, different value',
  "User's favorite color is blue",
  "User's favorite color is green");

// ---------------------------------------------------------------------------
// 3. The traps — a past-tense CLAUSE inside a present-tense fact
// ---------------------------------------------------------------------------
judged('T1 an embedded past clause does not make the sentence historical',
  'User has a dog that was born in 2019',
  'User has a cat that was born in 2019');

check('T2 "User has a dog that was born in 2019" reads as current state',
  rules.isCurrentState('User has a dog that was born in 2019'), true);

check('T3 "User used to have a dog" reads as historical',
  rules.isHistorical('User used to have a dog').historical, true);

check('T4 "User has previously paid $2,000 per car" reads as historical',
  rules.isHistorical('User has previously paid $2,000 per car for professional automotive services.').historical, true);

check('T5 past wins over present when both could match',
  rules.isCurrentState('User used to have a Tundra'), false);

check('T6 "User no longer uses Tailscale" reads as historical',
  rules.isHistorical('User no longer uses Tailscale').historical, true);

// ---------------------------------------------------------------------------
// 4. The rule must not fire on a pair where the "present" side is not a state
// ---------------------------------------------------------------------------
judged('N1 past vs a non-state sentence is not exempted',
  'User traded the GR86 for a 2026 Tundra',
  'Inn At Spanish Head (ISH) is located in Lincoln City on the Oregon coast');

// ---------------------------------------------------------------------------

const line = '='.repeat(74);
console.log(`\n${line}\nHISTORY COEXISTS WITH CURRENT STATE — rule test\n${line}\n`);
for (const r of results) {
  console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`);
  if (!r.ok) console.log(`        wanted ${r.want}, got ${r.got}`);
  if (r.ok && r.detail) console.log(`        ${r.detail.slice(0, 120)}`);
}
console.log(`\n${line}`);
console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
console.log(`${line}\n`);
process.exit(fail === 0 ? 0 : 1);
