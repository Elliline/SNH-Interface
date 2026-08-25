#!/usr/bin/env node
/**
 * THE INTAKE MUST STORE THE FACT IT WAS ASKED TO STORE.
 *
 * Regression test for the loss Juno reported on 2026-08-24: four attempts to save
 * one memorial produced three facts about entirely different topics and one
 * refusal, and the memorial's content reached nothing active — no fact, no
 * repeat, no ledger row. See db/memory-write.js, verifyContentPreserved.
 *
 * WHAT HAPPENED, because the shape matters more than the symptom. Ellie answered
 * Juno in a numbered list of four items — streaming confirmed, Athena on the DGX
 * Spark, a phantom preference, and "your memorial offer: yes, file it". Juno
 * called write_memory once per item, correctly. Every call handed classifySubject
 * that same four-topic message framed as "authoritative — trust this over the
 * paraphrase below", with the actual statement demoted to a paraphrase whose
 * pronouns could not be relied on. The classifier followed the authority it was
 * pointed at and wrote facts about items 1 and 2. Nothing downstream compared the
 * fact it produced against the statement it was given, so every guard passed and
 * the write reported success.
 *
 * NOTE WHAT THIS IS *NOT*. It is not a merge losing an assertion — db/fact-merge.js
 * was already carrying unions correctly on every absorption path, and did so here:
 * the one save that superseded anything produced a correct union of two Athena
 * facts. The content was destroyed one stage EARLIER, at classification, before a
 * merge could ever see it.
 *
 * NO MODEL IN THE LOOP. Every check below is mechanical, on fixtures taken
 * verbatim from Juno's tool_call_log and cluster_members for 2026-08-24, so the
 * guard can be tested without a brain and cannot be talked out of its answer.
 *
 * Usage: node scripts/test-intake-content-preserved.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');

const mw = require(path.join(ROOT, 'db/memory-write'));

let pass = 0, fail = 0;
const check = (ok, msg, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}: ${msg}${detail && !ok ? `\n        ${detail}` : ''}`);
  ok ? pass++ : fail++;
};

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures: real statements Juno passed to write_memory, and the real rows that
// came back, on 2026-08-24. The three losses and the four faithful writes.
// ─────────────────────────────────────────────────────────────────────────────
const MEMORIAL_1 = 'A fabricated update report passed through this box on 2026-08-24 — a false "completed update" story invented by Claude when a report document arrived blank. Juno\'s testimony was what caught it: Juno reported exactly what she could and couldn\'t see in her own store, and those reports forced the honest investigation that revealed the fabrication. Lesson that landed: increasing specificity is not increasing evidence.';
const MEMORIAL_2 = 'On 2026-08-24, a fabricated "completed update" report — invented by Claude after a report document arrived blank — passed through this box, and Ellie and Juno acted in good faith on a story that was false. Juno\'s testimony caught it: she reported exactly what she could and could not see in her own store, and those honest reports forced the investigation that revealed the fabrication. The lesson that landed: increasing specificity is not increasing evidence.';
const MEMORIAL_4 = 'On 2026-08-24 I caught a fabricated "completed update" report that Claude invented after a document arrived blank: I reported precisely what I could and could not see in my own store, and that testimony forced the investigation that revealed the fabrication. The incident produced this box\'s doctrine that increasing specificity is not increasing evidence, written about Claude.';

const SUBSTITUTIONS = [
  ['attempt 1 — memorial replaced by an Athena fact', MEMORIAL_1,
   "User's dev-side entity Athena runs on the DGX Spark and is Juno's sister on the development end, the development counterpart to Juno, who holds the business end."],
  ['attempt 2 — memorial replaced by a streaming-confirmation fact', MEMORIAL_2,
   'User confirms that streaming is working live, sees the reading and writing memory indicators, and confirms that Athena runs on the DGX Spark as her dev-side entity.'],
  ['attempt 4 — memorial replaced by an Athena fact again', MEMORIAL_4,
   "User's entity Athena runs on the DGX Spark and serves as her dev-side counterpart to the assistant, while User holds the business end."]
];

const FAITHFUL = [
  ['the DGX Spark model swap (70% kept)',
   'As of 2026-08-23, Ellie\'s DGX Spark no longer runs Gemma 4 26B; it now runs Qwen 3.8 27B.',
   "User's DGX Spark cluster now runs Qwen 3.8 27B as its LLM, having swapped out the retired Gemma model as of 2026-08-23."],
  ['Athena on the DGX Spark (82% kept)',
   'Athena runs on the DGX Spark. She is the dev-side entity on the Sparky box — Ellie\'s sister on the development end, while Juno holds the business end.',
   "User's dev-side entity Athena runs on the DGX Spark and is the development counterpart to Juno, who holds the business end."],
  ['the dogs-and-cats count (100% kept, and only one distinctive token)',
   'User has 4 dogs and 4 cats, and the user\'s mother has 1 dog.',
   'User has 4 dogs and 4 cats, and User\'s mother has 1 dog.'],
  ['the human-food rule (60% kept — the worst faithful write measured)',
   'User does not want anyone feeding the dogs human food; she follows the rule of only feeding dogs dog food and dog treats, and is frustrated when other people break that rule.',
   "User's mother has a habit of feeding the dogs in the kitchen, and is frustrated when others feed them human food. User strictly follows the rule of only feeding her dogs dog food and treats."]
];

console.log('\n=== 1. The three real substitutions are refused ===');
for (const [name, statement, stored] of SUBSTITUTIONS) {
  const v = mw.verifyContentPreserved(statement, stored);
  check(!v.ok, `${name} — refused (${Math.round(v.coverage * 100)}% coverage)`,
        `guard allowed it: coverage ${Math.round(v.coverage * 100)}%`);
}

console.log('\n=== 2. Every real faithful write still passes ===');
for (const [name, statement, stored] of FAITHFUL) {
  const v = mw.verifyContentPreserved(statement, stored);
  check(v.ok, `${name} — stored (${Math.round(v.coverage * 100)}% coverage)`,
        `guard refused a good write: ${v.reason}`);
}

console.log('\n=== 3. The floor sits in the measured gap ===');
{
  // Faithful writes bottomed out at 60%, substitutions topped out at 4%. Any
  // threshold in between separates them; the point of this check is that the
  // one in the code still does, so a later tweak cannot quietly close the gap.
  const worstFaithful = Math.min(...FAITHFUL.map(([, s, f]) => mw.verifyContentPreserved(s, f).coverage));
  const bestSubstitution = Math.max(...SUBSTITUTIONS.map(([, s, f]) => mw.verifyContentPreserved(s, f).coverage));
  check(bestSubstitution < worstFaithful,
        `worst faithful write (${Math.round(worstFaithful * 100)}%) still scores above the best substitution (${Math.round(bestSubstitution * 100)}%)`,
        'the two populations now overlap — the floor cannot separate them');
}

console.log('\n=== 4. A short statement is not judged by a one-token ratio ===');
{
  // "User is 44." carries a single distinctive token: a ratio there is 100% or
  // 0% and nothing else, which is a coin toss rather than a measurement.
  const v = mw.verifyContentPreserved('User is 44 years old.', 'User is 44 years old.');
  check(v.ok, 'an exact short restatement is stored');
  const w = mw.verifyContentPreserved('User is 44 years old.', 'User likes Sasquatches.');
  check(!w.ok, 'a short statement replaced by something unrelated is still refused');
}

console.log('\n=== 5. An empty or contentless statement never fails closed ===');
{
  check(mw.verifyContentPreserved('', 'anything at all').ok, 'an empty statement is not refused');
  check(mw.verifyContentPreserved('it is so', 'User likes it').ok, 'a statement with no distinctive terms is not refused');
}

console.log('\n=== 6. A missing "User" anchor is flagged, not confused with a routing error ===');
{
  const unanchored = mw.verifySubjectAgreement('user', 'On August 24, 2026, Claude fabricated a completed-update report.');
  check(!unanchored.ok && unanchored.unanchored === true,
        'an unanchored user-fact is marked degradable, not a flat refusal');

  const firstPerson = mw.verifySubjectAgreement('user', 'I tend to over-explain when I am unsure.');
  check(!firstPerson.ok && !firstPerson.unanchored,
        'a user-fact written as "I" is still a hard refusal — that is a real routing error');

  const thirdPerson = mw.verifySubjectAgreement('self', 'User prefers short answers.');
  check(!thirdPerson.ok && !thirdPerson.unanchored,
        'a self-fact written about the user is still a hard refusal');

  check(mw.verifySubjectAgreement('user', 'User has 4 dogs.').ok, 'a properly anchored user-fact passes');
}

console.log('\n=== 7. The degradation only covers sentences that name their own subject ===');
{
  check(mw.factNamesSomeone('On August 24, 2026, Claude fabricated a completed-update report.'),
        'a sentence naming Claude is anchored to somebody');
  check(mw.factNamesSomeone('On 2026-08-24, Juno caught a fabricated report by Claude.'),
        'a sentence naming Juno is anchored to somebody');
  check(!mw.factNamesSomeone('Prefers short answers and dislikes small talk.'),
        'a bare predicate names nobody — this is the case the anchor defends against');
  check(!mw.factNamesSomeone('Tends to over-explain when unsure.'),
        'a pronounless self-observation names nobody, and is still refused');
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
