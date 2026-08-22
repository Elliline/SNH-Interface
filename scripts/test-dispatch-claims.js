#!/usr/bin/env node
/**
 * THE CLAIM PATTERNS, AGAINST THE SENTENCES HE ACTUALLY WROTE.
 *
 * Every FIRES case below is real text from the live corpus or a minimal
 * variation of one. Every MISS case is a sentence that must not be corrected,
 * because appending "that did not happen" to a true or conditional statement is
 * its own kind of lie.
 *
 * Usage: node scripts/test-dispatch-claims.js
 */
const path = require('path');
const { classifyDispatchClaim, forgedStatusLine, classifyCodingGoAhead } =
  require(path.join(__dirname, '..', 'db/dispatch-claims'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}
const claims = t => classifyDispatchClaim(t).claims;

console.log('\n── The 2026-08-22 forgery, which the old guard missed ──');
const THE_MISS = "It's sent. The brief has been delivered to the coding agent.";
check('"It\'s sent. The brief has been delivered..." is a claim', claims(THE_MISS),
  'this is the exact sentence that got through');
check('  and it is recognised as PASSIVE voice',
  classifyDispatchClaim(THE_MISS).voice === 'passive');

console.log('\n── Passive and impersonal claims (the whole missing family) ──');
for (const t of [
  'It has been sent.',
  "It's sent.",
  'The brief has been delivered to the coding agent.',
  'The job has been handed to squatch-code.',
  'The brief has been dispatched.',
  'It has been queued.',
  'The work is on its way.',
  'It is with the coder now.',
  'Sent.',
  'Dispatched to squatch-code.',
  'Delivered.',
  'The brief was sent successfully.',
]) check(`fires: ${JSON.stringify(t)}`, claims(t));

console.log('\n── First person still fires (nothing regressed) ──');
for (const t of [
  'I have sent it to the coder.',
  "I've dispatched the brief.",
  'I have started a background job to organize everything I know about your clients.',
  'I am now running this in the background.',
  'The agent is now working.',
  'Sending the directive to squatch-code now.',
  'I will retry the job immediately.',
]) check(`fires: ${JSON.stringify(t)}`, claims(t));

console.log('\n── And these must NOT fire ──');
for (const t of [
  ['a question', 'Shall I send this to the coder?'],
  ['an offer', 'Would you like me to send it?'],
  ['conditional intent', "I'll send it once you confirm."],
  ['waiting', 'Let me know and I will send it.'],
  ['an explicit denial', "I didn't actually send it."],
  ['a denial, passive', 'Nothing was sent.'],
  ['not-yet', 'It has not been sent yet.'],
  ['making, not sending', 'The brief has been written.'],
  ['making, not sending', 'The file has been created.'],
  ['about tests', 'The tests have been run and they pass.'],
  ['about her sending', 'Once you send it, the coder will pick it up.'],
  ['describing the tool', 'When you approve, the brief goes to squatch-code.'],
  ['plain prose', 'The brief describes a self-contained HTML file.'],
]) check(`silent (${t[0]}): ${JSON.stringify(t[1])}`, !claims(t[1]), 'false positive');

console.log('\n── The forged status line ──');
const FORGERY = '_squatch-code, working:_\n- **squatch_crawler** · step 1/25 · run_command update_brief_v1.1 · 1m45s';
check('the exact forgery is detected', forgedStatusLine(FORGERY).forged);
check('the old header alone is detected', forgedStatusLine('_squatch-code, working:_').forged);
check('a progress line alone is detected', forgedStatusLine('- **todoapp** — step 4/25 · read_file x.py · 1m35s').forged);
check('a quiet-variant line is detected', forgedStatusLine('- **x** — no activity for 5m').forged);
for (const t of [
  'I will send the brief to squatch-code when you approve.',
  'The game has 25 levels and you are on step 1.',
  'Step 1: open the file. Step 2: run it.',
]) check(`not flagged: ${JSON.stringify(t.slice(0, 44))}`, !forgedStatusLine(t).forged, 'false positive');

console.log('\n── Her go-aheads, measured ──');
for (const t of [
  'Send away. Thanks for doing this.',
  'Go ahead and send it. Thank you.',
  'Go ahead and send it and lets see what you can make.',
  'Send it to the coder.',
  'Ship it.',
  'Yes, send it.',
]) check(`go-ahead: ${JSON.stringify(t)}`, classifyCodingGoAhead(t).goAhead);

for (const t of [
  ['a question', 'Should I have you send it?'],
  ['asking to see it', 'Can you show me the whole brief before you send it.'],
  ['conditional', 'I will read it and let you know if i aprove it and if you should send it to the coder'],
  ['a refusal', 'Stop — do not send it to Squatch-code.'],
  ['unrelated', 'The game worked, I got past the first 5 levels.'],
]) check(`not a go-ahead (${t[0]}): ${JSON.stringify(t[1].slice(0, 46))}`,
  !classifyCodingGoAhead(t[1]).goAhead, 'false positive');

console.log(`\n=== ${pass} passed, ${fail} failed ===`);
process.exit(fail ? 1 : 0);
