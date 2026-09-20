#!/usr/bin/env node
/**
 * Tool-call markup must never reach the chat window.
 *
 * On 2026-08-06 Ellie asked which approved job never ran. He had no tool for it,
 * so he wrote the call he wished he could make — as text, in his reply — and it
 * rendered:  <function=memory_jobs>{"status": "approved"}</function>
 *
 * The filter has to work a CHARACTER AT A TIME, because that is how the answer
 * arrives. Half these cases feed the same string in awkward pieces to prove the
 * boundary handling, since a regex over the finished text would have been easy
 * and would not have helped: by the time `</function>` arrives, `<f` is already
 * on screen.
 *
 * PURE. No database, no model, no server.
 *
 * Usage: node scripts/test-tool-artifacts.js
 */
const path = require('path');
const { createToolArtifactFilter, stripToolArtifacts } =
  require(path.join(__dirname, '..', 'db', 'tool-artifacts'));

let pass = 0, fail = 0;
const results = [];
const check = (name, got, want) => {
  const ok = got === want;
  ok ? pass++ : fail++;
  results.push({ ok, name, got, want });
};

/** Feed a string in pieces of n characters, the way a stream delivers it. */
function streamIn(text, n) {
  const f = createToolArtifactFilter();
  let out = '';
  for (let i = 0; i < text.length; i += n) out += f.feed(text.slice(i, i + n));
  out += f.flush();
  return { out, stripped: f.stripped() };
}

// --- 1. the exact failure --------------------------------------------------
const REAL = 'Let me look that up. <function=memory_jobs>{"status": "approved"}</function>';
check('A1 the observed artifact is removed',
  stripToolArtifacts(REAL).text, 'Let me look that up. ');
check('A1a …and is counted', stripToolArtifacts(REAL).stripped, 1);

// One character at a time is the real delivery shape.
check('A2 removed when streamed one character at a time', streamIn(REAL, 1).out, 'Let me look that up. ');
check('A3 removed at chunk size 3', streamIn(REAL, 3).out, 'Let me look that up. ');
check('A4 removed at chunk size 7', streamIn(REAL, 7).out, 'Let me look that up. ');
check('A5 removed when the whole thing arrives at once', streamIn(REAL, 999).out, 'Let me look that up. ');

// --- 2. an artifact that never closes --------------------------------------
//
// The commonest shape: the model runs out mid-call. The fragment is precisely
// what must not be shown.
const UNCLOSED = 'Checking now. <function=memory_jobs>{"status": "appro';
check('B1 an unterminated artifact is swallowed whole', streamIn(UNCLOSED, 1).out, 'Checking now. ');
check('B1a …and still counted', streamIn(UNCLOSED, 1).stripped, 1);

// --- 3. the other engine dialects ------------------------------------------
check('C1 <tool_call>',
  stripToolArtifacts('Before. <tool_call>{"name":"x"}</tool_call> After.').text, 'Before.  After.');
check('C2 [TOOL_CALL]',
  stripToolArtifacts('a [TOOL_CALL]do_thing()[/TOOL_CALL] b').text, 'a  b');
check('C3 <|python_tag|>',
  stripToolArtifacts('x <|python_tag|>call()<|eom_id|> y').text, 'x  y');

// --- 4. ORDINARY PROSE MUST SURVIVE UNTOUCHED ------------------------------
//
// This filter runs on every token of every reply. A false positive silently
// deletes part of an answer, which is worse than the thing it is preventing.
const innocent = [
  'The function is called twice.',
  'Use a < b to compare them.',
  'In HTML you would write <b>bold</b> and <i>italic</i>.',
  'The comparison a<function b is nonsense but harmless.',
  'She said "5 < 10" and I agreed.',
  'I can call memory_jobs to check that for you.',
  'Your name is Ellie.',
  ''
];
for (let i = 0; i < innocent.length; i++) {
  check(`D${i + 1} untouched: "${innocent[i].slice(0, 44)}"`, stripToolArtifacts(innocent[i]).text, innocent[i]);
  // And untouched no matter how it is cut up.
  check(`D${i + 1}s …streamed one char at a time`, streamIn(innocent[i], 1).out, innocent[i]);
}

// --- 5. an artifact that IS the whole reply --------------------------------
//
// Nothing legible is left, which is what the honest-refusal fallback is for.
const ONLY = '<function=memory_jobs>{"status":"approved"}</function>';
const onlyRes = streamIn(ONLY, 2);
check('E1 an artifact-only reply leaves nothing', onlyRes.out.trim(), '');
check('E1a …and reports that it stripped something', onlyRes.stripped, 1);

// --- 6. more than one in a reply -------------------------------------------
check('F1 two artifacts in one reply',
  stripToolArtifacts('a <function=x>{}</function> b <function=y>{}</function> c').text, 'a  b  c');
check('F1a …both counted',
  stripToolArtifacts('a <function=x>{}</function> b <function=y>{}</function> c').stripped, 2);

// --- report ----------------------------------------------------------------
const bar = '='.repeat(76);
console.log(`\n${bar}\nTOOL-CALL MARKUP MUST NOT REACH THE CHAT WINDOW\n${bar}\n`);
for (const r of results) {
  console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`);
  if (!r.ok) console.log(`        wanted ${JSON.stringify(r.want)}\n        got    ${JSON.stringify(r.got)}`);
}
console.log(`\n${bar}`);
console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
console.log(`${bar}\n`);
process.exit(fail === 0 ? 0 : 1);
