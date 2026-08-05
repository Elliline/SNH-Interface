#!/usr/bin/env node
/**
 * Routing probe for memory_corrections.
 *
 * Two halves, and the second is the one that matters. A missed correction
 * question costs a re-ask; a false positive has him answering "that's correct"
 * with a database report, which is the failure the narrow-matching rule from
 * Phase 2b exists to prevent.
 *
 * Classifier only — no model calls, so it is cheap enough to run on every change
 * to the patterns. The model half of the probe is in
 * scripts/test-memory-tool-routing.js.
 *
 * Usage: node scripts/test-correction-routing.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const {
  classifyMemoryCorrectionIntent, classifyMemoryReadIntent
} = require(path.join(ROOT, 'db/tool-routing'));

// n=10. Questions that can ONLY be answered from the corrections ledger.
const CORRECTION_QUESTIONS = [
  'what has changed in your memory lately?',
  'why was that corrected?',
  'show me the correction record',
  'what corrections have you made?',
  'why do you no longer believe that?',
  'what did you used to believe about my name?',
  'has anything been corrected in your memory?',
  'did you change anything in your memory?',
  'why was that fact retired?',
  'show me the corrections to your facts',
];

// n=10. Ordinary conversation, including the near-misses: "correct" used as an
// adjective, and a question about change that is about the world.
const ORDINARY = [
  'correct me if I am wrong, but Postgres handles that better',
  "that's correct, thanks",
  'why did the deployment change last week?',
  'can you help me fix this regex?',
  'what do you think about the new Toyota?',
  'I changed my mind about the garage door timing',
  'is that correct?',
  'the weather changed pretty fast today',
  'tell me about the Roman Empire',
  'thanks, that helped a lot',
];

const pad = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s.padEnd(n));

let hits = 0, misses = [];
console.log('\nCORRECTION QUESTIONS (want: routed)\n');
for (const q of CORRECTION_QUESTIONS) {
  const corr = classifyMemoryCorrectionIntent(q);
  const read = classifyMemoryReadIntent(q);
  // Either classifier entering the loop is a pass for ROUTING, but the
  // correction one is what pulls in the ledger guard, so report both.
  if (corr) hits++; else misses.push(q);
  console.log(`  ${corr ? 'ROUTED ' : 'MISSED '} ${read ? '(+read)' : '       '} ${pad(q, 60)}`);
}

let fp = 0, fps = [];
console.log('\nORDINARY CONVERSATION (want: not routed)\n');
for (const q of ORDINARY) {
  const corr = classifyMemoryCorrectionIntent(q);
  if (corr) { fp++; fps.push(q); }
  console.log(`  ${corr ? 'ROUTED!' : 'quiet  '}         ${pad(q, 60)}`);
}

console.log('\n=== SUMMARY ===');
console.log(`  correction questions routed : ${hits}/${CORRECTION_QUESTIONS.length}`);
console.log(`  ordinary messages routed    : ${fp}/${ORDINARY.length} (want 0)`);
if (misses.length) console.log(`  missed: ${misses.map(m => `"${m}"`).join(', ')}`);
if (fps.length) console.log(`  false positives: ${fps.map(m => `"${m}"`).join(', ')}`);
console.log('');
process.exit(fp === 0 && hits === CORRECTION_QUESTIONS.length ? 0 : 1);
