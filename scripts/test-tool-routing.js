#!/usr/bin/env node
/**
 * Unit test for classifyToolNeed — the tool-routing classifier.
 *
 * Guards the regression that shipped this fix: an explicit "look it up" request
 * about "the new Grok 4.5" was being classified DIRECT (no search), three turns
 * running. Run: node scripts/test-tool-routing.js
 */
const { classifyToolNeed } = require('../db/tool-routing');

let pass = 0, fail = 0;
function check(msg, expected, note) {
  const got = classifyToolNeed(msg, false); // superSearch OFF — exercise the classifier
  const ok = got === expected;
  if (ok) pass++; else fail++;
  const tag = ok ? 'PASS' : 'FAIL';
  console.log(`  [${tag}] want ${expected ? 'TOOLS ' : 'DIRECT'} got ${got ? 'TOOLS ' : 'DIRECT'}  ${note}\n         "${msg}"`);
}

console.log('\n=== The reported regression (must all route TOOLS) ===');
// The exact opening message from the bug report:
check('you vs the new Grok 4.5 — look it up on the interwebs if you need to', true, 'reported bug: explicit + recency');
check('has the new Grok 4.5 been released yet?', true, 'recency: new + proper noun + version');
check('is Grok 4.5 out?', true, 'version token');

console.log('\n=== Explicit search requests — hard override ===');
check('look it up', true, 'look it up');
check('look this up for me', true, 'look this up');
check('can you look that up?', true, 'look that up');
check('search for the best pizza in town', true, 'search for');
check('google it', true, 'google it');
check('check the web', true, 'check the web');
check('check online and tell me', true, 'check online');
check('find out on the internet', true, 'find out online');
check('look up the capital of France', true, 'bare look up still works');

console.log('\n=== Recency + entity ===');
check('what is the latest iOS version', true, 'latest + version keyword');
check('tell me about the new Vision Pro', true, 'new + proper noun (overrides tell-me-about)');
check('just released: Sora 2, what is it', true, 'just released + proper noun');
check('the newest Llama model', true, 'newest + proper noun');

console.log('\n=== Must stay DIRECT (no over-triggering) ===');
check('hello', false, 'greeting');
check('thanks!', false, 'thanks');
check('write a python function to reverse a string', false, 'coding');
check('explain how recursion works', false, 'conceptual');
check('what is the difference between a stack and a queue', false, 'conceptual');
check('write me a short poem about the ocean', false, 'creative');
check('what model are you', false, 'about the model');
check('what do you remember about my project', false, 'memory reference');
check('this is a whole new experience for me', false, 'recency word but no entity');
check('can you help me refactor this code', false, 'plain coding ask');

console.log(`\n=== ${pass} passed, ${fail} failed ===`);
process.exit(fail === 0 ? 0 : 1);
