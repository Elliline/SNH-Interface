#!/usr/bin/env node
/**
 * test-brief-shown — the guard that a dispatched brief was on her screen.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-brief-shown.js
 */

const assert = require('assert');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-brief-shown.js');
  process.exit(1);
}

const database = require('../db/database');
database.initDatabase();
const briefShown = require('../db/brief-shown');

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

const BRIEF =
  'Refactor the auth module in src/auth.py to use signed tokens instead of ' +
  'session cookies, keep the existing tests passing, and add a test for ' +
  'token expiry.';

function conversationShowing(text) {
  const id = database.createConversation('test', 'test-model');
  database.addMessage(id, 'assistant', text, 'test-model');
  return id;
}

console.log('\nwhat counts as having been shown\n');

test('her own message counts — "send the coder this: ..."', () => {
  const r = briefShown.check(BRIEF, { userMessage: BRIEF });
  assert.ok(r.ok, r.reason);
  assert.match(String(r.source), /her own message/);
});

test('an earlier reply in this conversation counts', () => {
  const convo = conversationShowing('Here is the brief:\n\n' + BRIEF);
  const r = briefShown.check(BRIEF, { conversationId: convo, userMessage: 'send it' });
  assert.ok(r.ok, r.reason);
  assert.match(String(r.source), /earlier reply/);
});

test('a brief inside a longer reply still counts', () => {
  const convo = conversationShowing(
    'I think the cleanest thing is this.\n\n' + BRIEF +
    '\n\nI can send that whenever you want.');
  assert.ok(briefShown.check(BRIEF, { conversationId: convo }).ok);
});

console.log('\nwhat does not\n');

test('a brief she never saw is refused', () => {
  const r = briefShown.check(BRIEF, { userMessage: 'yes go ahead, sounds right' });
  assert.ok(!r.ok);
  assert.match(r.reason, /has not seen this brief/i);
});

test('the drafting turn is refused — the reply is not stored yet', () => {
  // The turn where the model FIRST writes the brief: her message asks
  // for the work, the assistant reply does not exist in the database
  // yet, so there is nothing for the brief to match.
  const convo = database.createConversation('test', 'test-model');
  database.addMessage(convo, 'user', 'the auth module needs sorting out', 'test-model');
  const r = briefShown.check(BRIEF, {
    conversationId: convo,
    userMessage: 'the auth module needs sorting out',
  });
  assert.ok(!r.ok, 'dispatched on the turn the brief was written');
});

test('a different brief in the same conversation is refused', () => {
  const convo = conversationShowing(BRIEF);
  const other = 'Rewrite the billing exporter to emit CSV and delete the old ' +
                'XML path, updating the fixtures as you go.';
  assert.ok(!briefShown.check(other, { conversationId: convo }).ok);
});

test('a revision she has not seen is refused', () => {
  const convo = conversationShowing(BRIEF);
  const revised = BRIEF.replace('signed tokens', 'refresh tokens with rotation') +
    ' Also migrate the sessions table and drop the cookie helper entirely.';
  assert.ok(!briefShown.check(revised, { conversationId: convo }).ok,
    '"change X and send it" must not dispatch the unseen revision');
});

test('no context at all fails CLOSED', () => {
  const r = briefShown.check(BRIEF, {});
  assert.ok(!r.ok);
  assert.match(r.reason, /no way to tell/i);
});

test('a too-short brief cannot match by coincidence', () => {
  const r = briefShown.check('fix it', { userMessage: 'fix it please' });
  assert.ok(!r.ok);
  assert.match(r.reason, /too short/i);
});

console.log('\nformatting is not content\n');

test('markdown added around the brief still matches', () => {
  const shown = '## The brief\n\n> ' + BRIEF.replace(/\. /g, '.\n> ');
  const r = briefShown.check(BRIEF, { userMessage: shown });
  assert.ok(r.ok, `formatting broke the match (${r.ratio})`);
});

test('re-wrapped whitespace still matches', () => {
  const shown = BRIEF.replace(/ /g, '\n   ');
  assert.ok(briefShown.check(BRIEF, { userMessage: shown }).ok);
});

test('an exact send is reported as exact', () => {
  const r = briefShown.check(BRIEF, { userMessage: 'Send this: ' + BRIEF });
  assert.ok(r.ok && r.exact, 'verbatim send was not marked exact');
});

test('a reworded send is reported as NOT exact', () => {
  const r = briefShown.check('Please ' + BRIEF.toLowerCase(), { userMessage: BRIEF });
  assert.ok(r.ok, r.reason);
  assert.ok(!r.exact, 'a paraphrase claimed to be word for word');
  assert.ok(r.ratio >= briefShown.MATCH_THRESHOLD && r.ratio < 1);
});

console.log('\nthe ratio is reported so the threshold is tunable\n');

test('a refusal carries the best ratio it found', () => {
  const convo = conversationShowing('We should probably look at auth at some point.');
  const r = briefShown.check(BRIEF, { conversationId: convo });
  assert.ok(!r.ok);
  assert.ok(typeof r.ratio === 'number' && r.ratio < briefShown.MATCH_THRESHOLD);
});

test('coverage is asymmetric: a long reply containing the brief matches', () => {
  const long = 'x '.repeat(500) + BRIEF;
  assert.ok(briefShown.coverage(BRIEF, long) === 1);
});

test('...but a brief containing a short remark does not', () => {
  assert.ok(briefShown.coverage(BRIEF, 'send it') < 0.2);
});

test('stopwords alone cannot carry a match', () => {
  const filler = 'the a an and or but if then to of in on for with is are';
  assert.ok(briefShown.coverage(BRIEF, filler) < 0.2);
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
