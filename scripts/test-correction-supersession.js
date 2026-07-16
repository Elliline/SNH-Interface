#!/usr/bin/env node
/**
 * End-to-end test of correction-aware supersession, using the ACTUAL Bernice
 * transcripts from the live DB (the 7/8 correction and the 7/16 re-correction
 * that both failed to supersede the misspelled "Bernie" fact).
 *
 * Read-only: exercises extractFacts + judgeContradiction against the real
 * exchanges without writing any facts, supersessions, or questions.
 *
 * Asserts:
 *  1. extractFacts returns at least one fact with corrects metadata for each
 *     correction message.
 *  2. judgeContradiction, given the source message + corrects context, returns
 *     YES against the old "Bernie" fact — the verdict that was missed live.
 *  Also reports the bare (no-context) verdict for comparison; not asserted,
 *  since that blindness is exactly what this change works around.
 *
 * Usage: node scripts/test-correction-supersession.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

const OLD_BERNIE_FACT = 'User is setting up a computer for Bernie, who is a manager at ISH.';

// The two real correction exchanges, located by their stored user messages.
const CASES = [
  { label: '7/8 correction', userLike: 'Its Bernice, i was spelling it wrong%' },
  { label: '7/16 re-correction', userLike: 'It was Bernice not Bernie%' }
];

function getExchange(sql, userLike) {
  const user = sql.prepare(
    "SELECT id, conversation_id, content, timestamp FROM messages WHERE role='user' AND content LIKE ? ORDER BY timestamp LIMIT 1"
  ).get(userLike);
  if (!user) return null;
  const assistant = sql.prepare(
    "SELECT content FROM messages WHERE conversation_id = ? AND role='assistant' AND timestamp > ? ORDER BY timestamp LIMIT 1"
  ).get(user.conversation_id, user.timestamp);
  return { userMessage: user.content, assistantMessage: assistant ? assistant.content : '' };
}

(async () => {
  db.initDatabase();
  const sql = db.getSqliteDb();
  const fx = require(path.join(ROOT, 'db/fact-extractor'));

  const config = getConfig();
  const ext = config.models.extraction;
  const inst = getProviderInstance(ext.provider, ext.instance);
  const host = inst ? inst.host : null;

  let pass = 0, fail = 0;
  const check = (ok, msg) => {
    console.log(`${ok ? '  PASS' : '  FAIL'}: ${msg}`);
    ok ? pass++ : fail++;
  };

  for (const c of CASES) {
    console.log(`\n=== ${c.label} ===`);
    const exchange = getExchange(sql, c.userLike);
    if (!exchange) { check(false, `transcript not found for LIKE "${c.userLike}"`); continue; }
    console.log(`  user: "${exchange.userMessage.slice(0, 100)}"`);

    // 1. Extraction should flag the corrective intent.
    const extracted = await fx.extractFacts(
      exchange.userMessage, exchange.assistantMessage,
      ext.provider, ext.model, null, host
    );
    for (const f of extracted) {
      console.log(`  extracted: "${f.text}"${f.corrects ? ` [corrects: ${f.corrects}]` : ''}`);
    }
    const corrective = extracted.filter(f => f.corrects && /bernice/i.test(f.text));
    check(corrective.length > 0, 'extraction emits a Bernice fact with corrects metadata');

    // 2. With the source message + corrects context, the judge should now see
    //    the correction and call the old Bernie fact contradicted.
    const newFact = corrective[0] ? corrective[0].text
      : (extracted.find(f => /bernice/i.test(f.text)) || {}).text
      || "User's colleague Bernice is the Director of Rooms at ISH.";

    const bare = await fx.judgeContradiction(newFact, OLD_BERNIE_FACT);
    console.log(`  bare verdict (old behavior, informational): ${bare.verdict.toUpperCase()}`);

    const informed = await fx.judgeContradiction(newFact, OLD_BERNIE_FACT, {
      userMessage: exchange.userMessage,
      corrects: corrective[0] ? corrective[0].corrects : null
    });
    check(informed.verdict === 'yes',
      `judge with correction context returns YES against old Bernie fact (got ${informed.verdict.toUpperCase()})`);
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
