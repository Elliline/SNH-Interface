#!/usr/bin/env node
/**
 * Does the flush summary preserve the EARLY part of a long thread on its own?
 *
 * WHY THIS IS ASKED NOW. Today's-log render no longer injects entries stamped
 * with the active conversation's id, because they echo message history that is
 * already in the request. On a short thread that is pure duplication. On a thread
 * long enough to trip `memory-flush`, the old turns are no longer in the request
 * — they were compacted away — so the question is whether the flush summary
 * carries them, since the log echo no longer backs it up.
 *
 * This is a MEASUREMENT, not a unit test: it calls the configured chat model, so
 * it needs the brain up. It writes to a throwaway SNH_DATA_DIR — never the live
 * corpus.
 *
 * Usage: node scripts/check-flush-early-content.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-flush-check-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();

const memoryFlush = require(path.join(ROOT, 'db/memory-flush'));
const { getConfig } = require(path.join(ROOT, 'db/config'));

const config = getConfig();
const chat = config.models.chat;
const inst = (config.providers[chat.provider] || []).find(p => p.name === chat.instance)
  || (config.providers[chat.provider] || [])[0];
const host = inst ? inst.host : null;
const model = chat.model;

// Three planted needles: one in the opening turns, one in the middle, one near
// the end. Each is a concrete detail of the kind a person would expect to be
// remembered, and each is stated exactly once.
const NEEDLES = {
  early: { probe: 'SQ-4417', text: 'The staging cutover is blocked on ticket SQ-4417, and nothing ships until that clears.' },
  middle: { probe: 'VACUUM INTO', text: 'We settled on keeping VACUUM INTO from a readonly connection for the seed copy, with no checkpoint first.' },
  late: { probe: 'Friday', text: 'Ellie needs the staging report before the Friday call at 2pm.' }
};

// Filler with enough substance that the model is summarising a conversation
// rather than a chant. Each exchange is ~120-160 tokens.
const TOPICS = [
  'the corrector pass and how it memoises pair verdicts',
  'why retired facts keep their row but lose their embedding',
  'the identity lock and which categories it covers',
  'the anomaly memo and why unchanged conditions are counted, not repeated',
  'cluster naming after a split, and who gets the old name',
  'the ops ledger versus the day\'s log',
  'the heartbeat tool budget and which steps get hands',
  'salience scoring and what a 2 out of 10 actually means',
  'the replay redirect and why it is a process, not a flag',
  'the agent pool throttling background work under live chat'
];

function buildConversation(targetTokens) {
  const messages = [];
  const push = (role, content) => messages.push({ role, content });

  push('user', `Morning. Before anything else: ${NEEDLES.early.text} Keep that in mind for the rest of this.`);
  push('assistant', `Understood — ${NEEDLES.early.text} I will treat it as the gate on everything else we discuss today.`);

  let i = 0;
  const est = memoryFlush.estimateMessagesTokens;
  while (est(messages) < targetTokens) {
    const topic = TOPICS[i % TOPICS.length];
    push('user', `Walk me through ${topic}. I want the reasoning, not just the behaviour, and tell me where it would break if the corpus doubled.`);
    push('assistant',
      `On ${topic}: the mechanism is deterministic where it can be and a model call only where a judgement is genuinely required. ` +
      `Enumeration comes from vector neighbours and marker regexes, the decision comes from the model, and the write is bounded so a bad pass ` +
      `costs a log line rather than a row. If the corpus doubled, the pass would still be bounded and resumable, because the memo table records ` +
      `both the per-row scan marks and the pair verdicts, so a second run starts where the first stopped instead of re-judging the first rows forever. ` +
      `The failure mode worth watching is a rehearsal that writes to the memo table, because that would make the live pass skip work it never did.`);
    if (i === 4) {
      push('user', `Note this: ${NEEDLES.middle.text}`);
      push('assistant', `Noted — ${NEEDLES.middle.text} Checkpointing would be a write, which is the thing we are avoiding.`);
    }
    i++;
  }

  push('user', `One more thing: ${NEEDLES.late.text}`);
  push('assistant', `Got it — ${NEEDLES.late.text} I will have it ready.`);
  return messages;
}

(async () => {
  if (!host) { console.error('No host configured for the chat provider.'); process.exit(1); }

  const contextLimit = memoryFlush.getModelContextLimit(model);
  // Just past the 80% trigger, which is what a real thread looks like when flush fires.
  const messages = buildConversation(Math.ceil(contextLimit * 0.85));
  const gate = memoryFlush.shouldFlush(messages, model);

  console.log(`\nModel:         ${chat.provider}/${model} @ ${host}`);
  console.log(`Context limit: ${contextLimit} tokens (as memory-flush computes it)`);
  console.log(`Conversation:  ${messages.length} messages, ${gate.tokenCount} tokens (${(gate.usage * 100).toFixed(1)}% of limit)`);
  console.log(`Flush fires:   ${gate.needsFlush}`);
  if (!gate.needsFlush) { console.error('Built a thread that does not trip the flush — check the builder.'); process.exit(1); }

  // What the extraction call will actually be shown, computed the way performFlush does.
  const convText = messages.filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
  const maxExtractionTokens = Math.floor(contextLimit * 0.5);
  const truncated = memoryFlush.estimateTokens(convText) > maxExtractionTokens;
  const shown = truncated ? convText.slice(-(maxExtractionTokens * 4)) : convText;
  console.log(`\nExtraction prompt cap: ${maxExtractionTokens} tokens; conversation is ${memoryFlush.estimateTokens(convText)} tokens.`);
  console.log(`Truncated before the model sees it: ${truncated}` +
    (truncated ? ` — the OLDEST ${memoryFlush.estimateTokens(convText) - maxExtractionTokens} tokens (~${(100 - (maxExtractionTokens / memoryFlush.estimateTokens(convText)) * 100).toFixed(0)}%) are cut, because the slice keeps the TAIL.` : ''));
  for (const [where, n] of Object.entries(NEEDLES)) {
    console.log(`  ${where.padEnd(7)} "${n.probe}" reaches the extraction model: ${shown.includes(n.probe)}`);
  }

  console.log('\nRunning the real performFlush…');
  const t0 = process.hrtime.bigint();
  const result = await memoryFlush.performFlush(
    messages, chat.provider, model, null, host, path.join(TMP, 'memory'));
  const secs = Number(process.hrtime.bigint() - t0) / 1e9;

  const summary = result.flushSummary || '';
  console.log(`\nFlush summary: ${summary.length} chars in ${secs.toFixed(1)}s`);
  console.log('-'.repeat(74));
  console.log(summary.slice(0, 2000));
  console.log('-'.repeat(74));

  console.log('\nDoes the summary preserve each planted detail?');
  const verdict = {};
  for (const [where, n] of Object.entries(NEEDLES)) {
    verdict[where] = summary.includes(n.probe);
    console.log(`  ${where.padEnd(7)} "${n.probe}": ${verdict[where] ? 'PRESERVED' : 'LOST'}`);
  }

  // The compacted messages are the other half of the answer: what the next turn
  // still has in front of it.
  const kept = result.compactedMessages.map(m => m.content).join('\n');
  console.log('\nAnd in the compacted message history the next turn actually gets:');
  for (const [where, n] of Object.entries(NEEDLES)) {
    console.log(`  ${where.padEnd(7)} "${n.probe}": ${kept.includes(n.probe) ? 'still present' : 'gone'}`);
  }

  console.log(`\n${'='.repeat(74)}`);
  console.log(verdict.early
    ? 'EARLY CONTENT SURVIVES the flush summary on its own.'
    : 'EARLY CONTENT IS LOST by the flush summary on its own.');
  console.log(`${'='.repeat(74)}\n`);
  process.exit(verdict.early ? 0 : 2);
})().catch(err => {
  console.error('[check-flush-early-content] error:', err);
  process.exit(1);
});
