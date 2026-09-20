#!/usr/bin/env node
/**
 * What does the new window do to a thread that used to be compacted?
 *
 * The 8/13 report built a 68-message thread to trip the flush, because the
 * static table said the window was 8,192 tokens. The engine was serving 131,072
 * and the app now uses 24,576 of it. So the same thread is no longer long — the
 * question is how many times it would be compacted now (the answer should be
 * none), and what a turn costs deep inside it, since a bigger window is paid for
 * in prefill on every turn.
 *
 * Replays the thread turn by turn: at each user turn it asks shouldFlush the way
 * checkAndFlush does, then sends the whole thread so far to the engine and times
 * the reply. Read-only — no flush is performed, nothing is stored.
 *
 * Usage: node scripts/measure-flush-under-window.js [--target-tokens 6963]
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-flush-window-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const memoryFlush = require(path.join(ROOT, 'db/memory-flush'));
const modelContext = require(path.join(ROOT, 'db/model-context'));
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

const cfg = getConfig();
const chat = cfg.models.chat;
const inst = getProviderInstance(chat.provider, chat.instance);
const HOST = inst ? inst.host : 'http://localhost:7070';

// The thread from the 8/13 report, rebuilt: it was sized at 85% of the window
// the static table claimed (8,192), which is where 68 messages came from.
const OLD_TABLE_LIMIT = 8192;
const targetArg = process.argv.indexOf('--target-tokens');
const TARGET = targetArg > -1 ? Number(process.argv[targetArg + 1]) : Math.ceil(OLD_TABLE_LIMIT * 0.85);

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
  push('user', 'Morning. Before anything else: the staging cutover is blocked on ticket SQ-4417, and nothing ships until that clears.');
  push('assistant', 'Understood — SQ-4417 gates everything else we discuss today.');
  let i = 0;
  while (memoryFlush.estimateMessagesTokens(messages) < targetTokens) {
    const topic = TOPICS[i % TOPICS.length];
    push('user', `Walk me through ${topic}. I want the reasoning, not just the behaviour, and tell me where it would break if the corpus doubled.`);
    push('assistant',
      `On ${topic}: the mechanism is deterministic where it can be and a model call only where a judgement is genuinely required. ` +
      `Enumeration comes from vector neighbours and marker regexes, the decision comes from the model, and the write is bounded so a bad pass ` +
      `costs a log line rather than a row. If the corpus doubled, the pass would still be bounded and resumable, because the memo table records ` +
      `both the per-row scan marks and the pair verdicts, so a second run starts where the first stopped instead of re-judging the first rows forever. ` +
      `The failure mode worth watching is a rehearsal that writes to the memo table, because that would make the live pass skip work it never did.`);
    i++;
  }
  return messages;
}

async function timeTurn(messages) {
  const t0 = Date.now();
  const res = await fetch(`${HOST}/v1/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: chat.model, messages, max_tokens: 64, stream: false }),
    signal: AbortSignal.timeout(180000)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return { ms: Date.now() - t0, promptTokens: data.usage?.prompt_tokens ?? null };
}

(async () => {
  await modelContext.ensureProbed(chat.provider, HOST, chat.model);
  const usable = memoryFlush.getModelContextLimit(chat.model, chat.provider, HOST);
  const messages = buildConversation(TARGET);
  const total = memoryFlush.estimateMessagesTokens(messages);

  console.log(`\nModel ${chat.provider}/${chat.model} @ ${HOST}`);
  console.log(`Engine ceiling ${modelContext.engineLimit(chat.model, chat.provider, HOST)}, ` +
    `usable window ${usable} (memory.contextTokens ${cfg.memory.contextTokens})`);
  console.log(`Thread: ${messages.length} messages, ${total} tokens — built to 85% of the OLD table limit (${OLD_TABLE_LIMIT}).\n`);

  // How many times would this thread be compacted, replayed turn by turn?
  let flushes = 0, firstFlushAt = null;
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role !== 'user') continue;
    const soFar = messages.slice(0, i + 1);
    const g = memoryFlush.shouldFlush(soFar, chat.model, chat.provider, HOST);
    if (g.needsFlush) { flushes++; if (firstFlushAt === null) firstFlushAt = i + 1; }
  }
  const old = { limit: OLD_TABLE_LIMIT };
  let oldFlushes = 0;
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role !== 'user') continue;
    const t = memoryFlush.estimateMessagesTokens(messages.slice(0, i + 1).filter(m => m.role !== 'system'));
    if (t > old.limit * 0.80) oldFlushes++;
  }

  console.log(`Turns that would trip the flush:`);
  console.log(`  under the old table limit (${OLD_TABLE_LIMIT}):  ${oldFlushes}` +
    `${oldFlushes ? ` — first at message ${messages.findIndex((m, i) => m.role === 'user' &&
      memoryFlush.estimateMessagesTokens(messages.slice(0, i + 1)) > old.limit * 0.8) + 1}` : ''}`);
  console.log(`  under the usable window (${usable}):        ${flushes}` +
    `${firstFlushAt ? ` — first at message ${firstFlushAt}` : ''}`);
  console.log(`  (the whole thread is ${(total / usable * 100).toFixed(1)}% of the new window)\n`);

  // What a turn costs deep in the thread.
  console.log('Turn latency by depth (max_tokens=64, so this is prefill plus a short reply):');
  const depths = [2, Math.floor(messages.length / 4), Math.floor(messages.length / 2), messages.length];
  for (const d of depths) {
    const slice = messages.slice(0, d);
    if (slice[slice.length - 1].role !== 'user') slice.push({ role: 'user', content: 'Summarise where we got to in one sentence.' });
    const warm = await timeTurn(slice);          // pay the cold prefill once
    const hot = await timeTurn(slice);           // then measure with the prefix cached
    console.log(`  ${String(d).padStart(3)} messages  ~${String(warm.promptTokens).padStart(6)} prompt tokens   ` +
      `cold ${String(warm.ms).padStart(5)}ms   cached ${String(hot.ms).padStart(5)}ms`);
  }
  console.log('');
})().catch(err => { console.error('[measure-flush-under-window] error:', err.message); process.exit(1); });
