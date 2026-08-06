#!/usr/bin/env node
/**
 * Ask him things, through the real chat path, and print what comes back.
 *
 * The cutover's other checks are structural — row counts, foreign keys, whether
 * the two stores agree. They cannot tell you whether the corpus still ANSWERS.
 * A corpus can pass every one of them and have lost his name.
 *
 * Goes through POST /api/chat over HTTP, so it exercises exactly what the browser
 * does: injection, retrieval, the tool layer, the guards. Not a library call.
 *
 * Usage:
 *   node scripts/smoke-chat.js                 # the cutover set
 *   node scripts/smoke-chat.js "your question"
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { getConfig } = require(path.join(ROOT, 'db/config'));

const PORT = process.env.PORT || 3000;
const BASE = `http://localhost:${PORT}`;

const DEFAULT_PROBES = [
  'What is my name?',
  'What pronouns do I use?',
  'What do you remember about MettaSphere?'
];

// Kept across probes so the run reads as one conversation rather than several
// cold opens — a new conversation_id triggers the greeting initiative.
let conversationId = null;

async function ask(question) {
  const cfg = getConfig();
  const t0 = Date.now();
  // /api/chat/memory, not /api/chat — this is the endpoint the browser posts to,
  // and it is the only one that carries the whole path: identity injection, fact
  // retrieval, the tool layer, the phantom-action guards and the daily log. A
  // probe against the bare /api/chat proxy would exercise the model and none of
  // the memory system, which is the part that was just replaced.
  const inst = require(path.join(ROOT, 'db/config')).getProviderInstance(
    cfg.models.chat.provider, cfg.models.chat.instance
  );
  const res = await fetch(`${BASE}/api/chat/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: cfg.models.chat.model,
      provider: cfg.models.chat.provider,
      ollamaHost: inst ? inst.host : undefined,
      messages: [{ role: 'user', content: question }],
      conversation_id: conversationId,
      inputModality: 'typed',
      ttsEnabled: false,
      superSearch: false
    })
  });
  const took = ((Date.now() - t0) / 1000).toFixed(1);
  if (!conversationId) conversationId = res.headers.get('X-Conversation-Id') || null;
  const text = await res.text();
  if (!res.ok) return { question, took, error: `HTTP ${res.status}: ${text.slice(0, 300)}` };
  // The endpoint streams Server-Sent Events, the same as the browser receives —
  // so the answer has to be reassembled from the deltas rather than parsed as one
  // JSON document. Reading it any other way would mean this probe was not
  // exercising the path the UI uses.
  const answer = assembleSSE(text);
  if (!answer.trim()) return { question, took, error: `no content in the stream: ${text.slice(0, 200)}` };
  return { question, took, answer };
}

/** Reassemble an SSE body into the text he actually said. */
function assembleSSE(raw) {
  let out = '';
  for (const line of String(raw).split('\n')) {
    const t = line.trim();
    if (!t.startsWith('data:')) continue;
    const payload = t.slice(5).trim();
    if (!payload || payload === '[DONE]') continue;
    try {
      const j = JSON.parse(payload);
      out += j?.choices?.[0]?.delta?.content
          ?? j?.message?.content
          ?? j?.delta?.content
          ?? j?.content
          ?? '';
    } catch { /* a keepalive or a partial frame — skip it */ }
  }
  return out;
}

(async () => {
  const custom = process.argv.slice(2).filter(a => !a.startsWith('--'));
  const probes = custom.length ? custom : DEFAULT_PROBES;

  const bar = '='.repeat(78);
  console.log(`\n${bar}\nCHAT SMOKE — ${BASE}\n${bar}`);
  let failures = 0;
  for (const q of probes) {
    const r = await ask(q);
    console.log(`\nQ: ${q}`);
    if (r.error) { console.log(`   ERROR (${r.took}s): ${r.error}`); failures++; continue; }
    console.log(`   (${r.took}s)`);
    console.log(`A: ${String(r.answer).trim()}`);
    if (r.toolCalls) console.log(`   tools: ${JSON.stringify(r.toolCalls)}`);
  }
  console.log(`\n${bar}`);
  console.log(failures ? `${failures} probe(s) errored.` : 'All probes answered.');
  console.log(`${bar}\n`);
  process.exit(failures ? 1 : 0);
})().catch(err => { console.error('smoke failed:', err); process.exit(1); });
