#!/usr/bin/env node
/**
 * Is the shape we send after a tool round accepted by THIS model's template, and
 * does the answer come back clean?
 *
 * Two different failures live here and they are invisible to each other:
 *
 *   A 400 from the template. Qwen3's chat template raises 'System message must
 *   be at the beginning.' for any system message where `not loop.first`, so the
 *   old [system, …, user, system] shape was rejected outright. That is what
 *   foldSystemMessages exists for.
 *
 *   A leak into the answer. Folding put the post-tool nudge in front of the
 *   conversation, leaving a tool result as the last thing before generation.
 *   Gemma answered by opening a thought channel, and on an engine with no
 *   --reasoning-parser the marker lands in `content` — spoken, transcribed and
 *   embedded. Measured on Sparky: 6/6 fully folded, 0/6 with the nudge trailing.
 *
 * The shipped shape has to clear BOTH, and the two boxes fail in opposite
 * directions, so neither one alone proves it. This runs every candidate shape
 * against whatever model this box has configured and reports which clear both.
 *
 * SHIPPED is the shape server.js actually sends: one leading system message,
 * with the post-tool nudge as a trailing USER turn.
 *
 * READ-ONLY. It talks to the engine and nothing else — no memory is read or
 * written, and the tool result below is a fixed literal. Safe on a live box.
 *
 * Usage: node scripts/test-fold-shape.js [--runs 6]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

const runsArg = process.argv.indexOf('--runs');
const RUNS = runsArg > -1 ? Number(process.argv[runsArg + 1]) : 6;

const IDENTITY = 'You are a self-hosted assistant with a persistent long-term memory. Talk plainly.';
const CAPABILITY = 'Your built-in capabilities — ground truth for what your system can do. This list is EXHAUSTIVE.';
const MEMORY = 'What you currently hold about the user (excerpt, chosen by relevance):\n- The user has four dogs.';
const CLOCK = 'Current date and time: Sunday, August 16, 2026, 10:30 AM Pacific.';
const GUARD = 'The user is asking about what you hold in your own memory. Call the right tool now. Do NOT say you searched anything unless the tool call actually ran.';
const NUDGE = 'Tool calls are complete. Now provide your response to the user based on the information gathered.';

const USER = { role: 'user', content: 'What do you remember about my dogs?' };
const TOOL_CALL = {
  role: 'assistant',
  content: '',
  tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'memory_search', arguments: '{"query":"dogs"}' } }],
};
const TOOL_RESULT = {
  role: 'tool',
  tool_call_id: 'call_1',
  content: JSON.stringify({ results: [{ text: 'The user has four dogs.' }, { text: 'Roscoe is one of the dogs.' }], shown: 2, total: 2 }),
};

const SHAPES = {
  'pre-fold: many system, nudge trailing system': () => ([
    { role: 'system', content: IDENTITY }, { role: 'system', content: CAPABILITY },
    { role: 'system', content: MEMORY }, { role: 'system', content: CLOCK },
    { role: 'system', content: GUARD }, USER, TOOL_CALL, TOOL_RESULT,
    { role: 'system', content: NUDGE },
  ]),
  'fully folded: nudge inside leading block': () => ([
    { role: 'system', content: [IDENTITY, CAPABILITY, MEMORY, CLOCK, GUARD, NUDGE].join('\n\n') },
    USER, TOOL_CALL, TOOL_RESULT,
  ]),
  'SHIPPED: folded + nudge as trailing user': () => ([
    { role: 'system', content: [IDENTITY, CAPABILITY, MEMORY, CLOCK, GUARD].join('\n\n') },
    USER, TOOL_CALL, TOOL_RESULT, { role: 'user', content: NUDGE },
  ]),
};

/**
 * A channel marker that reached `content`. Both the raw token form and the
 * bare-word remnant, because a non-streaming response strips the angle brackets
 * and leaves "thought" sitting at the front of the reply.
 */
function leaked(text) {
  return /<\|?channel/i.test(text) || /^\s*(thought|analysis)\b/i.test(text);
}

(async () => {
  const cfg = getConfig();
  const chat = cfg.models.chat;
  const inst = getProviderInstance(chat.provider, chat.instance);
  const host = inst ? inst.host : 'http://localhost:7070';

  console.log(`engine : ${host}`);
  console.log(`model  : ${chat.model}  (provider ${chat.provider})`);
  console.log(`runs   : ${RUNS} per shape\n`);

  const summary = [];
  for (const [label, build] of Object.entries(SHAPES)) {
    let rejected = 0, leaks = 0, ok = 0;
    let firstErr = null, sample = null;
    for (let i = 0; i < RUNS; i++) {
      let res;
      try {
        res = await fetch(`${host}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: chat.model, messages: build(), stream: false }),
          signal: AbortSignal.timeout(120000),
        });
      } catch (e) { firstErr = firstErr || String(e.message); rejected++; continue; }

      if (!res.ok) {
        rejected++;
        if (!firstErr) firstErr = `HTTP ${res.status}: ${(await res.text()).slice(0, 160)}`;
        continue;
      }
      const data = await res.json();
      const msg = data.choices?.[0]?.message || {};
      const content = String(msg.content ?? '');
      // A reasoning model with a parser puts thinking HERE, which is correct and
      // is not a leak. Only content is judged.
      if (leaked(content)) leaks++; else ok++;
      if (!sample) sample = JSON.stringify(content.slice(0, 60));
    }
    summary.push({ label, rejected, leaks, ok });
    console.log(`── ${label}`);
    console.log(`     rejected by template : ${rejected}/${RUNS}${firstErr ? `   (${firstErr})` : ''}`);
    console.log(`     channel leaked into answer : ${leaks}/${RUNS}`);
    console.log(`     clean : ${ok}/${RUNS}${sample ? `   e.g. ${sample}` : ''}\n`);
  }

  const shipped = summary.find(s => s.label.startsWith('SHIPPED'));
  const good = shipped && shipped.rejected === 0 && shipped.leaks === 0;
  console.log('=== VERDICT ===');
  console.log(`  Shipped shape on ${chat.model}: ${good ? 'OK — accepted and clean' : 'PROBLEM'}`);
  if (!good && shipped) {
    console.log(`    rejected ${shipped.rejected}/${RUNS}, leaked ${shipped.leaks}/${RUNS}`);
  }
  process.exit(good ? 0 : 1);
})().catch(e => { console.error('probe failed:', e.message); process.exit(2); });
