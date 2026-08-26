#!/usr/bin/env node
/**
 * A BLANK BUBBLE IS NOT AN ANSWER, AND IT IS NOT SILENT ANY MORE.
 *
 * On 2026-08-26 Juno produced three empty assistant turns in one conversation.
 * Two of them left NOTHING behind: no text on the wire, no row in `messages`,
 * no line in the log. The engine had returned `finish_reason: stop` with zero
 * content and zero reasoning (vLLM: 0.1 tokens/s against a full prefill, 09:32:13
 * and 09:47:30), and server.js's existing guard for "thought but never answered"
 * required the reasoning to be non-empty — so the wholly-empty case fell through
 * it, and then through the `if (fullResponse)` save below it.
 *
 * This asserts the two halves of the fix: the classifier that stops treating
 * silence as an answer, and the wiring that gives it exactly one retry.
 *
 * Usage: node scripts/test-empty-reply-retry.js
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');
const R = require(path.join(ROOT, 'db/empty-reply-retry'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// ── The classification ────────────────────────────────────────────────────
console.log('\n── What counts as empty ──');

check('a real answer is not empty', R.classifyEmptyReply('Here you go.', '') === null);
check('an answer with reasoning beside it is not empty',
  R.classifyEmptyReply('Here you go.', 'thinking...') === null);

check('THE 2026-08-15 CASE: all thinking, no answer',
  R.classifyEmptyReply('', 'eight thousand characters of thinking') === 'reasoning-only');

check('THE 2026-08-26 CASE: nothing at all',
  R.classifyEmptyReply('', '') === 'nothing-at-all',
  'this is the one that used to fall through the guard and vanish');

check('  and whitespace is nothing, on both channels',
  R.classifyEmptyReply('   \n\t ', ' \n ') === 'nothing-at-all');
check('  a whitespace-only answer with real thinking is still reasoning-only',
  R.classifyEmptyReply('  \n ', 'real thinking') === 'reasoning-only');
check('  null/undefined do not throw', R.classifyEmptyReply(null, undefined) === 'nothing-at-all');

// ── The sentence sent when the retry did not save it ──────────────────────
console.log('\n── What she is told when it stays broken ──');

const ro = R.noteForEmptyReply('reasoning-only');
const na = R.noteForEmptyReply('nothing-at-all');
check('the two cases say different things', ro !== na);
check('neither is empty — an empty message reads as answered',
  ro.trim().length > 40 && na.trim().length > 40);
check('both say it happened more than once, because it did (one retry ran)',
  /twice|two/i.test(ro) && /twice|two/i.test(na));
check('the nothing-at-all note does not blame her',
  /not (anything|something) you did|fault on my side/i.test(na));

// ── Reading a streamChat result, in either wire shape ─────────────────────
console.log('\n── Reading the retry back ──');

const openAi = { choices: [{ message: { role: 'assistant', content: 'answered', reasoning: 'thought' }, finish_reason: 'stop' }] };
const ollama = { message: { role: 'assistant', content: 'answered' }, done_reason: 'stop' };
check('OpenAI shape (vllm/llamacpp)', R.replyFromStreamResult(openAi).content === 'answered');
check('  and its reasoning', R.replyFromStreamResult(openAi).reasoning === 'thought');
check('Ollama shape', R.replyFromStreamResult(ollama).content === 'answered');
check('  with no reasoning channel, that is an empty string not undefined',
  R.replyFromStreamResult(ollama).reasoning === '');
check('an empty completion reads as empty, not as a crash',
  R.replyFromStreamResult({ choices: [{ message: { role: 'assistant', content: '' } }] }).content === '');
check('a garbage body does not throw', R.replyFromStreamResult(null).content === '');
check('  and classifies as nothing-at-all',
  R.classifyEmptyReply(...Object.values(R.replyFromStreamResult(undefined))) === 'nothing-at-all');

// ── The wiring in server.js ───────────────────────────────────────────────
console.log('\n── The wiring ──');
const src = fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8');

check('server.js uses the shared classifier rather than its own condition',
  /emptyReplyRetry\.classifyEmptyReply\(fullResponse, fullReasoning\)/.test(src));

check('THE REGRESSION IS GONE: no branch still requires reasoning to notice an empty reply',
  !/!fullResponse\.trim\(\)\s*&&\s*fullReasoning\.trim\(\)/.test(src),
  'the old guard let the wholly-empty case through');

check('a retry request is prepared for the vllm/llamacpp path',
  /retryRequest = \{[\s\S]{0,200}v1\/chat\/completions/.test(src));
check('a retry request is prepared for the ollama path',
  /retryRequest = \{[\s\S]{0,120}\/api\/chat/.test(src));
check('it is seeded null, so a provider we cannot re-ask falls back to the note',
  /let retryRequest = null;/.test(src));

const retryCalls = (src.match(/empty-reply retry/g) || []).length;
check('EXACTLY ONE retry call site — no loop, no second attempt', retryCalls === 1,
  `found ${retryCalls}`);
check('and no loop construct wraps it',
  !/(while|for)\s*\([^)]*\)\s*\{[^}]*empty-reply retry/.test(src));

check('the retry is capped by the same chat timeouts as the turn it rescues',
  /label: `\$\{providerType\} empty-reply retry`/.test(src) && /firstTokenMs: rt\.firstTokenMs/.test(src));
check('a thrown retry does not take out the turn',
  /catch \(retryErr\)[\s\S]{0,400}the retry call failed/.test(src));

check('every occurrence is logged — the failures',
  /EMPTY REPLY NOT RECOVERED/.test(src));
check('  and the rescues, so recurrence stays measurable',
  /EMPTY REPLY RESCUED/.test(src));
check('  and it reaches the ops log, not only the journal',
  /appendToOpsLog\(\s*`EMPTY REPLY/.test(src));

check('the rescued text is stored, so the turn is in the transcript',
  /fullResponse = text;/.test(src));



// ── A MODEL THAT IS THINKING IS NOT STALLED ───────────────────────────────
//
// The 49800ms chat.stallTimeoutMs was the first suspect for the 2026-08-26
// blanks and the evidence cleared it: no stall or first-token kill appears
// anywhere in the journal, and successful turns of 113.9s and 549s that same
// day were never touched. The reason they were never touched is this property
// — a reasoning delta counts as progress — and it is worth pinning, because a
// stall timer that ignored the reasoning channel WOULD kill a healthy turn
// that thinks quietly past the window before its first content token. That is
// the bug this section exists to stop anyone reintroducing.
(async () => {
  console.log('\n── Reasoning deltas reset the stall timer ──');
  const http = require('http');
  const { streamChat } = require(path.join(ROOT, 'db/memory-manager'));
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const thinker = http.createServer(async (req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const send = o => res.write(`data: ${JSON.stringify(o)}\n\n`);
    send({ choices: [{ delta: { role: 'assistant' }, finish_reason: null }] });
    // Six reasoning-only deltas, each landing well after the stall window
    // would have expired if reasoning did not count. No content the whole time.
    for (let i = 0; i < 6; i++) {
      await sleep(120);
      send({ choices: [{ delta: { reasoning: `thinking ${i} ` }, finish_reason: null }] });
    }
    await sleep(120);
    send({ choices: [{ delta: { content: 'and here is the answer' }, finish_reason: null }] });
    send({ choices: [{ delta: {}, finish_reason: 'stop' }] });
    res.write('data: [DONE]\n\n');
    res.end();
  });
  await new Promise(r => thinker.listen(0, '127.0.0.1', r));

  try {
    // stallMs deliberately SHORTER than the gap between deltas: if reasoning
    // did not reset the clock, this call would be killed before the answer.
    const data = await streamChat({
      url: `http://127.0.0.1:${thinker.address().port}/v1/chat/completions`,
      openAiStyle: true, body: { model: 'stub', messages: [] },
      firstTokenMs: 5000, stallMs: 80, label: 'reasoning-progress probe'
    });
    const r = R.replyFromStreamResult(data);
    check('a stream of reasoning-only deltas is NOT killed by an 80ms stall limit',
      r.content === 'and here is the answer',
      'the reasoning channel has stopped counting as progress — a quiet thinker would be killed');
    check('  and the reasoning it streamed is kept', /thinking 5/.test(r.reasoning));
  } catch (err) {
    check('a stream of reasoning-only deltas is NOT killed by an 80ms stall limit', false, `killed: ${err.message}`);
    check('  and the reasoning it streamed is kept', false, 'the call did not complete');
  }

  // The other half of the contract: genuine silence IS still caught.
  const wedged = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write('data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n');
    // ...and then nothing, forever.
  });
  await new Promise(r => wedged.listen(0, '127.0.0.1', r));
  try {
    await streamChat({
      url: `http://127.0.0.1:${wedged.address().port}/v1/chat/completions`,
      openAiStyle: true, body: { model: 'stub', messages: [] },
      firstTokenMs: 400, stallMs: 400, label: 'wedged probe'
    });
    check('a genuinely silent engine is still killed', false, 'it returned instead of throwing');
  } catch (err) {
    check('a genuinely silent engine is still killed', err.name === 'TimeoutError', err.message);
  }

  thinker.close(); wedged.close();
  console.log(`\n=== ${pass} passed, ${fail} failed ===`);
  process.exit(fail ? 1 : 0);
})();
