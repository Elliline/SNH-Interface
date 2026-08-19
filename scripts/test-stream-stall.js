#!/usr/bin/env node
/**
 * The streaming layer: tool-call reassembly, and a clock that measures GAPS.
 *
 * Both halves are invisible from outside and expensive to get wrong. The
 * reassembly runs on every agent job, every scheduled run and the corrector —
 * the one background step with write access — and a bug in it does not look like
 * a bug, it looks like a model that stopped calling tools. The stall clock
 * replaced a timeout derived from an assumed tokens-per-second rate, which got
 * stricter the more agents ran; the whole point is that the new one does not
 * care how many are running, and that is a property you can only test by
 * controlling the timing of the bytes.
 *
 * So the engine is a STUB: `fetch` is replaced with one that returns a stream
 * this file writes, chunk by chunk, on a schedule it chooses. No brain, no
 * network, and the failure cases (a stall, a stream that never starts, a burst)
 * are reachable on purpose rather than by luck.
 *
 * Usage: node scripts/test-stream-stall.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-stream-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
require(path.join(ROOT, 'db/database')).initDatabase();
const { streamChat } = require(path.join(ROOT, 'db/memory-manager'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Serve a scripted stream. `script` is [{ afterMs, text }] — `text` goes out
 * verbatim, so a test can send SSE comments and malformed lines as easily as
 * data frames. `neverEnd` holds the stream open after the script, which is how
 * a stall is distinguished from a stream that simply finished.
 */
function stubFetch(script, { neverEnd = false, status = 200 } = {}) {
  global.fetch = async (_url, init) => {
    if (status !== 200) return { ok: false, status, body: null };
    const signal = init && init.signal;
    const stream = new ReadableStream({
      async start(controller) {
        const enc = new TextEncoder();
        try {
          for (const step of script) {
            await sleep(step.afterMs || 0);
            if (signal && signal.aborted) { controller.close(); return; }
            controller.enqueue(enc.encode(step.text));
          }
          if (neverEnd) {
            // Hold it open until the caller gives up. This is a wedged engine:
            // the socket is fine, the bytes stopped. On abort the real fetch
            // makes the reader REJECT, so the stub does too — a stub that closed
            // cleanly here would have hidden the bug this found on first run.
            await new Promise((resolve, reject) => {
              if (signal) signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
            });
          }
          controller.close();
        } catch (e) {
          try { controller.error(e); } catch { /* already closed */ }
        }
      }
    });
    return { ok: true, status: 200, body: stream };
  };
}

const sse = (obj) => `data: ${JSON.stringify(obj)}\n\n`;
const chunk = (delta, finish) => sse({ choices: [{ delta, finish_reason: finish ?? null }] });
const FAST = { firstTokenMs: 5000, stallMs: 5000 };

(async () => {
  console.log(`\nStreaming + stall tests (throwaway data dir: ${TMP})\n`);

  // =========================================================================
  console.log('── Content and reasoning accumulate, in the non-streaming shape ──');
  stubFetch([
    { afterMs: 1, text: chunk({ role: 'assistant' }) },
    { afterMs: 1, text: chunk({ reasoning: 'let me think' }) },
    { afterMs: 1, text: chunk({ content: 'Two facts ' }) },
    { afterMs: 1, text: chunk({ content: 'merged.' }) },
    { afterMs: 1, text: chunk({}, 'stop') },
    { afterMs: 1, text: 'data: [DONE]\n\n' }
  ]);
  let r = await streamChat({ url: 'x', body: {}, openAiStyle: true, ...FAST });
  check('content is joined in order', r.choices[0].message.content === 'Two facts merged.',
    JSON.stringify(r.choices[0].message.content));
  check('reasoning is kept separate, never folded into content',
    r.choices[0].message.reasoning === 'let me think' && !/think/.test(r.choices[0].message.content));
  check('finish_reason survives', r.choices[0].finish_reason === 'stop');
  check('the shape is what a NON-streaming reader expects — choices[0].message',
    !!r.choices && !!r.choices[0].message && r.choices[0].message.role === 'assistant');

  // =========================================================================
  console.log('\n── A tool call split across chunks is put back together ──');
  // This is how vLLM actually sends one: name once, then the JSON in pieces
  // that are not valid JSON on their own.
  stubFetch([
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'web_search', arguments: '' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, function: { arguments: '{"qu' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, function: { arguments: 'ery":"octo' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, function: { arguments: 'pus cognition"' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, function: { arguments: '}' } }] }) },
    { afterMs: 1, text: chunk({}, 'tool_calls') },
    { afterMs: 1, text: 'data: [DONE]\n\n' }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: true, ...FAST });
  const calls = r.choices[0].message.tool_calls;
  check('exactly one call came out of five fragments', calls && calls.length === 1, JSON.stringify(calls));
  check('the id is kept', calls[0].id === 'call_a');
  check('the name is kept', calls[0].function.name === 'web_search');
  check('the arguments reassemble into VALID json — the thing that breaks silently',
    JSON.parse(calls[0].function.arguments).query === 'octopus cognition',
    calls[0].function.arguments);

  // =========================================================================
  console.log('\n── Two calls at once, interleaved, stay separate and in order ──');
  stubFetch([
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, id: 'a', function: { name: 'memory_search', arguments: '{"q":' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 1, id: 'b', function: { name: 'web_search', arguments: '{"query"' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 1, function: { arguments: ':"cars"}' } }] }) },
    { afterMs: 1, text: chunk({ tool_calls: [{ index: 0, function: { arguments: '"roscoe"}' } }] }) },
    { afterMs: 1, text: chunk({}, 'tool_calls') }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: true, ...FAST });
  const two = r.choices[0].message.tool_calls;
  check('both calls survive interleaving', two.length === 2, JSON.stringify(two));
  check('they come back in index order, not arrival order',
    two[0].id === 'a' && two[1].id === 'b', two.map(c => c.id).join(','));
  check('neither call stole the other\'s arguments',
    JSON.parse(two[0].function.arguments).q === 'roscoe' &&
    JSON.parse(two[1].function.arguments).query === 'cars',
    JSON.stringify(two.map(c => c.function.arguments)));

  // =========================================================================
  console.log('\n── Truncation still reports itself through the stream ──');
  stubFetch([
    { afterMs: 1, text: chunk({ content: 'def parse(line):' }) },
    { afterMs: 1, text: chunk({}, 'length') }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: true, ...FAST });
  check('finish_reason length arrives, so the partial-result work still fires',
    r.choices[0].finish_reason === 'length');

  // =========================================================================
  console.log('\n── A BURST is not a stall — gaps are what is measured ──');
  // Fifty tokens, a long quiet, fifty more. A rate-based limit fails this; a gap
  // limit does not, and this is the case that made the old design punish load.
  stubFetch([
    { afterMs: 10, text: chunk({ content: 'first burst ' }) },
    { afterMs: 220, text: chunk({ content: 'second burst' }) },
    { afterMs: 10, text: chunk({}, 'stop') }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: true, firstTokenMs: 3000, stallMs: 400 });
  check('a 220ms gap under a 400ms limit completes normally',
    r.choices[0].message.content === 'first burst second burst', JSON.stringify(r.choices[0].message.content));

  // =========================================================================
  console.log('\n── A stall is caught, and named ──');
  stubFetch([
    { afterMs: 5, text: chunk({ content: 'started fine' }) }
  ], { neverEnd: true });
  let err = null;
  const t0 = Date.now();
  try {
    await streamChat({ url: 'x', body: {}, openAiStyle: true, firstTokenMs: 5000, stallMs: 400, label: 'job-1' });
  } catch (e) { err = e; }
  const stallMs = Date.now() - t0;
  check('a stream that goes quiet is abandoned', !!err, 'it returned instead of throwing');
  check('the error says it STALLED, not that it was slow', /stalled/i.test(err.message), err && err.message);
  check('it names the limit that bound', /limit 0s|limit 1s/.test(err.message) || /no tokens for/.test(err.message), err && err.message);
  check('it carries the caller label, so a log says which job', /job-1/.test(err.message), err && err.message);
  check('it is typed TimeoutError, so the circuit breaker still counts a wedge',
    err.name === 'TimeoutError', err && err.name);
  check('and it gave up promptly rather than waiting out a long budget',
    stallMs < 3000, `${stallMs}ms`);

  // =========================================================================
  console.log('\n── Before the first token, silence is normal — and separately bounded ──');
  stubFetch([], { neverEnd: true });
  err = null;
  try {
    await streamChat({ url: 'x', body: {}, openAiStyle: true, firstTokenMs: 400, stallMs: 60000 });
  } catch (e) { err = e; }
  check('a stream that never starts is abandoned on the FIRST-TOKEN limit', !!err);
  check('and says so — a queue wait is not a stall',
    /first token/i.test(err.message) && !/stalled/i.test(err.message), err && err.message);

  // A long queue wait followed by real tokens must NOT be killed: this is the
  // case that matters when many agents are running and requests sit in `waiting`.
  stubFetch([
    { afterMs: 300, text: chunk({ content: 'queued, then answered' }) },
    { afterMs: 5, text: chunk({}, 'stop') }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: true, firstTokenMs: 2000, stallMs: 200 });
  check('a 300ms wait for the first token passes under a 200ms STALL limit — the two clocks are separate',
    r.choices[0].message.content === 'queued, then answered', JSON.stringify(r.choices[0].message.content));

  // =========================================================================
  console.log('\n── Keep-alives do not count as progress ──');
  // An engine emitting heartbeats while producing nothing is the stall, not the
  // cure. SSE comments and empty deltas must not reset the clock.
  stubFetch([
    { afterMs: 5, text: chunk({ content: 'begun' }) },
    { afterMs: 120, text: ': keep-alive\n\n' },
    { afterMs: 120, text: chunk({}) },
    { afterMs: 120, text: ': keep-alive\n\n' },
    { afterMs: 120, text: chunk({}) }
  ], { neverEnd: true });
  err = null;
  try {
    await streamChat({ url: 'x', body: {}, openAiStyle: true, firstTokenMs: 5000, stallMs: 300 });
  } catch (e) { err = e; }
  check('a stream of keep-alives with no tokens still stalls', !!err && /stalled/i.test(err.message),
    err ? err.message : 'it completed, so keep-alives reset the clock');

  // =========================================================================
  console.log('\n── The Ollama wire shape works too ──');
  stubFetch([
    { afterMs: 1, text: JSON.stringify({ message: { content: 'ollama ' }, done: false }) + '\n' },
    { afterMs: 1, text: JSON.stringify({ message: { content: 'answer' }, done: false }) + '\n' },
    { afterMs: 1, text: JSON.stringify({ message: {}, done: true, done_reason: 'stop' }) + '\n' }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: false, ...FAST });
  check('NDJSON accumulates on message.content', r.message.content === 'ollama answer', JSON.stringify(r.message));
  check('and reports done_reason where the old reader looked for it', r.done_reason === 'stop');

  // =========================================================================
  console.log('\n── A chunk split mid-line is not lost ──');
  const whole = chunk({ content: 'split across packets' });
  stubFetch([
    { afterMs: 1, text: whole.slice(0, 12) },
    { afterMs: 5, text: whole.slice(12) },
    { afterMs: 1, text: chunk({}, 'stop') }
  ]);
  r = await streamChat({ url: 'x', body: {}, openAiStyle: true, ...FAST });
  check('a frame arriving in two packets is buffered and parsed once whole',
    r.choices[0].message.content === 'split across packets', JSON.stringify(r.choices[0].message.content));

  console.log('\n── A dead endpoint is still an ordinary error ──');
  stubFetch([], { status: 503 });
  err = null;
  try { await streamChat({ url: 'x', body: {}, openAiStyle: true, ...FAST }); } catch (e) { err = e; }
  check('HTTP failure throws with the status, not a stall', !!err && /503/.test(err.message), err && err.message);
  check('and is NOT typed as a timeout — the engine answered, it just said no',
    err.name !== 'TimeoutError', err && err.name);

  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
