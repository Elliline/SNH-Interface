#!/usr/bin/env node
/**
 * THE CHAT PATH TELLS A WEDGED ENGINE FROM A SLOW TURN, AND SAYS WHICH.
 *
 * Two failures on 2026-08-22, one morning apart in cause and identical in
 * effect — she sat in front of a dead screen and the system said nothing
 * useful:
 *
 *   1. The engine stopped generating at 07:03. Her turn was killed by our own
 *      120s deadline at 07:09 and arrived as a TimeoutError, which matched
 *      neither arm of the upstream test, answered 500, and skipped the
 *      watchdog block entirely. Bare "The operation was aborted due to timeout".
 *   2. That same flat 120s was the ONLY deadline a turn had, so a turn still
 *      producing tokens died on the same clock as a corpse.
 *
 * Usage: node scripts/test-chat-timeouts.js
 */
const path = require('path');
const http = require('http');
const ROOT = path.join(__dirname, '..');
const { classifyChatFailure, chatFailureBody } = require(path.join(ROOT, 'db/chat-failure'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// ── The classification ────────────────────────────────────────────────────
console.log('\n── A timeout waiting on the engine is UPSTREAM ──');

const timeoutErr = Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' });
const v1 = classifyChatFailure(timeoutErr);
check('the exact 07:09 error is upstream', v1.upstream === true);
check('and answers 502, not 500', v1.status === 502,
  `got ${v1.status} — this is the regression that produced the bare error`);
check('and is recognised specifically as an engine timeout', v1.engineTimeout === true);

const stallErr = Object.assign(new Error('vllm tool-round 1: stalled — no tokens for 61s (limit 60s)'), { name: 'TimeoutError' });
check('streamChat\'s own stall is upstream too', classifyChatFailure(stallErr).status === 502);

const netErr = Object.assign(new Error('fetch failed'), { name: 'TypeError', cause: { code: 'ECONNREFUSED' } });
check('a connection failure is still upstream (unchanged)', classifyChatFailure(netErr).status === 502);

const flagged = Object.assign(new Error('engine said no'), { upstream: true });
check('an explicitly flagged upstream error is still upstream', classifyChatFailure(flagged).status === 502);

const ourBug = new TypeError("Cannot read properties of null (reading 'name')");
check('OUR OWN BUG IS STILL A 500', classifyChatFailure(ourBug).status === 500,
  'a real bug in here must not be dressed up as an engine problem');

// ── The words ─────────────────────────────────────────────────────────────
console.log('\n── And she is told something she can act on ──');

const wedged = { healthy: false, state: 'wedged', message: 'The model engine has stopped responding (about 6 minutes now). I restart it automatically after 3 failed checks; this is 1. Try again in a minute.' };
const b1 = chatFailureBody(timeoutErr, wedged);
check('the watchdog\'s account is what she reads', b1.error === wedged.message);
check('the raw error is kept, not shown', b1.technical === timeoutErr.message);
check('and the state is machine-readable for the UI', b1.brain === 'wedged');

// The window that made the classifier fix insufficient on its own: the
// watchdog needs consecutive failures and reports healthy until it has them.
const b2 = chatFailureBody(timeoutErr, { healthy: true, state: 'ok', message: null });
check('with the watchdog still silent, she STILL gets plain language',
  /stopped responding/i.test(b2.error) && !/aborted due to timeout/i.test(b2.error),
  `got: ${b2.error}`);
check('and that case is labelled unresponsive', b2.brain === 'unresponsive');
check('the raw text survives in technical', b2.technical === timeoutErr.message);

const b3 = chatFailureBody(ourBug, null);
check('an unclassified failure is NOT dressed up as an engine problem',
  b3.error === ourBug.message && b3.brain === undefined,
  'a real bug must read as a real bug');

// ── The two limits actually behave differently ────────────────────────────
console.log('\n── A slow turn survives; a wedged one does not ──');

const { streamChat } = require(path.join(ROOT, 'db/memory-manager'));

/** An engine that emits a token every `gapMs`, `n` times, then finishes. */
function slowEngine(gapMs, n) {
  return http.createServer(async (req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    for (let i = 0; i < n; i++) {
      await new Promise(r => setTimeout(r, gapMs));
      res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: 'tok ' } }] })}\n\n`);
    }
    res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  });
}

/** An engine that sends one token and then goes silent forever. */
function wedgingEngine() {
  return http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: 'starting' } }] })}\n\n`);
    // and then nothing, ever
  });
}

/** An engine that accepts and never sends a byte — the 07:03 state. */
function deadEngine() {
  return http.createServer(() => { /* accept, never respond */ });
}

function listen(server) {
  return new Promise(res => server.listen(0, '127.0.0.1', () => res(server.address().port)));
}

(async () => {
  // Slow but progressing: 12 tokens, 150ms apart = 1.8s total, well past a
  // 400ms flat deadline but never a 400ms GAP.
  {
    const s = slowEngine(150, 12);
    const port = await listen(s);
    const t0 = Date.now();
    let ok = false, err = null;
    try {
      const data = await streamChat({
        url: `http://127.0.0.1:${port}/v1/chat/completions`,
        body: { messages: [] }, openAiStyle: true,
        firstTokenMs: 2000, stallMs: 400, label: 'slow'
      });
      ok = (data.choices[0].message.content || '').trim() === 'tok tok tok tok tok tok tok tok tok tok tok tok';
    } catch (e) { err = e; }
    s.close();
    check('a turn that keeps producing is NOT killed for taking longer than the gap limit',
      ok, err ? `threw: ${err.message}` : 'content did not survive intact');
    check('  (and it really did outlast the stall limit)', Date.now() - t0 > 400);
  }

  // Wedged after the first token: killed by stallMs, not firstTokenMs.
  {
    const s = wedgingEngine();
    const port = await listen(s);
    const t0 = Date.now();
    let name = null, msg = '';
    try {
      await streamChat({
        url: `http://127.0.0.1:${port}/v1/chat/completions`,
        body: { messages: [] }, openAiStyle: true,
        firstTokenMs: 60000, stallMs: 500, label: 'wedged'
      });
    } catch (e) { name = e.name; msg = e.message; }
    const took = Date.now() - t0;
    s.close();
    check('an engine that dies mid-answer is caught by the STALL limit', name === 'TimeoutError', `got ${name}`);
    check('  quickly — not on the first-token clock', took < 5000, `took ${took}ms with firstTokenMs=60000`);
    check('  and says which limit bound', /stalled/i.test(msg), msg);
    check('  and classifies as upstream, so she gets the 502 and the explanation',
      classifyChatFailure({ name, message: msg }).status === 502);
  }

  // Never starts: killed by firstTokenMs, and says so differently.
  {
    const s = deadEngine();
    const port = await listen(s);
    let msg = '', name = null;
    try {
      await streamChat({
        url: `http://127.0.0.1:${port}/v1/chat/completions`,
        body: { messages: [] }, openAiStyle: true,
        firstTokenMs: 500, stallMs: 60000, label: 'dead'
      });
    } catch (e) { name = e.name; msg = e.message; }
    s.close();
    check('an engine that never answers is caught by the FIRST-TOKEN limit', name === 'TimeoutError', `got ${name}`);
    check('  and says that, rather than reporting a stall', /first token/i.test(msg), msg);
  }

  // A refusal must stay distinguishable from a stall, or the forced
  // tool_choice retry turns into a lost turn.
  {
    const s = http.createServer((req, res) => {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end('{"error":"tool_choice not supported"}');
    });
    const port = await listen(s);
    let err = null;
    try {
      await streamChat({
        url: `http://127.0.0.1:${port}/v1/chat/completions`,
        body: { messages: [] }, openAiStyle: true,
        firstTokenMs: 5000, stallMs: 5000, label: 'refusal'
      });
    } catch (e) { err = e; }
    s.close();
    check('a REFUSAL carries its status, so the round can be retried unforced',
      err && err.status === 400, `status was ${err && err.status}`);
    check('  and its body, for the log and the ops line',
      err && /tool_choice not supported/.test(err.body || ''));
    check('  and is NOT mistaken for a stall', err && err.name !== 'TimeoutError');
  }

  // ── The config is real and defensive ───────────────────────────────────
  console.log('\n── The knobs ──');
  const { getConfig } = require(path.join(ROOT, 'db/config'));
  const c = getConfig().chat || {};
  check('chat.stallTimeoutMs ships with a value', Number.isFinite(c.stallTimeoutMs) && c.stallTimeoutMs > 0);
  check('chat.firstTokenTimeoutMs ships with a value', Number.isFinite(c.firstTokenTimeoutMs) && c.firstTokenTimeoutMs > 0);
  check('a person waiting is given a shorter leash than a background job',
    c.firstTokenTimeoutMs < (getConfig().generation || {}).firstTokenTimeoutMs,
    'chat should not wait as long as a queued background run');

  console.log(`\n=== ${pass} passed, ${fail} failed ===`);
  process.exit(fail ? 1 : 0);
})();
