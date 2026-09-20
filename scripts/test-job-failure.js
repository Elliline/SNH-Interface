#!/usr/bin/env node
/**
 * "TERMINATED" IS NOT A DIAGNOSIS.
 *
 * On 2026-09-09 a background job's card said `terminated` and nothing else —
 * the word Node's fetch uses for a body cut off mid-stream, written verbatim.
 * It took three logs to establish that SNH's own brain watchdog had restarted
 * the engine underneath a healthy run. db/job-failure.js is the mapping from
 * "what was thrown" to "who stopped it, which clock, in words", and this is
 * the proof that every shape of failure lands on a side and a sentence.
 *
 * Pure: no engine, no database. Usage: node scripts/test-job-failure.js
 */
const path = require('path');
const F = require(path.join(__dirname, '..', 'db/job-failure'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}
const fmt = (ms) => new Date(ms).toISOString().slice(11, 16) + ' UTC';

console.log('\n── The 9/9 case: a stream cut mid-answer ──');
const terminated = new TypeError('terminated');
let v = F.classifyThrown(terminated, { round: 8, calls: 27 });
check('with no restart on record it is the ENGINE side', v.source === 'engine' && v.kind === 'connection-cut', `${v.source}/${v.kind}`);
check('  the sentence never contains the bare word', !/\bterminated\b/.test(v.plain), v.plain);
check('  it says where the run was', /round 8/.test(v.plain) && /27 tool calls/.test(v.plain), v.plain);
check('  the raw text survives in `technical`', v.technical === 'terminated');

v = F.classifyThrown(terminated, { round: 8, calls: 27, recentRestart: { issuedAt: Date.parse('2026-09-10T04:11:05Z'), reason: '3 failed liveness checks in a row, verdict "stalled"' }, formatTime: fmt });
check('with the watchdog\'s restart on record it is the RUNNER side', v.source === 'runner' && v.kind === 'watchdog-restart', `${v.source}/${v.kind}`);
check('  and it names the watchdog, the time and the reason',
  /brain watchdog restarted the engine at 04:11 UTC/.test(v.plain) && /3 failed liveness checks/.test(v.plain), v.plain);
check('  and says the engine was not the problem', /NOT the problem/.test(v.plain), v.plain);

console.log('\n── The runner\'s own clocks ──');
const stall = Object.assign(new Error('agent-job:abc round 3: stalled — no tokens for 106s (limit 105s)'), { name: 'TimeoutError' });
v = F.classifyThrown(stall, { round: 3 });
check('a stall is the runner, kind `stall`', v.source === 'runner' && v.kind === 'stall', `${v.source}/${v.kind}`);
check('  said in seconds, with the limit and where to change it', /106 seconds/.test(v.plain) && /105 seconds/.test(v.plain) && /Settings/.test(v.plain), v.plain);
const first = Object.assign(new Error('x: timed out waiting for the first token after 528s (limit 528s)'), { name: 'TimeoutError' });
v = F.classifyThrown(first);
check('a first-token timeout is the runner, kind `first-token`', v.source === 'runner' && v.kind === 'first-token', `${v.source}/${v.kind}`);
check('  said in minutes and seconds', /8 minutes 48 seconds/.test(v.plain), v.plain);
v = F.classifyThrown(new Error('Brain circuit open — skipping LLM call (engine wedged)'));
check('the circuit breaker is the runner, kind `circuit-open`', v.source === 'runner' && v.kind === 'circuit-open', `${v.source}/${v.kind}`);

console.log('\n── The engine\'s own failures ──');
v = F.classifyThrown(Object.assign(new Error('HTTP 400'), { status: 400, body: '{"error":"context length"}' }));
check('a refusal is the engine, kind `refused`, with the status', v.source === 'engine' && v.kind === 'refused' && /HTTP 400/.test(v.plain), v.plain);
v = F.classifyThrown(Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED' } }));
check('nothing listening is the engine, kind `unreachable`', v.source === 'engine' && v.kind === 'unreachable', `${v.source}/${v.kind}`);
v = F.classifyThrown(Object.assign(new Error('fetch failed'), { cause: { code: 'ECONNRESET', message: 'other side closed' } }));
check('a reset socket is a connection cut', v.kind === 'connection-cut', v.kind);

console.log('\n── The budgets, from the summary the session wrote ──');
v = F.classifyBudgetStop({ exhausted: 'call budget spent (40.75/40 billed over 43 call(s), 3 of them empty or failed)' });
check('the call budget is the runner, kind `call-budget`', v.source === 'runner' && v.kind === 'call-budget', `${v.source}/${v.kind}`);
check('  the door-watch wording survives: what ran out and by how much', /40\.75\/40 billed over 43 call\(s\), 3 of them empty or failed/.test(v.plain), v.plain);
v = F.classifyBudgetStop({ exhausted: 'time budget spent (900s of 900s)' });
check('the clock is the runner, kind `wall-clock`, in minutes', v.kind === 'wall-clock' && /15 minutes/.test(v.plain), v.plain);
v = F.classifyBudgetStop({ exhausted: 'attempt ceiling reached (80/80 calls, 80 of them empty or failed — nothing is coming back, so it stopped trying)' });
check('the attempt ceiling is named as the runaway guard', v.kind === 'attempt-ceiling' && /runaway guard/.test(v.plain), v.plain);
v = F.classifyBudgetStop({ exhausted: 'round budget spent (16 rounds)' }, { maxRounds: 16 });
check('rounds are the runner, kind `rounds`', v.kind === 'rounds' && /16 tool rounds/.test(v.plain), v.plain);
v = F.classifyCutShort({ truncated: true, outOfRounds: true, budget: { exhausted: 'round budget spent (16 rounds)' }, answerTokens: 8192 });
check('truncation wins over every other cut-short reason (it says WHERE the text stops)', v.kind === 'answer-budget' && /8192 tokens/.test(v.plain), v.plain);
check('nothing cut short → null, never a sentence', F.classifyCutShort({}) === null);

console.log('\n── Nothing matched ──');
v = F.classifyThrown(new Error('ENOSPC: no space left on device'));
check('is `unknown`, and SAYS SNH could not place it', v.source === 'unknown' && /could not tell which side/.test(v.plain), v.plain);
check('  with the raw text inside the sentence, quoted', /"ENOSPC: no space left on device"/.test(v.plain), v.plain);

console.log('\n── The label the card leads with ──');
check('runner', F.sourceLabel('runner') === "stopped by SNH's job runner");
check('engine', F.sourceLabel('engine') === 'stopped on the engine side');
check('dispatched', /handed to/.test(F.sourceLabel('dispatched')));
check('unknown never claims a side', /could not place/.test(F.sourceLabel('unknown')));
check('sayDuration: 105000 → "105 seconds"', F.sayDuration(105000) === '105 seconds', F.sayDuration(105000));
check('sayDuration: 14400000 → "4 hours"', F.sayDuration(14400000) === '4 hours', F.sayDuration(14400000));

console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
process.exit(fail ? 1 : 0);
