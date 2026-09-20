#!/usr/bin/env node
/**
 * EVERY WAY A BACKGROUND JOB CAN STOP, CAUSED FOR REAL, AND WHAT THE CARD SAYS.
 *
 * The 2026-09-09 card said "terminated" and nothing else; Ellie could not route
 * the bug because the system did not say which side it happened on. This suite
 * drives the REAL runner — db/agent-jobs.js through memory-manager's real tool
 * loop and streamChat — against a scripted engine that misbehaves on demand,
 * and asserts on the fields the card reads: status, stop_source, stop_kind,
 * the plain sentence, the partial output, the checkpoint on disk.
 *
 * What is caused, not simulated:
 *   - a stall: the engine sends a token and goes silent past the stall limit
 *   - the wall clock: the job's time limit binds while it is still asking
 *   - a provider down: a tool throws, then answers on the free retry; or never
 *   - the engine cut off mid-answer, with and without the watchdog's restart
 *   - nothing listening at the engine
 *   - a KILLED PROCESS mid-run, and a restart that resumes from the checkpoint
 *   - the budget ask, answered yes, no, and neither
 *   - a retry from a partial, whose brief carries the last attempt
 *
 * Model calls are real HTTP to the stub; only the classifier that reads her
 * yes/no is injected, because that one bit is not what is under test here.
 *
 * Refuses to run without SNH_DATA_DIR — it writes jobs, conversations, bell
 * items and checkpoint files. Tears the directory down at exit.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-job-failures.js
 */
process.env.TZ = 'America/Los_Angeles';
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-job-failures.js');
  process.exit(1);
}
const TMP = process.env.SNH_DATA_DIR;
const ROOT = path.join(__dirname, '..');
const ROLE = process.env.JOBTEST_ROLE || 'main';   // main | child-run | child-restart

// ---------------------------------------------------------------------------
// Config stub — installed BEFORE memory-manager is required (it destructures
// getConfig at load). `over` is mutated per test; every read sees the latest.
// data/config.json is NOT redirected by SNH_DATA_DIR, so nothing here writes it.
// ---------------------------------------------------------------------------
const config = require(path.join(ROOT, 'db/config'));
const realGetConfig = config.getConfig;
const over = {
  enginePort: Number(process.env.JOBTEST_ENGINE_PORT || 0),
  agentJobs: {},
  generation: {},
  toolBudget: {}
};
config.getConfig = () => {
  const c = realGetConfig();
  return {
    ...c,
    models: { ...c.models, heartbeat: { provider: 'vllm', instance: 'stub', model: 'stub-model' } },
    agentJobs: Object.assign({
      enabled: true, maxQueued: 10, maxStartsPerHour: 100,
      maxToolCallsPerJob: 40, maxWallClockMs: 900000, maxRoundsPerJob: 16,
      maxAttempts: 2, retryGraceMinutes: 30, retentionDays: 90,
      askBeforeCeiling: false, askAtPercent: 80, extensionPercent: 50,
      // Phases and compaction are scripts/test-job-phases.js's subject. Off
      // here so every engine request in these scripts is the one the failure
      // under test expects — the planner would be one more.
      phases: { enabled: false }, compaction: { enabled: false }
    }, over.agentJobs),
    generation: Object.assign({}, c.generation, {
      agentJobResponseTokens: 512, agentJobThinkingTokens: null,
      stallTimeoutMs: 20000, firstTokenTimeoutMs: 20000
    }, over.generation),
    heartbeat: { ...c.heartbeat, toolBudget: Object.assign({}, c.heartbeat.toolBudget, { failedCallRetries: 1, failedCallCost: 0.25, attemptCeilingMultiple: 2 }, over.toolBudget) },
    // Off: the watchdog and the liveness loop are not under test, and the
    // circuit breaker must not open across tests that deliberately time out.
    watchdog: { ...(c.watchdog || {}), enabled: false },
    brainCircuit: { consecutiveTimeoutsToOpen: 1000 },
    agentPool: c.agentPool
  };
};
config.getProviderInstance = () => ({ host: `http://127.0.0.1:${over.enginePort}` });

const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();
const mm = require(path.join(ROOT, 'db/memory-manager'));
const agentJobs = require(path.join(ROOT, 'db/agent-jobs'));
const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
const budgetAsk = require(path.join(ROOT, 'db/job-budget-ask'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${String(detail).slice(0, 300)}` : ''}`); }
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const job = (id) => db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id);
async function settle(id, ms = 30000, statuses = agentJobs.TERMINAL) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    const j = job(id);
    if (j && statuses.includes(j.status)) return j;
    await sleep(25);
  }
  return job(id);
}

// ---------------------------------------------------------------------------
// The scripted engine. Each POST /v1/chat/completions consumes the next step
// of `script`; an empty script answers with plain text. Every request's body
// is kept so a test can read what the model was shown.
// ---------------------------------------------------------------------------
const engine = { script: [], requests: [], server: null, port: 0 };
function sse(res, obj) { res.write(`data: ${JSON.stringify(obj)}\n\n`); }
function toolDelta(calls) {
  return { choices: [{ delta: { tool_calls: calls.map((c, i) => ({ index: i, id: `call_${Date.now()}_${i}`, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args || {}) } })) }, finish_reason: null }] };
}
async function serve(req, res) {
  // The runner probes GET /v1/models once per process for the context window
  // (db/job-window). That is not a scripted step: a stub that knows no window
  // answers 404, and the runner runs without one — the request it always sent.
  if (req.method !== 'POST' || !/\/chat\/completions/.test(req.url || '')) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ error: 'not a chat completion' }));
  }
  let raw = '';
  for await (const chunk of req) raw += chunk;
  let body = {};
  try { body = JSON.parse(raw); } catch { /* not json */ }
  engine.requests.push(body);
  const step = engine.script.length ? engine.script.shift() : { text: 'done.' };
  if (typeof step === 'function') return step(req, res, body);
  if (step.refuse) { res.writeHead(step.refuse, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify({ error: 'refused by the stub' })); }
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  for (let i = 0; i < (step.preamble || 0); i++) {
    await sleep(step.gapMs || 100);
    sse(res, { choices: [{ delta: { reasoning: 'hm ' }, finish_reason: null }] });
  }
  if (step.stall) { sse(res, { choices: [{ delta: { content: 'starting' }, finish_reason: null }] }); return; /* silent forever */ }
  if (step.cut) { sse(res, { choices: [{ delta: { content: 'half an ans' }, finish_reason: null }] }); await sleep(50); return req.socket.destroy(); }
  if (step.tools) {
    sse(res, toolDelta(step.tools));
    sse(res, { choices: [{ delta: {}, finish_reason: 'tool_calls' }] });
  } else {
    sse(res, { choices: [{ delta: { content: step.text || 'done.' }, finish_reason: null }] });
    sse(res, { choices: [{ delta: {}, finish_reason: step.length ? 'length' : 'stop' }] });
  }
  res.write('data: [DONE]\n\n');
  res.end();
}
function startEngine() {
  return new Promise(resolve => {
    engine.server = http.createServer((req, res) => { serve(req, res).catch(() => { try { res.end(); } catch { /* gone */ } }); });
    engine.server.listen(0, '127.0.0.1', () => { engine.port = engine.server.address().port; over.enginePort = engine.port; resolve(engine.port); });
  });
}

// ---------------------------------------------------------------------------
// Tools: the real registry, with executeTool scripted per name so a provider
// can be down on demand. memory_count is left real (SQLite, cheap, productive).
// ---------------------------------------------------------------------------
const client = MCPClient.shared();
const realExecute = client.executeTool.bind(client);
const toolScript = { memory_search: [] };   // queue of () => result | throw
client.executeTool = async (name, args, ctx) => {
  if (toolScript[name] && toolScript[name].length) {
    const step = toolScript[name].shift();
    return step();
  }
  return realExecute(name, args, ctx);
};
const systemMessageOf = (req) => (req.messages || []).find(m => m.role === 'system')?.content || '';
const lastUserMessageOf = (req) => [...(req.messages || [])].reverse().find(m => m.role === 'user')?.content || '';

// ===========================================================================
// CHILD ROLES — a separate process so a job can genuinely be killed mid-run,
// and a genuinely new process can pick it up.
// ===========================================================================
if (ROLE === 'child-run') {
  (async () => {
    const started = agentJobs.enqueue({ title: 'the one that gets killed', task: 'Count things, then count again, then hang.', conversationId: null });
    process.send && process.send({ id: started.id });
    await settle(started.id, 60000);
    process.exit(0);
  })();
} else if (ROLE === 'child-restart') {
  (async () => {
    over.agentJobs = JSON.parse(process.env.JOBTEST_AGENTJOBS || '{}');
    const swept = agentJobs.startup();
    process.send && process.send({ swept });
    const id = process.env.JOBTEST_JOB_ID;
    await settle(id, 20000);
    process.exit(0);
  })();
} else {
  main().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
}

async function runChild(role, env, onMessage) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [__filename], {
      env: { ...process.env, JOBTEST_ROLE: role, JOBTEST_ENGINE_PORT: String(engine.port), SNH_DATA_DIR: TMP, ...env },
      stdio: ['ignore', 'pipe', 'pipe', 'ipc']
    });
    let out = '';
    child.stdout.on('data', d => { out += d; });
    child.stderr.on('data', d => { out += d; });
    child.on('message', m => onMessage && onMessage(m, child));
    child.on('exit', (code, signal) => resolve({ code, signal, out }));
    if (onMessage) onMessage(null, child);
  });
}

async function main() {
  await startEngine();
  console.log(`\nBackground-job failure modes, caused for real (store: ${TMP}, engine on :${engine.port})\n`);

  // =========================================================================
  console.log('── 1. A STALL: the engine goes quiet mid-answer ──');
  over.generation = { stallTimeoutMs: 700, firstTokenTimeoutMs: 5000 };
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }] },
    { stall: true },
    // the salvage writeup, once the loop has thrown
    { text: 'Salvaged: I had counted the store before it went quiet.' }
  ];
  let s = agentJobs.enqueue({ title: 'stalls in round two', task: 'Count the facts, then think.' });
  let j = await settle(s.id);
  check('the job is `failed`', j.status === 'failed', j.status);
  check('  stopped by the RUNNER, kind `stall`', j.stop_source === 'runner' && j.stop_kind === 'stall', `${j.stop_source}/${j.stop_kind}`);
  check('  the sentence says the engine went quiet and names the limit in words', /went quiet for \d+ seconds? in the middle of an answer/.test(j.error) && /stall limit is 1 second/.test(j.error), j.error);
  check('  and where it was: round 2, after 1 tool call', /tool round 2/.test(j.error) && /after 1 tool call/.test(j.error), j.error);
  check('  the tool call count survived the throw (was 0 on the 9/9 card)', j.tool_calls === 1, String(j.tool_calls));
  check('  partial output is on the card, from the salvage writeup', /Salvaged: I had counted/.test(j.result_text || ''), j.result_text);
  const feed1 = agentJobs.feed().find(f => f.id === s.id);
  check('  the feed marks it as offering PARTIAL output', feed1 && feed1.partial_output === true, JSON.stringify(feed1 && feed1.partial_output));
  check('  the checkpoint on disk holds the tool record', (agentJobs.readCheckpoint(s.id) || {}).toolCalls?.length === 1);
  const stalledId = s.id;

  // =========================================================================
  console.log('\n── 2. THE ENGINE CUT OFF MID-ANSWER — the 9/9 shape ──');
  over.generation = { stallTimeoutMs: 5000, firstTokenTimeoutMs: 5000 };
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }] },
    { cut: true },
    { refuse: 503 }, { refuse: 503 }   // both salvage calls find the engine gone too
  ];
  s = agentJobs.enqueue({ title: 'cut off', task: 'Count, then get cut off.' });
  j = await settle(s.id);
  check('the job is `failed`', j.status === 'failed', j.status);
  check('  with no restart on record it is the ENGINE side, `connection-cut`', j.stop_source === 'engine' && j.stop_kind === 'connection-cut', `${j.stop_source}/${j.stop_kind}`);
  check('  the sentence never says "terminated"', !/\bterminated\b/.test(j.error || ''), j.error);
  check('  the text it had streamed before the cut is on the card, marked cut off', /half an ans/.test(j.result_text || '') && /cut off, not finished/.test(j.result_text || ''), j.result_text);
  check('  and the tool record is there too', /memory_count/.test(j.result_text || '') && j.tool_calls === 1, j.result_text);

  console.log('\n── 3. THE SAME CUT, WITH THE WATCHDOG\'S RESTART ON RECORD ──');
  const watchdog = require(path.join(ROOT, 'db/brain-watchdog'));
  const realRecent = watchdog.recentRestart;
  watchdog.recentRestart = () => ({ issuedAt: Date.now() - 90000, reason: '3 failed liveness checks in a row, verdict "stalled"' });
  engine.script = [{ tools: [{ name: 'memory_count', args: {} }] }, { cut: true }, { text: 'The engine is back: I had one count done before the cut.' }];
  s = agentJobs.enqueue({ title: 'cut by our own watchdog', task: 'Count, then be killed by the watchdog.' });
  j = await settle(s.id);
  watchdog.recentRestart = realRecent;
  check('it is the RUNNER side, kind `watchdog-restart`', j.stop_source === 'runner' && j.stop_kind === 'watchdog-restart', `${j.stop_source}/${j.stop_kind}`);
  check('  and the card says SNH\'s own watchdog restarted the engine, and when', /own brain watchdog restarted the engine at \d/.test(j.error || ''), j.error);
  check('  with the engine back, the writeup came from the transcript itself', /The engine is back/.test(j.result_text || ''), j.result_text);
  const salvageReq = engine.requests[engine.requests.length - 1] || {};
  check('  and that writeup turn was given the half-streamed text to keep', /half an ans/.test(lastUserMessageOf(salvageReq)) && !salvageReq.tools, lastUserMessageOf(salvageReq).slice(0, 200));

  console.log('\n── 4. NOTHING LISTENING AT THE ENGINE ──');
  const livePort = over.enginePort;
  over.enginePort = 1;   // nothing there
  s = agentJobs.enqueue({ title: 'engine down', task: 'Try anyway.' });
  j = await settle(s.id);
  over.enginePort = livePort;
  check('it is the ENGINE side, kind `unreachable`', j.stop_source === 'engine' && j.stop_kind === 'unreachable', `${j.stop_source}/${j.stop_kind} ${j.error}`);
  check('  and the card still has an account, not a blank', !!(j.result_text || '').trim());

  // =========================================================================
  console.log('\n── 5. A PROVIDER DOWN: the free retry ──');
  over.generation = {};
  toolScript.memory_search = [
    () => { throw new Error('ECONNREFUSED: exa is down'); },            // first try: dead
    () => ({ results: [{ id: 'x', text: 'a fact' }] }),                  // free retry: answers
    () => { throw new Error('still down'); },                             // second call: dead
    () => { throw new Error('still down'); },                             //   and its retry: dead
    () => ({ results: [] })                                               // third call: ran fine, found nothing
  ];
  engine.script = [
    { tools: [{ name: 'memory_search', args: { query: 'one' } }, { name: 'memory_search', args: { query: 'two' } }, { name: 'memory_search', args: { query: 'three' } }] },
    { text: 'Three lookups done.' }
  ];
  s = agentJobs.enqueue({ title: 'provider flaps', task: 'Search three times.' });
  j = await settle(s.id);
  const b = JSON.parse(j.budget_json || '{}');
  check('the job finished ok', j.status === 'ok', j.status);
  check('  three calls, as the model asked', b.calls === 3, String(b.calls));
  check('  two free retries happened', b.retries === 2, String(b.retries));
  check('  billed 1 (retry answered) + 0.25 (retry failed too) + 0.25 (empty, an answer) = 1.5', b.billed === 1.5, String(b.billed));
  check('  two of them count as failed/empty — the retried-and-answered one does not', b.failedCalls === 2, String(b.failedCalls));
  check('  the checkpoint is gone — an ok job\'s record is its result', !fs.existsSync(agentJobs.checkpointPath(s.id)));
  toolScript.memory_search = [() => { throw new Error('down once'); }, () => ({ results: [{ id: 'y' }] }), () => ({ results: [] })];
  engine.script = [{ tools: [{ name: 'memory_search', args: { query: 'p' } }, { name: 'memory_search', args: { query: 'q' } }] }, { text: 'ok' }];
  const direct = await mm.callLLM('sys', 'user', { toolSession: mm.createToolSession('direct', ['memory_search'], { maxCalls: 10 }) });
  check('  the tool record marks which call was retried', direct.toolCalls[0].retried === true && direct.toolCalls[0].productive === true && direct.toolCalls[1].retried === false && direct.toolCalls[1].productive === false, JSON.stringify(direct.toolCalls.map(r => [r.retried, r.productive])));

  console.log('\n── 5b. THE ATTEMPT CEILING STILL COUNTS RETRIES ──');
  const sess = mm.createToolSession('t', ['memory_search'], { maxCalls: 2, maxAttempts: 3 });
  toolScript.memory_search = [() => { throw new Error('down'); }, () => { throw new Error('down'); }, () => { throw new Error('down'); }];
  await mm.executeBackgroundTool(sess, 'memory_search', { query: 'a' });   // call 1 + retry = 2 attempts
  await mm.executeBackgroundTool(sess, 'memory_search', { query: 'b' });   // call 2 → 3 attempts; no room to retry
  check('a retry is refused once the raw attempt ceiling is reached', sess.calls === 2 && sess.retries === 1 && !!sess.spent(), `calls ${sess.calls} retries ${sess.retries} spent ${sess.spent()}`);
  check('  and the ceiling reason counts the retry', /3\/3 calls/.test(sess.spent() || ''), sess.spent());

  // =========================================================================
  console.log('\n── 6. THE WALL CLOCK BINDS while it is still asking ──');
  over.agentJobs = { maxWallClockMs: 5000 };
  over.generation = { stallTimeoutMs: 10000, firstTokenTimeoutMs: 10000 };
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }] },
    // keeps producing reasoning tokens for ~5.2s so the stall clock never fires, then asks again
    { tools: [{ name: 'memory_count', args: {} }], preamble: 26, gapMs: 200 },
    { text: 'What I had before the clock ran out.' }
  ];
  s = agentJobs.enqueue({ title: 'runs out of time', task: 'Keep going until the clock stops you.' });
  j = await settle(s.id, 40000);
  over.agentJobs = {};
  check('the job is `partial` — it wrote what it had', j.status === 'partial', j.status);
  check('  stopped by the RUNNER, kind `wall-clock`', j.stop_source === 'runner' && j.stop_kind === 'wall-clock', `${j.stop_source}/${j.stop_kind}`);
  check('  the sentence says how long it ran and what the limit is, in words', /running \d+ seconds/.test(j.error || '') && /limit for one job is 5 seconds/.test(j.error || ''), j.error);
  check('  and points at the Settings row', /Time limit for one job/.test(j.error || ''), j.error);

  // =========================================================================
  console.log('\n── 7. A KILLED PROCESS MID-RUN, then a RESTART that resumes it ──');
  over.generation = {};
  engine.requests.length = 0;
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }] },
    { tools: [{ name: 'memory_count', args: {} }] },
    // round 3: never answers — the child is killed while it waits here
    (req, res) => { res.writeHead(200, { 'Content-Type': 'text/event-stream' }); }
  ];
  let killedId = null;
  const run = await runChild('child-run', {}, async (m, child) => {
    if (m && m.id) {
      killedId = m.id;
      // Wait until the checkpoint shows two completed rounds, then kill -9.
      for (let i = 0; i < 200; i++) {
        const ck = agentJobs.readCheckpoint(killedId);
        if (ck && (ck.toolCalls || []).length >= 2 && engine.requests.length >= 3) break;
        await sleep(50);
      }
      child.kill('SIGKILL');
    }
  });
  check('the child was killed, not finished', run.signal === 'SIGKILL', `${run.code}/${run.signal}`);
  j = job(killedId);
  check('  the row is still `running` — nothing closed it', j && j.status === 'running', j && j.status);
  const ckKilled = agentJobs.readCheckpoint(killedId);
  check('  the checkpoint on disk has both completed rounds and nothing of the third', ckKilled && ckKilled.toolCalls.length === 2 && ckKilled.session.roundsUsed === 2, JSON.stringify(ckKilled && { calls: ckKilled.toolCalls.length, rounds: ckKilled.session.roundsUsed }));

  engine.requests.length = 0;
  engine.script = [{ text: 'Finished after the restart, from where I left off.' }];
  const restart = await runChild('child-restart', { JOBTEST_JOB_ID: killedId });
  j = job(killedId);
  check('a new process swept it: re-queued, then resumed and finished ok', j.status === 'ok', `${j.status} — ${restart.out.slice(-300)}`);
  check('  it counts as the second (and last) attempt', j.attempts === 2, String(j.attempts));
  check('  the result is the post-restart answer', /after the restart/.test(j.result_text || ''), j.result_text);
  const resumedReq = engine.requests[0] || {};
  const roles = (resumedReq.messages || []).map(m => m.role).join(',');
  check('  the resumed request carried the earlier tool results (round 1 and 2)', (resumedReq.messages || []).filter(m => m.role === 'tool').length === 2, roles);
  check('  and told the model the restart lost the round in progress', /SNH restarted while you were in the middle of a tool round/.test(lastUserMessageOf(resumedReq)), lastUserMessageOf(resumedReq));
  check('  the checkpoint is gone once the job is ok', !fs.existsSync(agentJobs.checkpointPath(killedId)));

  console.log('\n── 7b. KILLED, and NOT run again — the card still offers what it had ──');
  engine.requests.length = 0;
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }] },
    { tools: [{ name: 'memory_count', args: {} }] },
    (req, res) => { res.writeHead(200, { 'Content-Type': 'text/event-stream' }); }
  ];
  killedId = null;
  await runChild('child-run', {}, async (m, child) => {
    if (m && m.id) {
      killedId = m.id;
      for (let i = 0; i < 200; i++) {
        const ck = agentJobs.readCheckpoint(killedId);
        if (ck && (ck.toolCalls || []).length >= 2 && engine.requests.length >= 3) break;
        await sleep(50);
      }
      child.kill('SIGKILL');
    }
  });
  engine.requests.length = 0;
  // maxAttempts 1: the sweep may not re-run it.
  await runChild('child-restart', { JOBTEST_JOB_ID: killedId, JOBTEST_AGENTJOBS: JSON.stringify({ maxAttempts: 1 }) });
  j = job(killedId);
  check('closed as `interrupted` with the reason', j.status === 'interrupted' && /already been retried|NOT run again/.test(j.error || ''), `${j.status} ${j.error}`);
  check('  stopped by the RUNNER, kind `service-restart`', j.stop_source === 'runner' && j.stop_kind === 'service-restart', `${j.stop_source}/${j.stop_kind}`);
  check('  its two tool calls are on the card as partial output — no model call needed', /2 tool call\(s\): memory_count, memory_count/.test(j.result_text || '') && j.tool_calls === 2, j.result_text);
  check('  no engine request was made to write it (the engine may not be back yet)', engine.requests.length === 0, String(engine.requests.length));
  check('  the feed offers it as partial output', agentJobs.feed().find(f => f.id === killedId)?.partial_output === true);

  // =========================================================================
  console.log('\n── 8. THE BUDGET ASK, answered YES ──');
  const convId = database.createConversation('Door watch build', 'stub', 'user');
  database.addMessage(convId, 'user', 'go research the Hailo stack', 'stub');
  over.agentJobs = { maxToolCallsPerJob: 5, maxRoundsPerJob: 16, askBeforeCeiling: true, askAtPercent: 60, extensionPercent: 50 };
  engine.requests.length = 0;
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }] },   // 3 billed = 60% of 5
    { tools: [{ name: 'memory_count', args: { again: true } }] },                                                             // asks for more → pause
    { text: 'So far: three counts, all consistent. Left: the fourth count and the writeup. I need a little more.\nNEEDED: 4 more tool calls' }
  ];
  s = agentJobs.enqueue({ title: 'Hailo door watch research', task: 'Count four times and write it up.', conversationId: convId });
  j = await settle(s.id, 20000, ['paused']);
  check('the job PAUSED instead of running into the wall', j.status === 'paused', j.status);
  const ask = JSON.parse(j.ask_json || '{}');
  check('  the ask says what it has, what is left, and what it wants', /three counts/.test(ask.text) && /fourth count/.test(ask.text), ask.text);
  check('  the NEEDED line was read and became the grant: 4 calls', ask.grant && ask.grant.calls === 4 && ask.wants === 4, JSON.stringify(ask.grant));
  check('  the footer tells her what yes and no do, in the entity\'s voice', /If you say yes, I get 4 more tool calls/.test(ask.text) && /If you say no, I'll write up what I have/.test(ask.text), ask.text);
  check('  the NEEDED line itself is not shown to her', !/NEEDED:/.test(ask.text), ask.text);
  check('  it was near the CALLS limit at 3 of 5', ask.near && ask.near.limit === 'calls' && ask.near.used === 3 && ask.near.max === 5, JSON.stringify(ask.near));
  const askMsg = db.prepare("SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY timestamp DESC LIMIT 1").get(convId);
  check('  the ask is an assistant message IN THE CONVERSATION that dispatched it', askMsg && askMsg.content === ask.text, askMsg && askMsg.content);
  check('  and the delivery record says so', ask.delivery && ask.delivery.conversationId === convId && ask.delivery.opened === false, JSON.stringify(ask.delivery));
  const bell = db.prepare("SELECT * FROM initiatives WHERE source_kind = 'job-budget-ask' AND source_ref = ?").get(s.id);
  check('  the bell got a PROPOSAL pointing at it', bell && bell.type === 'proposal' && bell.status === 'pending', JSON.stringify(bell && { type: bell.type, status: bell.status }));
  check('  which holds no content beyond "waiting on you, in that conversation"', bell && /waiting on your answer/.test(bell.content) && !/three counts/.test(bell.content), bell && bell.content);
  check('  the checkpoint holds the pending call it did not run', (agentJobs.readCheckpoint(s.id) || {}).pendingCalls?.length === 1);
  check('  the job holds no lane while it waits', !agentJobs._inFlight.has(s.id) && agentJobs.activeCount() === 0, `inFlight ${agentJobs._inFlight.size} active ${agentJobs.activeCount()}`);
  check('  the live block tells the entity it is paused, waiting on her', /PAUSED, WAITING ON HER: "Hailo door watch research"/.test((agentJobs.renderActiveJobsBlock() || {}).text || ''));
  check('  pendingAsk finds it by conversation', agentJobs.pendingAsk(convId)?.job.id === s.id);

  // Her message that is NOT an answer.
  const classifier = (verdict) => async () => ({ content: verdict });
  let outcome = await budgetAsk.decideFromMessage({ conversationId: convId, message: 'how is it going?', callLLM: classifier('NEITHER') });
  check('a message that is neither leaves it waiting', outcome.decision === 'neither' && job(s.id).status === 'paused', JSON.stringify(outcome && outcome.decision));
  check('  and the guidance block says so', /Still Waiting/.test(budgetAsk.renderGuidance(outcome).text));
  outcome = await budgetAsk.decideFromMessage({ conversationId: convId, message: 'yes', callLLM: async () => { throw new Error('engine wedged'); } });
  check('a classifier that cannot answer fails to NEITHER, never to a decision', outcome.decision === 'neither' && job(s.id).status === 'paused', outcome.reason);

  // YES.
  engine.requests.length = 0;
  engine.script = [{ text: 'Four counts done with the extra budget. All consistent.' }];
  outcome = await budgetAsk.decideFromMessage({ conversationId: convId, message: 'yes go ahead', callLLM: classifier('YES') });
  check('YES is acted on: granted', outcome.decision === 'yes' && outcome.ok === true, JSON.stringify(outcome && { d: outcome.decision, ok: outcome.ok, e: outcome.error }));
  j = await settle(s.id, 20000);
  check('  the job resumed and finished ok', j.status === 'ok', `${j.status} ${j.error}`);
  check('  the result is the post-yes answer', /extra budget/.test(j.result_text || ''), j.result_text);
  const bud = JSON.parse(j.budget_json || '{}');
  check('  the budget summary records the extension: +4 calls, and 4 calls made in all', bud.extended && bud.extended.calls === 4 && bud.calls === 4, JSON.stringify({ ext: bud.extended, calls: bud.calls, max: bud.maxCalls }));
  check('  it stayed one attempt — a resume is not a new start', j.attempts === 1, String(j.attempts));
  const resumeReq = engine.requests[0] || {};
  check('  the pending call ran first and its result is in the transcript', (resumeReq.messages || []).filter(m => m.role === 'tool').length === 4, (resumeReq.messages || []).map(m => m.role).join(','));
  check('  and the transcript told the model she said yes, with the numbers', /Ellie said yes\. You have 4 more tool call/.test(lastUserMessageOf(resumeReq)), lastUserMessageOf(resumeReq));
  const bellAfter = db.prepare('SELECT status FROM initiatives WHERE id = ?').get(bell.id);
  check('  the bell item is decided (delivered), not left ringing', bellAfter && bellAfter.status === 'delivered', bellAfter && bellAfter.status);
  check('  the guidance tells the entity what was done', /she just said YES/.test(budgetAsk.renderGuidance(outcome).text) && /4 more tool call/.test(budgetAsk.renderGuidance(outcome).text));

  console.log('\n── 9. THE BUDGET ASK, answered NO ──');
  engine.requests.length = 0;
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }] },
    { tools: [{ name: 'memory_count', args: { again: true } }] },
    { text: 'So far: three counts. Left: one more.\nNEEDED: 2 more tool calls' }
  ];
  s = agentJobs.enqueue({ title: 'Hailo door watch research, take two', task: 'Count four times.', conversationId: convId });
  j = await settle(s.id, 20000, ['paused']);
  check('paused again', j.status === 'paused', j.status);
  engine.requests.length = 0;
  engine.script = [{ text: 'Wrapping up with three counts; the fourth was not run.' }];
  outcome = await budgetAsk.decideFromMessage({ conversationId: convId, message: 'no, wrap it up', callLLM: classifier('NO') });
  check('NO is acted on: declined', outcome.decision === 'no' && outcome.ok === true, JSON.stringify(outcome && { d: outcome.decision, ok: outcome.ok }));
  j = await settle(s.id, 20000);
  check('  it wrote up what it had and finished PARTIAL', j.status === 'partial' && /Wrapping up with three counts/.test(j.result_text || ''), `${j.status} ${j.result_text}`);
  check('  stopped by HER, kind `budget-declined` — not a failure of anything', j.stop_source === 'user' && j.stop_kind === 'budget-declined', `${j.stop_source}/${j.stop_kind}`);
  check('  the sentence says it was her decision and what it had used', /at your decision/.test(j.error || '') && /3\.0 of 5 tool calls/.test(j.error || ''), j.error);
  const wrapReq = engine.requests[0] || {};
  check('  the writeup turn had NO tools and told the model she said no', !wrapReq.tools && /Ellie said no to more budget/.test(lastUserMessageOf(wrapReq)), lastUserMessageOf(wrapReq));
  check('  the pending call was marked not run in the transcript', (wrapReq.messages || []).some(m => m.role === 'tool' && /Ellie chose to stop here/.test(m.content)));
  check('  the guidance tells the entity not to argue for more', /do not argue for more/.test(budgetAsk.renderGuidance(outcome).text));
  check('  an unanswered ask still routes to nobody: pendingAsk is empty now', agentJobs.pendingAsk(convId) === null);

  console.log('\n── 9b. The ask goes to a NEW conversation when the old one is archived ──');
  const archivedConv = database.createConversation('an old one', 'stub', 'user');
  db.prepare("UPDATE conversations SET status = 'archived' WHERE id = ?").run(archivedConv);
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }] },
    { tools: [{ name: 'memory_count', args: {} }] },
    { text: 'Three so far.\nNEEDED: 1 more tool calls' }
  ];
  s = agentJobs.enqueue({ title: 'orphaned ask', task: 'Count.', conversationId: archivedConv });
  j = await settle(s.id, 20000, ['paused']);
  const a2 = JSON.parse(j.ask_json || '{}');
  check('a new conversation was opened for it, titled so she knows', a2.delivery && a2.delivery.opened === true && /needs your answer/.test(a2.delivery.conversationTitle || ''), JSON.stringify(a2.delivery));
  check('  and pendingAsk finds it by the NEW conversation', agentJobs.pendingAsk(a2.delivery.conversationId)?.job.id === s.id);
  const r2 = agentJobs.cancel(s.id);
  check('  cancelling a paused job from the panel is a NO — it writes up', r2.ok === true);
  engine.script = [{ text: 'wrapped.' }];
  j = await settle(s.id, 20000);
  check('  and finishes partial by her decision', j.status === 'partial' && j.stop_kind === 'budget-declined', `${j.status}/${j.stop_kind}`);
  over.agentJobs = {};

  console.log('\n── 9c. Asking switched off: the hard stop behaves as before ──');
  over.agentJobs = { maxToolCallsPerJob: 2, maxRoundsPerJob: 16, askBeforeCeiling: false };
  engine.script = [
    { tools: [{ name: 'memory_count', args: {} }, { name: 'memory_count', args: {} }] },
    { tools: [{ name: 'memory_count', args: {} }] },
    { text: 'Out of budget, here is what I had.' }
  ];
  s = agentJobs.enqueue({ title: 'no asking', task: 'Count.' });
  j = await settle(s.id, 20000);
  check('it runs into the call budget and writes up: partial, runner/call-budget', j.status === 'partial' && j.stop_source === 'runner' && j.stop_kind === 'call-budget', `${j.status} ${j.stop_source}/${j.stop_kind}`);
  check('  no ask was made', !j.ask_json && !db.prepare("SELECT COUNT(*) n FROM initiatives WHERE source_ref = ?").get(s.id).n);
  over.agentJobs = {};

  // =========================================================================
  console.log('\n── 10. RETRY FROM THE CARD, carrying the last attempt ──');
  engine.requests.length = 0;
  engine.script = [{ text: 'Second attempt: picked up from the salvaged count and finished.' }];
  const r = agentJobs.retry(stalledId);
  check('retry returns a NEW job id', r.ok && r.id && r.id !== stalledId, JSON.stringify(r));
  j = await settle(r.id, 20000);
  const old = job(stalledId);
  check('  the new row points back at the old one', j.retry_of === stalledId);
  check('  and the old row points forward', old.retried_by === r.id);
  check('  the old row itself is untouched: still failed, same text', old.status === 'failed' && /Salvaged/.test(old.result_text));
  check('  the new run finished ok', j.status === 'ok', `${j.status} ${j.error}`);
  const sys = systemMessageOf(engine.requests[0] || {});
  check('  its brief says THIS IS A RETRY', /THIS IS A RETRY/.test(sys));
  check('  and carries the previous attempt\'s reason (the stall, by the runner)', /went quiet/.test(sys) && /stall limit/.test(sys), sys.slice(0, 400));
  check('  and its partial output', /Salvaged: I had counted/.test(sys));
  check('  and what it had already looked up', /WHAT IT HAD ALREADY LOOKED UP \(1 call\(s\)\)/.test(sys) && /memory_count/.test(sys));
  check('  the feed no longer offers a retry on the old card, and offers none on the ok one', agentJobs.feed().find(f => f.id === stalledId)?.retryable === false && agentJobs.feed().find(f => f.id === r.id)?.retryable === false);
  const again = agentJobs.retry(stalledId);
  check('  retrying the old card twice is refused with the reason', again.ok === false && /already/.test(again.error), JSON.stringify(again));
  const codingId = require('crypto').randomUUID();
  db.prepare("INSERT INTO agent_jobs (id, title, task, status, source, error, stop_source, stop_kind, finished_at) VALUES (?, 'squatch-code: x', 'edit files', 'failed', ?, 'The coding agent exited without writing a report.', 'dispatched', 'died', ?)")
    .run(codingId, require(path.join(ROOT, 'db/coding-jobs')).SOURCE, new Date().toISOString());
  const rc = agentJobs.retry(codingId);
  check('  a coding job cannot be retried from the panel, and the refusal says where to', rc.ok === false && /from the conversation/.test(rc.error), JSON.stringify(rc));
  const rok = agentJobs.retry(r.id);
  check('  an ok job is not offered a retry either', rok.ok === false && /nothing to retry/.test(rok.error), JSON.stringify(rok));

  // =========================================================================
  console.log('\n── 11. THE CHANNEL RULE STILL HOLDS ──');
  const src = fs.readFileSync(path.join(ROOT, 'db/agent-jobs.js'), 'utf8');
  check('db/agent-jobs.js does not require db/initiatives', !/require\(['"]\.\/initiatives['"]\)/.test(src));
  check('db/agent-jobs.js does not require db/conversation-channel', !/require\(['"]\.\/conversation-channel['"]\)/.test(src));
  const finishedIds = db.prepare("SELECT id FROM agent_jobs WHERE status IN ('ok','partial','failed','interrupted') AND ask_json IS NULL").all().map(x => x.id);
  const leaked = db.prepare(`SELECT COUNT(*) n FROM initiatives WHERE source_ref IN (${finishedIds.map(() => '?').join(',') || "''"})`).get(...finishedIds).n;
  check('no finished job that never paused has a bell item', leaked === 0, String(leaked));
  check('every bell item from a job is a PROPOSAL from a pause, nothing else', db.prepare("SELECT COUNT(*) n FROM initiatives WHERE source_kind != 'job-budget-ask' OR type != 'proposal'").get().n === 0);

  // =========================================================================
  console.log('\n── 12. The wording never regresses to a bare word ──');
  const bare = db.prepare("SELECT id, status, error FROM agent_jobs WHERE status IN ('failed','partial','interrupted') AND (error IS NULL OR length(error) < 40)").all();
  check('no failed, partial or interrupted row has a one-word reason', bare.length === 0, JSON.stringify(bare));
  const unplaced = db.prepare("SELECT id, status FROM agent_jobs WHERE status IN ('failed','partial','interrupted') AND stop_source IS NULL").all();
  check('every one of them names a side', unplaced.length === 0, JSON.stringify(unplaced));

  engine.server.close();
  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
}

process.on('exit', () => {
  if (ROLE !== 'main') return;
  // CLEANUP IS PART OF THE TEST. The store, the checkpoints, the lancedb dir.
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});
