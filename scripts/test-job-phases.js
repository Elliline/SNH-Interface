#!/usr/bin/env node
/**
 * A LONG JOB SURVIVES ITS WINDOW — the three parts, caused for real.
 *
 * 2026-09-17: a research job on Juno died in tool round 11, 40 tool calls in,
 * on HTTP 400 — prompt 81,857 tokens plus a 49,216-token output reservation
 * against a 131,072 ceiling. This suite drives the REAL runner (db/agent-jobs
 * through memory-manager's tool loop and streamChat) against a scripted engine
 * that reports a small window and a `usage` count, and asserts:
 *
 *   PART 1 — the output reservation fits the room left: rounds are squeezed as
 *            the prompt grows, the engine's own count calibrates the estimate,
 *            a 400 on context length is read, refitted and retried once.
 *   PART 2 — tool results are compacted as the job runs: old fetches become
 *            digests WITH title/url/date attached by the runner, searches keep
 *            every hit's provenance, nothing is silent (the report is on the
 *            row), and NO URL present before compaction is missing after it.
 *   PART 3 — a job runs in phases with the runner writing a findings document
 *            between them; a process KILLED mid-phase 2 leaves phase 1's
 *            findings and sources on disk; a restart resumes phase 2; a failed
 *            phased job's retry carries the finished phases.
 *
 * Refuses to run without SNH_DATA_DIR — it writes jobs, checkpoint files and
 * documents (into the throwaway's own documents folder). Tears down at exit.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-job-phases.js
 */
process.env.TZ = 'America/Los_Angeles';
const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-job-phases.js');
  process.exit(1);
}
const TMP = process.env.SNH_DATA_DIR;
const ROOT = path.join(__dirname, '..');
const ROLE = process.env.PHASETEST_ROLE || 'main';   // main | child-run | child-restart

// ---------------------------------------------------------------------------
// Config stub — installed BEFORE memory-manager is required. data/config.json
// is NOT redirected by SNH_DATA_DIR, so nothing here writes it.
// ---------------------------------------------------------------------------
const config = require(path.join(ROOT, 'db/config'));
const realGetConfig = config.getConfig;
const over = { enginePort: Number(process.env.PHASETEST_ENGINE_PORT || 0), window: 9000, agentJobs: {}, generation: {} };
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
      context: { windowTokens: null, marginTokens: 100, floorAnswerTokens: 256, floorThinkingTokens: 0, charsPerToken: 3 },
      compaction: { enabled: true, recentRounds: 1, hardAfter: 6, snippetChars: 120, truncateChars: 300, digestFetches: true },
      phases: { enabled: true, maxPhases: 6, carryChars: 12000, splitAtPercent: 0, partsPerPhase: 4 }
    }, over.agentJobs),
    generation: Object.assign({}, c.generation, {
      agentJobResponseTokens: 2000, agentJobThinkingTokens: 1000,
      stallTimeoutMs: 20000, firstTokenTimeoutMs: 20000
    }, over.generation),
    heartbeat: { ...c.heartbeat, toolBudget: Object.assign({}, c.heartbeat.toolBudget, { failedCallRetries: 0, failedCallCost: 0.25, attemptCeilingMultiple: 2 }) },
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
const jobWindow = require(path.join(ROOT, 'db/job-window'));
const compaction = require(path.join(ROOT, 'db/job-compaction'));
const findings = require(path.join(ROOT, 'db/job-findings'));
const artifacts = require(path.join(ROOT, 'db/job-artifacts'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${String(detail).slice(0, 400)}` : ''}`); }
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const job = (id) => db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id);
async function settle(id, ms = 60000, statuses = agentJobs.TERMINAL) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    const j = job(id);
    if (j && statuses.includes(j.status)) return j;
    await sleep(25);
  }
  return job(id);
}
const progressOf = (j) => { try { return JSON.parse(j.progress_json); } catch { return null; } };
// The runner reads the window through db/model-context, which caches an
// engine's answer for ten minutes. A scenario that changes the stub's window
// clears that cache, or the runner keeps planning against the old one.
const setWindow = (n) => { over.window = n; require(path.join(ROOT, 'db/model-context'))._cache.clear(); };

// ---------------------------------------------------------------------------
// The scripted engine. GET /v1/models reports the window; every POST
// /v1/chat/completions consumes the next step of `script` and ends with a
// `usage` chunk counting the prompt at ~3 chars/token, so the runner's
// estimate has something real to calibrate against.
// ---------------------------------------------------------------------------
const engine = { script: [], requests: [], server: null, port: 0 };
function sse(res, obj) { res.write(`data: ${JSON.stringify(obj)}\n\n`); }
function toolDelta(calls) {
  return { choices: [{ delta: { tool_calls: calls.map((c, i) => ({ index: i, id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 6)}_${i}`, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args || {}) } })) }, finish_reason: null }] };
}
function promptTokensOf(body) {
  const msgs = body.messages || [];
  let chars = 0;
  for (const m of msgs) { chars += (typeof m.content === 'string' ? m.content.length : JSON.stringify(m.content || '').length) + JSON.stringify(m.tool_calls || '').length + 12; }
  if (body.tools) chars += JSON.stringify(body.tools).length;
  return Math.ceil(chars / 3);
}
async function serve(req, res) {
  if (req.method === 'GET' && /\/v1\/models/.test(req.url || '')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ data: [{ id: 'stub-model', max_model_len: over.window }] }));
  }
  if (req.method !== 'POST' || !/\/chat\/completions/.test(req.url || '')) { res.writeHead(404); return res.end('{}'); }
  let raw = '';
  for await (const chunk of req) raw += chunk;
  let body = {};
  try { body = JSON.parse(raw); } catch { /* not json */ }
  engine.requests.push(body);
  const step = engine.script.length ? engine.script.shift() : { text: 'done.' };
  if (typeof step === 'function') return step(req, res, body);
  const prompt = promptTokensOf(body);
  // THE REAL WALL. A request whose prompt plus reservation exceeds the window
  // is refused exactly as vLLM refuses it — the sentence the 9/17 card quoted.
  if (step.enforceWindow !== false && Number.isFinite(body.max_tokens) && prompt + body.max_tokens > over.window) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ error: { message: `This model's maximum context length is ${over.window} tokens. However, you requested ${body.max_tokens} output tokens and your prompt contains at least ${prompt} in`, type: 'BadRequestError' } }));
  }
  if (step.refuse) { res.writeHead(step.refuse, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify({ error: step.message || 'refused by the stub' })); }
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  if (step.stall) { sse(res, { choices: [{ delta: { content: 'starting' }, finish_reason: null }] }); return; }
  if (step.hang) { return; }   // never sends anything — the child is killed while it waits here
  let completion = 20;
  if (step.tools) {
    sse(res, { choices: [{ delta: { reasoning: 'thinking about which tool. ' }, finish_reason: null }] });
    sse(res, toolDelta(step.tools));
    sse(res, { choices: [{ delta: {}, finish_reason: 'tool_calls' }] });
  } else {
    const text = step.text || 'done.';
    completion = Math.ceil(text.length / 3);
    sse(res, { choices: [{ delta: { content: text }, finish_reason: null }] });
    sse(res, { choices: [{ delta: {}, finish_reason: step.length ? 'length' : 'stop' }] });
  }
  sse(res, { choices: [], usage: { prompt_tokens: prompt, completion_tokens: completion } });
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
// Tools: the real registry, with web_search/web_fetch scripted to return the
// SHAPE the real ones return, at the size the 9/17 pages had.
// ---------------------------------------------------------------------------
const client = MCPClient.shared();
const realExecute = client.executeTool.bind(client);
let fetchCounter = 0;
const PAGE = (n) => `Qwen3.8-Flash-Next on 2x3090 — report ${n} · Inovello\n` +
  `Published 2026-09-0${(n % 9) + 1}. Config: llama.cpp b7100, UD-Q4_K_XL, -ngl 48, ctx 131072, ubatch 2048. ` +
  `Decode ${17 + n} tok/s, prompt ${1200 + n * 10} tok/s. Expert cache PR #${5000 + n}. ` +
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. '.repeat(45);
const SEARCH = (q) => ({
  results: [1, 2, 3, 4, 5, 6].map(i => ({
    title: `${q} — result ${i}`, url: `https://example.test/${encodeURIComponent(q)}/${i}`,
    snippet: `Snippet ${i} for ${q}: `.padEnd(400, 'x'), publishedDate: `2026-09-0${i}T00:00:00.000Z`
  })),
  provider: 'stub', providers_tried: ['stub']
});
client.executeTool = async (name, args, ctx) => {
  if (name === 'web_search') return SEARCH(String(args && args.query || 'q'));
  if (name === 'web_fetch') { fetchCounter++; return { url: String(args && args.url || `https://example.test/page/${fetchCounter}`), content: PAGE(fetchCounter), truncated: false }; }
  return realExecute(name, args, ctx);
};
const hasWeb = client.backgroundToolsAmong(['web_search', 'web_fetch']).length === 2;

const BRIEF =
  'Research a substrate switch for the assistant. Cite title/who/date/URL for every number.\n' +
  '1) QUANTS: confirm GB sizes of the published quants.\n' +
  '2) MEASURED THROUGHPUT on dual-3090-class hardware.\n' +
  '3) KNOWN PROBLEMS: bugs, crashes, version pinning.\n' +
  'Output: numbered sections, each claim cited.';
const PLAN = { text: JSON.stringify({ phases: ['Quant sizes', 'Measured throughput', 'Known problems'] }) };
const search = (q) => ({ tools: [{ name: 'web_search', args: { query: q } }] });
const fetch2 = (a, b) => ({ tools: [{ name: 'web_fetch', args: { url: a } }, { name: 'web_fetch', args: { url: b } }] });
const DIGEST = { text: JSON.stringify({ sources: [{ i: 0, findings: '- decode 17 tok/s (Inovello)' }, { i: 1, findings: '- expert cache PR' }] }) };
const digestStep = () => (req, res, body) => {
  // The digest call asks for JSON keyed by SOURCE index; answer for every
  // source in the request so nothing falls to truncation.
  const idx = [...String((body.messages || []).map(m => m.content).join('\n')).matchAll(/=== SOURCE (\d+) ===/g)].map(m => Number(m[1]));
  const out = { sources: idx.map(i => ({ i, findings: `- decode ${17 + i} tok/s on 2x3090 (Inovello, 2026-09)\n- llama.cpp b7100, UD-Q4_K_XL` })) };
  engine.script.unshift({ text: JSON.stringify(out) });
  return serve(req, res).catch(() => {});
};
const isDigestReq = (b) => /compacting your own research notes/.test((b.messages || [])[0]?.content || '');
const isPlanReq = (b) => /Split THE JOB/.test((b.messages || [])[0]?.content || '');
const isSynthesisReq = (b) => /FINDINGS BY PHASE \(your own writeups\)/.test([...(b.messages || [])].reverse()[0]?.content || '');
const isPartWriteup = (b) => /this sitting's transcript has grown large/.test([...(b.messages || [])].reverse()[0]?.content || '');

// A phase of `rounds` tool rounds then a writeup. The engine answers whatever
// request comes — a digest request is recognised by its prompt and answered
// with findings for every source in it, out of band of the script.
function phaseSteps(n, rounds, writeup) {
  const steps = [];
  for (let r = 0; r < rounds; r++) steps.push(r % 2 === 0 ? search(`phase ${n} query ${r}`) : fetch2(`https://example.test/phase-${n}/${r}a`, `https://example.test/phase-${n}/${r}b`));
  steps.push({ text: writeup });
  return steps;
}
// Requests that are not scripted steps: digests get answered from their own
// content, the planner from PLAN. This wrapper looks at the body first.
function scriptedServe(req, res) {
  const chunks = [];
  req.on('data', d => chunks.push(d));
  req.on('end', () => {
    const raw = Buffer.concat(chunks).toString();
    let body = {};
    try { body = JSON.parse(raw); } catch { /* not json */ }
    const fake = { method: req.method, url: req.url, socket: req.socket, [Symbol.asyncIterator]: async function* () { yield raw; } };
    if (req.method === 'POST' && isDigestReq(body)) {
      const idx = [...String((body.messages || []).map(m => m.content).join('\n')).matchAll(/=== SOURCE (\d+) ===/g)].map(m => Number(m[1]));
      engine.script.unshift({ text: JSON.stringify({ sources: idx.map(i => ({ i, findings: `- decode ${17 + i} tok/s on 2x3090 (Inovello, 2026-09)\n- llama.cpp b7100, UD-Q4_K_XL` })) }) });
    } else if (req.method === 'POST' && isPlanReq(body)) {
      engine.script.unshift(over.plan || PLAN);
    }
    serve(fake, res).catch(() => { try { res.end(); } catch { /* gone */ } });
  });
}

// ===========================================================================
// CHILD ROLES — a separate process so a job can genuinely be killed mid-phase,
// and a genuinely new process can pick it up.
// ===========================================================================
if (ROLE === 'child-run') {
  (async () => {
    await new Promise(resolve => { engine.server = http.createServer(scriptedServe); engine.server.listen(0, '127.0.0.1', () => { engine.port = engine.server.address().port; over.enginePort = engine.port; resolve(); }); });
    // Phase 1 completes; phase 2 hangs in its second round.
    engine.script = [...phaseSteps(1, 2, 'Phase 1 findings: UD-Q4_K_XL is 111.3 GB (Unsloth, https://example.test/phase-1/1a).'),
      search('phase 2 query 0'), { hang: true }];
    const started = agentJobs.enqueue({ title: 'the one that gets killed in phase 2', task: BRIEF, conversationId: null });
    process.send && process.send({ id: started.id, port: engine.port });
    await settle(started.id, 60000);
    process.exit(0);
  })();
} else if (ROLE === 'child-restart') {
  (async () => {
    await new Promise(resolve => { engine.server = http.createServer(scriptedServe); engine.server.listen(0, '127.0.0.1', () => { engine.port = engine.server.address().port; over.enginePort = engine.port; resolve(); }); });
    // What the resumed phase 2 needs, then phase 3, then the synthesis.
    engine.script = [fetch2('https://example.test/phase-2/1a', 'https://example.test/phase-2/1b'), { text: 'Phase 2 findings after the restart.' },
      ...phaseSteps(3, 1, 'Phase 3 findings.'), { text: 'FINAL after restart: all three phases.' }];
    const swept = agentJobs.startup();
    process.send && process.send({ swept, port: engine.port });
    const id = process.env.PHASETEST_JOB_ID;
    await settle(id, 40000);
    process.send && process.send({ requests: engine.requests.map(b => ({ plan: isPlanReq(b), sys: (b.messages || [])[0]?.content?.slice(0, 3000), roles: (b.messages || []).map(m => m.role) })) });
    await sleep(100);
    process.exit(0);
  })();
} else {
  main().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
}

async function runChild(role, env, onMessage) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [__filename], {
      env: { ...process.env, PHASETEST_ROLE: role, SNH_DATA_DIR: TMP, ...env },
      stdio: ['ignore', 'pipe', 'pipe', 'ipc']
    });
    let out = '';
    child.stdout.on('data', d => { out += d; });
    child.stderr.on('data', d => { out += d; });
    child.on('message', m => onMessage && onMessage(m, child));
    child.on('exit', (code, signal) => resolve({ code, signal, out }));
  });
}

async function main() {
  await new Promise(resolve => { engine.server = http.createServer(scriptedServe); engine.server.listen(0, '127.0.0.1', () => { engine.port = engine.server.address().port; over.enginePort = engine.port; resolve(); }); });
  console.log(`\nLong jobs survive their window (store: ${TMP}, engine on :${engine.port}, web tools ${hasWeb ? 'registered' : 'NOT registered — web sections will be thin'})\n`);

  // =========================================================================
  console.log('── PART 1a. The reservation fits the room left (pure) ──');
  const full = jobWindow.fitReservation({ window: 131072, promptTokens: 20000, answerTokens: 16448, thinkingTokens: 32768, margin: 512 });
  check('with room, a round reserves the full answer + thinking', full.fits && !full.squeezed && full.maxTokens === 49216, JSON.stringify(full));
  const nine17 = jobWindow.fitReservation({ window: 131072, promptTokens: 81857, answerTokens: 16448, thinkingTokens: 32768, floorAnswer: 1024, floorThinking: 1024, margin: 512 });
  check('the 9/17 round (prompt 81,857) FITS once the answer is squeezed — it did not have to die', nine17.fits && nine17.squeezed && nine17.maxTokens === 131072 - 81857 - 512 && nine17.thinking === 32768, JSON.stringify(nine17));
  const tight = jobWindow.fitReservation({ window: 131072, promptTokens: 120000, answerTokens: 16448, thinkingTokens: 32768, floorAnswer: 1024, floorThinking: 1024, margin: 512 });
  check('  tighter: the thinking shrinks too, down to its floor', tight.fits && tight.thinking < 32768 && tight.thinking >= 1024 && tight.maxTokens === 131072 - 120000 - 512, JSON.stringify(tight));
  const none = jobWindow.fitReservation({ window: 131072, promptTokens: 130000, answerTokens: 16448, thinkingTokens: 32768, floorAnswer: 1024, floorThinking: 1024, margin: 512 });
  check('  below the floors it says so instead of sending a request the engine will refuse', none.fits === false && /floor/.test(none.why), JSON.stringify(none));
  const noWin = jobWindow.fitReservation({ window: null, promptTokens: 1e6, answerTokens: 100, thinkingTokens: 50 });
  check('  no window known = the request the runner always sent', noWin.fits && noWin.maxTokens === 150 && !noWin.squeezed);
  const said = jobWindow.parseContextRefusal("This model's maximum context length is 131072 tokens. However, you requested 49216 output tokens and your prompt contains at least 81857 in");
  check('the engine\'s refusal is read back into numbers (the real 9/17 sentence)', said && said.limit === 131072 && said.requested === 49216 && said.prompt === 81857, JSON.stringify(said));
  const cal = jobWindow.createCalibrator(4);
  const before = cal.estimate(199073);
  cal.observe(199073, 81857);
  const after = cal.estimate(199073);
  check('the calibrator learns the real ratio from the engine\'s count (9/17 ran 2.4 chars/token, not 4)', before < 60000 && after >= 81857 && after < 90000, `${before} → ${after}, ratio ${cal.ratio.toFixed(2)}`);
  check('the loud sentence carries the numbers', /131,072-token context window/.test(jobWindow.refusalSentence({ window: 131072, promptTokens: 81857, requested: 49216 })) && /81,857/.test(jobWindow.refusalSentence({ window: 131072, promptTokens: 81857 })));

  // =========================================================================
  console.log('\n── PART 2a. Compaction keeps the finding and the source, drops the page (pure) ──');
  const convo = [{ role: 'system', content: 'sys' }, { role: 'user', content: BRIEF }];
  const addRound = (n) => {
    convo.push({ role: 'assistant', content: '', reasoning: `round ${n} thoughts`, tool_calls: [{ id: `s${n}`, type: 'function', function: { name: 'web_search', arguments: '{}' } }, { id: `f${n}`, type: 'function', function: { name: 'web_fetch', arguments: '{}' } }] });
    convo.push({ role: 'tool', tool_call_id: `s${n}`, name: 'web_search', content: JSON.stringify(SEARCH(`q${n}`)) });
    convo.push({ role: 'tool', tool_call_id: `f${n}`, name: 'web_fetch', content: JSON.stringify({ url: `https://example.test/q${n}/2`, content: PAGE(n), truncated: false }) });
  };
  for (let n = 1; n <= 5; n++) addRound(n);
  const urlsBefore = compaction.urlsIn(convo);
  const charsBefore = jobWindow.transcriptChars(convo);
  let digestCalls = 0;
  const pinnedDigest = async (sys, user) => {
    digestCalls++;
    const idx = [...user.matchAll(/=== SOURCE (\d+) ===/g)].map(m => Number(m[1]));
    return { content: JSON.stringify({ sources: idx.map(i => ({ i, findings: `- decode 17 tok/s, ubatch 2048 (source ${i})` })) }) };
  };
  const c1 = await compaction.compact(convo, { round: 6, recentRounds: 2, hardAfter: 6, task: BRIEF, callLLM: pinnedDigest, snippetChars: 120, truncateChars: 300 });
  const r1 = c1.report;
  check('rounds older than the recent window were compacted, the recent two were not', JSON.stringify(r1.roundsCompacted) === '[1,2,3]', JSON.stringify(r1.roundsCompacted));
  check('  6 entries: 3 searches trimmed, 3 pages digested in ONE model call', r1.entries === 6 && r1.trimmed === 3 && r1.digested === 3 && digestCalls === 1, JSON.stringify({ ...r1, digestCalls }));
  check('  and the three compacted rounds shrank by more than two thirds', (r1.charsBefore - r1.charsAfter) > r1.charsBefore * 0.66 && jobWindow.transcriptChars(c1.convo) < charsBefore * 0.6, `${charsBefore} → ${jobWindow.transcriptChars(c1.convo)}; compacted ${r1.charsBefore} → ${r1.charsAfter}`);
  const urlsAfter = compaction.urlsIn(c1.convo);
  const lost = [...urlsBefore].filter(u => !urlsAfter.has(u));
  check('PROVENANCE SURVIVES: every URL present before compaction is present after it', lost.length === 0, `lost ${lost.length}: ${lost.slice(0, 3).join(', ')}`);
  const digest = JSON.parse(c1.convo[4].content);
  check('  a digested page carries source {title, url, date} put there by the RUNNER, not the model',
    digest._compacted === 'digest' && digest.source && digest.source.url === 'https://example.test/q1/2' && digest.source.title === 'q1 — result 2' && /^2026-09-02/.test(digest.source.date) && /decode 17 tok\/s/.test(digest.findings),
    JSON.stringify(digest).slice(0, 300));
  const trimmed = JSON.parse(c1.convo[3].content);
  check('  a trimmed search keeps every hit\'s title, url and date, with a short snippet', trimmed._compacted === 'trim' && trimmed.results.length === 6 && trimmed.results.every(h => h.url && h.title && h.date) && trimmed.results[0].snippet.length <= 120, JSON.stringify(trimmed).slice(0, 200));
  check('  the recent rounds are byte-identical', c1.convo[13].content === convo[13].content && c1.convo[15].content === convo[15].content);
  check('  the original array was not mutated', !compaction.isCompacted(convo[4]));
  const c2 = await compaction.compact(c1.convo, { round: 6, recentRounds: 2, task: BRIEF, callLLM: pinnedDigest });
  check('a second pass over the same rounds changes nothing (idempotent)', c2.report.entries === 0 && digestCalls === 1, JSON.stringify(c2.report));
  const failing = async () => { throw new Error('engine busy'); };
  const c3 = await compaction.compact(convo, { round: 6, recentRounds: 2, task: BRIEF, callLLM: failing, truncateChars: 300 });
  check('when the digest call fails, pages are TRUNCATED instead, marked, and the report says so', c3.report.truncated === 3 && c3.report.digestFailed === 3 && c3.report.digested === 0 && /_compacted":"truncate/.test(c3.convo[4].content) && /chars dropped/.test(c3.convo[4].content), JSON.stringify(c3.report));
  check('  and provenance still survives the fallback', [...urlsBefore].every(u => compaction.urlsIn(c3.convo).has(u)));
  const c4 = await compaction.compact(convo, { round: 6, recentRounds: 2, hard: true, task: BRIEF, callLLM: pinnedDigest });
  check('pressure mode compacts everything eligible, drops old reasoning, cuts old searches to five hits', c4.report.mode === 'pressure' && c4.report.roundsCompacted.length === 5 && c4.report.reasoningDropped >= 4 && JSON.parse(c4.convo[3].content).results.length === 5, JSON.stringify(c4.report));
  const srcs = compaction.sourcesIn(c1.convo);
  check('sourcesIn lists every page read and every hit found, with title/url/date, from a compacted transcript', srcs.filter(s => s.how === 'read').length === 5 && srcs.filter(s => s.how === 'found').length >= 25 && srcs.every(s => s.url), `${srcs.length} sources`);

  // =========================================================================
  console.log('\n── PART 3a. A three-phase job, end to end ──');
  setWindow(9000);
  engine.requests = [];
  fetchCounter = 0;
  engine.script = [
    ...phaseSteps(1, 3, 'Phase 1 findings: UD-Q4_K_XL is 111.3 GB (Unsloth, https://example.test/phase-1/1a, 2026-09-02).'),
    ...phaseSteps(2, 2, 'Phase 2 findings: 17–25 tok/s decode on 2x3090 (Inovello, https://example.test/phase-2/1a).'),
    ...phaseSteps(3, 1, 'Phase 3 findings: expert cache PR pinned to b7100.'),
    { text: 'FINAL: 1) quants 111.3 GB; 2) 17–25 tok/s; 3) pin b7100. Sources as cited.' }
  ];
  let s = agentJobs.enqueue({ title: 'Qwen3.8-Flash-Next substrate research (stub)', task: BRIEF });
  let j = await settle(s.id, 90000);
  check('the job is `ok`', j.status === 'ok', `${j.status}: ${j.error}`);
  check('  the result is the synthesis written from the document', /FINAL: 1\) quants/.test(j.result_text || ''), (j.result_text || '').slice(0, 120));
  check('  the planner was asked once, and its answer became three phases', engine.requests.filter(isPlanReq).length === 1 && progressOf(j) && progressOf(j).phases.length === 3, JSON.stringify(progressOf(j) && progressOf(j).phases.map(p => p.status)));
  let prog = progressOf(j);
  check('  every phase is `done`, with its calls and rounds on the record', prog.phases.every(p => p.status === 'done') && prog.phases[0].calls === 4 && prog.phases[0].rounds === 3, JSON.stringify(prog.phases));
  check('  tool_calls on the row is the sum over phases', j.tool_calls === 4 + 3 + 1, String(j.tool_calls));
  check('  the window the runner planned against is the one the engine reported', prog.window && prog.window.tokens === 9000 && prog.window.source === 'engine', JSON.stringify(prog.window));
  check('  peak prompt size is on the record, from the engine\'s own count', Number.isFinite(prog.peakPromptTokens) && prog.peakPromptTokens > 1000, String(prog.peakPromptTokens));
  check('  compaction is on the record: rounds compacted and chars dropped', prog.compaction && prog.compaction.roundsCompacted >= 1 && prog.compaction.droppedChars > 2000, JSON.stringify(prog.compaction));
  check('  checkpoints were written — one per phase at least', prog.checkpoints >= 3, String(prog.checkpoints));
  check('every request asked the engine for usage', engine.requests.every(b => b.stream_options && b.stream_options.include_usage === true));
  const fitted = engine.requests.filter(b => Number.isFinite(b.max_tokens) && b.max_tokens < 3000);
  check('PART 1b: rounds were SQUEEZED as the prompt grew — max_tokens below the 3,000 full reservation', fitted.length >= 1 && prog.squeezedRounds >= 1, `${fitted.length} fitted request(s), squeezedRounds ${prog.squeezedRounds}`);
  check('  and none was ever refused: prompt + reservation stayed under 9,000 on every request', engine.requests.every(b => !Number.isFinite(b.max_tokens) || promptTokensOf(b) + b.max_tokens <= 9000));
  const doc = j.findings_path;
  check('the findings document exists, in the throwaway\'s own documents folder', doc && fs.existsSync(doc) && doc.startsWith(artifacts.outputDir()), doc);
  const text = doc ? fs.readFileSync(doc, 'utf8') : '';
  check('  with a section per phase, each carrying its writeup', /## Phase 1: Quant sizes/.test(text) && /111\.3 GB/.test(text) && /## Phase 2: Measured throughput/.test(text) && /## Phase 3: Known problems/.test(text));
  check('  and a sources list per phase from the RUNNER\'s record, URLs intact', /### Sources \(phase 1\)/.test(text) && /<https:\/\/example\.test\/phase-1\/1a>/.test(text) && /read/.test(text) && /found in search/.test(text));
  const feed = agentJobs.feed().find(f => f.id === s.id);
  check('  the feed names the document (and never its path)', feed && feed.findings_name === path.basename(doc) && feed.findings_location === path.dirname(doc) && !('findings_path' in feed), JSON.stringify({ n: feed && feed.findings_name, l: feed && feed.findings_location }));
  check('  the phase-2 sitting was told what phase 1 found, from the document, not the transcript', engine.requests.some(b => /PHASE 2 OF 3/.test((b.messages || [])[0]?.content || '') && /FINDINGS SO FAR/.test((b.messages || [])[0]?.content || '') && /111\.3 GB/.test((b.messages || [])[0]?.content || '') && (b.messages || []).filter(m => m.role === 'tool').length === 0));
  const synth = engine.requests.find(isSynthesisReq);
  check('  the synthesis had every phase\'s writeup and every source, and no tools', synth && /Phase 3: Known problems/.test(synth.messages.slice(-1)[0].content) && /EVERY SOURCE THE PHASES READ/.test(synth.messages.slice(-1)[0].content) && !synth.tools);
  check('  the checkpoint is gone once the job is ok, the document stays', !agentJobs.readCheckpoint(s.id) && fs.existsSync(doc));

  // =========================================================================
  console.log('\n── PART 3b. A phase past the split point is written up and continued fresh ──');
  setWindow(12000);
  over.agentJobs = { phases: { enabled: true, maxPhases: 6, carryChars: 12000, splitAtPercent: 55, partsPerPhase: 4 }, compaction: { enabled: true, recentRounds: 1, digestFetches: true, truncateChars: 300, snippetChars: 120 } };
  over.plan = { text: JSON.stringify({ phases: ['Quant sizes', 'Known problems'] }) };
  engine.requests = [];
  engine.script = [
    search('phase 1 q0'), fetch2('https://example.test/p1/a', 'https://example.test/p1/b'),
    // the split's part writeup lands wherever the runner asks for it; the
    // continued sitting then finishes the phase
    { text: 'Part findings so far: pages a–d read (https://example.test/p1/a).' },
    { text: 'Phase 1 findings, continued: done.' },
    ...phaseSteps(2, 1, 'Phase 2 findings.'),
    { text: 'FINAL with a split.' }
  ];
  s = agentJobs.enqueue({ title: 'split me', task: BRIEF });
  j = await settle(s.id, 90000);
  prog = progressOf(j);
  check('the job is `ok` after a split', j.status === 'ok', `${j.status}: ${j.error}`);
  check('  phase 1 ran in two sittings', prog && prog.phases[0].parts >= 1 && prog.phases[0].status === 'done', JSON.stringify(prog && prog.phases[0]));
  check('  the part writeup was a no-tools turn on the transcript that still fit', engine.requests.some(isPartWriteup) && engine.requests.filter(isPartWriteup).every(b => !b.tools));
  const doc2 = fs.readFileSync(j.findings_path, 'utf8');
  check('  the document carries both sittings, marked', /_\(sitting 1\)_/.test(doc2) && /Part findings so far/.test(doc2) && /continued: done/.test(doc2));
  check('  the continued sitting saw the part findings in its system prompt', engine.requests.some(b => /continued — part 2/.test((b.messages || [])[0]?.content || '') && /Part findings so far/.test((b.messages || [])[0]?.content || '')));
  over.plan = null;

  // =========================================================================
  console.log('\n── PART 1c. The engine refuses on context length — refit and retry, once ──');
  setWindow(9000);
  over.agentJobs = { phases: { enabled: false }, compaction: { enabled: true, recentRounds: 1, digestFetches: false, truncateChars: 300 }, context: { windowTokens: 20000, marginTokens: 100, floorAnswerTokens: 256, floorThinkingTokens: 0, charsPerToken: 3 } };
  // The runner is told 20,000 (pinned) while the engine enforces 9,000 — so
  // its first fit is wrong, the engine says so in vLLM's words, and the
  // runner must read the sentence, recalibrate, compact and refit.
  engine.requests = [];
  engine.script = [search('a'), fetch2('https://example.test/r/1', 'https://example.test/r/2'), fetch2('https://example.test/r/3', 'https://example.test/r/4'), { text: 'Refit answer.' }];
  s = agentJobs.enqueue({ title: 'refit me', task: 'Look up a thing and say what it is.' });
  j = await settle(s.id, 60000);
  prog = progressOf(j);
  check('the job is `ok` although the engine refused a round', j.status === 'ok' && /Refit answer/.test(j.result_text || ''), `${j.status}: ${j.error}`);
  check('  one refit retry is on the record, and the runner learned the real window', prog && prog.refitRetries === 1 && prog.window.tokens === 20000, JSON.stringify({ refit: prog && prog.refitRetries, win: prog && prog.window }));
  check('  the retried request fit under the real ceiling', engine.requests.slice(-1)[0] && promptTokensOf(engine.requests.slice(-1)[0]) + engine.requests.slice(-1)[0].max_tokens <= 9000);

  console.log('\n── 1d. The wall, when nothing more can be done, fails LOUDLY with the numbers ──');
  over.agentJobs = { phases: { enabled: false }, compaction: { enabled: false }, context: { windowTokens: 8000, marginTokens: 100, floorAnswerTokens: 2000, floorThinkingTokens: 0, charsPerToken: 3 } };
  setWindow(8000);
  engine.requests = [];
  engine.script = [search('a'), fetch2('https://example.test/w/1', 'https://example.test/w/2'), { text: 'never reached' }, { refuse: 503 }, { refuse: 503 }];
  s = agentJobs.enqueue({ title: 'hit the wall', task: 'Look up a thing.' });
  j = await settle(s.id, 60000);
  check('the job FAILED, stop_kind `context-window`, on the runner\'s side', j.status === 'failed' && j.stop_kind === 'context-window' && j.stop_source === 'runner', `${j.status} ${j.stop_source}/${j.stop_kind}`);
  check('  and the sentence says the window, the prompt and the floor in words', /8,000-token context window/.test(j.error || '') && /floor/.test(j.error || ''), j.error);
  check('  what it had is on the card as partial output', j.tool_calls === 3 && /web_search|web_fetch/.test(j.result_text || ''), `${j.tool_calls} calls`);
  over.agentJobs = {};

  // =========================================================================
  console.log('\n── PART 3c. KILLED mid-phase 2: phase 1\'s findings survive on disk, with sources ──');
  setWindow(9000);
  let killedId = null, childPort = null;
  const run = await runChild('child-run', {}, async (m, child) => {
    if (!m || !m.id) return;
    killedId = m.id; childPort = m.port;
    // Wait until phase 1 is on disk and phase 2 is in its hanging round.
    const until = Date.now() + 30000;
    while (Date.now() < until) {
      const ck = agentJobs.readCheckpoint(killedId);
      if (ck && Array.isArray(ck.phases) && ck.phases[0].status === 'done' && ck.phase === 2 && (ck.toolCalls || []).length >= 1) break;
      await sleep(50);
    }
    await sleep(300);
    child.kill('SIGKILL');
  });
  check('the child was killed, not finished', run.signal === 'SIGKILL', `${run.code}/${run.signal}`);
  j = job(killedId);
  check('  the row is still `running` — nothing closed it', j && j.status === 'running', j && j.status);
  let ck = agentJobs.readCheckpoint(killedId);
  check('  the checkpoint holds the phases: 1 done, 2 running', ck && ck.phases && ck.phases[0].status === 'done' && ck.phases[1].status === 'running' && ck.phase === 2, JSON.stringify(ck && ck.phases && ck.phases.map(p => p.status)));
  const killedDoc = ck && ck.findingsPath;
  const killedText = killedDoc && fs.existsSync(killedDoc) ? fs.readFileSync(killedDoc, 'utf8') : '';
  check('  PHASE 1\'S FINDINGS ARE ON DISK, written by the runner before the kill', /## Phase 1: Quant sizes/.test(killedText) && /111\.3 GB/.test(killedText), killedDoc);
  check('  with its sources intact', /### Sources \(phase 1\)/.test(killedText) && /<https:\/\/example\.test\/phase-1\/1a>/.test(killedText) && /<https:\/\/example\.test\/phase-1\/1b>/.test(killedText));

  console.log('\n── 3d. A new process resumes phase 2 from the checkpoint, and phase 1 is NOT redone ──');
  let restartMsgs = [];
  const restart = await runChild('child-restart', { PHASETEST_JOB_ID: killedId }, (m) => { if (m) restartMsgs.push(m); });
  j = job(killedId);
  const swept = restartMsgs.find(m => m.swept);
  check('the sweep re-queued it to resume', swept && swept.swept && swept.swept.requeued === 1, JSON.stringify(swept));
  check('  it finished ok in the new process', j && j.status === 'ok', `${j && j.status}: ${j && j.error}\n${restart.out.slice(-600)}`);
  check('  the result is the post-restart synthesis', /FINAL after restart/.test(j && j.result_text || ''), (j && j.result_text || '').slice(0, 100));
  const reqs = (restartMsgs.find(m => m.requests) || {}).requests || [];
  check('  no planner call: the phases came from the checkpoint', reqs.length > 0 && !reqs.some(r => r.plan), `${reqs.length} requests`);
  check('  the first resumed request was phase 2, carrying its earlier tool result', reqs[0] && /PHASE 2 OF 3/.test(reqs[0].sys || '') && reqs[0].roles.includes('tool'), JSON.stringify(reqs[0]));
  check('  and told the model the restart lost the round in progress', restart.out.includes('resuming after a restart'));
  prog = progressOf(j);
  check('  all three phases done on the row; phase 1 kept its original writeup', prog && prog.phases.every(p => p.status === 'done') && /111\.3 GB/.test(fs.readFileSync(j.findings_path, 'utf8')) && /Phase 2 findings after the restart/.test(fs.readFileSync(j.findings_path, 'utf8')), JSON.stringify(prog && prog.phases.map(p => p.status)));

  // =========================================================================
  console.log('\n── 3e. A phased job that FAILS reports what the finished phases found; its RETRY carries them ──');
  setWindow(9000);
  engine.requests = [];
  engine.script = [
    ...phaseSteps(1, 1, 'Phase 1 findings: 111.3 GB.'),
    search('phase 2 q'), { refuse: 503 }, { refuse: 503 }, { refuse: 503 }   // the round, then both salvage attempts find the engine gone
  ];
  s = agentJobs.enqueue({ title: 'fails in phase 2', task: BRIEF });
  j = await settle(s.id, 60000);
  check('the job is `failed`', j.status === 'failed', `${j.status}: ${j.error}`);
  check('  the card carries phase 1\'s findings, names where it stopped, and lists what was not reached', /Findings from 1 of 3 phases/.test(j.result_text || '') && /111\.3 GB/.test(j.result_text || '') && /stopped in phase 2/.test(j.result_text || '') && /Not reached/.test(j.result_text || '') && /Phase 3: Known problems/.test(j.result_text || ''), (j.result_text || '').slice(0, 300));
  check('  the feed marks it as partial output with a findings document', (() => { const f = agentJobs.feed().find(f => f.id === s.id); return f && f.partial_output && f.findings_name; })());
  const failedDoc = fs.readFileSync(j.findings_path, 'utf8');
  check('  the document says the job stopped, and why', /\*\*The job stopped:\*\*/.test(failedDoc) && /Phase 1: Quant sizes/.test(failedDoc));

  engine.requests = [];
  engine.script = [...phaseSteps(2, 1, 'Phase 2 findings, second attempt.'), ...phaseSteps(3, 1, 'Phase 3 findings.'), { text: 'FINAL from the retry.' }];
  const r = agentJobs.retry(s.id);
  check('retry() made a new row', r && r.ok && r.id, JSON.stringify(r));
  const rj = await settle(r.id, 60000);
  check('  the retry finished ok', rj && rj.status === 'ok' && /FINAL from the retry/.test(rj.result_text || ''), `${rj && rj.status}: ${rj && rj.error}`);
  prog = progressOf(rj);
  check('  it CARRIED phase 1 and ran only 2 and 3 — no planner, no phase-1 requests', prog && /carried 1 finished phase/.test(prog.plan) && !engine.requests.some(isPlanReq) && !engine.requests.some(b => /PHASE 1 OF 3/.test((b.messages || [])[0]?.content || '')), prog && prog.plan);
  check('  phase 1\'s writeup is in the retry\'s document and its result', /111\.3 GB/.test(fs.readFileSync(rj.findings_path, 'utf8')) && /Phase 2 findings, second attempt/.test(fs.readFileSync(rj.findings_path, 'utf8')));

  // =========================================================================
  console.log('\n── A short brief is ONE phase, with no planning call and no document ──');
  engine.requests = [];
  engine.script = [{ tools: [{ name: 'memory_count', args: {} }] }, { text: 'There are N facts.' }];
  s = agentJobs.enqueue({ title: 'count', task: 'Count the facts in memory.' });
  j = await settle(s.id, 30000);
  check('ok, no planner request, no findings document, result on the card', j.status === 'ok' && !engine.requests.some(isPlanReq) && !j.findings_path && /N facts/.test(j.result_text || ''), `${j.status} ${j.findings_path}`);
  prog = progressOf(j);
  check('  but the window and the peak prompt are still on the record', prog && prog.multi === false && prog.window.tokens === 9000 && prog.peakPromptTokens > 0, JSON.stringify(prog));

  // =========================================================================
  engine.server.close();
  console.log(`\n=== ${pass} passed, ${fail} failed ===`);
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
  process.exit(fail ? 1 : 0);
}
