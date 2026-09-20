#!/usr/bin/env node
/**
 * THE CONVERSATION REVIEW: one step at a time, memory saved before anything
 * closes, resumable, and reported back where she asked.
 *
 * 2026-09-09: Athena reviewed five open conversations in one chat turn and the
 * turn died 31 minutes in. This drives the review as it runs now — a
 * background job through the REAL queue (db/agent-jobs.js), the real channel
 * (archive, requestRetire, sendInto), the real judge call over HTTP to a
 * scripted engine — and asserts the things that were luck last time:
 *
 *   - a conversation the entity opened is closed by the entity; one Ellie
 *     opened becomes a request to her — from the stored `initiated_by`
 *   - the memory save lands BEFORE the close, every time, and a save that
 *     fails leaves the conversation open
 *   - a conversation with a paused job in it is skipped, and nobody can
 *     archive it — not the entity, not the sidebar
 *   - a process killed mid-review is resumed by a new process from the exact
 *     step it reached: nothing re-judged, nothing skipped
 *   - the report is one assistant message in the conversation she asked in,
 *     unread until she opens or acknowledges it, with a link per conversation
 *   - a self-archived conversation can be reopened
 *   - the budget ask, answered yes and no
 *
 * Only the memory write is stubbed (its classifier and embedder are their own
 * suites); the stub records WHEN it ran so the order can be asserted.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-conversation-review.js
 */
process.env.TZ = 'America/Los_Angeles';
const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-conversation-review.js');
  process.exit(1);
}
const TMP = process.env.SNH_DATA_DIR;
const ROOT = path.join(__dirname, '..');
const ROLE = process.env.REVIEWTEST_ROLE || 'main';

const config = require(path.join(ROOT, 'db/config'));
const realGetConfig = config.getConfig;
const over = { enginePort: Number(process.env.REVIEWTEST_ENGINE_PORT || 0), agentJobs: {}, review: {}, tools: {} };
config.getConfig = () => {
  const c = realGetConfig();
  return {
    ...c,
    models: { ...c.models, heartbeat: { provider: 'vllm', instance: 'stub', model: 'stub-model' } },
    agentJobs: Object.assign({
      enabled: true, maxQueued: 10, maxStartsPerHour: 100, maxToolCallsPerJob: 40, maxWallClockMs: 900000, maxRoundsPerJob: 16,
      maxAttempts: 2, retryGraceMinutes: 30, retentionDays: 90, askBeforeCeiling: false, askAtPercent: 80, extensionPercent: 50
    }, over.agentJobs),
    conversationReview: Object.assign({ enabled: true, maxFactsPerConversation: 3, transcriptChars: 12000, maxConversationsPerReview: 25, maxConsecutiveFailures: 3 }, over.review),
    generation: { ...c.generation, agentJobResponseTokens: 512, agentJobThinkingTokens: null, stallTimeoutMs: 20000, firstTokenTimeoutMs: 20000 },
    tools: { ...c.tools, conversations: Object.assign({ enabled: true, maxSendsPerHour: 100, selfArchive: true }, over.tools) },
    watchdog: { ...(c.watchdog || {}), enabled: false },
    brainCircuit: { consecutiveTimeoutsToOpen: 1000 }
  };
};
config.getProviderInstance = () => ({ host: `http://127.0.0.1:${over.enginePort}` });

const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();
const mm = require(path.join(ROOT, 'db/memory-manager'));
const agentJobs = require(path.join(ROOT, 'db/agent-jobs'));
const channel = require(path.join(ROOT, 'db/conversation-channel'));
const review = require(path.join(ROOT, 'db/conversation-review'));
const memoryWrite = require(path.join(ROOT, 'db/memory-write'));
const budgetAsk = require(path.join(ROOT, 'db/job-budget-ask'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${String(detail).slice(0, 320)}` : ''}`); }
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const job = (id) => db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id);
async function settle(id, ms = 30000, statuses = agentJobs.TERMINAL) {
  const until = Date.now() + ms;
  while (Date.now() < until) { const j = job(id); if (j && statuses.includes(j.status)) return j; await sleep(25); }
  return job(id);
}

// ---- the scripted engine (judge answers) ------------------------------------
const engine = { script: [], requests: [], server: null, port: 0 };
function sse(res, obj) { res.write(`data: ${JSON.stringify(obj)}\n\n`); }
async function serve(req, res) {
  let raw = ''; for await (const chunk of req) raw += chunk;
  let body = {}; try { body = JSON.parse(raw); } catch { /* */ }
  engine.requests.push(body);
  const step = engine.script.length ? engine.script.shift() : { text: '{"finished": false, "why": "default: not finished", "keep": []}' };
  if (typeof step === 'function') return step(req, res, body);
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  sse(res, { choices: [{ delta: { content: step.text }, finish_reason: null }] });
  sse(res, { choices: [{ delta: {}, finish_reason: 'stop' }] });
  res.write('data: [DONE]\n\n'); res.end();
}
function startEngine() {
  return new Promise(resolve => {
    engine.server = http.createServer((req, res) => { serve(req, res).catch(() => { try { res.end(); } catch { /* */ } }); });
    engine.server.listen(0, '127.0.0.1', () => { engine.port = engine.server.address().port; over.enginePort = engine.port; resolve(); });
  });
}
const judge = (finished, why, keep = []) => ({ text: JSON.stringify({ finished, why, keep }) });

// ---- the memory write, stubbed and recorded ---------------------------------
const events = [];   // the order things happened in
const writeStub = { failFor: new Set() };
memoryWrite.write = async ({ statement, conversationId }) => {
  events.push({ kind: 'save', conversationId, statement, at: Date.now() });
  if (writeStub.failFor.has(conversationId)) return { ok: false, error: 'embedding provider down' };
  return { ok: true, subject: 'user', factId: `stub-${events.length}` };
};
const realArchive = channel.archive;
channel.archive = (id, opts) => { const r = realArchive(id, opts); events.push({ kind: 'archive', conversationId: id, at: Date.now() }); return r; };
const realRequest = channel.requestRetire;
channel.requestRetire = async (id, reason) => { const r = await realRequest(id, reason); events.push({ kind: 'request', conversationId: id, at: Date.now() }); return r; };

// ---- fixtures --------------------------------------------------------------
function makeConversation(title, by, turns, { minutesAgo = 60 } = {}) {
  const id = database.createConversation(title, 'stub', by);
  for (const [role, content] of turns) database.addMessage(id, role, content, 'stub');
  const t = new Date(Date.now() - minutesAgo * 60000).toISOString().slice(0, 19).replace('T', ' ');
  db.prepare('UPDATE conversations SET updated_at = ? WHERE id = ?').run(t, id);
  return id;
}

// ===========================================================================
if (ROLE === 'child-run') {
  (async () => {
    const started = review.enqueueReview({ conversationId: process.env.REVIEWTEST_ASK_CONV });
    process.send && process.send({ id: started.id, error: started.error });
    await settle(started.id, 60000);
    process.exit(0);
  })();
} else if (ROLE === 'child-restart') {
  (async () => {
    over.agentJobs = JSON.parse(process.env.REVIEWTEST_AGENTJOBS || '{}');
    agentJobs.startup();
    await settle(process.env.REVIEWTEST_JOB_ID, 30000);
    process.exit(0);
  })();
} else {
  main().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
}

function runChild(role, env, onMessage) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [__filename], {
      env: { ...process.env, REVIEWTEST_ROLE: role, REVIEWTEST_ENGINE_PORT: String(engine.port), SNH_DATA_DIR: TMP, ...env },
      stdio: ['ignore', 'pipe', 'pipe', 'ipc']
    });
    let out = '';
    child.stdout.on('data', d => { out += d; }); child.stderr.on('data', d => { out += d; });
    child.on('message', m => onMessage && onMessage(m, child));
    child.on('exit', (code, signal) => resolve({ code, signal, out }));
  });
}

async function main() {
  await startEngine();
  console.log(`\nConversation review (store: ${TMP}, engine :${engine.port})\n`);

  // =========================================================================
  console.log('── 0. Her ask is recognised, narrowly ──');
  check('the 9/9 sentence itself', review.looksLikeReviewAsk('if you would like to look at all your open messages to me and if you are satisfied with the conversation send me a request for them to be archived.'));
  check('"go through your open conversations and close the finished ones"', review.looksLikeReviewAsk('Can you go through your open conversations and close the finished ones?'));
  check('"how did the job go?" is not', !review.looksLikeReviewAsk('how did the job go?'));
  check('"can I archive this message?" is not (one conversation, hers)', !review.looksLikeReviewAsk('can i archive this message?'));

  // =========================================================================
  console.log('\n── 1. A review that closes some, requests others, leaves the rest ──');
  const ask = makeConversation('Tidy up', 'user', [['user', 'look through your open conversations and archive the ones you are done with']], { minutesAgo: 1 });
  const A = makeConversation('On breathing and thinking', 'snh', [['assistant', 'I turned a remark into a project.'], ['user', 'that is who you are'], ['assistant', 'accepted, no framework']], { minutesAgo: 500 });
  const B = makeConversation('Hey if we build you a sandbox', 'user', [['user', 'what would you use it for?'], ['assistant', 'a place to run what I write'], ['user', 'filed on the todo list']], { minutesAgo: 400 });
  const C = makeConversation('Two notes out of step', 'snh', [['assistant', 'are these two notes in conflict?'], ['user', 'let me think about it']], { minutesAgo: 300 });
  const D = makeConversation('The door watch', 'snh', [['assistant', 'shall I plan it?'], ['user', 'yes']], { minutesAgo: 200 });
  const E = makeConversation('The 4 Dog Army song', 'user', [['user', 'write me a song about the dogs'], ['assistant', 'here it is'], ['user', 'perfect, thanks']], { minutesAgo: 100 });
  // D has a paused job tied to it — asked her there, waiting.
  const pausedJobId = require('crypto').randomUUID();
  db.prepare("INSERT INTO agent_jobs (id, title, task, status, source, conversation_id, paused_at, ask_json) VALUES (?, 'door-watch plan', 't', 'paused', 'chat-handoff', ?, ?, ?)")
    .run(pausedJobId, D, new Date().toISOString(), JSON.stringify({ delivery: { conversationId: D }, text: 'may I have more?' }));
  writeStub.failFor.add(E);

  engine.script = [
    judge(true, 'Settled: you said this is who I am and I accepted it.', ['I tend to turn a passing remark into a project.', 'Ellie has said that turning remarks into projects is simply who I am.']),   // A → close
    judge(true, 'Answered, and the next step lives on the todo list.', ['Ellie keeps her follow-ups for me on her todo list.']),   // B → request
    judge(false, 'You said you would think about it and have not come back yet.'),   // C → open
    // D is skipped before any call
    judge(true, 'Done — the song was delivered and you liked it.', ['Ellie has four dogs she calls the 4 Dog Army.'])   // E → save fails
  ];
  events.length = 0;
  let r = review.enqueueReview({ conversationId: ask });
  check('the review is a job on the queue', r.ok && r.id, JSON.stringify(r));
  check('a second review is refused while one is going', review.enqueueReview({ conversationId: ask }).ok === false);
  let j = await settle(r.id);
  check('it finished ok', j.status === 'ok', `${j.status} ${j.error}`);
  const st = (id) => channel.getState(id);
  check('A (mine, finished) is ARCHIVED, by the entity', st(A).status === 'archived' && st(A).archived_by === 'snh', JSON.stringify({ s: st(A).status, by: st(A).archived_by }));
  check('  and the ledger says the entity closed its own conversation', !!db.prepare("SELECT 1 FROM corrections_ledger WHERE action = 'conversation-archive' AND target_id = ? AND subject = 'snh' AND reason LIKE '%closed by the entity on its own%'").get(A));
  check('B (hers, finished) is NOT archived — it is a REQUEST to her', st(B).status === 'active' && !!st(B).retire_requested_at && !!st(B).retire_initiative_id, JSON.stringify({ s: st(B).status, req: st(B).retire_requested_at }));
  check('  and the request is a proposal on the bell', db.prepare("SELECT type FROM initiatives WHERE id = ?").get(st(B).retire_initiative_id)?.type === 'proposal');
  check('C (mine, not finished) is left open, untouched', st(C).status === 'active' && !st(C).retire_requested_at);
  check('D (paused job in it) was SKIPPED with no model call', st(D).status === 'active' && !st(D).retire_requested_at);
  check('E (save failed) is NOT closed and NOT requested', st(E).status === 'active' && !st(E).retire_requested_at, JSON.stringify({ s: st(E).status, req: st(E).retire_requested_at }));
  check('four judge calls were made (D needed none)', engine.requests.length === 4, String(engine.requests.length));
  const ev = (k, c) => events.filter(e => e.kind === k && e.conversationId === c);
  const archivedAtMs = (id) => { const row = db.prepare("SELECT created_at FROM corrections_ledger WHERE action = 'conversation-archive' AND target_id = ? ORDER BY created_at DESC LIMIT 1").get(id); return row ? Date.parse(row.created_at) : null; };
  check('A: both saves landed BEFORE the archive', ev('save', A).length === 2 && archivedAtMs(A) !== null && ev('save', A).every(s => s.at <= archivedAtMs(A) + 1), JSON.stringify({ saves: ev('save', A).map(s => s.at), archived: archivedAtMs(A) }));
  check('B: the save landed BEFORE the request', ev('save', B).length === 1 && ev('request', B).length === 1 && ev('save', B)[0].at <= ev('request', B)[0].at);
  check('E: the save was attempted and nothing followed it', ev('save', E).length === 1 && ev('archive', E).length === 0 && ev('request', E).length === 0);
  check('C: no save was attempted for an unfinished conversation', ev('save', C).length === 0);
  const ck = agentJobs.readCheckpoint(r.id);
  const stateOf = (id) => (ck.items.find(i => i.id === id) || {}).state;
  check('the record on disk says what happened to each', stateOf(A) === 'closed' && stateOf(B) === 'requested' && stateOf(C) === 'left-open' && stateOf(D) === 'skipped' && stateOf(E) === 'save-failed', JSON.stringify(ck.items.map(i => [i.title, i.state])));
  check('  the skip names the paused job', /paused, waiting on your answer/.test(ck.items.find(i => i.id === D).why), ck.items.find(i => i.id === D).why);
  check('  and the save failure carries its reason', /embedding provider down/.test(ck.items.find(i => i.id === E).error));
  check('the conversation she asked in was not reviewed', !ck.items.some(i => i.id === ask));

  console.log('\n── 1b. The report: one message where she asked, unread, with a link per conversation ──');
  const msgs = db.prepare("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC").all(ask);
  const report = msgs[msgs.length - 1];
  check('one assistant message landed in the conversation she asked in', msgs.length === 2 && report.role === 'assistant', JSON.stringify(msgs.map(m => m.role)));
  check('  it is the job\'s result text too', job(r.id).result_text === report.content);
  check('  it says what was closed on its own, with a link', new RegExp(`Closed on my own[\\s\\S]*\\[\\[conversation:${A}\\|On breathing and thinking\\]\\]`).test(report.content), report.content);
  check('  and what is waiting on her approval, with a link', new RegExp(`Waiting on your approval[\\s\\S]*\\[\\[conversation:${B}\\|`).test(report.content));
  check('  and what was left open, and why', new RegExp(`Left open, not finished[\\s\\S]*${C}[\\s\\S]*have not come back`).test(report.content));
  check('  and the one whose save failed, with the reason', /could not save what I wanted to keep[\s\S]*embedding provider down/.test(report.content));
  check('  and the skipped one, naming what is using it', /Skipped[\s\S]*door-watch plan/.test(report.content));
  check('  and how many things it saved first', /saved 3 things to memory/.test(report.content), report.content.match(/saved [^\n]*/)?.[0]);
  check('the conversation is UNREAD by one — the blinking count', channel.unreadFor(ask) === 1, String(channel.unreadFor(ask)));
  check('  and the sidebar total counts it', channel.totalUnread() >= 1);
  check('  the job is already stamped announced — the message IS the announcement', !!job(r.id).announced_at);
  check('  so the next turn will not announce it a second time', !agentJobs.pendingAnnouncements().some(p => p.id === r.id));
  channel.markRead(ask);
  check('opening it (or the read route) clears it', channel.unreadFor(ask) === 0);

  console.log('\n── 2. Nobody can archive a conversation something is using ──');
  let threw = null;
  try { channel.archive(D, { by: 'user' }); } catch (e) { threw = e; }
  check('Ellie\'s archive of D is refused with the reason', threw && threw.code === 'OPEN_ITEMS' && /door-watch plan/.test(threw.message), threw && threw.message);
  const self = await channel.archiveBySelf(D, { reason: 'done' }).catch(e => ({ error: e.message }));
  check('  and so is the entity\'s', self.error && /door-watch plan/.test(self.error), JSON.stringify(self));
  check('  D is still active', st(D).status === 'active');
  check('openItemsFor names the paused job', channel.openItemsFor(D).some(i => i.id === pausedJobId && /paused/.test(i.state)));
  check('  and nothing for a quiet conversation', channel.openItemsFor(C).length === 0);

  console.log('\n── 3. The entity closing hers is turned into a request ──');
  const F = makeConversation('Her thread', 'user', [['user', 'hi'], ['assistant', 'hello']]);
  const rF = await channel.archiveBySelf(F, { reason: 'seems done' });
  check('archiveBySelf on a conversation ELLIE started: not archived, requested', rF.archived === false && rF.requested === true && st(F).status === 'active' && !!st(F).retire_requested_at, JSON.stringify(rF));
  check('  and the note says why', /hers to close/.test(rF.note));
  const { ConversationArchiveTool } = require(path.join(ROOT, 'mcp/tools/conversations'));
  const G = makeConversation('My thread', 'snh', [['assistant', 'a thought'], ['user', 'noted']]);
  const tool = new ConversationArchiveTool();
  const tr = await tool.execute({ conversation_id: G, reason: 'noted and done' });
  check('the conversation_archive tool closes one the entity opened', tr.archived === true && st(G).status === 'archived' && st(G).archived_by === 'snh', JSON.stringify(tr));
  over.tools = { selfArchive: false };
  const H = makeConversation('Mine, but the switch is off', 'snh', [['assistant', 'x'], ['user', 'y']]);
  const th = await tool.execute({ conversation_id: H, reason: 'done' });
  check('with the Settings switch off, even its own goes to her as a request', th.archived === false && th.requested === true && st(H).status === 'active' && !!st(H).retire_requested_at, JSON.stringify(th));
  over.tools = {};

  console.log('\n── 4. She can reopen anything the entity closed ──');
  const back = channel.unarchive(A, { by: 'user' });
  check('A is active again', back.status === 'active' && st(A).archived_by === null);
  check('  ledgered as reopened by Ellie', !!db.prepare("SELECT 1 FROM corrections_ledger WHERE action = 'conversation-unarchive' AND target_id = ?").get(A));
  channel.archive(A, { by: 'snh', reason: 'closed again for the next test' });

  // =========================================================================
  console.log('\n── 5. A process killed mid-review, resumed by a new process ──');
  // Fresh conversations, all mine, all finished; the third judge call hangs
  // forever and the child is killed while it waits there.
  const ask2 = makeConversation('Second tidy', 'user', [['user', 'go through them again']], { minutesAgo: 1 });
  for (const id of [B, C, E, F, H]) db.prepare("UPDATE conversations SET status = 'archived', archived_by = 'user', archived_at = ? WHERE id = ?").run(channel.sqlNow(), id);
  db.prepare("UPDATE agent_jobs SET status = 'cancelled' WHERE id = ?").run(pausedJobId);
  channel.archive(D, { by: 'user' });
  channel.archive(ask, { by: 'user' });
  const K1 = makeConversation('K1', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 90 });
  const K2 = makeConversation('K2', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 80 });
  const K3 = makeConversation('K3', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 70 });
  const K4 = makeConversation('K4', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 60 });
  engine.requests.length = 0;
  events.length = 0;
  engine.script = [
    judge(true, 'K1 done', ['fact from K1']),
    judge(true, 'K2 done', ['fact from K2', 'second fact from K2']),
    (req, res) => { res.writeHead(200, { 'Content-Type': 'text/event-stream' }); }   // K3: hangs
  ];
  let killedId = null;
  const run = await runChild('child-run', { REVIEWTEST_ASK_CONV: ask2 }, async (m, child) => {
    if (m && m.id) {
      killedId = m.id;
      for (let i = 0; i < 400; i++) {
        const ck2 = agentJobs.readCheckpoint(killedId);
        if (ck2 && ck2.items && ck2.items.filter(x => x.state === 'closed').length >= 2 && engine.requests.length >= 3) break;
        await sleep(50);
      }
      child.kill('SIGKILL');
    }
  });
  check('the child was killed mid-review', run.signal === 'SIGKILL' && !!killedId, `${run.signal} ${run.out.slice(-200)}`);
  let ck2 = agentJobs.readCheckpoint(killedId);
  check('the record shows K1 and K2 closed and K3/K4 pending', ck2 && ck2.items.filter(x => x.state === 'closed').length === 2 && ck2.items.filter(x => x.state === 'pending').length === 2, JSON.stringify(ck2 && ck2.items.map(i => [i.title, i.state])));
  check('K1 and K2 are archived; K3 and K4 are not', st(K1).status === 'archived' && st(K2).status === 'archived' && st(K3).status === 'active' && st(K4).status === 'active');
  // NOTE: the child's memory-write stub lives in the child; here only the
  // archive/request records matter, and the transcript of judge calls.
  engine.requests.length = 0;
  engine.script = [judge(true, 'K3 done', []), judge(true, 'K4 done', ['fact from K4'])];
  const restart = await runChild('child-restart', { REVIEWTEST_JOB_ID: killedId });
  j = job(killedId);
  check('a new process swept it, resumed it and finished ok', j.status === 'ok', `${j.status} ${j.error} ${restart.out.slice(-300)}`);
  check('  only K3 and K4 were judged after the restart — K1 and K2 were not re-judged', engine.requests.length === 2, String(engine.requests.length));
  check('  the two judged were K3 then K4', /K3/.test((engine.requests[0].messages || []).map(m => m.content).join(' ')) && /K4/.test((engine.requests[1].messages || []).map(m => m.content).join(' ')));
  check('  all four are archived now', [K1, K2, K3, K4].every(id => st(id).status === 'archived' && st(id).archived_by === 'snh'));
  check('  it counted as the second attempt', j.attempts === 2, String(j.attempts));
  check('  and the new process picked up FROM THE RECORD: 2 done, 2 to go', /resuming \(restart\): 2 done, 2 to go of 4/.test(restart.out), restart.out.slice(-400));
  const rep2 = db.prepare("SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY timestamp DESC, rowid DESC LIMIT 1").get(ask2);
  check('  the report lists all four as closed', rep2 && [K1, K2, K3, K4].every(id => rep2.content.includes(`[[conversation:${id}|`)), rep2 && rep2.content);
  check('  and she has it unread', channel.unreadFor(ask2) === 1);

  console.log('\n── 5b. Killed, and NOT run again: the card still says where each one got to ──');
  const ask2b = makeConversation('Second tidy, again', 'user', [['user', 'once more']], { minutesAgo: 1 });
  channel.archive(ask2, { by: 'user' });
  const N1 = makeConversation('N1', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 90 });
  const N2 = makeConversation('N2', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 80 });
  engine.requests.length = 0;
  engine.script = [judge(true, 'N1 done', []), (req, res) => { res.writeHead(200, { 'Content-Type': 'text/event-stream' }); }];
  killedId = null;
  await runChild('child-run', { REVIEWTEST_ASK_CONV: ask2b }, async (m, child) => {
    if (m && m.id) {
      killedId = m.id;
      for (let i = 0; i < 400; i++) {
        const c3 = agentJobs.readCheckpoint(killedId);
        if (c3 && c3.items && c3.items.some(x => x.state === 'closed') && engine.requests.length >= 2) break;
        await sleep(50);
      }
      child.kill('SIGKILL');
    }
  });
  engine.requests.length = 0;
  await runChild('child-restart', { REVIEWTEST_JOB_ID: killedId, REVIEWTEST_AGENTJOBS: JSON.stringify({ maxAttempts: 1 }) });
  j = job(killedId);
  check('closed as interrupted, by the runner, with no engine call', j.status === 'interrupted' && j.stop_kind === 'service-restart' && engine.requests.length === 0, `${j.status} ${j.stop_kind} ${engine.requests.length}`);
  check('  the card is the report: N1 closed, N2 not looked at', /Closed on my own[\s\S]*N1/.test(j.result_text || '') && /Not looked at yet[\s\S]*N2/.test(j.result_text || ''), j.result_text);
  check('  N1 archived, N2 untouched', st(N1).status === 'archived' && st(N2).status === 'active');
  check('  and it can be retried from the card', agentJobs.feed().find(f => f.id === killedId)?.retryable === true);
  channel.archive(ask2b, { by: 'user' });
  channel.archive(N2, { by: 'user' });

  // =========================================================================
  console.log('\n── 6. The budget ask, both ways ──');
  over.agentJobs = { maxToolCallsPerJob: 4, askBeforeCeiling: true, askAtPercent: 50, extensionPercent: 50 };
  const ask3 = makeConversation('Third tidy', 'user', [['user', 'once more']], { minutesAgo: 1 });
  const L1 = makeConversation('L1', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 50 });
  const L2 = makeConversation('L2', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 40 });
  const L3 = makeConversation('L3', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 30 });
  engine.requests.length = 0;
  engine.script = [judge(true, 'L1 done', ['f1']), judge(true, 'L2 done', ['f2']), judge(true, 'L3 done', [])];
  r = review.enqueueReview({ conversationId: ask3 });
  j = await settle(r.id, 20000, ['paused']);
  check('near its budget with conversations left, the review PAUSED', j.status === 'paused', `${j.status}`);
  const a = JSON.parse(j.ask_json || '{}');
  check('  the ask says what it has done and what is left', /1 closed/.test(a.text) && /2 still to look at/.test(a.text), a.text);
  check('  and asks for what the rest needs (2 left × 4 calls)', a.wants === 8 && a.grant.calls === 8, JSON.stringify({ wants: a.wants, grant: a.grant }));
  check('  the ask landed in the conversation she asked in, unread', channel.unreadFor(ask3) === 1);
  check('  L1 is closed, L2 and L3 untouched', st(L1).status === 'archived' && st(L2).status === 'active' && st(L3).status === 'active');
  const yes = await budgetAsk.decideFromMessage({ conversationId: ask3, message: 'yes', callLLM: async () => ({ content: 'YES' }) });
  check('YES resumes it', yes.decision === 'yes' && yes.ok);
  j = await settle(r.id, 20000);
  check('  and it finishes ok with all three closed', j.status === 'ok' && [L1, L2, L3].every(id => st(id).status === 'archived'), `${j.status} ${j.error}`);
  check('  one attempt still', j.attempts === 1);
  check('  the report says three closed', (db.prepare("SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY timestamp DESC, rowid DESC LIMIT 1").get(ask3).content.match(/\[\[conversation:/g) || []).length === 3);

  channel.markRead(ask3);
  channel.archive(ask3, { by: 'user' });
  const ask4 = makeConversation('Fourth tidy', 'user', [['user', 'and again']], { minutesAgo: 1 });
  const M1 = makeConversation('M1', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 50 });
  const M2 = makeConversation('M2', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 40 });
  const M3 = makeConversation('M3', 'snh', [['assistant', 'a'], ['user', 'b']], { minutesAgo: 30 });
  engine.script = [judge(true, 'M1 done', ['f1']), judge(true, 'M2 done', ['f2'])];
  r = review.enqueueReview({ conversationId: ask4 });
  j = await settle(r.id, 20000, ['paused']);
  check('paused again', j.status === 'paused');
  const no = await budgetAsk.decideFromMessage({ conversationId: ask4, message: 'no, that is enough', callLLM: async () => ({ content: 'NO' }) });
  check('NO stops it', no.decision === 'no' && no.ok);
  j = await settle(r.id, 20000);
  check('  partial, by her decision', j.status === 'partial' && j.stop_source === 'user' && j.stop_kind === 'budget-declined', `${j.status} ${j.stop_source}/${j.stop_kind}`);
  check('  M1 closed, M2 and M3 untouched', st(M1).status === 'archived' && st(M2).status === 'active' && st(M3).status === 'active');
  const rep4 = db.prepare("SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY timestamp DESC, rowid DESC LIMIT 1").get(ask4);
  check('  the report says it stopped where she said, and lists what was not looked at', /stopped the review where you said/.test(rep4.content) && /Not looked at yet[\s\S]*M2[\s\S]*M3/.test(rep4.content), rep4.content);
  over.agentJobs = {};

  // =========================================================================
  console.log('\n── 7. Retry from the card carries on with what is left ──');
  engine.script = [judge(true, 'M2 done', []), judge(true, 'M3 done', [])];
  const rr = agentJobs.retry(r.id);
  check('a partial review can be retried', rr.ok, JSON.stringify(rr));
  j = await settle(rr.id, 20000);
  check('  the retry reviews only what is still open and closes it', j.status === 'ok' && st(M2).status === 'archived' && st(M3).status === 'archived', `${j.status} ${j.error}`);

  console.log('\n── 8. The doctrine still holds ──');
  const src = fs.readFileSync(path.join(ROOT, 'mcp/mcp-client.js'), 'utf8');
  const bg = src.slice(src.indexOf('const BACKGROUND_TOOLS'), src.indexOf('];', src.indexOf('const BACKGROUND_TOOLS')));
  check('no write tool was added to BACKGROUND_TOOLS for this', !/write_memory|conversation_archive|conversation_request_retire|conversation_send/.test(bg));
  check('the review runner never hands the model a tool', !/toolSession/.test(fs.readFileSync(path.join(ROOT, 'db/conversation-review.js'), 'utf8')));
  check('db/agent-jobs.js still requires neither initiatives nor the channel', !/require\(['"]\.\/(initiatives|conversation-channel)['"]\)/.test(fs.readFileSync(path.join(ROOT, 'db/agent-jobs.js'), 'utf8')));

  engine.server.close();
  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
}

process.on('exit', () => {
  if (ROLE !== 'main') return;
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});
