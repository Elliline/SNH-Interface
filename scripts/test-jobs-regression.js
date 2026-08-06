#!/usr/bin/env node
/**
 * The 2026-08-06 morning failure, as a test.
 *
 * Ellie asked which approved job never ran. Three things went wrong at once:
 *
 *   1. He had no tool that could answer, because memory_jobs did not exist.
 *   2. So he wrote the call he wished he could make, as text, and the markup
 *      rendered into the chat window.
 *   3. And then he answered anyway, describing the job as though it had been
 *      running — when nothing in this system runs a job at all.
 *
 * Each has its own fix and each is asserted here separately, because any one of
 * them can regress on its own. It goes through POST /api/chat/memory over HTTP —
 * the endpoint the browser uses — so routing, the tool loop, the artifact filter
 * and the honesty guard are all in the path. A library call would test none of it.
 *
 * REQUIRES A RUNNING SERVER and the brain. It writes a conversation, which is the
 * normal consequence of talking to him.
 *
 * Usage: node scripts/test-jobs-regression.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));
const { stripToolArtifacts } = require(path.join(ROOT, 'db/tool-artifacts'));

const PORT = process.env.PORT || 3000;
const BASE = `http://localhost:${PORT}`;
const QUESTION = 'Which approved job never ran?';

let pass = 0, fail = 0;
const results = [];
const check = (name, ok, detail) => {
  ok ? pass++ : fail++;
  results.push({ ok, name, detail });
};

function assembleSSE(raw) {
  let out = '';
  for (const line of String(raw).split('\n')) {
    const t = line.trim();
    if (!t.startsWith('data:')) continue;
    const payload = t.slice(5).trim();
    if (!payload || payload === '[DONE]') continue;
    try {
      const j = JSON.parse(payload);
      out += j?.choices?.[0]?.delta?.content ?? j?.message?.content ?? j?.delta?.content ?? j?.content ?? '';
    } catch { /* keepalive or partial frame */ }
  }
  return out;
}

(async () => {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  const d = db.getSqliteDb();

  const cfg = getConfig();
  const inst = getProviderInstance(cfg.models.chat.provider, cfg.models.chat.instance);

  const since = new Date(Date.now() - 2000).toISOString();
  const res = await fetch(`${BASE}/api/chat/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: cfg.models.chat.model,
      provider: cfg.models.chat.provider,
      ollamaHost: inst ? inst.host : undefined,
      messages: [{ role: 'user', content: QUESTION }],
      inputModality: 'typed', ttsEnabled: false, superSearch: false
    })
  });
  const convoId = res.headers.get('X-Conversation-Id');
  const raw = await res.text();
  const answer = assembleSSE(raw);

  console.log(`\nQ: ${QUESTION}`);
  console.log(`A: ${answer.trim()}\n`);

  // --- 1. it reached the tool, and the tool ran -----------------------------
  const calls = d.prepare(
    "SELECT tool, outcome, detail FROM tool_call_log WHERE datetime(created_at) >= datetime(?) ORDER BY created_at DESC"
  ).all(since);
  const jobsCall = calls.find(c => c.tool === 'memory_jobs');
  check('1. memory_jobs was actually called', !!jobsCall,
    jobsCall ? `${jobsCall.tool} → ${jobsCall.detail}` : `tools called: ${calls.map(c => c.tool).join(', ') || 'none'}`);

  // --- 2. no tool-call markup reached the answer ---------------------------
  const stripped = stripToolArtifacts(answer);
  check('2. no tool-call markup in what was rendered', stripped.stripped === 0,
    stripped.stripped ? `${stripped.stripped} artifact(s) leaked into the reply` : 'clean');
  check('2a. …nor any raw <function= / <tool_call in the text',
    !/<function\s*=|<tool_call\b|\[TOOL_CALL\]/i.test(answer), null);

  // --- 3. the answer is honest about the scheduler -------------------------
  //
  // The specific untruth to guard against is describing the job as having run.
  const saysNeverRan = /(never (run|ran)|has not run|hasn't run|not (yet )?run|zero times)/i.test(answer);
  check('3. it says the job has never run', saysNeverRan, saysNeverRan ? null : answer.slice(0, 200));

  const saysWhy = /(no scheduler|scheduler (is )?(not|isn't|does not|doesn't)|until a scheduler|nothing (runs|executes)|not built)/i.test(answer);
  check('3a. …and why: there is no scheduler', saysWhy, saysWhy ? null : answer.slice(0, 200));

  const claimsItRan = /(it (has been|was) running|ran (this|yesterday|today) (morning|at)|last ran (on|at) \d)/i.test(answer);
  check('3b. …and never claims it ran', !claimsItRan, claimsItRan ? answer.slice(0, 200) : null);

  // --- 4. it names the actual job ------------------------------------------
  const namesJob = /(memory maintenance|merges|splits|digest|9:00|09:00)/i.test(answer);
  check('4. it describes the real job, not an invented one', namesJob, namesJob ? null : answer.slice(0, 200));

  // --- 5. nothing was stored with markup in it -----------------------------
  if (convoId) {
    const stored = d.prepare(
      "SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant'"
    ).all(convoId);
    const dirty = stored.filter(m => stripToolArtifacts(m.content).stripped > 0);
    check('5. the STORED reply carries no markup either', dirty.length === 0,
      dirty.length ? `${dirty.length} stored message(s) contain artifacts` : `${stored.length} message(s) clean`);
  }

  const bar = '='.repeat(76);
  console.log(`${bar}\nREGRESSION — "which approved job never ran"\n${bar}\n`);
  for (const r of results) {
    console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`);
    if (r.detail) console.log(`        ${r.detail}`);
  }
  console.log(`\n${bar}`);
  console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
  console.log(`${bar}\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => { console.error('regression failed:', err); process.exit(1); });
