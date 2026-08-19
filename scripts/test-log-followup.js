#!/usr/bin/env node
/**
 * Tests for the daily-log follow-up reader.
 *
 * Runs entirely against a throwaway SNH_DATA_DIR and a throwaway daily-log
 * directory, with the model call mocked. Nothing here touches the live corpus,
 * the live log, or the live initiative queue — the whole point of this source is
 * that it queues things a person will be shown, and a test that queued into the
 * real pool would be putting words in SNH's mouth. (2026-08-16: test chat turns
 * silently consumed a real notice and two priority-7 initiatives. Not again.)
 *
 * Run: node scripts/test-log-followup.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

// A disposable store, before anything requires the database.
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-logfollowup-'));
process.env.SNH_DATA_DIR = TMP;

const DAILY = path.join(TMP, 'daily');
fs.mkdirSync(DAILY, { recursive: true });

let failures = 0;
function check(cond, label) {
  if (cond) console.log(`  ✅ ${label}`);
  else { console.log(`  ❌ ${label}`); failures++; }
}

function stamp(daysAgo) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// ---- Fixture: a realistic day, events mixed with the bookkeeping ------------
const PROV = '[conversation 3a1a0c38, message a03eefc8, typed]';
function writeDay(daysAgo, lines) {
  fs.writeFileSync(path.join(DAILY, `${stamp(daysAgo)}.md`),
    `# Daily Log - ${stamp(daysAgo)}\n\n` + lines.map(l => `### 09:15\n- ${l}\n`).join('\n'), 'utf8');
}

writeDay(0, [
  `User let the dogs out ${PROV}`,
  `user found Aurelius' philosophical discussions can get too deep for them sometimes ${PROV}`,
  `Superseded fact: "User's systems run AMD" → replaced by "User has 2 RTX3090 cards" (correction). ${PROV}`,
  `Already knew this, so I did not write it down twice — "x" restates "y" ${PROV}`,
  `Did not record "User's name is Mike" — the identity anchor refused it. ${PROV}`,
  'Reflection: reviewed 10 message(s) across 1 conversation(s) → 0 self-fact(s) stored, 0 superseded.',
  'Scored fact salience 8/10: "something" — because reasons',
  'Follow-up: considered 3 candidate(s) → none (nothing cleared the bar)',
]);
writeDay(1, [`User said they would call their mom this weekend ${PROV}`]);
// Outside a 3-day window; must never be read.
writeDay(20, [`User mentioned something from three weeks ago ${PROV}`]);

const reader = require('../db/daily-log-reader');

console.log('\n── Parser: events only, bookkeeping excluded ──');
const parsed = reader.readRecentEvents({ days: 3, dailyDir: DAILY });
const texts = parsed.events.map(e => e.text);
check(texts.some(t => /let the dogs out/.test(t)), 'plain event kept');
check(texts.some(t => /philosophical discussions can get too deep/.test(t)), 'the event worth following up is kept');
check(texts.some(t => /call their mom/.test(t)), 'yesterday is inside the window');
check(!texts.some(t => /^Superseded fact:/.test(t)), 'supersession bookkeeping excluded');
check(!texts.some(t => /^Already knew this/.test(t)), 'repeat-fold bookkeeping excluded');
check(!texts.some(t => /^Did not record/.test(t)), 'intake refusal excluded');
check(!texts.some(t => /^Reflection:/.test(t)), 'reflection (no provenance) excluded');
check(!texts.some(t => /^Scored fact salience/.test(t)), 'salience scoring excluded');
check(!texts.some(t => /^Follow-up:/.test(t)), "the source's own log line excluded — no feedback loop");
check(!texts.some(t => /three weeks ago/.test(t)), 'entry outside the window is not read');
check(parsed.events.length === 3, `exactly the 3 real events (got ${parsed.events.length})`);
check(parsed.events.every(e => e.conversationId === '3a1a0c38'), 'provenance parsed off each entry');

console.log('\n── Parser: stable ids ──');
const again = reader.readRecentEvents({ days: 3, dailyDir: DAILY });
check(JSON.stringify(again.events.map(e => e.id)) === JSON.stringify(parsed.events.map(e => e.id)),
  'ids are stable across reads (dedup depends on this)');
check(new Set(parsed.events.map(e => e.id)).size === parsed.events.length, 'ids are unique per entry');

// ---- Now the judgement half, with the model mocked -------------------------
const db = require('../db/database');
db.initDatabase();

const memoryManager = require('../db/memory-manager');
let mockReply = null;
let llmCalls = 0;
memoryManager.callLLM = async () => { llmCalls++; return { content: mockReply }; };

const agentPool = require('../db/agent-pool');
agentPool.schedule = async (fn) => fn();

const initiatives = require('../db/initiatives');
const engine = require('../db/initiative-engine');

(async () => {
  console.log('\n── It can decline, and declining is RECORDED ──');
  mockReply = JSON.stringify({
    candidates: ['dogs out', 'call their mom'],
    sourceEntry: null,
    followup: null,
    reasoning: 'nothing here is worth interrupting them for'
  });
  let trace = await engine.generateLogFollowup({ days: 3, dailyDir: DAILY });
  check(trace.skipped === true, 'declined');
  check(trace.initiativeId === null, 'nothing queued');
  let traces = initiatives.listLogFollowupTraces({ limit: 5 });
  check(traces.length === 1, 'a trace row was written for the decline');
  check(traces[0].skipped === true, 'trace says skipped');
  check(/worth interrupting/.test(traces[0].reasoning), 'trace carries the model\'s reason');
  check(traces[0].entriesConsidered === 3, 'trace records how many entries it read');
  check(traces[0].windowDays === 3, 'trace records the window used');
  check(traces[0].filesRead.length >= 2, 'trace records which files it read');
  check(initiatives.listPending().length === 0, 'pending queue untouched by a decline');

  console.log('\n── It can raise, through the normal initiative queue ──');
  const target = parsed.events.findIndex(e => /philosophical/.test(e.text)) + 1;
  mockReply = JSON.stringify({
    candidates: ['the philosophical depth comment'],
    sourceEntry: target,
    followup: "I've been sitting with something you said — that I can get too deep sometimes. I'd rather meet you where you are; tell me when I'm doing it.",
    reasoning: 'feedback about how we talk deserves a response'
  });
  trace = await engine.generateLogFollowup({ days: 3, dailyDir: DAILY });
  check(trace.skipped === false, 'raised');
  check(!!trace.initiativeId, 'an initiative id came back');
  check(!!trace.sourceEntryId, 'the source entry was identified');
  const pending = initiatives.listPending();
  check(pending.length === 1, 'exactly one initiative queued');
  check(pending[0] && pending[0].type === 'followup', "queued as a normal 'followup', not a parallel path");
  check(pending[0] && pending[0].source_kind === 'daily-log', 'source_kind = daily-log');
  check(pending[0] && pending[0].source_ref === trace.sourceEntryId, 'source_ref is the log entry id');
  const usedSourceRef = trace.sourceEntryId;

  console.log('\n── It will not stack a second one ──');
  llmCalls = 0;
  trace = await engine.generateLogFollowup({ days: 3, dailyDir: DAILY });
  check(trace.skipped === true, 'declined while one is already pending');
  check(llmCalls === 0, 'did not even ask the model — refused before spending a call');
  check(/already waiting/.test(trace.reasoning), 'the reason names the pending item');
  check(initiatives.listPending().length === 1, 'still exactly one queued');
  traces = initiatives.listLogFollowupTraces({ limit: 10 });
  check(traces.length === 3, 'every pass wrote a trace, including the refusals');

  console.log('\n── An entry already followed up on is never re-raised ──');
  // Clear the pending guard, leaving the delivered//used source behind.
  const sqlite = db.getSqliteDb();
  sqlite.prepare("UPDATE initiatives SET status = 'delivered' WHERE status = 'pending'").run();
  mockReply = JSON.stringify({
    candidates: ['the philosophical depth comment'],
    sourceEntry: 1,
    followup: 'Trying to raise the same thing again.',
    reasoning: 'should never get the chance'
  });
  trace = await engine.generateLogFollowup({ days: 3, dailyDir: DAILY });
  check(trace.entries.every(e => e.id !== usedSourceRef),
    'the already-used entry was withheld from the model');
  check(trace.entries.length === 2, `only the 2 unused entries were offered (got ${trace.entries.length})`);

  console.log('\n── But an entry whose follow-up expired UNREAD comes back ──');
  // The pool cap and the stale sweep both expire a pending initiative without
  // anyone having seen it. Withholding on source alone burned the entry forever
  // on behalf of a question that was never asked, and decayed the window pass by
  // pass. Expired means unread means still actionable.
  sqlite.prepare("UPDATE initiatives SET status = 'expired'").run();
  mockReply = JSON.stringify({
    candidates: ['the philosophical depth comment'],
    sourceEntry: 1,
    followup: 'Asking again, because nobody ever saw the first one.',
    reasoning: 'the earlier raise expired unread'
  });
  trace = await engine.generateLogFollowup({ days: 3, dailyDir: DAILY });
  check(trace.entries.some(e => e.id === usedSourceRef),
    'the entry behind an expired-unread follow-up was offered again');
  check(trace.entries.length === 3, `all 3 entries back in view (got ${trace.entries.length})`);
  check(trace.skipped === false, 'and it could raise on it again');

  console.log('\n── An error is not a decline ──');
  memoryManager.callLLM = async () => { throw new Error('engine unreachable'); };
  sqlite.prepare("UPDATE initiatives SET status = 'dismissed'").run();
  trace = await engine.generateLogFollowup({ days: 3, dailyDir: DAILY });
  check(trace.skipped === true, 'nothing raised on error');
  check(/^error: /.test(trace.reasoning), 'the trace says it ERRORED, not that it declined');
  traces = initiatives.listLogFollowupTraces({ limit: 1 });
  check(/engine unreachable/.test(traces[0].reasoning), 'the failure reason is queryable');

  console.log('\n── Empty window ──');
  const EMPTY = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-emptylog-'));
  trace = await engine.generateLogFollowup({ days: 3, dailyDir: EMPTY });
  check(trace.skipped === true, 'no entries → nothing raised');
  check(/no event entries/.test(trace.reasoning), 'and it says so rather than being silent');
  fs.rmSync(EMPTY, { recursive: true, force: true });

  fs.rmSync(TMP, { recursive: true, force: true });
  console.log(`\n${failures === 0 ? '✅ ALL PASSED' : `❌ ${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => {
  console.error('test crashed:', e);
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {}
  process.exit(1);
});
