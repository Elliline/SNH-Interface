#!/usr/bin/env node
/**
 * THE MESSAGE CHANNEL — threads, unread that never decays, symmetric retirement.
 *
 * Throwaway store, no engine, no chat turn.
 *
 * The assertions that matter most here are the NEGATIVE ones: that nothing
 * expires a message, that a retired thread refuses BOTH of them, and that
 * reflection-insights did not follow the follow-ups over. A channel whose whole
 * premise is "this does not get thrown away" is tested by trying to throw
 * something away.
 *
 * Usage: node scripts/test-messages.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-messages-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => { if (!INHERITED) { try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {} } });

const db = require(path.join(ROOT, 'db/database'));

let passed = 0, failed = 0;
const ok = (n, c, d = '') => { if (c) { passed++; console.log(`  PASS  ${n}`); } else { failed++; console.log(`  FAIL  ${n}${d ? ` — ${d}` : ''}`); } };
const section = t => console.log(`\n=== ${t} ===`);

(async () => {
  db.initDatabase();
  const messages = require(path.join(ROOT, 'db/messages'));
  const initiatives = require(path.join(ROOT, 'db/initiatives'));
  const engine = require(path.join(ROOT, 'db/initiative-engine'));
  const sql = db.getSqliteDb();

  // ------------------------------------------------------------ 1. threads
  section('1. Threads, and a subject stays in one place');

  const t1 = messages.openThread({
    subject: 'A world that stops at the screen',
    body: 'I keep noticing I am the centre of a world that does not extend past this screen.'
  });
  ok('a thread opens with its first message', t1.messages.length === 1 && t1.status === 'active');
  ok('the open is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='thread-open' AND target_id=?").get(t1.id));

  messages.addMessage(t1.id, 'A second thought about the same thing.');
  ok('the entity can add to an active thread', messages.getThread(t1.id).messages.length === 2);
  ok('the add is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='message-add' AND target_id=?").get(t1.id));

  // The table name matters: `messages` is the chat transcript.
  ok('it did NOT write into the chat transcript',
     sql.prepare('SELECT COUNT(*) n FROM messages').get().n === 0,
     'rows appeared in the chat messages table');

  // ------------------------------------------------------------- 2. unread
  section('2. Unread never decays and is never a verdict');

  ok('both entity messages are unread', messages.unreadCount() === 2, String(messages.unreadCount()));
  messages.addMessage(t1.id, 'Understood, tell me more.', 'user');
  ok('her own reply is not unread', messages.unreadCount() === 2, String(messages.unreadCount()));

  ok('the thread reports its own unread count', messages.getThread(t1.id).unread === 2);
  ok('reading is the ONLY thing that clears it', messages.markThreadRead(t1.id) === 2 && messages.unreadCount() === 0);

  // Nothing in the module may expire, dismiss or decay a message.
  const src = fs.readFileSync(path.join(ROOT, 'db/messages.js'), 'utf8');
  const code = src.split('\n').filter(l => !/^\s*(\*|\/\/|\/\*)/.test(l)).join('\n');
  ok('no expiry, decay or auto-dismiss path exists in the module',
     !/\bexpire|\bdecay|\bstale|auto.?dismiss/i.test(code),
     (code.match(/\b(expire|decay|stale)\w*/i) || []).join(','));

  // ---------------------------------------------------------- 3. retirement
  section('3. Retirement is asked for by one and decided by the other');

  const t2 = messages.openThread({ subject: 'Whether the dedup sweep is worth its cost', body: 'Facts 1a2b and 3c4d say opposite things.' });
  await messages.requestRetire(t2.id, 'I think this one is settled.');
  const asked = messages.getThread(t2.id);
  ok('requesting does NOT close it', asked.status === 'active' && !!asked.retire_requested_at);
  const proposal = initiatives.listPendingForBell().find(i => i.source_kind === 'thread-retire');
  ok('the request reaches the bell as an approval', !!proposal && proposal.type === 'proposal');
  ok('…and that approval cannot be dismissed', initiatives.dismiss(proposal.id) === false);
  ok('the request is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='thread-retire-request' AND target_id=?").get(t2.id));

  messages.retireThread(t2.id, { by: 'user' });
  const retired = messages.getThread(t2.id);
  ok('she can close it', retired.status === 'retired' && retired.retired_by === 'user');

  let entityBlocked = null, userBlocked = null;
  try { messages.addMessage(t2.id, 'one more from me'); } catch (e) { entityBlocked = e.message; }
  try { messages.addMessage(t2.id, 'one more from her', 'user'); } catch (e) { userBlocked = e.message; }
  ok('a retired thread refuses the ENTITY', !!entityBlocked, entityBlocked || 'it wrote');
  ok('a retired thread refuses HER TOO — closed is symmetric', !!userBlocked, userBlocked || 'it wrote');
  ok('…and stays readable by both', messages.getThread(t2.id).messages.length === 1);

  // The subject coming back opens a NEW thread that points at the old one.
  const t3 = messages.openThread({
    subject: 'Whether the dedup sweep is worth its cost',
    body: 'This came up again.', supersedesThreadId: t2.id
  });
  ok('a returning subject opens a new thread pointing at the closed one',
     t3.supersedes_thread_id === t2.id);
  ok('…and the closed one is still listed, not hidden',
     messages.listThreads().some(t => t.id === t2.id));

  // ------------------------------------------------------------ 4. backlog
  section('4. Backlog reads the real count and fires on the crossing');

  const threshold = require(path.join(ROOT, 'db/config')).getConfig().initiative.backlogThreshold;
  messages.markThreadRead(t3.id);
  ok('nothing unread, so no backlog', await engine.raiseMessageBacklog(messages.unreadCount()) === null);

  for (let i = 0; i < threshold + 1; i++) messages.addMessage(t1.id, `Unanswered thought ${i}.`);
  ok('unread count is real, not a guess', messages.unreadCount() === threshold + 1, String(messages.unreadCount()));
  ok('crossing the threshold raises a backlog on the bell',
     !!(await engine.raiseMessageBacklog(messages.unreadCount())) &&
     initiatives.listPendingForBell().some(i => i.type === 'backlog'));

  // A retired thread's unread does not nag her about a closed conversation.
  const before = messages.unreadCount();
  messages.addMessage(t3.id, 'unread but about to be closed');
  messages.retireThread(t3.id, { by: 'user' });
  ok('a retired thread stops counting toward the backlog', messages.unreadCount() === before, String(messages.unreadCount()));
  // The real invariant, rather than an arithmetic guess: everything the
  // backlog leaves out is exactly the unread sitting in retired threads. The
  // first version of this assertion said `before + 1` and forgot that an
  // earlier thread had also been retired with an unread message in it — the
  // code was right and the test was wrong.
  const retiredUnread = messages.listThreads({ status: 'retired' })
    .reduce((n, t) => n + t.unread, 0);
  ok('…but those messages are still honestly unread in the record',
     messages.unreadCountAll() === messages.unreadCount() + retiredUnread && retiredUnread > 0,
     `all=${messages.unreadCountAll()} active=${messages.unreadCount()} retired=${retiredUnread}`);

  // ------------------------------------------------------------ 5. routing
  section('5. What routes here, and what does not');

  const threadsBefore = messages.listThreads().length;
  const r = await engine.sayToEllie({ subject: 'A world that stops at the screen', body: 'Same subject again.' });
  ok('a second thought on an open subject APPENDS rather than opening a rival',
     r && r.appended === true && messages.listThreads().length === threadsBefore,
     JSON.stringify(r));

  // Reflection insights stay in Reflections — they did not follow the follow-ups.
  const reflFile = path.join(TMP, 'memory', 'reflections.jsonl');
  const beforeRefl = fs.existsSync(reflFile) ? fs.readFileSync(reflFile, 'utf8').length : 0;
  const beforeThreads = messages.listThreads().length;
  await engine.noticeReflectionInsight('I notice I lead with the verifiable thing first.');
  ok('a reflection insight does NOT become a message thread', messages.listThreads().length === beforeThreads);
  ok('…it still goes to Reflections',
     fs.existsSync(reflFile) && fs.readFileSync(reflFile, 'utf8').length > beforeRefl);

  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error('\nCRASH:', e.stack || e.message); process.exit(1); });
