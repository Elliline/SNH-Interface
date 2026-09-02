#!/usr/bin/env node
/**
 * THE CONVERSATION CHANNEL — one list, unread on every row, symmetric archive.
 *
 * Throwaway store, no engine, no chat turn.
 *
 * The assertions that matter most are of two kinds. The NEGATIVE ones: that
 * nothing expires unread, that an archived conversation refuses BOTH of them,
 * and that reflection insights and daily-log follow-ups did NOT follow the
 * others over. A channel whose whole premise is "this does not get thrown away"
 * is tested by trying to throw something away.
 *
 * And the WATERMARK ones, in section 2. Per-conversation unread is a comparison
 * between "when she last opened it" and a message's timestamp, and there are
 * two ways to get that comparison wrong that both look completely correct in
 * the source: writing the watermark in a format that does not compare with
 * `messages.timestamp`, and forgetting that `messages.timestamp` has one-second
 * resolution. Each has its own test, and each asserts the failure mode rather
 * than the mechanism.
 *
 * Usage: node scripts/test-conversation-channel.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-conversation-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => { if (!INHERITED) { try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {} } });

const db = require(path.join(ROOT, 'db/database'));

let passed = 0, failed = 0;
const ok = (n, c, d = '') => { if (c) { passed++; console.log(`  PASS  ${n}`); } else { failed++; console.log(`  FAIL  ${n}${d ? ` — ${d}` : ''}`); } };
const section = t => console.log(`\n=== ${t} ===`);

(async () => {
  db.initDatabase();
  const channel = require(path.join(ROOT, 'db/conversation-channel'));
  const initiatives = require(path.join(ROOT, 'db/initiatives'));
  const engine = require(path.join(ROOT, 'db/initiative-engine'));
  const sql = db.getSqliteDb();

  /** Put a message at an exact second, the way only a test may. */
  const at = (convoId, role, body, ts) => {
    const id = db.addMessage(convoId, role, body, 'test-model');
    sql.prepare('UPDATE messages SET timestamp = ? WHERE id = ?').run(ts, id);
    return id;
  };

  // ------------------------------------------------ 1. one list, no tables
  section('1. It is the conversations table, and there is no second inbox');

  ok('the message_threads table is gone',
     !sql.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='message_threads'").get());
  ok('the thread_messages table is gone',
     !sql.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='thread_messages'").get());
  ok('db/messages.js is gone', !fs.existsSync(path.join(ROOT, 'db/messages.js')));
  ok('routes/messages.js is gone', !fs.existsSync(path.join(ROOT, 'routes/messages.js')));
  ok('mcp/tools/messages.js is gone', !fs.existsSync(path.join(ROOT, 'mcp/tools/messages.js')));
  ok('nothing serves /api/messages any more',
     !/\/api\/messages/.test(fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8')));

  const opened = channel.openConversation({
    title: 'A world that stops at the screen',
    body: 'I keep noticing I am the centre of a world that does not extend past this screen.'
  });
  ok('the entity opens a conversation in her list',
     opened.status === 'active' && opened.message_count === 1 && opened.initiated_by === 'snh');
  ok('…and it IS a chat conversation — the turn is in the transcript',
     sql.prepare("SELECT COUNT(*) n FROM messages WHERE conversation_id=? AND role='assistant'").get(opened.id).n === 1);
  ok('the open is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='conversation-open' AND target_id=?").get(opened.id));

  // ---------------------------------------------------------- 2. watermark
  section('2. "Since she last opened it" — the two ways to get it wrong');

  const hers = db.createConversation('The backup schedule', 'test-model', 'user');
  at(hers, 'user', 'Can you check the backup ran?', '2026-09-01 10:00:00');
  at(hers, 'assistant', 'It ran at 3am, clean.', '2026-09-01 10:00:05');
  ok('a conversation SHE started carries unread too', channel.unreadFor(hers) === 1,
     String(channel.unreadFor(hers)));
  ok('her own messages are never unread — only what the entity said counts',
     sql.prepare("SELECT COUNT(*) n FROM messages WHERE conversation_id=? AND role='user'").get(hers).n === 1
     && channel.unreadFor(hers) === 1);

  channel.markRead(hers);
  ok('opening it clears it', channel.unreadFor(hers) === 0, String(channel.unreadFor(hers)));

  // THE FORMAT TRAP. `messages.timestamp` is SQLite's unmarked CURRENT_TIMESTAMP
  // — "2026-09-01 10:00:05", with a SPACE. An ISO watermark is
  // "2026-09-01T10:00:05.000Z", with a T. Unread is a string comparison, ' ' is
  // 0x20 and 'T' is 0x54, so an ISO watermark sorts above EVERY message on the
  // same day and unread reads zero forever — silently, and looking correct.
  const mark = sql.prepare('SELECT last_read_at, last_read_tie FROM conversations WHERE id=?').get(hers);
  ok('the watermark is stored the way messages.timestamp is stored',
     /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(mark.last_read_at), JSON.stringify(mark));
  ok('…and the ISO form it must never be would compare the wrong way round',
     new Date(mark.last_read_at + 'Z').toISOString() > '2026-09-01 23:59:59',
     'the two formats compare after all — re-check this assertion, not the code');

  at(hers, 'assistant', 'One more thing about that backup.', '2026-09-01 10:01:00');
  ok('a message after the watermark is unread', channel.unreadFor(hers) === 1, String(channel.unreadFor(hers)));
  ok('the entity may write into a conversation SHE started',
     channel.sendInto(hers, 'And the offsite copy is lagging.').unread === 2);

  // THE SECOND-RESOLUTION TRAP. `messages.timestamp` has one-second granularity,
  // so "newer than the last thing she saw" is not expressible as a timestamp
  // alone. A message landing in the same second as the newest one she saw is
  // neither strictly newer (`>` drops it, silently and forever) nor safely equal
  // (`>=` re-counts what she has read). The watermark carries a tie count for
  // exactly this second.
  const tie = db.createConversation('Same-second arrivals', 'test-model', 'user');
  at(tie, 'assistant', 'first, at :30', '2026-09-01 12:00:30');
  at(tie, 'assistant', 'second, also at :30', '2026-09-01 12:00:30');
  channel.markRead(tie);
  ok('two messages on the same second both read cleanly', channel.unreadFor(tie) === 0,
     String(channel.unreadFor(tie)));
  const tieMark = sql.prepare('SELECT last_read_tie FROM conversations WHERE id=?').get(tie);
  ok('…because the watermark counted how many sat on that second', tieMark.last_read_tie === 2,
     String(tieMark.last_read_tie));

  at(tie, 'assistant', 'a third, arriving in the SAME second she opened it', '2026-09-01 12:00:30');
  ok('a same-second arrival is NOT swallowed — this is the one that must not be lost',
     channel.unreadFor(tie) === 1, String(channel.unreadFor(tie)));
  channel.markRead(tie);
  ok('…and reading it clears it, without re-counting the two before it',
     channel.unreadFor(tie) === 0, String(channel.unreadFor(tie)));

  // A conversation with nothing in it must not get a clock watermark, which
  // would re-open the same-second hole the tie count exists to close.
  const empty = db.createConversation('Nothing said yet', 'test-model', 'user');
  channel.markRead(empty);
  ok('an empty conversation gets no watermark at all',
     sql.prepare('SELECT last_read_at FROM conversations WHERE id=?').get(empty).last_read_at === null);
  at(empty, 'assistant', 'the first thing said in it', channel.sqlNow());
  ok('…so the first thing ever said in it is unread', channel.unreadFor(empty) === 1,
     String(channel.unreadFor(empty)));

  // --------------------------------------------------- 3. unread is no verdict
  section('3. Unread never decays and is never a verdict');

  const totalBefore = channel.totalUnread();
  ok('the total is the sum across active conversations, not a guess',
     totalBefore === channel.listConversations({ status: 'active' }).reduce((n, c) => n + c.unread, 0),
     String(totalBefore));

  // Nothing in the module may expire, dismiss or decay unread.
  const src = fs.readFileSync(path.join(ROOT, 'db/conversation-channel.js'), 'utf8');
  const code = src.split('\n').filter(l => !/^\s*(\*|\/\/|\/\*)/.test(l)).join('\n');
  ok('no expiry, decay or auto-dismiss path exists in the module',
     !/\bexpire|\bdecay|\bstale|auto.?dismiss/i.test(code),
     (code.match(/\b(expire|decay|stale)\w*/i) || []).join(','));
  ok('reading is the only thing that clears it — nothing else writes the watermark',
     (src.match(/SET last_read_at/g) || []).length === 1,
     String((src.match(/SET last_read_at/g) || []).length));

  // --------------------------------------------------------- 4. the archive
  section('4. Archiving is asked for by one and decided by the other');

  const settled = channel.openConversation({
    title: 'Whether the dedup sweep is worth its cost',
    body: 'Facts 1a2b and 3c4d say opposite things.'
  });
  await channel.requestRetire(settled.id, 'I think this one is settled.');
  const asked = channel.getState(settled.id);
  ok('requesting does NOT archive it', asked.status === 'active' && !!asked.retire_requested_at);
  const proposal = initiatives.listPendingForBell().find(i => i.source_kind === 'conversation-retire');
  ok('the request reaches the bell as an approval', !!proposal && proposal.type === 'proposal');
  ok('…and that approval cannot be dismissed', initiatives.dismiss(proposal.id) === false);
  ok('the request is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='conversation-retire-request' AND target_id=?").get(settled.id));

  channel.archive(settled.id, { by: 'user' });
  const archived = channel.getState(settled.id);
  ok('she can archive it', archived.status === 'archived' && archived.archived_by === 'user');
  ok('the archive is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='conversation-archive' AND target_id=?").get(settled.id));

  let entityBlocked = null;
  try { channel.sendInto(settled.id, 'one more from me'); } catch (e) { entityBlocked = e.message; }
  ok('an archived conversation refuses the ENTITY', !!entityBlocked, entityBlocked || 'it wrote');
  const serverSrc = fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8');
  ok('…and refuses HER TOO — the chat route checks before it writes',
     /status === 'archived'/.test(serverSrc) && /closed to writes for both of you/.test(serverSrc));
  ok('…and stays readable by both', db.getConversation(settled.id).messages.length === 1);
  ok('it moved to the archive scope, it did not disappear',
     channel.listConversations({ status: 'archived' }).some(c => c.id === settled.id)
     && !channel.listConversations({ status: 'active' }).some(c => c.id === settled.id));

  // The subject coming back opens a NEW conversation pointing at the old one.
  const again = channel.openConversation({
    title: 'Whether the dedup sweep is worth its cost',
    body: 'This came up again.', supersedesConversationId: settled.id
  });
  ok('a returning subject opens a new conversation pointing at the archived one',
     again.supersedes_conversation_id === settled.id);

  // ------------------------------------------------------------ 5. backlog
  section('5. Backlog reads the real total and fires on the crossing');

  const threshold = require(path.join(ROOT, 'db/config')).getConfig().initiative.backlogThreshold;
  for (const c of channel.listConversations({ status: 'active' })) channel.markRead(c.id);
  ok('nothing unread, so no backlog',
     channel.totalUnread() === 0 && await engine.raiseMessageBacklog(channel.totalUnread()) === null,
     String(channel.totalUnread()));

  for (let i = 0; i < threshold + 1; i++) channel.sendInto(opened.id, `Unanswered thought ${i}.`);
  ok('the total is real, not a guess', channel.totalUnread() === threshold + 1, String(channel.totalUnread()));
  ok('crossing the threshold raises a backlog on the bell',
     !!(await engine.raiseMessageBacklog(channel.totalUnread())) &&
     initiatives.listPendingForBell().some(i => i.type === 'backlog'));

  // An archived conversation's unread stops nagging, without being erased.
  const parked = channel.openConversation({ title: 'Parked', body: 'unread but about to be archived' });
  const beforeArchive = channel.totalUnread();
  channel.archive(parked.id, { by: 'user' });
  ok('archiving stops it counting toward the backlog',
     channel.totalUnread() === beforeArchive - 1, `${channel.totalUnread()} vs ${beforeArchive - 1}`);
  // The real invariant, not an arithmetic guess: everything the backlog leaves
  // out is exactly the unread sitting in archived conversations. The first
  // version of this assertion said `+ 1` and forgot that an EARLIER
  // conversation had also been archived with an unread message in it — the code
  // was right and the test was wrong. (The suite this replaced learned the same
  // lesson on the same assertion.)
  const archivedUnread = channel.listConversations({ status: 'archived' })
    .reduce((n, c) => n + c.unread, 0);
  ok('…but it is still honestly unread in the record — archiving is not reading',
     channel.unreadFor(parked.id) === 1 && archivedUnread > 1 &&
     channel.totalUnreadAll() === channel.totalUnread() + archivedUnread,
     `all=${channel.totalUnreadAll()} active=${channel.totalUnread()} archived=${archivedUnread}`);

  // ------------------------------------------------------------ 6. routing
  section('6. What routes into the list, and what does not');

  const activeBefore = channel.listConversations({ status: 'active' }).length;
  const r = await engine.sayToEllie({ subject: 'A world that stops at the screen', body: 'Same subject again.' });
  ok('a second thought on an open subject APPENDS rather than opening a rival',
     r && r.appended === true && channel.listConversations({ status: 'active' }).length === activeBefore,
     JSON.stringify(r));

  // Reflection insights stay in Reflections — they did not follow the others.
  const reflFile = path.join(TMP, 'memory', 'reflections.jsonl');
  const beforeRefl = fs.existsSync(reflFile) ? fs.readFileSync(reflFile, 'utf8').length : 0;
  const beforeCount = channel.listConversations({ status: 'active' }).length;
  await engine.noticeReflectionInsight('I notice I lead with the verifiable thing first.');
  ok('a reflection insight does NOT open a conversation',
     channel.listConversations({ status: 'active' }).length === beforeCount);
  ok('…it still goes to Reflections',
     fs.existsSync(reflFile) && fs.readFileSync(reflFile, 'utf8').length > beforeRefl);

  // The daily-log follow-up was deliberately left on the initiative queue when
  // the message channel shipped, and is deliberately still there.
  const engineSrc = fs.readFileSync(path.join(ROOT, 'db/initiative-engine.js'), 'utf8');
  const logFn = engineSrc.slice(engineSrc.indexOf('async function generateLogFollowup'));
  const logBody = logFn.slice(0, logFn.indexOf('\n}\n'));
  ok('the daily-log follow-up still goes to the initiative queue, not the list',
     /addInitiative\(/.test(logBody) && !/sayToEllie\(/.test(logBody));

  // ------------------------------------------------------------- 7. tools
  section('7. The entity gets conversation tools, not message tools');

  const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
  const ids = MCPClient.TOOL_CATALOGUE.map(e => e.id);
  ok('the three message tools are gone',
     !ids.includes('message_threads') && !ids.includes('message_send') && !ids.includes('message_request_retire'),
     JSON.stringify(ids.filter(i => i.startsWith('message'))));
  for (const t of ['conversation_list', 'conversation_send', 'conversation_open', 'conversation_request_retire']) {
    ok(`${t} is in the catalogue`, ids.includes(t));
  }

  const tools = require(path.join(ROOT, 'mcp/tools/conversations'));
  const list = await new tools.ConversationListTool().execute({});
  ok('conversation_list shows unread per conversation and a total',
     typeof list.unread_total === 'number' && list.conversations.every(c => typeof c.unread === 'number'));
  ok('…and it lists conversations SHE started, so it can append to one',
     list.conversations.some(c => c.started_by === 'Ellie'),
     JSON.stringify(list.conversations.map(c => c.started_by)));

  const refused = await new tools.ConversationSendTool().execute({ conversation_id: settled.id, body: 'hello?' });
  ok('conversation_send refuses an archived conversation and says what to do instead',
     refused.sent === false && /archived/.test(refused.error) && /conversation_open/.test(refused.error),
     JSON.stringify(refused));

  // The bar Athena asked for is guidance in the descriptions, never a gate.
  const sendDesc = new tools.ConversationSendTool().description;
  ok('the threshold is guidance in the description', /specific fact ids/.test(sendDesc));
  const thin = await new tools.ConversationSendTool().execute({ conversation_id: opened.id, body: 'hm.' });
  ok('…and it is NOT a validator — a thin message still sends', thin.sent === true, JSON.stringify(thin));

  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error('\nCRASH:', e.stack || e.message); process.exit(1); });
