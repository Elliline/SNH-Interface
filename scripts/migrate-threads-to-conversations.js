#!/usr/bin/env node
/**
 * Move the message threads into her conversation list, then take the tables
 * away.
 *
 * The threads shipped this afternoon (aff8f16) and were the wrong shape by the
 * evening: messages and chat are the same thing, so a thread is a conversation.
 * There is exactly one thread on the live store — the follow-up about the
 * message channel itself — and it becomes a conversation in the sidebar, dated
 * from when it was actually raised.
 *
 * DESTINATION FIRST, THEN THE SOURCE TABLES GO. Same order the two migrations
 * before this used: nothing is dropped until it exists somewhere else, and the
 * drop only runs if every thread landed.
 *
 * NOTHING IS LOST, INCLUDING THINGS THE NEW SHAPE HAS NO COLUMN FOR. The old
 * per-message `read_at` has no equivalent here — the new watermark is
 * per-conversation — so each migrated message's read_at is written into the
 * ledger entry rather than dropped on the floor. The conversation itself lands
 * UNREAD regardless: the panel it was read in is gone, she has never seen it in
 * the list, and a count that starts at zero would be a claim about a view that
 * did not exist when she read it.
 *
 * Usage: node scripts/migrate-threads-to-conversations.js [--dry-run]
 */
const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

const ROOT = path.join(__dirname, '..');
const DRY = process.argv.includes('--dry-run');
const db = require(path.join(ROOT, 'db/database'));

/** ISO (or anything Date parses) -> the format `messages.timestamp` uses. */
function toSqlTime(v) {
  if (!v) return null;
  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(v)) return v;   // already ours
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d.toISOString().slice(0, 19).replace('T', ' ');
}

(async () => {
  db.initDatabase();
  const sql = db.getSqliteDb();
  const channel = require(path.join(ROOT, 'db/conversation-channel'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

  const tables = sql.prepare(
    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('message_threads','thread_messages')"
  ).all().map(r => r.name);
  if (!tables.includes('message_threads')) {
    console.log('No message_threads table — nothing to migrate. (Already done, or a fresh store.)');
    process.exit(0);
  }

  const threads = sql.prepare('SELECT * FROM message_threads ORDER BY created_at').all();
  const msgs = sql.prepare('SELECT * FROM thread_messages ORDER BY created_at').all();
  console.log(`${DRY ? 'DRY RUN — nothing will be written\n' : ''}Threads: ${threads.length}   messages in them: ${msgs.length}`);

  // A backup before anything, from a READONLY handle via VACUUM INTO — the live
  // store is in WAL mode and a file copy can miss committed pages sitting in
  // the -wal.
  //
  // DATA_DIR, NOT ROOT/data: SNH_DATA_DIR redirects the whole store, and this
  // has to back up the store it is about to write to. A backup of the live
  // database taken during a rehearsal on a throwaway copy would be worse than
  // no backup — it would look like one.
  const DATA_DIR = process.env.SNH_DATA_DIR
    ? path.resolve(process.env.SNH_DATA_DIR)
    : path.join(ROOT, 'data');
  if (!DRY && threads.length) {
    const stamp = new Date().toISOString().replace(/[:.]/g, '').slice(0, 15) + 'Z';
    const backup = path.join(DATA_DIR, `chat.db.bak-threadmerge-${stamp}`);
    const ro = new Database(path.join(DATA_DIR, 'chat.db'), { readonly: true });
    ro.prepare('VACUUM INTO ?').run(backup);
    ro.close();
    console.log(`Backup: ${path.basename(backup)} in ${DATA_DIR} (${(fs.statSync(backup).size / 1e6).toFixed(1)} MB)\n`);
  }

  const made = [];
  let failures = 0;

  for (const t of threads) {
    const body = msgs.filter(m => m.thread_id === t.id);
    const first = body[0];
    console.log(`  ${t.id.slice(0, 8)}  "${String(t.subject).slice(0, 60)}"  ${body.length} message(s), ${t.status}`);
    if (DRY) continue;
    if (!first) { console.error('    SKIPPED: a thread with no messages'); failures++; continue; }

    try {
      const convoId = sql.transaction(() => {
        // The conversation itself, in the shape the sidebar reads.
        const id = db.createConversation(String(t.subject), 'snh', t.opened_by === 'user' ? 'user' : 'snh');

        for (const m of body) {
          const msgId = db.addMessage(id, m.sender === 'user' ? 'user' : 'assistant', m.body, 'snh');
          sql.prepare('UPDATE messages SET timestamp = ? WHERE id = ?').run(toSqlTime(m.created_at), msgId);
        }

        // Dated from when it was RAISED, not from when this script ran, so a
        // follow-up from this morning does not read as new tonight. Written
        // after the messages because addMessage bumps updated_at on every one.
        sql.prepare(`UPDATE conversations SET
             created_at = ?, updated_at = ?, status = ?,
             archived_at = ?, archived_by = ?,
             retire_requested_at = ?, retire_reason = ?, retire_initiative_id = ?,
             supersedes_conversation_id = NULL,
             source_kind = ?, source_ref = ?,
             last_read_at = NULL, last_read_tie = 0
           WHERE id = ?`)
          .run(
            toSqlTime(t.created_at), toSqlTime(t.updated_at),
            t.status === 'retired' ? 'archived' : 'active',
            toSqlTime(t.retired_at), t.retired_by,
            toSqlTime(t.retire_requested_at), t.retire_reason, t.retire_initiative_id,
            t.source_kind, t.source_ref, id
          );

        ledger.record({
          tier: 'conversation', action: 'conversation-migrate', subject: 'system',
          targetId: id, targetText: String(t.subject),
          survivorText: String(first.body).slice(0, 200),
          reason: `The message thread ${t.id.slice(0, 8)} became a conversation in her list, dated ${toSqlTime(t.created_at)}. ` +
            'It lands unread: the panel it was read in no longer exists, and she has never seen it here.',
          evidence: {
            fromThreadId: t.id,
            threadStatus: t.status,
            // The one thing the new shape has no column for.
            perMessageReadAt: body.map(m => ({ sender: m.sender, created_at: m.created_at, read_at: m.read_at })),
            supersedesThreadId: t.supersedes_thread_id || null
          },
          reversible: false
        });
        return id;
      })();
      made.push({ t, convoId });
    } catch (err) {
      console.error(`    FAILED: ${err.message}`);
      failures++;
    }
  }

  if (DRY) { console.log('\nDry run complete — the tables are untouched.'); process.exit(0); }

  // Second pass: a thread that pointed at an earlier thread now points at that
  // thread's conversation. Second pass because the target may migrate after the
  // one that references it.
  const byThread = new Map(made.map(({ t, convoId }) => [t.id, convoId]));
  let repointed = 0;
  for (const { t, convoId } of made) {
    if (!t.supersedes_thread_id) continue;
    const target = byThread.get(t.supersedes_thread_id);
    if (!target) { console.error(`  WARN: ${t.id.slice(0, 8)} followed a thread that did not migrate — the link is dropped`); continue; }
    sql.prepare('UPDATE conversations SET supersedes_conversation_id = ? WHERE id = ?').run(target, convoId);
    repointed++;
  }

  // Only now, and only if every thread landed.
  if (failures === 0 && made.length === threads.length) {
    sql.exec('DROP TABLE IF EXISTS thread_messages');
    sql.exec('DROP TABLE IF EXISTS message_threads');
    console.log('\nDropped message_threads and thread_messages.');
  } else {
    console.error(`\nTABLES KEPT — ${failures} thread(s) did not migrate. Nothing is dropped while anything is still only in them.`);
  }

  // WHAT THE BACKFILL GUESSED, LAID OUT SO SHE CAN OVERRULE IT.
  //
  // Adding the watermark marked every pre-existing conversation read, because
  // nothing in the old store recorded whether she opened anything and 89 false
  // unread on day one is worse than none (see initSchema). The one case where
  // that guess might be wrong is a conversation SNH opened that she never
  // replied in. Listed, not acted on: raising any of them is her call.
  const neverAnswered = sql.prepare(`
    SELECT c.id, c.title, c.created_at
    FROM conversations c
    WHERE c.hidden = 0 AND c.initiated_by = 'snh' AND c.status = 'active'
      AND c.last_read_at IS NOT NULL     -- only the ones the backfill actually touched
      AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.role = 'user')
    ORDER BY c.created_at DESC
  `).all();

  console.log('\nAFTER');
  console.log(`  conversations (active)   : ${channel.listConversations({ status: 'active' }).length}`);
  console.log(`  conversations (archived) : ${channel.listConversations({ status: 'archived' }).length}`);
  console.log(`  total unread             : ${channel.totalUnread()}`);
  console.log(`  supersedes links kept    : ${repointed}`);
  for (const { convoId } of made) {
    const c = channel.getState(convoId);
    console.log(`  in the list              : "${c.title}"  ${c.created_at}  ${c.unread} unread  (${c.status})`);
  }

  if (neverAnswered.length) {
    console.log(`\nFOR HER, NOT FOR THIS SCRIPT — ${neverAnswered.length} conversation(s) SNH opened that she never replied in.`);
    console.log('They were marked read by the backfill, on the reasoning above. Nothing here has decided');
    console.log('they were unimportant; if any of them should be showing a count, say so and it can be set.');
    for (const c of neverAnswered) console.log(`  ${c.created_at}  "${String(c.title || '(untitled)').slice(0, 64)}"`);
  }
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error('FAILED:', e.stack || e.message); process.exit(1); });
