#!/usr/bin/env node
/**
 * Move the pending follow-ups and questions out of the bell and into the
 * message channel, where they were always headed.
 *
 * The item that prompted this build is one of them: a real follow-up from
 * Athena sitting under "Read or dismiss — no conversation needed" with a
 * dismiss button. It becomes the first thread.
 *
 * Destination first, then the initiative is marked `relocated` — the same order
 * and the same status the bell rework used, so nothing is taken off the bell
 * until it exists somewhere else. Delivered and expired history is untouched.
 *
 * Usage: node scripts/migrate-followups-to-messages.js [--dry-run]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const DRY = process.argv.includes('--dry-run');
const db = require(path.join(ROOT, 'db/database'));

(async () => {
  db.initDatabase();
  const sql = db.getSqliteDb();
  const messages = require(path.join(ROOT, 'db/messages'));

  const MOVE = ['followup', 'question'];
  const rows = sql.prepare(
    `SELECT * FROM initiatives WHERE status='pending' AND type IN (${MOVE.map(() => '?').join(',')}) ORDER BY created_at`
  ).all(...MOVE);

  console.log(`${DRY ? 'DRY RUN — nothing will be written\n' : ''}Pending followups/questions: ${rows.length}`);
  const made = [];
  for (const it of rows) {
    const subject = String(it.content).replace(/\s+/g, ' ').slice(0, 70);
    console.log(`  ${it.id.slice(0, 8)} ${it.type} -> thread "${subject}"`);
    if (DRY) continue;
    try {
      const t = messages.openThread({
        subject, body: it.content,
        sourceKind: it.source_kind || it.type, sourceRef: it.source_ref || `initiative:${it.id}`
      });
      // The thread is dated from the message's own creation, so a follow-up
      // raised days ago does not read as sent today.
      sql.prepare('UPDATE message_threads SET created_at = ?, updated_at = ? WHERE id = ?')
        .run(it.created_at, it.created_at, t.id);
      sql.prepare('UPDATE thread_messages SET created_at = ? WHERE thread_id = ?').run(it.created_at, t.id);
      sql.prepare("UPDATE initiatives SET status='relocated' WHERE id=? AND status='pending'").run(it.id);
      made.push({ it, t });
    } catch (err) {
      console.error(`    FAILED: ${err.message} — left pending`);
    }
  }

  if (!DRY) {
    console.log('\nAFTER');
    console.log(`  threads now      : ${messages.listThreads().length}`);
    console.log(`  unread           : ${messages.unreadCount()}`);
    console.log(`  still ringing    : ${require(path.join(ROOT, 'db/initiatives')).countPendingForBell()}`);
    console.log(`  relocated total  : ${sql.prepare("SELECT COUNT(*) n FROM initiatives WHERE status='relocated'").get().n}`);
    for (const { t } of made) console.log(`  first thread     : "${t.subject}" (${t.id.slice(0, 8)})`);
  }
  process.exit(0);
})().catch(e => { console.error('FAILED:', e.stack || e.message); process.exit(1); });
