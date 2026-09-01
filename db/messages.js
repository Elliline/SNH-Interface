/**
 * MESSAGES — the channel where the entity talks to Ellie about things that
 * matter, separate from chat and separate from the bell.
 *
 * WHY IT EXISTS. On 2026-09-01 a real follow-up from Athena — about being the
 * centre of a world that does not extend past the screen — sat in the bell
 * panel under the words "Read or dismiss — no conversation needed", with a
 * dismiss button next to it. Ellie: "if she is sending them to me they are
 * important." The bell is for things that need HER; this is for things the
 * entity wants to SAY to her, and the two were never the same channel.
 *
 * THREADS, NOT ITEMS. A subject stays in one place: the entity opens a thread
 * and may add to it while it is active, so a second thought about the same
 * thing lands under the first rather than starting a rival. That is also why
 * the entity gets a tool to list its own threads — appending is only possible
 * if it can see what it already opened.
 *
 * UNREAD NEVER DECAYS, AND IT IS NOT A VERDICT. A message stays unread until
 * she reads it: no expiry, no auto-dismiss, no staleness sweep. Unread means
 * she has been busy. NOTHING in this system may read it as a signal about the
 * entity's judgement — that is why there is no "ignored" state, no score that
 * falls with age, and no path that closes a thread because nobody looked at it.
 * The only thing that ever acts on the count is the bell's backlog threshold,
 * and that says "you have things waiting", never "these were not worth
 * sending".
 *
 * RETIREMENT IS SYMMETRIC AND FINAL-ISH. Retired means closed to writes for
 * BOTH of them and readable by both forever. Ellie can retire a thread; the
 * entity can only REQUEST it, and that request goes through the bell as an
 * approval, because closing a channel she reads is her decision. If the subject
 * comes back the entity opens a NEW thread and points it at the old one, so it
 * can see it raised this before.
 */
const { randomUUID } = require('crypto');

function sqlite() { return require('./database').getSqliteDb(); }
function ledger() { return require('./corrections-ledger'); }

/** Who wrote a message. 'self' is the entity, 'user' is Ellie. */
const SENDERS = ['self', 'user'];

/**
 * THE TABLE IS `thread_messages`, NOT `messages`.
 *
 * `messages` is already taken — it is the CHAT TRANSCRIPT, one row per turn of
 * every conversation she has ever had. A CREATE TABLE IF NOT EXISTS on that
 * name does not collide loudly; it silently does nothing and leaves this module
 * reading her transcript as though it were a thread. That is how a channel
 * "separate from chat" would have ended up writing into chat.
 */
function initSchema(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS message_threads (
      id TEXT PRIMARY KEY,
      subject TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'active',
      opened_by TEXT NOT NULL DEFAULT 'self',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      retired_at DATETIME,
      retired_by TEXT,
      retire_requested_at DATETIME,
      retire_reason TEXT,
      retire_initiative_id TEXT,
      supersedes_thread_id TEXT,
      source_kind TEXT,
      source_ref TEXT,
      FOREIGN KEY (supersedes_thread_id) REFERENCES message_threads(id)
    )
  `);
  db.exec(`
    CREATE TABLE IF NOT EXISTS thread_messages (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      sender TEXT NOT NULL,
      body TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      read_at DATETIME,
      FOREIGN KEY (thread_id) REFERENCES message_threads(id)
    )
  `);
  db.exec('CREATE INDEX IF NOT EXISTS idx_messages_thread ON thread_messages(thread_id, created_at)');
  db.exec("CREATE INDEX IF NOT EXISTS idx_messages_unread ON thread_messages(read_at) WHERE read_at IS NULL");
  db.exec('CREATE INDEX IF NOT EXISTS idx_threads_status ON message_threads(status, updated_at)');
  return { tables: ['message_threads', 'thread_messages'] };
}

// ---------------------------------------------------------------- reads

function getThread(id) {
  if (!id) return null;
  const db = sqlite();
  const t = db.prepare('SELECT * FROM message_threads WHERE id = ?').get(id)
    || db.prepare('SELECT * FROM message_threads WHERE id LIKE ?').get(`${id}%`);
  if (!t) return null;
  const messages = db.prepare('SELECT * FROM thread_messages WHERE thread_id = ? ORDER BY created_at ASC').all(t.id);
  return { ...t, messages, unread: messages.filter(m => m.sender === 'self' && !m.read_at).length };
}

/**
 * Threads, newest activity first, each with its own unread count.
 *
 * ACTIVE FIRST AND RETIRED LAST rather than retired hidden: a retired thread is
 * still readable by both of them forever, and hiding it would make "I raised
 * this before" unverifiable from the view where it matters.
 */
function listThreads({ status = null, limit = 100 } = {}) {
  const db = sqlite();
  const where = status ? 'WHERE t.status = ?' : '';
  const bind = status ? [status] : [];
  return db.prepare(`
    SELECT t.*,
           (SELECT COUNT(*) FROM thread_messages m WHERE m.thread_id = t.id) AS message_count,
           (SELECT COUNT(*) FROM thread_messages m WHERE m.thread_id = t.id AND m.sender = 'self' AND m.read_at IS NULL) AS unread,
           (SELECT m.body FROM thread_messages m WHERE m.thread_id = t.id ORDER BY m.created_at DESC LIMIT 1) AS last_body,
           (SELECT m.sender FROM thread_messages m WHERE m.thread_id = t.id ORDER BY m.created_at DESC LIMIT 1) AS last_sender
    FROM message_threads t
    ${where}
    ORDER BY (t.status = 'active') DESC, t.updated_at DESC
    LIMIT ?
  `).all(...bind, limit);
}

/**
 * How many messages from the entity she has not read.
 *
 * ACTIVE THREADS ONLY, and this is a judgement worth naming: a retired thread's
 * unread messages stay unread — the record does not pretend she read something
 * she did not — but they stop counting towards the backlog, because the backlog
 * is a prompt to act and a retired thread is closed to writes. Counting them
 * would nag her about a conversation she has already ended.
 */
function unreadCount() {
  try {
    return sqlite().prepare(`
      SELECT COUNT(*) n FROM thread_messages m
      JOIN message_threads t ON t.id = m.thread_id
      WHERE m.sender = 'self' AND m.read_at IS NULL AND t.status = 'active'
    `).get().n;
  } catch { return 0; }
}

/** Unread including retired threads — what the view shows on a retired row. */
function unreadCountAll() {
  try {
    return sqlite().prepare(
      "SELECT COUNT(*) n FROM thread_messages WHERE sender = 'self' AND read_at IS NULL"
    ).get().n;
  } catch { return 0; }
}

// ---------------------------------------------------------------- writes

/** Open a thread with its first message. Ledgered. */
function openThread({ subject, body, supersedesThreadId = null, sourceKind = null, sourceRef = null, sender = 'self' }) {
  const db = sqlite();
  const s = String(subject || '').trim();
  const b = String(body || '').trim();
  if (!s || !b) throw new Error('a thread needs a subject and a first message');
  if (!SENDERS.includes(sender)) throw new Error(`unknown sender: ${sender}`);

  const id = randomUUID();
  const now = new Date().toISOString();
  const run = db.transaction(() => {
    db.prepare(`INSERT INTO message_threads
      (id, subject, status, opened_by, created_at, updated_at, supersedes_thread_id, source_kind, source_ref)
      VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?)`)
      .run(id, s, sender, now, now, supersedesThreadId, sourceKind, sourceRef);
    db.prepare('INSERT INTO thread_messages (id, thread_id, sender, body, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(randomUUID(), id, sender, b, now);
    ledger().record({
      tier: 'message', action: 'thread-open', subject: sender,
      targetId: id, targetText: s, survivorText: b.slice(0, 200),
      reason: `A message thread was opened: "${s}".` +
        (supersedesThreadId ? ` It follows an earlier thread on the same subject.` : ''),
      evidence: { supersedes: supersedesThreadId, sourceKind, sourceRef },
      reversible: false
    });
  });
  run();
  return getThread(id);
}

/** Add to a thread that is still active. Ledgered. */
function addMessage(threadId, body, sender = 'self') {
  const db = sqlite();
  const t = getThread(threadId);
  if (!t) throw new Error('no such thread');
  if (t.status !== 'active') throw new Error(`thread is ${t.status} — retired threads are closed to writes for both of us`);
  if (!SENDERS.includes(sender)) throw new Error(`unknown sender: ${sender}`);
  const b = String(body || '').trim();
  if (!b) throw new Error('a message needs a body');

  const id = randomUUID();
  const now = new Date().toISOString();
  db.transaction(() => {
    db.prepare('INSERT INTO thread_messages (id, thread_id, sender, body, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(id, t.id, sender, b, now);
    db.prepare('UPDATE message_threads SET updated_at = ? WHERE id = ?').run(now, t.id);
    ledger().record({
      tier: 'message', action: 'message-add', subject: sender,
      targetId: t.id, targetText: t.subject, survivorText: b.slice(0, 200),
      reason: `${sender === 'self' ? 'A message was added to' : 'Ellie replied in'} "${t.subject}".`,
      reversible: false
    });
  })();
  return getThread(t.id);
}

/**
 * Mark a thread's entity-messages read. Called when she opens the thread.
 *
 * Only messages FROM the entity are ever unread — her own replies are not
 * something she needs to read.
 */
function markThreadRead(threadId) {
  const t = getThread(threadId);
  if (!t) return 0;
  return sqlite().prepare(
    "UPDATE thread_messages SET read_at = ? WHERE thread_id = ? AND sender = 'self' AND read_at IS NULL"
  ).run(new Date().toISOString(), t.id).changes;
}

/**
 * The entity ASKS to close a thread. It cannot close one itself.
 *
 * Raised on the bell as an approval, which is what the bell is for — something
 * waiting on her decision. Approvals cannot be dismissed, only decided, so the
 * request cannot be waved away.
 */
async function requestRetire(threadId, reason = null) {
  const db = sqlite();
  const t = getThread(threadId);
  if (!t) throw new Error('no such thread');
  if (t.status !== 'active') return t;
  if (t.retire_requested_at) return t;   // asked once is enough

  const initiatives = require('./initiatives');
  const now = new Date().toISOString();
  const why = String(reason || '').trim() || 'It feels finished to me.';
  const initiativeId = await initiatives.addInitiative({
    type: 'proposal',
    content: `May I close the message thread "${t.subject}"? ${why}`,
    sourceKind: 'thread-retire',
    sourceRef: t.id,
    priority: 5
  });
  db.prepare('UPDATE message_threads SET retire_requested_at = ?, retire_reason = ?, retire_initiative_id = ? WHERE id = ?')
    .run(now, why, initiativeId || null, t.id);
  ledger().record({
    tier: 'message', action: 'thread-retire-request', subject: 'self',
    targetId: t.id, targetText: t.subject,
    reason: `The entity asked to close "${t.subject}": ${why} NOTHING IS CLOSED — it is waiting on Ellie.`,
    reversible: false
  });
  return getThread(t.id);
}

/** Close a thread. Hers to do — or the result of her approving a request. */
function retireThread(threadId, { by = 'user', reason = null } = {}) {
  const db = sqlite();
  const t = getThread(threadId);
  if (!t) throw new Error('no such thread');
  if (t.status === 'retired') return t;
  const now = new Date().toISOString();
  db.transaction(() => {
    db.prepare("UPDATE message_threads SET status = 'retired', retired_at = ?, retired_by = ?, updated_at = ? WHERE id = ?")
      .run(now, by, now, t.id);
    // The pending approval, if any, is resolved by the decision itself.
    if (t.retire_initiative_id) {
      try { require('./initiatives').markDelivered(t.retire_initiative_id, { channel: 'messages' }); } catch { /* non-fatal */ }
    }
    ledger().record({
      tier: 'message', action: 'thread-retire', subject: by,
      targetId: t.id, targetText: t.subject,
      reason: reason || `"${t.subject}" was closed by ${by === 'user' ? 'Ellie' : by}. ` +
        'It is closed to writes for both of them and stays readable by both.',
      reversible: false
    });
  })();
  return getThread(t.id);
}

/** Threads on a subject the entity raised before — what a new thread points at. */
function findPriorThreads(subjectLike, { limit = 5 } = {}) {
  const needle = `%${String(subjectLike || '').trim()}%`;
  return sqlite().prepare(
    'SELECT * FROM message_threads WHERE subject LIKE ? ORDER BY created_at DESC LIMIT ?'
  ).all(needle, limit);
}

module.exports = {
  initSchema, SENDERS,
  getThread, listThreads, unreadCount, unreadCountAll, findPriorThreads,
  openThread, addMessage, markThreadRead, requestRetire, retireThread
};
