/**
 * THE CONVERSATION CHANNEL — messages and chat are the same thing.
 *
 * WHY THIS REPLACED A MESSAGES PANEL (2026-09-01, same day). The morning's
 * problem was real: a follow-up from Athena sat in the bell under "Read or
 * dismiss — no conversation needed", with a dismiss button beside it. The
 * afternoon's fix was a second inbox with its own tables, its own view and its
 * own badge. Ellie, on seeing it: "I want all conversation in this area." Two
 * places to read what the entity says is one place too many, and the one it
 * built was the one she does not live in.
 *
 * So there is ONE list — the conversation sidebar — and everything the entity
 * wants to say lands in it: either by opening a conversation there, the way the
 * SNH-prefixed unprompted ones already appear, or by adding to an active
 * conversation that already exists, INCLUDING ONE SHE STARTED. She reads and
 * replies where she already reads and replies.
 *
 * THIS MODULE DOES NOT OWN A TABLE. It adds state to `conversations` and reads
 * `messages` — the chat transcript, the real one. The previous build went out
 * of its way NOT to touch `messages` because a channel "separate from chat"
 * must not write into chat; that reasoning was correct for that design and is
 * exactly backwards for this one. An entity message IS a chat turn now.
 *
 * UNREAD NEVER DECAYS AND IT IS NOT A VERDICT. No expiry, no auto-dismiss, no
 * staleness sweep, no "ignored" state, no score that falls with age. Unread
 * means she has been busy. NOTHING in this system may read the count as a
 * signal about the entity's judgement. The only thing that ever acts on it is
 * the bell's backlog threshold, and that says "you have things waiting", never
 * "these were not worth sending".
 *
 * ARCHIVE IS SYMMETRIC. Archived means closed to writes for BOTH of them and
 * readable by both forever. Ellie archives; the entity can only ASK, and the
 * ask lands on the bell as an approval, which cannot be dismissed, only
 * decided. When a subject comes back the entity opens a NEW conversation
 * pointing at the archived one, so it can see it raised this before.
 */
const { getSqliteDb } = require('./database');

function sqlite() { return getSqliteDb(); }
function ledger() { return require('./corrections-ledger'); }

/**
 * TIME IS STORED THE WAY `messages.timestamp` STORES IT, AND THIS IS THE WHOLE
 * BALLGAME FOR UNREAD.
 *
 * `messages.timestamp` is SQLite's unmarked CURRENT_TIMESTAMP: UTC, and
 * formatted "2026-09-01 21:47:33" with a SPACE. Everything the previous
 * messages module wrote was `new Date().toISOString()` — "2026-09-01T21:47:33.985Z"
 * with a T. Unread is a string comparison between a watermark and those
 * timestamps, and those two formats DO NOT COMPARE: ' ' is 0x20 and 'T' is
 * 0x54, so any ISO watermark sorts ABOVE every space-formatted message on the
 * same day. A watermark written the obvious way would have made every
 * conversation read zero unread forever, silently, and looked completely
 * correct in the code.
 *
 * So every datetime this module writes to `conversations` goes through here,
 * and the unread comparison is a comparison between two things written the
 * same way.
 */
function sqlNow() {
  return new Date().toISOString().slice(0, 19).replace('T', ' ');
}

/** Columns this module adds to `conversations`. Additive; nothing beside it. */
function initSchema(db) {
  const have = new Set(db.prepare('PRAGMA table_info(conversations)').all().map(c => c.name));
  const add = (name, decl) => { if (!have.has(name)) db.exec(`ALTER TABLE conversations ADD COLUMN ${name} ${decl}`); };

  // 'active' | 'archived'. Archived is closed to writes for both of them.
  add('status', "TEXT NOT NULL DEFAULT 'active'");
  add('archived_at', 'DATETIME');
  add('archived_by', 'TEXT');

  // The entity's ask to archive, and the bell approval it is waiting on.
  add('retire_requested_at', 'DATETIME');
  add('retire_reason', 'TEXT');
  add('retire_initiative_id', 'TEXT');

  // A subject that came back: the archived conversation this one follows.
  add('supersedes_conversation_id', 'TEXT');

  // Where a conversation the entity opened came from — a question row, a
  // reflection follow-up, a self-coherence finding, a tool call.
  add('source_kind', 'TEXT');
  add('source_ref', 'TEXT');

  // THE READ WATERMARK. See sqlNow() above for the format, and markRead() for
  // why there are two columns and not one.
  const watermarkIsNew = !have.has('last_read_at');
  add('last_read_at', 'DATETIME');
  add('last_read_tie', 'INTEGER NOT NULL DEFAULT 0');

  db.exec('CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status, updated_at DESC)');

  // EXISTING CONVERSATIONS START READ, ONCE, WHEN THE COLUMN FIRST APPEARS.
  //
  // A null watermark means "she has never opened this", which is right for a
  // conversation the entity opens tomorrow and wrong for every conversation she
  // has ever had. Left alone, the day this shipped would have opened with 89
  // unread across 29 conversations she has demonstrably been reading and
  // replying in — a number that is false, and false in the direction that
  // teaches her to ignore the badge.
  //
  // The counter-argument is real: an unprompted conversation she never opened
  // IS honestly unread, and this marks it read. But nothing in the old store
  // recorded whether she opened anything, so both answers are guesses, and this
  // is the guess that does not cry wolf on day one. The migration reports which
  // SNH-opened conversations never got a reply so she can raise any of them
  // deliberately — that is hers to decide, not something to infer here.
  if (watermarkIsNew) {
    const n = db.prepare(`
      UPDATE conversations SET
        last_read_at = (SELECT MAX(timestamp) FROM messages WHERE conversation_id = conversations.id),
        last_read_tie = (SELECT COUNT(*) FROM messages m
                          WHERE m.conversation_id = conversations.id AND m.role = 'assistant'
                            AND m.timestamp = (SELECT MAX(timestamp) FROM messages WHERE conversation_id = conversations.id))
      WHERE EXISTS (SELECT 1 FROM messages WHERE conversation_id = conversations.id)
    `).run().changes;
    if (n) console.log(`Migration: ${n} existing conversation(s) start read — see db/conversation-channel.js initSchema for why`);
  }

  return { table: 'conversations' };
}

// ---------------------------------------------------------------- unread

/**
 * UNREAD, AS ONE SQL EXPRESSION, IN TWO PARTS — AND THE SECOND PART IS NOT
 * CLEVERNESS FOR ITS OWN SAKE.
 *
 * Part one is the obvious half: assistant messages STRICTLY NEWER than the
 * watermark. Part two exists because `messages.timestamp` has one-SECOND
 * resolution, so "newer than the last thing she saw" cannot be expressed by a
 * timestamp alone. A message that lands in the same second as the newest
 * message present when she opened the conversation is neither strictly newer
 * (so `>` drops it — silently, forever, which is the one thing this feature
 * must never do) nor safely equal (so `>=` would re-count messages she has
 * demonstrably read, every time, until she opened it again).
 *
 * So the watermark records HOW MANY assistant messages already sat on that
 * second, and the tie term counts only the ones that arrived after. Both halves
 * are load-bearing: drop part two and same-second arrivals vanish; drop the
 * tie count and everything on the boundary second reappears as unread.
 *
 * A NULL watermark means she has never opened it: COALESCE to '' , which sorts
 * below every real timestamp, and everything the entity said is unread. That is
 * correct for an unprompted conversation the entity has just opened.
 */
const UNREAD_SQL = `
  ((SELECT COUNT(*) FROM messages m
      WHERE m.conversation_id = c.id AND m.role = 'assistant'
        AND m.timestamp > COALESCE(c.last_read_at, ''))
   + MAX(0,
       (SELECT COUNT(*) FROM messages m
          WHERE m.conversation_id = c.id AND m.role = 'assistant'
            AND c.last_read_at IS NOT NULL AND m.timestamp = c.last_read_at)
       - COALESCE(c.last_read_tie, 0)))`;

const LIST_COLUMNS = `
  c.id, c.title, c.created_at, c.updated_at, c.model_used, c.initiated_by,
  c.status, c.archived_at, c.archived_by,
  c.retire_requested_at, c.retire_reason, c.retire_initiative_id,
  c.supersedes_conversation_id, c.source_kind, c.source_ref, c.last_read_at,
  (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY timestamp ASC LIMIT 1) AS preview,
  (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS message_count,
  ${UNREAD_SQL} AS unread`;

/**
 * The sidebar list. One list, and the only list.
 *
 * ACTIVE AND ARCHIVED ARE SEPARATE SCOPES rather than one list with the
 * archived greyed out, because the archive is a place she goes deliberately —
 * but nothing is hidden: an archived conversation is readable forever, which is
 * what makes "I raised this before" checkable from the view where it matters.
 *
 * `hidden` conversations stay out of both. That flag marks a row as not really
 * hers (verification turns, clone artifacts); it is already the one thing
 * history search consults, and a synthetic turn must not be able to show up in
 * her sidebar carrying an unread count.
 */
function listConversations({ status = 'active', limit = 500 } = {}) {
  const db = sqlite();
  const where = status ? 'WHERE c.hidden = 0 AND c.status = ?' : 'WHERE c.hidden = 0';
  const bind = status ? [status] : [];
  return db.prepare(`
    SELECT ${LIST_COLUMNS}
    FROM conversations c
    ${where}
    ORDER BY c.updated_at DESC
    LIMIT ?
  `).all(...bind, limit);
}

/** One conversation's channel state, unread included. */
function getState(id) {
  if (!id) return null;
  return sqlite().prepare(`SELECT ${LIST_COLUMNS} FROM conversations c WHERE c.id = ?`).get(id) || null;
}

/** Unread on one conversation. */
function unreadFor(id) {
  const s = getState(id);
  return s ? s.unread : 0;
}

/**
 * The number at the top of the list, and the number the bell's backlog reads.
 *
 * ACTIVE ONLY, and that is a judgement worth naming: an archived conversation's
 * unread messages stay honestly unread in the record — nothing pretends she
 * read what she did not — but they stop counting toward the total, because the
 * total is a prompt to act and an archived conversation is closed to writes.
 * Counting them would nag her about a conversation she has already ended.
 */
function totalUnread() {
  try {
    return sqlite().prepare(`
      SELECT COALESCE(SUM(${UNREAD_SQL}), 0) AS n
      FROM conversations c
      WHERE c.hidden = 0 AND c.status = 'active'
    `).get().n;
  } catch { return 0; }
}

/** Unread across every conversation, archived included — the honest record. */
function totalUnreadAll() {
  try {
    return sqlite().prepare(`
      SELECT COALESCE(SUM(${UNREAD_SQL}), 0) AS n FROM conversations c WHERE c.hidden = 0
    `).get().n;
  } catch { return 0; }
}

/**
 * She opened it. This is the ONLY thing that clears unread — there is no
 * expiry, no sweep, no auto-dismiss and no path anywhere that closes or clears
 * a conversation because nobody looked at it.
 *
 * The watermark is set to the newest message PRESENT RIGHT NOW, not to the
 * clock. Using the clock would mean anything written between the last message
 * and this instant is marked read without ever having been on screen; using the
 * newest message means the watermark names something she actually saw. The
 * companion tie count is explained on UNREAD_SQL.
 *
 * An empty conversation is left alone: there is nothing to have read, unread is
 * already zero, and writing a clock watermark into it would open exactly the
 * same-second hole the tie count exists to close.
 */
function markRead(id) {
  const db = sqlite();
  const conv = db.prepare('SELECT id FROM conversations WHERE id = ?').get(id);
  if (!conv) return 0;
  const before = unreadFor(conv.id);
  const newest = db.prepare('SELECT MAX(timestamp) AS mx FROM messages WHERE conversation_id = ?').get(conv.id);
  if (!newest || !newest.mx) return 0;
  const tie = db.prepare(
    "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ? AND role = 'assistant' AND timestamp = ?"
  ).get(conv.id, newest.mx).n;
  db.prepare('UPDATE conversations SET last_read_at = ?, last_read_tie = ? WHERE id = ?')
    .run(newest.mx, tie, conv.id);
  return before;
}

// ---------------------------------------------------------------- writes

/** Whether the entity may write here at all. */
function assertWritable(conv) {
  if (!conv) throw new Error('no such conversation');
  if (conv.status === 'archived') {
    throw new Error('that conversation is archived — archived conversations are closed to writes for both of us, and stay readable by both');
  }
  return conv;
}

function chatModel() {
  try { return require('./config').getConfig().models?.chat?.model || 'snh'; }
  catch { return 'snh'; }
}

/**
 * The entity says something into an EXISTING conversation. Its own or hers —
 * that is what "adding to a thread" means now.
 *
 * It goes in as an assistant turn in the real transcript, because that is what
 * it is. Ledgered, like every other write it makes.
 */
function sendInto(conversationId, body, { model = null, sourceKind = null, sourceRef = null } = {}) {
  const db = require('./database');
  const conv = assertWritable(getState(conversationId));
  const text = String(body || '').trim();
  if (!text) throw new Error('a message needs a body');

  const messageId = db.addMessage(conv.id, 'assistant', text, model || conv.model_used || chatModel());
  ledger().record({
    tier: 'conversation', action: 'conversation-send', subject: 'self',
    targetId: conv.id, targetText: conv.title || '(untitled conversation)',
    survivorId: messageId, survivorText: text.slice(0, 200),
    reason: `The entity added a message to "${conv.title || 'an untitled conversation'}"` +
      `${conv.initiated_by === 'user' ? ', a conversation Ellie started' : ''}.`,
    evidence: { sourceKind, sourceRef, conversationId: conv.id },
    reversible: false
  });
  return getState(conv.id);
}

/**
 * The entity opens a NEW conversation in her list — the same way the
 * SNH-prefixed unprompted ones already appear, because it is the same thing.
 *
 * `supersedesConversationId` is how a returning subject shows it raised this
 * before: the new conversation points at the archived one.
 */
function openConversation({ title, body, supersedesConversationId = null, sourceKind = null, sourceRef = null, model = null }) {
  const db = require('./database');
  const sql = sqlite();
  const text = String(body || '').trim();
  if (!text) throw new Error('a conversation needs a first message');
  const heading = String(title || '').trim() || text.slice(0, 48) + (text.length > 48 ? '…' : '');

  const id = db.createConversation(heading, model || chatModel(), 'snh');
  sql.prepare(`UPDATE conversations
     SET supersedes_conversation_id = ?, source_kind = ?, source_ref = ?
     WHERE id = ?`).run(supersedesConversationId, sourceKind, sourceRef, id);
  const messageId = db.addMessage(id, 'assistant', text, model || chatModel());

  ledger().record({
    tier: 'conversation', action: 'conversation-open', subject: 'self',
    targetId: id, targetText: heading,
    survivorId: messageId, survivorText: text.slice(0, 200),
    reason: `The entity opened a conversation: "${heading}".` +
      (supersedesConversationId ? ' The subject came back — it points at the archived one.' : ''),
    evidence: { supersedes: supersedesConversationId, sourceKind, sourceRef },
    reversible: false
  });
  return getState(id);
}

/**
 * The entity ASKS to archive a conversation. It cannot archive one itself.
 *
 * The ask reaches the bell as an approval — the bell is for things waiting on
 * her decision, and an approval cannot be dismissed, only decided, so the
 * request cannot be waved away.
 */
async function requestRetire(conversationId, reason = null) {
  const sql = sqlite();
  const conv = getState(conversationId);
  if (!conv) throw new Error('no such conversation');
  if (conv.status !== 'active') return conv;
  if (conv.retire_requested_at) return conv;    // asked once is enough

  const initiatives = require('./initiatives');
  const why = String(reason || '').trim() || 'It feels finished to me.';
  const initiativeId = await initiatives.addInitiative({
    type: 'proposal',
    content: `May I archive "${conv.title || 'an untitled conversation'}"? ${why}`,
    sourceKind: 'conversation-retire',
    sourceRef: conv.id,
    priority: 5
  });
  sql.prepare(`UPDATE conversations
     SET retire_requested_at = ?, retire_reason = ?, retire_initiative_id = ?
     WHERE id = ?`).run(sqlNow(), why, initiativeId || null, conv.id);
  ledger().record({
    tier: 'conversation', action: 'conversation-retire-request', subject: 'self',
    targetId: conv.id, targetText: conv.title || '(untitled conversation)',
    reason: `The entity asked to archive "${conv.title || 'an untitled conversation'}": ${why} ` +
      'NOTHING IS ARCHIVED — it is waiting on Ellie.',
    reversible: false
  });
  return getState(conv.id);
}

/**
 * Archive it. Hers to do — directly, or by approving a request.
 *
 * Unread is NOT cleared by archiving. What she did not read she did not read,
 * and the record says so; archiving only stops it counting toward the total she
 * is being asked to act on.
 */
function archive(conversationId, { by = 'user', reason = null } = {}) {
  const sql = sqlite();
  const conv = getState(conversationId);
  if (!conv) throw new Error('no such conversation');
  if (conv.status === 'archived') return conv;
  const now = sqlNow();
  sql.transaction(() => {
    sql.prepare("UPDATE conversations SET status = 'archived', archived_at = ?, archived_by = ? WHERE id = ?")
      .run(now, by, conv.id);
    if (conv.retire_initiative_id) {
      // The decision itself resolves the approval that was waiting for it.
      try { require('./initiatives').markDelivered(conv.retire_initiative_id, { channel: 'conversations' }); }
      catch { /* non-fatal */ }
    }
    ledger().record({
      tier: 'conversation', action: 'conversation-archive', subject: by,
      targetId: conv.id, targetText: conv.title || '(untitled conversation)',
      reason: reason || `"${conv.title || 'an untitled conversation'}" was archived by ${by === 'user' ? 'Ellie' : by}. ` +
        'It is closed to writes for both of them and stays readable by both.',
      reversible: false
    });
  })();
  return getState(conv.id);
}

/** Put it back on the active list. Hers alone; the entity has no path here. */
function unarchive(conversationId, { by = 'user' } = {}) {
  const sql = sqlite();
  const conv = getState(conversationId);
  if (!conv) throw new Error('no such conversation');
  if (conv.status !== 'archived') return conv;
  sql.prepare("UPDATE conversations SET status = 'active', archived_at = NULL, archived_by = NULL WHERE id = ?")
    .run(conv.id);
  ledger().record({
    tier: 'conversation', action: 'conversation-unarchive', subject: by,
    targetId: conv.id, targetText: conv.title || '(untitled conversation)',
    reason: `"${conv.title || 'an untitled conversation'}" was reopened by ${by === 'user' ? 'Ellie' : by}.`,
    reversible: false
  });
  return getState(conv.id);
}

/**
 * Conversations on a subject, newest first — what "have I raised this before"
 * reads, and what a new conversation points at when the subject comes back.
 */
function findPrior(subjectLike, { limit = 5 } = {}) {
  const needle = `%${String(subjectLike || '').trim()}%`;
  return sqlite().prepare(`
    SELECT ${LIST_COLUMNS} FROM conversations c
    WHERE c.hidden = 0 AND c.title LIKE ?
    ORDER BY c.created_at DESC LIMIT ?
  `).all(needle, limit);
}

module.exports = {
  initSchema, sqlNow, UNREAD_SQL,
  listConversations, getState, unreadFor, totalUnread, totalUnreadAll, findPrior,
  markRead, sendInto, openConversation, requestRetire, archive, unarchive
};
