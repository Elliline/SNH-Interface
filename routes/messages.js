/**
 * The MESSAGES view's API — her side of the channel.
 *
 * Reading a thread MARKS IT READ, because opening it is what reading is. That
 * is the only place unread ever changes: there is no expiry, no auto-dismiss
 * and no sweep, by design (see db/messages.js).
 */
const express = require('express');
const router = express.Router();
const messages = require('../db/messages');

/** GET /api/messages — the thread list, with unread counts. */
router.get('/', (req, res) => {
  try {
    res.json({
      threads: messages.listThreads({ status: req.query.status || null }),
      unread: messages.unreadCount(),
      unreadAll: messages.unreadCountAll()
    });
  } catch (error) {
    console.error('[MessagesAPI] list failed:', error.message);
    res.status(500).json({ error: 'Failed to load messages' });
  }
});

/** GET /api/messages/:id — one thread. Opening it marks its messages read. */
router.get('/:id', (req, res) => {
  try {
    const before = messages.getThread(req.params.id);
    if (!before) return res.status(404).json({ error: 'No such thread' });
    const marked = req.query.peek === '1' ? 0 : messages.markThreadRead(before.id);
    res.json({ thread: messages.getThread(before.id), markedRead: marked, unread: messages.unreadCount() });
  } catch (error) {
    console.error('[MessagesAPI] get failed:', error.message);
    res.status(500).json({ error: 'Failed to load the thread' });
  }
});

/** POST /api/messages/:id/reply — she answers. */
router.post('/:id/reply', (req, res) => {
  try {
    const body = String((req.body && req.body.body) || '').trim();
    if (!body) return res.status(400).json({ error: 'A reply needs a body' });
    const t = messages.addMessage(req.params.id, body, 'user');
    res.json({ ok: true, thread: t });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

/**
 * POST /api/messages/:id/retire — she closes it.
 *
 * Hers alone. The entity can only ask, and its ask arrives on the bell as an
 * approval; approving there lands here.
 */
router.post('/:id/retire', (req, res) => {
  try {
    const t = messages.retireThread(req.params.id, { by: 'user' });
    res.json({ ok: true, thread: t });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

module.exports = router;
