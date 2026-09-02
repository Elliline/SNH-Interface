/**
 * Conversation Management API Routes
 *
 * THE SIDEBAR LIST IS THE ONLY LIST (2026-09-01). What the entity wants to say
 * to her arrives here, not in a second inbox — so these routes carry the unread
 * counts, the total, and the archive scope alongside the ordinary CRUD.
 *
 * OPENING A CONVERSATION MARKS IT READ, because opening it is what reading is,
 * and it is the ONLY thing that clears unread. There is no expiry, no
 * auto-dismiss and no sweep anywhere in this file or the module behind it.
 */

const express = require('express');
const router = express.Router();

// Import database functions (will be implemented in db/database.js)
const {
  getConversations,
  getConversation,
  createConversation,
  deleteConversation,
  updateConversationTitle,
  deleteConversationEmbeddings
} = require('../db/database.js');
const channel = require('../db/conversation-channel');

/**
 * GET /api/conversations
 * List conversations for the sidebar, with per-conversation unread counts.
 *
 * ?status=archived for the Archive tab; active is the default.
 *
 * Still returns a bare ARRAY, because that is what the sidebar has always been
 * handed and every field it needs now rides on the rows themselves. The totals
 * the header shows come from GET /api/conversations/unread — one small request
 * a badge can poll without pulling the whole list.
 */
router.get('/', (req, res) => {
  try {
    const status = req.query.status === 'archived' ? 'archived'
      : req.query.status === 'all' ? null : 'active';
    res.json(getConversations(status));
  } catch (error) {
    console.error('Error fetching conversations:', error.message);
    res.status(500).json({
      error: 'Failed to fetch conversations',
      details: error.message
    });
  }
});

/**
 * GET /api/conversations/unread
 * The number at the top of the list. Active conversations only — see
 * totalUnread() for why an archived conversation's unread stops counting.
 *
 * Declared BEFORE /:id so "unread" is never read as a conversation id.
 */
router.get('/unread', (req, res) => {
  try {
    res.json({
      unread: channel.totalUnread(),
      unreadAll: channel.totalUnreadAll(),
      archived: getConversations('archived').length
    });
  } catch (error) {
    console.error('Error counting unread:', error.message);
    res.status(500).json({ error: 'Failed to count unread' });
  }
});

/**
 * POST /api/conversations
 * Create a new conversation
 * Body: { model_used }
 * Returns: { id, title, created_at, model_used }
 */
router.post('/', (req, res) => {
  try {
    const { model_used } = req.body;

    // Validate required field
    if (!model_used || typeof model_used !== 'string') {
      return res.status(400).json({
        error: 'model_used is required and must be a string'
      });
    }

    // Validate model name length
    if (model_used.length > 100) {
      return res.status(400).json({
        error: 'model_used exceeds maximum length of 100 characters'
      });
    }

    const id = createConversation(null, model_used);
    res.status(201).json({ id, title: null, model_used });
  } catch (error) {
    console.error('Error creating conversation:', error.message);
    res.status(500).json({
      error: 'Failed to create conversation',
      details: error.message
    });
  }
});

/**
 * GET /api/conversations/:id
 * Get full conversation with all messages
 * Returns: { id, title, created_at, updated_at, model_used, messages: [{id, role, content, timestamp, model}] }
 */
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Validate ID format (UUID)
    if (!id || typeof id !== 'string' || id.length > 50) {
      return res.status(400).json({
        error: 'Invalid conversation ID'
      });
    }

    // Opening it IS reading it, and this is the only place unread ever clears.
    // ?peek=1 for anything that needs to look without claiming she looked.
    const unreadBefore = req.query.peek === '1' ? channel.unreadFor(id) : channel.markRead(id);

    const conversation = getConversation(id);

    if (!conversation) {
      return res.status(404).json({
        error: 'Conversation not found'
      });
    }

    res.json({ ...conversation, unread_cleared: unreadBefore, unread: channel.unreadFor(id) });
  } catch (error) {
    console.error('Error fetching conversation:', error.message);
    res.status(500).json({
      error: 'Failed to fetch conversation',
      details: error.message
    });
  }
});

/**
 * DELETE /api/conversations/:id
 * Delete a conversation and its embeddings
 * Returns: { success: true }
 */
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Validate ID format (UUID)
    if (!id || typeof id !== 'string' || id.length > 50) {
      return res.status(400).json({
        error: 'Invalid conversation ID'
      });
    }

    // Check if conversation exists
    const existing = getConversation(id);
    if (!existing) {
      return res.status(404).json({
        error: 'Conversation not found'
      });
    }

    // Delete embeddings first (if they exist)
    try {
      await deleteConversationEmbeddings(id);
    } catch (embeddingError) {
      // Log but don't fail if embeddings deletion fails
      console.warn('Warning: Failed to delete embeddings for conversation', id, ':', embeddingError.message);
    }

    // Delete the conversation
    deleteConversation(id);
    const deleted = true;

    if (!deleted) {
      return res.status(404).json({
        error: 'Conversation not found'
      });
    }

    res.json({ success: true });
  } catch (error) {
    console.error('Error deleting conversation:', error.message);
    res.status(500).json({
      error: 'Failed to delete conversation',
      details: error.message
    });
  }
});

/**
 * PUT /api/conversations/:id/title
 * Rename a conversation
 * Body: { title }
 * Returns: { success: true, title }
 */
router.put('/:id/title', async (req, res) => {
  try {
    const { id } = req.params;
    const { title } = req.body;

    // Validate ID format (UUID)
    if (!id || typeof id !== 'string' || id.length > 50) {
      return res.status(400).json({
        error: 'Invalid conversation ID'
      });
    }

    // Validate title
    if (!title || typeof title !== 'string') {
      return res.status(400).json({
        error: 'title is required and must be a string'
      });
    }

    if (title.trim().length === 0) {
      return res.status(400).json({
        error: 'title cannot be empty'
      });
    }

    if (title.length > 255) {
      return res.status(400).json({
        error: 'title exceeds maximum length of 255 characters'
      });
    }

    // Check if conversation exists
    const existing = getConversation(id);
    if (!existing) {
      return res.status(404).json({
        error: 'Conversation not found'
      });
    }

    updateConversationTitle(id, title.trim());

    res.json({ success: true, title: title.trim() });
  } catch (error) {
    console.error('Error updating conversation title:', error.message);
    res.status(500).json({
      error: 'Failed to update conversation title',
      details: error.message
    });
  }
});

/**
 * POST /api/conversations/:id/archive
 * She retires it. Hers alone — the entity can only ask, and its ask arrives on
 * the bell as an approval; approving there lands here.
 *
 * Archived means closed to writes for BOTH of them and readable by both
 * forever. Nothing is deleted and no unread is cleared.
 */
router.post('/:id/archive', (req, res) => {
  try {
    const conversation = channel.archive(req.params.id, { by: 'user' });
    res.json({ success: true, conversation });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

/**
 * POST /api/conversations/:id/unarchive
 * She changes her mind. Also hers alone: the entity has no path back either,
 * because "closed to writes for both" has to mean something.
 */
router.post('/:id/unarchive', (req, res) => {
  try {
    const conversation = channel.unarchive(req.params.id, { by: 'user' });
    res.json({ success: true, conversation });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

module.exports = router;
