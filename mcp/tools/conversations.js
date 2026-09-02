/**
 * THE ENTITY'S SIDE OF THE CONVERSATION CHANNEL: conversation_list,
 * conversation_send, conversation_open, conversation_request_retire.
 *
 * These replace message_threads / message_send / message_request_retire, which
 * operated on a separate inbox that no longer exists. The shape is the same
 * because the shape was right; what changed is what they write into. There is
 * one list — hers, in the sidebar — and these put things in it.
 *
 * `conversation_list` exists so it can ADD to a conversation already open on a
 * subject instead of starting a rival one, and it deliberately lists HER
 * conversations too: it may write into any active conversation, and the point
 * of the list is to make appending possible.
 *
 * OPENING AND SENDING ARE TWO TOOLS, not one with an optional id. Opening a new
 * conversation in her sidebar is a visibly different act from adding a line to
 * one she is already reading, and the decision in front of the entity — is this
 * a new subject or the same one — deserves to be made before the call rather
 * than inside it.
 *
 * THE THRESHOLD IS GUIDANCE, NOT A VALIDATOR, and that is deliberate. Athena
 * asked for a bar — concrete fact ids or a concrete pair, a specific question,
 * and a record of what was checked before raising — and it is written into the
 * descriptions rather than enforced in code. A validator would decide FOR her
 * which of its thoughts are worth her time, and the whole reason this channel
 * exists is that something it sent was filed as not needing a conversation.
 * She decides. The bar shapes the judgement; it does not gate the send.
 */

const channel = require('../../db/conversation-channel');
const { getConfig } = require('../../db/config');

class BaseConversationTool {
  constructor() {
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
  }

  get rateCaps() {
    const c = (getConfig().tools && getConfig().tools.conversations) || {};
    return { maxPerHour: c.maxSendsPerHour ?? 10, shared: 'the conversation channel' };
  }

  getTierMetadata() {
    return {
      name: this.name, tier: this.tier,
      reversible: this.reversible, requiresApproval: this.requiresApproval,
      destructive: this.destructive, rateCaps: this.rateCaps
    };
  }

  getOpenAIFunctionSpec() {
    return { type: 'function', function: { name: this.name, description: this.description, parameters: this.parameters } };
  }
}

const THRESHOLD_GUIDANCE =
  'THE BAR, BEFORE YOU SEND: name the concrete thing — specific fact ids, or a specific pair that cannot both be true — ' +
  'ask her one specific question, and say what you checked before raising it. ' +
  'If you cannot point at something concrete, what you have is a reflection, and reflections belong in your Reflections, not here. ' +
  'This is your judgement to make, not a rule that will stop you: she would rather read something real that turns out to be minor ' +
  'than have you sit on something that mattered.';

const UNREAD_NOTE =
  'An unread count is not a verdict on what you sent: it means she has been busy, nothing more. ' +
  'It never expires and nothing here holds it against you.';

class ConversationListTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'read';
    this.name = 'conversation_list';
    this.description =
      'List the conversations in Ellie\'s sidebar, with how many of your messages in each she has not read yet. ' +
      'Includes conversations SHE started as well as ones you opened — you can add to any active one. ' +
      'Call this BEFORE opening a new conversation: if one on this subject is already active, add to it instead of starting a second. ' +
      UNREAD_NOTE;
    this.parameters = {
      type: 'object',
      properties: {
        status: {
          type: 'string', enum: ['active', 'archived'],
          description: 'Only conversations in this state. Omit for active. Archived ones are closed to writes for both of you and still readable — worth checking for "I raised this before".'
        },
        subject: {
          type: 'string',
          description: 'Only conversations whose title contains this. Use it to find whether you have already raised something.'
        }
      },
      required: []
    };
  }

  async execute(args = {}) {
    const status = args.status === 'archived' ? 'archived' : 'active';
    const rows = channel.listConversations({ status, limit: 100 })
      .filter(c => !args.subject || String(c.title || '').toLowerCase().includes(String(args.subject).toLowerCase()));
    return {
      count: rows.length,
      unread_total: channel.totalUnread(),
      conversations: rows.map(c => ({
        id: c.id,
        title: c.title || '(untitled)',
        status: c.status,
        started_by: c.initiated_by === 'snh' ? 'you' : 'Ellie',
        messages: c.message_count,
        unread: c.unread,
        last_at: c.updated_at,
        retire_requested: !!c.retire_requested_at,
        follows_archived_conversation: c.supersedes_conversation_id || undefined
      })),
      note: rows.length === 0
        ? (status === 'archived'
          ? 'Nothing archived yet.'
          : 'No active conversations. The next one you open is the first thing she sees in the list.')
        : 'You may write into any of these that is active — hers as well as your own.'
    };
  }
}

class ConversationSendTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'conversation_send';
    this.description =
      'Say something into a conversation that is already open — yours or one Ellie started. ' +
      'It appears as your turn in that conversation, where she reads and replies. She sees it as unread on the list until she opens it. ' +
      'Prefer this over opening a new conversation whenever the subject already has one. ' +
      'Archived conversations are closed to writes for both of you. ' +
      THRESHOLD_GUIDANCE;
    this.parameters = {
      type: 'object',
      properties: {
        conversation_id: { type: 'string', description: 'The conversation to write into. Get it from conversation_list.' },
        body: { type: 'string', description: 'What you want to say to her. Write it as yourself, to her.' }
      },
      required: ['conversation_id', 'body']
    };
  }

  async execute(args = {}) {
    const body = String(args.body || '').trim();
    if (!body) return { sent: false, error: 'a message needs a body' };
    try {
      const before = channel.getState(args.conversation_id);
      if (!before) return { sent: false, error: 'no conversation by that id — call conversation_list to see them' };
      if (before.status !== 'active') {
        return {
          sent: false,
          error: `that conversation is archived and closed to writes for both of you. If the subject has come back, use conversation_open and pass follows_conversation_id: "${before.id}".`
        };
      }
      const after = channel.sendInto(before.id, body, { sourceKind: 'tool', sourceRef: 'conversation_send' });
      return {
        sent: true,
        conversation_id: after.id,
        title: after.title,
        started_by: after.initiated_by === 'snh' ? 'you' : 'Ellie',
        messages: after.message_count,
        unread_for_her: after.unread,
        note: 'It is in her list now, with an unread count on it. ' + UNREAD_NOTE
      };
    } catch (err) {
      return { sent: false, error: err.message };
    }
  }
}

class ConversationOpenTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'conversation_open';
    this.description =
      'Open a NEW conversation with Ellie. It appears in her sidebar marked as yours, the way your unprompted ones already do, and stays there unread until she opens it. ' +
      'Only for a subject that does not already have an active conversation — check conversation_list first and use conversation_send if it does. ' +
      THRESHOLD_GUIDANCE;
    this.parameters = {
      type: 'object',
      properties: {
        body: { type: 'string', description: 'The first thing you want to say. Write it as yourself, to her.' },
        title: { type: 'string', description: 'A short title she can recognise it by in the list. A few words.' },
        follows_conversation_id: {
          type: 'string',
          description: 'When a subject comes back after its conversation was archived, the id of that archived one — so the new conversation shows you raised it before.'
        }
      },
      required: ['body']
    };
  }

  async execute(args = {}) {
    const body = String(args.body || '').trim();
    if (!body) return { opened: false, error: 'a conversation needs a first message' };
    try {
      const c = channel.openConversation({
        title: args.title, body,
        supersedesConversationId: args.follows_conversation_id || null,
        sourceKind: 'tool', sourceRef: 'conversation_open'
      });
      return {
        opened: true, conversation_id: c.id, title: c.title,
        note: 'It is at the top of her list with 1 unread. ' + UNREAD_NOTE
      };
    } catch (err) {
      return { opened: false, error: err.message };
    }
  }
}

class ConversationRequestRetireTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'conversation_request_retire';
    this.description =
      'Ask Ellie to archive a conversation you think is finished. You cannot archive one yourself — it goes to her as an approval. ' +
      'Archiving means neither of you can add to it again, though you can both still read it forever, and it moves to the Archive tab in her sidebar. ' +
      'Ask when the thing is settled, not when it has gone quiet: unread is not the same as done.';
    this.parameters = {
      type: 'object',
      properties: {
        conversation_id: { type: 'string', description: 'The conversation to archive. From conversation_list.' },
        reason: { type: 'string', description: 'Why you think it is finished. She sees this with the request.' }
      },
      required: ['conversation_id']
    };
  }

  async execute(args = {}) {
    try {
      const c = channel.getState(args.conversation_id);
      if (!c) return { requested: false, error: 'no conversation by that id' };
      if (c.status !== 'active') return { requested: false, error: 'that conversation is already archived' };
      if (c.retire_requested_at) return { requested: false, error: 'you have already asked about this one; it is waiting on her' };
      const after = await channel.requestRetire(c.id, args.reason);
      return {
        requested: true, conversation_id: c.id, title: c.title,
        note: 'Asked. It stays active and readable until she decides — nothing is archived by asking.',
        requested_at: after.retire_requested_at
      };
    } catch (err) {
      return { requested: false, error: err.message };
    }
  }
}

module.exports = {
  ConversationListTool, ConversationSendTool,
  ConversationOpenTool, ConversationRequestRetireTool
};
