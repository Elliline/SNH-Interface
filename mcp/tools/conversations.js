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
 * BOTH BARS ARE GUIDANCE, NOT VALIDATORS, and that is deliberate. Athena
 * asked for a bar — concrete fact ids or a concrete pair, a specific question,
 * and a record of what was checked before raising — and it is written into the
 * descriptions rather than enforced in code. A validator would decide FOR her
 * which of its thoughts are worth her time, and the whole reason this channel
 * exists is that something it sent was filed as not needing a conversation.
 * She decides. The bar shapes the judgement; it does not gate the send.
 *
 * The stand-alone bar added on 2026-09-02 sits here on the same terms. It IS
 * enforced on the automatic follow-up writer in db/initiative-engine.js, and
 * the difference is not inconsistency: nothing is in that loop, so nothing
 * catches what it writes before Ellie does. Here, the entity is the judgement.
 */

const channel = require('../../db/conversation-channel');
const { getConfig } = require('../../db/config');
const standards = require('../../db/message-standards');

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

/**
 * THE THRESHOLD DECIDES WHETHER IT IS A MESSAGE. THIS DECIDES HOW IT READS.
 *
 * Added 2026-09-02, after a morning in which four messages reached her that
 * she could not read at all — not because they were thin, but because every
 * one of them was written from inside the hour of thinking that produced it.
 *
 * The two bars appear to conflict on one point and do not. The threshold says
 * NAME THE FACT IDS; this says KEEP THEM OUT OF THE TEXT. Both hold, because
 * they are about different things: having the ids is what earns the send, and
 * printing them is what makes the message unreadable. They go in the record
 * behind it — the ledger row this call writes already carries them.
 *
 * Guidance, like the threshold, and for the same reason: you have read the
 * conversation and you are making a judgement. Nothing here will stop a send.
 */
const STANDALONE_GUIDANCE =
  standards.STANDALONE_BAR + '\n' +
  'This does NOT contradict the bar above. Having the concrete thing is what earns the send; ' +
  'printing it is what makes the message unreadable. The ids and the receipts are already in the record ' +
  'this call writes — leave them there.\n' +
  standards.PROVENANCE_RULE;

/**
 * The worked pair goes on conversation_send ONLY, and that is a cost decision.
 *
 * Tool schemas are attached to every chat turn — 3,432 tokens across eleven
 * tools, measured — so a description is not free the way a comment is. The
 * example is the most instructive part of the bar and the cheapest way to
 * show what "written both ways" means, but it is ~1,400 characters, and
 * putting it on both send and open would pay for it twice on every turn for
 * one lesson. Sending is the far more common call, and open carries the same
 * rules in prose above.
 */
const STANDALONE_GUIDANCE_WITH_EXAMPLE = STANDALONE_GUIDANCE + '\n' + standards.WORKED_EXAMPLE;

const UNREAD_NOTE =
  'An unread count is not a verdict on what you sent: it means she has been busy, nothing more. ' +
  'It never expires and nothing here holds it against you.';

/**
 * WHOSE UNREAD IT IS, SAID IN THE FIELD NAME — because a bare `unread` was read
 * backwards, in the direction that costs her something.
 *
 * 2026-09-02, 16:58: Athena called conversation_list, saw `unread: 3`, and
 * opened an unprompted conversation saying "just a quick nudge that I've got
 * three of your messages waiting in the queue." Nothing was waiting. Asked what
 * they were, she listed three of ELLIE'S OWN messages back at her — ones she
 * had already answered. Ellie: "nothing's waiting, those were mine and you
 * answered them — the count you saw was my unread of yours, not the other way."
 *
 * There is exactly ONE reading state in this system and it is Ellie's: the
 * watermark on `conversations` records what SHE has opened. Nothing anywhere
 * tracks what the entity has or has not seen, because the entity sees a
 * conversation by taking a turn in it. So the honest fix is not a better
 * adjective on an ambiguous number — it is to name the direction in the field
 * itself, and to say plainly that the count in the other direction does not
 * exist rather than leaving a gap for it to be inferred into.
 */
const UNREAD_DIRECTION_NOTE =
  'WHOSE UNREAD THIS IS: every count here is ELLIE\'S reading state — messages YOU sent that SHE has not opened yet. ' +
  'None of it is a queue of things waiting on you. Nothing in this system tracks what you have seen, ' +
  'so a number here can never mean "she is waiting for a reply from me".';

class ConversationListTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'read';
    this.name = 'conversation_list';
    this.description =
      'List the conversations in Ellie\'s sidebar, with how many of your messages in each she has not read yet. ' +
      'Includes conversations SHE started as well as ones you opened — you can add to any active one. ' +
      'Call this BEFORE opening a new conversation: if one on this subject is already active, add to it instead of starting a second. ' +
      UNREAD_DIRECTION_NOTE + ' ' +
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
    const total = channel.totalUnread();
    return {
      count: rows.length,
      // NAMED, NOT BARE. `unread_total` was read as "things waiting on me".
      your_messages_ellie_has_not_read: total,
      // Stated as a fact rather than left as an absence, so it cannot be filled
      // in by inference the way the bare count was.
      messages_waiting_on_you: 'not tracked — nothing in this system counts what you have seen, so there is never a backlog here for you to work through',
      conversations: rows.map(c => ({
        id: c.id,
        title: c.title || '(untitled)',
        status: c.status,
        started_by: c.initiated_by === 'snh' ? 'you' : 'Ellie',
        messages: c.message_count,
        your_messages_ellie_has_not_read: c.unread,
        last_at: c.updated_at,
        retire_requested: !!c.retire_requested_at,
        follows_archived_conversation: c.supersedes_conversation_id || undefined
      })),
      note: rows.length === 0
        ? (status === 'archived'
          ? 'Nothing archived yet.'
          : 'No active conversations. The next one you open is the first thing she sees in the list.')
        : 'You may write into any of these that is active — hers as well as your own. ' +
          (total > 0
            ? `The ${total} unread ${total === 1 ? 'message is one YOU sent' : 'messages are ones YOU sent'} that she has not opened — not anything of hers awaiting you.`
            : 'She has opened everything you have sent.')
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
      THRESHOLD_GUIDANCE + '\n' + STANDALONE_GUIDANCE_WITH_EXAMPLE;
    this.parameters = {
      type: 'object',
      properties: {
        conversation_id: { type: 'string', description: 'The conversation to write into. Get it from conversation_list.' },
        body: { type: 'string', description: 'What you want to say to her. Write it as yourself, to her — and for someone who has not seen anything you did to arrive at it.' }
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
        // Already directional, and it stays that way — this is the naming
        // conversation_list should have had from the start.
        your_messages_ellie_has_not_read: after.unread,
        note: 'It is in her list now, with an unread count on it — YOUR messages she has not opened, never anything of hers awaiting you. ' + UNREAD_NOTE
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
      THRESHOLD_GUIDANCE + '\n' + STANDALONE_GUIDANCE;
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
      'Ask Ellie to archive a conversation SHE started that you think is finished — it goes to her as an approval. ' +
      '(One YOU opened you may close yourself with conversation_archive.) ' +
      'Archiving means neither of you can add to it again, though you can both still read it forever, and it moves to the Archive tab in her sidebar. ' +
      'Ask when the thing is settled, not when it has gone quiet: unread is not the same as done. ' +
      'ONE AT A TIME ONLY. If she asks you to go through your open conversations, use review_conversations instead — that runs in the background, one conversation at a time, and does not lose the work if the turn dies.';
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

/**
 * THE ENTITY CLOSES ONE OF ITS OWN (Ellie's call, 2026-09-10).
 *
 * Only a conversation it STARTED — `initiated_by = 'snh'` on the row, a stored
 * fact and never the entity's own reading. Asked to close one of hers, the
 * channel turns the archive into a retirement request and the result says
 * so; nothing is refused into silence. A conversation something is still
 * using (a running or paused job, an ask waiting on her) cannot be closed by
 * anyone, and the refusal names what is using it.
 */
class ConversationArchiveTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'conversation_archive';
    this.description =
      'Archive a conversation YOU opened, on your own — no approval needed. It moves to her Archive tab, closed to writes for both of you, readable forever, and she can reopen it. ' +
      'Save anything worth keeping from it to memory FIRST: nothing else is saved when it closes. ' +
      'A conversation ELLIE started is hers to close — this tool turns that into a request to her and tells you so. ' +
      'One that still has work running in it (a job, an ask waiting on her) is refused with the reason. ' +
      'For going through many at once, use review_conversations.';
    this.parameters = {
      type: 'object',
      properties: {
        conversation_id: { type: 'string', description: 'The conversation to archive. From conversation_list.' },
        reason: { type: 'string', description: 'Why it is finished, in a sentence. She sees this in the record (and with the request, if it becomes one).' }
      },
      required: ['conversation_id']
    };
  }

  async execute(args = {}) {
    try {
      const c = (getConfig().tools && getConfig().tools.conversations) || {};
      const state = channel.getState(args.conversation_id);
      if (!state) return { archived: false, error: 'no conversation by that id — call conversation_list to see them' };
      if (state.status !== 'active') return { archived: false, error: 'it is already archived' };
      if (c.selfArchive === false) {
        const r = await channel.archiveBySelf(args.conversation_id, { reason: args.reason, forceRequest: true });
        return { archived: false, requested: r.requested, conversation_id: state.id, title: state.title,
          note: 'Closing your own conversations is switched off in Settings, so this went to her as a request instead.' };
      }
      const r = await channel.archiveBySelf(args.conversation_id, { reason: args.reason });
      return {
        archived: r.archived, requested: r.requested,
        conversation_id: state.id, title: state.title,
        started_by: state.initiated_by === 'snh' ? 'you' : 'Ellie',
        note: r.note
      };
    } catch (err) {
      return { archived: false, error: err.message };
    }
  }
}

/**
 * THE REVIEW, DISPATCHED. When she asks the entity to go through its open
 * conversations, the work is a background job — one conversation at a time,
 * memory saved before anything closes, checkpointed, resumable, and reported
 * back to her in the conversation she asked in. See db/conversation-review.js
 * for why it is not one chat turn any more.
 */
class ReviewConversationsTool extends BaseConversationTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'review_conversations';
    this.description =
      'Go through ALL your open conversations in the background — use this whenever Ellie asks you to look at, review, tidy or close your open conversations. ' +
      'It runs as a job, one conversation at a time: reads it, decides whether it is finished (your judgement), saves anything worth keeping to memory, ' +
      'and only then closes it (if you opened it) or asks her to (if she did). Anything unfinished or still in use is left alone. ' +
      'When it is done it sends her one message in this conversation listing what was closed and what is waiting on her. ' +
      'Do NOT do the review by hand in this turn — that is how a 31-minute turn was lost. Call this, then tell her it is underway and will report back here.';
    this.parameters = {
      type: 'object',
      properties: {
        why: { type: 'string', description: 'One line on what she asked for, in your words.' }
      },
      required: []
    };
  }

  async execute(args = {}, context = {}) {
    try {
      const review = require('../../db/conversation-review');
      const r = review.enqueueReview({ conversationId: context.conversationId || null, messageId: context.messageId || null, why: args.why || null });
      if (!r.ok) return { started: false, error: r.error };
      return {
        started: true, job_id: r.id, short_id: String(r.id).slice(0, 8),
        message: 'Started — not finished. It works through the conversations one at a time in the background and will send Ellie one message HERE when it is done, ' +
          'saying which it closed and which are waiting on her. Tell her that plainly now; do not describe conversations as closed until that message exists.'
      };
    } catch (err) {
      return { started: false, error: err.message };
    }
  }
}

module.exports = {
  ConversationListTool, ConversationSendTool,
  ConversationOpenTool, ConversationRequestRetireTool,
  ConversationArchiveTool, ReviewConversationsTool
};
