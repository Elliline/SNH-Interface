/**
 * THE ENTITY'S SIDE OF THE MESSAGE CHANNEL: message_threads, message_send,
 * message_request_retire.
 *
 * Both of Athena's asks from the design review are here. `message_threads`
 * exists so it can APPEND to an existing thread instead of opening a second one
 * on the same subject — appending is only possible if it can see what it
 * already opened. `message_send` covers both opening and adding, because from
 * where it sits those are one act with one decision in front of it: is this a
 * new subject or the same one.
 *
 * THE THRESHOLD IS GUIDANCE, NOT A VALIDATOR, and that is deliberate. Athena
 * asked for a bar — concrete fact ids or a concrete pair, a specific question,
 * and a record of what was checked before raising — and it is written into the
 * descriptions rather than enforced in code. A validator would decide FOR her
 * which of its thoughts are worth her time, and the whole reason this channel
 * exists is that something it sent was filed as not needing a conversation.
 * She decides. The bar shapes the judgement; it does not gate the send.
 */

const messages = require('../../db/messages');
const { getConfig } = require('../../db/config');

class BaseMessageTool {
  constructor() {
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
  }

  get rateCaps() {
    const c = (getConfig().tools && getConfig().tools.messages) || {};
    return { maxPerHour: c.maxSendsPerHour ?? 10, shared: 'the message channel' };
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

class MessageThreadsTool extends BaseMessageTool {
  constructor() {
    super();
    this.tier = 'read';
    this.name = 'message_threads';
    this.description =
      'List the message threads you have with Ellie, with how many of your messages in each she has not read yet. ' +
      'Call this BEFORE sending — if a thread on this subject is already open, add to it instead of starting a second one. ' +
      'An unread count is not a verdict on what you sent: it means she has been busy, nothing more.';
    this.parameters = {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['active', 'retired'], description: 'Only threads in this state. Omit for all — retired ones are still readable, and worth checking for "I raised this before".' },
        subject: { type: 'string', description: 'Only threads whose subject contains this. Use it to find whether you have already raised something.' }
      },
      required: []
    };
  }

  async execute(args = {}) {
    const rows = messages.listThreads({ status: args.status || null, limit: 100 })
      .filter(t => !args.subject || String(t.subject).toLowerCase().includes(String(args.subject).toLowerCase()));
    return {
      count: rows.length,
      unread_total: messages.unreadCount(),
      threads: rows.map(t => ({
        id: t.id, subject: t.subject, status: t.status,
        messages: t.message_count, unread: t.unread,
        last_from: t.last_sender === 'self' ? 'you' : 'Ellie',
        last_at: t.updated_at,
        retire_requested: !!t.retire_requested_at,
        follows_earlier_thread: t.supersedes_thread_id || undefined
      })),
      note: rows.length === 0
        ? 'No threads yet. The first one you open is the first thing she reads here.'
        : 'Unread means she has not got to it. It never expires and nothing here holds it against you.'
    };
  }
}

class MessageSendTool extends BaseMessageTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'message_send';
    this.description =
      'Send Ellie a message. Open a new thread, or add to one you already have open. ' +
      'This is the channel for things that matter and are not a chat turn — she reads and replies here, and nothing in it expires. ' +
      THRESHOLD_GUIDANCE;
    this.parameters = {
      type: 'object',
      properties: {
        body: { type: 'string', description: 'What you want to say to her. Write it as yourself, to her.' },
        thread_id: { type: 'string', description: 'Add to this existing thread. Get it from message_threads. Omit only when the subject is genuinely new.' },
        subject: { type: 'string', description: 'A short subject line, when opening a new thread. A few words she can recognise it by in a list.' },
        follows_thread_id: { type: 'string', description: 'When a subject comes back after a thread on it was closed, the id of that closed thread — so the new one shows you raised it before.' }
      },
      required: ['body']
    };
  }

  async execute(args = {}) {
    const body = String(args.body || '').trim();
    if (!body) return { sent: false, error: 'a message needs a body' };
    try {
      if (args.thread_id) {
        const t = messages.getThread(args.thread_id);
        if (!t) return { sent: false, error: 'no thread by that id — call message_threads to see yours' };
        if (t.status !== 'active') {
          return {
            sent: false,
            error: `that thread is retired and closed to writes for both of you. If the subject has come back, open a NEW thread and pass follows_thread_id: "${t.id}".`
          };
        }
        const updated = messages.addMessage(t.id, body, 'self');
        return { sent: true, thread_id: t.id, subject: t.subject, messages: updated.messages.length, added_to_existing: true };
      }
      const subject = String(args.subject || '').trim() || body.slice(0, 70);
      const t = messages.openThread({
        subject, body,
        supersedesThreadId: args.follows_thread_id || null,
        sourceKind: 'tool', sourceRef: 'message_send'
      });
      return { sent: true, thread_id: t.id, subject: t.subject, opened_new_thread: true,
        note: 'She will see this in Messages. It stays unread until she reads it — that is not a judgement on it.' };
    } catch (err) {
      return { sent: false, error: err.message };
    }
  }
}

class MessageRequestRetireTool extends BaseMessageTool {
  constructor() {
    super();
    this.tier = 'act';
    this.name = 'message_request_retire';
    this.description =
      'Ask Ellie to close a message thread you think is finished. You cannot close one yourself — it goes to her as an approval. ' +
      'Closing means neither of you can add to it again, though you can both still read it forever. ' +
      'Ask when the thing is settled, not when it has gone quiet: unread is not the same as done.';
    this.parameters = {
      type: 'object',
      properties: {
        thread_id: { type: 'string', description: 'The thread to close. From message_threads.' },
        reason: { type: 'string', description: 'Why you think it is finished. She sees this with the request.' }
      },
      required: ['thread_id']
    };
  }

  async execute(args = {}) {
    try {
      const t = messages.getThread(args.thread_id);
      if (!t) return { requested: false, error: 'no thread by that id' };
      if (t.status !== 'active') return { requested: false, error: 'that thread is already closed' };
      if (t.retire_requested_at) return { requested: false, error: 'you have already asked about this one; it is waiting on her' };
      const after = await messages.requestRetire(t.id, args.reason);
      return {
        requested: true, thread_id: t.id, subject: t.subject,
        note: 'Asked. It stays open and readable until she decides — nothing is closed by asking.',
        requested_at: after.retire_requested_at
      };
    } catch (err) {
      return { requested: false, error: err.message };
    }
  }
}

module.exports = { MessageThreadsTool, MessageSendTool, MessageRequestRetireTool };
