/**
 * history_search — the entity's own conversation archive, read by proxy.
 *
 * Three tools in one file because they are one capability: the chat-facing one
 * the entity calls, and the two the background run uses to do the digging. They
 * share a config flag, a backing module (db/history-search.js) and a contract,
 * and splitting them would put three copies of "this only ever reads" in three
 * places to drift apart. It is the same call mcp/tools/memory-inspect.js makes.
 *
 * TIER. All three are `read`. Nothing here writes, proposes, or acts — every
 * statement is a SELECT. The rate cap on the chat-facing one is not a safety
 * bound; it is the same shared read budget the memory-inspect tools spend, for
 * the same reason: a model that CAN look things up will look up forty things.
 *
 * WHY THE OTHER TWO ARE BACKGROUND-ONLY. history_find and history_read are the
 * archaeology, and the archaeology is precisely what must not happen in the
 * chat turn — a dozen hits and five read windows is most of a context window
 * spent on material that gets discarded. Offering them in chat would give the
 * entity a cheaper-looking way to do the expensive thing, and it would take it.
 * They are marked backgroundOnly, so MCPClient.getToolsForOpenAI() drops them
 * from every schema a conversation is ever handed — the same structural
 * absence the corrector's write tools have.
 *
 * DESCRIPTIONS ARE BARE IMPERATIVES, measured on this brain: a hedged tool
 * description scores near zero on selection. Name the trigger, say call it.
 */

const historySearch = require('../../db/history-search');
const memoryInspect = require('../../db/memory-inspect');
const { getConfig } = require('../../db/config');

class BaseHistoryTool {
  constructor() {
    this.tier = 'read';
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
  }

  getTierMetadata() {
    return {
      name: this.name,
      tier: this.tier,
      reversible: this.reversible,
      requiresApproval: this.requiresApproval,
      destructive: this.destructive,
      rateCaps: this.rateCaps || null
    };
  }

  getOpenAIFunctionSpec() {
    return {
      type: 'function',
      function: { name: this.name, description: this.description, parameters: this.parameters }
    };
  }
}

/**
 * THE ONE THE ENTITY CALLS.
 *
 * The description has to do two jobs at once and they pull against each other:
 * make it call this instead of answering from impression, and stop it calling
 * this for things that are in front of it already. So the trigger is written as
 * something observable in her message — she is asking about a PAST conversation
 * — and the cost is stated plainly, because a tool that takes a minute and is
 * described as free gets called for everything.
 */
class HistorySearchTool extends BaseHistoryTool {
  constructor() {
    super();
    this.name = 'history_search';
    this.description =
      'Search your past conversations with Ellie and get back what was actually said, in quotes. ' +
      'CALL THIS WHENEVER SHE ASKS ABOUT AN EARLIER CONVERSATION — "what did that script do", ' +
      '"what did we decide about X", "when did I tell you about Y", "remind me what you said about Z" — ' +
      'and whenever answering would mean recalling a specific exchange rather than a fact you hold. ' +
      'YOUR MEMORY IS FACTS, NOT TRANSCRIPTS: memory_search finds what you know, this finds what was said. ' +
      'If she is asking what was SAID, this is the tool. ' +
      'A background agent does the digging on its own budget, so this costs you a wait and a short digest ' +
      'rather than your context. ' +
      'WHAT COMES BACK: verbatim quotes, each with the conversation and timestamp it came from, plus a ' +
      'short framing. Quotes are checked against the database before you see them. ' +
      'IT WILL SOMETIMES SAY IT FOUND NOTHING, and that is a real result — it means there is no record, ' +
      'and you tell her that. Never fill in what a conversation probably said. ' +
      'It takes up to a minute and it counts against your hourly limit on looking things up, so use it ' +
      'for a real question about the past, not to double-check something already in front of you.';
    this.parameters = {
      type: 'object',
      properties: {
        question: {
          type: 'string',
          description:
            'The question, in your own words, as a full question — "what did the script for Lincoln City ' +
            'Animal Clinic do?", "what did we decide about the backup schedule?". Include the distinctive ' +
            'names and terms; the agent searches on those. Not a bare keyword.'
        }
      },
      required: ['question']
    };
  }

  get rateCaps() {
    const c = (getConfig().tools && getConfig().tools.memoryInspect) || {};
    return { maxPerHour: c.maxCallsPerHour ?? 40, shared: 'all memory read tools' };
  }

  /**
   * Cap, log, run, log. Every path through here writes exactly one
   * tool_call_log row — including the refusals, because a search that was
   * refused and a search that found nothing are indistinguishable from the
   * answer, and only the log can tell them apart.
   */
  async execute(args, context = {}) {
    const conversationId = context.conversationId || null;
    const log = (outcome, detail, refId = null) => {
      try {
        require('../../db/cron-jobs').logToolCall({
          tool: 'history_search', args: args || {}, outcome, detail, refId, conversationId
        });
      } catch (e) { console.error('[HistorySearch] logToolCall failed:', e.message); }
    };

    if (!historySearch.cfg().enabled) {
      log('error', 'tool disabled in config');
      return { error: 'Conversation-history search is switched off.' };
    }

    // The SHARED read budget, not one of its own — see CAP_TOOLS in
    // db/memory-inspect.js for why there is only ever one counter for this.
    const cap = memoryInspect.checkCap();
    if (!cap.ok) {
      log('rejected-cap', cap.reason);
      return {
        error: `Not searched — ${cap.reason}. Say plainly that you have hit your own limit for looking ` +
          `things up, rather than answering from impression as though you had checked.`
      };
    }

    let result;
    try {
      result = await historySearch.ask({
        question: args && args.question,
        conversationId,
        messageId: context.messageId || null
      });
    } catch (err) {
      console.error('[HistorySearch] ask failed:', err.message);
      log('error', err.message);
      return { error: `The history search failed to start: ${err.message}. You have nothing from your history — say so.` };
    }

    if (!result.ok) {
      log('error', String(result.error).slice(0, 200));
      return { error: result.error };
    }

    log('read',
      `${result.status} — ${result.verified || 0} quote(s) verified, ${result.rejected || 0} rejected`,
      result.job_id);
    console.log(`[HistorySearch] history_search: ${result.status}, ${result.verified || 0} verified quote(s) (${cap.used + 1}/${cap.maxCallsPerHour} reads this hour)`);

    return {
      status: result.status,
      job_id: result.short_id,
      digest: result.digest,
      // Said to it as well as enforced in the digest text, because this is the
      // sentence that decides whether the digest becomes an honest answer or a
      // seed for a plausible one.
      how_to_use:
        'The quotes are the evidence and they are verbatim. Quote or paraphrase them freely, and say which ' +
        'conversation something came from if it matters. Anything NOT in the digest is not something you ' +
        'found — if it says nothing was found, tell her that plainly rather than reconstructing the answer.'
    };
  }
}

/** Shared by the two background tools: they run inside a job, never in chat. */
class BackgroundHistoryTool extends BaseHistoryTool {
  constructor() {
    super();
    this.backgroundOnly = true;
  }
}

class HistoryFindTool extends BackgroundHistoryTool {
  constructor() {
    super();
    this.name = 'history_find';
    this.description =
      'Search every message in your conversations with Ellie and get ranked hits back. ' +
      'Search on the distinctive words — names, project names, technical terms — not on the filler. ' +
      'Each hit gives you a message id, the conversation it is in, when it was said, and a snippet. ' +
      'The snippet is a locator, not a quote: read the message with history_read before you quote anything. ' +
      'If a search comes back thin, try the same thing worded differently before concluding there is nothing.';
    this.parameters = {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The words to look for. "Lincoln City Animal Clinic script", "backup schedule decision".' },
        limit: { type: 'integer', description: 'How many hits. Default 12, which is also the maximum.' }
      },
      required: ['query']
    };
  }

  async execute(args, context = {}) {
    const result = historySearch.find({ query: args && args.query, limit: args && args.limit });
    if (!result.error) {
      historySearch.noteRead(context.caller, { hits: result.returned });
    }
    return result;
  }
}

class HistoryReadTool extends BackgroundHistoryTool {
  constructor() {
    super();
    this.name = 'history_read';
    this.description =
      'Read the messages around one search hit, so you can see the exchange and copy an exact quote. ' +
      'Call this on every hit you intend to quote from — quoting a snippet is how a quote comes out wrong, ' +
      'and a wrong quote is thrown away. Returns the message plus the ones before and after it. ' +
      'Long messages come back shortened; a quote must be copied from the text you were actually given.';
    this.parameters = {
      type: 'object',
      properties: {
        message_id: { type: 'string', description: 'The message id from a history_find hit, in full as it was given to you.' },
        before: { type: 'integer', description: 'How many messages before it. Default 1, maximum 4.' },
        after: { type: 'integer', description: 'How many messages after it. Default 2, maximum 4 — an answer usually follows the question.' }
      },
      required: ['message_id']
    };
  }

  async execute(args, context = {}) {
    const result = historySearch.readAround({
      message_id: args && args.message_id,
      before: args && args.before,
      after: args && args.after
    });
    if (!result.error) {
      historySearch.noteRead(context.caller, { messages: result.returned });
    }
    return result;
  }
}

module.exports = { HistorySearchTool, HistoryFindTool, HistoryReadTool };
