/**
 * write_memory — the second action tool, and the first that DIRECT-EXECUTES.
 *
 * create_cron_job proposes because it acts on the world outside the
 * conversation. This one records something Ellie just said and asked to have
 * recorded, so an approval queue would defeat the point: the whole reason it
 * exists is that "write this to your memory" used to produce agreement and no
 * write. It happens in the moment or it hasn't solved anything.
 *
 * Bounded the same way everything else here is: reversible tier (supersede-
 * never-delete, and the returned fact id can be retired), rate-capped from the
 * DB so a restart grants no fresh budget, and every call logged to
 * tool_call_log — including the refusals.
 *
 * The tool itself is deliberately thin. It hands the statement to
 * db/memory-write.js, which dispatches the classifier agent that decides who the
 * fact is about, whether it replaces something, and how much it matters.
 */

const memoryWrite = require('../../db/memory-write');
const { getConfig } = require('../../db/config');

class WriteMemoryTool {
  constructor() {
    this.name = 'write_memory';
    this.description =
      'Save something to your long-term memory when you are asked to remember it. ' +
      'Use this the moment Ellie says "remember this", "write this to your memory", ' +
      '"don\'t forget", or corrects something you already hold. It writes immediately — ' +
      'it is not a proposal. It works out on its own whether the fact is about her or ' +
      'about you, and whether it replaces something you already believe, so pass her ' +
      'statement through plainly rather than reformatting it.';
    this.parameters = {
      type: 'object',
      properties: {
        statement: {
          type: 'string',
          description:
            'What to remember, copied from her message as close to VERBATIM as you can. ' +
            'Do not change the pronouns. "Remember that YOU prefer X" is a fact about ' +
            'YOU and must stay "you prefer X" — rewriting it as "I prefer X" turns it ' +
            'into a fact about HER and files it under her preferences, which is wrong. ' +
            'Likewise leave her "I"/"my" alone; do not convert it to "you" or "the user". ' +
            'Copy the sentence; do not translate its point of view.'
        },
        context: {
          type: 'string',
          description:
            'Optional. A line or two of the surrounding conversation, if the statement ' +
            'needs it to make sense on its own (e.g. it uses "it" or "that").'
        }
      },
      required: ['statement']
    };

    // ---- tier metadata ----
    this.tier = 'reversible';
    this.reversible = true;
    this.requiresApproval = false;   // direct-execute, unlike create_cron_job
    this.destructive = false;
  }

  /** Rate caps read live from config so declared metadata never drifts from behavior. */
  get rateCaps() {
    const c = (getConfig().tools && getConfig().tools.memoryWrite) || {};
    return { maxPerHour: c.maxWritesPerHour ?? 20 };
  }

  getTierMetadata() {
    return {
      name: this.name,
      tier: this.tier,
      reversible: this.reversible,
      requiresApproval: this.requiresApproval,
      destructive: this.destructive,
      rateCaps: this.rateCaps
    };
  }

  getOpenAIFunctionSpec() {
    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.description,
        parameters: this.parameters
      }
    };
  }

  /**
   * @param {Object} args - { statement, context? }
   * @param {Object} context - { conversationId }
   */
  async execute(args, context = {}) {
    const result = await memoryWrite.write({
      statement: args?.statement,
      context: args?.context || '',
      conversationId: context.conversationId || null,
      // The verbatim user message, supplied by the chat path rather than by the
      // model. This is what the subject classifier actually trusts — see the
      // note on toolContext in server.js.
      userMessage: context.userMessage || '',
      messageId: context.messageId || null,
      inputModality: context.inputModality || null
    });

    // An identity-lock refusal is returned as its own status, not folded into
    // the generic error. Nothing was written and the model must SAY so — a lock
    // that produces a shrug and a warm "I've updated that" is the phantom-action
    // bug wearing a new hat.
    if (!result.ok && result.locked) {
      return {
        status: 'refused_locked',
        locked_category: result.category,
        written: false,
        message: result.error
      };
    }
    if (!result.ok) return { error: result.error };

    // Tell the model exactly what happened so it reports it truthfully — in
    // particular whether this REPLACED a belief, which is the part it would
    // otherwise gloss as "saved".
    const about = result.subject === 'self' ? 'yourself' : 'Ellie';

    if (result.unchanged) {
      return {
        status: 'already_held',
        fact_id: result.memberId,
        stored_as: result.fact,
        about,
        message: `Nothing to do — you already hold exactly this, and it is a locked identity fact. Say that you already have it and that it is unchanged.`
      };
    }

    return {
      status: 'saved',
      fact_id: result.memberId,
      // What the row NOW reads, which differs from what was asked for when a
      // supersession carried the old fact's uncontested assertions across.
      stored_as: result.fact,
      requested_as: result.requestedFact || result.fact,
      carried_over: result.carriedOver || null,
      about,
      salience: result.salience,
      replaced: result.superseded || null,
      locked: result.lockedCategories || null,
      message: (result.carriedOver
        ? `Saved to long-term memory as a fact about ${about}. It replaced what you held before ("${result.superseded}"), and rather than dropping that fact's other details they were carried into this one, which now reads: "${result.carriedOver}". Tell her you have updated it, what it replaced, and that nothing from the old one was lost.`
        : result.superseded
        ? `Saved to long-term memory as a fact about ${about}, replacing what you held before: "${result.superseded}". Tell her you have updated it, and say what it replaced.`
        : `Saved to long-term memory as a fact about ${about}. It is stored now, not pending — you can say so plainly.`)
        + (result.lockedCategories
          ? ` This set your ${result.lockedCategories.join(' and ')} for the first time, so it is now LOCKED — say that too: it can't be changed from a conversation any more, only through the Self tab or the identity-lock script.`
          : '')
    };
  }
}

module.exports = WriteMemoryTool;
