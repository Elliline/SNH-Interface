/**
 * start_background_job — the handoff.
 *
 * The first tool that does NOT block the turn. Every other tool he has runs
 * inside the seconds Ellie is sitting there waiting, which made the turn the
 * unit of work and put anything larger than a turn out of reach. This one writes
 * a row, hands the run to the agent pool, and returns a job id immediately. The
 * turn ends normally; the work carries on without it.
 *
 * WHAT THE RESULT DOES NOT DO — and the tool says so to his face, because this
 * is the rule he should know as a limit rather than discover: the result lands
 * in Ellie's jobs panel. It does not message her, does not open a conversation,
 * and does not interrupt her. If a finding turns out to be worth SAYING, that is
 * an ordinary decision he makes in an ordinary conversation, with the same
 * judgement as anything else he might raise.
 *
 * TWO PHANTOM-ACTION GUARDS, in opposite directions, in one tool:
 *   - He must not claim a job he did not start. The row is written before this
 *     returns, so "I started it" is checkable, and every refusal comes back as a
 *     reason in his hands rather than as silence.
 *   - He must not claim a RESULT he does not have. The return message says, in
 *     as many words, that he does not have one yet and must not describe what it
 *     will say.
 *
 * DIRECT-EXECUTE, not propose-only, unlike create_cron_job. Starting a read-only
 * background lookup is not a decision that needs Ellie's approval — it changes
 * nothing, costs a few minutes of a GPU that is already his to use, and the
 * caps live on the queue. What needed her approval about a cron job was that it
 * would recur forever; this happens once and stops.
 */

const agentJobs = require('../../db/agent-jobs');
const { getConfig } = require('../../db/config');

class StartBackgroundJobTool {
  constructor() {
    this.name = 'start_background_job';
    this.description =
      'Start a piece of work in the background and come back to the conversation right away. ' +
      'Use it when answering properly would need more digging than she should sit and wait for — ' +
      'several searches, a sweep through memory, anything you would otherwise apologise for the length of. ' +
      'Calling this does NOT give you the answer: it starts the work and returns a job id, and the result ' +
      'goes to her jobs panel, which does not notify her and never opens a conversation. ' +
      'Do not use it for something you can simply answer now, and do not use it as a way to promise her a ' +
      'message later. Starting a job does not excuse you from answering what you can in this turn.';
    this.parameters = {
      type: 'object',
      properties: {
        title: {
          type: 'string',
          description: 'Short label for her jobs panel, a few words, e.g. "check what changed in memory this week".'
        },
        task: {
          type: 'string',
          description:
            'The full instruction the background run will follow. It IS the prompt — the run sees this and ' +
            'not the conversation, so write it so it stands alone: what to look at, what to look for, and ' +
            'what period it covers if that matters.'
        },
        why: {
          type: 'string',
          description: 'One line on why this is worth doing away from the conversation rather than answering now.'
        }
      },
      required: ['title', 'task']
    };

    // ---- tier metadata (same shape create_cron_job declares) ----
    this.tier = 'reversible';
    this.reversible = true;        // it reads; there is nothing to undo
    this.requiresApproval = false; // read-only and one-shot — see the header
    this.destructive = false;
  }

  /** Rate caps read live from config so the declared metadata never drifts from behavior. */
  get rateCaps() {
    const c = getConfig().agentJobs || {};
    return {
      maxPerHour: c.maxStartsPerHour ?? 6,
      maxConcurrent: c.maxConcurrent ?? 2,
      maxQueued: c.maxQueued ?? 10
    };
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
   * @param {Object} args - { title, task, why }
   * @param {Object} context - { conversationId, messageId }
   */
  async execute(args, context = {}) {
    const result = agentJobs.enqueue({
      title: args && args.title,
      task: args && args.task,
      why: args && args.why,
      conversationId: context.conversationId || null,
      messageId: context.messageId || null,
      source: 'chat-handoff'
    });

    if (!result.ok) return { error: result.error };

    return {
      status: 'started',
      job_id: result.id,
      short_id: result.id.slice(0, 8),
      message:
        'Started — not finished. The work is running in the background now and will keep running after ' +
        'this conversation ends. You do NOT have the result and must not describe what it will say. ' +
        'Tell her you have started it and roughly what it will cover. It lands in her jobs panel when it ' +
        'is done — it will not message her — and you will be told about it at the start of a later reply, ' +
        'which is when you can tell her what it found.'
    };
  }
}

module.exports = StartBackgroundJobTool;
