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
 * SHE MAY NAME THE MECHANISM, AND THEN IT IS NOT A JUDGEMENT CALL (2026-08-18).
 * "Use and agent and write me a python script for a calculator": tier 1 fired,
 * the guidance block fired, this tool was fourth of eleven in the payload — and
 * the model returned no tool calls and the sentence "I have started a background
 * job to write a Python calculator script." Two things were missing and both are
 * fixed here: the description was written entirely about RESEARCH, so a request
 * to produce something matched nothing in it; and it never said that her naming
 * an agent settles the question. The server also forces the call on that signal
 * and backstops it after the reply, because a tool description is an argument and
 * an argument can be lost.
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
    // WHEN, IN OBSERVABLE TERMS — rewritten 2026-08-18 after it failed live.
    //
    // The first version described mechanics and prohibitions and never said
    // when. Every reason to delegate was self-assessed ("more digging than she
    // should sit and wait for"), while the reason not to was concrete and always
    // satisfiable — it CAN answer now, it has web_search in the same tool list.
    // Two research prompts that explicitly granted time were answered inline
    // instead, in 10s and 82s, and no job was ever queued.
    //
    // So: the triggers are things he can observe in her message or count on his
    // fingers, the trade is stated as a gain rather than a cost, and the "do not
    // punt" guard is BOTH/AND rather than a reason to stay inline — because the
    // failure it prevents also happened live: handed a job, he told her he could
    // not answer this turn and gave her nothing, while knowing plenty already.
    this.description =
      'Hand a piece of work to a background agent and carry on talking. ' +
      'IF SHE ASKS FOR AN AGENT, CALL THIS. "Use an agent", "send an agent", "start a background job" — ' +
      'that is her decision already made, and it does not matter what the work is: research, a writeup, ' +
      'a script, a plan. Do not weigh it up and do not answer inline instead. ' +
      'ALSO USE IT WHEN ANY OF THESE IS TRUE: she has said she is not waiting — "take your time", ' +
      '"take as long as you need", "I\'ll keep chatting", "while I\'m out"; or the answer needs ' +
      'more than about two searches, or several sources compared against each other; or the work ' +
      'would take you more than a minute. ' +
      'WHAT YOU GAIN: the agent can run a dozen searches and read whole pages. Answering inline ' +
      'gets you two or three searches and the snippets around them. For anything needing current ' +
      'detail from several places, the handed-off version is simply the better answer. ' +
      'IT CAN PRODUCE AS WELL AS LOOK UP: a run can write the thing she asked for — a script, a draft, ' +
      'a writeup — out of what it knows, and use its tools to check the facts that thing depends on. ' +
      'It cannot RUN what it writes; nothing in a job executes code. ' +
      'BOTH, NOT EITHER: starting a job never means saying nothing now. Tell her what you already ' +
      'know from memory and training in this same turn — she asked you, not the agent — and hand ' +
      'off only the part that needs real digging. A turn that just says "I will come back to you" ' +
      'has given her nothing, and that is a failure whether or not the job succeeds. ' +
      'A single lookup — one search, one fact — is not a job: answer that yourself. ' +
      'Calling this does NOT return the answer to you: it starts the work and returns a job id. ' +
      'The result goes to her jobs panel, which does not notify her and never opens a conversation; ' +
      'you are told what it found at the start of a later reply, and that is when you tell her.';
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
        'NOW ANSWER HER: say you have started it and what it will cover, and then tell her what you ' +
        'already know about this from memory and training. Do not end the turn having said only that ' +
        'you will come back to her — that leaves her with nothing, and she asked you. It lands in her ' +
        'jobs panel when it is done — it will not message her — and you are told about it at the start ' +
        'of a later reply, which is when you tell her what it found.'
    };
  }
}

module.exports = StartBackgroundJobTool;
