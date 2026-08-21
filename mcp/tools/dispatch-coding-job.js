/**
 * dispatch_coding_job — handing coding work to squatch-code.
 *
 * PROPOSE ONLY, like create_cron_job and unlike start_background_job, and the
 * difference is the whole reason this tool is separate from that one.
 * start_background_job argues its own case for skipping approval: "starting a
 * read-only background lookup is not a decision that needs Ellie's approval —
 * it changes nothing". This one changes files on her machine, unattended, with
 * nobody at squatch-code's approval prompt. So it asks first.
 *
 * ONE COARSE TOOL, NOT FILE OPERATIONS. squatch-code has its own agentic loop,
 * its own model and its own tools; what it wants is a brief, not a driver.
 * There is deliberately no read_file/write_file surface here — a tool layer
 * that drove it step by step would be a worse copy of the thing it is calling.
 *
 * THE BRIEF IS PROSE. No schema, no fields to fill in. She and the entity have
 * just talked the problem through in chat; the brief is him writing down what
 * they settled on, the way one person briefs another. A form would lose exactly
 * the context that makes the handoff worth anything.
 */

const codingJobs = require('../../db/coding-jobs');

class DispatchCodingJobTool {
  constructor() {
    this.name = 'dispatch_coding_job';
    this.description =
      'Send a piece of coding work to squatch-code, the local coding agent, to do ' +
      'on its own in one of the projects under Projects/. This does NOT start ' +
      'anything: it writes a brief and shows it to Ellie, and she approves, edits ' +
      'or rejects it. If she approves, squatch-code works unattended for a few ' +
      'minutes — it can edit files in that project and run test commands there — ' +
      'and the write-up lands in her jobs panel, not in this conversation. ' +
      'Use it when the two of you have settled on a concrete change to a project ' +
      'and she says to send it, hand it over, or get squatch-code to do it. ' +
      'Write the brief as prose, the way you would explain the job to a person: ' +
      'what needs doing and what it is for. It is the only instruction the run ' +
      'gets, so include what you both worked out, not just the last sentence she ' +
      'said. Do not use it to ask questions about code, and do not use it for ' +
      'work outside a project directory.';

    this.parameters = {
      type: 'object',
      properties: {
        project: {
          type: 'string',
          description:
            'The project name under Projects/, e.g. "todoapp". A name, not a path.'
        },
        brief: {
          type: 'string',
          description:
            'What needs doing, as prose. Everything the run will know — it cannot ' +
            'see this conversation. Say what to change, where, and why, and name ' +
            'anything you already worked out together about how.'
        }
      },
      required: ['project', 'brief']
    };

    // Tier metadata, declared the way create_cron_job declares its own.
    this.tier = 'action';
    // Reversible in the sense that matters: a git restore point is committed
    // inside the project before the run starts, and the report carries the
    // command that undoes the whole job. Not reversible if the project is not
    // a repository, which is why every project under Projects/ was made one.
    this.reversible = true;
    this.requiresApproval = true;
    this.destructive = false;
    this.rateCaps = null;
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

  async execute(args = {}, context = {}) {
    const { project, brief } = args;

    const result = await codingJobs.propose({
      project,
      brief,
      conversationId: context.conversationId || null,
      messageId: context.messageId || null
    });

    if (!result.ok) {
      return {
        success: false,
        error: result.error,
        message:
          `Nothing was sent. ${result.error} Tell her plainly that you did not ` +
          `dispatch it, and why — do not describe work that is not happening.`
      };
    }

    return {
      success: true,
      proposal_id: result.id,
      status: 'awaiting-approval',
      message:
        'The brief is written and waiting for her approval — nothing has been ' +
        'sent to squatch-code yet and no file has been touched. Show her the ' +
        'brief in your reply so she can approve or correct it without leaving ' +
        'the conversation. You do NOT have a result and must not describe one; ' +
        'when the job finishes, the write-up appears in her jobs panel.'
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
}

module.exports = DispatchCodingJobTool;
