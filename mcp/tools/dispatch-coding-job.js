/**
 * dispatch_coding_job — sending coding work to squatch-code.
 *
 * THE INTERFACE IS THE CONVERSATION. Ellie and the entity talk a problem
 * through; he writes the brief in his reply where she can read it; she
 * says send it; this fires. No panel, no button, nothing to leave the
 * chat for.
 *
 * This shipped as a bell proposal with an approve button and that was a
 * mechanism failure, not a matter of taste. Of 390 bell items ever
 * raised on this instance, 223 expired unactioned; exactly one proposal
 * has ever been raised there and it was dismissed. Briefs would have sat
 * until they aged out. The rule Ellie drew from it is general and lives
 * in CLAUDE.md: nothing that has to be ACTED ON goes on the bell.
 *
 * SO THIS TOOL IS NOT PROPOSE-ONLY, and it is not direct-execute either.
 * Her go-ahead already happened, in words, in the conversation - the
 * call is what carries it out. What makes that safe is not a gate in
 * the UI but a check on the brief itself: db/brief-shown.js refuses
 * anything she has not already been shown, which is what stops this
 * firing on the turn where the brief is still being written.
 *
 * ONE COARSE TOOL, NOT FILE OPERATIONS. squatch-code has its own agentic
 * loop, its own model and its own tools; it wants a brief, not a driver.
 */

const codingJobs = require('../../db/coding-jobs');

class DispatchCodingJobTool {
  constructor() {
    this.name = 'dispatch_coding_job';
    // SHORTER ON PURPOSE. This grew into a wall of prohibitions - do not
    // substitute a project, do not ask for directories, do not, do not -
    // and on the night it mattered the model returned tool_calls: [] and
    // narrated the work instead. A tool described mostly by what it
    // refuses is a tool that is easier not to call. The prohibitions are
    // now ENFORCED in db/coding-jobs.js (validateBrief, validateProject),
    // which is where they belong, so this can say what the tool is for.
    this.description =
      'Send coding work to squatch-code, the local coding agent, which does it ' +
      'on its own in one project and writes up what it did. ' +
      'Use it when Ellie tells you to send work: "send that to the coder", ' +
      '"go ahead", "ship it". ' +
      'Two rules, both checked for you: she must have READ the brief in an ' +
      'earlier reply, and the brief must say WHAT to build, never where to put ' +
      'it. Where it goes is decided by the project field alone — give it a ' +
      'plain name like "squatch_crawler" and it is created if it does not ' +
      'exist. If a call is refused, the refusal tells you what to change; do ' +
      'that and call again in the same reply. ' +
      'The write-up arrives in her jobs panel, so do not describe a result you ' +
      'do not have, and do not say you have sent anything unless this returned ' +
      'success.';

    this.parameters = {
      type: 'object',
      properties: {
        project: {
          type: 'string',
          description:
            'The project this work belongs in, e.g. "todoapp". A NAME, never a ' +
            'path and never nested. If it does not exist it is created, so name ' +
            'the project the work actually belongs to rather than an existing ' +
            'one that happens to be nearby.'
        },
        brief: {
          type: 'string',
          description:
            'The brief, exactly as Ellie read it. Everything the run will know — ' +
            'it cannot see this conversation. Copy the text you already showed ' +
            'her rather than composing a fresh version of it.'
        }
      },
      required: ['project', 'brief']
    };

    this.tier = 'action';
    // A git restore point is committed inside the project before the run
    // starts, and the report carries the command that undoes the job.
    this.reversible = true;
    // Her approval is real but conversational: it happened in words
    // before this was called, and the brief-shown check is what ties the
    // call to it. There is no pending-approval state anywhere.
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

    const result = codingJobs.dispatch({
      project,
      brief,
      conversationId: context.conversationId || null,
      messageId: context.messageId || null,
      userMessage: context.userMessage || null,
    });

    if (!result.ok) {
      // Every refusal ends with something to DO this turn, because a
      // refusal the model cannot act on costs a round trip and, three
      // times tonight, produced a claim that work had started instead.
      let next;
      if (result.unseen) {
        next = ' Write the brief out in this reply so she can read it, and send ' +
               'it once she says to. Do not claim anything has been sent.';
      } else if (result.briefRejected) {
        next = ' Rewrite the brief with the directory instructions removed, put ' +
               'the project name in the project field instead, show her the ' +
               'corrected brief, and send that in this same reply if she has ' +
               'already said go.';
      } else if (result.suggestion) {
        next = ` Call this again with project: "${result.suggestion}".`;
      } else {
        next = ' Tell her plainly that you did not send it, and why.';
      }
      return {
        success: false,
        error: result.error,
        retry_with_project: result.suggestion || undefined,
        message: 'NOT SENT. ' + result.error + next,
      };
    }

    // A paraphrase is dispatched but never passes silently: both texts
    // are in the scrollback, so a divergence is hers to see.
    const fidelity = result.exact
      ? 'The brief you sent is word for word what she read.'
      : 'NOTE: what you sent is not word for word what she read (' +
        Math.round(result.ratio * 100) + '% match). Quote the brief you ' +
        'actually sent in your reply so she can see the difference.';

    // A project that did not exist a moment ago is something she should
    // hear in this reply, not discover later. And if the name is close to
    // one she already has, that is where a typo gets caught.
    let newProject = '';
    if (result.renamed) {
      newProject += ` You asked for "${result.renamed}"; the project is ` +
        `Projects/${result.project}. Use that name when you refer to it.`;
    }
    if (result.isNewProject) {
      newProject += ` This project did not exist, so a new one was created at Projects/${result.project}.`;
      if (result.nearMatches && result.nearMatches.length) {
        newProject += ` Say so plainly, and mention that she already has ` +
          `Projects/${result.nearMatches.join(' and Projects/')} — if that is ` +
          `what was meant, this one can be deleted.`;
      } else {
        newProject += ' Say so in your reply.';
      }
    }

    return {
      success: true,
      dispatch_id: result.id,
      job_id: result.agentJobId,
      status: 'running',
      new_project: !!result.isNewProject,
      near_matches: result.nearMatches || [],
      message:
        'Sent to squatch-code and it is running now. ' + fidelity + newProject +
        ' Tell her it has gone and quote the brief that was sent. You do NOT ' +
        'have a result and must not describe one — the write-up will appear ' +
        'in her jobs panel when the job finishes.',
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
