/**
 * create_cron_job — the first action tool.
 *
 * PROPOSE ONLY. Calling this does not create anything: it records a proposal and
 * raises it in the bell panel for Ellie to approve or reject.
 *
 * What approving MEANS changed on 2026-08-12. It used to record the job and
 * nothing more; there is a scheduler now (db/scheduler.js), so approving arms
 * the job and it starts running on its schedule. The propose-only boundary is
 * unchanged and is the whole point of this tool — he asks, she decides — but the
 * stakes of her yes are higher than they were, and the description below says so
 * rather than letting him propose as though nothing would happen.
 *
 * TIER METADATA (tier/reversible/rateCaps/requiresApproval) is declared here now
 * even though nothing reads it. The tool registry will consume it later; putting
 * it on the tool at birth means the first tool is not a special case that has to
 * be retrofitted when the registry lands.
 */

const cronJobs = require('../../db/cron-jobs');
const { getConfig } = require('../../db/config');

class CreateCronJobTool {
  constructor() {
    this.name = 'create_cron_job';
    this.description =
      'Propose a recurring scheduled job. This does NOT create the job — it sends a ' +
      'proposal to Ellie for approval, and she decides. If she approves it, it is ' +
      'scheduled and WILL run on the schedule you give: each run is you, in the ' +
      'background, doing what your description says using your read-only memory ' +
      'tools, and reporting the result to her notification panel. Propose accordingly ' +
      '— the description IS the instruction the run follows. Use it when she asks for ' +
      'something to happen on a schedule.';
    this.parameters = {
      type: 'object',
      properties: {
        schedule: {
          type: 'string',
          description: "Cron expression, 5 fields (minute hour day-of-month month day-of-week), e.g. '0 6 * * *' for 6am every day."
        },
        description: {
          type: 'string',
          description: 'Plain-language description of what the job does, e.g. "check disk space".'
        },
        enabled: {
          type: 'boolean',
          description: 'Whether the job should start running as soon as she approves it. False means it is recorded but not armed until she enables it.'
        }
      },
      required: ['schedule', 'description', 'enabled']
    };

    // ---- tier metadata (declared now; consumed by the registry later) ----
    this.tier = 'reversible';
    this.reversible = true;
    this.requiresApproval = true;
    this.destructive = false;
  }

  /** Rate caps read live from config so the declared metadata never drifts from behavior. */
  get rateCaps() {
    const c = (getConfig().tools && getConfig().tools.cron) || {};
    return {
      maxPerHour: c.maxProposalsPerHour ?? 3,
      maxTotal: c.maxKidCreatedJobs ?? 10
    };
  }

  /** Full tier descriptor — the shape the future registry will read. */
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
   * @param {Object} args - { schedule, description, enabled }
   * @param {Object} context - { conversationId }
   */
  async execute(args, context = {}) {
    const result = await cronJobs.propose({
      schedule: args?.schedule,
      description: args?.description,
      enabled: args?.enabled,
      conversationId: context.conversationId || null
    });

    if (!result.ok) return { error: result.error };

    return {
      status: 'proposed',
      proposal_id: result.id,
      message:
        'Proposed — not created. This is now waiting for Ellie to approve or reject it ' +
        'in her bell panel. Tell her you have put it to her for approval; do not tell ' +
        'her the job exists or is scheduled, because it does not yet.'
    };
  }
}

module.exports = CreateCronJobTool;
