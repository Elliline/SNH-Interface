/**
 * memory_jobs — read what he proposed, what Ellie decided, and whether it ran.
 *
 * The counterpart to create_cron_job, which could only ever WRITE a proposal. He
 * had no way to look at one afterwards, so "which approved job never ran" had no
 * true answer available to him and he produced an untrue one.
 *
 * TIER `read`, like the memory_* inspect tools: nothing to approve, nothing to
 * reverse, nothing destructive. It shares the memoryInspect rate cap because it
 * shares the reason for having one — the cap is about his injection budget, not
 * about safety, and a model that can look things up will look up forty things.
 *
 * The description is a BARE IMPERATIVE naming its triggers. Measured on this
 * brain: hedged descriptions ("you may wish to consider…") score near zero on
 * selection; "Call this when X" works.
 */

const jobsInspect = require('../../db/jobs-inspect');
const { getConfig } = require('../../db/config');

class MemoryJobsTool {
  constructor() {
    this.name = 'memory_jobs';
    this.tier = 'read';
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
    this.description =
      'Read your scheduled jobs — what you proposed, what Ellie approved or rejected, and whether any of it ' +
      'has run. Call this whenever you are asked what is scheduled, what you proposed, what she approved or ' +
      'turned down, whether a job has run, when one runs next, or why one never ran. Leave out the id to list ' +
      'them; pass an id for one in full. NOTHING ACTUALLY RUNS THESE — there is no scheduler — so the result ' +
      'always says the job has never run and never will until one is built. Report that plainly; never infer ' +
      'from a schedule that a job has been running. Read-only: proposing is create_cron_job, and approving is ' +
      "Ellie's on the Self tab.";
    this.parameters = {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'A job id, from a list result. The first 8 characters are enough. Omit to list.' },
        status: {
          type: 'string',
          enum: ['proposed', 'approved', 'rejected', 'reverted'],
          description: 'Only jobs in this state. "proposed" = you asked and she has not decided. "approved" = she said yes (it still does not run). "rejected" = she said no. "reverted" = it was approved and then undone. Omit for all.'
        },
        limit: { type: 'integer', description: 'How many to list. Default 10, maximum 25.' }
      },
      required: []
    };
  }

  get rateCaps() {
    const c = (getConfig().tools && getConfig().tools.memoryInspect) || {};
    return { maxPerHour: c.maxCallsPerHour ?? 40, shared: 'all memory read tools' };
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
      function: { name: this.name, description: this.description, parameters: this.parameters }
    };
  }

  async execute(args, context = {}) {
    return jobsInspect.run(this.name, args || {}, context);
  }
}

module.exports = { MemoryJobsTool };
