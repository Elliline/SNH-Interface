/**
 * The corrector's WRITE actions: memory_merge_facts, memory_expire_fact,
 * memory_supersede_fact.
 *
 * BACKGROUND ONLY. `backgroundOnly = true` keeps these out of
 * getToolsForOpenAI(), so they are never in the schema handed to a chat turn.
 * Nothing Ellie says can reach them, and the entity cannot invoke them mid-
 * conversation — the corrector runs on the heartbeat, deliberately, with a
 * budget and a ledger, and that is the only way these fire.
 *
 * WHY TOOLS AT ALL, when the corrector's control flow is deterministic and could
 * just call fact-store directly? Because routing every write through the tool
 * layer buys three things that are otherwise re-implemented per caller and drift:
 * the per-step budget counts writes as well as reads, the allowlist is a real
 * boundary rather than a convention, and there is one audited path. The
 * enumeration of what to correct stays deterministic — a model cannot scan 580
 * facts — while the judgment calls stay with the model and the writes stay here.
 *
 * Each one is a thin wrapper over db/fact-store.js. None of them delete: the row
 * survives as inactive with a reason and a successor, which is what makes the
 * semantic tier revertible at all.
 */

const factStore = require('../../db/fact-store');

class BaseCorrectTool {
  constructor() {
    this.tier = 'reversible';
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
    // The flag that keeps this out of every chat turn.
    this.backgroundOnly = true;
  }

  getTierMetadata() {
    return {
      name: this.name, tier: this.tier, reversible: this.reversible,
      requiresApproval: this.requiresApproval, destructive: this.destructive,
      backgroundOnly: true
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
 * Fold a near-duplicate or subset fact into the one that survives.
 *
 * The loser goes inactive/superseded pointing at the survivor, so the history
 * says what happened rather than the row simply vanishing. The survivor picks up
 * a corroboration, because two facts asserting the same thing IS a second
 * assertion of it, and that is evidence the corrector's own dominance rules read
 * later.
 */
class MergeFactsTool extends BaseCorrectTool {
  constructor() {
    super();
    this.name = 'memory_merge_facts';
    this.description =
      'Fold one fact into another that says the same thing. The loser is marked superseded and points at the survivor; nothing is deleted.';
    this.parameters = {
      type: 'object',
      properties: {
        loser_id: { type: 'string', description: 'The fact to fold away.' },
        survivor_id: { type: 'string', description: 'The fact that remains.' }
      },
      required: ['loser_id', 'survivor_id']
    };
  }

  async execute(args = {}) {
    const { loser_id: loserId, survivor_id: survivorId } = args;
    if (!loserId || !survivorId) return { error: 'loser_id and survivor_id are both required' };
    if (loserId === survivorId) return { error: 'a fact cannot be merged into itself' };

    const loser = factStore.getMember(loserId);
    const survivor = factStore.getMember(survivorId);
    if (!loser) return { error: `no fact with id ${loserId}` };
    if (!survivor) return { error: `no fact with id ${survivorId}` };
    if (survivor.status !== 'active') return { error: 'the survivor is not active' };

    const res = await factStore.supersede(loserId, survivorId);
    if (res.locked) return { status: 'refused_locked', written: false, message: res.reason };
    if (!res.ok) return { error: res.reason || 'merge failed' };

    // The loser's own assertion becomes a corroboration of the survivor, with the
    // loser's provenance carried across — otherwise folding a duplicate would
    // DESTROY evidence ("she said this twice") in the act of tidying it up.
    factStore.recordCorroboration(survivorId, {
      conversationId: loser.conversation_id,
      messageId: loser.message_id,
      verbatimSourceText: loser.verbatim_source_text,
      inputModality: loser.input_modality,
      restatedAs: loser.content,
      similarity: null,
      detectedBy: 'corrector-merge'
    });

    // ledger_id rides back so the corrector can ENRICH the entry the write
    // already filed (2026-08-18) rather than filing a second one for the same
    // change. The funnel records that the change happened; the corrector alone
    // knows which pass made it and on what evidence.
    return { status: 'merged', loser_id: loserId, survivor_id: survivorId, vector_cleared: res.vector, ledger_id: res.ledgerId || null };
  }
}

/** Retire a fact that was always an event. */
class ExpireFactTool extends BaseCorrectTool {
  constructor() {
    super();
    this.name = 'memory_expire_fact';
    this.description =
      'Retire a fact that was really a passing event, not a durable truth. It becomes inactive with reason "expired"; nothing is deleted.';
    this.parameters = {
      type: 'object',
      properties: { fact_id: { type: 'string', description: 'The fact to expire.' } },
      required: ['fact_id']
    };
  }

  async execute(args = {}) {
    if (!args.fact_id) return { error: 'fact_id is required' };
    const res = await factStore.expire(args.fact_id);
    if (res.locked) return { status: 'refused_locked', written: false, message: res.reason };
    if (!res.ok) return { error: res.reason || 'expire failed' };
    return { status: 'expired', fact_id: args.fact_id, vector_cleared: res.vector, ledger_id: res.ledgerId || null };
  }
}

/** Supersede a fact that a better-evidenced one contradicts. */
class SupersedeFactTool extends BaseCorrectTool {
  constructor() {
    super();
    this.name = 'memory_supersede_fact';
    this.description =
      'Mark a fact superseded by another that contradicts it and is better evidenced. The old fact stays as history and points at the new one.';
    this.parameters = {
      type: 'object',
      properties: {
        old_id: { type: 'string', description: 'The fact that loses.' },
        new_id: { type: 'string', description: 'The fact that wins.' }
      },
      required: ['old_id', 'new_id']
    };
  }

  async execute(args = {}) {
    const { old_id: oldId, new_id: newId } = args;
    if (!oldId || !newId) return { error: 'old_id and new_id are both required' };
    if (oldId === newId) return { error: 'a fact cannot supersede itself' };
    const res = await factStore.supersede(oldId, newId);
    // A refusal is a RESULT, not an error. The identity lock exists to be hit,
    // and the corrector has to be able to record that it was told no.
    if (res.locked) return { status: 'refused_locked', written: false, message: res.reason };
    if (!res.ok) return { error: res.reason || 'supersede failed' };
    return { status: 'superseded', old_id: oldId, new_id: newId, vector_cleared: res.vector, ledger_id: res.ledgerId || null };
  }
}

module.exports = { MergeFactsTool, ExpireFactTool, SupersedeFactTool };
