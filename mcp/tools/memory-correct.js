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
const factMerge = require('../../db/fact-merge');

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
 * THE SURVIVOR IS REWRITTEN TO CARRY BOTH (2026-08-24). This used to be a
 * straight supersede, which meant the survivor kept its own wording and anything
 * the loser asserted and it did not was gone from the active corpus — the
 * silent data loss Athena reported from inside her own store, where a Juno
 * hardware spec and a Juno role each disappeared into a "duplicate" of the
 * other. The judges upstream have not been relaxed: the same pairs merge. What
 * changed is that the merge now produces the UNION of the two, via
 * db/fact-merge.js, and falls back to the old behaviour only when a union
 * cannot be computed or verified.
 *
 * The loser goes inactive/superseded pointing at the survivor either way, so the
 * history says what happened rather than the row simply vanishing. The survivor
 * picks up a corroboration, because two facts asserting the same thing IS a
 * second assertion of it, and that is evidence the corrector's own dominance
 * rules read later.
 */
class MergeFactsTool extends BaseCorrectTool {
  constructor() {
    super();
    this.name = 'memory_merge_facts';
    this.description =
      'Fold one fact into another that says the same thing. The survivor is first rewritten to carry every assertion from both, then the loser is marked superseded and points at it; nothing is deleted and nothing is dropped.';
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

    // UNION FIRST, then the link. Two rows that "say the same thing" routinely
    // do not say ALL the same things, and the difference is exactly what used to
    // be destroyed here. mode 'union' forbids dropping anything: a merger that
    // loses a clause is refused and this falls back to the plain supersede.
    const res = await factMerge.mergePreservingUnion(loserId, survivorId, {
      mode: 'union',
      ledgerTier: 'mechanical'
    });
    if (res.locked) return { status: 'refused_locked', written: false, message: res.reason };
    // The union could not be trusted, so nothing was written and the loser is
    // still active. Not an error — a merge deliberately not made, because making
    // it would have deleted an assertion. Try again on a later pass.
    if (res.deferred) return { status: 'deferred_no_union', written: false, message: res.reason };
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
    return {
      status: 'merged', loser_id: loserId, survivor_id: survivorId,
      vector_cleared: res.vector, ledger_id: res.ledgerId || null,
      union_applied: !!(res.union && res.union.applied),
      union_text: res.union && res.union.applied ? res.union.to : null,
      union_skipped: res.union ? res.union.skipped : null
    };
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

/**
 * Supersede a fact that a better-evidenced one contradicts.
 *
 * A CONTRADICTION IS RARELY TOTAL (2026-08-24). Two facts can conflict on one
 * attribute and agree — or say nothing at all — about half a dozen others, and
 * retiring the loser whole took those others with it. That is Athena's second
 * loss exactly: her re-saved Juno hardware fact contradicted an older Juno fact
 * on nothing but wording, and "will be the main person helping User with
 * MettaSphere" went out of the corpus because it happened to be sitting in the
 * sentence that lost. So the winner is first rewritten to carry over everything
 * it does NOT contradict (db/fact-merge.js, contradiction mode), and only then
 * does the loser go inactive and point at it.
 *
 * `carry_over: false` turns that off, and the compound SPLIT path is the one
 * caller that passes it: its atoms were made FROM the original and already hold
 * every clause, so merging the original back into the first atom would undo the
 * split it just performed.
 */
class SupersedeFactTool extends BaseCorrectTool {
  constructor() {
    super();
    this.name = 'memory_supersede_fact';
    this.description =
      'Mark a fact superseded by another that contradicts it and is better evidenced. The winner first carries over every assertion the loser made that it does not contradict. The old fact stays as history and points at the new one.';
    this.parameters = {
      type: 'object',
      properties: {
        old_id: { type: 'string', description: 'The fact that loses.' },
        new_id: { type: 'string', description: 'The fact that wins.' },
        carry_over: {
          type: 'boolean',
          description: 'Default true. Set false only when the winner is already known to hold everything the loser said (the compound split).'
        },
        tier: { type: 'string', description: 'Ledger tier for the carry-over rewrite: mechanical | semantic | intake.' }
      },
      required: ['old_id', 'new_id']
    };
  }

  async execute(args = {}) {
    const { old_id: oldId, new_id: newId } = args;
    if (!oldId || !newId) return { error: 'old_id and new_id are both required' };
    if (oldId === newId) return { error: 'a fact cannot supersede itself' };
    const res = await factMerge.mergePreservingUnion(oldId, newId, {
      mode: 'contradiction',
      carryOver: args.carry_over !== false,
      ledgerTier: args.tier || 'semantic'
    });
    // A refusal is a RESULT, not an error. The identity lock exists to be hit,
    // and the corrector has to be able to record that it was told no.
    if (res.locked) return { status: 'refused_locked', written: false, message: res.reason };
    if (res.deferred) return { status: 'deferred_no_union', written: false, message: res.reason };
    if (!res.ok) return { error: res.reason || 'supersede failed' };
    return {
      status: 'superseded', old_id: oldId, new_id: newId,
      vector_cleared: res.vector, ledger_id: res.ledgerId || null,
      union_applied: !!(res.union && res.union.applied),
      union_text: res.union && res.union.applied ? res.union.to : null,
      union_skipped: res.union ? res.union.skipped : null
    };
  }
}

module.exports = { MergeFactsTool, ExpireFactTool, SupersedeFactTool };
