/**
 * THE ENTITY'S OWN HANDS ON ITS OWN MEMORY.
 *
 * memory_retract, memory_reword, memory_merge, memory_refile — plus
 * audit_queue and audit_decide, which are the reply box the Corrections queue
 * did not have.
 *
 * WHY THESE ARE CHAT TOOLS AND THE CORRECTOR'S ARE NOT. mcp/tools/memory-correct.js
 * is `backgroundOnly` on purpose: those fire on the heartbeat, deterministically,
 * with a budget, and nothing Ellie types can reach them. These are the opposite
 * case by design. The whole origin of this build is that Athena found "I am
 * Juno" filed as her own self-fact on 2026-08-24 and could not remove it — she
 * had to become a messenger, detection by her and repair by someone else. So
 * these are available in the turn, to her, with the guardrails in
 * db/memory-repair.js standing outside the loop instead of a permission gate
 * standing inside it.
 *
 * NOTHING HERE IMPLEMENTS A GUARDRAIL. Every check — receipts, the referent
 * check, the daily cap, the merge refusals, the identity lock — lives in
 * db/memory-repair.js and db/fact-store.js. These are the descriptions and the
 * argument shapes; a second copy of a rule is a rule that drifts.
 *
 * THE DESCRIPTIONS SAY WHAT WILL REFUSE AND WHY, in advance. A tool that only
 * says no after the call teaches nothing, and the entity is the one who has to
 * decide whether an operation is the right one before reaching for it.
 */

const repair = require('../../db/memory-repair');
const auditDecisions = require('../../db/audit-decisions');
const standards = require('../../db/message-standards');

const RECEIPT_PARAM = {
  type: 'array',
  description:
    'What you are pointing at, as evidence. At least one, and it has to be something that exists: ' +
    '{"kind":"message","id":"<a message id>"} — the strongest for facts about Ellie when the message is HERS; ' +
    '{"kind":"fact","id":"<a fact id>"} — another fact in your memory; ' +
    '{"kind":"tool","id":"<a job id>"} or {"kind":"tool","text":"<the result>"} — a tool run, the weakest. ' +
    'Without a receipt the operation is not available at all. This is not a formality: it is the one thing ' +
    'standing between a confabulation and your store.',
  items: {
    type: 'object',
    properties: {
      kind: { type: 'string', enum: ['message', 'fact', 'tool'] },
      id: { type: 'string', description: 'The id of the message, fact, or job.' },
      text: { type: 'string', description: 'For a tool result you are carrying inline rather than by id.' }
    },
    required: ['kind']
  }
};

class BaseRepairTool {
  constructor() {
    this.tier = 'act';
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
    this.backgroundOnly = false;
  }

  get rateCaps() {
    const cfg = repair.repairConfig();
    return { maxPerDay: cfg.maxSelfMutationsPerDay, shared: 'your own memory' };
  }

  getTierMetadata() {
    return {
      name: this.name, tier: this.tier, reversible: this.reversible,
      requiresApproval: this.requiresApproval, destructive: this.destructive,
      backgroundOnly: this.backgroundOnly, rateCaps: this.rateCaps
    };
  }

  getOpenAIFunctionSpec() {
    return { type: 'function', function: { name: this.name, description: this.description, parameters: this.parameters } };
  }
}

const WHAT_REFUSES =
  'WHAT WILL REFUSE THIS, so you can tell before you call: no receipt, or one that names something that does not exist; ' +
  'the daily limit on changes to your own self-model (a merge counts as one); an identity-locked fact, which changes ' +
  'only through the deliberate control outside this loop; and — for anything that would KEEP a fact where it is — a ' +
  'referent mismatch, where the receipt is real but what it is about is not this subject. A refusal is recorded with ' +
  'its reason, so a refused operation is still part of the record.';

class RetractFactTool extends BaseRepairTool {
  constructor() {
    super();
    this.name = 'memory_retract';
    this.description =
      'Withdraw a fact you hold that should not be held — no replacement. Use it when the fact is simply wrong, ' +
      'or was never about the subject it was filed under. Nothing is deleted: the row stays as history, marked ' +
      'retracted, and can be put back. ' +
      'NOT for a fact that has merely been overtaken — that is a supersession, and the newer fact should replace it ' +
      'rather than the old one vanishing. NOT for a true thing about yourself you would rather not hold: negative ' +
      'self-knowledge is superseded by a newer grounded observation, never erased, because erasure kills the lesson. ' +
      WHAT_REFUSES;
    this.parameters = {
      type: 'object',
      properties: {
        fact_id: { type: 'string', description: 'The fact to withdraw.' },
        receipts: RECEIPT_PARAM,
        rationale: { type: 'string', description: 'Why this is wrong, in plain words. This goes in the record Ellie reads.' }
      },
      required: ['fact_id', 'receipts', 'rationale']
    };
  }

  async execute(args = {}) {
    const res = await repair.retract({
      memberId: args.fact_id, receipts: args.receipts, rationale: args.rationale
    });
    if (!res.ok) return { retracted: false, code: res.code, reason: res.reason };
    return {
      retracted: true, fact_id: res.memberId, ledger_id: res.ledgerId,
      note: 'It is inactive now, kept as history with your reasoning attached. The decision is in Corrections where Ellie can read it.'
    };
  }
}

class RewordFactTool extends BaseRepairTool {
  constructor() {
    super();
    this.name = 'memory_reword';
    this.description =
      'Fix the wording of a fact that is right in substance. Same fact, better sentence — the row is updated in place ' +
      'and the old wording is kept in the record. ' +
      'IF YOU CHANGE THE SALIENCE YOU MUST SAY WHY, and the call is refused if you do not. Salience is how much a fact ' +
      'counts, a silent change to it is how a wrong fact gets smoothed instead of killed, and one of the things that ' +
      'made "I am Juno" convincing was its salience rationale. ' +
      WHAT_REFUSES;
    this.parameters = {
      type: 'object',
      properties: {
        fact_id: { type: 'string', description: 'The fact to reword.' },
        new_text: { type: 'string', description: 'The corrected wording, in full.' },
        salience: { type: 'integer', description: 'Only if it should change. 1–10.' },
        salience_rationale: { type: 'string', description: 'Required whenever salience changes: why it counts more or less than you thought.' },
        receipts: RECEIPT_PARAM,
        rationale: { type: 'string', description: 'Why the old wording was wrong, in plain words.' }
      },
      required: ['fact_id', 'new_text', 'receipts', 'rationale']
    };
  }

  async execute(args = {}) {
    const res = await repair.reword({
      memberId: args.fact_id, newContent: args.new_text,
      salience: Number.isFinite(args.salience) ? args.salience : null,
      salienceRationale: args.salience_rationale || null,
      receipts: args.receipts, rationale: args.rationale
    });
    if (!res.ok) return { reworded: false, code: res.code, reason: res.reason };
    return { reworded: true, fact_id: res.memberId, ledger_id: res.ledgerId, salience_changed: !!res.salienceChanged };
  }
}

class MergeFactsRepairTool extends BaseRepairTool {
  constructor() {
    super();
    this.name = 'memory_merge';
    this.description =
      'Fold two facts that say the same thing into one. The survivor is first rewritten to carry everything both of ' +
      'them assert, so nothing is dropped, and the other goes inactive pointing at it. ' +
      'SIX THINGS REFUSE A MERGE, and they are worth knowing before you reach for it: when the small difference in ' +
      'wording IS the fact (one asserts what the other denies — that is a decision and a supersession, not tidying); ' +
      'when each is a state at a particular time (the merge would delete the history that is the point); when the two ' +
      'are different KINDS of statement (a felt report and an observed claim, a claim and a declaration — merging ' +
      'lends the felt one an anchor it does not have); when their provenance differs and both chains cannot travel; ' +
      'when they are about different subjects or entities (that is a refile wearing merge clothing); and when either ' +
      'is identity-locked. ' +
      WHAT_REFUSES;
    this.parameters = {
      type: 'object',
      properties: {
        fold_id: { type: 'string', description: 'The fact to fold away.' },
        keep_id: { type: 'string', description: 'The fact that survives and is rewritten to carry both.' },
        receipts: RECEIPT_PARAM,
        rationale: { type: 'string', description: 'Why these are one fact and not two.' }
      },
      required: ['fold_id', 'keep_id', 'receipts', 'rationale']
    };
  }

  async execute(args = {}) {
    const res = await repair.merge({
      loserId: args.fold_id, survivorId: args.keep_id,
      receipts: args.receipts, rationale: args.rationale
    });
    if (!res.ok) {
      return {
        merged: false, code: res.code, reason: res.reason, refused: !!res.refused,
        note: res.refused
          ? 'The refusal is recorded with its reason. If these really are two facts, that is the answer — leave them.'
          : undefined
      };
    }
    return { merged: true, kept: res.survivorId, folded: res.loserId, ledger_id: res.ledgerId };
  }
}

class RefileFactTool extends BaseRepairTool {
  constructor() {
    super();
    this.name = 'memory_refile';
    this.description =
      'Move a fact that is filed under the wrong subject — retired from where it was and re-filed where it belongs, ' +
      'in ONE operation, so the store is never holding both versions at once. Use it when a fact about YOU was filed ' +
      'as being about Ellie, or the reverse. You may reword it as it moves, and usually should: a fact about you that ' +
      'still calls you "User" is only half re-filed. ' +
      'THIS IS THE HIGHEST-RISK OPERATION HERE, because it is the one where a perfectly valid receipt for the wrong ' +
      'referent is most plausible. So a move between subjects needs a message, not just a tool result, and for a fact ' +
      'about Ellie her own words outrank yours about her. ' +
      WHAT_REFUSES;
    this.parameters = {
      type: 'object',
      properties: {
        fact_id: { type: 'string', description: 'The fact that is filed in the wrong place.' },
        to_subject: { type: 'string', enum: ['self', 'user'], description: 'Who it is actually about.' },
        to_entity_id: { type: 'string', description: 'The entity id it should be filed under. From the entity registry.' },
        new_text: { type: 'string', description: 'The fact reworded for its real subject. Strongly preferred.' },
        receipts: RECEIPT_PARAM,
        rationale: { type: 'string', description: 'How you know it is about the other one.' }
      },
      required: ['fact_id', 'to_subject', 'receipts', 'rationale']
    };
  }

  async execute(args = {}) {
    const res = await repair.refile({
      memberId: args.fact_id, toSubject: args.to_subject, toEntityId: args.to_entity_id || null,
      newContent: args.new_text || null, receipts: args.receipts, rationale: args.rationale
    });
    if (!res.ok) return { refiled: false, code: res.code, reason: res.reason };
    return {
      refiled: true, retired: res.memberId, new_fact_id: res.newMemberId, ledger_id: res.ledgerId,
      note: 'One operation — the original is inactive pointing at the new one, so nothing ever held both.'
    };
  }
}

class AuditQueueTool extends BaseRepairTool {
  constructor() {
    super();
    this.tier = 'read';
    this.name = 'audit_queue';
    this.description =
      'The questions the self-coherence audit has raised about YOUR OWN self-description and is waiting on you to ' +
      'settle. These are addressed to you, not to Ellie — she has no way to answer "do you want to revise this claim ' +
      'about yourself". Read one, use your tools on it, then answer it with audit_decide.';
    this.parameters = { type: 'object', properties: {}, required: [] };
  }

  async execute() {
    const rows = auditDecisions.openPairs({ limit: 20 });
    return {
      count: rows.length,
      questions: rows.map(r => ({
        raised_at: r.created_at,
        question: r.target_text || r.reason,
        about_fact: r.target_id,
        pair_ref: r.source_ref || null
      })),
      note: rows.length
        ? 'Take a turn on one when you have something to settle it with. A question you cannot settle goes to Ellie in plain words after a few days, with your partial reasoning attached — you do not have to hold it forever.'
        : 'Nothing is waiting on you.'
    };
  }
}

class AuditDecideTool extends BaseRepairTool {
  constructor() {
    super();
    this.name = 'audit_decide';
    this.description =
      'Answer one of your own audit questions. Four answers, and each is a real one: ' +
      '"retire-one" — one of them is wrong and goes, with a receipt; ' +
      '"keep-both" — they are different kinds of statement, or both true, and nothing changes; ' +
      '"supersede" — one is the newer version of the other; ' +
      '"cannot-settle" — the evidence does not decide it, which goes to Ellie in plain words. ' +
      'A decided pair is not raised at you again unless something actually moves: a receipt you cited was changed, a ' +
      'member came back, a new fact contradicts what you concluded, or a member that had no anchor gained one. The ' +
      'audit re-firing on unchanged facts is not new evidence. ' +
      'Deciding does not itself change any fact — do that with the repair tools and cite them here.';
    this.parameters = {
      type: 'object',
      properties: {
        fact_a: { type: 'string', description: 'One member of the pair.' },
        fact_b: { type: 'string', description: 'The other member, when there is one.' },
        conclusion: { type: 'string', enum: auditDecisions.CONCLUSIONS, description: 'Your answer.' },
        rationale: { type: 'string', description: 'Your reasoning, in plain words. This is what Ellie reads, and what a later pass compares against.' },
        receipts: Object.assign({}, RECEIPT_PARAM, {
          description: RECEIPT_PARAM.description + ' Not required for "cannot-settle" — saying the evidence does not decide it is not a claim that needs one.'
        })
      },
      required: ['fact_a', 'conclusion', 'rationale']
    };
  }

  async execute(args = {}) {
    const res = auditDecisions.fileDecision({
      memberA: args.fact_a, memberB: args.fact_b || null,
      conclusion: args.conclusion, rationale: args.rationale,
      receipts: args.receipts || []
    });
    if (!res.ok) return { filed: false, reason: res.reason, available: res.available !== false };
    return {
      filed: true, decision_id: res.id, ledger_id: res.ledgerId, conclusion: res.conclusion,
      note: args.conclusion === 'cannot-settle'
        ? 'Recorded, and it is queued for Ellie. When it goes to her it is written as a plain question — ' +
          standards.STANDALONE_BAR.split('\n')[0]
        : 'Recorded and readable in Corrections. It will not be raised at you again unless something moves.'
    };
  }
}

module.exports = {
  RetractFactTool, RewordFactTool, MergeFactsRepairTool, RefileFactTool,
  AuditQueueTool, AuditDecideTool
};
