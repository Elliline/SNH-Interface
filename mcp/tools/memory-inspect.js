/**
 * The READ tools: memory_search, memory_list, memory_count, memory_get,
 * memory_corrections.
 *
 * One file rather than five, because they are one capability with one rate cap,
 * one config flag and one backing module (db/memory-inspect.js) — splitting them
 * would put five copies of the same "read-only, never writes" contract in five
 * places to drift apart.
 *
 * TIER. All of them are `read` — the first tools here that neither act on the world
 * nor change anything. No approval, no reversal needed, nothing destructive. The
 * rate cap is not a safety bound; it exists because his injection budget is small
 * and a model that can look things up will happily look up forty things.
 *
 * DESCRIPTIONS ARE BARE IMPERATIVES. Measured on this brain (Gemma-4-26B-A4B):
 * hedged tool descriptions — "you may wish to consider using this when…" —
 * score near zero on selection. "Call this when X" works. Every description
 * below names the trigger and says call it.
 */

const memoryInspect = require('../../db/memory-inspect');
const { getConfig } = require('../../db/config');

/** Shared tier + spec machinery. */
class BaseInspectTool {
  constructor() {
    this.tier = 'read';
    this.reversible = true;
    this.requiresApproval = false;
    this.destructive = false;
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
    return memoryInspect.run(this.name, args || {}, context);
  }
}

// Shared argument shapes, so the four tools describe the same filter the same way.
const SUBJECT_ARG = {
  type: 'string',
  enum: ['user', 'self', 'world'],
  description: 'Whose facts. "user" = facts about Ellie. "self" = facts about you. "world" = knowledge about external things that is about neither of you — how a service behaves, a tool\'s quirk. World facts are NOT in your injected context, so search for them rather than assuming you would already know. Omit for all.'
};
const STATUS_ARG = {
  type: 'string',
  enum: ['active', 'inactive', 'any'],
  description: 'Defaults to "active" — what you currently believe. "inactive" = facts you have superseded, retired or expired. "any" = both. Use "inactive" only when asked what you USED to believe.'
};

class MemorySearchTool extends BaseInspectTool {
  constructor() {
    super();
    this.name = 'memory_search';
    this.description =
      'Search your long-term memory for facts about a topic. Call this whenever you are asked what you know, ' +
      'remember, or have stored about something, and whenever answering would mean recalling a specific fact ' +
      'rather than talking generally. Your injected context is a small slice of your memory, not all of it — ' +
      'if the answer is not already in front of you, search instead of guessing. Returns one compact line per ' +
      'fact with its id; pass an id to memory_get for the full story behind it.';
    this.parameters = {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'What to look for. A topic, a name, or a phrase — "MettaSphere", "her dogs", "what she drives".' },
        subject: SUBJECT_ARG,
        status: STATUS_ARG,
        limit: { type: 'integer', description: 'How many facts to return. Default 10, maximum 20.' }
      },
      required: ['query']
    };
  }
}

class MemoryListTool extends BaseInspectTool {
  constructor() {
    super();
    this.name = 'memory_list';
    this.description =
      'List what is in your memory, without searching for anything in particular. Call this when asked what ' +
      'you hold about a whole area, what your memory is organised into, or to page through facts. ' +
      'Set mode to "clusters" to get the names of your memory clusters and how many facts each holds — do that ' +
      'first when you need a cluster name, rather than guessing one.';
    this.parameters = {
      type: 'object',
      properties: {
        mode: { type: 'string', enum: ['facts', 'clusters'], description: '"facts" (default) lists individual facts. "clusters" lists your cluster names and member counts.' },
        cluster: { type: 'string', description: 'Restrict to one cluster, by exact name. Get names from mode:"clusters".' },
        subject: SUBJECT_ARG,
        status: STATUS_ARG,
        limit: { type: 'integer', description: 'How many facts to return. Default 10, maximum 20.' },
        offset: { type: 'integer', description: 'Skip this many first, to page through. The previous result tells you the next offset.' }
      },
      required: []
    };
  }
}

class MemoryCountTool extends BaseInspectTool {
  constructor() {
    super();
    this.name = 'memory_count';
    this.description =
      'Count facts in your memory. Call this for any "how many" question about what you know, and before ' +
      'stating any number about your own memory — never estimate that number, count it. Returns counts only, ' +
      'no fact text, so it is the cheap way to answer "how much do you have on X".';
    this.parameters = {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Optional. Count only facts whose text contains this — "MettaSphere", "dog".' },
        cluster: { type: 'string', description: 'Optional. Count only facts in this cluster, by exact name.' },
        subject: SUBJECT_ARG,
        status: STATUS_ARG
      },
      required: []
    };
  }
}

class MemoryGetTool extends BaseInspectTool {
  constructor() {
    super();
    this.name = 'memory_get';
    this.description =
      'Open one fact in full. Call this whenever you are asked WHY you believe something, WHEN or HOW you ' +
      'learned it, where it came from, or whether it is still true. Returns the fact\'s salience and the reason ' +
      'it scored that, its status and what replaced it, when you learned it, the conversation and message it ' +
      'came from, the exact words that were said, whether they were spoken or typed, and every time it has ' +
      'been said again since. Answer questions about the origin of a belief from this, not from memory.';
    this.parameters = {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'The fact id, from a memory_search or memory_list result. The first 8 characters are enough.' }
      },
      required: ['id']
    };
  }
}

class MemoryCorrectionsTool extends BaseInspectTool {
  constructor() {
    super();
    this.name = 'memory_corrections';
    this.description =
      'Read the record of changes made to your own memory. Call this whenever you are asked what has changed ' +
      'in your memory, why a fact was corrected, replaced or removed, or what you used to believe and why you ' +
      'stopped. Leave out the id to list recent changes; pass a correction id for one in full — what was ' +
      'retired, what was kept, and the evidence it was decided on. This is the ONLY source for why something ' +
      'changed: if a change is not in here, there is no record of it, and you say so rather than working out ' +
      'an explanation. Read-only — you cannot undo a correction, that is Ellie\'s to do.';
    this.parameters = {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'A correction id, from a list result or from memory_get on a fact. The first 8 characters are enough. Omit to list.' },
        subject: {
          type: 'string',
          enum: ['user', 'self', 'world'],
          description: 'Whose facts the change was to. "user" = facts about Ellie. "self" = facts about you. "world" = external knowledge about neither of you. Omit for all.'
        },
        action: {
          type: 'string',
          enum: ['merge', 'expire', 'split', 'supersede', 'reconcile', 'retract', 'repoint'],
          description: 'Only changes of this kind. merge = duplicates folded together. expire = something that was really a passing event moved out. split = one fact that said two things separated. supersede = one belief replaced by a better-evidenced one. reconcile = an index repair. retract = a fact withdrawn with no replacement, e.g. one filed under the wrong person. repoint = a retired fact aimed at a different successor, where the retirement was right and the replacement named was wrong.'
        },
        tier: {
          type: 'string',
          enum: ['mechanical', 'semantic'],
          description: '"mechanical" repaired the record (duplicates, events, indexes). "semantic" changed what you believe.'
        },
        limit: { type: 'integer', description: 'How many to list. Default 10, maximum 20.' }
      },
      required: []
    };
  }
}

module.exports = { MemorySearchTool, MemoryListTool, MemoryCountTool, MemoryGetTool, MemoryCorrectionsTool };
