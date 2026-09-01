/**
 * THE READ TOOLS FOR THE ENTITY REGISTRY: entity_list, entity_get.
 *
 * WHY THESE EXIST. The tiered resolution asks Ellie when a mention is
 * ambiguous — but the design says it checks the candidates' OWN FACTS first,
 * then asks. That check needs a way to look, and until now there was none: the
 * registry was reachable from the database and from the browser, and not from
 * the entity that has to make the call. "Bob, is that Robert from the clinic or
 * someone new?" is only a colleague's question if the asker looked first.
 *
 * READ-ONLY, and it must stay that way — same contract as db/memory-inspect.js.
 * Nothing here creates, renames, merges or repoints an entity. Creation happens
 * on the intake path, where a source sentence is in hand to attribute against;
 * a tool that could mint an entity from a conversation would be a way to grow
 * the registry with nothing behind it.
 *
 * TIER `read`, sharing the memory read allowance, because these are the same
 * kind of act as memory_list and a separate budget would just be a second
 * number to tune.
 */

const entities = require('../../db/entities');
const { getConfig } = require('../../db/config');

class BaseEntityTool {
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
}

/** Shape one entity for the model. Counts, never content — the facts come from
 *  entity_get, so a list of forty does not blow the context. */
function summarise(e) {
  return {
    id: e.id,
    name: e.name,
    type: e.type,
    relationship: e.relationship || null,
    aliases: entities.aliasesOf(e),
    active_facts: entities.factCount(e.id, 'active'),
    org: e.org_id ? (entities.get(e.org_id) || {}).name || null : null
  };
}

class EntityListTool extends BaseEntityTool {
  constructor() {
    super();
    this.name = 'entity_list';
    this.description =
      'List the people, organisations, animals and devices you keep facts about. ' +
      'Call this when she mentions a client, a person, a pet or a device and you need to know whether you already know it, ' +
      'and before asking her which one she means — check what you hold first.';
    this.parameters = {
      type: 'object',
      properties: {
        type: {
          type: 'string',
          enum: entities.TYPES,
          description: 'Only entities of this kind. Omit for all.'
        },
        name: {
          type: 'string',
          description: 'Only entities whose name or alias contains this. Use it to check whether a mention is already known.'
        }
      },
      required: []
    };
  }

  async execute(args = {}) {
    const { type = null, name = null } = args || {};
    let rows = entities.list();
    if (type) rows = rows.filter(e => e.type === type);
    if (name) {
      const needle = String(name).toLowerCase();
      rows = rows.filter(e => entities.namesOf(e).some(n => n.includes(needle) || needle.includes(n)));
    }
    return {
      count: rows.length,
      entities: rows.map(summarise),
      note: rows.length === 0
        ? 'Nothing registered answers to that. If she has just introduced it, it is new — say you will keep it separately rather than asking permission.'
        : undefined
    };
  }
}

class EntityGetTool extends BaseEntityTool {
  constructor() {
    super();
    this.name = 'entity_get';
    this.description =
      'Everything you hold about one person, organisation, animal or device — its details and its facts. ' +
      'Call this before asking her to disambiguate a name: read the candidates first, and ask only if their own facts do not settle it.';
    this.parameters = {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'The entity id from entity_list. The first 8 characters are enough.' },
        name: { type: 'string', description: 'Or its name or alias, if you do not have the id.' },
        limit: { type: 'integer', description: 'How many facts to return. Default 20, maximum 50.' }
      },
      required: []
    };
  }

  async execute(args = {}) {
    const { id = null, name = null } = args || {};
    const limit = Math.min(Math.max(parseInt(args.limit, 10) || 20, 1), 50);

    let row = null;
    if (id) {
      row = entities.get(id) || entities.list({ includeMerged: true }).find(e => e.id.startsWith(String(id))) || null;
    }
    if (!row && name) {
      const r = entities.resolve(String(name));
      if (r.tier === 'clear') row = r.entity;
      else if (r.tier === 'ambiguous') {
        return {
          found: false,
          ambiguous: true,
          candidates: r.candidates.map(summarise),
          note: `More than one thing answers to "${name}". Read their facts and ask her which she meant — do not pick.`
        };
      }
    }
    if (!row) return { found: false, note: 'No entity by that id or name. If she has just introduced it, it is new.' };

    // A merged entity still answers, and says where it went, so a stale id in
    // an older fact does not read as a hole.
    const merged = row.status === 'merged' && row.merged_into ? entities.get(row.merged_into) : null;
    const facts = entities.getFactsForEntity(merged ? merged.id : row.id, { status: 'active', limit });
    const flagged = entities.getFactsForEntity(merged ? merged.id : row.id, { status: 'flagged-unverified-subject', limit: 10 });

    return {
      found: true,
      entity: summarise(row),
      merged_into: merged ? summarise(merged) : null,
      locks: entities.entityLocks(row.id).map(l => l.category),
      facts: facts.map(f => ({ id: f.id, content: f.content, salience: f.salience, created_at: f.created_at })),
      held_back: flagged.length
        ? flagged.map(f => ({ id: f.id, content: f.content,
            note: 'held as unverified-subject — the source did not attribute this to it, so it is not something you know' }))
        : undefined
    };
  }
}

module.exports = { EntityListTool, EntityGetTool };
