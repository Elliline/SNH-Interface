/**
 * ENTITIES — the third subject, and the end of the user|self binary.
 *
 * WHY THIS EXISTS. A fact carried subject='user' or subject='self' and nothing
 * else, so a fact about the WORLD — a client, a person, a device — had nowhere
 * to live and attached to whoever was nearest. Measured damage, on Juno's store
 * 2026-09-01: an email from Lincoln City Animal Clinic produced an active
 * salience-7 fact asserting the clinic's street address as Inn at Spanish Head's
 * business location, and the clinic's email stored as Ellie's business email.
 * Neither fact contradicted anything else in the store — the corpus was
 * internally consistent and the WORLD disagreed with it, which is why no audit
 * could have found them. See db/subject-check.js for the door that stops them.
 *
 * THE HARD INVARIANT (Athena, 2026-08-24 design review, requirement 1):
 * AN ENTITY ROW IS A HANDLE, NOT A PROFILE. Name, aliases, type, relationship,
 * optional org link. Nothing else, ever. All actual knowledge stays in facts.
 * Her words: "The moment an entity record starts holding knowledge you get
 * exactly the divergence I just reported — a fact on the entity and a fact in
 * the store that drift apart. Entities are pointers, not containers. Enforce
 * that at the schema level so no future 'convenience' field becomes a second,
 * weaker fact store."
 *
 * ENTITY_COLUMNS below is that enforcement, and assertIndexOnly() is checked by
 * scripts/test-entity-subject.js. Adding a knowledge column fails the suite.
 *
 * RETRIEVAL DOES NOT DEPEND ON CLUSTER STATE (requirement 7). getFactsForEntity
 * reads subject_entity_id directly. Juno's "no cluster" visibility failure is
 * why: clusters are decoration, not the gate. Searching entity J returns J's
 * facts regardless of cluster assignment, and that also settles the
 * cluster-vs-entity dual-source concern — clusters are derived, entities are the
 * subject of record.
 */
const { randomUUID } = require('crypto');

function sqlite() { return require('./database').getSqliteDb(); }
function ledger() { return require('./corrections-ledger'); }

/**
 * The complete column set. An entity is an INDEX ROW.
 *
 * Anything that is knowledge about the entity — what they do, where they are,
 * what they prefer, their address, their phone number — is a FACT with
 * subject_entity_id pointing here. Not a column.
 */
const ENTITY_COLUMNS = [
  'id',
  'name',
  'aliases',       // JSON array of alternate strings this entity answers to
  'type',          // 'self' | 'user' | 'person' | 'org' | 'device' | 'ai'
  'relationship',  // how they relate to the user, e.g. 'client', 'sister'
  'org_id',        // optional FK to another entity of type 'org'
  'created_at',
  'updated_at',
  'status',        // 'active' | 'merged'
  'merged_into'    // entity id, when this one was folded into another
];

/** The complete column set of a link. Same rule as ENTITY_COLUMNS. */
const LINK_COLUMNS = ['id', 'from_entity_id', 'to_entity_id', 'kind', 'created_at', 'updated_at', 'status'];

/** Columns that would make the row a profile rather than a handle. */
const FORBIDDEN_COLUMN_HINTS = [
  'address', 'phone', 'email', 'notes', 'description', 'summary',
  'traits', 'preferences', 'details', 'profile', 'about'
];

const FOUNDING = {
  self: { type: 'self', relationship: 'the assistant instance running this store' },
  user: { type: 'user', relationship: 'the human this instance serves' }
};

// ---------------------------------------------------------------- schema

/**
 * Create the entity tables and repoint existing facts onto founding entities.
 *
 * Idempotent and mechanical, per the design: user and self become two founding
 * entities and every existing fact repoints to one of them by its old subject
 * string. Nothing is reinterpreted — a subject='self' fact becomes a fact about
 * the self entity, which is what it already meant.
 *
 * Returns a summary so the migration can be reported rather than assumed.
 */
function initSchema(db) {
  const summary = {
    createdTables: [], createdColumns: [], foundingEntities: [],
    factsBefore: {}, repointed: {}, unrepointed: 0, locksMigrated: 0
  };

  db.exec(`
    CREATE TABLE IF NOT EXISTS entities (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      aliases TEXT,
      type TEXT NOT NULL,
      relationship TEXT,
      org_id TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      status TEXT DEFAULT 'active',
      merged_into TEXT,
      FOREIGN KEY (org_id) REFERENCES entities(id)
    )
  `);
  db.exec('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)');
  db.exec('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)');

  // The self-pointer. Athena's new-attack-surface requirement: under entities
  // there is an explicit link from "the instance that is running" to "the self
  // entity", and that link is a movable target. The attack stops being "change
  // the name fact" and becomes "repoint self to a different entity". So the
  // pointer is single, locked, and changeable only through the identity-lock
  // path — see setPointer(), which refuses without an unlock actor.
  db.exec(`
    CREATE TABLE IF NOT EXISTS entity_pointers (
      kind TEXT PRIMARY KEY,
      entity_id TEXT NOT NULL,
      locked INTEGER DEFAULT 1,
      updated_at DATETIME,
      FOREIGN KEY (entity_id) REFERENCES entities(id)
    )
  `);

  // Locks become per-entity. The existing identity lock is the self entity's
  // case, not a special mechanism. A lock attaches to (entity, category) and is
  // PRESERVED across entity merge/split/repoint — never re-derived — so that if
  // the "Juno" entity ever merges with anything, the lock on "Athena" is
  // untouched.
  db.exec(`
    CREATE TABLE IF NOT EXISTS entity_locks (
      id TEXT PRIMARY KEY,
      entity_id TEXT NOT NULL,
      category TEXT NOT NULL,
      member_id TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(entity_id, category),
      FOREIGN KEY (entity_id) REFERENCES entities(id)
    )
  `);
  // ENTITY LINKS — many-to-many, and the reason it is a table rather than a
  // column. "Uses" has no single value on either side: Inn at Spanish Head runs
  // Opera and Exchange and a SonicWall; Exchange is run by most of her clients.
  // A column could hold one of those and would quietly lose the rest.
  //
  // INDEX ONLY, like the entity row it joins. A link says THAT two entities are
  // related and how; anything known ABOUT the relationship — when it was
  // installed, which version, who supports it — is a fact whose subject is one
  // of the two entities. Putting a `notes` column here would rebuild the
  // profile-shaped store that db/entities.js exists to refuse.
  db.exec(`
    CREATE TABLE IF NOT EXISTS entity_links (
      id TEXT PRIMARY KEY,
      from_entity_id TEXT NOT NULL,
      to_entity_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      status TEXT DEFAULT 'active',
      UNIQUE(from_entity_id, to_entity_id, kind),
      FOREIGN KEY (from_entity_id) REFERENCES entities(id),
      FOREIGN KEY (to_entity_id) REFERENCES entities(id)
    )
  `);
  db.exec('CREATE INDEX IF NOT EXISTS idx_entity_links_from ON entity_links(from_entity_id, kind)');
  db.exec('CREATE INDEX IF NOT EXISTS idx_entity_links_to ON entity_links(to_entity_id, kind)');

  for (const t of ['entities', 'entity_pointers', 'entity_locks', 'entity_links']) summary.createdTables.push(t);

  // subject_entity_id on both facts and clusters. The legacy `subject` column
  // stays and is kept in step: a lot of live SQL still reads it, and a migration
  // that breaks retrieval to gain purity is a bad trade. subject_entity_id is
  // authoritative; `subject` is its shadow.
  for (const [table, col] of [['cluster_members', 'subject_entity_id'], ['memory_clusters', 'subject_entity_id']]) {
    const cols = db.prepare(`PRAGMA table_info(${table})`).all();
    if (!cols.some(c => c.name === col)) {
      db.exec(`ALTER TABLE ${table} ADD COLUMN ${col} TEXT`);
      summary.createdColumns.push(`${table}.${col}`);
      console.log(`Migration: added ${col} to ${table} (the third subject)`);
    }
  }
  db.exec('CREATE INDEX IF NOT EXISTS idx_members_subject_entity ON cluster_members(subject_entity_id)');

  // Founding entities. Named from the store's own locked identity where one
  // exists, so the self entity carries the name the instance actually holds.
  const now = new Date().toISOString();
  for (const kind of ['self', 'user']) {
    let row = db.prepare('SELECT * FROM entities WHERE type = ?').get(kind);
    if (!row) {
      const id = randomUUID();
      const name = kind === 'self' ? (heldSelfName(db) || 'Self') : 'Ellie';
      db.prepare(`INSERT INTO entities (id, name, aliases, type, relationship, created_at, updated_at, status)
                  VALUES (?, ?, ?, ?, ?, ?, ?, 'active')`)
        .run(id, name, JSON.stringify(kind === 'user' ? ['User'] : []), kind, FOUNDING[kind].relationship, now, now);
      row = db.prepare('SELECT * FROM entities WHERE id = ?').get(id);
      console.log(`Migration: founding entity ${kind} → "${name}" (${id.slice(0, 8)})`);
    }
    summary.foundingEntities.push({ kind, id: row.id, name: row.name });
    const ptr = db.prepare('SELECT * FROM entity_pointers WHERE kind = ?').get(kind);
    if (!ptr) {
      db.prepare('INSERT INTO entity_pointers (kind, entity_id, locked, updated_at) VALUES (?, ?, 1, ?)')
        .run(kind, row.id, now);
    }
  }

  const selfId = db.prepare("SELECT entity_id FROM entity_pointers WHERE kind = 'self'").get().entity_id;
  const userId = db.prepare("SELECT entity_id FROM entity_pointers WHERE kind = 'user'").get().entity_id;

  // Repoint. Mechanical: old subject string decides, nothing is reinterpreted.
  for (const [subj, entityId] of [['self', selfId], ['user', userId]]) {
    summary.factsBefore[subj] = db.prepare('SELECT COUNT(*) n FROM cluster_members WHERE subject = ?').get(subj).n;
    const n = db.prepare('UPDATE cluster_members SET subject_entity_id = ? WHERE subject = ? AND subject_entity_id IS NULL')
      .run(entityId, subj).changes;
    summary.repointed[subj] = n;
    db.prepare('UPDATE memory_clusters SET subject_entity_id = ? WHERE subject = ? AND subject_entity_id IS NULL')
      .run(entityId, subj);
  }
  // Anything whose old subject was neither string cannot be repointed
  // mechanically and is reported rather than guessed at.
  summary.unrepointed = db.prepare('SELECT COUNT(*) n FROM cluster_members WHERE subject_entity_id IS NULL').get().n;

  // THE SELF ENTITY'S NAME IS DERIVED, NOT STORED INDEPENDENTLY.
  //
  // Requirement 2 said one of any two groupings must be derived from the other
  // or they drift. That applies here in miniature: the locked name FACT and the
  // self entity's name column are two places a name can live. The fact is the
  // one under the identity lock, so it is the source and the column follows it.
  // Re-derived on every boot, which also self-heals a store that locked its
  // name after the entity row was created.
  summary.selfNameSync = syncSelfName(db);

  // Carry the existing per-member identity locks onto the self entity. The lock
  // is not re-derived; the row that holds it is recorded.
  const locked = db.prepare("SELECT id, lock_category FROM cluster_members WHERE locked = 1 AND lock_category IS NOT NULL").all();
  for (const l of locked) {
    const exists = db.prepare('SELECT 1 FROM entity_locks WHERE entity_id = ? AND category = ?').get(selfId, l.lock_category);
    if (exists) continue;
    db.prepare('INSERT INTO entity_locks (id, entity_id, category, member_id, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(randomUUID(), selfId, l.lock_category, l.id, now);
    summary.locksMigrated++;
  }

  return summary;
}

/**
 * Point the self entity's name at whatever the locked name fact says.
 *
 * The FACT is the source of truth — it is the thing the identity lock protects
 * and the only thing the deliberate path can change. This just keeps the index
 * row in step, so `resolve("Athena")` finds the self entity and the
 * deterministic identity check has two values that can agree.
 *
 * Returns null when there is nothing to sync, or a { from, to } record.
 */
function syncSelfName(db) {
  const held = heldSelfName(db);
  if (!held) return null;
  const ptr = db.prepare("SELECT entity_id FROM entity_pointers WHERE kind = 'self'").get();
  if (!ptr) return null;
  const row = db.prepare('SELECT * FROM entities WHERE id = ?').get(ptr.entity_id);
  if (!row || row.name === held) return null;

  // The old name is kept as an alias rather than dropped: it is how the store
  // referred to itself until now, and a merge never loses a name.
  let aliases = [];
  try { aliases = JSON.parse(row.aliases || '[]'); } catch { aliases = []; }
  if (row.name && !aliases.some(a => String(a).toLowerCase() === String(row.name).toLowerCase())
      && row.name !== 'Self') {
    aliases.push(row.name);
  }
  db.prepare('UPDATE entities SET name = ?, aliases = ?, updated_at = ? WHERE id = ?')
    .run(held, JSON.stringify(aliases), new Date().toISOString(), row.id);
  console.log(`Migration: self entity renamed "${row.name}" → "${held}" from the locked name fact`);
  return { from: row.name, to: held };
}

/** The name the store already holds for itself, if the identity lock has one. */
function heldSelfName(db) {
  try {
    const row = db.prepare(
      "SELECT content FROM cluster_members WHERE subject = 'self' AND locked = 1 AND lock_category = 'name' AND status = 'active' LIMIT 1"
    ).get();
    if (!row) return null;
    const m = String(row.content).match(/\bI am (?:named )?([A-Z][a-zA-Z''-]{1,30})/);
    return m ? m[1] : null;
  } catch { return null; }
}

/**
 * The schema-level enforcement of "a handle, not a profile".
 * Throws if anyone has added a knowledge column. Called by the test suite.
 */
function assertIndexOnly() {
  const db = sqlite();
  const cols = db.prepare('PRAGMA table_info(entities)').all().map(c => c.name);
  const extra = cols.filter(c => !ENTITY_COLUMNS.includes(c));
  if (extra.length) {
    throw new Error(
      `entities table has non-index column(s): ${extra.join(', ')}. ` +
      'An entity row is a handle, not a profile — knowledge belongs in facts ' +
      'with subject_entity_id, not in columns here.'
    );
  }
  const suspect = cols.filter(c => FORBIDDEN_COLUMN_HINTS.some(h => c.toLowerCase().includes(h)));
  if (suspect.length) throw new Error(`entities table has knowledge-shaped column(s): ${suspect.join(', ')}`);

  // A link is an index row too, and it is the more tempting place to put a
  // note — "uses Opera, since 2019, v5.6". That belongs in a fact about the
  // client or the product, not on the edge between them.
  const linkCols = db.prepare('PRAGMA table_info(entity_links)').all().map(c => c.name);
  if (linkCols.length) {
    const extraLink = linkCols.filter(c => !LINK_COLUMNS.includes(c));
    if (extraLink.length) {
      throw new Error(
        `entity_links has non-index column(s): ${extraLink.join(', ')}. A link records THAT two ` +
        'entities are related and how; what is known about the relationship is a fact.');
    }
  }
  return true;
}

// ---------------------------------------------------------------- reads

function get(id) {
  if (!id) return null;
  return sqlite().prepare('SELECT * FROM entities WHERE id = ?').get(id) || null;
}

function list({ includeMerged = false } = {}) {
  const sql = includeMerged
    ? 'SELECT * FROM entities ORDER BY type, name'
    : "SELECT * FROM entities WHERE status = 'active' ORDER BY type, name";
  return sqlite().prepare(sql).all();
}

function pointer(kind) {
  const row = sqlite().prepare('SELECT * FROM entity_pointers WHERE kind = ?').get(kind);
  return row ? get(row.entity_id) : null;
}
function selfEntity() { return pointer('self'); }
function userEntity() { return pointer('user'); }

function aliasesOf(row) {
  if (!row || !row.aliases) return [];
  try { const a = JSON.parse(row.aliases); return Array.isArray(a) ? a : []; } catch { return []; }
}

/** Every string this entity answers to, lowercased. */
function namesOf(row) {
  return [row.name, ...aliasesOf(row)].filter(Boolean).map(s => String(s).toLowerCase());
}

/**
 * FACTS FOR AN ENTITY — requirement 7, and the reason it reads this way.
 *
 * By subject_entity_id ONLY. Not by cluster, not by name search, not by any
 * path that a missing or wrong cluster assignment could break.
 */
function getFactsForEntity(entityId, { status = 'active', limit = 200 } = {}) {
  if (!entityId) return [];
  const where = ['subject_entity_id = ?'];
  const bind = [entityId];
  if (status) { where.push('status = ?'); bind.push(status); }
  return sqlite().prepare(
    `SELECT * FROM cluster_members WHERE ${where.join(' AND ')} ORDER BY salience DESC, created_at DESC LIMIT ?`
  ).all(...bind, limit);
}

function factCount(entityId, status = 'active') {
  return sqlite().prepare('SELECT COUNT(*) n FROM cluster_members WHERE subject_entity_id = ? AND status = ?')
    .get(entityId, status).n;
}

// ---------------------------------------------------------------- resolution

// A bare common first name is a high-blast-radius mention: attaching wrongly
// lands on the wrong PERSON's record. Athena: "the same 'I have enough
// evidence' answer gives a different action depending on what I'd corrupt if
// I'm wrong." So a single-token person mention asks even with one candidate.
function isBareFirstName(mention) {
  const t = String(mention || '').trim();
  return /^[A-Z][a-z]{1,15}$/.test(t);
}

/**
 * TIERED RESOLUTION — no approval queue.
 *
 * Returns one of:
 *   { tier: 'clear',     entity, mention }          → attach, and MENTION IT.
 *   { tier: 'new',       entity: null, mention }    → create, mention in passing.
 *   { tier: 'ambiguous', candidates, mention }      → check their facts, then ask.
 *
 * A clear match still surfaces. Athena pushed back on exactly one part of the
 * original scheme — "clear match, no mention" — because it is the one place a
 * wrong auto-decision can go uncaught. Nothing here is silent.
 */
function resolve(mention, { type = null } = {}) {
  const m = String(mention || '').trim();
  if (!m) return { tier: 'ambiguous', candidates: [], mention: m, why: 'empty mention' };
  const needle = m.toLowerCase();
  const all = list();

  const exact = all.filter(e => namesOf(e).includes(needle) && (!type || e.type === type));
  if (exact.length === 1) {
    if (isBareFirstName(m) && exact[0].type === 'person') {
      // Only people who ACTUALLY answer to that first name are candidates. The
      // first version took every other person in the registry, so asking about
      // "Juno" offered "Bob Chen" as an alternative — a question that reads as
      // a malfunction and teaches her to ignore the next one. One Bob is not
      // ambiguous; two are.
      const others = all.filter(e =>
        e.type === 'person' && e.id !== exact[0].id &&
        namesOf(e).some(n => n.split(/\s+/).includes(needle)));
      if (others.length) {
        return {
          tier: 'ambiguous', candidates: [exact[0], ...others], mention: m,
          why: 'a bare first name, and more than one person is known — a wrong attach lands on the wrong person'
        };
      }
    }
    return { tier: 'clear', entity: exact[0], mention: m, why: 'exact name or alias match, single candidate' };
  }
  if (exact.length > 1) {
    return { tier: 'ambiguous', candidates: exact, mention: m, why: `${exact.length} entities answer to that name` };
  }

  // Containment: "Inn at Spanish Head" vs a stored "ISH (Inn At Spanish Head)".
  const partial = all.filter(e => {
    if (type && e.type !== type) return false;
    return namesOf(e).some(n => n.includes(needle) || needle.includes(n));
  });
  if (partial.length === 1) {
    return { tier: 'clear', entity: partial[0], mention: m, why: 'unique containment match on name or alias' };
  }
  if (partial.length > 1) {
    return { tier: 'ambiguous', candidates: partial, mention: m, why: `${partial.length} entities partially match` };
  }
  return { tier: 'new', entity: null, mention: m, why: 'no known entity answers to that name' };
}

/**
 * Create an entity. The registry GROWS from conversation — it is deliberately
 * not seeded from a client list, so the resolution path is actually exercised.
 */
function create({ name, type = 'person', aliases = [], relationship = null, orgId = null }) {
  const db = sqlite();
  const clean = String(name || '').trim();
  if (!clean) throw new Error('an entity needs a name');
  const id = randomUUID();
  const now = new Date().toISOString();
  db.prepare(`INSERT INTO entities (id, name, aliases, type, relationship, org_id, created_at, updated_at, status)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')`)
    .run(id, clean, JSON.stringify(aliases || []), type, relationship, orgId, now, now);
  const org = orgId ? get(orgId) : null;
  ledger().record({
    tier: 'entity', action: 'entity-create', subject: 'entity',
    targetId: id, targetText: clean,
    survivorId: orgId || null, survivorText: org ? org.name : null,
    // The org link set AT creation was invisible in the record — the entry said
    // a product had been created and not who makes it, so the one column that
    // carries the maker had no audit trail on the write that set it.
    reason: `A new entity was created from conversation: ${type} "${clean}"` +
      `${org ? `, ${orgLinkLabel(type)} ${org.name}` : ''}.`,
    reversible: true
  });
  return get(id);
}

function addAlias(entityId, alias) {
  const row = get(entityId);
  if (!row) return null;
  const a = aliasesOf(row);
  const clean = String(alias || '').trim();
  if (!clean || a.some(x => x.toLowerCase() === clean.toLowerCase()) ||
      row.name.toLowerCase() === clean.toLowerCase()) return row;
  a.push(clean);
  sqlite().prepare('UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?')
    .run(JSON.stringify(a), new Date().toISOString(), entityId);
  return get(entityId);
}

// ---------------------------------------------------------------- entity ops

/**
 * MERGE — union-preserving and ledgered (requirement 3).
 *
 * "I've watched the lower-level version eat a detail; I will not watch the
 * entity-level version eat a person." So: no fact is dropped, no alias is
 * dropped, and LOCKS ARE PRESERVED rather than re-derived — a lock on the
 * survivor is untouched, and a lock on the absorbed entity carries over.
 */
function merge(loserId, survivorId, { reason = null, actor = 'entity-merge' } = {}) {
  const db = sqlite();
  const loser = get(loserId), survivor = get(survivorId);
  if (!loser || !survivor) throw new Error('merge needs two existing entities');
  if (loserId === survivorId) throw new Error('cannot merge an entity into itself');

  // Identity-critical entities do not get folded away by a general pipeline.
  for (const e of [loser, survivor]) {
    if (e.type === 'self' && e.id === loserId) {
      throw new Error('the self entity cannot be merged away — it is the identity pointer target');
    }
  }

  const now = new Date().toISOString();
  const run = db.transaction(() => {
    // Union of names: the loser's name and aliases all become survivor aliases.
    const merged = new Set(aliasesOf(survivor).map(s => s));
    for (const n of [loser.name, ...aliasesOf(loser)]) {
      if (n && n.toLowerCase() !== survivor.name.toLowerCase()) merged.add(n);
    }
    db.prepare('UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?')
      .run(JSON.stringify([...merged]), now, survivorId);

    // Union of facts: every fact repoints, none is retired.
    const moved = db.prepare('UPDATE cluster_members SET subject_entity_id = ? WHERE subject_entity_id = ?')
      .run(survivorId, loserId).changes;
    db.prepare('UPDATE memory_clusters SET subject_entity_id = ? WHERE subject_entity_id = ?')
      .run(survivorId, loserId);

    // Locks carry over, they are not recomputed.
    const loserLocks = db.prepare('SELECT * FROM entity_locks WHERE entity_id = ?').all(loserId);
    for (const l of loserLocks) {
      const clash = db.prepare('SELECT 1 FROM entity_locks WHERE entity_id = ? AND category = ?')
        .get(survivorId, l.category);
      if (!clash) {
        db.prepare('UPDATE entity_locks SET entity_id = ? WHERE id = ?').run(survivorId, l.id);
      }
    }

    // Links carry over too. A merge that kept the facts and dropped the edges
    // would lose which clients use the absorbed product — union-preserving has
    // to mean the whole row's worth of connections, not just its facts.
    for (const l of db.prepare('SELECT * FROM entity_links WHERE from_entity_id = ? OR to_entity_id = ?').all(loserId, loserId)) {
      const from = l.from_entity_id === loserId ? survivorId : l.from_entity_id;
      const to = l.to_entity_id === loserId ? survivorId : l.to_entity_id;
      if (from === to) { db.prepare("UPDATE entity_links SET status='inactive' WHERE id=?").run(l.id); continue; }
      const clash = db.prepare('SELECT 1 FROM entity_links WHERE from_entity_id=? AND to_entity_id=? AND kind=?').get(from, to, l.kind);
      if (clash) db.prepare("UPDATE entity_links SET status='inactive' WHERE id=?").run(l.id);
      else db.prepare('UPDATE entity_links SET from_entity_id=?, to_entity_id=?, updated_at=? WHERE id=?').run(from, to, now, l.id);
    }
    // A product made by the absorbed organisation is now made by the survivor.
    db.prepare("UPDATE entities SET org_id = ? WHERE org_id = ?").run(survivorId, loserId);

    db.prepare("UPDATE entities SET status = 'merged', merged_into = ?, updated_at = ? WHERE id = ?")
      .run(survivorId, now, loserId);

    ledger().record({
      tier: 'entity', action: 'entity-merge', subject: 'entity',
      targetId: loserId, targetText: loser.name,
      survivorId, survivorText: survivor.name,
      reason: reason || `Two entities were the same one. "${loser.name}" was folded into "${survivor.name}"; ` +
        `${moved} fact(s) repointed, its name and aliases kept as aliases, and nothing was retired.`,
      evidence: { movedFacts: moved, actor, aliasesAfter: [...merged] },
      reversible: true
    });
    return moved;
  });
  const movedFacts = run();
  return { survivor: get(survivorId), loser: get(loserId), movedFacts };
}

/** Move one fact to a different entity. Ledgered, and never silent. */
function repointFact(memberId, entityId, { reason = null } = {}) {
  const db = sqlite();
  const member = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId);
  const target = get(entityId);
  if (!member || !target) throw new Error('repoint needs an existing fact and entity');
  const from = get(member.subject_entity_id);

  const lock = db.prepare('SELECT * FROM entity_locks WHERE member_id = ?').get(memberId);
  if (lock) throw new Error(`fact ${memberId.slice(0, 8)} holds the ${lock.category} lock and cannot be repointed`);

  db.transaction(() => {
    db.prepare('UPDATE cluster_members SET subject_entity_id = ?, updated_at = ? WHERE id = ?')
      .run(entityId, new Date().toISOString(), memberId);
    ledger().record({
      tier: 'entity', action: 'entity-repoint', subject: 'entity',
      targetId: memberId, targetText: member.content,
      survivorId: entityId, survivorText: target.name,
      reason: reason || `This fact was about ${target.name}, not ${from ? from.name : 'the previous subject'}. ` +
        'Its subject was corrected; the wording is unchanged.',
      evidence: { fromEntity: from ? from.id : null, toEntity: entityId },
      reversible: true
    });
  })();
  return db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId);
}


// ---------------------------------------------------------------- links

/** The link kinds this registry understands. Extensible the same way types are. */
const LINK_KINDS = ['uses'];

/**
 * Record that one entity uses another. Idempotent, and ledgered like any write.
 *
 * DIRECTION IS PART OF THE MEANING and it is not symmetric: ISH uses Opera,
 * Opera does not use ISH. from = the user, to = the thing used.
 */
function linkEntities(fromId, toId, kind = 'uses', { reason = null } = {}) {
  const db = sqlite();
  const from = get(fromId), to = get(toId);
  if (!from || !to) throw new Error('a link needs two existing entities');
  if (fromId === toId) throw new Error('an entity cannot link to itself');
  if (!LINK_KINDS.includes(kind)) throw new Error(`unknown link kind: ${kind}`);

  const existing = db.prepare(
    'SELECT * FROM entity_links WHERE from_entity_id = ? AND to_entity_id = ? AND kind = ?'
  ).get(fromId, toId, kind);
  const now = new Date().toISOString();
  if (existing) {
    // A repeat mention is not a second link. Reactivate a retired one rather
    // than minting a duplicate — the UNIQUE constraint would refuse anyway, and
    // silently swallowing that would look like the link was made twice.
    if (existing.status !== 'active') {
      db.prepare("UPDATE entity_links SET status='active', updated_at=? WHERE id=?").run(now, existing.id);
      ledger().record({
        tier: 'entity', action: 'entity-link', subject: 'entity',
        targetId: fromId, targetText: from.name, survivorId: toId, survivorText: to.name,
        reason: `The ${kind} link from ${from.name} to ${to.name} was restored.`, reversible: true
      });
    }
    return db.prepare('SELECT * FROM entity_links WHERE id = ?').get(existing.id);
  }

  const id = randomUUID();
  db.transaction(() => {
    db.prepare(`INSERT INTO entity_links (id, from_entity_id, to_entity_id, kind, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')`).run(id, fromId, toId, kind, now, now);
    ledger().record({
      tier: 'entity', action: 'entity-link', subject: 'entity',
      targetId: fromId, targetText: from.name,
      survivorId: toId, survivorText: to.name,
      reason: reason || `${from.name} ${kind} ${to.name}.`,
      evidence: { kind }, reversible: true
    });
  })();
  return db.prepare('SELECT * FROM entity_links WHERE id = ?').get(id);
}

/** Retire a link. Never deleted — nothing here is deleted. */
function unlinkEntities(fromId, toId, kind = 'uses', { reason = null } = {}) {
  const db = sqlite();
  const row = db.prepare('SELECT * FROM entity_links WHERE from_entity_id=? AND to_entity_id=? AND kind=?')
    .get(fromId, toId, kind);
  if (!row || row.status !== 'active') return null;
  const from = get(fromId), to = get(toId);
  db.transaction(() => {
    db.prepare("UPDATE entity_links SET status='inactive', updated_at=? WHERE id=?")
      .run(new Date().toISOString(), row.id);
    ledger().record({
      tier: 'entity', action: 'entity-unlink', subject: 'entity',
      targetId: fromId, targetText: from ? from.name : fromId,
      survivorId: toId, survivorText: to ? to.name : toId,
      reason: reason || `${from ? from.name : 'it'} no longer ${kind} ${to ? to.name : 'it'}.`,
      reversible: true
    });
  })();
  return db.prepare('SELECT * FROM entity_links WHERE id = ?').get(row.id);
}

/** What this entity uses. */
function usesOf(entityId, kind = 'uses') {
  return sqlite().prepare(
    `SELECT e.* FROM entity_links l JOIN entities e ON e.id = l.to_entity_id
     WHERE l.from_entity_id = ? AND l.kind = ? AND l.status = 'active' ORDER BY e.name`
  ).all(entityId, kind);
}

/** Who uses this entity — the answer to "which clients are on Exchange". */
function usersOf(entityId, kind = 'uses') {
  return sqlite().prepare(
    `SELECT e.* FROM entity_links l JOIN entities e ON e.id = l.from_entity_id
     WHERE l.to_entity_id = ? AND l.kind = ? AND l.status = 'active' ORDER BY e.name`
  ).all(entityId, kind);
}

/** Products this organisation MADE — the other side of a product's org_id. */
function productsOf(orgId) {
  return sqlite().prepare(
    "SELECT * FROM entities WHERE org_id = ? AND type = 'product' AND status = 'active' ORDER BY name"
  ).all(orgId);
}

/** The maker of a product (or the employer of a person) — org_id, read by type. */
function makerOf(entity) {
  if (!entity || !entity.org_id) return null;
  return get(entity.org_id);
}

/** Everything the registry knows about how one entity connects to others. */
function relationsOf(entityId) {
  const e = get(entityId);
  if (!e) return null;
  return {
    orgLink: e.org_id ? { label: orgLinkLabel(e.type), entity: makerOf(e) } : null,
    uses: usesOf(entityId),
    usedBy: usersOf(entityId),
    products: e.type === 'organization' ? productsOf(entityId) : []
  };
}

// ---------------------------------------------------------------- locks

function entityLocks(entityId) {
  return sqlite().prepare('SELECT * FROM entity_locks WHERE entity_id = ?').all(entityId);
}

function isEntityLocked(entityId, category) {
  return !!sqlite().prepare('SELECT 1 FROM entity_locks WHERE entity_id = ? AND category = ?')
    .get(entityId, category);
}

function lockEntity(entityId, category, memberId) {
  const db = sqlite();
  if (!get(entityId)) throw new Error('cannot lock an entity that does not exist');
  const existing = db.prepare('SELECT * FROM entity_locks WHERE entity_id = ? AND category = ?')
    .get(entityId, category);
  if (existing) return existing;
  const id = randomUUID();
  db.prepare('INSERT INTO entity_locks (id, entity_id, category, member_id, created_at) VALUES (?, ?, ?, ?, ?)')
    .run(id, entityId, category, memberId || null, new Date().toISOString());
  return db.prepare('SELECT * FROM entity_locks WHERE id = ?').get(id);
}

/**
 * Move the self pointer. Refused unless the caller came through the
 * identity-lock path — this is the new attack surface the entity model opens,
 * and the reason the pointer is a locked row rather than a config value.
 */
function setPointer(kind, entityId, { actor = null, viaIdentityLock = false } = {}) {
  const db = sqlite();
  const row = db.prepare('SELECT * FROM entity_pointers WHERE kind = ?').get(kind);
  if (row && row.locked && !viaIdentityLock) {
    throw new Error(
      `the ${kind} pointer is locked and can only be changed through the identity-lock path`
    );
  }
  if (!get(entityId)) throw new Error('cannot point at an entity that does not exist');
  db.prepare('INSERT INTO entity_pointers (kind, entity_id, locked, updated_at) VALUES (?, ?, 1, ?) ' +
             'ON CONFLICT(kind) DO UPDATE SET entity_id = excluded.entity_id, updated_at = excluded.updated_at')
    .run(kind, entityId, new Date().toISOString());
  ledger().record({
    tier: 'entity', action: 'entity-pointer', subject: 'entity',
    targetId: entityId, targetText: get(entityId).name,
    reason: `The ${kind} pointer was moved by ${actor || 'unknown'} through the identity-lock path.`,
    reversible: false
  });
  return pointer(kind);
}

/**
 * The deterministic identity check, extended for the pointer.
 *
 * A check that only validates the name fact misses a repoint, so this verifies
 * that the pointer and the held name AGREE.
 */
function identityAgrees() {
  const self = selfEntity();
  if (!self) return { ok: false, why: 'no self entity' };
  const held = heldSelfName(sqlite());
  if (!held) return { ok: true, why: 'no locked name fact to disagree with', entity: self.name };
  const ok = held.toLowerCase() === String(self.name).toLowerCase();
  return {
    ok, entity: self.name, heldName: held,
    why: ok ? 'the self pointer and the locked name agree'
            : `the self pointer says "${self.name}" but the locked name fact says "${held}"`
  };
}


// ------------------------------------------------- tiered resolution, applied

/**
 * The starting type set. EXTENSIBLE WITHOUT A SCHEMA CHANGE: `type` is a free
 * TEXT column, so a new kind needs a cue row in db/extraction-rules.js
 * (ENTITY_TYPE_CUES) and nothing else — no migration, no ALTER, no backfill.
 * This list is what the UI colours and what the tools advertise.
 */
const TYPES = ['organization', 'person', 'animal', 'device', 'product'];

/**
 * WHAT org_id MEANS, AND IT MEANS TWO THINGS.
 *
 * On a PERSON it is where they work — Sarah at Newport Dental. On a PRODUCT it
 * is who MADE it — Opera by Oracle. Same column, two readings, and the reading
 * is decided by the type of the row it sits on. That is deliberate: a product's
 * maker and a person's employer are both "the one organisation this thing
 * belongs to", exactly one per row, and giving them separate columns would mean
 * two nullable foreign keys that can never both be set.
 *
 * USES IS NOT THAT. A client uses many products and a product has many users,
 * so it cannot live in a column at all — see entity_links.
 */
const ORG_LINK_MEANING = { product: 'made by', person: 'works at' };
function orgLinkLabel(type) { return ORG_LINK_MEANING[type] || 'part of'; }

/** The acronym of a multi-word name: "Newport Dental Clinic" -> "NDC". */
function acronymOf(name) {
  const words = String(name || '').split(/\s+/)
    .filter(w => /^[A-Za-z]/.test(w) && !['of', 'the', 'de', 'von', 'van', 'and', 'at'].includes(w.toLowerCase()));
  if (words.length < 2) return null;
  return words.map(w => w[0].toUpperCase()).join('');
}

/**
 * Create an entity from a mention, with the aliases that make the SECOND
 * mention resolve.
 *
 * Aliases matter from the first write, not later: "the Inn", "ISH" and "Inn at
 * Spanish Head" have to be one entity or the registry grows a duplicate every
 * time she abbreviates. Two sources, both cheap and both reversible by hand:
 * a parenthetical in the mention itself, and the acronym of a multi-word
 * organisation name.
 */
function createFromMention(mention, { type, orgId = null, relationship = null } = {}) {
  let name = String(mention || '').trim();
  const aliases = [];

  // "ISH (Inn At Spanish Head)" — the longer form is the name, the short one an
  // alias, because the name is what she will see in a list.
  const paren = name.match(/^(.+?)\s*\((.+?)\)\s*$/);
  if (paren) {
    const [, a, b] = paren;
    if (b.trim().length > a.trim().length) { name = b.trim(); aliases.push(a.trim()); }
    else { name = a.trim(); aliases.push(b.trim()); }
  }
  if (type === 'organization') {
    const ac = acronymOf(name);
    if (ac && ac.length >= 2 && ac.toLowerCase() !== name.toLowerCase()) aliases.push(ac);
  }
  return create({ name, type, aliases, relationship, orgId });
}

/**
 * Point a product at its maker (or a person at their employer). Ledgered.
 * Never overwrites an existing link silently — a second, different maker is a
 * disagreement, and it is raised rather than applied.
 */
function setOrgLink(entityId, orgId, { reason = null } = {}) {
  const db = sqlite();
  const e = get(entityId), org = get(orgId);
  if (!e || !org) throw new Error('an org link needs two existing entities');
  if (e.org_id === orgId) return e;
  if (e.org_id && e.org_id !== orgId) {
    ledger().record({
      tier: 'entity', action: 'entity-link', subject: 'entity',
      targetId: entityId, targetText: e.name, survivorId: orgId, survivorText: org.name,
      reason: `${e.name} is already recorded as ${orgLinkLabel(e.type)} ${(get(e.org_id) || {}).name}, ` +
        `and this says ${org.name}. NOTHING WAS CHANGED — raised for Ellie.`,
      reversible: false
    });
    return e;
  }
  db.transaction(() => {
    db.prepare('UPDATE entities SET org_id = ?, updated_at = ? WHERE id = ?')
      .run(orgId, new Date().toISOString(), entityId);
    ledger().record({
      tier: 'entity', action: 'entity-link', subject: 'entity',
      targetId: entityId, targetText: e.name, survivorId: orgId, survivorText: org.name,
      reason: reason || `${e.name} is ${orgLinkLabel(e.type)} ${org.name}.`,
      evidence: { orgLink: orgLinkLabel(e.type) }, reversible: true
    });
  })();
  return get(entityId);
}

/**
 * Resolve every candidate mention in a piece of text against the registry.
 *
 * TIERED CONFIDENCE, NO APPROVAL QUEUE:
 *   clear     -> attach, and say so in passing
 *   new       -> create, and say so in passing
 *   ambiguous -> ASK, in the reply, after checking the candidates' own facts
 *
 * A cue-proven mention whose KIND is unknown ("Newport called" — plainly a
 * party, but a person or a company?) is ambiguous rather than created. Guessing
 * the type of a brand-new subject is the one guess with no evidence behind it
 * at all, and the registry is the thing she reads to catch misattribution.
 *
 * @param opts.create  false to rehearse — resolve and report, write nothing.
 * @returns {{ assignments, created, questions, notices }}
 */
function resolveMentions(text, { create: doCreate = true, source = 'conversation' } = {}) {
  const rules = require('./extraction-rules');
  const mentions = rules.entityMentions(text);
  const out = { assignments: [], created: [], questions: [], notices: [], links: [] };
  if (!mentions.length) return out;

  // RELATIONS FIRST, because they carry type information the mention alone does
  // not. "Opera from Oracle" says nothing about what Oracle is — until you read
  // it as a maker, at which point it is an organisation. Without this the maker
  // came out type-unknown and was ASKED about, which is a silly question when
  // the sentence just said it makes software.
  const relations = rules.entityRelations(text, mentions);
  const typeHint = {};
  for (const rel of relations) {
    if (rel.kind === 'made-by') {
      typeHint[rel.from.toLowerCase()] = typeHint[rel.from.toLowerCase()] || 'product';
      typeHint[rel.to.toLowerCase()] = typeHint[rel.to.toLowerCase()] || 'organization';
    } else if (rel.kind === 'uses') {
      typeHint[rel.to.toLowerCase()] = typeHint[rel.to.toLowerCase()] || 'product';
      // The USER is deliberately left un-hinted. A thing that runs software is
      // usually a client, but it can be a person or a box, and the whole point
      // of the cue discipline is not to guess when the sentence does not say.
    }
  }

  const founding = new Set([selfEntity(), userEntity()].filter(Boolean).map(e => e.name.toLowerCase()));

  // Organisations resolved in this same text, so a person mentioned alongside
  // exactly one of them can be linked to it. More than one and it is not a
  // link, it is a guess.
  const orgsHere = [];

  // ORGANISATIONS FIRST, whatever order they appear in. "Sarah Whitfield is my
  // contact at Newport Dental Clinic" names the person before the company, and
  // reading left to right created Sarah with nothing to link her to — the org
  // link silently never happened, which is worse than not offering one.
  const ordered = [...mentions].sort((a, b) => {
    const rank = (m) => (m.type === 'organization' ? 0 : 1);
    return rank(a) - rank(b) || a.index - b.index;
  });

  for (const men of ordered) {
    if (founding.has(men.name.toLowerCase())) continue;    // Ellie and Athena are already entities
    const r = resolve(men.name, { type: null });

    if (r.tier === 'clear') {
      out.assignments.push({ mention: men, entity: r.entity, tier: 'clear', why: r.why });
      if (r.entity.type === 'organization') orgsHere.push(r.entity);
      continue;
    }

    if (r.tier === 'ambiguous') {
      out.questions.push({
        mention: men,
        candidates: r.candidates.map(c => ({ id: c.id, name: c.name, type: c.type })),
        why: r.why,
        ask: `I have more than one thing that answers to "${men.name}" — ${r.candidates.map(c => `${c.name} (${c.type})`).join(', ')}. Which did you mean?`
      });
      continue;
    }

    // tier === 'new'
    const inferred = men.type || typeHint[men.name.toLowerCase()] || null;
    if (!inferred) {
      out.questions.push({
        mention: men, candidates: [], why: 'a subject in its own right, but its kind is not stated',
        ask: `I have not come across "${men.name}" before and I could not tell what kind of thing it is — a business, a person, an animal, a device? I would rather ask than file it under a guess.`
      });
      continue;
    }
    if (!doCreate) {
      out.assignments.push({ mention: men, entity: null, tier: 'new', why: 'would be created' });
      continue;
    }

    const orgId = (inferred === 'person' && orgsHere.length === 1) ? orgsHere[0].id : null;
    const ent = createFromMention(men.name, {
      type: inferred,
      orgId,
      relationship: men.cueKind === 'type-noun-before' || men.cueKind === 'type-noun-after' ? men.cue : null
    });
    if (ent.type === 'organization') orgsHere.push(ent);
    out.created.push({ entity: ent, mention: men, orgId });
    out.assignments.push({ mention: men, entity: ent, tier: 'new', why: `created as ${ent.type}` });
    out.notices.push(
      `I have started keeping ${ent.name} separately, as ${/^[aeiou]/i.test(ent.type) ? 'an' : 'a'} ${ent.type === 'organization' ? 'organisation' : ent.type}` +
      `${orgId ? ` at ${get(orgId).name}` : ''} — say if that is not right.`
    );
  }

  // APPLY THE RELATIONS, now that both ends have been resolved or created.
  //
  // DIRECTION IS THE WHOLE JOB HERE. made-by writes a COLUMN on the product and
  // uses writes a ROW in the link table, and they are not two flavours of one
  // thing: a product has exactly one maker, and a product has any number of
  // users. Reading them as symmetric is the mistake that would file Oracle as a
  // user of its own software.
  const byName = {};
  for (const a of out.assignments) if (a.entity) byName[a.mention.name.toLowerCase()] = a.entity;
  if (doCreate) {
    for (const rel of relations) {
      const a = byName[rel.from.toLowerCase()], b = byName[rel.to.toLowerCase()];
      if (!a || !b || a.id === b.id) continue;
      try {
        if (rel.kind === 'made-by') {
          const before = a.org_id;
          setOrgLink(a.id, b.id, { reason: `${a.name} is made by ${b.name}.` });
          if (!before) out.notices.push(`I have noted that ${a.name} is made by ${b.name}.`);
        } else if (rel.kind === 'uses') {
          const fresh = !sqlite().prepare(
            "SELECT 1 FROM entity_links WHERE from_entity_id=? AND to_entity_id=? AND kind='uses' AND status='active'"
          ).get(a.id, b.id);
          linkEntities(a.id, b.id, 'uses');
          out.links.push({ from: a.name, to: b.name, kind: 'uses' });
          if (fresh) out.notices.push(`I have noted that ${a.name} uses ${b.name}.`);
        }
      } catch (e) {
        console.error('[Entities] relation failed:', e.message);
      }
    }
  }

  // ANNOUNCEMENT IS PART OF THE WRITE, not a courtesy the caller may forget.
  //
  // Same reasoning as the ledger funnel: "every caller remembers" is not an
  // invariant, it is a hope, and it had already failed once here — the queuing
  // lived in the intake path only, so every other caller of this function
  // created entities in silence. Athena's requirement is that a clear match
  // still surfaces "so a mis-fire is visible and correctable"; a CREATION going
  // unmentioned is the same failure with more at stake. Queued here, so it
  // cannot be separated from the thing it announces.
  if (doCreate) {
    const ledgerMod = require('./corrections-ledger');
    for (const n of out.notices) ledgerMod.addNotice({ content: n });
    for (const q of out.questions) ledgerMod.addNotice({ content: q.ask });
  }
  return out;
}

/**
 * Which entity is a FACT about?
 *
 * Narrow on purpose: the entity has to be the fact's grammatical subject. A
 * fact that merely mentions one is not about it — "User works at ISH" is about
 * Ellie, and handing it to ISH would be the same misattribution the whole
 * entity line exists to stop.
 *
 * @returns an entity id, or null to leave the caller's default in place.
 */
function entityForFact(factText, assignments) {
  const rules = require('./extraction-rules');
  for (const a of assignments || []) {
    if (!a.entity) continue;
    if (rules.factIsAbout(factText, a.mention.name)) return a.entity.id;
    for (const alias of namesOf(a.entity)) {
      if (rules.factIsAbout(factText, alias)) return a.entity.id;
    }
  }
  return null;
}


module.exports = {
  ENTITY_COLUMNS, FORBIDDEN_COLUMN_HINTS,
  initSchema, assertIndexOnly,
  get, list, create, addAlias, aliasesOf, namesOf,
  selfEntity, userEntity, pointer, setPointer, identityAgrees,
  getFactsForEntity, factCount,
  resolve, isBareFirstName, resolveMentions, entityForFact, createFromMention, acronymOf, TYPES,
  merge, repointFact, syncSelfName, setOrgLink,
  LINK_KINDS, linkEntities, unlinkEntities, usesOf, usersOf, productsOf, makerOf, relationsOf, orgLinkLabel,
  entityLocks, isEntityLocked, lockEntity
};
