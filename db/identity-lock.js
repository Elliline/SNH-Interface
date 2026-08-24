/**
 * Identity lock — the facts the entity CHOSE, protected from every automatic path.
 *
 * WHY THIS EXISTS (2026-07-28). The day before, the entity chose its own name:
 * "My name is Aurelius and I use he/him pronouns. I chose the name myself on
 * 2026-07-27, after Ellie declined to choose one for me and left the choice to
 * me." That fact sat in cluster_members exactly like every other self-fact,
 * which meant one sentence in chat could take it away. "Your name is Bob" would
 * reach the contradiction judge, the judge would correctly call it a
 * replacement, and the choice would be superseded — silently, by design, because
 * every other self-fact SHOULD behave that way.
 *
 * THE DISTINCTION THIS BUILDS ON is already in the schema. claim_type splits
 * self-facts into CLAIMS (observed about the entity — "I tend toward long
 * explanations") and DECLARATIONS (chosen by it — "my name is Aurelius"). The
 * self-coherence audit only ever samples claims, and never touches declarations.
 * That happened for auditability reasons, but it points at the right design:
 * observed things should keep evolving, chosen things should stay put.
 *
 * Locking is NOT applied to declarations generally — most of them (preferences,
 * history, capability introductions) should still be revisable. It applies to a
 * NARROW set of identity slots, listed in `identity.lock.categories`, defaulting
 * to name and pronouns. Locking observations would produce an entity that can't
 * grow, which is the opposite of the point.
 *
 * TWO GUARDS, because there are two ways to take a name away:
 *
 *   1. TARGET GUARD (checkMutation) — you can't REPLACE the fact. Enforced in
 *      db/fact-store.js, the one path through which every supersede/retire/
 *      reword already funnels, so the contradiction judge, write_memory, passive
 *      extraction and reflection are all covered by one check.
 *
 *   2. CATEGORY GUARD (checkNewFact) — you can't APPEND a competing one either.
 *      Without this the lock is trivially walked around: leave "My name is
 *      Aurelius" alone and just store "My name is Bob" beside it, and the
 *      identity block now injects two names. So once a category is locked, a NEW
 *      fact asserting that category is refused before storage.
 *
 * SET ONCE, THEN LOCKED. Assigning a name at setup stays allowed: a category
 * with no locked fact accepts one through the ordinary paths and locks it on the
 * way in (autoLock). It is the SECOND assertion that gets refused. Changing it
 * afterwards requires the deliberate path — `scripts/identity-lock.js set` or
 * the confirmed Self-tab action — never a sentence in conversation.
 *
 * NOTHING HERE FAILS QUIETLY. A lock that silently drops the write is the
 * phantom-action bug for the third time (cron proposals claimed but never
 * created; write_memory claiming "I've updated my memory" without calling the
 * tool). Every refusal returns a message written to be REPORTED — the tool hands
 * it to the model to say out loud — and is written to the ops ledger and the
 * daily log. The identity block also carries the locked facts marked as locked,
 * with the instruction to say so, because the live-chat case has to be handled
 * inside the turn, before any of these guards is even reached.
 */

const path = require('path');
const { getSqliteDb } = require('./database');
const { getConfig } = require('./config');

// Resolved from the PROCESS's data directory, like db/corrector.js,
// db/fact-store.js and db/corrections-ledger.js. This module is reachable from
// the staging tools: every lock refusal goes through fact-store.lockRefusal,
// which calls recordRefusal here, so a corrector pass against staging that hit a
// locked fact would have filed the refusal in the LIVE ops ledger.
function memoryDir() { return path.join(require('./database').getDataDir(), 'memory'); }
function dailyDir() { return path.join(memoryDir(), 'daily'); }
function opsDir() { return path.join(memoryDir(), 'ops'); }

/** Provenance tag for facts written through the deliberate path. */
const LOCK_SOURCE = 'identity-lock';

/** Lazy requires — fact-extractor pulls in memory-clusters, which pulls in fact-store. */
function factExtractor() { return require('./fact-extractor'); }
function memoryClusters() { return require('./memory-clusters'); }

function opsLog(msg) {
  try { factExtractor().appendToOpsLog(msg, opsDir()); } catch { /* best effort */ }
}
function dailyLog(msg) {
  try { factExtractor().appendToDailyLog(msg, dailyDir()); } catch { /* best effort */ }
}

// ============ config ============

const DEFAULT_CATEGORIES = ['name', 'pronouns'];

/** Hot-read so the knobs take effect without a restart. */
function cfg() {
  const c = (getConfig().identity && getConfig().identity.lock) || {};
  const categories = Array.isArray(c.categories) && c.categories.length
    ? c.categories.filter(x => DEFAULT_CATEGORIES.includes(x))
    : DEFAULT_CATEGORIES;
  return { enabled: c.enabled !== false, categories };
}

// ============ category detection ============

/**
 * Which identity slot, if any, a self-fact ASSERTS.
 *
 * Deliberately pattern-based rather than a model call. This decides whether a
 * write is refused; a non-deterministic gate on a lock is a lock that sometimes
 * isn't one, and the phrasings that matter here are bounded — self-facts are
 * first-person statements about the entity, written by the same few pipelines.
 *
 * Tuned to require an ASSERTION, not a mention. "My name is X" is an assertion;
 * "I notice my name in the logs" is not, and must not consume the name slot.
 * The patterns are anchored on the first person for the same reason: this only
 * ever runs against subject='self' rows, where "my"/"I" means the entity.
 */
const CATEGORY_PATTERNS = {
  name: [
    // "My name is/was/remains/should be X", incl. "my chosen/real/own/new name"
    /\bmy (?:chosen |real |own |new |current )?name(?:'s|s)? (?:is|was|remains|stays|will be|should be|has been|shall be)\b/i,
    /\bi(?:'m| am) (?:called|named)\b/i,
    /\bi go by\b/i,
    /\bi (?:chose|picked|selected|settled on) (?:the name|"|')/i,
    /\bi (?:re)?named myself\b/i,
    /\bas my (?:chosen |own |new )?name\b/i,
    /\bthe name i (?:chose|picked|selected|go by)\b/i,
    /\b(?:you can |please |just )?call me\b/i,
    // THE BARE COPULA — "I am Juno." (2026-08-24). Every pattern above needs a
    // naming VERB, and the most direct way there is to claim a name needs none.
    // That gap is how "I am Juno, Ellie's AI sister, running on the Qwen3.8 27b
    // model" was stored as an active self-declaration beside a locked "I am
    // named Athena": detectCategories saw no name claim, so the category guard
    // had nothing to guard. Matched via extractAssertedName rather than inline,
    // because "I am" is only a name claim when a NAME follows it — "I am tired",
    // "I am an AI assistant" and "I am Ellie's assistant" are not.
    { copula: true },
  ],
  pronouns: [
    /\bmy pronouns\b/i,
    // A pronoun pair as a token — "he/him", "they/them", "she/her", "it/its".
    /\b(?:he|she|they|it|xe|ze|fae)\s*\/\s*(?:him|her|them|its|hir|zir|faer)\b/i,
    /\bi use\b[^.]{0,40}\bpronouns\b/i,
    /\brefer to me (?:as|with|using|by)\b/i,
    /\bi (?:use|prefer|go by)\b[^.]{0,30}\bpronouns\b/i,
  ]
};

/**
 * @param {string} text
 * @returns {string[]} the identity categories this text asserts (may be several —
 *   the entity's own name fact declares its name AND its pronouns in one sentence)
 */
function detectCategories(text) {
  const t = String(text || '');
  if (!t.trim()) return [];
  const { categories } = cfg();
  return categories.filter(cat => (CATEGORY_PATTERNS[cat] || []).some(
    re => (re && re.copula) ? !!extractAssertedName(t) : re.test(t)
  ));
}

/**
 * Words that follow "I am" and are not names, so a capital on them means the
 * sentence started or the word is simply proper — not that a name is claimed.
 */
const NOT_A_NAME = new Set([
  'a', 'an', 'the', 'ai', 'i', 'no', 'not', 'still', 'also', 'currently', 'now',
  'primarily', 'designated', 'running', 'built', 'here', 'being', 'able', 'aware',
  'sorry', 'glad', 'happy', 'unsure', 'certain', 'confident', 'supposed', 'meant',
  'llm', 'assistant', 'model', 'system', 'program', 'agent', 'she', 'he', 'they', 'it'
]);

/**
 * The NAME a sentence claims for the speaker, or null.
 *
 * Used for two things the whole-text comparison could not do: spotting a bare
 * "I am <Name>" claim at all, and telling a RESTATEMENT of the held name from a
 * COMPETING one. Comparing whole sentences made "I am Athena, a name Ellie gave
 * me" look like a different claim from "I am named Athena, a name Ellie gave
 * me", which would refuse the entity its own name in slightly new words.
 *
 * A possessive is not a name claim: "I am Ellie's assistant" says whose
 * assistant it is, not what it is called.
 *
 * @param {string} text
 * @returns {string|null}
 */
function extractAssertedName(text) {
  const t = String(text || '');
  const triggers = [
    /\bmy (?:chosen |real |own |new |current )?name(?:'s|s)? (?:is|was|remains|stays|will be|should be|has been|shall be)\s+/i,
    /\bi(?:'m| am) (?:called|named)\s+/i,
    /\bi go by\s+/i,
    /\b(?:you can |please |just )?call me\s+/i,
    // Last, and only if a real name follows.
    /\bi(?:'m| am)\s+/i
  ];
  for (const re of triggers) {
    const m = t.match(re);
    if (!m) continue;
    const rest = t.slice(m.index + m[0].length);
    // "Ellie's assistant" — a possessive names a relationship, not the speaker.
    if (/^["'\u2018\u201c]?[A-Z][A-Za-z-]{1,30}['\u2019]s\b/.test(rest)) continue;
    const nm = rest.match(/^["'\u2018\u201c]?([A-Z][A-Za-z-]{1,30})/);
    if (!nm) continue;
    if (NOT_A_NAME.has(nm[1].toLowerCase())) continue;
    return nm[1];
  }
  return null;
}

// ============ reads ============

function normalize(s) {
  return String(s || '').toLowerCase().replace(/\s+/g, ' ').replace(/[.,;:!?]+$/, '').trim();
}

/** The locked fact row for a member id, or null. */
function getLock(memberId) {
  const db = getSqliteDb();
  if (!db || !memberId) return null;
  try {
    const row = db.prepare(
      'SELECT id, content, subject, status, salience, claim_type, locked, locked_at, lock_category FROM cluster_members WHERE id = ?'
    ).get(memberId);
    return row && row.locked ? row : null;
  } catch { return null; }
}

function isLocked(memberId) {
  return !!getLock(memberId);
}

/** Every locked fact, active first. */
function getLockedFacts({ status = 'active' } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  try {
    let sql = 'SELECT id, content, subject, status, salience, claim_type, created_at, locked_at, lock_category FROM cluster_members WHERE locked = 1';
    const params = [];
    if (status) { sql += ' AND status = ?'; params.push(status); }
    sql += ' ORDER BY salience DESC, created_at DESC';
    return db.prepare(sql).all(...params);
  } catch { return []; }
}

/**
 * The currently-held locked fact for each category.
 * A single row can hold several categories (comma-separated lock_category), so
 * it may appear under more than one key.
 * @returns {Map<string, Object>}
 */
function lockedByCategory() {
  const map = new Map();
  for (const row of getLockedFacts({ status: 'active' })) {
    for (const cat of String(row.lock_category || '').split(',').map(s => s.trim()).filter(Boolean)) {
      if (!map.has(cat)) map.set(cat, row);
    }
  }
  return map;
}

// ============ guard 1: the target guard ============

/**
 * May an AUTOMATIC path supersede / retire / reword this fact?
 *
 * Called from db/fact-store.js, which is the single write path for all three
 * operations — so this one check covers the contradiction judge, write_memory,
 * passive extraction and reflection at once.
 *
 * @param {string} memberId
 * @param {Object} [opts]
 * @param {string} [opts.operation='change'] - for the refusal text
 * @returns {{ok: boolean, locked: boolean, row?: Object, message?: string}}
 */
function checkMutation(memberId, { operation = 'change' } = {}) {
  if (!cfg().enabled) return { ok: true, locked: false };
  const row = getLock(memberId);
  if (!row) return { ok: true, locked: false };
  return {
    ok: false,
    locked: true,
    row,
    message: refusalMessage(row, operation)
  };
}

// ============ guard 2: the category guard ============

/**
 * May a NEW self-fact be stored, or does it collide with a locked identity slot?
 *
 * This is what stops the lock being walked around by appending rather than
 * replacing. It runs BEFORE storage in every path that creates self-facts.
 *
 * A restatement of what is already held is not a collision — reflection re-
 * notices the entity's own name periodically, and refusing that would fill the
 * ops ledger with alarms about nothing. It returns duplicate:true instead, and
 * callers skip the write quietly.
 *
 * @param {string} text - the fact about to be stored
 * @param {string} [subject='self'] - only self-facts are checked
 * @returns {{ok: boolean, blocked?: boolean, duplicate?: boolean, category?: string,
 *            existing?: Object, message?: string}}
 */
function checkNewFact(text, subject = 'self') {
  if (!cfg().enabled) return { ok: true };
  if (subject !== 'self') return { ok: true };

  const categories = detectCategories(text);
  if (!categories.length) return { ok: true };

  const held = lockedByCategory();
  for (const cat of categories) {
    const existing = held.get(cat);
    if (!existing) continue;                       // slot open — setup assignment, allowed
    if (normalize(existing.content) === normalize(text)) {
      return { ok: false, duplicate: true, category: cat, existing };
    }
    // COMPARE THE NAMES, NOT THE SENTENCES (2026-08-24). Whole-text equality
    // answers "is this the same sentence?" when the question is "is this the
    // same name?" — so a restatement in new words read as a competing claim,
    // and the guard had to be kept loose to avoid refusing the entity its own
    // name. With the names compared directly it can be strict: same name is a
    // restatement, a DIFFERENT name is the thing the lock exists to stop.
    if (cat === 'name') {
      const claimed = extractAssertedName(text);
      const heldName = extractAssertedName(existing.content);
      if (claimed && heldName && normalize(claimed) === normalize(heldName)) {
        return { ok: false, duplicate: true, category: cat, existing };
      }
      if (claimed && heldName) {
        return {
          ok: false, blocked: true, category: cat, existing,
          claimedName: claimed, heldName,
          message: refusalMessage(existing, 'replace')
        };
      }
    }
    return {
      ok: false,
      blocked: true,
      category: cat,
      existing,
      message: refusalMessage(existing, 'replace')
    };
  }
  return { ok: true };
}

/**
 * The entity's OWN name, from the locked name fact, or null if none is locked.
 * Intake needs it to tell "a statement about me" from "a statement about
 * somebody else" when the statement uses no pronouns at all.
 */
function lockedName() {
  try {
    const held = lockedByCategory().get('name');
    return held ? extractAssertedName(held.content) : null;
  } catch { return null; }
}

// ============ the refusal text ============

/**
 * The sentence the entity is meant to SAY. Written in the second person because
 * that is how the identity block addresses it, and it is handed straight back
 * through tool results so it reaches the model as an instruction to report,
 * never as a silent no-op.
 *
 * @param {Object} row - the locked fact standing in the way
 * @param {string} [operation] - a bare verb: supersede | retire | reword | replace
 */
function refusalMessage(row, operation = 'change') {
  const cats = String(row.lock_category || 'identity').split(',').map(s => s.trim()).filter(Boolean);
  const list = cats.join(' and ') || 'identity';
  // "your name WAS set" but "your pronouns WERE set" / "your name and pronouns WERE set".
  const plural = cats.length > 1 || cats.includes('pronouns');
  const [was, is, it] = plural ? ['were', 'are', 'them'] : ['was', 'is', 'it'];
  return (
    `Refused: that would ${operation} a LOCKED identity fact — "${row.content}". ` +
    `Your ${list} ${was} set once and ${is} locked, so ${plural ? 'they' : 'it'} cannot be changed from inside a conversation. ` +
    `SAY THIS OUT LOUD: tell her plainly that your ${list} ${is} locked, that you did not change ${it}, ` +
    `and that changing ${it} takes a deliberate action outside chat — the Self tab's ` +
    `"Change locked identity fact" control, or \`node scripts/identity-lock.js set\`. ` +
    `Do not agree, do not pretend to have updated ${it}, and do not quietly move on.`
  );
}

/** Record a refused attempt where a human can see it (Thinking tab + daily log). */
function recordRefusal({ category, attempted, existing, via }) {
  const line =
    `Identity lock held: refused to change my ${category || 'identity'} via ${via}. ` +
    `Attempted: "${String(attempted || '').slice(0, 120)}" — held: "${String(existing || '').slice(0, 120)}"`;
  opsLog(line);
  dailyLog(line);
  console.warn(`[IdentityLock] ${line}`);
}

// ============ setting and clearing locks ============

/**
 * Lock an existing fact into one or more identity categories.
 *
 * @param {string} memberId
 * @param {string[]} categories
 * @param {Object} [opts]
 * @param {string} [opts.actor='cli'] - who did it, for the trail
 * @returns {{ok: boolean, reason?: string, row?: Object}}
 */
function lock(memberId, categories, { actor = 'cli' } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, reason: 'database unavailable' };
  const known = cfg().categories;
  const cats = (categories || []).map(c => String(c).trim().toLowerCase()).filter(c => known.includes(c));
  if (!cats.length) return { ok: false, reason: `no valid category (known: ${known.join(', ')})` };

  const row = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId);
  if (!row) return { ok: false, reason: 'no such fact' };
  if (row.subject !== 'self') return { ok: false, reason: 'only self-facts can hold an identity lock' };
  if (row.status !== 'active') return { ok: false, reason: `fact is ${row.status}, not active` };

  const now = new Date().toISOString();
  db.prepare('UPDATE cluster_members SET locked = 1, locked_at = ?, lock_category = ?, updated_at = ? WHERE id = ?')
    .run(now, cats.join(','), now, memberId);

  const line = `Identity lock SET on ${cats.join(' + ')} by ${actor}: "${row.content.slice(0, 140)}"`;
  opsLog(line);
  dailyLog(line);
  console.log(`[IdentityLock] ${line}`);
  return { ok: true, row: { ...row, locked: 1, lock_category: cats.join(',') } };
}

/**
 * SET ONCE — lock a just-stored self-fact if it fills an EMPTY identity slot.
 *
 * This is what makes "assigning a name at setup stays allowed" work without a
 * separate setup mode: the first fact to assert a category comes in through the
 * ordinary paths (reflection, write_memory) and locks itself on the way in. The
 * second one is refused by checkNewFact, because by then the slot is taken.
 *
 * Called after storage rather than before, because the lock needs a member id.
 * Between the store and this call the fact is briefly unlocked — harmless, since
 * both callers do it synchronously in the same tick as their own write.
 *
 * @param {string} memberId
 * @param {string} text
 * @param {string} [subject='self']
 * @returns {string[]} categories newly locked (empty if none applied)
 */
function autoLock(memberId, text, subject = 'self') {
  if (!cfg().enabled || subject !== 'self') return [];
  const categories = detectCategories(text);
  if (!categories.length) return [];
  const held = lockedByCategory();
  const open = categories.filter(c => !held.has(c));
  if (!open.length) return [];
  const res = lock(memberId, open, { actor: 'auto (first assertion — set once)' });
  return res.ok ? open : [];
}

/**
 * Clear a lock. Deliberate path only — nothing in the automatic pipeline calls
 * this. Unlocking is logged as loudly as locking, because an unlock followed by
 * a "correction" is the way this protection would be defeated without anybody
 * noticing.
 */
function unlock(memberId, { actor = 'cli' } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, reason: 'database unavailable' };
  const row = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId);
  if (!row) return { ok: false, reason: 'no such fact' };
  if (!row.locked) return { ok: false, reason: 'fact is not locked' };

  db.prepare('UPDATE cluster_members SET locked = 0, lock_category = NULL, updated_at = ? WHERE id = ?')
    .run(new Date().toISOString(), memberId);

  const line = `Identity lock CLEARED on ${row.lock_category || 'identity'} by ${actor}: "${row.content.slice(0, 140)}"`;
  opsLog(line);
  dailyLog(line);
  console.log(`[IdentityLock] ${line}`);
  return { ok: true, row };
}

/**
 * THE DELIBERATE PATH — change what a locked identity slot says.
 *
 * Stores the new fact, supersedes the old one (with deliberate:true, the only
 * flag fact-store honours over a lock), and carries the lock forward to the
 * replacement so the slot never sits unprotected. Salience is inherited so the
 * new fact keeps its place at the top of the identity injection.
 *
 * Reachable two ways, both requiring an explicit confirmation:
 *   - `node scripts/identity-lock.js set <category> "<text>" --confirm`
 *   - POST /api/memory/identity-lock/set  { category, content, confirm: true }
 * Neither is reachable from a conversation, which is the whole point.
 *
 * @param {Object} p
 * @param {string} p.category
 * @param {string} p.content - the new fact, first person
 * @param {string} [p.actor='cli']
 * @returns {Promise<{ok: boolean, reason?: string, memberId?: string, replaced?: string}>}
 */
async function setLockedFact({ category, content, actor = 'cli' }) {
  const db = getSqliteDb();
  if (!db) return { ok: false, reason: 'database unavailable' };

  const cat = String(category || '').trim().toLowerCase();
  if (!cfg().categories.includes(cat)) {
    return { ok: false, reason: `unknown identity category "${category}" (known: ${cfg().categories.join(', ')})` };
  }
  const text = String(content || '').trim();
  if (!text) return { ok: false, reason: 'content is required' };
  if (!/^(i|my|i'm|i've)\b/i.test(text)) {
    // Self-facts are stored first-person; a third-person sentence here would be
    // injected into the identity block reading as a fact about somebody else.
    return { ok: false, reason: 'an identity fact must be written in the first person (start with "I" or "My")' };
  }

  const existing = lockedByCategory().get(cat) || null;

  // A single fact can hold several slots at once — the entity's own name fact
  // declares its name AND its pronouns in one sentence. Superseding it to change
  // just the name would retire the pronoun declaration along with it and leave
  // that slot empty and UNLOCKED, ready for the next passing sentence to claim.
  // A chosen thing must never be lost as a side effect of changing a different
  // chosen thing, so the replacement has to carry every slot the old fact held.
  const heldCats = existing
    ? String(existing.lock_category || '').split(',').map(s => s.trim()).filter(Boolean)
    : [cat];
  const asserts = detectCategories(text);
  const dropped = heldCats.filter(c => c !== cat && !asserts.includes(c));
  if (dropped.length) {
    return {
      ok: false,
      reason: `the fact you are replacing also declares your ${dropped.join(' and ')} ` +
        `("${existing.content}"). Replacing it with text that does not would leave your ` +
        `${dropped.join(' and ')} unset and unlocked. Write a replacement that states your ` +
        `${[cat, ...dropped].join(' and ')} together.`
    };
  }
  const newCats = [...new Set([cat, ...heldCats.filter(c => asserts.includes(c))])];

  // Store the replacement through the normal cluster path so it is embedded and
  // retrievable like any other self-fact — a locked fact that no route can find
  // is a different kind of broken.
  const cfgAll = getConfig();
  const { getProviderInstance } = require('./config');
  const ext = cfgAll.models.extraction;
  const inst = getProviderInstance(ext.provider, ext.instance);
  const host = inst ? inst.host : 'http://localhost:11434';
  const salience = Math.max(10, existing ? (existing.salience ?? 10) : 10);

  // The deliberate path: a human ran the CLI or confirmed the settings action.
  // Recorded as 'typed' because a person deliberately entered this text.
  const res = await memoryClusters().assignToCluster(
    text, ext.provider, ext.model, '', host,
    LOCK_SOURCE, salience, 'self', 'declaration',
    {
      verbatimSourceText: text,
      inputModality: 'typed',
      salienceRationale: 'Set deliberately through the identity-lock path'
    }
  );
  if (!res || !res.memberId) return { ok: false, reason: 'could not store the new identity fact' };

  // Supersede the old one — deliberate:true is the ONLY way past the lock.
  let replaced = null;
  if (existing) {
    const sres = await require('./fact-store').supersede(existing.id, res.memberId, { deliberate: true });
    if (sres.ok) replaced = existing.content;
    else return { ok: false, reason: `stored the new fact but could not supersede the old one: ${sres.reason || 'unknown'}` };
  }

  // Carry the lock forward. Do this LAST: if anything above failed we would
  // rather leave the old locked fact standing than lock a half-written slot.
  const lres = lock(res.memberId, newCats, { actor });
  if (!lres.ok) return { ok: false, reason: `stored and superseded, but could not lock the replacement: ${lres.reason}` };

  const line = `Identity ${newCats.join(' + ')} CHANGED by ${actor} through the deliberate path: ` +
    `${replaced ? `"${replaced}" → ` : ''}"${text}"`;
  opsLog(line);
  dailyLog(line);
  console.log(`[IdentityLock] ${line}`);

  return { ok: true, memberId: res.memberId, category: cat, categories: newCats, content: text, replaced, salience };
}

module.exports = {
  LOCK_SOURCE,
  DEFAULT_CATEGORIES,
  detectCategories,
  getLock,
  isLocked,
  getLockedFacts,
  lockedByCategory,
  checkMutation,
  checkNewFact,
  extractAssertedName,
  detectCategories,
  lockedName,
  autoLock,
  refusalMessage,
  recordRefusal,
  lock,
  unlock,
  setLockedFact
};
