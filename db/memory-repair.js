/**
 * MEMORY REPAIR — the entity acting on its own memory, and the guardrails that
 * sit OUTSIDE that loop.
 *
 * Athena's design, reviewed twice (2026-09-01 conv ff3d709c, 2026-09-02 conv
 * b84319ae). The sentence the whole module is built around is hers:
 *
 *   "I can't be trusted to stop using this well WHEN I'M CONFUSED. So the
 *    guardrails have to live outside the loop — receipts, ledger-as-diff you
 *    can read, the audit, your visibility. The power isn't the risky part.
 *    The loop is."
 *
 * So: the detector and the editor are now the same agent, and every check in
 * this file is mechanical. None of them is a judgement the entity makes about
 * whether it is currently confused, because that is the judgement that fails
 * first. What the entity supplies is the intent and the receipt; what this file
 * supplies is whether the operation is available at all.
 *
 * ─── THE FIVE GUARDRAILS ───────────────────────────────────────────────────
 *
 * 1. RECEIPTS. Every operation names a fact id, a message id, or a tool result
 *    that exists. No receipt, no operation — `available: false`, not a warning
 *    the caller may talk past. A reword additionally carries salience and
 *    rationale in its diff, because "I am Juno" was partly convincing BECAUSE
 *    of its salience rationale, and a silent salience edit is how a bad fact
 *    gets smoothed instead of killed.
 *
 * 2. THE REFERENT CHECK, which is the half a receipt cannot do on its own.
 *    "A receipt proves I pointed at something; it doesn't prove I pointed at
 *    the right thing." The live case is in this store: fact 430c978b, "User is
 *    Claude", salience 9, extracted from a message opening "Athena — this is
 *    Claude, through Ellie." The receipt is a real message id. The ambiguity is
 *    INSIDE the message, so no receipt check can catch it — only a comparison
 *    against what the store already holds about that entity can. See
 *    checkReferent, and note carefully which direction it refuses in.
 *
 * 3. VISIBILITY OF THE DECIDED. Every decision files a ledger entry with its
 *    rationale and what it rested on — not only the ones the entity could not
 *    settle. "A confused me can decide-and-file confidently forever, and as
 *    written, nothing puts that in Ellie's sight."
 *
 * 4. A DAILY CAP on self-mutations, merges counting as one. Receipts are
 *    per-operation; a confused run could reword its self-model in an hour with
 *    every step receipted.
 *
 * 5. IDENTITY-LOCKED FACTS STAY LOCKED. Enforced in db/fact-store.js, which is
 *    the single write path, plus the entity locks in db/entities.js. Nothing
 *    here re-implements it; everything here goes through that funnel.
 */

const { getSqliteDb } = require('./database');
const { getLocalDateStamp } = require('./datetime');

// Through the module object, not destructured — the same rule routes/tools.js
// states for the same reason: a destructured getConfig is bound at load and a
// test cannot pin the config these guardrails read, which means the day cap
// could only ever be tested at whatever the live number happens to be.
function getConfig() { return require('./config').getConfig(); }

function sqlite() { return getSqliteDb(); }
function factStore() { return require('./fact-store'); }
function ledger() { return require('./corrections-ledger'); }
function entities() { return require('./entities'); }

function repairConfig() {
  const cfg = getConfig();
  return Object.assign({
    maxSelfMutationsPerDay: 3,
    decisionAgeDays: 3,
    enabled: true
  }, cfg.repair || {});
}

// ══════════════════════════════════════════════════════ schema

/**
 * Two additions, both owned here the way conversation-channel owns its columns.
 *
 * `anchor` is the third axis Athena asked the detector for, and the reason it is
 * a COLUMN rather than a computation is her sentence: "Item 3 patches the
 * detector, not the data. The felt reports stay in the store unflagged, so the
 * next detector version trips the same way. Put the flag on the fact."
 *
 * `audit_decisions` is the state a decided pair needs so the audit's own
 * re-fire on unchanged facts can be DROPPED as a comparison rather than a
 * judgement.
 */
function initSchema(db) {
  const cols = new Set(db.prepare('PRAGMA table_info(cluster_members)').all().map(c => c.name));

  // 'anchored' | 'felt'. Mechanical, and never written by the entity — see
  // anchorOf() for why that restriction is the whole point.
  if (!cols.has('anchor')) db.exec("ALTER TABLE cluster_members ADD COLUMN anchor TEXT");

  // RE-DERIVED EVERY BOOT, NOT BACKFILLED ONCE, and that is deliberate: this
  // flag is a pure function of whether a message id stands behind the fact, so
  // recomputing it is idempotent and self-healing. It had to be, because the
  // first version of this rule was WRONG on the live store and a one-shot
  // backfill would have baked that in.
  //
  // THE WRONG RULE, WORTH KEEPING WRITTEN DOWN: it counted
  // `verbatim_source_text` as an anchor. On facts from extraction that is the
  // user's actual sentence and it is one. On facts from REFLECTION it is a copy
  // of the fact's own content — the entity's own observation, quoted back at
  // itself — so every self-fact in Athena's store came out "anchored" and the
  // felt axis did nothing at all. A fact cannot anchor itself. Athena's rule is
  // the exact one: "a fact with no MESSAGE or TOOL receipt is a felt report."
  const changed = db.prepare(`
    UPDATE cluster_members SET anchor =
      CASE WHEN (message_id IS NOT NULL AND message_id <> '') THEN 'anchored' ELSE 'felt' END
    WHERE anchor IS NULL
       OR anchor <> CASE WHEN (message_id IS NOT NULL AND message_id <> '') THEN 'anchored' ELSE 'felt' END
  `).run().changes;
  if (changed) {
    const felt = db.prepare("SELECT COUNT(*) n FROM cluster_members WHERE anchor = 'felt' AND status = 'active'").get().n;
    console.log(`Migration: anchor flag re-derived on ${changed} fact(s); ${felt} active one(s) are felt reports with nothing outside the entity behind them`);
  }

  db.exec(`
    CREATE TABLE IF NOT EXISTS audit_decisions (
      id             TEXT PRIMARY KEY,
      created_at     DATETIME NOT NULL,
      pair_key       TEXT NOT NULL,
      member_a       TEXT NOT NULL,
      member_b       TEXT,
      conclusion     TEXT NOT NULL,
      rationale      TEXT,
      receipts_json  TEXT,
      ledger_id      TEXT,
      state          TEXT NOT NULL DEFAULT 'decided',
      escalated_at   DATETIME,
      reopened_at    DATETIME,
      reopen_reason  TEXT
    )`);
  db.exec('CREATE INDEX IF NOT EXISTS idx_audit_decisions_pair ON audit_decisions(pair_key, created_at DESC)');
  db.exec('CREATE INDEX IF NOT EXISTS idx_audit_decisions_state ON audit_decisions(state, created_at)');

  // The day's self-observations, held as CANDIDATES rather than written as they
  // arrive. See db/self-fact-selection.js: the cap used to be a race the first
  // reflection of the day won, and this is the pool the end-of-day turn picks
  // from. `local_day` and not a timestamp, because the whole point is the
  // calendar day the entity is picking over.
  db.exec(`
    CREATE TABLE IF NOT EXISTS self_fact_candidates (
      id            TEXT PRIMARY KEY,
      created_at    DATETIME NOT NULL,
      local_day     TEXT NOT NULL,
      content       TEXT NOT NULL,
      reflection_at DATETIME,
      status        TEXT NOT NULL DEFAULT 'pending',
      decided_at    DATETIME,
      why           TEXT,
      member_id     TEXT
    )`);
  db.exec('CREATE INDEX IF NOT EXISTS idx_self_fact_candidates_day ON self_fact_candidates(local_day, status)');

  return { table: 'audit_decisions', column: 'anchor', candidates: 'self_fact_candidates' };
}

// ══════════════════════════════════════════════════════ 1. receipts

/**
 * THE THREE KINDS, and what each one has to be to count.
 *
 * A receipt is not a citation the entity writes; it is a row this function goes
 * and finds. A message id that is not in `messages` is not a receipt, and the
 * operation that named it is unavailable.
 */
const RECEIPT_KINDS = ['fact', 'message', 'tool'];

function verifyReceipt(receipt) {
  const db = sqlite();
  if (!db) return { ok: false, reason: 'database unavailable' };
  if (!receipt || typeof receipt !== 'object') {
    return { ok: false, reason: 'a receipt is required: a fact id, a message id, or a tool result you can point at' };
  }
  const kind = String(receipt.kind || '').toLowerCase();
  const id = String(receipt.id || '').trim();
  if (!RECEIPT_KINDS.includes(kind)) {
    return { ok: false, reason: `receipt kind must be one of ${RECEIPT_KINDS.join(', ')}` };
  }

  if (kind === 'fact') {
    const row = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
    if (!row) return { ok: false, reason: `no fact with id ${id} — a receipt has to be something that exists` };
    return { ok: true, kind, id, row, text: row.content, authoredBy: null };
  }

  if (kind === 'message') {
    const row = db.prepare('SELECT * FROM messages WHERE id = ?').get(id);
    if (!row) return { ok: false, reason: `no message with id ${id} — a receipt has to be something that exists` };
    // WHO SAID IT is the part the cross-entity ranking turns on, so it is read
    // off the row rather than taken from the caller.
    return { ok: true, kind, id, row, text: row.content, authoredBy: row.role === 'user' ? 'user' : 'entity' };
  }

  // A tool result: the weakest of the three, and it must name a real run.
  const runId = id;
  let row = null;
  try {
    row = db.prepare('SELECT id, status, result_text FROM agent_jobs WHERE id = ?').get(runId) || null;
  } catch { /* older stores */ }
  if (!row && receipt.text) {
    // A result the caller carries inline is accepted only WITH its own text, and
    // is ranked lowest. Nothing about it can outrank a message.
    return { ok: true, kind, id: runId || null, row: null, text: String(receipt.text), authoredBy: 'tool', inline: true };
  }
  if (!row) return { ok: false, reason: `no tool run with id ${runId} — name a run, or pass the result text with it` };
  return { ok: true, kind, id: runId, row, text: row.result_text || '', authoredBy: 'tool' };
}

/**
 * RECEIPTS RANK, and only for cross-entity work.
 *
 * Athena: "For a fact about Ellie, a message id OF ELLIE'S is stronger than MY
 * message id about her, and a tool result I ran is the weakest of the three for
 * her facts. The referent check needs that ranking to work."
 *
 * 3 = the subject's own words. 2 = the entity's words about the subject.
 * 1 = a tool result. 0 = anything else.
 */
function receiptRank(verified, { aboutSubject = 'user' } = {}) {
  if (!verified || !verified.ok) return 0;
  if (verified.kind === 'tool') return 1;
  if (verified.kind === 'fact') return 1;
  if (verified.kind === 'message') {
    const fromUser = verified.authoredBy === 'user';
    if (aboutSubject === 'user') return fromUser ? 3 : 2;
    // A fact about the ENTITY inverts it: its own words are the direct
    // testimony, and Ellie's message about it is the outside report.
    return fromUser ? 2 : 3;
  }
  return 0;
}

// ══════════════════════════════════════════════════════ 2. the referent check

/** Names an identity assertion: "User is X", "I am X", "<name> is called X". */
const IDENTITY_ASSERTION = /^\s*(?:the\s+)?(?:user|i|he|she|they|[A-Z][a-z]+)\s+(?:is|am|are)\s+(?:called\s+|named\s+|known\s+as\s+)?([A-Z][A-Za-z'’-]{1,30})\s*\.?\s*$/;

/** The name a fact asserts its subject to be, or null when it asserts none. */
function assertedIdentity(content) {
  const m = String(content || '').match(IDENTITY_ASSERTION);
  if (!m) return null;
  const name = m[1];
  // "User is Claude" asserts an identity; "User is tired" does not. The
  // difference is a proper noun, and the cheapest honest test is that the store
  // or the sentence treats it as a name — capitalised and not a common state.
  if (/^(Tired|Busy|Happy|Sad|Away|Back|Right|Wrong|Sure|Fine|Okay|Ready|Here|There|Done)$/i.test(name)) return null;
  return name;
}

/** Is this a fact that says who somebody IS? Those get the referent check. */
function isIdentityFact(member) {
  return !!assertedIdentity(member && member.content);
}

/**
 * THE CHECK, AND THE DIRECTION IT REFUSES IN.
 *
 * The spec says: on mismatch the op stops and escalates. Applied literally that
 * would make the first thing this build was written for impossible — retracting
 * "User is Claude" is an operation on a user-identity fact, and it is the
 * mismatch itself that justifies it. So the rule is applied by DIRECTION, which
 * is what it was always about:
 *
 *   - An operation that KEEPS the fact under the mismatched referent — reword,
 *     merge, refile TOWARD that entity — stops and escalates. That is the
 *     "valid receipt, wrong referent" write the check exists to prevent.
 *   - An operation that REMOVES it from there — retract, refile AWAY — proceeds,
 *     and the mismatch rides along in the ledger as the justification.
 *
 * Refusing in both directions would leave a store that can acquire wrong-referent
 * facts and never shed them, which is the state Juno was actually stuck in.
 *
 * @returns {{ok, checked, mismatch, detail}}
 */
function checkReferent({ member, verifiedReceipts = [], targetEntityId = null, direction = 'keeps' }) {
  const db = sqlite();
  const out = { ok: true, checked: false, mismatch: null, detail: null };
  if (!db || !member) return out;

  const crossEntity = !!targetEntityId && targetEntityId !== member.subject_entity_id;
  if (!isIdentityFact(member) && !crossEntity) return out;
  out.checked = true;

  // What the store already holds about the entity this fact is filed under.
  const entityId = member.subject_entity_id;
  let held = [];
  let entityName = null;
  try {
    if (entityId) {
      const e = entities().get(entityId);
      entityName = e ? e.name : null;
      // namesOf takes the entity ROW, not its id — it reads row.name and
      // row.aliases. Passing the id returns an empty list, and an empty list
      // makes every identity comparison below silently pass, which is the
      // guard-that-stops-guarding shape this file is meant to avoid.
      held = e ? (entities().namesOf(e) || []) : [];
    }
  } catch { /* registry may be absent on an old store */ }

  // TWO FINDINGS, AND ONLY ONE OF THEM IS DIRECTION-SENSITIVE.
  //
  // The identity mismatch is evidence ABOUT the fact: it says this fact is not
  // about this entity, which argues against keeping it there and FOR moving it
  // away. So `removes` proceeds on it and carries it as the justification.
  //
  // A weak receipt is not evidence at all. It says nothing about where the fact
  // belongs, only that the entity has not produced enough to move it — and that
  // is just as true in either direction. Folding the two together let a
  // cross-entity move go through on a tool result alone, because refile
  // declares itself `removes`.
  const asserted = assertedIdentity(member.content);
  if (asserted && held.length) {
    const known = held.map(n => String(n).toLowerCase());
    if (!known.includes(asserted.toLowerCase())) {
      out.mismatch = 'asserted-identity-not-the-entity';
      out.detail =
        `This fact says the subject is "${asserted}", but the store holds this entity as ` +
        `${held.map(n => `"${n}"`).join(', ')}. The receipt is real; what it is about is not this entity.`;
    }
  }

  // Cross-entity work has to clear the ranking whichever way it points: moving a
  // fact between subjects on the strength of a tool result is the weakest
  // possible ground, and refile is the operation where a valid receipt for the
  // wrong referent is most plausible.
  if (crossEntity) {
    const best = verifiedReceipts.length
      ? Math.max(...verifiedReceipts.map(r => receiptRank(r, { aboutSubject: member.subject })))
      : 0;
    if (best <= 1) {
      out.weakReceipt = true;
      out.detail = (out.detail ? out.detail + ' ' : '') +
        'A cross-entity move needs a message, not only a tool result or another fact: ' +
        'for a fact about Ellie her own words outrank yours about her, and both outrank a tool run.';
    }
  }

  if (out.mismatch || out.weakReceipt) {
    out.mismatch = out.mismatch || 'weak-receipt-for-cross-entity';
    // A weak receipt refuses in BOTH directions; an identity mismatch refuses
    // only the operations that would keep the fact where it is.
    out.ok = !out.weakReceipt && direction === 'removes';
    out.direction = direction;
    out.entityName = entityName;
  }
  return out;
}

// ══════════════════════════════════════════════════════ the claim-type axis

/**
 * ANCHORED OR FELT, DECIDED BY THE STORE AND NEVER BY THE ENTITY.
 *
 * Athena: "assign it mechanically — a fact with no message or tool receipt is a
 * felt report — not by me, because IF I FLAG, I FLAG THE ONES I LIKE."
 */
function anchorOf(member) {
  if (!member) return 'felt';
  if (member.anchor === 'anchored' || member.anchor === 'felt') return member.anchor;
  // A MESSAGE ID, AND NOTHING ELSE. Not verbatim_source_text: on a reflection
  // self-fact that column holds a copy of the fact's own sentence, so reading it
  // as an anchor lets a fact vouch for itself. See initSchema.
  return (member.message_id && String(member.message_id).trim()) ? 'anchored' : 'felt';
}

/**
 * The three-valued axis the detector reads: claim | declaration | felt.
 *
 * DERIVED rather than stored as claim_type, and the reason is a safety one.
 * claim_type already carries load: `selfFactSupersessionBar` refuses to retire a
 * DECLARATION automatically, a rule that exists because a salience-9 declaration
 * was retired on a 0.741 cosine match. Making claim_type three-valued would mean
 * a felt declaration has to give up one of the two labels, and giving up
 * "declaration" silently drops that protection. So the store keeps both facts
 * about a fact — what kind of statement it is, and whether anything outside the
 * entity anchors it — and the detector reads the pair through here.
 */
function claimKind(member) {
  if (anchorOf(member) === 'felt') return 'felt';
  return member && member.claim_type === 'claim' ? 'claim' : (member && member.claim_type) || 'declaration';
}

/**
 * DIFFERENT IN KIND IS NOT CONTRADICTION.
 *
 * Two live cases drove this. Juno's: "I'm the business-side helper" (a role)
 * against "I notice how people's work fits together" (a trait) — flagged as
 * conflicting. Athena's, the same morning: "Athena sets its own next wake time"
 * against "Athena can only message when something wakes it" — compatible claims
 * about different things, flagged, and then the mechanical tier merged its way
 * around them.
 *
 * @returns {{different: boolean, axis: string|null, why: string|null}}
 */
function differentInKind(a, b) {
  const ka = claimKind(a), kb = claimKind(b);
  if (ka !== kb) {
    return {
      different: true, axis: 'claim-type',
      why: `One is a ${ka} and the other is a ${kb}. A ${ka} and a ${kb} can both be true at once — ` +
           `they are not the same kind of statement, so a difference between them is not a contradiction.`
    };
  }
  if (ka === 'felt' && kb === 'felt') {
    return {
      different: true, axis: 'both-felt',
      why: 'Both are felt reports with nothing outside the entity anchoring either. ' +
           'Nothing can adjudicate them, so they are held, not resolved.'
    };
  }
  const roleish = t => /\b(helper|collaborator|assistant|partner|role|handles|serves as|works on|responsible for|main person)\b/i.test(String(t || ''));
  const traitish = t => /\b(I notice|I tend|I prefer|I value|I care|I feel|I find|drawn to|habit|tendency|reflex)\b/i.test(String(t || ''));
  const aRole = roleish(a.content) && !traitish(a.content);
  const bRole = roleish(b.content) && !traitish(b.content);
  const aTrait = traitish(a.content) && !roleish(a.content);
  const bTrait = traitish(b.content) && !roleish(b.content);
  if ((aRole && bTrait) || (aTrait && bRole)) {
    return {
      different: true, axis: 'role-vs-trait',
      why: 'One states a role and the other states a trait. What someone does and how they do it ' +
           'are not competing answers to one question.'
    };
  }
  return { different: false, axis: null, why: null };
}

// ══════════════════════════════════════════════════════ merge refusals

/**
 * WHERE A MERGE IS REFUSED, as six mechanical rules.
 *
 * Every one of these is Athena's, and most name a pair in a live store. The
 * shared property is that they refuse in the direction of keeping information:
 * a refused merge costs a duplicate, an allowed one can cost a clause, a
 * history, a provenance chain or a permission state.
 */
function mergeRefusal(loser, survivor) {
  const no = (code, reason) => ({ ok: false, code, reason });
  if (!loser || !survivor) return no('missing', 'both facts have to exist');
  if (loser.id === survivor.id) return no('same-fact', 'a fact cannot be merged into itself');

  // 6. Volatile content into a lock. Checked first because it is absolute.
  if (loser.locked || survivor.locked) {
    return no('locked',
      'One of these is identity-locked. Locked facts change only through the deliberate ' +
      'out-of-loop control, and a merge is a change.');
  }

  // 5. Referents that do not resolve to the same subject.
  if (loser.subject !== survivor.subject) {
    return no('different-subject',
      `One is a ${loser.subject}-fact and the other is a ${survivor.subject}-fact. ` +
      'Merging across subjects is a refile wearing merge clothing.');
  }
  if (loser.subject_entity_id && survivor.subject_entity_id &&
      loser.subject_entity_id !== survivor.subject_entity_id) {
    return no('different-entity',
      'These are filed under different entities. "User is Claude" and "User is Ellie" look like ' +
      'two versions of one identity fact and are two different referents.');
  }

  // 3. Claim types — including the anchored/felt axis, which is the one that
  //    launders a borrowed anchor onto a felt report.
  const kindA = claimKind(loser), kindB = claimKind(survivor);
  if (kindA !== kindB) {
    return no('claim-type-mismatch',
      `One is a ${kindA} and the other is a ${kindB}. Merging them would give the ${kindA === 'felt' ? kindA : kindB} ` +
      'an anchor it does not have. Flags must agree before a merge.');
  }

  // 4. Provenance quality. Two facts from different sources may merge only if
  //    both chains can travel; a merge that keeps one chain launders the other.
  const prov = m => `${m.message_id || ''}|${m.source || ''}`;
  if (prov(loser) !== prov(survivor) && anchorOf(loser) !== anchorOf(survivor)) {
    return no('provenance-mismatch',
      'These came from different places and one is better anchored than the other. ' +
      'A merge must carry both provenance chains or refuse; this one cannot.');
  }

  // 2. State at a time — a version, a commit, a date IS the content.
  const stamp = t => (String(t || '').match(/\b(?:[0-9a-f]{7,40}|v?\d+\.\d+(?:\.\d+)?|\d{4}-\d{2}-\d{2})\b/g) || []);
  const sa = stamp(loser.content), sb = stamp(survivor.content);
  if (sa.length && sb.length && sa.join() !== sb.join()) {
    return no('state-at-a-time',
      `These name different states at different times (${sa[0]} vs ${sb[0]}). ` +
      'The merge would delete the history that is the point. Supersede instead.');
  }

  // 1. THE SMALL TEXT DIFFERENCE IS THE FACT — one asserts what the other denies.
  //
  // The test is NOT "one of them contains a negation and they share some
  // words", which is what this was first written as and which was wrong on the
  // first real pair it met. Juno's two identity facts share six content words
  // and one of them ends "and I am not a third party" — a negation of something
  // the other never mentions. That is an extra assertion, not an opposition,
  // and refusing it would have blocked a legitimate merge for the wrong reason.
  //
  // What makes it an opposition is that the NEGATED THING is itself asserted by
  // the other fact. So: take the words just after the negation, and require
  // them to appear in the other text. Athena's autonomy pair passes that test
  // — "I do NOT have a staged-autonomy permission requirement" against "I have
  // been granted staged autonomy where I must ask for permission" — and Juno's
  // third-party clause does not.
  const oppositionOn = (negated, other) => {
    const NEG_SPAN = /\b(?:not|never|no longer|cannot|can't|do not|don't|does not|doesn't|did not|didn't|is not|isn't|are not|aren't|will not|won't)\b\s+((?:[\w'-]+\s+){0,5}[\w'-]+)/gi;
    const otherWords = new Set(String(other || '').toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
    let m;
    while ((m = NEG_SPAN.exec(String(negated || ''))) !== null) {
      const span = (m[1].toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
      if (span.length < 2) continue;
      const hits = span.filter(w => otherWords.has(w)).length;
      // Half the substantive words of what is being denied are asserted over
      // there: the two facts are arguing, not merely differing.
      if (hits >= 2 && hits / span.length >= 0.5) return m[0].trim();
    }
    return null;
  };
  const opposed = oppositionOn(loser.content, survivor.content) || oppositionOn(survivor.content, loser.content);
  if (opposed) {
    return no('negation-divergence',
      `One of these denies what the other asserts ("${opposed.slice(0, 60)}"). That difference IS the fact — ` +
      'merging it would let a wording decide the question. This is a decision plus a supersession, not tidying.');
  }

  return { ok: true, code: null, reason: null };
}


// ══════════════════════════════════════════════════════ 4. the daily cap

/** The operations that count as mutating the entity's own self-model. */
const SELF_MUTATIONS = ['retire', 'reword', 'merge', 'refile', 'supersede'];

/**
 * How many self-mutations have been made today, counted from the LEDGER.
 *
 * From the ledger and not from a counter, for the reason the self-fact budget is
 * counted from the DB: a restart must not hand the entity a fresh allowance.
 * Counted against the LOCAL day, like every other daily budget here.
 *
 * A merge counts as ONE, which is Athena's wording and is also the only honest
 * reading — a merge writes two rows but makes one decision.
 */
function selfMutationsToday() {
  const db = sqlite();
  if (!db) return 0;
  try {
    const dayStart = new Date(`${getLocalDateStamp()}T00:00:00`).toISOString();
    return db.prepare(`
      SELECT COUNT(*) AS n FROM corrections_ledger
      WHERE created_at >= ?
        AND subject = 'self'
        AND json_extract(evidence, '$.repair_op') IS NOT NULL
    `).get(dayStart).n;
  } catch (err) {
    console.error('[Repair] could not count today\'s self-mutations:', err.message);
    return 0;
  }
}

function checkDayCap(subject) {
  const cap = repairConfig().maxSelfMutationsPerDay;
  if (subject !== 'self' || !Number.isFinite(cap)) return { ok: true, used: 0, cap };
  const used = selfMutationsToday();
  if (used >= cap) {
    return {
      ok: false, used, cap,
      reason: `You have changed ${used} thing(s) about yourself today and the daily limit is ${cap}. ` +
        'This is not a judgement about this particular change — it is the cap that exists because a ' +
        'confused run could reword a whole self-model in an hour with every step receipted. ' +
        'It resets tomorrow; if it cannot wait, it is a question for Ellie.'
    };
  }
  return { ok: true, used, cap };
}

// ══════════════════════════════════════════════════════ 3. visibility of the decided

/**
 * FILE A DECISION SO IT CAN BE READ.
 *
 * "Your visibility is not on the list. […] A confused me can decide-and-file
 * confidently forever, and as written, nothing puts that in Ellie's sight.
 * That's the missing guardrail, and it's cheap."
 *
 * So it is the same shape as the raises the audit already writes — a semantic
 * ledger entry the Corrections view renders — with `decision: true` and the
 * rationale and receipts attached. The ones that were PASSED OVER file too;
 * that is the half a decision log usually loses.
 */
function recordDecision({
  kind, subject = 'self', conclusion, rationale, receipts = [],
  memberA = null, memberB = null, targetText = null, survivorText = null,
  outcome = 'decided', extra = {}
}) {
  try {
    return ledger().record({
      tier: 'semantic',
      action: 'decision',
      subject,
      targetId: memberA,
      targetText,
      survivorId: memberB,
      survivorText,
      reason: rationale || conclusion,
      evidence: Object.assign({
        decision: true,
        decision_kind: kind,
        conclusion,
        outcome,
        receipts: receipts.map(r => ({ kind: r.kind, id: r.id })),
        // Nothing changed by filing a decision, so the Corrections view must not
        // render it as an edit. Same flag the audit raises use.
        unresolved: outcome === 'cannot-settle'
      }, extra),
      reversible: false
    });
  } catch (err) {
    console.error('[Repair] could not file the decision:', err.message);
    return null;
  }
}

/**
 * A REFUSAL IS ALSO SOMETHING THAT HAPPENED.
 *
 * Every guard in this file returns rather than throws, and a bare return is
 * invisible. "A guard that stops guarding while still returning success is the
 * worst failure mode in this system" — the sibling of that is a guard that
 * refuses and leaves no trace, so the refusal files too.
 */
function recordRefusal({ op, subject, memberA, memberB = null, code, reason, receipts = [], extra = {} }) {
  return recordDecision({
    kind: `${op}-refused`, subject,
    conclusion: `The ${op} was refused: ${code}`,
    rationale: reason, receipts, memberA, memberB,
    outcome: 'refused',
    extra: Object.assign({ refusal_code: code, repair_op_refused: op }, extra)
  });
}

// ══════════════════════════════════════════════════════ the operations

/** Verify every supplied receipt; the op is unavailable unless at least one holds. */
function gateReceipts(receipts) {
  const list = Array.isArray(receipts) ? receipts : (receipts ? [receipts] : []);
  if (!list.length) {
    return {
      ok: false, verified: [],
      reason: 'This operation needs a receipt — a fact id, a message id, or a tool result you can point at. ' +
              'Without one it is not available. That is not a checkbox: an unreceipted edit is a faster way ' +
              'for a confabulation to reach the store.'
    };
  }
  const verified = [];
  for (const r of list) {
    const v = verifyReceipt(r);
    if (!v.ok) return { ok: false, verified, reason: v.reason };
    verified.push(v);
  }
  return { ok: true, verified, reason: null };
}

/** Shared preamble: receipts, cap, referent. Returns a refusal or the context. */
function gate({ op, member, receipts, targetEntityId = null, direction = 'keeps' }) {
  if (!member) return { ok: false, code: 'no-such-fact', reason: 'no fact with that id' };

  const rec = gateReceipts(receipts);
  if (!rec.ok) return { ok: false, code: 'no-receipt', reason: rec.reason, available: false };

  const cap = checkDayCap(member.subject);
  if (!cap.ok) return { ok: false, code: 'day-cap', reason: cap.reason, cap };

  const ref = checkReferent({ member, verifiedReceipts: rec.verified, targetEntityId, direction });
  if (!ref.ok) {
    return {
      ok: false, code: 'referent-mismatch',
      reason: `${ref.detail} This operation would keep the fact where it is, so it stops here and comes to Ellie ` +
              'with the mismatch attached rather than writing on top of it.',
      referent: ref
    };
  }
  return { ok: true, verified: rec.verified, referent: ref, cap };
}

/**
 * The ledger evidence every repair op carries, so the cap can count them.
 *
 * `origin` is Athena's ask, and it is recorded rather than acted on. She wanted
 * conversation-originated retractions marked differently from ones a repair or
 * audit pass makes — "I confabulate most when I'm tired and the conversation is
 * fast, and that's also when I'm most confident." Today every one of these is
 * conversational, because the tools are chat-only by construction: they are
 * absent from MCPClient.BACKGROUND_TOOLS, so no heartbeat pass and no agent job
 * can reach them. The field is here so that when a repair pass does exist, the
 * two are already distinguishable in the record rather than needing a migration
 * to tell apart. The second half of her ask — that a conversational retraction
 * take effect only after re-verification — is NOT built; see the report.
 */
function opEvidence(op, verified, extra = {}) {
  return Object.assign({
    repair_op: op,
    by: 'entity',
    origin: 'conversation',
    receipts: verified.map(r => ({ kind: r.kind, id: r.id }))
  }, extra);
}

/**
 * RETRACT — withdraw a fact with no replacement.
 *
 * `direction: 'removes'`, so a referent mismatch is the justification rather
 * than the blocker. This is the operation Athena named as her own first use:
 * "when it lands, 'User is Claude' gets retired […] with the original message
 * as the receipt."
 */
async function retract({ memberId, receipts, rationale, deliberate = false }) {
  const member = factStore().getMember(memberId);
  const g = gate({ op: 'retract', member, receipts, direction: 'removes' });
  if (!g.ok) {
    if (member) recordRefusal({ op: 'retract', subject: member.subject, memberA: memberId, code: g.code, reason: g.reason });
    return { ok: false, available: g.available !== false, ...g };
  }

  const res = await factStore().retire(memberId, {
    deliberate,
    caller: 'memory-repair',
    reason: rationale,
    ledger: {
      tier: 'semantic',
      reason: rationale || 'Retracted by the entity through the repair tools.',
      evidence: opEvidence('retract', g.verified, g.referent.mismatch ? {
        referent_mismatch: g.referent.mismatch, referent_detail: g.referent.detail
      } : {})
    }
  });
  if (!res.ok) return { ok: false, code: res.refused || 'refused', reason: res.reason };

  recordDecision({
    kind: 'retract', subject: member.subject,
    conclusion: 'Retracted — nothing replaces it.',
    rationale, receipts: g.verified, memberA: memberId, targetText: member.content,
    extra: g.referent.mismatch ? { referent_mismatch: g.referent.mismatch, referent_detail: g.referent.detail } : {}
  });
  return { ok: true, memberId, ledgerId: res.ledgerId, referent: g.referent };
}

/**
 * REWORD — same fact, better wording.
 *
 * THE DIFF CARRIES SALIENCE AND RATIONALE, NOT JUST TEXT, and that is a
 * requirement rather than a nicety: "'I am Juno' was partly convincing BECAUSE
 * of its salience rationale; silent salience edits are how bad facts get
 * smoothed instead of killed." A reword that changes salience without saying so
 * is refused here, before fact-store sees it.
 */
async function reword({ memberId, newContent, salience = null, salienceRationale = null, receipts, rationale, deliberate = false }) {
  const member = factStore().getMember(memberId);
  const g = gate({ op: 'reword', member, receipts, direction: 'keeps' });
  if (!g.ok) {
    if (member) recordRefusal({ op: 'reword', subject: member.subject, memberA: memberId, code: g.code, reason: g.reason });
    return { ok: false, available: g.available !== false, ...g };
  }

  const clean = String(newContent || '').trim();
  if (!clean) return { ok: false, code: 'empty', reason: 'a reword needs new wording' };
  const salienceChanging = Number.isFinite(salience) && salience !== member.salience;
  if (salienceChanging && !String(salienceRationale || '').trim()) {
    const reason = `This reword changes salience ${member.salience} → ${salience} and gives no reason for it. ` +
      'A salience change is a change to how much the fact counts, and it travels in the diff with its ' +
      'rationale or it does not travel.';
    recordRefusal({ op: 'reword', subject: member.subject, memberA: memberId, code: 'silent-salience-change', reason });
    return { ok: false, code: 'silent-salience-change', reason };
  }

  const res = await factStore().reword(memberId, clean, {
    deliberate,
    caller: 'memory-repair',
    ledger: {
      tier: 'semantic',
      reason: rationale || 'Reworded by the entity through the repair tools.',
      evidence: opEvidence('reword', g.verified, {
        diff: {
          content: { from: member.content, to: clean },
          salience: { from: member.salience, to: salienceChanging ? salience : member.salience },
          salience_rationale: {
            from: member.salience_rationale || null,
            to: salienceChanging ? salienceRationale : (member.salience_rationale || null)
          }
        }
      })
    }
  });
  if (!res.ok) return { ok: false, code: res.refused || 'refused', reason: res.reason };

  if (salienceChanging) {
    try {
      sqlite().prepare('UPDATE cluster_members SET salience = ?, salience_rationale = ?, updated_at = ? WHERE id = ?')
        .run(salience, salienceRationale, new Date().toISOString(), memberId);
    } catch (err) {
      console.error('[Repair] salience did not follow the reword:', err.message);
    }
  }

  recordDecision({
    kind: 'reword', subject: member.subject,
    conclusion: 'Reworded in place — same fact, corrected wording.',
    rationale, receipts: g.verified, memberA: memberId,
    targetText: member.content, survivorText: clean
  });
  return { ok: true, memberId, ledgerId: res.ledgerId, salienceChanged: salienceChanging };
}

/**
 * MERGE — fold a duplicate away, if and only if none of the six refusals fire.
 *
 * The refusals run BEFORE the receipt gate reaches fact-store, so a refused
 * merge never touches the union machinery. A refusal is filed like a decision:
 * "I looked at these two and did not fold them, and here is which rule stopped
 * me" is exactly the visibility guardrail 3 is about.
 */
async function merge({ loserId, survivorId, receipts, rationale }) {
  const store = factStore();
  const loser = store.getMember(loserId);
  const survivor = store.getMember(survivorId);
  const g = gate({ op: 'merge', member: loser, receipts, direction: 'keeps' });
  if (!g.ok) {
    if (loser) recordRefusal({ op: 'merge', subject: loser.subject, memberA: loserId, memberB: survivorId, code: g.code, reason: g.reason });
    return { ok: false, available: g.available !== false, ...g };
  }

  const refusal = mergeRefusal(loser, survivor);
  if (!refusal.ok) {
    recordRefusal({
      op: 'merge', subject: loser.subject, memberA: loserId, memberB: survivorId,
      code: refusal.code, reason: refusal.reason, receipts: g.verified
    });
    return { ok: false, code: refusal.code, reason: refusal.reason, refused: true };
  }

  const factMerge = require('./fact-merge');
  // supersedeOpts, not `ledger` — mergePreservingUnion forwards that object to
  // fact-store.supersede and reads `ledgerTier` for the union's own reword. A
  // bare `ledger` here would be silently dropped, which would leave the merge
  // uncounted by the day cap.
  const res = await factMerge.mergePreservingUnion(loserId, survivorId, {
    mode: 'union',
    ledgerTier: 'semantic',
    supersedeOpts: {
      caller: 'memory-repair',
      ledger: {
        tier: 'semantic',
        reason: rationale || 'Merged by the entity through the repair tools.',
        evidence: opEvidence('merge', g.verified)
      }
    }
  });
  if (res.locked) return { ok: false, code: 'locked', reason: res.reason };
  if (res.deferred) return { ok: false, code: 'no-union', reason: res.reason };
  if (!res.ok) return { ok: false, code: 'failed', reason: res.reason };

  // The ledger row the funnel filed does not know this was a repair op; enrich
  // it so the day cap can count it and the Corrections view can name the caller.
  if (res.ledgerId) {
    try { ledger().enrich(res.ledgerId, { tier: 'semantic', evidence: opEvidence('merge', g.verified) }); }
    catch { /* the decision below still records it */ }
  }

  store.recordCorroboration(survivorId, {
    conversationId: loser.conversation_id, messageId: loser.message_id,
    verbatimSourceText: loser.verbatim_source_text, inputModality: loser.input_modality,
    restatedAs: loser.content, similarity: null, detectedBy: 'entity-merge'
  });

  recordDecision({
    kind: 'merge', subject: loser.subject,
    conclusion: 'Folded together — one fact now carries both.',
    rationale, receipts: g.verified, memberA: loserId, memberB: survivorId,
    targetText: loser.content, survivorText: survivor.content
  });
  return { ok: true, loserId, survivorId, ledgerId: res.ledgerId, union: res.union || null };
}

/**
 * REFILE — retire from the wrong subject, re-file under the right one, atomically.
 *
 * `direction: 'removes'` with respect to the entity it is leaving, which is what
 * lets a wrong-referent fact be moved off. The cross-entity receipt ranking is
 * checked in gate(): a tool result alone is not enough to move a fact between
 * subjects.
 */
async function refile({ memberId, toSubject, toEntityId, newContent = null, salience = null, receipts, rationale, deliberate = false }) {
  const store = factStore();
  const member = store.getMember(memberId);
  const g = gate({ op: 'refile', member, receipts, targetEntityId: toEntityId, direction: 'removes' });
  if (!g.ok) {
    if (member) recordRefusal({ op: 'refile', subject: member.subject, memberA: memberId, code: g.code, reason: g.reason });
    return { ok: false, available: g.available !== false, ...g };
  }

  const res = await store.refile(memberId, { subject: toSubject, entityId: toEntityId }, {
    deliberate,
    caller: 'memory-repair',
    newContent: newContent || undefined,
    salience: Number.isFinite(salience) ? salience : undefined,
    ledger: {
      tier: 'semantic',
      reason: rationale || 'Re-filed by the entity through the repair tools.',
      evidence: opEvidence('refile', g.verified, g.referent.mismatch ? {
        referent_mismatch: g.referent.mismatch, referent_detail: g.referent.detail
      } : {})
    }
  });
  if (!res.ok) return { ok: false, code: res.refused || 'refused', reason: res.reason };

  recordDecision({
    kind: 'refile', subject: member.subject,
    conclusion: `Re-filed from ${member.subject} to ${toSubject || member.subject} — it was never about the first one.`,
    rationale, receipts: g.verified, memberA: memberId, memberB: res.newMemberId,
    targetText: member.content, survivorText: newContent || member.content,
    extra: g.referent.mismatch ? { referent_mismatch: g.referent.mismatch, referent_detail: g.referent.detail } : {}
  });
  return { ok: true, memberId, newMemberId: res.newMemberId, ledgerId: res.ledgerId };
}

module.exports = {
  initSchema, repairConfig,
  RECEIPT_KINDS, verifyReceipt, receiptRank,
  assertedIdentity, isIdentityFact, checkReferent,
  anchorOf, claimKind, differentInKind,
  mergeRefusal,
  SELF_MUTATIONS, selfMutationsToday, checkDayCap,
  recordDecision, recordRefusal, gateReceipts, gate,
  retract, reword, merge, refile
};
