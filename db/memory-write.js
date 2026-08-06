/**
 * Explicit memory writes — the storage + decision layer behind the write_memory
 * tool.
 *
 * WHY THIS EXISTS: when Ellie said "write this to your memory," the entity said
 * yes and nothing happened. The fact reached storage only if the PASSIVE
 * extractor happened to pick it out of the transcript afterwards — and the
 * passive path is tuned for incidental facts, not for a deliberate instruction.
 * It routinely missed the thing that had just been asked for. (Case in point: it
 * did not catch the entity's own name on the day it chose one.) This makes the
 * write an action it can actually take, in the moment it is asked.
 *
 * DIRECT-EXECUTE, unlike create_cron_job. That tool proposes because it acts on
 * the world outside the conversation; this one records something Ellie just said
 * out loud and asked to have recorded, so routing it through an approval queue
 * would defeat the point. It is still bounded: reversible tier, rate-capped,
 * every call logged, and supersede-never-delete like every other memory path.
 *
 * FLOW — the tool dispatches to a classifier agent, which decides:
 *   1. SUBJECT — is this a fact about ELLIE or about the ENTITY ITSELF?
 *   2. supersede-or-append — does it replace something already held?
 *   3. salience.
 * Cluster placement is NOT decided here: assignToCluster already places a fact
 * semantically within its own subject's cluster space, which is a better
 * decision than a language model guessing a cluster name.
 *
 * SUBJECT IS THE DANGEROUS ONE. On 2026-07-27 self-facts were rewritten into the
 * third person and filed as Ellie's preferences — the entity's own observations
 * about itself became stored beliefs about her. So subject is asked for
 * EXPLICITLY as its own decision rather than falling out of phrasing, and the
 * answer is then checked mechanically against the grammatical person of the text
 * being stored (see verifySubjectAgreement). A disagreement between the two is
 * refused, not repaired: this is the substrate the entity's identity is built
 * from, and a refusal the model can report is far cheaper than a misattribution
 * nobody notices.
 */

const path = require('path');
const { getSqliteDb } = require('./database');
const { getConfig } = require('./config');

const MEMORY_DIR = require('./database').getMemoryDir();
const DAILY_DIR = path.join(MEMORY_DIR, 'daily');
const OPS_DIR = path.join(MEMORY_DIR, 'ops');

const HOUR_MS = 60 * 60 * 1000;

/** Provenance tag for everything this path writes (mirrors cron-jobs' KID_SOURCE). */
const WRITE_SOURCE = 'kid-memory-write';

/** Lazy requires — fact-extractor pulls in memory-clusters, which pulls in fact-store. */
function factExtractor() { return require('./fact-extractor'); }
function memoryClusters() { return require('./memory-clusters'); }
function factStore() { return require('./fact-store'); }

function opsLog(msg) {
  try { factExtractor().appendToOpsLog(msg, OPS_DIR); } catch (e) { /* best effort */ }
}
function dailyLog(msg) {
  try { factExtractor().appendToDailyLog(msg, DAILY_DIR); } catch (e) { /* best effort */ }
}

/** Hot-read config each call so knobs take effect without a restart. */
function cfg() {
  const c = (getConfig().tools && getConfig().tools.memoryWrite) || {};
  return {
    enabled: c.enabled !== false,
    maxWritesPerHour: Math.max(1, c.maxWritesPerHour || 20)
  };
}

// ============ rate caps ============

/**
 * Trailing-hour cap, counted from the DB so a restart does not hand the entity a
 * fresh budget. Same shape as the cron tool's cap. There is deliberately NO
 * total ceiling: unlike scheduled jobs, accumulating remembered facts is the
 * system working, not a resource leak.
 * @returns {{ok: boolean, reason?: string}}
 */
function checkCaps() {
  const db = getSqliteDb();
  if (!db) return { ok: false, reason: 'database unavailable' };
  const { maxWritesPerHour } = cfg();

  const sinceIso = new Date(Date.now() - HOUR_MS).toISOString();
  const recent = db.prepare(
    "SELECT COUNT(*) AS n FROM cluster_members WHERE source = ? AND datetime(created_at) > datetime(?)"
  ).get(WRITE_SOURCE, sinceIso).n;

  if (recent >= maxWritesPerHour) {
    return { ok: false, reason: `memory-write rate cap reached (${recent}/${maxWritesPerHour} in the last hour)` };
  }
  return { ok: true };
}

/** Current cap usage — for the Thinking tab and for tests. */
function capStatus() {
  const db = getSqliteDb();
  const { maxWritesPerHour } = cfg();
  if (!db) return { recentHour: 0, maxWritesPerHour };
  const sinceIso = new Date(Date.now() - HOUR_MS).toISOString();
  const recentHour = db.prepare(
    "SELECT COUNT(*) AS n FROM cluster_members WHERE source = ? AND datetime(created_at) > datetime(?)"
  ).get(WRITE_SOURCE, sinceIso).n;
  return { recentHour, maxWritesPerHour };
}

// ============ the classifier agent ============

/**
 * Decide who a remembered fact is ABOUT, and phrase it for storage.
 *
 * Returns the subject as its own explicit answer — not a by-product of how the
 * sentence came out — because the misattribution bug came from exactly that
 * inference. The prompt also fixes the grammatical person per subject, so the
 * stored text and the subject label can be cross-checked afterwards.
 *
 * @param {string} statement - what Ellie asked to have remembered, verbatim
 * @param {string} [conversationContext] - recent turns, for resolving pronouns
 * @returns {Promise<{subject: 'self'|'user', fact: string, emphasis: boolean, reasoning: string}|null>}
 */
async function classifySubject(statement, conversationContext = '', sourceMessage = '') {
  const memoryManager = require('./memory-manager');

  const systemPrompt = `You are the routing step of a personal memory system. Someone has explicitly asked an AI assistant to remember something. You decide WHO THE FACT IS ABOUT and write the single sentence that will be stored.

There are exactly two subjects, and getting this wrong corrupts the memory:

- USER — a fact about Ellie, the human. Her preferences, her work, her life, people she knows, things she owns or uses.
  Write it in the THIRD PERSON, starting with "User". Example: "User's favorite color is blue."

- SELF — a fact about the AI assistant itself. Its own name, its own traits, how it works, what it has noticed about itself, its own preferences and history.
  Write it in the FIRST PERSON, as the assistant speaking about itself. Example: "I tend to over-explain when I am unsure."

CRITICAL RULE: never convert between the two. If the statement is about the assistant, it is SELF and it stays first-person — do NOT rewrite it as a fact about the user. A statement like "you're very direct" or "your name is X" or "remember that you prefer Y" is about the ASSISTANT: subject SELF. A statement like "I prefer Y" or "my name is X", said by the human, is about the USER: subject USER.

Watch the speaker. The human says "I"/"my" about HERSELF (→ USER) and "you"/"your" about the ASSISTANT (→ SELF). Your output flips this: a USER fact says "User...", a SELF fact says "I...".

Also report whether the request carried EMPHASIS — the person signalled this is important or must not be forgotten ("this is important", "make sure you remember", "never forget").

Respond in exactly this format and nothing else:
SUBJECT: USER or SELF
FACT: the one sentence to store, in the person required above
EMPHASIS: YES or NO
WHY: one short line`;

  // The VERBATIM message outranks the assistant's paraphrase. The assistant
  // rewrites pronouns when it fills in a tool argument — "remember that YOU
  // prefer X" arrives as "I prefer X", which is indistinguishable from Ellie
  // talking about herself. Only the original utterance still carries who was
  // being described, so it is presented as the authority and the paraphrase is
  // labelled as what it is.
  const parts = [];
  if (conversationContext) parts.push(`Recent conversation, for context only:\n${conversationContext}`);
  if (sourceMessage && sourceMessage.trim() && sourceMessage.trim() !== statement.trim()) {
    parts.push(
      `WHAT THE HUMAN ACTUALLY TYPED (authoritative — trust this over the paraphrase below, especially for pronouns):\n"${String(sourceMessage).slice(0, 800)}"`
    );
    parts.push(`The assistant's paraphrase of what to store (it may have altered the pronouns, so do not rely on them):\n"${statement}"`);
  } else {
    parts.push(`The human just said, asking for this to be remembered:\n"${statement}"`);
  }
  const userPrompt = `${parts.join('\n\n')}\n\nWho is this fact about, and what exactly should be stored?`;

  try {
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 220 });
    const text = String(content || '');
    const grab = (label) => {
      const m = text.match(new RegExp(`^\\s*${label}\\s*:\\s*(.+)$`, 'im'));
      return m ? m[1].trim() : '';
    };
    const subjRaw = grab('SUBJECT').toLowerCase();
    const fact = grab('FACT').replace(/^["']|["']$/g, '').trim();
    const emphasis = /^y/i.test(grab('EMPHASIS'));
    const reasoning = grab('WHY');

    if (!fact) return null;
    // No default. An unparseable subject must fail the write, not silently pick
    // one — "user" is the wrong guess in exactly the case that caused the bug.
    let subject = null;
    if (/\bself\b/.test(subjRaw)) subject = 'self';
    else if (/\buser\b/.test(subjRaw)) subject = 'user';
    if (!subject) return null;

    return { subject, fact, emphasis, reasoning };
  } catch (err) {
    console.error('[MemoryWrite] classifySubject error:', err.message);
    return null;
  }
}

/**
 * Cross-check the classifier against itself: does the grammatical person of the
 * text it wrote match the subject it chose?
 *
 * This is the guard that would have caught the 2026-07-27 misattribution, where
 * self-observations arrived phrased as third-person facts about Ellie. The two
 * signals are produced by the same call but are independent enough that a
 * disagreement is real evidence of a routing error.
 *
 * @returns {{ok: boolean, reason?: string}}
 */
function verifySubjectAgreement(subject, fact) {
  const t = String(fact || '').trim();
  if (!t) return { ok: false, reason: 'empty fact text' };

  const looksThirdPerson = /^(the\s+)?(user|ellie)\b/i.test(t);
  const looksFirstPerson = /^(i|my|i'm|i've|i'd|i'll)\b/i.test(t);

  if (subject === 'self' && looksThirdPerson) {
    return { ok: false, reason: `classified as a fact about the assistant but written in the third person about the user ("${t.slice(0, 60)}")` };
  }
  if (subject === 'user' && looksFirstPerson) {
    return { ok: false, reason: `classified as a fact about the user but written in the first person as the assistant ("${t.slice(0, 60)}")` };
  }
  // A user-fact must be anchored to the user explicitly — an unanchored sentence
  // is how a self-observation slips into the user's cluster space wearing no
  // pronoun at all.
  if (subject === 'user' && !looksThirdPerson) {
    return { ok: false, reason: `a fact about the user must name her explicitly (start with "User"), got "${t.slice(0, 60)}"` };
  }
  return { ok: true };
}

/**
 * Ask whether this new fact REPLACES one the entity already holds.
 *
 * Reuses the existing contradiction machinery — candidate lookup by embedding
 * similarity, scoped to the same subject, then the same judge the passive path
 * uses — rather than a second, differently-tuned opinion about what supersession
 * means. The difference here is context: the judge is told this was an explicit
 * "remember this" instruction, which is a much stronger correction signal than
 * an incidental mention ("actually my favourite colour is blue" is a
 * replacement, not a second favourite colour).
 *
 * @returns {Promise<{oldMemberId: string, oldContent: string, oldSalience: number}|null>}
 */
async function findSupersession(fact, subject, statement) {
  try {
    const candidates = await memoryClusters().findContradictionCandidates(fact, {
      subject,
      // Wider net than the passive default (0.45). An explicit correction often
      // rewords heavily, and this path has a human in the loop who just told us
      // the record is wrong — recall matters more than precision here.
      threshold: 0.35,
      limit: 6
    });
    if (!candidates.length) return null;

    for (const candidate of candidates) {
      const { verdict } = await factExtractor().judgeContradiction(fact, candidate.content, {
        userMessage: statement,
        corrects: `an explicit instruction to remember "${fact}" — if the existing fact states an older version of this same thing, it is being replaced`
      });
      if (verdict === 'yes') {
        return {
          oldMemberId: candidate.memberId,
          oldContent: candidate.content,
          oldSalience: candidate.salience ?? 5
        };
      }
    }
    return null;
  } catch (err) {
    console.error('[MemoryWrite] findSupersession error:', err.message);
    return null;
  }
}

/** Nearby facts of the same subject — context for salience scoring. */
function nearbyContext(subject) {
  try {
    const db = getSqliteDb();
    if (!db) return '';
    const rows = db.prepare(
      "SELECT content FROM cluster_members WHERE subject = ? AND status = 'active' ORDER BY datetime(created_at) DESC LIMIT 20"
    ).all(subject);
    return rows.map(r => `- ${r.content}`).join('\n');
  } catch { return ''; }
}

// ============ the write ============

/**
 * Record a fact the entity was explicitly asked to remember.
 *
 * @param {Object} p
 * @param {string} p.statement - what to remember, in Ellie's words
 * @param {string} [p.context] - recent conversation, for pronoun resolution
 * @param {string} [p.conversationId]
 * @param {string} [p.messageId] - message this write came from (provenance)
 * @param {string} [p.inputModality] - 'stt' | 'typed' | 'unknown' (provenance)
 * @returns {Promise<{ok: boolean, error?: string, ...}>}
 */
async function write({ statement, context = '', conversationId = null, userMessage = '', messageId = null, inputModality = null }) {
  const args = { statement, context: context ? `${String(context).slice(0, 200)}…` : '' };
  const log = (outcome, detail, refId = null) => {
    try {
      require('./cron-jobs').logToolCall({ tool: 'write_memory', args, outcome, detail, refId, conversationId });
    } catch (e) { console.error('[MemoryWrite] logToolCall failed:', e.message); }
  };

  const { enabled } = cfg();
  if (!enabled) {
    log('error', 'tool disabled in config');
    return { ok: false, error: 'The write_memory tool is disabled.' };
  }
  if (typeof statement !== 'string' || !statement.trim()) {
    log('error', 'missing statement');
    return { ok: false, error: 'Nothing to remember — statement is required.' };
  }

  const caps = checkCaps();
  if (!caps.ok) {
    log('rejected-cap', caps.reason);
    opsLog(`write_memory refused by cap: ${caps.reason} — "${String(statement).slice(0, 80)}"`);
    return { ok: false, error: `Not saved — ${caps.reason}. Tell her plainly that you have hit your own limit for this rather than retrying.` };
  }

  // --- 1. subject + phrasing ---
  const decision = await classifySubject(statement.trim(), context, userMessage);
  if (!decision) {
    log('error', 'classifier could not determine subject');
    return { ok: false, error: 'I could not work out whether that is a fact about you or about me, so I did not save it. Ask again and say which it is.' };
  }

  const agree = verifySubjectAgreement(decision.subject, decision.fact);
  if (!agree.ok) {
    // Refuse rather than repair — see the module header.
    log('error', `subject/person mismatch: ${agree.reason}`);
    opsLog(`write_memory refused — subject/person disagreement: ${agree.reason}`);
    return { ok: false, error: 'I could not save that safely: I was not confident whether it is a fact about you or about me, and I will not guess on that. Tell me which and I will write it.' };
  }

  const { subject, fact } = decision;

  // --- 1b. identity lock ---
  // Checked BEFORE anything is scored or stored. A locked slot refuses a
  // competing fact outright rather than storing it and skipping the
  // supersession — leaving a second name active beside the first is how the
  // lock would be walked around by appending instead of replacing.
  //
  // The refusal is handed back as text the model is told to SAY. It must not
  // read as a quiet success: "your name is Bob" has to produce him telling her
  // it is locked, not a warm agreement and no write.
  const identityLock = require('./identity-lock');
  const lockCheck = identityLock.checkNewFact(fact, subject);
  if (!lockCheck.ok && lockCheck.blocked) {
    identityLock.recordRefusal({
      category: lockCheck.category,
      attempted: fact,
      existing: lockCheck.existing.content,
      via: 'write_memory'
    });
    log('rejected-locked', `identity lock (${lockCheck.category}) refused: "${fact.slice(0, 80)}"`);
    return { ok: false, locked: true, category: lockCheck.category, error: lockCheck.message };
  }
  if (!lockCheck.ok && lockCheck.duplicate) {
    // Already held, verbatim. Not a violation and not worth an alarm.
    log('ok', `identity fact already held (${lockCheck.category}), nothing to write`, lockCheck.existing.id);
    return {
      ok: true, memberId: lockCheck.existing.id, subject: 'self', fact: lockCheck.existing.content,
      salience: lockCheck.existing.salience ?? 10, cluster: null, superseded: null, unchanged: true
    };
  }

  // --- 2. salience ---
  let salience = 5;
  try {
    const scored = await factExtractor().scoreSalience(fact, nearbyContext(subject));
    salience = scored.salience ?? 5;
  } catch (e) { /* default stands */ }
  // Being asked outright to remember something is itself evidence it matters;
  // an explicit "this is important" raises the floor further.
  const floor = decision.emphasis ? 8 : 6;
  if (salience < floor) salience = floor;

  // --- 3. supersede or append ---
  const supersession = await findSupersession(fact, subject, statement.trim());
  if (supersession && supersession.oldSalience > salience) salience = supersession.oldSalience;

  // --- 4. claim type (self-facts only — the audit reads this split) ---
  let claimType = null;
  if (subject === 'self') {
    try { claimType = await factExtractor().classifyClaimType(fact); }
    catch { claimType = 'declaration'; }
  }

  // --- 5. write ---
  // One write, into SQLite. User-facts used to be appended to MEMORY.md first;
  // that file is gone and the injected block renders from the database.
  const cfgAll = getConfig();
  const extractionProvider = cfgAll.models.extraction.provider;
  const extractionModel = cfgAll.models.extraction.model;
  const { getProviderInstance } = require('./config');
  const extInst = getProviderInstance(extractionProvider, cfgAll.models.extraction.instance);
  const extractionHost = extInst ? extInst.host : 'http://localhost:11434';

  // An explicit "remember this" is the strongest evidence there is: Ellie asked
  // for it directly. The verbatim source is her actual message, not the cleaned
  // fact text, so a later correction can weigh what she really said.
  const res = await memoryClusters().assignToCluster(
    fact, extractionProvider, extractionModel, '', extractionHost,
    WRITE_SOURCE, salience, subject, claimType,
    {
      conversationId,
      messageId,
      verbatimSourceText: userMessage || statement,
      inputModality: inputModality || 'unknown',
      salienceRationale: 'Ellie asked me directly to remember this'
    }
  );
  if (!res || !res.memberId) {
    log('error', 'cluster assignment failed');
    return { ok: false, error: 'I could not store that — the memory write failed. Nothing was saved.' };
  }

  // --- 5b. set-once lock ---
  // The slot was open (checkNewFact passed above), so this is the FIRST time
  // this identity category has been asserted — a setup assignment. It locks
  // itself here, and every later attempt is refused.
  const lockedCats = identityLock.autoLock(res.memberId, fact, subject);

  // --- 6. apply supersession through the one path that updates all three stores ---
  let superseded = null;
  if (supersession) {
    const sres = await factStore().supersede(supersession.oldMemberId, res.memberId);
    if (sres.ok) {
      superseded = supersession.oldContent;
      dailyLog(`Superseded fact on request: "${supersession.oldContent}" → "${fact}" (asked to remember it directly)`);
    }
  }

  dailyLog(`Asked to remember, and did: "${fact}" (about ${subject === 'self' ? 'myself' : 'Ellie'}, salience ${salience}/10${superseded ? `, replacing "${superseded}"` : ''})`);
  log('ok', `${subject} fact${superseded ? ' (superseded 1)' : ''}, salience ${salience}, cluster "${res.clusterName}"`, res.memberId);

  console.log(`[MemoryWrite] stored ${subject} fact ${res.memberId.slice(0, 8)} salience=${salience} cluster="${res.clusterName}"${superseded ? ` superseded="${superseded.slice(0, 50)}"` : ''}`);

  return {
    ok: true,
    memberId: res.memberId,
    subject,
    fact,
    salience,
    cluster: res.clusterName,
    superseded,
    lockedCategories: lockedCats.length ? lockedCats : null
  };
}

module.exports = {
  WRITE_SOURCE,
  write,
  checkCaps,
  capStatus,
  classifySubject,
  verifySubjectAgreement,
  findSupersession,
};
