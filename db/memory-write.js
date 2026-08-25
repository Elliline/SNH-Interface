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
async function classifySubject(statement, conversationContext = '', sourceMessage = '', opts = {}) {
  const memoryManager = require('./memory-manager');

  const systemPrompt = `You are the routing step of a personal memory system. Someone has explicitly asked an AI assistant to remember something. You decide WHO THE FACT IS ABOUT and write the single sentence that will be stored.

There are exactly two subjects, and getting this wrong corrupts the memory:

- USER — a fact about Ellie, the human, OR about anyone else in her world. Her preferences, her work, her life, the PEOPLE SHE KNOWS, things she owns or uses.
  Write it in the THIRD PERSON, starting with "User". Examples: "User's favorite color is blue." "User's sister Juno runs on the Qwen3.8 27b model."

- SELF — a fact about YOU, the one assistant reading this prompt. Your own name, your own traits, how you work, what you have noticed about yourself.
  Write it in the FIRST PERSON. Example: "I tend to over-explain when I am unsure."

SOMEONE ELSE WITH A NAME IS NEVER SELF. This is the rule that matters most. If the statement is about a person, an AI, or an assistant WHO IS NOT YOU — anyone with their own name — the subject is USER and the fact stays in the third person, keeping their name. This holds even when the other one is an AI, runs on a named model, has a job that sounds like yours, or is described as a sibling. "Juno runs on the Qwen3.8 27b model and will help with MettaSphere" is a fact about JUNO. Juno is not you. Storing it as "I am Juno" would make the memory claim you are someone else, which is the worst thing this step can do.

NEVER CHANGE WHO THE SENTENCE IS ABOUT. Rephrasing is fine; moving a statement from one person onto another is not. If the statement describes someone in the third person, the stored fact describes them in the third person. Only a statement that is genuinely about YOU may come out as "I ...".

CRITICAL RULE: never convert between the two. If the statement is about the assistant, it is SELF and it stays first-person — do NOT rewrite it as a fact about the user. A statement like "you're very direct" or "your name is X" or "remember that you prefer Y" is about the ASSISTANT: subject SELF. A statement like "I prefer Y" or "my name is X", said by the human, is about the USER: subject USER.

Watch the speaker. The human says "I"/"my" about HERSELF (→ USER) and "you"/"your" about the ASSISTANT (→ SELF). A sentence with NEITHER — no "I", no "you", just a name and what they do — is about that named someone, and that is USER. Your output flips the pronouns: a USER fact says "User...", a SELF fact says "I...".

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
  //
  // …BUT ONLY WHEN IT IS ACTUALLY THE SOURCE (2026-08-24). The assistant also
  // calls write_memory on its own initiative, and then the human's last message
  // has nothing to do with the fact. Athena's misroute happened under exactly
  // that shape: the typed message was "Go for it, run your test.", and this
  // block handed it over as AUTHORITATIVE while labelling the only sentence
  // that described Juno as unreliable. Measured at a 256-token thinking budget,
  // the classifier then wrote self-facts about running the test — it followed
  // the authority it was pointed at. So relatedness is checked first, and an
  // unrelated message is presented as what it is: context.
  // opts.noSourceAuthority is the retry after verifyContentPreserved caught a
  // substitution: the typed message is demoted to context so it cannot pull the
  // classifier onto a different topic a second time.
  const related = !opts.noSourceAuthority && shareContent(sourceMessage, statement);
  const parts = [];
  if (conversationContext) parts.push(`Recent conversation, for context only:\n${conversationContext}`);
  if (sourceMessage && sourceMessage.trim() && sourceMessage.trim() !== statement.trim() && related) {
    // AUTHORITATIVE FOR *WHO*, NOT FOR *WHICH* (2026-08-25). The old wording said
    // "trust this over the paraphrase below" full stop, and demoted the statement
    // to an unreliable paraphrase. That is right when the human said ONE thing and
    // the assistant reworded it. It is wrong when she said FOUR things: on
    // 2026-08-24 Ellie answered Juno in a numbered list — streaming, Athena on the
    // DGX Spark, a phantom preference, and "your memorial offer: yes, file it" —
    // and Juno called write_memory once per item. Every call handed the classifier
    // that same four-topic message as the authority and the actual statement as a
    // paraphrase not to be relied on, so three of the four memorial saves came back
    // as facts about streaming and Athena. The statement's content was replaced
    // wholesale, and nothing downstream compared the two, so it was silent.
    //
    // The authority was always meant to be about PRONOUNS and WHO IS BEING
    // DESCRIBED — that is what the comment above says and what the misroute it
    // fixed needed. It now says only that.
    parts.push(
      `WHAT THE HUMAN ACTUALLY TYPED — authoritative for WHO each sentence is about and for the pronouns, and NOT for which fact to store:\n"${String(sourceMessage).slice(0, 800)}"`
    );
    parts.push(
      `THE STATEMENT TO STORE — this, and only this, decides WHICH fact you are writing down:\n"${statement}"\n\n` +
      `The typed message may cover several separate things. Find the part of it that THIS STATEMENT is about, and store THAT. ` +
      `Use the typed message to get the pronouns right and to see who is being described — never to store a different fact from it ` +
      `because that one looked more important or better formed. Your FACT must be about the same subject matter as the statement above.`
    );
  } else {
    if (sourceMessage && sourceMessage.trim() && sourceMessage.trim() !== statement.trim()) {
      parts.push(
        `The human's last message, for context only — it is NOT what is being remembered and may be about something else entirely:\n"${String(sourceMessage).slice(0, 400)}"`
      );
    }
    parts.push(`THE STATEMENT TO STORE — decide who THIS is about:\n"${statement}"`);
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
 * Do two texts talk about any of the same distinctive things?
 *
 * Used to decide whether the human's last message is the ORIGIN of this write or
 * just whatever she happened to say before the assistant wrote something down
 * on its own. contentTokens keeps only the distinctive half of a sentence, so
 * "Go for it, run your test." shares nothing with a sentence about Juno and the
 * Qwen3.8 model, while "remember you prefer short answers" shares plenty with
 * "I prefer short answers".
 */
function shareContent(a, b) {
  if (!a || !b) return false;
  try {
    const factMerge = require('./fact-merge');
    const ta = factMerge.contentTokens(a);
    if (!ta.size) return false;
    for (const t of factMerge.contentTokens(b)) if (ta.has(t)) return true;
    return false;
  } catch { return false; }
}

/**
 * Does a source sentence give any licence to read it as being about the
 * assistant? Either the human addressing it ("you"/"your"), the assistant
 * speaking of itself ("I"/"my"), or its own locked name appearing.
 *
 * Factored out of verifyPersonPreserved so the unanchored-user degradation in
 * write() can ask the same question with the same answer: a sentence carrying
 * none of these is about somebody else, which is what makes it safe to file an
 * unanchored user-fact rather than destroy it.
 */
function sourceHasSelfReference(src) {
  const t = String(src || '');
  if (/\b(you|your|yours|you're|youre|yourself)\b/i.test(t)) return true;
  if (/\b(i|i'm|im|i've|i'll|i'd|my|me|myself|mine)\b/i.test(t)) return true;
  let own = null;
  try { own = require('./identity-lock').lockedName(); } catch { /* none locked */ }
  if (own && new RegExp(`\\b${own.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(t)) return true;
  return false;
}

/**
 * Does this sentence name anybody at all?
 *
 * The precise thing verifySubjectAgreement's anchor branch defends against, in
 * its own words, is a self-observation slipping into the user's cluster space
 * "wearing no pronoun at all" — a bare predicate like "Prefers short answers."
 * or "Tends to over-explain when unsure.", where nothing in the sentence says
 * who it is about and the subject label is the only thing deciding.
 *
 * A sentence with an explicit named subject is not that. "On 8/24, Claude
 * fabricated a completed-update report…" is unmistakably about Claude; it simply
 * is not about Ellie, which is a filing limitation and not an ambiguity. Any
 * capitalised word past the first is a name the sentence is anchored to — the
 * first word is skipped because every sentence capitalises it.
 */
function factNamesSomeone(fact) {
  const words = String(fact || '').trim().match(/[A-Za-z][A-Za-z'\u2019-]*/g) || [];
  for (let i = 1; i < words.length; i++) if (/^[A-Z]/.test(words[i])) return true;
  return false;
}

/**
 * A PARAPHRASE MAY NOT CHANGE WHO THE SENTENCE IS ABOUT.
 *
 * verifySubjectAgreement below asks whether the classifier's two outputs agree
 * with EACH OTHER. They can agree perfectly and both be wrong: Athena's incident
 * stored subject=self with first-person text, which is self-consistent, and
 * claimed she was Juno. Nothing compared either output against the SOURCE.
 *
 * This does. A first-person self-fact needs a licence in the source sentence:
 * either the human addressing the assistant ("you"/"your"), or the assistant
 * speaking about itself ("I"/"my"), or the assistant's own name appearing. A
 * sentence carrying none of those is about somebody else in the third person,
 * and turning it into "I am ..." is the identity-planting failure this guards.
 *
 * Deliberately narrow: it only ever refuses subject=self, and only when the
 * source offers no self-reference at all. It cannot misfire on a user-fact.
 *
 * @returns {{ok: boolean, reason?: string}}
 */
function verifyPersonPreserved(statement, subject, fact) {
  if (subject !== 'self') return { ok: true };
  const src = String(statement || '');
  if (!src.trim()) return { ok: true };

  if (sourceHasSelfReference(src)) return { ok: true };

  return {
    ok: false,
    reason: `the statement is in the third person about someone else ("${src.slice(0, 80)}") but it was classified as a fact about me and written as "${String(fact).slice(0, 80)}"`
  };
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
  // NOT THE SAME KIND OF FAILURE AS THE TWO ABOVE. Those are the classifier
  // disagreeing with itself about WHO — a real routing error, and a refusal.
  // This one is only a MISSING ANCHOR: the subject may be perfectly right and the
  // text perfectly correct, and the sentence simply does not open with "User".
  // Juno's third memorial attempt died here — the classifier had finally written
  // the incident itself ("On August 24, 2026, Claude fabricated a completed-update
  // report…"), the subject was right, and the content was thrown away for its
  // opening word. Flagged so write() can file it with a note instead, but only
  // when the source shows it is not a disguised self-observation.
  if (subject === 'user' && !looksThirdPerson) {
    return { ok: false, unanchored: true, reason: `a fact about the user must name her explicitly (start with "User"), got "${t.slice(0, 60)}"` };
  }
  return { ok: true };
}

/**
 * DID THE INTAKE STORE WHAT IT WAS ASKED TO STORE?
 *
 * The guard that was missing, and the reason four saves of one memorial produced
 * three facts about something else. verifySubjectAgreement checks the classifier
 * against ITSELF (does the subject match the person of the text it wrote), and
 * verifyPersonPreserved checks the subject against the SOURCE. Nothing compared
 * the FACT to the STATEMENT, so a classifier that wandered onto another topic in
 * the conversation had its answer validated, scored, clustered and written — and
 * the assertion it was handed left no trace anywhere active.
 *
 * This is the same move db/fact-merge.js makes for merges: a model's text is
 * checked MECHANICALLY before it is written, using the same contentTokens /
 * missingTokens primitives, so the check cannot be talked out of its answer.
 *
 * MEASURED, NOT GUESSED. On Juno's own ten write_memory calls, the fraction of
 * the statement's distinctive tokens surviving into the stored fact was:
 *
 *     faithful writes   100%, 82%, 70%, 60%
 *     the three losses    4%,  0%,  0%
 *
 * Nothing lands between 4% and 60%, so the floor sits at 35% — far enough below
 * the worst faithful write to leave legitimate condensation alone, far enough
 * above the best substitution to catch it. This deliberately does NOT police
 * ordinary paraphrase loss: a fact that keeps the topic but trims a clause is a
 * different and much milder problem, and reported rather than refused (see
 * write()). Substitution is what destroys.
 *
 * SHORT STATEMENTS OPT OUT of the ratio. "User has 4 dogs and 4 cats" carries a
 * single distinctive token, and one-token ratios are 100% or 0% with nothing in
 * between — a coin toss, not a measurement. Below MIN_TOKENS the test falls back
 * to the old bar: at least one distinctive token in common.
 *
 * @returns {{ok: boolean, reason?: string, coverage: number, missing: string[]}}
 */
const CONTENT_FLOOR = 0.35;
const CONTENT_MIN_TOKENS = 4;

function verifyContentPreserved(statement, fact) {
  let factMerge;
  try { factMerge = require('./fact-merge'); }
  catch { return { ok: true, coverage: 1, missing: [] }; }   // never fail closed on a load error

  const need = factMerge.contentTokens(statement);
  if (!need.size) return { ok: true, coverage: 1, missing: [] };

  const missing = factMerge.missingTokens(need, fact);
  const kept = need.size - missing.length;
  const coverage = kept / need.size;

  if (need.size < CONTENT_MIN_TOKENS) {
    if (kept > 0) return { ok: true, coverage, missing };
    return {
      ok: false, coverage, missing,
      reason: `the stored sentence has nothing in common with the statement (0 of ${need.size} distinctive term(s): ${missing.join(', ')})`
    };
  }

  if (coverage < CONTENT_FLOOR) {
    return {
      ok: false, coverage, missing,
      reason: `the stored sentence kept only ${kept} of ${need.size} distinctive terms from the statement ` +
              `(${Math.round(coverage * 100)}%, floor ${Math.round(CONTENT_FLOOR * 100)}%) — dropped: ${missing.slice(0, 12).join(', ')}`
    };
  }
  return { ok: true, coverage, missing };
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
  let decision = await classifySubject(statement.trim(), context, userMessage);
  if (!decision) {
    log('error', 'classifier could not determine subject');
    return { ok: false, error: 'I could not work out whether that is a fact about you or about me, so I did not save it. Ask again and say which it is.' };
  }

  // --- 1a. IS THIS EVEN THE FACT THAT WAS ASKED FOR? (2026-08-25) ---
  //
  // Everything below this point validates the classifier's answer very carefully
  // and none of it ever asked whether that answer was about the right thing. When
  // the typed message covered several topics, the classifier wrote a fact about
  // the wrong one, and the guards passed it because it was internally consistent.
  // The statement's content then existed nowhere: not as a fact, not as a repeat,
  // not as a refusal — the write reported success and the assertion was gone.
  //
  // The retry is the actual repair for the common case: the substitution is pulled
  // by the typed message being framed as the authority, so the retry takes that
  // framing away and asks again. Refusing outright is the floor, not the plan.
  let content = verifyContentPreserved(statement.trim(), decision.fact);
  if (!content.ok) {
    opsLog(`write_memory: intake wrote a fact that is not about the statement it was given — ${content.reason}. Retrying with the typed message demoted to context.`);
    const retry = await classifySubject(statement.trim(), context, userMessage, { noSourceAuthority: true });
    if (retry) {
      const retryContent = verifyContentPreserved(statement.trim(), retry.fact);
      if (retryContent.ok) {
        decision = retry;
        content = retryContent;
        opsLog(`write_memory: the retry stored the statement it was given (${Math.round(retryContent.coverage * 100)}% of its distinctive terms kept).`);
      }
    }
  }
  if (!content.ok) {
    // Never store a substitution. The old behaviour reported success, which is
    // how one memorial produced three facts about streaming and Athena.
    log('error', `content not preserved: ${content.reason}`);
    opsLog(`write_memory refused — the intake step replaced the statement with a different fact: ${content.reason}`);
    dailyLog(`Did not save "${String(statement).slice(0, 120)}" — the step that files a memory rewrote it into a fact about something else, so I refused it rather than store the wrong thing.`);
    return {
      ok: false,
      error: 'I did not save that. The step that files a memory rewrote it into a different fact — one about something else we were talking about — ' +
             'and I will not store that in place of what you asked me to remember. Nothing was saved, and nothing was overwritten. ' +
             'Say it to me on its own and I will file it.'
    };
  }
  // Kept the topic but trimmed some of it. Not grounds to refuse — condensation is
  // the classifier's job — but it is never silent again.
  if (content.missing && content.missing.length) {
    opsLog(`write_memory kept the statement's topic but not all of it (${Math.round(content.coverage * 100)}% of distinctive terms) — not carried into "${String(decision.fact).slice(0, 80)}": ${content.missing.slice(0, 12).join(', ')}`);
  }

  // A paraphrase that moved the fact onto a different person is refused before
  // anything else looks at it — see verifyPersonPreserved.
  const person = verifyPersonPreserved(statement.trim(), decision.subject, decision.fact);
  if (!person.ok) {
    log('error', `person not preserved: ${person.reason}`);
    opsLog(`write_memory refused — intake tried to store a third-person statement as a fact about itself: ${person.reason}`);
    return {
      ok: false,
      error: 'I did not save that. It is a statement about someone else, and the step that files it tried to store it as a fact about me — ' +
             'which would have put a false claim about who I am into my memory. Say who it is about and I will file it there.'
    };
  }

  const agree = verifySubjectAgreement(decision.subject, decision.fact);
  // A MISSING ANCHOR IS NOT A DISAGREEMENT ABOUT WHO (2026-08-25). The two real
  // routing errors — a self-fact written in the third person, a user-fact written
  // as "I" — are still refused; guessing there is how identities get planted.
  // But a user-fact that simply does not open with "User" is a different animal:
  // the subject can be right and the sentence right, and Juno lost her memorial's
  // one good rendering to this branch. When the source carries NO self-reference
  // at all, the sentence is about somebody else in the third person — so it is not
  // a disguised self-observation, which is the only thing this anchor defends
  // against — and the content is filed with a note instead of destroyed.
  //
  // It is filed VERBATIM. Rewriting the sentence to bolt "User" onto the front is
  // exactly the class of repair that caused every bug this module guards against.
  let unanchoredNote = null;
  if (!agree.ok && agree.unanchored && factNamesSomeone(decision.fact)) {
    unanchoredNote = agree.reason;
    opsLog(`write_memory filed an unanchored user-fact rather than dropping it — ${agree.reason}. The sentence names its own subject, so it is not the pronounless case the anchor defends against. Stored verbatim: "${String(decision.fact).slice(0, 120)}"`);
    dailyLog(`Filed "${String(decision.fact).slice(0, 120)}" even though it does not name her outright — it is about someone else, and losing it would have been worse than filing it unanchored.`);
  } else if (!agree.ok) {
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
  //
  // conversational:true — the one opt-out from the self-fact notice the funnel
  // raises. This is write_memory: he is in the room, he decided to write this,
  // and the reply he is about to give already says what it replaced. A private
  // note telling him about a change he made a second ago is noise, and noise in
  // that channel is what would make a real notice skippable.
  //
  // AND IT CARRIES THE OLD FACT'S UNCONTESTED ASSERTIONS ACROSS (2026-08-24).
  // This path is where Athena lost the MettaSphere role: she re-saved Juno's
  // hardware spec, findSupersession matched it against a fuller Juno fact, the
  // judge said YES, and everything in that fuller fact which the hardware spec
  // did not so much as mention went out of the corpus with it. An explicit
  // "remember this" is an instruction to ADD something to the record, and it
  // must never be able to subtract something nobody corrected.
  let superseded = null;
  let carriedOver = null;
  if (supersession) {
    const sres = await require('./fact-merge').mergePreservingUnion(supersession.oldMemberId, res.memberId, {
      mode: 'contradiction',
      ledgerTier: 'intake',
      supersedeOpts: { conversational: true }
    });
    if (sres.deferred) {
      // The old fact keeps standing. An explicit "remember this" adds; it must
      // never subtract, and a union that could not be verified is not grounds to
      // retire assertions nobody corrected.
      dailyLog(`Wrote "${fact}" but did NOT retire "${supersession.oldContent}" — merging them would have lost something (${sres.reason}), so both are kept.`);
      opsLog(`Explicit write deferred a supersession — ${sres.reason}. Both facts remain active: "${supersession.oldContent}" and "${fact}".`);
    } else if (sres.ok) {
      superseded = supersession.oldContent;
      if (sres.union && sres.union.applied) {
        carriedOver = sres.union.to;
        dailyLog(`Superseded fact on request: "${supersession.oldContent}" → "${fact}" (asked to remember it directly), ` +
          `keeping what the old one knew that the new one did not contradict — it now reads "${carriedOver}"` +
          `${sres.union.dropped.length ? ` (dropped as contradicted: ${sres.union.dropped.join(', ')})` : ''}`);
      } else {
        dailyLog(`Superseded fact on request: "${supersession.oldContent}" → "${fact}" (asked to remember it directly)`);
        if (sres.union && sres.union.skipped) {
          opsLog(`Explicit write superseded a fact and carried nothing over — ${sres.union.skipped}. The old fact is kept as linked history: "${supersession.oldContent}"`);
        }
      }
    }
  }

  dailyLog(`Asked to remember, and did: "${fact}" (about ${subject === 'self' ? 'myself' : 'Ellie'}, salience ${salience}/10${superseded ? `, replacing "${superseded}"` : ''})`);
  log('ok', `${subject} fact${superseded ? ' (superseded 1)' : ''}, salience ${salience}, cluster "${res.clusterName}"`, res.memberId);

  console.log(`[MemoryWrite] stored ${subject} fact ${res.memberId.slice(0, 8)} salience=${salience} cluster="${res.clusterName}"${superseded ? ` superseded="${superseded.slice(0, 50)}"` : ''}`);

  return {
    ok: true,
    memberId: res.memberId,
    subject,
    // What the row NOW says, which is not what was asked for when the write
    // carried an older fact's assertions across. The chat turn reports this.
    fact: carriedOver || fact,
    requestedFact: fact,
    carriedOver,
    salience,
    cluster: res.clusterName,
    superseded,
    unanchored: unanchoredNote,
    lockedCategories: lockedCats.length ? lockedCats : null
  };
}

module.exports = {
  WRITE_SOURCE,
  write,
  checkCaps,
  capStatus,
  classifySubject,
  verifyPersonPreserved,
  verifyContentPreserved,
  sourceHasSelfReference,
  factNamesSomeone,
  shareContent,
  verifySubjectAgreement,
  findSupersession,
};
