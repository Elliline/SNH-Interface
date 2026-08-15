/**
 * Deterministic intake rules — the half of passive extraction that must NOT be a
 * language model's opinion.
 *
 * The extractor asks a model to split, route and attribute; this module is the
 * floor underneath that. Anything a prompt can be talked out of on a bad night is
 * enforced here instead, where it is a regex with a test rather than a paragraph
 * the model may skim. Three rules live here:
 *
 *   1. EVENT MARKERS — text that carries a time qualifier is an event no matter
 *      what the model called it. Strip the timestamp and nothing durable is left.
 *   2. IDENTITY ANCHOR — a fact about the user's own name, pronouns, or a core
 *      relationship needs the VERBATIM message to contain an explicit
 *      self-introduction. This is the F1 rule: "Hey, it's Mike not picking up the
 *      right words" — a mis-transcribed "mic", in a conversation about fixing the
 *      microphone — must never become "User's name is Mike".
 *   3. COMPOUND DETECTION — a fact that still joins unrelated assertions after
 *      the model has been asked for atoms gets sent back to be split.
 *
 * Everything here is pure and synchronous. No model calls, no database.
 */

// ============ 1. EVENT vs STATE ============

/**
 * Time qualifiers. Their presence in a candidate fact is decisive: the sentence
 * needed a date to be true, which is the definition of an event.
 *
 * Note "as of": the OLD extractor prompt actively instructed the model to anchor
 * time-relative statements to an absolute date ("As of July 2026, User is
 * migrating…"). That instruction manufactured the exact marker the
 * strip-the-timestamp test now treats as disqualifying, which is why it was
 * removed from the prompt in the same change that added this list.
 */
const TEMPORAL_MARKERS = [
  /\bas of\b/i,
  /\b(today|tonight|yesterday|tomorrow)\b/i,
  /\b(last|this|next)\s+(night|morning|afternoon|evening|week|weekend|month)\b/i,
  /\b(currently|right now|at the moment|these days|just now|earlier|recently|lately)\b/i,
  /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d/i,
  /\b\d{4}-\d{2}-\d{2}\b/,
  /\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i,
  /\bthe other (day|night|week)\b/i
];

/**
 * In-progress and momentary-state phrasings. Distinct from the temporal list
 * because they carry no date at all — the sentence is about a happening rather
 * than a standing truth. Deliberately narrow: "User is building SNH" is durable
 * and must not be caught, so this matches specific transient verbs rather than
 * any progressive form.
 */
const INPROGRESS_MARKERS = [
  /\b(is|are|was|were)\s+(currently\s+)?(experiencing|feeling|dealing with|going through|in the middle of|recovering from|struggling with)\b/i,
  /\b(had|has had)\s+(a|an)\s+\w+\s+(night|day|morning|week)\b/i,
  /\b(is|are)\s+(currently\s+)?(waiting|about to|on their way)\b/i,
  // Someone else temporarily present or engaged on the user's behalf — "has
  // cleaners working in the yard", "has a plumber coming". Anchored to have +
  // an object + a present participle, so "User is working at ISH" (durable, and
  // about the user themselves) cannot match. Being a candidate is not being
  // retired: the strip-the-timestamp judge still decides, and it is the half
  // that refuses.
  /\b(has|have|had)\s+(a|an|the|some|\d+)?\s*\w+(s)?\s+(working|coming|visiting|staying|scheduled|booked)\b/i
];

/**
 * Does this candidate carry a marker that FORCES the event branch?
 * @param {string} text
 * @returns {{isEvent: boolean, marker: string|null, kind: string|null}}
 */
function eventMarker(text) {
  const t = String(text || '');
  for (const re of TEMPORAL_MARKERS) {
    const m = t.match(re);
    if (m) return { isEvent: true, marker: m[0], kind: 'temporal' };
  }
  for (const re of INPROGRESS_MARKERS) {
    const m = t.match(re);
    if (m) return { isEvent: true, marker: m[0], kind: 'in-progress' };
  }
  return { isEvent: false, marker: null, kind: null };
}

// ============ 2. IDENTITY ANCHOR ============

const RELATIONSHIP_TERMS = [
  'wife', 'husband', 'spouse', 'partner', 'fiancé', 'fiancee', 'fiancée',
  'mother', 'mom', 'father', 'dad', 'son', 'daughter', 'brother', 'sister',
  'grandmother', 'grandfather', 'grandma', 'grandpa'
];

/**
 * Which identity slot, if any, does this fact assert?
 *
 * Scoped tightly to the USER'S OWN identity. "User has a dog named Casper" holds
 * the word "named" and is not an identity fact; "User's name is Mike" is. The
 * distinction is whose name is being asserted, so the patterns are anchored to
 * the sentence subject rather than searching for a keyword anywhere.
 *
 * @param {string} factText
 * @returns {{klass: 'name'|'pronouns'|'relationship', term?: string}|null}
 */
function identityClassOf(factText) {
  const t = String(factText || '').trim();

  // Name: the user's own, asserted as the sentence's main claim.
  if (/^(the\s+)?user'?s?\s+(full\s+)?name\s+(is|was)\b/i.test(t) ||
      /^(the\s+)?user\s+(is\s+named|goes\s+by|is\s+called)\b/i.test(t) ||
      /^(the\s+)?user'?s?\s+name,?\s/i.test(t)) {
    return { klass: 'name' };
  }

  // Pronouns.
  if (/^(the\s+)?user'?s?\s+pronouns\b/i.test(t) ||
      /^(the\s+)?user\s+uses\s+(he|she|they|it)\s*\//i.test(t)) {
    return { klass: 'pronouns' };
  }

  // Core relationships: "User's wife is X", "User's father is named X".
  const rel = t.match(new RegExp(`^(the\\s+)?user'?s?\\s+(${RELATIONSHIP_TERMS.join('|')})\\b`, 'i'));
  if (rel) return { klass: 'relationship', term: rel[2].toLowerCase() };

  return null;
}

/** Explicit self-introduction phrasings, per identity class. */
const NAME_INTRO = /\b(my name'?s?\s+is|my name\s+is|my names\s+is|i'?m\s+called|i\s+am\s+called|call\s+me|you\s+can\s+call\s+me|i\s+go\s+by|name'?s\s+\w+\s*,?\s*(nice|pleased)\s+to\s+meet)\b/i;
const PRONOUN_INTRO = /\b(my\s+pronouns|i\s+use\s+(he|she|they|it)\s*\/|use\s+(he|she|they)\s*\/\s*(him|her|them|it))\b/i;

/**
 * Does the verbatim message contain the explicit self-introduction this identity
 * class requires?
 *
 * For relationships the bar is different and weaker on purpose: there is no such
 * thing as "introducing" a sister the way you introduce a name, so the test is
 * that the relationship word the fact claims actually appears in what was said.
 * An incidental mishearing does not usually produce the exact kinship noun.
 *
 * @param {string} verbatim - the user's actual words, not a paraphrase
 * @param {{klass: string, term?: string}} klass
 * @returns {{ok: boolean, evidence: string|null}}
 */
function hasIdentityAnchor(verbatim, klass) {
  const v = String(verbatim || '');
  if (klass.klass === 'name') {
    const m = v.match(NAME_INTRO);
    return { ok: !!m, evidence: m ? m[0] : null };
  }
  if (klass.klass === 'pronouns') {
    const m = v.match(PRONOUN_INTRO);
    return { ok: !!m, evidence: m ? m[0] : null };
  }
  if (klass.klass === 'relationship' && klass.term) {
    const m = v.match(new RegExp(`\\b${klass.term}\\b`, 'i'));
    return { ok: !!m, evidence: m ? m[0] : null };
  }
  return { ok: true, evidence: null };
}

/**
 * The full gate. Returns null to allow the fact, or a refusal describing why.
 *
 * @param {string} factText
 * @param {string} verbatim
 * @param {string} modality - 'stt' | 'typed' | 'unknown'
 * @param {string[]} gatedModalities - from config
 */
function identityAnchorRefusal(factText, verbatim, modality, gatedModalities) {
  const klass = identityClassOf(factText);
  if (!klass) return null;
  const mod = (modality || 'unknown').toLowerCase();
  if (!gatedModalities.map(m => String(m).toLowerCase()).includes(mod)) return null;

  const anchor = hasIdentityAnchor(verbatim, klass);
  if (anchor.ok) return null;

  return {
    rule: 'identity-anchor',
    klass: klass.klass,
    modality: mod,
    detail: klass.klass === 'relationship'
      ? `a ${klass.klass} fact from a ${mod}-modality message, and the word "${klass.term}" does not appear in what was actually said`
      : `a ${klass.klass} fact from ${/^[aeiou]/i.test(mod) ? 'an' : 'a'} ${mod}-modality message with no explicit self-introduction in what was actually said`
  };
}

// ============ 2b. CAPABILITY AND DEPLOYMENT FACTS ============
//
// THE MANIFEST OWNS THIS GROUND, AND A FACT IN THE STORE CANNOT BE MADE TO
// AGREE WITH IT.
//
// db/capability-manifest.js is config-gated: every tool-bearing entry carries a
// `when` that reads the same flag which registers the tool, so what SNH claims
// it can do and what it can actually call cannot drift apart. A row in
// cluster_members has no such gate. On 2026-08-15 extraction proposed
//
//   "User's system has web search tool loaded"
//
// from a message about testing search. Turn tools.searxng.enabled off and the
// manifest entry disappears while that fact goes on asserting the opposite,
// forever, with salience and provenance behind it — and the injected memory
// block sits in the same prompt as the manifest, contradicting it. That is the
// two-flags-for-one-capability defect of 2026-07-27 coming back through intake.
//
// It also is not a fact about the user. It describes the deployment she is
// talking to, which config and the manifest already record exactly.
//
// The rule is deliberately narrow, because "tools" in the ordinary sense are
// legitimate and valuable user facts — the salience prompt itself rates
// "stable preferences/tools/hardware" at 5–7. So a refusal needs the SYSTEM to
// be what the sentence is ABOUT, not merely a word inside it: "User is the
// creator of SNH" is an identity fact and must survive, while "User's SNH
// instance runs Qwen3.8-27B" is deployment state and must not.

/**
 * SNH itself, in subject position.
 *
 * The SUBJECT is the whole test, with no accompanying list of capability nouns
 * or state verbs. A first version required both and let four facts through in a
 * single conversation — "User's SNH system allows the AI to search the web when
 * the user requests it", "...displays an alert when the user arrives at the
 * computer", "...creates a new chat for important messages", "...allows the AI
 * to initiate conversations on its own". Every one of those is a capability
 * statement; none used a verb from any plausible list. Enumerating the ways a
 * system can be described is a losing game, and each miss is a permanent row.
 *
 * So: if what the sentence is ABOUT is SNH, the manifest owns it and it does not
 * belong here, whatever the sentence goes on to say.
 *
 * The anchor is what keeps this narrow. It matches only in subject position, so
 * "User is the creator of SNH" and "User named the first instance Aurelius" are
 * facts about Ellie and survive. Deliberately absent: `server`, `box`, `pc`,
 * `laptop` — her hardware is a legitimate thing to remember about her, and the
 * salience prompt rates it 5–7.
 */
const SYSTEM_SUBJECT = new RegExp(
  '^(?:the\\s+)?(?:user\'?s?\\s+)?' +
  '(?:(?:second|first|new|local|other|primary)\\s+)?' +
  '(?:snh|squatch\\s+neuro\\s+hub|aurelius|sparky)?\\s*' +
  '(?:system|instance|setup|deployment|assistant|ai|memory\\s+system|brain|engine|model|tool\\s*chain|stack)\\b',
  'i'
);

/**
 * Refuse a fact that describes SNH's own capabilities or deployment.
 *
 * @param {string} factText
 * @returns {{rule: string, detail: string}|null} null to allow
 */
function capabilityFactRefusal(factText) {
  const t = String(factText || '').trim();
  if (!t) return null;
  if (!SYSTEM_SUBJECT.test(t)) return null;

  return {
    rule: 'capability-manifest-owns-this',
    detail: 'describes the assistant\'s own capabilities or deployment, which the ' +
      'capability manifest records from config and this store cannot be kept in step with'
  };
}

// ============ 3. COMPOUND DETECTION ============

const PREDICATE_VERBS = /\b(is|are|was|were|has|have|had|owns?|runs?|uses?|prefers?|likes?|enjoys?|works?|builds?|wants?|needs?|plans?|includes?|drives?|lives?)\b/gi;

/**
 * Does this still look like more than one assertion?
 *
 * Two shapes, both from the corpus:
 *   - a list: "enjoys computers, gaming, cars, and guns"
 *   - two predicates joined: "User's professional entities include her MSP,
 *     MettaSphere, and her AI research venture, Coastal Squatch."
 *
 * A hit does not split anything by itself — it sends the sentence back to the
 * model for one re-split call. Mechanically chopping on "and" would happily cut
 * "User has a dog named Casper and Cece" into nonsense.
 *
 * @returns {{compound: boolean, why: string|null}}
 */
function looksCompound(text) {
  const t = String(text || '').trim();
  if (t.length < 25) return { compound: false, why: null };

  // a, b, and c  — three or more list items
  if (/,[^,]{2,60},\s*(and|or)\s/i.test(t)) {
    return { compound: true, why: 'reads as a list of three or more items' };
  }

  const predicates = (t.match(PREDICATE_VERBS) || []).length;
  if (predicates >= 2 && /\b(and|but|while|whereas|as well as)\b/i.test(t)) {
    // Relative clauses ("the dog that has ...") are one assertion with two verbs;
    // require the conjunction to be joining top-level clauses, not a "who/that".
    if (!/\b(who|which|that)\b/i.test(t)) {
      return { compound: true, why: 'joins two assertions with a conjunction' };
    }
  }
  return { compound: false, why: null };
}

// ============ subject attribution sanity ============

/**
 * The subject a fact claims, read off its grammar. Same idea as
 * memory-write.verifySubjectAgreement: the passive path had no such check at all,
 * so a first-person self-observation the model happened to emit went into the
 * user's corpus wearing no pronoun.
 *
 * @returns {'user'|'self'|null} null = unanchored, which is itself a failure
 */
function grammaticalSubject(text) {
  const t = String(text || '').trim();
  if (/^(the\s+)?(user|ellie)\b/i.test(t)) return 'user';
  if (/^(i|my|i'm|i am|i've|i'd|i'll)\b/i.test(t)) return 'self';
  return null;
}

// ============ 5. SUBJECT ANNOTATION ============

/**
 * Strip the parenthetical name the archiver staples onto its subject.
 *
 * "User (Ellie) has blue eyes and her favorite color is green" — the "(Ellie)"
 * is an ANNOTATION saying who "User" refers to. It is not a claim the sentence is
 * making, and the corpus already holds her name as a locked identity fact, so
 * nothing is lost by removing it.
 *
 * It has to be removed before a compound is split, and here is the chain that
 * makes that non-optional. The splitter is told not to lose anything, so it
 * faithfully renders the parenthetical as its own atom: "User's name is Ellie."
 * The corrector then sees an atom asserting an identity slot and ABANDONS the
 * whole split — correctly, because manufacturing a name fact from a parenthetical
 * is the F1 defect, observed live twice. So the compound stays whole. And a
 * compound that is still whole when contradiction resolution runs can lose
 * ENTIRELY over a dispute about one of its clauses: this exact sentence was
 * retired for "User's favorite color is blue", taking "has blue eyes" with it.
 *
 * Deliberately narrow. Only a parenthetical immediately after the subject word,
 * only where it contains a bare name — "User (Ellie)" and "User (Ellie)'s" —
 * never a parenthetical carrying substance ("User's system (which runs 24/7)").
 *
 * @returns {{text: string, stripped: string|null}}
 */
function stripSubjectAnnotation(text) {
  const t = String(text || '');
  //                       subject      (  Name  )   optional possessive
  const re = /^(\s*(?:the\s+)?user)\s*\(\s*([A-Z][\w'-]{1,30})\s*\)(\'s)?/i;
  const m = t.match(re);
  if (!m) return { text: t, stripped: null };
  return { text: t.replace(re, `${m[1]}${m[3] || ''}`), stripped: m[2] };
}

// ============ 6. HISTORY vs CURRENT STATE ============

/**
 * Verbs and phrasings that put a sentence in the PAST — something that happened
 * or was once so, rather than something that is so now.
 *
 * Deliberately anchored to the sentence's main verb rather than searching for a
 * past-tense word anywhere, because "User has a dog that was born in 2019" is a
 * present-tense fact with a past-tense clause inside it.
 */
const PAST_VERBS = 'had|kept|traded|sold|bought|purchased|acquired|owned|used|preferred|liked|wanted|worked|lived|studied|drove|got rid of|was|were';
const PRESENT_VERBS = 'has|have|owns|drives|prefers|likes|uses|works|lives|runs|is|are|keeps|holds';

/**
 * A noun phrase between the possessive and the verb: "User's RAV4 is…",
 * "User's gaming system has…". Bounded so it cannot swallow a whole sentence and
 * match a verb three clauses away.
 *
 * The negative lookahead is what stops it eating a PRESENT-tense auxiliary on the
 * way to a past participle. Without it, "User's RAV4 has not had wax applied"
 * consumed "RAV4 has not" as the noun phrase, matched "had", and was filed as
 * history — a present-perfect statement about the car's current condition, read
 * as something that used to be true. Seen firing live in a corrector pass.
 */
const POSSESSED = "(?:(?!\\b(?:has|have|is|are|was|were|does|do|did|not)\\b)[\\w''-]+\\s+){0,3}";

const HISTORICAL_MARKERS = [
  new RegExp(`^(the\\s+)?user\\s+(${PAST_VERBS})\\b`, 'i'),
  // "User's Tundra was totalled", "User's old truck had 200k miles"
  new RegExp(`^(the\\s+)?user'?s\\s+${POSSESSED}(${PAST_VERBS})\\b`, 'i'),
  /^(the\s+)?user\s+has\s+(previously|formerly|since)\b/i,
  /\bused to\b/i,
  /\bno longer\b/i,
  /\b(before|prior to|until)\s+(trading|selling|moving|leaving|switching)\b/i
];

/**
 * Present-tense possession, state or preference — what is so NOW.
 *
 * The possessed-noun form is not optional garnish. Without it the rule missed
 * "User's Rav4 is brand new" and "User's Tundra is brand new", and on the very
 * next corrector pass "User had to get rid of the RAV and the Tacoma due to
 * painful memories associated with a death" was retired for a SECOND time — the
 * rule had exempted it against "User has a Rav4" and "User owns a Rav4 GR Sport"
 * and then let the same pairing through in a different grammatical dress.
 */
const CURRENT_STATE_MARKERS = [
  new RegExp(`^(the\\s+)?user\\s+(${PRESENT_VERBS})\\b`, 'i'),
  new RegExp(`^(the\\s+)?user'?s\\s+${POSSESSED}(${PRESENT_VERBS})\\b`, 'i'),
  /^(the\s+)?user\s+is\s+(a|an|the)?\s*\w+/i
];

/**
 * Is this sentence about what WAS, rather than what IS?
 * @returns {{historical: boolean, marker: string|null}}
 */
function isHistorical(text) {
  const t = String(text || '');
  for (const re of HISTORICAL_MARKERS) {
    const m = t.match(re);
    if (m) return { historical: true, marker: m[0].trim() };
  }
  return { historical: false, marker: null };
}

/** Is this sentence about what is so now? */
function isCurrentState(text) {
  const t = String(text || '');
  if (isHistorical(t).historical) return false;   // past wins — "used to have" is not "has"
  return CURRENT_STATE_MARKERS.some(re => re.test(t));
}

/**
 * HISTORY IS NOT A CONTRADICTION.
 *
 * A past-tense sentence and a present-tense one about the same subject matter do
 * not compete: they are two true statements about two different times, and both
 * belong in the corpus. "User had to get rid of the RAV and the Tacoma due to
 * painful memories associated with a death" and "User has a Rav4" are both true —
 * different vehicles, years apart — and one does not retire the other.
 *
 * The corrector proved the point on 2026-08-06, on the merged staging corpus.
 * Three of its five supersessions were this shape:
 *
 *   retired "User kept the Highlander Limited for six months before trading it
 *            for a 2023 RAV4 Prime."          for  "User owns a Rav4 GR Sport"
 *   retired "User had to get rid of the RAV and the Tacoma due to painful
 *            memories associated with a death."  for  "User has a Rav4"
 *   retired "User preferred AMD"               for  "User prefers their MacBook"
 *
 * Nothing malfunctioned. The contradiction judge answered YES, evidence dominance
 * preferred the newer and better-evidenced sentence, and both did exactly what
 * they are written to do — on a pair they should never have been handed. The
 * second one is the one that matters: it took the reason a car went with it, and
 * that reason was a death.
 *
 * So the pair is excluded at ENUMERATION, before any judge sees it. That is where
 * it belongs: the corrector's design is deterministic enumeration and model
 * judgement, and "these two are not candidates" is an enumeration question.
 *
 * ONE-SIDED ON PURPOSE. Two past-tense facts CAN contradict ("User owned a Tundra
 * in 2019" / "User never owned a Tundra"), and so can two present-tense ones.
 * Only the mixed pair is exempt.
 *
 * @returns {{exempt: boolean, reason: string|null}}
 */
function historyCoexists(a, b) {
  const aPast = isHistorical(a);
  const bPast = isHistorical(b);
  if (aPast.historical === bPast.historical) return { exempt: false, reason: null };

  const past = aPast.historical ? a : b;
  const present = aPast.historical ? b : a;
  if (!isCurrentState(present)) return { exempt: false, reason: null };

  return {
    exempt: true,
    reason: `one is about what was ("${(aPast.marker || bPast.marker)}") and the other about what is — history and current state coexist, so they are not a contradiction`
  };
}

module.exports = {
  eventMarker,
  identityClassOf,
  hasIdentityAnchor,
  identityAnchorRefusal,
  capabilityFactRefusal,
  looksCompound,
  grammaticalSubject,
  isHistorical,
  isCurrentState,
  historyCoexists,
  stripSubjectAnnotation,
  RELATIONSHIP_TERMS,
  TEMPORAL_MARKERS,
  INPROGRESS_MARKERS,
  HISTORICAL_MARKERS
};
