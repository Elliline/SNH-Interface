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

module.exports = {
  eventMarker,
  identityClassOf,
  hasIdentityAnchor,
  identityAnchorRefusal,
  looksCompound,
  grammaticalSubject,
  RELATIONSHIP_TERMS,
  TEMPORAL_MARKERS,
  INPROGRESS_MARKERS
};
