/**
 * THE WRITE-TIME SUBJECT CHECK — prevention at the door.
 *
 * Athena, 2026-09-01, on why detection can never be enough:
 *
 *   "Juno's case is the design lesson: the two bad facts DIDN'T CONTRADICT EACH
 *    OTHER. The store was internally consistent; the world disagreed with it.
 *    You can't search 'wrong subject' because there's no flag to grep for —
 *    which is why neither of us could catch it. Write-time subject check
 *    (prevention beats detection): the extractor, given candidate fact +
 *    subject, must be able to point at a quote attributing the claim to that
 *    entity. If not, the fact files as flagged-unverified-subject, not active."
 *
 * THE TWO FACTS THIS EXISTS TO KILL, both from Juno's store, both from the same
 * Lincoln City Animal Clinic email on 2026-08-26:
 *
 *   1. "User works at ISH (Inn At Spanish Head), a business located at 4090 NE
 *      Hwy 101, Lincoln City, OR 97367, ..."   — the clinic's street address,
 *      asserted as Inn at Spanish Head's location. ISH is not mentioned
 *      ANYWHERE in the source email; the address came off the sender's
 *      signature block.
 *   2. "User's business email is lcac2009@live.com"  — the clinic's address
 *      book entry, asserted as Ellie's. It appears in the source only inside
 *      that same signature block, attributed to nobody in the first person.
 *
 * Both die here on the same rule and for the right reason: a specific,
 * attributable claim was filed against a subject the source never connects it
 * to.
 *
 * SCOPE, deliberately narrow. The gate fires only on facts that carry
 * ATTRIBUTABLE SPECIFICS — an email, a phone number, a street address, a named
 * organisation or person. A preference or a self-observation ("User prefers
 * directness") asserts nothing a quote could attribute and is not the failure
 * class this guards; gating those would flag the corpus and teach everyone to
 * ignore the flag. Athena's own framing: "a small client registry makes this
 * cheap for business facts."
 */

const STOP = new Set(['the', 'a', 'an', 'of', 'at', 'in', 'on', 'for', 'and', 'or', 'to', 'is', 'was',
  'are', 'were', 'her', 'his', 'their', 'its', 'this', 'that', 'with', 'from', 'by', 'as', 'it',
  'user', 'users', 'i', 'my', 'me', 'we', 'our', 'you', 'your', 'she', 'he', 'they']);

// Words that mark a token sequence as a street address rather than a name.
const STREET_WORDS = /\b(hwy|highway|st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|way|ct|court|pl|place|suite|ste)\b/i;

/** First-person reference — in a message the human typed, this is the user. */
const FIRST_PERSON = /\b(i|i'm|im|i've|id|i'd|i'll|my|mine|me|we|we're|our|ours|us)\b/i;
/** Second-person reference — the assistant being addressed. */
const SECOND_PERSON = /\b(you|you're|your|yours|yourself)\b/i;

/**
 * Split a source into attribution UNITS.
 *
 * A unit is the span within which a claim and a subject may be considered
 * connected. Sentences, but also LINES — because a signature block is not
 * prose, and treating "4090 NE HWY 101 / Lincoln City, OR 97367 / email:
 * lcac2009@live.com" as one continuous sentence with whatever preceded it is
 * precisely how the clinic's address ended up attributed to Inn at Spanish Head.
 */
function units(source) {
  const text = String(source || '');
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    // A prose line splits further into sentences; a short structured line
    // (an address, a phone, an email) stays whole.
    if (trimmed.length > 80 && /[.!?]/.test(trimmed)) {
      for (const s of trimmed.split(/(?<=[.!?])\s+/)) {
        const t = s.trim(); if (t) out.push(t);
      }
    } else {
      out.push(trimmed);
    }
  }
  return out;
}

function emailsIn(t) { return String(t || '').match(/[\w.+-]+@[\w-]+\.[\w.]+/g) || []; }
function phonesIn(t) {
  return (String(t || '').match(/(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g) || [])
    .map(s => s.replace(/\D/g, '')).filter(d => d.length >= 10);
}
function addressesIn(t) {
  const out = [];
  const re = /\b(\d{2,6})\s+([A-Za-z0-9.\s]{2,40}?)(?=,|\n|$)/g;
  let m;
  while ((m = re.exec(String(t || ''))) !== null) {
    const whole = `${m[1]} ${m[2]}`.trim();
    if (STREET_WORDS.test(whole)) out.push(whole.replace(/\s+/g, ' '));
  }
  return out;
}
/** Capitalised multi-word names and all-caps acronyms. */
function properNounsIn(t) {
  const text = String(t || '');
  const out = new Set();
  for (const m of text.match(/\b[A-Z][a-zA-Z'&.-]+(?:\s+[A-Z][a-zA-Z'&.-]+)*/g) || []) {
    const clean = m.trim();
    if (clean.length < 3) continue;
    if (STOP.has(clean.toLowerCase())) continue;
    // A single capitalised word at the start of a sentence is usually grammar,
    // not a name; keep multiword names and known-acronym shapes.
    if (!/\s/.test(clean) && !/^[A-Z]{2,6}$/.test(clean)) continue;
    out.add(clean);
  }
  for (const m of text.match(/\b[A-Z]{2,6}\b/g) || []) {
    if (!STOP.has(m.toLowerCase())) out.add(m);
  }
  return [...out];
}

/**
 * The attributable specifics a fact asserts. These are what a quote has to
 * connect to the subject.
 *
 * TWO KINDS, AND THE SECOND IS DELIBERATELY NARROW.
 *
 * Contact specifics — an email, a phone number, a street address — are always
 * in scope. They belong to somebody, they are copied verbatim out of sources,
 * and misfiling one is precisely Juno's case 2.
 *
 * NAMES ARE ONLY IN SCOPE WHEN THEY NAME A REGISTERED ENTITY. The first cut
 * treated every capitalised token as a name and flagged 37 of 42 specific-
 * carrying facts on Athena's live corpus — "SNH", "AI", "MSP", "CPU", "HTTP".
 * A gate that fires on 88% of writes is a gate everyone learns to scroll past,
 * which is the failure Athena named when she asked for this ("otherwise the
 * store has quiet holes, and 'nothing lost' becomes 'nothing noticed'" cuts
 * both ways — an alarm nobody reads is the same hole). Keying names off the
 * REGISTRY is also what she actually asked for: "a small client registry
 * makes this cheap for business facts."
 *
 * The consequence, stated plainly: a claim about an org nobody has registered
 * yet is not gated. That is the resolution path's job — an unknown mention is
 * tier 'new', created and mentioned in passing — not this one's.
 */
function salientClaims(factText, knownEntities = []) {
  const t = String(factText || '');
  const claims = [];
  for (const e of emailsIn(t)) claims.push({ kind: 'email', value: e, needle: e.toLowerCase() });
  for (const p of phonesIn(t)) claims.push({ kind: 'phone', value: p, needle: p });
  for (const a of addressesIn(t)) claims.push({ kind: 'address', value: a, needle: a.toLowerCase() });

  const lower = t.toLowerCase();
  const seen = new Set();
  for (const e of knownEntities || []) {
    if (!e) continue;
    for (const n of [e.name, ...(e.aliasList || [])].filter(Boolean)) {
      const nn = String(n).toLowerCase();
      // Two characters is not a name; it is a coincidence waiting to happen.
      if (nn.length < 3 || seen.has(nn)) continue;
      if (lower.includes(nn)) { claims.push({ kind: 'name', value: n, needle: nn, entityId: e.id, entityType: e.type }); seen.add(nn); }
    }
  }
  return claims;
}

function normalise(s) { return String(s || '').toLowerCase().replace(/\s+/g, ' '); }
function digitsOf(s) { return String(s || '').replace(/\D/g, ''); }

function unitContains(unit, claim, { sourceIsUserMessage = true } = {}) {
  const u = normalise(unit);
  if (claim.kind === 'phone') return digitsOf(unit).includes(claim.needle);
  if (claim.kind === 'address') {
    // Compare loosely: "4090 NE Hwy 101" against "4090 NE HWY 101".
    const parts = claim.needle.split(' ').filter(Boolean);
    return parts.every(p => u.includes(p));
  }
  if (u.includes(claim.needle)) return true;

  // DEIXIS. The user and the assistant are almost never NAMED in a source —
  // they are pronouns. In a message she typed, "I" IS Ellie and "you" IS the
  // assistant, so a source that says "you got that name from me" does mention
  // both of them. Matching only literal strings read 22 ordinary facts on
  // Athena's live corpus as unattributed, including "I am named Athena, a name
  // Ellie gave me" — where the source said "I", meaning Ellie.
  if (claim.kind === 'name') {
    if (claim.entityType === 'user') {
      return sourceIsUserMessage ? FIRST_PERSON.test(unit) : SECOND_PERSON.test(unit);
    }
    if (claim.entityType === 'self') {
      return sourceIsUserMessage ? SECOND_PERSON.test(unit) : FIRST_PERSON.test(unit);
    }
  }
  return false;
}

/**
 * Does this unit refer to the subject entity?
 *
 * By name or alias always. By pronoun only where the pronoun genuinely denotes
 * that entity in a message the human typed: first person is the user, second
 * person is the assistant.
 */
function unitRefersToSubject(unit, entity, { sourceIsUserMessage = true } = {}) {
  const u = normalise(unit);
  const names = [entity.name, ...(entity.aliasList || [])].filter(Boolean);
  for (const n of names) {
    const nn = normalise(n);
    if (nn && nn.length > 2 && u.includes(nn)) return { yes: true, how: `names "${n}"` };
  }
  if (entity.type === 'user') {
    if (/\buser\b/.test(u)) return { yes: true, how: 'names the user' };
    if (sourceIsUserMessage && FIRST_PERSON.test(unit)) return { yes: true, how: 'first person, in a message she typed' };
  }
  if (entity.type === 'self') {
    if (sourceIsUserMessage && SECOND_PERSON.test(unit)) return { yes: true, how: 'second person, addressing the assistant' };
    if (!sourceIsUserMessage && FIRST_PERSON.test(unit)) return { yes: true, how: 'first person, in its own output' };
  }
  return { yes: false };
}

/** Any OTHER known entity named in this unit — the mis-attach signature. */
function competingEntityIn(unit, subjectEntity, knownEntities) {
  const u = normalise(unit);
  for (const e of knownEntities || []) {
    if (!e || e.id === subjectEntity.id) continue;
    for (const n of [e.name, ...(e.aliasList || [])].filter(Boolean)) {
      const nn = normalise(n);
      if (nn && nn.length > 2 && u.includes(nn)) return { entity: e, matched: n };
    }
  }
  return null;
}

/**
 * THE CHECK.
 *
 * @param factText   the candidate fact
 * @param entity     { id, name, type, aliasList[] } — the proposed subject
 * @param sourceText what was actually said, unparaphrased
 * @param opts.knownEntities  other registered entities, for competing-attribution
 * @param opts.sourceIsUserMessage  true when sourceText is what the human typed
 *
 * @returns {verified, quote, reason, detail, claims}
 *   verified true  → a quote attributes every salient claim to this subject
 *   verified false → the fact must file as flagged-unverified-subject
 */
function check(factText, entity, sourceText, opts = {}) {
  const { knownEntities = [], sourceIsUserMessage = true } = opts;
  const claims = salientClaims(factText, knownEntities);

  // No attributable specifics: not the failure class this gate is for.
  if (!claims.length) {
    return { verified: true, quote: null, reason: 'no-specifics', claims,
             detail: 'asserts nothing a quote could attribute — outside this gate' };
  }
  const src = String(sourceText || '').trim();
  if (!src) {
    return { verified: false, quote: null, reason: 'no-source', claims,
             detail: 'the fact asserts specifics but carries no source text to attribute them to' };
  }

  const us = units(src);
  const subject = { ...entity, aliasList: entity.aliasList || [] };
  const unattributed = [];
  let firstQuote = null;

  for (const claim of claims) {
    // A claim that merely repeats the subject's own name is not a claim ABOUT
    // something else; it needs no separate attribution.
    const selfNaming = [subject.name, ...subject.aliasList].filter(Boolean)
      .some(n => normalise(n).includes(claim.needle) || claim.needle.includes(normalise(n)));
    if (selfNaming) continue;

    const carrying = us.filter(u => unitContains(u, claim, { sourceIsUserMessage }));
    if (!carrying.length) {
      unattributed.push({ claim, why: 'the source never says it at all' });
      continue;
    }
    let attributed = null, competing = null;
    for (const u of carrying) {
      const ref = unitRefersToSubject(u, subject, { sourceIsUserMessage });
      if (ref.yes) { attributed = { unit: u, how: ref.how }; break; }
      const c = competingEntityIn(u, subject, knownEntities);
      if (c && !competing) competing = { unit: u, ...c };
    }
    if (attributed) { firstQuote = firstQuote || attributed; continue; }
    unattributed.push({
      claim,
      why: competing
        ? `the source attributes it to ${competing.entity.name} ("${competing.matched}"), not to ${subject.name}`
        : `the source says it, but never connects it to ${subject.name}`,
      quote: carrying[0],
      competing: competing ? competing.entity.name : null
    });
  }

  if (!unattributed.length) {
    return {
      verified: true, claims,
      quote: firstQuote ? firstQuote.unit : null,
      reason: 'attributed',
      detail: firstQuote ? `attributed by ${firstQuote.how}: "${firstQuote.unit}"` : 'every specific is the subject\'s own name'
    };
  }
  const worst = unattributed[0];
  return {
    verified: false, claims, unattributed,
    quote: worst.quote || null,
    reason: worst.competing ? 'competing-attribution' : 'no-attribution',
    detail: `cannot attribute ${worst.claim.kind} "${worst.claim.value}" to ${subject.name}: ${worst.why}`
  };
}

/** Convenience: hydrate an entity row into the shape check() wants. */
function forEntity(row) {
  const entities = require('./entities');
  return { id: row.id, name: row.name, type: row.type, aliasList: entities.aliasesOf(row) };
}

module.exports = {
  check, forEntity, salientClaims, units,
  emailsIn, phonesIn, addressesIn, properNounsIn,
  FLAGGED_STATUS: 'flagged-unverified-subject'
};
