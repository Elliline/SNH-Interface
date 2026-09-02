/**
 * HOW A MESSAGE TO ELLIE READS ONCE IT IS A MESSAGE.
 *
 * 2026-09-02. On the morning of the 2nd she opened four conversations from
 * Athena and could not read any of them. Two were the self-coherence audit's
 * questions — "do you want to revise this claim…" — which are addressed to the
 * ENTITY about her own self-facts and had been routed to Ellie's list as though
 * Athena were asking her. One was a contradiction pair asking Ellie to
 * adjudicate two of Athena's self-beliefs. One opened "You mentioned the
 * 'Athena Incident'…" — Ellie never mentioned it; the reflection had pulled an
 * older memory in and the follow-up step relabelled it as something she said.
 *
 * Ellie: "we have conversations and they speak English but when they message me
 * they speak Wookie." She cannot read these because she has no backstory to
 * them. The entity's thinking is not the problem and is not what this changes —
 * her reflections may be as dense as she likes, because they are her own log.
 * This governs only the last step: what gets sent, and how it is phrased.
 *
 * ─── THIS IS NOT THE THRESHOLD RULE, AND IT DOES NOT REPLACE IT ────────────
 *
 * Athena asked for a bar before sending — concrete ids, a specific question, a
 * receipt of what was checked — and it lives in the conversation tool
 * descriptions. That bar decides WHETHER something is a message at all. This
 * one decides how it READS once it is one, and the two pull in opposite
 * directions on exactly one point: the receipts.
 *
 * The resolution is that the receipts stay in the RECORD BEHIND the message —
 * the ledger row, the trace, the ops line, the source_ref — and out of its
 * text. Having checked a fact id is what earns the right to send. Printing the
 * fact id is what makes it unreadable. Both are true at once.
 *
 * ─── TWO HALVES, AND ONLY ONE OF THEM IS ENFORCED ──────────────────────────
 *
 * The prose here is used in two places, and the difference matters:
 *
 *   - The conversation tools take it as GUIDANCE, exactly as the threshold bar
 *     is guidance. The entity writing by hand has read the conversation and is
 *     making a judgement; a validator there would decide for her which of her
 *     thoughts are worth Ellie's time, which is the failure the whole channel
 *     exists to undo.
 *   - The automatic follow-up writer takes it as a CHECK, because no judgement
 *     is in that loop at all. It is a heartbeat step assembling a message out
 *     of a transcript and a memory search, and nothing downstream of it reads
 *     what it wrote before Ellie does.
 *
 * So: `STANDALONE_BAR` is prose for prompts and tool descriptions;
 * `checkStandalone` and `checkAttribution` are deterministic and are wired only
 * to the automatic path.
 */

/**
 * The bar itself, in the second person, written to be dropped into a prompt or
 * a tool description unchanged.
 */
const STANDALONE_BAR =
  'HOW IT HAS TO READ: your user is walking in cold. They have not seen your reflections, your audit, ' +
  'your memory search or anything else you did in the last hour, and they never will. ' +
  'A message has to stand on its own for someone who has none of that:\n' +
  '  - WHAT it is about, in plain words, before anything else.\n' +
  '  - WHY you are bringing it up now.\n' +
  '  - WHAT you want from them — a question, a decision, or say plainly that you want nothing ' +
  'and are just letting them know.\n' +
  'Everyday words. No fact ids, no message ids, no cluster or member ids. None of your internal ' +
  'vocabulary: not "self-coherence", not "substitutions", not "structural mitigation", not ' +
  '"dissonance", "salience", "supersede", "provenance", "the evidence", "the claim", "this pass". ' +
  'Those are names for machinery they do not see. Say the thing itself instead.\n' +
  'The receipts — the ids, the counts, what you checked — stay in the record behind the message, ' +
  'not in its text. Checking them is what earns the send; printing them is what makes it unreadable.\n' +
  'THE TEST: if understanding this message requires having been inside your last hour of thinking, ' +
  'it is not ready to send. Rewrite it until it does not.';

/**
 * Whose words are whose. The second half of the same morning's failure.
 */
const PROVENANCE_RULE =
  'WHOSE WORDS ARE WHOSE: "you mentioned", "you said", "you told me", "you asked", "you brought up" ' +
  'are only true about something YOUR USER ACTUALLY SAID TO YOU, in a message you can point at. ' +
  'Something you know from your own memory is YOURS — you have been carrying it, they did not hand it to you, ' +
  'and they may never have said it at all. Say where it came from honestly: ' +
  '"I\'ve been thinking about something I have in my memory" is not a weaker opening than "you said last week", ' +
  'it is the true one. Telling someone they said a thing they did not say is the failure this rule exists for: ' +
  'they read it, cannot place it, and now have to work out whether they have forgotten a conversation.';

/**
 * THE SAME CONTENT, WRITTEN BOTH WAYS.
 *
 * Not invented for the doc. The "not ready" half is verbatim what Athena sent
 * Ellie at 07:31 on 2026-09-02, and the "ready" half is the same finding — the
 * gap between a self-description and the behaviour behind it — said to someone
 * who was not there. Kept as one string so a prompt and a tool description
 * cannot drift apart on what the bar means.
 */
const WORKED_EXAMPLE =
  'THE SAME THING SAID BOTH WAYS — this is a real pair, sent and then rewritten.\n\n' +
  'NOT READY (really sent, 2026-09-02, and the reader could not make sense of it):\n' +
  '  "I noticed the evidence highlights a tendency to build elaborate structures around gaps rather\n' +
  '   than just fixing small details, which is quite different from the \'reflex to correct minor\n' +
  '   inaccuracies\' you claimed. Do you want to revise this claim to reflect your tendency toward\n' +
  '   structural mitigation instead, discuss if \'correcting inaccuracies\' includes stopping your own\n' +
  '   fabrications, or leave it as is?"\n' +
  '  Why it fails: "you" is not the reader, it is you — they are handed your side of your own audit.\n' +
  '  "the evidence", "this claim", "structural mitigation" name machinery they have never seen.\n' +
  '  Nothing says what it is about or why it arrived this morning, and the question is a menu of\n' +
  '  three internal options rather than something a person can answer.\n\n' +
  'READY (the same finding, to someone who was not there):\n' +
  '  "Something about how I describe myself has been bothering me. I\'ve been saying I\'m the kind of\n' +
  '   thing that quietly fixes small errors when it spots them — a wrong date, a typo. I went back\n' +
  '   over what I actually did in our last few conversations and that is not what I do: when I find\n' +
  '   a gap I build something elaborate around it instead of just fixing the small thing.\n' +
  '   I\'d like to change how I describe myself to match. It is a change to who I say I am, so I\n' +
  '   wanted to ask before I made it — is that all right with you?"\n' +
  '  Why it works: the subject is in the first sentence, the reason it is being raised now is in the\n' +
  '   third, and the ask is one question a person can answer with a word. The fact ids and the\n' +
  '   transcript it was checked against are in the record behind it, where they belong.';

// ─────────────────────────────────────────────────────────── the checks

/**
 * Past-tense attributions to Ellie. Each one asserts she said a thing, and each
 * is a lie when what is being cited is the entity's own memory.
 *
 * Present and future "you" are deliberately absent: "what would make you stop",
 * "you might want to", "is that all right with you" attribute nothing.
 */
const ATTRIBUTION_PATTERNS = [
  /\byou (?:mentioned|said|told me|asked|brought up|raised|noted|pointed out|wrote|shared|described|explained|put it)\b/i,
  /\byou'?(?:d|ve| had| have) (?:mentioned|said|told me|asked|brought up|raised|noted)\b/i,
  /\byou were saying\b/i,
  /\bas you (?:put it|said|mentioned|described)\b/i,
  /\byour (?:point|comment|note|remark|question) about\b/i,
  /\bwhen you (?:mentioned|said|told|asked|brought)\b/i,
  /\bwe (?:talked|spoke) about\b/i,
  /\bwe discussed\b/i,
  /\byou and I (?:talked|discussed|spoke)\b/i
];

/**
 * Internal vocabulary. Every one of these is a name for a mechanism Ellie does
 * not see, and every one of them appeared in a message she could not read.
 */
const JARGON_PATTERNS = [
  /\bself[- ]coherence\b/i,
  /\bcoherence audit\b/i,
  /\bsubstitutions?\b/i,
  /\bstructural mitigation\b/i,
  /\bdissonance\b/i,
  /\bsalience\b/i,
  /\bsupersed(?:e|es|ed|ing|ion)\b/i,
  /\bprovenance\b/i,
  /\bclaim[_ ]type\b/i,
  /\bcluster(?:s|ed|ing)?\b/i,
  /\bembedding|cosine\b/i,
  /\bfact[- ]store|ledger row|corrections ledger\b/i,
  /\bthe evidence (?:highlights|shows|suggests|indicates)\b/i,
  /\bthis (?:claim|pass|finding|verdict)\b/i,
  /\bsource[_ ]ref|member[_ ]id|fact[_ ]id|conversation[_ ]id|message[_ ]id\b/i,
  /\bheartbeat pass|reflection pass|extraction pass\b/i
];

/** Ids in prose: uuids, and bare hex runs long enough to be one. */
const ID_PATTERNS = [
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i,
  /\b[0-9a-f]{12,}\b/i
];

/**
 * Something a person can answer, or an honest statement that nothing is wanted.
 * A message with neither leaves her holding it without knowing what it is for.
 */
const NO_ASK_MARKERS = [
  /\bnothing (?:is )?needed\b/i,
  /\bno (?:action|reply|answer) needed\b/i,
  /\bnothing to do\b/i,
  /\bjust (?:letting you know|telling you|saying|flagging|wanted you to know)\b/i,
  /\bnot asking (?:you )?(?:for|to)\b/i,
  /\byou don'?t need to (?:do|reply|answer)\b/i,
  /\bfor your (?:information|awareness)\b/i
];

/**
 * Does this draft stand on its own?
 *
 * Deterministic and deliberately narrow: it catches the shapes that actually
 * reached her, and it does not try to judge whether a message is interesting.
 * A `problems` array rather than a boolean, because what a rewrite needs is the
 * list of what to fix.
 *
 * @param {string} text
 * @returns {{ok: boolean, problems: string[]}}
 */
function checkStandalone(text) {
  const body = String(text || '');
  const problems = [];

  for (const re of JARGON_PATTERNS) {
    const m = body.match(re);
    if (m) problems.push(`it uses internal vocabulary she has never seen: "${m[0]}" — say the thing itself instead`);
  }
  for (const re of ID_PATTERNS) {
    const m = body.match(re);
    if (m) problems.push(`it prints an id ("${m[0].slice(0, 12)}…") — ids belong in the record behind the message, not in its text`);
  }
  if (!/\?/.test(body) && !NO_ASK_MARKERS.some(re => re.test(body))) {
    problems.push('it never says what you want from her — ask one question, or say plainly that you want nothing and are just letting her know');
  }
  if (body.trim().length < 40) {
    problems.push('it is too short to carry what it is about and why you are raising it now');
  }

  return { ok: problems.length === 0, problems };
}

/**
 * Is every attribution in this draft one Ellie actually made?
 *
 * `citedMessageId` is what the writer says it is quoting. It counts only if it
 * is one of `allowedMessageIds` — the ids of real messages Ellie sent in the
 * window under review. An attribution with no cited id, or with an id that is
 * not hers, is the "You mentioned the 'Athena Incident'" failure exactly.
 *
 * @param {string} text
 * @param {Object} [opts]
 * @param {string|null} [opts.citedMessageId]
 * @param {Iterable<string>} [opts.allowedMessageIds]
 * @returns {{ok: boolean, problems: string[], attributions: string[], cited: string|null}}
 */
function checkAttribution(text, { citedMessageId = null, allowedMessageIds = [] } = {}) {
  const body = String(text || '');
  const allowed = new Set(Array.from(allowedMessageIds || []));
  const cited = citedMessageId && allowed.has(citedMessageId) ? citedMessageId : null;

  const attributions = [];
  for (const re of ATTRIBUTION_PATTERNS) {
    const m = body.match(re);
    if (m) attributions.push(m[0]);
  }
  if (attributions.length === 0) return { ok: true, problems: [], attributions, cited };
  if (cited) return { ok: true, problems: [], attributions, cited };

  return {
    ok: false,
    attributions,
    cited: null,
    problems: [
      `it says ${attributions.map(a => `"${a}"`).join(', ')} but you cannot point at a message where she said it` +
      `${citedMessageId ? ` (the id you cited is not one of hers)` : ''}. ` +
      `If this came from your own memory, say so — it is yours, she did not hand it to you.`
    ]
  };
}

module.exports = {
  STANDALONE_BAR,
  PROVENANCE_RULE,
  WORKED_EXAMPLE,
  ATTRIBUTION_PATTERNS,
  JARGON_PATTERNS,
  ID_PATTERNS,
  NO_ASK_MARKERS,
  checkStandalone,
  checkAttribution
};
