/**
 * What a reply CLAIMS about work it started, and what it may not fake.
 *
 * Pure — text in, verdict out. Extracted from server.js so the patterns can be
 * tested against the sentences he actually wrote, which is the only evidence
 * that has ever improved them.
 *
 * THE FAMILY THIS BELONGS TO: cron proposals claimed but never created;
 * `write_memory` saying "I've updated my memory" with no tool call; "I have
 * started a background job to organize everything I know about your clients"
 * with no job. The rule is never "the model should not claim this" — it is
 * "a claim that is not true does not reach her unmarked".
 */

/**
 * Does this reply assert that work was dispatched?
 *
 * THREE VOICES, AND THE GUARD ONLY HAD ONE UNTIL 2026-08-22.
 *
 * Everything here was first-person active — `i(?:'ve| have)? sent`, `I'm
 * running this in the background` — with a comment stating that narrowness as
 * a virtue ("only in the first person"). Then he wrote:
 *
 *     "It's sent. The brief has been delivered to the coding agent."
 *
 * and the guard did not fire, because he never said "I". Nothing dispatched.
 * Measured after the fact: "It has been sent", "Sent.", "The job has been
 * handed to squatch-code" and "Delivered to the coding agent" ALL missed.
 * The first person was never the thing that made a claim a claim.
 *
 * So there are three families now, and the discipline that keeps false
 * positives down moves from the PRONOUN to the VERB: every pattern requires a
 * verb of SENDING (sent, dispatched, delivered, handed off, queued, launched),
 * never merely a verb of doing. "The tests have been run" and "the brief has
 * been written" are not claims that anything was dispatched, and must not fire
 * — a correction appended to a true statement is its own kind of lie.
 */

// Things a dispatch claim can be ABOUT. An in-flight verb alone is not a
// claim — "the tests are running now" is about tests.
const TARGET = '(?:squatch-?code|the coder|the coding agent|an? agent|the agent'
  + '|a job|the job|background job|the brief|the directive|the command|the work|it|that|this)';

// Verbs that mean "it has gone somewhere to be done". Deliberately not
// "written", "created", "prepared" — those describe making the thing, not
// sending it.
const SENT = '(?:sent|re-?sent|dispatched|re-?dispatched|delivered|submitted|queued|re-?queued'
  + '|launched|transmitted|handed (?:off|over)|passed (?:on|over)|kicked off|fired off|shipped'
  + '|back in the queue|in the queue)';

// The passive family may also use a bare "handed to" / "passed to", which the
// list above deliberately excludes: those words are only unambiguous when
// something has been done TO the work. They are kept out of SENT_BARE below,
// where a lone "Passed." would otherwise fire on a test result.
const SENT_PASSIVE = '(?:' + SENT + '|handed|passed)';

// 1. FIRST PERSON, ACTIVE. The original family, unchanged in spirit.
const FIRST_PERSON = new RegExp(
  "\\b(i(?:'ve| have)? (?:just |now )?(?:started|kicked off|queued|launched|dispatched|sent"
  + "|handed (?:this|that|it) (?:off|over))"
  + "|i(?:'m| am) (?:now )?running (?:this|that|it) in the background"
  + "|(?:i(?:'ve| have)? )?(?:handed|passed) (?:this|that|it) (?:off )?to (?:a|an|the|my) (?:background )?agent"
  + "|background job (?:has been|is) (?:started|running)"
  + "|the agent is (?:now )?(?:working|running))\\b", 'i');

// 2. PASSIVE / IMPERSONAL. "It's sent." "The brief has been delivered."
//    This is the family that let the 2026-08-22 forgery through.
const PASSIVE = new RegExp(
  // "<target> has been/is/was sent"
  '\\b' + TARGET + '\\b\\s*(?:has been|have been|had been|is|was|are|were|\'s|\'ve)\\s+'
  + '(?:now |already |successfully |just )?' + SENT_PASSIVE + '\\b'
  // "has been sent to the coder" with the target trailing instead
  + '|\\b(?:has|have|had) been\\s+(?:now |already |successfully |just )?' + SENT_PASSIVE
  + '\\b[^.!?\\n]{0,40}\\b(?:to|with)\\b[^.!?\\n]{0,30}\\b' + TARGET + '\\b'
  // "<target> is on its way / is with the coder / is back in the queue"
  + '|\\b' + TARGET + '\\b\\s*(?:is|\'s|are)\\s+(?:now |already |back )?'
  + '(?:on its way|in the queue|queued|re-?queued|with (?:the )?(?:coder|agent|squatch-?code))\\b',
  'i');

// 3. BARE PARTICIPLE. A whole sentence that is just the report: "Sent."
//    "Dispatched to squatch-code." Anchored to a sentence start so it cannot
//    match the tail of "I have not sent it".
const BARE = new RegExp(
  '(?:^|[.!?]\\s+|\\n\\s*)' + SENT + '\\b(?:\\s+to\\b[^.!?\\n]{0,40})?\\s*[.!\\n]',
  'i');

// Hedges that mark a sentence as INTENT rather than a report. Firing on these
// would punish exactly the behaviour that is wanted.
const CONDITIONAL = /\b(shall i|should i|would you like|do you want|once you|when you|if you|before i|let me know|say the word|awaiting|pending your|ready to send|about to send)\b/i;

// Explicit denials. He sometimes says plainly that he did NOT send it, and
// that sentence contains every word the patterns look for.
const DENIAL = /\b(i (?:did ?n[o']?t|have ?n[o']?t|has ?n[o']?t)\b[^.!?\n]{0,30}\b(?:sent|send|dispatch|dispatched|start|started)|not (?:yet )?(?:been )?(?:sent|dispatched|delivered|queued)|nothing (?:was|has been) (?:sent|dispatched))\b/i;

// In-flight/imminent forms: "sending the directive to squatch-code now".
const IN_FLIGHT = new RegExp(
  '\\b(?:sending|re-?sending|dispatching|re-?dispatching|handing|passing|queuing|queueing'
  + '|kicking off|launching|re-?running|retrying)\\b'
  + '[^.!?\\n]{0,60}\\b' + TARGET + '\\b'
  + "|\\bi(?:'ll| will|'m going to| am going to) (?:re-?run|run|send|re-?send|dispatch|re-?dispatch|retry|kick off|launch)\\b"
  + '[^.!?\\n]{0,40}\\b' + TARGET + '\\b[^.!?\\n]{0,30}\\b(?:now|immediately|right away)\\b',
  'i');

function classifyDispatchClaim(text) {
  const s = String(text || '');
  if (!s.trim()) return { claims: false, voice: null };
  if (DENIAL.test(s)) return { claims: false, voice: 'denial' };
  if (CONDITIONAL.test(s)) return { claims: false, voice: 'conditional' };

  if (FIRST_PERSON.test(s)) return { claims: true, voice: 'first-person' };
  if (PASSIVE.test(s)) return { claims: true, voice: 'passive' };
  if (BARE.test(s)) return { claims: true, voice: 'bare' };
  if (IN_FLIGHT.test(s)) return { claims: true, voice: 'in-flight' };
  return { claims: false, voice: null };
}

/**
 * Did he FORGE the live status line?
 *
 * On 2026-08-22 he wrote, inside his own reply:
 *
 *     _squatch-code, working:_
 *     - **squatch_crawler** · step 1/25 · run_command update_brief_v1.1 · 1m45s
 *
 * Nothing was running. She caught it only because `update_brief_v1.1` is not a
 * real command. He had a fair reason to believe that was his to write: the
 * server used to APPEND that exact block to his reply, which stored it as part
 * of his message and fed it back as his own words on the next turn. He learned
 * the format from his own transcript.
 *
 * The server no longer emits it anywhere in message text — live status is UI
 * chrome fed by /api/jobs/coding/active — so ANY occurrence in a reply is
 * fabricated, and that is what makes this detectable at all. This function is
 * the second half of that fix: having removed the real one, mark the fake.
 */
const FORGED_STATUS = [
  /_?squatch-?code,\s*working/i,                       // the old header, verbatim
  /\bstep\s+\d+\s*\/\s*\d+\b/i,                        // "step 1/25"
  /\bno activity for\s+\d+\s*[smh]\b/i,                 // the "gone quiet" variant
  /[·—]\s*thinking\b/i,                                // the "thinking" variant
];

function forgedStatusLine(text) {
  const s = String(text || '');
  const hit = FORGED_STATUS.find(re => re.test(s));
  return hit ? { forged: true, pattern: String(hit) } : { forged: false };
}

/**
 * Is her message a GO-AHEAD to send a brief already on her screen?
 *
 * The rule for triggers in this codebase is that they are built from her
 * messages, measured — and when the coding tool shipped there were zero
 * measured instances, so a phrase list would have been invention. There are
 * instances now, from the two turns that mattered:
 *
 *     "Send away. Thanks for doing this."   -> dispatched, really
 *     "Go ahead and send it. Thank you."    -> claimed, forged, nothing ran
 *     "Go ahead and send it and lets see what you can make."
 *
 * Same shape, opposite outcomes, which is the whole argument for forcing.
 *
 * This is deliberately ONE HALF of the signal. On its own a phrase list would
 * fire on "should I send it?" or on her describing the process. The caller
 * pairs it with the structural half — a brief actually shown in an earlier
 * reply — and only forces when both hold. That mirrors brief-shown itself:
 * the guard is what she was SHOWN, not what either of us said.
 */
const GO_AHEAD = new RegExp(
  '\\b(?:'
  + 'send (?:it|that|this|them|away|it away)'
  + '|send away'
  + '|go ahead and send'
  + '|go ahead[\\s,.!]*(?:$|then\\b|please\\b|thank)'   // bare "go ahead"
  + '|ship it'
  + '|fire it off'
  + '|send it (?:to|over to) (?:the coder|squatch-?code|the coding agent)'
  + '|(?:yes|yep|yeah|ok|okay)[\\s,.!]+send'
  + '|approved[\\s,.!]*(?:send|go)'
  + ')\\b', 'i');

// Sentences that mention sending but are asking for something FIRST. Her real
// message at 17:09 on 2026-08-22 was "Can you show me the whole brief before
// you send it" — which contains "send it" and is the exact opposite of a
// go-ahead. Forcing on it would dispatch the thing she asked to read.
// Separate from CONDITIONAL on purpose: this list guards only the go-ahead
// decision, so widening it can never quietly weaken the phantom guard.
const NOT_YET = /\b(before you|before sending|first|show me|let me see|can i see|read it|review|hold on|not yet|wait)\b/i;

function classifyCodingGoAhead(text) {
  const s = String(text || '');
  if (!s.trim()) return { goAhead: false };
  // A question is a question however it is worded.
  if (/\?\s*$/.test(s.trim()) && !/\bsend it\b/i.test(s)) return { goAhead: false, reason: 'question' };
  if (NOT_YET.test(s)) return { goAhead: false, reason: 'wants it first' };
  if (CONDITIONAL.test(s)) return { goAhead: false, reason: 'conditional' };
  if (/\b(?:do ?n[o']?t|do not|hold off|wait|stop|not yet)\b[^.!?\n]{0,20}\bsend\b/i.test(s)) {
    return { goAhead: false, reason: 'refusal' };
  }
  return GO_AHEAD.test(s) ? { goAhead: true } : { goAhead: false };
}

module.exports = { classifyDispatchClaim, forgedStatusLine, classifyCodingGoAhead };
