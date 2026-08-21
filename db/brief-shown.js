/**
 * brief-shown — the guard on dispatching coding work.
 *
 * A brief may only be sent to squatch-code if Ellie has already SEEN it.
 * That is the whole rule, and it is checked structurally rather than
 * inferred from how she phrased her go-ahead.
 *
 * WHY NOT A PHRASE CLASSIFIER. The obvious guard is to look for "send
 * it" in her message. CLAUDE.md's rule for that class of thing is that
 * triggers are built from HER messages, measured - and the corpus is why:
 * "take your time" appeared twice and both were Claude's own test
 * prompts typed that afternoon. There are ZERO measured instances of her
 * saying "send it to the coder", because the feature does not exist yet.
 * A phrase list written now would be invention, and it would fail the
 * first time she said it differently.
 *
 * WHAT IS CHECKED INSTEAD. The rule is not "she used approving words" -
 * it is "she can only approve something she was shown". So the brief is
 * compared against what is actually on her screen:
 *
 *   - every PRIOR assistant message in this conversation, and
 *   - her CURRENT message, because "send the coder this: ..." is her
 *     writing the brief herself, and refusing that would be wrong.
 *
 * The timing does the work. server.js stores her message BEFORE tools
 * run and the assistant's reply AFTER, so a brief the model is composing
 * this turn has nothing to match. The guard fires precisely on the case
 * it should - not because it recognised a phrase, but because the text
 * was never on screen.
 *
 * IT ALSO DOES THE FIDELITY JOB. The comparison that proves the brief
 * was shown also measures how closely it matches, so a paraphrase is
 * dispatched but MARKED rather than passing silently. One mechanism,
 * both jobs.
 */

const { getSqliteDb } = require('./database');

/**
 * Fraction of the brief's content words that must appear in whatever it
 * is claimed to have been shown as. Below this it is a different brief.
 * Recorded on every dispatch so this is tunable from evidence rather
 * than argued about.
 */
const MATCH_THRESHOLD = 0.8;

/**
 * A brief shorter than this cannot be matched reliably - a handful of
 * common words will hit any threshold by coincidence.
 */
const MIN_BRIEF_WORDS = 8;

/** Words too common to count as evidence that two texts are the same. */
const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'to', 'of', 'in',
  'on', 'for', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'it',
  'this', 'that', 'these', 'those', 'as', 'at', 'by', 'from', 'so',
  'we', 'i', 'you', 'he', 'she', 'they', 'its', 'not', 'do', 'does',
]);

function contentWords(text) {
  return String(text || '')
    .toLowerCase()
    // Markdown punctuation is formatting, not content: a brief that
    // gained a heading between being shown and being sent is the same
    // brief.
    .replace(/[`*_#>\[\]()~|]/g, ' ')
    .replace(/[^\p{L}\p{N}\s./-]/gu, ' ')
    .split(/\s+/)
    .filter(w => w && !STOPWORDS.has(w));
}

/**
 * How much of `brief` appears in `candidate`, 0..1.
 *
 * Deliberately asymmetric: a long message that CONTAINS the brief is a
 * match, because she was shown the brief inside a longer reply. The
 * reverse - a brief containing a short message - is not.
 */
function coverage(brief, candidate) {
  const words = contentWords(brief);
  if (!words.length) return 0;
  const have = new Set(contentWords(candidate));
  let hits = 0;
  for (const w of words) if (have.has(w)) hits++;
  return hits / words.length;
}

/**
 * Everything Ellie could have read before this tool call.
 *
 * Her current message is included and the assistant's in-flight reply is
 * not, which is what makes the guard work.
 */
function shownSoFar({ conversationId, userMessage }) {
  const candidates = [];

  if (userMessage && String(userMessage).trim()) {
    candidates.push({ source: 'her own message', text: String(userMessage) });
  }

  const db = getSqliteDb();
  if (db && conversationId) {
    const rows = db.prepare(`
      SELECT content, timestamp FROM messages
      WHERE conversation_id = ? AND role = 'assistant'
      ORDER BY timestamp DESC LIMIT 40
    `).all(conversationId);
    for (const row of rows) {
      candidates.push({ source: 'an earlier reply in this conversation',
                        text: row.content || '' });
    }
  }

  return candidates;
}

/**
 * Was this brief on her screen?
 *
 * Returns { ok, ratio, source, exact, reason }.
 */
function check(brief, { conversationId = null, userMessage = null } = {}) {
  const text = String(brief || '').trim();

  if (contentWords(text).length < MIN_BRIEF_WORDS) {
    return {
      ok: false, ratio: 0, source: null, exact: false,
      reason: `The brief is too short to dispatch (under ${MIN_BRIEF_WORDS} substantive words). Write it out properly first.`,
    };
  }

  if (!conversationId && !userMessage) {
    // Fail closed. There is no non-chat caller, and a dispatch that
    // cannot be traced to something she read is exactly what this
    // exists to stop.
    return {
      ok: false, ratio: 0, source: null, exact: false,
      reason: 'No conversation context, so there is no way to tell whether Ellie has seen this brief. Nothing was sent.',
    };
  }

  const candidates = shownSoFar({ conversationId, userMessage });
  let best = { ratio: 0, source: null, text: '' };

  for (const candidate of candidates) {
    const ratio = coverage(text, candidate.text);
    if (ratio > best.ratio) best = { ratio, source: candidate.source, text: candidate.text };
    if (ratio === 1) break;
  }

  if (best.ratio < MATCH_THRESHOLD) {
    return {
      ok: false, ratio: best.ratio, source: null, exact: false,
      reason:
        'Ellie has not seen this brief. Nothing was sent. Write the brief out ' +
        'in your reply so she can read it, and send it only once she says to. ' +
        'If she already approved one, send that text rather than a rewrite of it.',
    };
  }

  const exact = String(best.text).includes(text);
  return { ok: true, ratio: best.ratio, source: best.source, exact, reason: null };
}

module.exports = {
  MATCH_THRESHOLD,
  MIN_BRIEF_WORDS,
  check,
  coverage,
  contentWords,
  shownSoFar,
};
