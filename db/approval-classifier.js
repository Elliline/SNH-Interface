/**
 * "Is she telling me to send this brief?" — asked of a model, not a regex.
 *
 * WHY THIS EXISTS. Dispatch ran at 2 real out of 7 claimed. Every mechanism
 * built to fix it — the tool_choice pin, the server backstop — was gated on
 * `classifyCodingGoAhead`, a list of phrases: "send it", "send away", "ship
 * it". On 2026-08-22 she wrote "Please try sending the brief again. Something
 * did not work the last time and it should be fixed." That is an approval by
 * any reading. It matched nothing, so the pin did not fire, and because both
 * consumers shared one signal the backstop did not fire either.
 *
 * The list was never going to work. It is trying to enumerate one side of an
 * open-ended conversation, every miss is silent, and the only way anyone finds
 * out is Ellie noticing a job did not run. THE REQUIREMENT IS THAT SHE CAN TALK
 * LIKE A PERSON, so no wording may gate a dispatch.
 *
 * WHAT THIS IS, PRECISELY. A separate, constrained call the SERVER makes before
 * the turn is generated — not the chat model deciding mid-answer, which is the
 * thing that has been failing. One bit out, a handful of tokens, temperature 0.
 * It only runs when there is something to approve: a brief on her screen that
 * has not been dispatched. Most turns never reach it.
 *
 * WHAT IT CANNOT DO. It cannot cause an unapproved dispatch. A YES pins
 * `tool_choice`, which makes the model CALL dispatch_coding_job; the tool still
 * runs every check it always ran, and db/brief-shown.js still refuses any brief
 * she has not been shown. The pin decides WHEN to ask, never WHAT is allowed.
 *
 * IT FAILS CLOSED. An engine error, an unparseable answer, a wedged brain — all
 * return false, meaning "no pin", meaning the turn proceeds exactly as it does
 * today. A false NO costs a round trip and is caught by the claim-keyed
 * backstop; a false YES would pin a turn she did not authorise. Those are not
 * symmetric, so the ambiguous case takes the cheaper mistake.
 */

const { getSqliteDb } = require('./database');
const briefShown = require('./brief-shown');
const { validateBrief } = require('./coding-jobs');

// Enough of the brief for the question to be answerable. The decision is about
// HER MESSAGE; the brief is context for what "it" refers to, and a 4000-char
// brief would cost more than the turn it is deciding about.
const BRIEF_EXCERPT_CHARS = 700;

/**
 * Is there a brief waiting on her answer in this conversation?
 *
 * "Pending" means: an earlier reply contains something the dispatch path would
 * accept as a brief, and no coding job has been created since it was shown.
 * Structural, not verbal — it is about the state of the conversation, not about
 * anything either of them said.
 */
function pendingBrief({ conversationId }) {
  const db = getSqliteDb();
  if (!db || !conversationId) return null;

  const replies = db.prepare(
    `SELECT content, timestamp FROM messages
     WHERE conversation_id = ? AND role = 'assistant'
     ORDER BY timestamp DESC LIMIT 6`
  ).all(conversationId);

  for (const row of replies) {
    const text = String(row.content || '').trim();
    // Too short to be a brief the tool would take.
    if (briefShown.contentWords(text).length < 40) continue;
    // A report ABOUT a dispatch is not a brief awaiting one.
    if (/\bsquatch-?code, working\b/i.test(text)) continue;
    // If the tool would refuse it for naming a directory, it is not a brief
    // that can be dispatched as-is, and pinning on it would force a refusal.
    if (!validateBrief(text).ok) continue;

    // Already sent? A coding job created after this reply means this brief has
    // had its answer, whatever was said since.
    const since = db.prepare(
      `SELECT COUNT(*) n FROM coding_jobs
       WHERE conversation_id = ? AND created_at > ?`
    ).get(conversationId, row.timestamp);
    if (since && since.n > 0) return null;

    return { text, timestamp: row.timestamp };
  }
  return null;
}

const SYSTEM = [
  'You decide ONE THING and nothing else: has the person just told the assistant to send',
  'a coding brief to the coding agent?',
  '',
  'Answer with exactly one word: YES or NO.',
  '',
  'YES when she is authorising it to go now — approving it, telling it to go ahead,',
  'asking for it to be sent or re-sent, saying a previous attempt failed and should be',
  'retried, or otherwise indicating the brief should now be handed over. Approval is',
  'about INTENT, not wording; she may be brief, casual, or misspell things, and she may',
  'approve while also saying thank you or adding a small remark.',
  '',
  'NO when she is asking to see it first, asking a question about it, requesting a',
  'change, telling it to wait or stop, or talking about something else entirely.',
  'NO if she is describing what should happen later rather than authorising it now.',
  '',
  'Judge only her message. The brief is shown so you know what "it" refers to.',
  'Answer YES or NO with no punctuation and no explanation.',
].join('\n');

function buildUserPrompt(brief, message) {
  const excerpt = String(brief || '').slice(0, BRIEF_EXCERPT_CHARS);
  return [
    'THE BRIEF ALREADY ON HER SCREEN (for context only):',
    '"""', excerpt, '"""',
    '',
    'HER MESSAGE:',
    '"""', String(message || '').trim(), '"""',
    '',
    'Is she telling the assistant to send this brief now? YES or NO.',
  ].join('\n');
}

/**
 * Read one bit out of the answer.
 *
 * Deliberately strict AND deliberately tolerant of the two things a small model
 * actually does: leading whitespace, and a trailing full stop. Anything else —
 * an explanation, a hedge, an empty string — is not a decision and fails closed.
 */
function parseVerdict(raw) {
  const t = String(raw || '').trim().toLowerCase().replace(/[.!]+$/, '');
  if (t === 'yes') return true;
  if (t === 'no') return false;
  // Some models cannot resist a preamble. Accept a leading YES/NO token only:
  // "YES - she is approving" is a decision; "I think this is probably yes" is
  // a model reasoning out loud, and reading a bit out of that is guessing.
  const m = /^(yes|no)\b/.exec(t);
  if (m) return m[1] === 'yes';
  return null;
}

/**
 * @param {object}   args
 * @param {string}   args.brief    the pending brief
 * @param {string}   args.message  her message this turn
 * @param {function} args.callLLM  injected so tests need no engine
 */
async function isApproval({ brief, message, callLLM }) {
  if (!brief || !String(message || '').trim()) return { approved: false, reason: 'nothing to decide' };
  try {
    const answer = await callLLM(SYSTEM, buildUserPrompt(brief, message), {
      maxTokens: 4,
      temperature: 0,
      thinkingTokens: 0,
    });
    // callLLM resolves to { content, reasoning, provider, truncated } — not a
    // string. Reading it as one produced "[object Object]", which parsed as
    // unparseable and failed closed: correct behaviour, wrong reason, and it
    // would have looked exactly like a model that could not answer.
    const raw = (answer && typeof answer === 'object') ? answer.content : answer;
    const verdict = parseVerdict(raw);
    if (verdict === null) {
      console.warn(`[Approval] unparseable verdict ${JSON.stringify(String(raw).slice(0, 60))} — treating as NO`);
      return { approved: false, reason: 'unparseable', raw };
    }
    return { approved: verdict, reason: verdict ? 'classifier YES' : 'classifier NO', raw };
  } catch (err) {
    // Fails closed on purpose: a wedged brain must not start dispatching.
    console.warn('[Approval] classifier call failed — treating as NO:', err.message);
    return { approved: false, reason: `classifier unavailable: ${err.message}` };
  }
}

module.exports = { pendingBrief, isApproval, parseVerdict, buildUserPrompt, SYSTEM, BRIEF_EXCERPT_CHARS };
