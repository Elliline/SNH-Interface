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
function actionableBrief({ conversationId }) {
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

    // Has it been sent? A coding job created after this reply is this brief's
    // answer. That USED TO END THE STORY — the function returned null and the
    // conversation had no trigger left. See the note on rerun below.
    const job = db.prepare(
      `SELECT id, project, created_at, agent_job_id FROM coding_jobs
       WHERE conversation_id = ? AND created_at > ?
       ORDER BY created_at DESC LIMIT 1`
    ).get(conversationId, row.timestamp);

    if (!job) return { text, timestamp: row.timestamp, dispatched: false, project: null, lastJob: null };

    const outcome = db.prepare(
      'SELECT status, error, duration_ms FROM agent_jobs WHERE id = ?'
    ).get(job.agent_job_id) || {};
    return {
      text,
      timestamp: row.timestamp,
      dispatched: true,
      project: job.project,
      lastJob: {
        id: job.id, agentJobId: job.agent_job_id, createdAt: job.created_at,
        status: outcome.status || null, error: outcome.error || null,
      },
    };
  }
  return null;
}

/**
 * The un-dispatched case, kept under its old name and old contract.
 *
 * The claim-keyed backstop scopes itself with this, and its skip behaviour for
 * an ALREADY-DISPATCHED brief is deliberate: a repeat claim there is a phantom
 * with nothing left to send, and loosening it would let a false claim re-run
 * real work. That guard is unchanged; the re-run case is handled BEFORE
 * generation by a classifier, not after it by a claim.
 */
function pendingBrief({ conversationId }) {
  const b = actionableBrief({ conversationId });
  return b && !b.dispatched ? { text: b.text, timestamp: b.timestamp } : null;
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
    // A DEADLINE OF ITS OWN, because this runs BEFORE the turn's own clock.
    //
    // callLLM's defaults are the BACKGROUND ones — 60s stall, 300s first token
    // — which are right for a queued job and absurd for a four-token yes/no
    // that decides whether to pin a tool. On 2026-08-22 the engine wedged and
    // her chat sat on a spinner for 500+ seconds with no error: the 120s chat
    // deadline fired correctly, but only once the turn reached the engine, and
    // everything before that had no deadline at all. Measured the next morning:
    // 7m14s between this classifier returning and the first tool round going
    // out. The turn's clock had not started yet, so nothing could time out.
    //
    // A wedged engine now costs this call 15 seconds, not five minutes, and it
    // fails closed — no pin — which is exactly what should happen when the
    // engine cannot answer anyway.
    const answer = await callLLM(SYSTEM, buildUserPrompt(brief, message), {
      maxTokens: 4,
      temperature: 0,
      thinkingTokens: 0,
      firstTokenMs: 15000,
      stallMs: 10000,
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

/**
 * "Is she asking me to RUN IT AGAIN?" — a separate and deliberately harder
 * question than "is she approving this?".
 *
 * WHY IT IS ITS OWN CLASSIFIER. Once a brief has been dispatched the same words
 * mean different things. "Go ahead" before a dispatch is an approval; after one
 * it is more likely a reply to something else entirely. And the conversation
 * after a job is FULL of talk about that job — how it went, what it produced,
 * what is wrong with it — which is exactly the material a first-dispatch
 * classifier would misread as enthusiasm for sending.
 *
 * WHY IT IS BIASED TOWARD NO. The two mistakes are not the same size. A false
 * YES on first dispatch costs a round trip: the tool refuses, or a job runs
 * that she was about to ask for anyway. A false YES here BURNS A REAL RUN —
 * squatch-code edits files, commits a restore point, and does it on top of
 * whatever the last run left. She has to notice and undo it. So anything
 * ambiguous is NO, and the prompt says so in those words.
 *
 * THE GAP THIS FILLS. Before this, a genuine re-run request hit nothing at all.
 * The brief was no longer pending, so the approval classifier never ran; the
 * claim-keyed backstop skips already-dispatched briefs on purpose. A job could
 * fail in 32ms with a spawn error, she could say "try that again", and the
 * system had no path — not a refusal, not a correction, silence. The 2026-08-22
 * resend only worked because the first dispatch had been FAKE, which left the
 * brief pending.
 */
const RERUN_SYSTEM = [
  'A coding brief has ALREADY been sent to the coding agent and a job has already run.',
  'You decide ONE THING: is the person now asking for that same brief to be RUN AGAIN?',
  '',
  'Answer with exactly one word: YES or NO.',
  '',
  'YES only when she is asking for another run — retry it, run it again, try that once',
  'more, send it again, have another go, do it over. She may be brief or casual and may',
  'misspell things; judge intent, not wording.',
  '',
  'NO for everything else, and MOST things are NO:',
  '- asking how the job went, or what it did, or whether it finished',
  '- discussing, praising or complaining about the result',
  '- reporting a bug or describing what is wrong with the output',
  '- asking for a CHANGE to the brief, or for a new brief (that is not a re-run)',
  '- talking about anything else',
  '',
  'Complaining about a result is not a request to repeat it. Describing a problem is not',
  'a request to repeat it. Only an explicit ask for another run of the SAME work is YES.',
  '',
  'When in doubt, answer NO. Running again edits real files, so a wrong YES costs her',
  'work; a wrong NO costs one sentence.',
  '',
  'Answer YES or NO with no punctuation and no explanation.',
].join('\n');

function buildRerunPrompt(brief, message, lastJob) {
  const excerpt = String(brief || '').slice(0, BRIEF_EXCERPT_CHARS);
  const outcome = lastJob && lastJob.status
    ? `The last run of it ended: ${lastJob.status}${lastJob.error ? ` (${String(lastJob.error).slice(0, 120)})` : ''}.`
    : 'The last run of it has already happened.';
  return [
    'THE BRIEF THAT WAS ALREADY SENT (for context only):',
    '"""', excerpt, '"""',
    '',
    outcome,
    '',
    'HER MESSAGE:',
    '"""', String(message || '').trim(), '"""',
    '',
    'Is she asking for that same brief to be run again? YES or NO.',
  ].join('\n');
}

async function isRerunRequest({ brief, message, lastJob = null, callLLM }) {
  if (!brief || !String(message || '').trim()) return { rerun: false, reason: 'nothing to decide' };
  try {
    const answer = await callLLM(RERUN_SYSTEM, buildRerunPrompt(brief, message, lastJob), {
      maxTokens: 4,
      temperature: 0,
      thinkingTokens: 0,
      // Same short deadline as isApproval — see the note there.
      firstTokenMs: 15000,
      stallMs: 10000,
    });
    const raw = (answer && typeof answer === 'object') ? answer.content : answer;
    const verdict = parseVerdict(raw);
    if (verdict === null) {
      console.warn(`[Rerun] unparseable verdict ${JSON.stringify(String(raw).slice(0, 60))} — treating as NO`);
      return { rerun: false, reason: 'unparseable', raw };
    }
    return { rerun: verdict, reason: verdict ? 'rerun classifier YES' : 'rerun classifier NO', raw };
  } catch (err) {
    console.warn('[Rerun] classifier call failed — treating as NO:', err.message);
    return { rerun: false, reason: `classifier unavailable: ${err.message}` };
  }
}

module.exports = {
  pendingBrief, actionableBrief,
  isApproval, isRerunRequest,
  parseVerdict, buildUserPrompt, buildRerunPrompt,
  SYSTEM, RERUN_SYSTEM, BRIEF_EXCERPT_CHARS,
};
