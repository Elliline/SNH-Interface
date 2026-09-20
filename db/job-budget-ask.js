/**
 * THE BUDGET ASK — the one time a background job speaks, and why it may.
 *
 * ROBOT, NOT BELL is the rule db/agent-jobs.js is bent around: a RESULT never
 * opens a conversation and never rings. Nothing here changes that. What this
 * module carries is not a result — it is a job that has stopped short of its
 * ceiling with work left and needs a DECISION from Ellie before it can move,
 * and her standing rule for decisions is the other one in CLAUDE.md: "a
 * decision she must make happens in the CONVERSATION. He writes the thing in
 * his reply where she reads it; she answers in words; the next turn acts on
 * her answer." So:
 *
 *   - THE ASK IS A MESSAGE in the conversation that dispatched the job, in the
 *     entity's own voice (the model wrote it from its own transcript). It is an
 *     ordinary assistant turn in the real transcript — the 9/1 rule, messages
 *     and chat are the same thing — so she reads it where she already reads.
 *   - THE BELL POINTS AT IT. A `proposal` (a thing waiting on her decision,
 *     which cannot be dismissed, only decided) whose whole content is "this job
 *     is waiting on you, in that conversation". It never holds the ask itself.
 *   - HER ANSWER IS HER NEXT MESSAGE THERE. A yes/no classifier — same shape as
 *     the coding-brief approval — reads it before the turn is generated, the
 *     job is resumed or told to write up, and the entity's turn is told what
 *     happened so its words match. A message that is neither leaves the job
 *     waiting; a paused job does not expire.
 *
 * This is the ONLY module on the job side that requires db/initiatives or
 * db/conversation-channel, and db/agent-jobs.js reaches it lazily, for a pause
 * and for nothing else. scripts/test-agent-jobs.js still asserts that a
 * FINISHED job leaves the initiative table empty.
 */

const { getSqliteDb } = require('./database');

function agentJobs() { return require('./agent-jobs'); }
function channel() { return require('./conversation-channel'); }
function initiatives() { return require('./initiatives'); }

const SOURCE_KIND = 'job-budget-ask';

/**
 * Put the ask in front of her.
 *
 * The conversation that dispatched the job, if it is still open to writes;
 * otherwise a new one in her list, titled so she knows what it is. Then the
 * bell item pointing there. Returns what was done so the job can record it.
 */
async function deliver(job, askText) {
  const ch = channel();
  let conv = job.conversation_id ? ch.getState(job.conversation_id) : null;
  // A hidden conversation (a verification turn, a clone artifact) is not hers
  // to read, so an ask must not land there.
  let hidden = false;
  try { hidden = !!(conv && getSqliteDb().prepare('SELECT hidden FROM conversations WHERE id = ?').get(conv.id)?.hidden); } catch { hidden = false; }
  let opened = false;
  if (conv && conv.status === 'active' && !hidden) {
    conv = ch.sendInto(conv.id, askText, { sourceKind: SOURCE_KIND, sourceRef: job.id });
  } else {
    conv = ch.openConversation({
      title: `SNH: "${job.title}" needs your answer`,
      body: askText,
      sourceKind: SOURCE_KIND, sourceRef: job.id
    });
    opened = true;
  }

  let initiativeId = null;
  try {
    initiativeId = await initiatives().addInitiative({
      type: 'proposal',
      content: `"${job.title}" is paused and waiting on your answer. It is near its budget with work left, and asked you in ` +
        `"${conv.title || 'a conversation'}" whether it may have more. A yes or no there decides it.`,
      sourceKind: SOURCE_KIND,
      sourceRef: job.id,
      priority: 7,
      dedupe: false
    });
  } catch (err) {
    console.warn(`[BudgetAsk] bell item for ${String(job.id).slice(0, 8)} not raised: ${err.message}`);
  }
  return { conversationId: conv.id, conversationTitle: conv.title, opened, initiativeId, deliveredAt: new Date().toISOString() };
}

/** The bell item is decided, not left ringing. */
function settle(job, ask, decision) {
  const d = ask && ask.delivery;
  if (!d || !d.initiativeId) return false;
  return initiatives().markDelivered(d.initiativeId, { channel: 'conversation', conversationId: d.conversationId || null });
}

/** For the bell's button: where the ask lives. */
function conversationForAsk(jobId) {
  const j = agentJobs().getJob(jobId);
  if (!j || !j.ask_json) return null;
  try { const a = JSON.parse(j.ask_json); return (a.delivery && a.delivery.conversationId) || j.conversation_id || null; }
  catch { return j.conversation_id || null; }
}

// ---------------------------------------------------------------- her answer

const SYSTEM = [
  'A background job has PAUSED near its budget and asked the person whether it may use MORE (more tool',
  'calls, more time). The ask is on their screen. You decide ONE THING about their reply: is it a YES,',
  'a NO, or NEITHER.',
  '',
  'Answer with exactly one word: YES, NO or NEITHER.',
  '',
  'YES when they are granting it — go ahead, keep going, sure, do it, take what you need, yes 20 more,',
  'finish it, carry on. They may be brief or casual and may misspell things; judge intent, not wording.',
  'NO when they are declining — no, stop there, wrap it up, that is enough, write up what you have,',
  'leave it, do not bother.',
  'NEITHER for anything else: a question about the job, talk about something unrelated, a change to',
  'what the job should do, or an answer you cannot place.',
  '',
  'When in doubt, answer NEITHER. The job simply keeps waiting; nothing is lost by waiting.',
  'Answer YES, NO or NEITHER with no punctuation and no explanation.'
].join('\n');

function buildUserPrompt(askText, message) {
  return [
    'THE ASK ALREADY ON THEIR SCREEN (for context only):',
    '"""', String(askText || '').slice(0, 900), '"""',
    '',
    'THEIR REPLY:',
    '"""', String(message || '').trim(), '"""',
    '',
    'Is this a YES, a NO, or NEITHER?'
  ].join('\n');
}

function parseVerdict(raw) {
  const t = String(raw || '').trim().toLowerCase().replace(/[.!]+$/, '');
  const m = /^(yes|no|neither)\b/.exec(t);
  return m ? m[1] : null;
}

/**
 * A number she named — "yes, 30 more", "give it another 15 calls" — so the
 * grant is hers rather than the default. Only read on a YES.
 */
function namedNumber(message) {
  const m = /(\d{1,3})\s*(?:more\s*)?(?:tool\s*)?(?:calls?|searches|lookups)?/i.exec(String(message || ''));
  if (!m) return null;
  const n = parseInt(m[1], 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * Read her message against the ask waiting on this conversation, and ACT.
 *
 * Returns null when nothing was waiting (the common case, and free — no model
 * call). Otherwise `{ job, decision, grant, reason }`, where decision is
 * 'yes' | 'no' | 'neither'. The classifier fails to NEITHER: a wedged engine
 * leaves the job waiting rather than deciding for her.
 */
async function decideFromMessage({ conversationId, message, callLLM }) {
  const waiting = agentJobs().pendingAsk(conversationId);
  if (!waiting) return null;
  const { job, ask } = waiting;
  const text = String(message || '').trim();
  if (!text) return { job, decision: 'neither', reason: 'empty message' };

  let verdict = null, raw = null, reason = null;
  try {
    const answer = await callLLM(SYSTEM, buildUserPrompt(ask.text, text), {
      maxTokens: 4, temperature: 0, thinkingTokens: 0, firstTokenMs: 26400, stallMs: 17600
    });
    raw = (answer && typeof answer === 'object') ? answer.content : answer;
    verdict = parseVerdict(raw);
    reason = verdict ? `classifier ${verdict.toUpperCase()}` : 'unparseable';
  } catch (err) {
    reason = `classifier unavailable: ${err.message}`;
  }
  if (verdict === 'yes') {
    const named = namedNumber(text);
    const r = agentJobs().grantMore(job.id, { calls: named, via: 'conversation' });
    return { job, decision: 'yes', grant: r.grant || null, reason, ok: r.ok, error: r.error || null };
  }
  if (verdict === 'no') {
    const r = agentJobs().declineMore(job.id, { via: 'conversation' });
    return { job, decision: 'no', reason, ok: r.ok, error: r.error || null };
  }
  console.log(`[BudgetAsk] ${String(job.id).slice(0, 8)} still waiting — her message was ${reason}${raw ? ` (${JSON.stringify(String(raw).slice(0, 40))})` : ''}`);
  return { job, decision: 'neither', reason };
}

/**
 * What the entity is told about it, so its words match what the system did.
 * A guidance block, injected into the same turn.
 */
function renderGuidance(outcome) {
  if (!outcome || !outcome.job) return null;
  const { sayDuration } = require('./job-failure');
  const title = outcome.job.title;
  if (outcome.decision === 'yes' && outcome.ok) {
    const g = outcome.grant || {};
    return {
      label: 'budget ask answered',
      text:
        '=== Her Answer To Your Budget Ask ===\n' +
        `You paused "${title}" to ask for more budget, and she just said YES. The system has already resumed it ` +
        `with ${g.calls || 0} more tool call(s), ${g.rounds || 0} more round(s) and ${sayDuration(g.wallMs || 0)} more on the clock. ` +
        'Acknowledge it briefly in your own words and move on — do not describe progress you cannot see; it will land in her jobs panel when it finishes.'
    };
  }
  if (outcome.decision === 'no' && outcome.ok) {
    return {
      label: 'budget ask answered',
      text:
        '=== Her Answer To Your Budget Ask ===\n' +
        `You paused "${title}" to ask for more budget, and she just said NO. The system is having it write up what it ` +
        'had, and the result will land in her jobs panel as partial. Acknowledge that briefly in your own words; do not argue for more.'
    };
  }
  if (outcome.decision === 'yes' || outcome.decision === 'no') {
    return {
      label: 'budget ask answered',
      text:
        '=== Her Answer To Your Budget Ask ===\n' +
        `She answered your budget ask for "${title}" but the system could not act on it: ${outcome.error || 'unknown'}. Say so plainly.`
    };
  }
  return {
    label: 'budget ask still open',
    text:
      '=== Your Budget Ask Is Still Waiting ===\n' +
      `"${title}" is paused, waiting for her to say yes or no to more budget in this conversation. Her last message did not ` +
      'answer it. Do not treat it as answered, and do not repeat the whole ask — if it fits, remind her in one sentence that ' +
      'a plain yes or no here decides it. If she is asking about the job itself, answer from what you can see: it is paused, not running.'
  };
}

module.exports = { SOURCE_KIND, deliver, settle, conversationForAsk, decideFromMessage, renderGuidance, parseVerdict, namedNumber, buildUserPrompt, SYSTEM };
