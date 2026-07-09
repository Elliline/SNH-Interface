/**
 * Initiative engine — the noticing and the speaking.
 *
 * Runs inside the heartbeat cycle:
 *   1. notice*()        — turn heartbeat findings into candidate initiatives
 *   2. prioritize()     — a pooled agent re-scores, expires stale, caps the pool
 *   3. deliverUnprompted() — maybe start one SNH-initiated conversation
 *
 * The conversation-open greeting channel lives in the chat route; this module
 * owns the background half. LLM calls go through the agent pool (chat-priority
 * aware) and use the heartbeat model via memory-manager.callLLM (lazy-required
 * to avoid a load-time cycle).
 */

const { getConfig } = require('./config');
const { getSqliteDb } = require('./database');
const db = require('./database');
const agentPool = require('./agent-pool');
const initiatives = require('./initiatives');
const factExtractor = require('./fact-extractor');
const path = require('path');

const DAILY_DIR = path.join(__dirname, '../data/memory/daily');

function initiativeConfig() {
  const cfg = getConfig();
  return Object.assign({
    greetingThreshold: 7,
    followupThreshold: 5,
    unpromptedThreshold: 8,
    maxUnpromptedPerDay: 1,
    quietHours: { start: 22, end: 8 },
    questionAgeDays: 3,
    staleDays: 7,
    maxPending: 10
  }, cfg.initiative || {});
}

function daysAgoIso(days) {
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
}

// ============ 1. Noticing ============

/**
 * Turn stale/blocking questions into initiatives:
 *  - high-salience gap questions still pending after questionAgeDays
 *  - contradiction uncertainties (they block a memory decision) — no age wait
 * @returns {number} candidates added
 */
async function noticeFromQuestions() {
  const sql = getSqliteDb();
  if (!sql) return 0;
  const cfg = initiativeConfig();
  let added = 0;

  try {
    // Self-heal: retract any pending initiative whose backing question has since
    // been answered (or otherwise left 'pending'). Answer-detection retires the
    // question row, but a question/alert initiative already minted from it would
    // otherwise linger in the pending pool and get re-surfaced — this is how the
    // security-audit question resurfaced after the question-dedup fix landed.
    const orphaned = sql.prepare(`
      SELECT i.id, i.source_ref
      FROM initiatives i
      JOIN questions q ON q.id = i.source_ref
      WHERE i.status = 'pending'
        AND i.source_kind = 'question'
        AND i.type IN ('question', 'alert')
        AND q.status <> 'pending'
    `).all();
    for (const o of orphaned) {
      if (initiatives.dismiss(o.id)) {
        console.log(`[Initiatives] Dismissed pending initiative ${o.id} — backing question ${o.source_ref} is no longer pending`);
      }
    }

    // Stale gap questions, joined to their fact's salience.
    const cutoff = daysAgoIso(cfg.questionAgeDays);
    const staleGaps = sql.prepare(`
      SELECT q.id, q.question, q.member_id, q.cluster_id,
             COALESCE(cm.salience, 5) AS salience
      FROM questions q
      LEFT JOIN cluster_members cm ON cm.id = q.member_id
      WHERE q.reason = 'gap' AND q.status = 'pending' AND q.created_at < ?
    `).all(cutoff);

    for (const q of staleGaps) {
      // Only promote ones that matter (high-salience) — priority tracks salience,
      // nudged up for having gone unanswered.
      if ((q.salience ?? 5) < 6) continue;
      const priority = Math.min(10, (q.salience ?? 5) + 1);
      if (await initiatives.addInitiative({
        type: 'question',
        content: q.question,
        sourceKind: 'question',
        sourceRef: q.id,
        priority
      })) added++;
    }

    // Contradiction uncertainties block a memory decision — surface them.
    const conflicts = sql.prepare(`
      SELECT id, question FROM questions
      WHERE reason = 'contradiction-uncertainty' AND status = 'pending'
    `).all();
    for (const q of conflicts) {
      if (await initiatives.addInitiative({
        type: 'alert',
        content: q.question,
        sourceKind: 'question',
        sourceRef: q.id,
        priority: 8
      })) added++;
    }
  } catch (err) {
    console.error('[Initiatives] noticeFromQuestions error:', err.message);
  }

  if (added) console.log(`[Initiatives] noticeFromQuestions added ${added} candidate(s)`);
  return added;
}

/**
 * Turn cluster-audit findings that need the user into initiatives:
 *  - clusters found incoherent (drifted into different topics) → observation
 *
 * Audit *errors* (an unparseable LLM response or a thrown exception on one
 * cluster in one cycle) are deliberately NOT surfaced as user alerts: they are
 * transient, internal, and not actionable by the user — the next heartbeat
 * re-audits the same cluster and usually succeeds. They are already recorded in
 * the heartbeat report's Anomalies section for operators. Raising them as alerts
 * only produced noise like "I hit a snag making sense of my … memories".
 * @param {Array} auditResults - from runAuditPipeline
 * @returns {Promise<number>} candidates added
 */
async function noticeFromAudit(auditResults = []) {
  let added = 0;
  try {
    for (const r of auditResults) {
      if (r.error) {
        // Transient internal audit failure — log for operators, do not alert the user.
        console.warn(`[Initiatives] Audit error on "${r.clusterName}" (not surfaced as alert): ${r.error}`);
        continue;
      }
      if (r.coherent === false && Array.isArray(r.splits) && r.splits.length > 0) {
        const into = r.splits.map(s => `"${s.newClusterName}"`).join(', ');
        if (await initiatives.addInitiative({
          type: 'observation',
          content: `Heads up — I noticed my "${r.clusterName}" memories had drifted into a couple of different topics (${into}), so I reorganized them into separate clusters.`,
          sourceKind: 'cluster',
          sourceRef: r.clusterId,
          priority: 5
        })) added++;
      }
    }
  } catch (err) {
    console.error('[Initiatives] noticeFromAudit error:', err.message);
  }
  if (added) console.log(`[Initiatives] noticeFromAudit added ${added} candidate(s)`);
  return added;
}

/**
 * Record a reflection insight the model flagged as worth sharing.
 * Called from runReflection. type = 'reflection-insight'.
 */
async function noticeReflectionInsight(text, priority = 6) {
  if (!text || !text.trim()) return null;
  return initiatives.addInitiative({
    type: 'reflection-insight',
    content: text.trim(),
    sourceKind: 'reflection',
    sourceRef: `reflection:${new Date().toISOString().slice(0, 10)}`,
    priority
  });
}

/**
 * Parse the follow-up review response into { candidates, followup, reasoning }.
 * Accepts a JSON object (optionally fenced); degrades gracefully on any shape.
 * @param {string} raw
 */
function parseFollowupResponse(raw) {
  const out = { candidates: [], followup: null, reasoning: '' };
  try {
    const text = (raw || '').replace(/```(?:json)?\s*\n?([\s\S]*?)```/g, '$1').trim();
    const objMatch = text.match(/\{[\s\S]*\}/);
    if (!objMatch) return out;
    const parsed = JSON.parse(objMatch[0]);
    if (Array.isArray(parsed.candidates)) {
      out.candidates = parsed.candidates
        .filter(c => typeof c === 'string')
        .map(c => c.trim())
        .filter(Boolean)
        .slice(0, 5);
    }
    if (typeof parsed.reasoning === 'string') out.reasoning = parsed.reasoning.trim();
    const f = parsed.followup;
    if (typeof f === 'string') {
      const clean = f.trim();
      if (clean && !/^(none|null|n\/a)$/i.test(clean)) out.followup = clean;
    }
  } catch (err) {
    console.error('[Initiatives] parseFollowupResponse error:', err.message);
  }
  return out;
}

/**
 * Conversation-followup source — "I've been thinking about what you said".
 *
 * Runs as a pooled step inside the reflection cycle, after self-observation.
 * Reviews the conversations since the last reflection and decides whether ONE
 * thing deserves a follow-up: a thought that kept developing, an idea worth
 * returning to, or a genuine connection between something recent and something
 * older. For that last case it retrieves relevant OLDER memory clusters (by
 * embedding similarity to the recent conversation topics) and folds them into
 * the review, so a follow-up can bridge recent talk to older memory rather than
 * just echo yesterday.
 *
 * At most ONE follow-up per cycle; producing none is the common, expected case.
 * Every cycle records a structured trace (queryable) and returns it.
 *
 * @param {Object} args
 * @param {string} args.transcript - recent conversation transcript (already budgeted)
 * @param {Array}  [args.conversationsReviewed] - [{id,title,messageCount}]
 * @param {number} [args.messageCount]
 * @returns {Promise<Object>} the trace
 */
async function generateConversationFollowup({ transcript, conversationsReviewed = [], messageCount = 0 } = {}) {
  const cfg = initiativeConfig();
  const trace = {
    at: new Date().toISOString(),
    conversationsReviewed,
    messageCount,
    relatedClusters: [],
    candidates: [],
    generated: null,
    skipped: true,
    reasoning: '',
    initiativeId: null
  };

  if (!transcript || !transcript.trim()) {
    trace.reasoning = 'no recent conversation to review';
    initiatives.recordFollowupTrace(trace);
    return trace;
  }

  try {
    const { callLLM } = require('./memory-manager');
    const memoryClusters = require('./memory-clusters');

    // 1. Retrieve related OLDER memory clusters by similarity to recent topics.
    const topics = conversationsReviewed.map(c => c.title).filter(Boolean).join('; ')
      || transcript.slice(0, 600);
    try {
      const related = await memoryClusters.searchClusters(topics, 4);
      trace.relatedClusters = (related || []).map(r => ({
        name: r.cluster.name,
        members: r.members.slice(0, 4).map(m => m.content)
      }));
    } catch (searchErr) {
      console.error('[Initiatives] followup cluster retrieval skipped:', searchErr.message);
    }

    const relatedBlock = trace.relatedClusters.length
      ? trace.relatedClusters
          .map(c => `[${c.name}]\n${c.members.map(m => `- ${m}`).join('\n')}`)
          .join('\n\n')
      : '(no strongly related older memories surfaced)';

    // 2. Review + decide (pooled).
    const sys = `You are SNH, reviewing your RECENT conversations to decide whether anything deserves a follow-up with your user — the "I've been thinking about what you said" impulse.

Send a follow-up ONLY if it is genuinely one of these:
  - a thought that kept developing after the conversation ended,
  - an idea worth returning to,
  - a real connection between something the user said recently and something older in your memory (the RELATED OLDER MEMORIES below).

Quality bar — be strict. Only if it would genuinely be worth the user's attention. NEVER small talk, check-ins, pleasantries, or restating what was already said. Producing NO follow-up is common and completely fine — most cycles should produce none.

At most ONE follow-up. Write it as a short, warm, natural first-person message to the user (address them as "you", never by name). One or two sentences.

Return ONLY a JSON object, nothing else:
{
  "candidates": [up to 3 short strings naming thoughts you weighed],
  "followup": "the ONE message to send — or null if nothing clears the bar",
  "reasoning": "one sentence: why you're sending it, or why nothing cleared the bar"
}`;
    const user = `RECENT CONVERSATIONS (since your last reflection):\n${transcript}\n\nRELATED OLDER MEMORIES:\n${relatedBlock}\n\nDecide.`;

    const { content } = await agentPool.schedule(
      () => callLLM(sys, user, { maxTokens: 400 }),
      'reflection-followup'
    );

    const parsed = parseFollowupResponse(content);
    trace.candidates = parsed.candidates;
    trace.reasoning = parsed.reasoning || (parsed.followup ? 'generated a follow-up' : 'nothing cleared the bar');

    if (parsed.followup && parsed.followup.length >= 8) {
      trace.generated = parsed.followup;
      trace.skipped = false;
      // Queue above followupThreshold so it clears the lower greeting bar, but
      // below the unprompted bar unless the prioritizer later promotes it.
      const priority = Math.min(10, Math.max(cfg.followupThreshold, 5) + 1);
      const id = await initiatives.addInitiative({
        type: 'followup',
        content: parsed.followup,
        sourceKind: 'reflection',
        sourceRef: `followup:${trace.at}`,
        priority
      });
      trace.initiativeId = id;
      console.log(`[Initiatives] Follow-up generated (priority ${priority}): "${parsed.followup.slice(0, 80)}"`);
    } else {
      console.log(`[Initiatives] No follow-up this cycle — ${trace.reasoning}`);
    }
  } catch (err) {
    trace.reasoning = trace.reasoning || `error: ${err.message}`;
    console.error('[Initiatives] generateConversationFollowup error:', err.message);
  }

  initiatives.recordFollowupTrace(trace);
  return trace;
}

// ============ 2. Prioritizer (through the pool) ============

// Reflective initiatives are thoughts SNH is mulling over — never urgent by
// nature — so scoring them on urgency/actionability (the problem-shaped rubric)
// systematically buried good followups at 2/10. They get a thought-quality rubric.
const REFLECTIVE_TYPES = new Set(['followup', 'reflection-insight']);

/**
 * The prioritizer system prompt for a given initiative type. Reflective items
 * (followup, reflection-insight) are scored on thought quality; problem-shaped
 * items (question, alert, observation) keep the urgency/actionability rubric.
 * @param {string} type
 * @returns {string}
 */
function prioritizerSystemPrompt(type) {
  if (REFLECTIVE_TYPES.has(type)) {
    return `You are scoring ONE reflective thought an AI assistant is considering sharing with its user — a follow-up on a conversation or a realization about itself. This is NOT a task, alert, or question, and it is NEVER urgent by nature. DO NOT score on urgency or actionability — those do not apply here. Score purely on the QUALITY of the thought:
- Does it genuinely advance or extend an idea from a real conversation (not just restate it)?
- Does it connect ideas across domains, or link something recent to something older in memory?
- Does it bear meaningfully on the user's ongoing work, projects, or goals?

Score 1–10:
- 8–10: genuinely extends a conversation — a real insight, a non-obvious connection, or something that meaningfully bears on the user's work/goals. Clearly worth their attention.
- 5–7: a solid, relevant continuation of a real thread — worth surfacing.
- 3–4: loosely relevant but mostly restates what was already said.
- 1–2: generic, hollow, off-topic, or pure small talk.
Respond with ONLY the integer.`;
  }
  return `You are a prioritizer for an AI assistant deciding how important it is to raise something with its user, unprompted. Score 1–10:
- 9–10: time-sensitive or blocks the assistant's memory/decisions; the user would want to know now.
- 6–8: genuinely useful or clarifying; worth raising soon.
- 3–5: minor, nice-to-know.
- 1–2: trivial; probably not worth interrupting for.
Respond with ONLY the integer.`;
}

/**
 * Review pending initiatives: expire stale ones, re-score priority with a pooled
 * agent, and cap the pending pool so it never becomes a nag queue.
 * @returns {Promise<{expired:number, rescored:number, capped:number, pending:number}>}
 */
async function prioritize() {
  const sql = getSqliteDb();
  if (!sql) return { expired: 0, rescored: 0, capped: 0, pending: 0 };
  const cfg = initiativeConfig();
  const result = { expired: 0, rescored: 0, capped: 0, pending: 0 };

  try {
    // 1. Expire stale pending initiatives.
    const staleCutoff = daysAgoIso(cfg.staleDays);
    const stale = sql.prepare(
      "SELECT id FROM initiatives WHERE status = 'pending' AND created_at < ?"
    ).all(staleCutoff);
    for (const s of stale) if (initiatives.expire(s.id)) result.expired++;

    // 2. Re-score remaining pending initiatives concurrently through the pool.
    let pending = initiatives.listPending({ limit: 100 });
    if (pending.length > 0) {
      const { callLLM } = require('./memory-manager');
      const scored = await agentPool.runBatch(
        pending.map(it => async () => {
          const sys = prioritizerSystemPrompt(it.type);
          const user = `Item: "${it.content}"\n\nScore (1-10)?`;
          const { content } = await callLLM(sys, user, { maxTokens: 8 });
          const m = (content || '').match(/\d+/);
          return { id: it.id, priority: m ? parseInt(m[0], 10) : it.priority };
        }),
        'initiative-prioritize'
      );
      for (const s of scored) {
        if (s.status === 'fulfilled' && s.value) {
          if (initiatives.updatePriority(s.value.id, s.value.priority)) result.rescored++;
        }
      }
    }

    // 3. Cap the pool — keep the top maxPending by priority, expire the rest.
    pending = initiatives.listPending({ limit: 1000 });
    if (pending.length > cfg.maxPending) {
      const excess = pending.slice(cfg.maxPending); // listPending is priority DESC
      for (const it of excess) if (initiatives.expire(it.id)) result.capped++;
    }

    result.pending = initiatives.countPending();
    console.log(`[Initiatives] prioritize: expired ${result.expired} stale, re-scored ${result.rescored}, capped ${result.capped}; ${result.pending} pending`);
    return result;
  } catch (err) {
    console.error('[Initiatives] prioritize error:', err.message);
    return result;
  }
}

// ============ 3. Unprompted delivery ============

/** Current local Pacific hour (0–23). */
function pacificHour() {
  const s = new Date().toLocaleString('en-US', {
    timeZone: 'America/Los_Angeles', hour: '2-digit', hour12: false
  });
  return parseInt(s, 10) % 24;
}

/** Whether we are currently inside quiet hours. */
function inQuietHours(cfg) {
  const h = pacificHour();
  const { start, end } = cfg.quietHours || { start: 22, end: 8 };
  return start <= end ? (h >= start && h < end) : (h >= start || h < end);
}

/**
 * Maybe start ONE unprompted conversation from the top pending initiative.
 * Hard rules: not during quiet hours, respect maxUnpromptedPerDay, and only for
 * priority >= unpromptedThreshold.
 * @returns {Promise<Object>} outcome
 */
async function deliverUnprompted() {
  const cfg = initiativeConfig();

  if (inQuietHours(cfg)) {
    return { skipped: true, reason: 'quiet hours' };
  }
  const usedToday = initiatives.countUnpromptedDeliveredToday();
  if (usedToday >= cfg.maxUnpromptedPerDay) {
    return { skipped: true, reason: `daily cap reached (${usedToday}/${cfg.maxUnpromptedPerDay})` };
  }
  const top = initiatives.getTopPending(cfg.unpromptedThreshold);
  if (!top) {
    return { skipped: true, reason: `no pending initiative >= ${cfg.unpromptedThreshold}` };
  }

  try {
    const { conversationId, message } = await openInitiativeConversation(top, 'unprompted');
    factExtractor.appendToDailyLog(
      `Reached out unprompted (${top.type}, priority ${top.priority}): "${message}"`,
      DAILY_DIR
    );
    console.log(`[Initiatives] Delivered unprompted initiative ${top.id} → conversation ${conversationId}`);
    return { delivered: true, conversationId, initiativeId: top.id, message };
  } catch (err) {
    console.error('[Initiatives] deliverUnprompted error:', err.message);
    return { error: err.message };
  }
}

/**
 * Start a conversation from a specific initiative, on demand (the "Discuss"
 * action in the initiative panel). Unlike deliverUnprompted this ignores the
 * quiet-hours and daily-cap gates — the user explicitly asked for it. SNH opens
 * the conversation by raising the item naturally; the user's reply then flows
 * through the normal chat + extraction path.
 * @param {string} id - initiative id
 * @returns {Promise<Object>} { conversationId, initiativeId, message } or { error }
 */
async function startDiscussion(id) {
  const it = initiatives.get(id);
  if (!it) return { error: 'not found' };
  if (it.status !== 'pending') return { error: `not pending (${it.status})` };
  try {
    const { conversationId, message } = await openInitiativeConversation(it, 'discuss');
    factExtractor.appendToDailyLog(
      `Opened a discussion on request (${it.type}, priority ${it.priority}): "${message}"`,
      DAILY_DIR
    );
    console.log(`[Initiatives] Discuss initiative ${it.id} → conversation ${conversationId}`);
    return { conversationId, initiativeId: it.id, message };
  } catch (err) {
    console.error('[Initiatives] startDiscussion error:', err.message);
    return { error: err.message };
  }
}

/**
 * Phrase an initiative as a warm, natural opener and create the SNH-initiated
 * conversation that raises it, marking the initiative delivered.
 * @param {Object} it - the initiative row
 * @param {'unprompted'|'discuss'} channel
 * @returns {Promise<{conversationId:string, message:string}>}
 */
async function openInitiativeConversation(it, channel) {
  const { callLLM } = require('./memory-manager');
  const appConfig = getConfig();

  // Phrase it as a brief, warm opener — never a list dump.
  const sys = `You are SNH, opening a conversation with your user because something is on your mind. Write a short, warm, natural opening message (1–3 sentences) that raises this ONE thing in your own voice. Do not greet with "Hi" repeatedly or over-explain; sound like yourself. Return ONLY the message text.`;
  const user = `The thing on your mind (${it.type}): "${it.content}"`;
  const { content } = await agentPool.schedule(
    () => callLLM(sys, user, { maxTokens: 200 }),
    'initiative-phrase'
  );
  const message = (content || '').trim() || it.content;

  const model = appConfig.models?.chat?.model || 'snh';
  const title = message.slice(0, 48) + (message.length > 48 ? '…' : '');
  const conversationId = db.createConversation(title, model, 'snh');
  db.addMessage(conversationId, 'assistant', message, model);

  initiatives.markDelivered(it.id, { channel, conversationId });
  return { conversationId, message };
}

module.exports = {
  noticeFromQuestions,
  noticeFromAudit,
  noticeReflectionInsight,
  generateConversationFollowup,
  prioritize,
  prioritizerSystemPrompt,
  deliverUnprompted,
  startDiscussion,
  inQuietHours,
  pacificHour,
  initiativeConfig
};
