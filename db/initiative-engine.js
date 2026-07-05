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
function noticeFromQuestions() {
  const sql = getSqliteDb();
  if (!sql) return 0;
  const cfg = initiativeConfig();
  let added = 0;

  try {
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
      if (initiatives.addInitiative({
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
      if (initiatives.addInitiative({
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
 *  - audit errors (SNH couldn't make sense of a cluster) → alert
 *  - clusters found incoherent (drifted into different topics) → observation
 * @param {Array} auditResults - from runAuditPipeline
 * @returns {number} candidates added
 */
function noticeFromAudit(auditResults = []) {
  const cfg = initiativeConfig();
  let added = 0;
  try {
    for (const r of auditResults) {
      if (r.error) {
        if (initiatives.addInitiative({
          type: 'alert',
          content: `I had trouble making sense of your "${r.clusterName}" memory cluster while tidying up — you may want to look at it.`,
          sourceKind: 'cluster',
          sourceRef: r.clusterId,
          priority: 6
        })) added++;
      } else if (r.coherent === false && Array.isArray(r.splits) && r.splits.length > 0) {
        const into = r.splits.map(s => `"${s.newClusterName}"`).join(', ');
        if (initiatives.addInitiative({
          type: 'observation',
          content: `I noticed your "${r.clusterName}" memories had drifted into distinct topics (${into}), so I reorganized them.`,
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
function noticeReflectionInsight(text, priority = 6) {
  if (!text || !text.trim()) return null;
  return initiatives.addInitiative({
    type: 'reflection-insight',
    content: text.trim(),
    sourceKind: 'reflection',
    sourceRef: `reflection:${new Date().toISOString().slice(0, 10)}`,
    priority
  });
}

// ============ 2. Prioritizer (through the pool) ============

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
          const sys = `You are a prioritizer for an AI assistant deciding how important it is to raise something with its user, unprompted. Score 1–10:
- 9–10: time-sensitive or blocks the assistant's memory/decisions; the user would want to know now.
- 6–8: genuinely useful or clarifying; worth raising soon.
- 3–5: minor, nice-to-know.
- 1–2: trivial; probably not worth interrupting for.
Respond with ONLY the integer.`;
          const user = `Type: ${it.type}\nItem: "${it.content}"\n\nPriority (1-10)?`;
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
    const { callLLM } = require('./memory-manager');
    const appConfig = getConfig();

    // Phrase it as a brief, warm, unprompted opener — never a list dump.
    const sys = `You are SNH, reaching out to your user unprompted because something is on your mind. Write a short, warm, natural opening message (1–3 sentences) that raises this ONE thing. Do not greet with "Hi" repeatedly or over-explain; sound like yourself. Return ONLY the message text.`;
    const user = `The thing on your mind (${top.type}): "${top.content}"`;
    const { content } = await agentPool.schedule(
      () => callLLM(sys, user, { maxTokens: 200 }),
      'initiative-phrase'
    );
    const message = (content || '').trim() || top.content;

    // Create an SNH-initiated conversation and post the message.
    const model = appConfig.models?.chat?.model || 'snh';
    const title = message.slice(0, 48) + (message.length > 48 ? '…' : '');
    const convId = db.createConversation(title, model, 'snh');
    db.addMessage(convId, 'assistant', message, model);

    initiatives.markDelivered(top.id, { channel: 'unprompted', conversationId: convId });

    factExtractor.appendToDailyLog(
      `Reached out unprompted (${top.type}, priority ${top.priority}): "${message}"`,
      DAILY_DIR
    );
    console.log(`[Initiatives] Delivered unprompted initiative ${top.id} → conversation ${convId}`);

    return { delivered: true, conversationId: convId, initiativeId: top.id, message };
  } catch (err) {
    console.error('[Initiatives] deliverUnprompted error:', err.message);
    return { error: err.message };
  }
}

module.exports = {
  noticeFromQuestions,
  noticeFromAudit,
  noticeReflectionInsight,
  prioritize,
  deliverUnprompted,
  inQuietHours,
  pacificHour,
  initiativeConfig
};
