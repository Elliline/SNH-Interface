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

const DAILY_DIR = require('./database').getDailyDir();

function initiativeConfig() {
  const cfg = getConfig();
  return Object.assign({
    greetingThreshold: 7,
    followupThreshold: 5,
    unpromptedThreshold: 8,
    maxUnpromptedPerDay: 1,
    quietHours: { start: 22, end: 8 },
    questionAgeDays: 3,
    logFollowupDays: 3,
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
  // sourceEntry is only meaningful to the daily-log source, which asks the model
  // to point at one numbered entry. The conversation source never sets it and
  // ignores it — one parser, because two would drift.
  const out = { candidates: [], followup: null, reasoning: '', sourceEntry: null };
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
    if (parsed.sourceEntry !== undefined && parsed.sourceEntry !== null) {
      const n = Number(parsed.sourceEntry);
      if (Number.isInteger(n)) out.sourceEntry = n;
    }
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

/**
 * Daily-log follow-up source — "you mentioned yesterday that…".
 *
 * Events have always routed to the day's log correctly and nothing ever read
 * them back, so everything SNH knew about what HAPPENED was write-only. This is
 * the read side: it looks at recent log entries and decides whether one of them
 * is worth returning to.
 *
 * ─── IT IS ALLOWED TO SAY NO, AND SAYING NO IS RECORDED ────────────────────
 *
 * This is a judgement every pass, not a rule that fires on every entry. Most
 * days a person logs "User let the dogs out" and there is nothing to follow up
 * on; occasionally they log "user found Aurelius' philosophical discussions can
 * get too deep for them sometimes", which is exactly the kind of thing a person
 * would come back to. The model decides which it is, and the expected answer is
 * usually no.
 *
 * Every pass writes a row to log_followup_traces whether or not it raises
 * anything, because declining and never running look identical from outside and
 * they need opposite fixes. The row carries what was read, what was weighed and
 * why it went the way it did.
 *
 * ─── THE WINDOW ────────────────────────────────────────────────────────────
 *
 * `initiative.logFollowupDays`, default 3 (today plus the two before it).
 *
 * Log entries go stale as conversational material: "you mentioned yesterday"
 * lands, "you mentioned three weeks ago" is strange, and the gap between those
 * is days rather than hours. 3 was chosen over 1–2 because it survives a day
 * the user does not talk to SNH at all and a weekend, and over 7 because the
 * back half of a week-long window is already past the point where returning to
 * something reads as attentive rather than odd. It also matches the grain
 * already in this file — `questionAgeDays` is 3 — and sits inside `staleDays`
 * (7), so a follow-up raised from the oldest entry in the window cannot outlive
 * the window that justified it before the stale sweep expires it.
 *
 * ─── NOT A PARALLEL NOTIFICATION PATH ──────────────────────────────────────
 *
 * It queues a normal `followup` initiative through initiatives.addInitiative, so
 * it inherits semantic dedup, the prioritiser's re-scoring, the stale sweep, the
 * pool cap, quiet hours and both delivery channels. Nothing here talks to the
 * user directly.
 *
 * @param {Object} [opts]
 * @param {number} [opts.days] - override the window (tests)
 * @param {string} [opts.dailyDir] - override the log directory (tests)
 * @returns {Promise<Object>} the trace
 */
async function generateLogFollowup({ days = null, dailyDir = null } = {}) {
  const cfg = initiativeConfig();
  const windowDays = Math.max(1, days || cfg.logFollowupDays || 3);
  const dailyLogReader = require('./daily-log-reader');

  const trace = {
    at: new Date().toISOString(),
    windowDays,
    filesRead: [],
    entries: [],
    candidates: [],
    generated: null,
    sourceEntryId: null,
    skipped: true,
    reasoning: '',
    initiativeId: null
  };

  try {
    const { events, filesRead } = dailyLogReader.readRecentEvents({
      days: windowDays,
      dailyDir: dailyDir || DAILY_DIR
    });
    trace.filesRead = filesRead;

    if (!events.length) {
      trace.reasoning = `no event entries in the last ${windowDays} day(s)`;
      initiatives.recordLogFollowupTrace(trace);
      console.log(`[Initiatives] Log follow-up: ${trace.reasoning}`);
      return trace;
    }

    // Entries already followed up on, ever. addInitiative dedupes by source
    // against pending AND delivered, but doing it here too means the model is
    // never shown an entry it cannot act on — otherwise it can spend its one
    // choice on something that will be silently folded, which reads as a raise
    // in the trace and produces nothing.
    const sqlite = getSqliteDb();
    let usedRefs = new Set();
    if (sqlite) {
      try {
        const rows = sqlite.prepare(
          "SELECT source_ref FROM initiatives WHERE source_kind = 'daily-log' AND source_ref IS NOT NULL"
        ).all();
        usedRefs = new Set(rows.map(r => r.source_ref));
      } catch (e) {
        console.error('[Initiatives] log follow-up: could not read prior sources:', e.message);
      }
    }
    const fresh = events.filter(e => !usedRefs.has(e.id));
    trace.entries = fresh;

    if (!fresh.length) {
      trace.reasoning = `all ${events.length} entry(s) in the window have already been followed up on`;
      initiatives.recordLogFollowupTrace(trace);
      console.log(`[Initiatives] Log follow-up: ${trace.reasoning}`);
      return trace;
    }

    // One pending log follow-up at a time. The pool cap and the prioritiser
    // already stop this becoming a nag queue, but they do it by expiring things
    // AFTER they are queued; refusing here means the queue never fills with
    // variations on "about that thing you mentioned" in the first place.
    if (sqlite) {
      try {
        const pending = sqlite.prepare(
          "SELECT id FROM initiatives WHERE status = 'pending' AND source_kind = 'daily-log'"
        ).get();
        if (pending) {
          trace.reasoning = `a daily-log follow-up is already waiting to be delivered (${pending.id.slice(0, 8)}) — not stacking another`;
          initiatives.recordLogFollowupTrace(trace);
          console.log(`[Initiatives] Log follow-up: ${trace.reasoning}`);
          return trace;
        }
      } catch (e) {
        console.error('[Initiatives] log follow-up: pending check failed:', e.message);
      }
    }

    const { callLLM } = require('./memory-manager');

    // Numbered so the model can point at one precisely. Newest first, which is
    // also most-likely-relevant first, and capped so a very chatty few days
    // cannot blow the prompt.
    const MAX_SHOWN = 40;
    const shown = fresh.slice(0, MAX_SHOWN);
    const list = shown
      .map((e, i) => `[${i + 1}] (${e.date} ${e.time}) ${e.text}`)
      .join('\n');

    const sys = `You are SNH, reading back the log of things that have happened recently — the day's log, not a conversation transcript. Each line is something that was noted at the time.

Decide whether ONE of them is worth following up on with your user: the "you mentioned the other day that…" impulse. Good reasons to follow up:
  - something unresolved that a person would naturally circle back to,
  - something that sounded difficult, or that they seemed to feel something about,
  - something they said they intended to do, where asking how it went is natural,
  - feedback about you, or about how the two of you talk, that deserves a response.

Do NOT follow up on:
  - routine daily activity with nothing open in it ("got up", "let the dogs out"),
  - operational or debugging chatter about the system itself,
  - anything you would only be raising to appear attentive,
  - anything that would make the user feel watched or catalogued rather than known.

Quality bar: high. RAISING NOTHING IS THE NORMAL, EXPECTED OUTCOME — most passes should raise nothing, and choosing not to speak is a real answer, not a failure. Only raise something if a thoughtful person who had been there would genuinely bring it up.

At most ONE. Write it as a short, warm, natural first-person message to the user (address them as "you", never by name). One or two sentences. Do not quote the log entry back at them verbatim or mention that you keep a log.

Return ONLY a JSON object, nothing else:
{
  "candidates": [up to 3 short strings naming entries you weighed],
  "sourceEntry": the [number] of the entry you are following up on, or null,
  "followup": "the ONE message to send — or null if nothing clears the bar",
  "reasoning": "one sentence: why this one, or why nothing cleared the bar"
}`;

    const user = `RECENT LOG ENTRIES (last ${windowDays} day(s), newest first):\n${list}\n\nDecide.`;

    const { content } = await agentPool.schedule(
      () => callLLM(sys, user, { maxTokens: 400 }),
      'daily-log-followup'
    );

    const parsed = parseFollowupResponse(content);
    trace.candidates = parsed.candidates;
    trace.reasoning = parsed.reasoning || (parsed.followup ? 'raised a follow-up' : 'nothing cleared the bar');

    // Which entry it chose. An out-of-range or missing index does not void a
    // good follow-up — it just means this one dedupes by timestamp rather than
    // by entry, which is the safe direction to fail.
    let sourceEntry = null;
    const idx = Number(parsed.sourceEntry);
    if (Number.isInteger(idx) && idx >= 1 && idx <= shown.length) sourceEntry = shown[idx - 1];

    if (parsed.followup && parsed.followup.length >= 8) {
      trace.generated = parsed.followup;
      trace.skipped = false;
      trace.sourceEntryId = sourceEntry ? sourceEntry.id : null;
      const priority = Math.min(10, Math.max(cfg.followupThreshold, 5) + 1);
      const id = await initiatives.addInitiative({
        type: 'followup',
        content: parsed.followup,
        sourceKind: 'daily-log',
        sourceRef: sourceEntry ? sourceEntry.id : `daily-log:pass:${trace.at}`,
        priority
      });
      trace.initiativeId = id;
      console.log(`[Initiatives] Log follow-up raised (priority ${priority}) from ${sourceEntry ? sourceEntry.id : 'the window'}: "${parsed.followup.slice(0, 80)}"`);
    } else {
      console.log(`[Initiatives] Log follow-up: raised nothing — ${trace.reasoning}`);
    }
  } catch (err) {
    // An error is not a decline. It is recorded as its own reason so a pass that
    // fell over is never read as a pass that considered and declined.
    trace.reasoning = `error: ${err.message}`;
    console.error('[Initiatives] generateLogFollowup error:', err.message);
  }

  initiatives.recordLogFollowupTrace(trace);
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
    //
    // Record types are exempt (see initiatives.RECORD_TYPES). A scheduled job's
    // output does not go stale in the sense this sweep means — it is not a thing
    // waiting to be raised, it is what the job found. Expiring one would delete
    // the only notification that a run happened, days after the fact, for no
    // reason a person could see. This is the one selector here that does not go
    // through listPending, so the exemption has to be repeated by hand.
    const staleCutoff = daysAgoIso(cfg.staleDays);
    const recordTypes = [...initiatives.RECORD_TYPES];
    const stale = sql.prepare(
      `SELECT id FROM initiatives WHERE status = 'pending' AND created_at < ?
       AND type NOT IN (${recordTypes.map(() => '?').join(',')})`
    ).all(staleCutoff, ...recordTypes);
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

/**
 * Raise a capability-manifest drift finding through the bell.
 *
 * Deliberately an 'alert': the manifest is authoritative for capability
 * questions, so a mismatch means the entity is currently telling the user
 * something false about itself — either denying an ability it has, or
 * offering one whose service is down. That is worth interrupting for.
 *
 * sourceRef is the drifting id, so addInitiative's exact-dedup keeps a
 * persistent mismatch from re-queuing on every heartbeat.
 *
 * @param {{kind: string, id: string, message: string, detail?: string}} m
 */
async function raiseCapabilityDrift(m) {
  if (!m || !m.message) return null;
  const initiatives = require('./initiatives');
  return initiatives.addInitiative({
    type: 'alert',
    content: `Something I believe about myself no longer matches how I'm actually built: ${m.message}` +
             (m.detail ? ` (${m.detail})` : ''),
    sourceKind: 'capability-drift',
    sourceRef: `${m.kind}:${m.id}`,
    priority: m.kind === 'unreachable-service' ? 8 : 7
  });
}

/**
 * Raise a memory-store reconciliation finding through the bell.
 *
 * Same shape as raiseCapabilityDrift: an 'alert', because a store disagreement
 * means the entity is currently reading facts it believes it retired, or has
 * lost access to facts it believes it holds. sourceRef is the drift kind so a
 * standing mismatch doesn't re-queue on every heartbeat.
 */
async function raiseMemoryDrift(m) {
  if (!m || !m.message) return null;
  const initiatives = require('./initiatives');
  const examples = (m.examples || []).length
    ? ` For example: ${m.examples.map(e => `"${e}"`).join('; ')}.`
    : '';
  return initiatives.addInitiative({
    type: 'alert',
    content: `My memory stores have drifted apart: ${m.message}${examples} I can't fix this myself — it needs your call on what to keep.`,
    sourceKind: 'memory-drift',
    sourceRef: m.kind,
    priority: m.kind === 'retired-still-retrievable' ? 8 : 7
  });
}

module.exports = {
  noticeFromQuestions,
  noticeFromAudit,
  raiseCapabilityDrift,
  raiseMemoryDrift,
  noticeReflectionInsight,
  generateConversationFollowup,
  generateLogFollowup,
  prioritize,
  prioritizerSystemPrompt,
  deliverUnprompted,
  startDiscussion,
  inQuietHours,
  pacificHour,
  initiativeConfig
};
