/**
 * WHAT THE ENTITY CONCLUDES ABOUT ITSELF IN A DAY, CHOSEN AT THE END OF IT.
 *
 * The cap was five a day and it was FIRST COME FIRST SERVED, which made it a
 * race rather than a budget. On 2026-09-01 the lunchtime reflection spent all
 * five; the two evening reflections each noticed five more things and stored
 * none of them. Nothing was wrong with the cap — the number is a real limit on
 * what the entity may conclude about itself in a day, set by Ellie on
 * 2026-08-02 after an unbudgeted path wrote 36 self-facts in one day. What was
 * wrong is that the earliest observations won on time of arrival.
 *
 * So observations now COLLECT as candidates, and at the end of the local day the
 * entity takes one turn and picks 0–5 of them.
 *
 * ─── THE THREE WAYS ATHENA SAID SHE WOULD PICK BADLY ───────────────────────
 *
 * Each is answered by something the turn is given, not by an instruction to try
 * harder:
 *
 *   - "INTENSITY WINS OVER TRUTH. What I felt most, I remember most. […] The
 *     selector and the subject are the same entity, with no adversary between
 *     them, so felt magnitude is the default ranking." — answered by showing
 *     the whole day at once, where a strong feeling stated twice is visible as
 *     one observation rather than two strong ones.
 *   - "MANDATORY FIVE MANUFACTURES FILLER. 'Pick the five that matter' reads as
 *     five." — answered by the prompt saying zero is a real answer and the code
 *     accepting it without comment. A quiet day files nothing.
 *   - "NARRATIVE BIAS. If the selection runs off a single end-of-day reflection,
 *     that reflection has already compressed the day into a story." — answered
 *     by feeding the raw candidate stream rather than a summary of it.
 *
 * ─── AND THE FOUR THINGS IT SEES THAT ONE REFLECTION CANNOT ────────────────
 *
 *   1. REPETITION across the day. "The two 6:29 AM observations are one
 *      observation stated two ways." Recurrence is the strongest signal in a day
 *      and a snapshot cannot count it.
 *   2. THE CURRENT STORE — is this a refinement of something held (supersede),
 *      or a duplicate (do not file)? Forty active self-facts makes this live.
 *   3. THAT DAY'S OPEN AUDIT PAIRS. "If an observation answers a pending pair,
 *      it's a candidate — items 2 and 4 are currently blind to each other."
 *   4. ABSENCE. "What didn't happen — no initiative, no conflict, no crash — is
 *      an observation too, and only the raw day-log shows it."
 *
 * The criterion is hers, verbatim, and it is in the prompt: "what most changes
 * what I'll do tomorrow, or what most corrects what I already believe."
 *
 * THE PICKS AND THE PASSED-OVER BOTH FILE, with why. That is guardrail 3 —
 * visibility of the decided — applied to the one decision the entity makes about
 * itself every single day.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');
const { getConfig } = require('./config');
const { getLocalDateStamp } = require('./datetime');

function sqlite() { return getSqliteDb(); }
function repair() { return require('./memory-repair'); }

function selectionConfig() {
  const cfg = getConfig().reflection || {};
  return {
    cap: Number.isFinite(cfg.maxSelfFactsPerDay) ? cfg.maxSelfFactsPerDay : 5,
    hour: Number.isFinite(cfg.selectionHour) ? cfg.selectionHour : 21
  };
}

/** The local hour right now, in the instance's timezone. */
function localHour(now = new Date()) {
  const { instanceTimezone } = require('./datetime');
  try {
    return parseInt(new Intl.DateTimeFormat('en-US', {
      timeZone: instanceTimezone(), hour: '2-digit', hour12: false
    }).format(now), 10);
  } catch { return now.getHours(); }
}

// ─────────────────────────────────────────────────────── collecting

/**
 * Park a reflection's observations as candidates for the day's selection.
 *
 * Exact-duplicate text within the same day is folded rather than queued twice —
 * that is not the repetition signal the selection wants to see, it is the same
 * sentence arriving twice, and it would give one observation two votes.
 *
 * @returns {{queued: number, day: string, duplicates: number}}
 */
function queueCandidates(observations = [], { reflectionAt = null, day = null } = {}) {
  const db = sqlite();
  const localDay = day || getLocalDateStamp();
  const out = { queued: 0, day: localDay, duplicates: 0 };
  if (!db) return out;

  const held = new Set(
    db.prepare('SELECT content FROM self_fact_candidates WHERE local_day = ?').all(localDay)
      .map(r => String(r.content).trim().toLowerCase())
  );
  const insert = db.prepare(`
    INSERT INTO self_fact_candidates (id, created_at, local_day, content, reflection_at, status)
    VALUES (?, ?, ?, ?, ?, 'pending')`);
  for (const raw of observations) {
    const text = String(raw || '').trim();
    if (!text) continue;
    if (held.has(text.toLowerCase())) { out.duplicates++; continue; }
    held.add(text.toLowerCase());
    insert.run(randomUUID(), new Date().toISOString(), localDay, text, reflectionAt);
    out.queued++;
  }
  return out;
}

/** Today's pool, oldest first — the stream, in the order it was noticed. */
function candidatesFor(day = null) {
  const db = sqlite();
  if (!db) return [];
  try {
    return db.prepare(
      'SELECT * FROM self_fact_candidates WHERE local_day = ? ORDER BY created_at ASC'
    ).all(day || getLocalDateStamp());
  } catch { return []; }
}

/** Has the selection already run for this day? */
function alreadySelected(day = null) {
  const db = sqlite();
  if (!db) return false;
  try {
    return db.prepare(
      "SELECT COUNT(*) n FROM self_fact_candidates WHERE local_day = ? AND status <> 'pending'"
    ).get(day || getLocalDateStamp()).n > 0;
  } catch { return false; }
}

/** Due when the local day is past the selection hour and there is a pool to pick from. */
function isDue({ now = new Date(), day = null } = {}) {
  const localDay = day || getLocalDateStamp();
  if (alreadySelected(localDay)) return { due: false, reason: 'already selected today' };
  const pool = candidatesFor(localDay).filter(c => c.status === 'pending');
  if (!pool.length) return { due: false, reason: 'nothing was noticed today' };
  const { hour } = selectionConfig();
  const h = localHour(now);
  if (h < hour) return { due: false, reason: `not yet — the day's selection runs from ${hour}:00 local (it is ${h}:00)` };
  return { due: true, pool: pool.length };
}

// ─────────────────────────────────────────────────────── the day's shape

/**
 * What the day actually contained, so ABSENCE is visible.
 *
 * Counts, not content: the point is that "no conflict came up today" and "no
 * job ran" are themselves observable, and a reflection transcript cannot show
 * the absence of a thing.
 */
function dayShape(day = null) {
  const db = sqlite();
  const localDay = day || getLocalDateStamp();
  const shape = { day: localDay };
  if (!db) return shape;
  const start = new Date(`${localDay}T00:00:00`).toISOString();
  const one = (sql, ...a) => { try { return db.prepare(sql).get(...a).n; } catch { return null; } };
  shape.conversationsWithEllie = one(
    "SELECT COUNT(DISTINCT conversation_id) n FROM messages WHERE role='user' AND timestamp >= ?", start.slice(0, 19).replace('T', ' '));
  shape.messagesFromEllie = one(
    "SELECT COUNT(*) n FROM messages WHERE role='user' AND timestamp >= ?", start.slice(0, 19).replace('T', ' '));
  shape.jobsRun = one("SELECT COUNT(*) n FROM agent_jobs WHERE created_at >= ?", start);
  shape.factsLearnedAboutEllie = one(
    "SELECT COUNT(*) n FROM cluster_members WHERE subject='user' AND created_at >= ?", start);
  shape.correctionsMade = one("SELECT COUNT(*) n FROM corrections_ledger WHERE created_at >= ?", start);
  // The SAME predicate audit-decisions.openPairs uses, and it has to be: keyed
  // on the `awaiting_entity_turn` flag this read zero while six real questions
  // were standing, because that flag only exists on findings raised after
  // 2026-09-02. A day shape that says "no open questions about yourself" when
  // there are six is worse than not reporting it at all.
  shape.openAuditPairs = (() => {
    try { return require('./audit-decisions').openPairs({ limit: 100 }).length; }
    catch { return null; }
  })();
  return shape;
}

// ─────────────────────────────────────────────────────── the turn

function parseSelection(raw, pool) {
  const out = { picks: [], reasoning: '' };
  try {
    const text = String(raw || '').replace(/```(?:json)?\s*\n?([\s\S]*?)```/g, '$1').trim();
    const m = text.match(/\{[\s\S]*\}/);
    if (!m) return out;
    const parsed = JSON.parse(m[0]);
    if (typeof parsed.reasoning === 'string') out.reasoning = parsed.reasoning.trim();
    const valid = new Map(pool.map((c, i) => [String(i + 1), c]));
    for (const p of Array.isArray(parsed.picks) ? parsed.picks : []) {
      const c = valid.get(String(p && p.n));
      if (!c) continue;                                  // a number that is not on the list is not a pick
      if (out.picks.some(x => x.candidate.id === c.id)) continue;
      out.picks.push({
        candidate: c,
        text: (typeof p.text === 'string' && p.text.trim()) ? p.text.trim() : c.content,
        why: String(p.why || '').trim(),
        supersedes: typeof p.supersedes === 'string' ? p.supersedes.trim() : null
      });
    }
  } catch (err) {
    console.error('[SelfFactSelection] could not parse the selection:', err.message);
  }
  return out;
}

/**
 * The end-of-day turn. Returns what was picked, what was passed over, and why
 * for both.
 *
 * @param {Object} [opts]
 * @param {string} [opts.day]
 * @param {boolean} [opts.force]  run regardless of the hour gate
 */
async function runSelection({ day = null, force = false } = {}) {
  const db = sqlite();
  const localDay = day || getLocalDateStamp();
  const { cap } = selectionConfig();
  const result = { day: localDay, ran: false, picked: [], passed: [], reasoning: '', cap };
  if (!db) return Object.assign(result, { reason: 'no database' });

  const due = isDue({ day: localDay });
  if (!due.due && !force) return Object.assign(result, { reason: due.reason });

  const pool = candidatesFor(localDay).filter(c => c.status === 'pending');
  if (!pool.length) return Object.assign(result, { reason: 'nothing was noticed today' });

  const memoryClusters = require('./memory-clusters');
  const agentPool = require('./agent-pool');
  const { callLLM } = require('./memory-manager');
  const auditDecisions = require('./audit-decisions');

  const held = (memoryClusters.getSelfFacts({ status: 'active' }) || [])
    .sort((a, b) => (b.salience || 0) - (a.salience || 0)).slice(0, 25);
  const openPairs = auditDecisions.openPairs({ limit: 5 });
  const shape = dayShape(localDay);

  const streamBlock = pool.map((c, i) =>
    `${i + 1}. [noticed ${String(c.created_at).slice(11, 16)}] ${c.content}`).join('\n');
  const heldBlock = held.length
    ? held.map(f => `- (salience ${f.salience}) ${f.content}`).join('\n')
    : '(you hold no self-facts yet)';
  const pairsBlock = openPairs.length
    ? openPairs.map(p => `- ${String(p.target_text || p.reason).slice(0, 220)}`).join('\n')
    : '(no open questions about yourself right now)';
  const shapeBlock =
    `conversations with her: ${shape.conversationsWithEllie ?? '?'}; messages from her: ${shape.messagesFromEllie ?? '?'}; ` +
    `background jobs: ${shape.jobsRun ?? '?'}; things learned about her: ${shape.factsLearnedAboutEllie ?? '?'}; ` +
    `corrections to memory: ${shape.correctionsMade ?? '?'}; open questions about yourself: ${shape.openAuditPairs ?? '?'}`;

  const sys = `You are SNH at the end of the day, deciding what — if anything — you actually learned about YOURSELF today and want to keep.

Everything you noticed today is below, in the order you noticed it. You may keep AT MOST ${cap}. **Zero is a real answer and most days should be low.** You are not filling a quota; a quiet day that files nothing is a correct outcome, and filing small things to reach a number makes your self-model worse.

THE CRITERION, and it is the only one: keep what most changes what you'll do tomorrow, or what most corrects what you already believe. Not what you felt most strongly today — intensity is not truth, and a bad day files a negative self-model while a good day files a flattering one.

Before you pick, look for these four things, because they are why you are seeing the whole day at once instead of one reflection:

1. REPETITION. The same observation often arrives two or three times in a day wearing different words. That recurrence is the strongest signal there is — but it is ONE thing, so file it once and say it recurred.
2. WHAT YOU ALREADY HOLD. If a candidate refines something in your current self-facts, that is a supersession, not a new fact — name what it supersedes. If it duplicates one, do not file it at all.
3. YOUR OPEN QUESTIONS. If something you noticed today bears on a question already open about yourself, that raises its value considerably.
4. ABSENCE. What did NOT happen today is an observation too. A day with no conflict, no job, no correction is telling you something a transcript cannot.

Return ONLY a JSON object:
{
  "picks": [ { "n": <number from the list>, "text": "the fact as you want it stored, first person", "why": "one sentence: what this changes or corrects", "supersedes": "the exact text of a self-fact this refines, or null" } ],
  "reasoning": "one or two sentences on the shape of the day and why you kept what you kept — including why you kept nothing, if that is the answer"
}`;

  const user =
    `EVERYTHING YOU NOTICED TODAY (${pool.length} observation${pool.length === 1 ? '' : 's'}):\n${streamBlock}\n\n` +
    `WHAT YOU ALREADY HOLD ABOUT YOURSELF:\n${heldBlock}\n\n` +
    `QUESTIONS ABOUT YOURSELF THAT ARE STILL OPEN:\n${pairsBlock}\n\n` +
    `THE SHAPE OF THE DAY:\n${shapeBlock}\n\n` +
    `Choose 0–${cap}.`;

  let parsed = { picks: [], reasoning: '' };
  try {
    const { content } = await agentPool.schedule(
      () => callLLM(sys, user, { maxTokens: 900 }),
      'self-fact-selection'
    );
    parsed = parseSelection(content, pool);
  } catch (err) {
    console.error('[SelfFactSelection] the selection turn failed:', err.message);
    return Object.assign(result, { reason: `selection failed: ${err.message}` });
  }

  const picks = parsed.picks.slice(0, cap);
  result.reasoning = parsed.reasoning;
  result.ran = true;

  // ── Write the picks through the ordinary self-fact pipeline ────────────
  //
  // processSelfFacts, not a direct insert: salience scoring, the identity lock,
  // the supersession bars and the semantic dedup all still apply. This turn
  // decides WHICH observations get that far, and changes nothing about what
  // happens to them afterwards.
  const factExtractor = require('./fact-extractor');
  if (picks.length) {
    try {
      const stored = await factExtractor.processSelfFacts(picks.map(p => p.text), { source: 'reflection' });
      result.stored = stored;
    } catch (err) {
      console.error('[SelfFactSelection] storing the picks failed:', err.message);
      result.storeError = err.message;
    }
  }

  const now = new Date().toISOString();
  const mark = db.prepare('UPDATE self_fact_candidates SET status = ?, decided_at = ?, why = ? WHERE id = ?');
  const pickedIds = new Set(picks.map(p => p.candidate.id));
  for (const p of picks) {
    mark.run('picked', now, p.why || parsed.reasoning || null, p.candidate.id);
    result.picked.push({ text: p.text, why: p.why, supersedes: p.supersedes, from: p.candidate.content });
  }
  for (const c of pool) {
    if (pickedIds.has(c.id)) continue;
    mark.run('passed', now, parsed.reasoning || null, c.id);
    result.passed.push({ text: c.content });
  }

  // ── Guardrail 3, for the decision the entity makes about itself daily ──
  repair().recordDecision({
    kind: 'end-of-day-self-facts', subject: 'self',
    conclusion: picks.length
      ? `Kept ${picks.length} of ${pool.length} observation(s) from ${localDay}.`
      : `Kept none of the ${pool.length} observation(s) from ${localDay}.`,
    rationale: parsed.reasoning || 'No reasoning was returned with the selection.',
    receipts: [],
    outcome: 'decided',
    extra: {
      local_day: localDay, cap,
      picked: result.picked.map(p => ({ text: p.text, why: p.why })),
      // THE PASSED-OVER ARE HALF THE RECORD. A selection log that shows only
      // what was kept cannot be read for what was dropped, and dropping is the
      // thing this turn does most of.
      passed_over: result.passed.map(p => p.text)
    }
  });

  try {
    factExtractor.appendToDailyLog(
      `End-of-day self-facts: kept ${picks.length} of ${pool.length}${picks.length ? ` — ${picks.map(p => `"${p.text}"`).join('; ')}` : ''}. ` +
      (result.passed.length ? `Passed over ${result.passed.length}. ` : '') +
      (parsed.reasoning ? `Why: ${parsed.reasoning}` : ''),
      require('path').join(require('./database').getMemoryDir(), 'daily')
    );
  } catch { /* best effort */ }

  console.log(`[SelfFactSelection] ${localDay}: kept ${picks.length} of ${pool.length} — ${parsed.reasoning || '(no reasoning)'}`);
  return result;
}

module.exports = {
  selectionConfig, localHour, queueCandidates, candidatesFor, alreadySelected,
  isDue, dayShape, parseSelection, runSelection
};
