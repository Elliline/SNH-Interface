/**
 * Memory inspection — the READ layer behind memory_search / memory_list /
 * memory_count / memory_get.
 *
 * WHY THIS EXISTS: the entity could be given memory and could be given facts,
 * but it could not LOOK. Everything it knew arrived by injection — a budgeted
 * slice chosen for it before the turn started — so "how many facts do you have
 * about MettaSphere" and "why do you believe that" were answerable only by
 * guessing from whatever happened to be in the window. Guessing about your own
 * memory is the same failure family as the phantom action: a confident answer
 * with nothing behind it.
 *
 * READ-ONLY, ABSOLUTELY. Nothing in this module writes, and nothing in it may
 * ever write. Writes are `write_memory` → `db/memory-write.js` → the fact-store
 * funnel, and that stays the single path (CLAUDE.md). The spec says it plainly:
 * "Reads are unrestricted. Writes only through the fact-store funnel. No
 * inspection tool writes."
 *
 * NO REDACTION. memory_get returns the full provenance — conversation id,
 * message id, the verbatim words, the modality, the salience rationale, the
 * successor chain, every corroboration. That was Ellie's decision, taken with
 * Aurelius: it is his memory, and a memory he can be shown only a sanitised view
 * of is not really his. EXPLAIN in the spec depends on it too — the corrector's
 * evidence-dominance bars are defined in terms of provenance, so a redacted view
 * would leave the semantic tier adjudicating on nothing.
 *
 * RESULT DISCIPLINE. His injection budget is small and tool results land in the
 * same window. Rows are single-line and capped; counts return numbers, not
 * content; only memory_get is allowed to be verbose, and only for one fact.
 */

const { getSqliteDb } = require('./database');
const { getConfig } = require('./config');
const { formatFactTimestamp } = require('./datetime');

const HOUR_MS = 60 * 60 * 1000;

/** Tools counted against the shared inspection rate cap. */
const INSPECT_TOOLS = ['memory_search', 'memory_list', 'memory_count', 'memory_get'];

/**
 * Result-discipline caps. Deliberately split from the tool's on/off + rate cap
 * (which live in `tools.memoryInspect`, matching create_cron_job and
 * write_memory so the registry and the drift-check read one place per tool).
 * These are about the SHAPE of a result, not about permission to call.
 */
function limits() {
  const c = (getConfig().memory && getConfig().memory.inspect) || {};
  return {
    maxRows: Math.max(1, c.maxRows ?? 20),
    defaultRows: Math.max(1, c.defaultRows ?? 10),
    rowChars: Math.max(40, c.rowChars ?? 140),
    verbatimChars: Math.max(80, c.verbatimChars ?? 400),
    rationaleChars: Math.max(40, c.rationaleChars ?? 240),
    maxClusters: Math.max(1, c.maxClusters ?? 40),
    maxCorroborations: Math.max(1, c.maxCorroborations ?? 10),
    semanticFloor: Number.isFinite(c.semanticFloor) ? c.semanticFloor : 0.55
  };
}

function toolCfg() {
  const c = (getConfig().tools && getConfig().tools.memoryInspect) || {};
  return {
    enabled: c.enabled !== false,
    maxCallsPerHour: Math.max(1, c.maxCallsPerHour ?? 40)
  };
}

/**
 * Trailing-hour cap across all four tools, counted from `tool_call_log` so a
 * restart hands out no fresh budget — the same shape as write_memory's cap and
 * the cron tool's. Shared rather than per-tool: the thing worth bounding is how
 * much of a turn goes into rummaging, and four separate budgets would let a loop
 * spend 4× while each individual counter looked healthy.
 */
function checkCap() {
  const db = getSqliteDb();
  if (!db) return { ok: false, reason: 'database unavailable' };
  const { maxCallsPerHour } = toolCfg();
  const sinceIso = new Date(Date.now() - HOUR_MS).toISOString();
  const placeholders = INSPECT_TOOLS.map(() => '?').join(',');
  const recent = db.prepare(
    `SELECT COUNT(*) AS n FROM tool_call_log
     WHERE tool IN (${placeholders}) AND outcome != 'rejected-cap'
       AND datetime(created_at) > datetime(?)`
  ).get(...INSPECT_TOOLS, sinceIso).n;

  if (recent >= maxCallsPerHour) {
    return { ok: false, reason: `memory inspection rate cap reached (${recent}/${maxCallsPerHour} lookups in the last hour)` };
  }
  return { ok: true, used: recent, maxCallsPerHour };
}

/** Current cap usage — for the Thinking tab and for tests. */
function capStatus() {
  const { maxCallsPerHour } = toolCfg();
  const c = checkCap();
  return { recentHour: c.used ?? maxCallsPerHour, maxCallsPerHour };
}

// ============ shared helpers ============

const trim = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

/**
 * Normalise the filter arguments the model supplies. Everything is optional and
 * everything invalid becomes "no filter" rather than an error — a tool that
 * rejects a slightly-wrong argument just gets called again with another guess,
 * which costs a round trip and teaches it nothing.
 *
 * status defaults to 'active', because "what do I believe" is the question being
 * asked ninety-nine times in a hundred, and inactive facts reaching an answer as
 * though they were live is the drift this whole subsystem exists to prevent.
 */
function normalizeFilters(args = {}) {
  const subject = ['user', 'self'].includes(String(args.subject || '').toLowerCase())
    ? String(args.subject).toLowerCase() : null;

  const rawStatus = String(args.status || 'active').toLowerCase();
  const status = ['active', 'inactive', 'any'].includes(rawStatus) ? rawStatus : 'active';

  const cluster = args.cluster ? String(args.cluster).trim() : null;

  const L = limits();
  let limit = Number.isFinite(args.limit) ? Math.floor(args.limit) : L.defaultRows;
  limit = Math.max(1, Math.min(L.maxRows, limit));

  const offset = Number.isFinite(args.offset) ? Math.max(0, Math.floor(args.offset)) : 0;

  return { subject, status, cluster, limit, offset };
}

/** SQL WHERE fragment + bindings for a normalised filter set. */
function whereFor(f, { clusterIsId = false } = {}) {
  const where = [];
  const bind = [];
  if (f.status === 'active') where.push("m.status = 'active'");
  else if (f.status === 'inactive') where.push("m.status != 'active'");
  if (f.subject) { where.push('COALESCE(m.subject, ?) = ?'); bind.push('user', f.subject); }
  if (f.cluster) {
    if (clusterIsId) { where.push('m.cluster_id = ?'); bind.push(f.cluster); }
    else { where.push('(c.name = ? COLLATE NOCASE OR m.cluster_id = ?)'); bind.push(f.cluster, f.cluster); }
  }
  return { sql: where.length ? `WHERE ${where.join(' AND ')}` : '', bind };
}

/** One compact line per fact — the shape every list/search row takes. */
function compactRow(row) {
  const L = limits();
  return {
    id: row.id,
    text: trim(row.content, L.rowChars),
    subject: row.subject || 'user',
    salience: row.salience ?? 5,
    status: row.status === 'active' ? 'active' : `inactive (${row.inactive_reason || 'unknown'})`,
    cluster: row.cluster_name || null
  };
}

// ============ 1. memory_search ============

/**
 * Semantic search plus plain-text match, merged.
 *
 * Both, not either. Vector similarity finds "what she does for work" when the
 * fact says "MSP"; a LIKE finds the fact that literally says "MettaSphere" even
 * when the embedding puts it 30th. The failure modes are different enough that
 * running one of them is running the wrong one about a third of the time — and
 * the text half costs nothing.
 *
 * Ranking is by best available similarity, with exact-substring hits floored at
 * a similarity that keeps them visible, so a literal name match is never ranked
 * off the end of the list by a vaguely-related neighbour.
 */
async function search(args = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'database unavailable' };
  const query = String(args.query || '').trim();
  if (!query) return { error: 'query is required' };

  const f = normalizeFilters(args);
  const L = limits();
  const found = new Map(); // id -> {row, similarity, how}

  // --- text half: substring match, cheap and exact ---
  try {
    const { sql, bind } = whereFor(f);
    const textWhere = sql ? `${sql} AND m.content LIKE ?` : 'WHERE m.content LIKE ?';
    const rows = db.prepare(`
      SELECT m.*, c.name AS cluster_name
      FROM cluster_members m LEFT JOIN memory_clusters c ON c.id = m.cluster_id
      ${textWhere}
      ORDER BY m.salience DESC, datetime(m.created_at) DESC
      LIMIT ?
    `).all(...bind, `%${query}%`, L.maxRows);
    for (const r of rows) found.set(r.id, { row: r, similarity: 1, how: 'text' });
  } catch (err) {
    console.error('[MemoryInspect] text search failed:', err.message);
  }

  // --- semantic half ---
  try {
    const memoryClusters = require('./memory-clusters');
    // Floored, not top-k-at-threshold-zero. Without the floor a search for
    // "MettaSphere" reported 60 matches against 6 facts that mention it — the
    // nearest-neighbour tail dressed up as relevance. A tool that overstates
    // what it found teaches him to trust a number that is not true.
    const { candidates } = await memoryClusters.findActiveNeighbours(query, {
      subject: f.subject || 'user',
      threshold: L.semanticFloor,
      limit: L.maxRows * 2
    });
    // findActiveNeighbours is active-only and single-subject by construction, so
    // it can only contribute to an active-scoped search. Inactive and
    // cross-subject searches are served by the text half plus the direct lookup
    // below — say so rather than silently returning less.
    const semanticApplies = f.status === 'active';
    if (semanticApplies) {
      const stmt = db.prepare(`
        SELECT m.*, c.name AS cluster_name
        FROM cluster_members m LEFT JOIN memory_clusters c ON c.id = m.cluster_id
        WHERE m.id = ?
      `);
      for (const cand of candidates) {
        if (found.has(cand.memberId)) continue;
        const row = stmt.get(cand.memberId);
        if (!row) continue;
        if (f.cluster && row.cluster_name !== f.cluster && row.cluster_id !== f.cluster) continue;
        found.set(row.id, { row, similarity: cand.similarity, how: 'semantic' });
      }
    }
  } catch (err) {
    console.error('[MemoryInspect] semantic search failed:', err.message);
  }

  const ranked = Array.from(found.values())
    .sort((a, b) => b.similarity - a.similarity || (b.row.salience ?? 0) - (a.row.salience ?? 0))
    .slice(0, f.limit);

  return {
    query,
    filters: { subject: f.subject || 'any', status: f.status },
    matched: found.size,
    returned: ranked.length,
    not_shown: Math.max(0, found.size - ranked.length),
    // Never silent: if the cap cut the list, say so, so "that is all I have" is
    // only ever said when it is true.
    note: found.size > ranked.length
      ? `${found.size} facts matched; showing the top ${ranked.length}. Ask again with a higher limit or a narrower query to see the rest — do not say these are all of them.`
      : null,
    results: ranked.map(r => ({ ...compactRow(r.row), matched_by: r.how }))
  };
}

// ============ 2. memory_list ============

/**
 * Facts by cluster / subject / status, paginated — or, in mode 'clusters', the
 * cluster names and their member counts.
 *
 * The clusters mode is the one that makes the others usable: without it he has
 * to guess a cluster name before he can filter by one, and a guessed name that
 * matches nothing is indistinguishable from a cluster that is genuinely empty.
 */
function list(args = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'database unavailable' };
  const L = limits();
  const f = normalizeFilters(args);

  if (String(args.mode || '').toLowerCase() === 'clusters') {
    const where = [];
    const bind = [];
    if (f.status === 'active') where.push("m.status = 'active'");
    else if (f.status === 'inactive') where.push("m.status != 'active'");
    if (f.subject) { where.push('COALESCE(m.subject, ?) = ?'); bind.push('user', f.subject); }
    const rows = db.prepare(`
      SELECT c.id, c.name, COALESCE(c.subject, 'user') AS subject, COUNT(m.id) AS members
      FROM memory_clusters c LEFT JOIN cluster_members m ON m.cluster_id = c.id
      ${where.length ? `AND ${where.join(' AND ')}` : ''}
      GROUP BY c.id
      HAVING members > 0
      ORDER BY members DESC, c.name ASC
      LIMIT ?
    `).all(...bind, L.maxClusters);
    // Counted under the SAME filter as the rows, and excluding clusters with no
    // members under it. Counting every row in memory_clusters instead reported
    // 118 against a list of 40, which he read out as "118 active clusters" —
    // a number that was never true of anything he could have looked at.
    const total = db.prepare(`
      SELECT COUNT(*) AS n FROM (
        SELECT c.id FROM memory_clusters c LEFT JOIN cluster_members m ON m.cluster_id = c.id
        ${where.length ? `AND ${where.join(' AND ')}` : ''}
        GROUP BY c.id HAVING COUNT(m.id) > 0
      )
    `).get(...bind).n;
    return {
      mode: 'clusters',
      filters: { subject: f.subject || 'any', status: f.status },
      returned: rows.length,
      clusters_with_facts: total,
      not_shown: Math.max(0, total - rows.length),
      note: total > rows.length
        ? `${total} clusters hold facts under this filter; showing the ${rows.length} largest. Do not say these are all of them.`
        : null,
      clusters: rows.map(r => ({ name: r.name, subject: r.subject, members: r.members }))
    };
  }

  const { sql, bind } = whereFor(f);
  const total = db.prepare(`
    SELECT COUNT(*) AS n FROM cluster_members m
    LEFT JOIN memory_clusters c ON c.id = m.cluster_id ${sql}
  `).get(...bind).n;

  const rows = db.prepare(`
    SELECT m.*, c.name AS cluster_name
    FROM cluster_members m LEFT JOIN memory_clusters c ON c.id = m.cluster_id
    ${sql}
    ORDER BY m.salience DESC, datetime(m.created_at) DESC
    LIMIT ? OFFSET ?
  `).all(...bind, f.limit, f.offset);

  return {
    mode: 'facts',
    filters: { subject: f.subject || 'any', status: f.status, cluster: f.cluster || 'any' },
    total,
    offset: f.offset,
    returned: rows.length,
    more: f.offset + rows.length < total,
    next_offset: f.offset + rows.length < total ? f.offset + rows.length : null,
    results: rows.map(compactRow)
  };
}

// ============ 3. memory_count ============

/**
 * Counts by any filter combination, plus an optional `query` that counts the
 * facts whose text contains it. Never returns fact text — the whole point of a
 * separate count tool is that "how many do you have about X" costs a number
 * rather than twenty rows he then has to be trusted not to summarise wrongly.
 */
function count(args = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'database unavailable' };
  const f = normalizeFilters(args);
  const { sql, bind } = whereFor(f);
  const query = args.query ? String(args.query).trim() : null;

  const base = `FROM cluster_members m LEFT JOIN memory_clusters c ON c.id = m.cluster_id ${sql}`;
  const matching = query
    ? db.prepare(`SELECT COUNT(*) AS n ${base} ${sql ? 'AND' : 'WHERE'} m.content LIKE ?`).get(...bind, `%${query}%`).n
    : db.prepare(`SELECT COUNT(*) AS n ${base}`).get(...bind).n;

  // A breakdown is what makes the number interpretable — "31" means something
  // different when 30 of them are inactive.
  const breakdown = db.prepare(`
    SELECT COALESCE(m.subject, 'user') AS subject,
           CASE WHEN m.status = 'active' THEN 'active' ELSE 'inactive' END AS state,
           COUNT(*) AS n
    FROM cluster_members m
    GROUP BY subject, state
  `).all();

  return {
    filters: { subject: f.subject || 'any', status: f.status, cluster: f.cluster || 'any', query: query || null },
    count: matching,
    corpus: {
      user_active: breakdown.find(b => b.subject === 'user' && b.state === 'active')?.n ?? 0,
      user_inactive: breakdown.find(b => b.subject === 'user' && b.state === 'inactive')?.n ?? 0,
      self_active: breakdown.find(b => b.subject === 'self' && b.state === 'active')?.n ?? 0,
      self_inactive: breakdown.find(b => b.subject === 'self' && b.state === 'inactive')?.n ?? 0
    }
  };
}

// ============ 4. memory_get ============

/**
 * One fact, in full depth. The EXPLAIN capability: which conversation, which
 * message, the verbatim words, how they arrived, why it scored what it scored,
 * what it replaced or was replaced by, and every time it has been restated.
 *
 * Accepts an id or an id prefix — he sees 8-character prefixes in search and
 * list results, and requiring him to have kept the full UUID would make the
 * transparency theoretical.
 */
function get(args = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'database unavailable' };
  const raw = String(args.id || args.fact_id || '').trim();
  if (!raw) return { error: 'id is required' };
  const L = limits();

  let row = db.prepare(`
    SELECT m.*, c.name AS cluster_name
    FROM cluster_members m LEFT JOIN memory_clusters c ON c.id = m.cluster_id
    WHERE m.id = ?
  `).get(raw);

  if (!row) {
    const matches = db.prepare(`
      SELECT m.*, c.name AS cluster_name
      FROM cluster_members m LEFT JOIN memory_clusters c ON c.id = m.cluster_id
      WHERE m.id LIKE ? LIMIT 5
    `).all(`${raw}%`);
    if (matches.length === 1) row = matches[0];
    else if (matches.length > 1) {
      return { error: `"${raw}" matches ${matches.length} facts — use more of the id`,
               candidates: matches.map(m => ({ id: m.id, text: trim(m.content, 80) })) };
    }
  }
  if (!row) return { error: `no fact with id "${raw}"` };

  // --- successor chain: what replaced this, and what that in turn became ---
  const chain = [];
  let cursor = row.successor_id || row.superseded_by;
  const guard = new Set([row.id]);
  while (cursor && !guard.has(cursor) && chain.length < 8) {
    guard.add(cursor);
    const next = db.prepare('SELECT id, content, status, successor_id, superseded_by FROM cluster_members WHERE id = ?').get(cursor);
    if (!next) break;
    chain.push({ id: next.id, text: trim(next.content, L.rowChars), status: next.status });
    cursor = next.successor_id || next.superseded_by;
  }

  // --- what this fact itself replaced ---
  const replaced = db.prepare(
    'SELECT id, content FROM cluster_members WHERE successor_id = ? OR superseded_by = ? LIMIT 5'
  ).all(row.id, row.id).map(r => ({ id: r.id, text: trim(r.content, L.rowChars) }));

  // --- corroborations: every time this was said again ---
  let corroborations = [];
  let corroborationCount = 0;
  try {
    corroborationCount = db.prepare('SELECT COUNT(*) AS n FROM fact_corroborations WHERE member_id = ?').get(row.id).n;
    corroborations = db.prepare(
      'SELECT created_at, restated_as, similarity, detected_by, input_modality, conversation_id, message_id FROM fact_corroborations WHERE member_id = ? ORDER BY datetime(created_at) DESC LIMIT ?'
    ).all(row.id, L.maxCorroborations).map(c => ({
      when: c.created_at,
      restated_as: trim(c.restated_as, L.rowChars),
      similarity: c.similarity,
      detected_by: c.detected_by,
      input_modality: c.input_modality,
      conversation_id: c.conversation_id,
      message_id: c.message_id
    }));
  } catch { /* table absent on an un-migrated DB — not fatal for the rest */ }

  return {
    id: row.id,
    text: row.content,
    subject: row.subject || 'user',
    salience: row.salience ?? 5,
    // Stored since Phase 1, and the reason "why is this a 10?" is answerable at
    // all. It used to be written to the daily log as prose and discarded.
    salience_rationale: trim(row.salience_rationale, L.rationaleChars) || null,
    status: row.status,
    inactive_reason: row.inactive_reason || null,
    replaced_by: chain.length ? chain : null,
    replaced: replaced.length ? replaced : null,
    cluster: row.cluster_name || null,
    claim_type: row.claim_type || null,
    locked: !!row.locked,
    lock_category: row.lock_category || null,
    learned: row.created_at ? formatFactTimestamp(new Date(row.created_at)) : null,
    learned_iso: row.created_at || null,
    updated_iso: row.updated_at || null,
    source: row.source || null,
    provenance: {
      conversation_id: row.conversation_id || null,
      message_id: row.message_id || null,
      // Null is a real answer here and means "this fact predates provenance
      // capture" — distinct from an empty string, and worth him being able to
      // say out loud rather than inventing a source.
      verbatim_source_text: row.verbatim_source_text ? trim(row.verbatim_source_text, L.verbatimChars) : null,
      input_modality: row.input_modality || null
    },
    // TOP-LEVEL, and an imperative, because neither weaker form worked.
    //
    // Measured 2026-08-03 on this fact, which has null provenance: with no note
    // at all he read source='fact-extraction' plus the learned date and said it
    // was "pulled directly from our conversation on July 4th" — an inference
    // beyond the record. With the same warning nested inside `provenance` he did
    // worse, and invented the quote: 'The record shows you said: "My MSP is
    // MettaSphere LLC."' Nothing in the record says that. A null field does not
    // read as an absence of evidence; it reads as a blank to be filled.
    ...(row.conversation_id || row.verbatim_source_text ? {} : {
      provenance_warning:
        'THIS FACT HAS NO RECORDED SOURCE. No conversation, no message, and no record of the ' +
        'original wording were kept — it predates provenance capture (2026-08-02). You know WHEN ' +
        'you learned it and that it came from ordinary conversation, and nothing else. Do NOT name ' +
        'a conversation and do NOT quote or paraphrase what was said: there is no stored wording to ' +
        'quote, so any quote you give is invented. Say the original wording was not kept.'
    }),
    corroborations: corroborationCount,
    corroboration_detail: corroborations.length ? corroborations : null
  };
}

// ============ dispatch ============

/**
 * Single entry point for all four tools: enforces the shared rate cap, runs the
 * op, and logs the call. Every path through here writes exactly one
 * `tool_call_log` row, including the refusals — an inspection that was refused
 * and an inspection that returned nothing look identical from the answer, and
 * only the log can tell them apart.
 *
 * @param {'memory_search'|'memory_list'|'memory_count'|'memory_get'} tool
 * @param {Object} args
 * @param {Object} [context] - { conversationId, caller }
 */
async function run(tool, args = {}, context = {}) {
  const log = (outcome, detail) => {
    try {
      require('./cron-jobs').logToolCall({
        tool, args, outcome, detail,
        conversationId: context.conversationId || null
      });
    } catch (e) { console.error('[MemoryInspect] logToolCall failed:', e.message); }
  };

  if (!toolCfg().enabled) {
    log('error', 'tool disabled in config');
    return { error: 'Memory inspection is disabled.' };
  }
  if (!INSPECT_TOOLS.includes(tool)) {
    log('error', `unknown inspection tool ${tool}`);
    return { error: `Unknown inspection tool: ${tool}` };
  }

  const cap = checkCap();
  if (!cap.ok) {
    log('rejected-cap', cap.reason);
    return { error: `Not looked up — ${cap.reason}. Say plainly that you have hit your own limit for looking things up rather than answering from memory as though you had checked.` };
  }

  let result;
  try {
    if (tool === 'memory_search') result = await search(args);
    else if (tool === 'memory_list') result = list(args);
    else if (tool === 'memory_count') result = count(args);
    else result = get(args);
  } catch (err) {
    console.error(`[MemoryInspect] ${tool} failed:`, err.message);
    log('error', err.message);
    return { error: `The lookup failed: ${err.message}` };
  }

  if (result && result.error) {
    log('error', String(result.error).slice(0, 200));
    return result;
  }

  const detail =
    tool === 'memory_count' ? `count=${result.count}`
    : tool === 'memory_get' ? `fact ${String(result.id).slice(0, 8)} "${trim(result.text, 60)}"`
    : `${result.returned} row(s)${result.truncated ? ` (+${result.truncated} not shown)` : ''}`;
  log('read', detail);
  console.log(`[MemoryInspect] ${tool}: ${detail} (${cap.used + 1}/${cap.maxCallsPerHour} this hour)`);
  return result;
}

module.exports = {
  run, search, list, count, get,
  checkCap, capStatus, limits, toolCfg,
  INSPECT_TOOLS
};
