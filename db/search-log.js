/**
 * Every search call, with the provider named.
 *
 * WHY THIS EXISTS, precisely. On 2026-08-18 a background job made seven searches
 * that all failed, and the only trace anywhere was the model's own sentence to
 * Ellie: "there seems to be an issue with the search tool". Which provider ran,
 * what it was asked, whether it came back with anything, and why it did not were
 * all unrecorded — so answering "was it Exa or SearXNG, and did it return
 * anything" meant reading a journal by hand. It cost hours.
 *
 * So this is telemetry with one job: make that question answerable from data.
 * One row per PROVIDER ATTEMPT, not per tool call — a search that tried Exa and
 * fell through to SearXNG writes two rows, both marked with the same attempt
 * group, because "Exa was empty and SearXNG had four" is the exact shape of the
 * thing worth knowing.
 *
 * Operational, so it goes to the ops side of the house (Thinking tab) and is
 * NEVER injected into a chat. And it is best-effort by construction: every
 * function here swallows its own errors, because a telemetry table that can fail
 * a search is worse than no telemetry at all.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');

/**
 * Record one provider attempt.
 *
 * @param {Object} row
 * @param {string} row.provider   - 'exa' | 'searxng'
 * @param {string} row.query      - what it was asked, verbatim
 * @param {number} row.numResults - how many came back (0 is a result)
 * @param {string} row.outcome    - 'results' | 'empty' | 'error' | 'skipped'
 * @param {string} [row.detail]   - the error, or why it was skipped
 * @param {string} [row.caller]   - 'chat' | 'agent-job:3f9a1c2b' | 'heartbeat:corrector'
 * @param {number} [row.latencyMs]
 * @param {string} [row.attemptId]- groups the providers tried for ONE tool call
 * @param {boolean} [row.served]  - true for the attempt whose results were used
 * @param {number} [row.costUsd]
 * @returns {string|null} the row id
 */
function logSearchCall(row = {}) {
  const db = getSqliteDb();
  if (!db) return null;
  try {
    const id = randomUUID();
    db.prepare(`
      INSERT INTO search_call_log
        (id, created_at, provider, query, num_results, outcome, detail, caller, latency_ms, attempt_id, served, cost_usd)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      new Date().toISOString(),
      String(row.provider || 'unknown'),
      String(row.query || '').slice(0, 500),
      Number.isFinite(row.numResults) ? row.numResults : 0,
      String(row.outcome || 'unknown'),
      row.detail ? String(row.detail).slice(0, 500) : null,
      row.caller ? String(row.caller).slice(0, 120) : null,
      Number.isFinite(row.latencyMs) ? Math.round(row.latencyMs) : null,
      row.attemptId || null,
      row.served ? 1 : 0,
      Number.isFinite(row.costUsd) ? row.costUsd : null
    );
    return id;
  } catch {
    return null;   // telemetry never fails a search
  }
}

/** Newest first, for the Thinking tab and for answering the question by hand. */
function recentSearchCalls({ limit = 50 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  try {
    return db.prepare(
      'SELECT * FROM search_call_log ORDER BY datetime(created_at) DESC LIMIT ?'
    ).all(Math.min(Math.max(1, limit), 500));
  } catch {
    return [];
  }
}

/**
 * Provider health over a trailing window — the summary form of the question.
 * @returns {Array<{provider, attempts, with_results, empty, errors, avg_latency_ms, cost_usd}>}
 */
function providerSummary({ hours = 24 } = {}) {
  const db = getSqliteDb();
  if (!db) return [];
  try {
    return db.prepare(`
      SELECT provider,
             COUNT(*) AS attempts,
             SUM(CASE WHEN outcome = 'results' THEN 1 ELSE 0 END) AS with_results,
             SUM(CASE WHEN outcome = 'empty'   THEN 1 ELSE 0 END) AS empty,
             SUM(CASE WHEN outcome = 'error'   THEN 1 ELSE 0 END) AS errors,
             SUM(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) AS skipped,
             CAST(AVG(latency_ms) AS INTEGER) AS avg_latency_ms,
             ROUND(SUM(COALESCE(cost_usd, 0)), 4) AS cost_usd
      FROM search_call_log
      WHERE datetime(created_at) >= datetime('now', ?)
      GROUP BY provider
      ORDER BY attempts DESC
    `).all(`-${Math.max(1, Math.round(hours))} hours`);
  } catch {
    return [];
  }
}

/** Drop rows past the retention window. Called from the same boot prune as the rest. */
function pruneSearchLog({ days = 30 } = {}) {
  const db = getSqliteDb();
  if (!db) return 0;
  try {
    const res = db.prepare(
      "DELETE FROM search_call_log WHERE datetime(created_at) < datetime('now', ?)"
    ).run(`-${Math.max(1, Math.round(days))} days`);
    return res.changes;
  } catch {
    return 0;
  }
}

module.exports = { logSearchCall, recentSearchCalls, providerSummary, pruneSearchLog };
