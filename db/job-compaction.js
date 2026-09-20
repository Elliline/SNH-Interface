/**
 * COMPACTING A JOB'S TRANSCRIPT WHILE IT RUNS.
 *
 * The 9/17 job on Juno carried forty raw tool payloads forward into every
 * round: 32 fetched pages at ~4,100 chars each and 8 searches at ~5,800 — 180k
 * of the 199k chars in its transcript were tool results, most of them pages
 * the model had already read and taken what it needed from. That is what
 * filled the window. After a search or a fetch has been read, what the job
 * needs from it is what it LEARNED and WHERE IT CAME FROM — not the page.
 *
 * So a tool result older than the recent window is replaced in the transcript:
 *
 *   - a web_search result keeps every hit's title, url and date and a short
 *     snippet (the snippet IS the finding for a search);
 *   - a web_fetch result becomes a DIGEST — the findings relevant to the job,
 *     extracted by one model call per round of pages — with the page's title,
 *     url and date attached BY THE RUNNER from the raw result. The model
 *     writes the findings; it never gets to write the provenance, so a digest
 *     cannot lose its URL. When the digest call cannot run (engine busy, refused)
 *     the page is TRUNCATED instead, marked as such, and the report says so;
 *   - a memory_* result is kept unless it is large, then truncated;
 *   - older rounds compact harder: the assistant's own reasoning from those
 *     rounds is dropped and search hits are cut to the top few.
 *
 * TWO HARD REQUIREMENTS, both from the brief:
 *
 *   PROVENANCE SURVIVES. Every compacted entry carries `source: {title, url,
 *   date}` taken from the raw result (and cross-referenced against earlier
 *   search hits for a title and a date the fetch itself does not carry).
 *   `test-job-compaction.js` asserts no URL present before compaction is
 *   absent after it.
 *
 *   COMPACTION IS VISIBLE. `compact()` returns a report — rounds touched,
 *   entries, chars before and after, how many were digested vs truncated —
 *   and the runner writes it into the job's record and its card. A silent
 *   squeeze that loses something is worse than a loud death.
 *
 * Compacted content is JSON with `_compacted` set, inside the tool message's
 * string content — the engine sees text, the runner can tell a compacted
 * message from a raw one with no side table, and a checkpoint that is resumed
 * in a new process needs nothing restored.
 */

const MARK = '_compacted';

function parseJSON(s) {
  try { return JSON.parse(s); } catch { return null; }
}

/** Is this tool message already compacted? */
function isCompacted(msg) {
  if (!msg || msg.role !== 'tool' || typeof msg.content !== 'string') return false;
  const j = msg.content.startsWith('{') ? parseJSON(msg.content) : null;
  return !!(j && j[MARK]);
}

/** The first line of a fetched page, as its title. */
function titleOf(content, max = 120) {
  const first = String(content || '').split('\n').map(s => s.trim()).find(Boolean) || '';
  return first.length > max ? `${first.slice(0, max - 1)}…` : first;
}

/**
 * Every search hit seen so far, by URL — the one place a fetch's title and date
 * can be recovered from, since web_fetch returns neither.
 */
function buildSearchIndex(convo = []) {
  const idx = new Map();
  for (const m of convo) {
    if (!m || m.role !== 'tool' || m.name !== 'web_search') continue;
    const j = parseJSON(m.content);
    const hits = j && Array.isArray(j.results) ? j.results : [];
    for (const h of hits) {
      if (h && h.url && !idx.has(h.url)) idx.set(h.url, { title: h.title || null, date: h.publishedDate || h.date || null });
    }
  }
  return idx;
}

/** Which round each tool message belongs to: the count of tool-asking assistant turns before it. */
function roundsOf(convo = []) {
  const out = new Array(convo.length).fill(0);
  let round = 0;
  for (let i = 0; i < convo.length; i++) {
    const m = convo[i];
    if (m && m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) round++;
    out[i] = round;
  }
  return out;
}

/** A web_search result, trimmed to what a later round needs from it. */
function trimSearch(result, { snippetChars = 160, maxHits = 10 } = {}) {
  const hits = result && Array.isArray(result.results) ? result.results : [];
  const results = hits.slice(0, maxHits).map(h => ({
    title: h.title || null,
    url: h.url || null,
    date: h.publishedDate || h.date || null,
    snippet: h.snippet ? (h.snippet.length > snippetChars ? `${h.snippet.slice(0, snippetChars - 1)}…` : h.snippet) : null
  }));
  return { [MARK]: 'trim', results, hits_total: hits.length };
}

/** A web_fetch result with its findings replaced by whatever the caller extracted. */
function digestFetch(result, findings, source) {
  return { [MARK]: 'digest', source, findings: String(findings || '').trim() || '(nothing relevant on this page)' };
}

/** A web_fetch result cut to its head — the fallback when no digest could be made. */
function truncateFetch(result, source, { keepChars = 800 } = {}) {
  const text = String(result && result.content || '');
  return { [MARK]: 'truncate', source, head: text.slice(0, keepChars), note: `page text cut here — ${Math.max(0, text.length - keepChars)} chars dropped; only the head survives` };
}

/** Any other large result, cut. */
function truncateOther(content, { keepChars = 3000 } = {}) {
  return { [MARK]: 'truncate', head: String(content).slice(0, keepChars), note: `result cut here — ${Math.max(0, content.length - keepChars)} chars dropped` };
}

/** The source block a compacted fetch carries — from the raw result and the search index, never from the model. */
function sourceOf(result, searchIndex) {
  const url = result && result.url ? String(result.url) : null;
  const hit = url && searchIndex ? searchIndex.get(url) : null;
  return {
    title: (hit && hit.title) || titleOf(result && result.content) || null,
    url,
    date: (hit && hit.date) || null
  };
}

/**
 * The digest call: one model call per batch of pages, findings only.
 *
 * `callLLM` is injected so this module has no model dependency of its own and
 * a test can pin it. The prompt asks for JSON keyed by index; a page the model
 * skipped or the whole call failing leaves those pages to the truncation path.
 *
 * @returns {Promise<Map<number,string>>} index → findings
 */
async function digestPages(pages, { task, callLLM, maxTokens = 4000, thinkingTokens = 0, pageChars = 6000, batchChars = 24000 } = {}) {
  const out = new Map();
  if (typeof callLLM !== 'function' || !pages.length) return out;
  const batches = [];
  let cur = [], curChars = 0;
  for (const p of pages) {
    const chars = Math.min(pageChars, String(p.content || '').length);
    if (cur.length && curChars + chars > batchChars) { batches.push(cur); cur = []; curChars = 0; }
    cur.push(p); curChars += chars;
  }
  if (cur.length) batches.push(cur);

  const system =
    'You are compacting your own research notes in the middle of a background job. Nobody reads this but you, ' +
    'later in the same job. For each SOURCE below, write the findings that matter for THE JOB: every specific ' +
    'number, date, version, configuration, measurement or claim, and who reported it. Short bullets; copy exact ' +
    'figures and short phrases rather than paraphrasing them. Never invent. Never add anything that is not on ' +
    'the page. If a page has nothing relevant, write exactly: nothing relevant.\n' +
    'Return ONLY a JSON object: {"sources":[{"i":<number>,"findings":"- ...\\n- ..."}]}';
  for (const batch of batches) {
    const user =
      `THE JOB:\n${String(task || '').slice(0, 2500)}\n\n` +
      batch.map(p =>
        `=== SOURCE ${p.i} ===\nurl: ${p.url || '(unknown)'}\n${String(p.content || '').slice(0, pageChars)}\n`
      ).join('\n');
    try {
      // Room for the answer scales with the batch: the first live run lost 7
      // of 27 pages to truncation because a four-page batch was capped at
      // 1,500 tokens and the JSON was cut before its last entries.
      const res = await callLLM(system, user, { maxTokens: Math.max(600, Math.min(maxTokens, 700 * batch.length)), thinkingTokens });
      const text = String(res && res.content || '');
      const m = text.match(/\{[\s\S]*\}/);
      const parsed = m ? parseJSON(m[0]) : null;
      const list = parsed && Array.isArray(parsed.sources) ? parsed.sources : [];
      for (const s of list) {
        const i = Number(s && s.i);
        if (Number.isInteger(i) && typeof s.findings === 'string' && s.findings.trim()) out.set(i, s.findings.trim().slice(0, 4000));
      }
    } catch (err) {
      console.warn(`[JobCompaction] digest call failed for ${batch.length} page(s): ${err && err.message}`);
    }
  }
  return out;
}

/**
 * Compact a transcript in place of the caller's array.
 *
 * @param {Array} convo            the message array as the loop holds it
 * @param {Object} opts
 * @param {number} opts.round        the round about to run (1-based); results from rounds
 *                                   older than `round - recentRounds` are eligible
 * @param {number} [opts.recentRounds=2]  rounds left raw behind the current one
 * @param {number} [opts.hardAfter=6]     rounds behind the current one after which compaction is harder
 * @param {boolean} [opts.hard=false]     pressure mode: everything eligible, and harder everywhere
 * @param {string} [opts.task]            the job brief, for the digest prompt
 * @param {Function|null} [opts.callLLM]  the digest model call; null = truncate instead
 * @param {number} [opts.snippetChars=160]
 * @param {number} [opts.truncateChars=800]
 * @returns {Promise<{convo: Array, report: Object}>}
 */
async function compact(convo, opts = {}) {
  const round = Math.max(1, opts.round | 0);
  const recent = opts.hard ? 0 : Math.max(0, Number.isFinite(opts.recentRounds) ? opts.recentRounds : 2);
  const hardAfter = Math.max(1, Number.isFinite(opts.hardAfter) ? opts.hardAfter : 6);
  const snippetChars = Math.max(40, opts.snippetChars || 160);
  const truncateChars = Math.max(200, opts.truncateChars || 800);

  const rounds = roundsOf(convo);
  const searchIndex = buildSearchIndex(convo);
  const out = convo.map(m => m);
  const report = {
    round, mode: opts.hard ? 'pressure' : 'age',
    roundsCompacted: new Set(), entries: 0, charsBefore: 0, charsAfter: 0,
    digested: 0, trimmed: 0, truncated: 0, reasoningDropped: 0, digestFailed: 0
  };

  // Pass 1: what is eligible, and the pages that want a digest.
  const pages = [];
  const eligible = [];
  for (let i = 0; i < out.length; i++) {
    const m = out[i];
    const age = round - rounds[i];
    if (!m || age < recent + 1) continue;   // in the recent window, or the current round
    if (m.role === 'assistant' && m.reasoning && (opts.hard || age > hardAfter)) {
      const copy = { ...m }; delete copy.reasoning;
      out[i] = copy; report.reasoningDropped++; report.roundsCompacted.add(rounds[i]);
      continue;
    }
    if (m.role !== 'tool' || typeof m.content !== 'string') continue;
    if (isCompacted(m)) {
      // Harder, second pass on an old trim: fewer hits.
      const j = parseJSON(m.content);
      if (j && j[MARK] === 'trim' && (opts.hard || age > hardAfter) && Array.isArray(j.results) && j.results.length > 5) {
        const shorter = { ...j, results: j.results.slice(0, 5) };
        const before = m.content.length;
        out[i] = { ...m, content: JSON.stringify(shorter) };
        report.charsBefore += before; report.charsAfter += out[i].content.length; report.entries++; report.trimmed++;
        report.roundsCompacted.add(rounds[i]);
      }
      continue;
    }
    const j = parseJSON(m.content);
    if (!j || j.error) continue;   // an error result is already small and says something
    eligible.push(i);
    if (m.name === 'web_fetch' && typeof j.content === 'string' && j.content.length > truncateChars) {
      pages.push({ i, url: j.url, content: j.content });
    }
  }
  if (!eligible.length) {
    report.roundsCompacted = [...report.roundsCompacted].sort((a, b) => a - b);
    return { convo: out, report };
  }

  // Pass 2: the digests, one call per batch of pages.
  const digests = pages.length && typeof opts.callLLM === 'function'
    ? await digestPages(pages, { task: opts.task, callLLM: opts.callLLM, thinkingTokens: opts.digestThinkingTokens ?? 0 })
    : new Map();

  // Pass 3: rewrite.
  for (const i of eligible) {
    const m = out[i];
    const j = parseJSON(m.content);
    const before = m.content.length;
    let next = null;
    if (m.name === 'web_search') {
      next = trimSearch(j, { snippetChars, maxHits: (opts.hard || round - rounds[i] > hardAfter) ? 5 : 10 });
      report.trimmed++;
    } else if (m.name === 'web_fetch') {
      const source = sourceOf(j, searchIndex);
      if (digests.has(i)) { next = digestFetch(j, digests.get(i), source); report.digested++; }
      else if (typeof j.content === 'string' && j.content.length > truncateChars) {
        next = truncateFetch(j, source, { keepChars: truncateChars }); report.truncated++;
        if (pages.some(p => p.i === i)) report.digestFailed++;
      } else continue;   // a short page: nothing to gain
    } else if (before > 3000) {
      next = truncateOther(m.content); report.truncated++;
    } else continue;
    const content = JSON.stringify(next);
    if (content.length >= before) continue;   // never make it bigger
    out[i] = { ...m, content };
    report.entries++;
    report.charsBefore += before;
    report.charsAfter += content.length;
    report.roundsCompacted.add(rounds[i]);
  }
  report.roundsCompacted = [...report.roundsCompacted].sort((a, b) => a - b);
  return { convo: out, report };
}

/** Every URL that appears anywhere in the transcript — for the provenance assertion, and the sources list. */
function urlsIn(convo = []) {
  const urls = new Set();
  const re = /https?:\/\/[^\s"'<>\\)\]]+/g;
  for (const m of convo) {
    if (!m || m.role !== 'tool' || typeof m.content !== 'string') continue;
    for (const u of m.content.match(re) || []) urls.add(u.replace(/[.,;:]+$/, ''));
  }
  return urls;
}

/**
 * The sources a transcript drew on, in order of first appearance, with what
 * is known about each — for the findings document's sources list, so a phase
 * writeup that forgot a URL still has it on the record.
 */
function sourcesIn(convo = []) {
  const seen = new Map();
  const idx = buildSearchIndex(convo);
  for (const m of convo) {
    if (!m || m.role !== 'tool' || typeof m.content !== 'string') continue;
    const j = parseJSON(m.content);
    if (!j) continue;
    if (m.name === 'web_fetch' && (j.url || (j.source && j.source.url))) {
      // A page that was READ outranks the same URL merely found in a search:
      // the search hit's title and date are kept, the `how` becomes read.
      const url = j.url || j.source.url;
      const src = j[MARK] ? (j.source || { url }) : sourceOf(j, idx);
      const prev = seen.get(url) || {};
      seen.set(url, { ...prev, ...src, title: src.title || prev.title || null, date: src.date || prev.date || null, url, how: 'read' });
    } else if (m.name === 'web_search' && Array.isArray(j.results)) {
      for (const h of j.results) {
        if (h && h.url && !seen.has(h.url)) seen.set(h.url, { title: h.title || null, url: h.url, date: h.publishedDate || h.date || null, how: 'found' });
      }
    }
  }
  return [...seen.values()];
}

/** Sum two reports, for the job-wide total. */
function addReport(total, r) {
  const t = total || { rounds: new Set(), entries: 0, charsBefore: 0, charsAfter: 0, digested: 0, trimmed: 0, truncated: 0, reasoningDropped: 0, digestFailed: 0, passes: 0, pressurePasses: 0 };
  if (!r) return t;
  for (const n of r.roundsCompacted || []) t.rounds.add(n);
  for (const k of ['entries', 'charsBefore', 'charsAfter', 'digested', 'trimmed', 'truncated', 'reasoningDropped', 'digestFailed']) t[k] += r[k] || 0;
  if (r.entries || r.reasoningDropped) t.passes++;
  if (r.mode === 'pressure' && (r.entries || r.reasoningDropped)) t.pressurePasses++;
  return t;
}

/** The total as plain data, for a row and a card. */
function reportSummary(total) {
  if (!total) return null;
  return {
    roundsCompacted: total.rounds instanceof Set ? total.rounds.size : (total.roundsCompacted || 0),
    entries: total.entries, charsBefore: total.charsBefore, charsAfter: total.charsAfter,
    droppedChars: Math.max(0, total.charsBefore - total.charsAfter),
    digested: total.digested, trimmed: total.trimmed, truncated: total.truncated,
    reasoningDropped: total.reasoningDropped, digestFailed: total.digestFailed,
    passes: total.passes, pressurePasses: total.pressurePasses
  };
}

module.exports = {
  MARK, isCompacted, compact, digestPages, trimSearch, digestFetch, truncateFetch, sourceOf,
  buildSearchIndex, roundsOf, urlsIn, sourcesIn, addReport, reportSummary, titleOf
};
