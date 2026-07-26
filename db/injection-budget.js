/**
 * Injection budgeting: keep the per-chat system context small so prefill stays
 * fast. Whole daily logs + long-term memory used to be injected wholesale
 * (~17–27k tokens → 60–90s time-to-first-token). These helpers cap each source
 * to a configured token budget, and for the daily log specifically, inject the
 * most-recent entries verbatim plus a brief digest of the remainder + yesterday.
 *
 * Token counts are estimated at ~4 chars/token — good enough for budgeting.
 */

function estTokens(str) {
  return Math.ceil((str || '').length / 4);
}

/**
 * Split a daily-log markdown file into its H1 header and its time-stamped entry
 * blocks (newest first, matching how writers prepend). Each block starts with a
 * "### HH:MM" line or a "## Heartbeat Report" line.
 * @returns {{header: string, blocks: string[]}}
 */
function splitDailyBlocks(text) {
  if (!text) return { header: '', blocks: [] };
  // Peel the leading "# ... \n\n" H1 header if present.
  const headerMatch = text.match(/^(# [^\n]*\r?\n(?:\r?\n)?)/);
  const header = headerMatch ? headerMatch[1] : '';
  const body = text.slice(header.length);
  // Split before each level-2 or level-3 heading, keeping the heading with its block.
  const blocks = body
    .split(/(?=^#{2,3} )/m)
    .map(b => b.trim())
    .filter(Boolean);
  return { header, blocks };
}

/**
 * Pull the human-readable headline out of an entry block for the digest:
 * the first "- ..." bullet, or the heading line, trimmed to `maxChars`.
 */
function entryHeadline(block, maxChars = 120) {
  const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
  // Prefer the first bullet; skip the "### HH:MM" / "## Heartbeat" heading line.
  let line = lines.find(l => l.startsWith('- ')) || lines[0] || '';
  line = line.replace(/^[-#]+\s*/, '').trim();
  if (line.length > maxChars) line = line.slice(0, maxChars - 1).trimEnd() + '…';
  return line;
}

/**
 * Budget the daily logs. Returns the recent slice of today's log (verbatim, up
 * to `dailyTodayTokens`) and a brief digest of everything older (rest of today +
 * yesterday), capped at `dailySummaryTokens`.
 *
 * @returns {{recent: string, summary: string, stats: object}}
 */
function budgetDailyLogs(todayText, yesterdayText, opts = {}) {
  const dailyTodayTokens = opts.dailyTodayTokens ?? 1500;
  const dailySummaryTokens = opts.dailySummaryTokens ?? 400;

  const { blocks: todayBlocks } = splitDailyBlocks(todayText);

  // Accumulate newest-first today blocks up to the verbatim budget.
  const keptBlocks = [];
  let used = 0;
  let i = 0;
  for (; i < todayBlocks.length; i++) {
    const t = estTokens(todayBlocks[i]) + 2; // +2 for the joining blank line
    if (keptBlocks.length > 0 && used + t > dailyTodayTokens) break;
    keptBlocks.push(todayBlocks[i]);
    used += t;
    if (used >= dailyTodayTokens) { i++; break; }
  }
  const overflowTodayBlocks = todayBlocks.slice(i);
  const recent = keptBlocks.join('\n\n');

  // Build a brief digest of the remainder (older today) + all of yesterday.
  const { blocks: yesterdayBlocks } = splitDailyBlocks(yesterdayText);
  const digestSources = [
    ...overflowTodayBlocks.map(b => ['today', b]),
    ...yesterdayBlocks.map(b => ['yesterday', b]),
  ];

  const headlineLines = [];
  let digestTokens = 0;
  let omitted = 0;
  for (const [, block] of digestSources) {
    const headline = entryHeadline(block);
    if (!headline) continue;
    const line = `- ${headline}`;
    const t = estTokens(line) + 1;
    if (digestTokens + t > dailySummaryTokens) { omitted++; continue; }
    headlineLines.push(line);
    digestTokens += t;
  }
  let summary = headlineLines.join('\n');
  if (omitted > 0) summary += `\n- …and ${omitted} more earlier entr${omitted === 1 ? 'y' : 'ies'} (see Thinking tab / daily logs).`;

  return {
    recent,
    summary,
    stats: {
      todayBlocksTotal: todayBlocks.length,
      todayBlocksKept: keptBlocks.length,
      recentTokens: estTokens(recent),
      summaryTokens: estTokens(summary),
      digestOmitted: omitted,
    },
  };
}

/**
 * Cap arbitrary injected text to a token budget, truncating on a line boundary
 * and appending a marker. Returns the text unchanged if already within budget.
 */
function budgetText(text, budgetTokens, label = 'content') {
  if (!text) return { text: '', tokens: 0, truncated: false };
  if (estTokens(text) <= budgetTokens) {
    return { text, tokens: estTokens(text), truncated: false };
  }
  const maxChars = budgetTokens * 4;
  let slice = text.slice(0, maxChars);
  const lastNl = slice.lastIndexOf('\n');
  if (lastNl > maxChars * 0.5) slice = slice.slice(0, lastNl);
  slice = slice.trimEnd() + `\n…(${label} truncated to fit context budget)`;
  return { text: slice, tokens: estTokens(slice), truncated: true };
}

/**
 * Framing for the injected memory block.
 *
 * Every memory source above is a RETRIEVAL — top-k by relevance, then capped to
 * a token budget by these helpers. Without saying so, the block reads as the
 * entity's COMPLETE memory, and it answers "I don't have any information about
 * that" for anything that missed the cut. That is a confident false negative:
 * the fact is in memory, it just wasn't retrieved this turn. It affects every
 * conversation, not only tool-calling ones.
 *
 * There are two variants because the fix for each case actively breaks the
 * other. Measured on the 2026-07-26 probe (20 runs/case, ~8k injected memory,
 * production sampling):
 *
 *   - WITH a memory-search tool available, the only thing that makes the model
 *     actually search is a bare imperative pointing at the tools. Baseline was
 *     0/20 tool selection; WITH_TOOLS scores 20/20, false positives 0/20.
 *   - Adding ANY "otherwise, say it wasn't retrieved" fallback to that string
 *     collapses it back to 0/20 — the model takes the cheaper phrasing branch
 *     instead of searching. Three separate phrasings of the fallback were tried
 *     (subordinated, conditional, and as a pure phrasing rule); all three
 *     scored 0/20. The escape hatch always wins.
 *   - WITHOUT tools there is nothing to search, so the imperative is inert and
 *     the model still asserts false absence. NO_TOOLS drops the imperative and
 *     keeps only the phrasing correction: 5/5 honest hedging on an absent fact,
 *     while still answering a present fact directly 4/4 (it does not over-hedge).
 *
 * So the caller picks by whether the model is being handed tools this turn.
 * Do not merge these back into one string — that regression is measured, not
 * hypothetical.
 */
const MEMORY_EXCERPT_FRAMING_WITH_TOOLS =
  'The memory below is a PARTIAL excerpt selected by relevance, not everything you ' +
  'remember. Anything not shown here simply was not retrieved this turn. Before ' +
  'telling the user you have no memory of something, search your memory using the ' +
  'tools available to you.';

const MEMORY_EXCERPT_FRAMING_NO_TOOLS =
  'The memory below is a PARTIAL excerpt selected by relevance, not everything you ' +
  'remember. Anything not shown here simply was not retrieved this turn — that is ' +
  'not evidence it is absent from your memory. Do not tell the user you have no ' +
  'memory of something on that basis; say it is not in what you retrieved this ' +
  'turn, and offer to look properly.';

/**
 * Pick the memory framing for this turn.
 * @param {boolean} toolsAvailable - true when the model is being handed a tool
 *   schema this turn (i.e. it actually has something to search with).
 */
function memoryFraming(toolsAvailable) {
  return toolsAvailable ? MEMORY_EXCERPT_FRAMING_WITH_TOOLS : MEMORY_EXCERPT_FRAMING_NO_TOOLS;
}

module.exports = {
  estTokens, splitDailyBlocks, entryHeadline, budgetDailyLogs, budgetText,
  MEMORY_EXCERPT_FRAMING_WITH_TOOLS, MEMORY_EXCERPT_FRAMING_NO_TOOLS, memoryFraming
};
