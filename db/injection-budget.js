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

module.exports = { estTokens, splitDailyBlocks, entryHeadline, budgetDailyLogs, budgetText };
