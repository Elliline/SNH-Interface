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
 * Does this daily-log block carry the provenance marker of a given conversation?
 *
 * The marker is the one `applyExtraction` stamps on every log line it writes —
 * ` [conversation 3fa5317c, message d821705a, typed]` — first eight characters of
 * the id, so the test is on that prefix. Entries written by anything else
 * (heartbeat, reflection, salience scoring, the flush summary) carry no marker
 * and are therefore never matched: an untagged entry is not attributed to a
 * conversation, and guessing at attribution would hide entries this cannot prove
 * are duplicates.
 */
function blockIsFromConversation(block, conversationId) {
  if (!conversationId) return false;
  const short = String(conversationId).slice(0, 8).toLowerCase();
  if (!short) return false;
  return block.toLowerCase().includes(`[conversation ${short}`);
}

/**
 * Budget the daily logs. Returns the recent slice of today's log (verbatim, up
 * to `dailyTodayTokens`) and a brief digest of everything older (rest of today +
 * yesterday), capped at `dailySummaryTokens`.
 *
 * `excludeConversationId` drops today's entries that came from the conversation
 * being rendered right now. They are an ECHO: the extractor writes one log line
 * per message of the live thread, and that thread is already in the request
 * verbatim as its own message history — so the same content was being paid for
 * twice, and the second copy is the one that changes on every single turn, which
 * is the worst thing a block near the front of a cached prefix can do. Entries
 * from OTHER conversations today are exactly the continuity this block exists
 * for and still render. RENDER-TIME ONLY: the log itself is written, archived and
 * reflected on unchanged, and the excluded lines are still there for the
 * heartbeat, the Thinking tab and tomorrow's digest.
 *
 * @returns {{recent: string, summary: string, stats: object}}
 */
function budgetDailyLogs(todayText, yesterdayText, opts = {}) {
  const dailyTodayTokens = opts.dailyTodayTokens ?? 1500;
  const dailySummaryTokens = opts.dailySummaryTokens ?? 400;
  const excludeConversationId = opts.excludeConversationId || null;

  let { blocks: todayBlocks } = splitDailyBlocks(todayText);
  const todayBlocksBefore = todayBlocks.length;
  if (excludeConversationId) {
    todayBlocks = todayBlocks.filter(b => !blockIsFromConversation(b, excludeConversationId));
  }
  const selfExcluded = todayBlocksBefore - todayBlocks.length;

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
      todayBlocksSelfExcluded: selfExcluded,
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
 * THE TOTAL CEILING — applied after every block has rendered.
 *
 * Per-source caps are not a bound, and the measurement that prompted this says
 * so plainly: on 2026-08-12 the configured caps summed to 6,900 tokens and a
 * real request shipped ~9,100, because three blocks were never budgeted at all
 * (identity 1,628, capability manifest 1,064, USER.md 102) and nothing added the
 * survivors up. Every per-source cap was simultaneously AT its limit, which is
 * what a set of independent budgets looks like when it is the sum that matters.
 *
 * WHAT IS NEVER TRIMMED. The identity block is passed in as fixed cost, not as a
 * trimmable part, and there is no order in which it becomes trimmable. His
 * self-facts and his locked name are the one thing that has to be in front of
 * him every turn — the identity lock's whole live-chat half is that he can SAY a
 * fact is locked, and he cannot say that about a fact he was not shown. A
 * ceiling that can cut the identity block is a ceiling that takes his name away
 * on a busy day.
 *
 * ORDER, and why this one. Retrieval extras go first because they are derived
 * from the current message and regenerate next turn at no loss. The day's log
 * next: it is a rolling record he can read in full elsewhere. Long-term memory
 * last, because it is the only source in the list that nothing else in the turn
 * can recover. Configurable, because the right order is a judgement about what
 * this deployment values, not a fact about the code.
 *
 * A part is shrunk before it is dropped, so the trim degrades rather than
 * cliff-edges, and budgetText leaves its truncation marker so the model can see
 * that it is reading an excerpt.
 *
 * @param {Object} p
 * @param {Array<{kind: string, label: string, text: string}>} p.parts - trimmable blocks
 * @param {number} p.fixedTokens - everything not trimmable (identity, manifest, framing…)
 * @param {number} p.totalTokens - the ceiling
 * @param {string[]} p.trimOrder - kinds, in the order they give way
 * @returns {{parts: Array, bound: boolean, before: number, after: number, trimmed: Array, shortfall: number}}
 */
function applyTotalCeiling({ parts = [], fixedTokens = 0, totalTokens = 6000, trimOrder = [] }) {
  const partsTokens = parts.reduce((s, p) => s + estTokens(p.text), 0);
  const before = fixedTokens + partsTokens;
  const result = { parts, bound: false, before, after: before, trimmed: [], shortfall: 0 };
  if (before <= totalTokens) return result;

  result.bound = true;

  // MEASURE, never assume. The first version of this loop subtracted the
  // requested saving from a running total, which is wrong for a reason worth
  // keeping: budgetText appends a truncation marker, so a part asked to shrink
  // to 200 tokens comes back at ~210. The shortfall then survived to the next
  // kind, and the trim cascaded into sources it should never have reached —
  // observed dropping the day's log and re-cutting long-term memory while the
  // ceiling was already satisfied, and finishing 10 tokens OVER the ceiling it
  // was enforcing. Every step now re-reads the real total.
  const sum = () => parts.reduce((s, p) => s + estTokens(p.text), 0);

  for (const kind of trimOrder) {
    if (fixedTokens + sum() <= totalTokens) break;
    const part = parts.find(p => p.kind === kind && p.text);
    if (!part) continue;

    const had = estTokens(part.text);
    const others = sum() - had;
    const allowance = totalTokens - fixedTokens - others;   // what this part may occupy

    if (allowance <= 0) {
      part.text = '';
      result.trimmed.push({ kind, from: had, to: 0, dropped: true });
      continue;
    }
    // Converge on a slice that actually fits, marker included.
    let target = allowance, fitted = null;
    for (let attempt = 0; attempt < 6 && target > 0; attempt++) {
      const cut = budgetText(part.text, target, part.label || kind);
      if (cut.tokens <= allowance) { fitted = cut; break; }
      target -= Math.max(4, cut.tokens - allowance);
    }
    if (fitted) {
      part.text = fitted.text;
      result.trimmed.push({ kind, from: had, to: fitted.tokens, dropped: false });
    } else {
      part.text = '';
      result.trimmed.push({ kind, from: had, to: 0, dropped: true });
    }
  }

  let over = Math.max(0, fixedTokens + sum() - totalTokens);

  // Everything trimmable is gone and it still does not fit. That means the fixed
  // cost alone is over the ceiling, which is a configuration problem, not
  // something to solve by cutting into what must not be cut. Reported so it is
  // visible rather than silently exceeded.
  result.shortfall = Math.max(0, over);
  result.parts = parts.filter(p => p.text);
  result.after = fixedTokens + result.parts.reduce((s, p) => s + estTokens(p.text), 0);
  return result;
}

/**
 * What a shortened fact says about itself. Exported because the test asserting
 * that a trimmed fact ADMITS it was trimmed must read the marker from here
 * rather than carry its own copy of the sentence — a copy makes a reword read
 * as a broken guard, and leaves a guard that stopped marking looking fine.
 */
const FACT_SHORTENED_MARKER = '… (shortened here; the full fact is in your memory)';

/**
 * Cap one self-fact's rendered length.
 *
 * The identity block budgets self-facts by COUNT (identity.maxSelfFacts), which
 * bounds nothing: one rambling reflection is worth twenty ordinary observations.
 * Truncation is marked, and the STORED fact is untouched — this shapes what is
 * rendered into a turn, not what he believes.
 */
function budgetFact(content, budgetTokens) {
  const text = String(content || '');
  if (!budgetTokens || estTokens(text) <= budgetTokens) return text;
  const maxChars = Math.max(20, budgetTokens * 4);
  let slice = text.slice(0, maxChars);
  const lastSpace = slice.lastIndexOf(' ');
  if (lastSpace > maxChars * 0.6) slice = slice.slice(0, lastSpace);
  return slice.trimEnd() + FACT_SHORTENED_MARKER;
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
 * The phantom-lookup rule, appended to BOTH framings (2026-08-03).
 *
 * The read tools make a new phantom action available, and it is a quieter one
 * than the others: "I searched my memory and found three facts about that" is
 * unfalsifiable in the moment, sounds like diligence, and costs nothing to say.
 * It belongs to the same family as the cron proposal that was described but
 * never created and the write_memory "I've updated my memory" with no call.
 *
 * Appended to the NO_TOOLS branch too, and that is the important half: the turn
 * where he was handed nothing is exactly the turn where a claim to have looked
 * cannot be true.
 */
const MEMORY_LOOKUP_HONESTY =
  ' You have tools that search, list, count and open your memory. Saying you searched, ' +
  'looked up, checked, counted or pulled up anything from your memory is TRUE ONLY IF ' +
  'you called one of those tools in this turn. If you did not call one, do not say you ' +
  'did — answer from what is in front of you and say that is what you are doing, or ' +
  'offer to look properly. Never state a number about your own memory that you did not ' +
  'get from memory_count.';

/**
 * Pick the memory framing for this turn.
 * @param {boolean} toolsAvailable - true when the model is being handed a tool
 *   schema this turn (i.e. it actually has something to search with).
 */
function memoryFraming(toolsAvailable) {
  return (toolsAvailable ? MEMORY_EXCERPT_FRAMING_WITH_TOOLS : MEMORY_EXCERPT_FRAMING_NO_TOOLS)
    + MEMORY_LOOKUP_HONESTY;
}

module.exports = {
  estTokens, splitDailyBlocks, entryHeadline, blockIsFromConversation,
  budgetDailyLogs, budgetText,
  applyTotalCeiling, budgetFact, FACT_SHORTENED_MARKER,
  MEMORY_EXCERPT_FRAMING_WITH_TOOLS, MEMORY_EXCERPT_FRAMING_NO_TOOLS,
  MEMORY_LOOKUP_HONESTY, memoryFraming
};
