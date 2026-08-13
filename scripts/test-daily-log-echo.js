#!/usr/bin/env node
/**
 * The day's log does not read the live thread back to him.
 *
 * WHAT THIS IS FOR. `applyExtraction` writes a log line per message of the
 * conversation it just processed, stamped with that conversation's id. Injected
 * into the SAME conversation, those lines are an echo: the turns they summarise
 * are already in the request verbatim as message history. Two costs, one of them
 * not obvious — the duplicate tokens, and the fact that this block then changed
 * on every single turn while sitting near the front of a cached prefix, so
 * everything after it was re-prefilled each time.
 *
 * The rules on trial here are the ones that make the exclusion safe:
 *
 *   - it drops the ACTIVE conversation's entries and nothing else. Another
 *     conversation from today is the continuity this block exists for; hiding it
 *     would be a memory loss dressed up as an optimisation.
 *   - an UNTAGGED entry always renders. The heartbeat, reflection, salience
 *     scoring and — the one that matters — the memory-flush summary carry no
 *     conversation marker, and the flush summary is what stands in for the early
 *     turns once the thread is compacted.
 *   - it is render-time only: the file on disk is untouched, so the excluded
 *     lines are still there for the heartbeat, the Thinking tab, tomorrow's
 *     digest and the archiver.
 *
 * Runs against a throwaway SNH_DATA_DIR. No model, no network, no live corpus.
 *
 * Usage: node scripts/test-daily-log-echo.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-daily-echo-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const injectionBudget = require(path.join(ROOT, 'db/injection-budget'));
const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const ACTIVE = '3fa5317c-9d21-4a55-b0e2-7c1e9f0a1b2d';
const OTHER = '8b7c1d02-11aa-4c3e-9f80-2d4e6a8b0c11';

// A day's log the way it is actually written: newest first, "### HH:MM" blocks,
// extraction lines carrying the conversation marker, everything else bare.
const todayLog = [
  '# Daily Log - 2026-08-13',
  '',
  '### 14:20',
  `- User asked about the staging cutover schedule [conversation ${ACTIVE.slice(0, 8)}, message aaaa1111, typed]`,
  '',
  '### 14:05',
  '- Scored fact salience 6/10: "User prefers local models." — a stable technical preference.',
  '',
  '### 13:40',
  `- User said the vacation is now the second week of September [conversation ${OTHER.slice(0, 8)}, message bbbb2222, typed]`,
  '',
  '### 12:00',
  '- Context was compacted; key points from earlier in the conversation: the SQ-4417 blocker, the decision to keep VACUUM INTO, and Ellie asking for the report by Friday.',
  '',
  '### 09:00',
  '- My scheduled job ran and I reported the overnight memory maintenance to Ellie.',
  ''
].join('\n');

const yesterdayLog = [
  '# Daily Log - 2026-08-12',
  '',
  '### 16:30',
  `- User asked me to re-run the corrector on staging [conversation ${ACTIVE.slice(0, 8)}, message cccc3333, typed]`,
  ''
].join('\n');

console.log('\n1. The active conversation\'s own entries do not come back to it');
{
  const r = injectionBudget.budgetDailyLogs(todayLog, '', { excludeConversationId: ACTIVE });
  check('its own entry is gone from the verbatim slice',
    !r.recent.includes('staging cutover schedule'), r.recent);
  check('…and gone from the digest too, not just pushed into it',
    !r.summary.includes('staging cutover schedule'), r.summary);
  check('…and it is counted, so the log line can say so',
    r.stats.todayBlocksSelfExcluded === 1, `${r.stats.todayBlocksSelfExcluded}`);
}

console.log('\n2. Everything else from today still renders');
{
  const r = injectionBudget.budgetDailyLogs(todayLog, '', { excludeConversationId: ACTIVE });
  check('another conversation from today is kept — that is the continuity this block is for',
    r.recent.includes('second week of September'));
  check('an untagged entry (salience scoring) is kept',
    r.recent.includes('Scored fact salience 6/10'));
  check('an untagged entry (the scheduled job) is kept',
    r.recent.includes('scheduled job'));
  check('THE FLUSH SUMMARY IS KEPT — it carries no marker, and it is what stands in ' +
    'for the early turns once the thread is compacted',
    r.recent.includes('SQ-4417 blocker'), r.recent);
}

console.log('\n3. Only today is filtered, and only when asked');
{
  const r = injectionBudget.budgetDailyLogs(todayLog, yesterdayLog, { excludeConversationId: ACTIVE });
  check('yesterday is not filtered — a thread that crossed midnight keeps its history',
    r.summary.includes('corrector on staging'), r.summary);

  const off = injectionBudget.budgetDailyLogs(todayLog, '', {});
  check('with no id passed, nothing is excluded (the config flag can turn this off)',
    off.recent.includes('staging cutover schedule') && off.stats.todayBlocksSelfExcluded === 0);

  const otherThread = injectionBudget.budgetDailyLogs(todayLog, '', { excludeConversationId: OTHER });
  check('a DIFFERENT conversation sees the first entry and loses its own',
    otherThread.recent.includes('staging cutover schedule') &&
    !otherThread.recent.includes('second week of September'));
}

console.log('\n4. The budget is spent on what is left, not on the hole');
{
  // Fifty echo entries in front of one entry that matters: unfiltered, the
  // verbatim budget is eaten by the echo and the real entry never renders.
  const noise = [];
  for (let i = 0; i < 50; i++) {
    noise.push(`### 15:${String(i).padStart(2, '0')}`);
    noise.push(`- User said something in the live thread, turn ${i}, and it was long enough to matter for the budget by a fair margin [conversation ${ACTIVE.slice(0, 8)}, message dddd${String(i).padStart(4, '0')}, typed]`);
    noise.push('');
  }
  const busy = ['# Daily Log - 2026-08-13', '', ...noise,
    '### 08:00',
    '- Ellie asked me to have the staging report ready before the Friday call.',
    ''].join('\n');

  const before = injectionBudget.budgetDailyLogs(busy, '', { dailyTodayTokens: 400 });
  const after = injectionBudget.budgetDailyLogs(busy, '', { dailyTodayTokens: 400, excludeConversationId: ACTIVE });
  check('unfiltered, the echo crowds the real entry out of the verbatim slice',
    !before.recent.includes('Friday call'));
  check('filtered, the real entry renders',
    after.recent.includes('Friday call'), after.recent);
  check('…and the block is far smaller',
    after.stats.recentTokens < before.stats.recentTokens / 3,
    `${after.stats.recentTokens} vs ${before.stats.recentTokens} tok`);
  check('…having excluded all fifty', after.stats.todayBlocksSelfExcluded === 50,
    `${after.stats.todayBlocksSelfExcluded}`);
}

console.log('\n5. Render-time only: the file on disk is unchanged');
{
  const dailyDir = path.join(TMP, 'memory', 'daily');
  fs.mkdirSync(dailyDir, { recursive: true });
  factExtractor.appendToDailyLog(
    `User asked about the staging cutover schedule [conversation ${ACTIVE.slice(0, 8)}, message aaaa1111, typed]`,
    dailyDir);
  const file = fs.readdirSync(dailyDir).find(f => f.endsWith('.md'));
  const written = fs.readFileSync(path.join(dailyDir, file), 'utf8');

  const r = injectionBudget.budgetDailyLogs(written, '', { excludeConversationId: ACTIVE });
  check('the entry is written to disk exactly as before',
    written.includes('staging cutover schedule') && written.includes(`[conversation ${ACTIVE.slice(0, 8)}`));
  check('…and it is only the RENDER that leaves it out',
    r.recent === '' && r.stats.todayBlocksSelfExcluded === 1);
  const reread = fs.readFileSync(path.join(dailyDir, file), 'utf8');
  check('…the render did not touch the file', reread === written);
}

console.log('\n6. The marker test is exact');
{
  const mk = (text) => `### 10:00\n- ${text}`;
  check('a full id matches on its first eight characters',
    injectionBudget.blockIsFromConversation(
      mk(`x [conversation ${ACTIVE.slice(0, 8)}, message aaaa1111, typed]`), ACTIVE));
  check('a different conversation does not match',
    !injectionBudget.blockIsFromConversation(
      mk(`x [conversation ${OTHER.slice(0, 8)}, message aaaa1111, typed]`), ACTIVE));
  check('the id appearing as ordinary prose is not a marker',
    !injectionBudget.blockIsFromConversation(
      mk(`Ellie mentioned conversation ${ACTIVE.slice(0, 8)} while debugging.`), ACTIVE));
  check('no id means no exclusion',
    !injectionBudget.blockIsFromConversation(mk('x [conversation 3fa5317c, message a, typed]'), null));
}

const bar = '='.repeat(74);
console.log(`\n${bar}`);
console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
console.log(`${bar}\n`);
process.exit(fail === 0 ? 0 : 1);
