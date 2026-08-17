/**
 * Reading the day's log back.
 *
 * Events already route to the daily log correctly — "User let the dogs out",
 * "User found the philosophical discussions can get too deep for them
 * sometimes". Nothing ever read them back, so the half of memory that holds what
 * HAPPENED was write-only while the fact store was doing all the remembering.
 * This is the read side, and it is deliberately dumb: it finds and parses
 * entries and makes no decision about them. The judgement lives in
 * db/initiative-engine.js, where it can be a model call.
 *
 * That split is the same one the extraction pipeline keeps between
 * planExtraction and applyExtraction, and for the same reason: a deterministic
 * half can be tested against the real corpus without a model in the loop.
 *
 * ─── WHAT COUNTS AS AN EVENT ───────────────────────────────────────────────
 *
 * The daily log is not only events. It also carries salience scores,
 * supersessions, reflections, follow-up decisions, self-observation limits and
 * refusals — the cognitive bookkeeping that makes the log worth keeping. Handing
 * those to a follow-up judge would have it "following up" on its own filing.
 *
 * The discriminator is PROVENANCE, not a list of prefixes. Intake stamps every
 * line it writes with `[conversation abcd1234, message ef567890, typed]`
 * (applyExtraction's `src`); background passes — reflection, prioritiser,
 * scheduler, audit — write no provenance because they came from no conversation.
 * So requiring the stamp removes the entire background half by construction,
 * rather than by an exclusion list that has to be maintained forever. Measured
 * over the last six days of real logs, that single rule leaves only two
 * machine-written shapes behind, and those two are named below.
 *
 * This matters because this codebase has already learned that enumerating the
 * ways a system can describe itself is a losing game (see
 * db/extraction-rules.js). A positive structural signal that removes ~13 shapes
 * at once beats a negative list that must anticipate the fourteenth.
 *
 * Nothing here writes. It reads files and returns objects.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

/**
 * Machine-written lines that DO carry intake provenance, because they are
 * emitted by applyExtraction alongside the events. Verified against six days of
 * real logs: these are the only two provenance-carrying shapes that are not
 * events. Each is anchored at the start of the entry so an event that merely
 * mentions one of these words is unaffected.
 *
 * If a third is ever added to applyExtraction it belongs here — that is a
 * two-line change in one file, which is why the list is allowed to exist at all.
 */
const MACHINE_PREFIXES = [
  /^Superseded fact:/i,          // supersession bookkeeping
  /^Already knew this,/i,        // repeat folded into an existing fact
  /^Did not record ["“]/i,       // intake refusal (extraction-rules)
  /^I did not change a locked fact:/i, // identity-lock refusal
];

/** The stamp intake puts on everything it writes. */
const PROVENANCE_RE = /\[conversation ([0-9a-f]{6,}),?(?:\s*message ([0-9a-f]{6,}),?)?\s*([a-z]+)?\]\s*$/i;

/** `### HH:MM` section headers inside a daily file. */
const TIME_HEADER_RE = /^###\s+(\d{1,2}:\d{2})\s*$/;

/** Local YYYY-MM-DD, N days back from `from`. */
function dateStampsBack(days, from = new Date()) {
  const out = [];
  for (let i = 0; i < days; i++) {
    const d = new Date(from.getFullYear(), from.getMonth(), from.getDate() - i);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    out.push(`${y}-${m}-${day}`);
  }
  return out;
}

/**
 * A stable identity for one log entry, so the same entry can never raise a
 * second follow-up. Derived from the content rather than a row id because the
 * daily log is a file, not a table — there is no id to borrow. Date and time are
 * included so a genuinely repeated event on a different day is a different
 * entry, which it is.
 */
function entryId(date, time, text) {
  const h = crypto.createHash('sha1').update(`${date} ${time} ${text}`).digest('hex').slice(0, 12);
  return `daily-log:${date}:${h}`;
}

/**
 * Parse one daily-log file's text into event entries.
 * @param {string} content - the file's contents
 * @param {string} date - YYYY-MM-DD this file represents
 * @returns {Array<{id,date,time,text,conversationId,messageId,modality}>}
 */
function parseDailyLog(content, date) {
  const events = [];
  let currentTime = null;

  for (const rawLine of String(content || '').split('\n')) {
    const line = rawLine.trimEnd();

    const timeMatch = line.match(TIME_HEADER_RE);
    if (timeMatch) { currentTime = timeMatch[1]; continue; }

    if (!line.startsWith('- ')) continue;
    const body = line.slice(2).trim();
    if (!body) continue;

    const prov = body.match(PROVENANCE_RE);
    if (!prov) continue;                       // background pass, not intake

    const text = body.slice(0, prov.index).trim();
    if (!text) continue;
    if (MACHINE_PREFIXES.some(re => re.test(text))) continue;

    events.push({
      id: entryId(date, currentTime || '00:00', text),
      date,
      time: currentTime || '00:00',
      text,
      conversationId: prov[1] || null,
      messageId: prov[2] || null,
      modality: prov[3] || null,
    });
  }
  return events;
}

/**
 * Event entries from the last `days` daily-log files, newest day first.
 *
 * `days` counts calendar files including today, so days:1 is today only and
 * days:3 is today plus the two before it. A missing file is a day with no log,
 * not an error.
 *
 * @param {Object} [opts]
 * @param {number} [opts.days=3]
 * @param {string} [opts.dailyDir] - defaults to the configured daily dir
 * @param {Date}   [opts.from] - for tests; defaults to now
 * @returns {{events: Array, filesRead: Array<string>, daysRequested: number}}
 */
function readRecentEvents({ days = 3, dailyDir = null, from = new Date() } = {}) {
  const dir = dailyDir || require('./database').getDailyDir();
  const stamps = dateStampsBack(Math.max(1, days), from);
  const events = [];
  const filesRead = [];

  for (const stamp of stamps) {
    const file = path.join(dir, `${stamp}.md`);
    let content;
    try {
      if (!fs.existsSync(file)) continue;
      content = fs.readFileSync(file, 'utf8');
    } catch (err) {
      console.error(`[DailyLogReader] could not read ${file}: ${err.message}`);
      continue;
    }
    filesRead.push(`${stamp}.md`);
    events.push(...parseDailyLog(content, stamp));
  }

  return { events, filesRead, daysRequested: Math.max(1, days) };
}

module.exports = {
  readRecentEvents,
  parseDailyLog,
  dateStampsBack,
  entryId,
  MACHINE_PREFIXES,
  PROVENANCE_RE,
};
