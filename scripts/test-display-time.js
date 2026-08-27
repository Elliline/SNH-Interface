#!/usr/bin/env node
/**
 * The display-time layer, tested where getting it wrong is invisible.
 *
 * The bug that produced this layer was invisible to everything except an entity
 * who happened to know what time she had been talking: a digest said 1:09 PM
 * about a conversation that happened at 6:09 AM, and every part of the system
 * agreed with it. Nothing was throwing, nothing was empty, nothing looked odd.
 * So the tests here are mostly about values that LOOK fine and are wrong.
 *
 * Four things under test:
 *   1. THE UNMARKED-UTC TRAP. messages.timestamp is SQLite CURRENT_TIMESTAMP —
 *      UTC with no marker — and `new Date()` reads it as local. The obvious fix
 *      for the reported bug is still wrong, and this asserts the difference.
 *   2. DST, WITHOUT SEASONAL LOGIC. The same wall-clock moment in January and
 *      August must render as the same wall-clock time, with the zone label
 *      changing rather than the hour.
 *   3. THE SETTING IS READ LIVE. Change it, and the next render moves. No
 *      restart, no cached formatter, no module-load capture.
 *   4. THE DIGEST, END TO END. The surface where it was caught.
 *
 * Runs against a throwaway SNH_DATA_DIR; the model is stubbed. Config is stubbed
 * on the module object, because data/config.json is deliberately NOT redirected
 * by SNH_DATA_DIR and a test must never write the live one.
 *
 * Usage: node scripts/test-display-time.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-display-time-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const config = require(path.join(ROOT, 'db/config'));
const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
const datetime = require(path.join(ROOT, 'db/datetime'));
const historySearch = require(path.join(ROOT, 'db/history-search'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- config stub: the instance clock is what we are varying -----------------
const realGetConfig = config.getConfig;
let tz = 'America/Los_Angeles';
config.getConfig = () => {
  const c = realGetConfig();
  c.instance = { ...(c.instance || {}), timezone: tz };
  c.tools = { ...c.tools, historySearch: { ...(c.tools && c.tools.historySearch), waitMs: 20000 } };
  return c;
};

(async () => {
console.log('\n1. An unmarked timestamp is UTC, and the obvious fix is not');
{
  // 13:09 UTC is 6:09 AM Pacific. This is the exact shape Athena read back.
  const stored = '2026-08-27 13:09:12';

  check('the layer renders it on the instance clock',
    datetime.formatLocalTime(stored) === '2026-08-27 6:09 AM PDT',
    datetime.formatLocalTime(stored));

  check('…which is what the entity actually reported as wrong',
    /6:09 AM/.test(datetime.formatLocalTime(stored)) &&
    !/1:09 PM/.test(datetime.formatLocalTime(stored)));

  // The point of the whole parser: the naive fix produces a plausible, wrong answer.
  const naive = new Date(stored).toLocaleTimeString('en-US', { timeZone: 'America/Los_Angeles', hour: 'numeric', minute: '2-digit' });
  check('THE NAIVE new Date() FIX IS STILL WRONG, and this proves it',
    naive === '1:09 PM' && datetime.formatLocalTime(stored, { style: 'time' }) === '6:09 AM PDT',
    `new Date() gave ${naive}; the layer gives ${datetime.formatLocalTime(stored, { style: 'time' })}`);

  check('toUtcDate reads the unmarked form as UTC',
    datetime.toUtcDate(stored).toISOString() === '2026-08-27T13:09:12.000Z',
    datetime.toUtcDate(stored).toISOString());

  check('…and leaves a marked ISO value exactly alone',
    datetime.toUtcDate('2026-08-27T13:09:12.000Z').toISOString() === '2026-08-27T13:09:12.000Z');

  check('…and honours an explicit offset rather than overriding it',
    datetime.toUtcDate('2026-08-27T13:09:12+02:00').toISOString() === '2026-08-27T11:09:12.000Z',
    datetime.toUtcDate('2026-08-27T13:09:12+02:00').toISOString());
}

console.log('\n2. DST is resolved per instant, with no seasonal logic anywhere');
{
  // Both are 6:09 AM Pacific. The UTC offset differs (7h in summer, 8h in
  // winter), so the two stored values differ — and both must read 6:09 AM.
  const august  = '2026-08-15 13:09:00';   // PDT, UTC-7
  const january = '2026-01-15 14:09:00';   // PST, UTC-8

  const a = datetime.formatLocalTime(august, { style: 'time' });
  const j = datetime.formatLocalTime(january, { style: 'time' });

  check('a Pacific 6:09 AM in August renders 6:09 AM', a === '6:09 AM PDT', a);
  check('a Pacific 6:09 AM in January renders 6:09 AM', j === '6:09 AM PST', j);
  check('…the CLOCK is identical and only the zone label moves',
    a.split(' ').slice(0, 2).join(' ') === j.split(' ').slice(0, 2).join(' ') && a !== j,
    `${a} vs ${j}`);

  // The label is not decoration: without it, a converted and an unconverted
  // time are the same string and the next regression is undetectable.
  check('the zone label is present, so a future regression is visible',
    /P[DS]T/.test(datetime.formatLocalTime(august)));

  // Right at the boundary. 2026 US DST ends Nov 1 at 2:00 AM local.
  const beforeFallBack = '2026-11-01 08:59:00'; // 01:59 PDT
  const afterFallBack  = '2026-11-01 09:01:00'; // 01:01 PST
  check('an instant before the fall-back boundary is PDT',
    /PDT/.test(datetime.formatLocalTime(beforeFallBack)), datetime.formatLocalTime(beforeFallBack));
  check('…and one after it is PST', /PST/.test(datetime.formatLocalTime(afterFallBack)),
    datetime.formatLocalTime(afterFallBack));
}

console.log('\n3. The styles say what they mean');
{
  const t = '2026-08-27 13:09:12';
  check('datetime is sortable date + clear clock',
    datetime.formatLocalTime(t) === '2026-08-27 6:09 AM PDT');
  check('date is the LOCAL calendar day',
    datetime.formatLocalTime(t, { style: 'date' }) === '2026-08-27');
  check('time is the clock alone', datetime.formatLocalTime(t, { style: 'time' }) === '6:09 AM PDT');
  check('full is prose, for prompts',
    /^Thursday, August 27, 2026, 6:09 AM Pacific$/.test(datetime.formatLocalTime(t, { style: 'full' })),
    datetime.formatLocalTime(t, { style: 'full' }));

  // A late-evening UTC instant is the PREVIOUS local day. The classic off-by-one
  // that makes a daily log land in tomorrow's file.
  check('a 03:29 UTC instant is the previous local day',
    datetime.formatLocalTime('2026-08-24 03:29:57', { style: 'date' }) === '2026-08-23',
    datetime.formatLocalTime('2026-08-24 03:29:57', { style: 'date' }));

  check('a bare calendar date stays that date, with no invented clock time',
    datetime.formatLocalTime('2026-08-27') === '2026-08-27');
  check('unparseable input returns the fallback, never "Invalid Date"',
    datetime.formatLocalTime('not a date', { fallback: '—' }) === '—' &&
    datetime.formatLocalTime(null, { fallback: '—' }) === '—');
}

console.log('\n4. The setting is read live — change it, next render moves');
{
  const t = '2026-08-27 13:09:12';
  const pacific = datetime.formatLocalTime(t);
  check('starts on Pacific', pacific === '2026-08-27 6:09 AM PDT', pacific);

  tz = 'America/New_York';
  const eastern = datetime.formatLocalTime(t);
  check('THE VERY NEXT RENDER IS EASTERN, with no restart and no cache flush',
    eastern === '2026-08-27 9:09 AM EDT', eastern);

  tz = 'UTC';
  check('…and UTC renders as UTC', datetime.formatLocalTime(t) === '2026-08-27 1:09 PM UTC',
    datetime.formatLocalTime(t));

  tz = 'Pacific/Honolulu';
  check('…and a zone with no DST works too', /HST$/.test(datetime.formatLocalTime(t)),
    datetime.formatLocalTime(t));

  // The entity's own "now" has to move with it, or it holds two clocks at once.
  tz = 'America/New_York';
  const nowEastern = datetime.getCurrentDateTimeString();
  tz = 'America/Los_Angeles';
  const nowPacific = datetime.getCurrentDateTimeString();
  check('the entity\'s injected sense of "now" follows the same setting',
    /Eastern/.test(nowEastern) && /Pacific/.test(nowPacific),
    `${nowEastern} | ${nowPacific}`);

  // A typo must not throw inside whatever was rendering.
  tz = 'Mars/Olympus_Mons';
  check('a nonsense timezone falls back rather than throwing',
    datetime.formatLocalTime(t) === '2026-08-27 6:09 AM PDT', datetime.formatLocalTime(t));
  check('…and instanceTimezone reports the fallback, not the bad value',
    datetime.instanceTimezone() === 'America/Los_Angeles');
  tz = 'America/Los_Angeles';
}

console.log('\n5. The digest — the surface where this was caught');
{
  // A conversation at 6:09 AM Pacific, stored the way SQLite stores it.
  const conv = database.createConversation('Morning chat about the dogs', 'test-model');
  const mid = database.addMessage(conv, 'assistant',
    'Cece was already at the door before the coffee finished, which is the most Rottweiler thing possible.');
  db.prepare('UPDATE messages SET timestamp = ? WHERE id = ?').run('2026-08-27 13:09:12', mid);

  memoryManager.callLLM = async () => ({
    content: JSON.stringify({
      found: true,
      summary: 'You talked about Cece in the morning.',
      quotes: [{ message_id: mid, quote: 'Cece was already at the door before the coffee finished' }],
      gaps: ''
    }),
    toolCalls: [], budget: null, truncated: false, outOfRounds: false, reasoningChars: 0, roundMs: [1]
  });

  const r = await historySearch.ask({
    question: 'What did we say about the dogs this morning?',
    conversationId: database.createConversation('asking', 'test-model')
  });

  check('the digest came back with its quote', r.status === 'ok' && r.verified === 1, JSON.stringify(r.status));
  check('THE REFERENCE READS 6:09 AM, NOT 1:09 PM', /6:09 AM/.test(r.digest) && !/1:09 PM/.test(r.digest),
    (r.digest.match(/at [^·]+/) || ['?'])[0]);
  check('…and it says which clock, so the next regression is visible',
    /6:09 AM PDT/.test(r.digest));
  check('…and no raw UTC string survives anywhere in it',
    !/13:09:12/.test(r.digest), 'a raw stored timestamp leaked into the digest');

  // The agent's own view of the store must agree with the digest's.
  const hits = historySearch.find({ query: 'Rottweiler coffee door' });
  const hit = hits.hits.find(h => h.message_id === mid);
  check('the search results handed to the RUN are on the same clock',
    hit && /6:09 AM PDT/.test(hit.timestamp), hit && hit.timestamp);
  const win = historySearch.readAround({ message_id: mid });
  check('…and so is the read window', win.messages.every(m => !/^\d{4}-\d{2}-\d{2} \d{2}:/.test(m.timestamp)),
    JSON.stringify(win.messages.map(m => m.timestamp)));

  // And it follows the setting like everything else.
  tz = 'America/New_York';
  const east = historySearch.find({ query: 'Rottweiler coffee door' }).hits.find(h => h.message_id === mid);
  check('a different instance clock moves the digest\'s times too',
    /9:09 AM EDT/.test(east.timestamp), east.timestamp);
  tz = 'America/Los_Angeles';
}

console.log('\n6. The legacy helpers now share the one implementation');
{
  check('formatFactTimestamp routes through the layer',
    datetime.formatFactTimestamp('2026-08-27 13:09:12') === '2026-08-27 6:09 AM PDT',
    datetime.formatFactTimestamp('2026-08-27 13:09:12'));
  check('…and still returns null for nothing, as its callers expect',
    datetime.formatFactTimestamp(null) === null && datetime.formatFactTimestamp('') === null);

  check('getLocalDateStamp buckets on the instance clock',
    datetime.getLocalDateStamp('2026-08-24 03:29:57') === '2026-08-23',
    datetime.getLocalDateStamp('2026-08-24 03:29:57'));

  tz = 'America/New_York';
  check('…and follows the setting (03:29 UTC is still the 23rd in Eastern)',
    datetime.getLocalDateStamp('2026-08-24 03:29:57') === '2026-08-23');
  check('…while an instant that straddles differently moves with the zone',
    datetime.getLocalDateStamp('2026-08-24 05:30:00') === '2026-08-24' &&
    (() => { tz = 'America/Los_Angeles'; return datetime.getLocalDateStamp('2026-08-24 05:30:00') === '2026-08-23'; })(),
    'the local calendar day should differ between Eastern and Pacific at 05:30 UTC');
  tz = 'America/Los_Angeles';
}

console.log('\n==========================================================================');
console.log(fail ? `${fail} FAILED, ${pass} passed.` : `All ${pass} checks pass.`);
console.log('==========================================================================');
process.exit(fail ? 1 : 0);
})();
