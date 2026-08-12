#!/usr/bin/env node
/**
 * The due-time rule, tested on its own.
 *
 * db/cron-eval.js is the only thing standing between "0 9 * * *" and a job that
 * fires at the wrong hour, twice, or not at all — and every one of those is a
 * silent failure: a scheduler that runs a digest at 08:00 instead of 09:00 looks
 * exactly like a scheduler that works. So the arithmetic is asserted here rather
 * than inferred from a job that seemed to go off at the right time.
 *
 * PURE. No SNH_DATA_DIR, no database, no model, no clock dependency — every case
 * names its own `from` instant. TZ is pinned to America/Los_Angeles before the
 * first Date is constructed, so the daylight-saving cases assert the same thing
 * on any machine.
 *
 * Usage: node scripts/test-cron-eval.js
 */
process.env.TZ = 'America/Los_Angeles';

const path = require('path');
const { nextRunAfter, nextRuns, parseExpression } = require(path.join(__dirname, '..', 'db/cron-eval'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

/** Local wall-clock string, so a failure reads in the same units the cron does. */
function local(dt) {
  if (!dt) return 'null';
  const p = (n) => String(n).padStart(2, '0');
  return `${dt.getFullYear()}-${p(dt.getMonth() + 1)}-${p(dt.getDate())} ${p(dt.getHours())}:${p(dt.getMinutes())}`;
}
function at(y, mo, d, h = 0, mi = 0) { return new Date(y, mo - 1, d, h, mi, 0, 0); }

/** `expr` from `from` must fire at exactly this local wall time. */
function expectNext(name, expr, from, wantLocal) {
  const got = nextRunAfter(expr, from);
  check(name, local(got) === wantLocal, `got ${local(got)}, wanted ${wantLocal}`);
  return got;
}

console.log('\n1. Field ranges and the basic daily case');
expectNext('a daily 09:00 job from the morning fires today', '0 9 * * *', at(2026, 8, 12, 8, 19), '2026-08-12 09:00');
expectNext('…from after it, tomorrow', '0 9 * * *', at(2026, 8, 12, 9, 1), '2026-08-13 09:00');
expectNext('…and strictly after: standing exactly on it advances', '0 9 * * *', at(2026, 8, 12, 9, 0), '2026-08-13 09:00');
// Seconds are not a field. A run stamped 09:00:37 must not re-match its own minute.
const withSeconds = new Date(2026, 7, 12, 9, 0, 37);
check('seconds are ignored rather than re-matching the same minute',
  local(nextRunAfter('0 9 * * *', withSeconds)) === '2026-08-13 09:00',
  `got ${local(nextRunAfter('0 9 * * *', withSeconds))}`);
expectNext('midnight rolls the date', '0 0 * * *', at(2026, 8, 12, 23, 59), '2026-08-13 00:00');
expectNext('a month boundary rolls the month', '0 0 * * *', at(2026, 8, 31, 12, 0), '2026-09-01 00:00');
expectNext('a year boundary rolls the year', '30 23 * * *', at(2026, 12, 31, 23, 45), '2027-01-01 23:30');

console.log('\n2. Steps, lists and ranges');
check('*/15 gives the four quarters',
  nextRuns('*/15 * * * *', at(2026, 8, 12, 9, 1), 4).map(local).join(' | ') ===
  '2026-08-12 09:15 | 2026-08-12 09:30 | 2026-08-12 09:45 | 2026-08-12 10:00',
  nextRuns('*/15 * * * *', at(2026, 8, 12, 9, 1), 4).map(local).join(' | '));
expectNext('a list of hours takes the next one', '0 9,17 * * *', at(2026, 8, 12, 10, 0), '2026-08-12 17:00');
expectNext('…and wraps to the first tomorrow', '0 9,17 * * *', at(2026, 8, 12, 18, 0), '2026-08-13 09:00');
check('a stepped range stops at the end of the range',
  nextRuns('0 0-12/6 * * *', at(2026, 8, 12, 0, 30), 4).map(local).join(' | ') ===
  '2026-08-12 06:00 | 2026-08-12 12:00 | 2026-08-13 00:00 | 2026-08-13 06:00',
  nextRuns('0 0-12/6 * * *', at(2026, 8, 12, 0, 30), 4).map(local).join(' | '));
check('N/S runs from N to the end of the field',
  nextRuns('5/15 * * * *', at(2026, 8, 12, 9, 0), 4).map(local).join(' | ') ===
  '2026-08-12 09:05 | 2026-08-12 09:20 | 2026-08-12 09:35 | 2026-08-12 09:50',
  nextRuns('5/15 * * * *', at(2026, 8, 12, 9, 0), 4).map(local).join(' | '));
expectNext('a month field skips eleven months', '0 0 1 1 *', at(2026, 8, 12, 9, 0), '2027-01-01 00:00');
expectNext('a list of months picks the nearer one', '0 8 1 3,9 *', at(2026, 8, 12, 9, 0), '2026-09-01 08:00');

console.log('\n3. Day-of-month vs day-of-week (the Vixie OR)');
// 2026-08-12 is a Wednesday; the following Monday is the 17th, and the 1st of
// September is a Tuesday.
expectNext('day-of-month alone decides when day-of-week is open', '0 9 17 * *', at(2026, 8, 12, 9, 0), '2026-08-17 09:00');
expectNext('day-of-week alone decides when day-of-month is open', '0 9 * * 1', at(2026, 8, 12, 9, 0), '2026-08-17 09:00');
// Both restricted → OR. From the 12th, the next match is Monday the 17th (dow),
// not the 1st (dom) — and from the 18th it is September 1st.
expectNext('both restricted: the nearer of the two matches wins', '0 9 1 * 1', at(2026, 8, 12, 9, 0), '2026-08-17 09:00');
expectNext('…and the other one still matches', '0 9 1 * 1', at(2026, 8, 25, 9, 0), '2026-08-31 09:00');
expectNext('? in a day field reads as open', '0 9 ? * 1', at(2026, 8, 12, 9, 0), '2026-08-17 09:00');
// Sunday is 0 and 7. 2026-08-16 is a Sunday.
expectNext('day-of-week 0 is Sunday', '0 9 * * 0', at(2026, 8, 12, 9, 0), '2026-08-16 09:00');
expectNext('day-of-week 7 is the same Sunday', '0 9 * * 7', at(2026, 8, 12, 9, 0), '2026-08-16 09:00');
expectNext('a weekday range skips the weekend', '0 9 * * 1-5', at(2026, 8, 14, 10, 0), '2026-08-17 09:00');

console.log('\n4. Spring forward — 2026-03-08, 02:00 → 03:00 local');
// The gap is real: 02:30 does not exist that day.
const gapProbe = new Date(2026, 2, 8, 2, 30, 0, 0);
check('02:30 on the transition day does not exist (the platform normalizes it away)',
  gapProbe.getHours() !== 2, `got ${local(gapProbe)}`);
const springRuns = nextRuns('30 2 * * *', at(2026, 3, 6, 12, 0), 3);
check('the day before is untouched', local(springRuns[0]) === '2026-03-07 02:30', local(springRuns[0]));
// The whole point: the job still happens that day, at the moment the clock jumps.
check('a job in the gap fires at the instant the clock jumps, not a day later',
  local(springRuns[1]) === '2026-03-08 03:00', local(springRuns[1]));
check('…and that instant is the transition itself (10:00 UTC)',
  springRuns[1].toISOString() === '2026-03-08T10:00:00.000Z', springRuns[1].toISOString());
check('the day after is back to normal', local(springRuns[2]) === '2026-03-09 02:30', local(springRuns[2]));
// One firing per calendar day across the transition — the property that matters.
check('a daily job still fires exactly once a day across spring forward',
  new Set(springRuns.map(d => d.toDateString())).size === 3,
  springRuns.map(local).join(' | '));
// An hour that is skipped entirely still cannot fire twice or vanish elsewhere.
const springNoon = nextRuns('0 12 * * *', at(2026, 3, 7, 13, 0), 2).map(local);
check('a job outside the gap is unaffected', springNoon.join(' | ') === '2026-03-08 12:00 | 2026-03-09 12:00', springNoon.join(' | '));

console.log('\n5. Fall back — 2026-11-01, 02:00 → 01:00 local (01:30 happens twice)');
const firstOne30 = new Date(2026, 10, 1, 1, 30, 0, 0);
check('the repeated hour is real: 01:30 exists twice, an hour apart',
  new Date(firstOne30.getTime() + 3600_000).getHours() === 1,
  `${firstOne30.toISOString()} / ${new Date(firstOne30.getTime() + 3600_000).toISOString()}`);
const fallRuns = nextRuns('30 1 * * *', at(2026, 10, 30, 12, 0), 3);
check('the day before is untouched', local(fallRuns[0]) === '2026-10-31 01:30', local(fallRuns[0]));
check('the transition day fires on the FIRST 01:30 (still on daylight time)',
  fallRuns[1].toISOString() === '2026-11-01T08:30:00.000Z',
  `${local(fallRuns[1])} = ${fallRuns[1].toISOString()}`);
check('…and does NOT fire again on the repeat an hour later',
  fallRuns[2].toISOString() === '2026-11-02T09:30:00.000Z',
  `${local(fallRuns[2])} = ${fallRuns[2].toISOString()}`);
check('a daily job still fires exactly once a day across fall back',
  new Set(fallRuns.map(d => d.toDateString())).size === 3,
  fallRuns.map(local).join(' | '));
// Asked from INSIDE the repeated hour, the answer is still tomorrow — the wall
// clock has already passed 01:30 once, and a second pass is not a new match.
check('asked from inside the repeated hour, the next run is the next day',
  nextRunAfter('30 1 * * *', new Date(2026, 10, 1, 1, 45, 0, 0)).toISOString() === '2026-11-02T09:30:00.000Z',
  local(nextRunAfter('30 1 * * *', new Date(2026, 10, 1, 1, 45, 0, 0))));

console.log('\n6. Refusals — an unreadable expression never gets a guessed time');
for (const [expr, why] of [
  ['0 9 * *', 'four fields'],
  ['0 9 * * * *', 'six fields'],
  ['60 9 * * *', 'minute out of range'],
  ['0 24 * * *', 'hour out of range'],
  ['0 9 0 * *', 'day-of-month 0'],
  ['0 9 * 13 *', 'month out of range'],
  ['0 9 * * 8', 'day-of-week 8'],
  ['0 9-5 * * *', 'backwards range'],
  ['*/0 * * * *', 'zero step'],
  ['every morning', 'prose'],
  ['', 'empty']
]) {
  check(`refuses ${why}: "${expr}"`, nextRunAfter(expr, at(2026, 8, 12, 9, 0)) === null);
}
check('an expression that can never match returns null rather than looping forever',
  nextRunAfter('0 0 30 2 *', at(2026, 8, 12, 9, 0)) === null);
check('a bad date in gives null, not a crash',
  nextRunAfter('0 9 * * *', new Date('nonsense')) === null);

console.log('\n7. parseExpression reports what it read');
const p = parseExpression('0 9 * * 1-5');
check('minutes/hours parsed', p && p.minutes.has(0) && p.minutes.size === 1 && p.hours.has(9) && p.hours.size === 1);
check('day-of-week restriction is recorded, day-of-month openness too',
  p && p.dowRestricted === true && p.domRestricted === false);
check('a star-step counts as a restriction, not as open',
  parseExpression('0 9 */2 * *').domRestricted === true);
check('the real job, "0 9 * * *", parses to one time a day',
  (() => { const q = parseExpression('0 9 * * *'); return q && q.minutes.size === 1 && q.hours.size === 1 && !q.domRestricted && !q.dowRestricted; })());

const bar = '='.repeat(74);
console.log(`\n${bar}`);
console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
console.log(`${bar}\n`);
process.exit(fail === 0 ? 0 : 1);
