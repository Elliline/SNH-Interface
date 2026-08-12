/**
 * Cron evaluation — when does a 5-field expression next fire, on THIS machine's
 * local calendar.
 *
 * No npm package. This codebase adds third-party dependencies reluctantly, and a
 * scheduler's due-time rule is the one part of it that must be readable: a
 * dependency here would mean the answer to "why did my job run at 3am" lives in
 * someone else's changelog. It is also small — the whole rule is "advance the
 * calendar until every field matches".
 *
 * LOCAL TIME, DELIBERATELY. The schedules are written by Aurelius and read by
 * Ellie, and both of them mean 9am here. That makes daylight saving a real case
 * rather than a curiosity, and the two transitions are the only places where
 * "advance a minute" and "advance a wall-clock minute" disagree:
 *
 *   SPRING FORWARD — 02:30 does not exist on the transition day. A daily job at
 *     02:30 must not silently skip a day, so it fires at the instant the clock
 *     jumps to (03:00), which is the first moment its wall time has passed. This
 *     is what Vixie cron does with a fixed-time job in the gap.
 *   FALL BACK — 01:30 happens twice. It fires on the FIRST one. The second is
 *     not a new match: the walk resumes from the instant that just ran, whose
 *     wall components are already 01:30, so the next candidate is the next day.
 *     A job that runs twice because the clock repeated is a job that lied about
 *     its schedule.
 *
 * Both are asserted in scripts/test-cron-eval.js against real 2026 US/Pacific
 * transition dates.
 *
 * The expression grammar matches what db/cron-jobs.js already validates at
 * propose time — *, N, N-M, star-step, N-M/S, N/S, lists, and `?` in the two day
 * fields — so a schedule that was accepted for storage is one this can evaluate.
 * Anything else returns null rather than guessing: a scheduler that invents a
 * due time for an expression it did not understand is worse than one that says
 * it cannot read it.
 */

const FIELDS = [
  { name: 'minute', min: 0, max: 59 },
  { name: 'hour', min: 0, max: 23 },
  { name: 'day-of-month', min: 1, max: 31 },
  { name: 'month', min: 1, max: 12 },
  { name: 'day-of-week', min: 0, max: 7 }   // 0 and 7 are both Sunday
];

/** How far ahead we are willing to look before calling an expression unsatisfiable. */
const HORIZON_YEARS = 5;

/**
 * One field → the set of values it matches.
 * @returns {Set<number>|null} null if the term is unparseable
 */
function parseField(raw, min, max) {
  const text = String(raw || '').trim();
  if (!text) return null;
  const out = new Set();
  for (const term of text.split(',')) {
    const m = term.trim().match(/^(\*|\?|\d+(?:-\d+)?)(?:\/(\d+))?$/);
    if (!m) return null;
    const step = m[2] === undefined ? 1 : Number(m[2]);
    if (!Number.isInteger(step) || step < 1) return null;

    let lo, hi;
    if (m[1] === '*' || m[1] === '?') {
      lo = min; hi = max;
    } else if (m[1].includes('-')) {
      const [a, b] = m[1].split('-').map(Number);
      lo = a; hi = b;
      if (lo > hi) return null;
    } else {
      lo = Number(m[1]);
      // Vixie's "N/S" means "from N to the end of the range, every S".
      hi = step > 1 ? max : lo;
    }
    if (lo < min || hi > max) return null;
    for (let v = lo; v <= hi; v += step) out.add(v);
  }
  return out.size ? out : null;
}

/**
 * Parse a 5-field expression into matchable sets.
 *
 * `domRestricted` / `dowRestricted` are read from the RAW text, not from the
 * resulting set, because the day rule below turns on whether the field was
 * WRITTEN as a restriction: "*" and "?" are open, everything else — including
 * a star-step that happens to match every day — is a restriction.
 *
 * @returns {Object|null}
 */
function parseExpression(expr) {
  const parts = String(expr || '').trim().split(/\s+/);
  if (parts.length !== 5) return null;

  const sets = [];
  for (let i = 0; i < 5; i++) {
    const s = parseField(parts[i], FIELDS[i].min, FIELDS[i].max);
    if (!s) return null;
    sets.push(s);
  }

  // Sunday is both 0 and 7 in cron. Normalize to 0 so the day check has one
  // representation to compare against Date#getDay.
  const dow = new Set([...sets[4]].map(v => (v === 7 ? 0 : v)));

  const open = (t) => t === '*' || t === '?';
  return {
    expr: String(expr).trim(),
    minutes: sets[0],
    hours: sets[1],
    doms: sets[2],
    months: sets[3],
    dows: dow,
    domRestricted: !open(parts[2]),
    dowRestricted: !open(parts[4])
  };
}

/**
 * Does this date match the day fields?
 *
 * The Vixie rule, which is the one every crontab(5) documents and the one a
 * person writing "0 9 1 * 1" expects: when BOTH day fields are restricted the
 * job runs when EITHER matches (the 1st of the month, and every Monday). When
 * only one is restricted, only that one decides.
 */
function dayMatches(parsed, y, mo, d) {
  const dom = parsed.doms.has(d);
  // Noon, so a day that has no midnight (some zones' DST gap) still resolves to
  // the right calendar day.
  const dow = parsed.dows.has(new Date(y, mo, d, 12, 0, 0).getDay());
  if (parsed.domRestricted && parsed.dowRestricted) return dom || dow;
  if (parsed.domRestricted) return dom;
  if (parsed.dowRestricted) return dow;
  return true;
}

/** Local wall-clock components as a comparable number, ignoring the offset. */
function wallKey(dt) {
  return Date.UTC(dt.getFullYear(), dt.getMonth(), dt.getDate(), dt.getHours(), dt.getMinutes());
}

/**
 * The first real instant whose wall-clock time is at or after a nominal one.
 *
 * Only reached when the nominal time does not exist — the spring-forward gap.
 * Binary search rather than arithmetic on the offset, because the size of the
 * jump is a property of the zone (some are 30 minutes) and reading it off the
 * calendar cannot be wrong about it. Safe here specifically because wall time is
 * monotonic across a gap day; the day that repeats an hour never takes this path.
 */
function firstInstantAtOrAfterWall(y, mo, d, h, mi) {
  const target = Date.UTC(y, mo, d, h, mi);
  const anchor = new Date(y, mo, d, 12, 0, 0).getTime();
  let lo = anchor - 14 * 3600_000;   // comfortably before local midnight
  let hi = anchor + 14 * 3600_000;   // comfortably after the following midnight
  if (wallKey(new Date(hi)) < target) return null;
  while (hi - lo > 60_000) {
    const mid = lo + Math.floor((hi - lo) / 2 / 60_000) * 60_000;
    if (mid === lo) break;
    if (wallKey(new Date(mid)) >= target) hi = mid; else lo = mid;
  }
  return new Date(hi);
}

/**
 * Turn matched calendar components into an instant.
 *
 * Normally this is just `new Date(...)`. The exception is the spring-forward
 * gap, where JS silently normalizes 02:30 into 03:30 — an hour later than the
 * transition, on a day the job should fire AT the transition. Detected by
 * reading the components back and comparing.
 */
function materialize(y, mo, d, h, mi) {
  const dt = new Date(y, mo, d, h, mi, 0, 0);
  if (dt.getFullYear() === y && dt.getMonth() === mo && dt.getDate() === d &&
      dt.getHours() === h && dt.getMinutes() === mi) {
    return dt;
  }
  return firstInstantAtOrAfterWall(y, mo, d, h, mi);
}

/** Calendar arithmetic on a {y,mo,d,h,mi} tuple, via noon so DST cannot shift the day. */
function addDays(t, n) {
  const dt = new Date(t.y, t.mo, t.d + n, 12, 0, 0);
  return { y: dt.getFullYear(), mo: dt.getMonth(), d: dt.getDate(), h: 0, mi: 0 };
}
function startOfNextMonth(t) {
  const dt = new Date(t.y, t.mo + 1, 1, 12, 0, 0);
  return { y: dt.getFullYear(), mo: dt.getMonth(), d: 1, h: 0, mi: 0 };
}

/**
 * The next firing STRICTLY AFTER `from`, on the local calendar.
 *
 * Strictly after, so feeding a run's own timestamp back in advances rather than
 * returning the same minute forever — that is how the scheduler re-arms.
 *
 * @param {string} expr - 5-field cron expression
 * @param {Date|string|number} [from=now]
 * @returns {Date|null} null if the expression is unreadable or matches nothing
 *   within HORIZON_YEARS (e.g. "0 0 30 2 *" — there is no 30th of February)
 */
function nextRunAfter(expr, from = new Date()) {
  const parsed = parseExpression(expr);
  if (!parsed) return null;

  const start = from instanceof Date ? from : new Date(from);
  if (isNaN(start.getTime())) return null;

  // Begin at the minute after `from`; seconds never participate in a 5-field
  // expression, so a run at 09:00:37 must not re-match 09:00.
  const base = new Date(start.getTime());
  base.setSeconds(0, 0);
  base.setMinutes(base.getMinutes() + 1);

  let t = {
    y: base.getFullYear(), mo: base.getMonth(), d: base.getDate(),
    h: base.getHours(), mi: base.getMinutes()
  };
  const limitYear = start.getFullYear() + HORIZON_YEARS;

  // Each pass fixes the coarsest field that does not match and zeroes everything
  // below it, so the walk skips whole months and days instead of ticking minutes.
  for (let guard = 0; guard < 500_000; guard++) {
    if (t.y > limitYear) return null;

    if (!parsed.months.has(t.mo + 1)) { t = startOfNextMonth(t); continue; }
    if (!dayMatches(parsed, t.y, t.mo, t.d)) { t = addDays(t, 1); continue; }
    if (!parsed.hours.has(t.h)) {
      t.h += 1; t.mi = 0;
      if (t.h > 23) t = addDays(t, 1);
      continue;
    }
    if (!parsed.minutes.has(t.mi)) {
      t.mi += 1;
      if (t.mi > 59) { t.mi = 0; t.h += 1; if (t.h > 23) t = addDays(t, 1); }
      continue;
    }

    const dt = materialize(t.y, t.mo, t.d, t.h, t.mi);
    // A gap-resolved instant can land at or before `from` (the transition
    // already passed); step past it rather than returning a stale time.
    if (dt && dt.getTime() > start.getTime()) return dt;
    t.mi += 1;
    if (t.mi > 59) { t.mi = 0; t.h += 1; if (t.h > 23) t = addDays(t, 1); }
  }
  return null;
}

/** The next `n` firings, for previews and tests. */
function nextRuns(expr, from = new Date(), n = 5) {
  const out = [];
  let cursor = from instanceof Date ? from : new Date(from);
  for (let i = 0; i < n; i++) {
    const next = nextRunAfter(expr, cursor);
    if (!next) break;
    out.push(next);
    cursor = next;
  }
  return out;
}

module.exports = { nextRunAfter, nextRuns, parseExpression, HORIZON_YEARS };
