/**
 * The display-time layer: one place that turns a stored instant into a clock
 * time a person or an entity reads.
 *
 * WHY THIS FILE GOT REWRITTEN (2026-08-27). Athena read a history_search digest
 * that labelled a run of early-morning conversations "1:09 PM" to "1:24 PM".
 * They happened at 6:09 and 6:24 in the morning. The digest was printing raw
 * UTC, and she caught it because she knew what time she and Ellie had been
 * talking — which is to say the only check on this was an entity noticing its
 * own memory did not match its own clock.
 *
 * The instance was a one-line bug. The CLASS is that there was no display layer
 * at all: every feature formatted timestamps its own way, so forgetting the
 * conversion was the default and remembering it was an act of care that had to
 * be repeated at every new call site forever. Three different notions of "local"
 * were already live in this one file — the system timezone, a hardcoded
 * `America/Los_Angeles`, and nothing. So the fix is not a conversion added to
 * the digest; it is this file becoming the only way any surface renders a time.
 *
 * ── THE TRAP THAT MAKES THE OBVIOUS FIX WRONG ──
 *
 * This store holds timestamps in TWO formats, and only one of them is
 * self-describing:
 *
 *   agent_jobs.created_at    "2026-08-27T15:41:52.480Z"   written by code,
 *                                                          ISO 8601, marked UTC
 *   messages.timestamp       "2026-08-25 00:38:07"         written by SQLite's
 *                                                          CURRENT_TIMESTAMP —
 *                                                          UTC, and NOT MARKED
 *
 * `new Date("2026-08-25 00:38:07")` does not parse that second form as UTC. V8
 * reads a space-separated date-time with no offset as LOCAL time, so on a
 * Pacific host the value silently moves seven hours in the WRONG DIRECTION. The
 * natural fix for the reported bug — `new Date(ts).toLocaleString()` — is
 * therefore still wrong, and wrong in a way that looks plausible on screen:
 * 00:38 UTC would render as 12:38 AM rather than 5:38 PM the previous day.
 *
 * That is what toUtcDate() exists for, and it is why every path goes through it
 * rather than through `new Date()`. A timestamp with no timezone marker on it is
 * UTC in this system — every writer of that form is SQLite, and SQLite's
 * CURRENT_TIMESTAMP is UTC — so the parser says so explicitly instead of letting
 * the host's timezone decide.
 *
 * ── WHAT IS NOT CHANGING ──
 *
 * Storage. Every stored value stays UTC and no stored value is rewritten. This
 * layer is display only, which is what makes it safe to change the setting at
 * any time: nothing that has been recorded means something different afterwards,
 * it is only shown on a different clock.
 */

const DEFAULT_TIMEZONE = 'America/Los_Angeles';

function getConfig() { return require('./config').getConfig(); }

// Map common US timezone abbreviations to friendly region names.
// Falls back to the raw abbreviation for anything not listed.
const TZ_FRIENDLY = {
  PST: 'Pacific', PDT: 'Pacific',
  MST: 'Mountain', MDT: 'Mountain',
  CST: 'Central', CDT: 'Central',
  EST: 'Eastern', EDT: 'Eastern',
  AKST: 'Alaska', AKDT: 'Alaska',
  HST: 'Hawaii'
};

/**
 * Timezone names that have already been proven to work, so a bad one is
 * complained about once rather than on every render.
 */
const tzChecked = new Map();

/**
 * Is this a timezone name the runtime actually knows?
 *
 * An unknown IANA name makes Intl throw a RangeError, and a throw inside a
 * formatter is a throw inside whatever was rendering — a chat turn, a digest, a
 * panel. A setting typed by hand must not be able to do that, so it is validated
 * once and falls back loudly.
 */
function isUsableTimezone(name) {
  if (!name || typeof name !== 'string') return false;
  if (tzChecked.has(name)) return tzChecked.get(name);
  let ok = true;
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: name }).format(new Date(0));
  } catch {
    ok = false;
  }
  tzChecked.set(name, ok);
  if (!ok) {
    console.warn(`[DateTime] instance.timezone "${name}" is not a timezone this system knows — falling back to ${DEFAULT_TIMEZONE}. Set a valid IANA name (e.g. "America/New_York") in Settings.`);
  }
  return ok;
}

/**
 * THE INSTANCE'S CLOCK — an IANA timezone name from config.
 *
 * PER INSTANCE, and deliberately not a global or a host setting. Both boxes
 * running today sit with Ellie in Oregon, so both read Pacific; an instance
 * stood up for someone on the east coast sets Eastern and nobody else's clock
 * moves. It is also not the HOST timezone, which is what the old code used by
 * omission: the host's clock is an accident of where the machine is, and it
 * happened to agree here, which is exactly why nothing caught the difference
 * between "correct" and "coincidentally correct".
 *
 * Read live on every call, never captured at module load, so changing it in
 * Settings takes effect on the next render with no restart.
 */
function instanceTimezone() {
  let configured = null;
  try {
    configured = (getConfig().instance || {}).timezone;
  } catch {
    // Config unavailable (very early boot, or a test with no store). The
    // default is a working clock, not an error.
  }
  return isUsableTimezone(configured) ? configured : DEFAULT_TIMEZONE;
}

/**
 * Parse any stored instant into a Date, treating an unmarked timestamp as UTC.
 *
 * The one function that knows about the two-format problem described in the
 * header. Everything that renders a stored time goes through here rather than
 * calling `new Date()`, because `new Date()` gets the SQLite form wrong and does
 * so silently.
 *
 * Accepts: a Date, an epoch number, an ISO string (with Z or an offset), and the
 * SQLite "YYYY-MM-DD HH:MM:SS" form. Returns null for anything unparseable, so
 * callers render a placeholder instead of "Invalid Date".
 *
 * @param {string|number|Date} value
 * @returns {Date|null}
 */
function toUtcDate(value) {
  if (value === null || value === undefined || value === '') return null;
  if (value instanceof Date) return isNaN(value.getTime()) ? null : value;
  if (typeof value === 'number') {
    const d = new Date(value);
    return isNaN(d.getTime()) ? null : d;
  }

  const raw = String(value).trim();
  if (!raw) return null;

  // Already self-describing: ends in Z, or carries a ±HH:MM offset after the
  // time. Left exactly as it is — this form is unambiguous and parsing it any
  // other way would be inventing a problem.
  const marked = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(raw);
  if (marked) {
    const d = new Date(raw);
    return isNaN(d.getTime()) ? null : d;
  }

  // "YYYY-MM-DD HH:MM:SS[.sss]" or "YYYY-MM-DDTHH:MM:SS[.sss]" with no zone.
  // Every writer of this shape in this system is SQLite's CURRENT_TIMESTAMP,
  // which is UTC. Say so, rather than letting the host's timezone answer.
  const bare = raw.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?(?:\.(\d{1,3}))?$/);
  if (bare) {
    const [, y, mo, d, h, mi, s, ms] = bare;
    return new Date(Date.UTC(+y, +mo - 1, +d, +h, +mi, +(s || 0), +(ms || 0)));
  }

  // A bare calendar date is a DAY, not an instant. Noon UTC rather than
  // midnight, so it stays on the day it names in every timezone from UTC-11 to
  // UTC+12 — midnight UTC would render "2026-08-27" as the 26th anywhere west
  // of Greenwich, which is how a date silently becomes the wrong date.
  const dateOnly = raw.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (dateOnly) {
    const [, y, mo, d] = dateOnly;
    return new Date(Date.UTC(+y, +mo - 1, +d, 12));
  }

  const fallback = new Date(raw);
  return isNaN(fallback.getTime()) ? null : fallback;
}

/** The timezone's short label (PDT/PST/EST…) AT a given instant, so DST is right. */
function zoneAbbrev(date, timeZone) {
  try {
    return new Intl.DateTimeFormat('en-US', { timeZone, timeZoneName: 'short' })
      .formatToParts(date)
      .find(p => p.type === 'timeZoneName')?.value || '';
  } catch {
    return '';
  }
}

/**
 * Derive a friendly timezone label (e.g. "Pacific") for an instant.
 * @param {Date} [now]
 * @param {string} [timeZone] - defaults to the instance timezone
 */
function friendlyTimezone(now = new Date(), timeZone = instanceTimezone()) {
  const short = zoneAbbrev(now, timeZone);
  return TZ_FRIENDLY[short] || short;
}

/**
 * ═══ THE TRANSLATION FUNCTION ═══
 *
 * Render a stored UTC instant as the instance's local clock time. Every
 * human-facing and entity-facing timestamp in this system goes through here.
 *
 * DST IS NOT SPECIAL-CASED and must not be: the offset is resolved by Intl for
 * the specific instant being rendered, so an August timestamp gets PDT and a
 * January one gets PST from the same code path with no seasonal logic anywhere.
 * Anything that computed an offset once and reused it would be right for half
 * the year, which is the worst available failure mode — it looks fine when you
 * write it and breaks months later.
 *
 * THE ZONE LABEL IS PART OF THE OUTPUT, not decoration. A bare "6:38 AM" is
 * indistinguishable from an unconverted "6:38 AM", so the next time this goes
 * wrong nobody will be able to tell by looking. "6:38 AM PDT" can be checked.
 *
 * @param {string|number|Date} value - a stored instant, in any form toUtcDate accepts
 * @param {Object}  [opts]
 * @param {'datetime'|'full'|'date'|'time'} [opts.style='datetime']
 *        datetime  "2026-08-25 6:38 AM PDT"          — the default; sortable date, clear clock
 *        full      "Tuesday, August 25, 2026, 6:38 AM Pacific" — prose, for prompts
 *        date      "2026-08-25"                      — the local calendar day
 *        time      "6:38 AM PDT"                     — clock only, same-day contexts
 * @param {string}  [opts.timeZone] - override the instance timezone (tests, exports)
 * @param {string}  [opts.fallback=''] - returned when the value cannot be parsed
 * @returns {string}
 */
function formatLocalTime(value, opts = {}) {
  const { style = 'datetime', fallback = '' } = opts;
  const timeZone = opts.timeZone && isUsableTimezone(opts.timeZone)
    ? opts.timeZone
    : instanceTimezone();

  const d = toUtcDate(value);
  if (!d) return fallback;

  // A DAY HAS NO CLOCK TIME, so asking for one back is a question with no honest
  // answer. Callers pass bare dates around (daily-log filenames, `passId`
  // stamps), and rendering "2026-08-27" as "2026-08-27 12:00 PM PDT" would be
  // inventing a precision the value never had. The day is returned instead,
  // whatever style was requested.
  const dayOnly = typeof value === 'string' && /^\s*\d{4}-\d{2}-\d{2}\s*$/.test(value);

  try {
    if (style === 'date' || dayOnly) {
      // en-CA gives YYYY-MM-DD; timeZone decides which calendar day it is.
      return d.toLocaleDateString('en-CA', { timeZone });
    }

    const clock = d.toLocaleTimeString('en-US', {
      timeZone, hour: 'numeric', minute: '2-digit', hour12: true
    });

    if (style === 'time') {
      const z = zoneAbbrev(d, timeZone);
      return z ? `${clock} ${z}` : clock;
    }

    if (style === 'full') {
      const datePart = d.toLocaleDateString('en-US', {
        timeZone, weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
      });
      const z = friendlyTimezone(d, timeZone);
      return `${datePart}, ${clock}${z ? ' ' + z : ''}`;
    }

    // 'datetime' — the default.
    const day = d.toLocaleDateString('en-CA', { timeZone });
    const z = zoneAbbrev(d, timeZone);
    return `${day} ${clock}${z ? ' ' + z : ''}`;
  } catch (err) {
    // A formatter must never take down the thing that was rendering.
    console.error('[DateTime] formatLocalTime failed:', err.message);
    return fallback;
  }
}

/**
 * Build the "it is now …" line injected into every LLM system prompt.
 *
 * ON THE INSTANCE'S CLOCK, which is the point of routing this through the same
 * layer as everything else. If the entity's sense of "now" runs on a different
 * clock from the timestamps it reads back, then "this morning" means two things
 * inside one head — it would recall a conversation as having happened in the
 * afternoon and simultaneously believe it is currently morning. Matching Ellie's
 * clock is what makes "earlier today" a shared phrase rather than two.
 */
function getCurrentDateTimeString() {
  return `Current date and time: ${formatLocalTime(new Date(), { style: 'full' })}`;
}

/**
 * Prepend the current date/time line to an existing system prompt, so callers
 * that build their own prompt string (extraction, heartbeat) stay date-aware.
 * @param {string} systemPrompt
 * @returns {string}
 */
function withDateTime(systemPrompt) {
  return `${getCurrentDateTimeString()}\n\n${systemPrompt}`;
}

/**
 * Current calendar date on the instance's clock, as YYYY-MM-DD.
 *
 * The single source of truth for "which day's file" a daily-log entry belongs
 * to. Every daily-log writer AND reader buckets by this, so an entry written at
 * 6pm local lands in — and is read back from — today's local file rather than
 * tomorrow's UTC file.
 *
 * This used to hardcode America/Los_Angeles. It now reads the instance setting,
 * which is the same value on both boxes running today. Worth knowing before
 * changing that setting on a live instance: it moves the boundary for FUTURE
 * entries only. Nothing already written is renamed or re-bucketed, and an entry
 * near midnight either side of the change can land in a neighbouring day's file.
 *
 * @param {Date|string|number} [date=new Date()] - instant to stamp
 * @returns {string} e.g. "2026-07-04"
 */
function getLocalDateStamp(date = new Date()) {
  return formatLocalTime(date, { style: 'date' }) ||
    new Date().toLocaleDateString('en-CA', { timeZone: DEFAULT_TIMEZONE });
}

/**
 * 24-hour "HH:MM" on the instance clock, for the daily and ops log headers.
 *
 * Those files are a fixed format — `### HH:MM` blocks that db/daily-log-reader,
 * db/injection-budget and db/fact-extractor all parse — so this is a bare clock
 * rather than one of the display styles. It used to be `toTimeString()`, which
 * is the HOST's clock; on this box the two agree, which is exactly why it needed
 * finding rather than noticing.
 *
 * @param {Date|string|number} [date=new Date()]
 * @returns {string} e.g. "06:09"
 */
function formatLogClock(date = new Date()) {
  const d = toUtcDate(date);
  if (!d) return '00:00';
  try {
    return d.toLocaleTimeString('en-GB', {
      timeZone: instanceTimezone(), hour: '2-digit', minute: '2-digit', hour12: false
    });
  } catch {
    return '00:00';
  }
}

/**
 * Compact "learned"/"recorded" annotation for a stored timestamp.
 * Example: "2026-07-04 6:51 AM PDT". Returns null for missing/invalid input,
 * which is the contract its callers already rely on.
 * @param {string} iso
 * @returns {string|null}
 */
function formatFactTimestamp(iso) {
  const out = formatLocalTime(iso, { style: 'datetime', fallback: '' });
  return out || null;
}

module.exports = {
  DEFAULT_TIMEZONE,
  instanceTimezone,
  isUsableTimezone,
  toUtcDate,
  formatLocalTime,
  getCurrentDateTimeString,
  withDateTime,
  friendlyTimezone,
  getLocalDateStamp,
  formatLogClock,
  formatFactTimestamp
};
