/**
 * Shared date/time awareness helper.
 *
 * Single source of truth for injecting the current date and time into every
 * LLM system prompt (chat, extraction, heartbeat). Uses the system's local
 * timezone so the model always knows what "today" is.
 */

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
 * Derive a friendly timezone label from the system timezone (e.g. "Pacific").
 * @param {Date} now
 * @returns {string}
 */
function friendlyTimezone(now) {
  const short = new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' })
    .formatToParts(now)
    .find(p => p.type === 'timeZoneName')?.value || '';
  return TZ_FRIENDLY[short] || short;
}

/**
 * Build a human-readable current date/time string in the system timezone.
 * Example: "Current date and time: Saturday, July 4, 2026, 8:15 AM Pacific"
 * @returns {string}
 */
function getCurrentDateTimeString() {
  const now = new Date();
  const datePart = now.toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
  });
  const timePart = now.toLocaleTimeString('en-US', {
    hour: 'numeric', minute: '2-digit', hour12: true
  });
  const tz = friendlyTimezone(now);
  return `Current date and time: ${datePart}, ${timePart}${tz ? ' ' + tz : ''}`;
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
 * Format a stored timestamp (ISO string, usually UTC) as a compact
 * "learned" annotation in the system's local (Pacific) timezone.
 * Example: "2026-07-04 6:51 AM". Returns null for missing/invalid input.
 * @param {string} iso
 * @returns {string|null}
 */
function formatFactTimestamp(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  const date = d.toLocaleDateString('en-CA'); // YYYY-MM-DD in local tz
  const time = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
  return `${date} ${time}`;
}

module.exports = {
  getCurrentDateTimeString,
  withDateTime,
  friendlyTimezone,
  formatFactTimestamp
};
