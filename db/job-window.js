/**
 * THE WINDOW A JOB RUNS IN, AND HOW MUCH OF IT A TURN MAY RESERVE.
 *
 * 2026-09-17, Juno. A research job died in tool round 11, forty tool calls in,
 * fourteen minutes in, on HTTP 400: "maximum context length is 131072 tokens.
 * However, you requested 49216 output tokens and your prompt contains at least
 * 81857". The runner behaved and reported honestly. The job had outgrown the
 * window — and it had outgrown it a third early, because every round reserved
 * the full answer-plus-thinking budget (16,448 + 32,768 on that box) whether or
 * not the turn was going to produce anything like it. A tool round on that
 * transcript produced ~250 tokens of tool call and ~200 of reasoning.
 *
 * Three things this module does, all pure except the probe:
 *
 *   1. KNOWS THE CEILING. `engineWindow()` asks the engine (db/model-context,
 *      the same probe the chat window uses) and honours a pinned
 *      agentJobs.context.windowTokens — which is also how a 262k box rehearses
 *      a 131k one. A ceiling nobody could see is how 131,072 became a surprise.
 *   2. ESTIMATES THE PROMPT, and gets better every round. The first estimate
 *      is chars / charsPerToken; after that the engine's own `usage` count
 *      calibrates the ratio. The 9/17 transcript ran 2.4 chars/token — dense
 *      JSON, URLs, code — against the usual 4, which is exactly the kind of
 *      thing a fixed constant gets wrong.
 *   3. FITS THE RESERVATION. `fitReservation` gives a round the full budget
 *      when the window has room and only what fits when it does not — the
 *      answer shrinks first, then the thinking, down to floors. Below the
 *      floors it says so, and the caller compacts or checkpoints instead of
 *      sending a request the engine will refuse.
 *
 * WHAT IT DOES NOT DO: cap how much a job may do. A limit that binds here
 * fails loudly with the numbers in the sentence (`refusalSentence`).
 */

const VLLM_REFUSAL = /maximum context length is (\d+) tokens[^0-9]*requested (\d+) output tokens[^0-9]*prompt contains at least (\d+)/i;
const GENERIC_REFUSAL = /context (?:length|window)[^0-9]*(\d{4,})/i;

/** Rough chars per token when nothing has been measured yet. Deliberately low: dense tool JSON. */
const DEFAULT_CHARS_PER_TOKEN = 3.0;

/**
 * A running estimate of chars-per-token for ONE transcript, corrected by the
 * engine's own count whenever a round reports one.
 */
function createCalibrator(charsPerToken = DEFAULT_CHARS_PER_TOKEN) {
  const start = Number.isFinite(charsPerToken) && charsPerToken > 0.5 ? charsPerToken : DEFAULT_CHARS_PER_TOKEN;
  return {
    ratio: start,
    samples: 0,
    /** The engine said this many prompt tokens for this many chars. */
    observe(chars, promptTokens) {
      if (!Number.isFinite(chars) || !Number.isFinite(promptTokens) || chars <= 0 || promptTokens <= 0) return this.ratio;
      const seen = chars / promptTokens;
      // Weighted toward the newest sample: the transcript's mix changes as
      // fetches are compacted, and an old ratio is a stale ratio.
      this.ratio = this.samples === 0 ? seen : (this.ratio + seen * 2) / 3;
      this.samples++;
      return this.ratio;
    },
    /** Tokens for this many chars, with a small safety factor. */
    estimate(chars) {
      if (!Number.isFinite(chars) || chars <= 0) return 0;
      return Math.ceil((chars / this.ratio) * 1.05);
    },
    state() { return { ratio: this.ratio, samples: this.samples }; },
    restore(st) {
      if (st && Number.isFinite(st.ratio) && st.ratio > 0.5) { this.ratio = st.ratio; this.samples = st.samples | 0; }
      return this;
    }
  };
}

/** Characters a message array will put on the wire, roughly: content, tool calls, names. */
function transcriptChars(messages = []) {
  let n = 0;
  for (const m of messages) {
    if (!m) continue;
    const c = m.content;
    if (typeof c === 'string') n += c.length;
    else if (c) n += JSON.stringify(c).length;
    if (m.tool_calls) n += JSON.stringify(m.tool_calls).length;
    if (m.name) n += m.name.length + 8;
    n += 12; // role and framing
  }
  return n;
}

/** Chars the tool schemas add to every round. */
function specsChars(specs = []) {
  try { return specs && specs.length ? JSON.stringify(specs).length : 0; } catch { return 0; }
}

/**
 * How much of the window this round may reserve for output.
 *
 * @param {Object} a
 * @param {number|null} a.window       the engine's ceiling (null = unknown: reserve the full budget, as before)
 * @param {number} a.promptTokens      the estimated prompt
 * @param {number} a.answerTokens      the configured answer budget
 * @param {number|null} a.thinkingTokens the configured thinking budget (null = none sent)
 * @param {number} [a.floorAnswer]     the smallest answer a squeezed round may have
 * @param {number} [a.floorThinking]   the smallest thinking a squeezed round may keep
 * @param {number} [a.margin]          headroom under the ceiling the estimate cannot see (template tokens)
 * @returns {{fits: boolean, maxTokens: number, thinking: number|null, room: number, squeezed: boolean, why: string|null}}
 */
function fitReservation({ window, promptTokens, answerTokens, thinkingTokens = null, floorAnswer = 1024, floorThinking = 1024, margin = 512 }) {
  const answer = Math.max(1, answerTokens | 0);
  const thinking = Number.isFinite(thinkingTokens) && thinkingTokens > 0 ? thinkingTokens | 0 : null;
  const full = answer + (thinking || 0);
  if (!Number.isFinite(window) || window <= 0) {
    return { fits: true, maxTokens: full, thinking, room: Infinity, squeezed: false, why: null };
  }
  const room = window - Math.max(0, promptTokens | 0) - Math.max(0, margin | 0);
  if (room >= full) return { fits: true, maxTokens: full, thinking, room, squeezed: false, why: null };

  // Squeeze the answer first — a tool round produces a few hundred tokens and
  // the writeup gets re-asked with room made — then the thinking.
  const minAnswer = Math.min(answer, Math.max(64, floorAnswer | 0));
  if (thinking === null) {
    if (room >= minAnswer) return { fits: true, maxTokens: room, thinking: null, room, squeezed: true, why: `answer budget squeezed to ${room}` };
    return { fits: false, maxTokens: 0, thinking: null, room, squeezed: true, why: `only ${room} tokens of room, below the ${minAnswer}-token answer floor` };
  }
  if (room >= thinking + minAnswer) {
    return { fits: true, maxTokens: room, thinking, room, squeezed: true, why: `answer budget squeezed to ${room - thinking}` };
  }
  const minThinking = Math.min(thinking, Math.max(0, floorThinking | 0));
  if (room >= minThinking + minAnswer) {
    const think = Math.max(minThinking, room - minAnswer);
    return { fits: true, maxTokens: room, thinking: think, room, squeezed: true, why: `thinking squeezed to ${think}, answer to ${room - think}` };
  }
  return { fits: false, maxTokens: 0, thinking, room, squeezed: true, why: `only ${room} tokens of room, below the ${minThinking + minAnswer}-token floor (thinking ${minThinking} + answer ${minAnswer})` };
}

/**
 * The engine's refusal, read back into numbers. vLLM's sentence carries all
 * three; anything else that names a context length gives at least the limit.
 * @returns {{limit: number, requested: number|null, prompt: number|null}|null}
 */
function parseContextRefusal(text) {
  const s = String(text || '');
  const m = VLLM_REFUSAL.exec(s);
  if (m) return { limit: Number(m[1]), requested: Number(m[2]), prompt: Number(m[3]) };
  const g = GENERIC_REFUSAL.exec(s);
  if (g) return { limit: Number(g[1]), requested: null, prompt: null };
  return null;
}

/** True when an error from streamChat is the engine refusing on context length. */
function isContextRefusal(err) {
  if (!err) return false;
  const status = err.status;
  const body = String(err.body || err.message || '');
  return (status === 400 || status === 413 || /HTTP 4\d\d/.test(String(err.message))) && !!parseContextRefusal(body);
}

/** The loud sentence for a window a job could not be fitted into. */
function refusalSentence({ window, promptTokens, requested, floor }) {
  const w = Number.isFinite(window) ? window.toLocaleString('en-US') : 'an unknown number of';
  const p = Number.isFinite(promptTokens) ? promptTokens.toLocaleString('en-US') : 'an unmeasured number of';
  return `The job outgrew the model's ${w}-token context window: its prompt had reached ${p} tokens` +
    `${Number.isFinite(requested) ? ` with ${requested.toLocaleString('en-US')} reserved for the answer` : ''}` +
    `${Number.isFinite(floor) ? `, and even the ${floor.toLocaleString('en-US')}-token floor for one more turn would not fit` : ''}. ` +
    `Compacting what it had read was not enough.`;
}

/**
 * The ceiling the job path should plan against. A pinned
 * agentJobs.context.windowTokens wins; otherwise the engine is asked through
 * db/model-context (cached, 2 s probe, never throws); otherwise null, which
 * means "reserve the full budget and let the engine say no" — the behaviour
 * before this module existed.
 */
async function engineWindow({ getConfig, getProviderInstance } = {}) {
  const cfgMod = require('./config');
  const config = (getConfig || cfgMod.getConfig)();
  const pinned = config.agentJobs && config.agentJobs.context && config.agentJobs.context.windowTokens;
  if (Number.isFinite(pinned) && pinned > 0) return { window: pinned, source: 'pinned' };
  try {
    const hb = config.models && config.models.heartbeat;
    if (!hb) return { window: null, source: 'none' };
    const inst = (getProviderInstance || cfgMod.getProviderInstance)(hb.provider, hb.instance);
    const host = inst ? inst.host : null;
    const entry = await require('./model-context').ensureProbed(hb.provider, host, hb.model);
    // ONLY THE ENGINE'S OWN ANSWER COUNTS HERE. model-context falls back to a
    // table keyed on the model's name, which is fine for sizing a chat window
    // and wrong for this: a table that says 32k for a "qwen" served at 131k
    // would have the runner squeezing and splitting a job that had room to
    // spare — a cap by another name. No engine answer means no window, and no
    // window means the request the runner always sent.
    if (entry && entry.source === 'engine' && Number.isFinite(entry.limit) && entry.limit > 0) return { window: entry.limit, source: 'engine', model: hb.model };
    if (entry) console.log(`[JobWindow] the engine did not report a window for ${hb.model} (the table would say ${entry.limit}); running without one`);
  } catch (err) {
    console.warn(`[JobWindow] could not read the engine's window: ${err.message}`);
  }
  return { window: null, source: 'none' };
}

module.exports = {
  DEFAULT_CHARS_PER_TOKEN,
  createCalibrator, transcriptChars, specsChars,
  fitReservation, parseContextRefusal, isContextRefusal, refusalSentence,
  engineWindow
};
