/**
 * How big is the window, actually?
 *
 * `memory-flush` carried a static table keyed on substrings of the model name.
 * On 2026-08-13 that table answered 8,192 for the running model because the name
 * contains "gemma", while the engine serving it reported max_model_len 131,072 —
 * so the flush was compacting threads at roughly a twentieth of the real window,
 * throwing away conversation that would have fitted with room to spare. A name is
 * not a capability. The engine knows, so ask the engine.
 *
 * TWO NUMBERS, and they do different jobs:
 *
 *   - the ENGINE LIMIT is a hard ceiling. Exceed it and the request fails or is
 *     silently truncated, so nothing may ever be configured above it.
 *   - the USABLE WINDOW (`memory.contextTokens`) is a deliberate choice BELOW
 *     that ceiling. KV cache is real memory and prefill is real latency: a 131k
 *     window is not free just because it is available, and a thread allowed to
 *     grow into it gets slower every turn for the whole rest of its life. The
 *     default is 32,768 — chosen because the routing and honesty probes were
 *     re-measured at that size and hold, not because it is a round number.
 *
 * The effective window is min(engine, configured). Detection failing is not an
 * error: it falls back to the static table, which is what shipped before.
 */
const { getConfig } = require('./config');

/** The old static table, kept as the fallback when an engine cannot be asked. */
const TABLE = [
  [/o1|o3|o4/, 200000],
  [/claude-opus-4|claude-sonnet-4|claude-haiku-4/, 200000],
  [/claude-opus-5|claude-sonnet-5|claude-fable-5/, 200000],
  [/gpt-5|gpt-4o|gpt-4-turbo/, 128000],
  [/gpt-4/, 8192],
  [/command/, 128000],
  [/grok-4|grok-3/, 131072],
  [/scout|llama|deepseek/, 131072],
  [/qwen/, 32768],
  [/mistral/, 32768],
  [/phi/, 16384],
  [/gemma/, 8192],
];

const DEFAULT_LIMIT = 8192;

/** Providers whose served window can be read over HTTP. */
const PROBEABLE = new Set(['vllm', 'llamacpp', 'squatchserve', 'ollama']);

// key `${provider}|${host}|${model}` → { limit, at, source }
const cache = new Map();
const inflight = new Map();
const PROBE_TTL_MS = 10 * 60 * 1000;
const PROBE_TIMEOUT_MS = 2000;

function tableLimit(model) {
  if (!model || typeof model !== 'string') return DEFAULT_LIMIT;
  const m = model.toLowerCase();
  for (const [re, n] of TABLE) if (re.test(m)) return n;
  return DEFAULT_LIMIT;
}

function key(provider, host, model) {
  return `${String(provider || '').toLowerCase()}|${host || ''}|${model || ''}`;
}

async function getJSON(url, init) {
  const res = await fetch(url, { ...init, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Ask an engine what window it is actually serving. Returns null when the engine
 * cannot say — never throws, because a failed probe must degrade to the table
 * rather than take the chat path down with it.
 */
async function probeEngine(provider, host, model) {
  const p = String(provider || '').toLowerCase();
  if (!PROBEABLE.has(p) || !host) return null;
  try {
    if (p === 'ollama') {
      const data = await getJSON(`${host}/api/show`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
      // num_ctx is what this model is actually SERVED with when the Modelfile
      // sets it; the architecture's context_length is only the ceiling. Prefer
      // the served value — over-reading here means a conversation grows past
      // what Ollama will accept and gets silently cut at the front.
      const params = String(data.parameters || '');
      const numCtx = params.match(/^\s*num_ctx\s+(\d+)/m);
      if (numCtx) return Number(numCtx[1]);
      const info = data.model_info || {};
      const ctxKey = Object.keys(info).find(k => k.endsWith('.context_length'));
      return ctxKey ? Number(info[ctxKey]) : null;
    }

    // OpenAI-compatible engines: vLLM puts the served window on the model entry.
    const models = await getJSON(`${host}/v1/models`);
    const list = Array.isArray(models.data) ? models.data : [];
    const entry = list.find(m => m.id === model) || list[0];
    const fromModels = entry && (entry.max_model_len || entry.context_length || entry.max_context_length);
    if (fromModels) return Number(fromModels);

    if (p === 'llamacpp') {
      // llama.cpp does not put it on /v1/models; /props carries the loaded n_ctx.
      const props = await getJSON(`${host}/props`);
      const n = props?.default_generation_settings?.n_ctx || props?.n_ctx;
      if (n) return Number(n);
    }
    return null;
  } catch (err) {
    console.warn(`[ModelContext] Could not read the window from ${p} @ ${host}: ${err.message}`);
    return null;
  }
}

/**
 * Probe (or reuse a recent probe) and cache. Concurrent callers share one
 * in-flight request. Safe to await on the chat path: local HTTP, 2s timeout,
 * and any failure lands on the table.
 */
async function ensureProbed(provider, host, model) {
  const k = key(provider, host, model);
  const hit = cache.get(k);
  if (hit && Date.now() - hit.at < PROBE_TTL_MS) return hit;
  if (inflight.has(k)) return inflight.get(k);

  const run = (async () => {
    const limit = await probeEngine(provider, host, model);
    const entry = limit && limit > 0
      ? { limit, at: Date.now(), source: 'engine' }
      : { limit: tableLimit(model), at: Date.now(), source: 'table' };
    // Report the CHANGE, not the state: a line every turn would be wallpaper.
    const prev = cache.get(k);
    if (!prev || prev.limit !== entry.limit || prev.source !== entry.source) {
      console.log(`[ModelContext] ${model} @ ${provider}: ${entry.source === 'engine'
        ? `engine reports ${entry.limit} tokens` : `no engine answer — table says ${entry.limit} tokens`}` +
        `${entry.source === 'engine' && entry.limit !== tableLimit(model)
          ? ` (the static table would have said ${tableLimit(model)})` : ''}`);
    }
    cache.set(k, entry);
    inflight.delete(k);
    return entry;
  })();
  inflight.set(k, run);
  return run;
}

/**
 * The hard ceiling for a model: the engine's answer if one has been cached,
 * otherwise the table. Synchronous — callers on a hot path never wait.
 */
function engineLimit(model, provider = null, host = null) {
  if (provider || host) {
    const hit = cache.get(key(provider, host, model));
    if (hit) return hit.limit;
  }
  for (const [k, v] of cache) if (k.endsWith(`|${model}`)) return v.limit;
  return tableLimit(model);
}

/**
 * The window the app will actually use: the configured usable window, never
 * above what the engine will serve.
 */
function usableWindow(model, provider = null, host = null) {
  const cfg = getConfig();
  const configured = Number(cfg.memory?.contextTokens) || 32768;
  const ceiling = engineLimit(model, provider, host);
  return Math.max(1024, Math.min(configured, ceiling));
}

/** Boot-time probe of the configured chat model, so turn one already knows. */
async function startupProbe() {
  try {
    const cfg = getConfig();
    const chat = cfg.models?.chat;
    if (!chat) return null;
    const { getProviderInstance } = require('./config');
    const inst = getProviderInstance(chat.provider, chat.instance);
    const host = inst ? inst.host : null;
    const entry = await ensureProbed(chat.provider, host, chat.model);
    const usable = usableWindow(chat.model, chat.provider, host);
    console.log(`  - Context window: ${usable} tokens usable (engine ceiling ${entry.limit}, source: ${entry.source}` +
      `, configured memory.contextTokens ${cfg.memory?.contextTokens})`);
    return { ...entry, usable };
  } catch (err) {
    console.error('[ModelContext] startup probe failed:', err.message);
    return null;
  }
}

module.exports = {
  tableLimit, probeEngine, ensureProbed, engineLimit, usableWindow, startupProbe,
  _cache: cache,
};
