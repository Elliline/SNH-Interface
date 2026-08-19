/**
 * The search providers behind web_search.
 *
 * ONE TOOL, TWO PROVIDERS, ROUTING IN CODE. The model sees a single `web_search`
 * function and never chooses a provider — the order is a config list, tried in
 * turn, and the first provider that comes back with results wins. That is
 * deliberate: a provider choice offered to the model is a decision it would have
 * to make on every search with nothing to base it on, and it would double the
 * tool schema for a question that has one right answer (try the fast one first,
 * fall back when it fails).
 *
 * Each provider is a plain function with ONE shape of return value, so the tool
 * above it never asks which provider it is holding:
 *
 *   { ok: boolean, results: [{title, url, snippet}], error: string|null,
 *     meta: {…}, latencyMs: number }
 *
 * `ok: true` with an empty `results` array is a REAL ANSWER — the provider
 * worked and the web had nothing — and it is distinguished from `ok: false`,
 * which means the provider itself failed. The fallback treats both as a reason
 * to try the next provider, and the LOG keeps them apart, because "Exa returned
 * nothing" and "Exa was down" are different facts and today the difference cost
 * hours.
 *
 * EXA'S SEARCH ENDPOINT ONLY. `type` is pinned to a non-agentic value and any
 * `deep*` type is refused in code (see `resolveExaType`). Deep Search and the
 * Agent endpoint do the multi-step research SNH is supposed to do itself with
 * its own tools, on its own GPU, in its own memory — buying that from an API
 * would move the thinking off the machine, and it also bills per step against a
 * $10/month ceiling.
 */

/** Snippets are for a model's context window, not for reading. */
const SNIPPET_CHARS = 300;

/**
 * Exa search types this may use. `deep`, `deep-lite` and `deep-reasoning` are
 * the agentic ones and are NOT here — a rule in code rather than a note in a
 * comment, because a config file is exactly where someone would try one.
 */
const EXA_ALLOWED_TYPES = ['auto', 'fast', 'instant'];

function resolveExaType(configured) {
  const t = String(configured || 'auto').trim().toLowerCase();
  if (EXA_ALLOWED_TYPES.includes(t)) return { type: t, refused: null };
  return {
    type: 'auto',
    refused: `search type "${t}" is not one this uses (${EXA_ALLOWED_TYPES.join(', ')}) — ran as "auto" instead`
  };
}

/** Trim to a snippet without pretending a truncated paragraph is a whole one. */
function snippet(text) {
  const s = String(text || '').replace(/\s+/g, ' ').trim();
  return s.length > SNIPPET_CHARS ? `${s.slice(0, SNIPPET_CHARS - 1)}…` : s;
}

/**
 * Exa — POST https://api.exa.ai/search, key in the `x-api-key` header.
 *
 * The key comes from the environment (EXA_API_KEY) and never from
 * data/config.json: that file is read by routes, written by the settings UI, and
 * copied to staging seeds. A secret in it would leak through all three.
 *
 * @param {string} query
 * @param {number} numResults
 * @param {Object} cfg - { url, apiKey, type, timeoutMs, textChars }
 */
async function exaSearch(query, numResults, cfg) {
  const t0 = Date.now();
  const fail = (error, meta = {}) => ({ ok: false, results: [], error, meta, latencyMs: Date.now() - t0 });

  if (!cfg || !cfg.apiKey) {
    return fail('no EXA_API_KEY in the environment — Exa was not called', { skipped: true });
  }

  const { type, refused } = resolveExaType(cfg.type);

  try {
    const response = await fetch(cfg.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': cfg.apiKey },
      body: JSON.stringify({
        query,
        type,
        numResults,
        // Text is what makes a result usable as a snippet. Capped, because the
        // whole page would be billed for and then thrown away by the trim below.
        contents: { text: { maxCharacters: cfg.textChars || 1000 } }
      }),
      signal: AbortSignal.timeout(cfg.timeoutMs || 8000)
    });

    if (!response.ok) {
      // 402 is the one that matters operationally: the free tier has no payment
      // method on file, so it STOPS rather than billing. Said in words rather
      // than as a status code, because this is the failure that will look like
      // "search is broken" six weeks from now.
      const body = await response.text().catch(() => '');
      const why = response.status === 402
        ? 'Exa refused the call with 402 Payment Required — the monthly credit is spent, or the rate limit was hit. Nothing was billed.'
        : `Exa returned HTTP ${response.status}`;
      return fail(why, { status: response.status, body: body.slice(0, 200), refusedType: refused });
    }

    const data = await response.json();
    const raw = Array.isArray(data && data.results) ? data.results : [];
    const results = raw.slice(0, numResults).map(r => ({
      title: r.title || '',
      url: r.url || '',
      snippet: snippet(r.text || r.summary || (Array.isArray(r.highlights) ? r.highlights.join(' ') : '')),
      publishedDate: r.publishedDate || null
    }));

    return {
      ok: true,
      results,
      error: null,
      meta: {
        type,
        refusedType: refused,
        requestId: (data && data.requestId) || null,
        // Read defensively: it is worth having on a metered account, and it is
        // not worth failing a search over if the field ever moves.
        costUsd: (data && data.costDollars && (data.costDollars.total ?? null)) || null
      },
      latencyMs: Date.now() - t0
    };
  } catch (error) {
    return fail(`Exa call failed: ${error.message}`, { refusedType: refused });
  }
}

/**
 * SearXNG — the self-hosted fallback, unchanged in behaviour from when it was
 * the only provider. GET /search?q=…&format=json against a local instance.
 *
 * @param {string} query
 * @param {number} numResults
 * @param {Object} cfg - { url, timeoutMs }
 */
async function searxngSearch(query, numResults, cfg) {
  const t0 = Date.now();
  const fail = (error, meta = {}) => ({ ok: false, results: [], error, meta, latencyMs: Date.now() - t0 });

  if (!cfg || !cfg.url) return fail('no SearXNG url configured', { skipped: true });

  try {
    const searchUrl = `${cfg.url}/search?q=${encodeURIComponent(query)}&format=json`;
    const response = await fetch(searchUrl, { signal: AbortSignal.timeout(cfg.timeoutMs || 8000) });
    if (!response.ok) return fail(`SearXNG returned HTTP ${response.status}`, { status: response.status });

    const data = await response.json();
    const raw = Array.isArray(data && data.results) ? data.results : [];
    const results = raw.slice(0, numResults).map(r => ({
      title: r.title || '',
      url: r.url || '',
      snippet: snippet(r.content || ''),
      publishedDate: r.publishedDate || null
    }));
    return { ok: true, results, error: null, meta: { url: cfg.url }, latencyMs: Date.now() - t0 };
  } catch (error) {
    return fail(`SearXNG call failed: ${error.message}`, {});
  }
}

/** name → provider function. The order they are tried in is config, not here. */
const PROVIDERS = {
  exa: exaSearch,
  searxng: searxngSearch
};

/**
 * WHAT THE SETTINGS PAGE SHOWS FOR EACH PROVIDER — declared here, beside the
 * provider it describes, for the same reason the tool catalogue lives beside
 * registration: a page with its own list of providers is a list that falls behind.
 * Add a provider to PROVIDERS and to this array and it appears in the UI, with its
 * own switch, its own place in the order, and its own key field if it needs one.
 *
 * `secret` is the env-style name db/secrets.js stores it under. Declaring it here
 * is what makes the key field appear; nothing in the page knows about Exa.
 */
const SEARCH_PROVIDER_SPECS = [
  {
    id: 'exa',
    label: 'Exa',
    blurb: 'Hosted search index. Fast, and it returns page text with each result.',
    toggle: 'tools.exa.enabled',
    secret: {
      env: 'EXA_API_KEY',
      label: 'Exa API key',
      hint: 'From exa.ai. Stored encrypted on this machine and never shown again after saving. The free tier stops with a 402 rather than billing.'
    },
    fields: [
      { path: 'tools.exa.numResults', label: 'Results per search', type: 'number', min: 1, max: 25 },
      { path: 'tools.exa.timeoutMs', label: 'Timeout (ms)', type: 'number', min: 1000, max: 30000 },
      { path: 'tools.exa.url', label: 'Endpoint', type: 'text', placeholder: 'https://api.exa.ai/search',
        hint: 'Only change this for a proxy. The Search endpoint only — deep/agentic types are refused in code.' }
    ]
  },
  {
    id: 'searxng',
    label: 'SearXNG',
    blurb: 'Your own instance. No account, no metering, and it keeps working when a hosted index does not.',
    toggle: 'tools.searxng.enabled',
    fields: [
      { path: 'tools.searxng.url', label: 'Instance URL', type: 'text', placeholder: 'http://localhost:8888' }
    ]
  }
];

module.exports = {
  PROVIDERS, SEARCH_PROVIDER_SPECS,
  exaSearch, searxngSearch, resolveExaType, EXA_ALLOWED_TYPES, SNIPPET_CHARS
};
