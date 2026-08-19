/**
 * web_search — one tool, a provider chain behind it.
 *
 * Replaces mcp/tools/searxng.js (2026-08-18). Two things changed and both were
 * load-bearing:
 *
 * 1. THE SIGNATURE IS NOW (args, context), like every other tool.
 *
 *    The old one was `execute(args, endpointOverride)` — a positional STRING
 *    where every other tool in the registry takes a context object. On
 *    2026-08-18 that difference broke the agent worker: the chat path passed
 *    { searxngHost } and worked, the background tool loop passed { caller } and
 *    did not, and the context object went into a parameter used as a URL base:
 *
 *        Search failed: Failed to parse URL from [object Object]/search?q=…
 *
 *    Seven of those inside one job in 11 seconds, reported to Ellie as "an issue
 *    with the search tool", which was all the model could see. It was patched at
 *    the call site in MCPClient.executeTool; that special case is now GONE,
 *    because a shape that has to be remembered at one call site is a shape that
 *    will be forgotten at the next. `context.searxngHost` is still honoured as an
 *    override — as a named field of the context object, where a string cannot be
 *    confused for one.
 *
 * 2. THE PROVIDER IS RESOLVED HERE, from config, on every call.
 *
 *    Exa first, SearXNG as the fallback, order from config.tools.search.order.
 *    The tool asks each provider in turn and returns the FIRST one that comes
 *    back with results. Chat and agent jobs get the same chain for the same
 *    reason — one of them being quietly slower or quietly stale is precisely the
 *    thing nobody would notice.
 *
 * A FALLBACK IS NOT A SILENT FALLBACK. Every attempt writes a row to
 * search_call_log with its provider, query, count and outcome, and a fallback
 * also writes an ops line: "which provider served this, and did it find
 * anything" has to be answerable from data afterwards, because on 2026-08-18 it
 * was not, and that cost hours.
 *
 * BOTH EMPTY IS SAID PLAINLY. When no provider returns anything, the result says
 * which ones were tried and that the answer is nothing — never a shrug the model
 * can read as permission to fill the gap from memory.
 */

const { randomUUID } = require('crypto');
const path = require('path');
const { PROVIDERS } = require('./search-providers');

// Read through the module object rather than destructuring at load time — the
// same rule db/agent-jobs.js follows, for the same two reasons: the config seen
// here is always the one the process currently holds (a settings change reaches
// it without a restart), and a test can substitute one without writing to the
// live data/config.json, which is deliberately NOT redirected by SNH_DATA_DIR.
function searchConfig() { return require('../../db/config').getSearchConfig(); }
function logSearchCall(row) { return require('../../db/search-log').logSearchCall(row); }

/** Ops telemetry — resolved per call from the PROCESS's data dir, never a constant. */
function opsLog(msg) {
  try {
    const { getDataDir } = require('../../db/database');
    require('../../db/fact-extractor').appendToOpsLog(msg, path.join(getDataDir(), 'memory', 'ops'));
  } catch { /* console is the floor */ }
}

class WebSearchTool {
  constructor() {
    this.name = 'web_search';
    // Provider-agnostic on purpose: the model does not choose a provider and
    // must not start reasoning about one. What it needs to know is that this
    // searches the live web.
    this.description = 'Search the web for current information. Use this when you need up-to-date information, current events, news, or facts you are unsure about.';
    this.parameters = {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The search query'
        },
        num_results: {
          type: 'number',
          description: 'Number of results to return (default 5)'
        }
      },
      required: ['query']
    };
  }

  getOpenAIFunctionSpec() {
    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.description,
        parameters: this.parameters
      }
    };
  }

  /**
   * @param {Object} args - { query: string, num_results?: number }
   * @param {Object} context - the standard tool context. Read here:
   *   `caller` for the log, and `searxngHost` (a STRING) as an endpoint override
   *   for the SearXNG provider only.
   * @returns {Promise<Object>} { results, provider, providers_tried } or { error }
   */
  async execute(args, context = {}) {
    const query = args && args.query;
    if (!query || typeof query !== 'string') {
      return { error: 'Missing or invalid search query' };
    }
    const numResults = Math.min(Math.max(parseInt(args && args.num_results) || 5, 1), 20);
    const caller = (context && typeof context.caller === 'string' && context.caller) || 'chat';

    const chain = searchConfig();
    if (!chain.providers.length || !chain.any) {
      const why = chain.providers.map(p => `${p.name}: ${p.why}`).join('; ') || 'no providers configured';
      logSearchCall({ provider: 'none', query, numResults: 0, outcome: 'skipped', detail: why, caller });
      return { error: `No search provider is available (${why}). Say you could not search rather than answering from memory as though you had.` };
    }

    // One id for the whole tool call, so the rows for Exa-then-SearXNG read as
    // one attempt with two steps rather than two unrelated searches.
    const attemptId = randomUUID();
    const tried = [];

    for (const p of chain.providers) {
      if (!p.available) {
        logSearchCall({ provider: p.name, query, numResults: 0, outcome: 'skipped', detail: p.why, caller, attemptId });
        tried.push({ provider: p.name, outcome: 'skipped', detail: p.why });
        continue;
      }

      // The SearXNG endpoint override, still honoured, now as a named field.
      const pCfg = p.name === 'searxng' && typeof context.searxngHost === 'string' && context.searxngHost
        ? { ...p.config, url: context.searxngHost }
        : p.config;

      const res = await PROVIDERS[p.name](query, numResults, pCfg);
      const outcome = !res.ok ? 'error' : (res.results.length ? 'results' : 'empty');
      const served = outcome === 'results';

      logSearchCall({
        provider: p.name, query, numResults: res.results.length, outcome,
        detail: res.error || (res.meta && res.meta.refusedType) || null,
        caller, latencyMs: res.latencyMs, attemptId, served,
        costUsd: (res.meta && res.meta.costUsd) || null
      });
      tried.push({ provider: p.name, outcome, detail: res.error || null, results: res.results.length });

      const label = `[Search] ${p.name} "${String(query).slice(0, 60)}" → ${outcome}` +
        `${outcome === 'results' ? ` (${res.results.length})` : ''}` +
        `${res.error ? ` — ${res.error}` : ''} [${caller}, ${Math.round(res.latencyMs)}ms]`;
      console.log(label);

      if (served) {
        // A fallback that WORKED is still news: it means the provider before it
        // is failing, and nothing else would say so.
        if (tried.length > 1) {
          opsLog(`Web search fell back to ${p.name} and got ${res.results.length} result(s). ` +
            `Tried first: ${tried.slice(0, -1).map(t => `${t.provider} (${t.outcome}${t.detail ? `: ${t.detail}` : ''})`).join(', ')}. Query: "${String(query).slice(0, 120)}"`);
        }
        return {
          results: res.results,
          provider: p.name,
          providers_tried: tried.map(t => `${t.provider}:${t.outcome}`)
        };
      }
    }

    // Nobody had anything. Say that, and say who was asked — never a bare empty
    // array, which reads as "search is not working" or as nothing at all.
    const summary = tried.map(t => `${t.provider} (${t.outcome}${t.detail ? `: ${t.detail}` : ''})`).join(', ');
    opsLog(`Web search found NOTHING for "${String(query).slice(0, 120)}" [${caller}]. Providers: ${summary}.`);
    return {
      results: [],
      provider: null,
      providers_tried: tried.map(t => `${t.provider}:${t.outcome}`),
      message: `No results. Searched with: ${summary}. This is a real empty answer — report that you found nothing rather than filling it in from memory.`
    };
  }
}

module.exports = WebSearchTool;
