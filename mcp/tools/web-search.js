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
      return { error: WebSearchTool.HONESTY.noProvider(why) };
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

    // NOBODY SERVED IT — AND THERE ARE TWO WAYS THAT HAPPENS.
    //
    // Measured 2026-08-18, on this very code: with SearXNG pointed at a dead port
    // a job reported to Ellie that "the search tools returned no results for
    // specific pricing or listings". Nothing had been searched. Both providers
    // had FAILED, and the aggregate return said `results: []` with a message
    // about finding nothing — so the same rule this file keeps between providers
    // ("empty and broken are different facts") was being broken one level up, at
    // the only place the model actually reads.
    //
    // So the two are split here too:
    //   at least one provider RAN and found nothing → an empty answer about the
    //     world, returned as a result, and he should report finding nothing.
    //   every provider failed or was unavailable → NO SEARCH HAPPENED, returned
    //     as an error, and he must say the search could not run. This also bills
    //     as an error rather than as an empty search, which is the honest price.
    const summary = tried.map(t => `${t.provider} (${t.outcome}${t.detail ? `: ${t.detail}` : ''})`).join(', ');
    const anyProviderRan = tried.some(t => t.outcome === 'empty');
    const namesTried = tried.map(t => `${t.provider}:${t.outcome}`);

    if (!anyProviderRan) {
      opsLog(`Web search COULD NOT RUN for "${String(query).slice(0, 120)}" [${caller}]. Providers: ${summary}.`);
      return {
        error: `The search did not run: ${summary}. This is NOT "no results found" — nothing was searched, ` +
          `so nothing about the world follows from it. Say that you could not search, and do not report this ` +
          `as having found nothing.`,
        results: [],
        provider: null,
        providers_tried: namesTried
      };
    }

    opsLog(`Web search found NOTHING for "${String(query).slice(0, 120)}" [${caller}]. Providers: ${summary}.`);
    return {
      results: [],
      provider: null,
      providers_tried: namesTried,
      message: WebSearchTool.HONESTY.noResults(summary)
    };
  }
}

/**
 * THE TWO THINGS THE SEARCH TOOL SAYS WHEN IT FOUND NOTHING, AND WHY THEY ARE
 * NAMED.
 *
 * Both exist to stop one failure: answering from memory as though a search had
 * happened. "No provider" and "a real empty answer" are different facts and
 * must not be reported as each other — an empty result set is evidence, an
 * absent provider is not. Exported so the suites asserting that each is said
 * plainly read the sentence from here; a copy in a test turns a reword into a
 * red build and lets a message that stopped warning off confabulation pass.
 */
WebSearchTool.HONESTY = {
  // The warnings are named apart from the sentences that carry them, because the
  // warning is the invariant: what precedes it is a provider list or a reason
  // code that varies per call, and a test cannot reconstruct either.
  NO_PROVIDER_WARNING:
    'Say you could not search rather than answering from memory as though you had.',
  NO_RESULTS_WARNING:
    'This is a real empty answer — report that you found nothing rather than filling it in from memory.',
  noProvider: (why) =>
    `No search provider is available (${why}). ${WebSearchTool.HONESTY.NO_PROVIDER_WARNING}`,
  noResults: (summary) =>
    `No results. Searched with: ${summary}. ${WebSearchTool.HONESTY.NO_RESULTS_WARNING}`,
};

module.exports = WebSearchTool;
