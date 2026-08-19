#!/usr/bin/env node
/**
 * The search provider chain: Exa first, SearXNG as the fallback, and both of
 * those facts recorded rather than assumed.
 *
 * Every rule asserted here has a failure mode that is INVISIBLE while it happens
 * and expensive afterwards. A silent fallback looks exactly like a fast primary.
 * An empty result looks exactly like a broken provider. A tool whose signature
 * differs from its neighbours' works on the one path someone tested and hands
 * "[object Object]" to the other — which is precisely what happened to the agent
 * worker on 2026-08-18.
 *
 * Runs against a throwaway SNH_DATA_DIR and never touches the live corpus.
 * `fetch` is stubbed, so no provider is really called and no Exa credit is spent.
 * The search config is stubbed on db/config's module object — data/config.json is
 * deliberately NOT redirected by SNH_DATA_DIR and a test must never write to it.
 *
 * Usage: node scripts/test-search-providers.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-search-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const config = require(path.join(ROOT, 'db/config'));
const searchLog = require(path.join(ROOT, 'db/search-log'));
const WebSearchTool = require(path.join(ROOT, 'mcp/tools/web-search'));
const { resolveExaType } = require(path.join(ROOT, 'mcp/tools/search-providers'));
// The cost rule lives with the budget; asserted here because "an unreachable
// provider must not bill as an empty search" is a fact about this file's output.
const mm = require(path.join(ROOT, 'db/memory-manager'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- the config stub ------------------------------------------------------
// Both providers available by default; each case narrows it.
function stubChain({ exa = true, searxng = true } = {}) {
  config.getSearchConfig = () => {
    const providers = [
      {
        name: 'exa',
        available: exa,
        config: { url: 'https://api.exa.ai/search', apiKey: exa ? 'test-key' : null, type: 'auto', timeoutMs: 2000, textChars: 400 },
        why: exa ? null : 'no EXA_API_KEY in the environment'
      },
      {
        name: 'searxng',
        available: searxng,
        config: { url: 'http://searxng.test', timeoutMs: 2000 },
        why: searxng ? null : 'config.tools.searxng.enabled is false'
      }
    ];
    return { any: providers.some(p => p.available), order: providers.map(p => p.name), providers };
  };
}

// --- the fetch stub -------------------------------------------------------
// Answers per host, and RECORDS what it was asked, because "did Exa get called
// at all" is one of the things under test.
const calls = [];
function stubFetch({ exa, searxng }) {
  global.fetch = async (url, opts = {}) => {
    const u = String(url);
    const which = u.includes('api.exa.ai') ? 'exa' : 'searxng';
    calls.push({ which, url: u, body: opts.body ? JSON.parse(opts.body) : null, headers: opts.headers || {} });
    const plan = which === 'exa' ? exa : searxng;
    if (typeof plan === 'function') return plan();
    return plan;
  };
}
const ok = (json) => ({ ok: true, status: 200, json: async () => json, text: async () => JSON.stringify(json) });
const httpErr = (status) => ({ ok: false, status, json: async () => ({}), text: async () => `status ${status}` });
const boom = (msg) => () => { throw new Error(msg); };

const EXA_TWO = ok({
  requestId: 'req-1',
  costDollars: { total: 0.005 },
  results: [
    { title: 'Exa hit one', url: 'https://one.test', text: 'first exa result text', publishedDate: '2026-08-01' },
    { title: 'Exa hit two', url: 'https://two.test', text: 'second exa result text' }
  ]
});
const EXA_EMPTY = ok({ requestId: 'req-2', results: [] });
const SX_THREE = ok({
  results: [
    { title: 'SearXNG hit one', url: 'https://sx1.test', content: 'sx one' },
    { title: 'SearXNG hit two', url: 'https://sx2.test', content: 'sx two' },
    { title: 'SearXNG hit three', url: 'https://sx3.test', content: 'sx three' }
  ]
});
const SX_EMPTY = ok({ results: [] });

const tool = new WebSearchTool();
function rowsFor(query) {
  return db.prepare('SELECT * FROM search_call_log WHERE query = ? ORDER BY datetime(created_at) ASC, rowid ASC').all(query);
}

(async () => {
  console.log('\n=== The search chain ===\n');

  // ---------------------------------------------------------------------
  console.log('1. Exa first, and SearXNG is not called when Exa answers');
  // The routing claim itself. A chain that "prefers" Exa but calls both is
  // spending a credit AND the local instance's time on every search.
  stubChain();
  calls.length = 0;
  stubFetch({ exa: EXA_TWO, searxng: SX_THREE });
  let out = await tool.execute({ query: 'exa first', num_results: 5 }, { caller: 'chat' });
  check('the served provider is exa', out.provider === 'exa', out.provider);
  check('its results come back', out.results.length === 2 && out.results[0].url === 'https://one.test');
  check('the snippet is built from Exa\'s text field', /first exa result text/.test(out.results[0].snippet));
  check('searxng was never called', !calls.some(c => c.which === 'searxng'), JSON.stringify(calls.map(c => c.which)));
  check('the key rides in the x-api-key header', calls[0].headers['x-api-key'] === 'test-key');
  check('it is a POST to the /search endpoint', /api\.exa\.ai\/search$/.test(calls[0].url));

  let rows = rowsFor('exa first');
  check('one row per attempt — exactly one here', rows.length === 1, `${rows.length}`);
  check('the row names the provider, count and outcome',
    rows[0].provider === 'exa' && rows[0].num_results === 2 && rows[0].outcome === 'results',
    JSON.stringify(rows[0]));
  check('and it is marked as the one that served the call', rows[0].served === 1);
  check('the caller is recorded', rows[0].caller === 'chat', rows[0].caller);
  check('the reported cost is kept — the account is metered', rows[0].cost_usd === 0.005, `${rows[0].cost_usd}`);

  // ---------------------------------------------------------------------
  console.log('\n2. Exa FAILS → SearXNG serves the call');
  // The whole point of keeping SearXNG. A failure here must not fail the turn.
  stubChain();
  calls.length = 0;
  stubFetch({ exa: httpErr(500), searxng: SX_THREE });
  out = await tool.execute({ query: 'exa broken' }, { caller: 'agent-job:abc12345' });
  check('the fallback served it', out.provider === 'searxng', out.provider);
  check('the results are real', out.results.length === 3);
  check('and the turn was not failed', !out.error);
  check('both providers are named in the result', (out.providers_tried || []).join(',') === 'exa:error,searxng:results',
    JSON.stringify(out.providers_tried));

  rows = rowsFor('exa broken');
  check('two rows — the failure is kept, not overwritten', rows.length === 2, `${rows.length}`);
  check('the exa row says error, with the reason', rows[0].provider === 'exa' && rows[0].outcome === 'error' && /500/.test(rows[0].detail || ''),
    JSON.stringify(rows[0]));
  check('the exa row is NOT marked as serving', rows[0].served === 0);
  check('the searxng row is', rows[1].provider === 'searxng' && rows[1].served === 1);
  check('both rows share one attempt id — one tool call, two steps',
    rows[0].attempt_id && rows[0].attempt_id === rows[1].attempt_id);
  check('the job that asked is on the row', rows[1].caller === 'agent-job:abc12345', rows[1].caller);

  // ---------------------------------------------------------------------
  console.log('\n3. A 402 from Exa is said in words, not left as a status code');
  // The free tier stops rather than billing. Six weeks from now this will look
  // like "search is broken", and the log line is what prevents that.
  stubChain();
  stubFetch({ exa: httpErr(402), searxng: SX_THREE });
  out = await tool.execute({ query: 'exa spent' }, { caller: 'chat' });
  check('it still falls back', out.provider === 'searxng');
  rows = rowsFor('exa spent');
  check('and the row explains the 402 in plain words',
    /credit is spent|Payment Required/i.test(rows[0].detail || ''), rows[0].detail);
  check('and says nothing was billed', /nothing was billed/i.test(rows[0].detail || ''), rows[0].detail);

  // ---------------------------------------------------------------------
  console.log('\n4. Exa EMPTY also falls through — empty is not "served"');
  // A provider that worked and found nothing is a provider that did not answer
  // the question. The next one gets a turn.
  stubChain();
  calls.length = 0;
  stubFetch({ exa: EXA_EMPTY, searxng: SX_THREE });
  out = await tool.execute({ query: 'exa quiet' }, { caller: 'chat' });
  check('searxng serves it', out.provider === 'searxng', out.provider);
  check('searxng really was called', calls.some(c => c.which === 'searxng'));
  rows = rowsFor('exa quiet');
  check('the exa row reads empty, not error', rows[0].outcome === 'empty' && rows[0].num_results === 0);

  // ---------------------------------------------------------------------
  console.log('\n5. BOTH empty is stated plainly, never a shrug');
  // The invention risk. An empty array with no words around it is what a model
  // fills in from memory.
  stubChain();
  stubFetch({ exa: EXA_EMPTY, searxng: SX_EMPTY });
  out = await tool.execute({ query: 'nobody knows' }, { caller: 'chat' });
  check('no results', Array.isArray(out.results) && out.results.length === 0);
  check('no provider is claimed to have served it', out.provider === null);
  check('the message names both providers tried', /exa/.test(out.message) && /searxng/.test(out.message), out.message);
  check('and tells him not to fill it in from memory', /rather than filling it in from memory/i.test(out.message), out.message);
  rows = rowsFor('nobody knows');
  check('both attempts are on the record', rows.length === 2 && rows.every(r => r.outcome === 'empty'));
  check('neither is marked as serving', rows.every(r => r.served === 0));

  // ---------------------------------------------------------------------
  console.log('\n5b. EVERY PROVIDER BROKEN is not "no results" — it is no search');
  // Measured live on this code before it was fixed: with SearXNG on a dead port,
  // a job told Ellie "the search tools returned no results for specific pricing".
  // Nothing had been searched. The distinction this file keeps between providers
  // was being lost in the aggregate return, which is the only part the model reads.
  stubChain();
  stubFetch({ exa: httpErr(500), searxng: boom('connect ECONNREFUSED 127.0.0.1:9') });
  out = await tool.execute({ query: 'nothing ran' }, { caller: 'chat' });
  check('it comes back as an ERROR, not as an empty result', !!out.error, JSON.stringify(out).slice(0, 200));
  check('it says explicitly that this is not "no results found"',
    /NOT "no results found"/.test(out.error), out.error);
  check('and tells him nothing about the world follows from it',
    /nothing about the world follows/.test(out.error), out.error);
  check('both failures are still named', /exa/.test(out.error) && /searxng/.test(out.error));
  check('an error bills as an error, not as an empty search',
    mm.toolCallCost('web_search', out, 0.25).why === 'the call returned an error',
    mm.toolCallCost('web_search', out, 0.25).why);

  // One provider RAN and found nothing → that IS an answer about the world.
  stubChain();
  stubFetch({ exa: httpErr(500), searxng: SX_EMPTY });
  out = await tool.execute({ query: 'one ran and found nothing' }, { caller: 'chat' });
  check('a provider that ran and found nothing is still an empty RESULT, not an error',
    !out.error && Array.isArray(out.results) && out.results.length === 0, JSON.stringify(out).slice(0, 200));
  check('and it is reported as a real empty answer', /real empty answer/.test(out.message || ''), out.message);

  // ---------------------------------------------------------------------
  console.log('\n6. A provider with no prerequisite is SKIPPED, and says why');
  stubChain({ exa: false, searxng: true });
  stubFetch({ exa: boom('exa must not be called without a key'), searxng: SX_THREE });
  out = await tool.execute({ query: 'no key' }, { caller: 'chat' });
  check('searxng serves it', out.provider === 'searxng');
  rows = rowsFor('no key');
  check('the exa row says skipped, with the reason',
    rows[0].outcome === 'skipped' && /EXA_API_KEY/.test(rows[0].detail || ''), JSON.stringify(rows[0]));

  // ---------------------------------------------------------------------
  console.log('\n7. No provider at all → an error that forbids answering as if searched');
  stubChain({ exa: false, searxng: false });
  out = await tool.execute({ query: 'nothing available' }, { caller: 'chat' });
  check('it is an error, not an empty success', !!out.error, JSON.stringify(out));
  check('and it says to admit the search did not happen',
    /rather than answering from memory/i.test(out.error), out.error);

  // ---------------------------------------------------------------------
  console.log('\n8. Deep/agentic search types are refused in code');
  // Ellie\'s rule: the Search endpoint only. Deep Search and the Agent endpoint
  // do the multi-step research SNH is supposed to do itself.
  check('"deep" is refused and downgraded', resolveExaType('deep').type === 'auto' && !!resolveExaType('deep').refused);
  check('"deep-reasoning" too', resolveExaType('deep-reasoning').type === 'auto');
  check('"auto" is left alone', resolveExaType('auto').type === 'auto' && !resolveExaType('auto').refused);
  check('"fast" is allowed', resolveExaType('fast').type === 'fast');

  stubChain();
  calls.length = 0;
  config.getSearchConfig = ((real) => () => {
    const c = real();
    c.providers[0].config.type = 'deep';       // as if someone set it in config
    return c;
  })(config.getSearchConfig);
  stubFetch({ exa: EXA_TWO, searxng: SX_THREE });
  out = await tool.execute({ query: 'deep refused' }, { caller: 'chat' });
  check('a configured "deep" never reaches the wire', calls[0].body.type === 'auto', calls[0].body.type);
  rows = rowsFor('deep refused');
  check('and the refusal is recorded', /not one this uses/.test(rows[0].detail || ''), rows[0].detail);

  // ---------------------------------------------------------------------
  console.log('\n9. THE SIGNATURE: (args, context), like every other tool');
  // The 2026-08-18 bug, as a test. The background loop passes { caller } and no
  // endpoint; under the old positional shape that object became a URL base and
  // produced "Failed to parse URL from [object Object]/search?q=…". Nothing in
  // the chain may reshape the second argument.
  stubChain({ exa: false, searxng: true });
  calls.length = 0;
  stubFetch({ exa: boom('not called'), searxng: SX_THREE });
  out = await tool.execute({ query: 'context object' }, { caller: 'heartbeat:corrector' });
  check('a context object with no endpoint still searches', out.results.length === 3, JSON.stringify(out));
  check('and the url is the configured instance, not an object',
    calls.find(c => c.which === 'searxng').url.startsWith('http://searxng.test/search?'),
    calls.find(c => c.which === 'searxng').url);
  check('no "[object Object]" reached the wire', !calls.some(c => /\[object Object\]/.test(c.url)));

  calls.length = 0;
  out = await tool.execute({ query: 'host override' }, { caller: 'chat', searxngHost: 'http://other.test' });
  check('a string searxngHost in the context is still honoured',
    calls.find(c => c.which === 'searxng').url.startsWith('http://other.test/search?'),
    calls.find(c => c.which === 'searxng').url);

  // And the structural half: nothing may special-case web_search's arguments in
  // the registry any more, and no file may construct the deleted tool.
  const clientSrc = fs.readFileSync(path.join(ROOT, 'mcp/mcp-client.js'), 'utf8');
  const execBody = clientSrc.slice(clientSrc.indexOf('async executeTool'));
  check('executeTool no longer branches on a tool name',
    !/if\s*\(\s*toolName\s*===/.test(execBody.slice(0, execBody.indexOf('hasTools()'))),
    'a per-tool branch is back in executeTool');
  check('the old SearXNGTool module is gone', !fs.existsSync(path.join(ROOT, 'mcp/tools/searxng.js')));

  // ---------------------------------------------------------------------
  console.log('\n10. The summary answers the question that cost hours');
  const summary = searchLog.providerSummary({ hours: 24 });
  const exaRow = summary.find(r => r.provider === 'exa');
  const sxRow = summary.find(r => r.provider === 'searxng');
  check('exa attempts are counted with their outcomes split',
    !!exaRow && exaRow.attempts > 0 && exaRow.errors >= 2 && exaRow.empty >= 2, JSON.stringify(exaRow));
  check('so are searxng\'s', !!sxRow && sxRow.with_results >= 4, JSON.stringify(sxRow));
  check('recentSearchCalls reads back newest-first',
    searchLog.recentSearchCalls({ limit: 5 }).length === 5);

  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => {
  console.error('\nTEST HARNESS ERROR:', err);
  process.exit(1);
});
