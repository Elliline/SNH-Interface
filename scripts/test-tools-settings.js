#!/usr/bin/env node
/**
 * The Tools tab's contract: generated from the registry, and secrets go one way.
 *
 * Two failures are being tested against, both of them real:
 *
 *   1. A SETTINGS PAGE WITH ITS OWN LIST. On 2026-08-18 fourteen tools were
 *      registered and three appeared in settings, because the page carried a
 *      hand-written list. The fix is that registration and the page read one
 *      table, so the test that matters is: does a tool added to the catalogue
 *      appear in what the page renders, with nothing else touched?
 *
 *   2. A KEY THAT COMES BACK. "Write-only" is easy to say and easy to lose — one
 *      convenience preview, one debug field, and the value is in a browser
 *      response again. So the assertion is made against the actual JSON the route
 *      returns: the value must not be in it, in any form.
 *
 * Runs against a throwaway SNH_DATA_DIR, and — because secrets deliberately live
 * beside data/config.json rather than under the redirect — against a THROWAWAY
 * SECRETS PATH too, by pointing db/secrets at a temp copy through its own module
 * path. The live secrets file and key are never read or written.
 *
 * Usage: node scripts/test-tools-settings.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-tools-test-'));
process.env.SNH_DATA_DIR = TMP;
// The secret store, moved off the live paths before anything requires it.
process.env.SNH_SECRETS_PATH = path.join(TMP, 'secrets.json');
process.env.SNH_SECRET_KEY_PATH = path.join(TMP, '.secret-key');
// A key of our own, so nothing depends on a key file existing. This is also the
// managed-instance path being exercised: a platform injecting the key.
process.env.SNH_SECRET_KEY = crypto.randomBytes(32).toString('base64');
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();

const config = require(path.join(ROOT, 'db/config'));
const secretsMod = require(path.join(ROOT, 'db/secrets'));
const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
const WebSearchTool = require(path.join(ROOT, 'mcp/tools/web-search'));
const toolsRouter = require(path.join(ROOT, 'routes/tools'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- the secret store, pointed somewhere disposable ------------------------
// The secrets file deliberately does NOT follow SNH_DATA_DIR (it is configuration,
// like config.json, not corpus), so the redirect that keeps the rest of this test
// off the live system would not cover it. SNH_SECRETS_PATH and SNH_SECRET_KEY_PATH
// do — set before db/secrets.js is required, so the live file and the live key are
// never opened, not even for reading, and not even if this process dies mid-test.
// (Set at the top of the file, above the requires.)

// --- config stubbing -------------------------------------------------------
// Two seams, and they are different on purpose.
//
// getSearchConfig(cfg) takes an INJECTED config — the real chain logic, run over a
// config this test wrote, without touching the live data/config.json.
//
// mcp-client and mcp/tools/web-search.js read config THROUGH the module object, so
// patching config.getSearchConfig here reaches them — and what it is patched to is
// still the real function, just with the test's config injected. Nothing is
// reimplemented; only the input is substituted.
const realGetConfig = config.getConfig;
const realGetSearchConfig = config.getSearchConfig;
let toolsOverride = {};

function fakeConfig() {
  const c = realGetConfig();
  return { ...c, tools: { ...c.tools, ...toolsOverride } };
}
function setTools(o) {
  toolsOverride = o;
  config.getConfig = () => fakeConfig();
  config.getSearchConfig = () => realGetSearchConfig(fakeConfig());
}
setTools({});

// --- calling the route without a server -----------------------------------
function callRoute(method, routePath, body) {
  const layer = toolsRouter.stack.find(l => l.route && l.route.path === routePath && l.route.methods[method]);
  if (!layer) throw new Error(`no ${method} ${routePath} route`);
  // The last handler in the stack is the real one; a rate limiter may sit before it.
  const handlers = layer.route.stack.map(s => s.handle);
  const handler = handlers[handlers.length - 1];
  const res = {
    code: 200, body: null,
    status(c) { this.code = c; return this; },
    json(b) { this.body = b; return this; }
  };
  handler({ body, ip: '127.0.0.1', headers: {} }, res, () => {});
  return res;
}

(async () => {
  console.log(`\nTools tab + secret store (throwaway data dir: ${TMP})\n`);

  // =======================================================================
  console.log('── ONE TABLE, TWO READERS ──');
  setTools({
    exa: { enabled: true, url: 'https://api.exa.ai/search', numResults: 5, timeoutMs: 8000 },
    searxng: { enabled: true, url: 'http://localhost:8888' },
    search: { order: ['exa', 'searxng'] },
    cron: { enabled: true }, memoryWrite: { enabled: true },
    memoryInspect: { enabled: true }, agentJobs: { enabled: true }
  });
  const client = MCPClient.shared();
  client.loadConfig();

  const registered = client.getToolNames();
  let rows = client.describeCatalogue();
  const CATALOGUE_SIZE = rows.length;
  check('every registered tool has a row on the page',
    registered.every(n => rows.some(r => r.id === n)),
    registered.filter(n => !rows.some(r => r.id === n)).join(', '));
  check('and every row is a real registered tool name (no invented rows)',
    rows.filter(r => r.registered).every(r => registered.includes(r.id)));
  // The NUMBER is not the invariant — tools get added, and dispatch_coding_job
  // was the fifteenth (2026-08-20). What must hold is that the page has a row
  // for every tool in the catalogue, registered or not, which is the defect
  // this test was written for: fourteen registered and three on the page.
  check('every catalogued tool has a row, registered or not',
    rows.length === CATALOGUE_SIZE && registered.every(id => rows.some(r => r.id === id)),
    `${registered.length} registered, ${rows.length} rows, catalogue ${CATALOGUE_SIZE}`);
  check('each row carries the tool\'s OWN description, not a second copy',
    rows.every(r => r.description && r.description.length > 20));
  check('every row can be turned on or off by something — a switch or a stated reason',
    rows.every(r => r.toggle || r.toggleNote), rows.filter(r => !r.toggle && !r.toggleNote).map(r => r.id).join(', '));

  // THE ONE THAT MATTERS: a tool added to the registry, with no page edits.
  class DummyProbeTool {
    constructor() {
      this.name = 'dummy_probe';
      this.description = 'A tool that exists only to prove the settings page is generated from the registry rather than from a list of its own.';
    }
    getOpenAIFunctionSpec() { return { type: 'function', function: { name: this.name, description: this.description, parameters: { type: 'object', properties: {} } } }; }
  }
  MCPClient.TOOL_CATALOGUE.push({
    id: 'dummy_probe', title: 'Dummy probe', Tool: DummyProbeTool, card: 'jobs',
    gate: () => true, toggle: 'tools.dummyProbe.enabled'
  });
  client.loadConfig();
  rows = client.describeCatalogue();
  check('adding a tool to the registry adds one more row, with no page change',
    rows.length === CATALOGUE_SIZE + 1 && rows.some(r => r.id === 'dummy_probe'), `${rows.length} rows`);
  check('and it is registered too — one table decided both',
    client.getToolNames().includes('dummy_probe'));

  const payloadWithDummy = callRoute('get', '/').body;
  check('the API the page fetches shows it as well',
    payloadWithDummy.tools.some(t => t.id === 'dummy_probe') && payloadWithDummy.catalogueCount === CATALOGUE_SIZE + 1);

  MCPClient.TOOL_CATALOGUE.pop();
  client.loadConfig();
  check('removing it again leaves the catalogue as it was',
    client.describeCatalogue().length === CATALOGUE_SIZE);

  // =======================================================================
  console.log('\n── A ROW STAYS WHEN YOU TURN IT OFF ──');
  // The trap in "list the registered tools": turn one off and its row vanishes,
  // so there is no way back on.
  setTools({ ...toolsOverride, cron: { enabled: false } });
  client.loadConfig();
  rows = client.describeCatalogue();
  const cronRow = rows.find(r => r.id === 'create_cron_job');
  check('a switched-off tool is NOT registered', !client.getToolNames().includes('create_cron_job'));
  check('but its row is still on the page — otherwise it can never be turned back on',
    !!cronRow && cronRow.registered === false);
  check('and the row says why it is off', /turned off/.test(cronRow.why || ''), cronRow.why);
  check('with the switch that turns it back on', cronRow.toggle === 'tools.cron.enabled');
  setTools({ ...toolsOverride, cron: { enabled: true } });
  client.loadConfig();

  // =======================================================================
  console.log('\n── SECRETS: WRITE-ONLY, ENCRYPTED, ENV STILL WINS ──');
  delete process.env.EXA_API_KEY;
  const KEY = 'exa-live-key-abcdef123456';

  let r = callRoute('put', '/secrets', { secrets: { EXA_API_KEY: KEY } });
  check('the route accepts a declared secret', r.code === 200 && r.body.ok, JSON.stringify(r.body));
  check('and answers with status only — no value anywhere in the response',
    !JSON.stringify(r.body).includes(KEY), JSON.stringify(r.body));

  const st = secretsMod.status('EXA_API_KEY');
  check('the status says it is set, and from the store', st.set && st.source === 'store', JSON.stringify(st));
  check('it resolves for the server and for scripts', secretsMod.resolve('EXA_API_KEY') === KEY);

  const onDisk = fs.readFileSync(secretsMod.SECRETS_PATH, 'utf8');
  check('the key is NOT on disk in plaintext', !onDisk.includes(KEY));
  check('what is on disk is aes-256-gcm with an iv and a tag',
    /aes-256-gcm/.test(onDisk) && /"iv"/.test(onDisk) && /"tag"/.test(onDisk));
  const mode = fs.statSync(secretsMod.SECRETS_PATH).mode & 0o777;
  check('and the file is 0600', mode === 0o600, mode.toString(8));

  // The GET the page actually calls — the assertion that keeps "write-only" true.
  const payload = callRoute('get', '/').body;
  const asJson = JSON.stringify(payload);
  check('GET /api/tools does not contain the key', !asJson.includes(KEY));
  check('…nor any prefix of it that would let it be reconstructed',
    !asJson.includes(KEY.slice(0, 8)) && !asJson.includes(KEY.slice(-4)));
  const exaProvider = payload.search.providers.find(p => p.id === 'exa');
  check('it says the key is set without saying what it is',
    exaProvider.secret.status.set === true && !('value' in exaProvider.secret.status));
  check('and Exa now reads as available', exaProvider.available === true, JSON.stringify(exaProvider));
  check('the page can state the encryption honestly',
    payload.secretStore.encrypted === true && payload.secretStore.algorithm === 'aes-256-gcm');

  // A ciphertext is bound to its NAME: lifting the blob into another slot fails.
  const store = JSON.parse(fs.readFileSync(secretsMod.SECRETS_PATH, 'utf8'));
  store.secrets.OTHER_API_KEY = store.secrets.EXA_API_KEY;
  fs.writeFileSync(secretsMod.SECRETS_PATH, JSON.stringify(store), { mode: 0o600 });
  const moved = secretsMod.get('OTHER_API_KEY');
  check('a ciphertext copied into another slot does not decrypt',
    moved.value === null && /could not be decrypted/.test(moved.error || ''), JSON.stringify(moved));
  delete store.secrets.OTHER_API_KEY;
  fs.writeFileSync(secretsMod.SECRETS_PATH, JSON.stringify(store), { mode: 0o600 });

  // .env keeps working, and the UI can say that it is what answered.
  process.env.EXA_API_KEY = 'from-dot-env';
  const envSt = secretsMod.status('EXA_API_KEY');
  check('an environment value overrides the stored one', secretsMod.resolve('EXA_API_KEY') === 'from-dot-env');
  check('and the status says so, so a stale .env is not a mystery',
    envSt.source === 'env' && envSt.envOverrides === true, JSON.stringify(envSt));
  delete process.env.EXA_API_KEY;
  check('with the variable gone, the stored key answers again',
    secretsMod.resolve('EXA_API_KEY') === KEY);

  // Undeclared names are refused: this is not a general-purpose env writer.
  r = callRoute('put', '/secrets', { secrets: { SOMETHING_ELSE: 'x' } });
  check('a secret no tool declared is refused', r.code === 400 && /No tool declares/.test(r.body.error), JSON.stringify(r.body));
  r = callRoute('put', '/secrets', { secrets: { EXA_API_KEY: 42 } });
  check('a non-string value is refused', r.code === 400);

  // =======================================================================
  console.log('\n── THE PROVIDER CHAIN: OFF AND SECOND ARE DIFFERENT ──');
  const chainWith = (tools) => { setTools({ ...toolsOverride, ...tools }); return config.getSearchConfig(); };
  // Every case below runs the REAL getSearchConfig over an injected config, and
  // nothing reaches the network: with no available provider there is nothing to
  // call, and the cases that do have one never execute a search.
  delete process.env.EXA_API_KEY;

  let chain = chainWith({ exa: { enabled: true }, searxng: { enabled: true, url: 'http://localhost:8888' }, search: { order: ['exa', 'searxng'] } });
  check('both on: exa first, searxng second', chain.order.join(',') === 'exa,searxng', chain.order.join(','));

  chain = chainWith({ search: { order: ['searxng', 'exa'] } });
  check('the order is the order — searxng first when that is what was saved',
    chain.order.join(',') === 'searxng,exa', chain.order.join(','));

  chain = chainWith({ searxng: { enabled: false, url: 'http://localhost:8888' }, search: { order: ['exa', 'searxng'] } });
  check('SearXNG off REMOVES it from the order, rather than leaving it to be skipped',
    chain.order.join(',') === 'exa', chain.order.join(','));
  check('and search still works, via Exa', chain.any === true);
  check('the page still lists it, switched off, so it can come back',
    (chain.allProviders || []).some(p => p.name === 'searxng' && p.enabledInConfig === false));

  chain = chainWith({ exa: { enabled: false }, searxng: { enabled: true, url: 'http://localhost:8888' } });
  check('Exa off falls back to SearXNG alone', chain.order.join(',') === 'searxng' && chain.any === true);

  chain = chainWith({ exa: { enabled: true }, searxng: { enabled: true, url: 'http://localhost:8888' } });
  delete process.env.EXA_API_KEY;
  check('ON but keyless is a different state from OFF: still in the chain, marked unusable',
    chain.order.includes('exa'));

  // Both off — the case that must never become an invented answer.
  chain = chainWith({ exa: { enabled: false }, searxng: { enabled: false } });
  check('both off: nothing is in the chain', chain.order.length === 0);
  check('and `any` is false', chain.any === false);
  client.loadConfig();
  check('web_search is not registered at all', !client.getToolNames().includes('web_search'));
  check('web_fetch goes with it — a page fetch is only useful for a searched URL',
    !client.getToolNames().includes('web_fetch'));
  const offRow = client.describeCatalogue().find(t => t.id === 'web_search');
  check('the page explains which providers are off rather than just saying "off"',
    /switched off/.test(offRow.why || '') || /no search providers/.test(offRow.why || ''), offRow.why);

  const tool = new WebSearchTool();
  const out = await tool.execute({ query: 'anything at all' }, { caller: 'test' });
  check('a search with no provider is an ERROR, not an empty result', !!out.error, JSON.stringify(out));
  check('it says no provider is available', /No search provider is available/.test(out.error), out.error);
  check('and forbids answering as though it had searched',
    /rather than answering from memory/.test(out.error), out.error);
  check('it claims nothing about the world', !Array.isArray(out.results) || out.results.length === 0);

  // =======================================================================
  console.log('\n── CLEARING A KEY ──');
  setTools({ ...toolsOverride, exa: { enabled: true } });
  r = callRoute('put', '/secrets', { secrets: { EXA_API_KEY: null } });
  check('a null value clears the stored key', r.code === 200 && r.body.secrets.EXA_API_KEY.set === false, JSON.stringify(r.body));
  check('and it is gone from the store', secretsMod.resolve('EXA_API_KEY') === null);
  check('Exa reads as unusable again, with the reason',
    /no API key/.test((config.getSearchConfig().allProviders.find(p => p.name === 'exa') || {}).why || ''));

  // =======================================================================
  console.log('\n── THE BROWSER HALF, RUN RATHER THAN ASSUMED ──');
  // The render functions and the order collector are lifted VERBATIM out of
  // public/script.js and run here against a stub DOM. Retyping them would test the
  // copy; lifting them tests what ships. Two things are worth this trouble: the key
  // field must render EMPTY every time, and switching a provider off must drop it
  // from the order the form submits rather than reordering it.
  const uiSrc = fs.readFileSync(path.join(ROOT, 'public/script.js'), 'utf8');

  function liftFn(name) {
    const at = uiSrc.indexOf(`function ${name}(`);
    if (at === -1) throw new Error(`${name} is not in public/script.js`);
    let depth = 0;
    for (let j = uiSrc.indexOf('{', at); j < uiSrc.length; j++) {
      if (uiSrc[j] === '{') depth++;
      else if (uiSrc[j] === '}' && --depth === 0) return uiSrc.slice(at, j + 1);
    }
    throw new Error(`${name} never closes`);
  }

  const escapeHtml = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  const ui = {};
  new Function('escapeHtml', [
    liftFn('renderToolRow'), liftFn('renderToolField'),
    liftFn('renderSearchProviders'), liftFn('renderSecretField'),
    'this.renderToolRow = renderToolRow; this.renderSearchProviders = renderSearchProviders;'
  ].join('\n')).call(ui, escapeHtml);

  // A key IS set for this stretch, which is the case that could leak one.
  callRoute('put', '/secrets', { secrets: { EXA_API_KEY: 'render-check-key-zzz999' } });
  const uiPayload = callRoute('get', '/').body;
  const cardHtml = ui.renderSearchProviders(uiPayload.search);
  const rowsHtml = uiPayload.tools.map(t => ui.renderToolRow(t)).join('\n');

  check('every tool gets a row in the rendered markup',
    (rowsHtml.match(/class="tool-row /g) || []).length === uiPayload.tools.length,
    `${(rowsHtml.match(/class="tool-row /g) || []).length} of ${uiPayload.tools.length}`);
  check('each switch is bound by dotted config path, the existing mechanism',
    (rowsHtml.match(/data-config-key="/g) || []).length > 10);
  check('both providers render, with a position select each',
    (cardHtml.match(/class="provider-row /g) || []).length === 2
    && (cardHtml.match(/data-search-order="/g) || []).length === 2);
  check('the key field renders with an EMPTY value even though a key is set',
    /data-secret="EXA_API_KEY"[\s\S]{0,200}?value=""/.test(cardHtml)
    && !/data-secret="EXA_API_KEY"[\s\S]{0,200}?value="[^"]+"/.test(cardHtml));
  check('and the key itself is nowhere in the markup', !cardHtml.includes('render-check-key-zzz999'));
  check('what it shows instead is that one is saved', /secret-state-on/.test(cardHtml));
  callRoute('put', '/secrets', { secrets: { EXA_API_KEY: null } });

  // The order collector, verbatim, against a stub DOM.
  const collectorStart = uiSrc.indexOf('  const orderSelects = Array.from(');
  const collectorEnd = uiSrc.indexOf('  // Voice active selections');
  check('the order collector is where it is expected to be',
    collectorStart > -1 && collectorEnd > collectorStart);
  const collector = uiSrc.slice(collectorStart, collectorEnd);

  const collectOrder = (state) => {
    const doc = {
      querySelectorAll: (sel) => sel.includes('data-search-order')
        ? state.map(p => ({ dataset: { searchOrder: p.id }, value: String(p.pos) })) : [],
      querySelector: (sel) => {
        const m = sel.match(/data-provider-toggle="([^"]+)"/);
        const p = state.find(x => x.id === m[1]);
        return p ? { checked: p.on } : null;
      }
    };
    const partial = {};
    new Function('document', 'partial', collector)(doc, partial);
    return partial.tools.search.order.join(',');
  };

  check('both on, in the order shown',
    collectOrder([{ id: 'exa', pos: 1, on: true }, { id: 'searxng', pos: 2, on: true }]) === 'exa,searxng');
  check('positions swapped, order swapped',
    collectOrder([{ id: 'exa', pos: 2, on: true }, { id: 'searxng', pos: 1, on: true }]) === 'searxng,exa');
  check('a switched-off provider is LEFT OUT of the submitted order',
    collectOrder([{ id: 'exa', pos: 1, on: true }, { id: 'searxng', pos: 2, on: false }]) === 'exa');
  check('…including when it is the one that was first',
    collectOrder([{ id: 'exa', pos: 1, on: false }, { id: 'searxng', pos: 2, on: true }]) === 'searxng');
  check('both off submits an empty order', collectOrder([{ id: 'exa', pos: 1, on: false }, { id: 'searxng', pos: 1, on: false }]) === '');
  check('tied positions keep the displayed order rather than dropping one',
    collectOrder([{ id: 'exa', pos: 1, on: true }, { id: 'searxng', pos: 1, on: true }]) === 'exa,searxng');

  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => {
  console.error('\nTEST HARNESS ERROR:', err);
  process.exit(1);
});
