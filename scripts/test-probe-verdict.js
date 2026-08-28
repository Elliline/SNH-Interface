#!/usr/bin/env node
/**
 * A SLOW PROBE AND A DEAD ENGINE ARE DIFFERENT FACTS.
 *
 * On 2026-08-27 the watchdog restarted an engine that was generating at
 * 79.8 tok/s with 16 requests running and 10 queued. Nothing was wrong with it.
 * The liveness probe is an ordinary /v1/chat/completions, so it queues behind
 * whatever else the engine is doing, and its 8000ms deadline was reading queue
 * depth as death.
 *
 * The numbers that show a flat deadline cannot work here, measured over 6,503
 * successful probes on this box: p50 150ms, p99 540ms, p99.9 4607ms, and a
 * maximum SUCCESSFUL probe of 7940ms — 60ms under the cutoff. There is no gap
 * between healthy and dead for a single number to sit in.
 *
 * What is under test is the adjudicator that replaced it: given a failed probe,
 * does it reach the right verdict from the engine's own metrics? Four outcomes,
 * and only two of them are the engine's fault.
 *
 * Every assertion here is on a FIELD — the verdict, the counters, the stored
 * row, the resulting priority — never on a rendered sentence.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-probe-verdict.js
 */
const path = require('path');
const http = require('http');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-probe-verdict.js');
  process.exit(1);
}

const ROOT = path.join(__dirname, '..');
let pass = 0, fail = 0;
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

/**
 * A stub engine that serves /metrics the way vLLM does, with whatever counters
 * the test wants. `tokens` is read fresh on every request, so a test can decide
 * whether the counter MOVES between the adjudicator's two samples — which is the
 * whole discriminator between a busy engine and a stuck one.
 */
function stubEngine(state) {
  const server = http.createServer((req, res) => {
    if (req.url === '/metrics') {
      const t = typeof state.tokens === 'function' ? state.tokens() : state.tokens;
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end(
        `vllm:num_requests_running{engine="0",model_name="m"} ${state.running}\n` +
        `vllm:num_requests_waiting{engine="0",model_name="m"} ${state.waiting}\n` +
        `vllm:generation_tokens_total{engine="0",model_name="m"} ${t}\n` +
        `vllm:prompt_tokens_total{engine="0",model_name="m"} 0.0\n`
      );
      return;
    }
    res.writeHead(404); res.end();
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

(async () => {
  const config = require(path.join(ROOT, 'db/config'));
  const database = require(path.join(ROOT, 'db/database'));
  database.initDatabase();

  console.log('\n1. The adjudicator, against an engine that answers /metrics\n');

  // ---- saturated: holding work AND the token counter is moving --------------
  let ticking = 1000;
  const busy = await stubEngine({ running: 16, waiting: 10, tokens: () => (ticking += 50) });
  const busyPort = busy.address().port;
  // db/memory-manager DESTRUCTURES getConfig and getProviderInstance at load, so
  // the stubs have to be installed once, before it is required, and then aim
  // through a mutable variable. Replacing the module property afterwards changes
  // nothing — memory-manager is holding the old reference. (This cost four
  // spurious failures on the first run of this suite.)
  let currentPort = busyPort;
  const pointAt = (port) => { currentPort = port; };
  const realGetConfig = config.getConfig;
  config.getConfig = () => {
    const c = realGetConfig();
    return { ...c, models: { ...c.models, heartbeat: { provider: 'vllm', instance: 'stub', model: 'm' } } };
  };
  config.getProviderInstance = () => ({ host: `http://127.0.0.1:${currentPort}` });

  // Required AFTER the config stub, because it reads through the module object.
  const mm = require(path.join(ROOT, 'db/memory-manager'));

  const slow = { ok: false, ms: 8001, error: 'timeout after 8000ms', kind: 'slow' };
  let v = await mm.adjudicateProbe(slow, { metricsTimeoutMs: 2000 });
  check('a timed-out probe against a BUSY, PROGRESSING engine is `saturated`',
    v.verdict === 'saturated', v.verdict);
  check('…and it carries the queue depth it decided on',
    v.engine.running === 16 && v.engine.waiting === 10, JSON.stringify(v.engine));
  check('…and records that the engine was making progress',
    v.engine.generating === true, String(v.engine.generating));
  busy.close();

  // ---- stalled: holding work, counter frozen -------------------------------
  const stuck = await stubEngine({ running: 3, waiting: 0, tokens: 4242 });
  pointAt(stuck.address().port);
  v = await mm.adjudicateProbe(slow, { metricsTimeoutMs: 2000 });
  check('holding work with a FROZEN token counter is `stalled` — the 8/21-8/23 shape',
    v.verdict === 'stalled', v.verdict);
  check('…and it says so in the engine record',
    v.engine.generating === false && v.engine.running === 3, JSON.stringify(v.engine));
  stuck.close();

  // ---- idle but not answering ----------------------------------------------
  const idle = await stubEngine({ running: 0, waiting: 0, tokens: 7 });
  pointAt(idle.address().port);
  v = await mm.adjudicateProbe(slow, { metricsTimeoutMs: 2000 });
  check('an IDLE engine that still cannot answer a one-token completion is `stalled`',
    v.verdict === 'stalled', v.verdict);
  idle.close();

  // ---- metrics endpoint gone too -------------------------------------------
  console.log('\n2. When the engine is not there at all\n');
  pointAt(1);   // nothing listening
  v = await mm.adjudicateProbe(slow, { metricsTimeoutMs: 500 });
  check('a slow probe whose engine cannot serve /metrics either is `unreachable`',
    v.verdict === 'unreachable', v.verdict);
  check('…and the failure to read metrics is recorded, not swallowed',
    v.engine && v.engine.reachable === false, JSON.stringify(v.engine));

  const dead = { ok: false, ms: 2, error: 'fetch failed', kind: 'unreachable' };
  v = await mm.adjudicateProbe(dead, { metricsTimeoutMs: 500 });
  check('a connection failure is `unreachable` without consulting metrics at all',
    v.verdict === 'unreachable' && v.engine === null, v.verdict);

  const good = { ok: true, ms: 150, kind: 'ok' };
  v = await mm.adjudicateProbe(good, { metricsTimeoutMs: 500 });
  check('a probe that answered is `ok` and costs no extra call',
    v.verdict === 'ok' && v.engine === null, v.verdict);

  // ---- the probe itself classifies its own failure --------------------------
  console.log('\n3. The probe distinguishes a refused socket from an expired deadline\n');
  const black = http.createServer(() => { /* accept, never answer */ });
  await new Promise(r => black.listen(0, '127.0.0.1', r));
  pointAt(black.address().port);
  const timedOut = await mm.probeBrainLiveness(300);
  check('a socket that accepts and never answers yields kind `slow`',
    timedOut.kind === 'slow' && timedOut.ok === false, JSON.stringify(timedOut));
  black.close();

  pointAt(1);
  const refused = await mm.probeBrainLiveness(2000);
  check('a refused connection yields kind `unreachable`',
    refused.kind === 'unreachable' && refused.ok === false, JSON.stringify(refused));
  check('…and it fails fast rather than burning the deadline',
    refused.ms < 1000, `${refused.ms}ms`);

  config.getConfig = realGetConfig;

  // ---- the record keeps the evidence ---------------------------------------
  console.log('\n4. The verdict and its evidence are stored, not just logged\n');
  const db = database.getSqliteDb();
  const cols = db.prepare('PRAGMA table_info(liveness_probes)').all().map(c => c.name);
  for (const c of ['verdict', 'engine_running', 'engine_waiting', 'engine_generating']) {
    check(`liveness_probes carries \`${c}\``, cols.includes(c), cols.join(','));
  }

  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
