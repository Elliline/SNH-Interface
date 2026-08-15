#!/usr/bin/env node
/**
 * What does the brain actually deliver as concurrency climbs, and where does
 * it stop scaling?
 *
 * vLLM batches continuously: requests join and leave the running batch every
 * step, so aggregate throughput rises with concurrency until something binds —
 * either the scheduler's own seat count (--max-num-seqs) or KV cache space. On
 * this box the engine is FP8 Marlin W8A16 on two Ampere cards with no native
 * FP8 compute, so the decode side is dequant-bound rather than memory-bound,
 * and the interesting question is how much of the seat count is usable before
 * per-request latency degrades faster than aggregate throughput improves.
 *
 * HOW THIS MEASURES. It fires N chat completions at the real engine at once,
 * streaming, and times them from the client. TTFT is the wall time to the first
 * token of a response — the number a person actually feels. Aggregate tok/s is
 * total completion tokens across the level divided by the level's wall clock,
 * which is the throughput number, NOT the sum of per-request rates. Peak KV is
 * read from the engine's own Prometheus gauge (vllm:gpu_cache_usage_perc),
 * sampled while the level runs, because the client cannot see it.
 *
 * THE PIN IS READ FROM THE ENGINE, NOT ASSUMED. --max-num-seqs is the scheduler's
 * seat count: ask for more than that and the extra requests queue instead of
 * running, so the level would report a concurrency it never actually had. The
 * pin is recovered from the running process's own argv, and any level above it
 * is REFUSED rather than silently clamped — a quietly clamped level is a wrong
 * measurement that looks like a real one.
 *
 * Prompts differ per request on purpose. Identical prompts would share a KV
 * prefix and inflate throughput with cache hits that a real mixed workload
 * would not get.
 *
 * Usage: node scripts/measure-brain-concurrency.js [--levels 1,32,64,128]
 *                                                  [--tokens 256]
 *                                                  [--host http://127.0.0.1:8000]
 */

const DEFAULTS = {
  host: process.env.VLLM_HOST || 'http://127.0.0.1:8000',
  levels: [1, 32, 64, 128],
  tokens: 256,
};

function parseArgs() {
  const out = { ...DEFAULTS };
  const a = process.argv.slice(2);
  for (let i = 0; i < a.length; i++) {
    if (a[i] === '--levels') out.levels = a[++i].split(',').map(s => parseInt(s.trim(), 10));
    else if (a[i] === '--tokens') out.tokens = parseInt(a[++i], 10);
    else if (a[i] === '--host') out.host = a[++i];
  }
  return out;
}

/** The served model id, straight from the engine. */
async function getModel(host) {
  const r = await fetch(`${host}/v1/models`);
  if (!r.ok) throw new Error(`GET /v1/models -> ${r.status}`);
  const body = await r.json();
  if (!body.data || !body.data.length) throw new Error('no models served');
  return body.data[0].id;
}

/**
 * Recover --max-num-seqs from the running engine's argv. Returns null if the
 * process cannot be found, in which case the caller must refuse to guess.
 */
function readSeatPin() {
  const { execSync } = require('child_process');
  let out;
  try {
    out = execSync('ps -eo args', { encoding: 'utf8' });
  } catch {
    return null;
  }
  for (const line of out.split('\n')) {
    if (!line.includes('vllm') || !line.includes('serve')) continue;
    let m = line.match(/--max-num-seqs[= ]+(\d+)/);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

/**
 * KV usage gauge. vLLM renamed this: 0.27.x exposes vllm:kv_cache_usage_perc,
 * older builds vllm:gpu_cache_usage_perc. Accept either, and take the max over
 * whatever engine/model labels are present.
 */
const KV_GAUGES = ['vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc'];
async function readKvUsage(host) {
  try {
    const r = await fetch(`${host}/metrics`);
    if (!r.ok) return null;
    const text = await r.text();
    let peak = 0;
    let seen = false;
    for (const line of text.split('\n')) {
      if (line.startsWith('#')) continue;
      if (!KV_GAUGES.some(g => line.startsWith(g))) continue;
      const v = parseFloat(line.slice(line.lastIndexOf(' ') + 1));
      if (!Number.isNaN(v)) { peak = Math.max(peak, v); seen = true; }
    }
    return seen ? peak : null;
  } catch {
    return null;
  }
}

/** Distinct prompts so the level does not ride a shared KV prefix. */
function promptFor(i) {
  const subjects = [
    'a tidal estuary at dawn', 'the maintenance of a bicycle drivetrain',
    'why sourdough starters fail', 'the acoustics of a small concert hall',
    'how a mechanical watch escapement works', 'the ecology of urban crows',
    'the history of the shipping container', 'how frost forms on a window',
    'the logistics of a mountain rescue', 'why some bridges hum in wind',
  ];
  return `Request ${i}: write a careful, concrete paragraph about ${subjects[i % subjects.length]}. Vary your wording.`;
}

/**
 * One streaming request. Returns TTFT and completion-token count. Tokens are
 * counted from the stream's own deltas rather than a tokenizer, which is what
 * the engine actually emitted.
 */
async function oneRequest(host, model, i, maxTokens) {
  const started = process.hrtime.bigint();
  let ttftMs = null;
  let tokens = 0;

  const res = await fetch(`${host}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: promptFor(i) }],
      max_tokens: maxTokens,
      temperature: 1.0,
      top_p: 0.95,
      stream: true,
      stream_options: { include_usage: true },
      // The recipe's model ships adaptive thinking on by default; a benchmark
      // wants the answer path, not a variable-length reasoning preamble.
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let usageTokens = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') continue;
      let obj;
      try { obj = JSON.parse(payload); } catch { continue; }
      if (obj.usage && typeof obj.usage.completion_tokens === 'number') {
        usageTokens = obj.usage.completion_tokens;
      }
      const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
      if (delta && delta.content) {
        if (ttftMs === null) ttftMs = Number(process.hrtime.bigint() - started) / 1e6;
        tokens++;
      }
    }
  }

  const totalMs = Number(process.hrtime.bigint() - started) / 1e6;
  // Prefer the engine's own usage count; fall back to counted deltas.
  return { ttftMs, tokens: usageTokens !== null ? usageTokens : tokens, totalMs };
}

function p50(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

async function runLevel(host, model, n, maxTokens) {
  let peakKv = 0;
  let sampling = true;
  const sampler = (async () => {
    while (sampling) {
      const v = await readKvUsage(host);
      if (v !== null) peakKv = Math.max(peakKv, v);
      await new Promise(r => setTimeout(r, 250));
    }
  })();

  const started = process.hrtime.bigint();
  const results = await Promise.allSettled(
    Array.from({ length: n }, (_, i) => oneRequest(host, model, i, maxTokens))
  );
  const wallMs = Number(process.hrtime.bigint() - started) / 1e6;

  sampling = false;
  await sampler;

  const ok = results.filter(r => r.status === 'fulfilled').map(r => r.value);
  const failed = results.length - ok.length;
  const firstErr = results.find(r => r.status === 'rejected');
  const totalTokens = ok.reduce((s, r) => s + r.tokens, 0);

  return {
    concurrency: n,
    completed: ok.length,
    failed,
    completionRate: ok.length / results.length,
    wallSec: wallMs / 1000,
    totalTokens,
    aggregateTokS: totalTokens / (wallMs / 1000),
    ttftP50Ms: p50(ok.map(r => r.ttftMs).filter(v => v !== null)),
    peakKvPerc: peakKv,
    firstError: firstErr ? String(firstErr.reason).slice(0, 160) : null,
  };
}

(async () => {
  const cfg = parseArgs();

  const pin = readSeatPin();
  if (pin === null) {
    console.error('REFUSING: could not read --max-num-seqs from the running engine.');
    console.error('The sweep would report concurrency levels the scheduler never actually ran.');
    process.exit(2);
  }
  const over = cfg.levels.filter(l => l > pin);
  if (over.length) {
    console.error(`REFUSING: level(s) ${over.join(', ')} exceed the engine pin --max-num-seqs=${pin}.`);
    console.error('Above the pin, requests queue instead of running concurrently, so the');
    console.error('measurement would not mean what its label says. Re-pin the engine or drop the level.');
    process.exit(2);
  }

  const model = await getModel(cfg.host);
  console.log(`engine     : ${cfg.host}`);
  console.log(`model      : ${model}`);
  console.log(`seat pin   : --max-num-seqs=${pin} (read from engine argv)`);
  console.log(`levels     : ${cfg.levels.join(', ')}   max_tokens=${cfg.tokens}`);
  console.log();

  const rows = [];
  for (const n of cfg.levels) {
    process.stderr.write(`  running concurrency ${n} ...\n`);
    rows.push(await runLevel(cfg.host, model, n, cfg.tokens));
  }

  const pad = (s, w) => String(s).padStart(w);
  console.log('conc   done/tot   rate    agg tok/s   TTFT p50   peak KV   wall s');
  console.log('----   --------   -----   ---------   --------   -------   ------');
  for (const r of rows) {
    console.log([
      pad(r.concurrency, 4),
      pad(`${r.completed}/${r.concurrency}`, 10),
      pad((r.completionRate * 100).toFixed(0) + '%', 7),
      pad(r.aggregateTokS.toFixed(1), 11),
      pad(r.ttftP50Ms === null ? 'n/a' : r.ttftP50Ms.toFixed(0) + 'ms', 10),
      pad((r.peakKvPerc * 100).toFixed(1) + '%', 9),
      pad(r.wallSec.toFixed(1), 8),
    ].join(''));
    if (r.firstError) console.log(`       first error: ${r.firstError}`);
  }
  console.log();
  console.log(JSON.stringify({ pin, model, rows }, null, 2));
})().catch(e => { console.error('FAILED:', e.message); process.exit(1); });
