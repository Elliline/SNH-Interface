#!/usr/bin/env node
/**
 * How many concurrent requests can the brain actually take?
 *
 * The PIECEWISE root fix in scripts/launch-brain.sh was carried from an upstream
 * report — "clean over 200+ requests" — with no bound of our own. This measures
 * one. It fires N identical completions at the engine at once, times them, and
 * watches the engine's own scheduler counters while they run, so a level that
 * fails says WHY: a full batch (num_requests_running pinned, kv low) is not the
 * same failure as a full cache (kv high, preemptions climbing).
 *
 * Measured 2026-08-14, before --max-num-seqs was pinned: clean to 192 (2,040
 * tok/s aggregate, 0 preemptions, KV never above 7.5%), wedged at 256 — 0 of 256
 * completed, every request read-timed-out, and all 256 had been ADMITTED at 4.5%
 * KV. That is what the pin is for.
 *
 * READ-ONLY against the engine: it sends inference requests and scrapes
 * /metrics. It changes no config, touches no SNH data, and stores nothing.
 *
 * A LEVEL THAT WEDGES COSTS ~15 MINUTES. The engine stops answering, the
 * liveness probe fails three times, and the brain-watchdog restarts the
 * container and reloads the model. That is the designed recovery and it works —
 * but do not run this while you need the assistant.
 *
 * Every request is ignore_eos with a fixed max_tokens, so each level does
 * exactly the same amount of generation work and the aggregate throughputs are
 * comparable across levels.
 *
 * Usage:
 *   node scripts/measure-brain-concurrency.js                 # 1 4 8 16 32 64 128
 *   node scripts/measure-brain-concurrency.js 128 192         # specific levels
 *   node scripts/measure-brain-concurrency.js 256 --allow-wedge
 *   node scripts/measure-brain-concurrency.js --max-tokens 128 --timeout 60
 *
 * Levels above --max-num-seqs are refused without --allow-wedge, because past
 * the pin the interesting outcome is that they QUEUE (peak_waiting > 0, every
 * request still completes) rather than wedge — and confirming that is exactly
 * what the flag is for.
 */
const path = require('path');

const ROOT = path.join(__dirname, '..');
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

const cfg = getConfig();
const chat = cfg.models.chat;
const inst = getProviderInstance(chat.provider, chat.instance);
const HOST = (inst && inst.host) || 'http://localhost:7070';
const MODEL = chat.model;

// One pass over argv: a number is a concurrency level unless it is the value of
// a value-taking flag, so `--timeout 60 64` means one level of 64, not two.
const VALUE_FLAGS = new Set(['--max-tokens', '--timeout', '--max-num-seqs']);
const flags = {};
const levels = [];
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (VALUE_FLAGS.has(arg)) { flags[arg] = Number(process.argv[++i]); continue; }
  if (/^\d+$/.test(arg)) { levels.push(Number(arg)); continue; }
  flags[arg] = true;
}

const MAX_TOKENS = flags['--max-tokens'] ?? 256;
const TIMEOUT_MS = (flags['--timeout'] ?? 120) * 1000;
const ALLOW_WEDGE = flags['--allow-wedge'] === true;

// The batch ceiling the engine was launched with, if it was launched with one.
// Not reported in vLLM's startup config, so it is read from the container's
// command line — the same place launch-brain.sh writes it.
const PINNED_LIMIT = flags['--max-num-seqs'] ?? readPinnedLimit();

const SWEEP = levels.length ? [...new Set(levels)] : [1, 4, 8, 16, 32, 64, 128];

// Long enough that prefill is not the whole story, identical for every request
// so the prefix cache serves all of them and the level measures decode.
const PROMPT = 'Write a detailed technical explanation of how continuous batching '
  + 'works in an LLM inference server, covering the scheduler, the KV cache, and '
  + 'how requests are admitted and evicted.';

const METRIC_KEYS = [
  'vllm:num_requests_running',
  'vllm:num_requests_waiting',
  'vllm:kv_cache_usage_perc',
  'vllm:num_preemptions_total'
];

function readPinnedLimit() {
  try {
    const { execFileSync } = require('child_process');
    const container = (cfg.watchdog && cfg.watchdog.container) || 'sparky-brain';
    const cmd = execFileSync('docker', ['inspect', '-f', '{{json .Config.Cmd}}', container],
      { encoding: 'utf8', timeout: 5000 });
    const m = cmd.match(/--max-num-seqs[ =]+(\d+)/);
    return m ? Number(m[1]) : null;
  } catch {
    return null; // no docker, not our container, or it is not pinned — say so rather than guess
  }
}

async function scrape() {
  try {
    const res = await fetch(`${HOST}/metrics`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return null;
    const txt = await res.text();
    const out = {};
    for (const line of txt.split('\n')) {
      for (const k of METRIC_KEYS) {
        if (line.startsWith(k + '{')) {
          const v = Number(line.slice(line.lastIndexOf(' ') + 1));
          if (Number.isFinite(v)) out[k] = v;
        }
      }
    }
    return out;
  } catch {
    return null;
  }
}

/** Samples the engine's scheduler counters for the life of one level. */
function startPoller() {
  const samples = [];
  let stopped = false;
  const loop = (async () => {
    while (!stopped) {
      const s = await scrape();
      if (s) samples.push(s);
      await new Promise(r => setTimeout(r, 250));
    }
  })();
  return { samples, async stop() { stopped = true; await loop; } };
}

/** One streaming completion. Returns { ttft, wall, tokens, error }. */
async function oneRequest() {
  const t0 = process.hrtime.bigint();
  const since = () => Number(process.hrtime.bigint() - t0) / 1e9;
  let ttft = null;
  let tokens = 0;
  let usageTokens = null;
  try {
    const res = await fetch(`${HOST}/v1/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        prompt: PROMPT,
        max_tokens: MAX_TOKENS,
        temperature: 0,
        seed: 1234,
        stream: true,
        ignore_eos: true, // every request generates exactly MAX_TOKENS
        stream_options: { include_usage: true }
      }),
      signal: AbortSignal.timeout(TIMEOUT_MS)
    });
    if (!res.ok) {
      return { ttft: null, wall: since(), tokens: 0, error: `HTTP ${res.status}` };
    }
    let buf = '';
    for await (const chunk of res.body) {
      buf += Buffer.from(chunk).toString('utf8');
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;
        let obj;
        try { obj = JSON.parse(payload); } catch { continue; }
        const ch = obj.choices || [];
        if (ch.length && ch[0].text) {
          if (ttft === null) ttft = since();
          tokens++;
        }
        if (obj.usage) usageTokens = obj.usage.completion_tokens;
      }
    }
    return { ttft, wall: since(), tokens: usageTokens || tokens, error: null };
  } catch (err) {
    const name = err && err.name === 'TimeoutError' ? `timeout after ${TIMEOUT_MS / 1000}s` : String(err && err.message || err);
    return { ttft, wall: since(), tokens, error: name };
  }
}

async function runLevel(c) {
  const poller = startPoller();
  const t0 = process.hrtime.bigint();
  const results = await Promise.all(Array.from({ length: c }, () => oneRequest()));
  const wall = Number(process.hrtime.bigint() - t0) / 1e9;
  await poller.stop();

  const ok = results.filter(r => !r.error);
  const errors = results.filter(r => r.error).map(r => r.error);
  const ttfts = ok.map(r => r.ttft).filter(t => t !== null).sort((a, b) => a - b);
  const totalTokens = ok.reduce((s, r) => s + r.tokens, 0);
  const peak = key => poller.samples.reduce((m, s) => Math.max(m, s[key] || 0), 0);

  return {
    concurrency: c,
    wall_s: +wall.toFixed(2),
    completed: ok.length,
    n_errors: errors.length,
    errors: [...new Set(errors)].slice(0, 3),
    total_output_tokens: totalTokens,
    aggregate_tok_s: wall ? +(totalTokens / wall).toFixed(1) : 0,
    ttft_p50_s: ttfts.length ? +ttfts[Math.floor(ttfts.length / 2)].toFixed(2) : null,
    ttft_max_s: ttfts.length ? +ttfts[ttfts.length - 1].toFixed(2) : null,
    peak_running: peak('vllm:num_requests_running'),
    peak_waiting: peak('vllm:num_requests_waiting'),
    peak_kv_usage_perc: +(peak('vllm:kv_cache_usage_perc') * 100).toFixed(2),
    preemptions_total: peak('vllm:num_preemptions_total'),
    metric_samples: poller.samples.length
  };
}

function verdict(r) {
  if (r.completed === r.concurrency && !r.n_errors) {
    return r.peak_waiting > 0
      ? `clean — ${r.peak_waiting} queued at peak, so the pin held the batch and nothing was dropped`
      : 'clean';
  }
  if (!r.completed) return 'WEDGED — nothing completed';
  return `PARTIAL — ${r.completed}/${r.concurrency} completed`;
}

(async () => {
  console.log(`Engine: ${HOST}  model: ${MODEL}`);
  console.log(`Per request: max_tokens=${MAX_TOKENS} (ignore_eos), timeout ${TIMEOUT_MS / 1000}s`);
  console.log(PINNED_LIMIT
    ? `Engine batch pin: --max-num-seqs ${PINNED_LIMIT}`
    : 'Engine batch pin: NONE FOUND — the engine is using its default max_num_seqs');

  if (PINNED_LIMIT && !ALLOW_WEDGE) {
    const over = SWEEP.filter(c => c > PINNED_LIMIT);
    if (over.length) {
      console.error(`\nRefusing ${over.join(', ')}: above the pinned batch of ${PINNED_LIMIT}.`);
      console.error('Past the pin these should QUEUE rather than wedge. Re-run with --allow-wedge to confirm that,');
      console.error('knowing a wedge costs ~15 minutes while the watchdog restarts the brain.');
      process.exit(1);
    }
  }

  const idle = await scrape();
  if (!idle) {
    console.error('\nCould not read /metrics — is the engine up?');
    process.exit(1);
  }
  console.log(`Idle: running=${idle['vllm:num_requests_running']} waiting=${idle['vllm:num_requests_waiting']} `
    + `kv=${(idle['vllm:kv_cache_usage_perc'] * 100).toFixed(2)}%\n`);

  // One throwaway call so the prefix cache is warm for every level equally.
  process.stdout.write('warmup… ');
  const warm = await oneRequest();
  console.log(warm.error ? `FAILED: ${warm.error}` : `${warm.wall.toFixed(2)}s, ${warm.tokens} tokens\n`);
  if (warm.error) process.exit(1);

  const out = [];
  for (const c of SWEEP) {
    await new Promise(r => setTimeout(r, 5000)); // let the engine settle between levels
    process.stdout.write(`c=${String(c).padStart(4)} … `);
    const r = await runLevel(c);
    out.push(r);
    console.log(`${String(r.aggregate_tok_s).padStart(7)} tok/s  `
      + `ttft p50 ${r.ttft_p50_s === null ? '   —' : r.ttft_p50_s.toFixed(2) + 's'}  `
      + `run/wait ${r.peak_running}/${r.peak_waiting}  kv ${r.peak_kv_usage_perc}%  `
      + `preempt ${r.preemptions_total}  — ${verdict(r)}`);
    if (r.errors.length) console.log(`       errors: ${r.errors.join(' | ')}`);
    if (!r.completed) {
      console.log('\nStopping the sweep: the engine is not answering. The watchdog should restart it within');
      console.log('~15 minutes (3 liveness failures, then `docker restart`). Nothing here needs to be undone.');
      break;
    }
  }

  console.log(`\n${JSON.stringify(out, null, 2)}`);
})();
