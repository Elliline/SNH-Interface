#!/usr/bin/env node
/**
 * What does excluding the active conversation's own log entries do to the
 * prefix cache on a busy thread?
 *
 * vLLM caches the KV of a request's token PREFIX and the cache holds up to the
 * first token that differs from a previous request. Today's log sits early in
 * that prefix — before the retrieval blocks, before the date line, before the
 * conversation — and it used to grow by one entry per message OF THE THREAD
 * BEING RENDERED. So on every turn it changed, and everything after it was
 * re-prefilled. Excluding the echo makes the block change only when something
 * else happened today, which on a busy thread is rarely.
 *
 * HOW THIS MEASURES. It replays a real thread turn by turn against the real
 * engine and reads vllm:prefix_cache_{hits,queries}_total around each request.
 * The prompt is assembled the way server.js assembles it, from the real daily
 * log (reconstructed to its state at each turn, by entry timestamp), the real
 * identity block, the real capability manifest and the real long-term memory —
 * against a SNAPSHOT of the live store, so the measurement cannot write to it.
 *
 * The retrieval blocks (clusters, past conversations) are stand-ins of realistic
 * size that differ every turn, which is what the real ones do. They are IDENTICAL
 * between the two variants, so they are not what is being compared — they are
 * there because they sit after today's log in the prefix and therefore bound how
 * much a stable log block can actually save.
 *
 * Usage: node scripts/measure-daily-echo-cache.js [conversationId]
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const Database = require('better-sqlite3');

const ROOT = path.join(__dirname, '..');
const LIVE_DATA = path.join(ROOT, 'data');

// ---- snapshot the live store, then point the whole process at the snapshot ----
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-cache-measure-'));
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});
{
  // READONLY + VACUUM INTO, same reasoning as the replay seed: the live db is in
  // WAL mode under a running server, and checkpointing first would be a write.
  const live = new Database(path.join(LIVE_DATA, 'chat.db'), { readonly: true });
  live.prepare('VACUUM INTO ?').run(path.join(TMP, 'chat.db'));
  live.close();
  fs.cpSync(path.join(LIVE_DATA, 'memory'), path.join(TMP, 'memory'), { recursive: true });
  if (fs.existsSync(path.join(LIVE_DATA, 'config.json'))) {
    fs.copyFileSync(path.join(LIVE_DATA, 'config.json'), path.join(TMP, 'config.json'));
  }
}
process.env.SNH_DATA_DIR = TMP;

const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const injectionBudget = require(path.join(ROOT, 'db/injection-budget'));
const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
const identity = require(path.join(ROOT, 'db/identity'));
const capabilityManifest = require(path.join(ROOT, 'db/capability-manifest'));
const { getConfig } = require(path.join(ROOT, 'db/config'));

const config = getConfig();
const chat = config.models.chat;
const inst = (config.providers[chat.provider] || []).find(p => p.name === chat.instance)
  || (config.providers[chat.provider] || [])[0];
const HOST = inst && inst.host;
const MODEL = chat.model;
const injCfg = (config.memory && config.memory.injection) || {};

// ---- pick the thread ----
let convoId = process.argv[2];
if (!convoId) {
  const row = db.prepare(`
    SELECT conversation_id, DATE(timestamp) d, COUNT(*) n
    FROM messages WHERE timestamp >= DATE('now', '-7 days')
    GROUP BY conversation_id, DATE(timestamp) ORDER BY n DESC LIMIT 1`).get();
  if (!row) { console.error('No recent conversation to measure.'); process.exit(1); }
  convoId = row.conversation_id;
}
const msgs = db.prepare(
  `SELECT id, role, content, timestamp FROM messages
   WHERE conversation_id = ? AND role IN ('user','assistant') ORDER BY timestamp ASC`).all(convoId);
if (msgs.length < 4) { console.error(`Thread ${convoId} has ${msgs.length} messages — too short to measure.`); process.exit(1); }

// The messages table stores SQLite's CURRENT_TIMESTAMP, which is UTC; the daily
// log stamps its "### HH:MM" headings in local Pacific time. Comparing the two
// raw is a seven-hour error that quietly hands every turn the FINISHED day's log
// — which makes the block look static and hides the entire effect being
// measured. It did exactly that on the first run of this script.
const LOCAL_TZ = 'America/Los_Angeles';
const asLocal = (ts) => new Date(`${String(ts).replace(' ', 'T')}Z`)
  .toLocaleString('sv-SE', { timeZone: LOCAL_TZ });      // "YYYY-MM-DD HH:MM:SS"
const day = asLocal(msgs[0].timestamp).slice(0, 10);

// ---- the day's log, and its state at any moment of that day ----
const dailyDir = path.join(TMP, 'memory', 'daily');
const readDay = (d) => {
  const f = path.join(dailyDir, `${d}.md`);
  return fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : '';
};
const todayFull = readDay(day);
const yesterdayFull = (() => {
  const d = new Date(`${day}T12:00:00Z`); d.setUTCDate(d.getUTCDate() - 1);
  return readDay(d.toISOString().slice(0, 10));
})();

const { header, blocks } = injectionBudget.splitDailyBlocks(todayFull);
const blockTime = (b) => {
  const m = b.match(/^#{2,3} (\d{2}):(\d{2})/);
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
};
/** The log as it stood at HH:MM — entries stamped later had not been written yet. */
function logAsOf(minutes) {
  const kept = blocks.filter(b => {
    const t = blockTime(b);
    return t === null ? false : t <= minutes;
  });
  return header + kept.join('\n\n');
}
const msgMinutes = (m) => {
  const t = asLocal(m.timestamp).slice(11, 16).split(':');
  return Number(t[0]) * 60 + Number(t[1]);
};

// ---- the stable front of the prompt, exactly as server.js orders it ----
const identityBlock = identity.buildIdentityBlock();
const manifestBlock = capabilityManifest.buildInjectionBlock();
const ltm = memoryClusters.renderLongTermMemory({
  subject: ['user'], budgetTokens: injCfg.longTermTokens ?? 3000
}) || '';

/** Stand-in retrieval text: realistic size, different every turn, same in both variants. */
function retrievalFor(turn) {
  const line = (i) => `[Memory ${i}] user: turn-${turn} retrieval line ${i} — a passage pulled back by relevance for this particular message, of about the length these actually run to in practice.`;
  return Array.from({ length: 6 }, (_, i) => line(i)).join('\n');
}

/**
 * A per-run, per-variant salt at the very FRONT of the prompt.
 *
 * Without it the second run of this script reports 99.8% for both variants: the
 * engine still holds the exact prompts the first run sent, so every request is a
 * repeat of itself and the measurement measures its own history. Salting the
 * first token makes the whole prefix cold. A DIFFERENT salt per variant is the
 * fair form — otherwise whichever variant runs second inherits the other's warm
 * identity/manifest/long-term prefix and looks better for it.
 */
const RUN_SALT = process.env.SNH_CACHE_SALT || String(process.hrtime.bigint());

function buildPrompt(turnIdx, { excludeEcho }) {
  const upTo = msgs.slice(0, turnIdx + 1);
  const minutes = msgMinutes(upTo[upTo.length - 1]);
  const { recent, summary, stats } = injectionBudget.budgetDailyLogs(
    logAsOf(minutes), yesterdayFull,
    { dailyTodayTokens: injCfg.dailyTodayTokens ?? 1500,
      dailySummaryTokens: injCfg.dailySummaryTokens ?? 400,
      excludeConversationId: excludeEcho ? convoId : null });

  const parts = [];
  if (ltm) parts.push(`=== Long-Term Memory ===\n${ltm}`);
  if (recent) parts.push(`=== Today's Session Log (most recent) ===\n${recent}`);
  if (summary) parts.push(`=== Earlier / Yesterday (brief) ===\n${summary}`);
  parts.push(`=== Associated Memory Clusters ===\n${retrievalFor(turnIdx)}`);

  const system = [
    { role: 'system', content: `(measurement run ${RUN_SALT}/${excludeEcho ? 'after' : 'before'})` },
    { role: 'system', content: identityBlock.stableText || identityBlock.text },
    { role: 'system', content: manifestBlock.text },
    { role: 'system', content: parts.join('\n\n') },
    { role: 'system', content: `Use this as the current date/time: ${day} ${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')} Pacific` }
  ];
  return {
    messages: [...system, ...upTo.map(m => ({ role: m.role, content: m.content }))],
    dailyTokens: injectionBudget.estTokens(recent) + injectionBudget.estTokens(summary),
    excluded: stats.todayBlocksSelfExcluded
  };
}

// ---- engine metrics ----
async function metrics() {
  const text = await (await fetch(`${HOST}/metrics`)).text();
  const grab = (name) => {
    const m = text.match(new RegExp(`^vllm:${name}\\{[^}]*\\}\\s+([0-9.e+-]+)$`, 'm'));
    return m ? Number(m[1]) : null;
  };
  return { queries: grab('prefix_cache_queries_total'), hits: grab('prefix_cache_hits_total') };
}

async function turn(payload) {
  const before = await metrics();
  const res = await fetch(`${HOST}/v1/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: payload.messages, max_tokens: 1, temperature: 0 })
  });
  if (!res.ok) throw new Error(`engine ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const after = await metrics();
  return {
    promptTokens: data.usage.prompt_tokens,
    queries: after.queries - before.queries,
    hits: after.hits - before.hits
  };
}

async function runVariant(label, excludeEcho) {
  const rows = [];
  for (let i = 0; i < msgs.length; i++) {
    if (msgs[i].role !== 'user') continue;              // one request per user turn
    const payload = buildPrompt(i, { excludeEcho });
    const r = await turn(payload);
    rows.push({ turn: rows.length + 1, ...r, dailyTokens: payload.dailyTokens, excluded: payload.excluded });
  }
  const q = rows.reduce((s, r) => s + r.queries, 0);
  const h = rows.reduce((s, r) => s + r.hits, 0);
  const p = rows.reduce((s, r) => s + r.promptTokens, 0);
  const noise = Math.abs(q - p) / Math.max(1, p);
  console.log(`\n${label}`);
  console.log('  turn  prompt_tok  queried  cached  hit%   dailyBlock  echoExcluded');
  for (const r of rows) {
    console.log(`  ${String(r.turn).padStart(4)}  ${String(r.promptTokens).padStart(10)}  ${String(r.queries).padStart(7)}  ${String(r.hits).padStart(6)}  ` +
      `${(r.queries ? (r.hits / r.queries) * 100 : 0).toFixed(1).padStart(5)}  ${String(r.dailyTokens).padStart(10)}  ${String(r.excluded).padStart(12)}`);
  }
  console.log(`  TOTAL prompt=${p} queried=${q} cached=${h} → hit rate ${(h / q * 100).toFixed(1)}%` +
    `  | prefilled ${q - h} tokens` +
    (noise > 0.05 ? `  [WARN: queried vs prompt differ by ${(noise * 100).toFixed(1)}% — other traffic hit the engine during this run]` : ''));
  return { queries: q, hits: h, prompt: p, rows };
}

(async () => {
  if (!HOST) { console.error('No host configured for the chat provider.'); process.exit(1); }
  const userTurns = msgs.filter(m => m.role === 'user').length;
  console.log(`\nThread ${convoId} on ${day}: ${msgs.length} messages, ${userTurns} user turns`);
  console.log(`Engine ${chat.provider}/${MODEL} @ ${HOST}`);
  console.log(`Today's log: ${blocks.length} entries, ${injectionBudget.estTokens(todayFull)} tokens on disk; ` +
    `${blocks.filter(b => injectionBudget.blockIsFromConversation(b, convoId)).length} of them stamped with THIS conversation`);

  // Prime the shared stable prefix so neither variant is charged for being first.
  console.log('\nWarming the shared prefix…');
  await turn(buildPrompt(0, { excludeEcho: false }));
  await turn(buildPrompt(0, { excludeEcho: true }));

  // The engine is shared with the running server, so blocks can be evicted
  // between our requests. Order is therefore a confound: whichever variant runs
  // second could inherit or lose warmth that has nothing to do with the change.
  // --reverse runs them the other way round; if the result flips, the number is
  // an artifact of ordering rather than a property of the change.
  const reverse = process.argv.includes('--reverse');
  const labelBefore = 'BEFORE — today\'s log includes this conversation\'s own entries';
  const labelAfter = 'AFTER — this conversation\'s own entries excluded from the render';
  let before, after;
  if (reverse) {
    console.log('\n(order control: AFTER runs first)');
    after = await runVariant(labelAfter, true);
    before = await runVariant(labelBefore, false);
  } else {
    before = await runVariant(labelBefore, false);
    after = await runVariant(labelAfter, true);
  }

  const bRate = before.hits / before.queries * 100;
  const aRate = after.hits / after.queries * 100;
  const bPrefill = before.queries - before.hits;
  const aPrefill = after.queries - after.hits;
  const bar = '='.repeat(74);
  console.log(`\n${bar}`);
  console.log(`Prefix cache hit rate over the thread: ${bRate.toFixed(1)}%  →  ${aRate.toFixed(1)}%  (+${(aRate - bRate).toFixed(1)} pts)`);
  console.log(`Tokens actually prefilled:            ${bPrefill}  →  ${aPrefill}  ` +
    `(${bPrefill ? ((1 - aPrefill / bPrefill) * 100).toFixed(1) : '0'}% less)`);
  console.log(`Prompt tokens sent:                   ${before.prompt}  →  ${after.prompt}  ` +
    `(${before.prompt ? ((1 - after.prompt / before.prompt) * 100).toFixed(1) : '0'}% smaller)`);
  console.log(`${bar}\n`);
})().catch(err => {
  console.error('[measure-daily-echo-cache] error:', err.message);
  process.exit(1);
});
