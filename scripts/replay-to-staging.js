#!/usr/bin/env node
/**
 * The replay — rebuild the user-fact corpus from source, into STAGING.
 *
 * The spec's central claim is that this corpus can be rebuilt correctly from the
 * conversations that produced it. This is the thing that tests it. Every stored
 * exchange goes back through the CURRENT intake pipeline — extraction-rules,
 * atomicity, event-vs-state routing, the identity anchor, repeat folding,
 * provenance — and what comes out is a fresh corpus built by the rules as they
 * are now rather than as they were on the day each fact was written.
 *
 * ZERO WRITES TO THE LIVE STORE, and not by being careful: this process points
 * SNH_DATA_DIR at a staging directory, so `getSqliteDb()` and the vector handle
 * ARE the staging ones for every module in the process. There is no staging flag
 * threaded through fact-store or the extractor, because a flag is a thing one
 * call site can forget. It refuses to start if the data dir resolves to live.
 *
 * WHAT CARRIES OVER, untouched, never regenerated:
 *   - every self-fact (the spec: self-facts are curated WITH Aurelius, not
 *     replayed — replaying the reflection loop reproduces the defect)
 *   - identity-locked rows, verbatim
 *   - every inactive row, any subject — supersession history IS provenance
 *   - the corrections ledger and its notices
 *   - fact_corroborations belonging to rows that survive
 * What is discarded and rebuilt: ACTIVE user facts, and only those.
 *
 * CONCURRENCY. planExtraction decides and writes nothing; applyExtraction
 * writes. So planning runs in a pool (memory.replay.concurrency, default 6) and
 * applying is strictly serial in source order, which is what keeps repeat-folding
 * and supersession looking at a consistent corpus. Plans inside one window are
 * computed against a corpus missing that window's own writes; byte-identical
 * duplicates are still caught at the SQLite write, and the residue is reported
 * rather than hidden.
 *
 * Usage:
 *   node scripts/replay-to-staging.js [--limit N] [--concurrency N] [--fresh]
 *
 *   --fresh   rebuild the staging store from scratch (default: reuse if present)
 *   --limit   stop after N exchanges (for a smoke run)
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

const LIVE_DATA = path.join(ROOT, 'data');
const STAGING_DATA = process.env.SNH_STAGING_DIR || path.join(ROOT, 'data-staging');

// The redirect must be in place BEFORE db/database.js is required, because the
// paths are resolved at module load. This is the whole mechanism.
process.env.SNH_DATA_DIR = STAGING_DATA;

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const LIMIT = parseInt(argVal('--limit', '0'), 10) || 0;
const FRESH = args.includes('--fresh');

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

// ---------------------------------------------------------------------------
// Seed: copy live → staging, then discard exactly the active user facts
// ---------------------------------------------------------------------------

function seedStaging() {
  const Database = require('better-sqlite3');

  if (path.resolve(STAGING_DATA) === path.resolve(LIVE_DATA)) {
    console.error('ABORT: staging dir resolves to the live data dir.');
    process.exit(2);
  }

  if (FRESH && fs.existsSync(STAGING_DATA)) {
    fs.rmSync(STAGING_DATA, { recursive: true, force: true });
    console.log(`[Replay] removed previous staging store at ${STAGING_DATA}`);
  }
  if (fs.existsSync(path.join(STAGING_DATA, 'chat.db'))) {
    console.log('[Replay] staging store already exists — reusing it (pass --fresh to rebuild)');
    return { reused: true };
  }

  fs.mkdirSync(path.join(STAGING_DATA, 'memory', 'daily'), { recursive: true });
  fs.mkdirSync(path.join(STAGING_DATA, 'memory', 'ops'), { recursive: true });

  // VACUUM INTO rather than a file copy: the live database is in WAL mode and
  // being written by the running server, so copying chat.db alone can capture a
  // torn page or miss committed pages still sitting in the -wal. VACUUM INTO
  // writes a consistent snapshot of what this connection can see, WAL included.
  // The connection is READONLY — checkpointing first would be a write to the
  // live store, which this script must never make.
  const live = new Database(path.join(LIVE_DATA, 'chat.db'), { readonly: true });
  const target = path.join(STAGING_DATA, 'chat.db');
  live.prepare('VACUUM INTO ?').run(target);
  live.close();
  console.log(`[Replay] copied live SQLite → ${target}`);

  const st = new Database(target);
  st.pragma('foreign_keys = ON');

  const before = st.prepare("SELECT COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user'").get().n;

  // Corroborations belonging to the rows about to be discarded go with them;
  // every other corroboration carries over. (A corroboration is evidence ABOUT a
  // fact — it cannot outlive the row it is evidence for.)
  const discardIds = st.prepare("SELECT id FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user'").all().map(r => r.id);
  const delCorr = st.prepare('DELETE FROM fact_corroborations WHERE member_id = ?');
  let corrDropped = 0;
  const delRow = st.prepare('DELETE FROM cluster_members WHERE id = ?');
  const tx = st.transaction((ids) => {
    for (const id of ids) { corrDropped += delCorr.run(id).changes; delRow.run(id); }
  });
  tx(discardIds);

  // Supersession chains that ended at a discarded row now dangle. The pointer is
  // LEFT AS IT IS and reported at the gate rather than nulled: "replaced by
  // something that is no longer here" is the true statement after a replay, and
  // nulling it would quietly convert a superseded fact into one that was retired
  // for no recorded reason. memory_get's chain walk already stops at a missing
  // successor. Ellie decides at the gate whether to leave them.
  const dangling = st.prepare(`
    SELECT COUNT(*) n FROM cluster_members m
    WHERE m.successor_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM cluster_members s WHERE s.id = m.successor_id)
  `).get().n;

  const kept = st.prepare("SELECT COALESCE(subject,'user') s, status, COUNT(*) n FROM cluster_members GROUP BY s, status").all();
  st.close();

  console.log(`[Replay] discarded ${before} active user fact(s) and ${corrDropped} corroboration(s) belonging to them`);
  console.log(`[Replay] carried over: ${kept.map(k => `${k.s}/${k.status}=${k.n}`).join(', ')}`);
  console.log(`[Replay] dangling successor pointers after the discard: ${dangling}`);
  return { reused: false, discarded: before, corrDropped, dangling, carried: kept };
}

// ---------------------------------------------------------------------------
// Re-embed what carried over, so staging is a whole corpus and not half of one
// ---------------------------------------------------------------------------

async function embedCarriedOver(db) {
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const d = db.getSqliteDb();
  const rows = d.prepare("SELECT id, cluster_id, content FROM cluster_members WHERE status='active'").all();
  let ok = 0, fail = 0;
  for (const r of rows) {
    // replaceVector is the funnel's own writer — same path the corrector's
    // reconcile uses when it finds an active fact with no embedding.
    if (await factStore.replaceVector(r.id, r.cluster_id, r.content)) ok++; else fail++;
  }
  console.log(`[Replay] embedded ${ok} carried-over active fact(s) into staging (${fail} failed)`);
  return { embedded: ok, failed: fail };
}

// ---------------------------------------------------------------------------
// The exchanges, in the order they happened
// ---------------------------------------------------------------------------

function collectExchanges(d) {
  const convos = d.prepare('SELECT id, created_at FROM conversations ORDER BY datetime(created_at) ASC').all();
  const out = [];
  for (const c of convos) {
    const msgs = d.prepare(
      'SELECT id, role, content, timestamp, input_modality FROM messages WHERE conversation_id = ? ORDER BY datetime(timestamp) ASC, rowid ASC'
    ).all(c.id);
    for (let i = 0; i < msgs.length; i++) {
      if (msgs[i].role !== 'user') continue;
      // The assistant reply that followed, if there was one. An unanswered user
      // message still gets replayed — intake reads the user's words, and the
      // reply is context.
      const next = msgs[i + 1] && msgs[i + 1].role === 'assistant' ? msgs[i + 1] : null;
      out.push({
        conversationId: c.id,
        conversationDate: String(msgs[i].timestamp || c.created_at || '').slice(0, 10),
        messageId: msgs[i].id,
        userMessage: msgs[i].content,
        assistantMessage: next ? next.content : '',
        // Historical messages carry 'unknown', per the ratified decision: it is a
        // real answer ("we do not know how this arrived"), and the evidence rule
        // ships knowing unknown loses to typed.
        inputModality: ['stt', 'typed'].includes(msgs[i].input_modality) ? msgs[i].input_modality : 'unknown'
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

(async () => {
  const t0 = Date.now();
  const seed = seedStaging();

  const db = require(path.join(ROOT, 'db/database'));
  if (path.resolve(db.getDataDir()) !== path.resolve(STAGING_DATA)) {
    console.error(`ABORT: data dir is ${db.getDataDir()}, expected ${STAGING_DATA}`);
    process.exit(2);
  }
  db.initDatabase();
  await db.initVectorStore();

  const { getConfig } = require(path.join(ROOT, 'db/config'));
  const CONCURRENCY = parseInt(argVal('--concurrency', ''), 10)
    || getConfig().memory?.replay?.concurrency || 6;

  const fx = require(path.join(ROOT, 'db/fact-extractor'));
  const d = db.getSqliteDb();
  const stagingMemoryDir = path.join(STAGING_DATA, 'memory');

  if (!seed.reused) await embedCarriedOver(db);

  let exchanges = collectExchanges(d);
  if (LIMIT) exchanges = exchanges.slice(0, LIMIT);
  console.log(`[Replay] ${exchanges.length} exchange(s) to replay, concurrency ${CONCURRENCY}`);

  const stats = {
    exchanges: exchanges.length, planned: 0, applied: 0, planErrors: 0, applyErrors: 0,
    stored: 0, repeats: 0, events: 0, superseded: 0, refusals: 0
  };

  // Plan in a window, apply the window in order. The window is what bounds how
  // stale a plan can be: at most CONCURRENCY-1 exchanges behind the corpus it is
  // applied against.
  for (let i = 0; i < exchanges.length; i += CONCURRENCY) {
    const window = exchanges.slice(i, i + CONCURRENCY);
    const plans = await Promise.all(window.map(async (ex) => {
      try {
        const plan = await fx.planExtraction({
          userMessage: ex.userMessage,
          assistantMessage: ex.assistantMessage,
          conversationId: ex.conversationId,
          messageId: ex.messageId,
          inputModality: ex.inputModality
        });
        stats.planned++;
        return { ex, plan };
      } catch (err) {
        stats.planErrors++;
        console.error(`[Replay] plan failed (${ex.messageId.slice(0, 8)}): ${err.message}`);
        return { ex, plan: null };
      }
    }));

    for (const { ex, plan } of plans) {
      if (!plan) continue;
      try {
        const res = await fx.applyExtraction(plan, {
          memoryDir: stagingMemoryDir,
          dateStamp: ex.conversationDate || null
        });
        stats.applied++;
        stats.stored += res.stored || 0;
        stats.repeats += res.repeats || 0;
        stats.events += res.events || 0;
        stats.superseded += res.superseded || 0;
        stats.refusals += res.refusals || 0;
      } catch (err) {
        stats.applyErrors++;
        console.error(`[Replay] apply failed (${ex.messageId.slice(0, 8)}): ${err.message}`);
      }
    }

    const done = Math.min(i + CONCURRENCY, exchanges.length);
    if (done % 30 === 0 || done === exchanges.length) {
      const rate = done / ((Date.now() - t0) / 1000);
      const eta = rate > 0 ? Math.round((exchanges.length - done) / rate) : 0;
      console.log(`[Replay] ${done}/${exchanges.length} — ${stats.stored} stored, ${stats.repeats} folded, ` +
        `${stats.events} events, ${stats.refusals} refused (${rate.toFixed(2)}/s, ~${eta}s left)`);
    }
  }

  stats.durationMs = Date.now() - t0;
  stats.seed = seed;
  const outPath = path.join(STAGING_DATA, 'replay-stats.json');
  fs.writeFileSync(outPath, JSON.stringify(stats, null, 2));

  console.log(`\n${'='.repeat(74)}`);
  console.log(`REPLAY COMPLETE in ${(stats.durationMs / 1000 / 60).toFixed(1)} min`);
  console.log(`  exchanges  : ${stats.applied}/${stats.exchanges} applied (${stats.planErrors} plan errors, ${stats.applyErrors} apply errors)`);
  console.log(`  stored     : ${stats.stored}`);
  console.log(`  folded     : ${stats.repeats} (repeats absorbed into a fact already held)`);
  console.log(`  events     : ${stats.events} routed to the day's log`);
  console.log(`  superseded : ${stats.superseded}`);
  console.log(`  refused    : ${stats.refusals} (identity anchor and speaker rule)`);
  console.log(`  stats      : ${outPath}`);
  console.log(`${'='.repeat(74)}\n`);
  process.exit(0);
})().catch(err => { console.error('replay failed:', err); process.exit(1); });
