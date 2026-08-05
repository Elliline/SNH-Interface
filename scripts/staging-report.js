#!/usr/bin/env node
/**
 * The staging gate report — what the replay produced, against what is live.
 *
 * This is the artefact the cutover decision is made from, so it is built to be
 * argued with: every number is counted from the two stores rather than carried
 * over from the replay's own bookkeeping, and every "why is this fact gone"
 * is a classification with its evidence attached rather than an assertion.
 *
 * It runs POINTED AT STAGING (SNH_DATA_DIR), which gives it the staging vector
 * index, and opens the live database read-only alongside for comparison. That
 * asymmetry is deliberate: the interesting question is "what happened to each
 * live fact", and answering it needs semantic search over the REBUILT corpus.
 *
 * READ-ONLY against both stores.
 *
 * Usage:
 *   SNH_DATA_DIR=data-staging node scripts/staging-report.js [--json] [--sample N]
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

const STAGING_DATA = process.env.SNH_DATA_DIR || path.join(ROOT, 'data-staging');
process.env.SNH_DATA_DIR = STAGING_DATA;

const LIVE_DB = path.join(ROOT, 'data', 'chat.db');

const args = process.argv.slice(2);
const JSON_OUT = args.includes('--json');
const SAMPLE = parseInt((args[args.indexOf('--sample') + 1] || ''), 10) || 20;

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

// Words distinctive enough to look for in a log line. Drops the scaffolding
// every fact shares ("user", "the", "have") so a match means something.
const keyWords = (text) => String(text).toLowerCase()
  .replace(/[^a-z0-9 ]/g, ' ')
  .split(/\s+/)
  .filter(w => w.length > 4 && !['user', 'their', 'there', 'about', 'which', 'would', 'these', 'those'].includes(w));

(async () => {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  const staging = db.getSqliteDb();

  const Database = require('better-sqlite3');
  const live = new Database(LIVE_DB, { readonly: true });

  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));

  // ---- 1. corpus stats -----------------------------------------------------
  const countBy = (conn) => {
    const rows = conn.prepare(
      "SELECT COALESCE(subject,'user') s, status, COUNT(*) n FROM cluster_members GROUP BY s, status"
    ).all();
    const get = (s, st) => rows.find(r => r.s === s && r.status === st)?.n ?? 0;
    return {
      user_active: get('user', 'active'), user_inactive: get('user', 'inactive'),
      self_active: get('self', 'active'), self_inactive: get('self', 'inactive'),
      world_active: get('world', 'active'), world_inactive: get('world', 'inactive'),
      clusters: conn.prepare('SELECT COUNT(*) n FROM memory_clusters').get().n,
      user_clusters: conn.prepare("SELECT COUNT(*) n FROM memory_clusters WHERE COALESCE(subject,'user')='user'").get().n,
      corroborations: conn.prepare('SELECT COUNT(*) n FROM fact_corroborations').get().n,
      ledger: conn.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n
    };
  };
  const liveStats = countBy(live);
  const stagingStats = countBy(staging);

  let replayStats = {};
  try { replayStats = JSON.parse(fs.readFileSync(path.join(STAGING_DATA, 'replay-stats.json'), 'utf8')); } catch { /* not finished */ }

  // Compound splits are not in applyExtraction's return — they happen inside the
  // plan half. Counted from the run's own log rather than guessed.
  let compoundsSplit = null;
  try {
    const log = fs.readFileSync(path.join(ROOT, 'data-staging-replay.log'), 'utf8');
    compoundsSplit = (log.match(/Split compound into/g) || []).length;
  } catch { /* log may be elsewhere */ }

  // ---- 2. the diff ---------------------------------------------------------
  const liveFacts = live.prepare(
    "SELECT id, content, salience, conversation_id, source FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user' ORDER BY salience DESC"
  ).all();
  const stagingFacts = staging.prepare(
    "SELECT id, content, salience, conversation_id FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user' ORDER BY salience DESC"
  ).all();

  const norm = (t) => String(t).trim().toLowerCase().replace(/\s+/g, ' ');
  const stagingByText = new Set(stagingFacts.map(f => norm(f.content)));
  const liveByText = new Set(liveFacts.map(f => norm(f.content)));

  const liveConvos = new Set(live.prepare('SELECT id FROM conversations').all().map(r => r.id));

  // Daily logs the replay wrote — the destination for anything that was an event.
  let dailyText = '';
  try {
    const dir = path.join(STAGING_DATA, 'memory', 'daily');
    for (const f of fs.readdirSync(dir)) dailyText += fs.readFileSync(path.join(dir, f), 'utf8').toLowerCase();
  } catch { /* none */ }

  const onlyLive = liveFacts.filter(f => !stagingByText.has(norm(f.content)));
  const onlyStaging = stagingFacts.filter(f => !liveByText.has(norm(f.content)));

  // ---- 2a. COVERAGE — the number the cutover decision turns on --------------
  //
  // "How many live facts are absent from the rebuilt corpus, and why" is the
  // question, and the answer has to be per-PRODUCER. A replay rebuilds what
  // INTAKE produced; anything written by another path (the daily-log archive
  // import, the corrector's splits, a manual correction) has no conversation to
  // be replayed from and cannot come back by this route no matter how good the
  // pipeline is. Reporting a single "202 vs 256" hides that distinction, and it
  // is the distinction Ellie needs.
  const twinFloor = 0.80;
  const coverage = {};   // source -> { total, represented, missing, samples }
  for (const f of liveFacts) {
    const src = f.source || '(null)';
    coverage[src] = coverage[src] || { total: 0, represented: 0, missing: 0, samples: [] };
    coverage[src].total++;
    const { candidates } = await memoryClusters.findActiveNeighbours(f.content, {
      subject: 'user', threshold: twinFloor, limit: 1
    });
    if (candidates.length) {
      coverage[src].represented++;
      f._twin = candidates[0];
    } else {
      coverage[src].missing++;
      if (coverage[src].samples.length < 6) coverage[src].samples.push({ text: f.content, salience: f.salience });
    }
  }
  const totalMissing = Object.values(coverage).reduce((a, c) => a + c.missing, 0);

  // Classify a sample, with evidence. Semantic lookup runs against the STAGING
  // index, so "reworded" means a fact that survives saying the same thing.
  const classified = [];
  for (const f of onlyLive.slice(0, SAMPLE)) {
    let why = 'not re-derived', detail = 'The pipeline did not produce this fact from the surviving source.';

    const near = f._twin || null;

    if (near) {
      why = 'reworded / folded';
      detail = `staging holds "${trunc(near.content, 74)}" at ${near.similarity.toFixed(3)}`;
    } else if (f.conversation_id && !liveConvos.has(f.conversation_id)) {
      why = 'source deleted';
      detail = `its conversation ${String(f.conversation_id).slice(0, 8)} is no longer in the database — a replay cannot rebuild what was deleted`;
    } else if (f.source && f.source !== 'fact-extraction') {
      // NOT the same thing as a missing conversation id. 252 of 256 live facts
      // predate provenance capture and carry no conversation_id, yet the
      // conversations themselves are all still there — the replay reads those
      // directly. What actually cannot come back is a fact no CONVERSATION
      // produced: the daily-log archive import, a corrector split, a manual
      // correction.
      why = `not produced by intake (${f.source})`;
      detail = 'written by a path other than conversation extraction, so a replay of conversations has nothing to rebuild it from';
    } else {
      const kw = keyWords(f.content);
      const hits = kw.filter(w => dailyText.includes(w)).length;
      if (kw.length && hits / kw.length >= 0.6) {
        why = 'event-routed';
        detail = `${hits}/${kw.length} of its distinctive words appear in the replayed day's logs — routed to the log rather than stored as a fact`;
      }
    }
    classified.push({ id: f.id, text: f.content, salience: f.salience, why, detail });
  }

  // ---- 2b. what the corrector would still have to do -----------------------
  //
  // The instruction is that the corrector does NOT run against staging, which
  // makes the two corpora unlike in a way the headline counts hide: live's
  // number is post-corrector and staging's is pre-corrector. Counting the
  // near-duplicate pairs left in staging says how much of that gap is work the
  // mechanical tier would do on its first pass, rather than a real difference in
  // what was learned.
  const dupFloor = 0.86;
  let nearDupPairs = 0;
  const nearDupExamples = [];
  const seenPair = new Set();
  for (const f of stagingFacts) {
    const { candidates } = await memoryClusters.findActiveNeighbours(f.content, {
      subject: 'user', threshold: dupFloor, limit: 4, includeVerbatim: true
    });
    for (const c of candidates) {
      if (c.memberId === f.id) continue;
      const key = [f.id, c.memberId].sort().join('|');
      if (seenPair.has(key)) continue;
      seenPair.add(key);
      nearDupPairs++;
      if (nearDupExamples.length < 8) {
        nearDupExamples.push({ a: f.content, b: c.content, similarity: c.similarity });
      }
    }
  }

  // ---- 3. ambiguities the replay could not decide ---------------------------
  const ambiguities = [];

  const dangling = staging.prepare(`
    SELECT COUNT(*) n FROM cluster_members m
    WHERE m.successor_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM cluster_members s WHERE s.id = m.successor_id)
  `).get().n;
  if (dangling) {
    ambiguities.push({
      what: `${dangling} inactive fact(s) point at a successor that no longer exists`,
      why: 'They were superseded by an active user-fact, and every active user-fact was discarded and rebuilt. The pointer is TRUE ("replaced by something no longer here") but leads nowhere.',
      options: 'Leave them (history stays readable, chain walk stops early), or null the pointers (loses the fact that they were replaced rather than retired). Not decided by the replay.'
    });
  }

  const noSource = staging.prepare(
    "SELECT COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user' AND conversation_id IS NULL"
  ).get().n;
  if (noSource) {
    ambiguities.push({
      what: `${noSource} rebuilt fact(s) carry no conversation id`,
      why: 'Every replayed fact should carry the conversation it came from. Any that do not are worth looking at before cutover.',
      options: 'Inspect before cutover.'
    });
  }

  const orphanClusters = staging.prepare(`
    SELECT COUNT(*) n FROM memory_clusters c
    WHERE NOT EXISTS (SELECT 1 FROM cluster_members m WHERE m.cluster_id = c.id AND m.status='active')
  `).get().n;
  if (orphanClusters) {
    ambiguities.push({
      what: `${orphanClusters} cluster(s) in staging hold no active member`,
      why: 'Left behind when the live active user-facts were discarded. Clusters are display-only, so this is cosmetic, but the Map would draw empty groups.',
      options: 'Prune at cutover, or leave — they cost nothing but a row.'
    });
  }

  const out = {
    generatedAt: new Date().toISOString(),
    stagingDir: STAGING_DATA,
    replay: replayStats,
    compoundsSplit,
    stats: { live: liveStats, staging: stagingStats },
    coverage: { floor: twinFloor, totalMissing, bySource: coverage },
    correctorBacklog: { floor: dupFloor, nearDuplicatePairs: nearDupPairs, examples: nearDupExamples },
    diff: {
      onlyLive: onlyLive.length, onlyStaging: onlyStaging.length,
      identicalText: liveFacts.length - onlyLive.length,
      sampleOnlyLive: classified,
      sampleOnlyStaging: onlyStaging.slice(0, SAMPLE).map(f => ({ id: f.id, text: f.content, salience: f.salience }))
    },
    ambiguities
  };

  if (JSON_OUT) { console.log(JSON.stringify(out, null, 2)); process.exit(0); }

  const line = (l = '=') => console.log(l.repeat(78));
  line();
  console.log('STAGING GATE REPORT');
  console.log(`staging: ${STAGING_DATA}`);
  line();

  console.log('\n--- CORPUS ---');
  const row = (label, a, b) => console.log(`  ${label.padEnd(26)} live ${String(a).padStart(6)}    staging ${String(b).padStart(6)}   ${b - a >= 0 ? '+' : ''}${b - a}`);
  row('user facts (active)', liveStats.user_active, stagingStats.user_active);
  row('user facts (inactive)', liveStats.user_inactive, stagingStats.user_inactive);
  row('self facts (active)', liveStats.self_active, stagingStats.self_active);
  row('self facts (inactive)', liveStats.self_inactive, stagingStats.self_inactive);
  row('world facts (active)', liveStats.world_active, stagingStats.world_active);
  row('clusters (all)', liveStats.clusters, stagingStats.clusters);
  row('clusters (user)', liveStats.user_clusters, stagingStats.user_clusters);
  row('corroborations', liveStats.corroborations, stagingStats.corroborations);
  row('corrections ledger', liveStats.ledger, stagingStats.ledger);

  console.log('\n--- REPLAY ---');
  console.log(`  exchanges replayed : ${replayStats.applied ?? '?'} / ${replayStats.exchanges ?? '?'}` +
    `  (${replayStats.planErrors ?? '?'} plan errors, ${replayStats.applyErrors ?? '?'} apply errors)`);
  console.log(`  facts stored       : ${replayStats.stored ?? '?'}`);
  console.log(`  repeats folded     : ${replayStats.repeats ?? '?'}`);
  console.log(`  events → day's log : ${replayStats.events ?? '?'}`);
  console.log(`  supersessions      : ${replayStats.superseded ?? '?'}`);
  console.log(`  intake refusals    : ${replayStats.refusals ?? '?'}`);
  console.log(`  compounds split    : ${compoundsSplit ?? '?'}`);
  if (replayStats.durationMs) console.log(`  duration           : ${(replayStats.durationMs / 60000).toFixed(1)} min`);

  console.log('\n--- COVERAGE: is each live fact represented in the rebuild? ---');
  console.log(`  A live fact counts as represented if staging holds one within ${twinFloor} cosine.`);
  console.log(`  Broken out by what PRODUCED it, because a replay rebuilds what intake`);
  console.log(`  produced and nothing else.\n`);
  console.log(`  ${'source'.padEnd(22)} ${'total'.padStart(6)} ${'represented'.padStart(12)} ${'missing'.padStart(8)}`);
  for (const [src, c] of Object.entries(coverage).sort((a, b) => b[1].total - a[1].total)) {
    console.log(`  ${src.padEnd(22)} ${String(c.total).padStart(6)} ${String(c.represented).padStart(12)} ${String(c.missing).padStart(8)}`);
  }
  console.log(`  ${'TOTAL'.padEnd(22)} ${String(liveFacts.length).padStart(6)} ${String(liveFacts.length - totalMissing).padStart(12)} ${String(totalMissing).padStart(8)}`);
  for (const [src, c] of Object.entries(coverage)) {
    if (!c.samples.length) continue;
    console.log(`\n  missing from ${src}:`);
    for (const sm of c.samples) console.log(`    (sal ${String(sm.salience).padStart(2)}) "${trunc(sm.text, 66)}"`);
  }

  console.log('\n--- WHAT THE CORRECTOR WOULD STILL DO ---');
  console.log(`  The corrector does not run against staging, by instruction. Live's count is`);
  console.log(`  post-corrector and staging's is pre-corrector, so part of the gap below is`);
  console.log(`  work its mechanical tier would do on a first pass rather than a difference`);
  console.log(`  in what was learned.`);
  console.log(`  near-duplicate pairs at >= ${dupFloor}: ${nearDupPairs}`);
  for (const e of nearDupExamples) {
    console.log(`    ${e.similarity.toFixed(3)}  "${trunc(e.a, 58)}"`);
    console.log(`           "${trunc(e.b, 58)}"`);
  }

  console.log('\n--- DIFF ---');
  console.log(`  identical in both  : ${out.diff.identicalText}`);
  console.log(`  live only          : ${onlyLive.length}`);
  console.log(`  staging only       : ${onlyStaging.length}`);

  console.log(`\n  LIVE ONLY — ${classified.length} sampled, with why:\n`);
  for (const c of classified) {
    console.log(`   [${c.why}] "${trunc(c.text, 72)}"`);
    console.log(`      ${c.detail}`);
  }

  console.log(`\n  STAGING ONLY — ${out.diff.sampleOnlyStaging.length} sampled:\n`);
  for (const c of out.diff.sampleOnlyStaging) {
    console.log(`   (sal ${String(c.salience).padStart(2)}) "${trunc(c.text, 72)}"`);
  }

  console.log('\n--- AMBIGUOUS — for Ellie, not guessed ---');
  if (!ambiguities.length) console.log('  none');
  for (const a of ambiguities) {
    console.log(`\n  ${a.what}`);
    console.log(`    why    : ${a.why}`);
    console.log(`    options: ${a.options}`);
  }
  console.log('');
  line();
  process.exit(0);
})().catch(err => { console.error('report failed:', err); process.exit(1); });
