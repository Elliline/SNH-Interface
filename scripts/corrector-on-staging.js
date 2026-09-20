#!/usr/bin/env node
/**
 * Run the corrector against the STAGING corpus, to convergence.
 *
 * The first gate report measured staging pre-corrector against live post-
 * corrector, which made the two corpora unlike in a way the headline counts hid:
 * 26 near-duplicate pairs sat in staging that the mechanical tier would fold on
 * its first pass. The merge has since added 63 carried facts on top, some of
 * which necessarily overlap what the replay rebuilt. This runs the real
 * corrector over the merged result so the second gate compares like with like.
 *
 * SAME CODE, DIFFERENT CORPUS. `corrector.runPass` — the function the heartbeat
 * calls, unmodified. The only thing that differs is SNH_DATA_DIR, set before
 * db/database.js loads. Same rule as the replay: redirect the process, do not
 * flag the call site.
 *
 * TO CONVERGENCE, not once. A pass is deliberately bounded — 60 writes, 5
 * minutes, 40 near-duplicate pairs, 25 expiries, 10 splits — and the design's
 * answer to "what about the rest" is that resume is just running it again. So
 * this loops until a pass changes nothing, which is what "a full pass over the
 * corpus" actually means here. It stops early if a pass reports no progress or
 * the round cap is hit, and says which.
 *
 * Usage:
 *   node scripts/corrector-on-staging.js [--max-rounds N] [--json]
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

const LIVE_DATA = path.join(ROOT, 'data');
const STAGING_DATA = process.env.SNH_STAGING_DIR || path.join(ROOT, 'data-staging');
process.env.SNH_DATA_DIR = STAGING_DATA;

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const MAX_ROUNDS = parseInt(argVal('--max-rounds', '12'), 10);
const JSON_OUT = args.includes('--json');

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

(async () => {
  if (path.resolve(STAGING_DATA) === path.resolve(LIVE_DATA)) {
    console.error('ABORT: staging dir resolves to the live data dir.');
    process.exit(2);
  }
  if (!fs.existsSync(path.join(STAGING_DATA, 'chat.db'))) {
    console.error(`ABORT: no staging store at ${STAGING_DATA}.`);
    process.exit(2);
  }

  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  if (path.resolve(db.getDataDir()) !== path.resolve(STAGING_DATA)) {
    console.error(`ABORT: data dir is ${db.getDataDir()}, expected ${STAGING_DATA}`);
    process.exit(2);
  }

  const corrector = require(path.join(ROOT, 'db/corrector'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const d = db.getSqliteDb();

  const before = d.prepare(
    "SELECT COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user'"
  ).get().n;

  const t0 = Date.now();
  const rounds = [];
  let stopReason = 'converged';

  for (let r = 1; r <= MAX_ROUNDS; r++) {
    console.log(`\n${'─'.repeat(74)}\n[Staging corrector] round ${r}/${MAX_ROUNDS}\n${'─'.repeat(74)}`);
    const res = await corrector.runPass({});
    const changes = (res.merged || 0) + (res.expired || 0) + (res.split || 0) + (res.superseded || 0);
    rounds.push({
      round: r, passId: res.passId, changes,
      merged: res.merged || 0, expired: res.expired || 0, split: res.split || 0,
      superseded: res.superseded || 0, unresolved: res.unresolved || 0,
      refusedLocked: res.refusedLocked || 0,
      reconciled: res.reconciled || null,
      stopped: res.stopped || null,
      durationMs: res.durationMs || null
    });
    console.log(`[Staging corrector] round ${r}: ${changes} change(s)` +
      `${res.stopped ? ` — pass stopped early: ${res.stopped}` : ''}`);

    if (changes === 0) { stopReason = 'converged'; break; }
    if (r === MAX_ROUNDS) stopReason = `round cap reached (${MAX_ROUNDS}) with work still being done`;
  }

  const after = d.prepare(
    "SELECT COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user'"
  ).get().n;

  // The ledger for THIS run — the pass ids we just made, nothing older.
  const passIds = new Set(rounds.map(r => r.passId));
  const entries = d.prepare(
    'SELECT * FROM corrections_ledger ORDER BY datetime(created_at) ASC'
  ).all().filter(e => passIds.has(e.pass_id));

  // A refusal is not a correction, and the ledger does not distinguish them by
  // action — an unresolved raise is recorded as `semantic/supersede` with
  // reversible = 0, beside the supersessions that actually happened. Grouping on
  // action alone reports 35 supersessions when 5 rows changed and 30 pairs were
  // handed back untouched. Split on reversibility, which is the field that
  // actually carries the difference.
  const byAction = {};
  for (const e of entries) {
    const k = `${e.tier}/${e.action}${e.reversible ? '' : ':raised'}`;
    (byAction[k] = byAction[k] || []).push(e);
  }

  const unresolvedEntries = entries.filter(e => !e.reversible);
  const totals = rounds.reduce((a, r) => ({
    merged: a.merged + r.merged, expired: a.expired + r.expired,
    split: a.split + r.split, superseded: a.superseded + r.superseded,
    unresolved: a.unresolved + r.unresolved, refusedLocked: a.refusedLocked + r.refusedLocked
  }), { merged: 0, expired: 0, split: 0, superseded: 0, unresolved: 0, refusedLocked: 0 });

  const out = {
    generatedAt: new Date().toISOString(),
    stagingDir: STAGING_DATA,
    rounds, totals, stopReason,
    activeUserFacts: { before, after, delta: after - before },
    ledgerEntries: entries.length,
    durationMs: Date.now() - t0
  };
  fs.writeFileSync(path.join(STAGING_DATA, 'corrector-staging-result.json'), JSON.stringify({ ...out, entries }, null, 2));

  if (JSON_OUT) { console.log(JSON.stringify({ ...out, entries }, null, 2)); process.exit(0); }

  const line = (c = '=') => console.log(c.repeat(78));
  console.log('');
  line();
  console.log('CORRECTOR PASS AGAINST STAGING');
  console.log(`${rounds.length} round(s), ${((Date.now() - t0) / 1000 / 60).toFixed(1)} min — ${stopReason}`);
  line();

  console.log('\n--- WHAT IT DID ---');
  console.log(`  duplicates folded      : ${totals.merged}`);
  console.log(`  events moved to a log  : ${totals.expired}`);
  console.log(`  compounds split        : ${totals.split}`);
  console.log(`  contradictions resolved: ${totals.superseded}`);
  console.log(`  left unresolved        : ${totals.unresolved}`);
  console.log(`  refused by the lock    : ${totals.refusedLocked}`);
  console.log(`  active user facts      : ${before} → ${after} (${after - before >= 0 ? '+' : ''}${after - before})`);

  console.log('\n--- PER ROUND ---');
  for (const r of rounds) {
    console.log(`  ${String(r.round).padStart(2)}  merged ${String(r.merged).padStart(2)}  expired ${String(r.expired).padStart(2)}  ` +
      `split ${String(r.split).padStart(2)}  superseded ${String(r.superseded).padStart(2)}  unresolved ${String(r.unresolved).padStart(2)}` +
      `${r.stopped ? `   [${r.stopped}]` : ''}`);
  }

  console.log('\n--- THE LEDGER ---');
  const titles = {
    'mechanical/merge': 'MERGE — one fact held twice, folded into one',
    'mechanical/expire': "EXPIRE — an event stored as a fact, moved to the day's log",
    'mechanical/split': 'SPLIT — a compound broken into atoms',
    'mechanical/reconcile': 'RECONCILE — SQLite and the vector index brought back into step',
    'semantic/supersede': 'SUPERSEDE — a contradiction resolved on evidence, and a fact retired',
    'semantic/supersede:raised': 'RAISED, NOTHING CHANGED — contradictions the evidence could not separate',
    'mechanical/merge:raised': 'MERGE REFUSED — nothing changed',
    'mechanical/carry': 'CARRY — brought across from the live corpus',
    'mechanical/carry:raised': 'CARRY — brought across from the live corpus (adds a row, so not revertible by the ledger)'
  };
  for (const [key, items] of Object.entries(byAction)) {
    console.log(`\n── ${titles[key] || key}  (${items.length})`);
    for (const e of items) {
      const changed = e.reversible ? '' : '   [nothing changed]';
      console.log(`\n   ${e.action.toUpperCase()}  ${String(e.target_id || '').slice(0, 8)}  "${trunc(e.target_text, 96)}"${changed}`);
      if (e.survivor_text) console.log(`     survivor: ${String(e.survivor_id || '').slice(0, 8)}  "${trunc(e.survivor_text, 96)}"`);
      console.log(`     why: ${trunc(e.reason, 240)}`);
      if (e.evidence) {
        try {
          const ev = JSON.parse(e.evidence);
          if (ev.deciding_axis) {
            console.log(`     decided on: ${ev.deciding_axis}`);
            console.log(`       winner: ${JSON.stringify(ev.winner)}`);
            console.log(`       loser : ${JSON.stringify(ev.loser)}`);
          }
        } catch { /* not JSON */ }
      }
    }
  }
  if (!entries.length) console.log('  (no entries — the corpus was already in the state the corrector wants)');

  if (unresolvedEntries.length) {
    console.log(`\n--- ${unresolvedEntries.length} ENTRY/ENTRIES WHERE NOTHING CHANGED ---`);
    console.log('  Refusals and unresolved raises land in the same ledger as real changes,');
    console.log('  with reversible = 0. They are recorded, not applied.');
  }

  // Every retirement, in one place, because this is the part with consequences.
  const applied = entries.filter(e => e.reversible && e.action === 'supersede');
  if (applied.length) {
    console.log(`\n--- ${applied.length} FACT(S) ACTUALLY RETIRED — check these ---`);
    console.log('  Each is reversible:  node scripts/revert-correction.js <ledger-id>');
    console.log('  (run it with SNH_DATA_DIR=data-staging to act on this corpus)\n');
    for (const e of applied) {
      console.log(`  ${e.id}`);
      console.log(`    retired : "${trunc(e.target_text, 92)}"`);
      console.log(`    in favour of: "${trunc(e.survivor_text, 92)}"`);
      try { console.log(`    decided on: ${JSON.parse(e.evidence).deciding_axis}`); } catch { /* none */ }
      console.log('');
    }
  }

  console.log('');
  line();
  process.exit(0);
})().catch(err => { console.error('staging corrector run failed:', err); process.exit(1); });
