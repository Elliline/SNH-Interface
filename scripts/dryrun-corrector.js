#!/usr/bin/env node
/**
 * Dry-run a full corrector pass against the LIVE corpus.
 *
 * Same function the heartbeat calls (`corrector.runPass`), same enumeration,
 * same judges, same evidence-dominance rules — `dryRun` only stops the writes.
 * The Phase 2a rule again: a rehearsal that runs different code proves nothing
 * about the code that will run.
 *
 * SAFE. Belt and braces: every fact-store mutator is replaced with a thrower
 * before the pass starts, so a missed dryRun branch fails loudly instead of
 * quietly editing the corpus this is supposed to be measuring.
 *
 * Usage:
 *   node scripts/dryrun-corrector.js            # full plan
 *   node scripts/dryrun-corrector.js --json
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

(async () => {
  const json = process.argv.includes('--json');
  db.initDatabase();
  await db.initVectorStore();

  // Guard rail: nothing may write.
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  for (const fn of ['supersede', 'retire', 'expire', 'restore', 'reword', 'dropVector', 'replaceVector', 'absorbRepeat', 'absorbDuplicate', 'recordCorroboration']) {
    factStore[fn] = () => { throw new Error(`dry-run: fact-store.${fn} must not be called`); };
  }
  const clusters = require(path.join(ROOT, 'db/memory-clusters'));
  clusters.assignToCluster = () => { throw new Error('dry-run: assignToCluster must not be called'); };

  const corrector = require(path.join(ROOT, 'db/corrector'));
  const t0 = Date.now();
  const res = await corrector.runPass({ dryRun: true });

  if (json) { console.log(JSON.stringify(res, null, 2)); process.exit(0); }

  console.log(`\n${'='.repeat(78)}`);
  console.log('CORRECTOR DRY RUN — nothing was written');
  console.log(`pass ${res.passId}`);
  console.log('='.repeat(78));

  const byAction = {};
  for (const p of res.plan) {
    const k = `${p.tier}/${p.action}`;
    (byAction[k] = byAction[k] || []).push(p);
  }

  const order = ['mechanical/merge', 'mechanical/expire', 'mechanical/split', 'mechanical/reconcile', 'semantic/supersede'];
  const titles = {
    'mechanical/merge': 'MERGE — duplicate and subset facts folded into one',
    'mechanical/expire': 'EXPIRE — events that were stored as facts, moved to the day\'s log',
    'mechanical/split': 'SPLIT — compound facts broken into atoms',
    'mechanical/reconcile': 'RECONCILE — SQLite and the vector index brought back into step',
    'semantic/supersede': 'SUPERSEDE — contradictions resolved on evidence'
  };

  for (const key of order) {
    const items = byAction[key];
    if (!items || !items.length) continue;
    console.log(`\n── ${titles[key]}  (${items.length})`);
    for (const p of items) {
      console.log(`\n   ${p.action.toUpperCase()}  ${String(p.targetId || '').slice(0, 8)}  "${trunc(p.targetText, 100)}"`);
      if (p.survivorText) console.log(`     survivor: ${String(p.survivorId || '').slice(0, 8)}  "${trunc(p.survivorText, 100)}"`);
      if (p.atoms) for (const a of p.atoms) console.log(`     atom: "${a}"`);
      console.log(`     why: ${trunc(p.reason, 220)}`);
      if (p.evidence && p.evidence.deciding_axis) {
        console.log(`     decided on: ${p.evidence.deciding_axis}`);
        console.log(`       winner: ${JSON.stringify(p.evidence.winner)}`);
        console.log(`       loser : ${JSON.stringify(p.evidence.loser)}`);
      }
    }
  }

  if (res.unresolvedPairs && res.unresolvedPairs.length) {
    console.log(`\n── UNRESOLVED — contradictions the evidence cannot separate  (${res.unresolvedPairs.length})`);
    console.log('   These are NOT corrected. They are raised for Ellie.');
    for (const u of res.unresolvedPairs) {
      console.log(`\n   "${trunc(u.a.text, 90)}"`);
      console.log(`   vs "${trunc(u.b.text, 90)}"`);
    }
  }

  console.log(`\n${'='.repeat(78)}`);
  console.log(`PLAN  ${res.merged} merge(s), ${res.expired} expiry(ies), ${res.split} split(s), ` +
              `${res.superseded} supersession(s), ${res.unresolved} unresolved`);
  if (res.reconciled) {
    console.log(`      reconcile would fix: ${res.reconciled.inactiveWithVector} inactive-with-vector, ` +
                `${res.reconciled.activeNoVector} active-without-vector, ${res.reconciled.orphanVectors} orphan vector(s)`);
  }
  if (res.stopped) console.log(`      STOPPED EARLY: ${res.stopped}`);
  console.log(`      ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.log('='.repeat(78));
  process.exit(0);
})().catch(err => { console.error('dry run failed:', err); process.exit(1); });
