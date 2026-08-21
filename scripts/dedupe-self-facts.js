#!/usr/bin/env node
/**
 * One-off cleanup: collapse near-identical ACTIVE self-facts left behind by the
 * reflection stutter (multiple cycles restamping the identity with reworded-
 * identical observations). Within each near-duplicate group we KEEP the strongest
 * representative (highest salience, then earliest observed) and SUPERSEDE the rest
 * — history is preserved, nothing is deleted.
 *
 * Uses the same embedding-similarity + threshold as the live self-fact dedup
 * (identity.selfFactDedupThreshold, default 0.88). Idempotent: re-running finds
 * nothing once the duplicates are superseded.
 *
 * Usage: node scripts/dedupe-self-facts.js [--apply]
 *   (dry-run by default; pass --apply to actually supersede)
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
const { getConfig } = require(path.join(ROOT, 'db/config'));

function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den === 0 ? 0 : dot / den;
}

(async () => {
  const apply = process.argv.includes('--apply');
  db.initDatabase();
  await db.initVectorStore();
  const mc = require(path.join(ROOT, 'db/memory-clusters'));

  const cfg = getConfig();
  const threshold = Number.isFinite(cfg.identity?.selfFactDedupThreshold)
    ? cfg.identity.selfFactDedupThreshold : 0.88;

  // getSelfFacts already orders by salience DESC, created_at DESC. We want the
  // KEPT canonical to be highest salience then EARLIEST, so re-sort ascending by
  // time within equal salience.
  // LOCKED facts are excluded outright — they can be neither the duplicate that
  // gets superseded nor the canonical that absorbs others. This script calls
  // memoryClusters.supersedeFact directly, so it does NOT pass through
  // db/fact-store.js and the identity lock's guard never sees it; a near-
  // duplicate of the name fact would otherwise retire a chosen name during
  // routine cleanup, which is precisely the silent loss the lock exists to stop.
  const allActive = mc.getSelfFacts({ status: 'active' });
  const lockedOut = allActive.filter(f => f.locked);
  if (lockedOut.length) {
    console.log(`Skipping ${lockedOut.length} locked identity fact(s) — not eligible for dedup:`);
    for (const f of lockedOut) console.log(`  [${f.lock_category}] "${f.content.slice(0, 80)}"`);
    console.log('');
  }
  const facts = allActive
    .filter(f => !f.locked)
    .sort((a, b) => (b.salience - a.salience) || (new Date(a.created_at) - new Date(b.created_at)));

  // READ the stored vectors rather than regenerating them. This loop had
  // the same defect as the one in processSelfFacts: one embedding round
  // trip per active self-fact, ~950ms each, to reproduce vectors already
  // sitting in cluster_embeddings (verified identical, cosine 1.000000).
  // At 402 facts that is over six minutes before this script decides
  // anything; it grows with the corpus and this script exists to be run
  // on a corpus that has grown.
  const storedEmbs = await mc.getStoredEmbeddings(facts.map(f => f.id));
  const embs = [];
  let embeddedOnDemand = 0;
  for (const f of facts) {
    let e = storedEmbs.get(f.id);
    if (!e) { e = await mc.generateEmbedding(f.content); embeddedOnDemand++; }
    embs.push(e ? Array.from(e) : null);
  }
  const missingVectors = embs.filter(e => !e).length;
  if (embeddedOnDemand) {
    console.log(`${embeddedOnDemand}/${facts.length} fact(s) had no stored vector and were embedded on demand.`);
  }
  // A dedup run that cannot see part of the corpus is not a dedup run.
  // Refusing is the only safe answer here: this script SUPERSEDES facts,
  // and deciding which of two is a duplicate while blind to some of them
  // is how the July near-miss on his name happened.
  if (missingVectors) {
    console.error(`\nREFUSING TO RUN: ${missingVectors} of ${facts.length} self-facts have no usable embedding.`);
    console.error('Deduplicating while blind to part of the corpus can supersede a fact whose');
    console.error('near-duplicate was never compared. Fix the embedding provider and re-run.');
    process.exit(2);
  }

  const kept = []; // { id, content, emb }
  const supersessions = []; // { dupId, dupContent, keepId, keepContent, sim }
  for (let i = 0; i < facts.length; i++) {
    const emb = embs[i];
    let dupOf = null, dupSim = 0;
    if (emb) {
      for (const k of kept) {
        if (!k.emb) continue;
        const s = cosine(emb, k.emb);
        if (s >= threshold && s > dupSim) { dupSim = s; dupOf = k; }
      }
    }
    if (dupOf) {
      supersessions.push({ dupId: facts[i].id, dupContent: facts[i].content, keepId: dupOf.id, keepContent: dupOf.content, sim: dupSim });
    } else {
      kept.push({ id: facts[i].id, content: facts[i].content, emb });
    }
  }

  console.log(`Active self-facts: ${facts.length} | threshold: ${threshold}`);
  console.log(`Near-duplicate groups collapsed: ${supersessions.length} supersession(s)\n`);
  for (const s of supersessions) {
    console.log(`  sim ${s.sim.toFixed(3)}`);
    console.log(`    SUPERSEDE: "${s.dupContent.slice(0, 80)}"`);
    console.log(`    KEEP:      "${s.keepContent.slice(0, 80)}"`);
  }

  if (!apply) {
    console.log(`\n(dry-run — pass --apply to supersede these ${supersessions.length} duplicate(s))`);
    process.exit(0);
  }

  // THROUGH THE FUNNEL (2026-08-18), not straight at memoryClusters. That call
  // writes the row and nothing else: it skips the identity lock, and since the
  // ledger entry became part of the write it skips that too — so a re-run of
  // this script would mint exactly the unrevertable changes the funnel exists to
  // prevent. Locked facts are already excluded above; going through fact-store
  // means they would be refused anyway, which is the belt this wants.
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  let done = 0;
  for (const s of supersessions) {
    const res = await factStore.supersede(s.dupId, s.keepId, {
      caller: 'scripts/dedupe-self-facts.js',
      ledger: {
        tier: 'mechanical',
        reason: `Two self-facts said the same thing; this one was folded into "${String(s.keepContent).slice(0, 100)}" and kept as history.`,
        evidence: { repair: 'dedupe-self-facts', kept_id: s.keepId }
      }
    });
    if (res.ok) done++;
    else console.log(`  !! ${s.dupId.slice(0, 8)} not superseded: ${res.reason || 'unknown'}`);
  }
  console.log(`\nApplied: ${done} self-fact(s) superseded (kept ${kept.length} active).`);
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
