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
  const facts = mc.getSelfFacts({ status: 'active' })
    .sort((a, b) => (b.salience - a.salience) || (new Date(a.created_at) - new Date(b.created_at)));

  const embs = [];
  for (const f of facts) { const e = await mc.generateEmbedding(f.content); embs.push(e ? Array.from(e) : null); }

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

  let done = 0;
  for (const s of supersessions) {
    if (mc.supersedeFact(s.dupId, s.keepId)) done++;
  }
  console.log(`\nApplied: ${done} self-fact(s) superseded (kept ${kept.length} active).`);
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
