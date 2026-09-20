#!/usr/bin/env node
/**
 * One-time (idempotent) pass tagging existing active self-facts as behavioral
 * CLAIM or DECLARATION — the auditability split the self-coherence audit reads.
 * New self-facts are tagged at extraction time (fact-extractor.classifyClaimType);
 * this backfills everything written before the tag existed. Safe to re-run — once
 * everything is tagged it returns immediately. Needs the brain (LLM) up.
 *
 * Usage: node scripts/classify-self-facts.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
const selfAudit = require(path.join(ROOT, 'db/self-audit'));

(async () => {
  db.initDatabase();
  const res = await selfAudit.classifyExistingSelfFacts();
  console.log(`[classify-self-facts] classified ${res.classified} self-fact(s)`);
  process.exit(0);
})().catch(err => {
  console.error('[classify-self-facts] error:', err);
  process.exit(1);
});
