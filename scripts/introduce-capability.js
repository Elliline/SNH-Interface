#!/usr/bin/env node
/**
 * Ship-day introduction: turn a capability's manifest entry into a stored
 * self-fact, so the entity KNOWS it has the organ instead of only having it.
 * The introduction is a first-person, plain-language DECLARATION (a fact about
 * what's built), processed through SNH's normal self-fact pipeline — where the
 * existing classifier tags it 'declaration' (not an auditable claim).
 *
 * Run this ON SHIP DAY for a NEWLY-shipped capability, after adding its manifest
 * entry. Do NOT run it in bulk for the existing backfill — Ellie introduces those
 * organs in conversation (see capability-briefing.md), so the knowing happens
 * through dialogue and reflection, not database inserts.
 *
 * Usage:
 *   node scripts/introduce-capability.js <capability-id>          # store the self-fact
 *   node scripts/introduce-capability.js <capability-id> --dry    # print only, store nothing
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
const manifest = require(path.join(ROOT, 'db/capability-manifest'));
const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));

(async () => {
  const id = process.argv[2];
  const dry = process.argv.includes('--dry');
  if (!id) {
    console.error('Usage: node scripts/introduce-capability.js <capability-id> [--dry]');
    console.error('Known ids:', manifest.getAll().map(c => c.id).join(', '));
    process.exit(2);
  }
  const entry = manifest.getById(id);
  if (!entry) {
    console.error(`Unknown capability id "${id}". Known:`, manifest.getAll().map(c => c.id).join(', '));
    process.exit(2);
  }

  const intro = manifest.introSentence(id);
  console.log(`[introduce-capability] "${entry.name}"`);
  console.log(`  introduction: ${intro}`);

  if (dry) {
    console.log('  (--dry) nothing stored.');
    process.exit(0);
  }

  db.initDatabase();
  // Process through the normal self-fact pipeline — classify (→declaration),
  // dedup, cluster, store. Same path reflection uses.
  const result = await factExtractor.processSelfFacts([intro], { source: 'capability-intro' });
  try {
    factExtractor.appendToOpsLog(
      `Capability manifest: introduced "${entry.name}" as a self-fact (${result.stored} stored)`,
      path.join(ROOT, 'data/memory/ops')
    );
  } catch (_) { /* best-effort */ }
  console.log(`  stored: ${result.stored} self-fact(s), superseded: ${result.superseded}`);
  process.exit(0);
})().catch(err => {
  console.error('[introduce-capability] error:', err);
  process.exit(1);
});
