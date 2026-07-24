#!/usr/bin/env node
/**
 * Regenerate capability-briefing.md from the manifest — Ellie's plain-language
 * script for introducing SNH's organs to it in conversation (the backfill path:
 * knowing through dialogue + reflection, not database inserts). Run after the
 * manifest changes so the briefing stays in sync.
 *
 * Usage: node scripts/write-capability-briefing.js
 */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const manifest = require(path.join(ROOT, 'db/capability-manifest'));

const OUT = path.join(ROOT, 'capability-briefing.md');
fs.writeFileSync(OUT, manifest.getBriefing() + '\n', 'utf8');
console.log(`[write-capability-briefing] wrote ${OUT} (${manifest.getAll().length} organs)`);
