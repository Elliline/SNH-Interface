#!/usr/bin/env node
/**
 * One-time migration: split operational noise out of the daily memory logs.
 *
 * Errors, timeouts, liveness/circuit-breaker events, heartbeat maintenance
 * reports, agent-pool pass telemetry, and per-exchange "N facts extracted"
 * markers were historically written into the daily log — which is injected into
 * every chat's system context. This moves those entries into a parallel ops log
 * (data/memory/ops/<date>.md, surfaced in the Thinking tab, never injected),
 * leaving the daily log with only cognitively meaningful entries (facts,
 * supersessions, salience reasoning, reflections, initiatives, questions).
 *
 * Idempotent: re-running only moves entries that still match the ops patterns.
 * Originals are backed up to <file>.pre-ops-split.bak before rewriting.
 *
 * Usage:
 *   node scripts/migrate-ops-split.js                 # today + yesterday
 *   node scripts/migrate-ops-split.js 2026-07-08 ...  # explicit dates
 *   node scripts/migrate-ops-split.js --dry-run
 */
const fs = require('fs');
const path = require('path');
const { splitDailyBlocks } = require('../db/injection-budget');
const { getLocalDateStamp } = require('../db/datetime');

const DAILY_DIR = path.join(__dirname, '../data/memory/daily');
const OPS_DIR = path.join(__dirname, '../data/memory/ops');
const DRY_RUN = process.argv.includes('--dry-run');

// A block is operational if any of these match its text.
const OPS_PATTERNS = [
  /^##\s*Heartbeat Report/m,      // periodic maintenance report (tables/anomalies)
  /Chat exchange with .* facts extracted/,
  /Agent pool pass/,
  /Heartbeat: brain (unreachable|wedged)/,
  /brain wedged mid-cycle/,
  /Brain liveness/i,
  /liveness probe/i,
  /probe FAILED/i,
  /engine may be wedged/i,
  /⚠️/,
  /All LLM providers failed/,
  /Cross-link batch LLM failed/,
  /Audit error for/,
  /aborted due to timeout/i,
];

function isOpsBlock(block) {
  return OPS_PATTERNS.some(re => re.test(block));
}

function dates() {
  const explicit = process.argv.slice(2).filter(a => /^\d{4}-\d{2}-\d{2}$/.test(a));
  if (explicit.length) return explicit;
  return [getLocalDateStamp(), getLocalDateStamp(new Date(Date.now() - 86400000))];
}

function prependOpsBlocks(date, opsBlocks) {
  if (!opsBlocks.length) return;
  if (!fs.existsSync(OPS_DIR)) { if (!DRY_RUN) fs.mkdirSync(OPS_DIR, { recursive: true }); }
  const opsFile = path.join(OPS_DIR, `${date}.md`);
  const header = `# Ops Log - ${date}\n\n`;
  let existingBody = '';
  if (fs.existsSync(opsFile)) {
    const cur = fs.readFileSync(opsFile, 'utf8');
    const { blocks } = splitDailyBlocks(cur);
    existingBody = blocks.join('\n\n');
  }
  // Migrated blocks are older than anything live-written post-restart, so append
  // them after any existing ops entries (newest-first order preserved).
  const merged = [existingBody, opsBlocks.join('\n\n')].filter(Boolean).join('\n\n');
  const out = header + merged + '\n';
  if (DRY_RUN) { console.log(`  [dry-run] would write ${opsFile} (${opsBlocks.length} block(s) moved)`); return; }
  fs.writeFileSync(opsFile, out, 'utf8');
  console.log(`  → wrote ${opsFile} (+${opsBlocks.length} block(s))`);
}

function migrate(date) {
  const dailyFile = path.join(DAILY_DIR, `${date}.md`);
  if (!fs.existsSync(dailyFile)) { console.log(`- ${date}: no daily log, skipping`); return; }

  const content = fs.readFileSync(dailyFile, 'utf8');
  const { header, blocks } = splitDailyBlocks(content);
  const keep = [], ops = [];
  for (const b of blocks) (isOpsBlock(b) ? ops : keep).push(b);

  const estTok = s => Math.ceil(s.length / 4);
  const beforeTok = estTok(content);
  const keptText = (header || `# Daily Log - ${date}\n\n`) + keep.join('\n\n') + (keep.length ? '\n' : '');
  const afterTok = estTok(keptText);

  console.log(`- ${date}: ${blocks.length} blocks → keep ${keep.length}, move ${ops.length} to ops  (~${beforeTok} → ~${afterTok} tok)`);

  if (ops.length === 0) return;

  if (!DRY_RUN) {
    fs.copyFileSync(dailyFile, dailyFile + '.pre-ops-split.bak');
    fs.writeFileSync(dailyFile, keptText, 'utf8');
    console.log(`  → rewrote ${dailyFile} (backup: ${path.basename(dailyFile)}.pre-ops-split.bak)`);
  }
  prependOpsBlocks(date, ops);
}

console.log(`Ops-split migration${DRY_RUN ? ' (DRY RUN)' : ''} — dates: ${dates().join(', ')}`);
for (const d of dates()) migrate(d);
console.log('Done.');
