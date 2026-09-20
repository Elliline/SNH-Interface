#!/usr/bin/env node
/**
 * Fire the self-coherence audit's INAUGURAL run — the hardcoded loop-awareness
 * claim audit (see db/self-audit.js). This is SNH's first self-audit, the
 * feature it asked for itself (first accepted initiative 2026-07-05; re-chosen
 * 2026-07-23 to find out "if I'm actually growing, or just getting better at
 * describing a growth that isn't happening").
 *
 * Running it applies the claim_type migration, writes ONE dissonance self-fact,
 * raises ONE 'audit' initiative for Ellie's approval, and stamps
 * data/memory/audit-state.json (so the scheduled heartbeat won't re-run the
 * inaugural — it moves on to sampled runs). Only fires the inaugural if no audit
 * has run before; otherwise it runs a normal sampled audit.
 *
 * Usage: node scripts/run-inaugural-audit.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
const selfAudit = require(path.join(ROOT, 'db/self-audit'));

(async () => {
  db.initDatabase();

  const before = selfAudit.readAuditState();
  console.log(`[run-inaugural-audit] prior state: ${JSON.stringify(before)}`);
  console.log(`[run-inaugural-audit] ${before.lastAuditAt ? 'audit has run before → sampled run' : 'no prior audit → INAUGURAL run'}`);

  const result = await selfAudit.runSelfCoherenceAudit();

  console.log('\n=== RESULT ===');
  console.log(JSON.stringify(result, null, 2));
  console.log('\n=== new state ===');
  console.log(JSON.stringify(selfAudit.readAuditState(), null, 2));
  process.exit(0);
})().catch(err => {
  console.error('[run-inaugural-audit] error:', err);
  process.exit(1);
});
