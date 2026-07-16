#!/usr/bin/env node
/**
 * One-time backlog sweep: run every pending question through the answer-aware
 * gate (gapAlreadyAnswered) and retire the ones memory already answers.
 *
 * The mint-time gate (added 7/8) only screens NEW questions — everything queued
 * before it landed was grandfathered in ("What is ISH?" sat pending 8 days with
 * a defining fact on record the whole time). This runs the same gate over the
 * existing backlog. The heartbeat now repeats this every cycle (Task B2), so
 * this script is for the initial clear-out and ad-hoc re-runs.
 *
 * Retired questions flip to 'answered' via the normal markAnswered path —
 * never deleted. Requires the brain (LLM) and embedding backends to be up.
 *
 * Usage: node scripts/sweep-pending-questions.js [--apply]
 *   (dry-run by default; pass --apply to actually retire)
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

(async () => {
  const apply = process.argv.includes('--apply');
  db.initDatabase();
  await db.initVectorStore();
  const fx = require(path.join(ROOT, 'db/fact-extractor'));

  const { swept, retired } = await fx.sweepPendingQuestions({ dryRun: !apply });

  console.log(`\nChecked ${swept} pending question(s); ${retired.length} ${apply ? 'retired' : 'would be retired'}:`);
  for (const r of retired) {
    console.log(`- [${String(r.id).slice(0, 8)}] "${r.question}"`);
    console.log(`    ← answered by: "${r.evidence}"`);
  }
  if (!apply) console.log('\nDry run — pass --apply to retire these.');
  process.exit(0);
})();
