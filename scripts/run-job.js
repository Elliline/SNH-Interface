#!/usr/bin/env node
/**
 * Run a scheduled job by hand, or look at what its runs have done.
 *
 * The deliberate path for a job: a person naming one, at a keyboard. Used to
 * prove a newly-approved job actually works without waiting for its hour, and
 * to read back what it produced afterwards.
 *
 * A forced run is recorded with trigger='manual', so nothing here can ever be
 * mistaken for evidence that the timer fired on its own — that is what the
 * 'schedule' rows are for.
 *
 * Usage:
 *   node scripts/run-job.js                 # list jobs, their next run and last run
 *   node scripts/run-job.js <id> --run-now  # run one now, print its output
 *   node scripts/run-job.js <id> --runs     # the run log for one job
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

const args = process.argv.slice(2);
const flags = new Set(args.filter(a => a.startsWith('--')));
const target = args.find(a => !a.startsWith('--')) || null;

function when(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return isNaN(d.getTime()) ? '—' : d.toLocaleString();
}

(async () => {
  db.initDatabase();
  await db.initVectorStore();

  const scheduler = require(path.join(ROOT, 'db/scheduler'));
  const sql = db.getSqliteDb();

  if (!target) {
    const state = scheduler.schedulerState();
    console.log(`\nScheduler: ${state.enabled ? 'ENABLED' : 'DISABLED'}, tick every ${state.tickSeconds}s, ` +
                `catch-up window ${state.catchupGraceMinutes} min, ${state.armedJobs} job(s) armed.`);
    console.log(`Tools a job may use: ${state.tools.join(', ')}\n`);
    for (const job of sql.prepare('SELECT * FROM cron_jobs ORDER BY datetime(created_at) DESC').all()) {
      const rt = scheduler.runtimeState(job.id);
      console.log(`${job.id.slice(0, 8)}  [${job.status}${job.enabled ? '' : ', disabled'}]  "${job.description}"`);
      console.log(`          schedule ${job.schedule}   next ${when(rt.nextRunAt)}   last ${when(rt.lastRunAt)}${rt.lastStatus ? ` (${rt.lastStatus})` : ''}   runs ${rt.timesRun}`);
      if (rt.disabledReason) console.log(`          disabled: ${rt.disabledReason}`);
    }
    console.log('');
    process.exit(0);
  }

  const job = sql.prepare('SELECT * FROM cron_jobs WHERE id = ? OR id LIKE ?').get(target, `${target}%`);
  if (!job) { console.error(`No job matching "${target}".`); process.exit(2); }

  if (flags.has('--runs')) {
    const runs = scheduler.listRuns({ jobId: job.id, limit: 50 });
    console.log(`\n"${job.description}" (${job.id.slice(0, 8)}) — ${runs.length} run(s), newest first:\n`);
    for (const r of runs) {
      console.log(`${when(r.started_at)}  ${r.status.toUpperCase().padEnd(8)} ${String(r.trigger || '').padEnd(8)} ` +
                  `${r.duration_ms != null ? `${(r.duration_ms / 1000).toFixed(1)}s` : '—'}  ${r.tool_calls || 0} tool call(s)`);
      if (r.scheduled_for) console.log(`    for the ${when(r.scheduled_for)} firing`);
      if (r.error) console.log(`    ${r.error}`);
      if (r.output_text) console.log(`    → ${r.output_text.replace(/\n/g, '\n      ')}`);
    }
    console.log('');
    process.exit(0);
  }

  if (!flags.has('--run-now')) {
    console.error('Nothing to do — pass --run-now to run it, or --runs to see its history.');
    process.exit(2);
  }

  console.log(`\nRunning "${job.description}" (${job.id.slice(0, 8)}) now…\n`);
  const res = await scheduler.runNow(job.id);
  if (res.error && !res.status) { console.error(`Failed: ${res.error}`); process.exit(1); }

  console.log(`Status: ${res.status}${res.error ? ` — ${res.error}` : ''}`);
  console.log(`Took ${(res.durationMs / 1000).toFixed(1)}s, ${res.toolCalls} tool call(s)` +
              `${res.budget ? ` (budget: ${res.budget.calls}/${res.budget.maxCalls} calls${res.budget.exhausted ? `, ${res.budget.exhausted}` : ''})` : ''}`);
  if (res.output) console.log(`\n--- what it wrote ---\n${res.output}\n---------------------`);
  if (res.initiativeId) console.log(`\nDelivered to the bell panel as initiative ${res.initiativeId.slice(0, 8)}.`);

  const rt = scheduler.runtimeState(job.id);
  console.log(`\nNext run: ${when(rt.nextRunAt)}   (run ${rt.timesRun} total, ${rt.consecutiveFailures} consecutive failure(s))\n`);
  process.exit(res.status === 'ok' ? 0 : 1);
})().catch(err => {
  console.error('[run-job] error:', err);
  process.exit(1);
});
