#!/usr/bin/env node
/**
 * The scheduler's rules, tested where they can actually be observed.
 *
 * Everything asserted here is a rule whose failure is INVISIBLE in production
 * until it has already cost something: a job that runs twice, a job that runs
 * eleven times to catch up after a redeploy, a job that fails every night for a
 * month into a log nobody reads, a run that vanishes because the process
 * restarted mid-flight. None of those show up in a happy-path run, and waiting
 * for the real 09:00 firing tests exactly one of them.
 *
 * Runs against a throwaway SNH_DATA_DIR (same pattern as
 * test-cluster-audit-quiet.js) and never touches the live corpus. The model is
 * STUBBED — memory-manager.callLLM is replaced on the module object, which is
 * the same object db/scheduler.js resolves at call time. That is deliberate:
 * what is under test is the scheduler's decision-making, and a real model call
 * would make the failure cases (throws, empty output) unreachable.
 *
 * Usage: node scripts/test-scheduler.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-scheduler-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
const scheduler = require(path.join(ROOT, 'db/scheduler'));
const cronJobs = require(path.join(ROOT, 'db/cron-jobs'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- the stub -------------------------------------------------------------
// Replaces the model. `mode` decides what this run does; every test sets it
// before acting, so no case depends on a live brain.
let mode = 'ok';
let lastPrompt = null;
let callCount = 0;
memoryManager.callLLM = async (systemPrompt, userPrompt) => {
  callCount++;
  lastPrompt = { systemPrompt, userPrompt };
  if (mode === 'throw') throw new Error('Brain circuit open — skipping LLM call (engine wedged)');
  if (mode === 'empty') return { content: '   ', provider: 'stub', truncated: false, toolCalls: [], budget: {} };
  if (mode === 'slow') {
    await new Promise(r => setTimeout(r, 150));
    return { content: 'a slow but real answer', provider: 'stub', truncated: false, toolCalls: [], budget: {} };
  }
  return {
    content: 'Two facts merged and one event moved to the day log since yesterday.',
    provider: 'stub', truncated: false,
    toolCalls: [{ name: 'memory_corrections', args: {}, ok: true }],
    budget: { calls: 1, maxCalls: 12 }
  };
};

function makeJob({ schedule = '0 9 * * *', description = 'Test job', status = 'approved', enabled = 1 } = {}) {
  const id = randomUUID();
  db.prepare(`
    INSERT INTO cron_jobs (id, schedule, description, enabled, source, status, created_at, decided_at)
    VALUES (?, ?, ?, ?, 'kid-proposed', ?, ?, ?)
  `).run(id, schedule, description, enabled, status, new Date().toISOString(), new Date().toISOString());
  return id;
}
const job = (id) => db.prepare('SELECT * FROM cron_jobs WHERE id = ?').get(id);
const runs = (id) => db.prepare('SELECT * FROM job_runs WHERE job_id = ? ORDER BY datetime(started_at) ASC, rowid ASC').all(id);
const minutesAgo = (n) => new Date(Date.now() - n * 60_000).toISOString();

(async () => {
  console.log('\n1. Arming: approval is what makes a job real');
  const armed = makeJob({ description: 'Daily digest', status: 'proposed' });
  check('a proposed job is not armed', !job(armed).next_run_at);
  const appr = cronJobs.approve(armed);
  check('approving arms it', appr.ok && !!job(armed).next_run_at, JSON.stringify(appr.error || job(armed).next_run_at));
  const nextLocal = new Date(job(armed).next_run_at);
  check('…at the hour the expression says', nextLocal.getHours() === 9 && nextLocal.getMinutes() === 0,
    nextLocal.toLocaleString());
  check('…in the future', nextLocal.getTime() > Date.now(), nextLocal.toLocaleString());

  const disabledJob = makeJob({ status: 'proposed', enabled: 0, description: 'Proposed disabled' });
  cronJobs.approve(disabledJob);
  check('an approved-but-disabled job is recorded and NOT armed', !job(disabledJob).next_run_at);

  const unreadable = makeJob({ schedule: '0 0 30 2 *', description: 'The 30th of February' });
  scheduler.armJob(unreadable);
  check('a schedule nothing can satisfy is left disarmed rather than retried forever',
    !job(unreadable).next_run_at);

  console.log('\n2. A run that works');
  mode = 'ok';
  const daily = makeJob({ description: 'Summarize background memory maintenance' });
  scheduler.armJob(daily);
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), daily);
  const before = job(daily).next_run_at;
  let res = await scheduler.tick();
  check('a due job runs', res.ran === 1, JSON.stringify(res));
  const r1 = runs(daily);
  check('exactly one run row', r1.length === 1, `${r1.length} rows`);
  check('…with status ok and a duration', r1[0].status === 'ok' && r1[0].duration_ms >= 0, JSON.stringify(r1[0]));
  check('…recording the firing it was for, not just when it started',
    r1[0].scheduled_for === before, `${r1[0].scheduled_for} vs ${before}`);
  check('…and the output it produced', /merged/.test(r1[0].output_text || ''), r1[0].output_text);
  check('the job counts the run and clears its failure streak',
    job(daily).run_count === 1 && job(daily).consecutive_failures === 0 && job(daily).last_status === 'ok');
  check('it re-arms forward, not to the firing it just did',
    new Date(job(daily).next_run_at).getTime() > Date.now(), job(daily).next_run_at);
  check('…on the same schedule (tomorrow at 09:00)',
    new Date(job(daily).next_run_at).getHours() === 9, new Date(job(daily).next_run_at).toLocaleString());

  const bell = db.prepare("SELECT * FROM initiatives WHERE type = 'job-result'").all();
  check('the output reached the bell panel', bell.length === 1, `${bell.length} item(s)`);
  check('…marked as a job result with the job id attached',
    bell[0] && /Scheduled job result/.test(bell[0].content) && bell[0].content.includes(daily.slice(0, 8)),
    bell[0] && bell[0].content.slice(0, 120));
  check('…pointing at the RUN, so a later run is not read as a duplicate of it',
    bell[0] && bell[0].source_kind === 'scheduled-job' && bell[0].source_ref === r1[0].id);
  check('the run row points back at the bell item, so "it ran" and "she was told" are separable',
    r1[0].output_initiative_id === bell[0].id);
  check('the job description is what the model was asked to do',
    lastPrompt.userPrompt === 'Summarize background memory maintenance', lastPrompt.userPrompt);
  check('…and it was told to report only what the tools returned',
    /must come from a tool result/i.test(lastPrompt.systemPrompt));

  console.log('\n3. A second run is a second bell item, never folded into the first');
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), daily);
  await scheduler.tick();
  const bell2 = db.prepare("SELECT * FROM initiatives WHERE type = 'job-result'").all();
  check('two runs, two notifications', bell2.length === 2, `${bell2.length} item(s)`);
  check('…even though the text is identical',
    bell2[0].content.split('\n\n')[1] === bell2[1].content.split('\n\n')[1]);

  console.log('\n4. The prioritizer leaves records alone');
  // Age both job results past the stale window and cap the pool at nothing:
  // neither pass may touch them.
  db.prepare("UPDATE initiatives SET created_at = ? WHERE type = 'job-result'")
    .run(new Date(Date.now() - 90 * 86400_000).toISOString());
  const initiatives = require(path.join(ROOT, 'db/initiatives'));
  check('a job result is not offered to the pool machinery',
    initiatives.listPending({ limit: 100 }).every(i => i.type !== 'job-result'));
  check('…but the panel, which asks for records, sees it',
    initiatives.listPending({ limit: 100, includeRecords: true }).filter(i => i.type === 'job-result').length === 2);
  check('…and it is not chosen for an unprompted conversation',
    !initiatives.getTopPending(0) || initiatives.getTopPending(0).type !== 'job-result');

  console.log('\n5. Catch-up: one missed run, and only if it is still worth having');
  mode = 'ok';
  const missed = makeJob({ description: 'Missed while down' });
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(30), missed);
  res = await scheduler.tick();
  const mr = runs(missed);
  check('a run missed 30 min ago (inside the 120 min window) runs once',
    mr.length === 1 && mr[0].status === 'ok', JSON.stringify(mr.map(r => r.status)));
  check('…and is marked as the catch-up it was', mr[0].trigger === 'catchup', mr[0].trigger);
  check('…exactly once — the next tick finds nothing due',
    (await scheduler.tick()).due === 0);

  const stale = makeJob({ description: 'Missed long ago' });
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(60 * 9), stale);
  const callsBefore = callCount;
  await scheduler.tick();
  const sr = runs(stale);
  check('a run missed 9 hours ago is NOT run late', callCount === callsBefore, `${callCount - callsBefore} model call(s)`);
  check('…it is recorded as skipped, with the reason', sr.length === 1 && sr[0].status === 'skipped', JSON.stringify(sr[0]));
  check('…naming how late it was and the window it missed',
    /missed by 5\d\d min, past the 120 min catch-up window/.test(sr[0].error || ''), sr[0].error);
  check('…and the job is re-armed forward rather than left due',
    new Date(job(stale).next_run_at).getTime() > Date.now(), job(stale).next_run_at);
  check('…with no backfill: one skipped row, no run rows', runs(stale).length === 1);

  console.log('\n6. Re-entrancy: a job never starts on top of itself');
  const busy = makeJob({ description: 'Long runner' });
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), busy);
  // An open row is what a run in flight looks like from disk.
  db.prepare(`INSERT INTO job_runs (id, job_id, scheduled_for, started_at, status, trigger)
              VALUES (?, ?, ?, ?, 'running', 'schedule')`)
    .run(randomUUID(), busy, minutesAgo(61), minutesAgo(60));
  const callsBefore2 = callCount;
  await scheduler.tick();
  check('a job whose previous run is still open is not started again', callCount === callsBefore2);
  const deferrals = runs(busy).filter(r => r.status === 'deferred');
  check('…the deferral is recorded rather than silently dropped', deferrals.length === 1, `${deferrals.length}`);
  check('…naming why', /its own previous run has not finished/.test(deferrals[0].error || ''), deferrals[0].error);
  await scheduler.tick();
  await scheduler.tick();
  check('…and three ticks of waiting leave ONE line, not three',
    runs(busy).filter(r => r.status === 'deferred').length === 1,
    `${runs(busy).filter(r => r.status === 'deferred').length}`);

  console.log('\n7. An interrupted run is closed out, not left blocking forever');
  const swept = scheduler.sweepInterruptedRuns();
  check('the open row is swept', swept === 1, `${swept}`);
  const busyRuns = runs(busy);
  check('…as skipped, not as a failure (nothing about the job went wrong)',
    busyRuns.find(r => r.trigger === 'schedule' && r.status === 'skipped'), JSON.stringify(busyRuns.map(r => r.status)));
  check('…saying a restart interrupted it',
    /interrupted by a restart/.test((busyRuns.find(r => r.status === 'skipped') || {}).error || ''));
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), busy);
  await scheduler.tick();
  check('…and the job runs again on the next tick', runs(busy).some(r => r.status === 'ok'));

  console.log('\n8. Serial: two due jobs never run at once');
  mode = 'slow';
  const a = makeJob({ description: 'Job A' });
  const b = makeJob({ description: 'Job B' });
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), a);
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), b);
  // Two ticks overlapping is the real case: the timer does not wait for the
  // previous tick to finish.
  const [t1, t2] = await Promise.all([scheduler.tick(), scheduler.tick()]);
  const ranTotal = (t1.ran || 0) + (t2.ran || 0);
  const deferredTotal = (t1.deferred || 0) + (t2.deferred || 0);
  check('the overlapping tick defers instead of running in parallel',
    deferredTotal >= 1, JSON.stringify([t1, t2]));
  check('no job ran twice',
    runs(a).filter(r => r.status === 'ok').length <= 1 && runs(b).filter(r => r.status === 'ok').length <= 1,
    `A=${runs(a).filter(r => r.status === 'ok').length} B=${runs(b).filter(r => r.status === 'ok').length}`);
  check('and the work still gets done on a later tick', ranTotal >= 1, `${ranTotal}`);

  console.log('\n9. Failure: counted, named, and eventually stopped');
  mode = 'throw';
  const failing = makeJob({ description: 'Always fails' });
  for (let i = 1; i <= 2; i++) {
    db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), failing);
    await scheduler.tick();
    check(`failure ${i} is counted (${i}/3)`, job(failing).consecutive_failures === i, `${job(failing).consecutive_failures}`);
    check(`…and the job stays armed while it has attempts left`, !!job(failing).next_run_at && !!job(failing).enabled);
  }
  const failRun = runs(failing)[0];
  check('the run row carries the actual error text',
    /circuit open/.test(failRun.error || ''), failRun.error);

  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), failing);
  await scheduler.tick();
  check('the third consecutive failure disables the job', !job(failing).enabled);
  check('…disarms it, so it stops claiming a next run', !job(failing).next_run_at);
  check('…and says why, naming the error',
    /3 consecutive failures/.test(job(failing).disabled_reason || '') && /circuit open/.test(job(failing).disabled_reason || ''),
    job(failing).disabled_reason);
  const alert = db.prepare("SELECT * FROM initiatives WHERE type = 'alert' AND source_kind = 'scheduled-job-disabled'").get();
  check('…and raises it with Ellie rather than only in a log',
    !!alert && /disabled itself/.test(alert.content) && /circuit open/.test(alert.content),
    alert && alert.content.slice(0, 140));
  const callsBefore3 = callCount;
  await scheduler.tick();
  check('a disabled job is not picked up again', callCount === callsBefore3);

  console.log('\n10. Output that is not output');
  mode = 'empty';
  const silent = makeJob({ description: 'Says nothing' });
  db.prepare('UPDATE cron_jobs SET next_run_at = ? WHERE id = ?').run(minutesAgo(1), silent);
  await scheduler.tick();
  const silentRun = runs(silent)[0];
  check('a run that produced no text is a failure, not a quiet success',
    silentRun.status === 'failed' && /returned no text/.test(silentRun.error || ''), JSON.stringify(silentRun));
  check('…and nothing was posted to the panel for it',
    !db.prepare("SELECT id FROM initiatives WHERE source_ref = ?").get(silentRun.id));

  console.log('\n11. A successful run clears the streak');
  mode = 'ok';
  const recovering = makeJob({ description: 'Recovers' });
  db.prepare('UPDATE cron_jobs SET consecutive_failures = 2, next_run_at = ? WHERE id = ?').run(minutesAgo(1), recovering);
  await scheduler.tick();
  check('two failures then a success resets the counter to zero',
    job(recovering).consecutive_failures === 0 && job(recovering).last_status === 'ok');

  console.log('\n12. Forcing a run by hand does not consume the schedule');
  const forced = makeJob({ description: 'Forced' });
  scheduler.armJob(forced);
  const armedFor = job(forced).next_run_at;
  const fres = await scheduler.runNow(forced.slice(0, 8));
  check('runNow accepts a short id and runs', fres.status === 'ok', JSON.stringify(fres.error || fres.status));
  check('…marked manual, so it is never read as evidence the timer works',
    runs(forced)[0].trigger === 'manual', runs(forced)[0].trigger);
  // Re-arming computes from now, so a forced run lands back on the firing that
  // was already pending: forcing one at 08:33 does not cancel today's 09:00.
  check('…and the pending firing is still pending — forcing does not consume it',
    job(forced).next_run_at === armedFor && new Date(job(forced).next_run_at).getTime() > Date.now(),
    `${job(forced).next_run_at} vs ${armedFor}`);
  check('runNow on an unknown id says so rather than doing nothing quietly',
    (await scheduler.runNow('nope')).error);
  const proposed = makeJob({ status: 'proposed', description: 'Not approved' });
  check('runNow refuses a job Ellie has not approved',
    /not approved/.test((await scheduler.runNow(proposed)).error || ''));

  console.log('\n13. Reverting disarms');
  const toRevert = makeJob({ description: 'To revert' });
  scheduler.armJob(toRevert);
  cronJobs.revertAllKidCreated({ note: 'test' });
  check('a reverted job stops claiming a next run', !job(toRevert).next_run_at);
  check('…and is no longer due', !scheduler.dueJobs(new Date(Date.now() + 86400_000)).some(j => j.id === toRevert));

  const bar = '='.repeat(74);
  console.log(`\n${bar}`);
  console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
  console.log(`${bar}\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('[test-scheduler] error:', err);
  process.exit(1);
});
