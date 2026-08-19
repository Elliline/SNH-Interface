#!/usr/bin/env node
/**
 * Lanes, and the one property that justifies them: NOTHING STARVES.
 *
 * The pool was one bucket with one cap. Eight agent jobs would take every slot
 * and the heartbeat, the corrector and fact extraction queued behind them — and
 * that is invisible until the day the memory quietly stops being tidied while a
 * swarm runs. The only protection chat had was blunter still: everything
 * throttled to 1 the moment a reply started being written, so background work
 * stopped dead every time the user typed.
 *
 * What is asserted here is behavioural, not structural: a full lane must not be
 * able to hold a slot that another lane's work is waiting for. Tasks are
 * promises this file releases by hand, because the only way to observe a
 * scheduling rule is to hold work open and look at what got started.
 *
 * Usage: node scripts/test-agent-pool-lanes.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-pool-lanes-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const config = require(path.join(ROOT, 'db/config'));

let poolCfg = { lanes: { agentJobs: 3, scheduled: 1, background: 2 }, maxTotalBackground: 5, backgroundDuringChat: 1 };
const realGetConfig = config.getConfig;
config.getConfig = () => { const c = realGetConfig(); c.agentPool = poolCfg; return c; };

const agentPool = require(path.join(ROOT, 'db/agent-pool'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}
const tick = () => new Promise(r => setImmediate(r));
async function settle() { for (let i = 0; i < 8; i++) await tick(); }

/** A task that starts, records that it started, and blocks until released. */
function held(startedList, name) {
  let release;
  const gate = new Promise(r => { release = r; });
  const fn = async () => { startedList.push(name); await gate; return name; };
  return { fn, release: () => release() };
}

(async () => {
  console.log(`\nAgent-pool lane tests (throwaway data dir: ${TMP})\n`);

  // =========================================================================
  console.log('── A lane runs up to its own cap and no further ──');
  let started = [];
  const jobs = Array.from({ length: 6 }, (_, i) => held(started, `job${i}`));
  jobs.forEach(h => agentPool.schedule(h.fn, 'j', 'agentJobs'));
  await settle();
  check('3 of 6 agent jobs started — the lane cap held',
    started.length === 3, `${started.length}: ${started.join(',')}`);
  check('and they are the first three, in order — the lane is a FIFO',
    started.join(',') === 'job0,job1,job2', started.join(','));

  // =========================================================================
  console.log('\n── THE POINT: a full agent-job lane does not starve the others ──');
  // The agentJobs lane is saturated and has three more waiting. Background and
  // scheduled work arriving now must still start, because they are not queued
  // behind those jobs — they were never in the same queue.
  const bg = held(started, 'background0');
  const sched = held(started, 'scheduled0');
  agentPool.schedule(bg.fn, 'b', 'background');
  agentPool.schedule(sched.fn, 's', 'scheduled');
  await settle();
  check('background work started while 3 agent jobs still queue',
    started.includes('background0'), started.join(','));
  check('scheduled work started too',
    started.includes('scheduled0'), started.join(','));
  check('and the queued agent jobs did NOT jump their own cap to do it',
    started.filter(n => n.startsWith('job')).length === 3, started.join(','));

  // =========================================================================
  console.log('\n── The total cap bounds the lanes added together ──');
  // Caps are 3 + 1 + 2 = 6, but maxTotalBackground is 5. Five are running.
  const bg2 = held(started, 'background1');
  agentPool.schedule(bg2.fn, 'b', 'background');
  await settle();
  check('a 6th task waits even though its own lane has room',
    !started.includes('background1'),
    `three caps summing past the total would have started it: ${started.join(',')}`);
  check('exactly maxTotalBackground are running', started.length === 5, String(started.length));

  // Releasing one frees exactly one slot.
  jobs[0].release();
  await settle();
  check('freeing a slot lets the waiting task in', started.includes('background1'), started.join(','));

  // =========================================================================
  console.log('\n── Chat reserves headroom; it never queues ──');
  jobs.slice(1).forEach(h => h.release());
  bg.release(); sched.release(); bg2.release();
  await settle();

  agentPool.beginChat();
  started = [];
  const during = Array.from({ length: 4 }, (_, i) => held(started, `during${i}`));
  during.forEach(h => agentPool.schedule(h.fn, 'd', 'background'));
  await settle();
  check('while a reply is being written, background is held to its reserved headroom',
    started.length === 1, `${started.length} started: ${started.join(',')}`);
  check('which is backgroundDuringChat, not zero — background inches on rather than stopping dead',
    agentPool.stats().totalCap === 1, JSON.stringify(agentPool.stats().totalCap));

  agentPool.endChat();
  await settle();
  check('when the reply is done the lanes come back to full width',
    started.length > 1, `${started.length} started: ${started.join(',')}`);
  during.forEach(h => h.release());
  await settle();

  // =========================================================================
  console.log('\n── Back-compat: everything that was already written still works ──');
  started = [];
  const plain = held(started, 'unlabelled');
  agentPool.schedule(plain.fn, 'legacy');            // no lane argument at all
  await settle();
  check('a schedule() with no lane runs, in the background lane',
    started.includes('unlabelled') && agentPool.stats().lanes.background.active === 1,
    JSON.stringify(agentPool.stats().lanes));
  plain.release(); await settle();

  const results = await agentPool.runBatch([
    async () => 'a', async () => 'b'
  ], 'batch', 'background');
  check('runBatch still isolates errors and returns settled shapes',
    results.length === 2 && results[0].status === 'fulfilled' && results[0].value === 'a',
    JSON.stringify(results));

  const withFailure = await agentPool.runBatch([
    async () => { throw new Error('boom'); }, async () => 'ok'
  ], 'batch2', 'agentJobs');
  check('one failed task never rejects the batch',
    withFailure[0].status === 'rejected' && withFailure[1].value === 'ok',
    JSON.stringify(withFailure.map(r => r.status)));

  // =========================================================================
  console.log('\n── An unknown lane name lands somewhere real, not nowhere ──');
  started = [];
  const odd = held(started, 'odd');
  agentPool.schedule(odd.fn, 'x', 'no-such-lane');
  await settle();
  check('an unrecognised lane falls back to background rather than vanishing',
    started.includes('odd'), started.join(','));
  odd.release(); await settle();

  // =========================================================================
  console.log('\n── Round-robin: a busy lane cannot take every FREED slot ──');
  // In-flight work is never preempted — an LLM call cannot be cleanly cancelled,
  // and that was true before lanes. So the fairness question is not "who is
  // running now", it is "who gets the next slot that opens". A fixed drain order
  // would hand every freed slot back to the lane with six things waiting, and
  // the one background task would sit there forever behind them.
  poolCfg = { lanes: { agentJobs: 8, scheduled: 8, background: 8 }, maxTotalBackground: 2, backgroundDuringChat: 1 };
  started = [];
  const many = Array.from({ length: 6 }, (_, i) => held(started, `J${i}`));
  many.forEach(h => agentPool.schedule(h.fn, 'j', 'agentJobs'));
  const oneBg = held(started, 'B0');
  agentPool.schedule(oneBg.fn, 'b', 'background');
  await settle();
  check('the two slots are taken by the work that arrived first',
    started.length === 2 && started.join(',') === 'J0,J1', started.join(','));

  many[0].release();
  await settle();
  check('the freed slot goes to the WAITING LANE, not to the queue of six',
    started.includes('B0'),
    `${started.join(',')} — a fixed drain order would have started J2 and left B0 behind five more`);
  check('and it really was the background task that got it, not just any task',
    started[2] === 'B0', started.join(','));

  many.slice(1).forEach(h => h.release()); oneBg.release();
  await settle();

  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
