#!/usr/bin/env node
/**
 * THE BELL CARRIES NOTIFICATIONS AND NOTHING ELSE.
 *
 * Four kinds stay — alert, proposal, followup, backlog — and four leave:
 * audits to Corrections, reflection insights to Reflections, housekeeping to
 * the activity log, job results to the robot queue. This asserts both halves,
 * because "it no longer rings" is only half a fix: a thing pulled out of the
 * bell has to land somewhere, and a test that only checked the bell would pass
 * just as well if the item had been dropped on the floor.
 *
 * Throwaway store, no engine, no chat turn.
 *
 * Usage: node scripts/test-bell-routing.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-bell-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => { if (!INHERITED) { try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {} } });

const db = require(path.join(ROOT, 'db/database'));

let passed = 0, failed = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { passed++; console.log(`  PASS  ${name}`); }
  else { failed++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
};
const section = t => console.log(`\n=== ${t} ===`);

(async () => {
  db.initDatabase();
  const initiatives = require(path.join(ROOT, 'db/initiatives'));
  const engine = require(path.join(ROOT, 'db/initiative-engine'));
  const sql = db.getSqliteDb();

  const bellTypes = () => initiatives.listPendingForBell().map(i => i.type);
  const allPending = () => sql.prepare("SELECT type FROM initiatives WHERE status='pending'").all().map(r => r.type);

  // -------------------------------------------------------- kept
  section('1. The four kept categories reach the bell');

  await initiatives.addInitiative({ type: 'alert', content: 'The nightly backup did not run.', priority: 8, dedupe: false });
  await initiatives.addInitiative({ type: 'proposal', content: 'Shall I create a cron job for the weekly digest?', priority: 6, dedupe: false });
  await initiatives.addInitiative({ type: 'followup', content: 'You asked me what I would want and I never heard back.', priority: 6, dedupe: false });

  ok('an alert rings', bellTypes().includes('alert'));
  ok('a proposal rings', bellTypes().includes('proposal'));
  ok('a followup rings', bellTypes().includes('followup'));

  // Backlog fires on a THRESHOLD crossing, not on arrival.
  const threshold = require(path.join(ROOT, 'db/config')).getConfig().initiative.backlogThreshold;
  const under = await engine.raiseMessageBacklog(threshold - 1);
  ok('a backlog UNDER the threshold does not ring', under === null && !bellTypes().includes('backlog'),
     JSON.stringify(bellTypes()));
  const over = await engine.raiseMessageBacklog(threshold + 1);
  ok('a backlog OVER the threshold rings', !!over && bellTypes().includes('backlog'), JSON.stringify(bellTypes()));
  ok('the backlog seam is inert until something calls it — nothing polls',
     !/setInterval|setTimeout/.test(String(engine.raiseMessageBacklog)));

  // -------------------------------------------------------- moved
  section('2. The four moved categories no longer ring');

  ok('audit is not a bell type', !initiatives.isBellType('audit'));
  ok('reflection-insight is not a bell type', !initiatives.isBellType('reflection-insight'));
  ok('observation is not a bell type', !initiatives.isBellType('observation'));
  ok('job-result is not a bell type', !initiatives.isBellType('job-result'));

  // Even a row of a moved type sitting pending — history, or something a
  // migration has not reached — stays out of the bell.
  await initiatives.addInitiative({ type: 'audit', content: 'A stray audit row.', priority: 9, dedupe: false });
  await initiatives.addInitiative({ type: 'job-result', content: 'A stray job result.', priority: 9, dedupe: false });
  ok('a pending audit does not ring', !bellTypes().includes('audit'), JSON.stringify(bellTypes()));
  ok('a pending job-result does not ring', !bellTypes().includes('job-result'), JSON.stringify(bellTypes()));
  ok('…but both still EXIST — nothing was deleted',
     allPending().includes('audit') && allPending().includes('job-result'), JSON.stringify(allPending()));

  // -------------------------------------------------------- destinations
  section('3. What left the bell landed somewhere');

  const reflectionsFile = path.join(TMP, 'memory', 'reflections.jsonl');
  const beforeRefl = fs.existsSync(reflectionsFile) ? fs.readFileSync(reflectionsFile, 'utf8') : '';
  const rid = await engine.noticeReflectionInsight('I notice I lead with the verifiable thing first.');
  const afterRefl = fs.existsSync(reflectionsFile) ? fs.readFileSync(reflectionsFile, 'utf8') : '';
  ok('a reflection insight does NOT become an initiative', rid === null);
  ok('…it is written to the reflections record',
     afterRefl.length > beforeRefl.length && /lead with the verifiable/.test(afterRefl));
  ok('…and no reflection-insight row was created', !allPending().includes('reflection-insight'),
     JSON.stringify(allPending()));

  // Housekeeping goes to the ops log.
  const opsDir = path.join(TMP, 'memory', 'ops');
  const readOps = () => {
    try { return fs.readdirSync(opsDir).flatMap(f => fs.readFileSync(path.join(opsDir, f), 'utf8').split('\n')); }
    catch { return []; }
  };
  const opsBefore = readOps().length;
  await engine.noticeFromAudit([{ clusterName: 'Athenas Freedom', clusterId: 'c1', coherent: false,
    splits: [{ newClusterName: 'Freedom of Speech' }, { newClusterName: 'Self Protection' }] }]);
  const opsAfter = readOps();
  ok('cluster housekeeping does NOT become an initiative', !allPending().includes('observation'),
     JSON.stringify(allPending()));
  ok('…it is written to the activity log',
     opsAfter.length > opsBefore && opsAfter.some(l => /Cluster reorganisation/.test(l)),
     JSON.stringify(opsAfter.slice(-2)));

  // -------------------------------------------------------- no cap
  section('4. No cap — a notification queue may not drop notifications');

  for (let i = 0; i < 25; i++) {
    await initiatives.addInitiative({ type: 'alert', content: `Disk warning number ${i}.`, priority: 4, dedupe: false });
  }
  ok('all 25 alerts are in the bell, plus the earlier ones',
     initiatives.listPendingForBell().filter(i => i.type === 'alert').length === 26,
     String(initiatives.listPendingForBell().filter(i => i.type === 'alert').length));
  ok('countPendingForBell agrees with the list',
     initiatives.countPendingForBell() === initiatives.listPendingForBell().length);

  // The cap lived in prioritize(); assert it cannot come back by accident.
  const engineSrc = fs.readFileSync(path.join(ROOT, 'db/initiative-engine.js'), 'utf8');
  const capLines = engineSrc.split('\n').filter(l => /maxPending/.test(l) && !/^\s*(\/\/|\*)/.test(l.trim()));
  ok('prioritize() no longer reads maxPending in code', capLines.length === 0, JSON.stringify(capLines));

  // -------------------------------------------------------- dismiss
  section('5. Dismiss is for things seen; an approval is decided');

  const alertRow = initiatives.listPendingForBell().find(i => i.type === 'alert');
  const proposalRow = initiatives.listPendingForBell().find(i => i.type === 'proposal');
  ok('an alert can be dismissed', initiatives.dismiss(alertRow.id) === true);
  ok('a proposal CANNOT be dismissed', initiatives.dismiss(proposalRow.id) === false);
  ok('…and it is still pending afterwards',
     sql.prepare('SELECT status FROM initiatives WHERE id=?').get(proposalRow.id).status === 'pending');

  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error('\nCRASH:', e.stack || e.message); process.exit(1); });
