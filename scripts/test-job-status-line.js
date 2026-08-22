#!/usr/bin/env node
/**
 * test-job-status-line — knowing a job is running, without going hunting.
 *
 * She found out a job had finished by opening a panel afterwards. A
 * dispatched job now reports where it is at the end of any turn taken
 * while it runs, in the conversation she is already reading.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-job-status-line.js
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  process.exit(1);
}

const database = require('../db/database');
database.initDatabase();
const codingJobs = require('../db/coding-jobs');
const db = database.getSqliteDb();

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'prog-'));
let n = 0;

function writeProgress(fields) {
  const p = path.join(dir, `progress-${++n}.json`);
  const now = Date.now() / 1000;
  fs.writeFileSync(p, JSON.stringify({
    schema: 1, project: 'todoapp', iteration: 1, max_iterations: 25,
    tool_calls: 0, last_action: null, started_at: now, updated_at: now,
    finished: false, ...fields,
  }));
  return p;
}

function liveJob(progressPath, status = 'running', project = 'todoapp') {
  const id = `j${++n}`;
  db.prepare("INSERT INTO agent_jobs (id,title,task,status,source,created_at) VALUES (?,?,?,?,'squatch-code',datetime('now'))")
    .run(id, `squatch-code: ${project}`, 'brief', status);
  db.prepare("INSERT INTO coding_jobs (id,project,brief,status,agent_job_id,progress_path,created_at) VALUES (?,?,?,'dispatched',?,?,datetime('now'))")
    .run(`c${id}`, project, 'brief', id, progressPath);
  return id;
}

function clean() {
  db.prepare('DELETE FROM coding_jobs').run();
  db.prepare('DELETE FROM agent_jobs').run();
}

console.log('\nthe line itself\n');

test('nothing running renders nothing', () => {
  clean();
  assert.strictEqual(codingJobs.statusBlock(), null);
});

test('a running job names the project and where it is', () => {
  clean();
  liveJob(writeProgress({ iteration: 4, last_action: 'read_file src/auth.py' }));
  const block = codingJobs.statusBlock();
  assert.match(block, /todoapp/);
  assert.match(block, /step 4\/25/);
  assert.match(block, /read_file src\/auth\.py/);
});

test('before the first tool call it says thinking, not nothing', () => {
  clean();
  liveJob(writeProgress({ iteration: 1, last_action: null }));
  assert.match(codingJobs.statusBlock(), /thinking/);
});

test('elapsed comes from the clock, not the file', () => {
  // The file is written on events; a stored elapsed freezes between
  // them and read "0s" for seventy seconds of real waiting.
  clean();
  const p = writeProgress({ started_at: Date.now() / 1000 - 95, elapsed_seconds: 0 });
  liveJob(p);
  assert.match(codingJobs.statusBlock(), /1m35s/);
});

test('a quiet job says so in the same line', () => {
  clean();
  const now = Date.now() / 1000;
  liveJob(writeProgress({ started_at: now - 400, updated_at: now - 300,
                          iteration: 4, last_action: 'run_command pytest -q' }));
  const block = codingJobs.statusBlock();
  assert.match(block, /no activity for 5m/);
  assert.match(block, /run_command pytest -q/,
    'the last known action must still be shown alongside the silence');
});

test('a job that is still queued says so', () => {
  clean();
  liveJob(writeProgress({}), 'queued');
  assert.match(codingJobs.statusBlock(), /queued, not started yet/);
});

test('a job with no progress file yet says starting', () => {
  clean();
  liveJob(path.join(dir, 'does-not-exist.json'));
  assert.match(codingJobs.statusBlock(), /starting/);
});

test('a finished job drops out of the line entirely', () => {
  clean();
  const id = liveJob(writeProgress({ iteration: 3 }));
  db.prepare("UPDATE agent_jobs SET status='ok' WHERE id=?").run(id);
  assert.strictEqual(codingJobs.statusBlock(), null);
});

test('two jobs both appear', () => {
  clean();
  liveJob(writeProgress({ iteration: 2 }), 'running', 'todoapp');
  liveJob(writeProgress({ iteration: 7 }), 'running', 'csvtool');
  const block = codingJobs.statusBlock();
  assert.match(block, /todoapp/);
  assert.match(block, /csvtool/);
});

test('a corrupt progress file does not break the line', () => {
  clean();
  const p = path.join(dir, 'corrupt.json');
  fs.writeFileSync(p, '{not json');
  liveJob(p);
  assert.match(codingJobs.statusBlock(), /starting/);
});

test('it never claims to know how far along it is', () => {
  clean();
  liveJob(writeProgress({ iteration: 4, last_action: 'read_file x.py' }));
  const block = codingJobs.statusBlock();
  assert.ok(!/%|percent|complete|remaining|eta/i.test(block),
    'the line implies progress it cannot know');
});

console.log('\nwired into the reply\n');

test('server.js appends it to the turn', () => {
  const src = fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8');
  assert.ok(/statusBlock\(\)/.test(src), 'the reply never renders a status line');
  assert.ok(/if \(status\) append\(status\)/.test(src),
    'a null status must append nothing at all');
});

test('a failure in the status line cannot break the reply', () => {
  const src = fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8');
  const i = src.indexOf('statusBlock()');
  assert.ok(/catch \(statusErr\)/.test(src.slice(i, i + 400)),
    'the status line is not wrapped — a convenience could take out a turn');
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
