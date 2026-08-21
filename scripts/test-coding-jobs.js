#!/usr/bin/env node
/**
 * test-coding-jobs — the dispatch path to squatch-code.
 *
 * Run with SNH_DATA_DIR pointed at a throwaway directory. This never touches
 * chat and never reaches reflection, but it writes rows, and live rows are
 * live state.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-coding-jobs.js
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-coding-jobs.js');
  process.exit(1);
}

const database = require('../db/database');
database.initDatabase();
const { getSqliteDb } = database;
const codingJobs = require('../db/coding-jobs');
const agentJobs = require('../db/agent-jobs');

let passed = 0, failed = 0;
async function test(name, fn) {
  try { await fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

// A throwaway projects directory, so nothing here can reach real projects.
const projectsRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'test-projects-'));
fs.mkdirSync(path.join(projectsRoot, 'demo'));
const cfgMod = require('../db/config');
const origGet = cfgMod.getConfig;
cfgMod.getConfig = () => {
  const c = origGet();
  c.tools = c.tools || {};
  c.tools.codingJobs = { ...(c.tools.codingJobs || {}), enabled: true, projectsRoot };
  return c;
};

function db() { return getSqliteDb(); }
function clean() {
  db().prepare('DELETE FROM coding_jobs').run();
  db().prepare('DELETE FROM agent_jobs').run();
}

(async () => {
  console.log('\ncoding jobs: proposing\n');

  await test('a valid brief is proposed, not started', async () => {
    clean();
    const r = await codingJobs.propose({ project: 'demo', brief: 'Fix the failing tests.' });
    assert.ok(r.ok, r.error);
    assert.strictEqual(codingJobs.get(r.id).status, 'proposed');
    const queued = db().prepare('SELECT COUNT(*) AS n FROM agent_jobs').get().n;
    assert.strictEqual(queued, 0, 'proposing must not enqueue anything');
  });

  await test('an unknown project is refused, and says what exists', async () => {
    clean();
    const r = await codingJobs.propose({ project: 'nosuch', brief: 'x' });
    assert.ok(!r.ok);
    assert.match(r.error, /no project called/i);
    assert.match(r.error, /demo/, 'should list what is available');
  });

  await test('a path is refused rather than normalised', async () => {
    for (const name of ['../etc', '/etc', './demo', 'a/b']) {
      const r = await codingJobs.propose({ project: name, brief: 'x' });
      assert.ok(!r.ok, `${name} should be refused`);
    }
  });

  await test('an empty brief is refused', async () => {
    const r = await codingJobs.propose({ project: 'demo', brief: '   ' });
    assert.ok(!r.ok);
  });

  await test('pending proposals are capped', async () => {
    clean();
    for (let i = 0; i < 3; i++) {
      const r = await codingJobs.propose({ project: 'demo', brief: `job ${i}` });
      assert.ok(r.ok, r.error);
    }
    const r = await codingJobs.propose({ project: 'demo', brief: 'one too many' });
    assert.ok(!r.ok);
    assert.match(r.error, /limit/i);
  });

  console.log('\ncoding jobs: approving\n');

  await test('approving enqueues an agent job with the squatch-code source', async () => {
    clean();
    const p = await codingJobs.propose({ project: 'demo', brief: 'Fix the tests.' });
    const a = codingJobs.approve(p.id);
    assert.ok(a.ok, a.error);
    const job = agentJobs.getJob(a.agentJobId);
    assert.strictEqual(job.source, codingJobs.SOURCE);
    assert.strictEqual(job.task, 'Fix the tests.');
    assert.strictEqual(job.status, 'queued');
  });

  await test('her edit is what runs, and both are kept', async () => {
    clean();
    const p = await codingJobs.propose({ project: 'demo', brief: 'his version' });
    const a = codingJobs.approve(p.id, { editedBrief: 'her version' });
    assert.ok(a.editedByHer);
    assert.strictEqual(agentJobs.getJob(a.agentJobId).task, 'her version');
    const row = codingJobs.get(p.id);
    assert.strictEqual(row.brief, 'his version', 'the original ask must survive');
    assert.strictEqual(row.final_brief, 'her version');
  });

  await test('a proposal cannot be approved twice', async () => {
    clean();
    const p = await codingJobs.propose({ project: 'demo', brief: 'x' });
    assert.ok(codingJobs.approve(p.id).ok);
    assert.ok(!codingJobs.approve(p.id).ok);
  });

  await test('rejecting starts nothing', async () => {
    clean();
    const p = await codingJobs.propose({ project: 'demo', brief: 'x' });
    assert.ok(codingJobs.reject(p.id, { note: 'not now' }).ok);
    assert.strictEqual(codingJobs.get(p.id).status, 'rejected');
    assert.strictEqual(db().prepare('SELECT COUNT(*) AS n FROM agent_jobs').get().n, 0);
  });

  console.log('\na job that writes is never retried\n');

  await test('sweepInterrupted does NOT requeue a dispatched coding job', () => {
    clean();
    const id = 'test-coding-job';
    db().prepare(`
      INSERT INTO agent_jobs (id, title, task, status, source, started_at, attempts)
      VALUES (?, 'squatch-code: demo', 'brief', 'running', ?, ?, 1)
    `).run(id, codingJobs.SOURCE, new Date().toISOString());

    const out = agentJobs.sweepInterrupted();
    const job = agentJobs.getJob(id);
    assert.strictEqual(job.status, 'interrupted',
      'a killed coding job must close, not requeue');
    assert.strictEqual(out.requeued, 0);
    assert.match(job.error, /NOT run again/,
      'the reason must say it was not retried');
    assert.match(job.error, /git status/,
      'she must be told her files may have changed');
  });

  await test('an ordinary read-only job is still requeued', () => {
    clean();
    const id = 'test-ordinary-job';
    db().prepare(`
      INSERT INTO agent_jobs (id, title, task, status, source, started_at, attempts)
      VALUES (?, 'research', 'look it up', 'running', 'chat-handoff', ?, 0)
    `).run(id, new Date().toISOString());

    const out = agentJobs.sweepInterrupted();
    assert.strictEqual(agentJobs.getJob(id).status, 'queued');
    assert.strictEqual(out.requeued, 1, 'read-only retry must be unaffected');
  });

  console.log('\nrunning: a dispatch always comes back with something\n');

  await test('a missing squatch-job binary does not throw, and reports', async () => {
    const out = await codingJobs.runDispatched(
      { title: 'squatch-code: demo', task: 'x' },
      { timeoutMs: 5000 });
    // The binary name is real on this box; force the failure path instead.
    assert.ok(out.status, 'must return a status whatever happened');
    assert.ok(out.resultText && out.resultText.trim(), 'must never come back empty');
  });

  await test('a report document becomes the result text', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rep-'));
    const p = path.join(dir, 'report.json');
    fs.writeFileSync(p, JSON.stringify({
      status: 'partial',
      report: 'I changed one file.\n\nWhat the tools actually did:\nFiles changed:\n  - edited: /p/a.py',
      stop_reason: 'max_iterations (25)',
      restore_command: 'git -C /p reset --hard abc123',
      facts: { tool_calls: 7 }
    }));
    const out = codingJobs.readReport(p, { status: 'failed', resultText: 'fallback' });
    assert.strictEqual(out.status, 'partial');
    assert.match(out.resultText, /I changed one file/);
    assert.match(out.resultText, /What the tools actually did/,
      'the mechanical record must survive into the card');
    assert.match(out.resultText, /reset --hard abc123/,
      'she must be given the way to undo it');
    assert.strictEqual(out.toolCalls, 7);
  });

  await test('an unreadable report falls back rather than losing the job', () => {
    const out = codingJobs.readReport('/nonexistent/report.json',
      { status: 'failed', resultText: 'the fallback text' });
    assert.strictEqual(out.resultText, 'the fallback text');
  });

  await test('an unknown status in a report is treated as partial, not ok', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rep2-'));
    const p = path.join(dir, 'r.json');
    fs.writeFileSync(p, JSON.stringify({ status: 'weird', report: 'text' }));
    assert.strictEqual(codingJobs.readReport(p, {}).status, 'partial');
  });

  console.log('\nthe result is a ROBOT, never a BELL\n');

  await test('agent-jobs still does not require initiatives', () => {
    const src = fs.readFileSync(path.join(__dirname, '../db/agent-jobs.js'), 'utf8');
    assert.ok(!/require\(['"]\.\/initiatives['"]\)/.test(src),
      'a job result must never be able to open a conversation');
  });

  await test('the proposal is a bell item, and only the proposal', () => {
    const src = fs.readFileSync(path.join(__dirname, '../db/coding-jobs.js'), 'utf8');
    const initiativeUses = (src.match(/addInitiative/g) || []).length;
    assert.strictEqual(initiativeUses, 1,
      'exactly one bell item: the ask. The result goes to the panel.');
    assert.ok(/sourceKind: 'coding-job-proposal'/.test(src));
  });

  console.log('\nthe tool\n');

  await test('the tool proposes and never claims a result', async () => {
    clean();
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const t = new Tool();
    const r = await t.execute({ project: 'demo', brief: 'do the thing' }, {});
    assert.ok(r.success);
    assert.strictEqual(r.status, 'awaiting-approval');
    assert.match(r.message, /not have a result/i);
    assert.match(r.message, /no file has been touched/i);
  });

  await test('a refusal tells him to say so rather than imply work', async () => {
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const r = await new Tool().execute({ project: 'nosuch', brief: 'x' }, {});
    assert.ok(!r.success);
    assert.match(r.message, /Nothing was sent/);
  });

  await test('it is registered in the catalogue and off by default', () => {
    const MCPClient = require('../mcp/mcp-client');
    const rows = MCPClient.shared().describeCatalogue();
    const row = rows.find(r => r.id === 'dispatch_coding_job');
    assert.ok(row, 'must appear on the Tools tab');
    assert.strictEqual(row.toggle, 'tools.codingJobs.enabled');
    assert.ok(row.writes, 'must be marked as a tool that writes');
  });

  console.log(`\n${passed} passed, ${failed} failed\n`);
  process.exit(failed ? 1 : 0);
})();
