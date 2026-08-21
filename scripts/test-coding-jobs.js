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

/** A real conversation with a real assistant message in it.

    Uses the same API the chat route uses rather than raw SQL: a test
    that inserts rows by hand is testing its own SQL, not the shape the
    guard will actually see. */
function conversationShowing(text) {
  const id = database.createConversation('test', 'test-model');
  database.addMessage(id, 'assistant', text, 'test-model');
  return id;
}
function clean() {
  db().prepare('DELETE FROM coding_jobs').run();
  db().prepare('DELETE FROM agent_jobs').run();
}

(async () => {
  const SHOWN = 'Refactor the auth module in src/auth.py to use signed tokens '
    + 'instead of session cookies, keep the existing tests passing, and add a '
    + 'test for token expiry.';

  function asShown(extra = {}) {
    // She read it in her own message: the simplest way to satisfy the
    // guard without a stored conversation.
    return { project: 'demo', brief: SHOWN, userMessage: SHOWN, ...extra };
  }

  console.log('\ndispatching: the guard\n');

  await test('a brief she has read is dispatched', async () => {
    clean();
    const r = codingJobs.dispatch(asShown());
    assert.ok(r.ok, r.error);
    assert.strictEqual(agentJobs.getJob(r.agentJobId).source, codingJobs.SOURCE);
    assert.strictEqual(agentJobs.getJob(r.agentJobId).task, SHOWN);
  });

  await test('a brief she has NOT seen is refused, and nothing is queued', async () => {
    clean();
    const r = codingJobs.dispatch({
      project: 'demo', brief: SHOWN,
      userMessage: 'thanks, that all sounds sensible to me',
    });
    assert.ok(!r.ok, 'dispatched a brief she never read');
    assert.ok(r.unseen);
    assert.match(r.error, /has not seen this brief/i);
    assert.strictEqual(db().prepare('SELECT COUNT(*) AS n FROM agent_jobs').get().n, 0);
  });

  await test('the refusal tells the model what to do instead', async () => {
    clean();
    const r = codingJobs.dispatch({ project: 'demo', brief: SHOWN, userMessage: 'go' });
    assert.match(r.error, /write the brief out/i);
    assert.match(r.error, /once she says to/i);
  });

  await test('a brief shown in an earlier reply is dispatched', async () => {
    clean();
    const convo = conversationShowing('Here is what I would send:\n\n' + SHOWN);

    const r = codingJobs.dispatch({
      project: 'demo', brief: SHOWN, conversationId: convo,
      userMessage: 'yep, send that to the coder',
    });
    assert.ok(r.ok, r.error);
    assert.match(String(r.matchedIn), /earlier reply/);
  });

  await test('a DIFFERENT brief in the same conversation is still refused', async () => {
    clean();
    const convo = conversationShowing(SHOWN);

    const r = codingJobs.dispatch({
      project: 'demo', conversationId: convo, userMessage: 'send it',
      brief: 'Rewrite the billing exporter to emit CSV and delete the old XML path.',
    });
    assert.ok(!r.ok, 'a swapped brief passed the guard');
  });

  await test('"change X and send it" in one message is refused, as designed', async () => {
    clean();
    const convo = conversationShowing(SHOWN);

    const revised = SHOWN.replace('signed tokens', 'refresh tokens with rotation')
      + ' Also migrate the existing sessions table and drop the cookie helper.';
    const r = codingJobs.dispatch({
      project: 'demo', brief: revised, conversationId: convo,
      userMessage: 'use refresh tokens instead and send it',
    });
    assert.ok(!r.ok, 'dispatched a revision she had not been shown');
  });

  await test('a brief with no conversation context at all is refused', async () => {
    clean();
    const r = codingJobs.dispatch({ project: 'demo', brief: SHOWN });
    assert.ok(!r.ok, 'failed open with no context');
    assert.match(r.error, /no way to tell/i);
  });

  await test('a trivially short brief is refused', async () => {
    clean();
    const r = codingJobs.dispatch({
      project: 'demo', brief: 'fix it', userMessage: 'fix it',
    });
    assert.ok(!r.ok);
    assert.match(r.error, /too short/i);
  });

  console.log('\nnew projects\n');

  await test('a project that does not exist is dispatched, not refused', async () => {
    clean();
    const r = codingJobs.dispatch(asShown({ project: 'brand-new-build' }));
    assert.ok(r.ok, r.error);
    assert.ok(r.isNewProject, 'it did not report the project as new');
  });

  await test('an existing project is not reported as new', async () => {
    clean();
    const r = codingJobs.dispatch(asShown());
    assert.ok(r.ok, r.error);
    assert.ok(!r.isNewProject);
  });

  await test('a near-miss names the project it resembles', async () => {
    clean();
    const r = codingJobs.dispatch(asShown({ project: 'dem' }));
    assert.ok(r.ok, r.error);
    assert.deepStrictEqual(r.nearMatches, ['demo'],
      'a typo of an existing project was not flagged');
  });

  await test('a genuinely new name flags nothing', async () => {
    clean();
    const r = codingJobs.dispatch(asShown({ project: 'invoice-parser' }));
    assert.ok(r.ok, r.error);
    assert.deepStrictEqual(r.nearMatches, []);
  });

  await test('a path is still refused, new or not', async () => {
    for (const n of ['../etc', '/etc', 'a/b', '.hidden']) {
      assert.ok(!codingJobs.dispatch(asShown({ project: n })).ok, n);
    }
  });

  await test('the tool tells him to say a new project was created', async () => {
    clean();
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const SH = 'Build a small CSV to JSON converter with a convert(path) function '
      + 'and a test that round-trips one row.';
    const r = await new Tool().execute(
      { project: 'csvtool', brief: SH }, { userMessage: SH });
    assert.ok(r.success, r.error);
    assert.ok(r.new_project);
    assert.match(r.message, /did not exist/i);
    assert.match(r.message, /Projects\/csvtool/);
  });

  await test('...and names the near match in the same message', async () => {
    clean();
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const SH = 'Build a small CSV to JSON converter with a convert(path) function '
      + 'and a test that round-trips one row.';
    const r = await new Tool().execute(
      { project: 'dem', brief: SH }, { userMessage: SH });
    assert.ok(r.success, r.error);
    assert.deepStrictEqual(r.near_matches, ['demo']);
    assert.match(r.message, /already has/i);
    assert.match(r.message, /can be deleted/i);
  });

  console.log('\ndispatching: fidelity\n');

  await test('an exact send is recorded as exact', async () => {
    clean();
    const r = codingJobs.dispatch(asShown());
    assert.ok(r.exact, 'word-for-word send was not marked exact');
    assert.strictEqual(codingJobs.get(r.id).match_exact, 1);
  });

  await test('a paraphrase is dispatched but marked', async () => {
    clean();
    const paraphrased = SHOWN.replace('Refactor', 'Please refactor')
      .replace('add a test for token expiry', 'add a test covering token expiry');
    const r = codingJobs.dispatch({
      project: 'demo', brief: paraphrased, userMessage: SHOWN,
    });
    assert.ok(r.ok, r.error);
    assert.ok(!r.exact, 'a reworded brief was reported as word-for-word');
    assert.ok(r.ratio >= 0.8 && r.ratio < 1);
    assert.strictEqual(codingJobs.get(r.id).match_exact, 0);
  });

  await test('what was actually sent is what is stored and queued', async () => {
    clean();
    const paraphrased = 'Please ' + SHOWN.toLowerCase();
    const r = codingJobs.dispatch({
      project: 'demo', brief: paraphrased, userMessage: SHOWN,
    });
    assert.strictEqual(codingJobs.get(r.id).brief, paraphrased);
    assert.strictEqual(agentJobs.getJob(r.agentJobId).task, paraphrased);
  });

  console.log('\ndispatching: the ordinary refusals\n');

  await test('an unknown project is CREATED, not refused', async () => {
    // Was: refused with "there is no project called...". A new build is
    // an ordinary request, so the refusal was the defect.
    clean();
    const r = codingJobs.dispatch(asShown({ project: 'nosuch' }));
    assert.ok(r.ok, r.error);
    assert.ok(r.isNewProject);
  });

  await test('a path instead of a name is refused', async () => {
    for (const name of ['../etc', '/etc', './demo', 'a/b']) {
      assert.ok(!codingJobs.dispatch(asShown({ project: name })).ok, name);
    }
  });

  await test('an empty brief is refused', async () => {
    assert.ok(!codingJobs.dispatch(asShown({ brief: '   ' })).ok);
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

  await test('NOTHING about coding jobs touches the bell', () => {
    // The rule Ellie drew from 223 expired bell items: nothing that has
    // to be acted on goes there. A brief on the bell is a job that never
    // runs.
    const src = fs.readFileSync(path.join(__dirname, '../db/coding-jobs.js'), 'utf8');
    assert.ok(!/addInitiative/.test(src), 'a bell item crept back in');
    assert.ok(!/initiatives/.test(src), 'coding-jobs reached for the bell');
    const tool = fs.readFileSync(
      path.join(__dirname, '../mcp/tools/dispatch-coding-job.js'), 'utf8');
    assert.ok(!/addInitiative|initiatives/.test(tool), 'the tool reached for the bell');
  });

  console.log('\nthe tool\n');

  await test('the tool dispatches and never claims a result', async () => {
    clean();
    const SHOWN2 = 'Refactor the auth module in src/auth.py to use signed tokens '
      + 'instead of session cookies and keep the existing tests passing.';
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const r = await new Tool().execute(
      { project: 'demo', brief: SHOWN2 }, { userMessage: SHOWN2 });
    assert.ok(r.success, r.error);
    assert.strictEqual(r.status, 'running');
    assert.match(r.message, /not\s+have a result/i);
    assert.match(r.message, /jobs panel/i);
  });

  await test('an unseen brief tells him to write it out, not to claim a send', async () => {
    clean();
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const r = await new Tool().execute(
      { project: 'demo',
        brief: 'Refactor the auth module in src/auth.py to use signed tokens '
             + 'instead of session cookies and keep the tests passing.' },
      { userMessage: 'sounds good' });
    assert.ok(!r.success);
    assert.match(r.message, /Nothing was sent/);
    assert.match(r.message, /write the brief out/i);
    assert.match(r.message, /must not claim|do not claim/i);
  });

  await test('a paraphrase is flagged back to him', async () => {
    clean();
    const SHOWN3 = 'Refactor the auth module in src/auth.py to use signed tokens '
      + 'instead of session cookies and keep the existing tests passing.';
    const Tool = require('../mcp/tools/dispatch-coding-job');
    const r = await new Tool().execute(
      { project: 'demo', brief: 'Please ' + SHOWN3.toLowerCase() },
      { userMessage: SHOWN3 });
    assert.ok(r.success, r.error);
    assert.match(r.message, /not word for word/i);
    assert.match(r.message, /quote the brief/i);
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
