/**
 * coding-jobs — handing coding work to squatch-code.
 *
 * Ellie and the entity talk a problem through in chat. He writes the
 * brief IN HIS REPLY, where she reads it. She says send it. It
 * dispatches. That is the whole interface.
 *
 * NOTHING GOES THROUGH THE BELL. This shipped as a bell proposal with an
 * approve button, and that was wrong - not as a preference, as a
 * mechanism that does not work here. Measured on the live corpus: of 390
 * bell items ever raised, 223 EXPIRED (57%), and of proposals - the ones
 * that actually require an action - exactly one has ever been raised and
 * it was DISMISSED. There is no instance of the bell-approval path
 * producing an approved thing. A brief routed there would have sat until
 * it aged out, and the job would never have run.
 *
 * So the approval is conversational. Her "send it" IS the approval, and
 * the tool call is what it authorises. There is no proposal row waiting
 * on a decision, because by the time this is called the decision is made.
 *
 * THE GUARD IS THAT SHE MUST HAVE SEEN IT. db/brief-shown.js refuses any
 * brief that does not already appear in this conversation - in an earlier
 * reply, or in her own message. That is what stops a dispatch on the
 * drafting turn, structurally rather than by guessing at how she phrases
 * a go-ahead. See that file for why not a phrase classifier.
 *
 * WHAT KEEPS A BAD JOB SURVIVABLE, in the order it matters:
 *
 *   1. A git restore point committed inside the project before the run
 *      starts. Ellie named this as the risk she actually cares about.
 *   2. Path containment inside squatch-code, confining every file tool
 *      to Projects/<name>.
 *   3. A command allowlist - a speed bump on the restore point, and she
 *      accepted it as one.
 *
 * THE RESULT REUSES agent_jobs, so it inherits the jobs panel, the badge,
 * the announcement block and the ok/partial/failed vocabulary. The panel
 * is the right home for a RESULT: nothing there has to be acted on, so
 * nothing rots by being ignored.
 */

const { randomUUID } = require('crypto');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { getSqliteDb } = require('./database');
const briefShown = require('./brief-shown');

function cfg() {
  const c = require('./config').getConfig();
  return (c.tools && c.tools.codingJobs) || {};
}

function opsLog(line) {
  try { require('./ops-log').write(line); } catch (_) { /* optional */ }
}

const SOURCE = 'squatch-code';

/**
 * Projects live at Projects/<name>. Not a new convention — every saved
 * squatch-code session used that layout, so this follows what is already
 * true rather than declaring something.
 */
function projectsRoot() {
  return cfg().projectsRoot || path.join(os.homedir(), 'Projects');
}

/**
 * Names only. A path is refused rather than normalised, because the useful
 * error is "that is not a project name", and squatch-code refuses escapes
 * again at its own boundary.
 */
function validateProject(name) {
  const n = String(name || '').trim();
  if (!n) return { ok: false, error: 'A project name is required.' };
  if (n.includes('/') || n.includes('\\') || n.startsWith('.')) {
    return { ok: false, error: `Give the project NAME, not a path: got ${JSON.stringify(n)}.` };
  }
  const dir = path.join(projectsRoot(), n);
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    const available = listProjects();
    return {
      ok: false,
      error: `There is no project called "${n}" in ${projectsRoot()}.` +
        (available.length ? ` Available: ${available.join(', ')}.` : '')
    };
  }
  return { ok: true, dir };
}

function listProjects() {
  try {
    return fs.readdirSync(projectsRoot(), { withFileTypes: true })
      .filter(d => d.isDirectory() && !d.name.startsWith('.'))
      .map(d => d.name)
      .sort();
  } catch (_) {
    return [];
  }
}


/**
 * Send a brief to squatch-code. Her go-ahead already happened, in chat.
 *
 * Refuses unless the brief is already on her screen. A refusal comes back
 * in words the model can act on in the same turn - write the brief out,
 * ask - rather than as silence.
 */
function dispatch({ project, brief, conversationId = null, messageId = null,
                    userMessage = null }) {
  const c = cfg();
  if (c.enabled === false) {
    return { ok: false, error: 'Dispatching coding jobs is switched off in configuration.' };
  }

  const v = validateProject(project);
  if (!v.ok) return { ok: false, error: v.error };

  const text = String(brief || '').trim();
  if (!text) return { ok: false, error: 'A brief is required - say what needs doing.' };
  if (text.length > 4000) {
    return { ok: false, error: 'That brief is too long to dispatch (4000 characters max).' };
  }

  // THE GUARD.
  const seen = briefShown.check(text, { conversationId, userMessage });
  if (!seen.ok) {
    // Loud, never silent. A guard that quietly declines reproduces the
    // failure the bell had: work that never happens and nothing says so.
    opsLog('dispatch_coding_job REFUSED for ' + project + ': ' + seen.reason +
           ' (best match ' + (seen.ratio * 100).toFixed(0) + '%)');
    return { ok: false, error: seen.reason, unseen: true, ratio: seen.ratio };
  }

  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'The database is unavailable, so nothing was sent.' };

  const agentJobs = require('./agent-jobs');
  const queued = agentJobs.enqueue({
    title: 'squatch-code: ' + project.trim(),
    task: text,
    why: 'Sent to squatch-code after Ellie gave the go-ahead in conversation.',
    conversationId,
    messageId,
    source: SOURCE,
  });
  if (!queued.ok) return queued;

  const id = randomUUID();
  db.prepare(
    'INSERT INTO coding_jobs (id, project, brief, status, conversation_id, ' +
    'message_id, agent_job_id, match_ratio, match_exact, created_at) ' +
    "VALUES (?, ?, ?, 'dispatched', ?, ?, ?, ?, ?, ?)"
  ).run(id, project.trim(), text, conversationId, messageId, queued.id,
        seen.ratio, seen.exact ? 1 : 0, new Date().toISOString());

  opsLog('dispatch_coding_job sent to ' + project + ': "' + text.slice(0, 80) +
         '" (job ' + queued.id.slice(0, 8) + (seen.exact ? '' : ', paraphrased') + ')');

  return {
    ok: true, id, agentJobId: queued.id,
    exact: seen.exact, ratio: seen.ratio, matchedIn: seen.source,
  };
}


/** Fetch one dispatch record. */
function get(id) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare('SELECT * FROM coding_jobs WHERE id = ?').get(id) || null;
}


/**
 * Run a dispatched job by shelling out to squatch-job.
 *
 * Returns the shape agent-jobs' finish() wants. It never throws: a job that
 * started must always come back with something a person can read, and that
 * rule does not get an exception for the process failing to launch.
 */
function runDispatched(job, { timeoutMs = null } = {}) {
  const c = cfg();
  const limitMs = timeoutMs || (c.timeoutMinutes ?? 20) * 60 * 1000;
  const project = projectOf(job);

  return new Promise(resolve => {
    let reportPath;
    try {
      reportPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'squatch-job-')), 'report.json');
    } catch (err) {
      return resolve({
        status: 'failed',
        resultText: `Could not create a place for the job's report: ${err.message}. Nothing was dispatched.`,
        error: err.message
      });
    }

    const args = [
      '--project', project,
      '--projects-root', projectsRoot(),
      '--brief-file', '-',
      '--report-json', reportPath
    ];
    for (const cmd of (c.allowedCommands || DEFAULT_ALLOWED_COMMANDS)) {
      args.push('--allow-command', cmd);
    }

    const bin = c.binary || 'squatch-job';
    const child = spawn(bin, args, { stdio: ['pipe', 'pipe', 'pipe'] });

    let stderr = '';
    let settled = false;
    const done = (payload) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(payload);
    };

    const timer = setTimeout(() => {
      try { child.kill('SIGKILL'); } catch (_) { /* already gone */ }
      done(readReport(reportPath, {
        status: 'partial',
        resultText:
          `The job was still running after ${Math.round(limitMs / 60000)} minutes and was stopped.\n\n` +
          `Files it had already changed are still changed. Check the project's git status; ` +
          `the job commits a restore point before it starts, so "git log" in Projects/${project} ` +
          `will show it.`,
        error: `killed after ${limitMs}ms`
      }));
    }, limitMs);

    child.stderr.on('data', d => { stderr += d.toString().slice(0, 4000); });
    child.on('error', err => done({
      status: 'failed',
      resultText: `Could not start ${bin}: ${err.message}. Nothing ran, and nothing in the project was touched.`,
      error: err.message
    }));

    child.on('close', () => done(readReport(reportPath, {
      status: 'failed',
      resultText:
        'squatch-code exited without writing a report, so there is no account of what it did. ' +
        (stderr ? `It said:\n\n${stderr.trim()}` : 'It said nothing on stderr.') +
        `\n\nCheck git status in Projects/${project} before assuming nothing changed.`,
      error: stderr.trim() || 'no report written'
    })));

    try {
      child.stdin.write(job.task || '');
      child.stdin.end();
    } catch (err) {
      done({ status: 'failed', resultText: `Could not send the brief: ${err.message}`, error: err.message });
    }
  });
}

const DEFAULT_ALLOWED_COMMANDS = ['pytest', 'python', 'python3', 'node', 'npm', 'go', 'cargo', 'make', 'git', 'ls', 'cat'];

/** The project a dispatched agent job belongs to, from its title. */
function projectOf(job) {
  const m = /^squatch-code:\s*(.+)$/.exec(job.title || '');
  return m ? m[1].trim() : '';
}

/**
 * Read the report squatch-code wrote, or fall back to what we know.
 *
 * The report is already bound to its own mechanical record on the far side —
 * squatch-code appends what the tools actually did to whatever the model
 * wrote — so nothing here needs to re-derive it, and nothing here should
 * paraphrase it either.
 */
function readReport(reportPath, fallback) {
  try {
    const doc = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
    let text = doc.report || '';
    if (doc.restore_command) {
      text += `\n\nTo undo this whole job:\n\n    ${doc.restore_command}`;
    }
    return {
      status: ['ok', 'partial', 'failed'].includes(doc.status) ? doc.status : 'partial',
      resultText: text.trim() || fallback.resultText,
      error: doc.status === 'ok' ? null : (doc.stop_reason || null),
      toolCalls: (doc.facts && doc.facts.tool_calls) || 0,
      document: doc
    };
  } catch (_) {
    return fallback;
  }
}

module.exports = {
  SOURCE,
  DEFAULT_ALLOWED_COMMANDS,
  dispatch,
  get,
  runDispatched,
  readReport,
  validateProject,
  listProjects,
  projectsRoot,
  projectOf
};
