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
  // A project that does not exist yet is NOT an error. A new build is an
  // ordinary thing to ask for, and squatch-code creates the directory,
  // git-inits it and commits an empty baseline so the first job is as
  // undoable as any other. It also reports what it created, and names a
  // near-miss if the name is close to an existing project - which is
  // where a typo gets caught, without blocking a legitimate new project
  // whose name happens to resemble an old one.
  const exists = fs.existsSync(dir) && fs.statSync(dir).isDirectory();
  return { ok: true, dir, isNew: !exists };
}

/**
 * Existing projects whose names are close to this one.
 *
 * Auto-creating a project turns a typo into a new empty directory and a
 * job that builds in the wrong place. This does not block - a new
 * project may legitimately resemble an old one - it gives him something
 * to say at the moment she is reading the reply.
 */
function nearMatches(name) {
  const existing = listProjects().filter(p => p !== name);
  const a = String(name).toLowerCase();
  return existing.filter(p => {
    const b = p.toLowerCase();
    if (Math.abs(a.length - b.length) > 3) return false;
    // Cheap containment/prefix test: a typo is nearly always one of these.
    return a.startsWith(b.slice(0, Math.max(3, b.length - 2)))
        || b.startsWith(a.slice(0, Math.max(3, a.length - 2)));
  }).slice(0, 2);
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
 * Where squatch-job actually is, or null if it cannot be found.
 *
 * The first real dispatch failed with "spawn squatch-job ENOENT". Every
 * test until then had invoked it directly from a shell that had the
 * virtualenv on PATH; the systemd user service does not. Its PATH is the
 * default /usr/local/sbin:/usr/local/bin:... and nothing puts
 * ~/squatch-code/.venv/bin on it.
 *
 * Resolved rather than assumed, in this order:
 *   1. tools.codingJobs.binary, if it names a path
 *   2. PATH, for an installed entry point
 *   3. the conventional virtualenv beside the projects directory
 *
 * Returns an absolute path, or null. A null is not an error here - it is
 * the answer to "can this work at all", and the tool gate and the
 * capability manifest both ask.
 */
function resolveBinary(c = cfg()) {
  const configured = String(c.binary || 'squatch-job');

  const usable = (p) => {
    try { fs.accessSync(p, fs.constants.X_OK); return true; } catch (_) { return false; }
  };

  if (configured.includes('/')) {
    return usable(configured) ? path.resolve(configured) : null;
  }

  for (const dir of String(process.env.PATH || '').split(path.delimiter)) {
    if (!dir) continue;
    const candidate = path.join(dir, configured);
    if (usable(candidate)) return candidate;
  }

  // The layout on this machine, and the one squatch-code installs into.
  for (const candidate of [
    path.join(os.homedir(), 'squatch-code', '.venv', 'bin', configured),
    path.join(os.homedir(), '.local', 'bin', configured),
  ]) {
    if (usable(candidate)) return candidate;
  }

  return null;
}

/**
 * One line for boot and for the Tools tab: can this run, and from where.
 */
function binaryStatus(c = cfg()) {
  const resolved = resolveBinary(c);
  return resolved
    ? { ok: true, path: resolved }
    : {
        ok: false,
        path: null,
        why: `squatch-job was not found on PATH or at any known location. ` +
             `Set tools.codingJobs.binary to its full path ` +
             `(e.g. ${path.join(os.homedir(), 'squatch-code/.venv/bin/squatch-job')}).`
      };
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

  // Refuse before writing a row or queueing anything. A job that cannot
  // possibly start should not appear in the panel as one that failed.
  const bin = binaryStatus(c);
  if (!bin.ok) {
    opsLog(`dispatch_coding_job REFUSED: ${bin.why}`);
    return { ok: false, error: `squatch-code is not runnable from here. ${bin.why}`, unrunnable: true };
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
    isNewProject: !!v.isNew,
    nearMatches: v.isNew ? nearMatches(project.trim()) : [],
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

    const bin = resolveBinary(c) || c.binary || 'squatch-job';
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
  resolveBinary,
  binaryStatus,
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
