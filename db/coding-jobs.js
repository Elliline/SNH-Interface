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

const { randomUUID, createHash } = require('crypto');
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
  const raw = String(name || '').trim();
  if (!raw) return { ok: false, error: 'A project name is required.' };

  if (raw.includes('/') || raw.includes('\\') || raw.startsWith('.')) {
    const suggestion = suggestFromPath(raw);
    return {
      ok: false,
      error: `Give the project NAME, not a path: got ${JSON.stringify(raw)}.` +
        (suggestion
          ? ` Use "${suggestion}" as the project — it is created if it does not exist — and take the path out of the brief.`
          : ' A project is a single name under Projects/, never a path.'),
      suggestion,
    };
  }

  // "Squatch Crawler" -> "squatch_crawler", which is what she means and
  // what she expects to find on disk.
  const n = normaliseProjectName(raw);
  if (!n) return { ok: false, error: 'A project name is required.' };
  const dir = path.join(projectsRoot(), n);
  // A project that does not exist yet is NOT an error. A new build is an
  // ordinary thing to ask for, and squatch-code creates the directory,
  // git-inits it and commits an empty baseline so the first job is as
  // undoable as any other. It also reports what it created, and names a
  // near-miss if the name is close to an existing project - which is
  // where a typo gets caught, without blocking a legitimate new project
  // whose name happens to resemble an old one.
  const exists = fs.existsSync(dir) && fs.statSync(dir).isDirectory();
  return { ok: true, dir, isNew: !exists, name: n, renamed: n !== raw };
}

/**
 * Existing projects whose names are close to this one.
 *
 * Auto-creating a project turns a typo into a new empty directory and a
 * job that builds in the wrong place. This does not block - a new
 * project may legitimately resemble an old one - it gives him something
 * to say at the moment she is reading the reply.
 */
/**
 * The directory name for a project the user described in words.
 *
 * She says "build Squatch Crawler" and expects Projects/squatch_crawler.
 * Lowercased, spaces and hyphens to underscores, nothing else touched -
 * so a name that arrives already correct passes through unchanged.
 */
function normaliseProjectName(raw) {
  return String(raw || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
}

/**
 * What a path-shaped name was probably meant to be, for the refusal.
 *
 * A refusal that only says "no" costs a turn. This turns
 * "Projects\\squatch crawler" into "use squatch_crawler", which the
 * model can act on immediately.
 */
function suggestFromPath(raw) {
  const last = String(raw || '').split(/[\\/]+/).filter(Boolean).pop() || '';
  const name = normaliseProjectName(last);
  return name && name !== 'projects' ? name : null;
}


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
 * A one-line answer to "what is it doing", for the turn she is reading.
 *
 * She found out a job had finished by opening a panel afterwards. This
 * exists so she does not have to go looking: a job that is running says
 * so in the reply, and a job that has gone quiet says THAT in the same
 * line rather than as a new alert - a stale action shown as if it were
 * current is worse than no line at all.
 *
 * NOT A PERCENTAGE. The run does not know how much work is left, so any
 * fraction would be invented. What it knows is what it just did and how
 * long it has been going. This is also the line CLAUDE.md's rule allows:
 * THAT a job runs and for how long, never how far along.
 *
 * Elapsed is computed HERE from started_at rather than read from the
 * file, because the file is written on events and a stored elapsed
 * freezes between them - it read "0s" for seventy seconds while a run
 * waited on its first model response, which looks exactly like a job
 * that has died.
 */
const QUIET_AFTER_MS = 120000;

function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return s >= 60 ? `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s` : `${s}s`;
}

function progressLine(progressPath, now = Date.now()) {
  let data;
  try {
    data = JSON.parse(fs.readFileSync(progressPath, 'utf8'));
  } catch (_) {
    return null;                       // not started writing yet
  }
  if (!data || typeof data !== 'object') return null;

  if (data.finished) {
    return `finished after ${formatDuration(data.elapsed_seconds || 0)}`;
  }

  const startedMs = data.started_at ? data.started_at * 1000 : null;
  const elapsed = startedMs ? (now - startedMs) / 1000 : (data.elapsed_seconds || 0);

  const parts = [];
  if (data.iteration) {
    parts.push(`step ${data.iteration}` + (data.max_iterations ? `/${data.max_iterations}` : ''));
  }
  parts.push(data.last_action || 'thinking');
  parts.push(formatDuration(elapsed));

  const silentMs = data.updated_at ? now - data.updated_at * 1000 : 0;
  if (silentMs >= QUIET_AFTER_MS) {
    parts.push(`no activity for ${Math.floor(silentMs / 60000)}m`);
  }
  return parts.join(' · ');
}

/**
 * Every dispatched coding job currently running, with where it is.
 * Empty array when nothing is running, so a caller renders nothing.
 */
function running(now = Date.now()) {
  const db = getSqliteDb();
  if (!db) return [];
  let rows;
  try {
    rows = db.prepare(`
      SELECT cj.project, cj.progress_path, aj.id AS job_id, aj.status
      FROM coding_jobs cj
      JOIN agent_jobs aj ON aj.id = cj.agent_job_id
      WHERE aj.status IN ('queued', 'running')
      ORDER BY aj.created_at ASC
    `).all();
  } catch (_) {
    return [];
  }
  return rows.map(r => ({
    project: r.project,
    jobId: r.job_id,
    queued: r.status === 'queued',
    line: r.status === 'queued'
      ? 'queued, not started yet'
      : (r.progress_path ? progressLine(r.progress_path, now) : null) || 'starting',
  }));
}

/** The block appended to a reply while work is in flight. */
/**
 * ⛔ NEVER APPEND THIS TO A REPLY. It is not called by the chat path any more.
 *
 * It used to be, and putting it inside his message is what made it forgeable:
 * the message is stored whole, so the block came back to him next turn as his
 * own words, and on 2026-08-22 he wrote one himself with a command that does
 * not exist while nothing was running. She could not tell it from the real
 * thing, which was the one signal she had.
 *
 * Live status is UI chrome now — #coderStrip, fed by /api/jobs/coding/active.
 * This is kept because the LINE composition below is what the tests exercise
 * and what that endpoint's contents are checked against. If you want status in
 * the transcript again, it must arrive as its own frame the client renders as
 * chrome, never as characters in his message.
 */
function statusBlock(now = Date.now()) {
  const live = running(now);
  if (!live.length) return null;
  const lines = live.map(j => `- **${j.project}** — ${j.line}`);
  return `\n\n---\n\n_squatch-code, working:_\n${lines.join('\n')}`;
}


/**
 * Refuse a brief that tries to decide WHERE the work goes.
 *
 * The tool description said not to do this. It said so at length, and it
 * did not hold: three times the entity wrote directory instructions into
 * the brief instead of naming the project - "check if Projects\\squatch
 * crawler exists, create it if not" - with a backslash and a space that
 * match no project name, dispatched into an unrelated project.
 *
 * A description is an argument. This is the mechanism. The project field
 * is the ONLY thing that decides where work lands, and a brief that
 * reaches for a path is refused with something the model can act on in
 * the same turn.
 *
 * Deliberately narrow: it matches instructions ABOUT directories and
 * project paths, not ordinary code that happens to mention a folder. A
 * false refusal here blocks real work, so the patterns require both a
 * verb of creation-or-location AND a path-like or Projects-rooted
 * object.
 */
const BRIEF_PATH_PATTERNS = [
  {
    // "create a directory", "make a folder", "mkdir ..."
    re: /\b(creat(e|ing)|make|mkdir|set up|initialise|initialize)\b[^.\n]{0,40}\b(director(y|ies)|folder|sub-?folder)\b/i,
    why: 'it tells the job to create a directory',
  },
  {
    // any reference rooted at Projects/, with either slash
    re: /\bprojects[\\/][^\s.,;)]+/i,
    why: 'it points at a path under Projects/',
  },
  {
    // "in the <name> directory/folder" as a placement instruction
    re: /\b(in|into|under|inside)\b[^.\n]{0,30}\b(director(y|ies)|folder)\b[^.\n]{0,20}\b(exists?|create|if not)\b/i,
    why: 'it tells the job where to put itself',
  },
  {
    // cd / chdir out of the project
    re: /\bcd\s+\.\.|\bchdir\b|\.\.[\\/]/i,
    why: 'it tries to move outside the project',
  },
];

/**
 * Record every dispatch attempt, including the refused ones.
 *
 * "Did it even try?" was unanswerable tonight: the tool wrote nothing to
 * tool_call_log, so a refusal and a model that never called the tool
 * looked identical from the outside. They are different problems and
 * need different fixes.
 */
/**
 * A REFUSED CALL AND A CALL NEVER MADE MUST NOT LOOK THE SAME FROM OUTSIDE.
 *
 * This function existed from the first day and had NEVER ONCE WRITTEN A ROW.
 * Two bugs, both silent: the INSERT named a column `args` that does not exist
 * (the schema says `args_json`), and `id` is a PRIMARY KEY with no default, so
 * even a corrected column list would have failed on a null id. Every failure
 * went into `catch (_) {}`.
 *
 * The cost was a whole diagnosis. On 2026-08-21 the model called this tool,
 * was refused because the project did not exist, and worked around the refusal
 * by dispatching into an unrelated project — and the record showed nothing at
 * all, so the refusal was invisible and I reported "the tool has never been
 * called". A swallowed write is worse than no write: it looks like evidence.
 *
 * The catch stays — logging may never break a dispatch — but it now says so on
 * the console, because a logger that fails quietly is the thing being fixed.
 */
function logAttempt(outcome, project, detail, conversationId, extra = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return;
    db.prepare(`
      INSERT INTO tool_call_log (id, tool, args_json, outcome, detail, conversation_id, created_at)
      VALUES (?, 'dispatch_coding_job', ?, ?, ?, ?, ?)
    `).run(
      randomUUID(),
      JSON.stringify({ project: project || null, ...extra }),
      outcome,
      String(detail || '').slice(0, 300),
      conversationId,
      new Date().toISOString()
    );
  } catch (err) {
    console.error('[CodingJobs] attempt log FAILED — a refusal is going unrecorded:', err.message);
  }
}

/**
 * The refusal record, with the brief that was refused.
 *
 * Separate from logAttempt because a refusal needs the brief to be diagnosable
 * at all — "rejected-brief" without the text tells you a refusal happened and
 * nothing about why it kept happening. The brief is stored as a hash plus a
 * leading excerpt rather than whole: the full text is already in the reply that
 * proposed it, and tool_call_log is read in bulk.
 */
function logRefusal({ reason, project, brief, conversationId, kind = 'rejected' }) {
  const text = String(brief || '');
  const hash = createHash('sha256').update(text).digest('hex').slice(0, 16);
  logAttempt(kind, project, reason, conversationId, {
    brief_sha256: hash,
    brief_chars: text.length,
    brief_head: text.slice(0, 160).replace(/\s+/g, ' '),
  });
  opsLog(`dispatch_coding_job REFUSED (${kind}): ${String(reason || '').slice(0, 200)} `
    + `[project ${project || 'none'}, brief ${hash} ${text.length} chars]`);
}


function validateBrief(text) {
  const brief = String(text || '');
  for (const { re, why } of BRIEF_PATH_PATTERNS) {
    const m = brief.match(re);
    if (m) {
      return {
        ok: false,
        matched: m[0].slice(0, 60),
        error:
          `The brief was rejected because ${why} ("${m[0].slice(0, 60).trim()}"). ` +
          `A brief describes WHAT to build, never where to put it. ` +
          `Where the work goes is decided by the project field and nothing else: ` +
          `put the project name there — "squatch_crawler", not a path, not ` +
          `"Projects\\squatch crawler" — and it is created if it does not exist. ` +
          `Remove the directory instructions from the brief, show her the ` +
          `corrected brief, and send that.`,
      };
    }
  }
  return { ok: true };
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
    logRefusal({ reason: 'coding jobs are switched off in configuration', project, brief, conversationId, kind: 'rejected-disabled' });
    return { ok: false, error: 'Dispatching coding jobs is switched off in configuration.' };
  }

  // Refuse before writing a row or queueing anything. A job that cannot
  // possibly start should not appear in the panel as one that failed.
  const bin = binaryStatus(c);
  if (!bin.ok) {
    opsLog(`dispatch_coding_job REFUSED: ${bin.why}`);
    logRefusal({ reason: bin.why, project, brief, conversationId, kind: 'rejected-unrunnable' });
    return { ok: false, error: `squatch-code is not runnable from here. ${bin.why}`, unrunnable: true };
  }

  const v = validateProject(project);
  if (!v.ok) {
    logRefusal({ reason: v.error, project, brief, conversationId, kind: 'rejected-project' });
    return { ok: false, error: v.error, suggestion: v.suggestion };
  }
  // Everything downstream uses the NORMALISED name, so what she was told
  // and what is on disk are the same string.
  const projectName = v.name;

  const text = String(brief || '').trim();
  if (!text) {
    logRefusal({ reason: 'no brief text', project, brief, conversationId, kind: 'rejected-empty' });
    return { ok: false, error: 'A brief is required - say what needs doing.' };
  }
  if (text.length > 4000) {
    logRefusal({ reason: 'brief over 4000 characters', project, brief, conversationId, kind: 'rejected-long' });
    return { ok: false, error: 'That brief is too long to dispatch (4000 characters max).' };
  }

  // The project field decides where work goes. A brief that reaches for
  // a path is refused here rather than argued with in a description.
  const briefCheck = validateBrief(text);
  if (!briefCheck.ok) {
    logRefusal({ reason: `brief names a path: ${briefCheck.matched}`, project, brief, conversationId, kind: 'rejected-brief' });
    opsLog(`dispatch_coding_job REJECTED brief for ${project}: ${briefCheck.matched}`);
    return { ok: false, error: briefCheck.error, briefRejected: true };
  }

  // THE GUARD.
  const seen = briefShown.check(text, { conversationId, userMessage });
  if (!seen.ok) {
    // Loud, never silent. A guard that quietly declines reproduces the
    // failure the bell had: work that never happens and nothing says so.
    opsLog('dispatch_coding_job REFUSED for ' + project + ': ' + seen.reason +
           ' (best match ' + (seen.ratio * 100).toFixed(0) + '%)');
    logRefusal({ reason: `brief not shown to her (coverage ${Math.round((seen.ratio || 0) * 100)}%)`, project, brief, conversationId, kind: 'rejected-unseen' });
    return { ok: false, error: seen.reason, unseen: true, ratio: seen.ratio };
  }

  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'The database is unavailable, so nothing was sent.' };

  const agentJobs = require('./agent-jobs');
  const queued = agentJobs.enqueue({
    title: 'squatch-code: ' + projectName,
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
  ).run(id, projectName, text, conversationId, messageId, queued.id,
        seen.ratio, seen.exact ? 1 : 0, new Date().toISOString());

  opsLog('dispatch_coding_job sent to ' + projectName + ': "' + text.slice(0, 80) +
         '" (job ' + queued.id.slice(0, 8) + (seen.exact ? '' : ', paraphrased') + ')');

  return {
    ok: true, id, agentJobId: queued.id,
    exact: seen.exact, ratio: seen.ratio, matchedIn: seen.source,
    isNewProject: !!v.isNew,
    nearMatches: v.isNew ? nearMatches(projectName) : [],
    project: projectName,
    renamed: v.renamed ? String(project).trim() : null,
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

    const progressPath = path.join(path.dirname(reportPath), 'progress.json');
    try {
      const db = getSqliteDb();
      if (db) {
        db.prepare('UPDATE coding_jobs SET progress_path = ? WHERE agent_job_id = ?')
          .run(progressPath, job.id);
      }
    } catch (_) { /* the status line is a convenience, never the job */ }

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

/**
 * SHE APPROVED IT AND THE REPLY DID NOT SEND IT — so the server does.
 *
 * The same shape as the tier-1 backstop for start_background_job, and here for
 * the same reason: asking has failed enough. Of five claimed dispatches, two
 * were real. The forced tool_choice is the first line and it is not a
 * guarantee — an engine may refuse the pin, and a refused pin falls back to an
 * unforced round on purpose — so the invariant is closed where it is checkable.
 *
 * WHAT IT SENDS, AND WHY THOSE ARE KNOWABLE RATHER THAN GUESSED:
 *   brief   — the most recent assistant message before this turn. When she says
 *             "go ahead and send it", the thing she just read IS the previous
 *             reply; that is what made her say it. It also means the brief is,
 *             by construction, one she was shown — so it passes the same
 *             brief-shown check every other dispatch passes, rather than
 *             bypassing it.
 *   project — the project of the last coding job in THIS conversation. The case
 *             this exists for is iterating on something already dispatched.
 *
 * WHEN IT CANNOT KNOW, IT DOES NOTHING AND SAYS SO. A first dispatch in a fresh
 * conversation has no prior project and there is nothing to infer from; the
 * caller falls back to correcting the claim in words. Inventing a project name
 * would put her work somewhere she did not choose, which is the failure the
 * project field exists to prevent.
 */
function backstopDispatch({ conversationId, messageId = null, userMessage = null }) {
  const db = getSqliteDb();
  if (!db || !conversationId) return { ok: false, reason: 'no conversation' };

  const prior = db.prepare(
    `SELECT project FROM coding_jobs WHERE conversation_id = ?
     ORDER BY created_at DESC LIMIT 1`
  ).get(conversationId);
  if (!prior || !prior.project) {
    return { ok: false, reason: 'no earlier coding job in this conversation, so no project to send it to' };
  }

  // NOT SIMPLY "THE LAST REPLY". Measured the first time this ran: the most
  // recent assistant message was the previous dispatch's confirmation, which
  // quotes the brief AND names `Projects/<name>` — so validateBrief refused it
  // for pointing at a path, and the backstop declined a dispatch it could
  // perfectly well have made. Walk back until a message is something the tool
  // would actually accept, which is the same bar every other dispatch clears.
  const replies = db.prepare(
    `SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant'
     ORDER BY timestamp DESC LIMIT 8`
  ).all(conversationId);

  let brief = null;
  let lastRefusal = null;
  for (const row of replies) {
    const text = String(row.content || '').trim();
    if (text.split(/\s+/).filter(Boolean).length < 20) continue;
    // A reply that is itself a report ABOUT a dispatch is not the brief.
    if (/\bsquatch-?code, working\b/i.test(text)) continue;
    const v = validateBrief(text);
    if (!v.ok) { lastRefusal = lastRefusal || v.error; continue; }
    brief = text;
    break;
  }
  if (!brief) {
    return {
      ok: false,
      kind: 'no-brief',
      reason: lastRefusal
        ? 'the text I would have sent names a directory, and a brief may not decide where work goes'
        : 'I could not find the brief you approved in this conversation',
    };
  }

  const result = dispatch({
    project: prior.project, brief, conversationId, messageId, userMessage,
  });
  return result.ok
    ? { ok: true, ...result, project: prior.project }
    : { ok: false, kind: 'refused', reason: result.error };
}

/** Coding jobs created in this conversation since a timestamp — "did THIS turn
 *  dispatch?", which is what every guard actually wants to know. agent_jobs
 *  counts research jobs too, and a research job is not an answer to a brief. */
function dispatchedInTurn(conversationId, sinceIso) {
  const db = getSqliteDb();
  if (!db || !conversationId) return 0;
  const row = db.prepare(
    'SELECT COUNT(*) n FROM coding_jobs WHERE conversation_id = ? AND created_at >= ?'
  ).get(conversationId, sinceIso);
  return (row && row.n) || 0;
}

/**
 * Jobs still in flight for a project — the same-project concurrency hazard.
 *
 * Two runs in one project is a KNOWN open hazard, not a theoretical one: each
 * takes its own restore point, and a second run started while the first is
 * working takes its baseline from a half-finished tree, so the two undo
 * commands no longer describe recoverable states. squatch-code serialises
 * within a run; nothing serialises across dispatches.
 *
 * This is why a re-run is refused rather than queued. Queueing would look
 * helpful and would silently produce the same tangle a few minutes later.
 */
function activeForProject(project) {
  const db = getSqliteDb();
  if (!db || !project) return [];
  return db.prepare(`
    SELECT c.id, c.created_at, j.status, j.id AS agent_job_id
    FROM coding_jobs c JOIN agent_jobs j ON j.id = c.agent_job_id
    WHERE c.project = ? AND j.status IN ('queued', 'running')
    ORDER BY c.created_at DESC
  `).all(project);
}

module.exports = {
  SOURCE,
  validateBrief,
  normaliseProjectName,
  progressLine,
  running,
  statusBlock,
  backstopDispatch,
  logAttempt,
  logRefusal,
  dispatchedInTurn,
  activeForProject,
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
