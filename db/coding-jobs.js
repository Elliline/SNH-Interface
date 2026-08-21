/**
 * coding-jobs — handing work to squatch-code.
 *
 * Ellie and the entity settle on what to do in chat; she says "send it to
 * squatch-code"; he writes the brief, she approves it, and it goes. The
 * point is removing the clipboard from a loop she already runs by hand.
 *
 * THIS TOOL COMBINES TWO PATTERNS THAT HAVE ALWAYS BEEN SEPARATE HERE, and
 * the reason is worth stating because each existing tool argues the opposite
 * case for itself:
 *
 *   - create_cron_job is PROPOSE-ONLY. She decides, because a cron job
 *     recurs forever.
 *   - start_background_job is DIRECT-EXECUTE, and says so in its own header:
 *     "starting a read-only background lookup is not a decision that needs
 *     Ellie's approval — it changes nothing".
 *
 * That last clause is exactly what fails here. A coding job WRITES FILES on
 * her machine, unattended, with nobody at squatch-code's approval prompt. So
 * it takes the cron tool's gate and the job tool's handoff: proposed, shown
 * to her, approved, and only then dispatched — and once dispatched it runs
 * without further consent, because there is nobody to ask. WHAT SHE APPROVES
 * IS THE BRIEF, NOT THE ACTIONS. Everything downstream depends on that being
 * understood, so the proposal shows her the brief verbatim.
 *
 * WHAT KEEPS A BAD JOB SURVIVABLE, in the order it matters:
 *
 *   1. A git restore point, committed inside the project before the run
 *      starts. Ellie named this as the thing she actually cares about: a
 *      dispatched job with no way back is the risk. squatch-code takes it and
 *      reports the command that undoes the whole job.
 *   2. Path containment inside squatch-code, confining every file tool to
 *      Projects/<name>.
 *   3. A command allowlist. A speed bump on top of the restore point, and
 *      she accepted it as one — an allowed interpreter can run anything.
 *
 * THE RESULT REUSES agent_jobs ON PURPOSE. A dispatched job is enqueued with
 * source 'squatch-code', so it inherits the jobs panel, the unread badge, the
 * announcement block, the seen/announced stamps and the ok/partial/failed
 * vocabulary without a second panel that could disagree with the first.
 * squatch-code's own exit statuses were made to match this vocabulary rather
 * than the other way round.
 *
 * AND THE RESULT IS STILL A ROBOT, NEVER A BELL. A finished coding job lands
 * in the panel and does not open a conversation. The PROPOSAL is a bell item,
 * because asking is something he wants to say; the RESULT is not.
 */

const { randomUUID } = require('crypto');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { getSqliteDb } = require('./database');

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
 * Record a proposal and raise it for Ellie. Creates NOTHING that runs.
 */
async function propose({ project, brief, conversationId = null, messageId = null }) {
  const c = cfg();
  if (c.enabled === false) {
    return { ok: false, error: 'Dispatching coding jobs is switched off in configuration.' };
  }

  const v = validateProject(project);
  if (!v.ok) return { ok: false, error: v.error };

  const text = String(brief || '').trim();
  if (!text) return { ok: false, error: 'A brief is required — say what needs doing.' };
  if (text.length > 4000) {
    return { ok: false, error: 'That brief is too long to dispatch (4000 characters max).' };
  }

  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'The database is unavailable, so nothing was proposed.' };

  const pending = db.prepare(
    "SELECT COUNT(*) AS n FROM coding_jobs WHERE status = 'proposed'"
  ).get().n;
  const maxPending = c.maxPendingProposals ?? 3;
  if (pending >= maxPending) {
    return {
      ok: false,
      error: `There are already ${pending} coding jobs waiting for her approval, which is the limit (${maxPending}). Nothing was proposed — say so rather than proposing another.`
    };
  }

  const id = randomUUID();
  db.prepare(`
    INSERT INTO coding_jobs (id, project, brief, status, conversation_id, message_id, created_at)
    VALUES (?, ?, ?, 'proposed', ?, ?, ?)
  `).run(id, v.dir && project.trim(), text, conversationId, messageId, new Date().toISOString());

  // The bell carries the ASK. She sees the brief verbatim, because the brief
  // is the thing she is approving.
  let initiativeId = null;
  try {
    const initiatives = require('./initiatives');
    initiativeId = await initiatives.addInitiative({
      type: 'proposal',
      content: `I'd like to send this to squatch-code, working in Projects/${project.trim()}:\n\n${text}\n\nApprove it and it runs unattended — it can edit files in that project and run test commands there. A restore point is committed first, so the whole job can be undone. Reject it and nothing happens.`,
      sourceKind: 'coding-job-proposal',
      sourceRef: id,
      priority: 7
    });
    if (initiativeId) {
      db.prepare('UPDATE coding_jobs SET initiative_id = ? WHERE id = ?').run(initiativeId, id);
    }
  } catch (err) {
    console.error('[CodingJobs] could not raise initiative:', err.message);
  }

  opsLog(`dispatch_coding_job proposed for ${project}: "${text.slice(0, 80)}" (${id.slice(0, 8)}, awaiting approval)`);
  return { ok: true, id, status: 'proposed' };
}

function get(id) {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare('SELECT * FROM coding_jobs WHERE id = ?').get(id) || null;
}

/**
 * Approve a proposal: enqueue it as an agent job with source 'squatch-code'.
 *
 * `editedBrief` is the point of showing it to her. She approves or CORRECTS —
 * a brief she rewrote is the one that runs, and the row keeps both so the
 * record shows what he asked for and what she actually sent.
 */
function approve(id, { editedBrief = null } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'Database unavailable.' };
  const row = get(id);
  if (!row) return { ok: false, error: 'No such proposal.' };
  if (row.status !== 'proposed') {
    return { ok: false, error: `That proposal is already "${row.status}".` };
  }

  const finalBrief = (editedBrief && String(editedBrief).trim()) || row.brief;
  const v = validateProject(row.project);
  if (!v.ok) return { ok: false, error: v.error };

  const agentJobs = require('./agent-jobs');
  const queued = agentJobs.enqueue({
    title: `squatch-code: ${row.project}`,
    task: finalBrief,
    why: 'Dispatched to squatch-code after Ellie approved the brief.',
    conversationId: row.conversation_id,
    messageId: row.message_id,
    source: SOURCE
  });
  if (!queued.ok) return queued;

  db.prepare(`
    UPDATE coding_jobs
    SET status = 'approved', decided_at = ?, final_brief = ?, agent_job_id = ?
    WHERE id = ?
  `).run(new Date().toISOString(), finalBrief, queued.id, id);

  opsLog(`coding job ${id.slice(0, 8)} approved -> agent job ${queued.id.slice(0, 8)} (${row.project})`);
  return { ok: true, id, agentJobId: queued.id, editedByHer: finalBrief !== row.brief };
}

function reject(id, { note = null } = {}) {
  const db = getSqliteDb();
  if (!db) return { ok: false, error: 'Database unavailable.' };
  const row = get(id);
  if (!row) return { ok: false, error: 'No such proposal.' };
  if (row.status !== 'proposed') {
    return { ok: false, error: `That proposal is already "${row.status}".` };
  }
  db.prepare("UPDATE coding_jobs SET status = 'rejected', decided_at = ?, decided_note = ? WHERE id = ?")
    .run(new Date().toISOString(), note, id);
  opsLog(`coding job ${id.slice(0, 8)} rejected (${row.project})`);
  return { ok: true, id };
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
  propose,
  approve,
  reject,
  get,
  runDispatched,
  readReport,
  validateProject,
  listProjects,
  projectsRoot,
  projectOf
};
