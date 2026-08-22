/**
 * The jobs API — the ROBOT channel's read side.
 *
 * Its own router at /api/jobs rather than a corner of /api/memory, because the
 * separation is the feature. Job results are not initiatives, do not live in the
 * initiative table, and are not served by the endpoint the bell polls. Anything
 * that blurred the two here would be the first place the blur got in.
 *
 * READ, PLUS TWO SMALL HUMAN ACTIONS. Marking a result read and cancelling a job
 * that has not started are Ellie's, and neither is reachable from a conversation
 * — there is no tool for either, on purpose. Starting a job is the model's, via
 * start_background_job, and it goes through the queue's own caps rather than
 * through here: an HTTP route that could start jobs would be a second doorway
 * with a second set of limits.
 */

const express = require('express');
const rateLimit = require('express-rate-limit');
const router = express.Router();

const fs = require('fs');
const path = require('path');
const agentJobs = require('../db/agent-jobs');

/** Deliberate human actions: cheap, but not something to script in bulk. */
const actionLimiter = rateLimit({
  windowMs: 5 * 60 * 1000,
  max: 60,
  message: { error: 'Too many requests — slow down' }
});

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * GET /api/jobs
 * The panel feed: handed-off jobs and scheduled-job results, newest first, plus
 * the counts the badge needs. Read-only.
 *
 * `unseen` and `active` are separate numbers because they mean different things
 * to look at: unseen is something to read, active is something still happening.
 * A badge that added them together would tick up when a job STARTED, which reads
 * as "there is a result waiting" when there is not.
 */
router.get('/', (req, res) => {
  try {
    const limit = Math.min(200, Math.max(1, parseInt(req.query.limit, 10) || 50));
    const jobs = agentJobs.feed({ limit });
    res.json({ jobs, ...agentJobs.counts() });
  } catch (error) {
    console.error('[JobsAPI] Error loading jobs:', error.message);
    res.status(500).json({ error: 'Failed to load jobs' });
  }
});

/**
 * GET /api/jobs/counts
 * Just the badge numbers — what the 60s poll actually needs, without shipping
 * every result body on every tick.
 */
router.get('/counts', (req, res) => {
  try {
    res.json(agentJobs.counts());
  } catch (error) {
    console.error('[JobsAPI] Error counting jobs:', error.message);
    res.status(500).json({ error: 'Failed to count jobs' });
  }
});

/**
 * POST /api/jobs/:id/seen
 * Mark one result read. `kind` distinguishes a handed-off job from a scheduled
 * run, because the two live in different tables and share an id space only by
 * coincidence of both being UUIDs.
 */
router.post('/:id/seen', actionLimiter, (req, res) => {
  try {
    const { id } = req.params;
    if (!UUID_RE.test(id)) return res.status(400).json({ error: 'Invalid job ID' });
    const kind = req.body && req.body.kind === 'scheduled' ? 'scheduled' : 'handoff';
    const ok = kind === 'scheduled' ? agentJobs.markRunSeen(id) : agentJobs.markSeen(id);
    res.json({ success: ok });
  } catch (error) {
    console.error('[JobsAPI] Error marking job seen:', error.message);
    res.status(500).json({ error: 'Failed to mark job seen' });
  }
});

/**
 * GET /api/jobs/:id/file
 * Download what the run produced.
 *
 * ⚠ BY JOB ID, NEVER BY PATH. The route takes a UUID and looks the location up
 * in the row; there is no parameter that names a file and therefore nothing to
 * traverse out of. A `?path=` version of this endpoint would be a directory
 * traversal with a rate limit on it, and the temptation to add one — "so the
 * panel can link to older files too" — is exactly why the reason is written
 * down here rather than left as an obvious choice.
 *
 * The stored path is ours: it is written by db/job-artifacts.js and reaches the
 * database no other way. Nothing a browser sends is ever joined onto it.
 *
 * WHY THIS EXISTS AT ALL, when the file is already saved to a folder: the folder
 * is on the server and Ellie is usually not. A path in a card is a fact about a
 * machine she is not sitting at. Both halves are the feature — the file lands in
 * her documents folder AND the card downloads it.
 */
const MIME = {
  '.pdf': 'application/pdf',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.json': 'application/json',
  '.html': 'text/html; charset=utf-8'
};

router.get('/:id/file', (req, res) => {
  try {
    const { id } = req.params;
    if (!UUID_RE.test(id)) return res.status(400).json({ error: 'Invalid job ID' });

    const job = agentJobs.getJob(id);
    if (!job) return res.status(404).json({ error: 'No such job' });
    if (!job.artifact_path) {
      // The distinction matters to whoever is looking at this: a job with no
      // file is usually a short result that correctly stayed on the card, not a
      // missing file.
      return res.status(404).json({ error: job.artifact_error || 'This job did not produce a file — its result is on the card.' });
    }

    let stat;
    try {
      stat = fs.statSync(job.artifact_path);
    } catch {
      // Deleted, moved, or on a drive that is not mounted. Said plainly, and
      // NOT treated as data loss: the result itself is still in the row.
      return res.status(410).json({
        error: 'The file is no longer where it was saved. The result itself is still on the card.'
      });
    }
    if (!stat.isFile()) return res.status(410).json({ error: 'That is not a file any more.' });

    const name = job.artifact_name || path.basename(job.artifact_path);
    const type = MIME[path.extname(name).toLowerCase()] || 'application/octet-stream';
    res.setHeader('Content-Type', type);
    res.setHeader('Content-Length', stat.size);
    // `attachment` rather than `inline`: this is a download link and it should
    // download, on every browser, rather than opening a PDF viewer on some and
    // a save dialog on others. The filename is quoted and stripped of quotes and
    // control characters — it comes from a job title the model wrote.
    const safeName = name.replace(/["\\\r\n]/g, '');
    res.setHeader('Content-Disposition', `attachment; filename="${safeName}"`);
    // It is a generated document, not a page: nothing here should be sniffed,
    // framed, or cached by anything in between.
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Cache-Control', 'private, no-store');

    fs.createReadStream(job.artifact_path)
      .on('error', (err) => {
        console.error('[JobsAPI] Error streaming job file:', err.message);
        if (!res.headersSent) res.status(500).json({ error: 'Failed to read the file' });
        else res.destroy();
      })
      .pipe(res);
  } catch (error) {
    console.error('[JobsAPI] Error serving job file:', error.message);
    if (!res.headersSent) res.status(500).json({ error: 'Failed to serve the file' });
  }
});

/**
 * POST /api/jobs/:id/cancel
 * Cancel a job that has not started. A running job is refused WITH THE REASON —
 * an in-flight model call cannot be stopped cleanly, and a button that claimed
 * otherwise would leave the run going while the panel said it had stopped.
 */
router.post('/:id/cancel', actionLimiter, (req, res) => {
  try {
    const { id } = req.params;
    if (!UUID_RE.test(id)) return res.status(400).json({ error: 'Invalid job ID' });
    const result = agentJobs.cancel(id);
    if (!result.ok) return res.status(400).json({ error: result.error });
    res.json({ success: true });
  } catch (error) {
    console.error('[JobsAPI] Error cancelling job:', error.message);
    res.status(500).json({ error: 'Failed to cancel job' });
  }
});

/**
 * What squatch-code is doing right now.
 *
 * Feeds the strip in the header. Returns an empty array when nothing is
 * running, so the UI renders nothing at all rather than an empty shell -
 * "when nothing is running, I see nothing" is half the requirement.
 */
router.get('/coding/active', (req, res) => {
  try {
    res.json({ jobs: require('../db/coding-jobs').running() });
  } catch (err) {
    res.json({ jobs: [], error: err.message });
  }
});

module.exports = router;
