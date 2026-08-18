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

module.exports = router;
