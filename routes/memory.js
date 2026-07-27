/**
 * Memory Management API Routes
 * Provides endpoints for viewing, searching, adding, editing, and deleting memory facts and clusters
 */

const express = require('express');
const rateLimit = require('express-rate-limit');
const router = express.Router();
const fs = require('fs');
const path = require('path');

// Strict rate limiter for expensive LLM-driven operations
const heavyLimiter = rateLimit({
  windowMs: 10 * 60 * 1000, // 10 minutes
  max: 1,
  message: { error: 'Too many requests — this operation can only run once per 10 minutes' }
});

const db = require('../db/database');
const memoryClusters = require('../db/memory-clusters');
const factExtractor = require('../db/fact-extractor');
const questionQueue = require('../db/questions');
const identity = require('../db/identity');
const initiatives = require('../db/initiatives');
const { getConfig, getProviderInstance } = require('../db/config');

const MEMORY_DIR = path.join(__dirname, '../data/memory');

// ============ Validation Helpers ============

function isValidUUID(str) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(str);
}

function isValidDate(str) {
  return /^\d{4}-\d{2}-\d{2}$/.test(str) && !isNaN(Date.parse(str));
}

function sanitizeString(str, maxLength = 1000) {
  if (!str || typeof str !== 'string') return '';
  return str.trim().substring(0, maxLength);
}

// ============ Endpoints ============

/**
 * GET /api/memory
 * Load all memory files (MEMORY.md, USER.md, daily logs)
 */
router.get('/', (req, res) => {
  try {
    const memoryFiles = db.loadMemoryFiles();
    res.json(memoryFiles);
  } catch (error) {
    console.error('[MemoryAPI] Error loading memory files:', error.message);
    res.status(500).json({ error: 'Failed to load memory files' });
  }
});

/**
 * GET /api/memory/questions
 * List pending questions in the queue (for a future frontend to surface).
 */
router.get('/questions', (req, res) => {
  try {
    const questions = questionQueue.listPending();
    res.json({ questions });
  } catch (error) {
    console.error('[MemoryAPI] Error loading questions:', error.message);
    res.status(500).json({ error: 'Failed to load questions' });
  }
});

/**
 * GET /api/memory/daily/:date
 * Load a specific daily log by date (YYYY-MM-DD)
 */
router.get('/daily/:date', (req, res) => {
  try {
    const { date } = req.params;
    if (!isValidDate(date)) {
      return res.status(400).json({ error: 'Invalid date format. Use YYYY-MM-DD.' });
    }

    const dailyFile = path.join(MEMORY_DIR, 'daily', `${date}.md`);
    if (!fs.existsSync(dailyFile)) {
      return res.status(404).json({ error: 'Daily log not found for this date' });
    }

    const content = fs.readFileSync(dailyFile, 'utf8');
    res.json({ date, content });
  } catch (error) {
    console.error('[MemoryAPI] Error loading daily log:', error.message);
    res.status(500).json({ error: 'Failed to load daily log' });
  }
});

/**
 * GET /api/memory/ops        → list available ops-log dates (newest first)
 * GET /api/memory/ops/:date  → load one ops log (YYYY-MM-DD)
 *
 * The ops log holds operational events (errors, timeouts, liveness/circuit
 * events, heartbeat maintenance reports, background telemetry) that are
 * deliberately kept OUT of the injected chat context. Surfaced in the Thinking
 * tab so a wedged/slow brain stays observable.
 */
const OPS_DIR = path.join(MEMORY_DIR, 'ops');

router.get('/ops', (req, res) => {
  try {
    if (!fs.existsSync(OPS_DIR)) return res.json({ dates: [] });
    const dates = fs.readdirSync(OPS_DIR)
      .filter(f => /^\d{4}-\d{2}-\d{2}\.md$/.test(f))
      .map(f => f.replace(/\.md$/, ''))
      .sort()
      .reverse();
    res.json({ dates });
  } catch (error) {
    console.error('[MemoryAPI] Error listing ops logs:', error.message);
    res.status(500).json({ error: 'Failed to list ops logs' });
  }
});

router.get('/ops/:date', (req, res) => {
  try {
    const { date } = req.params;
    if (!isValidDate(date)) {
      return res.status(400).json({ error: 'Invalid date format. Use YYYY-MM-DD.' });
    }
    const opsFile = path.join(OPS_DIR, `${date}.md`);
    if (!fs.existsSync(opsFile)) {
      return res.status(404).json({ error: 'Ops log not found for this date' });
    }
    res.json({ date, content: fs.readFileSync(opsFile, 'utf8') });
  } catch (error) {
    console.error('[MemoryAPI] Error loading ops log:', error.message);
    res.status(500).json({ error: 'Failed to load ops log' });
  }
});

/**
 * GET /api/memory/clusters
 * Get clusters with member counts. Defaults to user-fact clusters so the Facts
 * and Clusters tabs don't mix in self-observation clusters (those surface in the
 * Self tab). Pass ?subject=self or ?subject=all to widen.
 */
router.get('/clusters', (req, res) => {
  try {
    const subject = req.query.subject === 'all' ? null : (req.query.subject || 'user');
    const clusters = memoryClusters.getClusters(subject);
    res.json({ clusters });
  } catch (error) {
    console.error('[MemoryAPI] Error loading clusters:', error.message);
    res.status(500).json({ error: 'Failed to load clusters' });
  }
});

/**
 * GET /api/memory/self
 * Read-only self-identity view: the injected identity block (seed + active
 * self-facts), superseded self-facts as history, and recent reflections.
 */
router.get('/self', (req, res) => {
  try {
    const memoryManager = require('../db/memory-manager');
    const block = identity.buildIdentityBlock();
    // The Self tab is the full read-only inventory, so it lists EVERY active
    // self-fact — not just block.selfFacts, which is capped at the injection
    // budget (maxSelfFacts) to keep the chat system prompt small. Using the
    // budgeted subset here would hide genuinely-active self-facts (e.g. ones
    // just reclassified from user territory) below the salience cutoff.
    const activeSelfFacts = memoryClusters.getSelfFacts({ status: 'active' });
    const supersededSelfFacts = memoryClusters.getSelfFacts({ status: 'superseded' });
    const reflections = memoryManager.getReflections(10);

    res.json({
      seed: block.seed,
      identityText: block.text,
      activeSelfFacts,
      injectedSelfFactCount: block.selfFacts.length,
      supersededSelfFacts,
      reflections
    });
  } catch (error) {
    console.error('[MemoryAPI] Error loading self view:', error.message);
    res.status(500).json({ error: 'Failed to load self view' });
  }
});

/**
 * GET /api/memory/capabilities
 * The capability manifest — SNH's machine-true registry of what it can actually
 * do. This is the "retrieved on demand" surface (full descriptions, schedules,
 * dates) behind the compact block injected into chat. Optional ?q= filters by
 * name/description. Read-only.
 */
router.get('/capabilities', (req, res) => {
  try {
    const capabilityManifest = require('../db/capability-manifest');
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    const capabilities = q ? capabilityManifest.find(q) : capabilityManifest.getAll();
    const injection = capabilityManifest.buildInjectionBlock();
    res.json({
      capabilities,
      count: capabilities.length,
      injection: { tokens: injection.tokens, text: injection.text }
    });
  } catch (error) {
    console.error('[MemoryAPI] Error loading capabilities:', error.message);
    res.status(500).json({ error: 'Failed to load capabilities' });
  }
});

/**
 * GET /api/memory/initiatives
 * List pending initiatives (highest priority first) plus the greeting threshold,
 * so the frontend bell can count those worth surfacing. Read-only view.
 */
router.get('/initiatives', (req, res) => {
  try {
    const cfg = getConfig().initiative || {};
    const threshold = Number.isFinite(cfg.greetingThreshold) ? cfg.greetingThreshold : 7;
    const pending = initiatives.listPending({ limit: 100 });
    res.json({
      initiatives: pending,
      threshold,
      aboveThreshold: pending.filter(i => i.priority >= threshold).length
    });
  } catch (error) {
    console.error('[MemoryAPI] Error loading initiatives:', error.message);
    res.status(500).json({ error: 'Failed to load initiatives' });
  }
});

/**
 * GET /api/memory/initiatives/history
 * Every initiative ever minted (newest first) — the full lifecycle
 * (pending/delivered/dismissed/expired) with timestamps and delivery channel.
 */
router.get('/initiatives/history', (req, res) => {
  try {
    const limit = Math.min(500, Math.max(1, parseInt(req.query.limit, 10) || 200));
    res.json({ initiatives: initiatives.listAll({ limit }) });
  } catch (error) {
    console.error('[MemoryAPI] Error loading initiative history:', error.message);
    res.status(500).json({ error: 'Failed to load initiative history' });
  }
});

/**
 * POST /api/memory/initiatives/:id/dismiss
 * Dismiss a pending initiative (user read it / doesn't want it raised).
 */
router.post('/initiatives/:id/dismiss', (req, res) => {
  try {
    const { id } = req.params;
    if (!isValidUUID(id)) {
      return res.status(400).json({ error: 'Invalid initiative ID' });
    }
    const ok = initiatives.dismiss(id);
    res.json({ success: ok });
  } catch (error) {
    console.error('[MemoryAPI] Error dismissing initiative:', error.message);
    res.status(500).json({ error: 'Failed to dismiss initiative' });
  }
});

/**
 * POST /api/memory/initiatives/:id/discuss
 * Start a conversation seeded with this initiative: SNH opens it by raising the
 * item naturally, the initiative leaves the pending pool (marked delivered with
 * the new conversation id), and the frontend navigates to it so the user can reply.
 */
router.post('/initiatives/:id/discuss', async (req, res) => {
  try {
    const { id } = req.params;
    if (!isValidUUID(id)) {
      return res.status(400).json({ error: 'Invalid initiative ID' });
    }
    const initiativeEngine = require('../db/initiative-engine');
    const result = await initiativeEngine.startDiscussion(id);
    if (result.error || !result.conversationId) {
      return res.status(400).json({ error: result.error || 'Could not start discussion' });
    }
    res.json({ success: true, conversationId: result.conversationId });
  } catch (error) {
    console.error('[MemoryAPI] Error starting initiative discussion:', error.message);
    res.status(500).json({ error: 'Failed to start discussion' });
  }
});

// ============ activity: everything scheduled in SNH ============

/**
 * GET /api/memory/activity
 *
 * Read-only model of every recurring process. Two categories, kept apart on
 * purpose:
 *   ACTIVE — the two in-process timers. Everything else people call "scheduled"
 *            is a STEP inside the heartbeat pass, so sub-tasks are nested under
 *            it rather than listed as peers.
 *   INERT  — cron_jobs rows. Nothing executes them; there is no scheduler.
 *
 * Reads real state (systemd process start, the run-history tables) rather than
 * keeping a parallel record. Where a process leaves no trace, it says so
 * instead of showing a blank.
 */
router.get('/activity', (req, res) => {
  try {
    const memoryManager = require('../db/memory-manager');
    const cronJobs = require('../db/cron-jobs');
    const { getSqliteDb } = require('../db/database');
    const cfg = getConfig();
    const db = getSqliteDb();

    // --- timer phase anchor -------------------------------------------------
    // Both timers are created at process start, so the whole cadence is
    // anchored there and NOT to the wall clock: every deploy re-phases them.
    // Surfaced explicitly so a shifted schedule after a restart reads as
    // expected rather than as a bug.
    const uptimeMs = Math.round(process.uptime() * 1000);
    const processStartedAt = new Date(Date.now() - uptimeMs).toISOString();

    const hbIntervalMs = (cfg.heartbeat?.intervalHours ?? 2) * 3600_000;
    const hbWarmupMs = (cfg.heartbeat?.warmupMinutes ?? 5) * 60_000;
    const lpIntervalMs = Math.max(1, cfg.livenessProbe?.intervalMinutes ?? 5) * 60_000;

    // nth firing strictly after now, counting from the anchor.
    const nextFrom = (anchorMs, offsetMs, intervalMs) => {
      const first = anchorMs + offsetMs;
      if (Date.now() < first) return new Date(first).toISOString();
      const n = Math.floor((Date.now() - first) / intervalMs) + 1;
      return new Date(first + n * intervalMs).toISOString();
    };
    const startMs = Date.now() - uptimeMs;

    // --- heartbeat ----------------------------------------------------------
    const hbRows = db
      ? db.prepare('SELECT * FROM heartbeat_reports ORDER BY datetime(created_at) DESC LIMIT 40').all()
      : [];

    // Pass duration is BIMODAL (~55s vs ~1060s observed), and a long row next to
    // a short one has to read as a different code path rather than a fault.
    //
    // The cluster-audit pipeline is NOT the discriminator: it engages on every
    // recorded pass and costs a steady 13-36s. What actually drives duration is
    // the cross-link audit's LLM workload — pairs whose content hash changed and
    // must be re-judged. Idle memory = every pair is a cache hit = fast pass.
    // Link add/update/remove counts are not a usable proxy (a 260.8s pass changed
    // zero links), so passes recorded before pairsJudged existed say so plainly
    // instead of guessing.
    const hbHistory = hbRows.map(r => {
      let parsed = {};
      try { parsed = JSON.parse(r.report_json || '{}'); } catch (e) { /* keep {} */ }
      const judged = parsed.pairsJudged;
      const known = Number.isFinite(judged);
      let pathLabel, pathKind;
      if (!known) {
        pathLabel = 'cross-link workload not recorded for this pass';
        pathKind = 'unknown';
      } else if (judged === 0) {
        pathLabel = `light pass — no cluster pairs needed re-judging (${parsed.pairsReused ?? 0} cached)`;
        pathKind = 'light';
      } else {
        pathLabel = `heavy pass — ${judged} cluster pair${judged === 1 ? '' : 's'} re-judged by the model` +
                    (Number.isFinite(parsed.pairsReused) ? ` (${parsed.pairsReused} cached)` : '');
        pathKind = 'heavy';
      }
      return {
        at: r.created_at,
        status: r.status || 'ok',
        statusReason: r.status_reason || null,
        durationMs: r.duration_ms ?? null,
        duration: r.duration || null,
        pairsJudged: known ? judged : null,
        pairsReused: Number.isFinite(parsed.pairsReused) ? parsed.pairsReused : null,
        pathKind,
        pathLabel,
        clusterPipelineEngaged: (r.clusters_audited || 0) > 0,
        clustersAudited: r.clusters_audited || 0,
        clustersSplit: r.clusters_split || 0,
        linksAdded: r.links_added || 0,
        linksUpdated: r.links_updated || 0,
        linksRemoved: r.links_removed || 0,
        anomalyCount: r.anomaly_count || 0,
        // steps only exist on passes recorded after the 2026-07-26 instrumentation
        steps: Array.isArray(parsed.steps) ? parsed.steps : null
      };
    });

    const lastHb = hbHistory[0] || null;
    const latestSteps = (hbHistory.find(h => h.steps && h.steps.length) || {}).steps || null;
    const stepByName = {};
    for (const s of latestSteps || []) stepByName[s.name] = s;

    // Sub-tasks, in the order runMaintenance executes them. These are STEPS in
    // one serial pass — not independently scheduled — hence nesting.
    const sub = (key, name, description, gate, opts = {}) => {
      const s = stepByName[key];
      return {
        key, name, description, gate,
        lastRun: s ? { at: lastHb?.at || null, ok: s.ok, durationMs: s.ms, result: s.result ?? null } : null,
        noHistory: !s && !opts.derived,
        noHistoryReason: !s && !opts.derived
          ? (latestSteps ? 'not reached on the most recent pass' : 'per-step timing recorded only from 2026-07-26')
          : null,
        ...opts
      };
    };

    const subTasks = [
      sub('preflight', 'Brain preflight probe',
          'Probes the model engine up to 3 times before committing to a full pass; aborts the cycle if it is unreachable.',
          'always — aborts the pass on failure',
          { derived: true,
            lastRun: lastHb ? { at: lastHb.at, ok: lastHb.status === 'ok', durationMs: null,
                                result: lastHb.status === 'ok' ? 'engine answered' : lastHb.statusReason } : null }),
      sub('clusterAudit', 'Cluster coherence audit + splits',
          'Asks the model whether each oversized cluster is really one topic, and splits the ones that are not.',
          'only when a cluster exceeds memory.maxFactsPerCluster',
          { derived: true,
            isClusterPipeline: true,
            lastRun: lastHb ? { at: lastHb.at, ok: lastHb.status === 'ok', durationMs: null,
                                result: lastHb.clusterPipeline
                                  ? `${lastHb.clustersAudited} audited, ${lastHb.clustersSplit} split`
                                  : 'skipped — no oversized clusters' } : null }),
      sub('crossLinks', 'Cross-link audit',
          'Scores how related each pair of clusters is and maintains the links between them. Pairs whose content has not changed are served from a cached judgment; only changed pairs go to the model. This step is what makes a pass fast or slow.',
          'always — but only re-judges pairs whose content changed',
          { derived: true,
            isDurationDriver: true,
            lastRun: lastHb ? { at: lastHb.at, ok: lastHb.status === 'ok', durationMs: null,
                                result: (lastHb.pairsJudged != null
                                  ? `${lastHb.pairsJudged} pair(s) re-judged, ${lastHb.pairsReused ?? 0} cached · `
                                  : '') + `links +${lastHb.linksAdded} / ~${lastHb.linksUpdated} / -${lastHb.linksRemoved}` } : null }),
      sub('mergeByName', 'Merge duplicate-name clusters',
          'Collapses clusters that ended up sharing a name.', 'always'),
      sub('cleanupFacts', 'Fact cleanup',
          'Dedups, rewords and merges facts in long-term memory.', 'always'),
      sub('summarizeDailyLogs', 'Daily log archival',
          'Archives daily logs older than the retention window, keeping a digest.', 'always'),
      sub('sweepPendingQuestions', 'Pending-question sweep',
          'Retires queued questions that memory can already answer.', 'always'),
      sub('reflection', 'Reflection',
          'SNH reviews the day\'s conversations and records observations about itself.',
          'only if there are new conversations'),
      sub('selfCoherenceAudit', 'Self-coherence audit',
          'Tests stored self-claims against how it actually behaved and raises gaps for approval. Never edits identity.',
          'at most once per audit.cadenceDays (currently ' + (cfg.audit?.cadenceDays ?? 1) + '/day)'),
      sub('capabilityDrift', 'Capability drift check',
          'Probes the services behind config-gated organs and reconciles the capability manifest against the live tool registry. Any disagreement is raised in the bell, because the manifest is authoritative for capability questions — a stale entry makes SNH deny something it can do, or offer something that is down.',
          'always'),
      sub('initiativeLayer', 'Initiative layer',
          'Turns findings into candidate initiatives, re-scores them, and may reach out once.',
          'quiet hours ' + (cfg.initiative?.quietHours?.start ?? 22) + ':00–' + (cfg.initiative?.quietHours?.end ?? 8) + ':00, max ' + (cfg.initiative?.maxUnpromptedPerDay ?? 1) + ' unprompted/day')
    ];

    // --- liveness -----------------------------------------------------------
    const lpRows = memoryManager.getLivenessProbes ? memoryManager.getLivenessProbes(60) : [];
    const lpHistory = lpRows.map(r => ({
      at: r.created_at, status: r.ok ? 'ok' : 'failed',
      durationMs: r.latency_ms ?? null, statusReason: r.error || null
    }));

    const active = [
      {
        id: 'heartbeat',
        name: 'Heartbeat maintenance pass',
        description: 'One serial pass that maintains memory: audits and splits clusters, maintains links, cleans up facts, archives logs, reflects, audits its own self-claims, and raises initiatives.',
        mechanism: 'node setInterval (in-process, db/memory-manager.js:1862)',
        schedule: `every ${cfg.heartbeat?.intervalHours ?? 2} hours (first run ${cfg.heartbeat?.warmupMinutes ?? 5} min after start)`,
        provenance: 'system',
        enabled: cfg.heartbeat?.enabled !== false,
        lastRun: lastHb,
        nextRun: cfg.heartbeat?.enabled !== false ? nextFrom(startMs, hbWarmupMs, hbIntervalMs) : null,
        history: hbHistory,
        subTasks
      },
      {
        id: 'liveness',
        name: 'Brain liveness probe',
        description: 'A tiny completion against the model engine to catch a wedged brain within minutes. Its result also feeds the watchdog.',
        mechanism: 'node setInterval (in-process, db/memory-manager.js:1892)',
        schedule: `every ${cfg.livenessProbe?.intervalMinutes ?? 5} minutes (timeout ${cfg.livenessProbe?.timeoutMs ?? 8000}ms)`,
        provenance: 'system',
        enabled: cfg.livenessProbe?.enabled !== false,
        lastRun: lpHistory[0] || null,
        nextRun: cfg.livenessProbe?.enabled !== false ? nextFrom(startMs, lpIntervalMs, lpIntervalMs) : null,
        history: lpHistory,
        retentionNote: `kept ${cfg.livenessProbe?.retentionDays ?? 14} days`,
        subTasks: []
      }
    ];

    // --- inert --------------------------------------------------------------
    const jobs = cronJobs.listKidCreated({ limit: 100 });

    // --- known gaps ---------------------------------------------------------
    // Things that run but leave no queryable trace. Listed rather than hidden.
    const gaps = [
      { name: 'Brain watchdog', what: 'Restarts the engine container after repeated probe failures.',
        why: 'Writes prose to the ops log only — no table, so runs and restarts cannot be listed here.',
        mechanism: 'reacts to liveness probe results (not independently scheduled)' },
      { name: 'Agent pool passes', what: 'Throttled queue that background LLM work runs through.',
        why: 'Each pass writes one unstructured ops-log line; counts are not queryable.',
        mechanism: 'used by the heartbeat (not independently scheduled)' },
      { name: 'Memory flush', what: 'Condenses a conversation when it approaches the context limit.',
        why: 'Leaves no run record. Also not on a timer — it fires per chat message.',
        mechanism: 'triggered per message (not scheduled)' }
    ];

    res.json({
      anchor: {
        processStartedAt, uptimeMs,
        note: 'Both timers are created at process start, so their cadence is anchored to it rather than to the wall clock. Restarting SNH re-phases every schedule below.'
      },
      active,
      inert: { jobs, caps: cronJobs.capStatus() },
      gaps
    });
  } catch (error) {
    console.error('[MemoryAPI] Error building activity view:', error.message);
    res.status(500).json({ error: 'Failed to build activity view' });
  }
});

// ============ cron proposals (create_cron_job, propose-only) ============

/**
 * GET /api/memory/cron
 * Every cron job the entity proposed, with current cap usage. Read-only.
 */
router.get('/cron', (req, res) => {
  try {
    const cronJobs = require('../db/cron-jobs');
    const status = typeof req.query.status === 'string' ? req.query.status : null;
    const allowed = new Set(['proposed', 'approved', 'rejected', 'reverted']);
    res.json({
      jobs: cronJobs.listKidCreated({ status: allowed.has(status) ? status : null }),
      caps: cronJobs.capStatus()
    });
  } catch (error) {
    console.error('[MemoryAPI] Error listing cron proposals:', error.message);
    res.status(500).json({ error: 'Failed to list cron proposals' });
  }
});

/**
 * POST /api/memory/cron/:id/approve
 * Approve a proposal. Records the job — nothing schedules or runs it.
 */
router.post('/cron/:id/approve', (req, res) => {
  try {
    const { id } = req.params;
    if (!isValidUUID(id)) return res.status(400).json({ error: 'Invalid proposal ID' });
    const cronJobs = require('../db/cron-jobs');
    const note = typeof req.body?.note === 'string' ? req.body.note.slice(0, 500) : null;
    const result = cronJobs.approve(id, { note });
    if (!result.ok) return res.status(400).json({ error: result.error });
    res.json({ success: true, job: result.job });
  } catch (error) {
    console.error('[MemoryAPI] Error approving cron proposal:', error.message);
    res.status(500).json({ error: 'Failed to approve proposal' });
  }
});

/**
 * POST /api/memory/cron/:id/reject
 * Reject a proposal. Writes a daily-log line so the entity learns the outcome.
 */
router.post('/cron/:id/reject', (req, res) => {
  try {
    const { id } = req.params;
    if (!isValidUUID(id)) return res.status(400).json({ error: 'Invalid proposal ID' });
    const cronJobs = require('../db/cron-jobs');
    const note = typeof req.body?.note === 'string' ? req.body.note.slice(0, 500) : null;
    const result = cronJobs.reject(id, { note });
    if (!result.ok) return res.status(400).json({ error: result.error });
    res.json({ success: true, job: result.job });
  } catch (error) {
    console.error('[MemoryAPI] Error rejecting cron proposal:', error.message);
    res.status(500).json({ error: 'Failed to reject proposal' });
  }
});

/**
 * POST /api/memory/cron/revert-all
 * Bulk revert every approved kid-created job. This is what the provenance tag
 * is for. Rows are kept and marked 'reverted' — supersede, never delete.
 */
router.post('/cron/revert-all', (req, res) => {
  try {
    const cronJobs = require('../db/cron-jobs');
    const note = typeof req.body?.note === 'string' ? req.body.note.slice(0, 500) : 'bulk revert';
    res.json({ success: true, ...cronJobs.revertAllKidCreated({ note }) });
  } catch (error) {
    console.error('[MemoryAPI] Error reverting cron jobs:', error.message);
    res.status(500).json({ error: 'Failed to revert cron jobs' });
  }
});

/**
 * GET /api/memory/tool-calls
 * Every tool call the entity made and what came of it — including calls refused
 * by a rate cap. Operational telemetry for the Thinking tab; never injected.
 */
router.get('/tool-calls', (req, res) => {
  try {
    const cronJobs = require('../db/cron-jobs');
    const limit = Math.min(500, Math.max(1, parseInt(req.query.limit, 10) || 100));
    res.json({ calls: cronJobs.listToolCalls({ limit }) });
  } catch (error) {
    console.error('[MemoryAPI] Error listing tool calls:', error.message);
    res.status(500).json({ error: 'Failed to list tool calls' });
  }
});

/**
 * GET /api/memory/followup-traces
 * Recent conversation-followup traces (newest first): what each reflection cycle
 * reviewed, the older memory it pulled in, the candidates it weighed, and what it
 * generated or skipped with reasoning. Read-only; consumed by the Self/Initiative UI.
 */
router.get('/followup-traces', (req, res) => {
  try {
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit, 10) || 20));
    const traces = initiatives.listFollowupTraces({ limit });
    res.json({ traces });
  } catch (error) {
    console.error('[MemoryAPI] Error loading followup traces:', error.message);
    res.status(500).json({ error: 'Failed to load followup traces' });
  }
});

/**
 * GET /api/memory/graph
 * Build the Memory Map graph (nodes + edges) from existing SQLite data in a
 * small, fixed number of query passes — no per-node LLM calls, pure data.
 *
 * Nodes:
 *   - clusters (memory_clusters), tagged by subject (user | self)
 *   - facts (cluster_members), with salience, status, subject, superseded_by,
 *     and a pendingQuestions count so the frontend can ring them
 * Edges:
 *   - membership: fact -> its cluster (built client-side from clusterId)
 *   - supersede:  old fact -> the fact that replaced it (directed, belief history)
 *   - cluster-link: cluster <-> cluster association (cluster_links, weighted)
 *
 * Response also includes per-cluster counts so the frontend can collapse large
 * clusters to a hub and expand members on demand.
 */
router.get('/graph', (req, res) => {
  try {
    const sqliteDb = db.getSqliteDb();
    if (!sqliteDb) return res.status(503).json({ error: 'Database not ready' });

    const clusterRows = sqliteDb.prepare(
      'SELECT id, name, subject, description FROM memory_clusters'
    ).all();

    const memberRows = sqliteDb.prepare(`
      SELECT id, cluster_id, content, salience, importance, status, subject,
             superseded_by, source, created_at, updated_at
      FROM cluster_members
    `).all();

    // Pending questions attached to a specific fact — one pass, aggregated in JS.
    const questionRows = sqliteDb.prepare(
      "SELECT member_id FROM questions WHERE status = 'pending' AND member_id IS NOT NULL"
    ).all();
    const pendingByMember = new Map();
    for (const q of questionRows) {
      pendingByMember.set(q.member_id, (pendingByMember.get(q.member_id) || 0) + 1);
    }

    const linkRows = sqliteDb.prepare(
      'SELECT cluster_a, cluster_b, strength FROM cluster_links'
    ).all();

    // Per-cluster tallies (for hub sizing + collapse decisions).
    const counts = new Map(); // clusterId -> { total, active, superseded }
    const bump = (cid, key) => {
      if (!cid) return;
      let c = counts.get(cid);
      if (!c) counts.set(cid, (c = { total: 0, active: 0, superseded: 0 }));
      c.total++;
      c[key]++;
    };

    const clusterIds = new Set(clusterRows.map(c => c.id));

    const nodes = memberRows.map(m => {
      const superseded = m.status === 'superseded' || !!m.superseded_by;
      bump(m.cluster_id, superseded ? 'superseded' : 'active');
      const pendingQuestions = pendingByMember.get(m.id) || 0;
      return {
        id: m.id,
        clusterId: clusterIds.has(m.cluster_id) ? m.cluster_id : null,
        content: m.content,
        salience: Number.isFinite(m.salience) ? m.salience : (Math.round((m.importance || 0.5) * 10) || 5),
        status: m.status || 'active',
        subject: m.subject || 'user',
        supersededBy: m.superseded_by || null,
        source: m.source || null,
        createdAt: m.created_at || null,
        updatedAt: m.updated_at || null,
        pendingQuestions
      };
    });

    const memberIds = new Set(memberRows.map(m => m.id));

    // Directed supersede edges — only when the replacement fact still exists.
    const supersedeEdges = [];
    for (const m of memberRows) {
      if (m.superseded_by && memberIds.has(m.superseded_by)) {
        supersedeEdges.push({ source: m.id, target: m.superseded_by, type: 'supersede' });
      }
    }

    // Undirected cluster associations — only between clusters we render.
    const linkEdges = [];
    for (const l of linkRows) {
      if (clusterIds.has(l.cluster_a) && clusterIds.has(l.cluster_b)) {
        linkEdges.push({
          source: l.cluster_a,
          target: l.cluster_b,
          type: 'cluster-link',
          strength: l.strength || 1
        });
      }
    }

    const clusters = clusterRows.map(c => {
      const cc = counts.get(c.id) || { total: 0, active: 0, superseded: 0 };
      return {
        id: c.id,
        name: c.name,
        subject: c.subject || 'user',
        description: c.description || null,
        total: cc.total,
        active: cc.active,
        superseded: cc.superseded
      };
    });

    res.json({
      clusters,
      nodes,
      edges: [...supersedeEdges, ...linkEdges],
      stats: {
        clusters: clusters.length,
        facts: nodes.length,
        superseded: nodes.filter(n => n.status === 'superseded' || n.supersededBy).length,
        pendingQuestions: questionRows.length
      }
    });
  } catch (error) {
    console.error('[MemoryAPI] Error building memory graph:', error.message);
    res.status(500).json({ error: 'Failed to build memory graph' });
  }
});

/**
 * POST /api/memory/reflect
 * Manually trigger a reflection pass (still requires new conversations).
 */
router.post('/reflect', heavyLimiter, async (req, res) => {
  try {
    const memoryManager = require('../db/memory-manager');
    const result = await memoryManager.runReflection({ force: true });
    res.json(result);
  } catch (error) {
    console.error('[MemoryAPI] Error running reflection:', error.message);
    res.status(500).json({ error: 'Failed to run reflection' });
  }
});

/**
 * GET /api/memory/thinking
 * Read-only "Thinking" feed: one entry per background cycle, newest first.
 *   - reflection cycles: what conversations were reviewed, self-observations
 *     noticed, follow-up candidates considered, and what was queued or skipped
 *     and why (fuses reflections.jsonl with the followup_traces table).
 *   - heartbeat passes: cluster-audit / link stats + anomalies + duration.
 */
router.get('/thinking', (req, res) => {
  try {
    const memoryManager = require('../db/memory-manager');
    const limit = Math.min(200, Math.max(1, parseInt(req.query.limit, 10) || 40));

    const reflections = memoryManager.getReflections(80);
    const traces = initiatives.listFollowupTraces({ limit: 80 });
    const heartbeats = memoryManager.getHeartbeatReports(80);

    const ms = (iso) => {
      if (!iso) return 0;
      const d = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z');
      return isNaN(d.getTime()) ? 0 : d.getTime();
    };
    const FUSE_WINDOW = 5 * 60 * 1000; // reflection + its follow-up trace share a cycle

    // Fuse each reflection with the nearest unused follow-up trace (same cycle).
    const usedTrace = new Set();
    const entries = [];
    for (const r of reflections) {
      let best = null, bestDelta = Infinity;
      for (let i = 0; i < traces.length; i++) {
        if (usedTrace.has(i)) continue;
        const delta = Math.abs(ms(traces[i].at) - ms(r.at));
        if (delta < bestDelta) { bestDelta = delta; best = i; }
      }
      const followup = (best !== null && bestDelta <= FUSE_WINDOW) ? traces[best] : null;
      if (followup) usedTrace.add(best);
      entries.push({
        kind: 'reflection',
        at: r.at,
        messageCount: r.messageCount || 0,
        conversationCount: r.conversationCount || 0,
        conversationsReviewed: followup ? followup.conversationsReviewed : [],
        observations: r.observations || [],
        stored: r.stored ?? null,
        superseded: r.superseded ?? null,
        followup: followup ? {
          candidates: followup.candidates || [],
          relatedClusters: followup.relatedClusters || [],
          generated: followup.generated,
          skipped: followup.skipped,
          reasoning: followup.reasoning,
          initiativeId: followup.initiativeId
        } : null
      });
    }
    // Orphan follow-up traces (a trace with no matching reflection record).
    traces.forEach((t, i) => {
      if (usedTrace.has(i)) return;
      entries.push({
        kind: 'reflection',
        at: t.at,
        messageCount: t.messageCount || 0,
        conversationCount: (t.conversationsReviewed || []).length,
        conversationsReviewed: t.conversationsReviewed || [],
        observations: [],
        stored: null,
        superseded: null,
        followup: {
          candidates: t.candidates || [],
          relatedClusters: t.relatedClusters || [],
          generated: t.generated,
          skipped: t.skipped,
          reasoning: t.reasoning,
          initiativeId: t.initiativeId
        }
      });
    });
    // Heartbeat passes.
    for (const h of heartbeats) {
      entries.push({ kind: 'heartbeat', ...h });
    }

    entries.sort((a, b) => ms(b.at) - ms(a.at));
    res.json({ entries: entries.slice(0, limit) });
  } catch (error) {
    console.error('[MemoryAPI] Error loading thinking feed:', error.message);
    res.status(500).json({ error: 'Failed to load thinking feed' });
  }
});

/**
 * GET /api/memory/clusters/:id
 * Get a specific cluster with all members and linked clusters
 */
router.get('/clusters/:id', (req, res) => {
  try {
    const { id } = req.params;
    if (!isValidUUID(id)) {
      return res.status(400).json({ error: 'Invalid cluster ID' });
    }

    const cluster = memoryClusters.getCluster(id);
    if (!cluster) {
      return res.status(404).json({ error: 'Cluster not found' });
    }

    res.json(cluster);
  } catch (error) {
    console.error('[MemoryAPI] Error loading cluster:', error.message);
    res.status(500).json({ error: 'Failed to load cluster' });
  }
});

/**
 * POST /api/memory/search
 * Search memory using hybrid search (vector + BM25)
 */
router.post('/search', async (req, res) => {
  try {
    const { query, limit } = req.body;
    const searchQuery = sanitizeString(query, 500);

    if (!searchQuery) {
      return res.status(400).json({ error: 'Search query is required' });
    }

    const searchLimit = Math.min(Math.max(parseInt(limit) || 10, 1), 50);
    const results = await db.hybridSearch(searchQuery, '', searchLimit);

    res.json({ query: searchQuery, results });
  } catch (error) {
    console.error('[MemoryAPI] Error searching memory:', error.message);
    res.status(500).json({ error: 'Failed to search memory' });
  }
});

/**
 * POST /api/memory/add
 * Add a new fact to memory (cluster assignment + MEMORY.md append)
 */
router.post('/add', async (req, res) => {
  try {
    const { fact } = req.body;
    const cleanFact = sanitizeString(fact, 2000);

    if (!cleanFact) {
      return res.status(400).json({ error: 'Fact text is required' });
    }

    // Assign to cluster using config-driven embedding provider
    const config = getConfig();
    const embeddingProvider = config.models.embedding.provider;
    const embeddingModel = config.models.embedding.model;
    const embInst = getProviderInstance(embeddingProvider, config.models.embedding.instance);
    const embeddingHost = embInst ? embInst.host : 'http://localhost:11434';
    const clusterResult = await memoryClusters.assignToCluster(
      cleanFact, embeddingProvider, embeddingModel, '', embeddingHost, 'manual'
    );

    // Append to MEMORY.md
    const memoryFile = path.join(MEMORY_DIR, 'MEMORY.md');
    await factExtractor.appendToMemory([cleanFact], memoryFile);

    res.status(201).json({
      fact: cleanFact,
      clusterId: clusterResult.clusterId,
      clusterName: clusterResult.clusterName,
      isNewCluster: clusterResult.isNew
    });
  } catch (error) {
    console.error('[MemoryAPI] Error adding fact:', error.message);
    res.status(500).json({ error: 'Failed to add fact' });
  }
});

/**
 * PUT /api/memory/edit
 * Edit an existing fact (update content in cluster_members + re-embed in LanceDB)
 */
router.put('/edit', async (req, res) => {
  try {
    const { memberId, content } = req.body;
    const cleanContent = sanitizeString(content, 2000);

    if (!memberId || !isValidUUID(memberId)) {
      return res.status(400).json({ error: 'Valid member ID is required' });
    }
    if (!cleanContent) {
      return res.status(400).json({ error: 'Content is required' });
    }

    const sqliteDb = db.getSqliteDb();
    if (!sqliteDb) {
      return res.status(500).json({ error: 'Database not available' });
    }

    // Verify member exists
    const member = sqliteDb.prepare('SELECT * FROM cluster_members WHERE id = ?').get(memberId);
    if (!member) {
      return res.status(404).json({ error: 'Fact not found' });
    }

    // Update content in SQLite (bump updated_at to reflect the edit)
    sqliteDb.prepare('UPDATE cluster_members SET content = ?, updated_at = ? WHERE id = ?')
      .run(cleanContent, new Date().toISOString(), memberId);

    // Re-embed in LanceDB
    const clusterTable = await db.getClusterEmbeddingsTable();
    if (clusterTable) {
      try {
        await clusterTable.delete(`member_id = "${memberId}"`);
        const embedding = await memoryClusters.generateEmbedding(cleanContent);
        if (embedding) {
          const { randomUUID } = require('crypto');
          await clusterTable.add([{
            id: randomUUID(),
            member_id: memberId,
            cluster_id: member.cluster_id,
            content: cleanContent,
            vector: Array.from(embedding)
          }]);
        }
      } catch (lanceErr) {
        console.error('[MemoryAPI] LanceDB re-embed error:', lanceErr.message);
      }
    }

    res.json({ success: true, memberId, content: cleanContent });
  } catch (error) {
    console.error('[MemoryAPI] Error editing fact:', error.message);
    res.status(500).json({ error: 'Failed to edit fact' });
  }
});

/**
 * DELETE /api/memory/fact/:id
 * Delete a fact from cluster_members and LanceDB, clean up empty cluster
 */
router.delete('/fact/:id', async (req, res) => {
  try {
    const { id } = req.params;
    if (!isValidUUID(id)) {
      return res.status(400).json({ error: 'Valid fact ID is required' });
    }

    const sqliteDb = db.getSqliteDb();
    if (!sqliteDb) {
      return res.status(500).json({ error: 'Database not available' });
    }

    // Get member info before deleting
    const member = sqliteDb.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
    if (!member) {
      return res.status(404).json({ error: 'Fact not found' });
    }

    // Delete from SQLite
    sqliteDb.prepare('DELETE FROM cluster_members WHERE id = ?').run(id);

    // Delete from LanceDB
    const clusterTable = await db.getClusterEmbeddingsTable();
    if (clusterTable) {
      try {
        await clusterTable.delete(`member_id = "${id}"`);
      } catch (lanceErr) {
        console.error('[MemoryAPI] LanceDB delete error:', lanceErr.message);
      }
    }

    // Check if cluster is now empty and clean up
    const remainingMembers = sqliteDb.prepare(
      'SELECT COUNT(*) as count FROM cluster_members WHERE cluster_id = ?'
    ).get(member.cluster_id);

    if (remainingMembers.count === 0) {
      sqliteDb.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?')
        .run(member.cluster_id, member.cluster_id);
      sqliteDb.prepare('DELETE FROM memory_clusters WHERE id = ?')
        .run(member.cluster_id);
      console.log(`[MemoryAPI] Cleaned up empty cluster ${member.cluster_id}`);
    }

    res.json({ success: true, deletedId: id });
  } catch (error) {
    console.error('[MemoryAPI] Error deleting fact:', error.message);
    res.status(500).json({ error: 'Failed to delete fact' });
  }
});

/**
 * POST /api/memory/maintain
 * Manually trigger a full maintenance cycle
 */
router.post('/maintain', heavyLimiter, async (req, res) => {
  try {
    const memoryManager = require('../db/memory-manager');
    const result = await memoryManager.runMaintenance();
    res.json(result);
  } catch (error) {
    console.error('[MemoryAPI] Error running maintenance:', error.message);
    res.status(500).json({ error: 'Failed to run maintenance' });
  }
});

/**
 * POST /api/memory/rebuild
 * Trigger a full intelligent cluster rebuild
 */
router.post('/rebuild', heavyLimiter, async (req, res) => {
  try {
    const memoryManager = require('../db/memory-manager');
    const result = await memoryManager.rebuildClusters();
    res.json(result);
  } catch (error) {
    console.error('[MemoryAPI] Error rebuilding clusters:', error.message);
    res.status(500).json({ error: 'Failed to rebuild clusters' });
  }
});

module.exports = router;
