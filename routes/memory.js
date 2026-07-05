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
    const supersededSelfFacts = memoryClusters.getSelfFacts({ status: 'superseded' });
    const reflections = memoryManager.getReflections(10);

    res.json({
      seed: block.seed,
      identityText: block.text,
      activeSelfFacts: block.selfFacts,
      supersededSelfFacts,
      reflections
    });
  } catch (error) {
    console.error('[MemoryAPI] Error loading self view:', error.message);
    res.status(500).json({ error: 'Failed to load self view' });
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
