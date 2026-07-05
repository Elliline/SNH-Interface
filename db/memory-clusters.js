const { randomUUID } = require('crypto');
const { getSqliteDb, getClusterEmbeddingsTable } = require('./database');
const { getConfig, getProviderInstance } = require('./config');

// UUID validation for safe LanceDB filter interpolation
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function safeId(id) {
  if (!UUID_RE.test(id)) throw new Error(`Invalid UUID for LanceDB filter: ${id}`);
  return id;
}

// Stop words filtered out during cluster naming
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
  'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
  'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
  'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
  'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
  'just', 'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this',
  'these', 'those', 'what', 'which', 'who', 'whom', 'its', 'his', 'her',
  'their', 'our', 'my', 'your', 'about', 'also', 'like', 'likes',
  'user', 'uses', 'using', 'runs', 'running', 'has', 'have', 'had',
  'loves', 'prefers', 'wants', 'enjoys', 'includes', 'named', 'called'
]);

/**
 * Generate embedding for text using Ollama's nomic-embed-text model
 * @param {string} text - Text to embed
 * @returns {Promise<number[]|null>} - Embedding vector or null on error
 */
async function generateEmbedding(text) {
  if (!text || typeof text !== 'string') {
    return null;
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const config = getConfig();
    const embInst = getProviderInstance(config.models.embedding.provider, config.models.embedding.instance);
    const embeddingHost = embInst ? embInst.host : 'http://localhost:11434';
    const embeddingModel = config.models.embedding.model;
    const response = await fetch(`${embeddingHost}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: embeddingModel,
        prompt: text
      }),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      console.error('[Clusters] Embedding generation failed:', response.status);
      return null;
    }

    const data = await response.json();
    if (!data.embedding || !Array.isArray(data.embedding)) {
      return null;
    }
    // Return Float32Array to match database.js format (LanceDB expects float32 precision)
    return new Float32Array(data.embedding);
  } catch (error) {
    if (error.name === 'AbortError') {
      console.error('[Clusters] Embedding generation timeout');
    } else {
      console.error('[Clusters] Embedding generation error:', error.message);
    }
    return null;
  }
}

/**
 * Calculate cosine similarity between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} - Similarity score (0-1)
 */
function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
    return 0;
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Generate a cluster name using LLM (ollama/llamacpp only)
 * @param {string} fact - The fact to generate a name for
 * @param {string} provider - Provider name
 * @param {string} model - Model name
 * @param {string} apiKey - API key (if needed)
 * @param {string} host - Host URL
 * @returns {Promise<string>} - Generated cluster name
 */
// Verbs that indicate a garbage cluster name (sentence fragment, not a category)
const REJECT_VERBS = new Set([
  'needs', 'avoid', 'having', 'doing', 'working', 'being', 'getting',
  'making', 'going', 'running', 'using', 'wants', 'takes', 'trying', 'looking'
]);

function isValidClusterName(name) {
  if (!name || name.trim().length < 3) return false;
  const words = name.trim().split(/\s+/);
  if (words.length > 4) return false;
  for (const word of words) {
    if (REJECT_VERBS.has(word.toLowerCase())) return false;
  }
  return true;
}

async function generateClusterName(fact, provider, model, apiKey, host) {
  // 1. Try curated category first — fast, deterministic, no LLM needed
  const curatedName = matchCuratedCategory(fact);
  if (curatedName) return curatedName;

  // 2. Try LLM with few-shot examples and constraints
  const prompt = `You are a category labeler. Given a fact about a user, return a short 1-3 word category name (like a folder label). Use nouns only, no verbs or sentences.

Examples:
- "User has two cats" → Pets & Animals
- "User runs an MSP business" → Business & MSP
- "User has an RTX 4090 GPU" → Hardware & Infrastructure
- "User plays Stellaris" → Gaming
- "User is building a memory system for their AI assistant" → AI & Projects

Return ONLY the category name, nothing else.

Fact: ${fact}`;

  try {
    let llmName = null;

    if (provider === 'ollama') {
      const response = await fetch(`${host}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model,
          prompt: prompt,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama request failed: ${response.status}`);
      }

      const data = await response.json();
      llmName = data.response?.trim()?.substring(0, 50);
    } else if (provider === 'llamacpp') {
      const response = await fetch(`${host}/completion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          n_predict: 20,
          temperature: 0.3,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Llama.cpp request failed: ${response.status}`);
      }

      const data = await response.json();
      llmName = data.content?.trim()?.substring(0, 50);
    }

    // 3. Post-LLM validation: reject names with >4 words or containing verbs
    if (llmName && isValidClusterName(llmName)) {
      return llmName;
    }

    // 4. Fall back to word-frequency extraction
    return extractNameFromFact(fact);
  } catch (error) {
    console.error('[Clusters] Cluster name generation error:', error.message);
    return extractNameFromFact(fact);
  }
}

/**
 * Extract a simple name from fact text (fallback)
 * Strips common prefixes, removes stop words, returns Title Case top words
 * @param {string} fact - The fact text
 * @returns {string} - Extracted name
 */
function extractNameFromFact(fact) {
  // Strip common fact prefixes
  const cleaned = fact
    .replace(/^(the\s+)?user('s)?\s+(has|is|loves|runs|uses|prefers|wants|enjoys|works|lives|owns|plays|likes)\s+/i, '')
    .replace(/^(the\s+)?user('s)?\s+/i, '')
    .replace(/^(I|My|This|That|There|The)\s+/i, '')
    .replace(/[.,!?;:].*$/, ''); // Trim from first punctuation

  // Split into words, filter stop words and short words
  const words = cleaned.split(/\s+/)
    .map(w => w.replace(/[^a-zA-Z0-9-]/g, ''))
    .filter(w => w.length > 2 && !STOP_WORDS.has(w.toLowerCase()));

  if (words.length === 0) return 'General';

  // Title case top 2-3 significant words
  const titleCase = w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
  return words.slice(0, 3).map(titleCase).join(' ').substring(0, 50);
}

// Simple plural/suffix stemming: dogs→dog, cats→cat, gaming→game, etc.
function stemWord(word) {
  const w = word.toLowerCase();
  if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'y'; // batteries→battery
  if (w.endsWith('ses') && w.length > 4) return w.slice(0, -2);       // buses→bus
  if (w.endsWith('ing') && w.length > 5) return w.slice(0, -3);       // gaming→gam → handled by map
  if (w.endsWith('tion') && w.length > 5) return w;                   // keep as-is
  if (w.endsWith('s') && !w.endsWith('ss') && w.length > 3) return w.slice(0, -1); // dogs→dog
  return w;
}

// Curated name map: if top stemmed words contain any key set → use that label
const CLUSTER_NAME_MAP = [
  { keys: ['dog', 'cat', 'pet', 'dragon', 'bearded', 'animal'], name: 'Pets & Animals' },
  { keys: ['wayne', 'eric', 'ellie', 'father', 'partner', 'family', 'wife', 'husband', 'brother', 'sister', 'kid', 'children'], name: 'People & Family' },
  { keys: ['battletech', 'mech', 'marauder', 'game', 'strategy', 'robot'], name: 'Gaming' },
  { keys: ['server', 'vram', 'gpu', 'cpu', 'ram', 'rtx', 'nvidia', 'amd', 'hardware', 'linux', 'garuda', 'strix', 'halo', 'ubiquiti', 'network', 'infrastructure'], name: 'Hardware & Infrastructure' },
  { keys: ['client', 'mettasphere', 'kaseya', 'syncro', 'msp', 'business', 'endpoint', 'autotask', 'rmm', 'psa', 'migrat', 'managed', 'service', 'provider', 'subscriptions', 'self-hosted', 'local-first'], name: 'Business & MSP', minHits: 1 },
  { keys: ['constantinople', 'opera', 'song', 'lyric', 'aria', 'arioso', 'recitative', 'hagia', 'sophia', 'chorus', 'mosaic'], name: 'Creative Projects' },
  { keys: ['story', 'fiction', 'transylvania', 'flee', 'journey', 'novel', 'young', 'woman'], name: 'Story & Fiction' },
  { keys: ['ai', 'ollama', 'llama', 'model', 'embedding', 'cluster', 'memory', 'coastal', 'squatch'], name: 'AI & Projects' },
  { keys: ['self-hosted', 'local', 'philosophy', 'build', 'prefer'], name: 'Preferences & Philosophy' },
  { keys: ['code', 'python', 'javascript', 'programming', 'software', 'docker', 'kubernetes', 'api', 'database'], name: 'Software & Dev' },
  { keys: ['music', 'band', 'guitar', 'piano', 'album'], name: 'Music' },
  { keys: ['food', 'cooking', 'homebrew', 'beer', 'recipe'], name: 'Food & Drink' },
];

/**
 * Generate a cluster name from all its members using word frequency analysis
 * with root-word deduplication and curated name map fallback
 * @param {Array} members - Array of {content} objects
 * @returns {string} - Generated cluster name
 */
function generateClusterNameFromMembers(members) {
  if (!members || members.length === 0) return 'General';

  const allText = members.map(m => m.content || m).join(' ');

  // Tokenize and stem (split on whitespace and / to handle "local/self-hosted" etc.)
  const rawWords = allText.split(/[\s/]+/)
    .map(w => w.replace(/[^a-zA-Z0-9-]/g, '').toLowerCase())
    .filter(w => w.length > 2 && !STOP_WORDS.has(w));

  // Group by stem, accumulate counts under the canonical (most frequent) form
  const stemGroups = {}; // stem → { forms: {word: count}, total: N }
  for (const word of rawWords) {
    const stem = stemWord(word);
    if (!stemGroups[stem]) stemGroups[stem] = { forms: {}, total: 0 };
    stemGroups[stem].forms[word] = (stemGroups[stem].forms[word] || 0) + 1;
    stemGroups[stem].total++;
  }

  // Build a set of all stems present for curated map matching
  const allStems = new Set(Object.keys(stemGroups));
  // Also add the raw words themselves (for partial matches like "migrat" in "migrating")
  for (const word of rawWords) allStems.add(word);

  // Try curated name map first: check if any map entry's keys overlap with our stems
  let bestMapMatch = null;
  let bestMapScore = 0;
  let bestMapMinHits = 2;
  for (const entry of CLUSTER_NAME_MAP) {
    let hits = 0;
    for (const key of entry.keys) {
      // Check exact stem match or if any stem starts with the key (for partial stems)
      for (const stem of allStems) {
        if (stem === key || stem.startsWith(key) || key.startsWith(stem)) {
          hits++;
          break;
        }
      }
    }
    const entryMinHits = entry.minHits || 2;
    if (hits > bestMapScore) {
      bestMapScore = hits;
      bestMapMatch = entry.name;
      bestMapMinHits = entryMinHits;
    }
  }

  // Use curated name if we got enough keyword hits (per-entry minHits, default 2)
  if (bestMapMatch && bestMapScore >= bestMapMinHits) {
    return bestMapMatch;
  }

  // Fall back to word frequency for novel clusters
  // Pick the most frequent form from each stem group
  const scored = Object.entries(stemGroups).map(([stem, group]) => {
    // Find the most common surface form for display
    const bestForm = Object.entries(group.forms)
      .sort((a, b) => b[1] - a[1])[0][0];
    return { word: bestForm, score: group.total };
  });

  scored.sort((a, b) => b.score - a.score);
  if (scored.length === 0) return 'General';

  // If curated map had 1 hit, use it as a prefix hint
  const titleCase = w => w.charAt(0).toUpperCase() + w.slice(1);
  const topWords = scored.slice(0, 3).map(s => titleCase(s.word));
  return topWords.join(' ').substring(0, 50);
}

/**
 * Match a single text against curated categories (for singleton merge)
 * @param {string} text - Text to match
 * @returns {string|null} - Category name or null if no match
 */
function matchCuratedCategory(text) {
  const words = text.toLowerCase().split(/[\s/]+/)
    .map(w => w.replace(/[^a-zA-Z0-9-]/g, ''))
    .filter(w => w.length > 2);

  const stems = new Set();
  for (const word of words) {
    stems.add(word);
    stems.add(stemWord(word));
  }

  let bestMatch = null;
  let bestScore = 0;
  let bestMinHits = 2;
  for (const entry of CLUSTER_NAME_MAP) {
    let hits = 0;
    for (const key of entry.keys) {
      for (const stem of stems) {
        if (stem === key || stem.startsWith(key) || key.startsWith(stem)) {
          hits++;
          break;
        }
      }
    }
    const entryMinHits = entry.minHits || 2;
    if (hits >= entryMinHits && hits > bestScore) {
      bestScore = hits;
      bestMatch = entry.name;
      bestMinHits = entryMinHits;
    }
  }

  return bestMatch;
}

/**
 * Rename all clusters using word frequency analysis of their members
 * @returns {Promise<number>} - Number of clusters renamed
 */
async function renameAllClusters() {
  try {
    const db = getSqliteDb();
    if (!db) return 0;

    const clusters = db.prepare('SELECT id, name FROM memory_clusters').all();
    let renamed = 0;

    for (const cluster of clusters) {
      const members = db.prepare(
        'SELECT content FROM cluster_members WHERE cluster_id = ?'
      ).all(cluster.id);

      if (members.length === 0) continue;

      const newName = generateClusterNameFromMembers(members);
      if (newName && newName !== cluster.name) {
        db.prepare('UPDATE memory_clusters SET name = ? WHERE id = ?')
          .run(newName, cluster.id);
        console.log(`[Clusters] Renamed "${cluster.name}" → "${newName}"`);
        renamed++;
      }
    }

    console.log(`[Clusters] Renamed ${renamed}/${clusters.length} clusters`);
    return renamed;
  } catch (error) {
    console.error('[Clusters] Error renaming clusters:', error.message);
    return 0;
  }
}

/**
 * Create or strengthen a link between two clusters
 * @param {string} clusterA - First cluster ID
 * @param {string} clusterB - Second cluster ID
 * @param {Object} db - SQLite database instance
 */
function createOrStrengthenLink(clusterA, clusterB, db) {
  if (clusterA === clusterB) {
    return; // Don't link a cluster to itself
  }

  try {
    // Check if link exists (in either direction)
    const existingLink = db.prepare(`
      SELECT id, strength FROM cluster_links
      WHERE (cluster_a = ? AND cluster_b = ?)
         OR (cluster_a = ? AND cluster_b = ?)
    `).get(clusterA, clusterB, clusterB, clusterA);

    if (existingLink) {
      // Strengthen existing link (max 1.0)
      const newStrength = Math.min(1.0, existingLink.strength + 0.1);
      db.prepare('UPDATE cluster_links SET strength = ? WHERE id = ?')
        .run(newStrength, existingLink.id);
      console.log(`[Clusters] Strengthened link between clusters (${newStrength.toFixed(2)})`);
    } else {
      // Create new link
      const linkId = randomUUID();
      db.prepare(`
        INSERT INTO cluster_links (id, cluster_a, cluster_b, strength)
        VALUES (?, ?, ?, 0.5)
      `).run(linkId, clusterA, clusterB);
      console.log(`[Clusters] Created new link between clusters`);
    }
  } catch (error) {
    console.error('[Clusters] Error creating/strengthening link:', error.message);
  }
}

/**
 * Assign a fact to a cluster (existing or new)
 * @param {string} fact - The fact to cluster
 * @param {string} provider - LLM provider for cluster naming
 * @param {string} model - Model name
 * @param {string} apiKey - API key
 * @param {string} host - Host URL
 * @param {string} source - Source of the fact
 * @returns {Promise<Object>} - {clusterId, clusterName, isNew}
 */
async function assignToCluster(fact, provider, model, apiKey, host, source = 'conversation', salience = 5, subject = 'user') {
  try {
    const config = getConfig();
    const db = getSqliteDb();
    if (!db) {
      console.error('[Clusters] Database not initialized');
      return { clusterId: null, clusterName: null, isNew: false };
    }

    // Generate embedding for the fact
    console.log('[Clusters] Generating embedding for fact');
    const embedding = await generateEmbedding(fact);
    if (!embedding) {
      console.error('[Clusters] Failed to generate embedding');
      return { clusterId: null, clusterName: null, isNew: false };
    }

    // Search for similar content in existing clusters
    const clusterTable = await getClusterEmbeddingsTable();
    let bestClusterId = null;
    let bestSimilarity = 0;
    const crossClusterCandidates = [];

    if (clusterTable) {
      console.log('[Clusters] Searching for similar cluster members');
      // Convert Float32Array to regular array for LanceDB compatibility
      const vectorArray = Array.from(embedding);
      const results = await clusterTable
        .search(vectorArray)
        .metricType('cosine')
        .limit(10)
        .execute();

      // Group by cluster and find best match
      const clusterScores = {};
      for (const result of results) {
        const similarity = 1 - (result._distance || 0); // Convert distance to similarity

        if (!clusterScores[result.cluster_id]) {
          clusterScores[result.cluster_id] = [];
        }
        clusterScores[result.cluster_id].push(similarity);

        // Track potential cross-cluster links
        if (similarity > config.memory.clusterLinkThreshold) {
          crossClusterCandidates.push({
            clusterId: result.cluster_id,
            similarity: similarity
          });
        }
      }

      // Restrict candidates to clusters of the SAME subject so self-observations
      // never merge into user-fact clusters (and vice versa). Self-facts and
      // user-facts live in separate cluster spaces.
      const candidateIds = Object.keys(clusterScores);
      if (candidateIds.length > 0) {
        const placeholders = candidateIds.map(() => '?').join(',');
        const subjRows = db.prepare(
          `SELECT id, subject FROM memory_clusters WHERE id IN (${placeholders})`
        ).all(...candidateIds);
        const subjById = new Map(subjRows.map(r => [r.id, r.subject || 'user']));
        for (const cid of candidateIds) {
          if ((subjById.get(cid) || 'user') !== subject) delete clusterScores[cid];
        }
        // Drop cross-cluster link candidates that are a different subject too.
        for (let i = crossClusterCandidates.length - 1; i >= 0; i--) {
          if ((subjById.get(crossClusterCandidates[i].clusterId) || 'user') !== subject) {
            crossClusterCandidates.splice(i, 1);
          }
        }
      }

      // Find cluster with highest max similarity (max is a better signal than
      // average — large clusters with some marginal members would otherwise
      // have their averages dragged down, causing duplicate cluster creation)
      for (const [clusterId, similarities] of Object.entries(clusterScores)) {
        const maxSimilarity = Math.max(...similarities);
        if (maxSimilarity > bestSimilarity) {
          bestSimilarity = maxSimilarity;
          bestClusterId = clusterId;
        }
      }

      console.log(`[Clusters] Best cluster match: ${bestClusterId} (similarity: ${bestSimilarity.toFixed(3)})`);
    }

    let clusterId = bestClusterId;
    let clusterName = null;
    let isNew = false;

    // Soft match band: if similarity is between clusterLinkThreshold (0.50) and
    // similarityThreshold (0.60), check whether the fact and the best cluster
    // share a curated category — if so, merge instead of creating a duplicate.
    const softMatchThreshold = config.memory.clusterLinkThreshold; // 0.50

    if (bestClusterId && bestSimilarity > softMatchThreshold && bestSimilarity <= config.memory.similarityThreshold) {
      const factCategory = matchCuratedCategory(fact);
      if (factCategory && bestClusterId) {
        const bestCluster = db.prepare('SELECT name FROM memory_clusters WHERE id = ?').get(bestClusterId);
        if (bestCluster && bestCluster.name === factCategory) {
          // Curated categories match — treat as a merge
          clusterId = bestClusterId;
          clusterName = bestCluster.name;
          console.log(`[Clusters] Soft match: "${factCategory}" category match (similarity: ${bestSimilarity.toFixed(3)}) → merging`);

          db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?')
            .run(new Date().toISOString(), clusterId);
        }
      }
    }

    // Create new cluster if no match (hard or soft)
    if (!clusterId || (!clusterName && bestSimilarity <= config.memory.similarityThreshold)) {
      clusterName = await generateClusterName(fact, provider, model, apiKey, host);

      // Name-collision check: if a cluster with this name AND subject already
      // exists, route there instead (name lookups are scoped per subject so a
      // "self" cluster and a "user" cluster may share a name harmlessly).
      const existingByName = db.prepare(
        'SELECT id, name FROM memory_clusters WHERE name = ? AND subject = ?'
      ).get(clusterName, subject);

      if (existingByName) {
        clusterId = existingByName.id;
        console.log(`[Clusters] Routing to existing cluster "${clusterName}" (name collision avoided)`);

        db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?')
          .run(new Date().toISOString(), clusterId);
      } else {
        console.log('[Clusters] Creating new cluster');
        clusterId = randomUUID();
        const now = new Date().toISOString();

        db.prepare(`
          INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject)
          VALUES (?, ?, '', ?, ?, ?)
        `).run(clusterId, clusterName, now, now, subject);

        isNew = true;
        console.log(`[Clusters] Created ${subject} cluster: ${clusterName}`);
      }
    } else if (!clusterName) {
      // Get existing cluster name
      const cluster = db.prepare('SELECT name FROM memory_clusters WHERE id = ?').get(clusterId);
      clusterName = cluster?.name || 'Unknown';

      // Update cluster timestamp
      db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?')
        .run(new Date().toISOString(), clusterId);
    }

    // Insert into cluster_members
    const memberId = randomUUID();
    const nowIso = new Date().toISOString();
    const salienceValue = Number.isFinite(salience) ? Math.max(1, Math.min(10, Math.round(salience))) : 5;
    db.prepare(`
      INSERT INTO cluster_members (id, cluster_id, content, source, importance, created_at, updated_at, salience, subject)
      VALUES (?, ?, ?, ?, 0.5, ?, ?, ?, ?)
    `).run(memberId, clusterId, fact, source, nowIso, nowIso, salienceValue, subject);

    console.log(`[Clusters] Added fact to cluster: ${clusterName}`);

    // Add embedding to LanceDB
    if (clusterTable) {
      // Convert Float32Array to regular array for LanceDB compatibility
      const vectorForStorage = Array.from(embedding);
      await clusterTable.add([{
        id: randomUUID(),
        member_id: memberId,
        cluster_id: clusterId,
        content: fact,
        vector: vectorForStorage
      }]);
    }

    // Cross-cluster linking — link when fact is similar to members in other clusters
    if (crossClusterCandidates.length > 0) {
      const uniqueClusters = [...new Set(crossClusterCandidates.map(c => c.clusterId))];
      const otherClusters = uniqueClusters.filter(id => id !== clusterId);

      if (otherClusters.length > 0) {
        console.log(`[Clusters] Creating/strengthening ${otherClusters.length} cross-cluster link(s)`);
        for (const otherClusterId of otherClusters) {
          createOrStrengthenLink(clusterId, otherClusterId, db);
        }
      }
    }

    return { clusterId, clusterName, isNew, memberId, salience: salienceValue };
  } catch (error) {
    console.error('[Clusters] Error in assignToCluster:', error);
    return { clusterId: null, clusterName: null, isNew: false, memberId: null };
  }
}

/**
 * Update a fact's salience (1–10). Used when a superseding fact must inherit
 * at least the salience of the fact it replaced.
 * @param {string} memberId
 * @param {number} salience
 * @returns {boolean}
 */
function updateFactSalience(memberId, salience) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const value = Math.max(1, Math.min(10, Math.round(salience)));
    const info = db.prepare(
      'UPDATE cluster_members SET salience = ?, updated_at = ? WHERE id = ?'
    ).run(value, new Date().toISOString(), memberId);
    return info.changes > 0;
  } catch (error) {
    console.error('[Clusters] updateFactSalience error:', error.message);
    return false;
  }
}

/**
 * Find existing ACTIVE facts that are semantically close to a candidate fact.
 * Used by the extraction pipeline to surface potential contradictions before
 * an LLM makes the final yes/no call. Superseded facts are excluded — we only
 * contradict against what is currently believed true.
 * @param {string} factText - The new candidate fact
 * @param {Object} [opts]
 * @param {number} [opts.threshold=0.45] - Min cosine similarity to consider
 * @param {number} [opts.limit=5] - Max candidates to return
 * @returns {Promise<Array<{memberId,content,clusterId,similarity}>>}
 */
async function findContradictionCandidates(factText, opts = {}) {
  const threshold = opts.threshold ?? 0.45;
  const limit = opts.limit ?? 5;
  const subject = opts.subject ?? 'user';
  try {
    const db = getSqliteDb();
    if (!db) return [];

    const embedding = await generateEmbedding(factText);
    if (!embedding) return [];

    const clusterTable = await getClusterEmbeddingsTable();
    if (!clusterTable) return [];

    const vectorArray = Array.from(embedding);
    const results = await clusterTable
      .search(vectorArray)
      .metricType('cosine')
      .limit(15)
      .execute();

    const normalizedNew = factText.trim().toLowerCase();
    const seen = new Set();
    const candidates = [];

    for (const result of results) {
      const similarity = 1 - (result._distance || 0);
      if (similarity < threshold) continue;
      const memberId = result.member_id;
      if (!memberId || seen.has(memberId)) continue;

      // Confirm the member still exists and is ACTIVE (LanceDB retains
      // embeddings of superseded facts for history; SQLite is the truth).
      const row = db.prepare(
        'SELECT id, content, cluster_id, status, salience, subject FROM cluster_members WHERE id = ?'
      ).get(memberId);
      if (!row) continue;
      if (row.status && row.status !== 'active') continue;
      // Only contradict within the same subject: self-observations can only
      // supersede other self-observations, user-facts only user-facts.
      if ((row.subject || 'user') !== subject) continue;

      // A verbatim duplicate is not a contradiction — skip the judge call.
      if (row.content.trim().toLowerCase() === normalizedNew) continue;

      seen.add(memberId);
      candidates.push({
        memberId: row.id,
        content: row.content,
        clusterId: row.cluster_id,
        salience: row.salience ?? 5,
        similarity
      });
      if (candidates.length >= limit) break;
    }

    return candidates;
  } catch (error) {
    console.error('[Clusters] Error in findContradictionCandidates:', error.message);
    return [];
  }
}

/**
 * Supersede an existing fact: mark it superseded and point it at the fact that
 * replaced it. History is preserved — the row is kept, never deleted.
 * @param {string} oldMemberId - Fact being replaced
 * @param {string} newMemberId - Fact that replaces it
 * @returns {boolean} - true if a row was updated
 */
function supersedeFact(oldMemberId, newMemberId) {
  try {
    const db = getSqliteDb();
    if (!db) return false;
    const info = db.prepare(
      "UPDATE cluster_members SET status = 'superseded', superseded_by = ?, updated_at = ? WHERE id = ? AND status = 'active'"
    ).run(newMemberId, new Date().toISOString(), oldMemberId);
    if (info.changes > 0) {
      console.log(`[Clusters] Superseded fact ${oldMemberId} → ${newMemberId}`);
      return true;
    }
    return false;
  } catch (error) {
    console.error('[Clusters] Error in supersedeFact:', error.message);
    return false;
  }
}

/**
 * Search clusters for relevant content
 * @param {string} query - Search query
 * @param {number} limit - Max number of clusters to return
 * @returns {Promise<Array>} - Array of cluster results with members and linked content
 */
async function searchClusters(query, limit = 3) {
  try {
    const db = getSqliteDb();
    if (!db) {
      console.log('[Clusters] Database not initialized');
      return [];
    }

    // Generate embedding for query
    const embedding = await generateEmbedding(query);
    if (!embedding) {
      console.error('[Clusters] Failed to generate query embedding');
      return [];
    }

    const clusterTable = await getClusterEmbeddingsTable();
    if (!clusterTable) {
      console.log('[Clusters] Cluster embeddings table not available');
      return [];
    }

    // Search for similar content
    console.log('[Clusters] Searching for relevant clusters');
    // Convert Float32Array to regular array for LanceDB compatibility
    const vectorArray = Array.from(embedding);
    const results = await clusterTable
      .search(vectorArray)
      .metricType('cosine')
      .limit(20)
      .execute();

    // Group by cluster and rank
    const clusterScores = {};
    for (const result of results) {
      const similarity = 1 - (result._distance || 0);
      if (!clusterScores[result.cluster_id]) {
        clusterScores[result.cluster_id] = {
          maxSimilarity: similarity,
          avgSimilarity: 0,
          count: 0
        };
      }
      const score = clusterScores[result.cluster_id];
      score.maxSimilarity = Math.max(score.maxSimilarity, similarity);
      score.avgSimilarity += similarity;
      score.count++;
    }

    // Rank by max similarity (consistent with assignToCluster scoring)
    const rankedClusters = Object.entries(clusterScores)
      .map(([clusterId, score]) => ({
        clusterId,
        score: score.maxSimilarity
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    console.log(`[Clusters] Found ${rankedClusters.length} relevant clusters`);

    // Build results with members and linked content
    const clusterResults = [];
    for (const { clusterId } of rankedClusters) {
      // Get cluster info
      const cluster = db.prepare(`
        SELECT id, name, description
        FROM memory_clusters
        WHERE id = ?
      `).get(clusterId);

      if (!cluster) continue;

      // Get all members (active only — superseded facts never enter model context).
      // Order by salience so that if the injected context has to be trimmed for
      // budget, the highest-salience facts survive.
      const members = db.prepare(`
        SELECT content, importance, salience, created_at
        FROM cluster_members
        WHERE cluster_id = ?
          AND (status = 'active' OR status IS NULL)
        ORDER BY salience DESC, importance DESC, created_at DESC
      `).all(clusterId);

      // A cluster whose facts have all been superseded contributes nothing
      if (members.length === 0) continue;

      // Get linked clusters
      const linkedClusters = db.prepare(`
        SELECT
          CASE
            WHEN cl.cluster_a = ? THEN cl.cluster_b
            ELSE cl.cluster_a
          END as linked_cluster_id,
          cl.strength
        FROM cluster_links cl
        WHERE (cl.cluster_a = ? OR cl.cluster_b = ?)
          AND cl.strength > 0.3
        ORDER BY cl.strength DESC
      `).all(clusterId, clusterId, clusterId);

      // Get members from linked clusters
      const linkedMembers = [];
      for (const link of linkedClusters) {
        const linkCluster = db.prepare('SELECT name FROM memory_clusters WHERE id = ?')
          .get(link.linked_cluster_id);

        const linkMembers = db.prepare(`
          SELECT content, created_at
          FROM cluster_members
          WHERE cluster_id = ?
            AND (status = 'active' OR status IS NULL)
          ORDER BY salience DESC, importance DESC, created_at DESC
          LIMIT 3
        `).all(link.linked_cluster_id);

        for (const member of linkMembers) {
          linkedMembers.push({
            content: member.content,
            created_at: member.created_at,
            clusterName: linkCluster?.name || 'Unknown',
            linkStrength: link.strength
          });
        }
      }

      clusterResults.push({
        cluster: {
          id: cluster.id,
          name: cluster.name,
          description: cluster.description
        },
        members: members.map(m => ({
          content: m.content,
          importance: m.importance,
          salience: m.salience ?? 5,
          created_at: m.created_at
        })),
        linkedMembers: linkedMembers
      });
    }

    return clusterResults;
  } catch (error) {
    console.error('[Clusters] Error in searchClusters:', error);
    return [];
  }
}

/**
 * Get all clusters with member counts
 * @returns {Array} - Array of clusters with metadata
 */
function getClusters(subject = null) {
  try {
    const db = getSqliteDb();
    if (!db) {
      return [];
    }

    const where = subject ? 'WHERE mc.subject = ?' : '';
    const params = subject ? [subject] : [];
    const clusters = db.prepare(`
      SELECT mc.*, COUNT(cm.id) as member_count
      FROM memory_clusters mc
      LEFT JOIN cluster_members cm ON mc.id = cm.cluster_id
      ${where}
      GROUP BY mc.id
      ORDER BY mc.updated_at DESC
    `).all(...params);

    return clusters;
  } catch (error) {
    console.error('[Clusters] Error in getClusters:', error);
    return [];
  }
}

/**
 * Get self-facts (SNH's observations about itself), salience-ordered.
 * @param {Object} [opts]
 * @param {string|null} [opts.status='active'] - 'active', 'superseded', or null for all
 * @param {number|null} [opts.limit=null] - max rows, or null for no limit
 * @returns {Array} cluster_member rows with cluster_name attached
 */
function getSelfFacts({ status = 'active', limit = null } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return [];

    let sql = `
      SELECT cm.id, cm.content, cm.salience, cm.status, cm.superseded_by,
             cm.created_at, cm.updated_at, cm.cluster_id, cm.source,
             mc.name AS cluster_name
      FROM cluster_members cm
      LEFT JOIN memory_clusters mc ON mc.id = cm.cluster_id
      WHERE cm.subject = 'self'`;
    const params = [];
    if (status) { sql += ' AND cm.status = ?'; params.push(status); }
    sql += ' ORDER BY cm.salience DESC, cm.created_at DESC';
    if (limit) { sql += ' LIMIT ?'; params.push(limit); }

    return db.prepare(sql).all(...params);
  } catch (error) {
    console.error('[Clusters] Error in getSelfFacts:', error.message);
    return [];
  }
}

/**
 * Get a specific cluster with all members and linked clusters
 * @param {string} id - Cluster ID
 * @returns {Object|null} - Cluster details or null
 */
function getCluster(id) {
  try {
    const db = getSqliteDb();
    if (!db) {
      return null;
    }

    const cluster = db.prepare(`
      SELECT * FROM memory_clusters WHERE id = ?
    `).get(id);

    if (!cluster) {
      return null;
    }

    // Get members
    const members = db.prepare(`
      SELECT * FROM cluster_members
      WHERE cluster_id = ?
      ORDER BY importance DESC, created_at DESC
    `).all(id);

    // Get linked clusters
    const linkedClusters = db.prepare(`
      SELECT
        mc.*,
        cl.strength
      FROM cluster_links cl
      JOIN memory_clusters mc ON mc.id = CASE
        WHEN cl.cluster_a = ? THEN cl.cluster_b
        ELSE cl.cluster_a
      END
      WHERE cl.cluster_a = ? OR cl.cluster_b = ?
      ORDER BY cl.strength DESC
    `).all(id, id, id);

    return {
      ...cluster,
      members,
      linkedClusters
    };
  } catch (error) {
    console.error('[Clusters] Error in getCluster:', error);
    return null;
  }
}

// Known person names for people-cluster detection
const PERSON_NAMES = new Set([
  'wayne', 'eric', 'ellie', 'casper', 'cece', 'calypso', 'erika', 'piff',
  'lucy', 'grey'
]);

/**
 * Check if a fact is about a person (contains known names or family terms)
 * @param {string} text - Fact text
 * @returns {boolean}
 */
function isPersonFact(text) {
  const lower = text.toLowerCase();
  for (const name of PERSON_NAMES) {
    if (lower.includes(name)) return true;
  }
  return /\b(father|mother|partner|wife|husband|son|daughter|brother|sister|cares?\s+for)\b/i.test(text);
}

/**
 * Merge singleton clusters into the most similar non-singleton cluster
 * @param {number} threshold - Minimum similarity to merge (default 0.50)
 * @returns {Promise<number>} - Number of singletons merged
 */
async function mergeSingletons(threshold) {
  if (threshold === undefined) {
    threshold = getConfig().memory.clusterLinkThreshold;
  }
  try {
    const db = getSqliteDb();
    if (!db) return 0;

    // Find singleton clusters (clusters with exactly 1 member)
    const singletons = db.prepare(`
      SELECT mc.id as cluster_id, mc.name, cm.id as member_id, cm.content
      FROM memory_clusters mc
      JOIN cluster_members cm ON cm.cluster_id = mc.id
      GROUP BY mc.id
      HAVING COUNT(cm.id) = 1
    `).all();

    if (singletons.length === 0) {
      console.log('[Clusters] No singleton clusters to merge');
      return 0;
    }

    // Find non-singleton cluster IDs and their names
    const nonSingletons = db.prepare(`
      SELECT mc.id, mc.name
      FROM memory_clusters mc
      JOIN cluster_members cm ON cm.cluster_id = mc.id
      GROUP BY mc.id
      HAVING COUNT(cm.id) > 1
    `).all();

    if (nonSingletons.length === 0) {
      console.log('[Clusters] No non-singleton clusters to merge into');
      return 0;
    }

    const nonSingletonSet = new Set(nonSingletons.map(r => r.id));

    // Find or identify a "People" cluster among non-singletons
    let peopleClusterId = null;
    for (const ns of nonSingletons) {
      if (/people|family|person/i.test(ns.name)) {
        peopleClusterId = ns.id;
        break;
      }
    }
    // Also check if any non-singleton has person-related members
    if (!peopleClusterId) {
      for (const ns of nonSingletons) {
        const members = db.prepare('SELECT content FROM cluster_members WHERE cluster_id = ?').all(ns.id);
        if (members.some(m => isPersonFact(m.content))) {
          peopleClusterId = ns.id;
          break;
        }
      }
    }

    const clusterTable = await getClusterEmbeddingsTable();
    if (!clusterTable) {
      console.log('[Clusters] Cluster embeddings table not available');
      return 0;
    }

    let merged = 0;

    for (const singleton of singletons) {
      const factIsPerson = isPersonFact(singleton.content);

      // If this is a person fact and we have a people cluster, prefer that
      if (factIsPerson && peopleClusterId && peopleClusterId !== singleton.cluster_id) {
        const targetCluster = db.prepare('SELECT name FROM memory_clusters WHERE id = ?')
          .get(peopleClusterId);

        console.log(`[Clusters] Merging person-fact singleton "${singleton.name}" → "${targetCluster?.name}" (person-name match)`);

        // Generate embedding for LanceDB update
        const embedding = await generateEmbedding(singleton.content);
        const vectorArray = embedding ? Array.from(embedding) : null;

        // Move member to people cluster
        db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?')
          .run(peopleClusterId, singleton.member_id);

        if (vectorArray) {
          try {
            await clusterTable.delete(`member_id = "${safeId(singleton.member_id)}"`);
            await clusterTable.add([{
              id: randomUUID(),
              member_id: singleton.member_id,
              cluster_id: peopleClusterId,
              content: singleton.content,
              vector: vectorArray
            }]);
          } catch (lanceErr) {
            console.error('[Clusters] LanceDB update error during merge:', lanceErr.message);
          }
        }

        db.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?')
          .run(singleton.cluster_id, singleton.cluster_id);
        db.prepare('DELETE FROM memory_clusters WHERE id = ?')
          .run(singleton.cluster_id);
        db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?')
          .run(new Date().toISOString(), peopleClusterId);

        merged++;
        continue;
      }

      // Standard embedding-based merge
      const embedding = await generateEmbedding(singleton.content);
      if (!embedding) continue;

      const vectorArray = Array.from(embedding);
      const results = await clusterTable
        .search(vectorArray)
        .metricType('cosine')
        .limit(20)
        .execute();

      // Find best non-singleton match
      let bestClusterId = null;
      let bestSimilarity = 0;

      for (const result of results) {
        if (result.cluster_id === singleton.cluster_id) continue;
        if (!nonSingletonSet.has(result.cluster_id)) continue;

        const similarity = 1 - (result._distance || 0);
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestClusterId = result.cluster_id;
        }
      }

      if (!bestClusterId || bestSimilarity < threshold) {
        console.log(`[Clusters] Keeping singleton "${singleton.name}" (best: ${bestSimilarity.toFixed(3)} < ${threshold})`);
        continue;
      }

      const targetCluster = db.prepare('SELECT name FROM memory_clusters WHERE id = ?')
        .get(bestClusterId);

      console.log(`[Clusters] Merging singleton "${singleton.name}" → "${targetCluster?.name}" (similarity: ${bestSimilarity.toFixed(3)})`);

      // Move member to target cluster
      db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?')
        .run(bestClusterId, singleton.member_id);

      // Update LanceDB: delete old entry, add with new cluster_id
      try {
        await clusterTable.delete(`member_id = "${safeId(singleton.member_id)}"`);
        await clusterTable.add([{
          id: randomUUID(),
          member_id: singleton.member_id,
          cluster_id: bestClusterId,
          content: singleton.content,
          vector: vectorArray
        }]);
      } catch (lanceErr) {
        console.error('[Clusters] LanceDB update error during merge:', lanceErr.message);
      }

      // Delete empty cluster and its links
      db.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?')
        .run(singleton.cluster_id, singleton.cluster_id);
      db.prepare('DELETE FROM memory_clusters WHERE id = ?')
        .run(singleton.cluster_id);

      // Update target cluster timestamp
      db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?')
        .run(new Date().toISOString(), bestClusterId);

      merged++;
    }

    // --- Second pass: category-based merge for remaining singletons ---
    // Re-query singletons that survived the embedding pass
    const remainingSingletons = db.prepare(`
      SELECT mc.id as cluster_id, mc.name, cm.id as member_id, cm.content
      FROM memory_clusters mc
      JOIN cluster_members cm ON cm.cluster_id = mc.id
      GROUP BY mc.id
      HAVING COUNT(cm.id) = 1
    `).all();

    if (remainingSingletons.length > 0) {
      // Group remaining singletons by curated category
      const categoryGroups = {}; // category name → [singleton]
      for (const s of remainingSingletons) {
        const category = matchCuratedCategory(s.content);
        if (category) {
          if (!categoryGroups[category]) categoryGroups[category] = [];
          categoryGroups[category].push(s);
        }
      }

      // Also check if any existing non-singleton cluster matches each category
      const currentNonSingletons = db.prepare(`
        SELECT mc.id, mc.name
        FROM memory_clusters mc
        JOIN cluster_members cm ON cm.cluster_id = mc.id
        GROUP BY mc.id
        HAVING COUNT(cm.id) > 1
      `).all();

      for (const [category, group] of Object.entries(categoryGroups)) {
        if (group.length < 2) {
          // Check if there's an existing non-singleton cluster for this category
          let targetId = null;
          for (const ns of currentNonSingletons) {
            const members = db.prepare('SELECT content FROM cluster_members WHERE cluster_id = ?').all(ns.id);
            const nsCategory = generateClusterNameFromMembers(members);
            if (nsCategory === category) {
              targetId = ns.id;
              break;
            }
          }
          if (!targetId) continue; // Only 1 singleton, no matching cluster — skip
          // Merge single singleton into matching non-singleton cluster
          const s = group[0];
          console.log(`[Clusters] Category merge: "${s.name}" → existing "${category}" cluster`);
          db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?').run(targetId, s.member_id);
          const embedding = await generateEmbedding(s.content);
          if (embedding) {
            const vectorArray = Array.from(embedding);
            try {
              await clusterTable.delete(`member_id = "${safeId(s.member_id)}"`)
;
              await clusterTable.add([{ id: randomUUID(), member_id: s.member_id, cluster_id: targetId, content: s.content, vector: vectorArray }]);
            } catch (e) { console.error('[Clusters] LanceDB error:', e.message); }
          }
          db.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?').run(s.cluster_id, s.cluster_id);
          db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(s.cluster_id);
          merged++;
        } else {
          // Merge multiple singletons sharing a category into the first one's cluster
          const target = group[0];
          console.log(`[Clusters] Category merge: grouping ${group.length} singletons into "${category}"`);
          for (let i = 1; i < group.length; i++) {
            const s = group[i];
            db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?').run(target.cluster_id, s.member_id);
            const embedding = await generateEmbedding(s.content);
            if (embedding) {
              const vectorArray = Array.from(embedding);
              try {
                await clusterTable.delete(`member_id = "${safeId(s.member_id)}"`)
;
                await clusterTable.add([{ id: randomUUID(), member_id: s.member_id, cluster_id: target.cluster_id, content: s.content, vector: vectorArray }]);
              } catch (e) { console.error('[Clusters] LanceDB error:', e.message); }
            }
            db.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?').run(s.cluster_id, s.cluster_id);
            db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(s.cluster_id);
            merged++;
          }
        }
      }
    }

    console.log(`[Clusters] Merged ${merged}/${singletons.length} singletons total`);
    return merged;
  } catch (error) {
    console.error('[Clusters] Error merging singletons:', error.message);
    return 0;
  }
}

/**
 * Merge clusters that share the same name (post-rename duplicates).
 * For each duplicate name group, keeps the cluster with the most members
 * and moves all members from the others into it. Updates LanceDB cluster_id
 * metadata in-place (no re-embedding).
 * @returns {Promise<number>} - Number of source clusters merged away
 */
async function mergeByName() {
  try {
    const db = getSqliteDb();
    if (!db) return 0;

    const rows = db.prepare(`
      SELECT mc.id, mc.name, COUNT(cm.id) AS member_count
      FROM memory_clusters mc
      LEFT JOIN cluster_members cm ON cm.cluster_id = mc.id
      GROUP BY mc.id
      ORDER BY mc.name ASC, member_count DESC
    `).all();

    // Group by name
    const byName = {};
    for (const row of rows) {
      if (!byName[row.name]) byName[row.name] = [];
      byName[row.name].push(row);
    }

    const clusterTable = await getClusterEmbeddingsTable();
    let merged = 0;

    for (const [name, group] of Object.entries(byName)) {
      if (group.length <= 1) continue;

      const [target, ...sources] = group; // sorted DESC by member_count

      for (const source of sources) {
        // Get member IDs before moving
        const members = db.prepare(
          'SELECT id FROM cluster_members WHERE cluster_id = ?'
        ).all(source.id);

        // Move members to target
        db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE cluster_id = ?')
          .run(target.id, source.id);

        // Re-point links (skip self-links and duplicates)
        const links = db.prepare(
          'SELECT * FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?'
        ).all(source.id, source.id);

        for (const link of links) {
          const newA = link.cluster_a === source.id ? target.id : link.cluster_a;
          const newB = link.cluster_b === source.id ? target.id : link.cluster_b;

          if (newA === newB) {
            db.prepare('DELETE FROM cluster_links WHERE id = ?').run(link.id);
            continue;
          }

          const existing = db.prepare(`
            SELECT id FROM cluster_links
            WHERE ((cluster_a = ? AND cluster_b = ?) OR (cluster_a = ? AND cluster_b = ?))
              AND id != ?
          `).get(newA, newB, newB, newA, link.id);

          if (existing) {
            db.prepare('DELETE FROM cluster_links WHERE id = ?').run(link.id);
          } else {
            db.prepare('UPDATE cluster_links SET cluster_a = ?, cluster_b = ? WHERE id = ?')
              .run(newA, newB, link.id);
          }
        }

        // Delete source cluster
        db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(source.id);

        // Update LanceDB cluster_id in-place for moved members
        if (clusterTable) {
          for (const m of members) {
            try {
              await clusterTable.update({
                where: `member_id = "${safeId(m.id)}"`,
                valuesSql: { cluster_id: `'${safeId(target.id)}'` }
              });
            } catch (e) {
              console.error(`[Clusters] LanceDB update error during mergeByName: ${e.message}`);
            }
          }
        }

        console.log(`[Clusters] mergeByName: "${name}" (${source.member_count} members) → target (${target.member_count} members)`);
        merged++;
      }
    }

    if (merged > 0) {
      console.log(`[Clusters] mergeByName: merged ${merged} duplicate-name cluster(s)`);
    }
    return merged;
  } catch (error) {
    console.error('[Clusters] Error in mergeByName:', error.message);
    return 0;
  }
}

module.exports = {
  assignToCluster,
  findContradictionCandidates,
  supersedeFact,
  updateFactSalience,
  searchClusters,
  getClusters,
  getCluster,
  getSelfFacts,
  generateEmbedding,
  cosineSimilarity,
  generateClusterNameFromMembers,
  matchCuratedCategory,
  isValidClusterName,
  renameAllClusters,
  mergeByName,
  mergeSingletons
};
