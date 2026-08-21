const { randomUUID } = require('crypto');
const { getSqliteDb, getClusterEmbeddingsTable } = require('./database');
const { getConfig, getProviderInstance } = require('./config');
const { formatFactTimestamp } = require('./datetime');
const { estTokens } = require('./injection-budget');

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

/**
 * Normalize raw LLM output into a clean cluster label: single line, Title Case,
 * no punctuation except "&", 2–4 words, ≤30 chars. Returns null if what's left
 * isn't a usable name (so callers can fall back or keep the existing name).
 * @param {string} raw
 * @returns {string|null}
 */
function sanitizeClusterName(raw) {
  if (!raw) return null;
  // Pick the first non-empty line that isn't an obvious preamble ("Here is the
  // label:"), so a chatty model's lead-in doesn't become the name.
  const lines = String(raw).split('\n').map(x => x.trim()).filter(Boolean);
  let s = lines.find(l => !/:$/.test(l) && !/^(sure|ok|okay|here|the label|category is)\b/i.test(l))
    || lines[0] || '';
  // Drop a leading "Label:" / "Category -" style prefix.
  s = s.replace(/^(category|label|name|group|answer|cluster)\s*[:\-–]\s*/i, '');
  // Keep only letters, digits, spaces and "&"; everything else → space.
  s = s.replace(/[^A-Za-z0-9 &]+/g, ' ').replace(/\s+/g, ' ').trim();
  if (!s) return null;

  // Trim leading/trailing function words so labels don't dangle on "And"/"The".
  const TRIM = new Set(['and', 'or', 'the', 'a', 'an', 'of', 'to', 'for', 'with', 'is', 'are', 'in', 'on']);
  let toks = s.split(' ').filter(w => w === '&' || w.length > 0);
  while (toks.length && (toks[0] === '&' || TRIM.has(toks[0].toLowerCase()))) toks.shift();
  while (toks.length && (toks[toks.length - 1] === '&' || TRIM.has(toks[toks.length - 1].toLowerCase()))) toks.pop();
  if (toks.length === 0) return null;

  // Cap at 4 meaningful (non-"&") words, preserving order and any "&".
  const out = [];
  let count = 0;
  for (const w of toks) {
    if (w === '&') { out.push(w); continue; }
    if (count >= 4) break;
    out.push(w);
    count++;
  }
  // A function word may now sit at the truncated tail ("...Users And") — trim it.
  while (out.length && (out[out.length - 1] === '&' || TRIM.has(out[out.length - 1].toLowerCase()))) out.pop();
  if (out.length === 0) return null;
  // Title-case each word, but leave short acronyms (AI, AGI, MSP, RAV4) intact.
  let words = out.map(w =>
    w === '&' ? '&'
      : /^[A-Z0-9]{2,4}$/.test(w) ? w
        : w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()
  );
  s = words.join(' ').replace(/\s*&\s*/g, ' & ').replace(/\s+/g, ' ').trim();

  // Length cap first, then strip any "&" left dangling at either end (a trailing
  // word may have been cut off, leaving "Foo &").
  if (s.length > 30) {
    s = s.slice(0, 30).replace(/\s+\S*$/, '').trim() || s.slice(0, 30).trim();
  }
  s = s.replace(/^&\s*/, '').replace(/\s*&\s*$/, '').trim();
  if (s.length < 3) return null;
  // Reject sentence fragments that slipped through (verbs, >4 words).
  if (!isValidClusterName(s)) return null;
  return s;
}

/**
 * The one shared cluster namer: given a cluster's member facts, ask the LLM for
 * a short natural-English noun-phrase label. Used by creation, the rename pass,
 * audit splits, and the repair script so every path produces the same style of
 * name. Returns null on failure so callers can fall back / keep the old name.
 * @param {Array<string|{content:string}>} facts - member facts (strings or rows)
 * @param {Object} [opts]
 * @param {string} [opts.subject] - 'user' | 'self' (tunes the framing slightly)
 * @returns {Promise<string|null>}
 */
async function generateClusterNameLLM(facts, opts = {}) {
  try {
    const contents = (facts || [])
      .map(f => (typeof f === 'string' ? f : (f && f.content)) || '')
      .map(s => s.trim())
      .filter(Boolean);
    if (contents.length === 0) return null;

    // Bound the prompt: at most 14 facts, each ≤200 chars.
    const sample = contents.slice(0, 14).map(s => (s.length > 200 ? s.slice(0, 200) : s));
    const isSelf = opts.subject === 'self';

    const memoryManager = require('./memory-manager');
    const systemPrompt = 'You label a group of related facts with a short category name, like a folder label.';
    const userPrompt =
      `${isSelf ? 'These are self-observations an AI made about itself' : 'These facts belong to one group'}:\n` +
      `${sample.map(s => `- ${s}`).join('\n')}\n\n` +
      'Give a short, natural English noun phrase (2–4 words) that a person would use to label this group.\n\n' +
      'Examples:\n' +
      '- facts about a Tundra, a RAV4, and a trailer → Vehicles & Trailer\n' +
      '- self-observations about validating others and reassurance habits → Validation Tendencies\n' +
      '- facts about GPUs, servers, and RAM → Hardware & Infrastructure\n' +
      '- facts about training LLMs and paths to AGI → AI Philosophy\n\n' +
      'Rules: nouns only, no verbs, no full sentences. Join two themes with "&". ' +
      'No quotes or punctuation other than "&". At most 4 words. Output ONLY the label, on one line.';

    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 24 });
    return sanitizeClusterName(content);
  } catch (error) {
    console.error('[Clusters] LLM name generation error:', error.message);
    return null;
  }
}

async function generateClusterName(fact, provider, model, apiKey, host) {
  // 1. Try curated category first — fast, deterministic, no LLM needed.
  const curatedName = matchCuratedCategory(fact);
  if (curatedName) return curatedName;

  // 2. Shared LLM namer (same one used by the rename pass and audit splits).
  const llmName = await generateClusterNameLLM([fact]);
  if (llmName) return llmName;

  // 3. Last resort if the LLM is unavailable: keyword extraction from the fact.
  return extractNameFromFact(fact);
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
 * Regenerate cluster names from their member facts using the shared LLM namer.
 * @param {Object} [options]
 * @param {string[]} [options.ids] - Only rename these cluster ids (default: all).
 * @returns {Promise<number>} - Number of clusters renamed
 */
async function renameAllClusters(options = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return 0;

    const clusters = Array.isArray(options.ids)
      ? options.ids
          .map(id => db.prepare('SELECT id, name, subject FROM memory_clusters WHERE id = ?').get(id))
          .filter(Boolean)
      : db.prepare('SELECT id, name, subject FROM memory_clusters').all();
    let renamed = 0;

    for (const cluster of clusters) {
      // Name from active facts only — superseded ones shouldn't shape the label.
      const members = db.prepare(
        "SELECT content FROM cluster_members WHERE cluster_id = ? AND (status = 'active' OR status IS NULL)"
      ).all(cluster.id);

      if (members.length === 0) continue;

      const newName = await generateClusterNameLLM(members, { subject: cluster.subject });
      // null → LLM unavailable/garbage; keep the existing name rather than clobber it.
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

// createOrStrengthenLink was removed 2026-08-02 along with the cross-link audit.
//
// It inserted every new cluster_links row at a hardcoded strength of 0.5 and bumped
// existing ones by +0.1 per co-occurrence, with no model involved. That constant is
// why 737 of 2,366 stored links (31%) sit at exactly 0.50 — a number that reads as a
// judgment and never was one.
//
// The table itself was dropped at the 2026-08-06 cutover. Association is a
// query-time vector lookup now, per selected cluster — see
// routes/memory.js GET /graph/neighbours/:clusterId.

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
async function assignToCluster(fact, provider, model, apiKey, host, source = 'conversation', salience = 5, subject = 'user', claimType = null, provenance = null) {
  try {
    const config = getConfig();
    const db = getSqliteDb();

    // Dedup BEFORE anything is spent: this used to insert unconditionally, so a
    // fact already held word-for-word still cost an embedding, a cluster search
    // and a row. The rule itself lives in fact-store, which is the single write
    // path — see findExactDuplicate for why exact-only is the MVP scope.
    if (db && fact) {
      const factStore = require('./fact-store');
      const dup = factStore.findExactDuplicate(fact, subject);
      if (dup) {
        factStore.absorbDuplicate(dup, salience);
        console.log(`[Clusters] Already held word-for-word, not storing again: "${String(fact).slice(0, 70)}"`);
        const existingCluster = db.prepare('SELECT name FROM memory_clusters WHERE id = ?').get(dup.cluster_id);
        return {
          clusterId: dup.cluster_id,
          clusterName: existingCluster?.name || 'Unknown',
          isNew: false,
          memberId: dup.id,
          salience: Math.max(dup.salience ?? 0, Number.isFinite(salience) ? salience : 0),
          duplicateOf: dup.id
        };
      }
    }
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
          // A vector may point at a cluster that no longer exists in SQLite —
          // LanceDB is not covered by the foreign key, so a deleted cluster
          // leaves its members' embeddings behind. Such a candidate has to be
          // dropped BEFORE the subject test, because `undefined || 'user'` reads
          // a ghost as a user-fact cluster and lets it win the match; the insert
          // then fails on the cluster_id foreign key and the caller loses the
          // write. Found by the corrector: the F5 compound split failed exactly
          // here, on a 0.951 match to a cluster with no row.
          if (!subjById.has(cid)) {
            console.warn(`[Clusters] Ignoring vector match on cluster ${cid.slice(0, 8)} — no such cluster in SQLite (orphan embedding)`);
            delete clusterScores[cid];
            continue;
          }
          if ((subjById.get(cid) || 'user') !== subject) delete clusterScores[cid];
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
    // Provenance: null-filled rather than omitted, so a caller that passes
    // nothing writes explicit nulls instead of silently inheriting a default.
    // 'unknown' modality is a real answer ("we don't know how this arrived"),
    // distinct from null ("this fact predates provenance").
    const prov = provenance || {};
    const p = {
      conversationId: prov.conversationId ?? null,
      messageId: prov.messageId ?? null,
      verbatimSourceText: prov.verbatimSourceText ?? null,
      inputModality: ['stt', 'typed', 'unknown'].includes(prov.inputModality) ? prov.inputModality : null,
      salienceRationale: prov.salienceRationale ?? null
    };
    db.prepare(`
      INSERT INTO cluster_members (
        id, cluster_id, content, source, importance, created_at, updated_at,
        salience, subject, claim_type, status,
        conversation_id, message_id, verbatim_source_text, input_modality, salience_rationale
      )
      VALUES (?, ?, ?, ?, 0.5, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
    `).run(
      memberId, clusterId, fact, source, nowIso, nowIso, salienceValue, subject, claimType,
      p.conversationId, p.messageId, p.verbatimSourceText, p.inputModality, p.salienceRationale
    );

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
 * Find existing ACTIVE facts of the same subject that are semantically close to a
 * candidate fact — the recall step the contradiction judge runs on.
 *
 * WHAT CHANGED (Phase 2a) AND WHY. This used to pull the top 15 RAW vector
 * neighbours (`.limit(15)`, a bare literal) and only afterwards discard the ones
 * that were superseded, of the wrong subject, or verbatim duplicates — then cap
 * what survived at 5 (`opts.limit ?? 5`, another literal). Both filters ran
 * AFTER the cap had already been spent, and LanceDB deliberately keeps the
 * embeddings of superseded facts as history, so a dense pocket of inactive rows
 * could consume all 15 slots and the judge would be handed nothing. "The judge
 * said no" and "the judge was never shown it" were indistinguishable from the
 * outside, and the second was happening.
 *
 * Now: fetch a large superset, filter to active + same subject FIRST, and select
 * by THRESHOLD rather than by rank — every surviving fact above the similarity
 * floor is a candidate. maxCandidates is a cost ceiling on judge calls, not a
 * selection rule, and when it truncates, the diagnostics say so.
 *
 * Every number here is a config key (`memory.contradiction.*`).
 *
 * @param {string} factText - The new candidate fact
 * @param {Object} [opts]
 * @param {number} [opts.threshold] - Min cosine similarity (default: config floor)
 * @param {number} [opts.limit] - Max candidates (default: config maxCandidates)
 * @param {string} [opts.subject='user']
 * @param {boolean} [opts.includeVerbatim=false] - keep word-for-word matches
 * @returns {Promise<{candidates: Array, diagnostics: Object}>}
 */
async function findActiveNeighbours(factText, opts = {}) {
  const cfg = getConfig().memory?.contradiction || {};
  const threshold = opts.threshold ?? (Number.isFinite(cfg.similarityFloor) ? cfg.similarityFloor : 0.45);
  const limit = opts.limit ?? (Number.isInteger(cfg.maxCandidates) ? cfg.maxCandidates : 15);
  const fetchLimit = Number.isInteger(cfg.vectorFetchLimit) ? cfg.vectorFetchLimit : 500;
  const subject = opts.subject ?? 'user';
  // The verbatim filter below is right for the question this function was built
  // for — "does anything already held contradict this?" — where an identical row
  // is a repeat and never a contradiction. It is exactly wrong for the corrector's
  // duplicate sweep, whose whole job is the rows that ARE identical: three
  // byte-identical machine-gun facts were structurally invisible to it, and the
  // merge phase's "identical pairs skip the judge" branch could never be reached.
  const includeVerbatim = opts.includeVerbatim === true;

  // Diagnostics exist so the record can answer "did the judge ever see it?".
  // nearestBelowFloor is the one that matters: it distinguishes "nothing was
  // close" from "something was close and the floor excluded it".
  const diagnostics = {
    subject, threshold, limit, fetchLimit,
    embedded: false, vectorHits: 0,
    rejectedInactive: 0, rejectedSubject: 0, rejectedVerbatim: 0, rejectedMissing: 0,
    aboveFloor: 0, belowFloor: 0, truncated: 0,
    nearestBelowFloor: null, error: null
  };

  try {
    const db = getSqliteDb();
    if (!db) { diagnostics.error = 'database unavailable'; return { candidates: [], diagnostics }; }

    const embedding = await generateEmbedding(factText);
    if (!embedding) { diagnostics.error = 'embedding unavailable'; return { candidates: [], diagnostics }; }
    diagnostics.embedded = true;

    const clusterTable = await getClusterEmbeddingsTable();
    if (!clusterTable) { diagnostics.error = 'vector table unavailable'; return { candidates: [], diagnostics }; }

    const results = await clusterTable
      .search(Array.from(embedding))
      .metricType('cosine')
      .limit(fetchLimit)
      .execute();
    diagnostics.vectorHits = results.length;

    const normalizedNew = factText.trim().toLowerCase();
    const seen = new Set();
    const eligible = [];

    // Prepared once, not per neighbour — the fetch limit is now in the hundreds
    // to low thousands rather than 15, so preparing inside the loop would be
    // paying the parse cost on every row.
    const memberStmt = db.prepare(
      'SELECT id, content, cluster_id, status, salience, subject FROM cluster_members WHERE id = ?'
    );

    // --- PASS 1: filter, with no cap in sight. ---
    for (const result of results) {
      const memberId = result.member_id;
      if (!memberId || seen.has(memberId)) continue;
      seen.add(memberId);

      const row = memberStmt.get(memberId);
      if (!row) { diagnostics.rejectedMissing++; continue; }
      // LanceDB retains the embeddings of inactive facts as history; SQLite is
      // the truth about what is still believed.
      if (row.status && row.status !== 'active') { diagnostics.rejectedInactive++; continue; }
      // Self-observations only contradict self-observations, user-facts only
      // user-facts.
      if ((row.subject || 'user') !== subject) { diagnostics.rejectedSubject++; continue; }
      // A verbatim duplicate is a REPEAT, not a contradiction — the repeat path
      // handles it, and putting it to the contradiction judge would only ever
      // waste a call. Callers looking FOR duplicates pass includeVerbatim.
      if (!includeVerbatim && row.content.trim().toLowerCase() === normalizedNew) { diagnostics.rejectedVerbatim++; continue; }

      const similarity = 1 - (result._distance || 0);
      eligible.push({
        memberId: row.id,
        content: row.content,
        clusterId: row.cluster_id,
        salience: row.salience ?? 5,
        similarity
      });
    }

    // --- PASS 2: threshold, then (only then) the cost ceiling. ---
    eligible.sort((a, b) => b.similarity - a.similarity);
    const above = eligible.filter(c => c.similarity >= threshold);
    const below = eligible.filter(c => c.similarity < threshold);
    diagnostics.aboveFloor = above.length;
    diagnostics.belowFloor = below.length;
    if (below.length) {
      diagnostics.nearestBelowFloor = {
        content: below[0].content, similarity: below[0].similarity, memberId: below[0].memberId
      };
    }

    const candidates = above.slice(0, limit);
    diagnostics.truncated = above.length - candidates.length;

    // --- PASS 3: pinned identity slots, past both the floor and the ceiling. ---
    // See memory.contradiction.pinIdentitySlots in db/config.js for why ranking
    // is not trusted here. Cheap: a scan of active same-subject rows, matched by
    // the same deterministic classifier the intake gate uses.
    if (opts.pinSlot && cfg.pinIdentitySlots !== false) {
      const { identityClassOf } = require('./extraction-rules');
      const already = new Set(candidates.map(c => c.memberId));
      const byId = new Map(eligible.map(c => [c.memberId, c]));
      const rows = db.prepare(
        "SELECT id, content, cluster_id, salience FROM cluster_members WHERE status = 'active' AND subject = ?"
      ).all(subject);
      for (const row of rows) {
        if (already.has(row.id)) continue;
        if (row.content.trim().toLowerCase() === normalizedNew) continue;
        const klass = identityClassOf(row.content);
        if (!klass || klass.klass !== opts.pinSlot) continue;
        const known = byId.get(row.id);
        candidates.push({
          memberId: row.id, content: row.content, clusterId: row.cluster_id,
          salience: row.salience ?? 5,
          similarity: known ? known.similarity : 0,
          pinned: opts.pinSlot
        });
        already.add(row.id);
        diagnostics.pinned = (diagnostics.pinned || 0) + 1;
      }
    }

    if (diagnostics.truncated > 0) {
      // No silent caps: a truncation that is never mentioned reads as complete
      // coverage.
      console.warn(`[Clusters] contradiction recall ceiling hit: ${above.length} active ${subject}-facts above ${threshold}, judging ${candidates.length} (${diagnostics.truncated} not judged)`);
    }

    return { candidates, diagnostics };
  } catch (error) {
    console.error('[Clusters] Error in findActiveNeighbours:', error.message);
    diagnostics.error = error.message;
    return { candidates: [], diagnostics };
  }
}

/**
 * Array-returning wrapper over findActiveNeighbours, kept because several
 * callers (gap-answer checks, the write_memory supersession lookup, the
 * self-fact path) only want the list.
 * @returns {Promise<Array<{memberId,content,clusterId,salience,similarity}>>}
 */
async function findContradictionCandidates(factText, opts = {}) {
  const { candidates } = await findActiveNeighbours(factText, opts);
  return candidates;
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
    // Lifecycle (2026-08-02): status becomes 'inactive' with an explicit reason,
    // and successor_id records what replaced it. superseded_by is written in
    // step so the Memory Map's supersede edges keep working — the two are kept
    // deliberately redundant rather than one being dropped.
    const info = db.prepare(`
      UPDATE cluster_members
      SET status = 'inactive', inactive_reason = 'superseded',
          successor_id = ?, superseded_by = ?, updated_at = ?
      WHERE id = ? AND status = 'active'
    `).run(newMemberId, newMemberId, new Date().toISOString(), oldMemberId);
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

      // No linked-cluster expansion. This used to pull three members from every
      // cluster `cluster_links` associated with this one, on a table nothing had
      // maintained since its writer was disabled — so it widened a search with
      // facts related to the query only by a stale edge. Retrieval already finds
      // related facts by similarity, which is the same question asked of data
      // that is actually current.
      const linkedMembers = [];

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
 * Get all clusters with member counts.
 *
 * TWO counts, deliberately. `member_count` is every row, ghosts included —
 * that is what the Memory Map draws, and dropping the ghosts from it would
 * empty half the map. `active_member_count` is the live corpus, and it is what
 * anything DECIDING about a cluster should read: the coherence audit spent four
 * days re-auditing two clusters that were 100% superseded facts, because
 * `member_count > maxFactsPerCluster` counted the ghosts. Same rule as the
 * split guard — inactive members are for looking at, not for acting on.
 *
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
      SELECT mc.*,
             COUNT(cm.id) as member_count,
             COALESCE(SUM(CASE WHEN cm.status = 'active' THEN 1 ELSE 0 END), 0) as active_member_count
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
 * Render the injected long-term memory block FROM SQLITE.
 *
 * This replaces reading data/memory/MEMORY.md, which until 2026-08-02 was the
 * thing actually injected as "=== Long-Term Memory ===". Keeping the injected
 * text in a file made the file a second system of record: it drifted from the
 * database (one machine-gun line in the file, three rows in SQLite), it was
 * edited by a cleanup step that could not match its own annotations, and a fact
 * retired in SQLite kept its line in the file and went on being read as current.
 *
 * Rendering per request means the injected block cannot disagree with the
 * database, because there is nothing left to disagree with.
 *
 * WHAT THE BUDGET LEAVES OUT IS RECORDED — see reportLtmTruncation below. 216
 * of 343 facts were being dropped from every single request and nothing said
 * so, which made the cap invisible to everyone including him.
 *
 * Shape is deliberately close to the old file so the chat system prompt's
 * contract still holds — "- fact (learned <when>)" lines under "## <heading>"
 * headings — including the rule that a "(learned ...)" annotation is the ONLY
 * thing SNH may quote when asked when it learned something.
 *
 * Ordering carries the truncation policy: clusters are ordered by their most
 * salient fact and facts by salience within a cluster, so when budgetText cuts
 * the tail it drops the least important facts rather than an arbitrary slice.
 *
 * SUBJECT is a list, because 'world' exists. World facts — knowledge about
 * external things that is not relational to either of them — are excluded from
 * the injected block by default (`memory.injection.includeWorld`), so the
 * caller asks for the subjects it wants rather than this function deciding.
 * Passing several renders them into one block; the cluster headings already
 * separate them, and a fact's subject is not something the reader needs spelled
 * out to use it.
 *
 * @param {Object} [opts]
 * @param {string|string[]} [opts.subject='user'] - which corpus (or corpora) to render
 * @param {number} [opts.budgetTokens] - cap, applied at CLUSTER boundaries. Omit
 *   for the whole corpus (the Self tab, exports, anything not injecting).
 * @returns {string} markdown, or '' when there are no active facts
 */
function reportLtmTruncation(kept, cut) {
  const line =
    `Injected memory is at its budget: ${kept.length} group(s) shown, ` +
    `${cut.length} left out (${cut.reduce((s, c) => s + c.facts, 0)} fact(s)). ` +
    `Not shown: ${cut.slice(0, 8).map(c => `${c.name} (${c.facts} facts, top salience ${c.top})`).join('; ')}` +
    `${cut.length > 8 ? `; and ${cut.length - 8} more` : ''}.`;
  try {
    const db = getSqliteDb();
    // Keyed on the CUT SET, so a stable corpus is silent after the first render
    // and a cluster crossing the line is news again.
    const key = `ltm-truncation:${cut.map(c => c.name).sort().join('|')}`;
    if (db) {
      const now = new Date().toISOString();
      const row = db.prepare('SELECT seen_count FROM heartbeat_anomaly_state WHERE anomaly_key = ?').get(key);
      if (row) {
        db.prepare('UPDATE heartbeat_anomaly_state SET last_seen_at = ?, seen_count = seen_count + 1 WHERE anomaly_key = ?')
          .run(now, key);
        return;   // reported once; the count is the record from here on
      }
      db.prepare('INSERT INTO heartbeat_anomaly_state (anomaly_key, first_seen_at, last_seen_at, seen_count, anomaly_text) VALUES (?, ?, ?, 1, ?)')
        .run(key, now, now, line);
    }
  } catch (err) {
    // Fail open, same as the heartbeat memo: losing the signal to bookkeeping
    // is worse than repeating it.
    console.error('[Clusters] truncation memo failed (reporting anyway):', err.message);
  }
  try {
    require('./fact-extractor').appendToOpsLog(
      line, require('path').join(require('./database').getDataDir(), 'memory', 'ops'));
  } catch { /* the console line is the floor */ }
  console.log(`[Clusters] ${line}`);
}

function renderLongTermMemory({ subject = 'user', budgetTokens = null } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return '';

    const subjects = (Array.isArray(subject) ? subject : [subject])
      .map(s => String(s || '').trim())
      .filter(Boolean);
    if (!subjects.length) return '';

    const rows = db.prepare(`
      SELECT cm.content, cm.salience, cm.created_at, cm.updated_at,
             COALESCE(mc.name, 'Other') AS cluster_name
      FROM cluster_members cm
      LEFT JOIN memory_clusters mc ON mc.id = cm.cluster_id
      WHERE cm.subject IN (${subjects.map(() => '?').join(',')}) AND cm.status = 'active'
    `).all(...subjects);
    if (rows.length === 0) return '';

    const byCluster = new Map();
    for (const r of rows) {
      if (!byCluster.has(r.cluster_name)) byCluster.set(r.cluster_name, []);
      byCluster.get(r.cluster_name).push(r);
    }

    // ORDER, and what happens at the margin.
    //
    // This sort decides what he remembers. Measured on 2026-08-12: 343 active
    // facts render to 8,770 tokens against a 3,000-token cap, so 79 clusters
    // compete for about 15 places and the ordering is not a presentation
    // detail — it is the selection rule.
    //
    // The tie-break used to be alphabetical, which meant that among clusters of
    // equal top salience, what reached him depended on the first letter of a
    // cluster name a background pass had chosen. "SNH Project Roadmap" was in;
    // "SNH System Architecture", same top salience, was out. Nobody decided
    // that. Now: top salience first, then the cluster touched most recently
    // (a corrected or re-asserted cluster is live in a way a dormant one is
    // not), and only then the name — kept as a final resort so the order is
    // total and a render is reproducible.
    const sal = r => (Number.isFinite(r.salience) ? r.salience : 5);
    const touched = f => String(f.updated_at || f.created_at || '');
    const groups = [...byCluster.entries()]
      .map(([name, facts]) => {
        facts.sort((a, b) => sal(b) - sal(a) || String(a.created_at).localeCompare(String(b.created_at)));
        return {
          name, facts,
          top: sal(facts[0]),
          touchedAt: facts.reduce((mx, f) => (touched(f) > mx ? touched(f) : mx), '')
        };
      })
      .sort((a, b) =>
        b.top - a.top ||
        b.touchedAt.localeCompare(a.touchedAt) ||
        a.name.localeCompare(b.name));

    const render = (g) => {
      const lines = [`## ${g.name}`];
      for (const f of g.facts) {
        const when = formatFactTimestamp(f.created_at);
        lines.push(when ? `- ${f.content} (learned ${when})` : `- ${f.content}`);
      }
      lines.push('');
      return lines;
    };

    // TRUNCATE AT CLUSTER BOUNDARIES, never mid-list.
    //
    // The cap used to be applied afterwards by budgetText, which slices on a
    // character offset: the last surviving cluster kept a heading and an
    // arbitrary prefix of its facts, so a cluster could appear to hold three
    // facts when it holds sixteen. Deciding here, where the groups are still
    // groups, means a cluster is either present in full or absent — and absent
    // is recorded rather than inferred.
    const out = ['# Long-Term Memory', ''];
    const kept = [];
    const cut = [];
    if (!budgetTokens) {
      for (const g of groups) { out.push(...render(g)); kept.push(g.name); }
    } else {
      let used = estTokens(out.join('\n'));
      let full = true;
      for (const g of groups) {
        const block = render(g);
        const cost = estTokens(block.join('\n'));
        // Always keep the first cluster, whatever it costs: an empty memory
        // block is worse than one over-budget block, and the total ceiling
        // downstream is what actually enforces the sum.
        if (full && (kept.length === 0 || used + cost <= budgetTokens)) {
          out.push(...block);
          used += cost;
          kept.push(g.name);
          continue;
        }
        full = false;
        cut.push({ name: g.name, facts: g.facts.length, top: g.top });
      }
    }

    if (cut.length) {
      const factsCut = cut.reduce((s, c) => s + c.facts, 0);
      // Said IN the block, because the alternative is a block that reads as
      // everything he knows. This is the same rule as the memory framing: an
      // excerpt that does not say it is an excerpt is a false negative waiting
      // to be stated with confidence.
      out.push(`…and ${factsCut} more fact(s) across ${cut.length} other group(s), not shown this turn ` +
               `to fit the context budget. They are still in your memory — look them up rather than ` +
               `concluding you have nothing.`);
      reportLtmTruncation(kept, cut);
    }
    return out.join('\n').trimEnd();
  } catch (err) {
    console.error('[Clusters] renderLongTermMemory failed:', err.message);
    return '';
  }
}

/**
 * Get self-facts (SNH's observations about itself), salience-ordered.
 * @param {Object} [opts]
 * @param {string|null} [opts.status='active'] - 'active', 'inactive', or null for
 *   all. 'superseded'/'retired' are accepted as legacy aliases and mapped onto
 *   the inactive_reason column, so existing callers keep working.
 * @param {number|null} [opts.limit=null] - max rows, or null for no limit
 * @param {string|null} [opts.claimType] - filter to one claim_type ('claim' |
 *   'declaration' | 'dissonance'); 'unclassified' matches rows with a NULL tag.
 *   Omit for all. Added for the self-coherence audit (samples 'claim' only).
 * @param {string|null} [opts.excludeClaimType] - drop rows with this claim_type
 *   (identity injection uses 'dissonance' to keep audit records out of chat).
 * @returns {Array} cluster_member rows with cluster_name + claim_type attached
 */
/**
 * Stored embeddings for a set of cluster members, keyed by member id.
 *
 * WHY THIS EXISTS. The self-fact dedup sweep called generateEmbedding()
 * for every active self-fact on every pass - 402 sequential round trips
 * to the embedding model at ~950ms each, 6.3 MINUTES per call, measured -
 * to reproduce vectors that were already sitting in cluster_embeddings.
 * Verified identical: cosine 1.000000 between a stored vector and a
 * freshly generated one, on every sample.
 *
 * It is O(n) in a corpus that only grows: 6.4 min at 402 facts, 20.6 at
 * 1,300, 35 at 2,200. Reflection pays it on every pass that concludes
 * anything about him, which is most days.
 *
 * Returns a Map(member_id -> Float32Array|number[]). A member with no
 * stored vector is simply absent, and the caller embeds that one.
 */
async function getStoredEmbeddings(memberIds = []) {
  const wanted = new Set((memberIds || []).filter(Boolean));
  const found = new Map();
  if (!wanted.size) return found;

  try {
    const table = getClusterEmbeddingsTable();
    if (!table) return found;

    // One scan, filtered in memory. The alternative - a filter string per
    // id - is n queries, which is the shape being removed.
    const rows = await table.filter('true').limit(100000).execute();
    for (const row of rows) {
      if (row && row.member_id && wanted.has(row.member_id) && row.vector) {
        found.set(row.member_id, row.vector);
      }
    }
  } catch (err) {
    // No vectors is a reason to embed, never a reason to fail.
    console.error('[Clusters] could not read stored embeddings:', err.message);
  }
  return found;
}


function getSelfFacts({ status = 'active', limit = null, claimType = null, excludeClaimType = null } = {}) {
  try {
    const db = getSqliteDb();
    if (!db) return [];

    let sql = `
      SELECT cm.id, cm.content, cm.salience, cm.status, cm.superseded_by,
             cm.inactive_reason, cm.successor_id,
             cm.created_at, cm.updated_at, cm.cluster_id, cm.source, cm.claim_type,
             cm.locked, cm.locked_at, cm.lock_category,
             cm.conversation_id, cm.message_id, cm.verbatim_source_text,
             cm.input_modality, cm.salience_rationale,
             mc.name AS cluster_name
      FROM cluster_members cm
      LEFT JOIN memory_clusters mc ON mc.id = cm.cluster_id
      WHERE cm.subject = 'self'`;
    const params = [];
    // Legacy aliases: callers that still ask for 'superseded' or 'retired' get
    // the matching inactive rows rather than silently getting none.
    if (status === 'superseded' || status === 'retired') {
      sql += " AND cm.status = 'inactive' AND cm.inactive_reason = ?";
      params.push(status === 'retired' ? 'retracted' : 'superseded');
    } else if (status) {
      sql += ' AND cm.status = ?'; params.push(status);
    }
    if (claimType === 'unclassified') {
      sql += ' AND cm.claim_type IS NULL';
    } else if (claimType) {
      sql += ' AND cm.claim_type = ?'; params.push(claimType);
    }
    if (excludeClaimType) {
      sql += ' AND (cm.claim_type IS NULL OR cm.claim_type != ?)'; params.push(excludeClaimType);
    }
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

    // Association is computed at query time now — GET /api/memory/graph/
    // neighbours/:clusterId — so there is no stored edge to return. The field is
    // kept and empty rather than removed: a caller that reads it gets "no stored
    // links", which is true, instead of undefined.
    return {
      ...cluster,
      members,
      linkedClusters: []
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
  findActiveNeighbours,
  supersedeFact,
  updateFactSalience,
  searchClusters,
  getClusters,
  getCluster,
  getSelfFacts,
  getStoredEmbeddings,
  generateEmbedding,
  cosineSimilarity,
  generateClusterNameFromMembers,
  generateClusterNameLLM,
  sanitizeClusterName,
  matchCuratedCategory,
  isValidClusterName,
  renameAllClusters,
  mergeByName,
  mergeSingletons,
  renderLongTermMemory
};
