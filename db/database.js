/**
 * Database layer for chat history and semantic memory
 * Combines SQLite (structured data) and LanceDB (vector embeddings)
 */

const Database = require('better-sqlite3');
const lancedb = require('vectordb');
const path = require('path');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { getConfig, getProviderInstance } = require('./config');

// Database paths
const DATA_DIR = path.join(__dirname, '../data');
const SQLITE_PATH = path.join(DATA_DIR, 'chat.db');
const LANCEDB_PATH = path.join(DATA_DIR, 'lancedb');

// Ensure data directories exist
function ensureDirectories() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  if (!fs.existsSync(LANCEDB_PATH)) {
    fs.mkdirSync(LANCEDB_PATH, { recursive: true });
  }
}

// Global database instances
let sqliteDb = null;
let vectorDb = null;
let embeddingsTable = null;
let clusterEmbeddingsTable = null;

// ============ SQLite Setup ============

/**
 * Initialize SQLite database with schema
 * @returns {Database} SQLite database instance
 */
function initDatabase() {
  try {
    ensureDirectories();

    // Initialize SQLite
    sqliteDb = new Database(SQLITE_PATH);
    sqliteDb.pragma('journal_mode = WAL'); // Write-Ahead Logging for better concurrency
    sqliteDb.pragma('foreign_keys = ON'); // Enable foreign key constraints

    // Create conversations table
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        model_used TEXT
      )
    `);

    // Create messages table
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model TEXT,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
      )
    `);

    // Create indexes for performance
    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
      ON messages(conversation_id)
    `);

    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
      ON conversations(updated_at DESC)
    `);

    // Create FTS5 virtual table for full-text search
    sqliteDb.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        conversation_id UNINDEXED,
        message_id UNINDEXED
      )
    `);

    // === UPGRADE 4: Memory Clustering tables ===
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS memory_clusters (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS cluster_members (
        id TEXT PRIMARY KEY,
        cluster_id TEXT NOT NULL,
        content TEXT NOT NULL,
        source TEXT,
        importance REAL DEFAULT 0.5,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (cluster_id) REFERENCES memory_clusters(id)
      )
    `);

    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS cluster_links (
        id TEXT PRIMARY KEY,
        cluster_a TEXT NOT NULL,
        cluster_b TEXT NOT NULL,
        strength REAL DEFAULT 0.5,
        FOREIGN KEY (cluster_a) REFERENCES memory_clusters(id),
        FOREIGN KEY (cluster_b) REFERENCES memory_clusters(id)
      )
    `);

    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster_id
      ON cluster_members(cluster_id)
    `);

    console.log('SQLite database initialized successfully');

    // Backfill FTS table with existing messages
    backfillFTS();

    return sqliteDb;
  } catch (error) {
    console.error('Failed to initialize SQLite database:', error.message);
    throw error;
  }
}

/**
 * Get list of all conversations with preview
 * @returns {Array} Array of conversation objects
 */
function getConversations() {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare(`
      SELECT
        c.id,
        c.title,
        c.created_at,
        c.updated_at,
        c.model_used,
        (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY timestamp ASC LIMIT 1) as preview
      FROM conversations c
      ORDER BY c.updated_at DESC
    `);

    return stmt.all();
  } catch (error) {
    console.error('Error fetching conversations:', error.message);
    throw error;
  }
}

/**
 * Get full conversation with all messages
 * @param {string} id - Conversation ID
 * @returns {Object|null} Conversation object with messages array
 */
function getConversation(id) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    // Get conversation metadata
    const conversationStmt = sqliteDb.prepare(`
      SELECT id, title, created_at, updated_at, model_used
      FROM conversations
      WHERE id = ?
    `);
    const conversation = conversationStmt.get(id);

    if (!conversation) {
      return null;
    }

    // Get all messages for this conversation
    const messagesStmt = sqliteDb.prepare(`
      SELECT id, conversation_id, role, content, timestamp, model
      FROM messages
      WHERE conversation_id = ?
      ORDER BY timestamp ASC
    `);
    const messages = messagesStmt.all(id);

    return {
      ...conversation,
      messages
    };
  } catch (error) {
    console.error('Error fetching conversation:', error.message);
    throw error;
  }
}

/**
 * Create a new conversation
 * @param {string} title - Conversation title
 * @param {string} model_used - Model identifier
 * @returns {string} New conversation ID
 */
function createConversation(title, model_used) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const id = randomUUID();
    const stmt = sqliteDb.prepare(`
      INSERT INTO conversations (id, title, model_used)
      VALUES (?, ?, ?)
    `);

    stmt.run(id, title, model_used);
    return id;
  } catch (error) {
    console.error('Error creating conversation:', error.message);
    throw error;
  }
}

/**
 * Delete a conversation and all its messages
 * @param {string} id - Conversation ID
 */
function deleteConversation(id) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare('DELETE FROM conversations WHERE id = ?');
    stmt.run(id);

    // Also delete vector embeddings
    deleteConversationEmbeddings(id).catch(err => {
      console.error('Error deleting embeddings:', err.message);
    });
  } catch (error) {
    console.error('Error deleting conversation:', error.message);
    throw error;
  }
}

/**
 * Update conversation title
 * @param {string} id - Conversation ID
 * @param {string} title - New title
 */
function updateConversationTitle(id, title) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare(`
      UPDATE conversations
      SET title = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `);

    stmt.run(title, id);
  } catch (error) {
    console.error('Error updating conversation title:', error.message);
    throw error;
  }
}

/**
 * Add a message to a conversation
 * @param {string} conversation_id - Conversation ID
 * @param {string} role - Message role (user/assistant/system)
 * @param {string} content - Message content
 * @param {string} model - Model used (optional)
 * @returns {string} New message ID
 */
function addMessage(conversation_id, role, content, model = null) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const id = randomUUID();
    const stmt = sqliteDb.prepare(`
      INSERT INTO messages (id, conversation_id, role, content, model)
      VALUES (?, ?, ?, ?, ?)
    `);

    stmt.run(id, conversation_id, role, content, model);

    // Also insert into FTS5 table for full-text search
    try {
      const ftsStmt = sqliteDb.prepare(`
        INSERT INTO messages_fts (content, conversation_id, message_id)
        VALUES (?, ?, ?)
      `);
      ftsStmt.run(content, conversation_id, id);
    } catch (ftsError) {
      console.error('[FTS] Error adding message to FTS:', ftsError.message);
    }

    // Update conversation's updated_at timestamp
    updateConversationTimestamp(conversation_id);

    return id;
  } catch (error) {
    console.error('Error adding message:', error.message);
    throw error;
  }
}

/**
 * Update conversation's updated_at timestamp
 * @param {string} id - Conversation ID
 */
function updateConversationTimestamp(id) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare(`
      UPDATE conversations
      SET updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `);

    stmt.run(id);
  } catch (error) {
    console.error('Error updating conversation timestamp:', error.message);
    throw error;
  }
}

// ============ LanceDB (Vector Store) Setup ============

/**
 * Initialize LanceDB for vector embeddings
 * @returns {Promise<Object>} LanceDB table instance
 */
async function initVectorStore() {
  try {
    ensureDirectories();

    // Connect to LanceDB
    vectorDb = await lancedb.connect(LANCEDB_PATH);

    // Check if table exists
    const tableNames = await vectorDb.tableNames();

    if (tableNames.includes('message_embeddings')) {
      // Open existing table
      embeddingsTable = await vectorDb.openTable('message_embeddings');
      console.log('LanceDB: Opened existing message_embeddings table');
    } else {
      // Create new table with schema
      // Note: We need at least one record to create the table with schema
      // Use regular array for LanceDB compatibility
      const sampleData = [{
        id: randomUUID(),
        message_id: 'sample',
        conversation_id: 'sample',
        text: 'Sample initialization text',
        role: 'system',
        vector: Array(768).fill(0) // nomic-embed-text produces 768-dim vectors
      }];

      embeddingsTable = await vectorDb.createTable('message_embeddings', sampleData);

      // Delete the sample record
      await embeddingsTable.delete('id = "' + sampleData[0].id + '"');

      console.log('LanceDB: Created new message_embeddings table');
    }

    // === UPGRADE 4: Cluster embeddings table ===
    if (tableNames.includes('cluster_embeddings')) {
      clusterEmbeddingsTable = await vectorDb.openTable('cluster_embeddings');
      console.log('LanceDB: Opened existing cluster_embeddings table');
    } else {
      const sampleCluster = [{
        id: randomUUID(),
        member_id: 'sample',
        cluster_id: 'sample',
        content: 'Sample cluster text',
        vector: Array(768).fill(0)
      }];
      clusterEmbeddingsTable = await vectorDb.createTable('cluster_embeddings', sampleCluster);
      await clusterEmbeddingsTable.delete('id = "' + sampleCluster[0].id + '"');
      console.log('LanceDB: Created new cluster_embeddings table');
    }

    return embeddingsTable;
  } catch (error) {
    console.error('Failed to initialize LanceDB:', error.message);
    throw error;
  }
}

/**
 * Generate embedding using Ollama's nomic-embed-text model
 * @param {string} text - Text to embed
 * @returns {Promise<Float32Array>} Embedding vector
 */
async function generateEmbedding(text) {
  try {
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
      })
    });

    if (!response.ok) {
      throw new Error(`Embedding API returned ${response.status}`);
    }

    const data = await response.json();

    if (!data.embedding || !Array.isArray(data.embedding)) {
      throw new Error('Invalid embedding response format');
    }

    return new Float32Array(data.embedding);
  } catch (error) {
    console.error('Error generating embedding:', error.message);
    throw error;
  }
}

/**
 * Add embedding to vector store
 * @param {string} message_id - Message ID from SQLite
 * @param {string} conversation_id - Conversation ID
 * @param {string} text - Message text
 * @param {string} role - Message role
 * @param {Float32Array} vector - Embedding vector
 * @returns {Promise<void>}
 */
async function addEmbedding(message_id, conversation_id, text, role, vector) {
  try {
    if (!embeddingsTable) {
      throw new Error('Vector store not initialized. Call initVectorStore() first.');
    }

    // Convert Float32Array to regular array for LanceDB compatibility
    const vectorArray = Array.from(vector);

    const record = {
      id: randomUUID(),
      message_id,
      conversation_id,
      text,
      role,
      vector: vectorArray
    };

    await embeddingsTable.add([record]);
  } catch (error) {
    console.error('Error adding embedding:', error.message);
    throw error;
  }
}

/**
 * Search for similar messages across conversations
 * @param {Float32Array} vector - Query vector
 * @param {string} excludeConversationId - Conversation ID to exclude from results
 * @param {number} limit - Maximum number of results
 * @param {number} threshold - Similarity threshold (0-1, higher is more similar)
 * @returns {Promise<Array>} Array of similar messages
 */
async function searchSimilar(vector, excludeConversationId, limit = 5, threshold) {
  if (threshold === undefined) {
    threshold = getConfig().memory.similarityThreshold;
  }
  try {
    if (!embeddingsTable) {
      throw new Error('Vector store not initialized. Call initVectorStore() first.');
    }

    // Convert Float32Array to regular array for LanceDB compatibility
    const vectorArray = Array.from(vector);

    // Perform vector search with cosine distance metric
    // Cosine distance returns values 0-1 where 0 = identical, 1 = opposite
    const results = await embeddingsTable
      .search(vectorArray)
      .metricType('cosine')
      .limit(limit * 2) // Get more results to filter
      .execute();

    // Filter out current conversation and apply threshold
    const filtered = results
      .filter(result => {
        // Exclude current conversation
        if (result.conversation_id === excludeConversationId) {
          return false;
        }
        // Apply similarity threshold
        // Convert cosine distance to similarity: similarity = 1 - distance
        const similarity = 1 - result._distance;
        return similarity >= threshold;
      })
      .slice(0, limit)
      .map(result => ({
        message_id: result.message_id,
        conversation_id: result.conversation_id,
        text: result.text,
        role: result.role,
        similarity: 1 - result._distance
      }));

    return filtered;
  } catch (error) {
    console.error('Error searching similar messages:', error.message);
    // Return empty array on error to not break the application
    return [];
  }
}

/**
 * Delete all embeddings for a conversation
 * @param {string} conversation_id - Conversation ID
 * @returns {Promise<void>}
 */
async function deleteConversationEmbeddings(conversation_id) {
  try {
    if (!embeddingsTable) {
      // If table not initialized, silently skip
      return;
    }

    await embeddingsTable.delete(`conversation_id = "${conversation_id}"`);
  } catch (error) {
    console.error('Error deleting conversation embeddings:', error.message);
    // Don't throw - embedding deletion is not critical
  }
}

// ============ Full-Text Search (FTS5) Functions ============

/**
 * Backfill FTS table with existing messages
 * Called during database initialization
 */
function backfillFTS() {
  try {
    if (!sqliteDb) {
      console.error('[FTS] Cannot backfill: database not initialized');
      return;
    }

    // Count rows in each table
    const ftsCount = sqliteDb.prepare('SELECT COUNT(*) as count FROM messages_fts').get().count;
    const messagesCount = sqliteDb.prepare('SELECT COUNT(*) as count FROM messages').get().count;

    console.log(`[FTS] Current state: ${ftsCount} FTS rows, ${messagesCount} total messages`);

    if (ftsCount >= messagesCount) {
      console.log('[FTS] FTS table is up to date, no backfill needed');
      return;
    }

    // Find messages not yet in FTS
    const missingMessages = sqliteDb.prepare(`
      SELECT m.id, m.conversation_id, m.content
      FROM messages m
      WHERE m.id NOT IN (SELECT message_id FROM messages_fts)
    `).all();

    if (missingMessages.length === 0) {
      console.log('[FTS] No missing messages to backfill');
      return;
    }

    console.log(`[FTS] Backfilling ${missingMessages.length} messages into FTS table`);

    const insertStmt = sqliteDb.prepare(`
      INSERT INTO messages_fts (content, conversation_id, message_id)
      VALUES (?, ?, ?)
    `);

    const insertMany = sqliteDb.transaction((messages) => {
      for (const msg of messages) {
        insertStmt.run(msg.content, msg.conversation_id, msg.id);
      }
    });

    insertMany(missingMessages);

    console.log(`[FTS] Successfully backfilled ${missingMessages.length} messages`);
  } catch (error) {
    console.error('[FTS] Error during backfill:', error.message);
    // Don't throw - backfill failure should not prevent app from starting
  }
}

/**
 * Sanitize FTS5 query to prevent MATCH syntax errors and column-name injection.
 * Wraps each term in double quotes so FTS5 treats them as literal strings
 * instead of interpreting them as column names or operators.
 * @param {string} query - Raw query string
 * @returns {string} Sanitized query safe for FTS5 MATCH
 */
function sanitizeFTSQuery(query) {
  if (!query || typeof query !== 'string') {
    return '"search"';
  }

  // Strip everything except word chars, spaces, and hyphens
  const cleaned = query.replace(/[^\w\s\-]/g, ' ');

  // Split into individual terms, quote each one for FTS5 safety
  const terms = cleaned
    .split(/\s+/)
    .filter(w => w.length > 0)
    .map(w => '"' + w.replace(/"/g, '') + '"');

  if (terms.length === 0) {
    return '"search"';
  }

  return terms.join(' ');
}

/**
 * Search messages using BM25 ranking (FTS5)
 * @param {string} query - Search query
 * @param {string} excludeConversationId - Conversation ID to exclude from results
 * @param {number} limit - Maximum number of results
 * @returns {Array} Array of search results with BM25 scores
 */
function searchBM25(query, excludeConversationId, limit = 10) {
  try {
    if (!sqliteDb) {
      console.error('[BM25] Database not initialized');
      return [];
    }

    const sanitized = sanitizeFTSQuery(query);
    console.log(`[BM25] Searching for: "${sanitized}" (original: "${query}")`);

    const stmt = sqliteDb.prepare(`
      SELECT
        message_id,
        conversation_id,
        content,
        bm25(messages_fts) as score
      FROM messages_fts
      WHERE messages_fts MATCH ? AND conversation_id != ?
      ORDER BY bm25(messages_fts)
      LIMIT ?
    `);

    const results = stmt.all(sanitized, excludeConversationId, limit);
    console.log(`[BM25] Found ${results.length} results`);

    // BM25 scores are negative (more negative = better match)
    // Normalize to 0-1 range
    if (results.length === 0) {
      return [];
    }

    // Find the best (most negative) score
    const maxAbsScore = Math.max(...results.map(r => Math.abs(r.score)));

    return results.map(result => ({
      message_id: result.message_id,
      conversation_id: result.conversation_id,
      content: result.content,
      score: maxAbsScore > 0 ? Math.abs(result.score) / maxAbsScore : 1.0
    }));
  } catch (error) {
    console.error('[BM25] Search error:', error.message);
    return [];
  }
}

/**
 * Hybrid search combining vector similarity and BM25 keyword search
 * @param {string} query - Search query
 * @param {string} excludeConversationId - Conversation ID to exclude from results
 * @param {number} limit - Maximum number of results to return
 * @param {number} threshold - Similarity threshold for vector search (0-1)
 * @returns {Promise<Array>} Array of search results with combined scores
 */
async function hybridSearch(query, excludeConversationId, limit = 5, threshold) {
  if (threshold === undefined) {
    threshold = getConfig().memory.similarityThreshold;
  }
  try {
    console.log(`[HybridSearch] Query: "${query}", limit: ${limit}, threshold: ${threshold}`);

    // Step 1: Generate embedding for query
    let vectorResults = [];
    try {
      const embedding = await generateEmbedding(query);
      console.log('[HybridSearch] Generated query embedding');

      // Step 2: Run vector search
      vectorResults = await searchSimilar(embedding, excludeConversationId, limit * 2, threshold);
      console.log(`[HybridSearch] Vector search found ${vectorResults.length} results`);
    } catch (embeddingError) {
      console.error('[HybridSearch] Vector search failed:', embeddingError.message);
      // Continue with BM25 only
    }

    // Step 3: Run BM25 search
    const bm25Results = searchBM25(query, excludeConversationId, limit * 2);
    console.log(`[HybridSearch] BM25 search found ${bm25Results.length} results`);

    // Step 4: Fuse results by message_id
    const resultsMap = new Map();

    // Add vector results
    for (const result of vectorResults) {
      resultsMap.set(result.message_id || result.text, {
        text: result.text,
        role: result.role,
        conversation_id: result.conversation_id,
        message_id: result.message_id,
        vectorScore: result.similarity,
        bm25Score: 0
      });
    }

    // Merge BM25 results
    for (const result of bm25Results) {
      const key = result.message_id || result.content;
      if (resultsMap.has(key)) {
        // Result appears in both searches
        resultsMap.get(key).bm25Score = result.score;
      } else {
        // BM25-only result - get full message details from SQLite
        try {
          const msgStmt = sqliteDb.prepare(`
            SELECT id, conversation_id, role, content
            FROM messages
            WHERE id = ?
          `);
          const msg = msgStmt.get(result.message_id);

          if (msg) {
            resultsMap.set(key, {
              text: msg.content,
              role: msg.role,
              conversation_id: msg.conversation_id,
              message_id: msg.id,
              vectorScore: 0,
              bm25Score: result.score
            });
          }
        } catch (dbError) {
          console.error('[HybridSearch] Error fetching message details:', dbError.message);
        }
      }
    }

    // Step 5: Calculate combined scores and sort
    const config = getConfig();
    const weightVector = config.memory.hybridSearchWeights.vector;
    const weightBM25 = config.memory.hybridSearchWeights.bm25;

    const combinedResults = Array.from(resultsMap.values()).map(result => {
      const finalScore = (weightVector * result.vectorScore) + (weightBM25 * result.bm25Score);

      // Determine source
      let source = 'hybrid';
      if (result.vectorScore > 0 && result.bm25Score === 0) {
        source = 'vector';
      } else if (result.bm25Score > 0 && result.vectorScore === 0) {
        source = 'bm25';
      }

      return {
        text: result.text,
        role: result.role,
        conversation_id: result.conversation_id,
        similarity: finalScore,
        source,
        // Include component scores for debugging
        _vectorScore: result.vectorScore,
        _bm25Score: result.bm25Score
      };
    });

    // Sort by combined score descending
    combinedResults.sort((a, b) => b.similarity - a.similarity);

    // Return top N results
    const topResults = combinedResults.slice(0, limit);
    console.log(`[HybridSearch] Returning ${topResults.length} results (${topResults.filter(r => r.source === 'hybrid').length} hybrid, ${topResults.filter(r => r.source === 'vector').length} vector-only, ${topResults.filter(r => r.source === 'bm25').length} bm25-only)`);

    return topResults;
  } catch (error) {
    console.error('[HybridSearch] Error:', error.message);
    return [];
  }
}

// ============ Memory Files Functions ============

/**
 * Load memory files for system prompt injection
 * @param {string} memoryDir - Directory containing memory files
 * @returns {Object} Object containing memory file contents
 */
function loadMemoryFiles(memoryDir = path.join(__dirname, '../data/memory')) {
  try {
    console.log(`[MemoryFiles] Loading from: ${memoryDir}`);

    const result = {
      memory: null,
      user: null,
      dailyToday: null,
      dailyYesterday: null
    };

    // Helper to safely read a file
    const readFileSafe = (filePath, label) => {
      try {
        if (fs.existsSync(filePath)) {
          const content = fs.readFileSync(filePath, 'utf-8');
          console.log(`[MemoryFiles] Loaded ${label}: ${content.length} chars`);
          return content;
        } else {
          console.log(`[MemoryFiles] ${label} not found: ${filePath}`);
          return null;
        }
      } catch (error) {
        console.error(`[MemoryFiles] Error reading ${label}:`, error.message);
        return null;
      }
    };

    // Load MEMORY.md
    result.memory = readFileSafe(path.join(memoryDir, 'MEMORY.md'), 'MEMORY.md');
    if (result.memory) {
      console.log(`[MemoryFiles] MEMORY.md preview (${result.memory.length} chars): "${result.memory.substring(0, 200)}${result.memory.length > 200 ? '...' : ''}"`);
    }

    // Load USER.md
    result.user = readFileSafe(path.join(memoryDir, 'USER.md'), 'USER.md');

    // Load today's daily note
    const today = new Date();
    const todayFilename = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}.md`;
    result.dailyToday = readFileSafe(path.join(memoryDir, 'daily', todayFilename), `daily/${todayFilename}`);

    // Load yesterday's daily note
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayFilename = `${yesterday.getFullYear()}-${String(yesterday.getMonth() + 1).padStart(2, '0')}-${String(yesterday.getDate()).padStart(2, '0')}.md`;
    result.dailyYesterday = readFileSafe(path.join(memoryDir, 'daily', yesterdayFilename), `daily/${yesterdayFilename}`);

    return result;
  } catch (error) {
    console.error('[MemoryFiles] Error loading memory files:', error.message);
    return {
      memory: null,
      user: null,
      dailyToday: null,
      dailyYesterday: null
    };
  }
}

// ============ Accessors for sub-modules ============

function getSqliteDb() { return sqliteDb; }
function getClusterEmbeddingsTable() { return clusterEmbeddingsTable; }

/**
 * Drop and recreate the cluster_embeddings LanceDB table
 * Used by rebuild scripts to get a clean table with a fresh index
 * @returns {Promise<Object>} New cluster embeddings table
 */
async function resetClusterEmbeddingsTable() {
  if (!vectorDb) {
    throw new Error('Vector store not initialized. Call initVectorStore() first.');
  }

  // Drop existing table
  const tableNames = await vectorDb.tableNames();
  if (tableNames.includes('cluster_embeddings')) {
    await vectorDb.dropTable('cluster_embeddings');
    console.log('LanceDB: Dropped cluster_embeddings table');
  }

  // Recreate with schema
  const sampleCluster = [{
    id: randomUUID(),
    member_id: 'sample',
    cluster_id: 'sample',
    content: 'Sample cluster text',
    vector: Array(768).fill(0)
  }];
  clusterEmbeddingsTable = await vectorDb.createTable('cluster_embeddings', sampleCluster);
  await clusterEmbeddingsTable.delete('id = "' + sampleCluster[0].id + '"');
  console.log('LanceDB: Recreated cluster_embeddings table');

  return clusterEmbeddingsTable;
}

// ============ Exports ============

module.exports = {
  // Initialization
  initDatabase,
  initVectorStore,

  // SQLite functions
  getConversations,
  getConversation,
  createConversation,
  deleteConversation,
  updateConversationTitle,
  addMessage,
  updateConversationTimestamp,

  // Vector store functions
  generateEmbedding,
  addEmbedding,
  searchSimilar,
  deleteConversationEmbeddings,

  // Full-text search functions
  searchBM25,
  hybridSearch,
  backfillFTS,

  // Memory files
  loadMemoryFiles,

  // Accessors for sub-modules
  getSqliteDb,
  getClusterEmbeddingsTable,
  resetClusterEmbeddingsTable
};
