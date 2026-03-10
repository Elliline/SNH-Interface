const fs = require('fs');
const path = require('path');
const { getConfig, getProviderInstance } = require('./config');

const MEMORY_DIR = path.join(__dirname, '../data/memory');
const DAILY_DIR = path.join(MEMORY_DIR, 'daily');

// ============ Embedding Helpers ============

/**
 * Generate embedding using the configured embedding provider/model (local to this module)
 * @param {string} text - Text to embed
 * @returns {Promise<number[]|null>} Embedding vector or null on failure
 */
async function generateFactEmbedding(text) {
  try {
    const config = getConfig();
    const embInst = getProviderInstance(config.models.embedding.provider, config.models.embedding.instance);
    const embeddingHost = embInst ? embInst.host : 'http://localhost:11434';
    const embeddingModel = config.models.embedding.model;
    const response = await fetch(`${embeddingHost}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: embeddingModel, prompt: text }),
      signal: AbortSignal.timeout(10000)
    });
    if (!response.ok) return null;
    const data = await response.json();
    return data.embedding || null;
  } catch (error) {
    console.error('[FactExtractor] Embedding error:', error.message);
    return null;
  }
}

/**
 * Cosine similarity between two vectors
 */
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Parse MEMORY.md into sections
 * Returns array of { heading, startIndex, endIndex, factLines }
 */
function parseMemorySections(content) {
  const sections = [];
  const lines = content.split('\n');
  let currentSection = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('## ')) {
      if (currentSection) {
        currentSection.endLine = i - 1;
        sections.push(currentSection);
      }
      currentSection = {
        heading: line.replace('## ', '').trim(),
        startLine: i,
        endLine: -1,
        factLines: []
      };
    } else if (currentSection && line.startsWith('- ')) {
      currentSection.factLines.push(line.substring(2).trim());
    }
  }
  if (currentSection) {
    currentSection.endLine = lines.length - 1;
    sections.push(currentSection);
  }

  return sections;
}

/**
 * Extract all fact lines (lines starting with "- ") from content
 */
function extractAllFactLines(content) {
  return content.split('\n')
    .filter(line => line.startsWith('- '))
    .map(line => line.substring(2).trim());
}

/**
 * Extract facts from a chat exchange using the same model/provider
 * @param {string} userMessage - The user's message
 * @param {string} assistantMessage - The assistant's response
 * @param {string} provider - LLM provider (ollama, claude, openai, grok, llamacpp, squatchserve)
 * @param {string} model - Model name
 * @param {string} apiKey - API key for the provider (if required)
 * @param {string} ollamaHost - Ollama host URL
 * @returns {Promise<string[]>} Array of extracted facts
 */
async function extractFacts(userMessage, assistantMessage, provider, model, apiKey, ollamaHost) {
  try {
    console.log(`[FactExtractor] Extracting facts using ${provider}/${model}`);

    const systemPrompt = `You are a fact extraction system. Extract facts about the USER from what the USER said in the chat exchange below.

RULES:
- Return ONLY a valid JSON array of strings, or [] if nothing worth remembering.
- Each fact MUST be a complete, self-contained sentence with full context.
- Only extract facts that the USER has stated about themselves, their life, their preferences, or their projects.
- Do NOT extract general knowledge, web search results, trivia, or information the AI provided.
- Do NOT extract facts from the Assistant's response — only from what the User said.
- Include: names, preferences, technical specs, relationships, decisions, project details the user mentions.
- Do NOT include: greetings, casual chat, temporary context, questions without answers.
- Write facts as "User has..." or "User prefers..." — never "Assistant has...".

GOOD examples (facts the user stated about themselves):
["User has 4 dogs: Casper, Cece, Calypso, and Erika", "User is migrating from Syncro to Kaseya for RMM", "User's AI server has dual RTX 3090s with 48GB total VRAM"]

BAD examples (AI-provided info, fragments, or general knowledge):
["A viral TikTok trend featuring Mini Huskies", "Constantinople fell in 1453", "RTX 3090", "The weather is nice"]

Every extracted fact must be something the USER told you, not something you told the user.`;

    const exchange = `USER MESSAGE:\n${userMessage}\n\nASSISTANT RESPONSE (for context only — do NOT extract facts from this):\n${assistantMessage}`;

    let response;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const config = getConfig();
      switch (provider.toLowerCase()) {
        case 'ollama': {
          const inst = getProviderInstance('ollama', config.models.extraction.instance);
          response = await extractFromOllama(systemPrompt, exchange, model, (inst && inst.host) || ollamaHost || 'http://localhost:11434', controller.signal);
          break;
        }
        case 'claude':
          response = await extractFromClaude(systemPrompt, exchange, model, apiKey, controller.signal);
          break;
        case 'grok':
          response = await extractFromGrok(systemPrompt, exchange, model, apiKey, controller.signal);
          break;
        case 'openai':
          response = await extractFromOpenAI(systemPrompt, exchange, model, apiKey, controller.signal);
          break;
        case 'llamacpp': {
          const inst = getProviderInstance('llamacpp', config.models.extraction.instance);
          response = await extractFromLlamacpp(systemPrompt, exchange, model, (inst && inst.host) || ollamaHost || 'http://localhost:8080', controller.signal);
          break;
        }
        case 'squatchserve': {
          const inst = getProviderInstance('squatchserve', config.models.extraction.instance);
          response = await extractFromSquatchServe(systemPrompt, exchange, model, (inst && inst.host) || ollamaHost || 'http://localhost:8080', controller.signal);
          break;
        }
        case 'vllm': {
          const inst = getProviderInstance('vllm', config.models.extraction.instance);
          response = await extractFromLlamacpp(systemPrompt, exchange, model, (inst && inst.host) || ollamaHost || 'http://localhost:8000', controller.signal);
          break;
        }
        default:
          console.log(`[FactExtractor] Unsupported provider: ${provider}`);
          return [];
      }
    } finally {
      clearTimeout(timeoutId);
    }

    // Parse JSON array from response (handle markdown code blocks)
    const facts = parseFactsFromResponse(response);
    console.log(`[FactExtractor] Extracted ${facts.length} facts`);
    return facts;

  } catch (error) {
    if (error.name === 'AbortError') {
      console.error('[FactExtractor] Extraction timeout after 30s');
    } else {
      console.error('[FactExtractor] Error extracting facts:', error.message);
    }
    return [];
  }
}

/**
 * Extract facts using Ollama
 */
async function extractFromOllama(systemPrompt, exchange, model, host, signal) {
  const response = await fetch(`${host}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: exchange }
      ],
      stream: false
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.status}`);
  }

  const data = await response.json();
  return data.message?.content || '';
}

/**
 * Extract facts using Claude
 */
async function extractFromClaude(systemPrompt, exchange, model, apiKey, signal) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      system: systemPrompt,
      messages: [
        { role: 'user', content: exchange }
      ],
      stream: false
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Claude API error: ${response.status}`);
  }

  const data = await response.json();
  return data.content?.[0]?.text || '';
}

/**
 * Extract facts using Grok
 */
async function extractFromGrok(systemPrompt, exchange, model, apiKey, signal) {
  const response = await fetch('https://api.x.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: exchange }
      ],
      stream: false
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Grok API error: ${response.status}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Extract facts using OpenAI
 */
async function extractFromOpenAI(systemPrompt, exchange, model, apiKey, signal) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: exchange }
      ],
      stream: false
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Extract facts using Llama.cpp
 */
async function extractFromLlamacpp(systemPrompt, exchange, model, host, signal) {
  const response = await fetch(`${host}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: exchange }
      ],
      stream: false
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Llama.cpp API error: ${response.status}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Extract facts using SquatchServe
 */
async function extractFromSquatchServe(systemPrompt, exchange, model, host, signal) {
  const response = await fetch(`${host}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: exchange }
      ],
      stream: false
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`SquatchServe API error: ${response.status}`);
  }

  const data = await response.json();
  return data.message?.content || '';
}

/**
 * Parse facts from LLM response (handles markdown code blocks and Python-style arrays)
 */
function parseFactsFromResponse(response) {
  try {
    // Strip markdown code fences before parsing
    response = response.replace(/```(?:json)?\s*\n?([\s\S]*?)```/g, '$1').trim();

    // Try to find JSON array in the response
    const jsonMatch = response.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      console.log('[FactExtractor] No JSON array found in response');
      return [];
    }

    let jsonStr = jsonMatch[0];

    // Try parsing as valid JSON first
    let facts;
    try {
      facts = JSON.parse(jsonStr);
    } catch (parseErr) {
      // Handle Python-style single-quoted arrays: ['fact', 'fact']
      // Convert structural single quotes to double quotes while preserving
      // apostrophes inside strings (e.g. "User's name")
      console.log('[FactExtractor] JSON parse failed, trying Python-style fixup');
      jsonStr = jsonStr
        .replace(/\[\s*'/g, '["')           // [' → ["
        .replace(/'\s*\]/g, '"]')           // '] → "]
        .replace(/'\s*,\s*'/g, '", "')     // ', ' → ", "
        .replace(/'\s*,\s*"/g, '", "')     // ', " → ", "
        .replace(/"\s*,\s*'/g, '", "');    // ", ' → ", "
      try {
        facts = JSON.parse(jsonStr);
      } catch (retryErr) {
        console.error('[FactExtractor] Python-style fixup also failed:', retryErr.message);
        return [];
      }
    }

    if (!Array.isArray(facts)) {
      console.log('[FactExtractor] Parsed JSON is not an array');
      return [];
    }

    // Filter out empty strings, non-strings, and non-personal facts
    return facts.filter(f => {
      if (typeof f !== 'string' || f.trim().length === 0) return false;
      const t = f.trim();

      // Reject facts where the assistant is the subject — these are hallucinations
      if (/^(the )?assistant\b/i.test(t)) {
        console.log(`[FactExtractor] Filtered out assistant-subject fact: "${f}"`);
        return false;
      }

      // Reject "User should be aware/know" — these are AI-provided info, not user facts
      if (/^user should (be aware|know)\b/i.test(t)) {
        console.log(`[FactExtractor] Filtered out AI-advice fact: "${f}"`);
        return false;
      }

      // Reject date citations from news/web results (e.g. "as of February 16, 2026")
      if (/\b(as of|on) [A-Z][a-z]+ \d{1,2},?\s*\d{4}\b/i.test(t)) {
        console.log(`[FactExtractor] Filtered out dated news item: "${f}"`);
        return false;
      }

      // Reject external/web-search content patterns
      if (/\b(study published|according to|trend featuring|organizations are reporting|journal of)\b/i.test(t)) {
        console.log(`[FactExtractor] Filtered out external info: "${f}"`);
        return false;
      }

      return true;
    });

  } catch (error) {
    console.error('[FactExtractor] Error parsing facts from response:', error.message);
    return [];
  }
}

/**
 * Append facts to MEMORY.md with embedding-based dedup and section-aware placement.
 * Facts are placed under the most semantically similar existing section.
 * @param {string[]} facts - Array of fact strings
 * @param {string} memoryFilePath - Path to MEMORY.md
 */
async function appendToMemory(facts, memoryFilePath) {
  try {
    if (!facts || facts.length === 0) return;

    console.log(`[FactExtractor] Processing ${facts.length} candidate facts`);

    // Ensure directory exists
    const memoryDir = path.dirname(memoryFilePath);
    if (!fs.existsSync(memoryDir)) {
      fs.mkdirSync(memoryDir, { recursive: true });
    }

    // Read and parse existing content
    let content = '';
    if (fs.existsSync(memoryFilePath)) {
      content = fs.readFileSync(memoryFilePath, 'utf8');
    } else {
      content = '# Long-Term Memory\n';
    }

    const sections = parseMemorySections(content);
    const existingFacts = extractAllFactLines(content);
    console.log(`[FactExtractor] Found ${sections.length} sections, ${existingFacts.length} existing facts`);

    // Generate embeddings for all existing facts (for dedup)
    const existingEmbeddings = [];
    for (const fact of existingFacts) {
      existingEmbeddings.push(await generateFactEmbedding(fact));
    }

    // Generate embeddings for each section (heading + content for matching)
    const sectionEmbeddings = [];
    for (const section of sections) {
      const sectionText = section.heading + ': ' + section.factLines.join('. ');
      sectionEmbeddings.push(await generateFactEmbedding(sectionText));
    }

    // Process each new fact: dedup then categorize
    // Map of sectionIndex → [fact strings to add]
    const factsPerSection = new Map();

    for (const fact of facts) {
      // Quick string-match dedup first
      const normalizedFact = fact.toLowerCase();
      if (existingFacts.some(ef => ef.toLowerCase() === normalizedFact)) {
        console.log(`[FactExtractor] Skipping exact duplicate: "${fact}"`);
        continue;
      }

      // Embedding-based dedup
      const factEmbedding = await generateFactEmbedding(fact);
      if (factEmbedding) {
        let isDuplicate = false;
        for (let i = 0; i < existingEmbeddings.length; i++) {
          if (!existingEmbeddings[i]) continue;
          const sim = cosineSimilarity(factEmbedding, existingEmbeddings[i]);
          if (sim > 0.85) {
            console.log(`[FactExtractor] Skipping semantic duplicate (${sim.toFixed(3)}): "${fact}" ≈ "${existingFacts[i]}"`);
            isDuplicate = true;
            break;
          }
        }
        if (isDuplicate) continue;

        // Find best matching section
        let bestIdx = -1;
        let bestScore = 0;
        for (let i = 0; i < sectionEmbeddings.length; i++) {
          if (!sectionEmbeddings[i]) continue;
          const score = cosineSimilarity(factEmbedding, sectionEmbeddings[i]);
          if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
          }
        }

        if (bestIdx >= 0 && bestScore > 0.3) {
          console.log(`[FactExtractor] Placing "${fact}" → ${sections[bestIdx].heading} (score: ${bestScore.toFixed(3)})`);
        } else {
          console.log(`[FactExtractor] Placing "${fact}" → Other (best score: ${bestScore.toFixed(3)})`);
          bestIdx = -1; // will go to Other
        }

        if (!factsPerSection.has(bestIdx)) factsPerSection.set(bestIdx, []);
        factsPerSection.get(bestIdx).push(fact);

        // Add to existing embeddings so subsequent facts in this batch dedup against it
        existingFacts.push(fact);
        existingEmbeddings.push(factEmbedding);
      } else {
        // Embedding failed — fall back to "Other" section, skip embedding dedup
        console.log(`[FactExtractor] Embedding unavailable, placing "${fact}" → Other`);
        if (!factsPerSection.has(-1)) factsPerSection.set(-1, []);
        factsPerSection.get(-1).push(fact);
      }
    }

    if (factsPerSection.size === 0) {
      console.log('[FactExtractor] No new facts to add (all duplicates)');
      return;
    }

    // Rebuild content with facts inserted into their sections
    const lines = content.split('\n');

    // Insert facts into existing sections (iterate in reverse to preserve line numbers)
    const sectionInserts = []; // [{lineIndex, facts}]
    for (const [sectionIdx, sectionFacts] of factsPerSection.entries()) {
      if (sectionIdx < 0) continue; // handle "Other" separately
      const section = sections[sectionIdx];
      // Insert after the last fact line in the section, or after the heading
      let insertAfter = section.startLine;
      for (let i = section.startLine; i <= section.endLine; i++) {
        if (lines[i].startsWith('- ')) insertAfter = i;
      }
      sectionInserts.push({ lineIndex: insertAfter, facts: sectionFacts });
    }

    // Sort inserts by line index descending so earlier inserts don't shift later ones
    sectionInserts.sort((a, b) => b.lineIndex - a.lineIndex);
    for (const insert of sectionInserts) {
      const newLines = insert.facts.map(f => `- ${f}`);
      lines.splice(insert.lineIndex + 1, 0, ...newLines);
    }

    // Handle "Other" section (facts with no good section match)
    const otherFacts = factsPerSection.get(-1);
    if (otherFacts && otherFacts.length > 0) {
      // Find or create ## Other section
      const hasOther = lines.some(l => l.startsWith('## Other'));
      if (hasOther) {
        const otherIdx = lines.findIndex(l => l.startsWith('## Other'));
        let insertAfter = otherIdx;
        for (let i = otherIdx; i < lines.length; i++) {
          if (lines[i].startsWith('- ')) insertAfter = i;
          if (i > otherIdx && lines[i].startsWith('## ')) break;
        }
        const newLines = otherFacts.map(f => `- ${f}`);
        lines.splice(insertAfter + 1, 0, ...newLines);
      } else {
        lines.push('', '## Other');
        for (const f of otherFacts) lines.push(`- ${f}`);
      }
    }

    const updatedContent = lines.join('\n');
    fs.writeFileSync(memoryFilePath, updatedContent, 'utf8');

    const totalAdded = Array.from(factsPerSection.values()).reduce((s, a) => s + a.length, 0);
    console.log(`[FactExtractor] Added ${totalAdded} new facts to memory`);

  } catch (error) {
    console.error('[FactExtractor] Error appending to memory:', error.message);
  }
}

/**
 * Append summary to daily log
 * @param {string} summary - Summary text
 * @param {string} dailyDir - Path to daily log directory
 */
function appendToDailyLog(summary, dailyDir) {
  try {
    if (!summary || summary.trim().length === 0) {
      return;
    }

    // Ensure directory exists
    if (!fs.existsSync(dailyDir)) {
      fs.mkdirSync(dailyDir, { recursive: true });
    }

    const now = new Date();
    const date = now.toISOString().split('T')[0]; // YYYY-MM-DD
    const time = now.toTimeString().slice(0, 5); // HH:MM

    const dailyFile = path.join(dailyDir, `${date}.md`);
    const entry = `### ${time}\n- ${summary}\n\n`;

    // Create file with header if it doesn't exist
    if (!fs.existsSync(dailyFile)) {
      const header = `# Daily Log - ${date}\n\n`;
      fs.writeFileSync(dailyFile, header, 'utf8');
    }

    fs.appendFileSync(dailyFile, entry, 'utf8');
    console.log(`[FactExtractor] Appended to daily log: ${dailyFile}`);

  } catch (error) {
    console.error('[FactExtractor] Error appending to daily log:', error.message);
  }
}

/**
 * Load memory context from files
 * @param {string} memoryDir - Path to memory directory
 * @returns {Object} Memory context object
 */
function loadMemoryContext(memoryDir) {
  try {
    const result = {
      memory: '',
      user: '',
      dailyToday: '',
      dailyYesterday: ''
    };

    // Read MEMORY.md
    const memoryFile = path.join(memoryDir, 'MEMORY.md');
    if (fs.existsSync(memoryFile)) {
      result.memory = fs.readFileSync(memoryFile, 'utf8');
    }

    // Read USER.md
    const userFile = path.join(memoryDir, 'USER.md');
    if (fs.existsSync(userFile)) {
      result.user = fs.readFileSync(userFile, 'utf8');
    }

    // Read today's daily log
    const today = new Date().toISOString().split('T')[0];
    const todayFile = path.join(memoryDir, 'daily', `${today}.md`);
    if (fs.existsSync(todayFile)) {
      result.dailyToday = fs.readFileSync(todayFile, 'utf8');
    }

    // Read yesterday's daily log
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
    const yesterdayFile = path.join(memoryDir, 'daily', `${yesterday}.md`);
    if (fs.existsSync(yesterdayFile)) {
      result.dailyYesterday = fs.readFileSync(yesterdayFile, 'utf8');
    }

    console.log(`[FactExtractor] Loaded memory context: memory=${result.memory.length} chars, user=${result.user.length} chars, today=${result.dailyToday.length} chars, yesterday=${result.dailyYesterday.length} chars`);

    return result;

  } catch (error) {
    console.error('[FactExtractor] Error loading memory context:', error.message);
    return {
      memory: '',
      user: '',
      dailyToday: '',
      dailyYesterday: ''
    };
  }
}

/**
 * Process fact extraction for a chat exchange (high-level orchestrator)
 * @param {string} userMessage - The user's message
 * @param {string} assistantMessage - The assistant's response
 * @param {string} provider - LLM provider
 * @param {string} model - Model name
 * @param {string} apiKey - API key
 * @param {string} ollamaHost - Ollama/llamacpp/squatchserve host
 * @param {string} memoryDir - Memory directory path
 */
async function processFactExtraction(userMessage, assistantMessage, provider, model, apiKey, ollamaHost, memoryDir = MEMORY_DIR) {
  try {
    // Always use the configured extraction model, independent of chat model
    const config = getConfig();
    const extractionProvider = config.models.extraction.provider;
    const extractionModel = config.models.extraction.model;
    const extInst = getProviderInstance(extractionProvider, config.models.extraction.instance);
    const extractionHost = extInst ? extInst.host : ollamaHost;
    console.log(`[FactExtractor] Using extraction model: ${extractionProvider}/${extractionModel}`);

    // Extract facts using the configured extraction model
    const facts = await extractFacts(userMessage, assistantMessage, extractionProvider, extractionModel, apiKey, extractionHost);

    // Append to memory if facts found
    if (facts.length > 0) {
      const memoryFile = path.join(memoryDir, 'MEMORY.md');
      await appendToMemory(facts, memoryFile);

      // === UPGRADE 4: Assign facts to memory clusters ===
      try {
        const memoryClusters = require('./memory-clusters');
        for (const fact of facts) {
          await memoryClusters.assignToCluster(fact, extractionProvider, extractionModel, apiKey, extractionHost, 'fact-extraction');
        }
        console.log(`[FactExtractor] Assigned ${facts.length} facts to clusters`);
      } catch (clusterError) {
        console.error('[FactExtractor] Cluster assignment error:', clusterError.message);
      }
    }

    // Create daily log summary
    const summary = `Chat exchange with ${extractionProvider}/${extractionModel} - ${facts.length} facts extracted`;
    const dailyDir = path.join(memoryDir, 'daily');
    appendToDailyLog(summary, dailyDir);

    console.log('[FactExtractor] Fact extraction complete');

  } catch (error) {
    console.error('[FactExtractor] Error in processFactExtraction:', error.message);
  }
}

module.exports = {
  extractFacts,
  extractAllFactLines,
  appendToMemory,
  appendToDailyLog,
  loadMemoryContext,
  processFactExtraction,
  MEMORY_DIR,
  DAILY_DIR
};
