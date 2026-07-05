const fs = require('fs');
const path = require('path');
const { getConfig, getProviderInstance } = require('./config');
const { getCurrentDateTimeString, formatFactTimestamp, getLocalDateStamp } = require('./datetime');
const agentPool = require('./agent-pool');

// A fact line written to MEMORY.md may carry a "(learned YYYY-MM-DD H:MM AM/PM)"
// annotation so the model can answer "when did I tell you this". Strip it when
// comparing/deduping/matching fact text so only the bare fact is considered.
function stripLearnedAnnotation(text) {
  return text.replace(/\s*\(learned\s+[^)]*\)\s*$/i, '').trim();
}

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
      currentSection.factLines.push(stripLearnedAnnotation(line.substring(2).trim()));
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
    .map(line => stripLearnedAnnotation(line.substring(2).trim()));
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

USER FACTS vs FACTS ABOUT ME (the AI) — this distinction is critical:
- A fact about the USER, their life, their preferences, or their WORK/PROJECTS is a user fact — extract it. This INCLUDES the SNH (Squatch Neuro Hub) project the user is building: "User's SNH system uses semantic clustering" or "User is building SNH, a cluster-memory tool" are legitimate user facts about what the USER built.
- A fact about ME — the AI/SNH's OWN nature, personality, feelings, values, self-image, or behavioral tendencies — is NOT a user fact. Never extract it. If the user says "you're really curious", "SNH tends to be verbose", or "you seem to care about accuracy", that is an observation about the AI, not a fact about the user — SKIP it. The AI records its own self-observations through a separate reflection pipeline, not here.
- The test for every candidate: "Is this a fact about the USER (or the user's work), or a fact about ME, the AI?" Keep only the former. When phrased in the first person ("I tend to...", "I care about...") it is about the AI — skip it.

GOOD examples (facts the user stated about themselves or their project):
["User has 4 dogs: Casper, Cece, Calypso, and Erika", "User is migrating from Syncro to Kaseya for RMM", "User's AI server has dual RTX 3090s with 48GB total VRAM", "User's SNH system uses salience scoring from 1-10 to prioritize facts"]

BAD examples (AI-provided info, fragments, general knowledge, or facts about the AI itself):
["A viral TikTok trend featuring Mini Huskies", "Constantinople fell in 1453", "RTX 3090", "The weather is nice", "SNH is curious and analytical", "I tend to reflect on my own cognitive processes", "The AI cares deeply about evolving truth"]

Every extracted fact must be something the USER told you about themselves or their work — not something you told the user, and not an observation about you, the AI.

${getCurrentDateTimeString()}. When the user's statement is time-relative ("I just bought", "last week", "recently", "starting next month"), anchor the fact to an absolute date using the current date above — e.g. "As of July 2026, User is migrating from Syncro to Kaseya".`;

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

      // Reject facts about the AI's OWN nature — first-person self-observations
      // ("I tend to...", "My purpose is...", "As an AI...") belong to the
      // reflection pipeline as subject:'self', never to user memory. User facts
      // are framed "User has..." / "User's ...", so this never drops a real one.
      if (/^(i|i'm|i am|my|as an ai|as a language model)\b/i.test(t)) {
        console.log(`[FactExtractor] Filtered out AI self-observation (first-person): "${f}"`);
        return false;
      }

      // Reject facts asserting the AI/SNH's own personality or feelings (as opposed
      // to the user's SNH *project*, which is framed "User's SNH ..."). "SNH is
      // curious" → dropped; "User's SNH system uses X" → kept (starts with "User").
      if (/^(snh|the ai)\b[^.]*\b(is|are|feels?|thinks?|believes?|wants?|enjoys?|cares?|values?|tends? to|prefers?|likes?|considers?|strives?)\b/i.test(t)) {
        console.log(`[FactExtractor] Filtered out AI-nature self-observation: "${f}"`);
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

    // Annotate each newly written fact with when it was learned (now), so the
    // markdown injection path carries timestamps like the cluster path does.
    const learnedAt = formatFactTimestamp(new Date().toISOString());
    const factLine = f => (learnedAt ? `- ${f} (learned ${learnedAt})` : `- ${f}`);

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
      const newLines = insert.facts.map(f => factLine(f));
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
        const newLines = otherFacts.map(f => factLine(f));
        lines.splice(insertAfter + 1, 0, ...newLines);
      } else {
        lines.push('', '## Other');
        for (const f of otherFacts) lines.push(factLine(f));
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
 * Insert a pre-formatted entry block at the TOP of the day's log, directly
 * under the "# Daily Log - <date>" H1 header, so the newest entry is first.
 * The file structure is otherwise unchanged: the H1 header stays at the top,
 * followed by "### HH:MM" / "## Heartbeat Report" blocks, newest → oldest.
 *
 * Shared by every daily-log writer (fact extraction, heartbeat report, agent
 * pool) so ordering stays consistent across all of them.
 *
 * @param {string} entry - Fully formatted block, ending with a blank line ("\n\n")
 * @param {string} dailyDir - Path to daily log directory
 * @param {string} date - YYYY-MM-DD for the target file
 * @returns {string} Path to the daily file written
 */
function prependDailyEntry(entry, dailyDir, date) {
  if (!fs.existsSync(dailyDir)) {
    fs.mkdirSync(dailyDir, { recursive: true });
  }

  const dailyFile = path.join(dailyDir, `${date}.md`);
  const header = `# Daily Log - ${date}\n\n`;

  if (!fs.existsSync(dailyFile)) {
    fs.writeFileSync(dailyFile, header + entry, 'utf8');
    return dailyFile;
  }

  const content = fs.readFileSync(dailyFile, 'utf8');
  // Match ONLY a level-1 header ("# " — a single hash + space), so a leading
  // "## Heartbeat Report" block is never mistaken for the header. Capture the
  // header line plus its trailing blank line.
  const headerMatch = content.match(/^(# [^\n]*\r?\n(?:\r?\n)?)/);
  if (headerMatch) {
    const head = headerMatch[1];
    const body = content.slice(head.length);
    fs.writeFileSync(dailyFile, head + entry + body, 'utf8');
  } else {
    // No recognizable H1 header (legacy file) — add one, then newest entry,
    // then the existing content.
    fs.writeFileSync(dailyFile, header + entry + content, 'utf8');
  }
  return dailyFile;
}

/**
 * Append summary to daily log (inserted at the top — newest first).
 * @param {string} summary - Summary text
 * @param {string} dailyDir - Path to daily log directory
 */
function appendToDailyLog(summary, dailyDir) {
  try {
    if (!summary || summary.trim().length === 0) {
      return;
    }

    const now = new Date();
    const date = getLocalDateStamp(now); // local Pacific YYYY-MM-DD
    const time = now.toTimeString().slice(0, 5); // HH:MM

    const entry = `### ${time}\n- ${summary}\n\n`;
    const dailyFile = prependDailyEntry(entry, dailyDir, date);
    console.log(`[FactExtractor] Prepended to daily log: ${dailyFile}`);

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

    // Read today's daily log (bucketed by local Pacific date)
    const today = getLocalDateStamp();
    const todayFile = path.join(memoryDir, 'daily', `${today}.md`);
    if (fs.existsSync(todayFile)) {
      result.dailyToday = fs.readFileSync(todayFile, 'utf8');
    }

    // Read yesterday's daily log
    const yesterday = getLocalDateStamp(new Date(Date.now() - 86400000));
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
 * Ask the local reasoning model whether a new user statement contradicts an
 * existing stored fact. The user is always the authority on their own life, so
 * a clear contradiction means the new statement wins and the old fact is
 * superseded. When it is genuinely ambiguous, the judge returns UNCERTAIN
 * rather than guessing — the caller queues a clarifying question instead.
 * @param {string} newFact - The fact just extracted from the user
 * @param {string} oldFact - An existing active stored fact
 * @returns {Promise<{verdict: 'yes'|'no'|'uncertain', reasoning: string}>}
 */
async function judgeContradiction(newFact, oldFact) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You are a fact contradiction detector for a personal memory system. You are given an EXISTING stored fact about the user and a NEW statement the user just made about themselves.

Decide the relationship between them:
- YES — they contradict: they cannot both be true of the user at the same time. Corrections and replacements count ("Actually my MSP is X, not Y", "I moved to Z", "I no longer use Q").
- NO — no contradiction: additional detail, refinement, or an unrelated fact.
- UNCERTAIN — you genuinely cannot tell whether they conflict without more information (e.g. they might refer to two different things, or one might update the other, but it is ambiguous).

Prefer UNCERTAIN over guessing when it is truly ambiguous.

Respond with exactly YES, NO, or UNCERTAIN on the first line, then one short line of reasoning.`;
    const userPrompt = `EXISTING fact: "${oldFact}"\nNEW statement: "${newFact}"\n\nDoes the NEW statement contradict the EXISTING fact?`;

    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 120 });
    const firstWord = (content.trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    let verdict = 'no';
    if (firstWord === 'yes') verdict = 'yes';
    else if (firstWord === 'uncertain') verdict = 'uncertain';
    const reasoning = content.trim().split('\n').slice(0, 2).join(' ').trim();
    console.log(`[FactExtractor] Contradiction judge: ${verdict.toUpperCase()} — "${newFact}" vs "${oldFact}" (${reasoning})`);
    return { verdict, reasoning };
  } catch (error) {
    console.error('[FactExtractor] Contradiction judge error:', error.message);
    return { verdict: 'no', reasoning: '' };
  }
}

/**
 * Score how much a fact matters (salience, 1–10) with a judgment call to the
 * local model. Higher = durable and decision-relevant; lower = trivia/ephemeral.
 * @param {string} fact - The new fact to score
 * @param {string} nearbyContext - Short summary of related existing facts/clusters
 * @returns {Promise<{salience: number, reasoning: string}>}
 */
async function scoreSalience(fact, nearbyContext = '') {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You score how much a fact about a user matters, for a long-term memory system. Return an integer salience from 1 (trivial/ephemeral) to 10 (defining/durable).

Judge using these criteria:
- Does this fact connect to or change existing memory clusters? Connected/changing → higher.
- Does it affect the user's decisions, projects, or work (high), or is it passing trivia (low)?
- Is it durable — a name, business, relationship, long-term preference (high) — or ephemeral, like today's mood or weather (low)?

Guidance: names/business/relationships/major projects = 8–10; stable preferences/tools/hardware = 5–7; incidental details = 3–4; momentary state ("tired today") = 1–2.

Respond with the integer on the first line, then one short line of reasoning.`;
    const userPrompt = `${nearbyContext ? `Related existing memory:\n${nearbyContext}\n\n` : ''}Fact to score: "${fact}"\n\nSalience (1-10)?`;

    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 120 });
    const match = content.match(/\d+/);
    let salience = match ? parseInt(match[0], 10) : 5;
    if (!Number.isFinite(salience)) salience = 5;
    salience = Math.max(1, Math.min(10, salience));
    const reasoning = content.trim().split('\n').slice(0, 2).join(' ').trim();
    console.log(`[FactExtractor] Salience ${salience}/10 — "${fact}" (${reasoning})`);
    return { salience, reasoning };
  } catch (error) {
    console.error('[FactExtractor] Salience scoring error:', error.message);
    return { salience: 5, reasoning: '' };
  }
}

/**
 * Given newly learned facts and their surrounding cluster context, decide
 * whether there is a single worthwhile clarifying question to ask the user —
 * a gap, something incomplete, or an odd inconsistency. Quality over quantity:
 * returns at most one question, or null if nothing is worth asking.
 * @param {string[]} facts - The new facts from this exchange
 * @param {string} nearbyContext - Related existing facts/clusters
 * @returns {Promise<{question: string}|null>}
 */
async function detectGapQuestion(facts, nearbyContext = '') {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You maintain a personal memory system for a user. Given facts just learned and related existing memory, decide whether there is ONE natural clarifying question worth asking the user — because something is unclear, incomplete, or oddly inconsistent (e.g. a project mentioned with no client, a tool with no purpose, two facts that don't quite line up).

Rules:
- Only propose a question if it would genuinely improve the memory and a person would find it natural to be asked.
- At most ONE question. Keep it short, specific, and conversational — never interrogation-style.
- If nothing is worth asking, respond with exactly NONE.

Respond with either NONE, or the single question text on one line (no preamble).`;
    const userPrompt = `Newly learned facts:\n${facts.map(f => `- ${f}`).join('\n')}\n\n${nearbyContext ? `Related existing memory:\n${nearbyContext}\n\n` : ''}Is there ONE clarifying question worth asking? If so, give just the question; otherwise NONE.`;

    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 100 });
    const text = content.trim().split('\n')[0].trim();
    if (!text || /^none\b/i.test(text)) return null;
    // Strip leading list markers/quotes the model might add.
    const question = text.replace(/^[-*\d.\s"]+/, '').replace(/"$/, '').trim();
    if (!question || question.length < 5) return null;
    console.log(`[FactExtractor] Gap question proposed: "${question}"`);
    return { question };
  } catch (error) {
    console.error('[FactExtractor] Gap detection error:', error.message);
    return null;
  }
}

/**
 * Judge whether a user's message answers a previously asked question.
 * @param {string} question - The question that was asked
 * @param {string} userMessage - The user's latest message
 * @returns {Promise<boolean>}
 */
async function judgeAnswered(question, userMessage) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You decide whether a user's message answers a specific question that was previously asked. Respond with exactly YES or NO on the first line.`;
    const userPrompt = `Question that was asked: "${question}"\nUser's message: "${userMessage}"\n\nDoes the user's message answer that question (even partially)?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 30 });
    const firstWord = (content.trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    return firstWord === 'yes';
  } catch (error) {
    console.error('[FactExtractor] Answer judge error:', error.message);
    return false;
  }
}

/**
 * Build a short context string of existing facts related to the new facts,
 * used to inform salience scoring and gap detection. Returns cluster name +
 * a few member facts. Also returns the ids of the clusters consulted.
 * @param {string[]} facts
 * @returns {Promise<{text: string, clusterIds: string[]}>}
 */
async function buildNearbyContext(facts) {
  try {
    const memoryClusters = require('./memory-clusters');
    const query = facts.join('. ');
    const clusters = await memoryClusters.searchClusters(query, 2);
    if (!clusters || clusters.length === 0) return { text: '', clusterIds: [] };
    const text = clusters.map(c => {
      const members = c.members.slice(0, 5).map(m => `  - ${m.content}`).join('\n');
      return `[${c.cluster.name}]\n${members}`;
    }).join('\n');
    const clusterIds = clusters.map(c => c.cluster.id).filter(Boolean);
    return { text, clusterIds };
  } catch (error) {
    console.error('[FactExtractor] buildNearbyContext error:', error.message);
    return { text: '', clusterIds: [] };
  }
}

/**
 * Remove a superseded fact's line from MEMORY.md so it stops entering model
 * context. The SQLite row is kept for history; only the injected markdown copy
 * is pruned. Matches the `- <fact>` bullet by trimmed content.
 * @param {string} factContent - The exact fact text to remove
 * @param {string} memoryFilePath - Path to MEMORY.md
 * @returns {boolean} - true if a line was removed
 */
function removeFactLineFromMemory(factContent, memoryFilePath) {
  try {
    if (!fs.existsSync(memoryFilePath)) return false;
    const target = factContent.trim().toLowerCase();
    const lines = fs.readFileSync(memoryFilePath, 'utf8').split('\n');
    const kept = lines.filter(line => {
      const m = line.match(/^\s*-\s+(.*)$/);
      if (!m) return true;
      // Compare on the bare fact, ignoring any "(learned ...)" annotation.
      return stripLearnedAnnotation(m[1].trim()).toLowerCase() !== target;
    });
    if (kept.length !== lines.length) {
      fs.writeFileSync(memoryFilePath, kept.join('\n'), 'utf8');
      console.log(`[FactExtractor] Removed superseded fact line from MEMORY.md: "${factContent}"`);
      return true;
    }
    return false;
  } catch (error) {
    console.error('[FactExtractor] Error removing fact line from memory:', error.message);
    return false;
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
 * @param {string} conversationId - Conversation this exchange belongs to
 * @param {string} memoryDir - Memory directory path
 */
async function processFactExtraction(userMessage, assistantMessage, provider, model, apiKey, ollamaHost, conversationId = null, memoryDir = MEMORY_DIR) {
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

    const memoryFile = path.join(memoryDir, 'MEMORY.md');
    const dailyDir = path.join(memoryDir, 'daily');
    const memoryClusters = require('./memory-clusters');
    const questions = require('./questions');

    // === Answer detection ===
    // If this conversation had an outstanding asked question, check whether the
    // user's message answers it (the answer itself is caught as facts normally).
    if (conversationId) {
      try {
        for (const q of questions.getAskedForConversation(conversationId)) {
          if (await judgeAnswered(q.question, userMessage)) {
            questions.markAnswered(q.id);
            appendToDailyLog(`Question answered: "${q.question}"`, dailyDir);
          }
        }
      } catch (answerErr) {
        console.error('[FactExtractor] Answer detection error:', answerErr.message);
      }
    }

    // At most ONE question per chat session, shared by contradiction-uncertainty
    // (higher priority) and gap detection.
    let questionQueued = false;

    // Append to memory if facts found
    if (facts.length > 0) {
      // Surrounding cluster context — informs salience scoring and gap detection.
      const nearby = await buildNearbyContext(facts);

      // === Contradiction detection (3-way: yes / no / uncertain) ===
      // Gather every (newFact, oldFact) candidate pair first — candidate lookup
      // is a cheap read — then judge them all concurrently through the agent
      // pool (vLLM batches them). Verdicts are applied afterward in gather
      // order so the outcome is deterministic and identical to a sequential pass:
      // the one-question-per-session slot still goes to the first uncertain pair,
      // and an old member already claimed by a 'yes' is never superseded twice.
      const supersessions = []; // {oldMemberId, oldContent, oldSalience, newFact}
      const seenOld = new Set();
      try {
        const contradictionPairs = [];
        for (const fact of facts) {
          const candidates = await memoryClusters.findContradictionCandidates(fact);
          for (const candidate of candidates) {
            contradictionPairs.push({ fact, candidate });
          }
        }

        if (contradictionPairs.length > 0) {
          const judged = await agentPool.runBatch(
            contradictionPairs.map(({ fact, candidate }) => async () => {
              const { verdict } = await judgeContradiction(fact, candidate.content);
              return verdict;
            }),
            'contradiction-judge'
          );

          for (let i = 0; i < contradictionPairs.length; i++) {
            const { fact, candidate } = contradictionPairs[i];
            if (seenOld.has(candidate.memberId)) continue;
            const verdict = judged[i].status === 'fulfilled' ? judged[i].value : 'no';
            if (verdict === 'yes') {
              seenOld.add(candidate.memberId);
              supersessions.push({
                oldMemberId: candidate.memberId,
                oldContent: candidate.content,
                oldSalience: candidate.salience ?? 5,
                newFact: fact
              });
            } else if (verdict === 'uncertain' && !questionQueued) {
              // Ambiguous conflict — ask the user rather than guess; both facts stay active.
              const q = `I have two things noted that might not line up: "${candidate.content}" and now "${fact}". Which is correct?`;
              if (questions.addQuestion({
                question: q,
                reason: 'contradiction-uncertainty',
                clusterId: candidate.clusterId,
                memberId: candidate.memberId,
                conversationId
              })) {
                questionQueued = true;
                appendToDailyLog(`Queued clarifying question (contradiction-uncertainty): "${q}"`, dailyDir);
              }
            }
          }
        }
      } catch (contradictionError) {
        console.error('[FactExtractor] Contradiction detection error:', contradictionError.message);
      }

      // === Salience scoring + gap detection (concurrent) ===
      // Score each new fact's salience concurrently, and run gap-question
      // detection alongside them as one more pool task rather than after — all
      // are independent, read-only LLM judgments over the same context. The DB
      // writes they inform stay sequential below, so there are no write races.
      const factToSalience = new Map();
      const runGap = !questionQueued; // skip if contradiction already claimed the one-question slot
      const [salienceSettled, gapCandidate] = await Promise.all([
        agentPool.runBatch(
          facts.map(fact => async () => {
            const { salience, reasoning } = await scoreSalience(fact, nearby.text);
            return { fact, salience, reasoning };
          }),
          'salience'
        ),
        runGap
          ? agentPool.schedule(() => detectGapQuestion(facts, nearby.text), 'gap').catch(err => {
              console.error('[FactExtractor] Gap detection error:', err.message);
              return null;
            })
          : Promise.resolve(null)
      ]);

      // Apply salience results in fact order for a stable daily-log trail.
      const salienceByFact = new Map();
      for (const s of salienceSettled) {
        if (s.status === 'fulfilled' && s.value) salienceByFact.set(s.value.fact, s.value);
      }
      for (const fact of facts) {
        const scored = salienceByFact.get(fact);
        const salience = scored ? scored.salience : 5;
        factToSalience.set(fact, salience);
        appendToDailyLog(`Scored fact salience ${salience}/10: "${fact}" — ${scored ? scored.reasoning : 'default (scoring failed)'}`, dailyDir);
      }

      // A superseding fact inherits at least the salience of the fact it replaces.
      for (const s of supersessions) {
        const cur = factToSalience.get(s.newFact) ?? 5;
        if (s.oldSalience > cur) {
          factToSalience.set(s.newFact, s.oldSalience);
          console.log(`[FactExtractor] "${s.newFact}" inherits salience ${s.oldSalience} from superseded fact`);
        }
      }

      await appendToMemory(facts, memoryFile);

      // === Assign facts to clusters (carrying salience) ===
      const factToMemberId = new Map();
      const factToClusterId = new Map();
      try {
        for (const fact of facts) {
          const res = await memoryClusters.assignToCluster(
            fact, extractionProvider, extractionModel, apiKey, extractionHost,
            'fact-extraction', factToSalience.get(fact) ?? 5
          );
          if (res && res.memberId) factToMemberId.set(fact, res.memberId);
          if (res && res.clusterId) factToClusterId.set(fact, res.clusterId);
        }
        console.log(`[FactExtractor] Assigned ${facts.length} facts to clusters`);
      } catch (clusterError) {
        console.error('[FactExtractor] Cluster assignment error:', clusterError.message);
      }

      // === Apply supersessions (after the replacing facts exist) ===
      for (const s of supersessions) {
        const newMemberId = factToMemberId.get(s.newFact);
        if (!newMemberId) continue; // replacing fact wasn't stored — skip
        const marked = memoryClusters.supersedeFact(s.oldMemberId, newMemberId);
        if (marked) {
          removeFactLineFromMemory(s.oldContent, memoryFile);
          appendToDailyLog(`Superseded fact: "${s.oldContent}" → replaced by "${s.newFact}" (user correction)`, dailyDir);
          console.log(`[FactExtractor] Supersession: "${s.oldContent}" → "${s.newFact}"`);
        }
      }

      // === Gap question queuing ===
      // The gap-detection LLM call already ran concurrently with salience scoring
      // above; here we only queue its result, anchored to the cluster the new
      // facts landed in so it surfaces when the user next chats about that topic.
      // Respect the one-question-per-session slot (contradiction takes priority).
      if (!questionQueued && gapCandidate && gapCandidate.question) {
        const anchorClusterId = factToClusterId.get(facts[0]) || nearby.clusterIds[0] || null;
        const anchorMemberId = factToMemberId.get(facts[0]) || null;
        if (questions.addQuestion({
          question: gapCandidate.question,
          reason: 'gap',
          clusterId: anchorClusterId,
          memberId: anchorMemberId,
          conversationId
        })) {
          questionQueued = true;
          appendToDailyLog(`Queued clarifying question (gap): "${gapCandidate.question}"`, dailyDir);
        }
      }
    }

    // Create daily log summary
    const summary = `Chat exchange with ${extractionProvider}/${extractionModel} - ${facts.length} facts extracted`;
    appendToDailyLog(summary, dailyDir);

    console.log('[FactExtractor] Fact extraction complete');

  } catch (error) {
    console.error('[FactExtractor] Error in processFactExtraction:', error.message);
  }
}

// ============ Self-facts (SNH's observations about itself) ============

/**
 * Parse a reflection response into a list of self-observation strings.
 * Unlike parseFactsFromResponse (which rejects assistant-subject facts), here
 * the AI itself IS the subject — first-person "I ..." statements are expected.
 * Accepts a JSON array or a bullet/numbered list; caps the count.
 * @param {string} response
 * @returns {string[]}
 */
function parseSelfObservations(response) {
  try {
    const text = (response || '').replace(/```(?:json)?\s*\n?([\s\S]*?)```/g, '$1').trim();
    let items = [];

    const arrMatch = text.match(/\[[\s\S]*\]/);
    if (arrMatch) {
      let jsonStr = arrMatch[0];
      try {
        items = JSON.parse(jsonStr);
      } catch {
        jsonStr = jsonStr
          .replace(/\[\s*'/g, '["')
          .replace(/'\s*\]/g, '"]')
          .replace(/'\s*,\s*'/g, '", "');
        try { items = JSON.parse(jsonStr); } catch { items = []; }
      }
    }

    if (!Array.isArray(items) || items.length === 0) {
      // Fallback: treat non-empty bullet/numbered lines as observations.
      items = text.split('\n')
        .map(l => l.replace(/^[-*\d.)\s]+/, '').trim())
        .filter(Boolean);
    }

    return items
      .filter(x => typeof x === 'string')
      .map(x => x.trim())
      .filter(x => x.length >= 4 && x.length <= 400)
      .slice(0, 8);
  } catch (error) {
    console.error('[SelfFacts] parseSelfObservations error:', error.message);
    return [];
  }
}

/**
 * Store self-observations through the SAME machinery as user facts — salience
 * scoring, contradiction/supersession against existing self-facts, cluster
 * assignment — but flagged subject:'self' and clustered separately. No MEMORY.md
 * write (self-facts inject via the identity block, not user memory) and no gap
 * questions. The AI can change its mind about itself: a new self-fact that
 * contradicts an old one supersedes it, keeping the old as history.
 *
 * @param {string[]} rawSelfFacts
 * @param {Object} [opts]
 * @param {string} [opts.source='reflection']
 * @param {string} [opts.memoryDir=MEMORY_DIR]
 * @returns {Promise<{stored:number, superseded:number, facts:Array}>}
 */
async function processSelfFacts(rawSelfFacts, opts = {}) {
  const source = opts.source || 'reflection';
  const memoryDir = opts.memoryDir || MEMORY_DIR;
  const dailyDir = path.join(memoryDir, 'daily');
  const result = { stored: 0, superseded: 0, facts: [] };

  try {
    const memoryClusters = require('./memory-clusters');

    // Normalize + dedup
    const facts = [];
    const seenText = new Set();
    for (const raw of rawSelfFacts || []) {
      const f = (raw || '').trim();
      if (!f) continue;
      const key = f.toLowerCase();
      if (seenText.has(key)) continue;
      seenText.add(key);
      facts.push(f);
    }
    if (facts.length === 0) return result;

    // Provider for embeddings + cluster naming (same as the extraction path).
    const config = getConfig();
    const extractionProvider = config.models.extraction.provider;
    const extractionModel = config.models.extraction.model;
    const extInst = getProviderInstance(extractionProvider, config.models.extraction.instance);
    const extractionHost = extInst ? extInst.host : 'http://localhost:11434';

    // Nearby context for salience scoring = existing self-facts.
    const existing = memoryClusters.getSelfFacts({ status: 'active', limit: 20 });
    const nearbyText = existing.length ? existing.map(f => `- ${f.content}`).join('\n') : '';

    // === Contradiction detection against existing self-facts (concurrent) ===
    const supersessions = [];
    const seenOld = new Set();
    try {
      const pairs = [];
      for (const fact of facts) {
        const candidates = await memoryClusters.findContradictionCandidates(fact, { subject: 'self' });
        for (const candidate of candidates) pairs.push({ fact, candidate });
      }
      if (pairs.length > 0) {
        const judged = await agentPool.runBatch(
          pairs.map(({ fact, candidate }) => async () => (await judgeContradiction(fact, candidate.content)).verdict),
          'self-contradiction-judge'
        );
        for (let i = 0; i < pairs.length; i++) {
          const { fact, candidate } = pairs[i];
          if (seenOld.has(candidate.memberId)) continue;
          const verdict = judged[i].status === 'fulfilled' ? judged[i].value : 'no';
          if (verdict === 'yes') {
            seenOld.add(candidate.memberId);
            supersessions.push({
              oldMemberId: candidate.memberId,
              oldContent: candidate.content,
              oldSalience: candidate.salience ?? 5,
              newFact: fact
            });
          }
        }
      }
    } catch (e) {
      console.error('[SelfFacts] Contradiction detection error:', e.message);
    }

    // === Salience scoring (concurrent) ===
    const factToSalience = new Map();
    const salienceSettled = await agentPool.runBatch(
      facts.map(fact => async () => {
        const { salience, reasoning } = await scoreSalience(fact, nearbyText);
        return { fact, salience, reasoning };
      }),
      'self-salience'
    );
    const byFact = new Map();
    for (const s of salienceSettled) if (s.status === 'fulfilled' && s.value) byFact.set(s.value.fact, s.value);
    for (const fact of facts) {
      const scored = byFact.get(fact);
      const salience = scored ? scored.salience : 5;
      factToSalience.set(fact, salience);
      appendToDailyLog(`Scored self-fact salience ${salience}/10: "${fact}" — ${scored ? scored.reasoning : 'default (scoring failed)'}`, dailyDir);
    }
    for (const s of supersessions) {
      const cur = factToSalience.get(s.newFact) ?? 5;
      if (s.oldSalience > cur) factToSalience.set(s.newFact, s.oldSalience);
    }

    // === Cluster assignment (subject:'self', sequential DB writes) ===
    const factToMemberId = new Map();
    for (const fact of facts) {
      const res = await memoryClusters.assignToCluster(
        fact, extractionProvider, extractionModel, '', extractionHost,
        source, factToSalience.get(fact) ?? 5, 'self'
      );
      if (res && res.memberId) {
        factToMemberId.set(fact, res.memberId);
        result.stored++;
        result.facts.push({
          content: fact,
          memberId: res.memberId,
          salience: factToSalience.get(fact) ?? 5,
          clusterName: res.clusterName
        });
      }
    }

    // === Apply supersessions (after replacing facts exist) ===
    for (const s of supersessions) {
      const newMemberId = factToMemberId.get(s.newFact);
      if (!newMemberId) continue;
      if (memoryClusters.supersedeFact(s.oldMemberId, newMemberId)) {
        result.superseded++;
        appendToDailyLog(`Superseded self-fact: "${s.oldContent}" → "${s.newFact}" (revised self-view)`, dailyDir);
      }
    }

    console.log(`[SelfFacts] Stored ${result.stored} self-fact(s), superseded ${result.superseded}`);
    return result;
  } catch (error) {
    console.error('[SelfFacts] processSelfFacts error:', error.message);
    return result;
  }
}

module.exports = {
  extractFacts,
  extractAllFactLines,
  appendToMemory,
  appendToDailyLog,
  prependDailyEntry,
  judgeContradiction,
  scoreSalience,
  detectGapQuestion,
  judgeAnswered,
  removeFactLineFromMemory,
  loadMemoryContext,
  processFactExtraction,
  parseSelfObservations,
  processSelfFacts,
  MEMORY_DIR,
  DAILY_DIR
};
