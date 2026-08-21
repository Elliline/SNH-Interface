const fs = require('fs');
const path = require('path');
const { getConfig, getProviderInstance } = require('./config');
const { getCurrentDateTimeString, formatFactTimestamp, getLocalDateStamp } = require('./datetime');
const agentPool = require('./agent-pool');

const MEMORY_DIR = require('./database').getMemoryDir();
const DAILY_DIR = path.join(MEMORY_DIR, 'daily');

// Cosine similarity that works for BOTH plain Arrays and Float32Arrays. The
// shared memoryClusters.cosineSimilarity guards on Array.isArray and so returns
// 0 for the Float32Array that memoryClusters.generateEmbedding produces — this
// index-based version avoids that trap for the self-fact dedup path.
function embeddingCosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den === 0 ? 0 : dot / den;
}

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
 * Extract candidates from a chat exchange, split into durable FACTS and
 * time-bound EVENTS.
 *
 * Phase 2a rewrite. Three things changed and each one has a defect behind it:
 *
 *  - ATOMICITY. "User's professional entities include her MSP, MettaSphere, and
 *    her AI research venture, Coastal Squatch." was stored as a single fact under
 *    a cluster named for one of the two entities, so a query about the other could
 *    not reach it. One fact now asserts one thing.
 *
 *  - EVENT vs STATE. "User has a pet named Roscoe who had a restless night as of
 *    July 2026" is in the corpus as a permanent fact. Events go to the day's log
 *    and never to the fact store.
 *
 *  - NO DATE STAMPING. The old prompt ended by instructing the model to anchor
 *    time-relative statements to an absolute date ("As of July 2026, User is
 *    migrating…"). That is the opposite of the rule above: it manufactured the
 *    very timestamp that marks a sentence as an event. It is gone.
 *
 * @returns {Promise<{facts: Array<{text: string, corrects: string|null}>,
 *                    events: Array<{text: string}>}>}
 */
async function extractCandidates(userMessage, assistantMessage, provider, model, apiKey, ollamaHost) {
  try {
    console.log(`[FactExtractor] Extracting candidates using ${provider}/${model}`);

    const systemPrompt = `You sort what the USER said in a chat exchange into two buckets: durable FACTS and time-bound EVENTS.

Return ONLY a JSON object in exactly this shape, and nothing else:
{"facts": [...], "events": [...]}
Both keys are always present. Either may be empty. Nothing appears in both.

=== THE TEST: STRIP THE TIMESTAMP ===
Take the statement and remove every reference to time. Is what remains still true next month, and still worth knowing?
- YES → it is a FACT (a standing state): has / owns / is / is named / prefers / works at / believes / is building.
- NO  → it is an EVENT.

These are ALWAYS events, with no exceptions:
- Something that happened: "washed the car", "the dog had a restless night", "bought dirt this morning".
- Anything qualified by today, tonight, last night, this week, currently, right now, or a specific date.
- An activity in progress: "the cleaners are filling the holes in the yard".
- A mood, an energy level, or a feeling: "is exhausted", "has no motivation at the moment", "is frustrated with the microphone".

When you cannot decide, put it in EVENTS. A durable fact that gets missed will be picked up the next time it comes up. A transient one that gets stored as a fact pollutes the memory forever.

=== ATOMIC ===
One fact asserts ONE thing about ONE subject. Split compounds into separate facts.
- "User enjoys computers, gaming, cars, and guns"
  → "User enjoys computers", "User enjoys gaming", "User enjoys cars", "User enjoys guns"
- "User's MSP is MettaSphere and her AI research venture is Coastal Squatch"
  → "User's MSP is MettaSphere", "User's AI research venture is Coastal Squatch"
Never join unrelated assertions with "and". A clause that describes the SAME one thing stays together: "User has a dog named Casper" is one fact, not two.

=== NO DATE STAMPING ===
Never write a date or a time reference into a fact. If a statement needs one to be true, it is an EVENT — put it in events with the time reference kept.

=== WHOSE FACT IT IS ===
Attribute from WHAT THE USER ACTUALLY SAID, quoted below. Never from the assistant's reply, and never from the assistant's restatement of the user's words — the assistant rewrites pronouns, and a detail that appears only in its reply is not the user's fact.

WATCH THE SPEAKER. The user says "I" and "my" about HERSELF, and "you" and "your" about the ASSISTANT.
- "my gaming PC is 850W" → a USER fact: "User's gaming PC is 850W".
- "you're on an ASUS GX10", "your box does 200W", "your name is X", "you tend to over-explain" → these are about the ASSISTANT'S own hardware, name or behaviour. They are NOT user facts. SKIP them entirely — do not rewrite them as "User's…". The assistant records what it learns about itself through a separate pipeline.
Rewriting a "you" statement into a "User" statement is the single most damaging mistake available here: it files a fact about the assistant as a belief about the user, where it can then contradict and retire things that are actually true of her.
- A fact about the USER, their life, their preferences, or their WORK/PROJECTS is a user fact — extract it. This INCLUDES the SNH (Squatch Neuro Hub) project the user is building: "User's SNH system uses semantic clustering" is a legitimate user fact about what the USER built.
- A fact about ME — the AI/SNH's OWN nature, personality, feelings, values, self-image, or behavioral tendencies — is NOT a user fact, and never belongs in either bucket. If the user says "you're really curious" or "SNH tends to be verbose", that is an observation about the AI — SKIP it entirely. The AI records its own self-observations through a separate pipeline.
- Do NOT extract general knowledge, web search results, trivia, or anything the AI told the user.
- Write facts in the third person, starting with "User": "User has…", "User's MSP is…". Never "Assistant has…", never first person.

=== NAMES AND IDENTITY ===
Only record the user's own name, pronouns, or family relationships when the user INTRODUCES them outright — "my name is…", "call me…", "my pronouns are…". A name that merely appears in passing, especially in a message that reads as garbled or mis-transcribed, is not evidence of anything. Leave it out.

=== CORRECTIONS ===
If the user's message carries corrective framing — "It was X not Y", "actually it's X", "I was spelling it wrong", "her name is actually…", "I no longer…", "we renamed…" — return that fact as an OBJECT instead of a string:
  {"fact": "<the corrected fact>", "corrects": "<one short line naming the outdated belief being replaced>"}
Example: "Its Bernice, i was spelling it wrong lol. She is the Director of Rooms here at ISH." →
  {"fact": "User's colleague Bernice is the Director of Rooms at ISH", "corrects": "earlier facts calling her 'Bernie' or describing her as a manager at ISH"}
Plain new facts stay plain strings.

=== ITEM SHAPES ===
- facts:  a string, or a correction object as above.
- events: a string — one plain sentence, time reference kept.

Skip greetings, small talk, and questions with no answer. Return {"facts": [], "events": []} when there is nothing worth recording.

${getCurrentDateTimeString()}. Use the current date only to make an EVENT's wording unambiguous — never to stamp a fact.`;

    const exchange = `WHAT THE USER ACTUALLY SAID (authoritative — attribute only from this):\n${userMessage}\n\nASSISTANT RESPONSE (for context only — do NOT extract anything from this):\n${assistantMessage}`;

    let response;
    const controller = new AbortController();
    // 30s was sized for a model that answers immediately. A reasoning model
    // thinks first, and on the three longest exchanges of a real conversation
    // the uncapped call took 30.8s worst case — just past the edge, which is
    // exactly the timeout seen live. Bounding the thinking brings that to 19.6s
    // while reproducing the same extraction, so the timeout is config now and
    // defaults to the old 30s for a model that needs none of this.
    const extractionTimeoutMs = Number.isFinite(getConfig().generation?.extractionTimeoutMs)
      ? getConfig().generation.extractionTimeoutMs
      : 30000;
    const timeoutId = setTimeout(() => controller.abort(), extractionTimeoutMs);

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
          return { facts: [], events: [] };
      }
    } finally {
      clearTimeout(timeoutId);
    }

    const parsed = parseCandidatesFromResponse(response);
    const corrections = parsed.facts.filter(f => f.corrects).length;
    console.log(`[FactExtractor] Extracted ${parsed.facts.length} fact(s), ${parsed.events.length} event(s)${corrections ? ` (${corrections} correction${corrections > 1 ? 's' : ''})` : ''}`);
    return parsed;

  } catch (error) {
    if (error.name === 'AbortError') {
      console.error('[FactExtractor] Extraction timeout after 30s');
    } else {
      console.error('[FactExtractor] Error extracting candidates:', error.message);
    }
    return { facts: [], events: [] };
  }
}

/**
 * Legacy array-returning wrapper. `scripts/test-correction-supersession.js`
 * asserts on the {text, corrects} shape, and that assertion is still the right
 * one — corrections are unchanged by this rewrite.
 * @returns {Promise<Array<{text: string, corrects: string|null}>>}
 */
async function extractFacts(userMessage, assistantMessage, provider, model, apiKey, ollamaHost) {
  const { facts } = await extractCandidates(userMessage, assistantMessage, provider, model, apiKey, ollamaHost);
  return facts;
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
  // No max_tokens here, deliberately — extraction's output is a JSON document
  // whose length depends on the exchange, and capping it truncates facts. What
  // IS bounded is the thinking, which is what ran the call past its timeout.
  // Measured over the three longest exchanges of a real conversation: uncapped
  // 30.8s worst, budget 768 gives 19.6s and the same 4 facts / 1 event.
  const gen = getConfig().generation || {};
  const think = Number.isFinite(gen.extractionThinkingTokens) ? gen.extractionThinkingTokens : null;
  const body = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: exchange }
    ],
    stream: false
  };
  if (think > 0) body.thinking_token_budget = think;
  if (gen.reasoningEffort) body.reasoning_effort = gen.reasoningEffort;

  const response = await fetch(`${host}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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
 * Pull the first JSON value out of an LLM response, tolerating markdown fences
 * and Python-style single quotes.
 * @param {string} response
 * @param {'object'|'array'} shape
 * @returns {any|null}
 */
function parseJsonBlob(response, shape) {
  const cleaned = String(response || '').replace(/```(?:json)?\s*\n?([\s\S]*?)```/g, '$1').trim();
  const re = shape === 'object' ? /\{[\s\S]*\}/ : /\[[\s\S]*\]/;
  const match = cleaned.match(re);
  if (!match) return null;

  let jsonStr = match[0];
  try {
    return JSON.parse(jsonStr);
  } catch {
    // Python-style single-quoted output: ['fact', 'fact']. Convert structural
    // single quotes while preserving apostrophes inside strings ("User's name").
    jsonStr = jsonStr
      .replace(/\[\s*'/g, '["')
      .replace(/'\s*\]/g, '"]')
      .replace(/'\s*,\s*'/g, '", "')
      .replace(/'\s*,\s*"/g, '", "')
      .replace(/"\s*,\s*'/g, '", "');
    try {
      return JSON.parse(jsonStr);
    } catch (retryErr) {
      console.error('[FactExtractor] JSON parse failed:', retryErr.message);
      return null;
    }
  }
}

/**
 * Parse the two-bucket extraction response: {"facts": [...], "events": [...]}.
 * A model that ignores the shape and returns a bare array is treated as having
 * returned facts only — the deterministic event markers in db/extraction-rules
 * still re-route anything time-bound, so a sloppy response degrades to the old
 * behaviour plus the mechanical floor, not to no routing at all.
 * @returns {{facts: Array<{text: string, corrects: string|null}>, events: Array<{text: string}>}}
 */
function parseCandidatesFromResponse(response) {
  try {
    const obj = parseJsonBlob(response, 'object');
    if (obj && !Array.isArray(obj) && (Array.isArray(obj.facts) || Array.isArray(obj.events))) {
      return {
        facts: normalizeFactItems(Array.isArray(obj.facts) ? obj.facts : []),
        events: normalizeEventItems(Array.isArray(obj.events) ? obj.events : [])
      };
    }
    const arr = parseJsonBlob(response, 'array');
    if (Array.isArray(arr)) {
      console.log('[FactExtractor] Response was a bare array — treating every item as a fact candidate');
      return { facts: normalizeFactItems(arr), events: [] };
    }
    console.log('[FactExtractor] No parseable JSON in extraction response');
    return { facts: [], events: [] };
  } catch (error) {
    console.error('[FactExtractor] Error parsing candidates from response:', error.message);
    return { facts: [], events: [] };
  }
}

/** Normalize event items to {text}. Items are strings or {event}/{text} objects. */
function normalizeEventItems(items) {
  return items.map(e => {
    if (typeof e === 'string') return { text: e.trim() };
    if (e && typeof e === 'object') {
      const t = e.event || e.text || e.description;
      if (typeof t === 'string' && t.trim()) return { text: t.trim() };
    }
    return null;
  }).filter(Boolean).filter(e => e.text.length >= 4);
}

/**
 * Normalize fact items to {text, corrects} and drop the ones that are not user
 * facts at all. Items are plain strings, or correction objects ({fact, corrects})
 * when the user's message carried corrective framing ("It was X not Y") — the
 * corrects note travels with the fact so the contradiction judge sees the intent.
 * @returns {Array<{text: string, corrects: string|null}>}
 */
function normalizeFactItems(facts) {
  try {
    const normalized = facts.map(f => {
      if (typeof f === 'string') return { text: f, corrects: null };
      if (f && typeof f === 'object' && typeof f.fact === 'string') {
        const corrects = typeof f.corrects === 'string' && f.corrects.trim() ? f.corrects.trim() : null;
        return { text: f.fact, corrects };
      }
      return null;
    }).filter(Boolean);

    // Filter out empty strings and non-personal facts
    return normalized.filter(({ text: f }) => {
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
    console.error('[FactExtractor] Error normalizing fact items:', error.message);
    return [];
  }
}

// appendToMemory was removed 2026-08-02 along with MEMORY.md as a store.
//
// It wrote facts as "- <fact> (learned <when>)" lines into data/memory/MEMORY.md,
// deduping against the FILE by exact text and embedding similarity. That dedup
// guarded the projection while assignToCluster inserted into SQLite regardless,
// so the two stores disagreed by construction — one machine-gun line in the file,
// three rows in the database.
//
// The injected block is now rendered from SQLite per request
// (memoryClusters.renderLongTermMemory) and dedup happens at the SQLite write
// (factStore.findExactDuplicate). Nothing writes a memory file.

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
function prependDailyEntry(entry, dailyDir, date, headerLabel = 'Daily Log') {
  if (!fs.existsSync(dailyDir)) {
    fs.mkdirSync(dailyDir, { recursive: true });
  }

  const dailyFile = path.join(dailyDir, `${date}.md`);
  const header = `# ${headerLabel} - ${date}\n\n`;

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
function appendToDailyLog(summary, dailyDir, dateStamp = null) {
  try {
    if (!summary || summary.trim().length === 0) {
      return;
    }

    // dateStamp exists for the REPLAY. An event pulled out of a conversation
    // from 4 July belongs in 4 July's log, not in today's — a replay that dumped
    // eight months of events into one day would be manufacturing a day that
    // never happened. Live intake passes nothing and gets today, as before.
    const now = new Date();
    const date = dateStamp || getLocalDateStamp(now); // local Pacific YYYY-MM-DD
    const time = now.toTimeString().slice(0, 5); // HH:MM

    const entry = `### ${time}\n- ${summary}\n\n`;
    const dailyFile = prependDailyEntry(entry, dailyDir, date);
    console.log(`[FactExtractor] Prepended to daily log: ${dailyFile}`);

  } catch (error) {
    console.error('[FactExtractor] Error appending to daily log:', error.message);
  }
}

/**
 * Append an OPERATIONAL event (errors, timeouts, liveness/circuit-breaker,
 * background telemetry) to a separate ops log — NOT the daily log. The daily
 * log stays cognitively meaningful (facts, supersessions, salience reasoning,
 * reflections, initiatives) so it can be injected into chat context cheaply;
 * ops events are surfaced only in the Thinking tab and are never injected.
 * @param {string} summary - Ops event text
 * @param {string} opsDir - Path to ops log directory (e.g. data/memory/ops)
 */
function appendToOpsLog(summary, opsDir) {
  try {
    if (!summary || summary.trim().length === 0) {
      return;
    }
    const now = new Date();
    const date = getLocalDateStamp(now);
    const time = now.toTimeString().slice(0, 5);
    const entry = `### ${time}\n- ${summary}\n\n`;
    prependDailyEntry(entry, opsDir, date, 'Ops Log');
  } catch (error) {
    console.error('[FactExtractor] Error appending to ops log:', error.message);
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

    // Long-term memory is rendered from SQLite, not read from a file. Callers
    // that need it use memoryClusters.renderLongTermMemory().
    result.memory = '';

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
 * @param {Object} [context] - Source context so corrective intent survives to the verdict
 * @param {string} [context.userMessage] - The user message the new fact was extracted from
 * @param {string} [context.corrects] - Extractor's note on what the new fact corrects/replaces
 * @returns {Promise<{verdict: 'yes'|'no'|'uncertain', reasoning: string}>}
 */
async function judgeContradiction(newFact, oldFact, context = {}) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You are a fact contradiction detector for a personal memory system. You are given an EXISTING stored fact about the user and a NEW statement the user just made about themselves.

Decide the relationship between them:
- YES — they contradict: they cannot both be true of the user at the same time. Corrections and replacements count ("Actually my MSP is X, not Y", "I moved to Z", "I no longer use Q"). A correction of a name, spelling, or title counts even when the two statements COULD describe different people — if the user is correcting the record, the old version is wrong, not a second person.
- NO — no contradiction: additional detail, refinement, or an unrelated fact.
- UNCERTAIN — you genuinely cannot tell whether they conflict without more information (e.g. they might refer to two different things, or one might update the other, but it is ambiguous).

Prefer UNCERTAIN over guessing when it is truly ambiguous.

Respond with exactly YES, NO, or UNCERTAIN on the first line, then one short line of reasoning.`;

    // Corrective intent lives in the user's message, not the extracted fact
    // strings ("It was Bernice not Bernie" extracts to a fact about Bernice
    // that no longer mentions Bernie) — so pass the source alongside.
    const contextParts = [];
    if (context.corrects) {
      contextParts.push(`The NEW statement was extracted from a message where the user was CORRECTING earlier information. It replaces: ${context.corrects}. If the EXISTING fact is (or contains) that outdated belief, they contradict.`);
    }
    if (context.userMessage) {
      const msg = String(context.userMessage).slice(0, 600);
      contextParts.push(`The user's original message, for context:\n"${msg}"`);
    }
    const userPrompt = `EXISTING fact: "${oldFact}"\nNEW statement: "${newFact}"\n${contextParts.length ? `\n${contextParts.join('\n\n')}\n` : ''}\nDoes the NEW statement contradict the EXISTING fact?`;

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
 * REPEAT detection, second half. Vector similarity finds the neighbours; this
 * decides whether a near-neighbour is the SAME ASSERTION restated, as opposed to
 * a related-but-different fact.
 *
 * The bar is deliberately strict and the default is NO. A false positive here
 * silently discards a genuinely new fact, which is the worst outcome available:
 * unlike a false negative (a second row, which the corrector can merge later),
 * nothing downstream can recover information that was never written.
 *
 * @returns {Promise<{same: boolean, reasoning: string}>}
 */
async function judgeSameAssertion(newFact, oldFact) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You decide whether two sentences about a user assert THE SAME THING, so that storing both would be storing one fact twice.

Answer SAME only when the second adds no information the first does not already carry — a rewording, a restatement, the same claim in different words.

Answer DIFFERENT when either sentence carries anything the other does not: an extra detail, a narrower or wider claim, a different object, a different attribute. "User has a dog named Casper" and "User has a dog named Casper who helps pull them up hills" are DIFFERENT — the second knows something the first does not.

If you are unsure, answer DIFFERENT.

Respond with exactly SAME or DIFFERENT on the first line, then one short line of reasoning.`;
    const userPrompt = `EXISTING fact: "${oldFact}"\nNEW statement: "${newFact}"\n\nSame assertion, or different?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 100 });
    const firstWord = (String(content).trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    const same = firstWord === 'same';
    const reasoning = String(content).trim().split('\n').slice(0, 2).join(' ').trim();
    console.log(`[FactExtractor] Repeat judge: ${same ? 'SAME' : 'DIFFERENT'} — "${newFact}" vs "${oldFact}"`);
    return { same, reasoning };
  } catch (error) {
    console.error('[FactExtractor] Repeat judge error:', error.message);
    return { same: false, reasoning: '' }; // a failed check must never eat a fact
  }
}

/**
 * Does one of these two facts already contain the other?
 *
 * A DIFFERENT question from judgeSameAssertion, and it has to be, because that
 * judge is defined the other way for exactly this case: at intake, "User has a
 * dog named Casper who helps pull them up hills" arriving against a stored "User
 * has a dog named Casper" carries new information and must be stored, so the
 * repeat judge answers DIFFERENT — the fixture pair is written into its prompt
 * as the worked example.
 *
 * The corrector is asking something else. Both rows are already held, and a
 * subset sitting beside its superset is one fact stored twice with one copy
 * impoverished. So: is everything the shorter one asserts already asserted by
 * the longer one? If yes, the shorter folds into it, and the DETAIL is what
 * survives — never the reverse.
 *
 * Deliberately strict. "Neither" is the safe answer and the default on any doubt,
 * because folding two facts that merely overlap loses whatever the loser knew.
 *
 * @returns {Promise<{relation: 'a-contains-b'|'b-contains-a'|'neither', reasoning: string}>}
 */
async function judgeSubsumption(a, b) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You are given two sentences about the same person, both currently stored as separate facts. Decide whether one of them ALREADY CONTAINS everything the other says.

Answer A if sentence A asserts everything B asserts and more.
Answer B if sentence B asserts everything A asserts and more.
Answer NEITHER if each one knows something the other does not, or if they are about different things.

"User has a dog named Casper" and "User has a dog named Casper who helps pull them up hills during walks" → the second contains the first entirely, so the answer is whichever letter that longer sentence is.
"User has a dog named Casper" and "User has a cat named Mia" → NEITHER.
"User works at ISH" and "User works 20 hours a week" → NEITHER; each carries something the other does not.

If you are unsure, answer NEITHER.

Respond with exactly A, B, or NEITHER on the first line, then one short line of reasoning.`;
    const userPrompt = `A: "${a}"\nB: "${b}"\n\nDoes one already contain the other?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 100 });
    const first = (String(content).trim().match(/[a-zA-Z]+/) || [''])[0].toUpperCase();
    const relation = first === 'A' ? 'a-contains-b' : first === 'B' ? 'b-contains-a' : 'neither';
    const reasoning = String(content).trim().split('\n').slice(0, 2).join(' ').trim();
    console.log(`[FactExtractor] Subsumption judge: ${relation} — "${String(a).slice(0, 70)}" / "${String(b).slice(0, 70)}"`);
    return { relation, reasoning };
  } catch (error) {
    console.error('[FactExtractor] Subsumption judge error:', error.message);
    return { relation: 'neither', reasoning: '' }; // a failed check must never fold a fact away
  }
}

/**
 * Who is a STORED fact actually about?
 *
 * A different question from memory-write's classifySubject, and it needs a
 * different judge. That one routes a live request: it has a speaker, it is handed
 * the verbatim utterance as the authority, and it keys on pronouns — "you're very
 * direct" is SELF, "I prefer Y" from the human is USER. It cannot answer this,
 * because by the time a row is in cluster_members the speaker is gone and the
 * grammar has already been normalised to the third person. Every row this is
 * asked about starts with "User", which is precisely what classifySubject would
 * read as the answer.
 *
 * So this asks about CONTENT rather than form: strip the "User" and ask whose
 * behaviour is being described. Found because the daily-log archiver had been
 * filing Aurelius's self-observations as Ellie's preferences — "User aims to be a
 * steady, non-judgmental presence that respects boundaries and cognitive load"
 * describes an assistant, and `verifySubjectAgreement` passes it happily because
 * the grammar is impeccable. Roughly ten a day at the time it was noticed.
 *
 * UNSURE MEANS USER. Reclassifying a fact about Ellie as a fact about the
 * assistant would take something true about her out of her own corpus, which is
 * the worse of the two errors by a distance.
 *
 * @param {string} text - the stored sentence, as held
 * @returns {Promise<{subject: 'user'|'self', reasoning: string}>}
 */
async function judgeStoredSubject(text) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `A memory system stores facts about a HUMAN and, separately, facts an AI assistant holds about ITSELF. Every stored sentence below is written in the third person starting with "User", because that is the format — the grammar tells you nothing about who it is really about. Your job is to read the CONTENT and say whose behaviour is being described.

Answer SELF when the sentence describes an assistant doing assistant things: adopting a tone for someone, holding space for someone, being a non-judgmental presence, asking probing or diagnostic questions to draw someone out, validating someone's feelings, acting as a sounding board or an interface for someone, protecting its own stored identity against being renamed, summarising in order to invite a reply. These are things a conversational assistant does FOR a person.

Answer USER when the sentence describes the human's own life, work, possessions, relationships, body, habits, opinions or preferences about the world. "User has blue eyes", "User has a pet named Roscoe", "User prefers to be told directly when she has made a mistake", "User works at ISH", "User is building SNH" — all USER.

The distinction is who is doing the thing. "Prefers to be told directly when she has made a mistake" is the human stating how she wants to be treated: USER. "Aims to be a steady, non-judgmental presence that respects boundaries" is the assistant describing how it treats someone: SELF.

If you are unsure, answer USER. Wrongly moving a fact out of the human's memory is much worse than leaving one in.

Respond with exactly SELF or USER on the first line, then one short line of reasoning.`;
    const userPrompt = `Stored sentence: "${text}"\n\nWhose behaviour does this describe — SELF or USER?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 120 });
    const firstWord = (String(content).trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    const subject = firstWord === 'self' ? 'self' : 'user';
    const reasoning = String(content).trim().split('\n').slice(0, 2).join(' ').trim();
    console.log(`[FactExtractor] Stored-subject judge: ${subject.toUpperCase()} — "${String(text).slice(0, 70)}"`);
    return { subject, reasoning };
  } catch (error) {
    console.error('[FactExtractor] judgeStoredSubject error:', error.message);
    return { subject: 'user', reasoning: '' };   // a failed check never moves a fact
  }
}

/**
 * The strip-the-timestamp test, asked properly.
 *
 * At intake a time marker can force the event branch outright, and that is fine:
 * a fresh statement carrying "as of July 2026" is almost always an event, and a
 * durable fact missed there re-extracts the next time it comes up. The cost of a
 * false positive is one lost extraction.
 *
 * For the CORRECTOR the arithmetic is reversed. It acts on facts already held,
 * and the cost of a false positive is retiring something true. The first dry run
 * made the case unanswerably: the marker heuristic alone proposed expiring
 * "User's partner passed away on January 24th, 2025, at 4:00 AM" — a date-bearing
 * sentence that is among the most durable facts in the corpus — along with every
 * dated capability declaration he holds about himself.
 *
 * So the marker is a CANDIDATE FILTER, and this is the actual test: remove the
 * time reference and ask whether anything durable is left. "Partner passed away"
 * survives it. "Dog had a restless night" does not.
 *
 * @returns {Promise<{isEvent: boolean, reasoning: string}>}
 */
async function judgeStripTheTimestamp(text) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You decide whether a stored sentence is a LASTING FACT or a PASSING EVENT.

The test: remove every reference to time from the sentence, then ask whether what remains is still true next year and still worth knowing.

- LASTING — something that stays true. "Her partner died on 24 January 2025" → strip the date → "her partner died" → still true forever. LASTING. Life events, deaths, births, when something was founded, a diagnosis, a permanent change, a standing habit, a capability someone has — all LASTING, even when the sentence names a date.
- PASSING — something that was only true around then. "The dog had a restless night" → strip the time → nothing durable remains. "She is visiting the cheese factory today", "the cleaners are filling holes this week", "she is tired at the moment" — all PASSING.

A date in the sentence does NOT make it passing. Plenty of permanent facts are dated. Ask only whether the underlying thing endures.

A MOOD or an ENERGY LEVEL is PASSING even with no date attached: "she is exhausted", "he has lost motivation", "she is experiencing burnout and low motivation" — how someone feels at a point in their life is not a standing truth about them. A DIAGNOSIS or a chronic condition is LASTING. If the sentence describes how someone is doing rather than how someone is, it is PASSING.

If you are unsure, answer LASTING. This decides whether a stored memory is retired, and wrongly retiring something true is far worse than keeping something stale.

Answer with exactly LASTING or PASSING on the first line, then one short line of reasoning.`;
    const userPrompt = `Stored sentence: "${text}"\n\nLASTING or PASSING?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 100 });
    const firstWord = (String(content).trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    const isEvent = firstWord === 'passing';
    const reasoning = String(content).trim().split('\n').slice(0, 2).join(' ').trim();
    console.log(`[FactExtractor] Strip-the-timestamp: ${isEvent ? 'PASSING' : 'LASTING'} — "${String(text).slice(0, 70)}"`);
    return { isEvent, reasoning };
  } catch (error) {
    console.error('[FactExtractor] judgeStripTheTimestamp error:', error.message);
    return { isEvent: false, reasoning: '' }; // a failed check never retires a fact
  }
}

/**
 * Which of two facts asserting the same thing should SURVIVE?
 *
 * Length was the first proxy and it is wrong often enough to matter: the dry run
 * picked "User's Managed Service Provider (MSP) is called MettaSphere." over
 * "User's MSP is MettaSphere LLC" because it is longer, silently dropping "LLC".
 * More words is not more information.
 *
 * @returns {Promise<'a'|'b'|null>} null when it genuinely does not matter
 */
async function judgeWhichSurvives(a, b) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `Two sentences record the same fact. One will be kept and the other discarded. Choose the one that should be KEPT.

Keep whichever carries MORE information — more specific names, qualifiers, or detail. Length is not information: a wordier sentence that drops a detail is the worse one.
If they carry exactly the same information, answer EITHER.

Answer with exactly A, B, or EITHER on the first line, then one short line of reasoning.`;
    const userPrompt = `A: "${a}"\nB: "${b}"\n\nWhich should be kept?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 80 });
    const first = (String(content).trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    if (first === 'a') return 'a';
    if (first === 'b') return 'b';
    return null;
  } catch (error) {
    console.error('[FactExtractor] judgeWhichSurvives error:', error.message);
    return null;
  }
}

/**
 * Split a candidate that still joins unrelated assertions after the extraction
 * prompt asked for atoms. Only called when db/extraction-rules.looksCompound
 * fires, so this costs nothing on the ordinary path.
 *
 * Returns the original text unchanged on any failure — a fact stored whole is a
 * worse fact, but a fact lost to a parse error is no fact at all.
 *
 * @returns {Promise<string[]>}
 */
async function splitCompoundFact(text, subject = 'user') {
  try {
    const memoryManager = require('./memory-manager');
    const rules = require('./extraction-rules');
    const isSelf = subject === 'self';

    // SUBJECT-AWARE, and this is not a nicety.
    //
    // The original prompt said "start each part with User", which is right for a
    // user fact and catastrophic for a self-fact. The corrector's first dry run
    // caught it turning "I tend to view my own existence through the lens of my
    // architectural design" into "User tends to view their own existence…" — his
    // self-observation rewritten as a third-person belief about Ellie. That is
    // the 2026-07-27 misattribution exactly, arriving by a new route.
    const person = isSelf
      ? 'Every part must be written in the FIRST PERSON, as the speaker talking about themselves, and must start with "I" or "My". Never write "User" or "the user" — these are the speaker\'s statements about THEMSELF.'
      : 'Every part must be written in the THIRD PERSON and must start with "User". Never write it in the first person.';

    const example = isSelf
      ? '- "I value rigor and I use architectural metaphors" → "I value rigor.", "I use architectural metaphors."'
      : '- "User\'s professional entities include her MSP, MettaSphere, and her AI research venture, Coastal Squatch" → "User\'s MSP is MettaSphere.", "User\'s AI research venture is Coastal Squatch."';

    const systemPrompt = `You split a sentence into atomic facts. One atomic fact asserts ONE thing about ONE subject.

Rules:
- ${person}
- REWRITE each part as a natural standalone sentence. Do not simply cut the original at its commas.
${example}
- A reader who sees only that one sentence must understand it.
- Do not invent anything, and do not LOSE anything. Every detail in the original must appear in one of the atoms, and every atom must be supported by the original.
- A clause describing the SAME single thing stays with it: "${isSelf ? 'I have a habit of pausing when I am unsure, which slows me down' : 'User has a dog named Casper who helps pull them up hills'}" is ONE fact.
- If the sentence is already atomic, return it unchanged as the only item.

Return ONLY a JSON array of strings.`;
    const userPrompt = `Sentence: "${text}"\n\nAtomic facts:`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 300 });
    const arr = parseJsonBlob(content, 'array');
    if (!Array.isArray(arr)) return [text];

    // Verify, do not trust. The grammatical subject of every atom must match the
    // subject of the fact being split; an atom that drifted is dropped, and if
    // any did, the whole split is abandoned rather than half-applied — a partial
    // split would supersede the original with an incomplete set.
    const parts = arr
      .filter(x => typeof x === 'string')
      .map(x => x.trim())
      .filter(x => x.length >= 8);
    const wellFormed = parts.filter(x => rules.grammaticalSubject(x) === subject);
    if (wellFormed.length !== parts.length) {
      console.warn(`[FactExtractor] compound split changed subject on ${parts.length - wellFormed.length} atom(s) — abandoning the split of "${text.slice(0, 60)}"`);
      return [text];
    }
    if (wellFormed.length === 0) return [text];
    if (wellFormed.length === 1) return wellFormed;

    // ENTAILMENT CHECK. Splitting an awkward sentence produces awkward atoms, and
    // an awkward atom is a false fact. The corrector's dry run turned "User has a
    // RAV4 that has been washed, but the wheels, tires, wax and ceramic coating
    // have not been done" into, among others, "User's RAV4 wheels have not been
    // applied" — wheels are not applied to anything. Six rows of near-nonsense
    // would then have superseded one coherent compound.
    //
    // One call, only on the split path, and it fails CLOSED: anything less than a
    // clean yes abandons the split and leaves the original alone.
    const check = await memoryManager.callLLM(
      `You check whether a set of split sentences faithfully covers an original sentence.

Answer NO if any part invents something the original does not say, if any part is garbled or not a sensible English sentence, or if the parts together lose something the original says.
Answer YES only if every part is sensible, supported by the original, and nothing is lost.

Answer with exactly YES or NO on the first line, then one short line of reasoning.`,
      `Original: "${text}"\n\nParts:\n${wellFormed.map(p => `- ${p}`).join('\n')}\n\nFaithful?`,
      { maxTokens: 100 }
    );
    const ok = /^\s*yes\b/i.test(String(check.content || '').trim());
    if (!ok) {
      console.warn(`[FactExtractor] compound split failed the faithfulness check, leaving original alone: "${text.slice(0, 60)}"`);
      return [text];
    }

    console.log(`[FactExtractor] Split compound into ${wellFormed.length} (${subject}): "${text.slice(0, 70)}"`);
    return wellFormed;
  } catch (error) {
    console.error('[FactExtractor] splitCompoundFact error:', error.message);
    return [text];
  }
}

/**
 * Classify a self-fact as a behavioral CLAIM or a DECLARATION — the auditability
 * split the self-coherence audit depends on. CLAIMS describe how SNH acts, values,
 * or approaches things ("I value precision", "I'm an analytical partner") and can
 * be tested against how it actually behaved, so the audit samples them.
 * DECLARATIONS are facts about SNH's name, stated preferences, or history — true
 * by assertion, not testable against behavior — so the audit leaves them alone.
 * Defaults to 'declaration' on any doubt: mis-tagging a claim as a declaration
 * just means it never gets audited (safe), while the reverse would audit an
 * untestable fact and manufacture false "gaps".
 * @param {string} text - the self-fact
 * @returns {Promise<'claim'|'declaration'>}
 */
async function classifyClaimType(text) {
  try {
    const memoryManager = require('./memory-manager');
    const systemPrompt = `You sort an AI's self-statements into two kinds:
- CLAIM: about how it behaves, what it values, or how it approaches things — something you could check against what it actually did. Examples: "I value precision", "I tend to reframe problems structurally", "I'm an analytical partner", "I notice when I'm repeating myself and change course".
- DECLARATION: about its name, its stated preferences, or its history — true because it's asserted, not something you'd verify by watching behavior. Examples: "My name is SNH", "I prefer to be addressed plainly", "I was first run on July 3rd", "I run on a machine called Squatch Neuro Hub".

Answer with exactly one word on the first line: CLAIM or DECLARATION. If it could be either, answer DECLARATION.`;
    const userPrompt = `Self-statement: "${text}"\n\nCLAIM or DECLARATION?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 8 });
    const firstWord = (String(content).trim().match(/[a-zA-Z]+/) || [''])[0].toLowerCase();
    const claimType = firstWord === 'claim' ? 'claim' : 'declaration';
    console.log(`[FactExtractor] Claim-type: ${claimType.toUpperCase()} — "${String(text).slice(0, 70)}"`);
    return claimType;
  } catch (error) {
    console.error('[FactExtractor] classifyClaimType error:', error.message);
    return 'declaration'; // safe default — an unaudited fact, never a false gap
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
 * Answer-aware gap check (Layer 2): before queueing a gap question, see whether
 * memory already answers it. Searches clusters for the question's topic and asks
 * the LLM whether the known facts make the question redundant. A gap question is
 * always topically similar to its own source facts, so a pure similarity gate
 * would suppress everything — the LLM judge is what distinguishes "the answer is
 * already here" from "this is a genuine open question about a known topic".
 * @param {string} question - The candidate gap question
 * @returns {Promise<{evidence: string}|null>} evidence fact if already answered, else null
 */
async function gapAlreadyAnswered(question) {
  try {
    const memoryClusters = require('./memory-clusters');
    // Member-level semantic search (not cluster-aggregated) so we actually pull
    // the individual facts most similar to the question. Restricted to active
    // user-facts, which also keeps self-observations out of the judge context.
    const candidates = await memoryClusters.findContradictionCandidates(question, {
      subject: 'user', limit: 8, threshold: 0.4
    });
    const facts = (candidates || []).map(c => c.content).filter(Boolean);
    if (facts.length === 0) return null;

    const memoryManager = require('./memory-manager');
    const systemPrompt = `You decide whether a set of already-known facts sufficiently answers a clarifying question, so it need NOT be asked. Line 1: exactly YES or NO. If YES, line 2: the single known fact that best answers it, verbatim.`;
    const userPrompt = `Clarifying question under consideration:\n"${question}"\n\nAlready-known facts:\n${facts.map(f => `- ${f}`).join('\n')}\n\nDo these facts already answer the question well enough that asking would be redundant?`;
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 120 });
    const lines = content.trim().split('\n');
    if (!/^\s*yes\b/i.test(lines[0] || '')) return null;
    const evidence = (lines[1] || '').replace(/^[-*\d.\s"]+/, '').replace(/"$/, '').trim() || facts[0];
    return { evidence };
  } catch (error) {
    console.error('[FactExtractor] gapAlreadyAnswered error:', error.message);
    return null;
  }
}

/**
 * Backlog sweep: run every pending (never-surfaced) question through the
 * answer-aware gate and retire the ones memory already answers. The mint-time
 * gate (gapAlreadyAnswered, Layer 2) only screens NEW questions — anything
 * queued before a gate improvement landed is grandfathered in and can sit
 * pending for days before being asked ("What is ISH?" sat 8 days despite a
 * defining fact predating it). Running this from the heartbeat makes every
 * future gate improvement retroactive too.
 *
 * Retired questions flip to 'answered' via the normal markAnswered path;
 * initiatives already minted from them are dismissed by noticeFromQuestions'
 * self-heal on the same heartbeat cycle (it runs after this sweep).
 * @param {Object} [opts]
 * @param {boolean} [opts.dryRun] - Report what would be retired without retiring
 * @param {string} [opts.dailyDir] - For the audit trail
 * @returns {Promise<{swept: number, retired: Array<{id: string, question: string, evidence: string}>}>}
 */
async function sweepPendingQuestions(opts = {}) {
  const questions = require('./questions');
  const dailyDir = opts.dailyDir || DAILY_DIR;
  const result = { swept: 0, retired: [] };
  try {
    const pending = questions.listPending(200);
    for (const q of pending) {
      result.swept++;
      try {
        const already = await gapAlreadyAnswered(q.question);
        if (!already) continue;
        if (opts.dryRun) {
          result.retired.push({ id: q.id, question: q.question, evidence: already.evidence });
          continue;
        }
        if (questions.markAnswered(q.id)) {
          result.retired.push({ id: q.id, question: q.question, evidence: already.evidence });
          appendToDailyLog(`Retired pending question (memory already answers it): "${q.question}" ← "${already.evidence}"`, dailyDir);
        }
      } catch (err) {
        // One bad question (or a wedged brain call) shouldn't sink the sweep.
        console.error(`[FactExtractor] Question sweep error on ${String(q.id).slice(0, 8)}:`, err.message);
      }
    }
    if (result.swept > 0) {
      console.log(`[FactExtractor] Question sweep: ${result.swept} pending checked, ${result.retired.length} ${opts.dryRun ? 'would be ' : ''}retired`);
    }
  } catch (error) {
    console.error('[FactExtractor] sweepPendingQuestions error:', error.message);
  }
  return result;
}

/**
 * Answer detection (Layer 3): retire questions the user's latest message
 * answers. Two passes:
 *  (a) precise — questions surfaced (asked) in THIS conversation, judged
 *      unconditionally (cheap, and the original narrow behavior).
 *  (b) broad — ANY outstanding question (pending or asked, in ANY conversation)
 *      whose topic matches this message, embedding-gated so only the few
 *      topically-close ones reach the LLM judge. This is the actual bug fix: a
 *      question answered in a different conversation, or before it was ever
 *      surfaced, now gets flipped to answered instead of lingering and re-asking.
 * @param {string} userMessage
 * @param {string} conversationId
 * @param {string} dailyDir - for the audit trail (optional)
 * @returns {Promise<string[]>} ids of questions marked answered
 */
async function detectAnswers(userMessage, conversationId = null, dailyDir = DAILY_DIR) {
  const questions = require('./questions');
  const answered = [];
  if (!userMessage || !userMessage.trim()) return answered;

  const markIt = (q, note) => {
    if (questions.markAnswered(q.id)) {
      answered.push(q.id);
      if (dailyDir) appendToDailyLog(`Question answered${note ? ` (${note})` : ''}: "${q.question}"`, dailyDir);
    }
  };

  // (a) Precise: questions asked in this conversation.
  if (conversationId) {
    try {
      for (const q of questions.getAskedForConversation(conversationId)) {
        if (await judgeAnswered(q.question, userMessage)) markIt(q);
      }
    } catch (err) {
      console.error('[FactExtractor] Answer detection (this-convo) error:', err.message);
    }
  }

  // (b) Broad: topic-matched outstanding questions across all conversations.
  try {
    const outstanding = questions.getOutstanding();
    if (outstanding.length > 0) {
      const memoryClusters = require('./memory-clusters');
      const msgEmb = await memoryClusters.generateEmbedding(userMessage);
      if (msgEmb) {
        const cfg = getConfig();
        const floor = Number.isFinite(cfg.questions?.answerMatchFloor) ? cfg.questions.answerMatchFloor : 0.40;
        const maxJudge = Number.isInteger(cfg.questions?.answerMaxJudge) ? cfg.questions.answerMaxJudge : 6;
        const scored = [];
        for (const q of outstanding) {
          const emb = questions.parseEmbedding(q.embedding) || await memoryClusters.generateEmbedding(q.question);
          if (!emb) continue;
          const sim = questions.cosineSim(msgEmb, emb);
          if (sim >= floor) scored.push({ q, sim });
        }
        scored.sort((a, b) => b.sim - a.sim);
        for (const { q, sim } of scored.slice(0, maxJudge)) {
          if (await judgeAnswered(q.question, userMessage)) {
            markIt(q, `topic match, sim ${sim.toFixed(2)}`);
          }
        }
      }
    }
  } catch (err) {
    console.error('[FactExtractor] Answer detection (broad) error:', err.message);
  }

  return answered;
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
 * API key for the EXTRACTION provider, from the environment.
 *
 * The old orchestrator threaded the CHAT request's key through to the extraction
 * call, which is wrong whenever the two providers differ — and they normally do,
 * since the extraction model is chosen in config independently of the chat
 * model. Local providers ignore the key entirely (which is why nothing broke),
 * but a cloud extraction model would have been handed the wrong one.
 */
function extractionApiKey(provider) {
  switch (String(provider || '').toLowerCase()) {
    case 'claude': return process.env.CLAUDE_API_KEY || '';
    case 'grok': return process.env.GROK_API_KEY || '';
    case 'openai': return process.env.OPENAI_API_KEY || '';
    default: return '';
  }
}

/**
 * ============================================================================
 * PASSIVE INTAKE — planned first, applied second
 * ============================================================================
 *
 * The pipeline is split in two on purpose. `planExtraction` decides EVERYTHING
 * and writes NOTHING: what the model proposed, which candidates were split,
 * which were routed to the day's log as events, which were refused by the
 * identity anchor, which are repeats of facts already held, what the
 * contradiction check considered and what the judge said about each candidate.
 * `applyExtraction` takes that plan and performs the writes, all of them through
 * the fact-store funnel.
 *
 * That split is what makes the dry-run harness (scripts/dryrun-extract.js) run
 * the REAL pipeline rather than a copy of it. The spec's rule for replay — "if
 * replay needs a special case, the pipeline is wrong" — applies just as much to
 * a dry run: a rehearsal that exercises different code proves nothing about the
 * code that will actually run.
 */

/**
 * Decide what an exchange should produce. Read-only: no rows, no vectors, no log
 * lines. Every LLM call it makes is a judgment, never a mutation.
 *
 * @param {Object} p
 * @param {string} p.userMessage
 * @param {string} p.assistantMessage
 * @param {string} [p.conversationId]
 * @param {string} [p.messageId]
 * @param {string} [p.inputModality] - 'stt' | 'typed' | 'unknown'
 * @returns {Promise<Object>} the plan (see the shape assembled at the end)
 */
async function planExtraction({
  userMessage, assistantMessage,
  conversationId = null, messageId = null, inputModality = 'unknown'
} = {}) {
  const config = getConfig();
  const rules = require('./extraction-rules');
  const memoryClusters = require('./memory-clusters');
  const factStore = require('./fact-store');

  const exCfg = config.memory?.extraction || {};
  const conCfg = config.memory?.contradiction || {};
  const gatedModalities = Array.isArray(exCfg.identityAnchorModalities)
    ? exCfg.identityAnchorModalities : ['stt', 'unknown'];
  const repeatFloor = Number.isFinite(exCfg.repeatSimilarityFloor) ? exCfg.repeatSimilarityFloor : 0.80;
  const repeatMax = Number.isInteger(exCfg.repeatMaxCandidates) ? exCfg.repeatMaxCandidates : 5;
  const maxFacts = Number.isInteger(exCfg.maxFactsPerExchange) ? exCfg.maxFactsPerExchange : 12;
  const maxEvents = Number.isInteger(exCfg.maxEventsPerExchange) ? exCfg.maxEventsPerExchange : 8;

  const extractionProvider = config.models.extraction.provider;
  const extractionModel = config.models.extraction.model;
  const extInst = getProviderInstance(extractionProvider, config.models.extraction.instance);
  const extractionHost = extInst ? extInst.host : 'http://localhost:11434';

  const modality = (inputModality || 'unknown').toLowerCase();
  const provenance = {
    conversationId,
    messageId,
    // The user's ACTUAL words, never the extractor's paraphrase. This is what
    // lets a later correction tell "User's name is Mike" — extracted from a
    // mis-transcribed "it's mic not picking up the right words" — apart from a
    // name Ellie actually stated.
    verbatimSourceText: userMessage,
    inputModality: modality
  };

  const plan = {
    conversationId, messageId, inputModality: modality,
    userMessage, assistantMessage,
    model: `${extractionProvider}/${extractionModel}`,
    proposed: { facts: [], events: [] },
    splits: [],      // {from, into[]}
    routedToLog: [], // {text, why, marker}
    refusals: [],    // {text, rule, detail}
    repeats: [],     // {text, existingId, existingContent, similarity, detectedBy, reasoning}
    facts: [],       // {text, corrects, salience, salienceRationale}
    recall: [],      // {fact, diagnostics, judged[]}
    supersessions: [],
    uncertainties: [],
    gapQuestion: null,
    truncated: { facts: 0, events: 0 },
    error: null
  };

  // ---- 1. Ask the model for atomic facts and time-bound events. ----
  const extracted = await extractCandidates(
    userMessage, assistantMessage, extractionProvider, extractionModel,
    extractionApiKey(extractionProvider), extractionHost
  );
  plan.proposed.facts = extracted.facts.map(f => ({ text: f.text, corrects: f.corrects }));
  plan.proposed.events = extracted.events.map(e => e.text);

  // ---- 2. Deterministic floor under the model's routing. ----
  // Atomicity first, THEN routing: a compound may hold a durable state and a
  // passing event in one sentence, and routing before splitting would throw the
  // state away with the event (or keep the event with the state).
  let candidates = [];
  for (const f of extracted.facts) {
    const compound = rules.looksCompound(f.text);
    if (!compound.compound) { candidates.push(f); continue; }
    const parts = await splitCompoundFact(f.text);
    if (parts.length <= 1) { candidates.push(f); continue; }
    plan.splits.push({ from: f.text, into: parts, why: compound.why });
    for (const part of parts) candidates.push({ text: part, corrects: f.corrects });
  }

  const eventTexts = [...plan.proposed.events];
  const survivors = [];
  for (const c of candidates) {
    // EVENT MARKERS — decisive regardless of which bucket the model chose.
    const marker = rules.eventMarker(c.text);
    if (marker.isEvent) {
      plan.routedToLog.push({
        text: c.text,
        why: `carries a ${marker.kind} time marker ("${marker.marker}") — strip it and nothing durable remains`,
        marker: marker.marker
      });
      eventTexts.push(c.text);
      continue;
    }

    // SUBJECT ATTRIBUTION — a user fact must name the user. An unanchored
    // sentence is how a self-observation slips into the user's corpus wearing no
    // pronoun at all (the 2026-07-27 misattribution, by a different route).
    const grammatical = rules.grammaticalSubject(c.text);
    if (grammatical !== 'user') {
      plan.refusals.push({
        text: c.text, rule: 'subject-attribution',
        detail: grammatical === 'self'
          ? 'written in the first person — an observation about the assistant, not about the user'
          : 'does not name the user, so it cannot be filed as a fact about her'
      });
      continue;
    }

    // IDENTITY ANCHOR — F1. A name, pronoun or core-relationship fact from a
    // message that was not provably typed needs an explicit self-introduction in
    // what was actually said.
    const refusal = rules.identityAnchorRefusal(c.text, userMessage, modality, gatedModalities);
    if (refusal) {
      plan.refusals.push({ text: c.text, rule: refusal.rule, detail: refusal.detail, klass: refusal.klass });
      continue;
    }

    // CAPABILITY / DEPLOYMENT — the manifest owns this ground and is config-gated,
    // so it cannot drift; a row here can, and would then contradict the manifest
    // from inside the same prompt. See db/extraction-rules.js for the full case.
    const capRefusal = rules.capabilityFactRefusal(c.text);
    if (capRefusal) {
      plan.refusals.push({ text: c.text, rule: capRefusal.rule, detail: capRefusal.detail });
      continue;
    }

    survivors.push(c);
  }

  if (survivors.length > maxFacts) {
    plan.truncated.facts = survivors.length - maxFacts;
    survivors.length = maxFacts;
  }
  if (eventTexts.length > maxEvents) {
    plan.truncated.events = eventTexts.length - maxEvents;
    eventTexts.length = maxEvents;
  }
  plan.events = eventTexts.map(t => ({ text: t }));

  if (survivors.length === 0) return plan;

  // ---- 3. REPEAT detection, before anything is written. ----
  // Exact match first (free), then semantic near-match (vector + judge). A
  // confirmed repeat never becomes a second row: it raises the held fact's
  // salience and records a corroboration against its provenance.
  const fresh = [];
  for (const c of survivors) {
    const exact = factStore.findExactDuplicate(c.text, 'user');
    if (exact) {
      plan.repeats.push({
        text: c.text, existingId: exact.id, existingContent: exact.content,
        existingSalience: exact.salience ?? 5, similarity: 1, detectedBy: 'exact',
        reasoning: 'byte-identical to a fact already held'
      });
      continue;
    }

    const { candidates: near } = await memoryClusters.findActiveNeighbours(c.text, {
      subject: 'user', threshold: repeatFloor, limit: repeatMax
    });
    let matched = null;
    for (const n of near) {
      const { same, reasoning } = await judgeSameAssertion(c.text, n.content);
      if (same) { matched = { n, reasoning }; break; }
    }
    if (matched) {
      plan.repeats.push({
        text: c.text, existingId: matched.n.memberId, existingContent: matched.n.content,
        existingSalience: matched.n.salience ?? 5, similarity: matched.n.similarity,
        detectedBy: 'semantic', reasoning: matched.reasoning
      });
      continue;
    }
    fresh.push(c);
  }

  // A repeat still gets scored. "She has said this before" can mean it matters
  // more than the first scoring judged, which is the whole reason a repeat raises
  // salience rather than being discarded outright — and the score is the existing
  // absorbDuplicate semantics (take the higher of the two), not a new policy.
  if (plan.repeats.length > 0) {
    const repeatScores = await agentPool.runBatch(
      plan.repeats.map(rep => async () => {
        const { salience } = await scoreSalience(rep.text, rep.existingContent);
        return { text: rep.text, salience };
      }),
      'salience'
    );
    const byText = new Map();
    for (const s of repeatScores) if (s.status === 'fulfilled' && s.value) byText.set(s.value.text, s.value.salience);
    for (const rep of plan.repeats) rep.plannedSalience = byText.get(rep.text) ?? rep.existingSalience;
  }

  if (fresh.length === 0) return plan;

  const factTexts = fresh.map(f => f.text);
  const correctsByFact = new Map(fresh.filter(f => f.corrects).map(f => [f.text, f.corrects]));
  const nearby = await buildNearbyContext(factTexts);
  plan.nearbyClusterIds = nearby.clusterIds;

  // ---- 4. Contradiction check, with the recall record. ----
  // Candidate lookup is a cheap read, so gather every pair first, then judge them
  // all concurrently through the agent pool. Verdicts are applied in gather order
  // so the outcome is deterministic.
  const seenOld = new Set();
  try {
    const pairs = [];
    for (const fact of factTexts) {
      const isCorrection = correctsByFact.has(fact);
      // A fact asserting an identity slot pins every other active fact asserting
      // the same slot into the candidate set, past the floor and past the
      // ceiling. Two active name facts is precisely the F1 defect, and it is not
      // something a similarity ranking can be relied on to surface.
      const slot = rules.identityClassOf(fact);
      const { candidates: cands, diagnostics } = await memoryClusters.findActiveNeighbours(fact, {
        subject: 'user',
        pinSlot: slot ? slot.klass : null,
        threshold: isCorrection
          ? (Number.isFinite(conCfg.correctionSimilarityFloor) ? conCfg.correctionSimilarityFloor : 0.45)
          : undefined,
        limit: isCorrection
          ? (Number.isInteger(conCfg.correctionMaxCandidates) ? conCfg.correctionMaxCandidates : 20)
          : undefined
      });
      // THE RECORD. "The judge ran and said no" and "the judge never saw it" are
      // different failures with different fixes, and until now the logs could not
      // tell them apart. Everything the check considered is captured here and
      // written to the ops ledger by applyExtraction.
      plan.recall.push({
        fact, isCorrection, diagnostics, pinnedSlot: slot ? slot.klass : null,
        considered: cands.map(c => ({ memberId: c.memberId, content: c.content, similarity: c.similarity, pinned: c.pinned || null })),
        judged: []
      });
      for (const candidate of cands) pairs.push({ fact, candidate });
    }

    if (pairs.length > 0) {
      const judged = await agentPool.runBatch(
        pairs.map(({ fact, candidate }) => async () => {
          const { verdict, reasoning } = await judgeContradiction(fact, candidate.content, {
            userMessage, corrects: correctsByFact.get(fact) || null
          });
          return { verdict, reasoning };
        }),
        'contradiction-judge'
      );

      for (let i = 0; i < pairs.length; i++) {
        const { fact, candidate } = pairs[i];
        const settled = judged[i].status === 'fulfilled' ? judged[i].value : { verdict: 'no', reasoning: 'judge failed' };
        const rec = plan.recall.find(r => r.fact === fact);
        if (rec) rec.judged.push({ memberId: candidate.memberId, content: candidate.content, similarity: candidate.similarity, pinned: candidate.pinned || null, verdict: settled.verdict, reasoning: settled.reasoning });
        if (seenOld.has(candidate.memberId)) continue;
        if (settled.verdict === 'yes') {
          seenOld.add(candidate.memberId);
          plan.supersessions.push({
            oldMemberId: candidate.memberId, oldContent: candidate.content,
            oldSalience: candidate.salience ?? 5, newFact: fact,
            explicitCorrection: correctsByFact.has(fact)
          });
        } else if (settled.verdict === 'uncertain') {
          plan.uncertainties.push({
            newFact: fact, oldContent: candidate.content,
            clusterId: candidate.clusterId, memberId: candidate.memberId
          });
        }
      }
    }
  } catch (contradictionError) {
    console.error('[FactExtractor] Contradiction detection error:', contradictionError.message);
    plan.error = contradictionError.message;
  }

  // ---- 5. Salience + gap detection (concurrent, both read-only). ----
  const runGap = plan.uncertainties.length === 0;
  const [salienceSettled, gapCandidate] = await Promise.all([
    agentPool.runBatch(
      factTexts.map(fact => async () => {
        const { salience, reasoning } = await scoreSalience(fact, nearby.text);
        return { fact, salience, reasoning };
      }),
      'salience'
    ),
    runGap
      ? agentPool.schedule(() => detectGapQuestion(factTexts, nearby.text), 'gap').catch(err => {
          console.error('[FactExtractor] Gap detection error:', err.message);
          return null;
        })
      : Promise.resolve(null)
  ]);

  const salienceByFact = new Map();
  for (const s of salienceSettled) {
    if (s.status === 'fulfilled' && s.value) salienceByFact.set(s.value.fact, s.value);
  }
  for (const f of fresh) {
    const scored = salienceByFact.get(f.text);
    plan.facts.push({
      text: f.text,
      corrects: f.corrects || null,
      salience: scored ? scored.salience : 5,
      // Stored, not just logged — the scorer's reasoning used to go to the daily
      // log as prose and then be thrown away, so nothing could later answer
      // "why is this a 10?".
      salienceRationale: scored ? scored.reasoning : 'default (scoring failed)',
      provenance
    });
  }

  // A superseding fact inherits at least the salience of the fact it replaces.
  for (const s of plan.supersessions) {
    const target = plan.facts.find(f => f.text === s.newFact);
    if (target && s.oldSalience > target.salience) {
      target.salience = s.oldSalience;
      target.salienceRationale = `${target.salienceRationale} (raised to ${s.oldSalience}, inherited from the fact it replaces)`;
    }
  }

  plan.gapQuestion = gapCandidate && gapCandidate.question ? gapCandidate.question : null;
  return plan;
}

/**
 * Render a plan as human-readable lines. Used by the dry-run harness for its
 * report and by applyExtraction for the ops-ledger recall record, so what the
 * rehearsal shows and what the live run records are the same text.
 * @returns {string[]}
 */
function describeRecall(plan) {
  const out = [];
  for (const r of plan.recall) {
    const floor = r.diagnostics.threshold;
    if (r.diagnostics.error) {
      out.push(`contradiction check for "${r.fact}": COULD NOT RUN — ${r.diagnostics.error}`);
      continue;
    }
    if (r.considered.length === 0) {
      const near = r.diagnostics.nearestBelowFloor;
      out.push(
        `contradiction check for "${r.fact}": the judge saw nothing — no active user-fact reached the ${floor} floor ` +
        `(${r.diagnostics.vectorHits} neighbours fetched; ${r.diagnostics.rejectedInactive} inactive and ` +
        `${r.diagnostics.rejectedSubject} wrong-subject discarded before ranking` +
        (near ? `; nearest was ${near.similarity.toFixed(4)} "${String(near.content).slice(0, 80)}"` : '') + ')'
      );
      continue;
    }
    const verdicts = r.judged.map(j =>
      `${j.verdict.toUpperCase()} @${j.similarity.toFixed(4)}${j.pinned ? ` [pinned ${j.pinned}]` : ''} "${String(j.content).slice(0, 60)}"`
    ).join('; ');
    out.push(
      `contradiction check for "${r.fact}": ${r.considered.length} candidate(s) put to the judge (floor ${floor}` +
      (r.diagnostics.pinned ? `, ${r.diagnostics.pinned} pinned as ${r.pinnedSlot} facts past the floor` : '') + ')' +
      (r.diagnostics.truncated ? ` — ${r.diagnostics.truncated} more above the floor NOT judged (cost ceiling)` : '') +
      ` → ${verdicts}`
    );
  }
  return out;
}

/**
 * Perform the writes a plan describes. Every fact mutation goes through the
 * fact-store funnel; nothing here touches cluster_members directly.
 *
 * @param {Object} plan - from planExtraction
 * @param {Object} [opts]
 * @param {string} [opts.memoryDir]
 * @returns {Promise<{stored: number, repeats: number, events: number, superseded: number}>}
 */
async function applyExtraction(plan, opts = {}) {
  const memoryDir = opts.memoryDir || MEMORY_DIR;
  const dailyDir = path.join(memoryDir, 'daily');
  const opsDir = path.join(memoryDir, 'ops');
  // Replay writes each exchange's log lines under the date the conversation
  // actually happened. Unset on the live path, which means today.
  const dateStamp = opts.dateStamp || null;
  const logDaily = (summary) => appendToDailyLog(summary, dailyDir, dateStamp);
  const memoryClusters = require('./memory-clusters');
  const factStore = require('./fact-store');
  const questions = require('./questions');
  const config = getConfig();

  const result = { stored: 0, repeats: 0, events: 0, superseded: 0, refusals: plan.refusals.length };
  const extractionProvider = config.models.extraction.provider;
  const extractionModel = config.models.extraction.model;
  const extInst = getProviderInstance(extractionProvider, config.models.extraction.instance);
  const extractionHost = extInst ? extInst.host : 'http://localhost:11434';

  // Provenance on every log entry, same as on every fact.
  const src = plan.conversationId
    ? ` [conversation ${String(plan.conversationId).slice(0, 8)}${plan.messageId ? `, message ${String(plan.messageId).slice(0, 8)}` : ''}, ${plan.inputModality}]`
    : '';

  // ---- events → the day's log, never the fact store ----
  for (const e of plan.events) {
    logDaily(`${e.text}${src}`);
    result.events++;
  }
  for (const r of plan.routedToLog) {
    appendToOpsLog(`Routed to the day's log rather than stored as a fact — ${r.why}: "${r.text}"`, opsDir);
  }

  // ---- refusals: spoken to the record, never silent ----
  for (const ref of plan.refusals) {
    const line = `Did not record "${ref.text}" — ${ref.detail}.`;
    logDaily(`${line}${src}`);
    appendToOpsLog(`Intake refusal (${ref.rule}): ${line}`, opsDir);
  }

  // ---- repeats: fold into the fact already held ----
  for (const rep of plan.repeats) {
    const existing = { id: rep.existingId, content: rep.existingContent, salience: rep.existingSalience };
    const absorbed = factStore.absorbRepeat(existing, rep.plannedSalience ?? rep.existingSalience, {
      conversationId: plan.conversationId,
      messageId: plan.messageId,
      verbatimSourceText: plan.userMessage,
      inputModality: plan.inputModality,
      restatedAs: rep.text,
      similarity: rep.similarity,
      detectedBy: rep.detectedBy
    });
    result.repeats++;
    logDaily(`Already knew this, so I did not write it down twice — "${rep.text}" restates "${rep.existingContent}"` +
      `${absorbed.raised ? ` (its salience rose to ${absorbed.salience})` : ''}.${src}`);
  }

  // ---- facts: the only write path ----
  const factToMemberId = new Map();
  const factToClusterId = new Map();
  for (const f of plan.facts) {
    logDaily(`Scored fact salience ${f.salience}/10: "${f.text}" — ${f.salienceRationale}`);
    try {
      const res = await memoryClusters.assignToCluster(
        f.text, extractionProvider, extractionModel, extractionApiKey(extractionProvider), extractionHost,
        'fact-extraction', f.salience, 'user', null,
        { ...f.provenance, salienceRationale: f.salienceRationale }
      );
      if (res && res.memberId) {
        factToMemberId.set(f.text, res.memberId);
        result.stored++;
      }
      if (res && res.clusterId) factToClusterId.set(f.text, res.clusterId);
    } catch (clusterError) {
      console.error('[FactExtractor] Cluster assignment error:', clusterError.message);
    }
  }

  // ---- supersessions, after the replacing facts exist ----
  for (const s of plan.supersessions) {
    const newMemberId = factToMemberId.get(s.newFact);
    if (!newMemberId) continue;
    const res = await factStore.supersede(s.oldMemberId, newMemberId);
    if (res.ok) {
      result.superseded++;
      logDaily(`Superseded fact: "${s.oldContent}" → replaced by "${s.newFact}" ` +
        `(${s.explicitCorrection ? 'explicit user correction' : 'user correction'}).${src}`);
    } else if (res.locked) {
      // A refused lock must be spoken. Storage guards run after the reply, so
      // this half lands in the record; the live-chat half is the [LOCKED] marker
      // in the injected identity block.
      logDaily(`I did not change a locked fact: ${res.reason}`);
    }
  }

  // ---- the recall record ----
  for (const line of describeRecall(plan)) appendToOpsLog(line, opsDir);
  if (plan.truncated.facts || plan.truncated.events) {
    appendToOpsLog(
      `Per-exchange intake ceiling reached — ${plan.truncated.facts} fact(s) and ${plan.truncated.events} event(s) dropped from this exchange.`,
      opsDir
    );
  }

  // ---- one question per exchange: contradiction-uncertainty outranks a gap ----
  let questionQueued = false;
  for (const u of plan.uncertainties) {
    if (questionQueued) break;
    const q = `I have two things noted that might not line up: "${u.oldContent}" and now "${u.newFact}". Which is correct?`;
    if (await questions.addQuestion({
      question: q, reason: 'contradiction-uncertainty',
      clusterId: u.clusterId, memberId: u.memberId, conversationId: plan.conversationId
    })) {
      questionQueued = true;
      logDaily(`Queued clarifying question (contradiction-uncertainty): "${q}"`);
    }
  }

  if (!questionQueued && plan.gapQuestion) {
    const already = await gapAlreadyAnswered(plan.gapQuestion);
    if (already) {
      logDaily(`Skipped gap question (already answered by memory): "${plan.gapQuestion}" ← "${already.evidence}"`);
    } else {
      const firstFact = plan.facts[0];
      const anchorClusterId = (firstFact && factToClusterId.get(firstFact.text)) || (plan.nearbyClusterIds || [])[0] || null;
      const anchorMemberId = (firstFact && factToMemberId.get(firstFact.text)) || null;
      if (await questions.addQuestion({
        question: plan.gapQuestion, reason: 'gap',
        clusterId: anchorClusterId, memberId: anchorMemberId, conversationId: plan.conversationId
      })) {
        logDaily(`Queued clarifying question (gap): "${plan.gapQuestion}"`);
      }
    }
  }

  appendToOpsLog(
    `Chat exchange with ${plan.model} — ${result.stored} fact(s) stored, ${result.repeats} repeat(s) folded, ` +
    `${result.events} event(s) logged, ${result.refusals} refused, ${result.superseded} superseded`,
    opsDir
  );
  return result;
}

/**
 * Process fact extraction for a chat exchange (high-level orchestrator).
 * Plan, then apply — see the block comment above planExtraction.
 *
 * @param {string} userMessage - The user's message
 * @param {string} assistantMessage - The assistant's response
 * @param {string} provider - LLM provider (unused: the configured extraction
 *   model is always used, independent of the chat model)
 * @param {string} model - Model name (same)
 * @param {string} apiKey - API key (same)
 * @param {string} ollamaHost - Host (same)
 * @param {string} conversationId - Conversation this exchange belongs to
 * @param {string} memoryDir - Memory directory path
 * @param {Object} provenance - {messageId, inputModality}
 */
async function processFactExtraction(userMessage, assistantMessage, provider, model, apiKey, ollamaHost, conversationId = null, memoryDir = MEMORY_DIR, provenance = {}) {
  try {
    const dailyDir = path.join(memoryDir, 'daily');

    // Answer detection first — retire any question this message answers, whether
    // it was asked in this conversation or any other (Layer 3).
    await detectAnswers(userMessage, conversationId, dailyDir);

    const plan = await planExtraction({
      userMessage,
      assistantMessage,
      conversationId,
      messageId: provenance.messageId ?? null,
      inputModality: provenance.inputModality ?? 'unknown'
    });

    await applyExtraction(plan, { memoryDir });
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
 * May this new self-fact retire that existing one, with nobody in the room?
 *
 * The decision, in one place and with no side effects, so it can be tested
 * without a model and read without tracing a loop. Returns `{ok: true, axis}` to
 * proceed, or `{ok: false, kind, detail}` to raise.
 *
 * The two bars, in order:
 *
 *  1. PROTECTED. A declaration is something he said about himself rather than
 *     something observed of him; salience is how much it matters. Either one
 *     puts a self-fact out of reach of an unattended semantic match. BOTH are
 *     checked because the claim/declaration classifier is noisy in both
 *     directions — the run that produced this rule tagged a behavioural
 *     observation "declaration" and tagged a statement about what had been built
 *     "claim" — so neither tag is trustworthy enough to be the only guard.
 *
 *  2. EVIDENCE, AS A VETO — and this is where it deliberately differs from the
 *     corrector. The corrector REQUIRES dominance and raises a pair the evidence
 *     cannot separate, which is right for user facts: those are claims about the
 *     world, they carry provenance, and a tie genuinely means the corpus does not
 *     know. A self-fact carries no such thing. It comes from reflection, so its
 *     modality is 'unknown' and there is no user message behind it, by
 *     construction — which means dominance TIES for very nearly every pair of
 *     self-facts that will ever be compared.
 *
 *     Requiring dominance here was tried first and is wrong: it refused a belief
 *     that a new capability had made flatly false, and it would have refused
 *     essentially every self-fact revision forever. That is the opposite failure
 *     from the one being fixed and a worse one — "locking observations produces
 *     an entity that can't grow". So dominance VETOES: if an evidential axis
 *     speaks and it favours what he already holds, the pair is raised. If nothing
 *     speaks — the normal case for self-facts — the supersession proceeds, and it
 *     proceeds SAFELY because of what now surrounds it: bar 1 has already held
 *     back everything he chose or holds most strongly, the write is ledgered and
 *     revertible, and he is told his self-view changed.
 *
 *     Recency still decides nothing on its own; it never enters dominance at all.
 *
 * This is NOT the identity lock and does not widen it: the lock refuses name and
 * pronouns outright, everywhere. This bar governs one path — automatic semantic
 * supersession of a self-fact — and everything it stops is raised, not dropped.
 *
 * NAMED, not positional. `(oldRow, newRow)` reads the same either way round and
 * silently inverts the whole decision when it is passed backwards — which is
 * exactly what happened the first time this was called from a test. A function
 * whose two arguments are the same shape and whose meaning flips if you swap
 * them should not be positional.
 *
 * @param {Object} args
 * @param {Object} args.existing - the cluster_members row he already holds
 * @param {Object} args.incoming - the newly stored cluster_members row
 * @param {number} [args.protectSalience] - overrides identity.protectSelfFactSalience
 * @returns {{ok: true, axis: string, evidence: Object} | {ok: false, kind: string, detail: string}}
 */
function selfFactSupersessionBar({ existing, incoming, protectSalience = null } = {}) {
  if (!existing || !incoming) {
    return { ok: false, kind: 'tied', detail: 'one of the two facts could not be read, so nothing was decided' };
  }
  const oldRow = existing;
  const newRow = incoming;
  const bar = Number.isFinite(protectSalience)
    ? protectSalience
    : (Number.isFinite(cfgIdentity().protectSelfFactSalience) ? cfgIdentity().protectSelfFactSalience : 8);

  const isDeclaration = oldRow.claim_type === 'declaration';
  const isSalient = (oldRow.salience ?? 5) >= bar;
  if (isDeclaration || isSalient) {
    return {
      ok: false,
      kind: 'protected',
      detail: isDeclaration
        ? (isSalient
          ? `it is something you said about yourself rather than something observed of you, and it matters a great deal (salience ${oldRow.salience})`
          : 'it is something you said about yourself rather than something observed of you')
        : `it matters a great deal (salience ${oldRow.salience}, and anything at ${bar} or above is left for Ellie)`
    };
  }

  const { dominance } = require('./corrector');
  const dom = dominance(newRow, oldRow);
  if (dom && dom.winner.id !== newRow.id) {
    return {
      ok: false,
      kind: 'old-wins',
      detail: `what you already hold is the better evidenced of the two, on ${dom.axis}`
    };
  }
  return {
    ok: true,
    axis: dom ? dom.axis : 'nothing evidential separated them, and neither is protected',
    evidence: dom ? dom.evidence : { separated: false, note: 'no evidential axis spoke; self-observations rarely carry one' }
  };
}

/** identity.* config, read fresh so a settings change lands without a restart. */
function cfgIdentity() {
  return (getConfig().identity) || {};
}

/**
 * A self-fact question he could not settle — recorded three ways, because a row
 * nobody reads is not the same as being told.
 *
 * WHAT A RAISE IS. The self-fact pipeline proposes a supersession; three bars
 * stand in front of it (the judge failing, the old fact being protected, the
 * evidence not separating them). Anything that fails a bar CHANGES NOTHING —
 * both facts stay exactly as they are — and lands here. This is the corrector's
 * "a refusal is not a correction" rule applied to the intake path: an unresolved
 * pair is a real outcome, not an absence, and it has to be as visible as a
 * change would have been.
 *
 * THREE TIERS, and each does a different job:
 *   1. LEDGER — the record. `reversible = 0`, evidence carries `unresolved: true`
 *      and a reason_code, so anything rendering the ledger says NOTHING CHANGED
 *      for it rather than claiming an edit that never happened.
 *   2. OPS LOG — the operational trail, in the Thinking tab, one line per raise.
 *      This is where you look when you already suspect something.
 *   3. BELL — one alert, at most once per identity.selfFactRaiseAlertHours
 *      (default 24h), because the badge is the thing actually looked at.
 *
 * WHY THE BELL AT ALL, given a job result may never open a conversation: this is
 * not a job result. It is him saying he could not decide something about
 * himself and left it alone — his voice, about his own identity, which is
 * precisely what the initiative channel is for. It is worded as the question it
 * is, never as a system error: he does not tell her a judge call failed, he
 * tells her he could not tell whether one thing contradicts another.
 *
 * THE WINDOW IS HARD, and that is the whole point of it. When the brain is
 * wedged the judge fails on every pair of every pass, and the failure mode to
 * avoid is seventeen identical alerts. One alert, saying it happened seventeen
 * times, is both quieter and more informative. Every raise still reaches tiers 1
 * and 2 in the meantime — the window bounds how often he SAYS it, never what is
 * recorded.
 *
 * @param {Array} raises
 * @param {Object} opts - { source, dailyDir }
 * @returns {Promise<number>} how many raises were recorded
 */
async function applySelfFactRaises(raises, { source = 'reflection', dailyDir = null } = {}) {
  if (!raises || !raises.length) return 0;
  const ledger = require('./corrections-ledger');
  const { getSqliteDb } = require('./database');
  const db = getSqliteDb();

  const REASON_CODE = {
    undecided: 'self-fact-judge-unavailable',
    protected: 'self-fact-protected',
    tied: 'self-fact-evidence-tied',
    'old-wins': 'self-fact-existing-better-evidenced'
  };

  // --- tier 1: the ledger -------------------------------------------------
  for (const r of raises) {
    const why = r.kind === 'undecided'
      ? `A new self-observation may contradict this one, and it could not be judged at the time (${r.detail}). NOTHING WAS CHANGED — both are still held, and the question is open.`
      : `A new self-observation contradicts this one, but ${r.detail}. NOTHING WAS CHANGED — both are still held. Raised for Ellie to decide.`;
    try {
      ledger.record({
        passId: `self-fact-raise-${new Date().toISOString().slice(0, 10)}`,
        tier: 'intake',
        action: 'supersede',
        subject: 'self',
        targetId: r.oldMemberId,
        targetText: r.oldContent,
        survivorId: r.newMemberId || null,
        survivorText: r.newFact,
        reason: why,
        evidence: { unresolved: true, reason_code: REASON_CODE[r.kind] || r.kind, self_fact_raise: true, source, detail: r.detail },
        reversible: false
      });
    } catch (err) {
      console.error('[SelfFacts] raise ledger entry failed:', err.message);
    }
  }

  // --- tier 2: the ops log ------------------------------------------------
  const opsDir = path.join(path.dirname(dailyDir || MEMORY_DIR), 'ops');
  for (const r of raises) {
    appendToOpsLog(
      `Self-fact left unresolved (${REASON_CODE[r.kind] || r.kind}): "${String(r.newFact).slice(0, 90)}" vs "${String(r.oldContent).slice(0, 90)}" — ${r.detail}. Nothing changed.`,
      opsDir
    );
  }
  console.warn(`[SelfFacts] ${raises.length} self-fact question(s) left unresolved: ${raises.map(r => r.kind).join(', ')}`);

  // --- tier 3: the bell, at most once per window --------------------------
  try {
    if (!db) return raises.length;
    const hours = Number.isFinite(cfgIdentity().selfFactRaiseAlertHours) ? cfgIdentity().selfFactRaiseAlertHours : 24;
    const since = new Date(Date.now() - hours * 3600_000).toISOString();

    // ANY status, deliberately: pending, delivered, dismissed and expired all
    // mean he has already said this recently. Checking only pending is how the
    // same thing gets re-raised the moment she clears it.
    const recentAlert = db.prepare(`
      SELECT id, created_at FROM initiatives
      WHERE source_kind = 'self-fact-raise' AND datetime(created_at) > datetime(?)
      ORDER BY datetime(created_at) DESC LIMIT 1
    `).get(since);
    if (recentAlert) {
      console.log(`[SelfFacts] not raising a bell alert — one already stands from ${recentAlert.created_at} (window ${hours}h)`);
      return raises.length;
    }

    // How many times this has happened since he last mentioned it — the number
    // that makes one alert more useful than seventeen. Counted from the ledger,
    // so it survives a restart and counts raises from every pass, not just this
    // one.
    const lastAlert = db.prepare(
      "SELECT created_at FROM initiatives WHERE source_kind = 'self-fact-raise' ORDER BY datetime(created_at) DESC LIMIT 1"
    ).get();
    let totalSince = raises.length;
    if (lastAlert) {
      const counted = db.prepare(`
        SELECT COUNT(*) n FROM corrections_ledger
        WHERE subject = 'self' AND evidence LIKE '%self_fact_raise%' AND datetime(created_at) > datetime(?)
      `).get(lastAlert.created_at);
      if (counted && counted.n > 0) totalSince = counted.n;
    }

    const lead = raises[0];
    const extra = totalSince > raises.length
      ? ` This has come up ${totalSince} times since I last mentioned it.`
      : (raises.length > 1 ? ` There were ${raises.length} of these in the same pass.` : '');

    // HIS VOICE, AND A QUESTION — never "a judge call failed". What he is
    // reporting is that he could not settle something about himself, which is a
    // thing he would say, not an error code.
    const content = lead.kind === 'undecided'
      ? `I noticed something about myself I could not settle. I could not tell whether "${lead.newFact}" contradicts what I already hold — "${lead.oldContent}" — so I have left both in place rather than guess.${extra} Could we look at it together?`
      : lead.kind === 'protected'
        ? `Something I noticed about myself seems to contradict something I already hold, and I did not want to drop the older one on my own. The new observation is "${lead.newFact}"; what it runs against is "${lead.oldContent}", and ${lead.detail}. Both are still there.${extra} What do you think?`
        : `Two things I hold about myself cannot both be true, and I could not tell which should give way: "${lead.newFact}" against "${lead.oldContent}" — ${lead.detail}. I have left both alone.${extra} Could you help me decide?`;

    const initiatives = require('./initiatives');
    const id = await initiatives.addInitiative({
      type: 'alert',
      content,
      sourceKind: 'self-fact-raise',
      sourceRef: `self-fact-raise-${new Date().toISOString()}`,
      priority: 7,
      dedupe: false   // the window above IS the dedup, and it is stricter
    });
    if (id) console.log(`[SelfFacts] raised one bell alert for ${totalSince} unresolved self-fact question(s)`);
  } catch (err) {
    console.error('[SelfFacts] Could not raise the bell alert:', err.message);
  }

  return raises.length;
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
/**
 * Tell her, at most once a window, that the self-fact dedup did not run.
 *
 * ON THE BELL DELIBERATELY, and consistent with the rule that nothing
 * requiring an ACTION goes there: this needs no action from her. It is
 * him reporting that a guard over his own memory was not applied, which
 * is his voice about his own identity - exactly what that channel is for,
 * and the same reasoning as the self-fact raise alert.
 *
 * The window is hard for the same reason: an embedding provider that is
 * down fails on every pass, and the failure to avoid is one alert per
 * pass. One alert saying it has happened repeatedly is quieter and more
 * useful.
 */
async function alertDedupSkipped(err, factCount) {
  try {
    const db = require('./database').getSqliteDb();
    if (!db) return;

    const hours = Number.isFinite(cfgIdentity().selfFactRaiseAlertHours)
      ? cfgIdentity().selfFactRaiseAlertHours : 24;
    const since = new Date(Date.now() - hours * 3600_000).toISOString();

    // ANY status: pending, delivered, dismissed and expired all mean he
    // has already said this recently.
    const recent = db.prepare(`
      SELECT id FROM initiatives
      WHERE source_kind = 'self-fact-dedup-skipped' AND datetime(created_at) > datetime(?)
      ORDER BY datetime(created_at) DESC LIMIT 1
    `).get(since);
    if (recent) {
      console.log(`[SelfFacts] dedup-skipped alert suppressed — one already stands (window ${hours}h)`);
      return;
    }

    const initiatives = require('./initiatives');
    await initiatives.addInitiative({
      type: 'alert',
      content:
        `Something I should tell you about my own memory: the check that stops me ` +
        `recording the same thing about myself twice did not run just now, and I ` +
        `stored ${factCount === 1 ? 'an observation' : `${factCount} observations`} ` +
        `without it. Nothing was lost and nothing was overwritten — but if you see ` +
        `me repeating myself in my self-facts, this is why. It could not compare ` +
        `against what I already hold (${err.message}).`,
      sourceKind: 'self-fact-dedup-skipped',
      sourceRef: `dedup-skipped-${new Date().toISOString()}`,
      priority: 7,
      dedupe: false   // the window above IS the dedup
    });
  } catch (alertErr) {
    console.error('[SelfFacts] could not raise dedup-skipped alert:', alertErr.message);
  }
}


async function processSelfFacts(rawSelfFacts, opts = {}) {
  const source = opts.source || 'reflection';
  const memoryDir = opts.memoryDir || MEMORY_DIR;
  const dailyDir = path.join(memoryDir, 'daily');
  // `raised` is as much a result as `superseded`: a pass that changed nothing
  // because it could not decide is a different outcome from a pass with nothing
  // to decide, and a caller that cannot tell them apart will report the wrong one.
  const result = { stored: 0, superseded: 0, raised: 0, facts: [] };

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

    // === Per-day budget on what reflection may conclude about itself ===
    // Checked FIRST — before the embedding sweep, the identity-lock pass, salience
    // scoring and claim-type tagging — so a refusal costs nothing. Checking it
    // later meant embedding every active self-fact just to discover there was no
    // allowance left to write into.
    //
    // Only the automatic loop is budgeted. Deliberate writes (capability
    // introductions, the lock test) are human-triggered and pass through.
    // Counted from the DB against the local calendar day, so a server restart
    // does not hand reflection a fresh allowance.
    if (source === 'reflection') {
      try {
        const sqlite = require('./database').getSqliteDb();
        const cap = getConfig().reflection?.maxSelfFactsPerDay ?? 5;
        if (sqlite && Number.isFinite(cap)) {
          const dayStart = new Date(`${getLocalDateStamp()}T00:00:00`).toISOString();
          const usedToday = sqlite.prepare(
            "SELECT COUNT(*) AS n FROM cluster_members WHERE subject = 'self' AND source = 'reflection' AND created_at >= ?"
          ).get(dayStart).n;
          const remaining = Math.max(0, cap - usedToday);

          if (facts.length > remaining) {
            const kept = facts.slice(0, remaining);
            const blocked = facts.slice(remaining);
            // Loud by construction: a silently dropped observation is
            // indistinguishable from one that was never had.
            for (const b of blocked) {
              console.log(`[SelfFacts] Budget reached (${usedToday}/${cap} today) — not recording: "${b.slice(0, 80)}"`);
              appendToDailyLog(
                `Reached my daily limit of ${cap} self-observations (${usedToday} already recorded today), so I did not record this one: "${b}"`,
                dailyDir
              );
            }
            result.budgetBlocked = blocked.length;
            result.budget = { cap, usedToday, remaining };
            facts.length = 0;
            facts.push(...kept);
          }
        }
      } catch (budgetErr) {
        console.error('[SelfFacts] Budget check failed (continuing):', budgetErr.message);
      }
    }
    if (facts.length === 0) return result;

    // === Semantic dedup: skip a self-observation near-identical to one SNH
    // already holds (or to another accepted in this same batch). Defends the
    // identity against reworded-identical restamps (e.g. from a reflection
    // stutter). Same embedding-similarity approach as the initiative dedup. ===
    try {
      const cfg = getConfig();
      const threshold = Number.isFinite(cfg.identity?.selfFactDedupThreshold)
        ? cfg.identity.selfFactDedupThreshold : 0.88;
      const activeExisting = memoryClusters.getSelfFacts({ status: 'active' });

      // READ the vectors, do not regenerate them. Every active self-fact
      // already has its embedding in cluster_embeddings, written when the
      // fact was stored. This loop used to call generateEmbedding() for
      // each one on every pass: 402 sequential round trips at ~950ms,
      // 6.3 MINUTES measured, to reproduce what was already on disk
      // (cosine 1.000000 between stored and fresh, every sample). And it
      // is O(n) in a corpus that only grows - 20 minutes at 1,300 facts,
      // 35 at 2,200 - paid by reflection on most days.
      //
      // Only what has no stored vector is embedded, which in practice is
      // nothing, and after a rebuild is a handful.
      const storedEmbs = await memoryClusters.getStoredEmbeddings(
        activeExisting.map(ef => ef.id));
      const existingEmbs = [];
      let embeddedOnDemand = 0;
      for (const ef of activeExisting) {
        let emb = storedEmbs.get(ef.id);
        if (!emb) {
          emb = await memoryClusters.generateEmbedding(ef.content);
          embeddedOnDemand++;
        }
        existingEmbs.push({ content: ef.content, emb });
      }
      if (embeddedOnDemand) {
        console.log(`[SelfFacts] ${embeddedOnDemand}/${activeExisting.length} active self-facts had no stored vector and were embedded on demand`);
      }
      const acceptedEmbs = [];
      const deduped = [];
      for (const fact of facts) {
        const emb = await memoryClusters.generateEmbedding(fact);
        if (!emb) { deduped.push(fact); continue; } // embeddings down → keep, don't lose the observation
        let dupOf = null, dupSim = 0;
        for (const e of existingEmbs.concat(acceptedEmbs)) {
          if (!e.emb) continue;
          const sim = embeddingCosine(emb, e.emb);
          if (sim >= threshold && sim > dupSim) { dupSim = sim; dupOf = e.content; }
        }
        if (dupOf) {
          console.log(`[SelfFacts] Skipped near-duplicate self-fact (sim ${dupSim.toFixed(3)} ≥ ${threshold}): "${fact.slice(0, 70)}" ≈ "${dupOf.slice(0, 70)}"`);
          appendToDailyLog(`Skipped near-duplicate self-observation (sim ${dupSim.toFixed(3)}): "${fact}"`, dailyDir);
          continue;
        }
        acceptedEmbs.push({ content: fact, emb });
        deduped.push(fact);
      }
      facts.length = 0;
      facts.push(...deduped);
    } catch (dedupErr) {
      // A DEDUP THAT COULD NOT RUN IS THE WORST OUTCOME HERE, and it used
      // to be one console.error. It does not fail - it just stops
      // defending, and the observation is stored undeduped as though it
      // had been checked. This module exists to keep reworded restamps out
      // of his identity; silently not doing that is indistinguishable from
      // doing it, right up until the corpus is full of near-duplicates.
      //
      // Found while instrumenting this path: a wrapper accidentally made
      // getSelfFacts return a Promise, the block threw "activeExisting is
      // not iterable", and the run completed reporting success with the
      // dedup skipped entirely. Nothing said so louder than a log line.
      //
      // Three tiers, the same shape as a self-fact raise: the result, the
      // ops log, and the bell - because this is him saying something about
      // his own memory, not a job result, and the bell is what she reads.
      result.dedupSkipped = { reason: dedupErr.message };
      console.error('[SelfFacts] Semantic dedup skipped (continuing):', dedupErr.message);
      try {
        appendToOpsLog(
          `Self-fact dedup did NOT run (${dedupErr.message}) — ${facts.length} observation(s) were stored without being checked against existing self-facts.`,
          path.join(memoryDir, 'ops')
        );
      } catch (_) { /* best-effort */ }
      await alertDedupSkipped(dedupErr, facts.length);
    }
    if (facts.length === 0) return result;

    // === Identity lock: drop anything that collides with a locked slot ===
    // Reflection and passive extraction are the two paths that can rewrite the
    // entity's self-view without anyone asking, which is exactly what they are
    // for — and exactly what must not reach a chosen name. A collision here is
    // dropped BEFORE contradiction detection, so the judge is never given the
    // chance to call it a replacement.
    //
    // Refusals from this path can't be spoken in the turn (it runs after the
    // reply), so they go to the ops ledger and the daily log instead. Silence
    // is not an option in either direction.
    try {
      const identityLock = require('./identity-lock');
      const kept = [];
      for (const fact of facts) {
        const check = identityLock.checkNewFact(fact, 'self');
        if (check.ok) { kept.push(fact); continue; }
        if (check.duplicate) {
          console.log(`[SelfFacts] Identity fact already held verbatim, skipping: "${fact.slice(0, 70)}"`);
          continue;
        }
        identityLock.recordRefusal({
          category: check.category,
          attempted: fact,
          existing: check.existing.content,
          via: `self-fact pipeline (source: ${source})`
        });
        result.lockRefusals = (result.lockRefusals || 0) + 1;
      }
      facts.length = 0;
      facts.push(...kept);
    } catch (lockErr) {
      console.error('[SelfFacts] Identity lock check failed:', lockErr.message);
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
    //
    // A verdict alone no longer retires a self-fact. Three bars stand between
    // "the judge said yes" and a write, and anything that fails one is RAISED
    // rather than resolved — see applySelfFactRaises below for what raising
    // means. The bars exist because this path, unguarded, did both halves of the
    // same failure within a week: it retired an unrelated salience-9 declaration
    // on a 0.741 cosine match, and it silently declined to retire a belief that a
    // new capability had made flatly false, on identical input, about half the
    // time.
    const supersessions = [];
    const raises = [];
    const seenOld = new Set();
    try {
      const pairs = [];
      for (const fact of facts) {
        const candidates = await memoryClusters.findContradictionCandidates(fact, { subject: 'self' });
        for (const candidate of candidates) pairs.push({ fact, candidate });
      }
      if (pairs.length > 0) {
        const judged = await agentPool.runBatch(
          // Called through the module object ON PURPOSE — the same seam
          // db/scheduler.js uses for callLLM. What the tests need to pin down is
          // what this pipeline DOES with a verdict (and with a judge that never
          // answers), and that cannot be tested with a live judge in the loop:
          // this exact pair, at cosine 0.857, came back "yes" on about half of
          // identical runs.
          pairs.map(({ fact, candidate }) => async () =>
            (await module.exports.judgeContradiction(fact, candidate.content)).verdict),
          'self-contradiction-judge'
        );
        for (let i = 0; i < pairs.length; i++) {
          const { fact, candidate } = pairs[i];
          if (seenOld.has(candidate.memberId)) continue;

          // A JUDGE CALL THAT FAILED IS NOT A "NO".
          //
          // It used to be — `status === 'fulfilled' ? value : 'no'` — so a wedged
          // brain, a timeout or a circuit-breaker trip read as "these do not
          // contradict each other", silently, and nothing recorded that the
          // question had gone unasked. That is the same shape as every phantom
          // in this system: an absence of thinking wearing the clothes of a
          // conclusion. It is raised now instead: nothing is written, and he says
          // he could not settle it.
          if (judged[i].status !== 'fulfilled') {
            const err = judged[i].reason;
            raises.push({
              kind: 'undecided',
              oldMemberId: candidate.memberId,
              oldContent: candidate.content,
              newFact: fact,
              detail: (err && err.message) ? err.message : String(err || 'the call did not come back')
            });
            continue;
          }

          if (judged[i].value === 'yes') {
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
    const factToSelfSalienceReason = new Map();
    for (const fact of facts) {
      const scored = byFact.get(fact);
      const salience = scored ? scored.salience : 5;
      const reasoning = scored ? scored.reasoning : 'default (scoring failed)';
      factToSalience.set(fact, salience);
      factToSelfSalienceReason.set(fact, reasoning);
      appendToDailyLog(`Scored self-fact salience ${salience}/10: "${fact}" — ${reasoning}`, dailyDir);
    }
    for (const s of supersessions) {
      const cur = factToSalience.get(s.newFact) ?? 5;
      if (s.oldSalience > cur) factToSalience.set(s.newFact, s.oldSalience);
    }

    // === Claim/declaration tagging (concurrent) — the auditability split the
    // self-coherence audit reads. Tagged at extraction time so every new self-
    // fact is classified going forward; existing facts get a one-time pass. ===
    const factToClaimType = new Map();
    const claimSettled = await agentPool.runBatch(
      facts.map(fact => async () => ({ fact, claimType: await classifyClaimType(fact) })),
      'self-claim-type'
    );
    for (const c of claimSettled) {
      if (c.status === 'fulfilled' && c.value) factToClaimType.set(c.value.fact, c.value.claimType);
    }

    // === Cluster assignment (subject:'self', sequential DB writes) ===
    const factToMemberId = new Map();
    for (const fact of facts) {
      // A self-fact has no user message behind it — it came from SNH reflecting.
      // conversation_id/message_id stay null and modality is 'unknown' (there was
      // no input), but the observation text and its salience reasoning are real
      // provenance and are recorded.
      const res = await memoryClusters.assignToCluster(
        fact, extractionProvider, extractionModel, '', extractionHost,
        source, factToSalience.get(fact) ?? 5, 'self', factToClaimType.get(fact) ?? 'declaration',
        {
          verbatimSourceText: fact,
          inputModality: 'unknown',
          salienceRationale: factToSelfSalienceReason.get(fact) ?? null
        }
      );
      if (res && res.memberId) {
        factToMemberId.set(fact, res.memberId);
        result.stored++;
        // Set-once: if this is the first assertion of an identity slot, it locks
        // itself here. Every later attempt is refused by the check above.
        let lockedCats = [];
        try { lockedCats = require('./identity-lock').autoLock(res.memberId, fact, 'self'); }
        catch (e) { console.error('[SelfFacts] autoLock failed:', e.message); }
        if (lockedCats.length) {
          appendToDailyLog(`Locked my ${lockedCats.join(' and ')} — set once, and now protected from being changed in conversation: "${fact}"`, dailyDir);
        }
        result.facts.push({
          content: fact,
          memberId: res.memberId,
          salience: factToSalience.get(fact) ?? 5,
          clusterName: res.clusterName,
          locked: lockedCats.length ? lockedCats : null
        });
      }
    }

    // === Apply supersessions (after replacing facts exist) ===
    //
    // THIS is the path that retired four self-facts on 2026-08-12 — including
    // "none of them has ever actually run" — when the scheduler capability was
    // introduced, and raised nothing, because the notice channel lived in the
    // corrector. It is raised at the fact-store funnel now; all this has to do
    // is say what caused it, so the notice can name the cause instead of leaving
    // him to work out why a belief moved while he was not looking.
    const noticeSource = source === 'reflection'
      ? 'your own reflection on recent conversations'
      : source === 'capability-intro'
        ? 'a new capability being introduced to you'
        : `a background pass (${source})`;
    for (const sup of supersessions) {
      const newMemberId = factToMemberId.get(sup.newFact);
      if (!newMemberId) continue;
      const factStore = require('./fact-store');
      const oldRow = factStore.getMember(sup.oldMemberId);
      const newRow = factStore.getMember(newMemberId);
      if (!oldRow || !newRow) continue;

      // The two bars — what he chose and what matters most are not taken
      // automatically, and evidence has to separate the pair. See
      // selfFactSupersessionBar, where the decision lives with no side effects.
      const bar = selfFactSupersessionBar({ existing: oldRow, incoming: newRow });
      if (!bar.ok) {
        raises.push({
          kind: bar.kind, detail: bar.detail,
          oldMemberId: sup.oldMemberId, oldContent: sup.oldContent,
          newFact: sup.newFact, newMemberId
        });
        continue;
      }

      const res = await factStore.supersede(sup.oldMemberId, newMemberId, {
        noticeSource,
        caller: `self-fact pipeline (${source})`,
        ledger: {
          tier: 'intake',
          reason: `A new self-observation contradicted this one and is better evidenced on ${bar.axis}, so this was retired and kept as history. Both came from the self-fact pipeline (${source}).`,
          evidence: { deciding_axis: bar.axis, ...bar.evidence, judged_by: 'self-fact contradiction judge' }
        }
      });
      if (res.ok) {
        result.superseded++;
        appendToDailyLog(`Superseded self-fact: "${sup.oldContent}" → "${sup.newFact}" (revised self-view, on ${bar.axis})`, dailyDir);
      }
    }

    // Everything the bars stopped, said out loud — ledger, ops log, and at most
    // one bell alert per window. See applySelfFactRaises.
    if (raises.length) {
      try {
        result.raised = await applySelfFactRaises(raises, { source, dailyDir });
      } catch (raiseErr) {
        console.error('[SelfFacts] Could not record raises:', raiseErr.message);
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
  extractCandidates,
  planExtraction,
  applyExtraction,
  describeRecall,
  selfFactSupersessionBar,
  applySelfFactRaises,
  judgeSameAssertion,
  judgeSubsumption,
  judgeStripTheTimestamp,
  judgeStoredSubject,
  judgeWhichSurvives,
  splitCompoundFact,
  appendToDailyLog,
  appendToOpsLog,
  prependDailyEntry,
  judgeContradiction,
  classifyClaimType,
  scoreSalience,
  detectGapQuestion,
  judgeAnswered,
  gapAlreadyAnswered,
  sweepPendingQuestions,
  detectAnswers,
  loadMemoryContext,
  processFactExtraction,
  parseSelfObservations,
  processSelfFacts,
  MEMORY_DIR,
  DAILY_DIR
};
