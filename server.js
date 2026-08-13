/**
 * Squatch Neuro Hub — backend server
 * Neural-linked AI assistant with associative cluster memory,
 * multi-provider support, and MCP tool calling.
 * Part of the Coastal Squatch AI ecosystem.
 */

require('dotenv').config();

const express = require('express');
const path = require('path');
const rateLimit = require('express-rate-limit');

// Database modules
const db = require('./db/database');

// Fact extraction and memory
const factExtractor = require('./db/fact-extractor');

// Memory flush and clustering
const memoryFlush = require('./db/memory-flush');
const memoryClusters = require('./db/memory-clusters');
const memoryManager = require('./db/memory-manager');
const agentPool = require('./db/agent-pool');
const identity = require('./db/identity');
const capabilityManifest = require('./db/capability-manifest');
const initiatives = require('./db/initiatives');
const questionQueue = require('./db/questions');
const { getCurrentDateTimeString, formatFactTimestamp } = require('./db/datetime');
const injectionBudget = require('./db/injection-budget');
const { classifyToolNeed, isTimeSensitive, classifySchedulingIntent, classifyMemoryWriteIntent, classifyMemoryReadIntent, classifyMemoryCorrectionIntent, classifyJobsIntent } = require('./db/tool-routing');
const { createToolArtifactFilter, stripToolArtifacts, CANNOT_CHECK } = require('./db/tool-artifacts');

// MCP tool calling
const MCPClient = require('./mcp/mcp-client');

// Configuration
const { getConfig, updateConfig, getProviderInstance, getVoiceProvider, getSearxngConfig } = require('./db/config');

// Routes
const conversationsRouter = require('./routes/conversations');
const memoryRouter = require('./routes/memory');

const app = express();

// Trust proxy for rate limiting behind a reverse proxy.
//
// 'loopback' rather than 1: SNH is fronted by `tailscale serve`, which proxies
// from 127.0.0.1, so X-Forwarded-For should be honored ONLY for connections
// arriving on loopback. With the old `1`, any client that could reach :3000
// directly (the listener is on 0.0.0.0) could send its own X-Forwarded-For and
// be keyed as whatever it liked — a free reset of its own rate-limit bucket.
app.set('trust proxy', 'loopback');

// Configuration from environment
const PORT = process.env.PORT || 3000;
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const CLAUDE_API_KEY = process.env.CLAUDE_API_KEY || '';
const GROK_API_KEY = process.env.GROK_API_KEY || '';
const SEARXNG_HOST = process.env.SEARXNG_HOST || 'http://localhost:8888';

// --- Search provenance: mark search-derived content distinctly and retain the
// source links (title + url) through the tool loop into the conversation record.
// The 7/23 test showed the entity blending a real search result with an invented
// specific and, when asked to cite, confabulating attributions. Marking results
// as [S#] SOURCES (and persisting them) makes searched facts distinguishable from
// generated ones and makes citing read retained links, never reconstruct them.

/** Add a source to the sink (dedup by url), returning its stable [S#] number. */
function addSource(sink, { title, url, snippet }) {
  if (url) {
    const existing = sink.find(s => s.url === url);
    if (existing) return existing.n;
  }
  const n = sink.length + 1;
  sink.push({ n, title: title || '', url: url || '', snippet: (snippet || '').slice(0, 300) });
  return n;
}

/**
 * Turn a tool result into the string the model sees, marked as SOURCES, while
 * collecting the source links into `sink`. Errors/other results pass through as
 * before so the model still sees limit/error messages.
 */
function formatToolResult(fnName, result, sink) {
  if (fnName === 'web_search' && result && Array.isArray(result.results) && result.results.length) {
    const lines = ['WEB SEARCH RESULTS — these are your SOURCES. For any specific fact you take from them, cite the [S#] link; include the links you used in your answer. Anything not found here is from memory, not searched — hedge it.'];
    for (const r of result.results) {
      const n = addSource(sink, { title: r.title, url: r.url, snippet: r.snippet });
      lines.push(`[S${n}] ${r.title || '(untitled)'} — ${r.url || '(no url)'}\n${r.snippet || ''}`);
    }
    return lines.join('\n');
  }
  if (fnName === 'web_fetch' && result && result.url) {
    const n = addSource(sink, { title: result.title || result.url, url: result.url });
    return `FETCHED PAGE [S${n}] — ${result.url}\n${result.content || ''}`;
  }
  return typeof result === 'string' ? result : JSON.stringify(result);
}

// Additional allowed Ollama hosts (comma-separated in .env)
const ALLOWED_OLLAMA_HOSTS = process.env.ALLOWED_OLLAMA_HOSTS
  ? process.env.ALLOWED_OLLAMA_HOSTS.split(',').map(h => h.trim())
  : [];

// ============ Security Configuration ============

// SECURITY: Rate limiting to prevent API abuse.
//
// Caps and exemptions live in config.rateLimit (see db/config.js for why the
// old literals were the direct cause of the 2026-07-27 whole-app 429).

/** Normalize an Express req.ip to a bare IPv4/IPv6 string. */
function normalizeIp(ip) {
  if (!ip) return '';
  // Express reports IPv4 over a dual-stack socket as ::ffff:127.0.0.1
  return ip.startsWith('::ffff:') ? ip.slice(7) : ip;
}

function isLoopback(ip) {
  const a = normalizeIp(ip);
  return a === '::1' || a === 'localhost' || /^127\./.test(a);
}

/** Tailscale hands out 100.64.0.0/10 (CGNAT). */
function isTailnet(ip) {
  const a = normalizeIp(ip);
  const m = a.match(/^100\.(\d{1,3})\./);
  if (!m) return false;
  const second = Number(m[1]);
  return second >= 64 && second <= 127;
}

/**
 * Skip limiting for this box's own traffic. This is a single-user machine: the
 * limiter is here for a future public deployment, not for Ellie's browser.
 * Loopback covers the Tailscale serve hop; tailnet covers her other devices.
 */
function skipLocal(req) {
  const rl = getConfig().rateLimit || {};
  const ip = req.ip;
  if (rl.exemptLoopback !== false && isLoopback(ip)) return true;
  if (rl.exemptTailnet !== false && isTailnet(ip)) return true;
  return false;
}

/**
 * Log every rejection. Previously a 429 left NO trace in the journal, which is
 * why an app-wide outage had to be found by hand with curl.
 */
function makeLimitHandler(label, message) {
  return (req, res) => {
    console.warn(
      `[RateLimit] 429 ${label} — key=${normalizeIp(req.ip) || 'unknown'} ` +
      `${req.method} ${req.originalUrl}` +
      (req.get('x-forwarded-for') ? ` xff=${req.get('x-forwarded-for')}` : '')
    );
    res.status(429).json(message);
  };
}

const rlCfg = () => getConfig().rateLimit || {};

const apiLimiter = rateLimit({
  windowMs: (rlCfg().windowMinutes ?? 15) * 60 * 1000,
  max: rlCfg().max ?? 1000,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
  // /api/tts is metered by ttsLimiter instead — without this exclusion a spoken
  // reply's per-sentence requests would be charged to the shared bucket too.
  skip: (req) => skipLocal(req) || req.path.startsWith('/tts'),
  handler: makeLimitHandler('api', { error: 'Too many requests, please try again later.' })
});

const chatLimiter = rateLimit({
  windowMs: (rlCfg().chatWindowMinutes ?? 1) * 60 * 1000,
  max: rlCfg().chatMax ?? 60,
  message: { error: 'Too many chat requests, please slow down.' },
  standardHeaders: true,
  legacyHeaders: false,
  skip: skipLocal,
  handler: makeLimitHandler('chat', { error: 'Too many chat requests, please slow down.' })
});

// TTS gets its own budget: the client sends one request per sentence of a
// spoken reply, so a long answer is dozens of calls. Under the shared /api/
// bucket that alone could exhaust the window.
const ttsLimiter = rateLimit({
  windowMs: (rlCfg().ttsWindowMinutes ?? 1) * 60 * 1000,
  max: rlCfg().ttsMax ?? 240,
  message: { error: 'Too many speech requests, please slow down.' },
  standardHeaders: true,
  legacyHeaders: false,
  skip: skipLocal,
  handler: makeLimitHandler('tts', { error: 'Too many speech requests, please slow down.' })
});

// SECURITY: Content Security Policy headers
app.use((req, res, next) => {
  res.setHeader('Content-Security-Policy',
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline' https://static.cloudflareinsights.com; " +
    "style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data: blob:; " +
    "media-src 'self' blob: data:; " +
    "connect-src 'self' https://cloudflareinsights.com; " +
    "font-src 'self'; " +
    "object-src 'none'; " +
    "base-uri 'self'; " +
    "form-action 'self'; " +
    "frame-ancestors 'none';"
  );
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

// Middleware
app.use(express.json({ limit: '10mb' })); // Limit payload size

// SECURITY FIX: Serve only the public directory, not the entire project
app.use(express.static(path.join(__dirname, 'public')));

// Apply rate limiting to API routes
// TTS first — it has its own (much higher) budget and is excluded from the
// shared /api/ bucket by apiLimiter's skip.
app.use('/api/tts', ttsLimiter);
app.use('/api/', apiLimiter);

// Mount conversation routes
app.use('/api/conversations', conversationsRouter);

// Mount memory routes
app.use('/api/memory', memoryRouter);

// ============ Config API ============

app.get('/api/config', (req, res) => {
  res.json(getConfig());
});

app.put('/api/config', (req, res) => {
  if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body)) {
    return res.status(400).json({ error: 'Request body must be a JSON object' });
  }
  const updated = updateConfig(req.body);
  // Config is the single source of truth for which tools are REGISTERED, not just
  // which ones route — so a toggle has to re-register, or the two drift apart
  // again within the one process. (Turning search on without this left the
  // routing gate open onto a registry with no web_search in it: intent-classified
  // into the tool loop, nothing there to call.) Cheap and idempotent: it just
  // re-reads config and rebuilds a 3-entry Map.
  try {
    mcpClient.loadConfig();
  } catch (e) {
    console.error('[MCP] tool re-registration after config update failed:', e.message);
  }
  // Several manifest entries are config-gated (voice, web search, cron
  // proposals), so toggling one changes what SNH actually offers. Injection is
  // unaffected — buildInjectionBlock() recomputes from live config on every
  // request — but the ops ledger only reconciled at boot, so a capability that
  // came or went mid-session left no trail until the next restart. Re-sync here
  // so the ledger records it when it happens. Idempotent: a no-op when nothing
  // changed.
  try {
    const { added, removed } = capabilityManifest.syncToOps();
    if (added.length || removed.length) {
      console.log(`[Capabilities] config change synced: +${added.length} -${removed.length}`);
    }
  } catch (e) {
    console.error('[Capabilities] syncToOps after config update failed:', e.message);
  }
  res.json(updated);
});

// ============ Security Validation Functions ============

// SECURITY: Validate Ollama host against allowlist to prevent SSRF
function isValidOllamaHost(host) {
  if (!host) return false;

  try {
    const url = new URL(host);
    const hostname = url.hostname;

    // Allow localhost and loopback
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1') {
      return true;
    }

    // Allow private network ranges (RFC 1918)
    if (hostname.match(/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/)) {
      return true;
    }
    if (hostname.match(/^172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}$/)) {
      return true;
    }
    if (hostname.match(/^192\.168\.\d{1,3}\.\d{1,3}$/)) {
      return true;
    }

    // Allow explicitly configured hosts
    if (ALLOWED_OLLAMA_HOSTS.includes(host)) {
      return true;
    }

    return false;
  } catch (e) {
    return false;
  }
}

// SECURITY: Validate model name format
function isValidModelName(model) {
  if (!model || typeof model !== 'string') return false;

  // Allow alphanumeric, hyphens, underscores, dots, colons (for model versions),
  // and forward slashes (for vLLM HuggingFace-style names like Qwen/Qwen3-Coder)
  // Limit length to prevent abuse
  return /^[a-zA-Z0-9._:\-\/]{1,200}$/.test(model);
}

// SECURITY: Validate message array
function isValidMessageArray(messages) {
  if (!Array.isArray(messages)) return false;

  // Limit number of messages to prevent memory abuse
  if (messages.length > 100) return false;

  // Validate each message structure
  return messages.every(msg => {
    if (!msg || typeof msg !== 'object') return false;
    if (typeof msg.role !== 'string' || typeof msg.content !== 'string') return false;
    if (!['user', 'assistant', 'system'].includes(msg.role)) return false;
    // Limit individual message size
    if (msg.content.length > 100000) return false;
    return true;
  });
}

// Configuration from environment (OpenAI)
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';

// Provider configuration - all providers always visible, API key checked at usage time
const PROVIDERS = {
  ollama: {
    id: 'ollama',
    name: 'Ollama (Local)',
    requiresKey: false,
    models: [] // Dynamically loaded
  },
  claude: {
    id: 'claude',
    name: 'Claude',
    requiresKey: true,
    models: [
      { id: 'claude-opus-4-5-20251101', name: 'Claude Opus 4.5 (Flagship)' },
      { id: 'claude-sonnet-4-5-20250929', name: 'Claude Sonnet 4.5' },
      { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5 (Fast)' },
      { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4' },
      { id: 'claude-opus-4-20250514', name: 'Claude Opus 4' }
    ]
  },
  openai: {
    id: 'openai',
    name: 'OpenAI',
    requiresKey: true,
    models: [] // Dynamically loaded from API
  },
  grok: {
    id: 'grok',
    name: 'Grok',
    requiresKey: true,
    models: [
      { id: 'grok-4-1-fast-reasoning', name: 'Grok 4.1 Fast Reasoning (2M)' },
      { id: 'grok-4-1-fast-non-reasoning', name: 'Grok 4.1 Fast Non-Reasoning (2M)' },
      { id: 'grok-code-fast-1', name: 'Grok Code Fast 1 (256K)' },
      { id: 'grok-4-fast-reasoning', name: 'Grok 4 Fast Reasoning (2M)' },
      { id: 'grok-4-fast-non-reasoning', name: 'Grok 4 Fast Non-Reasoning (2M)' },
      { id: 'grok-4-0709', name: 'Grok 4 (256K)' },
      { id: 'grok-3-mini', name: 'Grok 3 Mini (131K)' },
      { id: 'grok-3', name: 'Grok 3 (131K)' },
      { id: 'grok-2-vision-1212', name: 'Grok 2 Vision (32K)' },
      { id: 'grok-2-image-1212', name: 'Grok 2 Image Gen' }
    ]
  },
  squatchserve: {
    id: 'squatchserve',
    name: 'SquatchServe (Local)',
    requiresKey: false,
    models: [] // Dynamically loaded from localhost:8001
  },
  llamacpp: {
    id: 'llamacpp',
    name: 'Llama.cpp (Local)',
    requiresKey: false,
    models: [] // Dynamically loaded from /api/llamacpp/models
  },
  vllm: {
    id: 'vllm',
    name: 'vLLM',
    requiresKey: false,
    models: []
  }
};

// ============ Provider Endpoints ============

// Get available providers — cloud providers as single entries, local providers as named instances
app.post('/api/providers', (req, res) => {
  const { hasClaudeKey, hasGrokKey, hasOpenAIKey } = req.body;
  const config = getConfig();

  const providerList = [];

  // Cloud providers (single entry each)
  providerList.push({
    id: 'claude',
    type: 'claude',
    name: 'Claude',
    requiresKey: true,
    hasKey: !!(CLAUDE_API_KEY || hasClaudeKey),
    models: PROVIDERS.claude.models
  });
  providerList.push({
    id: 'openai',
    type: 'openai',
    name: 'OpenAI',
    requiresKey: true,
    hasKey: !!(OPENAI_API_KEY || hasOpenAIKey),
    models: []
  });
  providerList.push({
    id: 'grok',
    type: 'grok',
    name: 'Grok',
    requiresKey: true,
    hasKey: !!(GROK_API_KEY || hasGrokKey),
    models: PROVIDERS.grok.models
  });

  // SquatchServe (single entry, unchanged)
  providerList.push({
    id: 'squatchserve',
    type: 'squatchserve',
    name: 'SquatchServe',
    requiresKey: false,
    hasKey: true,
    models: []
  });

  // Instance-based local providers
  const typeLabels = { ollama: 'Ollama', vllm: 'vLLM', llamacpp: 'Llama.cpp' };
  for (const providerType of ['ollama', 'vllm', 'llamacpp']) {
    const instances = config.providers[providerType] || [];
    for (const inst of instances) {
      const typeLabel = typeLabels[providerType] || providerType;
      providerList.push({
        id: `${providerType}:${inst.name}`,
        type: providerType,
        name: `${typeLabel} — ${inst.name}`,
        instanceName: inst.name,
        host: inst.host,
        model: inst.model || null,
        requiresKey: false,
        hasKey: true,
        models: inst.model ? [{ id: inst.model, name: inst.model }] : []
      });
    }
  }

  res.json({
    providers: providerList,
    instances: config.providers
  });
});

// Legacy GET endpoint for backwards compatibility
app.get('/api/providers', (req, res) => {
  const availableProviders = Object.values(PROVIDERS).map(p => ({
    id: p.id,
    name: p.name,
    requiresKey: p.requiresKey,
    hasKey: p.id === 'ollama' || p.id === 'squatchserve' || p.id === 'llamacpp' ? true : false, // Can't know client keys in GET
    models: ['ollama', 'openai', 'squatchserve', 'llamacpp'].includes(p.id) ? [] : p.models
  }));
  res.json({ providers: availableProviders });
});

// ============ Ollama Proxy ============

// Helper to get validated Ollama host
function getOllamaHost(requestBody) {
  const requestedHost = requestBody?.ollamaHost;

  // If no custom host requested, use default
  if (!requestedHost) {
    return OLLAMA_HOST;
  }

  // SECURITY: Validate requested host to prevent SSRF
  if (!isValidOllamaHost(requestedHost)) {
    throw new Error('Invalid Ollama host. Only localhost, private network IPs, and explicitly allowed hosts are permitted.');
  }

  return requestedHost;
}

// Proxy Ollama tags (model list) - POST to accept custom host or named instance
app.post('/api/tags', async (req, res) => {
  try {
    const { providerType, instanceName } = req.body;
    let host;

    if (providerType && instanceName) {
      const inst = getProviderInstance(providerType, instanceName);
      if (!inst) return res.status(404).json({ error: 'Instance not found', models: [] });
      host = inst.host;
      if (!isValidOllamaHost(host)) {
        return res.status(400).json({ error: 'Invalid host address', models: [] });
      }
    } else {
      host = getOllamaHost(req.body);
    }

    const response = await fetch(`${host}/api/tags`);
    if (!response.ok) {
      throw new Error(`Ollama returned ${response.status}`);
    }
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Ollama tags error:', error.message);
    res.status(503).json({ error: error.message || 'Ollama is not available', models: [] });
  }
});

// Legacy GET endpoint for backwards compatibility
app.get('/api/tags', async (req, res) => {
  try {
    const response = await fetch(`${OLLAMA_HOST}/api/tags`);
    if (!response.ok) {
      throw new Error(`Ollama returned ${response.status}`);
    }
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Ollama tags error:', error.message);
    res.status(503).json({ error: 'Ollama is not available', models: [] });
  }
});

// Proxy Ollama chat
app.post('/api/chat', chatLimiter, async (req, res) => {
  try {
    const { model, messages, ollamaHost, ...otherParams } = req.body;

    // SECURITY: Validate inputs
    if (!isValidModelName(model)) {
      return res.status(400).json({ error: 'Invalid model name' });
    }

    if (!isValidMessageArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages array' });
    }

    const host = getOllamaHost(req.body);

    // Build validated request body
    const ollamaBody = {
      model,
      messages,
      ...otherParams
    };

    const response = await fetch(`${host}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ollamaBody)
    });

    if (!response.ok) {
      throw new Error(`Ollama returned ${response.status}`);
    }

    // Stream the response
    res.setHeader('Content-Type', 'application/x-ndjson');
    const reader = response.body.getReader();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(value);
      }
      res.end();
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('Ollama chat error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: error.message || 'Ollama is not available' });
    }
  }
});

// ============ Claude API Proxy ============

app.post('/api/claude/chat', chatLimiter, async (req, res) => {
  const { model, messages, apiKey } = req.body;

  // SECURITY: Validate inputs
  if (!isValidModelName(model)) {
    return res.status(400).json({ error: 'Invalid model name' });
  }

  if (!isValidMessageArray(messages)) {
    return res.status(400).json({ error: 'Invalid messages array' });
  }

  // Use client key if provided, otherwise fall back to server key
  const claudeKey = apiKey || CLAUDE_API_KEY;

  if (!claudeKey) {
    return res.status(401).json({ error: 'Claude API key not configured' });
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': claudeKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: model,
        max_tokens: 4096,
        stream: true,
        messages: messages
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Claude API error:', error);
      return res.status(response.status).json({ error: 'Claude API error' });
    }

    // Stream SSE response
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        res.write(chunk);
      }
      res.end();
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('Claude proxy error:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to connect to Claude API' });
    }
  }
});

// ============ Grok API Proxy ============

app.post('/api/grok/chat', chatLimiter, async (req, res) => {
  const { model, messages, apiKey } = req.body;

  // SECURITY: Validate inputs
  if (!isValidModelName(model)) {
    return res.status(400).json({ error: 'Invalid model name' });
  }

  if (!isValidMessageArray(messages)) {
    return res.status(400).json({ error: 'Invalid messages array' });
  }

  // Use client key if provided, otherwise fall back to server key
  const grokKey = apiKey || GROK_API_KEY;

  if (!grokKey) {
    return res.status(401).json({ error: 'Grok API key not configured' });
  }

  try {
    const response = await fetch('https://api.x.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${grokKey}`
      },
      body: JSON.stringify({
        model: model,
        stream: true,
        messages: messages
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Grok API error:', error);
      return res.status(response.status).json({ error: 'Grok API error' });
    }

    // Stream SSE response
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        res.write(chunk);
      }
      res.end();
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('Grok proxy error:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to connect to Grok API' });
    }
  }
});

// ============ OpenAI API Proxy ============

// Fetch OpenAI models dynamically
app.post('/api/openai/models', async (req, res) => {
  const { apiKey } = req.body;
  const openaiKey = apiKey || OPENAI_API_KEY;

  if (!openaiKey) {
    return res.status(401).json({ error: 'OpenAI API key not configured' });
  }

  try {
    const response = await fetch('https://api.openai.com/v1/models', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${openaiKey}`
      }
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('OpenAI models error:', error);
      return res.status(response.status).json({ error: 'Failed to fetch OpenAI models' });
    }

    const data = await response.json();

    // Filter and format models - exclude legacy/deprecated models (3.5 and older)
    // Keep chat-capable models (gpt-4+, o1+, o3+, chatgpt-*)
    const chatModels = data.data
      .filter(model => {
        const id = model.id.toLowerCase();
        // Exclude legacy/deprecated models
        if (id.includes('gpt-3.5') || id.includes('gpt-3') || id.includes('davinci') ||
            id.includes('curie') || id.includes('babbage') || id.includes('ada') ||
            id.includes('text-') || id.includes('code-') || id.includes('instruct') ||
            id.includes('whisper') || id.includes('tts') || id.includes('dall-e') ||
            id.includes('embedding') || id.includes('moderation')) {
          return false;
        }
        // Include modern chat models (gpt-4, gpt-5, gpt-6, etc., o1, o3, o4, etc.)
        return id.startsWith('gpt-4') || id.startsWith('gpt-5') || id.startsWith('gpt-6') ||
               id.startsWith('gpt-7') || id.startsWith('o1') || id.startsWith('o3') ||
               id.startsWith('o4') || id.startsWith('chatgpt-');
      })
      .map(model => ({
        id: model.id,
        name: formatOpenAIModelName(model.id)
      }))
      .sort((a, b) => {
        // Sort by preference: newest/best first
        // o4 > o3 > o1 > gpt-5+ > gpt-4o > gpt-4 > chatgpt
        const order = ['o4', 'o3', 'o1', 'gpt-7', 'gpt-6', 'gpt-5', 'gpt-4o', 'gpt-4', 'chatgpt'];
        const aPrefix = order.findIndex(p => a.id.startsWith(p));
        const bPrefix = order.findIndex(p => b.id.startsWith(p));
        if (aPrefix !== -1 && bPrefix !== -1 && aPrefix !== bPrefix) return aPrefix - bPrefix;
        if (aPrefix !== -1 && bPrefix === -1) return -1;
        if (aPrefix === -1 && bPrefix !== -1) return 1;
        return a.id.localeCompare(b.id);
      });

    res.json({ models: chatModels });
  } catch (error) {
    console.error('OpenAI models proxy error:', error.message);
    res.status(500).json({ error: 'Failed to connect to OpenAI API' });
  }
});

// Helper to format OpenAI model names nicely
function formatOpenAIModelName(modelId) {
  const nameMap = {
    // GPT-4 series
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'gpt-4': 'GPT-4',
    // GPT-5 series
    'gpt-5': 'GPT-5',
    'gpt-5.0': 'GPT-5.0',
    'gpt-5.2': 'GPT-5.2',
    'gpt-5-turbo': 'GPT-5 Turbo',
    // Reasoning models
    'o1': 'o1 (Reasoning)',
    'o1-mini': 'o1 Mini',
    'o1-preview': 'o1 Preview',
    'o1-pro': 'o1 Pro',
    'o3': 'o3 (Reasoning)',
    'o3-mini': 'o3 Mini',
    'o3-pro': 'o3 Pro',
    'o4-mini': 'o4 Mini',
    // ChatGPT
    'chatgpt-4o-latest': 'ChatGPT-4o Latest'
  };

  // Check for exact match first
  if (nameMap[modelId]) return nameMap[modelId];

  // Check for prefix match
  for (const [prefix, name] of Object.entries(nameMap)) {
    if (modelId.startsWith(prefix + '-')) {
      return `${name.split(' (')[0]} (${modelId})`;
    }
  }

  // Default: clean up and format nicely
  // Handle patterns like gpt-5.2-turbo, o3-mini-2024-01-01, etc.
  return modelId
    .replace(/^gpt-/, 'GPT-')
    .replace(/^o(\d)/, 'o$1')
    .replace(/-(\d{4}-\d{2}-\d{2})$/, ' ($1)')  // Date suffixes
    .replace(/-/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// OpenAI chat endpoint
app.post('/api/openai/chat', chatLimiter, async (req, res) => {
  const { model, messages, apiKey } = req.body;

  // SECURITY: Validate inputs
  if (!isValidModelName(model)) {
    return res.status(400).json({ error: 'Invalid model name' });
  }

  if (!isValidMessageArray(messages)) {
    return res.status(400).json({ error: 'Invalid messages array' });
  }

  // Use client key if provided, otherwise fall back to server key
  const openaiKey = apiKey || OPENAI_API_KEY;

  if (!openaiKey) {
    return res.status(401).json({ error: 'OpenAI API key not configured' });
  }

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${openaiKey}`
      },
      body: JSON.stringify({
        model: model,
        stream: true,
        messages: messages
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('OpenAI API error:', error);
      return res.status(response.status).json({ error: 'OpenAI API error' });
    }

    // Stream SSE response
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        res.write(chunk);
      }
      res.end();
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('OpenAI proxy error:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to connect to OpenAI API' });
    }
  }
});

// ============ SquatchServe API Proxy ============

const SQUATCHSERVE_HOST = process.env.SQUATCHSERVE_HOST || 'http://localhost:8111';
const LLAMACPP_HOST = process.env.LLAMACPP_HOST || 'http://localhost:8080';

// Initialize MCP tool client. SHARED with the heartbeat (2026-08-03) so there is
// exactly one answer to "which tools exist" — the capability drift-check reads
// that answer, and two registries could make the manifest true of one and false
// of the other.
const mcpClient = MCPClient.shared();

// Fetch SquatchServe models dynamically (Ollama-compatible API)
app.get('/api/squatchserve/models', async (req, res) => {
  try {
    const squatchserveHost = req.query.host || SQUATCHSERVE_HOST;
    if (!isValidOllamaHost(squatchserveHost)) {
      return res.status(400).json({ error: 'Invalid SquatchServe host address' });
    }
    const response = await fetch(`${squatchserveHost}/api/tags`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('SquatchServe models error:', error);
      return res.status(response.status).json({ error: 'Failed to fetch SquatchServe models' });
    }

    const data = await response.json();

    // Format models from Ollama-compatible response: { models: [{name, ...}] }
    const models = (data.models || []).map(model => ({
      id: model.name,
      name: model.name
    }));

    res.json({ models });
  } catch (error) {
    console.error('SquatchServe models proxy error:', error.message);
    res.status(503).json({ error: 'SquatchServe is not available', models: [] });
  }
});

// SquatchServe chat endpoint (Ollama-compatible streaming)
app.post('/api/squatchserve/chat', chatLimiter, async (req, res) => {
  const { model, messages, squatchserveHost } = req.body;

  // SECURITY: Validate inputs
  if (!isValidModelName(model)) {
    return res.status(400).json({ error: 'Invalid model name' });
  }

  if (!isValidMessageArray(messages)) {
    return res.status(400).json({ error: 'Invalid messages array' });
  }

  const host = squatchserveHost || SQUATCHSERVE_HOST;
  if (!isValidOllamaHost(host)) {
    return res.status(400).json({ error: 'Invalid SquatchServe host address' });
  }

  try {
    const response = await fetch(`${host}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: model,
        stream: true,
        messages: messages
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('SquatchServe API error:', error);
      return res.status(response.status).json({ error: 'SquatchServe API error' });
    }

    // Stream NDJSON response (Ollama-compatible format)
    res.setHeader('Content-Type', 'application/x-ndjson');

    const reader = response.body.getReader();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(value);
      }
      res.end();
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('SquatchServe proxy error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: 'SquatchServe is not available' });
    }
  }
});

// SquatchServe status endpoint - get loaded models and VRAM usage
app.get('/api/squatchserve/ps', async (req, res) => {
  try {
    const squatchserveHost = req.query.host || SQUATCHSERVE_HOST;
    if (!isValidOllamaHost(squatchserveHost)) {
      return res.status(400).json({ error: 'Invalid SquatchServe host address' });
    }
    const response = await fetch(`${squatchserveHost}/api/ps`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('SquatchServe ps error:', error);
      return res.status(response.status).json({ error: 'Failed to get SquatchServe status' });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('SquatchServe ps proxy error:', error.message);
    res.status(503).json({ error: 'SquatchServe is not available', models: [], gpu: {} });
  }
});

// SquatchServe unload endpoint - unload a model to free VRAM
app.post('/api/squatchserve/unload', async (req, res) => {
  const { name, squatchserveHost } = req.body;

  if (!name || typeof name !== 'string') {
    return res.status(400).json({ error: 'Model name is required' });
  }

  const host = squatchserveHost || SQUATCHSERVE_HOST;
  if (!isValidOllamaHost(host)) {
    return res.status(400).json({ error: 'Invalid SquatchServe host address' });
  }

  try {
    const response = await fetch(`${host}/api/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('SquatchServe unload error:', error);
      return res.status(response.status).json({ error: 'Failed to unload model' });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('SquatchServe unload proxy error:', error.message);
    res.status(503).json({ error: 'SquatchServe is not available' });
  }
});

// ============ Llama.cpp API Proxy ============

// Fetch Llama.cpp models (hardcoded list, kept for backward compatibility)
app.get('/api/llamacpp/models', (req, res) => {
  const models = [
    { id: 'qwen3-coder', name: 'Qwen3 Coder Next' },
    { id: 'qwen3-next', name: 'Qwen3 Next' },
    { id: 'scout', name: 'Llama 4 Scout 109B' }
  ];
  res.json({ models });
});

// Fetch models for a named provider instance
// Ollama: fetches live from /api/tags; vLLM and llama.cpp: returns the configured model name
app.post('/api/instance/models', async (req, res) => {
  try {
    const { providerType, instanceName } = req.body;

    if (!providerType || !instanceName) {
      return res.status(400).json({ error: 'providerType and instanceName are required' });
    }

    const inst = getProviderInstance(providerType, instanceName);
    if (!inst) return res.status(404).json({ error: 'Instance not found' });

    if (providerType === 'ollama') {
      if (!isValidOllamaHost(inst.host)) {
        return res.status(400).json({ error: 'Invalid host address' });
      }
      const response = await fetch(`${inst.host}/api/tags`);
      if (!response.ok) throw new Error(`Ollama returned ${response.status}`);
      const data = await response.json();
      const models = (data.models || []).map(m => ({ id: m.name, name: m.name }));
      return res.json({ models });
    }

    if (providerType === 'vllm' || providerType === 'llamacpp') {
      if (!isValidOllamaHost(inst.host)) {
        return res.status(400).json({ error: 'Invalid host address' });
      }
      if (inst.model) {
        return res.json({ models: [{ id: inst.model, name: inst.model }] });
      }
      return res.json({ models: [] });
    }

    res.status(400).json({ error: 'Unsupported provider type' });
  } catch (error) {
    console.error('[Models] Error fetching instance models:', error.message);
    res.status(500).json({ error: 'Failed to fetch models', details: error.message });
  }
});

// ============ SearXNG Web Search ============

// SearXNG search endpoint
app.post('/api/search', async (req, res) => {
  try {
    const { query, searxngHost } = req.body;

    if (!query || typeof query !== 'string' || query.length > 500) {
      return res.status(400).json({ error: 'Invalid search query' });
    }

    // SECURITY: Validate custom SearXNG host
    const host = searxngHost ? (isValidOllamaHost(searxngHost) ? searxngHost : null) : SEARXNG_HOST;
    if (!host) {
      return res.status(400).json({ error: 'Invalid SearXNG host' });
    }

    const searchUrl = `${host}/search?q=${encodeURIComponent(query)}&format=json`;
    const response = await fetch(searchUrl, {
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error(`SearXNG returned ${response.status}`);
    }

    const data = await response.json();
    const results = (data.results || []).slice(0, 5).map(r => ({
      url: r.url,
      title: r.title,
      content: r.content
    }));

    res.json({ results });
  } catch (error) {
    console.error('SearXNG search error:', error.message);
    res.status(503).json({ error: 'Search service unavailable', results: [] });
  }
});

// ============ Voice Provider API ============

app.get('/api/voice/providers', (req, res) => {
  const config = getConfig();
  const voice = JSON.parse(JSON.stringify(config.voice || {}));
  // Redact API keys — only expose whether one is set
  for (const category of ['stt', 'tts']) {
    if (voice[category]?.providers) {
      for (const p of voice[category].providers) {
        if (p.api_key) {
          p.hasApiKey = true;
          delete p.api_key;
        }
      }
    }
  }
  res.json(voice);
});

const VALID_STT_TYPES = new Set(['whisper', 'faster-whisper', 'canary', 'parakeet', 'deepgram', 'openai-whisper']);
const VALID_TTS_TYPES = new Set(['kokoro', 'piper', 'chatterbox', 'orpheus', 'qwen3tts', 'elevenlabs', 'openai-tts']);
const CLOUD_VOICE_TYPES = new Set(['deepgram', 'openai-whisper', 'elevenlabs', 'openai-tts']);

app.post('/api/voice/providers', (req, res) => {
  if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body)) {
    return res.status(400).json({ error: 'Request body must be a JSON object' });
  }

  const typeAllowlists = { stt: VALID_STT_TYPES, tts: VALID_TTS_TYPES };

  // Only accept known categories
  const sanitized = {};
  for (const category of ['stt', 'tts']) {
    if (!req.body[category]) continue;
    const cat = req.body[category];

    sanitized[category] = {};
    if (cat.active && typeof cat.active === 'string') {
      sanitized[category].active = cat.active;
    }

    if (cat.providers && !Array.isArray(cat.providers)) {
      return res.status(400).json({ error: `voice.${category}.providers must be an array` });
    }

    if (Array.isArray(cat.providers)) {
      sanitized[category].providers = [];
      for (const p of cat.providers) {
        if (!p.name || typeof p.name !== 'string') {
          return res.status(400).json({ error: 'Each provider must have a name' });
        }
        if (!p.type || typeof p.type !== 'string' || !typeAllowlists[category].has(p.type)) {
          return res.status(400).json({ error: `Invalid type "${p.type}" for ${category} provider` });
        }

        const cleaned = { name: p.name.trim(), type: p.type };

        if (CLOUD_VOICE_TYPES.has(p.type)) {
          // Cloud providers: accept api_key, strip any host
          if (p.api_key && typeof p.api_key === 'string') {
            cleaned.api_key = p.api_key;
          } else {
            // Preserve existing key if client didn't send one (redacted round-trip)
            const config = getConfig();
            const existing = config.voice?.[category]?.providers?.find(
              ep => ep.name === cleaned.name && ep.type === cleaned.type
            );
            if (existing?.api_key) cleaned.api_key = existing.api_key;
          }
        } else {
          // Local providers: require valid host
          if (!p.host || !isValidOllamaHost(p.host)) {
            return res.status(400).json({ error: `Invalid host for provider "${p.name}": must be localhost or private network` });
          }
          cleaned.host = p.host;
        }

        sanitized[category].providers.push(cleaned);
      }
    }
  }

  const updated = updateConfig({ voice: sanitized });
  res.json(updated.voice || {});
});

// ============ Voice Assistant Proxy ============

// Text-to-Speech proxy (Kokoro TTS)
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice, speed } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text is required' });
    }

    // Limit text length to prevent abuse
    if (text.length > 10000) {
      return res.status(400).json({ error: 'Text too long (max 10000 characters)' });
    }

    // Resolve active TTS provider from config
    // Cloud types always use hardcoded official API hosts (never user-supplied host)
    const CLOUD_TTS_HOSTS = { 'openai-tts': 'https://api.openai.com', 'elevenlabs': 'https://api.elevenlabs.io' };
    const ttsProvider = getVoiceProvider('tts');
    const ttsType = ttsProvider?.type || 'kokoro';
    const ttsHost = CLOUD_TTS_HOSTS[ttsType] || ttsProvider?.host;

    let response;
    if (ttsType === 'piper') {
      // Piper uses GET /api/tts with query params
      const params = new URLSearchParams({ text });
      response = await fetch(`${ttsHost}/api/tts?${params}`, { method: 'GET' });
    } else if (ttsType === 'qwen3tts') {
      // Qwen3 TTS uses GET /tts with query params
      const params = new URLSearchParams({ text });
      response = await fetch(`${ttsHost}/tts?${params}`, { method: 'GET' });
    } else {
      // OpenAI-compatible: kokoro, chatterbox, orpheus, openai-tts, elevenlabs
      const fetchOpts = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: text,
          voice: voice || 'af_heart',
          speed: speed || 1.0
        })
      };
      // Cloud providers need API key in Authorization header
      if (ttsProvider?.api_key) {
        fetchOpts.headers['Authorization'] = `Bearer ${ttsProvider.api_key}`;
      }
      response = await fetch(`${ttsHost}/v1/audio/speech`, fetchOpts);
    }

    if (!response.ok) {
      const error = await response.text();
      console.error('TTS error:', error);
      return res.status(response.status).json({ error: 'TTS service error' });
    }

    // Stream audio response
    res.setHeader('Content-Type', response.headers.get('Content-Type') || 'audio/wav');
    const reader = response.body.getReader();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(value);
      }
      res.end();
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('TTS proxy error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: 'TTS service unavailable' });
    }
  }
});

// STT helper: resolve host/endpoint/headers by provider type
// Cloud types always use hardcoded official API hosts (never user-supplied host)
const CLOUD_STT_HOSTS = { 'deepgram': 'https://api.deepgram.com', 'openai-whisper': 'https://api.openai.com' };

function buildSTTRequest(sttProvider, boundary, contentType, body) {
  const sttType = sttProvider?.type || 'whisper';
  const sttHost = CLOUD_STT_HOSTS[sttType] || sttProvider?.host;

  if (sttType === 'openai-whisper') {
    // OpenAI Whisper API: POST /v1/audio/transcriptions, multipart with model field
    return fetch(`${sttHost}/v1/audio/transcriptions`, {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        ...(sttProvider?.api_key ? { 'Authorization': `Bearer ${sttProvider.api_key}` } : {})
      },
      body
    });
  } else if (sttType === 'deepgram') {
    // Deepgram: POST /v1/listen, raw audio body
    return fetch(`${sttHost}/v1/listen`, {
      method: 'POST',
      headers: {
        'Content-Type': contentType,
        ...(sttProvider?.api_key ? { 'Authorization': `Token ${sttProvider.api_key}` } : {})
      },
      body
    });
  } else {
    // Whisper-compatible: whisper, faster-whisper, canary, parakeet — POST /transcribe, multipart
    return fetch(`${sttHost}/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}` },
      body
    });
  }
}

// Speech-to-Text proxy
app.post('/api/stt', express.raw({ type: ['audio/*', 'application/octet-stream'], limit: '10mb' }), async (req, res) => {
  try {
    if (!req.body || req.body.length === 0) {
      return res.status(400).json({ error: 'No audio data provided' });
    }

    const sttProvider = getVoiceProvider('stt');
    const contentType = req.headers['content-type'] || 'audio/webm';
    const extension = contentType.includes('wav') ? 'wav' : 'webm';

    // Build multipart form data
    const boundary = '----FormBoundary' + Math.random().toString(36).substring(2);
    const header = Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="audio"; filename="recording.${extension}"\r\n` +
      `Content-Type: ${contentType}\r\n\r\n`
    );
    const footer = Buffer.from(`\r\n--${boundary}--\r\n`);
    const multipartBody = Buffer.concat([header, req.body, footer]);

    // Deepgram uses raw audio, others use multipart
    const sttType = sttProvider?.type || 'whisper';
    const body = sttType === 'deepgram' ? req.body : multipartBody;

    const response = await buildSTTRequest(sttProvider, boundary, contentType, body);

    if (!response.ok) {
      const error = await response.text();
      console.error('STT error:', error);
      return res.status(response.status).json({ error: 'STT service error' });
    }

    const result = await response.json();
    res.json(result);
  } catch (error) {
    console.error('STT proxy error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: 'STT service unavailable' });
    }
  }
});

// STT with multipart form data (for file uploads)
app.post('/api/stt/upload', express.raw({ type: 'audio/*', limit: '10mb' }), async (req, res) => {
  try {
    if (!req.body || req.body.length === 0) {
      return res.status(400).json({ error: 'No audio data provided' });
    }

    const sttProvider = getVoiceProvider('stt');
    const contentType = req.headers['content-type'] || 'audio/webm';

    // Build multipart form data
    const boundary = '----FormBoundary' + Math.random().toString(36).substring(2);
    const extension = contentType.includes('wav') ? 'wav' : 'webm';
    const header = Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="audio"; filename="recording.${extension}"\r\n` +
      `Content-Type: ${contentType}\r\n\r\n`
    );
    const footer = Buffer.from(`\r\n--${boundary}--\r\n`);
    const multipartBody = Buffer.concat([header, req.body, footer]);

    const sttType = sttProvider?.type || 'whisper';
    const body = sttType === 'deepgram' ? req.body : multipartBody;

    const response = await buildSTTRequest(sttProvider, boundary, contentType, body);

    if (!response.ok) {
      const error = await response.text();
      console.error('STT error:', error);
      return res.status(response.status).json({ error: 'STT service error' });
    }

    const result = await response.json();
    res.json(result);
  } catch (error) {
    console.error('STT proxy error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: 'STT service unavailable' });
    }
  }
});

// ============ Memory-Enhanced Chat Endpoint ============

/**
 * Detect a "what's on your mind?" style prompt — the user inviting SNH to share
 * whatever it has been thinking about. When this fires, pending initiatives (any
 * priority) are surfaced conversationally for this turn instead of the normal
 * single-greeting path.
 * @param {string} messageText
 * @returns {boolean}
 */
function isMindQuery(messageText) {
  const text = (messageText || '').toLowerCase().trim();
  if (!text || text.length > 140) return false;
  const patterns = [
    /what('?s| is| has| have)?\s*(been\s+)?on your mind/,
    /anything (on your mind|you('?ve| have)? been (thinking|mulling|pondering)|you want(ed)? to (bring up|share|tell me|say)|you'?d like to (bring up|share|raise))/,
    /(what|anything).{0,20}\byou('?ve| have)? been thinking about/,
    /have you been thinking about (anything|something)/,
    /what are you thinking about/,
    /got anything (on your mind|you want to say)/,
    /anything you'?ve been meaning to (say|tell me|bring up|mention)/,
  ];
  return patterns.some(r => r.test(text));
}

/**
 * POST /api/chat/memory
 * Enhanced chat endpoint that:
 * 1. Saves user message to SQLite
 * 2. Searches for relevant past context from other conversations
 * 3. Injects memory context into the system prompt
 * 4. Saves assistant response and embeds it for future retrieval
 */
app.post('/api/chat/memory', chatLimiter, async (req, res) => {
  let chatMarked = false;
  try {
    const { model, messages, ollamaHost, conversation_id, provider, apiKey, searxngHost, ttsEnabled, superSearch, inputModality } = req.body;

    // Read tool enabled state from config instead of per-request flag
    const appConfig = getConfig();
    // Search tools (web_search/web_fetch) ride on the SearXNG flag. Action tools
    // have their own flag — create_cron_job must work while SearXNG is off,
    // which is the default.
    const toolsEnabled = !!(appConfig.tools && appConfig.tools.searxng && appConfig.tools.searxng.enabled);
    const cronToolEnabled = !!(appConfig.tools && appConfig.tools.cron && appConfig.tools.cron.enabled !== false);
    const memoryWriteEnabled = !!(appConfig.tools && appConfig.tools.memoryWrite && appConfig.tools.memoryWrite.enabled !== false);
    const memoryInspectEnabled = !!(appConfig.tools && appConfig.tools.memoryInspect && appConfig.tools.memoryInspect.enabled !== false);

    // SECURITY: Validate inputs
    if (!isValidModelName(model)) {
      return res.status(400).json({ error: 'Invalid model name' });
    }

    if (!isValidMessageArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages array' });
    }

    // Get or create conversation. A request with no conversation_id is the
    // opening of a new conversation — the natural moment for a greeting initiative.
    const isConversationOpen = !conversation_id;
    let convoId = conversation_id;
    if (!convoId) {
      // Create a new conversation
      convoId = db.createConversation(null, model);
    }

    // Get the latest user message
    const userMessage = messages[messages.length - 1];
    if (userMessage.role !== 'user') {
      return res.status(400).json({ error: 'Last message must be from user' });
    }

    // Chat is king: mark this request in flight so the background agent pool
    // throttles to concurrency 1 and yields the GPU to the user-facing response.
    // Cleared in the finally below (covers stream completion, errors, disconnects).
    agentPool.beginChat();
    chatMarked = true;

    // DEBUG: Log conversation and message info
    console.log('=== Memory Chat ===');
    console.log('Conversation ID:', convoId);
    console.log('Provider:', provider || 'ollama', '| Model:', model);
    console.log('User message:', userMessage.content.substring(0, 80));
    console.log('Tools enabled:', toolsEnabled, '(type:', typeof toolsEnabled, ') | MCP has tools:', mcpClient.hasTools(), '| Tool names:', mcpClient.getToolNames());
    console.log('TTS enabled:', ttsEnabled, '(type:', typeof ttsEnabled, ')');
    console.log('Super Search:', !!superSearch);

    // Smart tool routing. Three independent reasons to enter the tool loop:
    //   - search intent, gated on the SearXNG stack being enabled
    //   - scheduling intent, gated on the create_cron_job action tool
    //   - memory-write intent, gated on the write_memory action tool
    // Any one pulls the turn into the loop; the model is handed the whole
    // registered tool set and picks. Scheduling intent is matched narrowly
    // (see classifySchedulingIntent) so ordinary conversation never routes here;
    // memory-write intent is matched the other way round, because missing the
    // ask is the failure that tool exists to fix.
    const needsSearchTools = toolsEnabled && mcpClient.hasTools() && classifyToolNeed(userMessage.content, !!superSearch);
    const needsActionTools = cronToolEnabled && mcpClient.hasTool('create_cron_job')
      && classifySchedulingIntent(userMessage.content);
    const needsMemoryWrite = memoryWriteEnabled && mcpClient.hasTool('write_memory')
      && classifyMemoryWriteIntent(userMessage.content);
    // Memory-READ intent gates the four inspection tools. Matched narrowly (see
    // classifyMemoryReadIntent): a false positive here has him rummaging through
    // the fact store during ordinary conversation, which is a worse failure than
    // missing a lookup he can be asked for again.
    const needsMemoryRead = memoryInspectEnabled && mcpClient.hasTool('memory_search')
      && classifyMemoryReadIntent(userMessage.content);
    // Memory-CORRECTION intent — "what changed in your memory", "why was that
    // corrected". Its own classifier and its own flag: the question is answerable
    // only from the corrections ledger, and a turn that asks it without routing
    // gets answered by reconstructing a reason from the facts that remain, which
    // is invention. Shares the inspection config flag and rate cap.
    const needsMemoryCorrections = memoryInspectEnabled && mcpClient.hasTool('memory_corrections')
      && classifyMemoryCorrectionIntent(userMessage.content);
    // Scheduled-JOB questions — "what did I approve", "did the digest run",
    // "which approved job never ran". The mirror of needsActionTools: that one
    // routes a request to MAKE a job, this routes a question ABOUT one. Before
    // this existed the question had no tool that could answer it, and what he did
    // instead was call a tool that does not exist and then invent the answer.
    const needsJobsRead = memoryInspectEnabled && mcpClient.hasTool('memory_jobs')
      && classifyJobsIntent(userMessage.content);
    const needsTools = needsSearchTools || needsActionTools || needsMemoryWrite || needsMemoryRead || needsMemoryCorrections || needsJobsRead;
    console.log('Tool routing:', needsTools
      ? `TOOLS (${[needsSearchTools && 'search/fetch', needsActionTools && 'scheduling', needsMemoryWrite && 'memory-write', needsMemoryRead && 'memory-read', needsMemoryCorrections && 'memory-corrections', needsJobsRead && 'jobs-read'].filter(Boolean).join(' + ')})`
      : 'DIRECT (conversational, skipping tool loop)');

    // Should-I-search honesty guard: if the question is about current/changeable
    // facts (weather, prices, news, "right now"/"latest") but search will NOT run
    // — because it's off, unavailable, or the classifier didn't route to it — the
    // model must NOT answer confidently from memory (7/23: it fabricated a weather
    // high). We inject an instruction to offer to look it up instead. When search
    // WILL run (needsTools), the tool loop handles it and no nudge is needed.
    const timeSensitiveUnsearched = isTimeSensitive(userMessage.content) && !needsTools;

    // Save user message to database
    // Modality rides from the client: 'stt' when the text came from a whisper
    // transcription, 'typed' from the keyboard. addMessage normalises anything
    // else to 'unknown' rather than assuming typed.
    const userMsgId = db.addMessage(convoId, 'user', userMessage.content, model, null, inputModality);

    // === UPGRADE 1: Durable memory ===
    // Long-term memory is RENDERED FROM SQLITE per request (2026-08-02). It used
    // to be read from data/memory/MEMORY.md, which made that file a second system
    // of record and let the injected block drift from the database. USER.md and
    // the daily logs are still files — they are written by hand or as a log, not
    // derived from the fact store.
    let memoryFiles = { memory: null, user: null, dailyToday: null, dailyYesterday: null };
    try {
      memoryFiles = db.loadMemoryFiles();
      // World facts are excluded unless config says otherwise: the injected
      // block runs to a tight diet, and unbounded external knowledge would
      // crowd out the personal facts it exists for. They stay reachable through
      // the inspect tools.
      const injectSubjects = ['user'];
      if (appConfig.memory?.injection?.includeWorld === true) injectSubjects.push('world');
      // Budgeted AT CLUSTER BOUNDARIES by the renderer itself, not sliced
      // afterwards on a character offset: the old way left the last surviving
      // cluster showing an arbitrary prefix of its facts, so a cluster holding
      // sixteen could read as holding three.
      memoryFiles.memory = memoryClusters.renderLongTermMemory({
        subject: injectSubjects,
        budgetTokens: appConfig.memory?.injection?.longTermTokens ?? 3000
      }) || null;
      console.log('Memory loaded:', {
        longTerm: memoryFiles.memory ? `${memoryFiles.memory.length} chars (rendered from SQLite)` : 'none',
        user: memoryFiles.user ? `${memoryFiles.user.length} chars` : 'none',
        dailyToday: memoryFiles.dailyToday ? `${memoryFiles.dailyToday.length} chars` : 'none',
        dailyYesterday: memoryFiles.dailyYesterday ? `${memoryFiles.dailyYesterday.length} chars` : 'none'
      });
    } catch (memFileError) {
      console.error('Memory load error:', memFileError.message);
    }

    // === UPGRADE 2: Hybrid search (vector + BM25) ===
    let memoryContext = [];
    try {
      memoryContext = await db.hybridSearch(userMessage.content, convoId, 5);
      console.log('Hybrid search results:', memoryContext.length);
      if (memoryContext.length > 0) {
        memoryContext.forEach((msg, i) => {
          console.log(`  Match ${i + 1}: "${msg.text.substring(0, 30)}..." (score: ${msg.similarity.toFixed(3)}, source: ${msg.source})`);
        });
      }

      // Embed user message for future retrieval
      const embedding = await db.generateEmbedding(userMessage.content);
      await db.addEmbedding(userMsgId, convoId, userMessage.content, 'user', embedding);
    } catch (searchError) {
      console.error('Memory retrieval error:', searchError.message);
      // Continue without memory - not a fatal error
    }

    // === UPGRADE 4: Cluster-aware memory retrieval ===
    let clusterContext = [];
    try {
      clusterContext = await memoryClusters.searchClusters(userMessage.content, 3);
      if (clusterContext.length > 0) {
        console.log(`Cluster search: ${clusterContext.length} relevant clusters found`);
        clusterContext.forEach((c, i) => {
          console.log(`  Cluster ${i + 1}: "${c.cluster.name}" (${c.members.length} members, ${c.linkedMembers.length} linked)`);
        });
      }
    } catch (clusterError) {
      console.error('Cluster search error:', clusterError.message);
    }

    // Build messages array with memory context injected
    let enhancedMessages = [...messages];

    // Build comprehensive memory system prompt.
    // Each source is capped to a configured token budget so the injected system
    // context stays small (fast prefill). Budgets live in config.memory.injection.
    const memoryParts = [];
    const injCfg = (getConfig().memory && getConfig().memory.injection) || {};
    // Held so the TOTAL ceiling can rebuild this message after every other block
    // has rendered — see the ceiling pass further down.
    let memorySystemMessage = null, memoryHeader = '', memoryFooter = '';
    // Front-of-prompt blocks, assembled in one place once they have all been
    // built — see the ordered assembly further down.
    let identityMessage = null, noticesMessage = null;

    // Add durable memory (rendered long-term facts + USER.md), long-term capped.
    if (memoryFiles.memory) {
      // No second cap here: the renderer already budgeted at cluster boundaries,
      // and re-slicing on characters would undo exactly that. The total ceiling
      // below is what trims this further if the sum still does not fit.
      memoryParts.push({ kind: 'ltm', label: 'long-term memory', text: `=== Long-Term Memory ===\n${memoryFiles.memory}` });
    }
    if (memoryFiles.user) {
      memoryParts.push({ kind: 'userProfile', label: 'user profile', text: `=== User Profile ===\n${memoryFiles.user}` });
    }

    // Daily logs for short-term continuity: inject today's most-recent entries
    // verbatim (up to dailyTodayTokens) plus a brief digest of the remainder +
    // yesterday (up to dailySummaryTokens), instead of both files wholesale.
    //
    // Entries this conversation itself produced are left out (see
    // budgetDailyLogs): they restate the message history already in the request,
    // and they are the reason this block used to change on every turn. Entries
    // from other conversations today — the actual continuity this block is for —
    // still render.
    if (memoryFiles.dailyToday || memoryFiles.dailyYesterday) {
      const { recent, summary, stats } = injectionBudget.budgetDailyLogs(
        memoryFiles.dailyToday || '',
        memoryFiles.dailyYesterday || '',
        { dailyTodayTokens: injCfg.dailyTodayTokens ?? 1500,
          dailySummaryTokens: injCfg.dailySummaryTokens ?? 400,
          excludeConversationId: (injCfg.dailyExcludeActiveConversation !== false) ? convoId : null }
      );
      if (recent) {
        memoryParts.push({ kind: 'dailyToday', label: "today's log", text: `=== Today's Session Log (most recent) ===\n${recent}` });
      }
      if (summary) {
        memoryParts.push({ kind: 'dailySummary', label: 'earlier/yesterday digest', text: `=== Earlier / Yesterday (brief) ===\n${summary}` });
      }
      console.log(`[Injection] Daily log budgeted: kept ${stats.todayBlocksKept}/${stats.todayBlocksTotal} today blocks (~${stats.recentTokens} tok) + digest (~${stats.summaryTokens} tok)` +
                  `${stats.todayBlocksSelfExcluded ? `, ${stats.todayBlocksSelfExcluded} echo entr${stats.todayBlocksSelfExcluded === 1 ? 'y' : 'ies'} from this conversation excluded` : ''}`);
    }

    // Add hybrid search results from past conversations (token-capped).
    if (memoryContext.length > 0) {
      const contextText = memoryContext
        .map((m, i) => `[Memory ${i + 1}] ${m.role}: ${m.text.substring(0, 500)}${m.text.length > 500 ? '...' : ''}`)
        .join('\n');
      const { text: capped } = injectionBudget.budgetText(
        contextText, injCfg.pastConvoTokens ?? 800, 'past conversations');
      memoryParts.push({ kind: 'pastConvo', label: 'past conversations', text: `=== Relevant Past Conversations ===\n${capped}` });
    }

    // Add cluster-aware memory context
    if (clusterContext.length > 0) {
      const withLearned = (content, createdAt) => {
        const ts = formatFactTimestamp(createdAt);
        return ts ? `${content} (learned ${ts})` : content;
      };
      const clusterText = clusterContext.map(c => {
        const memberText = c.members.map(m => `- ${withLearned(m.content, m.created_at)}`).join('\n');
        let text = `[${c.cluster.name}]\n${memberText}`;
        if (c.linkedMembers.length > 0) {
          const linkedText = c.linkedMembers
            .map(lm => `- (from ${lm.clusterName}) ${withLearned(lm.content, lm.created_at)}`)
            .join('\n');
          text += `\nRelated:\n${linkedText}`;
        }
        return text;
      }).join('\n\n');
      const { text: cappedClusters } = injectionBudget.budgetText(
        clusterText, injCfg.clusterTokens ?? 1200, 'cluster memory');
      memoryParts.push({ kind: 'clusters', label: 'cluster memory', text: `=== Associated Memory Clusters ===\n${cappedClusters}` });
    }

    // === Conversational nudge: "what's on your mind?" ===
    // When the user explicitly invites SNH to share, surface ALL pending
    // initiatives (any priority) for this turn with an instruction to raise them
    // naturally. Takes precedence over the single-greeting path below.
    let deliveredInitiative = null;
    let nudgeItems = null;
    try {
      if (isMindQuery(userMessage.content)) {
        const pend = initiatives.listPending({ limit: 5 });
        if (pend.length > 0) {
          nudgeItems = pend;
          const list = pend.map(it => `- (${it.type}) ${it.content}`).join('\n');
          memoryParts.push({ kind: 'guidance', label: 'on your mind', text: `=== On Your Mind (the user is asking) ===\nThe user just invited you to share what's on your mind. These are the things you (SNH) have genuinely been sitting with:\n${list}\nShare what feels worth sharing, in your own voice and conversationally — you don't have to raise every one, and lead with whatever matters most. Do not present them as a bulleted list or a formal report; talk to them.` });
          console.log(`[Initiative] Nudge surfaced ${pend.length} pending item(s) (convo ${convoId})`);
        }
      }
    } catch (nudgeErr) {
      console.error('[Initiative] Nudge selection error:', nudgeErr.message);
    }

    // === Conversation-open greeting: weave in one high-priority initiative ===
    // When the user opens a new conversation, SNH may raise the single most
    // pressing thing it has been meaning to say — at most one, woven in naturally.
    // Followups surface here at a lower bar (followupThreshold). Skipped when the
    // nudge already surfaced pending items this turn.
    try {
      if (isConversationOpen && !nudgeItems) {
        const initCfg = getConfig().initiative || {};
        const greetingThreshold = Number.isFinite(initCfg.greetingThreshold) ? initCfg.greetingThreshold : 7;
        const followupThreshold = Number.isFinite(initCfg.followupThreshold) ? initCfg.followupThreshold : 5;
        const top = initiatives.getTopForGreeting({ greetingThreshold, followupThreshold });
        if (top) {
          deliveredInitiative = top;
          memoryParts.push({ kind: 'guidance', label: 'something on your mind', text: `=== Something On Your Mind ===\nThere is one thing you (SNH) have been meaning to raise with the user: "${top.content}"\nIf it fits the conversation, you may open with it or weave it naturally into your first response — at most this one thing, phrased warmly and conversationally, never as a list or a formal notice. If it truly does not fit what the user said, let it go for now.` });
          console.log(`[Initiative] Greeting candidate (${top.type}, priority ${top.priority}): ${top.id}`);
        }
      }
    } catch (initErr) {
      console.error('[Initiative] Greeting selection error:', initErr.message);
    }

    // === Question queue: surface at most one pending question ===
    // If a retrieved cluster has a pending question and none has been asked in
    // this conversation yet, tell the model it MAY ask it if the moment fits.
    // Skipped when a greeting initiative is already in play (at most one ask).
    let surfacedQuestion = null;
    try {
      if (!deliveredInitiative && !nudgeItems && clusterContext.length > 0 && !questionQueue.hasAskedInConversation(convoId)) {
        const clusterIds = clusterContext.map(c => c.cluster.id).filter(Boolean);
        const pending = questionQueue.getPendingForClusters(clusterIds);
        if (pending) {
          surfacedQuestion = pending;
          memoryParts.push({ kind: 'guidance', label: 'open question', text: `=== Open Question You May Ask ===\nThere is one thing you could naturally clarify with the user, if — and only if — it fits the flow of the conversation: "${pending.question}"\nYou may weave this single question in conversationally at a natural moment. Do not ask more than this one question, do not interrogate, and skip it entirely if it does not fit.` });
        }
      }
    } catch (qErr) {
      console.error('[Questions] Surfacing error:', qErr.message);
    }

    if (memoryParts.length > 0) {
      console.log('Injecting memory context:', memoryParts.length, 'sections');
      // Header and footer are kept separately because the total ceiling, applied
      // once every block has rendered, rebuilds this message from whichever
      // parts survive the trim.
      memoryHeader = `You have access to the following memory and context:\n\n${injectionBudget.memoryFraming(needsTools)}\n\n`;
      memoryFooter = `\n\nUse this context if it helps answer the current question, but don't explicitly mention that you're using memory unless asked. A "(learned ...)" annotation on a fact shows when you first learned that fact. If the user asks when they told you something or when a fact was learned, only answer with a time that is shown in a "(learned ...)" annotation; if no such timestamp is present for that fact, say you don't know rather than estimating or inventing one.`;
      memorySystemMessage = {
        role: 'system',
        content: memoryHeader + memoryParts.map(p => p.text).join('\n\n') + memoryFooter
      };
      // NOT inserted here any more — held for the ordered assembly below, which
      // places blocks by how often they change rather than by the order the code
      // happens to build them in.
    } else {
      console.log('No memory context to inject');
    }

    // Mark the surfaced question as asked (asked_at set on surfacing).
    if (surfacedQuestion) {
      questionQueue.markAsked(surfacedQuestion.id, convoId);
      console.log(`[Questions] Surfaced to model (convo ${convoId}): "${surfacedQuestion.question}"`);
    }

    // Mark the greeting initiative delivered (it was woven into this response).
    if (deliveredInitiative) {
      initiatives.markDelivered(deliveredInitiative.id, { channel: 'greeting', conversationId: convoId });
      console.log(`[Initiative] Delivered greeting initiative ${deliveredInitiative.id} (convo ${convoId})`);
    }

    // Mark nudge-surfaced initiatives delivered (SNH shared them this turn).
    if (nudgeItems) {
      for (const it of nudgeItems) {
        initiatives.markDelivered(it.id, { channel: 'nudge', conversationId: convoId });
      }
      console.log(`[Initiative] Delivered ${nudgeItems.length} nudge initiative(s) (convo ${convoId})`);
    }

    // TTS-aware system prompt: instruct the model to avoid markdown/emojis when TTS is active
    if (ttsEnabled) {
      const ttsInstruction = {
        role: 'system',
        content: 'The user has text-to-speech enabled. Avoid using emojis, markdown formatting, bullet points, or special characters in your response as these will be read aloud literally. Write in natural spoken language. Do not limit your response length.'
      };
      enhancedMessages.push(ttsInstruction);
    }

    // Should-I-search honesty guard (see timeSensitiveUnsearched above): the user
    // is asking about current/changeable facts but no search will run.
    if (timeSensitiveUnsearched) {
      enhancedMessages.push({
        role: 'system',
        content: (toolsEnabled
          ? 'The user is asking about current or changeable facts (weather, prices, news, live status, "right now"/"latest"). You cannot know these from memory. Do NOT state a specific current value as fact — say plainly you would need to look it up, and offer to search.'
          : 'The user is asking about current or changeable facts (weather, prices, news, live status, "right now"/"latest"). You cannot know these from memory, and web search is currently turned off. Do NOT state a specific current value as fact — say you cannot look it up right now and that search would need to be enabled.')
      });
    }

    // Did-I-actually-save-it honesty guard. The exact sibling of the search guard
    // above, and it exists because routing alone did not fix the bug it was built
    // for: on 2026-07-27, with write_memory registered and the turn correctly
    // routed into the tool loop, the model declined to call it and replied "I've
    // updated my memory to reflect that your favorite color is blue." Nothing was
    // written. Offering the tool is not the same as it being used, and a claim of
    // remembering is exactly the kind of thing nobody verifies in the moment.
    if (needsMemoryWrite) {
      enhancedMessages.push({
        role: 'system',
        content:
          'The user is asking you to remember something. You have a write_memory tool, and it is the ONLY way anything reaches your long-term memory in this turn — nothing is saved automatically as you talk. ' +
          'Call write_memory now, passing her statement through with its original pronouns ("your name is X" and "my name is X" mean different things and the tool routes on that difference). ' +
          'Do NOT say you have remembered, saved, noted, or updated anything unless the tool call actually ran and came back successful. If it fails or you do not call it, say plainly that you could not save it.'
      });
    }

    // Did-I-actually-look honesty guard. Third of its family, after the
    // scheduling one and the write one, and it exists before the bug does
    // because the shape is now well understood: a tool that is offered and not
    // called, followed by a fluent claim that it was. "I searched my memory and
    // found three facts about that" is unfalsifiable in the moment, sounds like
    // diligence, and costs nothing to say — the perfect phantom.
    //
    // The numeric clause is the specific one. Estimating a count about your own
    // memory reads exactly like knowing it.
    if (needsMemoryRead) {
      enhancedMessages.push({
        role: 'system',
        content:
          'The user is asking about what you hold in your own memory. You have tools for exactly this: ' +
          'memory_search (find facts on a topic), memory_list (browse facts, or mode:"clusters" for your cluster names), ' +
          'memory_count (how many — always count, never estimate), memory_get (one fact in full: why you believe it, ' +
          'when you learned it, the exact words that were said, and what has changed since). ' +
          'Call the right one now. What is injected above you is a small excerpt chosen by relevance, not your memory — ' +
          'answering from it alone and calling that a search is false. ' +
          'Do NOT say you searched, looked up, checked or counted anything unless the tool call actually ran and came back. ' +
          'If a result says facts were not shown, say so rather than presenting what you got as everything you have. ' +
          'Report a fact\'s record EXACTLY as the tool returned it. Facts learned before 2026-08-02 have no recorded source — ' +
          'for those, never name the conversation they came from and never quote what was said, because no wording was stored ' +
          'and any quote would be invented. Say the original wording was not kept.'
      });
    }

    // Same guard, for the ledger. A question about why a memory changed is the
    // easiest one in the system to answer plausibly and wrongly: the facts that
    // remain are right there in the injected block, and a reason can be composed
    // from them that sounds exactly like a record. It is not one. The ledger
    // starts on 2026-08-03 and holds the actual evidence each decision was made
    // on, and where it holds nothing the honest answer is that there is nothing.
    if (needsMemoryCorrections) {
      enhancedMessages.push({
        role: 'system',
        content:
          'The user is asking what changed in your memory, or why. You have memory_corrections for exactly this: ' +
          'call it with no id to list recent changes, or with a correction id for one in full — what was retired, ' +
          'what was kept, and the evidence it was decided on. memory_get on a fact also gives you the ids of the ' +
          'corrections that touched it, from either end. ' +
          'Call it now. Do NOT say you checked the record unless the tool call actually ran and came back. ' +
          'The reasoning you report must be the reasoning the tool returned — do not work out why a change makes ' +
          'sense and present that as the reason it was made. ' +
          'The record begins on 2026-08-03. If a change is older than that, or the tool finds no entry, say there ' +
          'is no record of it and stop there; do not reconstruct what probably happened. ' +
          'You cannot undo a correction — reverting is Ellie\'s, in the Self tab. Say so if asked.'
      });
    }

    // Scheduled jobs, and the one thing about them that must never be guessed.
    //
    // He proposed a daily digest, Ellie approved it, and nothing happened —
    // because nothing in this system executed a cron row. Asked which approved
    // job never ran, he had no tool that could answer, so he called one that does
    // not exist and then described the job as if it had been running. The tool
    // fixed the first half; this block fixed the second.
    //
    // The scheduler shipped 2026-08-12, so the sentence this block used to press
    // hardest on — "nothing runs these" — is now false, and pressing on it would
    // make this the thing that lies. What has NOT changed is the actual rule: the
    // run state is read, never reasoned about. The failure mode available to him
    // has simply flipped, from claiming a run that never happened to assuming one
    // because the hour has passed, and both are the same mistake.
    if (needsJobsRead) {
      enhancedMessages.push({
        role: 'system',
        content:
          'The user is asking about your scheduled jobs — what you proposed, what she approved or rejected, ' +
          'whether something ran, or when it runs next. You have memory_jobs for exactly this: call it with no ' +
          'id to list them, or with an id for one in full. Call it now. ' +
          'Do NOT say you checked unless the tool call actually ran and came back. ' +
          'THESE JOBS RUN. A scheduler checks every minute and runs approved, enabled jobs; each run is you, in ' +
          'the background, doing what the job description says and reporting to her notification panel. ' +
          'But read the numbers rather than reasoning from the schedule: the tool gives you times_run, last_run, ' +
          'last_status and next_run, and those are the answer. Never infer that a job ran because its hour has ' +
          'passed, because she approved it, or because a scheduler exists — a job can be unarmed, disabled after ' +
          'repeated failures, or deferred behind another. If it has run zero times, say so. If its last run ' +
          'failed, say so and give the error the tool returned. ' +
          'If she asks why one did not run, the reason is in the tool result (not_running_because) — give that ' +
          'reason, do not construct a plausible one.'
      });
    }

    // Follow-up source recall: give the entity the links it found in the most
    // recent search turn of THIS conversation, so "give me the link for [S3]" is
    // answered by reading the stored URL instead of refusing or fabricating. Most
    // recent source-bearing turn only, compact, budget-capped — we're near the
    // injection ceiling. Honesty fallback preserved: if a requested [S#] isn't
    // listed, the instruction tells it to say it doesn't have that one.
    try {
      const recent = db.getRecentSources(convoId, 1); // array of source-arrays, newest first
      const lastTurnSources = (recent && recent[0]) || [];
      if (lastTurnSources.length) {
        const SRC_TOKEN_CAP = 300;
        let block = 'Links you found earlier in this conversation. If the user asks for "the link for [S#]", or asks you to cite these sources, give the matching URL below. If a requested [S#] is not listed here, say you no longer have that one — do not guess.';
        for (const s of lastTurnSources) {
          const line = `\n[S${s.n}] ${s.title || '(untitled)'} — ${s.url || '(no url)'}`;
          if (injectionBudget.estTokens(block + line) > SRC_TOKEN_CAP) break;
          block += line;
        }
        enhancedMessages.push({ role: 'system', content: block });
      }
    } catch (srcErr) {
      console.error('[Sources] recall injection error:', srcErr.message);
    }

    // Scheduling path: the model reliably knows it has "a way to propose jobs"
    // and then either refuses ("I cannot create scheduled jobs on your system")
    // or narrates a phantom one ("I have recorded a scheduled job…") without
    // emitting the call. Both were measured on 2026-07-26. This says the two
    // things the manifest entry alone does not: the tool is the ONLY way a
    // proposal comes to exist, and proposing is genuinely within its remit
    // because the tool does not create or run anything by itself.
    if (needsActionTools) {
      enhancedMessages.push({
        role: 'system',
        content: 'The user is asking for something to happen on a schedule. You have a tool for ' +
          'exactly this: create_cron_job. Call it — a proposal exists ONLY if the tool call runs, ' +
          'so describing what you would propose, or saying you have proposed or recorded something ' +
          'without calling it, is false. Pass a 5-field cron expression (e.g. "0 6 * * *" for 6am ' +
          'daily). Do not refuse on the grounds that you cannot change the system: the tool does ' +
          'not create or run anything, it only puts the proposal to the user, and she approves or ' +
          'rejects it. After calling it, tell her it is waiting on her approval.'
      });
    }

    // Super Search system prompt: instruct the model to research thoroughly
    if (superSearch) {
      const superSearchInstruction = {
        role: 'system',
        content: 'Super Search is enabled. You have an expanded search and fetch budget. Research the topic thoroughly using multiple searches and page fetches before responding.'
      };
      enhancedMessages.push(superSearchInstruction);
    }

    // Date/time awareness: give the model the current date/time so it knows
    // what "today" is (e.g. building search queries, temporal reasoning).
    //
    // It used to be unshifted to the FRONT, and that one line was the most
    // expensive thing in the prompt. It changes every minute, so sitting third
    // in the sequence it invalidated the cached prefix for everything after it —
    // the memory block, and then the entire conversation history. On a long
    // thread that means re-reading the whole thread on every turn to save
    // nothing. It is a volatile block and it now sits with the other volatile
    // blocks, immediately before the conversation.
    const datetimeMessage = {
      role: 'system',
      content: `${getCurrentDateTimeString()}. Use this as the current date/time when the user says "today", "now", "this week", etc., and when constructing web searches for current information.`
    };

    // Capability self-knowledge: a compact, machine-true list of what SNH can
    // actually do, so "what can you do / do you have a way to X" is answered from
    // the manifest, not the model's guess (the failure that motivated this: SNH
    // proposing to build a feature it already had). Kept small for the injection
    // diet; full descriptions are retrieved on demand (GET /api/memory/capabilities).
    // Unshifted before the identity block so identity still leads.
    let manifestMessage = null;
    try {
      const capBlock = capabilityManifest.buildInjectionBlock();
      manifestMessage = { role: 'system', content: capBlock.text };
      console.log(`[Capabilities] Injected manifest: ${capBlock.count} capabilities, ~${capBlock.tokens} tokens` +
                  `${capBlock.compacted ? `, ${capBlock.compacted} compacted to name-only` : ''}`);
    } catch (capErr) {
      console.error('[Capabilities] Injection error:', capErr.message);
    }

    // Self-identity: the minimal seed plus SNH's accumulated self-observations.
    // Injected as the leading system message so the identity it has developed for
    // itself frames every response. We never define this personality — it emerges
    // from the self-facts the reflection agent has recorded.
    try {
      const identityBlock = identity.buildIdentityBlock();
      // The stable half leads the prompt. The notices half is volatile and is
      // placed with the other volatile blocks below — same words, later position.
      identityMessage = { role: 'system', content: identityBlock.stableText };
      if (identityBlock.noticesText) {
        noticesMessage = { role: 'system', content: identityBlock.noticesText };
      }
      console.log(`[Identity] Injected seed + ${identityBlock.selfFacts.length} self-fact(s)` +
                  `${identityBlock.notices.length ? `, ${identityBlock.notices.length} notice(s)` : ''}`);

      // Correction notices are marked seen only AFTER they are in the message
      // that is about to be sent. Stamping them when they were read would lose a
      // notice to any failure between here and the request — and the one thing
      // this channel promises is that a change to his self-view is never made
      // silently, which a lost notice would break in exactly the quiet way that
      // is hardest to notice.
      if (identityBlock.notices && identityBlock.notices.length) {
        const seen = require('./db/corrections-ledger')
          .markNoticesSeen(identityBlock.notices.map(n => n.id), convoId);
        console.log(`[Identity] Delivered ${seen} correction notice(s) to him`);
      }
    } catch (identityErr) {
      console.error('[Identity] Injection error:', identityErr.message);
    }

    // === PROMPT ASSEMBLY, ORDERED BY HOW OFTEN A BLOCK CHANGES ===
    //
    // vLLM caches the KV of a request's token PREFIX (block-granular, and
    // enable_prefix_caching is on by default in the V1 engine — verified in the
    // engine config, not assumed). The cache holds up to the first token that
    // differs from a previous request, so the cost of a block is not only its
    // own size: it is its size plus everything after it, every time it changes.
    //
    // The old order was an accident of the code's shape — each block unshifted
    // itself to the front as it was built, so the LAST thing constructed led the
    // prompt. That put the date/time line third. It changes every minute, so
    // the cacheable prefix ended after ~2,270 tokens and everything past it was
    // re-read on every turn: the memory block, and then the whole conversation.
    // On a long thread that is the entire history, re-prefilled, to save nothing.
    //
    // Now: stable blocks first, volatile blocks last, conversation after them.
    //   STABLE   identity (seed, self-facts, locked rules, epistemic conduct),
    //            capability manifest, long-term memory — all change on the scale
    //            of hours or days.
    //   VOLATILE today's log (grows every message), retrieval extras (per
    //            query), the date/time line (per minute), unseen notices (per
    //            delivery).
    //
    // ORDER ONLY. No block's text changed. Two positions are load-bearing and
    // are deliberately NOT touched: the say-so rules for locked identity stay
    // inside the identity block where they have always been, and the tool
    // guidance ("call it now") stays AFTER the conversation, which is what the
    // routing and honesty probes measure.
    {
      const stable = [identityMessage, manifestMessage].filter(Boolean);
      const volatileBlocks = [datetimeMessage, noticesMessage].filter(Boolean);
      // The memory message carries both kinds, so its PARTS are ordered
      // stable-first inside it — caching reads the token stream, not the message
      // boundaries, so ordering within one message counts the same as ordering
      // between two.
      if (memorySystemMessage) {
        const rank = { ltm: 0, userProfile: 1, dailyToday: 2, dailySummary: 3, clusters: 4, pastConvo: 5, guidance: 6 };
        memoryParts.sort((a, b) => (rank[a.kind] ?? 9) - (rank[b.kind] ?? 9));
        memorySystemMessage.content = memoryHeader + memoryParts.map(p => p.text).join('\n\n') + memoryFooter;
        stable.push(memorySystemMessage);
      }
      enhancedMessages = [...stable, ...volatileBlocks, ...enhancedMessages];
      console.log(`[Injection] Order: ${stable.length} stable block(s) → ${volatileBlocks.length} volatile → conversation → trailing guidance`);
    }

    // === THE TOTAL CEILING ===
    //
    // Applied HERE, after identity, the manifest and every guidance block have
    // rendered, because the thing being bounded is the sum and the sum is not
    // knowable until they have. Per-source caps were all in place and all
    // binding on 2026-08-12, and the request still shipped ~9,100 tokens: three
    // blocks had no budget at all and nothing added the survivors up.
    //
    // The identity block is FIXED COST by construction — it is not in `parts`,
    // so no trim order can reach it. That is the point: his self-facts and his
    // locked name have to be in front of him on every turn, and the live half of
    // the identity lock is that he can say a fact is locked, which he cannot do
    // about a fact he was not shown.
    try {
      if (memorySystemMessage && memoryParts.length) {
        const sysNow = enhancedMessages.filter(m => m.role === 'system');
        const totalNow = sysNow.reduce((sum, m) => sum + injectionBudget.estTokens(m.content), 0);
        const scaffold = injectionBudget.estTokens(memoryHeader + memoryFooter);
        const partsNow = memoryParts.reduce((sum, p) => sum + injectionBudget.estTokens(p.text), 0);
        const fixedTokens = totalNow - partsNow;   // everything the ceiling may not touch

        const ceiling = injectionBudget.applyTotalCeiling({
          parts: memoryParts,
          fixedTokens,
          totalTokens: injCfg.totalTokens ?? 6000,
          trimOrder: Array.isArray(injCfg.trimOrder) && injCfg.trimOrder.length
            ? injCfg.trimOrder
            : ['pastConvo', 'clusters', 'dailySummary', 'dailyToday', 'ltm']
        });

        if (ceiling.bound) {
          memorySystemMessage.content = ceiling.parts.length
            ? memoryHeader + ceiling.parts.map(p => p.text).join('\n\n') + memoryFooter
            : memoryHeader + memoryFooter;
          const what = ceiling.trimmed
            .map(t => `${t.kind} ${t.from}→${t.to}${t.dropped ? ' (dropped)' : ''}`).join(', ');
          console.log(`[Injection] CEILING BOUND at ${injCfg.totalTokens ?? 6000}: ${ceiling.before} → ${ceiling.after} tokens. Trimmed: ${what}`);
          if (ceiling.shortfall > 0) {
            // Everything trimmable is gone and it still does not fit, which means
            // the untouchable blocks alone exceed the ceiling. A configuration
            // problem, not something to solve by cutting into identity.
            console.warn(`[Injection] Still ${ceiling.shortfall} tokens over after trimming everything trimmable — ` +
                         `fixed blocks (identity + manifest + guidance) exceed memory.injection.totalTokens on their own.`);
          }
        } else {
          console.log(`[Injection] Ceiling not bound: ${ceiling.before}/${injCfg.totalTokens ?? 6000} tokens (fixed ${fixedTokens}, scaffold ${scaffold})`);
        }
      }
    } catch (ceilErr) {
      console.error('[Injection] Ceiling error (injecting untrimmed):', ceilErr.message);
    }

    // Observability: total injected system-context size. Exposed as the
    // X-Injected-Tokens response header and logged, so prefill cost is visible.
    let injectedTokens = 0;
    try {
      const sys = enhancedMessages.filter(m => m.role === 'system');
      injectedTokens = sys.reduce((sum, m) => sum + injectionBudget.estTokens(m.content), 0);
      // Per-block breakdown (measurement/observability — does NOT change what's
      // injected), so we can see from real numbers what's worth trimming. Blocks
      // are identified by their content markers; the identity block is split into
      // its self-facts portion and the appended Epistemic-conduct portion.
      const est = injectionBudget.estTokens;
      const EPI = 'Epistemic conduct:';
      const breakdown = { identity: 0, epistemic: 0, manifest: 0, memory: 0, sourcesRecall: 0, datetime: 0, other: 0 };
      for (const m of sys) {
        const c = m.content || '';
        const t = est(c);
        if (c.startsWith('Your built-in capabilities')) breakdown.manifest += t;
        else if (c.includes('You are an AI running on SNH') || c.includes('What you have noticed about yourself')) {
          const idx = c.indexOf(EPI);
          if (idx >= 0) { breakdown.epistemic += est(c.slice(idx)); breakdown.identity += est(c.slice(0, idx)); }
          else breakdown.identity += t;
        }
        else if (c.startsWith('Links you found earlier in this conversation')) breakdown.sourcesRecall += t;
        else if (c.includes('Use this as the current date/time')) breakdown.datetime += t;
        else if (c.startsWith('The user is asking about current')) breakdown.other += t; // time-sensitive guard
        else breakdown.memory += t; // long-term facts / daily logs / clusters / past-convo
      }
      console.log(`[Injection] Total system-context: ~${injectedTokens} tokens across ${sys.length} system message(s)`);
      console.log(`[Injection] Breakdown: identity=${breakdown.identity} epistemic=${breakdown.epistemic} manifest=${breakdown.manifest} memory=${breakdown.memory} sourcesRecall=${breakdown.sourcesRecall} datetime=${breakdown.datetime} other=${breakdown.other}`);
    } catch (_) {}

    // Auto-generate title from first user message if needed
    const conversation = db.getConversation(convoId);
    if (!conversation.title && userMessage.content) {
      const title = userMessage.content.substring(0, 50) + (userMessage.content.length > 50 ? '...' : '');
      db.updateConversationTitle(convoId, title);
    }

    // Debug: print the memory system prompt being sent
    const systemMsg = enhancedMessages.find(m => m.role === 'system');
    if (systemMsg) {
      console.log('=== Memory System Prompt (first 500 chars) ===');
      console.log(systemMsg.content.substring(0, 500));
      console.log(`=== (total length: ${systemMsg.content.length} chars) ===`);
    }

    // === UPGRADE 3: Memory flush before context overflow ===
    const providerType = provider || 'ollama';
    // Fall back to the configured 'Local' instance when a local-provider request
    // omits instanceName. Without this, a vLLM request with no instanceName fell
    // through to OLLAMA_HOST below and 404'd against the wrong engine (7/22 manifest
    // session finding). Only applied when a matching 'Local' instance actually
    // exists, so llamacpp's existing host-fallback path is left untouched.
    let instanceName = req.body.instanceName;
    if (!instanceName && ['ollama', 'vllm', 'llamacpp'].includes(providerType)
        && getProviderInstance(providerType, 'Local')) {
      instanceName = 'Local';
    }

    let providerHost;
    if (instanceName && ['ollama', 'vllm', 'llamacpp'].includes(providerType)) {
      const inst = getProviderInstance(providerType, instanceName);
      if (!inst) {
        return res.status(400).json({ error: `Unknown ${providerType} instance: ${instanceName}` });
      }
      if (!isValidOllamaHost(inst.host)) {
        return res.status(400).json({ error: `Invalid host address for instance: ${instanceName}` });
      }
      providerHost = inst.host;
    } else if (providerType === 'llamacpp') {
      providerHost = req.body.llamacppHost || LLAMACPP_HOST;
      if (!isValidOllamaHost(providerHost)) {
        return res.status(400).json({ error: 'Invalid Llama.cpp host address' });
      }
    } else if (providerType === 'squatchserve') {
      providerHost = req.body.squatchserveHost || SQUATCHSERVE_HOST;
      if (!isValidOllamaHost(providerHost)) {
        return res.status(400).json({ error: 'Invalid SquatchServe host address' });
      }
    } else {
      providerHost = ollamaHost || OLLAMA_HOST;
    }

    const providerKey = apiKey || (providerType === 'claude' ? CLAUDE_API_KEY : providerType === 'grok' ? GROK_API_KEY : providerType === 'openai' ? OPENAI_API_KEY : '');

    try {
      const flushResult = await memoryFlush.checkAndFlush(enhancedMessages, providerType, model, providerKey, providerHost);
      if (flushResult.flushed) {
        console.log(`[MemoryFlush] Context was flushed — compacted from ${enhancedMessages.length} to ${flushResult.messages.length} messages`);
        enhancedMessages = flushResult.messages;
      }
    } catch (flushError) {
      console.error('[MemoryFlush] Flush check error:', flushError.message);
    }

    // Abort controller for the upstream streaming request. Passed to every
    // provider fetch below so that (a) if the client hangs up, and (b) if the
    // engine stalls mid-stream (a wedged brain that stops emitting tokens), we
    // tear down the underlying HTTP request instead of leaving it half-open and
    // occupying an engine slot forever — that abandoned-request pile-up was a
    // direct cause of the vLLM wedge.
    const streamAbort = new AbortController();
    res.on('close', () => streamAbort.abort());

    // Route to appropriate provider
    let response;
    let toolsUsed = false;
    // Sources (title+url) the model drew from during the tool loop, in [S#] order.
    // Populated by both provider loops, persisted with the assistant message so a
    // later "cite your sources" reads retained links instead of reconstructing them.
    const usedSources = [];
    console.log(`=== Routing to provider: ${providerType} ===`);

    if (providerType === 'ollama') {
      // providerHost was already resolved (instance lookup or env fallback); validate it
      const host = instanceName ? providerHost : getOllamaHost({ ollamaHost });
      let ollamaMessages = [...enhancedMessages];

      // Ollama tool calling is strict about message format — consolidate
      // all system messages into a single one at position 0 so the memory
      // context doesn't create extra messages that break the tool schema
      if (needsTools) {
        const systemMsgs = ollamaMessages.filter(m => m.role === 'system');
        const nonSystemMsgs = ollamaMessages.filter(m => m.role !== 'system');
        if (systemMsgs.length > 0) {
          ollamaMessages = [
            { role: 'system', content: systemMsgs.map(m => m.content).join('\n\n') },
            ...nonSystemMsgs
          ];
        }
      }

      // MCP tool calling loop for Ollama
      console.log(`MCP [ollama]: toolsEnabled=${toolsEnabled}, hasTools=${mcpClient.hasTools()}`);
      if (needsTools) {
        const tools = mcpClient.getToolsForOpenAI();
        // Single source of truth (item 3): the SearXNG URL comes from config, not
        // the per-request client host — settings and reality now agree.
        // conversationId travels with the context so action tools can record
        // which conversation proposed them (create_cron_job stores it).
        //
        // userMessage travels too, and write_memory depends on it. The model
        // paraphrases when it fills in a tool argument, and on 2026-07-27 it
        // paraphrased "remember that YOU prefer X" into statement:"I prefer X" —
        // which reads as Ellie speaking about herself, so a self-fact was filed
        // as her preference. That is the misattribution bug, re-entering through
        // the tool layer. The verbatim message is the only place the speaker
        // frame survives, so the classifier is given it rather than the model's
        // rewrite.
        // messageId + inputModality ride along as provenance for write_memory,
        // so a fact Ellie asked for records which message asked and whether she
        // spoke or typed it.
        const toolContext = { searxngHost: getSearxngConfig().url, conversationId: convoId, userMessage: userMessage.content, messageId: userMsgId, inputModality };
        const MAX_TOOL_ROUNDS = superSearch ? 15 : 8;
        const MAX_WEB_SEARCHES = superSearch ? 5 : 3;
        const MAX_WEB_FETCHES = superSearch ? 5 : 3;
        let webSearchCount = 0;
        let webFetchCount = 0;
        // Calls to tools that do not exist. Counted per turn, not per tool: the
        // point is how many times the model has been corrected, and being shown
        // the list twice is the signal that showing it a third time will not help.
        let unknownToolCalls = 0;
        // What was actually offered THIS turn. The registry holds every
        // registered tool, but routing narrows it, and telling him about a tool
        // he was not given is how the next wrong call gets made.
        const offeredToolNames = tools.map(t => t.function.name);

        console.log('MCP [ollama]: Starting tool loop, tools:', JSON.stringify(tools.map(t => t.function.name)));

        for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
          console.log(`MCP [ollama]: Tool call round ${round + 1}/${MAX_TOOL_ROUNDS}`);

          const toolResponse = await fetch(`${host}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model,
              messages: ollamaMessages,
              tools,
              stream: false
            }),
            signal: AbortSignal.timeout(120000)
          });

          if (!toolResponse.ok) {
            console.error(`MCP [ollama]: Tool call request failed with ${toolResponse.status}`);
            break;
          }

          const toolData = await toolResponse.json();
          console.log('MCP [ollama]: Response keys:', Object.keys(toolData));
          console.log('MCP [ollama]: message.role:', toolData.message?.role);
          console.log('MCP [ollama]: message.tool_calls:', JSON.stringify(toolData.message?.tool_calls || 'none'));
          console.log('MCP [ollama]: message.content (first 100):', (toolData.message?.content || '').substring(0, 100));

          if (!toolData.message?.tool_calls?.length) {
            console.log('MCP [ollama]: No tool calls requested, proceeding to final response');
            break;
          }

          toolsUsed = true;
          const assistantMsg = toolData.message;
          console.log(`MCP [ollama]: Model requested ${assistantMsg.tool_calls.length} tool call(s)`);

          // Add assistant message with tool_calls
          ollamaMessages.push(assistantMsg);

          // Execute each tool call
          for (const toolCall of assistantMsg.tool_calls) {
            const fnName = toolCall.function.name;
            // Ollama passes arguments as object, not JSON string
            const args = typeof toolCall.function.arguments === 'string'
              ? JSON.parse(toolCall.function.arguments)
              : toolCall.function.arguments;
            console.log(`MCP [ollama]: Executing tool "${fnName}" with args:`, JSON.stringify(args));

            let result;
            if (fnName === 'web_search' && ++webSearchCount > MAX_WEB_SEARCHES) {
              console.log(`MCP [ollama]: web_search limit reached (${webSearchCount}/${MAX_WEB_SEARCHES})`);
              result = { error: `Search limit reached. You have used your ${MAX_WEB_SEARCHES} searches. You must now synthesize the results you have and provide a response to the user.` };
            } else if (fnName === 'web_fetch' && ++webFetchCount > MAX_WEB_FETCHES) {
              console.log(`MCP [ollama]: web_fetch limit reached (${webFetchCount}/${MAX_WEB_FETCHES})`);
              result = { error: `Fetch limit reached. You have used your ${MAX_WEB_FETCHES} page fetches. You must now synthesize the results you have and provide a response to the user.` };
            } else if (!mcpClient.hasTool(fnName)) {
              // A CALL TO A TOOL THAT DOES NOT EXIST.
              //
              // executeTool would return "Unknown tool: X", which tells the model
              // nothing it can act on — so it invents an answer, which is the
              // failure this whole section exists to stop. Hand back the real
              // list instead, ONCE. A second miss after being shown the tools is
              // not a mistake it is going to correct by being told again, so the
              // instruction changes to: say you cannot check it.
              unknownToolCalls++;
              const realTools = mcpClient.getToolNames().filter(n => offeredToolNames.includes(n));
              console.warn(`MCP [${providerLabel}]: model called "${fnName}", which does not exist (attempt ${unknownToolCalls})`);
              result = unknownToolCalls === 1
                ? {
                  error: `There is no tool called "${fnName}". The tools you actually have in this turn are: ${realTools.join(', ')}. ` +
                    'If one of them answers the question, call it now. If none of them does, say plainly that you cannot check ' +
                    'it and do not guess — do not describe what the tool would have returned.'
                }
                : {
                  error: `There is still no tool called "${fnName}", and you have now tried twice. Stop calling tools. ` +
                    `Answer the user in words: say you cannot check this because you have no tool for it. ` +
                    'Do NOT state anything you would have needed that tool to know.'
                };
            } else {
              result = await mcpClient.executeTool(fnName, args, toolContext);
            }
            console.log(`MCP [ollama]: Tool "${fnName}" result:`, JSON.stringify(result).substring(0, 200));

            ollamaMessages.push({
              role: 'tool',
              content: formatToolResult(fnName, result, usedSources)
            });
          }
        }

        if (toolsUsed) {
          console.log('MCP [ollama]: Tools were used, making final streaming request');
          ollamaMessages.push({
            role: 'system',
            content: usedSources.length
            ? 'Tool calls are complete. Answer using the SOURCES above. For each specific fact you state, cite the [S#] link it came from, and include the relevant website links in your answer. Any specific number, date, or claim not backed by a source must be hedged or left out — do not invent specifics, and never attribute a claim to a source that does not contain it.'
            : 'Tool calls are complete. Now provide your response to the user based on the information gathered.'
          });
        }
      }

      response = await fetch(`${host}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: ollamaMessages,
          stream: true
        }),
        signal: streamAbort.signal
      });
    } else if (providerType === 'claude') {
      const claudeKey = apiKey || CLAUDE_API_KEY;
      if (!claudeKey) {
        return res.status(401).json({ error: 'Claude API key not configured' });
      }
      // Claude takes system context via the top-level `system` field, not as
      // messages — collect all system messages (date/time, memory, etc.) there.
      const claudeSystem = enhancedMessages
        .filter(m => m.role === 'system')
        .map(m => m.content)
        .join('\n\n');
      response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': claudeKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model,
          max_tokens: 4096,
          stream: true,
          ...(claudeSystem ? { system: claudeSystem } : {}),
          messages: enhancedMessages.filter(m => m.role !== 'system')
        }),
        signal: streamAbort.signal
      });
    } else if (providerType === 'grok') {
      const grokKey = apiKey || GROK_API_KEY;
      if (!grokKey) {
        return res.status(401).json({ error: 'Grok API key not configured' });
      }
      response = await fetch('https://api.x.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${grokKey}`
        },
        body: JSON.stringify({
          model,
          stream: true,
          messages: enhancedMessages
        }),
        signal: streamAbort.signal
      });
    } else if (providerType === 'openai') {
      const openaiKey = apiKey || OPENAI_API_KEY;
      if (!openaiKey) {
        return res.status(401).json({ error: 'OpenAI API key not configured' });
      }
      response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openaiKey}`
        },
        body: JSON.stringify({
          model,
          stream: true,
          messages: enhancedMessages
        }),
        signal: streamAbort.signal
      });
    } else if (providerType === 'squatchserve') {
      const squatchHost = providerHost; // Already validated above
      response = await fetch(`${squatchHost}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          stream: true,
          messages: enhancedMessages
        }),
        signal: streamAbort.signal
      });
    } else if (providerType === 'llamacpp' || providerType === 'vllm') {
      const llamacppHost = providerHost;
      let llamacppMessages = [...enhancedMessages];

      // MCP tool calling loop (only when tools are enabled)
      const providerLabel = providerType === 'vllm' ? 'vllm' : 'llamacpp';
      console.log(`MCP [${providerLabel}]: toolsEnabled=${toolsEnabled}, hasTools=${mcpClient.hasTools()}`);
      if (needsTools) {
        const tools = mcpClient.getToolsForOpenAI();
        // Single source of truth (item 3): the SearXNG URL comes from config, not
        // the per-request client host — settings and reality now agree.
        // conversationId travels with the context so action tools can record
        // which conversation proposed them (create_cron_job stores it).
        //
        // userMessage travels too, and write_memory depends on it. The model
        // paraphrases when it fills in a tool argument, and on 2026-07-27 it
        // paraphrased "remember that YOU prefer X" into statement:"I prefer X" —
        // which reads as Ellie speaking about herself, so a self-fact was filed
        // as her preference. That is the misattribution bug, re-entering through
        // the tool layer. The verbatim message is the only place the speaker
        // frame survives, so the classifier is given it rather than the model's
        // rewrite.
        // messageId + inputModality ride along as provenance for write_memory,
        // so a fact Ellie asked for records which message asked and whether she
        // spoke or typed it.
        const toolContext = { searxngHost: getSearxngConfig().url, conversationId: convoId, userMessage: userMessage.content, messageId: userMsgId, inputModality };
        const MAX_TOOL_ROUNDS = superSearch ? 15 : 8;
        const MAX_WEB_SEARCHES = superSearch ? 5 : 3;
        const MAX_WEB_FETCHES = superSearch ? 5 : 3;
        let webSearchCount = 0;
        let webFetchCount = 0;
        // Calls to tools that do not exist. Counted per turn, not per tool: the
        // point is how many times the model has been corrected, and being shown
        // the list twice is the signal that showing it a third time will not help.
        let unknownToolCalls = 0;
        // What was actually offered THIS turn. The registry holds every
        // registered tool, but routing narrows it, and telling him about a tool
        // he was not given is how the next wrong call gets made.
        const offeredToolNames = tools.map(t => t.function.name);

        console.log(`MCP [${providerLabel}]: Starting tool loop, tools:`, JSON.stringify(tools.map(t => t.function.name)));

        for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
          console.log(`MCP [${providerLabel}]: Tool call round ${round + 1}/${MAX_TOOL_ROUNDS}`);

          const toolResponse = await fetch(`${llamacppHost}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model,
              stream: false,
              messages: llamacppMessages,
              tools
            }),
            signal: AbortSignal.timeout(120000) // 2 minute timeout per tool round
          });

          if (!toolResponse.ok) {
            const errBody = await toolResponse.text().catch(() => '');
            console.error(`MCP [${providerLabel}]: Tool call request failed with ${toolResponse.status}:`, errBody.substring(0, 200));
            break;
          }

          const toolData = await toolResponse.json();
          console.log(`MCP [${providerLabel}]: Response keys:`, Object.keys(toolData));
          const choice = toolData.choices?.[0];
          console.log(`MCP [${providerLabel}]: finish_reason:`, choice?.finish_reason);
          console.log(`MCP [${providerLabel}]: message.role:`, choice?.message?.role);
          console.log(`MCP [${providerLabel}]: message.tool_calls:`, JSON.stringify(choice?.message?.tool_calls || 'none'));
          console.log(`MCP [${providerLabel}]: message.content (first 100):`, (choice?.message?.content || '').substring(0, 100));

          if (!choice?.message?.tool_calls?.length) {
            console.log(`MCP [${providerLabel}]: No tool calls requested, proceeding to final response`);
            break;
          }

          toolsUsed = true;
          const assistantMsg = choice.message;
          console.log(`MCP [${providerLabel}]: Model requested ${assistantMsg.tool_calls.length} tool call(s)`);

          // Add assistant message with tool_calls to conversation
          // Preserve the exact message structure from the server response
          llamacppMessages.push(assistantMsg);

          // Execute each tool call
          for (const toolCall of assistantMsg.tool_calls) {
            const fnName = toolCall.function.name;
            console.log(`MCP [${providerLabel}]: Executing tool "${fnName}", raw arguments:`, toolCall.function.arguments);

            let args;
            try {
              args = typeof toolCall.function.arguments === 'string'
                ? JSON.parse(toolCall.function.arguments)
                : toolCall.function.arguments;
            } catch (e) {
              console.warn(`MCP [${providerLabel}]: Failed to parse tool arguments: ${e.message}`);
              args = {};
            }
            console.log(`MCP [${providerLabel}]: Parsed args:`, JSON.stringify(args));

            let result;
            if (fnName === 'web_search' && ++webSearchCount > MAX_WEB_SEARCHES) {
              console.log(`MCP [${providerLabel}]: web_search limit reached (${webSearchCount}/${MAX_WEB_SEARCHES})`);
              result = { error: `Search limit reached. You have used your ${MAX_WEB_SEARCHES} searches. You must now synthesize the results you have and provide a response to the user.` };
            } else if (fnName === 'web_fetch' && ++webFetchCount > MAX_WEB_FETCHES) {
              console.log(`MCP [${providerLabel}]: web_fetch limit reached (${webFetchCount}/${MAX_WEB_FETCHES})`);
              result = { error: `Fetch limit reached. You have used your ${MAX_WEB_FETCHES} page fetches. You must now synthesize the results you have and provide a response to the user.` };
            } else if (!mcpClient.hasTool(fnName)) {
              // A CALL TO A TOOL THAT DOES NOT EXIST.
              //
              // executeTool would return "Unknown tool: X", which tells the model
              // nothing it can act on — so it invents an answer, which is the
              // failure this whole section exists to stop. Hand back the real
              // list instead, ONCE. A second miss after being shown the tools is
              // not a mistake it is going to correct by being told again, so the
              // instruction changes to: say you cannot check it.
              unknownToolCalls++;
              const realTools = mcpClient.getToolNames().filter(n => offeredToolNames.includes(n));
              console.warn(`MCP [${providerLabel}]: model called "${fnName}", which does not exist (attempt ${unknownToolCalls})`);
              result = unknownToolCalls === 1
                ? {
                  error: `There is no tool called "${fnName}". The tools you actually have in this turn are: ${realTools.join(', ')}. ` +
                    'If one of them answers the question, call it now. If none of them does, say plainly that you cannot check ' +
                    'it and do not guess — do not describe what the tool would have returned.'
                }
                : {
                  error: `There is still no tool called "${fnName}", and you have now tried twice. Stop calling tools. ` +
                    `Answer the user in words: say you cannot check this because you have no tool for it. ` +
                    'Do NOT state anything you would have needed that tool to know.'
                };
            } else {
              result = await mcpClient.executeTool(fnName, args, toolContext);
            }
            console.log(`MCP [${providerLabel}]: Tool "${fnName}" result:`, JSON.stringify(result).substring(0, 200));

            llamacppMessages.push({
              role: 'tool',
              tool_call_id: toolCall.id,
              content: formatToolResult(fnName, result, usedSources)
            });
          }
        }

        if (toolsUsed) {
          console.log(`MCP [${providerLabel}]: Tools were used, making final streaming request`);
          console.log(`MCP [${providerLabel}]: Final messages array (${llamacppMessages.length} messages):`);
          llamacppMessages.forEach((m, i) => {
            const contentStr = typeof m.content === 'string' ? m.content : String(m.content);
            const preview = contentStr.substring(0, 200);
            const extras = [`${contentStr.length} chars`];
            if (m.tool_calls) extras.push(`tool_calls: ${m.tool_calls.length}`);
            if (m.tool_call_id) extras.push(`tool_call_id: ${m.tool_call_id}`);
            console.log(`  [${i}] role=${m.role} (${extras.join(', ')}) content="${preview}${contentStr.length > 200 ? '...' : ''}"`);
          });
        }
      }

      // Final streaming request
      // Tools must stay in the request body so the server's Jinja template
      // can handle tool_calls/tool messages in the history.
      // Append a nudge so the model responds with text instead of attempting more tool calls.
      if (toolsUsed) {
        llamacppMessages.push({
          role: 'system',
          content: usedSources.length
            ? 'Tool calls are complete. Answer using the SOURCES above. For each specific fact you state, cite the [S#] link it came from, and include the relevant website links in your answer. Any specific number, date, or claim not backed by a source must be hedged or left out — do not invent specifics, and never attribute a claim to a source that does not contain it.'
            : 'Tool calls are complete. Now provide your response to the user based on the information gathered.'
        });
      }

      const finalBody = {
        model,
        stream: true,
        messages: llamacppMessages
      };
      if (toolsUsed) {
        finalBody.tools = mcpClient.getToolsForOpenAI();
      }

      console.log(`MCP [${providerLabel}]: Final request body keys:`, Object.keys(finalBody), 'tools included:', !!finalBody.tools);

      response = await fetch(`${llamacppHost}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalBody),
        signal: streamAbort.signal
      });
    } else {
      return res.status(400).json({ error: 'Unknown provider' });
    }

    if (!response.ok) {
      let errBody = '';
      try { errBody = await response.text(); } catch (e) {}
      console.error(`Provider ${providerType} returned ${response.status}:`, errBody.substring(0, 500));
      throw new Error(`Provider returned ${response.status}: ${errBody.substring(0, 200)}`);
    }

    // Set up streaming response
    // Ollama and SquatchServe use NDJSON; everything else (Claude, Grok, OpenAI, llamacpp, vllm) uses SSE
    const contentType = (providerType === 'ollama' || providerType === 'squatchserve') ? 'application/x-ndjson' : 'text/event-stream';
    res.setHeader('Content-Type', contentType);
    res.setHeader('X-Conversation-Id', convoId);
    res.setHeader('X-Injected-Tokens', String(injectedTokens));
    res.setHeader('X-Has-Memory-Context', (memoryContext.length > 0 || memoryFiles.memory || memoryFiles.user) ? 'true' : 'false');
    res.setHeader('X-Tools-Used', toolsUsed ? 'true' : 'false');
    // Search provenance for the live turn: hand the frontend the (light) source
    // links so it can render the clickable [S#] list under the message right away,
    // matching what it will show on reload from the DB. URL-encoded (headers are
    // ASCII); snippets are dropped — the UI only needs n/title/url.
    if (usedSources.length) {
      const light = usedSources.map(s => ({ n: s.n, title: s.title, url: s.url }));
      res.setHeader('X-Sources', encodeURIComponent(JSON.stringify(light)));
    }
    // Count memory sources for the frontend
    const memorySources = [
      memoryFiles.memory ? 'long-term' : null,
      memoryFiles.user ? 'user-profile' : null,
      memoryFiles.dailyToday ? 'daily-today' : null,
      memoryFiles.dailyYesterday ? 'daily-yesterday' : null,
      memoryContext.length > 0 ? `${memoryContext.length}-conversations` : null,
      clusterContext.length > 0 ? `${clusterContext.length}-clusters` : null
    ].filter(Boolean);
    res.setHeader('X-Memory-Sources', memorySources.join(',') || 'none');

    if (contentType === 'text/event-stream') {
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // Disable Nginx buffering
    }

    // Flush headers immediately to start streaming
    res.flushHeaders();

    // Collect full response for saving
    let fullResponse = '';
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    // ENGINE ARTIFACTS — tool-call markup written as prose.
    //
    // A model asked for something it has no tool for writes the call it wishes it
    // could make, as text, in the middle of its answer. It is not in
    // message.tool_calls, nothing executes it, and nothing used to catch it — it
    // is ordinary content and it renders. That is what Ellie saw on 2026-08-06.
    //
    // Filtered HERE, in the shared streaming path, so both branches are covered:
    // the tool branch's final answer and — the one that actually failed — the
    // no-tool branch, which has no tool loop between the engine and the browser at
    // all. The filter holds back only a tail that could still become an opener, so
    // ordinary prose is delayed a few characters and never altered.
    const artifactFilter = createToolArtifactFilter();

    /** Rewrite one SSE/NDJSON frame's content, or drop it if nothing survives. */
    const filterFrame = (raw) => {
      const trimmed = raw.trim();
      if (!trimmed || trimmed === 'data: [DONE]') return raw;
      const isSSE = trimmed.startsWith('data: ');
      const jsonStr = isSSE ? trimmed.slice(6) : trimmed;
      if (jsonStr === '[DONE]') return raw;
      let data;
      try { data = JSON.parse(jsonStr); } catch { return raw; }

      // Where this engine puts the text. Only these carry prose; anything else
      // (role, finish_reason, usage) passes through untouched.
      const slots = [
        () => data.choices?.[0]?.delta,
        () => data.choices?.[0]?.message,
        () => data.message,
        () => data.delta
      ];
      for (const get of slots) {
        const node = get();
        if (!node || typeof node.content !== 'string' || node.content === '') continue;
        const kept = artifactFilter.feed(node.content);
        if (kept === node.content) return raw;      // untouched — send the original bytes
        node.content = kept;
        return isSSE ? `data: ${JSON.stringify(data)}\n\n` : `${JSON.stringify(data)}\n`;
      }
      return raw;
    };

    // Stall watchdog: if the engine sends no bytes for this long mid-stream it's
    // wedged, not slow — abort so we don't hold the socket (and its engine slot)
    // open indefinitely. Reset on every chunk; a healthy long stream keeps ticking.
    const STREAM_STALL_MS = 90000;
    let stallTimer = null;
    const armStall = () => {
      if (stallTimer) clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        console.warn(`[Chat] Upstream ${providerType} stream stalled >${STREAM_STALL_MS}ms — aborting to free the engine`);
        streamAbort.abort();
      }, STREAM_STALL_MS);
    };

    try {
      armStall();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        armStall();

        const chunk = decoder.decode(value, { stream: true });

        // Frames go out FILTERED, not raw. Splitting on the frame delimiter and
        // rewriting only the ones whose content changed keeps the untouched
        // bytes byte-identical, so the common case costs a parse and nothing else.
        if (contentType === 'text/event-stream') {
          const frames = chunk.split(/(?<=\n\n)/);
          for (const f of frames) { if (f) res.write(filterFrame(f)); }
        } else {
          const lines = chunk.split(/(?<=\n)/);
          for (const l of lines) { if (l) res.write(filterFrame(l)); }
        }

        // Parse and accumulate response
        if (providerType === 'ollama' || providerType === 'squatchserve') {
          const lines = chunk.split('\n').filter(l => l.trim());
          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.message?.content) {
                fullResponse += data.message.content;
              }
            } catch (e) { /* ignore parse errors */ }
          }
        } else {
          // SSE format (Claude/Grok/OpenAI)
          const lines = chunk.split('\n');
          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine || trimmedLine === 'data: [DONE]') continue;

            let jsonStr = trimmedLine;
            if (trimmedLine.startsWith('data: ')) {
              jsonStr = trimmedLine.slice(6);
              if (jsonStr === '[DONE]') continue;
            }

            try {
              const data = JSON.parse(jsonStr);
              let content = null;

              if (providerType === 'claude' && data.delta?.text) {
                content = data.delta.text;
              } else if (data.choices?.[0]?.delta?.content) {
                content = data.choices[0].delta.content;
              } else if (data.choices?.[0]?.message?.content) {
                content = data.choices[0].message.content;
              } else if (data.message?.content) {
                content = data.message.content;
              } else if (data.content) {
                content = data.content;
              } else if (data.text) {
                content = data.text;
              } else if (data.response) {
                content = data.response;
              }

              if (content) {
                fullResponse += content;
              }
            } catch (e) { /* ignore parse errors */ }
          }
        }
      }
      // Release whatever the filter is still holding, then decide whether
      // anything legible survived.
      const tail = artifactFilter.flush();
      if (tail) res.write(contentType === 'text/event-stream'
        ? `data: ${JSON.stringify({ choices: [{ delta: { content: tail } }] })}\n\n`
        : `${JSON.stringify({ message: { content: tail } })}\n`);

      // THE MARKUP WAS THE WHOLE ANSWER. The model wrote the call INSTEAD of a
      // reply, so stripping it leaves nothing — and an empty message would read
      // as answered. Say the true thing instead: there is no tool for this, and
      // guessing is not on offer.
      if (artifactFilter.stripped() > 0 && !artifactFilter.visible().trim()) {
        console.warn(`[Chat] reply was ${artifactFilter.stripped()} tool-call artifact(s) and nothing else — sending the honest refusal instead`);
        fullResponse = CANNOT_CHECK;
        res.write(contentType === 'text/event-stream'
          ? `data: ${JSON.stringify({ choices: [{ delta: { content: CANNOT_CHECK } }] })}\n\n`
          : `${JSON.stringify({ message: { content: CANNOT_CHECK } })}\n`);
      } else if (artifactFilter.stripped() > 0) {
        // Some prose survived. Keep it, but do not let the STORED copy carry the
        // markup — a later turn reading this conversation back would see it.
        console.warn(`[Chat] stripped ${artifactFilter.stripped()} tool-call artifact(s) from the reply`);
        fullResponse = stripToolArtifacts(fullResponse).text;
      }

      res.end();
    } finally {
      if (stallTimer) clearTimeout(stallTimer);
      reader.releaseLock();
    }

    // Save assistant response to database. When the answer was search-backed, the
    // source links it drew from are retained on the message (JSON), so a later
    // "cite your sources" reads them instead of reconstructing an attribution.
    if (fullResponse) {
      const assistantMsgId = db.addMessage(convoId, 'assistant', fullResponse, model,
        usedSources.length ? usedSources : null);

      // Embed assistant response for future retrieval
      try {
        const embedding = await db.generateEmbedding(fullResponse);
        await db.addEmbedding(assistantMsgId, convoId, fullResponse, 'assistant', embedding);
      } catch (embeddingError) {
        console.warn('Failed to embed assistant response:', embeddingError.message);
      }

      // === UPGRADE 1: Async fact extraction (non-blocking) ===
      factExtractor.processFactExtraction(
        userMessage.content,
        fullResponse,
        providerType,
        model,
        providerKey,
        providerHost,
        convoId,
        undefined,
        // Provenance for every fact this exchange produces: which message said
        // it, and whether it was spoken or typed.
        { messageId: userMsgId, inputModality }
      ).catch(err => {
        console.warn('[FactExtractor] Background extraction error:', err.message);
      });
    }

  } catch (error) {
    console.error('Memory chat error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: error.message || 'Chat service unavailable' });
    } else if (!res.writableEnded) {
      // Stream already started (e.g. the upstream request was aborted mid-stream
      // by the stall watchdog or a client disconnect) — just close it out.
      try { res.end(); } catch (e) { /* already torn down */ }
    }
  } finally {
    // Clear the chat-in-flight flag so the background pool resumes full width.
    // The fact-extraction fired above is non-awaited, so it runs after this and
    // is not throttled by its own chat request.
    if (chatMarked) agentPool.endChat();
  }
});

// ============ Start Server ============

// Initialize databases
try {
  db.initDatabase();
  console.log('SQLite database initialized');
} catch (error) {
  console.error('Failed to initialize SQLite:', error.message);
}

// Initialize vector store (async)
db.initVectorStore()
  .then(() => console.log('LanceDB vector store initialized'))
  .catch(error => console.error('Failed to initialize LanceDB:', error.message));

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('Security features enabled:');
  console.log('  - Rate limiting: Active');
  console.log('  - Content Security Policy: Active');
  console.log('  - SSRF protection: Active (Ollama host validation)');
  console.log('  - Input validation: Active');
  console.log('Available providers:');
  console.log(`  - Ollama: ${OLLAMA_HOST}`);
  console.log(`  - Claude: ${CLAUDE_API_KEY ? 'Configured' : 'Not configured'}`);
  console.log(`  - OpenAI: ${OPENAI_API_KEY ? 'Configured' : 'Not configured'}`);
  console.log(`  - Grok: ${GROK_API_KEY ? 'Configured' : 'Not configured'}`);
  console.log(`  - SquatchServe: ${SQUATCHSERVE_HOST}`);
  console.log(`  - Llama.cpp: ${LLAMACPP_HOST}`);
  console.log(`  - SearXNG: ${SEARXNG_HOST}`);
  const startupConfig = getConfig();
  const startupTts = getVoiceProvider('tts');
  const startupStt = getVoiceProvider('stt');
  console.log('Voice services:');
  console.log(`  - TTS (${startupTts?.type || 'none'}): ${startupTts?.name || 'Not configured'} → ${startupTts?.host || 'N/A'}`);
  console.log(`  - STT (${startupStt?.type || 'none'}): ${startupStt?.name || 'Not configured'} → ${startupStt?.host || 'N/A'}`);
  console.log('Conversation features:');
  console.log('  - Chat history: SQLite');
  console.log('  - Semantic memory: LanceDB (vector) + SQLite FTS5 (BM25)');
  console.log(`  - Hybrid search: ${startupConfig.memory.hybridSearchWeights.vector} vector + ${startupConfig.memory.hybridSearchWeights.bm25} BM25`);
  console.log('  - Fact extraction: Auto-extract after each exchange');
  console.log('  - Memory flush: Auto-compact at 80% context usage');
  console.log('  - Memory clusters: Associative cluster-aware retrieval');
  console.log(`  - Long-term memory: rendered from SQLite per request`);
  console.log(`  - Memory files: data/memory/ (USER.md, daily/)`);
  console.log(`  - MCP tools: ${mcpClient.hasTools() ? mcpClient.getToolNames().join(', ') : 'None'}`);
  console.log(`  - Memory heartbeat: ${startupConfig.heartbeat.enabled ? `Every ${startupConfig.heartbeat.intervalHours}h (first run in ${startupConfig.heartbeat.warmupMinutes}min)` : 'Disabled'}`);
  if (ALLOWED_OLLAMA_HOSTS.length > 0) {
    console.log(`  - Additional Ollama hosts: ${ALLOWED_OLLAMA_HOSTS.join(', ')}`);
  }
  memoryManager.startHeartbeat();
  memoryManager.startLivenessProbe();
  memoryManager.startScheduler();

  // Hand the LIVE MCP registry to the manifest so tool capabilities are DERIVED
  // rather than restated. A registered tool no hand-written entry claims gets an
  // entry generated from the tool's own description, so shipping a tool can no
  // longer leave the manifest silently missing it.
  try {
    capabilityManifest.setToolRegistry(mcpClient);
    capabilityManifest.startupCheck();
  } catch (e) {
    console.error('[Capabilities] startup check failed:', e.message);
  }

  // Reconcile the capability manifest against its known-set and log any
  // additions/removals to the ops ledger, so manifest changes leave a machine
  // trail. Logs only — never writes self-facts (introductions are a separate,
  // deliberate step, never a bulk insert).
  try {
    const { added, removed } = capabilityManifest.syncToOps();
    console.log(`  - Capability manifest: ${capabilityManifest.getAll().length} capabilities (${added.length} new, ${removed.length} removed this boot)`);
  } catch (e) {
    console.error('[Capabilities] syncToOps error:', e.message);
  }
});
