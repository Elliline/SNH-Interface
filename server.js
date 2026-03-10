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

// MCP tool calling
const MCPClient = require('./mcp/mcp-client');

// Configuration
const { getConfig, updateConfig, getProviderInstance, getVoiceProvider } = require('./db/config');

// Routes
const conversationsRouter = require('./routes/conversations');
const memoryRouter = require('./routes/memory');

const app = express();

// Trust proxy for rate limiting behind reverse proxy (Nginx, Cloudflare, etc.)
app.set('trust proxy', 1);

// Configuration from environment
const PORT = process.env.PORT || 3000;
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const CLAUDE_API_KEY = process.env.CLAUDE_API_KEY || '';
const GROK_API_KEY = process.env.GROK_API_KEY || '';
const TTS_HOST = process.env.TTS_HOST || 'http://localhost:5050';
const STT_HOST = process.env.STT_HOST || 'http://localhost:5051';
const SEARXNG_HOST = process.env.SEARXNG_HOST || 'http://192.168.4.97:8888';

// Additional allowed Ollama hosts (comma-separated in .env)
const ALLOWED_OLLAMA_HOSTS = process.env.ALLOWED_OLLAMA_HOSTS
  ? process.env.ALLOWED_OLLAMA_HOSTS.split(',').map(h => h.trim())
  : [];

// ============ Security Configuration ============

// SECURITY: Rate limiting to prevent API abuse
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

const chatLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 20, // Limit each IP to 20 chat requests per minute
  message: { error: 'Too many chat requests, please slow down.' },
  standardHeaders: true,
  legacyHeaders: false,
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

// Initialize MCP tool client
const mcpClient = new MCPClient();
mcpClient.loadConfig();

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

    // Resolve active TTS provider from config, fall back to env var
    // Cloud types always use hardcoded official API hosts (never user-supplied host)
    const CLOUD_TTS_HOSTS = { 'openai-tts': 'https://api.openai.com', 'elevenlabs': 'https://api.elevenlabs.io' };
    const ttsProvider = getVoiceProvider('tts');
    const ttsType = ttsProvider?.type || 'kokoro';
    const ttsHost = CLOUD_TTS_HOSTS[ttsType] || ttsProvider?.host || TTS_HOST;

    let response;
    if (ttsType === 'piper') {
      // Piper uses GET with query params
      const params = new URLSearchParams({ text });
      response = await fetch(`${ttsHost}/api/tts?${params}`, { method: 'GET' });
    } else {
      // OpenAI-compatible: kokoro, chatterbox, orpheus, qwen3tts, openai-tts, elevenlabs
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
  const sttHost = CLOUD_STT_HOSTS[sttType] || sttProvider?.host || STT_HOST;

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
 * POST /api/chat/memory
 * Enhanced chat endpoint that:
 * 1. Saves user message to SQLite
 * 2. Searches for relevant past context from other conversations
 * 3. Injects memory context into the system prompt
 * 4. Saves assistant response and embeds it for future retrieval
 */
app.post('/api/chat/memory', chatLimiter, async (req, res) => {
  try {
    const { model, messages, ollamaHost, conversation_id, provider, apiKey, searxngHost } = req.body;

    // Read tool enabled state from config instead of per-request flag
    const appConfig = getConfig();
    const toolsEnabled = !!(appConfig.tools && appConfig.tools.searxng && appConfig.tools.searxng.enabled);

    // SECURITY: Validate inputs
    if (!isValidModelName(model)) {
      return res.status(400).json({ error: 'Invalid model name' });
    }

    if (!isValidMessageArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages array' });
    }

    // Get or create conversation
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

    // DEBUG: Log conversation and message info
    console.log('=== Memory Chat ===');
    console.log('Conversation ID:', convoId);
    console.log('Provider:', provider || 'ollama', '| Model:', model);
    console.log('User message:', userMessage.content.substring(0, 80));
    console.log('Tools enabled:', toolsEnabled, '(type:', typeof toolsEnabled, ') | MCP has tools:', mcpClient.hasTools(), '| Tool names:', mcpClient.getToolNames());

    // Save user message to database
    const userMsgId = db.addMessage(convoId, 'user', userMessage.content, model);

    // === UPGRADE 1: Load durable memory files (MEMORY.md, USER.md, daily logs) ===
    let memoryFiles = { memory: null, user: null, dailyToday: null, dailyYesterday: null };
    try {
      memoryFiles = db.loadMemoryFiles();
      console.log('Memory files loaded:', {
        memory: memoryFiles.memory ? `${memoryFiles.memory.length} chars` : 'none',
        user: memoryFiles.user ? `${memoryFiles.user.length} chars` : 'none',
        dailyToday: memoryFiles.dailyToday ? `${memoryFiles.dailyToday.length} chars` : 'none',
        dailyYesterday: memoryFiles.dailyYesterday ? `${memoryFiles.dailyYesterday.length} chars` : 'none'
      });
    } catch (memFileError) {
      console.error('Memory file load error:', memFileError.message);
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

    // Build comprehensive memory system prompt
    const memoryParts = [];

    // Add durable memory (MEMORY.md + USER.md)
    if (memoryFiles.memory) {
      memoryParts.push(`=== Long-Term Memory ===\n${memoryFiles.memory}`);
    }
    if (memoryFiles.user) {
      memoryParts.push(`=== User Profile ===\n${memoryFiles.user}`);
    }

    // Add daily logs for short-term continuity
    if (memoryFiles.dailyYesterday) {
      memoryParts.push(`=== Yesterday's Session Log ===\n${memoryFiles.dailyYesterday}`);
    }
    if (memoryFiles.dailyToday) {
      memoryParts.push(`=== Today's Session Log ===\n${memoryFiles.dailyToday}`);
    }

    // Add hybrid search results from past conversations
    if (memoryContext.length > 0) {
      const contextText = memoryContext
        .map((m, i) => `[Memory ${i + 1}] ${m.role}: ${m.text.substring(0, 500)}${m.text.length > 500 ? '...' : ''}`)
        .join('\n');
      memoryParts.push(`=== Relevant Past Conversations ===\n${contextText}`);
    }

    // Add cluster-aware memory context
    if (clusterContext.length > 0) {
      const clusterText = clusterContext.map(c => {
        const memberText = c.members.map(m => `- ${m.content}`).join('\n');
        let text = `[${c.cluster.name}]\n${memberText}`;
        if (c.linkedMembers.length > 0) {
          const linkedText = c.linkedMembers
            .map(lm => `- (from ${lm.clusterName}) ${lm.content}`)
            .join('\n');
          text += `\nRelated:\n${linkedText}`;
        }
        return text;
      }).join('\n\n');
      memoryParts.push(`=== Associated Memory Clusters ===\n${clusterText}`);
    }

    if (memoryParts.length > 0) {
      console.log('Injecting memory context:', memoryParts.length, 'sections');
      const memorySystemMessage = {
        role: 'system',
        content: `You have access to the following memory and context:\n\n${memoryParts.join('\n\n')}\n\nUse this context if it helps answer the current question, but don't explicitly mention that you're using memory unless asked.`
      };
      enhancedMessages = [memorySystemMessage, ...enhancedMessages];
    } else {
      console.log('No memory context to inject');
    }

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
    const instanceName = req.body.instanceName;

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

    // Route to appropriate provider
    let response;
    let toolsUsed = false;
    console.log(`=== Routing to provider: ${providerType} ===`);

    if (providerType === 'ollama') {
      // providerHost was already resolved (instance lookup or env fallback); validate it
      const host = instanceName ? providerHost : getOllamaHost({ ollamaHost });
      let ollamaMessages = [...enhancedMessages];

      // Ollama tool calling is strict about message format — consolidate
      // all system messages into a single one at position 0 so the memory
      // context doesn't create extra messages that break the tool schema
      if (toolsEnabled) {
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
      if (toolsEnabled && mcpClient.hasTools()) {
        const tools = mcpClient.getToolsForOpenAI();
        const toolSearxngHost = searxngHost
          ? (isValidOllamaHost(searxngHost) ? searxngHost : SEARXNG_HOST)
          : SEARXNG_HOST;
        const toolContext = { searxngHost: toolSearxngHost };
        const MAX_TOOL_ROUNDS = 3;

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

            const result = await mcpClient.executeTool(fnName, args, toolContext);
            console.log(`MCP [ollama]: Tool "${fnName}" result:`, JSON.stringify(result).substring(0, 200));

            ollamaMessages.push({
              role: 'tool',
              content: typeof result === 'string' ? result : JSON.stringify(result)
            });
          }
        }

        if (toolsUsed) {
          console.log('MCP [ollama]: Tools were used, making final streaming request');
          ollamaMessages.push({
            role: 'system',
            content: 'Tool calls are complete. Now provide your response to the user based on the information gathered.'
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
        })
      });
    } else if (providerType === 'claude') {
      const claudeKey = apiKey || CLAUDE_API_KEY;
      if (!claudeKey) {
        return res.status(401).json({ error: 'Claude API key not configured' });
      }
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
          messages: enhancedMessages.filter(m => m.role !== 'system')
        })
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
        })
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
        })
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
        })
      });
    } else if (providerType === 'llamacpp' || providerType === 'vllm') {
      const llamacppHost = providerHost;
      let llamacppMessages = [...enhancedMessages];

      // MCP tool calling loop (only when tools are enabled)
      const providerLabel = providerType === 'vllm' ? 'vllm' : 'llamacpp';
      console.log(`MCP [${providerLabel}]: toolsEnabled=${toolsEnabled}, hasTools=${mcpClient.hasTools()}`);
      if (toolsEnabled && mcpClient.hasTools()) {
        const tools = mcpClient.getToolsForOpenAI();
        const toolSearxngHost = searxngHost
          ? (isValidOllamaHost(searxngHost) ? searxngHost : SEARXNG_HOST)
          : SEARXNG_HOST;
        const toolContext = { searxngHost: toolSearxngHost };
        const MAX_TOOL_ROUNDS = 3;

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

            const result = await mcpClient.executeTool(fnName, args, toolContext);
            console.log(`MCP [${providerLabel}]: Tool "${fnName}" result:`, JSON.stringify(result).substring(0, 200));

            llamacppMessages.push({
              role: 'tool',
              tool_call_id: toolCall.id,
              content: typeof result === 'string' ? result : JSON.stringify(result)
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
          content: 'Tool calls are complete. Now provide your response to the user based on the information gathered.'
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
        body: JSON.stringify(finalBody)
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
    res.setHeader('X-Has-Memory-Context', (memoryContext.length > 0 || memoryFiles.memory || memoryFiles.user) ? 'true' : 'false');
    res.setHeader('X-Tools-Used', toolsUsed ? 'true' : 'false');
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

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Write decoded text for SSE, raw bytes for NDJSON
        if (contentType === 'text/event-stream') {
          res.write(chunk);
        } else {
          res.write(value);
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
      res.end();
    } finally {
      reader.releaseLock();
    }

    // Save assistant response to database
    if (fullResponse) {
      const assistantMsgId = db.addMessage(convoId, 'assistant', fullResponse, model);

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
        providerHost
      ).catch(err => {
        console.warn('[FactExtractor] Background extraction error:', err.message);
      });
    }

  } catch (error) {
    console.error('Memory chat error:', error.message);
    if (!res.headersSent) {
      res.status(503).json({ error: error.message || 'Chat service unavailable' });
    }
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
  console.log('Voice services:');
  console.log(`  - TTS (Kokoro): ${TTS_HOST}`);
  console.log(`  - STT (Whisper): ${STT_HOST}`);
  console.log('Conversation features:');
  console.log('  - Chat history: SQLite');
  console.log('  - Semantic memory: LanceDB (vector) + SQLite FTS5 (BM25)');
  const startupConfig = getConfig();
  console.log(`  - Hybrid search: ${startupConfig.memory.hybridSearchWeights.vector} vector + ${startupConfig.memory.hybridSearchWeights.bm25} BM25`);
  console.log('  - Fact extraction: Auto-extract after each exchange');
  console.log('  - Memory flush: Auto-compact at 80% context usage');
  console.log('  - Memory clusters: Associative cluster-aware retrieval');
  console.log(`  - Memory files: data/memory/ (MEMORY.md, USER.md, daily/)`);
  console.log(`  - MCP tools: ${mcpClient.hasTools() ? mcpClient.getToolNames().join(', ') : 'None'}`);
  console.log(`  - Memory heartbeat: ${startupConfig.heartbeat.enabled ? `Every ${startupConfig.heartbeat.intervalHours}h (first run in ${startupConfig.heartbeat.warmupMinutes}min)` : 'Disabled'}`);
  if (ALLOWED_OLLAMA_HOSTS.length > 0) {
    console.log(`  - Additional Ollama hosts: ${ALLOWED_OLLAMA_HOSTS.join(', ')}`);
  }
  memoryManager.startHeartbeat();
});
