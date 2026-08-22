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
const dispatchClaims = require('./db/dispatch-claims');
const codingJobs = require('./db/coding-jobs');
const approvalClassifier = require('./db/approval-classifier');
const agentPool = require('./db/agent-pool');
const identity = require('./db/identity');
const capabilityManifest = require('./db/capability-manifest');
const initiatives = require('./db/initiatives');
// The agent-job queue. Deliberately NOT initiatives: results go to the jobs
// panel, which never opens a conversation — see db/agent-jobs.js.
const agentJobs = require('./db/agent-jobs');
const questionQueue = require('./db/questions');
const { getCurrentDateTimeString, formatFactTimestamp } = require('./db/datetime');
const injectionBudget = require('./db/injection-budget');
const { classifyToolNeed, isTimeSensitive, classifySchedulingIntent, classifyMemoryWriteIntent, classifyMemoryReadIntent, classifyMemoryCorrectionIntent, classifyJobsIntent, classifyHandoffIntent, classifyHandoffSignal } = require('./db/tool-routing');
const { createToolArtifactFilter, stripToolArtifacts, CANNOT_CHECK } = require('./db/tool-artifacts');

// MCP tool calling
const MCPClient = require('./mcp/mcp-client');

// Configuration
const { getConfig, updateConfig, getProviderInstance, getVoiceProvider, getSearxngConfig, getSearchConfig } = require('./db/config');

// Routes
const conversationsRouter = require('./routes/conversations');
const memoryRouter = require('./routes/memory');
const jobsRouter = require('./routes/jobs');
const toolsRouter = require('./routes/tools');

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
// Bind address. Defaults to loopback so an unconfigured instance is never
// exposed on the network by accident; LAN binding is opt-in by setting HOST
// to a specific address (this box uses 192.168.4.179 via .env).
const HOST = process.env.HOST || '127.0.0.1';
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

/**
 * Compact role sequence of a messages array, for logging: "0:system 1:system 2:user".
 * The order of roles is the thing that breaks against a strict chat template, and
 * it was not visible anywhere in the logs when it did.
 */
function roleSequence(messages) {
  return (messages || []).map((m, i) => `${i}:${m && m.role}`).join(' ');
}

/**
 * Canonical outbound shape for a chat-completions API: AT MOST ONE system
 * message, and it leads the array; every other message keeps conversation order.
 *
 * SNH builds its prompt as many separate system messages — identity, capability
 * manifest, memory, date/time, unseen notices, and a tail of per-turn guidance
 * blocks that are pushed AFTER the conversation on purpose. That is a fine
 * internal representation and a bad wire format: a chat template is free to
 * accept only one, and Qwen3's does exactly that. Its loop raises
 * 'System message must be at the beginning.' for any system message where
 * `not loop.first` — so the SECOND system message trips it even though every
 * system message is still, in plain English, at the beginning. Gemma's template
 * merged them silently, which is why this survived until the model changed.
 *
 * This is not a Qwen workaround. One leading system message is the shape every
 * OpenAI-compatible provider accepts, so it is what we send to all of them, and
 * the next template's rules cannot reach it.
 *
 * Concatenation order is the array's own order, which the assembly above has
 * already arranged stable → volatile → trailing guidance, so the cacheable
 * prefix that ordering exists to protect is preserved inside the folded message.
 */
function foldSystemMessages(messages) {
  const systemParts = [];
  const rest = [];
  for (const m of messages || []) {
    if (!m) continue;
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content : String(m.content ?? '');
      if (text.trim()) systemParts.push(text.trim());
    } else {
      rest.push(m);
    }
  }
  if (!systemParts.length) return rest;
  return [{ role: 'system', content: systemParts.join('\n\n') }, ...rest];
}

// The reasoning channel, shared with the background paths (db/memory-manager.js
// reads the same function). Reasoning is deliberately NOT answer text: it never
// joins the reply, is never embedded, and is never stored as what SNH said.
const { extractReasoning } = require('./db/reasoning-channel');

/**
 * Generation budgets for one provider request.
 *
 * `thinking_token_budget` is a vLLM extension and is only included for the local
 * OpenAI-compatible engines; sending an unknown field to a hosted provider is a
 * 400. `reasoning_effort` is understood by both vLLM and OpenAI, and is omitted
 * entirely when configured null so a non-reasoning model is left alone.
 */
function generationParams(providerType) {
  const gen = getConfig().generation || {};
  const thinking = Number.isFinite(gen.thinkingTokens) ? gen.thinkingTokens : null;
  const responseT = Number.isFinite(gen.responseTokens) ? gen.responseTokens : null;
  const body = {};

  // Unset means UNSENT, not a fallback number. An engine given no max_tokens
  // allows the rest of its window; an engine given 4096 stops at 4096 and cuts
  // the sentence in half. Defaulting here would apply that ceiling to every box
  // that never asked for one.
  if (thinking !== null || responseT !== null) {
    body.max_tokens = (thinking || 0) + (responseT || 0);
  }
  if (gen.reasoningEffort) body.reasoning_effort = gen.reasoningEffort;
  if (thinking !== null && thinking > 0 && ['vllm', 'llamacpp'].includes(providerType)) {
    body.thinking_token_budget = thinking;
  }
  return body;
}

/**
 * Fold + log, immediately before a provider request. Logs both sequences so the
 * shape actually sent is visible, not inferred.
 */
/**
 * A LIVE STATUS BLOCK IS STRIPPED OUT OF HISTORY BEFORE HE SEES IT AGAIN.
 *
 * Until 2026-08-22 the server appended `_squatch-code, working:_ …` to the
 * reply. That put it inside his message, the message is stored whole, and the
 * next turn handed it back to him as his own words. An hour after the first
 * real one he wrote a fake — right format, invented command, nothing running.
 *
 * The append is gone, but two of those blocks are in the stored transcript and
 * deleting her history to fix our bug is not on. So they are removed on the way
 * OUT instead: the record keeps what happened, and he stops being shown a
 * format that is no longer his to write. Same reasoning as never re-adding a
 * vector for an inactive fact — the row stays, the retrieval stops.
 */
const STORED_STATUS_BLOCK =
  /\n*(?:---\n+)?_squatch-?code, working:_\n(?:[ \t]*[-*].*\n?)*/gi;

function stripStoredStatusBlocks(messages) {
  let stripped = 0;
  const out = messages.map(m => {
    if (m.role !== 'assistant' || typeof m.content !== 'string') return m;
    if (!/squatch-?code, working/i.test(m.content)) return m;
    stripped++;
    return { ...m, content: m.content.replace(STORED_STATUS_BLOCK, '\n').trimEnd() };
  });
  return { out, stripped };
}

function prepareOutboundMessages(messages, label) {
  const { out: cleaned, stripped } = stripStoredStatusBlocks(messages);
  if (stripped) {
    console.log(`[Outbound ${label}] stripped ${stripped} stored status block(s) — he must not relearn that format`);
  }
  const folded = foldSystemMessages(cleaned);
  console.log(`[Outbound ${label}] roles in : ${roleSequence(messages)}`);
  console.log(`[Outbound ${label}] roles out: ${roleSequence(folded)}`);
  return folded;
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

// The markdown renderer, served from db/ rather than copied into public/.
//
// The panel card and the printable report have to agree on what a document
// looks like, and the only way to guarantee that is for both to run the same
// bytes. A copy under public/ would be a second answer that drifts — so the one
// file in db/ is served here, and there is nothing to keep in step.
app.get('/markdown.js', (req, res) => {
  res.type('application/javascript');
  res.setHeader('Cache-Control', 'no-cache');
  res.sendFile(path.join(__dirname, 'db', 'markdown.js'));
});

// Apply rate limiting to API routes
// TTS first — it has its own (much higher) budget and is excluded from the
// shared /api/ bucket by apiLimiter's skip.
app.use('/api/tts', ttsLimiter);
app.use('/api/', apiLimiter);

// Mount conversation routes
app.use('/api/conversations', conversationsRouter);

// Mount memory routes
app.use('/api/memory', memoryRouter);

// Mount the jobs routes — the ROBOT channel. Its own prefix, not a corner of
// /api/memory, because job results are not initiatives and must not be served
// by the endpoint the bell polls.
app.use('/api/jobs', jobsRouter);
// The Tools tab's data: the tool catalogue, the search provider chain, and secret
// STATUS. Its own router because secrets are its own contract — write-only, and
// never part of the /api/config payload that the browser reads whole.
app.use('/api/tools', toolsRouter);

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
/**
 * The two deadlines every engine call on the CHAT path is held to.
 *
 * One flat wall-clock cannot tell a wedged engine from a slow one, and chat
 * had three of them: 120s per tool round on each provider and 90s on the
 * final stream. See the `chat` block in db/config.js for what that cost on
 * 2026-08-22. These are read per call rather than captured at boot so a
 * change in Settings takes effect on the next turn, not the next restart.
 *
 * Falls back to the shipped defaults on a missing or nonsense value — a
 * timeout of 0 or NaN would mean "abort immediately", so a bad config would
 * take chat down entirely rather than degrade it.
 */
/**
 * HE SAID HE SENT IT. GIVE HIM ONE PINNED ROUND TO ACTUALLY DO IT.
 *
 * The second trigger, and it keys off HIS CLAIM rather than her wording — which
 * is the point. The claim is bounded and observable: there are only so many ways
 * to assert you dispatched something, and the phantom classifier already reads
 * them reliably. There is no bounded way to ASK.
 *
 * It cannot be a pin on the first round, because the claim does not exist until
 * the reply does. So it is a second round, run after the reply is written, with
 * `tool_choice` naming dispatch_coding_job and one instruction: make the call
 * you just described.
 *
 * WHY THIS BEATS THE SERVER INFERRING. He supplies the project and the brief, so
 * nothing is guessed. The previous backstop had to work out which project the
 * work belonged to from the conversation, and on a fresh conversation it could
 * not — the 2026-08-22 resend was in a brand-new conversation, so even had it
 * run, it would have declined. He knows; he just did not call.
 *
 * The tool's own guards are untouched: brief-shown still refuses anything she
 * has not read, validateBrief still refuses a brief that names a directory. A
 * pin decides when to ask, never what is allowed.
 */
async function forcedDispatchRound({ host, model, providerType, messages, assistantReply, toolContext }) {
  const openAiStyle = providerType !== 'ollama';
  const url = openAiStyle ? `${host}/v1/chat/completions` : `${host}/api/chat`;
  const spec = mcpClient.getToolsForOpenAI().find(t => t.function.name === 'dispatch_coding_job');
  if (!spec) return { attempted: false, reason: 'the tool is not registered' };

  const convo = [
    ...messages,
    { role: 'assistant', content: String(assistantReply || '').slice(0, 4000) },
    { role: 'user', content:
      'SYSTEM: Your reply above says the brief was sent to the coding agent. No job '
      + 'exists — you did not call the tool. Call dispatch_coding_job now, with the '
      + 'project this work belongs to and the brief exactly as she read it. Do not '
      + 'write a reply; make the call.' },
  ];

  const ct = chatTimeouts();
  let data;
  try {
    data = await memoryManager.streamChat({
      url,
      openAiStyle,
      body: {
        ...(openAiStyle ? { model } : { model }),
        messages: prepareOutboundMessages(convo, 'forced dispatch retry'),
        tools: [spec],
        tool_choice: { type: 'function', function: { name: 'dispatch_coding_job' } },
      },
      firstTokenMs: ct.firstTokenMs,
      stallMs: ct.stallMs,
      label: 'forced dispatch retry',
    });
  } catch (err) {
    // A refused pin or a stall must not take out the turn — the reply is
    // already written and she is reading it.
    return { attempted: true, called: false, reason: `the retry call failed: ${err.message}` };
  }

  const msg = openAiStyle ? (data.choices?.[0]?.message || {}) : (data.message || {});
  const call = (msg.tool_calls || []).find(c => c.function && c.function.name === 'dispatch_coding_job');
  if (!call) return { attempted: true, called: false, reason: 'he still did not call it' };

  let args;
  try {
    args = typeof call.function.arguments === 'string'
      ? JSON.parse(call.function.arguments) : (call.function.arguments || {});
  } catch (err) {
    return { attempted: true, called: false, reason: `his arguments did not parse: ${err.message}` };
  }

  const result = await mcpClient.executeTool('dispatch_coding_job', args, toolContext);
  return { attempted: true, called: true, result, args };
}

function chatTimeouts() {
  const c = (getConfig().chat) || {};
  const pick = (v, dflt) => (Number.isFinite(v) && v > 0 ? v : dflt);
  return {
    stallMs: pick(c.stallTimeoutMs, 60000),
    firstTokenMs: pick(c.firstTokenTimeoutMs, 120000)
  };
}

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

// ============ Web Search ============

/**
 * The same provider chain the model uses, over HTTP.
 *
 * It used to be its own hand-rolled SearXNG fetch — a SECOND search
 * implementation, which meant a second answer to "which provider ran and did it
 * find anything", and only one of the two was logged. It now goes through the
 * registered web_search tool, so Exa-then-SearXNG, the fallback and the
 * search_call_log row are identical whoever asked.
 */
app.post('/api/search', async (req, res) => {
  try {
    const { query, searxngHost } = req.body;

    if (!query || typeof query !== 'string' || query.length > 500) {
      return res.status(400).json({ error: 'Invalid search query' });
    }

    // SECURITY: a custom SearXNG host is still validated before it is honoured.
    // An invalid one is dropped rather than refusing the search — the chain has
    // a configured instance to fall back on.
    const host = searxngHost && isValidOllamaHost(searxngHost) ? searxngHost : undefined;

    const tool = mcpClient.tools.get('web_search');
    if (!tool) {
      return res.status(503).json({ error: 'No search provider is available', results: [] });
    }

    const out = await tool.execute({ query, num_results: 5 }, { caller: 'api', searxngHost: host });
    if (out && out.error) {
      return res.status(503).json({ error: out.error, results: [] });
    }

    // Legacy response shape kept: the client reads {url, title, content}.
    res.json({
      results: (out.results || []).map(r => ({ url: r.url, title: r.title, content: r.snippet })),
      provider: out.provider || null,
      providers_tried: out.providers_tried || []
    });
  } catch (error) {
    console.error('Web search error:', error.message);
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
  // When this turn began — the window the phantom-dispatch guard compares
  // against, so a job started in an earlier turn cannot vouch for this one.
  const turnStartedAt = new Date().toISOString();
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
    const agentJobsToolEnabled = !!(appConfig.tools && appConfig.tools.agentJobs && appConfig.tools.agentJobs.enabled !== false)
      && !!(appConfig.agentJobs && appConfig.agentJobs.enabled !== false);

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
    // HANDOFF intent — she is asking for work rather than for an answer. This is
    // only a reason to ENTER the loop; once in it he is handed the whole
    // registry, so a research or memory turn that turns out to be bigger than a
    // turn can be handed off without its own classifier firing. Whether a thing
    // is worth handing off is his judgement, in the tool description — a regex
    // cannot tell a two-second lookup from a twenty-minute sweep.
    const handoffSignal = agentJobsToolEnabled && mcpClient.hasTool('start_background_job')
      ? classifyHandoffSignal(userMessage.content, {
        allowBuild: !!(appConfig.tools && appConfig.tools.agentJobs && appConfig.tools.agentJobs.dispatchBuildRequests)
      })
      : { dispatch: false, tier: null, reason: null };
    const needsHandoff = handoffSignal.dispatch;

    // === THE PRE-GENERATION TOOL GATE IS GONE (2026-08-18) ===
    //
    // Tools are attached to every turn now, and the model decides. The classifiers
    // above still run — they are what the guidance nudges key off, and they are a
    // useful record of what the turn looked like — but they no longer decide
    // whether he is ALLOWED to reach for anything.
    //
    // WHY. On 2026-08-18 Ellie typed "Use an agent and write up any thing you
    // know about my clients." Every classifier returned false, the turn routed
    // DIRECT, and the model was handed an empty tools array — so it wrote a
    // detailed paragraph about the background job it had started, the categories
    // the report would use, and the fields within each. No job existed. It could
    // not have existed: nothing was callable.
    //
    // That is the shape of the whole class. A gate in front of generation has to
    // predict, from a regex, what the model will need before the model has read
    // the message — and every miss is invisible, because a tool that was never
    // offered leaves no trace of not being offered. The two dispatches that DID
    // work that day worked because the messages happened to mention a company
    // and a year, which tripped the SEARCH classifier and dragged the whole
    // registry along behind it. Working by coincidence of shape is not working.
    //
    // What the classifiers become is a SAFETY NET rather than the mechanism: they
    // add guidance ("she is not waiting", "she asked for an agent by name") on
    // top of a tool set the model already has. Right order.

    // === SHE NAMED THE AGENT: THE CALL IS FORCED, NOT SUGGESTED ===
    //
    // Tier 1 has never been gated — classifyHandoffSignal returns it before it
    // ever looks at allowBuild — and on 2026-08-18 that was not enough. "Use and
    // agent and write me a python script for a calculator": tier 1 fired, the
    // guidance block fired, start_background_job was fourth of eleven tools in
    // the payload, and the model returned `tool_calls: []` with the words "I have
    // started a background job to write a Python calculator script." The phantom
    // guard caught the claim. The job should have been real.
    //
    // So the third attempt at this stops arguing. On tier 1 the FIRST tool round
    // pins tool_choice to start_background_job, which takes the decision out of
    // the model's hands for exactly the case where she already made it. Rounds
    // after the first are free — it still answers her in the same turn, which is
    // the other half of the rule.
    //
    // Tier 1 ONLY. Tiers 2–4 are inferences about the shape of the work and
    // forcing one would dispatch a job she never asked for; tier 1 is her naming
    // the mechanism, and the cost of being wrong is one read-only job in a panel.
    const forceHandoffCall = needsHandoff && handoffSignal.tier === 1
      && mcpClient.hasTool('start_background_job');

    // === SHE APPROVED A BRIEF: THE DISPATCH IS FORCED TOO ===
    //
    // THE TRIGGER IS A CLASSIFIER, NOT A PHRASE LIST, and that is the whole
    // point. Dispatch ran at 2 real out of 7 claimed, and every fix was gated
    // on a list of phrases — "send it", "send away", "ship it". She wrote
    // "Please try sending the brief again. Something did not work the last time
    // and it should be fixed." That is an approval by any reading; it matched
    // nothing, and because the pin and the backstop shared one signal, neither
    // fired. She must be able to talk like a person, so no wording may gate a
    // dispatch.
    //
    // The classifier only runs when there is something to approve — a brief on
    // her screen with no coding job since — so most turns never pay for it. It
    // fails closed: an error or an unparseable answer means no pin, which is
    // exactly today's behaviour, and the claim-keyed backstop after the reply
    // is the second net.
    //
    // The phrase list is KEPT, and it gates nothing. When it happens to hit it
    // saves a round trip; when it misses, the classifier decides. If they
    // disagree the classifier wins, because the list is the thing that failed.
    let forceCodingCall = false;
    let codingPinReason = null;
    if (mcpClient.hasTool('dispatch_coding_job')) {
      const pending = approvalClassifier.pendingBrief({ conversationId: convoId });
      if (pending) {
        const phrase = dispatchClaims.classifyCodingGoAhead(userMessage.content);
        const verdict = await approvalClassifier.isApproval({
          brief: pending.text,
          message: userMessage.content,
          callLLM: memoryManager.callLLM,
        });
        forceCodingCall = verdict.approved;
        codingPinReason = verdict.reason;
        if (phrase.goAhead !== verdict.approved) {
          // Worth a line: this is the disagreement the rebuild exists for, and
          // it is how anyone learns the list is still drifting.
          console.log(`Tool routing: phrase list said ${phrase.goAhead ? 'YES' : 'NO'}, `
            + `classifier said ${verdict.approved ? 'YES' : 'NO'} — classifier wins`);
        }
      }
    }
    if (forceCodingCall) console.log(`Tool routing: SHE APPROVED THE PENDING BRIEF (${codingPinReason}) — pinning dispatch_coding_job this round`);

    const classifierWouldHaveGated = needsSearchTools || needsActionTools || needsMemoryWrite
      || needsMemoryRead || needsMemoryCorrections || needsJobsRead || needsHandoff;
    const needsTools = mcpClient.hasTools();
    const firedList = [needsSearchTools && 'search/fetch', needsActionTools && 'scheduling',
      needsMemoryWrite && 'memory-write', needsMemoryRead && 'memory-read',
      needsMemoryCorrections && 'memory-corrections', needsJobsRead && 'jobs-read',
      needsHandoff && `handoff:t${handoffSignal.tier}`].filter(Boolean);
    console.log(`Tool routing: ALL TOOLS ATTACHED (classifiers fired: ${firedList.length ? firedList.join(' + ') : 'none'}` +
      `${!classifierWouldHaveGated ? ' — this turn would have been DIRECT under the old gate' : ''})`);

    // Should-I-search honesty guard: if the question is about current/changeable
    // facts (weather, prices, news, "right now"/"latest") but search will NOT run
    // — because it's off, unavailable, or the classifier didn't route to it — the
    // model must NOT answer confidently from memory (7/23: it fabricated a weather
    // high). We inject an instruction to offer to look it up instead. When search
    // WILL run (needsTools), the tool loop handles it and no nudge is needed.
    // With the gate gone, "will search run" is no longer knowable up front — the
    // model decides. So this guard now fires on what IS knowable: whether search
    // is available at all. If the stack is off or unregistered, a question about
    // current facts still needs the honesty nudge; if it is available, the tool
    // loop is there and the model can use it.
    const searchAvailable = toolsEnabled && mcpClient.hasTool('web_search');
    const timeSensitiveUnsearched = isTimeSensitive(userMessage.content) && !searchAvailable;

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

    // === She has said she is not waiting ===
    //
    // The description tells him WHEN to hand work off; this tells him that the
    // condition is met right now, in this message. Both halves are needed, and
    // the reason is what happened on 2026-08-18: the tool was in the payload,
    // fourth of eleven, on a turn whose message said "Take your time, I'm going
    // to keep chatting while you work" — and he searched once and answered
    // inline anyway. A tool he can use immediately will always win an argument
    // conducted only inside a tool description.
    //
    // It fires exactly when classifyHandoffIntent does, so it cannot nudge a
    // turn that was never offered the tool, and it says BOTH things: hand off
    // the digging, and still answer what you already know. Ending a turn with
    // only "I'll come back to you" is the other failure, and it happened live
    // too.
    if (needsHandoff && mcpClient.hasTool('start_background_job')) {
      // The nudge says WHICH signal fired, because the four tiers are different
      // arguments. Tier 1 is not a suggestion — she named the mechanism.
      const head = handoffSignal.tier === 1
        ? '=== She Asked For An Agent, By Name ===\n' +
          'She has asked you to use an agent for this. That is not a judgement call: call ' +
          'start_background_job and hand the work to it.\n'
        : `=== This Looks Like Agent Work ===\nWhy: ${handoffSignal.reason}.\n` +
          'If answering it properly needs real digging — more than a couple of searches, or several ' +
          'sources compared — hand that part to a background agent with start_background_job. It can ' +
          'run a dozen searches and read whole pages, where you get two or three searches and snippets.\n';
      memoryParts.push({
        kind: 'guidance',
        label: `handoff signal (tier ${handoffSignal.tier})`,
        text: head +
          'Then answer her anyway in this same turn with what you already know. Starting a job is not ' +
          'a reason to say nothing — a reply that only promises to come back later leaves her with ' +
          'nothing, and she asked you.'
      });
      console.log(`[Handoff] tier ${handoffSignal.tier} (${handoffSignal.reason}) — nudging toward start_background_job (convo ${convoId})`);
    }

    // === What is running right now ===
    //
    // The live half of the jobs picture. The announcement block below covers
    // FINISHED work; this covers work in flight, and it exists because their
    // absence was being filled with invention: asked "are you still working on
    // this?", he described one job as slowed by a connection issue he was
    // "working through" and another as scanning a large volume of memory. Every
    // job he had was finished, and one had never existed.
    //
    // Costs nothing on a normal turn — renderActiveJobsBlock returns null when
    // the queue is empty, and that absence is what the standing instruction
    // below keys off.
    let activeJobsBlock = null;
    try {
      activeJobsBlock = agentJobs.renderActiveJobsBlock();
      if (activeJobsBlock) {
        memoryParts.push({ kind: 'guidance', label: 'jobs running now', text: activeJobsBlock.text });
        console.log(`[AgentJobs] ${activeJobsBlock.running} running / ${activeJobsBlock.queued} queued — live status injected (convo ${convoId})`);
      }
    } catch (activeErr) {
      console.error('[AgentJobs] Active-jobs render error:', activeErr.message);
    }

    // === Background work that finished since he last spoke ===
    //
    // The chat-awareness half of the agent-job queue. Jobs he handed off, and
    // scheduled jobs of his, that finished while nobody was talking to him, so
    // he can lead with "that finished, here's what it found" instead of learning
    // about his own work from her.
    //
    // NEXT TURN MEANS THE NEXT TURN SHE TAKES. This is the only place it fires:
    // the heartbeat is not told, because the heartbeat is not a conversation and
    // a finished job is not a reason to start one — that is the whole channel
    // rule. The jobs panel stays the only place a result is delivered; this
    // block just means he is not the last to know.
    //
    // Zero tokens on almost every turn: renderAnnouncementBlock returns null
    // unless something actually landed. Measured with real output, one job is
    // ~130 tokens and three are ~250, against memory.injection.jobTokens.
    let announcedJobs = null;
    try {
      const rendered = agentJobs.renderAnnouncementBlock({
        limit: injCfg.maxAnnouncedJobs ?? 3,
        tokenCap: injCfg.jobTokens ?? 400
      });
      if (rendered) {
        announcedJobs = rendered.items;
        memoryParts.push({ kind: 'guidance', label: 'finished background work', text: rendered.text });
        console.log(`[AgentJobs] Announcing ${rendered.items.length} finished job(s) to him (${rendered.tokens} tokens, convo ${convoId})`);
      }
    } catch (jobErr) {
      console.error('[AgentJobs] Announcement render error:', jobErr.message);
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
      console.log(`[Capabilities] Injected manifest: ${capBlock.count} capabilities, ~${capBlock.tokens} tokens`);
      // A shed one-liner is a capability he is being shown the name of and
      // nothing else — always the newest ones, since the list is in ship order.
      // It was a clause on the line above until 2026-08-20, which is how
      // job-documents spent a day in his context as four bare words.
      if (capBlock.compacted) {
        console.warn(`[Capabilities] ⚠ ${capBlock.compacted} listed by NAME ONLY — over budget ` +
                     `(${capBlock.tokens}/${capBlock.budget} tokens): ${capBlock.compactedNames.join(', ')}`);
      }
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

    // === Stamp the job announcements, and only now ===
    //
    // AFTER the ceiling, and only if the block is really in the message about to
    // be sent. Same rule as the correction notices, and the same reason: a job
    // stamped as announced by a block that was then trimmed is a result he is
    // never told about again. Nothing expires an unannounced job, so the cost of
    // stamping late is one turn's delay and the cost of stamping early is
    // permanent silence.
    if (announcedJobs && announcedJobs.length) {
      try {
        const inMessage = !!(memorySystemMessage && memorySystemMessage.content
          && memorySystemMessage.content.includes('=== Background Work That Finished ==='));
        if (inMessage) {
          const n = agentJobs.markAnnounced(announcedJobs);
          console.log(`[AgentJobs] Marked ${n} finished job(s) announced (convo ${convoId})`);
        } else {
          console.warn('[AgentJobs] Announcement block did not survive assembly — NOT stamping; it will be offered again next turn');
        }
      } catch (stampErr) {
        console.error('[AgentJobs] Announcement stamping error:', stampErr.message);
      }
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

      // Consolidation into a single leading system message used to live here,
      // gated on needsTools. It is now foldSystemMessages, applied at every
      // dispatch below for two reasons: the tool-free path needs it just as
      // much (the model behind Ollama has a chat template too), and folding
      // once up front does not hold — the tool loop appends the "tool calls
      // are complete" system nudge afterwards and would put it last again.

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

          // Tier 1 only, first round only: pin the call. A named tool_choice is
          // OpenAI-shaped and vLLM honours it; Ollama's /api/chat ignores an
          // unknown field, which is why the backstop after the reply exists and
          // is not optional. If the engine REFUSES the request outright, the same
          // round is retried without the pin rather than losing the turn — a
          // forced call is worth having, and it is not worth a failed reply.
          // TWO THINGS CAN BE PINNED, never both: a tool_choice names one
          // function. Handoff wins if somehow both fire, because she named the
          // mechanism explicitly there.
          const forcedToolName = forceHandoffCall ? 'start_background_job'
            : forceCodingCall ? 'dispatch_coding_job' : null;
          const forceThisRound = !!forcedToolName && round === 0;
          // STREAMED SO THE DEADLINE CAN SEE PROGRESS. streamChat returns the
          // NON-streaming shape, so everything below reads `toolData` exactly
          // as it did when this was `stream: false` — that shape-preservation
          // is the whole reason it is reusable here rather than copied.
          const ct = chatTimeouts();
          const postRound = (forced) => memoryManager.streamChat({
            url: `${host}/api/chat`,
            openAiStyle: false,
            body: {
              model,
              messages: prepareOutboundMessages(ollamaMessages, `ollama tool-round ${round + 1}`),
              tools,
              ...(forced ? { tool_choice: { type: 'function', function: { name: forcedToolName } } } : {})
            },
            firstTokenMs: ct.firstTokenMs,
            stallMs: ct.stallMs,
            label: `ollama tool-round ${round + 1}`
          });

          if (forceThisRound) console.log(`MCP [ollama]: FORCING ${forcedToolName} this round — ${forceHandoffCall ? 'she named the agent (tier 1)' : 'she approved a brief and said send it'}`);
          let toolData;
          try {
            toolData = await postRound(forceThisRound);
          } catch (err) {
            // A REFUSAL AND A STALL ARE DIFFERENT ANSWERS. `err.status` means the
            // engine answered and said no: retry unforced, or give up on tools and
            // let the turn finish. No status means the call never completed —
            // wedged engine, killed by one of the two limits above — and that must
            // propagate to the chat error handler, which turns it into a 502
            // carrying the watchdog's account. Swallowing it here would produce a
            // reply written with no tools and no sign anything went wrong.
            if (!err.status) throw err;
            if (forceThisRound) {
              console.warn(`MCP [ollama]: engine refused the forced tool_choice (${err.status}) — retrying this round unforced`);
              try {
                toolData = await postRound(false);
              } catch (err2) {
                if (!err2.status) throw err2;
                console.error(`MCP [ollama]: Tool call request failed with ${err2.status}`);
                break;
              }
            } else {
              console.error(`MCP [ollama]: Tool call request failed with ${err.status}`);
              break;
            }
          }
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
          // `user` for the same reason as the llamacpp/vllm path above: this nudge
          // has to still be the last thing before generation after the fold.
          ollamaMessages.push({
            role: 'user',
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
          messages: prepareOutboundMessages(ollamaMessages, 'ollama final'),
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
          messages: prepareOutboundMessages(enhancedMessages, 'grok')
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
          messages: prepareOutboundMessages(enhancedMessages, 'openai')
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
          messages: prepareOutboundMessages(enhancedMessages, 'squatchserve')
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

          // Tier 1 only, first round only: pin the call. A named tool_choice is
          // OpenAI-shaped and vLLM honours it; Ollama's /api/chat ignores an
          // unknown field, which is why the backstop after the reply exists and
          // is not optional. If the engine REFUSES the request outright, the same
          // round is retried without the pin rather than losing the turn — a
          // forced call is worth having, and it is not worth a failed reply.
          // TWO THINGS CAN BE PINNED, never both: a tool_choice names one
          // function. Handoff wins if somehow both fire, because she named the
          // mechanism explicitly there.
          const forcedToolName = forceHandoffCall ? 'start_background_job'
            : forceCodingCall ? 'dispatch_coding_job' : null;
          const forceThisRound = !!forcedToolName && round === 0;
          // STREAMED SO THE DEADLINE CAN SEE PROGRESS — see the ollama round
          // above. streamChat hands back the non-streaming shape, so `toolData`
          // below is byte-identical in structure to what `stream: false` gave.
          const ct = chatTimeouts();
          const postRound = (forced) => memoryManager.streamChat({
            url: `${llamacppHost}/v1/chat/completions`,
            openAiStyle: true,
            body: {
              model,
              // Folded per round, not once up front: the loop keeps appending to
              // llamacppMessages (assistant tool_calls, tool results, and the
              // post-tool nudge), so the working history stays as built and only
              // the wire format is canonicalised.
              messages: prepareOutboundMessages(llamacppMessages, `${providerLabel} tool-round ${round + 1}`),
              tools,
              ...(forced ? { tool_choice: { type: 'function', function: { name: forcedToolName } } } : {}),
              ...generationParams(providerType)
            },
            firstTokenMs: ct.firstTokenMs,
            stallMs: ct.stallMs,
            label: `${providerLabel} tool-round ${round + 1}`
          });

          if (forceThisRound) console.log(`MCP [${providerLabel}]: FORCING ${forcedToolName} this round — ${forceHandoffCall ? 'she named the agent (tier 1)' : 'she approved a brief and said send it'}`);
          let toolData;
          try {
            toolData = await postRound(forceThisRound);
          } catch (err) {
            // A refusal (`err.status` — the engine answered and said no) is
            // recoverable here. A stall is not ours to swallow: it propagates to
            // the chat error handler, which answers 502 with what the watchdog
            // knows. See the ollama round for the full reasoning.
            if (!err.status) throw err;
            if (forceThisRound) {
              console.warn(`MCP [${providerLabel}]: engine refused the forced tool_choice (${err.status}: ${String(err.body || '').substring(0, 120)}) — retrying this round unforced`);
              try { factExtractor.appendToOpsLog(`The engine refused a forced start_background_job (HTTP ${err.status}). The round was retried without it; the tier-1 backstop still applies.`, db.getOpsDir()); } catch { /* console is the floor */ }
              try {
                toolData = await postRound(false);
              } catch (err2) {
                if (!err2.status) throw err2;
                console.error(`MCP [${providerLabel}]: Tool call request failed with ${err2.status}:`, String(err2.body || '').substring(0, 200));
                break;
              }
            } else {
              console.error(`MCP [${providerLabel}]: Tool call request failed with ${err.status}:`, String(err.body || '').substring(0, 200));
              break;
            }
          }
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
      // ROLE user, NOT system, AND THAT IS THE WHOLE POINT.
      //
      // foldSystemMessages moves every system message into the leading block, so
      // a nudge pushed as `system` here stops being trailing — it lands in front
      // of the conversation and the model reaches the generation point with a
      // tool result as the last thing it saw. Measured on Gemma against this
      // engine, 6 runs per shape, the reply to a memory question after a tool
      // round:
      //
      //   all-trailing (pre-fold)          0/6 opened a thought channel
      //   folded, this nudge left trailing 0/6
      //   fully folded                     6/6   "thought\nI remember that you..."
      //   fully folded, no read guard      6/6   (so it is this nudge, not the guard)
      //   folded + this nudge as `user`    0/6
      //
      // The engine runs --tool-call-parser gemma4 with no --reasoning-parser, so
      // the channel marker lands in `content` and is spoken to the user, saved to
      // the transcript and embedded. `user` is what keeps the instruction at the
      // generation point while still satisfying the rule foldSystemMessages
      // exists for — Qwen3's template raises on any system message that is not
      // messages[0], and says nothing about a trailing user message.
      if (toolsUsed) {
        llamacppMessages.push({
          role: 'user',
          content: usedSources.length
            ? 'Tool calls are complete. Answer using the SOURCES above. For each specific fact you state, cite the [S#] link it came from, and include the relevant website links in your answer. Any specific number, date, or claim not backed by a source must be hedged or left out — do not invent specifics, and never attribute a claim to a source that does not contain it.'
            : 'Tool calls are complete. Now provide your response to the user based on the information gathered.'
        });
      }

      const finalBody = {
        model,
        stream: true,
        messages: prepareOutboundMessages(llamacppMessages, `${providerLabel} final`),
        ...generationParams(providerType)
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
      const upstreamErr = new Error(`Provider returned ${response.status}: ${errBody.substring(0, 200)}`);
      // Marked rather than string-matched in the handler: the status this turns
      // into is a decision about WHERE the failure was, and that is known here
      // and nowhere else.
      upstreamErr.upstream = true;
      throw upstreamErr;
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
    // The thinking channel, kept strictly apart from the answer. It is measured
    // and logged, forwarded to the browser to show or hide, and never saved as
    // the assistant's message — what SNH said is the answer, not the working out.
    let fullReasoning = '';
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
    // TWO LIMITS, NOT ONE — the same split the tool rounds and the background
    // path use. A flat 90s here meant "the answer has not started" and "the
    // answer died halfway" were the same event on the same clock, so the number
    // had to be generous enough for the first and was therefore slack for the
    // second. Before the first byte the wait is queue and prefill and silence is
    // normal; after it, silence means the engine is wedged and the socket (and
    // its engine slot) is being held for nothing.
    const streamT = chatTimeouts();
    let sawFirstByte = false;
    let stallTimer = null;
    const armStall = () => {
      if (stallTimer) clearTimeout(stallTimer);
      const limit = sawFirstByte ? streamT.stallMs : streamT.firstTokenMs;
      stallTimer = setTimeout(() => {
        console.warn(`[Chat] Upstream ${providerType} stream ${sawFirstByte
          ? `stalled >${limit}ms mid-answer`
          : `never started within ${limit}ms`} — aborting to free the engine`);
        streamAbort.abort();
      }, limit);
    };

    try {
      armStall();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        sawFirstByte = true;
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

              // Thinking arrives on its own field, interleaved with nothing else
              // — during the reasoning phase there is no `content` at all. This
              // is why an all-thinking reply used to read as a blank message.
              const reasoned = extractReasoning(data.choices?.[0]?.delta)
                || extractReasoning(data.choices?.[0]?.message);
              if (reasoned) fullReasoning += reasoned;

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

      // === PHANTOM DISPATCH: HE SAID HE STARTED A JOB, AND HE DID NOT ===
      //
      // The third member of a family this codebase keeps meeting. Cron proposals
      // claimed but never created. `write_memory` saying "I've updated my memory"
      // with no tool call. And on 2026-08-18: "I have started a background job to
      // organize and categorize everything I know about your clients", followed
      // by three paragraphs of what the report would contain. No job existed —
      // the turn had routed DIRECT and there was no tool to call. An hour later
      // she asked where it was.
      //
      // The narrow fix for that turn was the trigger list. This is the guard for
      // the CLASS, and it goes where the invariant cannot be forgotten: after the
      // reply is written, compare what it CLAIMS against what the queue actually
      // holds. Same doctrine as the ledger funnel — not "the model should not
      // claim this", but "a claim that is not true does not reach her unmarked".
      //
      // Deliberately narrow: only the assertion of a STARTED job, only in the
      // first person, only when this conversation created no row in this turn. A
      // false positive here would append a correction to a true statement, which
      // is its own kind of lie, so the patterns are the ones he actually writes.
      //
      // === AND THE TIER-1 BACKSTOP: SHE NAMED THE AGENT, SO IT GOES ===
      //
      // The forced tool_choice above is the first line and it is not a guarantee:
      // Ollama ignores the field, an engine may refuse it, and a refused pin
      // falls back to an unforced round on purpose. So the invariant is closed
      // HERE, where it is checkable rather than hoped for — if she named the
      // agent and this turn created no row, the server creates it from her own
      // words and SAYS SO in the reply she is reading.
      //
      // It runs whether or not he claimed a job. A tier-1 turn that quietly
      // answered inline is the same failure with better manners: she asked for an
      // agent, and "it goes, regardless of what the request is about" is the rule.
      // Same doctrine as the ledger funnel and the phantom guard — the outcome is
      // made true, and the truth is what reaches her.
      try {
        // THE CLAIM PATTERNS LIVE IN db/dispatch-claims.js — pure and tested
        // against the sentences he actually wrote, which is the only evidence
        // that has ever improved them.
        //
        // They were inline here, and entirely FIRST-PERSON ACTIVE, with a
        // comment calling that narrowness a virtue. On 2026-08-22 he wrote
        // "It's sent. The brief has been delivered to the coding agent." and
        // nothing fired, because he never said "I". Nothing had been
        // dispatched. The first person was never what made a claim a claim —
        // the discipline that keeps false positives down is the VERB (sent,
        // dispatched, delivered) rather than the pronoun.
        const claim = dispatchClaims.classifyDispatchClaim(fullResponse);
        const claimsDispatch = claim.claims;
        const created = agentJobs.jobsStartedInTurn(convoId, turnStartedAt);

        // === A CORRECTION IS CHROME, NOT PART OF HIS MESSAGE ===
        //
        // Every server-authored note here used to be APPENDED to the reply.
        // That is the same mechanism that taught him to forge the status line:
        // `fullResponse` is stored whole, so a correction became part of his
        // own words and came back next turn as history. He would have learned
        // to write "Correction — no job was actually started" exactly as he
        // learned to write a progress line — and a forged correction is the
        // worse object, because it is the thing she reads to decide whether to
        // believe the rest of the reply.
        //
        // `notice()` writes a frame the client renders as chrome and the
        // transcript never sees. It deliberately does NOT touch fullResponse:
        // not stored, not replayed to him, and not forgeable, because nothing
        // he can emit reaches this channel.
        //
        // There is deliberately NO append() any more. Nothing on this path
        // writes into his message: message text is the model's, corrections are
        // the system's, and the two travel on different channels so neither can
        // impersonate the other.
        const notice = (kind, text) => {
          const frame = { snh_notice: { kind, text } };
          res.write(contentType === 'text/event-stream'
            ? `data: ${JSON.stringify(frame)}\n\n`
            : `${JSON.stringify(frame)}\n`);
        };

        if (created.length === 0 && forceHandoffCall) {
          // Her message IS the task: it is what she asked for, in her words, and
          // the run sees the task and not the conversation. A model rewrite would
          // be the better prompt on a good day and a paraphrase of the wrong thing
          // on a bad one — and on a bad one is precisely when this path runs.
          const asked = String(userMessage.content || '').trim();
          const title = asked.length > 70 ? `${asked.slice(0, 67)}…` : asked;
          const started = agentJobs.enqueue({
            title: title || 'the work she asked an agent for',
            task: asked,
            why: 'She asked for an agent by name. The reply did not start one, so the server did.',
            conversationId: convoId,
            messageId: userMsgId,
            source: 'tier1-backstop'
          });

          if (started.ok) {
            const line = `TIER-1 BACKSTOP: she named the agent and the reply started no job, so one was queued from her message (${started.id.slice(0, 8)}, convo ${convoId}).`;
            console.warn(`[AgentJobs] ${line}`);
            try { factExtractor.appendToOpsLog(line + ` Task: "${asked.slice(0, 160).replace(/\s+/g, ' ')}"`, db.getOpsDir()); } catch { /* console is the floor */ }
            notice('job-queued', claimsDispatch
              // He said it; now it is true. She is told which, because "started"
              // and "started after the fact" are different facts about her turn.
              ? `Job \`${started.id.slice(0, 8)}\` is queued and running — it will land in the jobs panel. The reply above said it had started before it actually had; it is real now.`
              : `You asked for an agent, so one is queued — job \`${started.id.slice(0, 8)}\`, working from your message as written. It will land in the jobs panel when it is done; it will not message you.`);
          } else {
            // A refusal is a fact too, and the reason is the queue's, not mine.
            const line = `TIER-1 BACKSTOP REFUSED: ${started.error} (convo ${convoId}).`;
            console.warn(`[AgentJobs] ${line}`);
            try { factExtractor.appendToOpsLog(line, db.getOpsDir()); } catch { /* console is the floor */ }
            notice('nothing-started', `**No agent was started.** You asked for one and it could not be queued: ${started.error}`);
          }
        } else if (claimsDispatch && codingJobs.dispatchedInTurn(convoId, turnStartedAt) === 0
                   && mcpClient.hasTool('dispatch_coding_job')
                   && approvalClassifier.pendingBrief({ conversationId: convoId })) {
          // === THE CLAIM-KEYED BACKSTOP ===
          //
          // NOT gated on classifyCodingGoAhead any more. It was, and that was
          // the defect: the pin and the backstop read one `forceCodingCall`, so
          // a phrase miss killed both at once and the turn fell through to a
          // correction that only apologised. This keys off HIS CLAIM, which is
          // observable and already reliable, and is scoped to coding by the
          // pending brief rather than by anything either of them said.
          const retry = await forcedDispatchRound({
            host: providerHost, model, providerType,
            messages: enhancedMessages, assistantReply: fullResponse, toolContext: {
              searxngHost: getSearxngConfig().url, conversationId: convoId,
              userMessage: userMessage.content, messageId: userMsgId, inputModality,
            },
          });

          if (retry.called && retry.result && retry.result.success) {
            const line = `CLAIM BACKSTOP: he claimed a dispatch and had made none; a forced round sent it for real (${retry.args.project}, convo ${convoId}).`;
            console.warn(`[CodingJobs] ${line}`);
            try { factExtractor.appendToOpsLog(line, db.getOpsDir()); } catch { /* console is the floor */ }
            notice('sent-after-claim',
              `Your reply said the brief had gone before it had. It has now actually been sent to `
              + `**${retry.args.project}** — job \`${String(retry.result.job_id || '').slice(0, 8)}\`. `
              + `Watch the strip at the top; the write-up lands in the jobs panel.`);
          } else if (retry.called && retry.result && !retry.result.success) {
            // He called it and the TOOL refused. That is a real, recorded
            // refusal now (see logRefusal) rather than a silence.
            const line = `CLAIM BACKSTOP: forced round called the tool and it was refused: ${retry.result.error} (convo ${convoId}).`;
            console.warn(`[CodingJobs] ${line}`);
            try { factExtractor.appendToOpsLog(line, db.getOpsDir()); } catch { /* console is the floor */ }
            notice('claim-refused',
              `Your reply says the brief was sent. It was not, and when the server tried on your `
              + `behalf the dispatch was refused: ${String(retry.result.error).split(/(?<=\.)\s/)[0]} `
              + `Nothing is running.`);
          } else {
            // Last resort: the server infers, and says plainly when it cannot.
            const sent = codingJobs.backstopDispatch({
              conversationId: convoId, messageId: userMsgId, userMessage: userMessage.content,
            });
            if (sent.ok) {
              const line = `CLAIM BACKSTOP (inferred): ${retry.reason}; the server sent the brief to ${sent.project} (${String(sent.id).slice(0, 8)}, convo ${convoId}).`;
              console.warn(`[CodingJobs] ${line}`);
              try { factExtractor.appendToOpsLog(line, db.getOpsDir()); } catch { /* console is the floor */ }
              notice('sent-after-claim',
                `Your reply said the brief had gone before it had. The server sent it to `
                + `**${sent.project}** — job \`${String(sent.id).slice(0, 8)}\`. A restore point was `
                + `committed first, so it can be undone.`);
            } else {
              const line = `CLAIM BACKSTOP FAILED: ${retry.reason}; inference declined: ${sent.reason} (convo ${convoId}).`;
              console.error(`[CodingJobs] ${line}`);
              try { factExtractor.appendToOpsLog(line, db.getOpsDir()); } catch { /* console is the floor */ }
              notice('nothing-sent',
                `**Nothing was sent.** The reply says it was; no job exists. The server tried to send `
                + `it and could not: ${sent.reason}. Tell it which project this belongs to and it will go.`);
            }
          }
        } else if (claimsDispatch && created.length === 0) {
          const line = `PHANTOM DISPATCH: the reply claims a background job was started, and no agent_jobs row was created in this turn (convo ${convoId}).`;
          console.error(`[AgentJobs] ${line}`);
          try { factExtractor.appendToOpsLog(line + ` Reply began: "${fullResponse.slice(0, 160).replace(/\s+/g, ' ')}"`, db.getOpsDir()); } catch { /* console is the floor */ }

          // The correction is APPENDED TO THE TURN, not swallowed into a log,
          // because she is the one who was told the false thing and she is
          // reading this reply, not the ops ledger.
          notice('no-job', '**No job was actually started.** The reply above says work was handed to a '
            + 'background agent; that did not happen. Nothing is running, and nothing will appear in the '
            + 'jobs panel for it.');
        } else if (claimsDispatch) {
          console.log(`[AgentJobs] dispatch claim checks out — ${created.length} job(s) created in this turn`);
        }
        // === THE LIVE STATUS LINE IS NOT WRITTEN HERE ANY MORE ===
        //
        // It used to be: statusBlock() was APPENDED to the reply so she could
        // see a running job in the conversation she was already reading. That
        // was the right instinct and the wrong channel, and it produced a
        // forgery within a day.
        //
        // Appending it put the block INSIDE his message. The message is stored
        // whole, so the block became part of his own words, and on the next
        // turn it came back to him as conversation history — his own transcript
        // teaching him the format. An hour after the first real one he wrote,
        // unprompted and with nothing running:
        //
        //     _squatch-code, working:_
        //     - **squatch_crawler** · step 1/25 · run_command update_brief_v1.1 · 1m45s
        //
        // She caught it only because `update_brief_v1.1` is not a real command.
        // The line that existed to tell a real job from a story had become a
        // thing the story could contain.
        //
        // So live status is UI CHROME, not message text: #coderStrip polls
        // /api/jobs/coding/active and renders it outside the transcript, where
        // he cannot write and cannot read it back. Nothing this server puts in
        // message text ever looks like a status line again — which is precisely
        // what makes the check below meaningful. Having removed the real one,
        // ANY occurrence in a reply is fabricated.
        //
        // Do not re-add an append here. If she wants it in the transcript
        // again, it has to arrive as its own frame the client renders as
        // chrome, never as characters in his message.
        try {
          const forged = dispatchClaims.forgedStatusLine(fullResponse);
          if (forged.forged) {
            const line = `FORGED STATUS LINE: the reply contains a fabricated live-progress line (convo ${convoId}).`;
            console.error(`[CodingJobs] ${line}`);
            try { factExtractor.appendToOpsLog(line + ` Pattern ${forged.pattern}. Reply excerpt: "${fullResponse.slice(0, 200).replace(/\s+/g, ' ')}"`, db.getOpsDir()); } catch { /* console is the floor */ }
            notice('forged-status', '**The progress line in that reply was written by the model, not by the system.** '
              + 'It has no way to show live job status in a reply — the real one is the strip at the top of '
              + 'the window, from the job queue itself. Treat that line as invented.');
          }
        } catch (statusErr) {
          console.error('[CodingJobs] forged-status check failed:', statusErr.message);
        }
      } catch (phantomErr) {
        console.error('[AgentJobs] phantom-dispatch check failed:', phantomErr.message);
      }

      // THE MODEL THOUGHT AND NEVER ANSWERED.
      //
      // Same family as the artifact case above, and it cost the same thing: a
      // reply that is entirely reasoning leaves `fullResponse` empty, the save
      // below is guarded on truthiness, so nothing was stored and the browser
      // rendered a blank turn. Measured 2026-08-15: 8,062 characters of thinking,
      // zero of answer, finish_reason `stop`. Nothing in the logs said so.
      //
      // thinking_token_budget makes this rare rather than impossible, so the
      // honest sentence stays. An empty message reads as answered.
      if (!fullResponse.trim() && fullReasoning.trim()) {
        const msg = 'I thought about that but never actually wrote an answer — my reasoning ran to the end of its budget without producing a reply. Ask me again and I should get there.';
        console.warn(`[Chat] reply was ${fullReasoning.length} chars of reasoning and no answer — sending the honest note instead`);
        fullResponse = msg;
        res.write(contentType === 'text/event-stream'
          ? `data: ${JSON.stringify({ choices: [{ delta: { content: msg } }] })}\n\n`
          : `${JSON.stringify({ message: { content: msg } })}\n`);
      }

      if (fullReasoning) {
        console.log(`[Chat] reasoning ${fullReasoning.length} chars / answer ${fullResponse.length} chars`);
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
      // WHAT IT WAS, AND WHAT TO SAY, ARE BOTH IN db/chat-failure.js — pure,
      // and tested there. This handler's job is to fetch the one piece of live
      // state the words depend on and send the result.
      //
      // 502 vs 500 is the same distinction it always was: upstream failure is
      // Bad Gateway, anything else reaching here is our own bug. What changed on
      // 2026-08-22 is that a TIMEOUT counts as upstream — it did not, and that
      // is why a wedged engine answered 500 with a bare abort string.
      const { classifyChatFailure, chatFailureBody } = require('./db/chat-failure');
      const verdict = classifyChatFailure(error);

      let brain = null;
      if (verdict.upstream) {
        try {
          brain = require('./db/brain-watchdog').brainStatus();
        } catch (statusErr) {
          // A status lookup must never replace the error it explains.
          console.error('[Watchdog] brainStatus failed:', statusErr.message);
        }
      }

      res.status(verdict.status).json(chatFailureBody(error, brain));
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

app.listen(PORT, HOST, () => {
  console.log(`Server running on http://${HOST}:${PORT}`);
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
  // Search says the ORDER and whether each provider can actually be called — the
  // question that cost hours on 2026-08-18 was "which one ran", and the answer
  // starts at boot.
  {
    const chain = getSearchConfig();
    console.log('Web search providers (in order):');
    for (const p of chain.providers) {
      console.log(`  - ${p.name}: ${p.available ? 'available' : `UNAVAILABLE — ${p.why}`}` +
        (p.name === 'searxng' ? ` (${p.config.url})` : ''));
    }
    if (!chain.any) console.log('  - none available — web_search is not registered');
    // Where secrets are held, and how. Printed because "encrypted at rest" is a
    // claim, and a claim about security should be checkable from the log of the
    // process making it.
    const sh = require('./db/secrets').health();
    console.log(`  - secrets: ${sh.stored} stored, ${sh.algorithm}, key ${sh.keySource === 'env' ? 'from SNH_SECRET_KEY' : sh.keySource === 'file' ? 'on this machine' : 'not created yet'}`);
  }
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
  // Ask the engine what window it is actually serving, rather than guessing from
  // the model's name. Async so a slow or absent engine cannot delay the boot.
  require('./db/model-context').startupProbe();
  console.log('  - Memory clusters: Associative cluster-aware retrieval');
  console.log(`  - Long-term memory: rendered from SQLite per request`);
  console.log(`  - Memory files: data/memory/ (USER.md, daily/)`);
  // WHICH BUILD IS ACTUALLY RUNNING. Twice on 2026-08-21 a fix was made,
  // committed, and then debugged for a while against a process still on
  // the previous build. A line naming the commit at boot answers
  // "is the live process current?" without inferring it from timestamps.
  try {
    const { execSync } = require('child_process');
    const opts = { cwd: __dirname, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] };
    const sha = execSync('git rev-parse --short HEAD', opts).trim();
    const subject = execSync('git log -1 --format=%s', opts).trim();
    const dirty = execSync('git status --porcelain', opts).trim() ? ' +uncommitted changes' : '';
    console.log(`  - Build: ${sha}${dirty} — ${subject}`);
  } catch (_) {
    console.log('  - Build: unknown (not a git checkout)');
  }
  console.log(`  - MCP tools: ${mcpClient.hasTools() ? mcpClient.getToolNames().join(', ') : 'None'}`);
  console.log(`  - Memory heartbeat: ${startupConfig.heartbeat.enabled ? `Every ${startupConfig.heartbeat.intervalHours}h (first run in ${startupConfig.heartbeat.warmupMinutes}min)` : 'Disabled'}`);
  if (ALLOWED_OLLAMA_HOSTS.length > 0) {
    console.log(`  - Additional Ollama hosts: ${ALLOWED_OLLAMA_HOSTS.join(', ')}`);
  }
  memoryManager.startHeartbeat();
  memoryManager.startLivenessProbe();
  memoryManager.startScheduler();

  // The agent-job queue's recovery pass, beside the scheduler's own.
  //
  // A restart kills an in-flight run — an LLM call cannot be resumed — so this
  // closes every row a restart left open, WITH THE REASON, and re-queues the
  // ones still young enough to be worth redoing. Queued rows are the easy half:
  // they are just rows, so they survived, and they only need launching again.
  // Without this a killed job would read as in-flight forever, which is the
  // silent loss the whole table exists to refuse.
  try {
    const jobs = agentJobs.startup();
    console.log(`  - Agent jobs: ${startupConfig.agentJobs?.enabled === false ? 'Disabled' : 'Enabled'}` +
      ` (${jobs.closed} interrupted by the last restart, ${jobs.requeued} re-queued, ${jobs.resumed} launched)`);
  } catch (e) {
    console.error('[AgentJobs] startup failed:', e.message);
  }

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
