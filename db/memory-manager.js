/**
 * Memory Manager Heartbeat
 * Scheduled background job that maintains the memory system.
 *
 * The cluster pipeline runs ONLY when a cluster is oversized; on an idle memory
 * the pass does no model work at all. (Before 2026-08-02 the cross-link audit ran
 * unconditionally and dominated pass duration; it and cleanupFacts are gone —
 * see the tombstones below.)
 *   Step 1: auditClusterCoherence — per-cluster LLM coherence check, flags splits
 *   Step 2: executeSplits         — apply flagged splits, re-embed moved facts
 *   Step 3: generateReport        — build report object, log to console + ops file
 *   Task B2: sweepPendingQuestions — retire pending questions memory already answers
 *   Task C: summarizeDailyLogs    — archive daily logs older than retention window
 */

const fs = require('fs');
const path = require('path');
const { randomUUID, createHash } = require('crypto');
const { getConfig, getProviderInstance } = require('./config');
const { getCurrentDateTimeString, getLocalDateStamp } = require('./datetime');
const reasoningChannel = require('./reasoning-channel');

const { getSqliteDb, getClusterEmbeddingsTable } = require('./database');
const memoryClusters = require('./memory-clusters');
const factExtractor = require('./fact-extractor');
const agentPool = require('./agent-pool');
const initiativeEngine = require('./initiative-engine');
const selfAudit = require('./self-audit');
const brainWatchdog = require('./brain-watchdog');

const MEMORY_DIR = require('./database').getMemoryDir();
const DAILY_DIR = path.join(MEMORY_DIR, 'daily');
// Operational events (liveness/heartbeat failures, circuit-breaker trips,
// maintenance-pass telemetry) go here — surfaced in the Thinking tab, never
// injected into chat context. Keeps the daily log cognitively meaningful.
const OPS_DIR = path.join(MEMORY_DIR, 'ops');
const ARCHIVE_DIR = path.join(DAILY_DIR, 'archive');

let heartbeatTimer = null;
let warmupTimer = null;
let livenessTimer = null;
let schedulerTimer = null;
// Start optimistic — only write a daily-log warning on the transition from
// answering to not-answering (and a recovery note on the way back), so a wedged
// engine produces one alert rather than a warning every probe interval.
let lastLivenessOk = true;
let isRunning = false;

// Mid-cycle circuit breaker. The preflight probe in runMaintenance catches a
// brain that's already down when a cycle starts; this catches one that wedges
// PART WAY THROUGH. After this many consecutive callLLM timeouts the circuit
// opens: subsequent callLLM calls fast-fail (so an in-flight pass — e.g. a
// 231-pair cross-link audit — drains in milliseconds instead of grinding every
// remaining task against a dead engine), and runMaintenance aborts the cycle.
// Any successful call or a successful liveness probe closes it again.
const CIRCUIT_TIMEOUT_THRESHOLD = 3;
let consecutiveTimeouts = 0;
let circuitOpen = false;

function isTimeoutError(err) {
  return !!err && (err.name === 'TimeoutError' || err.name === 'AbortError' || /abort|timeout/i.test(err.message || ''));
}

/** Reset the mid-cycle breaker — brain is reachable again. */
function closeCircuit() {
  consecutiveTimeouts = 0;
  circuitOpen = false;
}
// Serializes reflection so a manual "Reflect now" can't run concurrently with a
// scheduled heartbeat cycle (or another manual trigger). Both paths call
// runReflection, which advances the watermark only at the END of a cycle — so
// without this lock two overlapping runs both read the same old watermark, both
// review the same conversations, and both store facts + queue followups (the
// "reflection stutter"). The check+set is atomic in Node (no await between them).
let isReflecting = false;

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Safely build a LanceDB delete filter for member_id, validating UUID format first */
function memberIdFilter(id) {
  if (!UUID_RE.test(id)) throw new Error(`Invalid member_id format: ${id}`);
  return `member_id = "${id}"`;
}

// ============ Background tool budget ============

/**
 * A per-step tool budget.
 *
 * Created by runStep when a step declares a tool allowlist, and threaded into
 * callLLM. Nothing declares one today: this is the scaffold the corrector
 * (Phase 2c) inherits, built now so the first consumer is not also the thing
 * that invents its own bounds.
 *
 * Two limits, both enforced. A call cap alone lets a step spend twenty minutes
 * on three slow lookups; a wall clock alone lets a fast loop make two hundred
 * calls. When either binds, the step keeps running — it just stops being offered
 * tools — and the fact that it was cut short is recorded rather than inferred
 * from a suspiciously thin result.
 */
function createToolSession(stepName, allowedTools = [], overrides = {}) {
  const cfg = (getConfig().heartbeat && getConfig().heartbeat.toolBudget) || {};
  return {
    stepName,
    allowedTools,
    // Overrides exist for the corrector, which legitimately makes far more calls
    // than any other background role. A single shared number would either starve
    // it or over-grant everything else, so its budget lives in corrector.* and is
    // handed in here — one session, one set of limits, no second accounting.
    maxCalls: Math.max(1, overrides.maxCalls ?? cfg.maxCallsPerStep ?? 12),
    maxWallMs: Math.max(1000, overrides.maxWallMs ?? cfg.maxWallClockMsPerStep ?? 120000),
    maxRounds: Math.max(1, overrides.maxRounds ?? cfg.maxRoundsPerCall ?? 5),
    calls: 0,
    startedMs: Date.now(),
    exhaustedReason: null,

    /** @returns {string|null} the reason the budget is spent, or null */
    spent() {
      if (this.calls >= this.maxCalls) return `call budget spent (${this.calls}/${this.maxCalls} tool calls)`;
      const elapsed = Date.now() - this.startedMs;
      if (elapsed >= this.maxWallMs) return `time budget spent (${Math.round(elapsed / 1000)}s of ${Math.round(this.maxWallMs / 1000)}s)`;
      return null;
    },

    /** Record that the budget bound. Loud by construction — never silent. */
    exhaust(reason) {
      if (this.exhaustedReason) return;
      this.exhaustedReason = reason;
      const line = `Heartbeat step "${this.stepName}" hit its tool budget: ${reason}. It carried on without tools for the rest of the step.`;
      console.warn(`[Heartbeat] ${line}`);
      try { factExtractor.appendToOpsLog(line, OPS_DIR); } catch (e) { /* console line is the floor */ }
    },

    summary() {
      return {
        step: this.stepName,
        tools: this.allowedTools,
        calls: this.calls,
        maxCalls: this.maxCalls,
        elapsedMs: Date.now() - this.startedMs,
        maxWallMs: this.maxWallMs,
        exhausted: this.exhaustedReason
      };
    }
  };
}

/**
 * Execute one tool call on behalf of a background step.
 *
 * Routed through the SHARED MCP registry, so a background step calls the exact
 * same tool implementation the chat path does — the spec's rule for INSPECT is
 * "same tools, same contract", and two implementations would be two contracts.
 * The allowlist is intersected with MCPClient.BACKGROUND_TOOLS, which is
 * read-only: a background agent that can write is one that can change what the
 * entity believes about itself with nobody in the room.
 */
async function executeBackgroundTool(session, name, args) {
  const MCPClient = require('../mcp/mcp-client');
  const client = MCPClient.shared();
  if (!session.allowedTools.includes(name)) {
    return { error: `Tool "${name}" is not available to this background step.` };
  }
  session.calls++;
  try {
    return await client.executeTool(name, args, { caller: `heartbeat:${session.stepName}` });
  } catch (err) {
    return { error: `Tool execution failed: ${err.message}` };
  }
}

/**
 * Strip channel/control markers from a tool-loop response.
 *
 * Observed on the live engine (vLLM serving Gemma-4-26B-A4B-NVFP4, 2026-08-03):
 * a plain callLLM returns "OK.", but the same model answering after a tool call
 * returns "<|channel>thought\n<channel|>I hold 6 active facts…". The markers
 * appear only on the tool path, so nothing upstream ever had to handle them —
 * and the corrector will be parsing this content, where a stray control token is
 * the difference between a parsed verdict and a skipped one.
 *
 * Deliberately narrow: only angle-bracket control tokens carrying a pipe. Real
 * prose does not contain "<|…>" and a broader strip would eat comparisons.
 */
function stripChannelMarkers(text) {
  return String(text || '')
    // The whole header span, name included: "<|channel>thought\n<channel|>".
    // Stripping only the two markers would leave the bare channel name
    // ("thought") glued to the front of the answer, which reads as content.
    .replace(/<\|channel>[\s\S]{0,40}?<channel\|>/g, '')
    // Any remaining lone control token.
    .replace(/<\|[^>|]*\|?>/g, '')
    .replace(/<[^<>|\s]*\|>/g, '')
    .trim();
}

/**
 * The background tool loop.
 *
 * Same shape as the chat path's loop, deliberately: model turn → tool calls →
 * results appended → model turn again, until it stops asking or the budget
 * binds. Kept here rather than shared with server.js because the two differ in
 * every way that matters — no streaming, no per-tool sub-caps, no user-visible
 * source cards — and a shared abstraction would have to be told which of those
 * it was doing on every call.
 *
 * When the budget binds mid-loop the tools are withdrawn and ONE more turn runs
 * without them, so the step gets an answer built from what it managed to look up
 * rather than nothing at all.
 */
async function runToolLoop({ session, openAiStyle, url, body, messages, timeoutMs, providerName }) {
  const MCPClient = require('../mcp/mcp-client');
  const client = MCPClient.shared();
  const specs = client.getToolsForOpenAISubset(session.allowedTools);
  const convo = [...messages];
  const toolCalls = [];

  if (specs.length === 0) {
    session.exhaust('no allowed tools are registered');
  }

  for (let round = 0; round < session.maxRounds; round++) {
    const spentReason = session.spent();
    if (spentReason) session.exhaust(spentReason);
    const offerTools = specs.length > 0 && !session.exhaustedReason;

    const roundBody = { ...body, messages: convo };
    if (offerTools) roundBody.tools = specs;

    console.log(`[Heartbeat] ${session.stepName} tool round ${round + 1}/${session.maxRounds}` +
                `${offerTools ? ` (${specs.length} tool(s) offered, ${session.calls}/${session.maxCalls} used)` : ' (no tools — budget spent)'}`);

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roundBody),
      signal: AbortSignal.timeout(timeoutMs)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    const msg = openAiStyle ? (data.choices?.[0]?.message || {}) : (data.message || {});
    const finishReason = openAiStyle ? (data.choices?.[0]?.finish_reason || '') : (data.done_reason || '');
    const requested = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];

    if (requested.length === 0 || !offerTools) {
      closeCircuit();
      return {
        content: stripChannelMarkers(msg.content),
        provider: providerName,
        truncated: finishReason === 'length',
        toolCalls,
        budget: session.summary()
      };
    }

    convo.push(msg);
    for (const call of requested) {
      const name = call.function?.name;
      // OpenAI-style providers send arguments as a JSON STRING; Ollama sends an
      // object. Handle both rather than assuming, because "arguments is a
      // string" is exactly the sort of thing that differs per engine build.
      let args = call.function?.arguments;
      if (typeof args === 'string') {
        try { args = JSON.parse(args); } catch { args = {}; }
      }
      args = args || {};

      const spent = session.spent();
      if (spent) {
        session.exhaust(spent);
        convo.push({
          role: 'tool', tool_call_id: call.id, name,
          content: JSON.stringify({ error: `Not run — ${spent}. Answer with what you already have and say you could not look further.` })
        });
        continue;
      }

      const result = await executeBackgroundTool(session, name, args);
      toolCalls.push({ name, args, ok: !result?.error });
      convo.push({ role: 'tool', tool_call_id: call.id, name, content: JSON.stringify(result) });
    }
  }

  // Rounds exhausted with the model still asking. Return what it last said
  // rather than throwing — a step that looked things up and ran out of rounds
  // has done real work.
  session.exhaust(`round budget spent (${session.maxRounds} rounds)`);
  closeCircuit();
  return {
    content: '', provider: providerName, truncated: false,
    toolCalls, budget: session.summary()
  };
}

// ============ LLM Helper ============

/**
 * Call an LLM with system + user prompts.
 * Uses the heartbeat model/provider from config.
 *
 * TOOLS (2026-08-03). Pass `options.toolSession` — from createToolSession — to
 * run this call as a tool loop. Without it the request body is exactly what it
 * has always been: no `tools` key on either provider branch, which is why no
 * background role could call anything. Every existing caller omits it, so every
 * existing step is byte-identical to before.
 *
 * @param {string} systemPrompt
 * @param {string} userPrompt
 * @param {Object} [options]
 * @param {number} [options.maxTokens]
 * @param {Object} [options.toolSession] - per-step budget from createToolSession
 * @returns {Promise<{content: string, provider: string, truncated: boolean, toolCalls?: Array}>}
 */
async function callLLM(systemPrompt, userPrompt, options = {}) {
  // Fast-fail while the mid-cycle breaker is open — the brain is wedged, so
  // don't spend another full timeout piling a doomed request onto a dead engine.
  if (circuitOpen) {
    throw new Error('Brain circuit open — skipping LLM call (engine wedged)');
  }

  const config = getConfig();
  const heartbeatModel = config.models.heartbeat;
  const inst = getProviderInstance(heartbeatModel.provider, heartbeatModel.instance);
  const host = inst ? inst.host : 'http://localhost:11434';
  const maxTokens = options.maxTokens ?? 1024;
  // Date/time awareness for all heartbeat/audit roles (single shared injection).
  const datedSystemPrompt = `${getCurrentDateTimeString()}\n\n${systemPrompt}`;
  const messages = [
    { role: 'system', content: datedSystemPrompt },
    { role: 'user', content: userPrompt }
  ];

  // Scale fetch timeout to token budget: max_tokens / 45 tok/s * 1000ms * 2x safety margin
  const timeoutMs = Math.max(60000, Math.ceil(maxTokens / 45 * 1000 * 2));

  // Build provider call based on config
  // THE CALLER'S maxTokens IS THE ANSWER BUDGET, AND ON A REASONING MODEL THAT
  // IS NOT THE WHOLE BILL.
  //
  // Every call site here sizes for the reply — 8 for a claim-type tag, 100 for a
  // gap question, 120 for a salience score. A reasoning model spends that budget
  // on thinking first and never reaches the answer, which is why salience and gap
  // detection returned "" on every run. The thinking allowance is therefore ADDED
  // to what the caller asked for rather than taken out of it: no call site
  // changes meaning, and none of the twenty of them need editing.
  //
  // null (the shipped default) sends neither field, so a non-reasoning box gets
  // byte-identical requests. vLLM extension, so OpenAI-style local engines only.
  const gen = config.generation || {};
  const bgThinking = Number.isFinite(gen.backgroundThinkingTokens) ? gen.backgroundThinkingTokens : null;
  const wireMaxTokens = bgThinking > 0 ? maxTokens + bgThinking : maxTokens;

  let url, body, extract, extractFinishReason;
  if (['llamacpp', 'vllm'].includes(heartbeatModel.provider)) {
    url = `${host}/v1/chat/completions`;
    body = { messages, stream: false, max_tokens: wireMaxTokens };
    if (bgThinking > 0) body.thinking_token_budget = bgThinking;
    if (gen.reasoningEffort) body.reasoning_effort = gen.reasoningEffort;
    extract = (data) => data.choices?.[0]?.message?.content || '';
    extractFinishReason = (data) => data.choices?.[0]?.finish_reason || '';
  } else {
    url = `${host}/api/chat`;
    body = { model: heartbeatModel.model, messages, stream: false, options: { num_predict: maxTokens } };
    extract = (data) => data.message?.content || '';
    extractFinishReason = (data) => data.done_reason || '';
  }

  // === Tool loop, when a step declared one. Everything below this block is the
  // === original single-shot path, byte-for-byte, and that is what every
  // === existing caller still runs: no toolSession, no `tools` key, no change.
  if (options.toolSession) {
    return runToolLoop({
      session: options.toolSession,
      openAiStyle: ['llamacpp', 'vllm'].includes(heartbeatModel.provider),
      url, body, messages, timeoutMs,
      providerName: `${heartbeatModel.provider}/${heartbeatModel.model}`
    });
  }

  const providers = [
    {
      name: `${heartbeatModel.provider}/${heartbeatModel.model}`,
      url,
      body,
      extract,
      extractFinishReason
    }
  ];

  let lastError = null;

  for (const provider of providers) {
    try {
      console.log(`[Heartbeat] Trying ${provider.name} → ${provider.url} (max_tokens: ${maxTokens}, timeout: ${timeoutMs}ms)`);
      const response = await fetch(provider.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(provider.body),
        signal: AbortSignal.timeout(timeoutMs)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const content = provider.extract(data);
      // The thinking channel, read through the one shared reader. It is never
      // folded into content — it is returned so a caller can show it, and named
      // in the error below so a model that thought instead of answering says so.
      const reasoning = reasoningChannel.reasoningFromResponse(data);
      const finishReason = provider.extractFinishReason(data);
      const truncated = finishReason === 'length';

      if (truncated) {
        console.warn(`[Heartbeat] WARNING: ${provider.name} finish_reason: "length" — response truncated at max_tokens (${wireMaxTokens}). Response: ${content.length} chars`);
      }

      if (content) {
        console.log(`[Heartbeat] ${provider.name} responded (${content.length} chars${reasoning ? `, ${reasoning.length} chars reasoning` : ''}, finish_reason: ${finishReason || 'n/a'})`);
        closeCircuit(); // a real response means the engine is alive
        return { content, reasoning, provider: provider.name, truncated };
      }
      // ALL THINKING, NO ANSWER. Distinguished from a genuinely empty reply,
      // because the two need opposite fixes and used to be the same message:
      // this one is a budget that the reasoning consumed before the answer
      // started, and it is fixed in config, not by retrying.
      if (reasoning) {
        throw new Error(
          `Model spent the whole budget reasoning and produced no answer ` +
          `(${reasoning.length} chars of reasoning, max_tokens ${wireMaxTokens}, finish_reason ${finishReason || 'n/a'}). ` +
          `Raise memory.generation.backgroundThinkingTokens.`
        );
      }
      throw new Error('Empty response');
    } catch (err) {
      // Track consecutive timeouts to trip the mid-cycle breaker. A non-timeout
      // error (e.g. HTTP 4xx/5xx) means the engine is answering, so it doesn't
      // count toward a wedge — reset the streak instead.
      if (isTimeoutError(err)) {
        consecutiveTimeouts++;
        if (consecutiveTimeouts >= CIRCUIT_TIMEOUT_THRESHOLD && !circuitOpen) {
          circuitOpen = true;
          console.warn(`[Heartbeat] Circuit opened after ${consecutiveTimeouts} consecutive timeouts — brain appears wedged; remaining calls will fast-fail`);
        }
      } else {
        consecutiveTimeouts = 0;
      }
      console.log(`[Heartbeat] ${provider.name} failed: ${err.message}`);
      lastError = err;
    }
  }

  throw new Error(`All LLM providers failed. Last error: ${lastError?.message}`);
}

/**
 * Lightweight brain liveness probe: a single tiny chat completion against the
 * heartbeat provider with a short timeout. Used by both the maintenance circuit
 * breaker (preflight) and the periodic liveness timer. Deliberately NOT routed
 * through the agent pool or callLLM's retry machinery — this is the low-level
 * check that decides whether the engine is answering at all.
 * @param {number} [timeoutMs=8000]
 * @returns {Promise<{ok: boolean, ms: number, error?: string}>}
 */
async function probeBrainLiveness(timeoutMs = 8000) {
  const config = getConfig();
  const hb = config.models.heartbeat;
  const inst = getProviderInstance(hb.provider, hb.instance);
  const host = inst ? inst.host : 'http://localhost:11434';
  const started = Date.now();

  let url, body;
  if (['llamacpp', 'vllm'].includes(hb.provider)) {
    url = `${host}/v1/chat/completions`;
    body = { model: hb.model, messages: [{ role: 'user', content: 'ping' }], stream: false, max_tokens: 1 };
  } else {
    url = `${host}/api/chat`;
    body = { model: hb.model, messages: [{ role: 'user', content: 'ping' }], stream: false, options: { num_predict: 1 } };
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs)
    });
    if (!response.ok) {
      return { ok: false, ms: Date.now() - started, error: `HTTP ${response.status}` };
    }
    // A 200 means the engine answered; the body may be empty at max_tokens 1,
    // which is still a live response.
    await response.json().catch(() => ({}));
    return { ok: true, ms: Date.now() - started };
  } catch (err) {
    const error = (err.name === 'TimeoutError' || err.name === 'AbortError')
      ? `timeout after ${timeoutMs}ms`
      : err.message;
    return { ok: false, ms: Date.now() - started, error };
  }
}

/**
 * Parse a JSON object from LLM response text.
 * Finds the last balanced JSON object (or array) in the text to avoid
 * capturing chain-of-thought braces like "the {Hardware} cluster".
 * @param {string} text - LLM response
 * @returns {Object|null}
 */
function parseJSON(text) {
  // Strip markdown code fences if present
  const stripped = text.replace(/```(?:json)?\s*/gi, '').replace(/```/g, '');

  // Find the last top-level '{' that starts a parseable JSON object (max 20 attempts)
  let attempts = 0;
  for (let i = stripped.lastIndexOf('{'); i >= 0 && attempts < 20; i = stripped.lastIndexOf('{', i - 1)) {
    attempts++;
    const candidate = stripped.slice(i);
    try {
      const parsed = JSON.parse(candidate);
      if (typeof parsed === 'object' && parsed !== null) return parsed;
    } catch { /* try earlier brace */ }
  }

  // Fallback: try last top-level '[' for array responses (max 10 attempts)
  attempts = 0;
  for (let i = stripped.lastIndexOf('['); i >= 0 && attempts < 10; i = stripped.lastIndexOf('[', i - 1)) {
    attempts++;
    const candidate = stripped.slice(i);
    try {
      const parsed = JSON.parse(candidate);
      if (Array.isArray(parsed)) return parsed;
    } catch { /* try earlier bracket */ }
  }

  return null;
}

/**
 * Repair a structurally-truncated JSON object and parse it. Local models
 * sometimes emit a well-formed opening (`{"links":[ {...}, {...},`) but stop
 * before the closing brackets — finish_reason 'stop', not 'length', so it isn't
 * a token-budget truncation, just the model deciding it was done. parseJSON()
 * can't recover that (its backward brace scan only finds inner complete objects
 * that lack the wrapper key). This walks from the first '{', tracking string
 * state and bracket depth, then closes any dangling string, drops the trailing
 * comma left by the cut, and appends the missing closers in order.
 *
 * Intended as a LAST-RESORT fallback after parseJSON() returns null — it only
 * runs on already-unparseable text, so it can recover more but never corrupt a
 * response that was already valid.
 * @param {string} text
 * @returns {object|array|null}
 */
function repairTruncatedJSON(text) {
  if (!text) return null;
  const s = text.replace(/```(?:json)?\s*/gi, '').replace(/```/g, '');
  const start = s.indexOf('{');
  if (start < 0) return null;

  const stack = [];
  let inStr = false, esc = false, closeIdx = -1;
  let lastSafe = null; // { idx, stack } right after a complete nested element closed
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{' || ch === '[') stack.push(ch);
    else if (ch === '}' || ch === ']') {
      stack.pop();
      if (stack.length === 0) { closeIdx = i; break; } // top-level object closed
      lastSafe = { idx: i, stack: stack.slice() }; // a complete element just closed
    }
  }

  const closeWith = (str, brackets) => {
    let out = str;
    for (let k = brackets.length - 1; k >= 0; k--) out += brackets[k] === '{' ? '}' : ']';
    return out;
  };
  const tryParse = (body) => {
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed === 'object' && parsed !== null) return parsed;
    } catch { /* fall through */ }
    return null;
  };

  // 1. Clean close (also discards any trailing prose the model appended).
  if (closeIdx >= 0) return tryParse(s.slice(start, closeIdx + 1));

  // 2. Truncated: shut a dangling string, drop the trailing comma, close the
  //    open brackets innermost-first.
  let body = s.slice(start);
  if (inStr) body += '"';
  body = body.replace(/[\s,]*$/, '');
  let parsed = tryParse(closeWith(body, stack));
  if (parsed) return parsed;

  // 3. The tail was an incomplete element (e.g. `"strength":` with no value).
  //    Fall back to the last point where a nested element closed cleanly, drop
  //    the partial trailing element, and close from there.
  if (lastSafe) {
    const trimmed = s.slice(start, lastSafe.idx + 1);
    parsed = tryParse(closeWith(trimmed, lastSafe.stack));
    if (parsed) return parsed;
  }

  return null;
}

// ============ Step 1: Audit Cluster Coherence ============

/**
 * Audit a single cluster for internal coherence using the LLM.
 * Returns an audit result object for that cluster, including any suggested splits.
 *
 * Designed as a self-contained unit so that future parallelization is trivial —
 * just swap the sequential loop for Promise.all().
 *
 * ACTIVE MEMBERS ONLY (2026-08-10). getCluster returns ghosts on purpose — the
 * Memory Map draws them — but showing them to the auditor asks it to reorganise
 * facts nobody believes any more. It duly proposed splits made entirely of
 * superseded rows, executeSplits refused every one (correctly: the write guard
 * is active-only), and the refusals were logged as anomalies. Identical
 * anomalies, every pass, for four days, while the two all-ghost clusters ate
 * 47s of the 69s pass. The audit is a DECISION about a cluster, so it reads the
 * live corpus; the ghost stays visible where it belongs, on the Map.
 *
 * Under two active members there is nothing to judge — one fact cannot be
 * incoherent with itself — so the cluster leaves the rotation without an LLM
 * call, and an all-ghost cluster keeps its name for the Map and goes quiet.
 *
 * @param {Object} cluster - Cluster row from getClusters() (has id, name, member_count, active_member_count)
 * @returns {Promise<{clusterId: string, clusterName: string, coherent: boolean, splits: Array, durationMs: number, skipped?: string, error?: string}>}
 */
async function auditClusterCoherence(cluster) {
  const startMs = Date.now();
  const base = { clusterId: cluster.id, clusterName: cluster.name, coherent: true, splits: [], durationMs: 0 };

  try {
    const detail = memoryClusters.getCluster(cluster.id);
    if (!detail || !Array.isArray(detail.members) || detail.members.length === 0) {
      base.durationMs = Date.now() - startMs;
      return base;
    }

    const activeMembers = detail.members.filter(m => (m.status || 'active') === 'active');
    if (activeMembers.length < 2) {
      base.durationMs = Date.now() - startMs;
      base.skipped = activeMembers.length === 0
        ? 'no active members — ghosts only'
        : 'only one active member';
      return base;
    }

    const factLines = activeMembers.map(m => {
      const ts = m.created_at ? m.created_at.split('T')[0] : 'unknown';
      const src = m.source || 'unknown';
      return `[id:${m.id}] [date:${ts}] [source:${src}] ${m.content}`;
    }).join('\n');

    const systemPrompt = `You are a memory cluster coherence auditor. Your job is to decide whether all the facts in a named cluster genuinely belong together.

Cluster name: "${cluster.name}"

Return ONLY valid JSON in this exact format:
{
  "coherent": true,
  "splits": []
}

If the cluster contains clearly distinct categories that do NOT belong under a single name, set "coherent" to false and list the splits:
{
  "coherent": false,
  "splits": [
    {
      "newClusterName": "Descriptive Category Name",
      "factIds": ["id1", "id2"]
    }
  ]
}

Rules:
- Only flag a split when the facts fall into genuinely different topics. Do NOT split if facts are loosely related to the same theme.
- Every fact id that appears in the cluster must appear in exactly one split group if you flag incoherence. Do not drop any.
- Split names should be concise noun phrases (2-4 words).
- If in doubt, return coherent: true.`;

    // Scale max_tokens: ~100 tokens per fact (40 visible JSON + ~60 model reasoning overhead) + 500 buffer
    const estOutputTokens = Math.min(12288, Math.max(1024, activeMembers.length * 100 + 500));
    const userPrompt = `Facts in cluster "${cluster.name}":\n${factLines}`;
    const { content, truncated } = await callLLM(systemPrompt, userPrompt, { maxTokens: estOutputTokens });
    let parsed = parseJSON(content);

    if (!parsed) {
      // Retry once before giving up. The usual cause is the model wrapping the
      // JSON in prose / <think> reasoning that then truncates before the closing
      // brace — so we (a) demand raw JSON only and (b) double the token budget.
      console.warn(`[Heartbeat] Parse failure for cluster "${cluster.name}" (${content.length} chars, truncated: ${truncated}) — retrying with stricter format`);
      const strictSystem = systemPrompt + `

CRITICAL OUTPUT RULE: Respond with ONLY the raw JSON object. No explanation, no reasoning, no <think> blocks, no markdown code fences, no text before or after. Your entire response must begin with { and end with }.`;
      const retryTokens = Math.min(12288, estOutputTokens * 2);
      const retry = await callLLM(strictSystem, userPrompt, { maxTokens: retryTokens });
      parsed = parseJSON(retry.content);
      if (!parsed) {
        console.warn(`[Heartbeat] Parse failure persisted after retry for "${cluster.name}" (${retry.content.length} chars, truncated: ${retry.truncated}), last 200: ...${retry.content.slice(-200)}`);
        base.durationMs = Date.now() - startMs;
        return { ...base, error: `LLM returned unparseable JSON after retry (${retry.content.length} chars, truncated: ${retry.truncated})` };
      }
      console.log(`[Heartbeat] Audit of "${cluster.name}" recovered on stricter-format retry`);
    }

    base.coherent = parsed.coherent !== false;
    base.splits = Array.isArray(parsed.splits) ? parsed.splits : [];
    base.durationMs = Date.now() - startMs;
    return base;

  } catch (err) {
    base.durationMs = Date.now() - startMs;
    return { ...base, error: err.message };
  }
}

// ============ Step 2: Execute Splits ============

/**
 * Apply all cluster splits flagged by auditClusterCoherence.
 * For each split: creates new clusters, moves facts in SQLite, re-embeds in LanceDB.
 * Runs renameAllClusters() once if any splits were applied.
 *
 * @param {Array} auditResults - Array of results from auditClusterCoherence
 * @returns {Promise<{clustersSplit: number, splitDetails: Array, anomalies: Array}>}
 */
async function executeSplits(auditResults) {
  console.log('[Heartbeat] Step 2: Executing cluster splits...');

  const results = { clustersSplit: 0, splitDetails: [], anomalies: [] };
  const db = getSqliteDb();
  if (!db) {
    results.anomalies.push('SQLite not available — skipping splits');
    return results;
  }

  const clusterTable = await getClusterEmbeddingsTable();
  const incoherentResults = auditResults.filter(r => !r.coherent && r.splits && r.splits.length > 0);

  if (incoherentResults.length === 0) {
    console.log('[Heartbeat] No splits to execute');
    return results;
  }

  console.log(`[Heartbeat] Applying splits for ${incoherentResults.length} incoherent cluster(s)`);

  // Clusters whose membership changed and therefore need a fresh name: every new
  // split-out cluster, plus any source cluster that retained facts.
  const touchedClusterIds = new Set();

  for (const auditResult of incoherentResults) {
    const { clusterId, clusterName, splits } = auditResult;

    try {
      // Verify source cluster still exists
      const sourceCluster = db.prepare('SELECT id, subject FROM memory_clusters WHERE id = ?').get(clusterId);
      if (!sourceCluster) {
        results.anomalies.push(`Source cluster ${clusterId} (${clusterName}) no longer exists, skipping splits`);
        continue;
      }
      // Split-out clusters inherit the source's subject — otherwise splitting a
      // self-cluster would create user-subject clusters (defaulting via schema),
      // leaking self-observations back into the user Facts/Clusters tabs.
      const srcSubject = sourceCluster.subject || 'user';

      const now = new Date().toISOString();
      const movedMemberIds = new Set();
      const splitDetail = { originalCluster: clusterName, newClusters: [] };

      for (const split of splits) {
        if (!split.newClusterName || !Array.isArray(split.factIds) || split.factIds.length === 0) {
          results.anomalies.push(`Invalid split spec in cluster "${clusterName}": missing name or factIds`);
          continue;
        }

        // Resolve members that actually exist in this cluster
        const membersToMove = [];
        for (const rawFactId of split.factIds) {
          // Strip "id:" prefix the LLM echoes back from the audit prompt
          const factId = rawFactId.replace(/^id:/, '');
          // ACTIVE ONLY (2026-08-03). This had no status filter, and the move
          // below deletes each member's vector and then RE-ADDS it — so a split
          // that happened to include an inactive fact resurrected its embedding
          // and put a superseded belief back into semantic retrieval. That is
          // the LanceDB-drift class the Phase 1 notes recorded as historical;
          // it was not historical, it had a live source. Found by reconcile()
          // reporting three superseded self-facts with vectors eight hours after
          // the same three had been cleared.
          //
          // memoryClusters.getCluster (which feeds the audit that proposes these
          // splits) deliberately returns inactive members too — the Memory Map
          // draws them as ghosts. So the filter belongs HERE, at the write, not
          // on the read. Same rule as the identity lock: guard the write path.
          const member = db.prepare(
            "SELECT * FROM cluster_members WHERE id = ? AND cluster_id = ? AND status = 'active'"
          ).get(factId, clusterId);
          if (member) {
            membersToMove.push(member);
          } else {
            const inactive = db.prepare(
              "SELECT id FROM cluster_members WHERE id = ? AND cluster_id = ? AND status != 'active'"
            ).get(factId, clusterId);
            results.anomalies.push(inactive
              ? `Fact id "${factId}" in cluster "${clusterName}" is inactive — not moved, and its embedding left alone`
              : `Fact id "${factId}" not found in cluster "${clusterName}"`);
          }
        }

        if (membersToMove.length === 0) {
          results.anomalies.push(`No valid facts found for split "${split.newClusterName}" in cluster "${clusterName}"`);
          continue;
        }

        // Create new cluster. The audit's newClusterName is a provisional label;
        // renameAllClusters below regenerates it from the actual moved facts via
        // the shared LLM namer, so splits get the same naming as every other path.
        const newClusterId = randomUUID();
        touchedClusterIds.add(newClusterId);
        db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?, ?, ?, ?, ?, ?)')
          .run(newClusterId, split.newClusterName, '', now, now, srcSubject);

        console.log(`[Heartbeat] Created cluster "${split.newClusterName}" (${newClusterId}), moving ${membersToMove.length} facts`);

        // Move facts
        for (const member of membersToMove) {
          db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?')
            .run(newClusterId, member.id);
          movedMemberIds.add(member.id);

          if (clusterTable) {
            try {
              await clusterTable.delete(memberIdFilter(member.id));
              const embedding = await memoryClusters.generateEmbedding(member.content);
              if (embedding) {
                await clusterTable.add([{
                  id: randomUUID(),
                  member_id: member.id,
                  cluster_id: newClusterId,
                  content: member.content,
                  vector: Array.from(embedding)
                }]);
              }
            } catch (e) {
              console.error('[Heartbeat] LanceDB re-embed error:', e.message);
              results.anomalies.push(`LanceDB re-embed failed for member ${member.id}: ${e.message}`);
            }
          }
        }

        splitDetail.newClusters.push({ name: split.newClusterName, factsCount: membersToMove.length });
      }

      // Check how many facts remain in the original cluster
      const remaining = db.prepare('SELECT COUNT(*) as cnt FROM cluster_members WHERE cluster_id = ?')
        .get(clusterId);
      const remainingCount = remaining ? remaining.cnt : 0;

      if (remainingCount === 0) {
        // All facts moved out — delete original cluster and its links
        db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(clusterId);
        console.log(`[Heartbeat] Deleted empty original cluster "${clusterName}"`);
        splitDetail.originalDeleted = true;
      } else {
        db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?').run(now, clusterId);
        touchedClusterIds.add(clusterId);
        splitDetail.originalRetained = true;
        splitDetail.originalRemainingFacts = remainingCount;
      }

      if (splitDetail.newClusters.length > 0) {
        results.clustersSplit++;
        results.splitDetails.push(splitDetail);
      }

    } catch (err) {
      console.error(`[Heartbeat] Error executing splits for cluster "${clusterName}":`, err.message);
      results.anomalies.push(`Split execution failed for "${clusterName}": ${err.message}`);
    }
  }

  if (results.clustersSplit > 0) {
    console.log(`[Heartbeat] Renaming ${touchedClusterIds.size} touched cluster(s) after ${results.clustersSplit} split(s)`);
    try {
      await memoryClusters.renameAllClusters({ ids: [...touchedClusterIds] });
    } catch (err) {
      console.error('[Heartbeat] renameAllClusters error:', err.message);
      results.anomalies.push(`renameAllClusters failed: ${err.message}`);
    }

    // Merge any clusters that ended up with the same name after renaming
    try {
      const mergedByName = await memoryClusters.mergeByName();
      if (mergedByName > 0) {
        console.log(`[Heartbeat] Merged ${mergedByName} duplicate-name cluster(s) after rename`);
      }
    } catch (err) {
      console.error('[Heartbeat] mergeByName error:', err.message);
      results.anomalies.push(`mergeByName failed: ${err.message}`);
    }
  }

  console.log(`[Heartbeat] Split execution complete: ${results.clustersSplit} cluster(s) split`);
  return results;
}
// ============ Cross-link audit — DELETED 2026-08-02 ============
//
// auditCrossLinks used to live here and ran on EVERY pass, oversized clusters or
// not. It scored how related each pair of clusters was and maintained
// cluster_links from the verdicts.
//
// Removed because the cost was O(n²) in clusters over a corpus that is O(n) in
// facts: 112 clusters = 6,216 pairs judged for 658 facts. It was the single
// biggest driver of pass duration (observed 736s at 438 pairs re-judged vs 51s
// when every pair was a cache hit), and the content-hash cache it needed was a
// mitigation of a cost that should not have existed.
//
// The tables were DROPPED at the 2026-08-06 cutover. Keeping them read-only was
// the wrong call: nothing maintained them, the replay rebuilt every active user
// fact into clusters no link had ever pointed at, and the Map went on drawing
// those edges as if they were current. A stale association is worse than none —
// it looks like knowledge. The supersede edges beside them are still live, and
// association is now computed on demand from the vector index.
//
// Associations become query-time vector neighbours in a later phase — computed
// when asked, never stored. See docs/memory-mvp-spec.md (RETRIEVE).
// ============ Step 4: Generate Report ============

/**
 * How long an anomaly may go unseen before its memo is forgotten. A condition
 * that clears and comes back weeks later is news again; one that is still true
 * on the next pass is not.
 */
const ANOMALY_STATE_TTL_DAYS = 30;

/**
 * Split this pass's anomalies into the ones worth printing and the ones already
 * on record, and update the memo.
 *
 * The corrector's gate reads its last pass off disk so a restart cannot hand it
 * a fresh turn; this is the same trick applied to reporting, so a restart cannot
 * hand an old anomaly a fresh voice either. State lives in SQLite, which means
 * it follows SNH_DATA_DIR — a staging replay memoises against staging and leaves
 * the live log's history alone.
 *
 * Fails OPEN: if the table cannot be read, every anomaly is treated as fresh.
 * Losing a warning to a bookkeeping error is worse than repeating one.
 *
 * @param {string[]} anomalies - every anomaly this pass observed
 * @returns {{fresh: string[], suppressed: number, oldestSuppressedAt: string|null}}
 */
function partitionAnomalies(anomalies) {
  if (!anomalies || anomalies.length === 0) {
    return { fresh: [], suppressed: 0, oldestSuppressedAt: null };
  }

  const db = getSqliteDb();
  if (!db) return { fresh: [...anomalies], suppressed: 0, oldestSuppressedAt: null };

  const now = new Date().toISOString();
  const fresh = [];
  let suppressed = 0;
  let oldestSuppressedAt = null;

  try {
    // Prune first, so a long-quiet anomaly is genuinely new again rather than
    // resurfacing as "seen 40 times" with a stale first_seen_at.
    const cutoff = new Date(Date.now() - ANOMALY_STATE_TTL_DAYS * 86400_000).toISOString();
    db.prepare('DELETE FROM heartbeat_anomaly_state WHERE last_seen_at < ?').run(cutoff);

    const get = db.prepare('SELECT first_seen_at, seen_count FROM heartbeat_anomaly_state WHERE anomaly_key = ?');
    const insert = db.prepare(
      'INSERT INTO heartbeat_anomaly_state (anomaly_key, first_seen_at, last_seen_at, seen_count, anomaly_text) VALUES (?, ?, ?, 1, ?)'
    );
    const bump = db.prepare(
      'UPDATE heartbeat_anomaly_state SET last_seen_at = ?, seen_count = seen_count + 1 WHERE anomaly_key = ?'
    );

    // One pass may legitimately observe the same anomaly text twice; count it
    // once so the memo tracks conditions, not occurrences.
    for (const key of new Set(anomalies.map(a => String(a)))) {
      const seen = get.get(key);
      if (seen) {
        bump.run(now, key);
        suppressed++;
        if (!oldestSuppressedAt || seen.first_seen_at < oldestSuppressedAt) {
          oldestSuppressedAt = seen.first_seen_at;
        }
      } else {
        insert.run(key, now, now, key);
        fresh.push(key);
      }
    }
  } catch (err) {
    console.error('[Heartbeat] anomaly dedup failed, reporting everything:', err.message);
    return { fresh: [...anomalies], suppressed: 0, oldestSuppressedAt: null };
  }

  return { fresh, suppressed, oldestSuppressedAt };
}

/**
 * Build a structured heartbeat report, log it to console and append to today's daily log file.
 *
 * @param {Object} opts
 * @param {number}  opts.cycleStartMs          - Date.now() at cycle start
 * @param {Array}   opts.auditResults          - Per-cluster audit result objects
 * @param {Object}  opts.splitResults          - Result from executeSplits
 * @returns {Object} The report object
 */
function generateReport({ cycleStartMs, auditResults, splitResults, steps = [] }) {
  const totalDurationMs = Date.now() - cycleStartMs;
  const totalDuration = (totalDurationMs / 1000).toFixed(1) + 's';

  // "Audited" means put to the model. A cluster that left the rotation for want
  // of two active members was not audited, and counting it as though it were is
  // how a pass reports twenty clusters reviewed while judging eighteen.
  const clustersAudited = auditResults.filter(r => !r.skipped).length;
  const clustersSkipped = auditResults.filter(r => r.skipped).length;
  const clustersSplit = splitResults.clustersSplit || 0;

  const perClusterTiming = auditResults
    .filter(r => !r.skipped)
    .map(r => ({
      clusterName: r.clusterName,
      durationMs: r.durationMs || 0
    }));

  const observedAnomalies = [
    ...auditResults.filter(r => r.error).map(r => `Audit error for "${r.clusterName}": ${r.error}`),
    ...(splitResults.anomalies || []),

  ];

  // Only what CHANGED goes in the report. The rest is counted, not repeated.
  const { fresh: anomalies, suppressed: anomaliesSuppressed, oldestSuppressedAt } =
    partitionAnomalies(observedAnomalies);
  const suppressedNote = anomaliesSuppressed > 0
    ? `${anomaliesSuppressed} unchanged anomaly(ies) still true, already reported${oldestSuppressedAt ? ` (oldest first seen ${oldestSuppressedAt.slice(0, 10)})` : ''}`
    : null;

  const report = {
    status: 'ok',
    clustersAudited,
    // Left the rotation without an LLM call — fewer than two active members.
    clustersSkipped,
    clustersSplit,
    splitDetails: splitResults.splitDetails || [],
    // Always 0 from 2026-08-02: the cross-link audit is deleted and nothing
    // maintains cluster_links. Kept in the report shape so historical rows in
    // heartbeat_reports stay comparable with new ones.
    linksUpdated: 0,
    linksRemoved: 0,
    linksAdded: 0,
    // Cross-link LLM workload — the actual driver of pass duration. See the
    // comment in auditCrossLinks: this, not the link deltas, is why a pass is
    // 55s or 17 minutes.
    pairsTotal: null,
    pairsReused: null,
    pairsJudged: null,
    // Marks a pass as running the post-cross-link pipeline, so the Activity view
    // can tell "no pairs judged because there was nothing to do" apart from
    // "no pairs judged because the step no longer exists".
    crossLinkAudit: 'deleted',
    totalDuration,
    // Numeric ms alongside the display string so the Activity view can trend
    // pass duration (the 2026-07-26 pass took 1062s — ~15% of the 2h interval).
    totalDurationMs,
    // Per-step results. These were already being computed and then dropped to
    // console: cleanupFacts, summarizeDailyLogs, sweepPendingQuestions and
    // mergeByName all return counts that nothing persisted.
    steps,
    perClusterTiming,
    // NEW anomalies only. `anomaliesObserved` is what the pass actually saw, so
    // nothing is lost — but a reader (human or Thinking tab) is shown change.
    anomalies,
    anomaliesObserved: observedAnomalies.length,
    anomaliesSuppressed,
    suppressedNote
  };

  // Console summary
  console.log('[Heartbeat] === Heartbeat Report ===');
  console.log(`[Heartbeat]   Clusters audited : ${clustersAudited}${clustersSkipped > 0 ? ` (${clustersSkipped} skipped — too few active members)` : ''}`);
  console.log(`[Heartbeat]   Clusters split   : ${clustersSplit}`);
  console.log(`[Heartbeat]   Links added      : ${report.linksAdded}`);
  console.log(`[Heartbeat]   Links updated    : ${report.linksUpdated}`);
  console.log(`[Heartbeat]   Links removed    : ${report.linksRemoved}`);
  console.log(`[Heartbeat]   Total duration   : ${totalDuration}`);
  if (anomalies.length > 0) {
    console.log(`[Heartbeat]   New anomalies (${anomalies.length}):`);
    for (const a of anomalies) console.log(`[Heartbeat]     - ${a}`);
  }
  if (suppressedNote) console.log(`[Heartbeat]   ${suppressedNote}`);
  console.log('[Heartbeat] === End Report ===');

  // Prepend to the OPS log (newest first) — this is maintenance telemetry, not
  // cognitive memory, so it stays out of the injected daily log. It remains
  // fully visible in the Thinking tab via getHeartbeatReports().
  try {
    const opsDir = OPS_DIR;
    const today = getLocalDateStamp(); // local Pacific date

    let splitSummary = '';
    if (report.splitDetails.length > 0) {
      splitSummary = '\n### Splits\n' + report.splitDetails.map(d => {
        const newNames = d.newClusters.map(c => `"${c.name}" (${c.factsCount} facts)`).join(', ');
        const fate = d.originalDeleted ? 'original deleted' : `original retained (${d.originalRemainingFacts} facts remaining)`;
        return `- "${d.originalCluster}" → ${newNames}; ${fate}`;
      }).join('\n');
    }

    // New anomalies get their line; ones already on record get one line between
    // them. Four days of identical warnings is what this replaces.
    let anomalySection = '';
    if (anomalies.length > 0) {
      anomalySection = '\n### Anomalies (new)\n' + anomalies.map(a => `- ${a}`).join('\n');
    }
    if (suppressedNote) {
      anomalySection += `\n\n_${suppressedNote}._`;
    }

    const timingRows = perClusterTiming
      .sort((a, b) => b.durationMs - a.durationMs)
      .slice(0, 10)
      .map(t => `| ${t.clusterName} | ${t.durationMs}ms |`)
      .join('\n');

    const reportBlock = [
      `## Heartbeat Report — ${new Date().toISOString()}`,
      '',
      `| Metric | Value |`,
      `|--------|-------|`,
      `| Clusters audited | ${clustersAudited} |`,
      `| Clusters skipped (too few active facts) | ${clustersSkipped} |`,
      `| Clusters split | ${clustersSplit} |`,
      `| Links added | ${report.linksAdded} |`,
      `| Links updated | ${report.linksUpdated} |`,
      `| Links removed | ${report.linksRemoved} |`,
      `| Total duration | ${totalDuration} |`,
      splitSummary,
      timingRows.length > 0 ? `\n### Per-cluster audit timing (top 10)\n| Cluster | Duration |\n|---------|----------|\n${timingRows}` : '',
      anomalySection
    ].join('\n').replace(/\s*$/, '') + '\n\n';

    // Prepend under the H1 header so the newest report is at the top.
    const opsFile = factExtractor.prependDailyEntry(reportBlock, opsDir, today, 'Ops Log');
    console.log(`[Heartbeat] Report prepended to ${opsFile}`);
  } catch (err) {
    console.error('[Heartbeat] Failed to write daily report:', err.message);
  }

  // Persist the pass stats so the Thinking view can render them per cycle.
  recordHeartbeatReport(report);

  return report;
}

/** Persist one heartbeat pass's stats to the heartbeat_reports table. */
function recordHeartbeatReport(report) {
  try {
    const db = getSqliteDb();
    if (!db) return;
    db.prepare(`
      INSERT INTO heartbeat_reports
        (id, created_at, clusters_audited, clusters_split, links_added, links_updated,
         links_removed, duration, anomaly_count, report_json, status, status_reason, duration_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', NULL, ?)
    `).run(
      randomUUID(),
      new Date().toISOString(),
      report.clustersAudited || 0,
      report.clustersSplit || 0,
      report.linksAdded || 0,
      report.linksUpdated || 0,
      report.linksRemoved || 0,
      report.totalDuration || null,
      // NEW anomalies, matching report.anomalies. The total this pass observed
      // is in report_json as anomaliesObserved — the column is the change
      // signal the Activity view trends, and a flat line of 29 is not one.
      (report.anomalies || []).length,
      JSON.stringify(report),
      report.totalDurationMs ?? null
    );
  } catch (err) {
    console.error('[Heartbeat] Failed to persist heartbeat report:', err.message);
  }
}

/**
 * Persist one liveness probe result and prune past the retention window.
 * Deliberately the cheapest possible row: when, ok, how long, why not.
 * @param {{ok: boolean, ms?: number, error?: string}} probe
 * @param {number} retentionDays
 */
function recordLivenessProbe(probe, retentionDays) {
  try {
    const db = getSqliteDb();
    if (!db) return;
    db.prepare(
      'INSERT INTO liveness_probes (id, created_at, ok, latency_ms, error) VALUES (?, ?, ?, ?, ?)'
    ).run(randomUUID(), new Date().toISOString(), probe.ok ? 1 : 0,
          Number.isFinite(probe.ms) ? probe.ms : null, probe.ok ? null : (probe.error || 'unknown'));
    // Prune inline. The table is small and created_at is indexed, so this is
    // cheaper than carrying a separate cleanup schedule for one table.
    const cutoff = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000).toISOString();
    db.prepare('DELETE FROM liveness_probes WHERE datetime(created_at) < datetime(?)').run(cutoff);
  } catch (err) {
    console.error('[Liveness] Failed to record probe:', err.message);
  }
}

/** Recent liveness probes (newest first) for the Activity view. */
function getLivenessProbes(limit = 100) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    return db.prepare(
      'SELECT * FROM liveness_probes ORDER BY datetime(created_at) DESC LIMIT ?'
    ).all(Math.min(Math.max(1, limit), 1000));
  } catch (err) {
    console.error('[Liveness] Failed to read probes:', err.message);
    return [];
  }
}

/**
 * Persist a heartbeat pass that did NOT reach the report step.
 *
 * generateReport only runs on the success path, so before this existed a pass
 * that bailed at the preflight probe (or was aborted mid-cycle by the breaker,
 * or threw) left no row — the table read as an unbroken run of healthy passes
 * through an outage. These rows are what make the Activity view trustworthy.
 *
 * @param {Object} o
 * @param {'failed'|'aborted'|'skipped'} o.status
 * @param {string} o.reason        - plain-language why
 * @param {number} o.cycleStartMs  - so a partial pass still reports its duration
 * @param {Object} [o.partial]     - whatever the pass did manage before bailing
 */
function recordHeartbeatOutcome({ status, reason, cycleStartMs, partial = {} }) {
  try {
    const db = getSqliteDb();
    if (!db) return;
    const durationMs = cycleStartMs ? Date.now() - cycleStartMs : null;
    const report = {
      status,
      statusReason: reason,
      clustersAudited: (partial.auditResults || []).length,
      clustersSplit: partial.splitResults?.clustersSplit || 0,
      linksAdded: 0,
      linksUpdated: 0,
      linksRemoved: 0,
      totalDurationMs: durationMs,
      totalDuration: durationMs != null ? (durationMs / 1000).toFixed(1) + 's' : null,
      anomalies: [reason].filter(Boolean)
    };
    db.prepare(`
      INSERT INTO heartbeat_reports
        (id, created_at, clusters_audited, clusters_split, links_added, links_updated,
         links_removed, duration, anomaly_count, report_json, status, status_reason, duration_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      randomUUID(), new Date().toISOString(),
      report.clustersAudited, report.clustersSplit, report.linksAdded,
      report.linksUpdated, report.linksRemoved, report.totalDuration,
      report.anomalies.length, JSON.stringify(report), status, reason, durationMs
    );
    console.log(`[Heartbeat] Recorded ${status} pass: ${reason}`);
  } catch (err) {
    console.error('[Heartbeat] Failed to persist heartbeat outcome:', err.message);
  }
}

/** Recent heartbeat pass stats (newest first) for the Thinking view. */
function getHeartbeatReports(limit = 30) {
  try {
    const db = getSqliteDb();
    if (!db) return [];
    const rows = db.prepare(
      'SELECT * FROM heartbeat_reports ORDER BY created_at DESC LIMIT ?'
    ).all(limit);
    return rows.map(r => {
      let full = {};
      try { full = JSON.parse(r.report_json || '{}'); } catch { /* ignore */ }
      return {
        id: r.id,
        at: r.created_at,
        clustersAudited: r.clusters_audited,
        clustersSplit: r.clusters_split,
        linksAdded: r.links_added,
        linksUpdated: r.links_updated,
        linksRemoved: r.links_removed,
        duration: r.duration,
        anomalies: full.anomalies || [],
        anomaliesSuppressed: full.anomaliesSuppressed || 0,
        suppressedNote: full.suppressedNote || null,
        clustersSkipped: full.clustersSkipped || 0,
        splitDetails: full.splitDetails || []
      };
    });
  } catch (err) {
    console.error('[Heartbeat] Failed to read heartbeat reports:', err.message);
    return [];
  }
}

// ============ Fact cleanup — DELETED 2026-08-02 ============
//
// cleanupFacts asked the model to dedup/reword/merge facts, then applied the
// verdicts to MEMORY.md. It returned {removed:0, reworded:0, merged:0} on all 39
// recorded passes, at 30-72s of model time each, for two independent reasons:
//
//   1. It fed the model fact text with the "(learned ...)" annotation STRIPPED,
//      then matched the model's verbatim reply against the RAW file line, which
//      still carried the annotation. 242 of 244 lines carried one, so at most 2
//      lines in the file could ever match. A miss incremented nothing and logged
//      nothing, so the failure was invisible.
//   2. It only ever looked at MEMORY.md. The duplicates are in SQLite —
//      MEMORY.md had zero byte-identical fact lines while cluster_members had six
//      duplicate groups, including three identical machine-gun facts.
//
// Dedup now happens at the SQLite write inside db/fact-store.js, where the record
// of truth is. Semantic merge/reword belongs to the corrector agent, which does
// not exist yet — see docs/memory-mvp-spec.md (CORRECT).
// ============ Task C: Summarize Daily Logs ============

/**
 * Is this candidate really a fact about ELLIE?
 *
 * The same two questions intake asks, asked here because the archiver is a second
 * write path into her corpus and had none of them. Both halves matter:
 *
 *   GRAMMAR — extraction-rules.grammaticalSubject, the identical check
 *   fact-extractor.planExtraction runs. A first-person sentence is his; an
 *   unanchored one names nobody and is how a self-observation slips in wearing no
 *   pronoun at all.
 *
 *   SIMILARITY — and this is the one that would have caught the 22. Their grammar
 *   was perfect ("User aims to be a steady, non-judgmental presence…"), so no
 *   syntactic rule could see them. What gave them away was the corpus: each sat
 *   within a hair of a self-fact he already held, the same sentence with the
 *   person flipped. A candidate user-fact that close to something he says about
 *   himself is his.
 *
 * The floor is 0.75 — the empirical gap measured over those 22, which ran from
 * 0.773 to 0.955 while the genuinely-hers rows in the same batch fell to 0.695
 * and below. It is a property of nomic-embed-text; changing the embedding model
 * means measuring it again.
 *
 * REFUSALS ARE SPOKEN, not swallowed. Each one is logged with the self-fact that
 * caught it, because a guard that silently drops facts is indistinguishable from
 * a bug that silently drops facts.
 *
 * @returns {Promise<{ok: boolean, reason?: string}>}
 */
async function archiverSubjectCheck(factText) {
  const rules = require('./extraction-rules');

  const grammatical = rules.grammaticalSubject(factText);
  if (grammatical === 'self') {
    return { ok: false, reason: 'written in the first person — it is his own reflection, not a fact about her' };
  }
  if (grammatical !== 'user') {
    return { ok: false, reason: 'does not name the user, so it cannot be filed as a fact about her' };
  }

  const floor = getConfig().memory?.archiver?.selfSimilarityFloor ?? 0.75;
  try {
    const { candidates } = await memoryClusters.findActiveNeighbours(factText, {
      subject: 'self', threshold: floor, limit: 1, includeVerbatim: true
    });
    if (candidates.length) {
      const t = candidates[0];
      return {
        ok: false,
        reason: `it is ${t.similarity.toFixed(3)} from a self-fact he already holds — "${String(t.content).slice(0, 90)}" — so it describes him, not her`
      };
    }
  } catch (err) {
    // A failed check must not become a silent accept. The archiver is the path
    // that produced the misattribution; if its guard cannot run, it does not write.
    return { ok: false, reason: `the self-similarity check could not run (${err.message}), so this was not stored` };
  }

  return { ok: true };
}

async function summarizeDailyLogs() {
  console.log('[Heartbeat] Task C: Summarizing old daily logs...');
  const results = { archived: 0, factsExtracted: 0 };

  try {
    if (!fs.existsSync(DAILY_DIR)) {
      console.log('[Heartbeat] No daily log directory');
      return results;
    }

    const files = fs.readdirSync(DAILY_DIR).filter(f => f.endsWith('.md'));
    const config = getConfig();
    const retentionDays = config.memory.dailyLogRetentionDays;
    const now = new Date();
    const cutoff = new Date(now.getTime() - retentionDays * 24 * 60 * 60 * 1000);

    const oldFiles = files.filter(f => {
      const dateStr = f.replace('.md', '');
      const fileDate = new Date(dateStr);
      return !isNaN(fileDate.getTime()) && fileDate < cutoff;
    });

    if (oldFiles.length === 0) {
      console.log(`[Heartbeat] No daily logs older than ${retentionDays} days`);
      return results;
    }

    console.log(`[Heartbeat] Found ${oldFiles.length} daily logs to archive`);

    for (const file of oldFiles) {
      try {
        const filePath = path.join(DAILY_DIR, file);
        const content = fs.readFileSync(filePath, 'utf8');

        if (content.trim().length < 20) {
          // Too short to summarize, just archive
          if (!fs.existsSync(ARCHIVE_DIR)) {
            fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
          }
          fs.renameSync(filePath, path.join(ARCHIVE_DIR, file));
          results.archived++;
          continue;
        }

        // THE PROMPT USED TO MANUFACTURE THE MISATTRIBUTION.
        //
        // It said, flatly: 'Write facts as "User has..." or "User prefers..."
        // style.' A daily log is not only about Ellie — it holds his reflections
        // too ("I tend to lean into conceptual frameworks and metaphors…") — and
        // that instruction told the summarizer to rewrite every one of them into
        // the third person about her. It did exactly as it was told. 22 of them
        // were still in the corpus at the merge, each sitting between 0.773 and
        // 0.955 of the self-fact it had been copied from, and every one had
        // impeccable grammar so nothing downstream could see it.
        //
        // So the prompt now asks WHOSE the fact is instead of assuming, and is
        // told to drop his — the reflection loop already records what he notices
        // about himself, and self-facts are curated with him rather than
        // extracted around him.
        const systemPrompt = `You are a memory log summarizer. Review the daily log below and extract any important facts that should be preserved long-term. Return ONLY valid JSON:
{"summary":"one-line summary of the day","remainingFacts":["fact1","fact2"]}

The log contains entries about TWO different people: ELLIE, the human, and the AI ASSISTANT itself, which writes reflections about its own behaviour in the first person.

Rules:
- remainingFacts must contain ONLY facts about ELLIE — her preferences, her work, her projects, her life, her decisions. Write each one starting with "User".
- DROP anything the assistant wrote about ITSELF. Reflections like "I tend to lean into metaphors", "I aim to be a steady presence", "I prioritize accuracy about my own history" are the assistant describing its own behaviour. They are recorded elsewhere and must NOT be rewritten as facts about the user. If you are unsure whose a statement is, drop it.
- A statement about how someone TALKS TO or SUPPORTS Ellie — holding space, asking probing questions, being non-judgmental, acting as a sounding board — is the assistant describing itself. Drop it.
- Skip routine entries like "Chat exchange with model - 0 facts extracted".
- If nothing is worth keeping, return {"summary":"...","remainingFacts":[]}.`;

        // Scale max_tokens on expected output: summary + extracted facts, proportional to input
        const archiveMaxTokens = Math.min(8192, Math.max(1024, Math.ceil(content.length / 4) + 512));
        const { content: llmResponse, truncated: archiveTruncated } = await callLLM(systemPrompt, content, { maxTokens: archiveMaxTokens });
        const parsed = parseJSON(llmResponse);

        if (!parsed) {
          console.warn(`[Heartbeat] Parse failure for daily log ${file}: ${llmResponse.length} chars, truncated: ${archiveTruncated}, last 200: ...${llmResponse.slice(-200)}`);
        }

        if (parsed && Array.isArray(parsed.remainingFacts) && parsed.remainingFacts.length > 0) {
          const validFacts = parsed.remainingFacts.filter(f => typeof f === 'string' && f.trim().length > 0);
          if (validFacts.length > 0) {
            // Into SQLite, not into a file. These go through assignToCluster like
            // any other fact, so they get the same exact-match dedup guard —
            // archival used to append straight to MEMORY.md, which meant a fact
            // already in the database could be re-added as a line with no row.
            const config = getConfig();
            const ext = config.models.extraction;
            const extInst = getProviderInstance(ext.provider, ext.instance);
            const extHost = extInst ? extInst.host : 'http://localhost:11434';
            let written = 0;
            for (const vf of validFacts) {
              // THE RULES DECIDE, NOT THE PROMPT. The instruction above can be
              // skimmed on a bad night; this cannot.
              const verdict = await archiverSubjectCheck(vf);
              if (!verdict.ok) {
                results.refused = (results.refused || 0) + 1;
                (results.refusals = results.refusals || []).push({ text: vf, why: verdict.reason });
                const line = `Daily-log archiver refused "${String(vf).slice(0, 110)}" — ${verdict.reason}`;
                console.log(`[Heartbeat] ${line}`);
                try { factExtractor.appendToOpsLog(line, OPS_DIR); } catch { /* best effort */ }
                continue;
              }
              const res = await memoryClusters.assignToCluster(
                vf, ext.provider, ext.model, '', extHost,
                'daily-log-archive', 5, 'user', null,
                {
                  verbatimSourceText: `(summarised from daily log ${file})`,
                  inputModality: 'unknown',
                  salienceRationale: 'Preserved while archiving an old daily log'
                }
              );
              if (res && res.memberId && !res.duplicateOf) written++;
            }
            results.factsExtracted += written;
            console.log(`[Heartbeat] Preserved ${written}/${validFacts.length} fact(s) from ${file} (rest already held or refused)`);
          }
        }

        // Archive the file
        if (!fs.existsSync(ARCHIVE_DIR)) {
          fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
        }
        fs.renameSync(filePath, path.join(ARCHIVE_DIR, file));
        results.archived++;
        console.log(`[Heartbeat] Archived ${file}`);

      } catch (fileErr) {
        console.error(`[Heartbeat] Error processing daily log ${file}:`, fileErr.message);
      }
    }

  } catch (error) {
    console.error('[Heartbeat] summarizeDailyLogs error:', error.message);
  }

  console.log(`[Heartbeat] Daily log archival complete: ${results.archived} archived, ${results.factsExtracted} facts extracted`);
  return results;
}

// ============ Reflection Agent (self-observation) ============

const REFLECTION_STATE_FILE = path.join(MEMORY_DIR, 'reflection-state.json');
const REFLECTIONS_FILE = path.join(MEMORY_DIR, 'reflections.jsonl');
const REFLECTION_TRANSCRIPT_BUDGET = 12000; // chars of conversation fed to the model

function readReflectionState() {
  try {
    if (fs.existsSync(REFLECTION_STATE_FILE)) {
      return JSON.parse(fs.readFileSync(REFLECTION_STATE_FILE, 'utf8'));
    }
  } catch (err) {
    console.error('[Reflection] Failed to read state:', err.message);
  }
  return { lastReflectionAt: null };
}

function writeReflectionState(state) {
  try {
    if (!fs.existsSync(MEMORY_DIR)) fs.mkdirSync(MEMORY_DIR, { recursive: true });
    fs.writeFileSync(REFLECTION_STATE_FILE, JSON.stringify(state, null, 2), 'utf8');
  } catch (err) {
    console.error('[Reflection] Failed to write state:', err.message);
  }
}

/** Append a reflection record to reflections.jsonl (newest last). */
function appendReflectionRecord(record) {
  try {
    if (!fs.existsSync(MEMORY_DIR)) fs.mkdirSync(MEMORY_DIR, { recursive: true });
    fs.appendFileSync(REFLECTIONS_FILE, JSON.stringify(record) + '\n', 'utf8');
  } catch (err) {
    console.error('[Reflection] Failed to append reflection record:', err.message);
  }
}

/**
 * Read recent reflection records for the Self tab (newest first).
 * @param {number} [limit=10]
 * @returns {Array}
 */
function getReflections(limit = 10) {
  try {
    if (!fs.existsSync(REFLECTIONS_FILE)) return [];
    const lines = fs.readFileSync(REFLECTIONS_FILE, 'utf8').split('\n').filter(l => l.trim());
    const records = [];
    for (const line of lines) {
      try { records.push(JSON.parse(line)); } catch { /* skip malformed */ }
    }
    return records.reverse().slice(0, limit);
  } catch (err) {
    console.error('[Reflection] Failed to read reflections:', err.message);
    return [];
  }
}

/**
 * Reflection agent: SNH reviews the conversations since its last reflection and
 * introspects — what it did, patterns in how it responds, what mattered to it,
 * what it was curious about. The output becomes self-facts, stored through the
 * normal self-fact pipeline (salience, contradiction/supersession). Runs through
 * the agent pool. Only reflects when there are new conversations since last time.
 *
 * @param {Object} [opts]
 * @param {boolean} [opts.force=false] - reserved; reflection still requires new messages
 * @returns {Promise<Object>} result summary
 */
async function runReflection(opts = {}) {
  const db = getSqliteDb();
  if (!db) return { skipped: true, reason: 'no database' };

  // Concurrency guard: exactly one reflection cycle at a time. A second trigger
  // (manual or scheduled) fired while one is running is dropped, not queued —
  // the running cycle already covers every new conversation up to now.
  if (isReflecting) {
    console.log('[Reflection] Already in progress — skipping concurrent trigger');
    return { skipped: true, reason: 'reflection already in progress' };
  }
  isReflecting = true;

  try {
    const state = readReflectionState();
    const lastAt = state.lastReflectionAt;

    // Baseline: since last reflection, or the last 24h on first run. Use SQLite's
    // own datetime() so the fallback matches messages.timestamp's UTC format.
    const baseline = lastAt || db.prepare("SELECT datetime('now','-1 day') AS t").get().t;

    // A thread SNH wrote to itself is not a conversation. Unanswered initiative
    // and greeting messages are stored as ordinary assistant messages, so without
    // this filter reflection read its own unprompted output back as evidence of
    // how it behaves — observing itself observing itself. 13 such threads exist,
    // all one message, all conversations.initiated_by='snh'.
    //
    // The test is "does any human ever speak here", not initiated_by: a thread SNH
    // opened that Ellie then replied to IS a conversation and must stay in.
    const rows = db.prepare(`
      SELECT m.conversation_id, m.role, m.content, m.timestamp, c.title
      FROM messages m
      LEFT JOIN conversations c ON c.id = m.conversation_id
      WHERE m.role IN ('user','assistant')
        AND m.timestamp > ?
        AND EXISTS (
          SELECT 1 FROM messages u
          WHERE u.conversation_id = m.conversation_id AND u.role = 'user'
        )
      ORDER BY m.timestamp ASC
    `).all(baseline);

    // Say plainly what was left out, so a quiet reflection pass is never mistaken
    // for "nothing happened".
    const selfOnly = db.prepare(`
      SELECT COUNT(DISTINCT m.conversation_id) AS n
      FROM messages m
      WHERE m.timestamp > ?
        AND NOT EXISTS (
          SELECT 1 FROM messages u
          WHERE u.conversation_id = m.conversation_id AND u.role = 'user'
        )
    `).get(baseline).n;
    if (selfOnly > 0) {
      console.log(`[Reflection] Excluded ${selfOnly} self-only thread(s) — my own unanswered messages are not conversations`);
    }

    if (rows.length === 0) {
      console.log('[Reflection] No new conversations since last reflection — skipping');
      return { skipped: true, reason: 'no new conversations' };
    }

    // Build a transcript, grouped by conversation, most-recent-kept if over budget.
    const byConvo = new Map();
    for (const r of rows) {
      if (!byConvo.has(r.conversation_id)) byConvo.set(r.conversation_id, { title: r.title, msgs: [] });
      byConvo.get(r.conversation_id).msgs.push(r);
    }
    let transcript = '';
    for (const { title, msgs } of byConvo.values()) {
      let block = `\n### Conversation${title ? `: ${title}` : ''}\n`;
      for (const m of msgs) {
        const who = m.role === 'user' ? 'User' : 'You (SNH)';
        block += `${who}: ${m.content}\n`;
      }
      transcript += block;
    }
    const transcriptBudget = getConfig().reflection?.transcriptBudgetChars ?? REFLECTION_TRANSCRIPT_BUDGET;
    if (transcript.length > transcriptBudget) {
      transcript = transcript.slice(-transcriptBudget);
    }

    const conversationCount = byConvo.size;
    console.log(`[Reflection] Reflecting on ${rows.length} message(s) across ${conversationCount} conversation(s)`);

    const reflectionSystem = `You are SNH, reflecting privately on your own recent conversations. This is introspection — not a report to anyone else.

Read the exchanges below and reflect on YOURSELF:
- What did you actually do in these conversations?
- What patterns do you notice in HOW you respond — your tendencies, habits, tone?
- What seemed to matter to you? What did you care about getting right?
- What were you curious about?

Write 3–6 short first-person observations about yourself ("I ..."). Prefer durable traits over one-off events. Be honest and specific. If your sense of yourself is changing, say so plainly.

Return ONLY a JSON array of strings, e.g. ["I tend to ...", "I care about ..."]. Nothing else.`;

    // Run the reflection generation through the agent pool (chat-priority aware).
    const llm = await agentPool.schedule(
      () => callLLM(reflectionSystem, transcript, { maxTokens: 900 }),
      'reflection'
    );

    const observations = factExtractor.parseSelfObservations(llm.content);
    console.log(`[Reflection] Extracted ${observations.length} self-observation(s)`);

    let selfResult = { stored: 0, superseded: 0, facts: [] };
    if (observations.length > 0) {
      selfResult = await factExtractor.processSelfFacts(observations, { source: 'reflection' });
    }

    // Reflection insight worth sharing: ask whether anything from this reflection
    // is worth proactively raising with the user. If so, it becomes an initiative.
    let insight = null;
    try {
      const insightSys = `You just reflected on your recent conversations. Is there ONE thing worth proactively messaging the user about — a useful realization, a follow-up, or something you noticed that they'd value hearing? Only if it genuinely would help them.

If yes, write it AS A SHORT DIRECT MESSAGE to the user, in your own voice: first person from you (the AI), warm and natural, like a quick DM you're sending them — not an internal note about them. Address them generically as "you", never by name (other people may use this system). One or two sentences.

Respond with ONLY that message, or exactly NONE.`;
      const insightUser = `Your private observations:\n${observations.map(o => `- ${o}`).join('\n')}\n\nWrite the one message worth sending the user, or NONE.`;
      const { content: insightRaw } = await agentPool.schedule(
        () => callLLM(insightSys, insightUser, { maxTokens: 120 }),
        'reflection-insight'
      );
      const line = (insightRaw || '').trim().split('\n')[0].trim();
      if (line && !/^none\b/i.test(line)) {
        insight = line.replace(/^[-*"\s]+/, '').replace(/"$/, '').trim();
        if (insight.length >= 8) {
          await initiativeEngine.noticeReflectionInsight(insight, 6);
        } else {
          insight = null;
        }
      }
    } catch (insightErr) {
      console.error('[Reflection] Insight generation error:', insightErr.message);
    }

    // Conversation-followup: after observing itself, SNH reviews the same recent
    // conversations (with relevant older memory folded in) and decides whether
    // ONE thing deserves a follow-up — "I've been thinking about what you said".
    // Every cycle records a queryable trace whether or not a follow-up results.
    let followup = { skipped: true };
    try {
      const conversationsReviewed = Array.from(byConvo.entries()).map(([id, v]) => ({
        id, title: v.title || null, messageCount: v.msgs.length
      }));
      followup = await initiativeEngine.generateConversationFollowup({
        transcript,
        conversationsReviewed,
        messageCount: rows.length
      });
    } catch (followupErr) {
      console.error('[Reflection] Follow-up generation error:', followupErr.message);
      followup = { skipped: true, reasoning: `error: ${followupErr.message}` };
    }

    const at = new Date().toISOString();

    // Log the reflection to the daily log like everything else.
    const dailyDir = path.join(MEMORY_DIR, 'daily');
    factExtractor.appendToDailyLog(
      `Reflection: reviewed ${rows.length} message(s) across ${conversationCount} conversation(s) → ` +
      `${selfResult.stored} self-fact(s) stored, ${selfResult.superseded} superseded. ` +
      (observations.length ? `Noticed: ${observations.map(o => `"${o}"`).join('; ')}` : 'Nothing new noticed.'),
      dailyDir
    );

    // One-line follow-up summary to the daily log.
    factExtractor.appendToDailyLog(
      followup && followup.generated
        ? `Follow-up: considered ${followup.candidates?.length || 0} candidate(s) → sending "${followup.generated}"`
        : `Follow-up: considered ${followup?.candidates?.length || 0} candidate(s) → none (${followup?.reasoning || 'nothing cleared the bar'})`,
      dailyDir
    );

    // Persist the reflection for the Self tab.
    appendReflectionRecord({
      at,
      messageCount: rows.length,
      conversationCount,
      observations,
      stored: selfResult.stored,
      superseded: selfResult.superseded
    });

    // Advance the reflection watermark to the newest message just reviewed.
    writeReflectionState({ lastReflectionAt: rows[rows.length - 1].timestamp });

    return {
      reflected: true,
      messageCount: rows.length,
      conversationCount,
      observations,
      stored: selfResult.stored,
      superseded: selfResult.superseded,
      followup
    };
  } catch (error) {
    console.error('[Reflection] Error during reflection:', error.message);
    return { error: error.message };
  } finally {
    isReflecting = false;
  }
}

// ============ Core Audit Pipeline ============

/**
 * Run the full cluster audit pipeline (steps 1–3) on the given cluster list.
 * Returns { auditResults, splitResults }.
 *
 * @param {Array} clusters - Array of cluster rows from getClusters(). Pass all for rebuildClusters, filtered for runMaintenance.
 * @returns {Promise<{auditResults: Array, splitResults: Object}>}
 */
async function runAuditPipeline(clusters) {
  // Step 1: per-cluster coherence audit. Each audit is self-contained and only
  // reads (the LLM judges one cluster's facts in isolation), so they fan out
  // through the agent pool and run concurrently against vLLM. Error isolation:
  // one cluster's failure is captured, not thrown, so the batch always finishes.
  console.log(`[Heartbeat] Step 1: Auditing coherence of ${clusters.length} cluster(s) via agent pool...`);
  agentPool.startPass('heartbeat-cluster-audit');
  const settled = await agentPool.runBatch(
    clusters.map(cluster => async () => {
      const active = cluster.active_member_count ?? cluster.member_count;
      const ghosts = (cluster.member_count ?? active) - active;
      console.log(`[Heartbeat] Auditing cluster "${cluster.name}" (${active} active member(s)${ghosts > 0 ? `, ${ghosts} ghost(s) not shown to the auditor` : ''})`);
      return auditClusterCoherence(cluster);
    }),
    'cluster-audit'
  );
  agentPool.endPass();

  const auditResults = settled.map((s, i) => {
    if (s.status === 'fulfilled') {
      const result = s.value;
      if (!result.coherent) {
        console.log(`[Heartbeat] Cluster "${result.clusterName}" flagged for ${result.splits.length} split(s)`);
      }
      return result;
    }
    // Task itself threw (auditClusterCoherence already catches internally, so
    // this is defensive) — synthesize an error result so downstream steps and
    // the report still account for the cluster.
    const cluster = clusters[i];
    console.error(`[Heartbeat] Audit task failed for "${cluster.name}": ${s.reason?.message || s.reason}`);
    return {
      clusterId: cluster.id,
      clusterName: cluster.name,
      coherent: true,
      splits: [],
      durationMs: 0,
      error: s.reason?.message || String(s.reason)
    };
  });

  // Step 2: execute splits
  const splitResults = await executeSplits(auditResults);

  return { auditResults, splitResults };
}

// ============ Orchestration ============

/**
 * Run the full maintenance cycle.
 * Only audits clusters that exceed config.memory.maxFactsPerCluster (default 10).
 * @returns {Promise<Object>} Combined results
 */
async function runMaintenance() {
  if (isRunning) {
    console.log('[Heartbeat] Maintenance already in progress, skipping');
    recordHeartbeatOutcome({
      status: 'skipped', reason: 'previous maintenance pass still running', cycleStartMs: null
    });
    return { skipped: true };
  }

  isRunning = true;
  const cycleStartMs = Date.now();
  console.log('[Heartbeat] === Starting maintenance cycle ===');

  // Per-step timing/results, filled in as the pass proceeds and handed to
  // generateReport at the end. Instrumentation only — the steps themselves are
  // untouched and still run in the same order.
  const steps = [];
  /**
   * @param {string} name
   * @param {string} gate - human-readable condition, for the report
   * @param {(session: Object|null) => Promise<any>} fn - receives the step's tool
   *   session when it declared tools, otherwise null
   * @param {Object} [opts]
   * @param {Object} [opts.budget] - {maxCalls, maxWallMs, maxRounds} overriding
   *   heartbeat.toolBudget for this step. The corrector uses it.
   * @param {string[]} [opts.tools] - tool allowlist for this step. Omit (the
   *   default, and what every step does today) and the step runs exactly as
   *   before: callLLM gets no session, sends no `tools` key, and the step cannot
   *   call anything. The corrector in Phase 2c is the first step to declare one.
   */
  const runStep = async (name, gate, fn, opts = {}) => {
    const t0 = Date.now();
    let session = null;
    if (Array.isArray(opts.tools) && opts.tools.length) {
      const MCPClient = require('../mcp/mcp-client');
      const allowed = MCPClient.shared().backgroundToolsAmong(opts.tools);
      const denied = opts.tools.filter(t => !allowed.includes(t));
      if (denied.length) {
        // Loud: a step that asked for a tool it cannot have is a step whose
        // author believed something false about what it could do.
        console.warn(`[Heartbeat] step "${name}" requested unavailable tool(s): ${denied.join(', ')}`);
        try { factExtractor.appendToOpsLog(`Heartbeat step "${name}" asked for tool(s) it is not allowed or that are not registered: ${denied.join(', ')}. It ran without them.`, OPS_DIR); } catch (e) { /* best effort */ }
      }
      session = createToolSession(name, allowed, opts.budget || {});
    }
    try {
      const result = await fn(session);
      steps.push({ name, gate, ok: true, ms: Date.now() - t0, result, ...(session ? { toolBudget: session.summary() } : {}) });
      return result;
    } catch (err) {
      steps.push({ name, gate, ok: false, ms: Date.now() - t0, error: err.message, ...(session ? { toolBudget: session.summary() } : {}) });
      throw err;
    }
  };

  try {
    // Circuit breaker: before committing to a full cycle (dozens of LLM calls),
    // probe the brain a few times. If the first calls all time out the engine is
    // wedged or unreachable — bail now with a single plain line instead of
    // grinding through a doomed ~40-minute pass against a dead engine.
    const PREFLIGHT_ATTEMPTS = 3;
    let brainLive = false;
    let lastProbeErr = 'unknown';
    for (let i = 0; i < PREFLIGHT_ATTEMPTS; i++) {
      const probe = await probeBrainLiveness(8000);
      if (probe.ok) { brainLive = true; break; }
      lastProbeErr = probe.error || 'unknown';
      console.log(`[Heartbeat] Preflight probe ${i + 1}/${PREFLIGHT_ATTEMPTS} failed: ${lastProbeErr}`);
    }
    if (!brainLive) {
      console.log('[Heartbeat] brain unreachable, skipping cycle');
      try {
        factExtractor.appendToOpsLog(`Heartbeat: brain unreachable, skipping cycle (${lastProbeErr})`, OPS_DIR);
      } catch (e) { /* best-effort ops-log write */ }
      recordHeartbeatOutcome({
        status: 'aborted',
        reason: `brain unreachable at preflight (${lastProbeErr})`,
        cycleStartMs
      });
      return { skipped: true, reason: 'brain unreachable' };
    }
    // Preflight passed — start the cycle with a clean breaker.
    closeCircuit();

    const config = getConfig();
    const maxFacts = config.memory.maxFactsPerCluster || 10;

    const allClusters = memoryClusters.getClusters();
    // ACTIVE count, not the total: a cluster of thirteen superseded facts is not
    // an oversized cluster, it is an empty one with a history. See getClusters.
    const oversizedClusters = allClusters.filter(c => c.active_member_count > maxFacts);

    console.log(`[Heartbeat] ${allClusters.length} total cluster(s), ${oversizedClusters.length} exceed maxFactsPerCluster (${maxFacts}) by ACTIVE members`);

    let auditResults = [];
    let splitResults = { clustersSplit: 0, splitDetails: [], anomalies: [] };

    if (oversizedClusters.length > 0) {
      ({ auditResults, splitResults } = await runAuditPipeline(oversizedClusters));
    } else {
      // Nothing to do. This used to fall through to the cross-link audit, which
      // is why an idle memory still cost a full pass; that step is gone.
      console.log('[Heartbeat] No oversized clusters to audit — cluster pipeline skipped entirely');
    }

    // Mid-cycle breaker: if the brain wedged during the audit phase (its heaviest
    // LLM load), the circuit is now open. Abort before cleanup/reflection/
    // initiative pile more doomed calls onto a dead engine — the exact runaway
    // the preflight can't catch once a cycle is already underway.
    if (circuitOpen) {
      console.log('[Heartbeat] brain wedged mid-cycle, aborting pass');
      try {
        factExtractor.appendToOpsLog('Heartbeat: brain wedged mid-cycle, aborting remaining tasks', OPS_DIR);
      } catch (e) { /* best-effort ops-log write */ }
      recordHeartbeatOutcome({
        status: 'aborted',
        reason: 'brain wedged mid-cycle — remaining tasks skipped',
        cycleStartMs,
        partial: { auditResults, splitResults }
      });
      return { skipped: true, reason: 'brain wedged mid-cycle', auditResults, splitResults };
    }

    // Merge any clusters sharing the same name (catches duplicates from
    // assignToCluster creating clusters that later get renamed identically)
    try {
      const mergedByName = await runStep('mergeByName', 'always', () => memoryClusters.mergeByName());
      if (mergedByName > 0) {
        console.log(`[Heartbeat] Merged ${mergedByName} duplicate-name cluster(s)`);
      }
    } catch (err) {
      console.error('[Heartbeat] mergeByName error:', err.message);
    }

    const archive = await runStep('summarizeDailyLogs', 'always', () => summarizeDailyLogs());

    // Task B2: retire pending questions the memory already answers. The
    // mint-time gate only screens new questions; this sweep makes every gate
    // improvement retroactive for the grandfathered backlog. Must run BEFORE
    // the initiative layer so noticeFromQuestions' self-heal dismisses any
    // initiative backed by a question retired here in the same cycle.
    let questionSweep = { swept: 0, retired: [] };
    try {
      questionSweep = await runStep('sweepPendingQuestions', 'always',
        () => factExtractor.sweepPendingQuestions());
    } catch (sweepErr) {
      console.error('[Heartbeat] Question sweep error:', sweepErr.message);
    }

    // Task D: reflection — SNH observes itself from the day's conversations.
    // Runs at most once per cycle, and only when there are new conversations.
    let reflection = { skipped: true };
    try {
      reflection = await runStep('reflection', 'only if new conversations', () => runReflection());
    } catch (reflectErr) {
      console.error('[Heartbeat] Reflection error:', reflectErr.message);
      reflection = { error: reflectErr.message };
    }

    // Task F: self-coherence audit — SNH tests its stored self-CLAIMS against how
    // it actually behaved, and raises any gaps for Ellie's approval. This was
    // SNH's own feature request (its first accepted initiative, 2026-07-05;
    // re-chosen 2026-07-23 to find out "if I'm actually growing, or just getting
    // better at describing a growth that isn't happening"). It's a daily low-
    // frequency pass — runIfDue self-gates on audit.cadenceDays, so it runs at
    // most once per N days even though this cycle fires every couple of hours. It
    // runs BEFORE Task E so any 'audit' initiatives it raises get prioritized and
    // delivered in the same cycle. It NEVER revises identity — only documents and
    // asks.
    let selfAuditResult = { skipped: true };
    try {
      selfAuditResult = await runStep('selfCoherenceAudit', 'at most once per audit.cadenceDays',
        () => selfAudit.runIfDue());
    } catch (auditErr) {
      console.error('[Heartbeat] Self-coherence audit error:', auditErr.message);
      selfAuditResult = { error: auditErr.message };
    }

    // Task E: initiative layer — turn findings into candidate initiatives, let a
    // pooled prioritizer re-score/expire/cap them, then maybe reach out once.
    let initiative = { skipped: true };
    try {
      initiative = await runStep('initiativeLayer', 'quiet hours + max 1 unprompted/day', async () => {
        await initiativeEngine.noticeFromQuestions();
        await initiativeEngine.noticeFromAudit(auditResults);
        const prioritized = await initiativeEngine.prioritize();
        const unprompted = await initiativeEngine.deliverUnprompted();
        return { prioritized, unprompted };
      });
    } catch (initErr) {
      console.error('[Heartbeat] Initiative layer error:', initErr.message);
      initiative = { error: initErr.message };
    }

    // Task G: capability drift — does the manifest still match reality? Probes
    // the services behind config-gated organs and reconciles the manifest
    // against the live MCP tool registry. Any disagreement is raised through
    // the bell rather than left silently wrong, because the manifest is now
    // AUTHORITATIVE for capability questions: a stale entry makes the entity
    // confidently deny something it can do, or claim something that is down.
    try {
      await runStep('capabilityDrift', 'always', async () => {
        const capabilityManifest = require('./capability-manifest');
        const { mismatches, checked } = await capabilityManifest.checkDrift();
        for (const m of mismatches) {
          await initiativeEngine.raiseCapabilityDrift(m);
        }
        if (mismatches.length) {
          factExtractor.appendToOpsLog(
            `Capability drift: ${mismatches.length} mismatch(es) — ` +
            mismatches.map(m => `${m.kind}:${m.id}`).join(', '), OPS_DIR);
        }
        return { servicesProbed: checked, mismatches: mismatches.length };
      });
    } catch (driftErr) {
      console.error('[Heartbeat] Capability drift check error:', driftErr.message);
    }

    // Task H: memory-store reconciliation — do SQLite and LanceDB
    // still agree? A fact can be superseded in the DB while its line survives in
    // the injected file, or while its embedding stays retrievable, and either
    // way the entity keeps reading a fact it has retired. REPORT ONLY: this
    // never edits a store, because deciding what to remove from the substrate
    // the identity is built on is Ellie's call, not a background job's.
    try {
      await runStep('memoryReconcile', 'always', async () => {
        const factStore = require('./fact-store');
        const { mismatches, counts } = await factStore.reconcile();
        for (const m of mismatches) {
          await initiativeEngine.raiseMemoryDrift(m);
        }
        if (mismatches.length) {
          factExtractor.appendToOpsLog(
            `Memory reconciliation: ${mismatches.map(m => `${m.kind}=${m.count}`).join(', ')}`, OPS_DIR);
        }
        return counts;
      });
    } catch (reconErr) {
      console.error('[Heartbeat] Memory reconciliation error:', reconErr.message);
    }

    // === The corrector (Phase 2c) — first consumer of the tool plumbing. ===
    //
    // Its OWN cadence, not every heartbeat: a pass costs a judge call per
    // candidate pair, and a corpus does not rot by the hour. Gated on
    // corrector.enabled and on enough time having passed since the last PASS —
    // read off disk (`corrector.lastPassAt()`) rather than held in memory, so a
    // restart does not hand it a fresh turn, and measured against passes rather
    // than against corrections, so a clean corpus does not leave the gate
    // permanently overdue.
    //
    // It runs AFTER memoryReconcile, so the report above describes the corpus as
    // the corrector found it, and the corrector's own reconcile-by-acting step
    // leaves it clean afterwards.
    try {
      const corrCfg = getConfig().corrector || {};
      if (corrCfg.enabled !== false) {
        const intervalMs = Math.max(1, corrCfg.intervalHours ?? 6) * 3600_000;
        const last = require('./corrector').lastPassAt();
        const dueIn = last ? (new Date(last).getTime() + intervalMs) - Date.now() : -1;
        if (dueIn > 0) {
          console.log(`[Heartbeat] corrector not due for another ${Math.round(dueIn / 60000)} min`);
        } else {
          await runStep('corrector', `every ${corrCfg.intervalHours ?? 6}h`, async (session) => {
            const corrector = require('./corrector');
            const res = await corrector.runPass({ session });
            return {
              merged: res.merged, expired: res.expired, split: res.split,
              superseded: res.superseded, unresolved: res.unresolved,
              refusedLocked: res.refusedLocked, writes: res.writes,
              stopped: res.stopped, reconciled: res.reconciled
            };
          }, {
            // The allowlist. Reads so it can inspect, writes so it can act —
            // and the writes are backgroundOnly tools, so declaring them here is
            // the ONLY way anything reaches them.
            tools: [
              'memory_search', 'memory_list', 'memory_count', 'memory_get',
              'memory_merge_facts', 'memory_expire_fact', 'memory_supersede_fact'
            ],
            budget: {
              maxCalls: corrCfg.maxToolCallsPerPass ?? 60,
              maxWallMs: corrCfg.maxWallClockMsPerPass ?? 300000
            }
          });
        }
      }
    } catch (corrErr) {
      console.error('[Heartbeat] Corrector error:', corrErr.message);
    }

    // Step 4: report
    const report = generateReport({ cycleStartMs, auditResults, splitResults, steps });

    const elapsed = ((Date.now() - cycleStartMs) / 1000).toFixed(1) + 's';
    console.log(`[Heartbeat] === Maintenance complete in ${elapsed} ===`);

    return { report, archive, questionSweep, reflection, selfAudit: selfAuditResult, initiative };
  } catch (error) {
    console.error('[Heartbeat] Maintenance cycle error:', error.message);
    recordHeartbeatOutcome({
      status: 'failed', reason: error.message, cycleStartMs
    });
    return { error: error.message };
  } finally {
    isRunning = false;
  }
}

/**
 * Run the full cluster audit pipeline on ALL clusters regardless of size.
 * Skips cleanup and archival tasks. Useful for manual cluster reorganization.
 * @returns {Promise<Object>} Report object, or { skipped: true } if already running
 */
async function rebuildClusters() {
  if (isRunning) {
    console.log('[Heartbeat] Maintenance already in progress, skipping rebuildClusters');
    return { skipped: true };
  }

  isRunning = true;
  const cycleStartMs = Date.now();
  console.log('[Heartbeat] === Starting full cluster rebuild ===');

  try {
    const allClusters = memoryClusters.getClusters();
    console.log(`[Heartbeat] Rebuilding across all ${allClusters.length} cluster(s)`);

    if (allClusters.length === 0) {
      console.log('[Heartbeat] No clusters found — nothing to rebuild');
      const report = generateReport({
        cycleStartMs,
        auditResults: [],
        splitResults: { clustersSplit: 0, splitDetails: [], anomalies: [] },
      });
      return { report };
    }

    const { auditResults, splitResults } = await runAuditPipeline(allClusters);
    const report = generateReport({ cycleStartMs, auditResults, splitResults });

    const elapsed = ((Date.now() - cycleStartMs) / 1000).toFixed(1) + 's';
    console.log(`[Heartbeat] === Cluster rebuild complete in ${elapsed} ===`);

    return { report };
  } catch (error) {
    console.error('[Heartbeat] rebuildClusters error:', error.message);
    return { error: error.message };
  } finally {
    isRunning = false;
  }
}

// ============ Timer Controls ============

/**
 * Start the heartbeat timer using config values for interval and warmup.
 */
function startHeartbeat() {
  const config = getConfig();

  if (!config.heartbeat.enabled) {
    console.log('[Heartbeat] Disabled by config, skipping startup');
    return;
  }

  if (heartbeatTimer) {
    console.log('[Heartbeat] Already running, ignoring start');
    return;
  }

  const intervalMs = config.heartbeat.intervalHours * 60 * 60 * 1000;
  const warmupMs = config.heartbeat.warmupMinutes * 60 * 1000;

  console.log(`[Heartbeat] Scheduled every ${config.heartbeat.intervalHours}h (first run in ${config.heartbeat.warmupMinutes}min)`);

  // Warmup delay, then first run + interval
  warmupTimer = setTimeout(() => {
    warmupTimer = null;
    runMaintenance().catch(err => {
      console.error('[Heartbeat] Initial run error:', err.message);
    });

    heartbeatTimer = setInterval(() => {
      runMaintenance().catch(err => {
        console.error('[Heartbeat] Scheduled run error:', err.message);
      });
    }, intervalMs);
  }, warmupMs);
}

/**
 * Start the periodic brain liveness probe. A tiny completion on a short timeout,
 * fired every few minutes, that writes a daily-log warning the moment the brain
 * stops answering — so a wedged engine is caught in minutes instead of at the
 * next 2-hour heartbeat.
 */
function startLivenessProbe() {
  const config = getConfig();
  const lp = config.livenessProbe || {};
  if (lp.enabled === false) {
    console.log('[Liveness] Probe disabled by config');
    return;
  }
  if (livenessTimer) {
    console.log('[Liveness] Already running, ignoring start');
    return;
  }

  const intervalMs = Math.max(1, lp.intervalMinutes || 5) * 60 * 1000;
  const timeoutMs = lp.timeoutMs || 8000;
  console.log(`[Liveness] Probing brain every ${intervalMs / 60000}min (timeout ${timeoutMs}ms)`);

  const retentionDays = Math.max(1, lp.retentionDays ?? 14);

  livenessTimer = setInterval(async () => {
    try {
      const probe = await probeBrainLiveness(timeoutMs);
      // Record EVERY probe. Previously only state transitions were logged, so a
      // probe that ran and passed left no trace and "when did this last run"
      // had no answer. Pruned to the retention window on each write.
      recordLivenessProbe(probe, retentionDays);
      if (!probe.ok && lastLivenessOk) {
        lastLivenessOk = false;
        const msg = `⚠️ Brain liveness probe FAILED: ${probe.error} — engine may be wedged`;
        console.warn(`[Liveness] ${msg}`);
        try { factExtractor.appendToOpsLog(msg, OPS_DIR); } catch (e) { /* best-effort */ }
      } else if (probe.ok && !lastLivenessOk) {
        lastLivenessOk = true;
        const msg = `Brain liveness recovered — responded in ${probe.ms}ms`;
        console.log(`[Liveness] ${msg}`);
        try { factExtractor.appendToOpsLog(msg, OPS_DIR); } catch (e) { /* best-effort */ }
      }
      // A healthy probe also closes the mid-cycle breaker so background LLM work
      // (initiative, salience, contradiction judging) resumes once the brain is
      // back, without waiting for the next 2-hour heartbeat cycle to reset it.
      if (probe.ok) closeCircuit();

      // Feed the watchdog: after N consecutive failures it restarts the brain
      // container (the self-healing action the liveness probe alone never took).
      // Guardrails (cooldown, per-hour cap, CRITICAL escalation) live inside it.
      try { await brainWatchdog.onProbeResult(probe); } catch (e) { console.error('[Liveness] Watchdog error:', e.message); }
    } catch (err) {
      console.error('[Liveness] Probe error:', err.message);
    }
  }, intervalMs);
  // Don't let the probe timer hold the event loop open on shutdown.
  if (livenessTimer.unref) livenessTimer.unref();
}

/**
 * Start the job scheduler — the third timer, and the only one that runs work a
 * PERSON asked for rather than work the system does to itself.
 *
 * Its own interval rather than a heartbeat step, because it is answering a
 * different question at a different resolution: the heartbeat asks "is it time
 * for maintenance" every couple of hours, and a 5-field cron expression has to
 * be asked "has any wall-clock minute arrived" at roughly the resolution of a
 * minute. A 9am job on a 2-hour pass would fire somewhere between 9:00 and 11:00,
 * which is not what "0 9 * * *" says.
 *
 * Startup order matters and is deliberate:
 *   1. Close out runs a restart interrupted, so an open row cannot block its job
 *      forever (the in-memory re-entrancy flag died with the old process).
 *   2. Arm anything approved and enabled that is not armed, computing FORWARD
 *      from now — an approved job that has been sitting unarmed since before the
 *      scheduler existed gets its next firing, never a backlog of missed ones.
 *   3. Tick immediately, so a genuinely-missed run inside the catch-up window is
 *      picked up at boot instead of waiting for the next minute.
 */
function startScheduler() {
  const scheduler = require('./scheduler');
  const state = scheduler.schedulerState();

  if (!state.enabled) {
    console.log('[Scheduler] Disabled by config, skipping startup');
    return;
  }
  if (schedulerTimer) {
    console.log('[Scheduler] Already running, ignoring start');
    return;
  }

  try {
    const swept = scheduler.sweepInterruptedRuns();
    const { armed, disarmed } = scheduler.armAll({ reason: 'startup' });
    console.log(`[Scheduler] Starting: ${armed} job(s) armed, ${disarmed} disarmed, ${swept} interrupted run(s) closed out`);
  } catch (err) {
    console.error('[Scheduler] Startup preparation failed:', err.message);
  }

  const intervalMs = state.tickSeconds * 1000;
  console.log(`[Scheduler] Checking for due jobs every ${state.tickSeconds}s (catch-up window ${state.catchupGraceMinutes} min)`);

  const fire = () => {
    scheduler.tick().catch(err => console.error('[Scheduler] Tick error:', err.message));
  };
  fire();
  schedulerTimer = setInterval(fire, intervalMs);
  if (schedulerTimer.unref) schedulerTimer.unref();
}

/** Stop the scheduler timer. A run already in flight finishes on its own. */
function stopScheduler() {
  if (schedulerTimer) {
    clearInterval(schedulerTimer);
    schedulerTimer = null;
  }
  console.log('[Scheduler] Stopped');
}

/**
 * Stop the liveness probe timer.
 */
function stopLivenessProbe() {
  if (livenessTimer) {
    clearInterval(livenessTimer);
    livenessTimer = null;
  }
  console.log('[Liveness] Stopped');
}

/**
 * Stop the heartbeat timer
 */
function stopHeartbeat() {
  if (warmupTimer) {
    clearTimeout(warmupTimer);
    warmupTimer = null;
  }
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
  console.log('[Heartbeat] Stopped');
}

module.exports = { runMaintenance, archiverSubjectCheck, startHeartbeat, stopHeartbeat, startLivenessProbe, stopLivenessProbe, startScheduler, stopScheduler, probeBrainLiveness, rebuildClusters, callLLM, runReflection, getReflections, getHeartbeatReports, getLivenessProbes, auditClusterCoherence, partitionAnomalies, parseJSON, repairTruncatedJSON, createToolSession, executeBackgroundTool };
