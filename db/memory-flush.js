const path = require('path');
const factExtractor = require('./fact-extractor');

/**
 * Estimate token count from text using rough 4-char-per-token heuristic
 * @param {string} text - Text to estimate tokens for
 * @returns {number} Estimated token count
 */
function estimateTokens(text) {
  if (!text || typeof text !== 'string') return 0;
  return Math.ceil(text.length / 4);
}

/**
 * Estimate total tokens across all messages
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @returns {number} Total estimated tokens
 */
function estimateMessagesTokens(messages) {
  if (!Array.isArray(messages)) return 0;

  return messages.reduce((total, msg) => {
    if (msg && msg.content) {
      return total + estimateTokens(msg.content);
    }
    return total;
  }, 0);
}

/**
 * Get context window limit for a given model.
 *
 * This is now the USABLE WINDOW, not a guess from the model's name: the engine
 * is asked what it actually serves (see db/model-context.js) and the answer is
 * capped by `memory.contextTokens`. The name-substring table survives inside
 * that module as the fallback for engines that cannot be asked — it is what
 * answered 8,192 for a model the engine was serving at 131,072.
 *
 * @param {string} model - Model name/identifier
 * @param {string} [provider] - narrows the lookup to one engine's cached answer
 * @param {string} [host]
 * @returns {number} Context window size in tokens
 */
function getModelContextLimit(model, provider = null, host = null) {
  return require('./model-context').usableWindow(model, provider, host);
}

/**
 * Check if conversation needs flushing based on token count
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {string} model - Model name
 * @returns {{needsFlush: boolean, tokenCount: number, contextLimit: number, usage: number}}
 */
function shouldFlush(messages, model, provider = null, host = null) {
  // Only user/assistant turns are compactable. A flush KEEPS the system message
  // (the injected memory context / identity prompt) verbatim, so counting it
  // toward the threshold made flush fire on every request once the memory prompt
  // grew large — even when there was nothing to compact — and each firing blocks
  // the chat on a synchronous extraction call. Measure the compactable portion
  // only, so flush triggers on genuinely long conversations, not a big prompt.
  const conversation = Array.isArray(messages)
    ? messages.filter(m => m && m.role !== 'system')
    : [];
  const tokenCount = estimateMessagesTokens(conversation);
  const contextLimit = getModelContextLimit(model, provider, host);
  const usage = contextLimit > 0 ? tokenCount / contextLimit : 0;
  const needsFlush = tokenCount > (contextLimit * 0.80);

  return {
    needsFlush,
    tokenCount,
    contextLimit,
    usage
  };
}

/**
 * How many trailing messages the compaction keeps verbatim.
 *
 * Load-bearing in two places that must agree: it defines what stays in the
 * request, and therefore what the extraction pass has to cover. Read from one
 * constant so a change to either half cannot silently un-cover the other.
 */
const KEEP_RECENT_MESSAGES = 10;

/**
 * Split messages into chunks that each fit a token budget, in order.
 *
 * A single message larger than the budget gets a chunk of its own rather than
 * being dropped — an oversized turn is exactly the kind of thing worth keeping,
 * and the extraction prompt is capped at half the window, so one long turn still
 * has room.
 * @param {Array<{role: string, content: string}>} list
 * @param {number} budgetTokens
 * @returns {Array<Array<{role: string, content: string}>>}
 */
function chunkMessages(list, budgetTokens) {
  const chunks = [];
  let current = [], used = 0;
  for (const m of list) {
    const t = estimateTokens(m.content) + 4;   // + the "ROLE: " framing
    if (current.length && used + t > budgetTokens) {
      chunks.push(current);
      current = []; used = 0;
    }
    current.push(m);
    used += t;
  }
  if (current.length) chunks.push(current);
  return chunks;
}

/**
 * Cap the head text for the fact-extraction pass, keeping the FRONT.
 *
 * Same reasoning as the summary itself: if something has to go, it is the part
 * nearest the turns that are staying in the request.
 * @private
 */
function budgetHeadForExtraction(text, budgetTokens) {
  if (estimateTokens(text) <= budgetTokens) return text;
  return text.slice(0, budgetTokens * 4) + '\n\n…(earlier-conversation excerpt truncated for this pass)';
}

/**
 * How long one summary may run.
 *
 * Generation was unbounded on every local provider, which matters more now that
 * a flush can make several calls: the pass is inline on the chat path with a 30s
 * timeout per call, and an unbounded summary spends that budget rambling. The
 * output is a bullet list of what was said, so this is generous for the job.
 */
const FLUSH_MAX_TOKENS = 1200;

/**
 * Make LLM API call for flush extraction (non-streaming)
 * @private
 */
async function callLLMForFlush(provider, model, messages, apiKey, host, maxTokens = FLUSH_MAX_TOKENS) {
  const controller = new AbortController();
  // Flush runs inline on the chat request path, so a slow/wedged brain here
  // stalls the user's reply. Bound it tightly — on timeout the flush fails
  // gracefully and the chat proceeds with the uncompacted messages.
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    let response;

    switch (provider.toLowerCase()) {
      case 'ollama': {
        const ollamaHost = host || 'http://localhost:11434';
        response = await fetch(`${ollamaHost}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            messages,
            stream: false,
            options: { num_predict: maxTokens }   // Ollama's name for the same bound
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`Ollama flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.message?.content || '';
      }

      case 'claude': {
        const systemMsg = messages.find(m => m.role === 'system');
        const userMessages = messages.filter(m => m.role !== 'system');

        response = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01'
          },
          body: JSON.stringify({
            model,
            max_tokens: maxTokens,
            system: systemMsg?.content || undefined,
            messages: userMessages,
            stream: false
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`Claude flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.content?.[0]?.text || '';
      }

      case 'openai': {
        response = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model,
            messages,
            stream: false,
            max_tokens: maxTokens
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`OpenAI flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.choices?.[0]?.message?.content || '';
      }

      case 'grok': {
        response = await fetch('https://api.x.ai/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model,
            messages,
            stream: false,
            max_tokens: maxTokens
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`Grok flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.choices?.[0]?.message?.content || '';
      }

      case 'llamacpp': {
        const llamaHost = host || 'http://localhost:8080';
        response = await fetch(`${llamaHost}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            messages,
            stream: false,
            max_tokens: maxTokens
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`Llama.cpp flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.choices?.[0]?.message?.content || '';
      }

      case 'vllm': {
        // vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint, so
        // this mirrors the llamacpp case but points at the vLLM host (sparky-brain).
        const vllmHost = host || 'http://localhost:8000';
        response = await fetch(`${vllmHost}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            messages,
            stream: false,
            max_tokens: maxTokens
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`vLLM flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.choices?.[0]?.message?.content || '';
      }

      case 'squatchserve': {
        const squatchHost = host || 'http://localhost:8000';
        response = await fetch(`${squatchHost}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            messages,
            stream: false,
            max_tokens: maxTokens
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`SquatchServe flush failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.message?.content || '';
      }

      default:
        throw new Error(`Unsupported provider for memory flush: ${provider}`);
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Perform memory flush: extract conversation summary, save to memory, compact messages
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {string} provider - LLM provider
 * @param {string} model - Model name
 * @param {string} apiKey - API key for provider
 * @param {string} host - Host URL for local providers
 * @param {string} memoryDir - Memory directory path
 * @returns {Promise<{compactedMessages: Array, flushSummary: string, factsExtracted: number}>}
 */
async function performFlush(messages, provider, model, apiKey, host, memoryDir = null) {
  const memDir = memoryDir || require('./database').getMemoryDir();
  const dailyDir = path.join(memDir, 'daily');

  try {
    const { tokenCount, contextLimit, usage } = shouldFlush(messages, model, provider, host);
    console.log(`[MemoryFlush] Starting flush - conversation at ${(usage * 100).toFixed(1)}% of context (${tokenCount}/${contextLimit} tokens)`);

    // THE PART BEING DISCARDED IS THE PART TO SUMMARISE.
    //
    // This used to be conversationText.slice(-maxChars) — the TAIL. Backwards,
    // and measured so on 2026-08-13: the tail is precisely what compaction keeps
    // in the request, while the head is what it is about to throw away. With the
    // extraction prompt capped at 50% of the window and the flush firing at 80%,
    // the earliest ~44% of a thread was cut before the model ever saw it, and the
    // summary that was supposed to stand in for those turns was written from the
    // turns that were staying. Planted details from the opening and middle of a
    // 68-message thread were both lost; only the late one survived, and it
    // survived because it was still in the compacted history anyway.
    //
    // The head is derived from the SAME slice the compaction uses, so the two can
    // never drift apart: whatever is not kept is what gets summarised.
    const conversationMessages = messages.filter(m => m.role === 'user' || m.role === 'assistant');
    const keptTail = new Set(messages.slice(-KEEP_RECENT_MESSAGES));
    const headMessages = conversationMessages.filter(m => !keptTail.has(m));
    const asText = (list) => list.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');

    // Nothing is being dropped (a short thread over budget on sheer message
    // size): summarise the whole thing rather than nothing.
    const toSummarise = headMessages.length ? headMessages : conversationMessages;
    const headText = asText(toSummarise);

    // CHUNKED, so nothing being discarded goes unseen. A cap that silently drops
    // the overflow is the bug this replaces; if the head does not fit in one
    // prompt it gets more than one prompt.
    const maxExtractionTokens = Math.floor(contextLimit * 0.5);
    const chunks = chunkMessages(toSummarise, maxExtractionTokens);
    console.log(`[MemoryFlush] Summarising the ${toSummarise.length} message(s) being dropped ` +
      `(${estimateTokens(headText)} tokens) in ${chunks.length} pass(es); ` +
      `${messages.length - toSummarise.length} message(s) stay in the request`);

    const summaries = [];
    for (let i = 0; i < chunks.length; i++) {
      const part = chunks.length > 1 ? ` (part ${i + 1} of ${chunks.length})` : '';
      const extractionMessages = [
        {
          role: 'system',
          content: 'You are a memory extraction system. Extract and save important facts, decisions, and context from this conversation.'
        },
        {
          role: 'user',
          content: `This conversation is getting long and the earlier turns are about to be dropped from context${part}. ` +
            `Extract all important facts, decisions, preferences, action items, and technical details from the following ` +
            `portion of it, including specifics like names, identifiers, ticket numbers and exact settings. Write them as bullet points.\n\n` +
            asText(chunks[i])
        }
      ];
      console.log(`[MemoryFlush] Requesting extraction from ${provider}/${model}${part}`);
      // A chunk that fails takes the whole flush with it, deliberately. The
      // caller then keeps the uncompacted messages, so the turns this pass could
      // not read are still in the conversation rather than dropped unread.
      const t0 = Date.now();
      const text = await callLLMForFlush(provider, model, extractionMessages, apiKey, host);
      console.log(`[MemoryFlush] Pass ${i + 1}/${chunks.length} returned in ${Date.now() - t0}ms`);
      if (!text || !text.trim()) {
        console.log(`[MemoryFlush] Warning: empty summary for chunk ${i + 1}/${chunks.length}`);
        continue;
      }
      summaries.push(chunks.length > 1 ? `**Earlier conversation, part ${i + 1} of ${chunks.length}**\n${text.trim()}` : text.trim());
    }

    const flushSummary = summaries.join('\n\n');
    if (!flushSummary) {
      console.log(`[MemoryFlush] Warning: Empty flush summary received`);
    } else {
      console.log(`[MemoryFlush] Received flush summary (${flushSummary.length} chars across ${summaries.length} pass(es))`);
    }

    // Save to daily log
    await factExtractor.appendToDailyLog(flushSummary, dailyDir);
    console.log(`[MemoryFlush] Appended flush summary to daily log`);

    // Extract and save facts from the flush response
    // processFactExtraction handles extraction, dedup, and saving internally
    try {
      // Fed the head text — the turns that are leaving — capped to the same
      // budget as one extraction pass, since this is its own model call with its
      // own prompt to fit. The summary above already covers the whole head; this
      // is the second reader, not the only one.
      await factExtractor.processFactExtraction(
        budgetHeadForExtraction(headText, maxExtractionTokens),
        flushSummary,
        provider,
        model,
        apiKey,
        host
      );
      console.log(`[MemoryFlush] Fact extraction from flush complete`);
    } catch (factError) {
      console.error(`[MemoryFlush] Error extracting facts from flush:`, factError.message);
    }

    // Compact messages: keep system message + the trailing window
    const systemMessage = messages.find(m => m.role === 'system');
    const recentMessages = messages.slice(-KEEP_RECENT_MESSAGES);

    const compactedMessages = [
      {
        role: 'system',
        content: '[Context was compacted to save space. Key points from earlier conversation were saved to memory.]'
      },
      ...(systemMessage && systemMessage !== recentMessages[0] ? [systemMessage] : []),
      ...recentMessages
    ];

    console.log(`[MemoryFlush] Compacted ${messages.length} messages to ${compactedMessages.length} messages`);

    return {
      compactedMessages,
      flushSummary
    };

  } catch (error) {
    console.error(`[MemoryFlush] Error during flush:`, error.message);
    // On error, return original messages unchanged
    return {
      compactedMessages: messages,
      flushSummary: ''
    };
  }
}

/**
 * Check if flush is needed and perform it if necessary
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {string} provider - LLM provider
 * @param {string} model - Model name
 * @param {string} apiKey - API key for provider
 * @param {string} host - Host URL for local providers
 * @param {string} memoryDir - Memory directory path
 * @returns {Promise<{messages: Array, flushed: boolean, flushResult: Object|null}>}
 */
async function checkAndFlush(messages, provider, model, apiKey, host, memoryDir = null) {
  try {
    // Ask the engine what it is serving before deciding anything. Cached for ten
    // minutes and bounded at 2s, so this is a no-op on all but the first turn
    // after a boot — and if it fails, the window falls back to the static table
    // and the flush behaves exactly as it did before.
    try { await require('./model-context').ensureProbed(provider, host, model); }
    catch (probeErr) { console.warn('[MemoryFlush] Context probe failed, using fallback:', probeErr.message); }

    const { needsFlush, usage, tokenCount, contextLimit } = shouldFlush(messages, model, provider, host);

    console.log(`[MemoryFlush] Context usage: ${(usage * 100).toFixed(1)}% (${tokenCount}/${contextLimit} tokens)`);

    if (needsFlush) {
      console.log(`[MemoryFlush] Flush threshold exceeded, performing memory flush`);
      const flushResult = await performFlush(messages, provider, model, apiKey, host, memoryDir);

      return {
        messages: flushResult.compactedMessages,
        flushed: true,
        flushResult
      };
    }

    return {
      messages,
      flushed: false,
      flushResult: null
    };

  } catch (error) {
    console.error(`[MemoryFlush] Error in checkAndFlush:`, error.message);
    // On error, return original messages
    return {
      messages,
      flushed: false,
      flushResult: null
    };
  }
}

module.exports = {
  estimateTokens,
  estimateMessagesTokens,
  getModelContextLimit,
  shouldFlush,
  performFlush,
  checkAndFlush,
  chunkMessages,
  KEEP_RECENT_MESSAGES
};
