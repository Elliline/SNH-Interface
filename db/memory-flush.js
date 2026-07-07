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
 * Get context window limit for a given model
 * @param {string} model - Model name/identifier
 * @returns {number} Context window size in tokens
 */
function getModelContextLimit(model) {
  if (!model || typeof model !== 'string') return 8192;

  const modelLower = model.toLowerCase();

  // Check from most specific to least specific
  if (modelLower.includes('o1') || modelLower.includes('o3') || modelLower.includes('o4')) {
    return 200000;
  }
  if (modelLower.includes('claude-opus-4')) return 200000;
  if (modelLower.includes('claude-sonnet-4')) return 200000;
  if (modelLower.includes('claude-haiku-4')) return 200000;
  if (modelLower.includes('gpt-5')) return 128000;
  if (modelLower.includes('gpt-4o')) return 128000;
  if (modelLower.includes('gpt-4-turbo')) return 128000;
  if (modelLower.includes('gpt-4')) return 8192;
  if (modelLower.includes('command')) return 128000;
  if (modelLower.includes('grok-4')) return 131072;
  if (modelLower.includes('grok-3')) return 131072;
  if (modelLower.includes('scout')) return 131072;
  if (modelLower.includes('llama')) return 131072;
  if (modelLower.includes('deepseek')) return 131072;
  if (modelLower.includes('qwen')) return 32768;
  if (modelLower.includes('mistral')) return 32768;
  if (modelLower.includes('phi')) return 16384;
  if (modelLower.includes('gemma')) return 8192;

  // Default fallback
  return 8192;
}

/**
 * Check if conversation needs flushing based on token count
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {string} model - Model name
 * @returns {{needsFlush: boolean, tokenCount: number, contextLimit: number, usage: number}}
 */
function shouldFlush(messages, model) {
  const tokenCount = estimateMessagesTokens(messages);
  const contextLimit = getModelContextLimit(model);
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
 * Make LLM API call for flush extraction (non-streaming)
 * @private
 */
async function callLLMForFlush(provider, model, messages, apiKey, host) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000);

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
            stream: false
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
            max_tokens: 4096,
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
            stream: false
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
            stream: false
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
            stream: false
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
            stream: false
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
            stream: false
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
  const memDir = memoryDir || path.join(__dirname, '../data/memory');
  const dailyDir = path.join(memDir, 'daily');

  try {
    const { tokenCount, contextLimit, usage } = shouldFlush(messages, model);
    console.log(`[MemoryFlush] Starting flush - conversation at ${(usage * 100).toFixed(1)}% of context (${tokenCount}/${contextLimit} tokens)`);

    // Build conversation text for extraction (user + assistant messages only)
    const conversationMessages = messages.filter(m => m.role === 'user' || m.role === 'assistant');
    const conversationText = conversationMessages
      .map(m => `${m.role.toUpperCase()}: ${m.content}`)
      .join('\n\n');

    // Truncate conversation text to keep extraction prompt under 50% of context
    const maxExtractionTokens = Math.floor(contextLimit * 0.5);
    let truncatedConversation = conversationText;
    if (estimateTokens(conversationText) > maxExtractionTokens) {
      const maxChars = maxExtractionTokens * 4;
      truncatedConversation = conversationText.slice(-maxChars);
      console.log(`[MemoryFlush] Truncated conversation for extraction to fit context`);
    }

    // Build extraction prompt
    const extractionMessages = [
      {
        role: 'system',
        content: 'You are a memory extraction system. Extract and save important facts, decisions, and context from this conversation.'
      },
      {
        role: 'user',
        content: `This conversation is getting long. Extract all important facts, decisions, preferences, action items, and technical details from the following conversation. Write them as bullet points.\n\n${truncatedConversation}`
      }
    ];

    // Call LLM for extraction
    console.log(`[MemoryFlush] Requesting extraction from ${provider}/${model}`);
    const flushSummary = await callLLMForFlush(provider, model, extractionMessages, apiKey, host);

    if (!flushSummary || flushSummary.trim().length === 0) {
      console.log(`[MemoryFlush] Warning: Empty flush summary received`);
    } else {
      console.log(`[MemoryFlush] Received flush summary (${flushSummary.length} chars)`);
    }

    // Save to daily log
    await factExtractor.appendToDailyLog(flushSummary, dailyDir);
    console.log(`[MemoryFlush] Appended flush summary to daily log`);

    // Extract and save facts from the flush response
    // processFactExtraction handles extraction, dedup, and saving internally
    try {
      await factExtractor.processFactExtraction(
        extractionMessages[1].content,
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

    // Compact messages: keep system message + last 10 messages
    const systemMessage = messages.find(m => m.role === 'system');
    const recentMessages = messages.slice(-10);

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
    const { needsFlush, usage, tokenCount, contextLimit } = shouldFlush(messages, model);

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
  checkAndFlush
};
