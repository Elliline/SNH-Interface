/**
 * The reasoning channel — one reader, used by every path that talks to a model.
 *
 * A reasoning model streams (and returns) its working out on a field of its own,
 * separate from the reply. There is no agreed name for that field: vLLM 0.27.1
 * with the qwen3 parser emits `reasoning`, verified against raw bytes in both the
 * streaming delta and the non-streaming message; DeepSeek-R1's API and several
 * other engines emit `reasoning_content`. Both are read here so this holds for
 * the next reasoning model rather than for this one.
 *
 * It lives in its own module because the chat path and the background paths both
 * need it, and the alternative — a copy in server.js and another in
 * db/memory-manager.js — is two implementations of one contract, which is the
 * shape of defect this codebase already has a rule about.
 *
 * REASONING IS NOT THE ANSWER. Nothing here folds it into content. It is read so
 * it can be measured, shown beside the reply, and named in an error when the
 * model spent its whole budget thinking and never answered — which otherwise
 * surfaces as an empty string with no cause attached.
 */

/**
 * The reasoning text on a streamed delta or a whole message, or '' if none.
 * @param {object} node - a `delta` or a `message` object from the provider
 * @returns {string}
 */
function extractReasoning(node) {
  if (!node) return '';
  const v = node.reasoning ?? node.reasoning_content;
  return typeof v === 'string' ? v : '';
}

/**
 * The reasoning on a whole non-streaming chat-completions response body,
 * whichever provider shape it came back in.
 * @param {object} data - parsed response JSON
 * @returns {string}
 */
function reasoningFromResponse(data) {
  if (!data) return '';
  return extractReasoning(data.choices?.[0]?.message)
    || extractReasoning(data.message)
    || '';
}

module.exports = { extractReasoning, reasoningFromResponse };
