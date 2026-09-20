/**
 * Engine artifacts — tool-call syntax that leaks into the assistant's PROSE.
 *
 * A model asked for something it has no tool for does not simply decline. It
 * writes the call it wishes it could make, as text, in the middle of its answer:
 *
 *     <function=memory_jobs>{"status": "approved"}</function>
 *
 * That is not a tool call. It is not in `message.tool_calls`, nothing executes
 * it, and nothing catches it — it is ordinary content, and it streams to the
 * browser and renders. On 2026-08-06 that is exactly what Ellie saw when she
 * asked which approved job never ran: the markup, and then an invented answer.
 *
 * There are two halves to that failure and this is the second one. The first is
 * that he had no memory_jobs to call; that is fixed by giving him one. But a
 * missing tool is a permanent condition of any system — there is always
 * something it cannot do — so the markup must never render regardless.
 *
 * WHY A STREAMING FILTER RATHER THAN A REGEX AT THE END. The answer is streamed
 * token by token, so by the time a complete `</function>` has arrived the `<f`
 * has already been written to the client. The filter therefore HOLDS BACK any
 * tail that could still turn into an opener, and releases it once it cannot.
 * Held text is emitted intact the moment it is proven innocent, so ordinary prose
 * containing a `<` is delayed by a few characters and never altered.
 *
 * PURE and synchronous. No database, no model, no network.
 */

/**
 * Openers that begin a text-form tool call, with the closer that ends each.
 *
 * Deliberately a short list of UNAMBIGUOUS engine syntax. A looser rule — say,
 * any JSON object with a "name" key — would eat legitimate answers about JSON,
 * and this filter runs on every token of every reply.
 */
const ARTIFACTS = [
  { open: /<function\s*=/i, close: '</function>', label: 'function=' },
  { open: /<tool_call\b/i, close: '</tool_call>', label: 'tool_call' },
  { open: /<\|tool_calls_begin\|>/i, close: '<|tool_calls_end|>', label: 'tool_calls_begin' },
  { open: /<\|python_tag\|>/i, close: '<|eom_id|>', label: 'python_tag' },
  { open: /\[TOOL_CALL\]/i, close: '[/TOOL_CALL]', label: 'TOOL_CALL' },
  { open: /<\|assistant\|>\s*<\|tool\|>/i, close: '<|end|>', label: 'assistant-tool' }
];

/**
 * The longest prefix of `s` that could still grow into one of the openers.
 * A partial `<fun` at the end of a chunk must be held; a `<b>` cannot become one
 * and is released immediately.
 */
const OPENER_HEADS = ['<function=', '<tool_call', '<|tool_calls_begin|>', '<|python_tag|>', '[TOOL_CALL]', '<|assistant|>'];
const MAX_HOLD = Math.max(...OPENER_HEADS.map(h => h.length)) + 2;

function pendingOpenerLength(s) {
  for (let n = Math.min(MAX_HOLD, s.length); n > 0; n--) {
    const tail = s.slice(-n).toLowerCase();
    if (OPENER_HEADS.some(h => h.toLowerCase().startsWith(tail))) return n;
  }
  return 0;
}

/**
 * A stateful filter over one streamed reply.
 *
 * @returns {{feed: (chunk: string) => string, flush: () => string, stripped: () => number, visible: () => string}}
 */
function createToolArtifactFilter() {
  let buffer = '';        // text seen but not yet released
  let inside = null;      // the artifact currently being swallowed, if any
  let strippedCount = 0;
  let visibleText = '';

  /** Release text to the caller and remember that it was shown. */
  const emit = (t) => { visibleText += t; return t; };

  function feed(chunk) {
    buffer += String(chunk ?? '');
    let out = '';

    for (;;) {
      if (inside) {
        // Swallowing. Look for the closer; if it has not arrived, keep everything
        // and wait — an unterminated artifact is swallowed whole at flush.
        const idx = buffer.toLowerCase().indexOf(inside.close.toLowerCase());
        if (idx === -1) return out;
        buffer = buffer.slice(idx + inside.close.length);
        inside = null;
        strippedCount++;
        continue;
      }

      // Not inside one. Find the earliest opener in what we hold.
      let best = null;
      for (const a of ARTIFACTS) {
        const m = buffer.match(a.open);
        if (m && (best === null || m.index < best.index)) best = { index: m.index, a };
      }

      if (best) {
        out += emit(buffer.slice(0, best.index));
        buffer = buffer.slice(best.index);
        inside = best.a;
        continue;
      }

      // No opener. Release everything except a tail that could still become one.
      const hold = pendingOpenerLength(buffer);
      if (hold === 0) { out += emit(buffer); buffer = ''; return out; }
      out += emit(buffer.slice(0, buffer.length - hold));
      buffer = buffer.slice(buffer.length - hold);
      return out;
    }
  }

  /**
   * End of stream. Anything still held is released, EXCEPT an artifact that never
   * closed — a reply that ends mid-call is the commonest shape of this failure and
   * the fragment is exactly what must not be shown.
   */
  function flush() {
    if (inside) { strippedCount++; inside = null; buffer = ''; return ''; }
    const rest = buffer;
    buffer = '';
    return emit(rest);
  }

  return {
    feed,
    flush,
    stripped: () => strippedCount,
    visible: () => visibleText
  };
}

/**
 * One-shot form, for text that is already complete.
 * @returns {{text: string, stripped: number}}
 */
function stripToolArtifacts(text) {
  const f = createToolArtifactFilter();
  const out = f.feed(String(text ?? '')) + f.flush();
  return { text: out, stripped: f.stripped() };
}

/**
 * What he says when the markup WAS the answer.
 *
 * Stripping an artifact can leave nothing behind, because the model wrote the
 * call INSTEAD of a reply rather than alongside one. Emitting an empty message
 * would be its own kind of lie — the question would look answered. This is the
 * honest floor, and it names the limit rather than apologising for it.
 */
const CANNOT_CHECK =
  "I can't check that — I don't have a tool for it, and I'm not going to guess at the answer.";

module.exports = { createToolArtifactFilter, stripToolArtifacts, CANNOT_CHECK, ARTIFACTS };
