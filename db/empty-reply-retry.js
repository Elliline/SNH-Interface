/**
 * AN EMPTY REPLY IS A FAILURE, NOT AN ANSWER — AND IT GETS ONE MORE GO.
 *
 * Measured on aiserver, 2026-08-26. Three turns in one conversation came back
 * from the engine with `finish_reason: stop` and nothing in them. Two of the
 * three carried no reasoning either — vLLM's own counter read 0.1 tokens/s
 * against a full prefill, so the model was handed the prompt and emitted a
 * single end-of-turn token. 09:32:13 and 09:47:30 both did exactly that.
 *
 * WHAT THAT COST, BEFORE THIS EXISTED. server.js had one guard for the case,
 * and it was written for the OTHER half of it: `!fullResponse && fullReasoning`
 * — a reply that was all thinking and no answer. That guard is right and it
 * still fires. But the wholly-empty case fails its second condition, so it fell
 * straight through: nothing was written to the stream, and the save is guarded
 * on `if (fullResponse)`, so no row was stored either. The browser had already
 * rendered the bubble. The result is a blank bubble, no message in the database,
 * and not one line in the log saying a turn was lost. Ellie hit it twice in
 * forty minutes and both times fixed it by hand, by sending the same message
 * again — which is the whole of what this file automates.
 *
 * ONE RETRY. NOT TWO, AND NEVER A LOOP. A resend is cheap and it worked by hand
 * every time; a second resend is a system arguing with an engine that has told
 * it twice. If the retry is also empty the turn SURFACES — the honest sentence
 * goes to the browser and gets stored like any other reply, so the transcript
 * records that the turn happened and failed rather than quietly skipping it.
 *
 * EVERY OCCURRENCE IS LOGGED, both the ones the retry rescues and the ones it
 * does not, because "how often is this happening" is the question the next
 * person will ask and the silent version of this bug is why nobody could.
 */

/** What the model said when it said nothing, in its own words after a failed retry. */
const REASONING_ONLY_NOTE =
  'I thought about that and never actually wrote an answer — twice. My reasoning ran to the end '
  + 'of its budget both times without producing a reply. Ask me again, or put it a different way, '
  + 'and I should get there.';

const NOTHING_AT_ALL_NOTE =
  'My engine came back with nothing at all — no answer and no reasoning, twice in a row. That is a '
  + 'fault on my side, not anything you did. Ask me again and it should come through.';

/**
 * Is this reply empty, and in which way?
 *
 * The distinction matters because the two have different causes and deserve
 * different sentences: thinking that never landed on an answer is a budget or
 * prompt interaction, while nothing at all is the engine declining to generate.
 *
 * @param {string} content   - the answer text accumulated for this turn
 * @param {string} reasoning - the reasoning-channel text for the same turn
 * @returns {'reasoning-only'|'nothing-at-all'|null} null when there IS an answer
 */
function classifyEmptyReply(content, reasoning) {
  if (String(content ?? '').trim()) return null;
  return String(reasoning ?? '').trim() ? 'reasoning-only' : 'nothing-at-all';
}

/**
 * The sentence to send when the retry did not rescue the turn.
 * @param {'reasoning-only'|'nothing-at-all'} kind
 * @returns {string}
 */
function noteForEmptyReply(kind) {
  return kind === 'reasoning-only' ? REASONING_ONLY_NOTE : NOTHING_AT_ALL_NOTE;
}

/**
 * The content of a streamChat result, whichever wire shape it came back in.
 * streamChat returns the NON-streaming shape: OpenAI-style callers get
 * `{ choices: [{ message }] }`, Ollama-style callers get `{ message }`.
 * @param {object} data
 * @returns {{ content: string, reasoning: string }}
 */
function replyFromStreamResult(data) {
  const msg = (data && (data.choices?.[0]?.message || data.message)) || {};
  const reasoning = msg.reasoning ?? msg.reasoning_content;
  return {
    content: typeof msg.content === 'string' ? msg.content : '',
    reasoning: typeof reasoning === 'string' ? reasoning : ''
  };
}

module.exports = {
  classifyEmptyReply,
  noteForEmptyReply,
  replyFromStreamResult,
  REASONING_ONLY_NOTE,
  NOTHING_AT_ALL_NOTE
};
