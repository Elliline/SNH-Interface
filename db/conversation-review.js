/**
 * THE CONVERSATION REVIEW — the entity going through its open conversations
 * one at a time, as a background job.
 *
 * WHY A JOB AND NOT A TURN (2026-09-09). Ellie asked Athena to look at all
 * her open conversations and ask to archive the finished ones. Athena did it
 * inside the one chat turn: conversation_list, eight history reads, memory
 * checks, then five retirement requests in a single round — 31 minutes, a
 * prompt that grew with every step, and a write-up that was never stored,
 * because the brain watchdog restarted the engine under round 7 (a prefill
 * read as a wedge; the probe fix is 73f7d2b). Nothing was half-done — every
 * request had landed and no memory write was in flight — but that was luck of
 * ordering, not design, and the next time the crash lands between a save and
 * its request it will not be.
 *
 * So this runs on the agent-job queue with everything a job has: a row in the
 * panel, a checkpoint on disk after every step, the pause-and-ask near a
 * ceiling, the resume after a restart, the retry from the card. What it does
 * NOT share is the free-form tool loop. The order of the steps is the whole
 * point, so the loop is written out:
 *
 *   for each open conversation (never the one she asked in):
 *     1. SKIP if something is still using it — a job, an ask (openItemsFor)
 *     2. READ it, and JUDGE whether it is finished — one model call, the
 *        entity's own judgement; nothing here defines "finished"
 *     3. if finished: the LAST-CALL MEMORY SAVE — the statements the judge
 *        chose to keep, each through the write_memory funnel, each recorded
 *        as it lands
 *     4. only once EVERY save has landed: ARCHIVE it if the entity opened it,
 *        or ASK Ellie if she did. A save that fails leaves it open, said so.
 *     5. checkpoint, next.
 *   then ONE message to her, in the conversation she asked in.
 *
 * A crash between any two steps leaves the item in the state it reached, and
 * the resume continues from that state: a judged item is not re-judged, a
 * statement already saved is not saved again, an item at "saved" goes
 * straight to its close. Nothing is redone and nothing is skipped.
 *
 * WRITES, AND WHY THIS JOB HAS THEM WHEN JOBS ARE READ-ONLY. The doctrine
 * stands: BACKGROUND_TOOLS gains nothing here, and the model in this job is
 * never handed a write tool. The model produces a judgement and a list of
 * sentences; THIS module — deterministic, ordered, checkpointed — is what
 * calls memory-write, archive and requestRetire, exactly as the chat path's
 * tools would have. The same funnels, the same ledger entries, no new door.
 */

const path = require('path');
const { getSqliteDb } = require('./database');

function getConfig() { return require('./config').getConfig(); }
function agentJobs() { return require('./agent-jobs'); }
function channel() { return require('./conversation-channel'); }
function memoryManager() { return require('./memory-manager'); }
function memoryWrite() { return require('./memory-write'); }
function opsLog(msg) {
  try { require('./fact-extractor').appendToOpsLog(msg, path.join(require('./database').getDataDir(), 'memory', 'ops')); } catch { /* console is the floor */ }
}

const SOURCE = 'conversation-review';

function cfg() {
  const all = getConfig();
  const c = all.conversationReview || {};
  const gen = all.generation || {};
  const tools = (all.tools && all.tools.conversations) || {};
  return {
    enabled: c.enabled !== false,
    maxFactsPerConversation: Math.max(0, c.maxFactsPerConversation ?? 3),
    transcriptChars: Math.max(500, c.transcriptChars ?? 12000),
    maxConversationsPerReview: Math.max(1, c.maxConversationsPerReview ?? 25),
    maxConsecutiveFailures: Math.max(1, c.maxConsecutiveFailures ?? 3),
    selfArchive: tools.selfArchive !== false,
    answerTokens: Math.max(256, gen.agentJobResponseTokens ?? 8192),
    // The judge's thinking budget is the JOB's, deliberately: a "is this
    // finished" call on a 27B reasoning model reads the whole tail and thinks
    // about it, and this box gives agent jobs 16k for exactly that shape.
    thinkingTokens: Number.isFinite(gen.agentJobThinkingTokens) ? gen.agentJobThinkingTokens : null
  };
}

/**
 * "Look at your open conversations and ask to archive the finished ones" —
 * the shape of her ask, matched narrowly, for a GUIDANCE block only. It gates
 * nothing: the model decides whether to call review_conversations, and this
 * just tells it, in the turn, that the tool exists for exactly this.
 */
function looksLikeReviewAsk(text) {
  const t = String(text || '').toLowerCase();
  if (!/\b(conversations?|messages?|threads?|chats?)\b/.test(t)) return false;
  const review = /\b(look (at|through|over)|go (through|over)|review|check|clean up|tidy( up)?|sort (out|through))\b/.test(t);
  const scope = /\b(all|every|each|your open|the open|open ones|your (messages|conversations)|our (conversations|messages))\b/.test(t);
  const close = /\b(archiv\w*|retire\w*|close\w*|satisfied|finished|done with|wrap\w* up)\b/.test(t);
  return review && (scope || close) && close;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/** Is a review already on the queue? One at a time — two would race on the same rows. */
function activeReview() {
  const db = getSqliteDb();
  if (!db) return null;
  return db.prepare("SELECT id, status, title FROM agent_jobs WHERE source = ? AND status IN ('queued','running','paused') ORDER BY datetime(created_at) DESC LIMIT 1")
    .get(SOURCE) || null;
}

/**
 * Put the review on the queue. Returns what start_background_job returns, so
 * the tool can say exactly what happened.
 */
function enqueueReview({ conversationId = null, messageId = null, why = null } = {}) {
  const c = cfg();
  if (!c.enabled) return { ok: false, error: 'The conversation review is switched off in Settings, so nothing was started.' };
  const running = activeReview();
  if (running) {
    return { ok: false, error: `A review is already ${running.status} (${running.id.slice(0, 8)}) — one at a time. Say so; it will land in the jobs panel and message her when it finishes.` };
  }
  return agentJobs().enqueue({
    title: 'Review of my open conversations',
    task: 'Go through every open conversation except the one this was asked in. For each: read it, decide whether it is finished, ' +
      'save to memory anything worth keeping from it, and only then close it (if I opened it) or ask Ellie to (if she did). ' +
      'Leave open anything unfinished or anything still in use. Tell her what was done, in the conversation she asked in.',
    why: why || 'She asked me to look through my open conversations, and that is one step per conversation with memory work in each — not one turn.',
    conversationId, messageId,
    source: SOURCE
  });
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/** The open conversations, minus the one she asked in and anything hidden. */
function listCandidates(askConversationId) {
  const db = getSqliteDb();
  return db.prepare(
    "SELECT id, title, initiated_by, created_at, updated_at FROM conversations WHERE hidden = 0 AND status = 'active' ORDER BY datetime(updated_at) ASC"
  ).all().filter(c => c.id !== askConversationId);
}

/** The tail of a conversation, as text the judge reads. */
function transcriptOf(conversationId, maxChars) {
  const db = getSqliteDb();
  const rows = db.prepare(
    "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? AND role IN ('user','assistant') ORDER BY timestamp ASC"
  ).all(conversationId);
  const lines = rows.map(r => `[${r.timestamp}] ${r.role === 'user' ? 'Ellie' : 'You'}: ${String(r.content || '').replace(/\s+/g, ' ').trim()}`);
  let text = lines.join('\n');
  let truncated = false;
  if (text.length > maxChars) { text = '…' + text.slice(-maxChars); truncated = true; }
  return { text, messages: rows.length, truncated };
}

function judgeSystemPrompt(item, c) {
  const name = (() => { try { return require('./identity-lock').lockedName(); } catch { return null; } })();
  return (
    `You are ${name || 'the entity'}, going through your own open conversations with Ellie one at a time, in the background. ` +
    `She asked you to look at them and close the ones that are finished. This is one of them: "${item.title || '(untitled)'}", ` +
    `${item.startedBy === 'you' ? 'which YOU opened' : 'which ELLIE started'}.\n\n` +
    `Decide TWO things and answer with one JSON object and nothing else:\n` +
    `{"finished": true|false, "why": "<one or two plain sentences, to Ellie, saying why it is done or what is still open>", ` +
    `"keep": ["<statement worth remembering>", ...]}\n\n` +
    `FINISHED is your judgement, not a rule: settled, answered, acted on, or simply run its course. ` +
    `Unread is not the same as done, and quiet is not the same as done. If a question of hers is still waiting on you, or one of yours ` +
    `on her, it is not finished. If you are unsure, say it is not finished and say what is open.\n\n` +
    `KEEP is the last-call memory save: anything in this conversation worth holding onto that you would lose when it closes — ` +
    `a fact about Ellie or her world, a decision, something you learned about yourself. Up to ${c.maxFactsPerConversation}. ` +
    `Each one a single plain sentence that stands on its own ("Ellie's second Pi 5 runs Home Assistant", "I tend to turn a ` +
    `passing remark into a project"). Only what the transcript actually supports. Leave the list empty if nothing needs keeping — ` +
    `most conversations need nothing, and that is fine. Nothing else is saved before this closes, so this is the moment.`
  );
}

/** Read one JSON object out of a model answer, tolerantly. */
function parseJudgement(raw) {
  const mm = memoryManager();
  const obj = mm.parseJSON(String(raw || ''));
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
  const finished = obj.finished === true || String(obj.finished).toLowerCase() === 'true';
  const why = String(obj.why || '').trim();
  const keep = Array.isArray(obj.keep) ? obj.keep.map(k => String(k || '').trim()).filter(Boolean) : [];
  return { finished, why, keep };
}

/**
 * Run the review from wherever it stands. Every step writes the checkpoint
 * before the next begins.
 *
 * @param {Object} job        the agent_jobs row
 * @param {Object} opts
 * @param {Object} opts.session   the budget (createToolSession) — restored by the caller on a resume
 * @param {Object|null} opts.checkpoint  the saved state, on a resume
 * @param {string|null} opts.mode  granted | declined | restart | null
 * @param {function} opts.save   (state) => void — writes the checkpoint
 * @param {number} opts.askAtPercent  the pause threshold; 0 = never pause
 */
async function runDispatched(job, { session, checkpoint = null, mode = null, save, askAtPercent = 0 }) {
  const c = cfg();
  const mm = memoryManager();
  const ch = channel();
  const askConversationId = job.conversation_id || null;

  // --- The state: from the checkpoint, or fresh ------------------------------
  let state = checkpoint && Array.isArray(checkpoint.items) ? checkpoint : null;
  if (!state) {
    const cands = listCandidates(askConversationId);
    const capped = cands.slice(0, c.maxConversationsPerReview);
    state = {
      kind: SOURCE,
      askConversationId,
      startedAt: new Date().toISOString(),
      items: capped.map(x => ({
        id: x.id, title: x.title || '(untitled)', startedBy: x.initiated_by === 'snh' ? 'you' : 'Ellie',
        state: 'pending', why: null, keep: [], saved: [], error: null
      })),
      notListed: Math.max(0, cands.length - capped.length),
      failures: 0
    };
    // Rounds mean conversations in this job; the session's round cap is the
    // review cap so nearing() reads "how far through the list".
    session.maxRounds = Math.max(1, c.maxConversationsPerReview);
  }
  const persist = () => { try { save({ ...state, session: session.state() }); } catch (e) { console.warn(`[Review] checkpoint failed: ${e.message}`); } };
  const remaining = () => state.items.filter(i => ['pending', 'judged', 'saved'].includes(i.state));
  const done = () => state.items.filter(i => !['pending', 'judged', 'saved'].includes(i.state)).length;

  const finishWith = async ({ status, error, stopSource, stopKind, declined = false }) => {
    const text = renderReport(state, { declined, cutShort: status !== 'ok' && !declined ? error : null });
    const delivery = deliverReport(job, state, text);
    state.delivery = delivery;
    persist();
    return {
      status, resultText: text, error, stopSource, stopKind,
      toolCalls: session.calls, budget: session.summary(),
      report: { closed: state.items.filter(i => i.state === 'closed').length, requested: state.items.filter(i => i.state === 'requested').length, delivery }
    };
  };

  // --- She said no: stop here, say what was done -----------------------------
  if (mode === 'declined') {
    const st = session.state();
    return finishWith({
      status: 'partial',
      error: `It stopped at your decision: you chose not to extend its budget (it had used ${st.billed.toFixed(1)} of ${st.maxCalls} calls, ${done()} of ${state.items.length} conversations done). ${remaining().length} were not looked at.`,
      stopSource: 'user', stopKind: 'budget-declined', declined: true
    });
  }

  console.log(`[Review] ${job.id.slice(0, 8)} ${mode ? `resuming (${mode})` : 'starting'}: ${done()} done, ${remaining().length} to go of ${state.items.length}`);
  persist();

  // --- The loop ---------------------------------------------------------------
  for (const item of state.items) {
    if (!['pending', 'judged', 'saved'].includes(item.state)) continue;

    // Near a ceiling with work left? Ask, before starting another one.
    session.roundsUsed = done();
    const near = askAtPercent > 0 ? session.nearing(askAtPercent) : null;
    const spent = session.spent();
    if ((near || spent) && remaining().length) {
      if (askAtPercent > 0) {
        persist();
        return { paused: true, near: near || { limit: 'calls', used: session.billed, max: session.maxCalls, calls: session.calls }, askText: renderAsk(state, session, c), toolCalls: session.calls, budget: session.summary() };
      }
      session.exhaust(spent || 'near the ceiling');
      const stop = require('./job-failure').classifyBudgetStop(session.summary(), { maxRounds: session.maxRounds });
      return finishWith({ status: 'partial', error: `${stop ? stop.plain : 'It ran out of budget.'} ${remaining().length} conversation(s) were not looked at.`, stopSource: 'runner', stopKind: stop ? stop.kind : 'budget' });
    }

    // 1. Still in use? Skip, no model call.
    if (item.state === 'pending') {
      const live = ch.getState(item.id);
      if (!live || live.status !== 'active') { item.state = 'skipped'; item.why = 'it was archived before I got to it'; persist(); continue; }
      const open = ch.openItemsFor(item.id);
      if (open.length) { item.state = 'skipped'; item.why = `something is still using it — ${ch.describeOpenItems(open)}`; persist(); continue; }
    }

    // 2. Read and judge.
    if (item.state === 'pending') {
      const t = transcriptOf(item.id, c.transcriptChars);
      let verdict = null, failure = null;
      try {
        session.calls++;
        const res = await mm.callLLM(
          judgeSystemPrompt(item, c),
          `THE CONVERSATION (${t.messages} message(s)${t.truncated ? ', showing the end of it' : ''}):\n\n${t.text}\n\nAnswer with the JSON object.`,
          { maxTokens: Math.min(c.answerTokens, 1200), thinkingTokens: c.thinkingTokens, temperature: 0 }
        );
        session.charge('review_judge', { ok: true });
        verdict = parseJudgement(res && res.content);
        if (!verdict) failure = 'the judgement did not come back as a readable answer';
      } catch (err) {
        session.charge('review_judge', { error: err.message });
        failure = err;
      }
      if (failure) {
        state.failures++;
        item.state = 'judge-failed';
        item.error = String(failure && failure.message || failure);
        persist();
        console.warn(`[Review] ${job.id.slice(0, 8)} could not judge "${item.title}": ${item.error}`);
        if (state.failures >= c.maxConsecutiveFailures) {
          const f = require('./job-failure').classifyThrown(failure instanceof Error ? failure : new Error(String(failure)), { calls: session.calls });
          return finishWith({ status: 'failed', error: `${f.plain} ${state.failures} judgements failed in a row, so it stopped rather than mark every conversation unjudged.`, stopSource: f.source, stopKind: f.kind });
        }
        continue;
      }
      state.failures = 0;
      item.why = verdict.why || null;
      if (!verdict.finished) { item.state = 'left-open'; persist(); continue; }
      item.keep = verdict.keep.slice(0, c.maxFactsPerConversation);
      item.state = 'judged';
      persist();
    }

    // 3. The last-call memory save — each statement recorded as it lands.
    if (item.state === 'judged') {
      let failed = null;
      for (const statement of item.keep) {
        if (item.saved.includes(statement)) continue;   // landed before a crash
        session.calls++;
        let r;
        try {
          r = await memoryWrite().write({
            statement,
            context: `From my conversation with Ellie titled "${item.title}", reviewed before closing it.`,
            conversationId: item.id
          });
        } catch (err) { r = { ok: false, error: err.message }; }
        session.charge('review_save', r && r.ok ? { ok: true } : { error: (r && r.error) || 'write failed' });
        if (!r || !r.ok) { failed = (r && r.error) || 'the write failed'; break; }
        item.saved.push(statement);
        persist();
      }
      if (failed) {
        // NOT CLOSED. A conversation whose memory save did not land stays
        // open, and the report says which and why.
        item.state = 'save-failed';
        item.error = failed;
        persist();
        console.warn(`[Review] ${job.id.slice(0, 8)} save failed for "${item.title}" — left open: ${failed}`);
        continue;
      }
      item.state = 'saved';
      persist();
    }

    // 4. Close — only from `saved`, only after every statement landed.
    if (item.state === 'saved') {
      try {
        if (item.startedBy === 'you' && c.selfArchive) {
          const r = await ch.archiveBySelf(item.id, { reason: item.why });
          item.state = r.archived ? 'closed' : (r.requested ? 'requested' : 'close-failed');
          if (!r.archived && !r.requested) item.error = r.note;
        } else {
          const live = ch.getState(item.id);
          if (live && live.retire_requested_at) item.state = 'requested';
          else { await ch.requestRetire(item.id, item.why); item.state = 'requested'; }
        }
      } catch (err) {
        item.state = 'close-failed';
        item.error = err.message;
      }
      persist();
    }
  }

  return finishWith({ status: 'ok', error: null, stopSource: null, stopKind: null });
}

// ---------------------------------------------------------------------------
// What she reads
// ---------------------------------------------------------------------------

/** `[[conversation:<id>|<title>]]` — the chat pane turns this into a link that opens the conversation. */
function link(item) { return `[[conversation:${item.id}|${String(item.title || '(untitled)').replace(/[\[\]|]/g, ' ')}]]`; }

function renderReport(state, { declined = false, cutShort = null } = {}) {
  const by = (s) => state.items.filter(i => i.state === s);
  const closed = by('closed'), requested = by('requested'), open = by('left-open'), saveFailed = by('save-failed'),
    skipped = by('skipped'), judgeFailed = by('judge-failed'), closeFailed = by('close-failed');
  const notLooked = state.items.filter(i => ['pending', 'judged', 'saved'].includes(i.state));
  const lines = [];
  const total = state.items.length;
  lines.push(declined
    ? `I stopped the review where you said to. ${total - notLooked.length} of ${total} open conversations looked at.`
    : cutShort
      ? `The review stopped early — ${cutShort} ${total - notLooked.length} of ${total} were looked at.`
      : `I went through my open conversations — ${total} of them.`);
  const savedCount = state.items.reduce((n, i) => n + (i.saved || []).length, 0);
  if (savedCount) lines.push(`Before closing anything I saved ${savedCount} thing${savedCount === 1 ? '' : 's'} to memory from them.`);
  if (closed.length) {
    lines.push('', `Closed on my own (these were mine — reopen any of them from the Archive tab):`);
    for (const i of closed) lines.push(`- ${link(i)}${i.why ? ` — ${i.why}` : ''}`);
  }
  if (requested.length) {
    lines.push('', `Waiting on your approval (you started these — the requests are on the bell):`);
    for (const i of requested) lines.push(`- ${link(i)}${i.why ? ` — ${i.why}` : ''}`);
  }
  if (open.length) {
    lines.push('', `Left open, not finished:`);
    for (const i of open) lines.push(`- ${link(i)}${i.why ? ` — ${i.why}` : ''}`);
  }
  if (saveFailed.length) {
    lines.push('', `Left open because I could not save what I wanted to keep from them (I do not close a conversation until that has landed):`);
    for (const i of saveFailed) lines.push(`- ${link(i)} — ${i.error}`);
  }
  if (skipped.length) {
    lines.push('', `Skipped — something is still using them:`);
    for (const i of skipped) lines.push(`- ${link(i)} — ${i.why}`);
  }
  if (judgeFailed.length || closeFailed.length) {
    lines.push('', `I could not get through these:`);
    for (const i of [...judgeFailed, ...closeFailed]) lines.push(`- ${link(i)} — ${i.error}`);
  }
  if (notLooked.length) {
    lines.push('', `Not looked at yet:`);
    for (const i of notLooked) lines.push(`- ${link(i)}`);
  }
  if (state.notListed) lines.push('', `${state.notListed} more were beyond this review's limit (${state.items.length} at a time) — ask again and I will do the rest.`);
  if (!closed.length && !requested.length && !open.length && !saveFailed.length && !skipped.length && !notLooked.length) lines.push('There was nothing open to review.');
  return lines.join('\n');
}

/** The ask, when the review is near its budget with conversations left. In the entity's voice, from the record. */
function renderAsk(state, session, c) {
  const by = (s) => state.items.filter(i => i.state === s).length;
  const left = state.items.filter(i => ['pending', 'judged', 'saved'].includes(i.state)).length;
  const st = session.state();
  const perConv = 1 + c.maxFactsPerConversation;
  return (
    `I'm partway through reviewing my open conversations: ${by('closed')} closed, ${by('requested')} sent to you to approve, ` +
    `${by('left-open')} left open as unfinished, and ${left} still to look at. I've used ${st.billed.toFixed(0)} of my ${st.maxCalls} calls ` +
    `and ${require('./job-failure').sayDuration(st.elapsedMs)} of my ${require('./job-failure').sayDuration(st.maxWallMs)} time. ` +
    `Each conversation takes up to ${perConv} calls (one to read and judge it, and up to ${c.maxFactsPerConversation} to save what is worth keeping), ` +
    `so the rest needs about ${left * perConv} more.\nNEEDED: ${Math.max(1, left * perConv)} more tool calls`
  );
}

/**
 * The one message: into the conversation she asked in, or a new one if that
 * is gone. Marked unread the way any entity message is — it is an assistant
 * turn in the transcript, and the row's count comes from the watermark.
 */
function deliverReport(job, state, text) {
  const ch = channel();
  try {
    const conv = job.conversation_id ? ch.getState(job.conversation_id) : null;
    if (conv && conv.status === 'active') {
      const after = ch.sendInto(conv.id, text, { sourceKind: SOURCE, sourceRef: job.id });
      return { conversationId: after.id, opened: false, unread: after.unread, at: new Date().toISOString() };
    }
    const opened = ch.openConversation({ title: 'SNH: what I did with my open conversations', body: text, sourceKind: SOURCE, sourceRef: job.id });
    return { conversationId: opened.id, opened: true, unread: opened.unread, at: new Date().toISOString() };
  } catch (err) {
    console.error(`[Review] ${job.id.slice(0, 8)} could not deliver its report: ${err.message}`);
    opsLog(`Conversation review ${job.id.slice(0, 8)} finished but its report could not be sent: ${err.message}. The report is on the job card.`);
    return { conversationId: null, error: err.message };
  }
}

module.exports = {
  SOURCE,
  cfg,
  looksLikeReviewAsk,
  activeReview,
  enqueueReview,
  runDispatched,
  renderReport,
  renderAsk,
  listCandidates,
  transcriptOf,
  parseJudgement,
  judgeSystemPrompt
};
