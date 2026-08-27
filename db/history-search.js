/**
 * Conversation-history search — asking about the past without paying for it in
 * context.
 *
 * WHY THIS EXISTS. The entity's own past conversations are on disk, indexed,
 * and effectively unreachable. The only ways in were the semantic slice the
 * injector chooses for it (a handful of messages, chosen before the question
 * exists) and its own impression of what it remembers. So "what did the script
 * for Lincoln City Animal Clinic do?" had exactly two possible answers: the
 * lucky one where the injector happened to pick the right thread, and the
 * plausible one. On this system the plausible one has a name — it is the same
 * failure the capability manifest and renderActiveJobsBlock were both built
 * against, and it is the one that costs the most, because a reconstructed
 * memory is indistinguishable from a real one to the entity holding it.
 *
 * The obvious fix — let it read its own archive in the turn — trades one
 * problem for another. Archaeology is expensive: a dozen FTS hits, five windows
 * read around them, most of it discarded. Spent in the chat turn that is
 * exactly the context the answer is supposed to be for, and Ellie's next
 * question arrives to an entity that has spent its window rummaging.
 *
 * SO THE RUMMAGING HAPPENS SOMEWHERE ELSE. One background agent job with its
 * own thinking budget searches, reads around what it finds, and hands back a
 * digest measured in a few hundred tokens. The entity pays for the digest, not
 * for the search.
 *
 * ── THE DIGEST CONTRACT, which is the whole point of the file ──
 *
 * A digest that says "you told her the script polled their PMS every fifteen
 * minutes" is worth less than nothing if no conversation says that. It is worth
 * less than nothing specifically because it ARRIVES AS A TOOL RESULT — the
 * entity treats it as something it checked, not as something it recalled, and
 * a fabrication laundered through a tool is a fabrication with a citation.
 *
 * Three things enforce the contract, and only the first is a prompt:
 *
 *   1. The agent is TOLD to search, read, and quote, and told that finding
 *      nothing is a real answer it is allowed to give.
 *   2. Every quote it returns is CHECKED against the store before the digest is
 *      built — literal substring, whitespace-normalised, against the message it
 *      claims to be quoting. A quote that does not match is dropped and
 *      counted. This is what makes the contract a property of the code rather
 *      than a hope about the model.
 *   3. The REFERENCES ARE NOT THE MODEL'S TO WRITE. The agent returns a
 *      message id and a quote; the conversation title, the timestamp and the
 *      role are looked up here, from the row. It cannot get a citation wrong
 *      because it never supplies one.
 *
 * And the load-bearing consequence: IF NO QUOTE SURVIVES VERIFICATION, THE
 * SUMMARY IS DISCARDED TOO. Not softened, not labelled "unverified" — dropped,
 * and the digest says nothing was found. Paraphrase frames quotes; it never
 * substitutes for them, and a paraphrase with nothing behind it is precisely
 * the reconstruction this file exists to refuse. House doctrine on this is
 * already written down (CLAUDE.md, "increasing specificity is not increasing
 * evidence") and it was written about a person's reports, not a model's — but
 * an agent gets no exemption from it for being ours.
 *
 * ── WHAT IT WILL NOT LOOK AT ──
 *
 * Hidden conversations, by the `conversations.hidden` flag and nothing else.
 * Verification residue and clone artifacts are the reason: this box has been
 * cloned for testing, and a store where 18 of 20 conversations were synthetic
 * has already happened here once. A search is a side door into context that
 * did not exist when the "delete the test turns afterwards" rule was written,
 * and the flag is what closes it. The filter is a JOIN, not a WHERE on the FTS
 * table, which also drops the other leak: `messages_fts` has no triggers and
 * foreign keys are off, so a deleted conversation can leave orphaned index rows
 * behind. Joining through `messages` and `conversations` means an orphan cannot
 * be a hit — there is no row to join to.
 *
 * ── SCOPE ──
 *
 * The entity's own conversations with Ellie. That is not a filter, it is what
 * this store IS: one instance, one entity, one human, and `data/chat.db` holds
 * nothing else. Conversations SNH itself opened (`initiated_by = 'snh'`) are
 * hers too and are included. If this store ever holds a second party's threads,
 * the scope stops being structural and this comment becomes a bug report.
 */

const { getSqliteDb } = require('./database');
const { formatLocalTime } = require('./datetime');

// Read config through the module object, not destructured at load — the rule
// db/agent-jobs.js follows, for the same two reasons: the process's current
// config, and a test that can substitute one without writing the live file.
function getConfig() { return require('./config').getConfig(); }
function memoryManager() { return require('./memory-manager'); }
function agentJobs() { return require('./agent-jobs'); }

/**
 * The `agent_jobs.source` for a history-search run.
 *
 * Same shape as coding-jobs' SOURCE and used the same way: it is what the job
 * runner dispatches on, and what keeps these rows out of Ellie's panel. They
 * are records of an in-turn read, not results waiting for her.
 */
const SOURCE = 'history-search';

/** The tools the run itself is given. Both read; neither can reach anything else. */
const RUN_TOOLS = ['history_find', 'history_read'];

function cfg() {
  const c = (getConfig().tools && getConfig().tools.historySearch) || {};
  const gen = getConfig().generation || {};
  return {
    enabled: c.enabled !== false,
    maxHits: Math.max(1, c.maxHits ?? 8),
    // HOW MUCH TEXT A HIT CARRIES. The lever that removes rounds — see find().
    snippetChars: Math.max(80, c.snippetChars ?? 700),
    windowBefore: Math.max(0, c.windowBefore ?? 1),
    windowAfter: Math.max(0, c.windowAfter ?? 2),
    maxWindow: Math.max(1, c.maxWindow ?? 4),
    messageChars: Math.max(200, c.messageChars ?? 1200),
    maxQuotes: Math.max(1, c.maxQuotes ?? 6),
    quoteChars: Math.max(80, c.quoteChars ?? 400),
    summaryChars: Math.max(100, c.summaryChars ?? 700),
    // THE CAP. See renderDigest() for what it costs and why it is this number.
    digestChars: Math.max(500, c.digestChars ?? 4000),
    maxToolCalls: Math.max(1, c.maxToolCalls ?? 6),
    maxRounds: Math.max(1, c.maxRounds ?? 2),
    maxWallClockMs: Math.max(5000, c.maxWallClockMs ?? 45000),
    undeliveredGraceMinutes: Math.max(0, c.undeliveredGraceMinutes ?? 30),
    // How long the chat turn will actually sit here. Longer than the run's own
    // wall clock so a run that stops cleanly is still reported as a result
    // rather than as a timeout — the gap is deliberate.
    waitMs: Math.max(5000, c.waitMs ?? 90000),
    answerTokens: Math.max(256, gen.agentJobResponseTokens ?? 8192),
    // ITS OWN THINKING BUDGET, AND THE ONE PLACE THAT DELIBERATELY DOES NOT
    // INHERIT generation.agentJobThinkingTokens.
    //
    // v1 read that key, and on this box it is set to 16384. The first live run
    // spent 74% of everything it generated on reasoning and came back at 192s
    // against a 90s wait — the digest was good and arrived where nobody could
    // see it. The budget was not wrong for what it was written for; a research
    // job that reads six web pages and forms a view should think. This job
    // searches, reads, and copies text out — the harness writes the references,
    // enforces the quotes and renders the digest, so there is nothing here for
    // a long deliberation to decide.
    //
    // The fallback is the LOCAL default, never the agent-job one: if this key
    // is ever removed from config, the right behaviour is a small budget, not a
    // silent return to 16k. 0 is honoured as "no thinking at all" by callLLM
    // (enable_thinking=false on this engine) and is a legitimate setting here.
    thinkingTokens: Number.isFinite(c.thinkingTokens) ? Math.max(0, c.thinkingTokens) : 128
  };
}

// ---------------------------------------------------------------------------
// Reading the store
// ---------------------------------------------------------------------------

/**
 * FTS5 needs its input quoted or it reads the question as syntax.
 *
 * Same treatment db/database.js gives a BM25 query and for the same reason: an
 * apostrophe, a colon or a bare `NOT` in a natural-language question is a MATCH
 * syntax error or, worse, a column reference. Every term becomes a quoted
 * literal. Terms of one character are dropped — they match everything and cost
 * the whole result set.
 */
function ftsQuery(text) {
  const cleaned = String(text || '').replace(/[^\w\s\-]/g, ' ');
  const terms = cleaned.split(/\s+/)
    .filter(w => w.length > 1)
    .map(w => `"${w.replace(/"/g, '')}"`);
  return terms.length ? terms.join(' OR ') : null;
}

const squash = (s) => String(s ?? '').replace(/\s+/g, ' ').trim();

function clip(text, n) {
  const t = String(text ?? '');
  return t.length > n ? `${t.slice(0, n)}…` : t;
}

/**
 * Search the message index.
 *
 * The two JOINs are the exclusion, and they are doing two different jobs. The
 * join to `conversations` is where `hidden = 0` is applied — a hidden thread
 * has no row that satisfies it and therefore cannot produce a hit. The join to
 * `messages` is the orphan guard: an FTS row whose message was deleted has
 * nothing to join to and drops out, which matters because nothing maintains
 * this index on delete.
 *
 * OR rather than AND across terms. A natural-language question carries filler
 * ("what did the script for Lincoln City Animal Clinic do") and an AND query
 * needs every word of it present in one message, which is almost never true.
 * BM25 does the discriminating — a message with "Lincoln", "City", "Clinic"
 * and "script" outranks one with only "do" — so the recall is worth having.
 *
 * @param {{query: string, limit?: number}} args
 * @returns {{hits: Array, returned: number, searched: string}|{error: string}}
 */
function find({ query, limit } = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'The conversation store is unavailable.' };

  const c = cfg();
  const n = Math.min(Math.max(1, limit || c.maxHits), c.maxHits);
  const match = ftsQuery(query);
  if (!match) {
    return { error: 'That search has no usable terms in it — give it words to look for.' };
  }

  let rows;
  try {
    rows = db.prepare(`
      SELECT
        messages_fts.message_id                AS message_id,
        m.conversation_id                      AS conversation_id,
        c.title                                AS conversation_title,
        m.role                                 AS role,
        m.timestamp                            AS timestamp,
        m.content                              AS content,
        LENGTH(m.content)                      AS content_len,
        snippet(messages_fts, 0, '', '', '…', 64) AS snippet,
        bm25(messages_fts)                     AS score
      FROM messages_fts
      JOIN messages m      ON m.id = messages_fts.message_id
      JOIN conversations c ON c.id = m.conversation_id
      WHERE messages_fts MATCH ?
        AND c.hidden = 0
      ORDER BY bm25(messages_fts)
      LIMIT ?
    `).all(match, n);
  } catch (err) {
    console.error('[HistorySearch] find failed:', err.message);
    return { error: `The search failed: ${err.message}` };
  }

  // A HIT CARRIES ENOUGH TEXT TO QUOTE FROM. This is the round-killer.
  //
  // v1 returned a 20-token snippet, which is a locator and nothing more — so
  // every hit worth quoting cost a history_read, and every read cost a round,
  // and a round on this engine costs tens of seconds. The dogs question spent
  // four rounds and 122 seconds to end up quoting seven messages, none of them
  // longer than 650 characters. All seven could have come back in the first
  // call.
  //
  // So a message shorter than snippetChars comes back WHOLE and is marked
  // `complete`, which means it can be quoted directly with no second call. A
  // longer one comes back as the match-centred FTS snippet (64 tokens is that
  // function's hard maximum) and is marked incomplete, which is the signal to
  // read it properly before quoting — an excerpt has elision in it, and a quote
  // that spans an elision is a quote that fails verification, correctly.
  //
  // The cost of being generous is prefill, which is milliseconds. The cost of
  // being stingy is a round, which is not.
  return {
    searched: match,
    returned: rows.length,
    hits: rows.map(r => {
      const whole = r.content_len <= c.snippetChars;
      return {
        message_id: r.message_id,
        conversation_id: r.conversation_id,
        conversation_title: squash(r.conversation_title) || '(untitled)',
        role: r.role,
        // The run READS these, so they are on the instance's clock like
        // everything else it is shown. Handing the agent UTC and the digest
        // local time would give one lookup two clocks.
        timestamp: formatLocalTime(r.timestamp, { fallback: r.timestamp }),
        // Whitespace-squashed, exactly as verifyQuote squashes the stored
        // content before comparing — so text copied from here verifies.
        text: whole ? squash(r.content) : squash(r.snippet),
        complete: whole,
        quotable: whole
          ? 'This is the whole message. Quote from it directly; no history_read needed.'
          : `Excerpt of a ${r.content_len}-character message. Call history_read before quoting.`
      };
    })
  };
}

/**
 * The messages around one hit, so a quote can be read in the exchange it
 * belongs to rather than out of a snippet.
 *
 * Ordered by rowid, not by timestamp. Both are insertion order here, but
 * timestamps tie at one-second resolution and a tie makes the window
 * non-deterministic — a question and its answer can land in the same second,
 * and "which of these two came first" is exactly what the window is for.
 *
 * The same hidden/orphan JOIN as find(): a message id from anywhere, including
 * a stale one the model kept from an earlier call, cannot open a hidden thread.
 */
function readAround({ message_id, before, after } = {}) {
  const db = getSqliteDb();
  if (!db) return { error: 'The conversation store is unavailable.' };

  const c = cfg();
  const b = Math.min(Math.max(0, before ?? c.windowBefore), c.maxWindow);
  const a = Math.min(Math.max(0, after ?? c.windowAfter), c.maxWindow);

  const anchorId = String(message_id || '').trim();
  const anchor = db.prepare(`
    SELECT m.rowid AS rid, m.conversation_id, c.title
    FROM messages m
    JOIN conversations c ON c.id = m.conversation_id
    WHERE m.id = ? AND c.hidden = 0
  `).get(anchorId);

  if (!anchor) {
    // One message for both causes, deliberately. "No such message" and "that
    // conversation is not searchable" are the same fact from the model's side —
    // there is nothing there for it — and telling the two apart would let a
    // hidden thread be probed for existence one id at a time.
    return { error: 'No readable message with that id. It may not exist, or it may be in a conversation that is not searchable.' };
  }

  const rows = db.prepare(`
    SELECT id, role, content, timestamp, rowid AS rid
    FROM messages
    WHERE conversation_id = ? AND rowid BETWEEN ? AND ?
    ORDER BY rowid ASC
  `).all(anchor.conversation_id, anchor.rid - b, anchor.rid + a);

  return {
    conversation_id: anchor.conversation_id,
    conversation_title: squash(anchor.title) || '(untitled)',
    returned: rows.length,
    messages: rows.map(r => ({
      message_id: r.id,
      role: r.role,
      timestamp: formatLocalTime(r.timestamp, { fallback: r.timestamp }),
      is_hit: r.id === anchorId,
      text: clip(r.content, c.messageChars),
      truncated: String(r.content || '').length > c.messageChars
    }))
  };
}

/**
 * The row behind a quote, or null.
 *
 * The `hidden = 0` join is repeated here rather than trusted from the search
 * that produced the id, because this is the last gate before something reaches
 * the entity: verification is also the point at which a message from a hidden
 * thread would be laundered into the digest if the agent ever produced an id
 * from somewhere other than a search result.
 */
function messageForQuote(id) {
  const db = getSqliteDb();
  if (!db) return null;
  const wanted = String(id || '').trim();
  if (!wanted) return null;

  const SELECT = `
    SELECT m.id, m.role, m.content, m.timestamp,
           m.conversation_id, c.title AS conversation_title
    FROM messages m
    JOIN conversations c ON c.id = m.conversation_id
    WHERE c.hidden = 0 AND `;

  const exact = db.prepare(`${SELECT} m.id = ?`).get(wanted);
  if (exact) return exact;

  // A UNIQUE PREFIX IS ALSO AN IDENTIFICATION, and refusing one would throw away
  // correct quotes for a clerical reason.
  //
  // Everything else in this system talks about ids by their first eight
  // characters — the panel, the ops log, every console line, this file's own
  // references — so a model reading its own digest format has been shown the
  // short form far more often than the long one, and shortening the id it hands
  // back is the natural mistake. Rejecting that quote would report "not in the
  // store" about a passage that is verbatim in the store, which is a lie in the
  // one direction this file is built to avoid: it would make the honest
  // nothing-found answer unreliable, and an unreliable nothing-found is worse
  // than no nothing-found at all.
  //
  // AND IT IS STILL AN IDENTIFICATION, not a guess. Eight hex characters or
  // more, and EXACTLY ONE row may match — two matches is ambiguity and ambiguity
  // resolves to null, because picking one of two would be the tool deciding
  // which conversation the entity meant. The hidden filter is inside the same
  // query, so this cannot become a way to reach a hidden thread by prefix.
  if (wanted.length < 8 || !/^[0-9a-fA-F-]+$/.test(wanted)) return null;
  const matches = db.prepare(`${SELECT} m.id LIKE ? || '%' LIMIT 2`).all(wanted);
  return matches.length === 1 ? matches[0] : null;
}

// ---------------------------------------------------------------------------
// Checking what came back
// ---------------------------------------------------------------------------

/**
 * Strip what a model adds to a quote without meaning to change it.
 *
 * Surrounding quotation marks, a leading or trailing ellipsis where it elided
 * something, and whitespace/newline differences from the JSON round trip. None
 * of these are the model altering the words — and refusing a correct quote over
 * a smart-quote character would make the check useless in exactly the cases it
 * is supposed to pass.
 *
 * What is NOT normalised away: any change to the words themselves. That is the
 * whole check.
 */
function normaliseQuote(text) {
  let t = String(text ?? '');
  t = t.replace(/[‘’]/g, "'").replace(/[“”]/g, '"');
  t = t.replace(/^\s*["'`]+/, '').replace(/["'`]+\s*$/, '');
  t = t.replace(/^\s*(?:\.\.\.|…)\s*/, '').replace(/\s*(?:\.\.\.|…)\s*$/, '');
  return squash(t);
}

/**
 * Check one claimed quote against the message it claims to be from.
 *
 * Literal containment after whitespace normalisation. Not fuzzy, not a
 * similarity score: a quote is either in the record or it is not, and a
 * threshold here would be a dial for how much invention is acceptable.
 *
 * @returns {{ok: true, ref: Object, quote: string} | {ok: false, why: string}}
 */
function verifyQuote({ message_id, quote }) {
  const claimed = normaliseQuote(quote);
  if (claimed.length < 8) {
    return { ok: false, why: 'the quote was too short to be evidence of anything' };
  }
  const row = messageForQuote(message_id);
  if (!row) {
    return { ok: false, why: `no readable message ${String(message_id || '').slice(0, 8)}` };
  }
  if (!squash(row.content).includes(claimed)) {
    return { ok: false, why: `not found in message ${String(message_id).slice(0, 8)}` };
  }
  return {
    ok: true,
    quote: claimed,
    ref: {
      message_id: row.id,
      conversation_id: row.conversation_id,
      conversation_title: squash(row.conversation_title) || '(untitled)',
      role: row.role,
      timestamp: row.timestamp
    }
  };
}

/**
 * Pull the run's JSON out of whatever it actually wrote.
 *
 * A fenced block, a bare object, or prose with an object in it. Returns null
 * rather than guessing at a shape — a run whose answer cannot be read is
 * reported as one, because the alternative is inventing structure for it and
 * this file does not do that anywhere else either.
 */
function parseRunOutput(text) {
  const raw = String(text || '').trim();
  if (!raw) return null;

  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidates = [];
  if (fenced) candidates.push(fenced[1]);
  candidates.push(raw);
  const first = raw.indexOf('{'), last = raw.lastIndexOf('}');
  if (first !== -1 && last > first) candidates.push(raw.slice(first, last + 1));

  for (const c of candidates) {
    try {
      const parsed = JSON.parse(String(c).trim());
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    } catch { /* try the next shape */ }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Building the digest
// ---------------------------------------------------------------------------

function refLine(n, ref) {
  // THE LINE THE WHOLE TIMEZONE LAYER CAME OUT OF (2026-08-27).
  //
  // This used to hand the stored value straight through with the 'T' swapped
  // for a space, which printed raw UTC: Athena read a morning's conversations
  // back as "1:09 PM" to "1:24 PM" and knew they had happened at breakfast.
  //
  // Note what it is NOT doing now, either: `new Date(ts).toLocaleString()` would
  // still be wrong here, because messages.timestamp is SQLite's
  // CURRENT_TIMESTAMP — UTC with no marker, which V8 reads as local. That is
  // what formatLocalTime is for; see db/datetime.js.
  const when = formatLocalTime(ref.timestamp, { fallback: 'time unknown' });
  return `[${n}] ${clip(ref.conversation_title, 60)} · conv ${ref.conversation_id.slice(0, 8)}` +
    ` · ${ref.role} at ${when} · msg ${ref.message_id.slice(0, 8)}`;
}

/**
 * Render the digest, and hold it under the cap.
 *
 * THE CAP IS 4000 CHARACTERS (~1000 tokens), `tools.historySearch.digestChars`.
 * Sized against what it displaces, not against what would be nice to have: the
 * whole capability manifest is injected into every turn at a 700-token budget,
 * and this lands in the same window. Six quotes at 400 characters plus their
 * references and a 700-character summary comes to roughly 3,800 — so in normal
 * use the per-part limits bind and this one never does. It is the backstop for
 * the case they miss, which is the case where it matters.
 *
 * ── WHAT GIVES WAY, IN ORDER, AND WHY THAT ORDER ──
 *
 * A cut quote is not a quote. It is a paraphrase with quotation marks around
 * it, presented as the record, and there is no size of context window worth
 * buying with one. So the cap is never paid for out of the quotes' accuracy —
 * it is paid, in this order, out of everything else:
 *
 *   1. THE SUMMARY SHRINKS, then goes entirely. It is the run's framing, and
 *      framing is the one part of a digest that is not evidence.
 *   2. WHOLE QUOTES ARE DROPPED, lowest-ranked first, and the digest says how
 *      many went so the reader knows the list is partial.
 *   3. FAILING BOTH — one quote, no summary, and still over — the digest keeps
 *      the REFERENCE and drops the quote text, saying it did. That is the last
 *      honest move available: it still tells the entity where to look, and it
 *      does not hand it a sentence that reads as the record and is not.
 *
 * This ordering was not the first version. The first one truncated whatever was
 * left at the cap, which under a small budget produced a digest ending
 * mid-reference with a half-quote above it — evidence-shaped, unverifiable, and
 * exactly the artefact the verification pass upstream exists to prevent. The
 * test that caught it is scripts/test-history-search.js §8, and it asserts the
 * property rather than the size: every rendered quote is still a literal
 * substring of the message it cites, at any cap.
 */
function renderDigest({ question, summary, quotes, gaps, stats, dropped }) {
  const c = cfg();
  const head = `Conversation-history search — "${clip(squash(question), 160)}"`;
  const scope = `Searched your own conversations with Ellie. Hidden and test conversations are excluded.`;

  // NOTHING FOUND. Said as one thing, whatever the reason, and never dressed up
  // with the summary — a summary with no surviving quote behind it IS the
  // reconstruction this tool exists to prevent, so it is discarded here rather
  // than shown with a caveat.
  if (!quotes.length) {
    const why = stats.hits > 0
      ? `${stats.hits} message(s) matched the search terms, and nothing in them answered this.`
      : `Nothing in your conversations matched the search terms.`;
    const lost = dropped.length
      ? `${dropped.length} passage(s) the run offered could not be matched back to the store and were dropped: ${dropped.slice(0, 3).map(d => d.why).join('; ')}.`
      : '';
    return [
      head,
      '',
      'NOTHING FOUND.',
      why,
      lost,
      '',
      'You have no quotes and no record here. Tell her you looked and found nothing.',
      'Do NOT describe what those conversations said, or what the thing she asked about probably was.'
    ].filter(Boolean).join('\n').replace(/\n{3,}/g, '\n\n');
  }

  const tail = [];
  if (gaps) tail.push(`Not found: ${clip(squash(gaps), 300)}`);
  tail.push(`Read ${stats.messagesRead} message(s) from ${stats.hits} search hit(s).`);
  if (dropped.length) {
    tail.push(`${dropped.length} passage(s) the run offered were NOT in the store and were dropped — treat that as a warning about the rest.`);
  }
  tail.push('Every line in quotes above is verbatim from the message it cites. Anything else here is the run\'s framing, not the record.');

  const render = (q, i) => `${refLine(i + 1, q.ref)}\n    "${clip(q.quote, c.quoteChars)}"`;
  const assemble = (summaryText, body, note) => [
    head,
    '',
    ...(summaryText ? [summaryText, ''] : []),
    ...body,
    '',
    ...(note ? [note] : []),
    ...tail,
    '',
    scope
  ].join('\n');

  const full = clip(squash(summary || ''), c.summaryChars);
  // Step 1: the summary is the first thing to give, because it is the only part
  // of this that is not evidence.
  for (const s of [full, full ? clip(squash(summary), 240) : '', '']) {
    const out = assemble(s, quotes.map(render), null);
    if (out.length <= c.digestChars) return out;
    if (!s) break;
  }

  // Step 2: whole quotes, lowest-ranked first. Never a partial one.
  let kept = quotes.slice();
  while (kept.length > 1) {
    kept.pop();
    const note = `(${quotes.length - kept.length} further quote(s) left out to keep this short — ask again, more narrowly, for those.)`;
    const out = assemble('', kept.map(render), note);
    if (out.length <= c.digestChars) return out;
  }

  // Step 3: one quote and still over. Keep the reference, drop the words, and
  // say which happened — a pointer is useful, a mangled quote is not.
  const only = kept[0];
  const note = quotes.length > 1
    ? `(${quotes.length - 1} further quote(s) left out to keep this short.)`
    : null;
  const stub = [
    refLine(1, only.ref),
    `    [the passage at this reference is longer than the ${c.digestChars}-character digest cap, so it is NOT quoted here. Do not reconstruct it — ask again for this one conversation.]`
  ];
  const out = assemble('', stub, note);
  // The reference block alone cannot realistically exceed a 500-character floor,
  // but a hard slice is kept as the last resort. It can now only ever cut the
  // bracketed note above, never a quote.
  return out.length <= c.digestChars ? out : `${out.slice(0, Math.max(0, c.digestChars - 30))}\n… [cut at the size cap]`;
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/**
 * The run's instructions. SHORT ON PURPOSE.
 *
 * v1's prompt was ~2,400 characters of careful argument about honesty, and it
 * worked — the first live run produced a clean, correctly-quoted digest. It
 * also took 192 seconds, and 74% of what it generated was reasoning. A long
 * prompt full of considerations invites a model to consider them, one at a
 * time, out loud, before doing a job that is: search, read what is marked as an
 * excerpt, copy sentences, emit JSON.
 *
 * So the argument moved into the code, where it belongs and where it cannot be
 * reasoned around. The quotes are checked whatever this prompt says; the
 * references are written by the harness whether or not it is told not to write
 * them; a digest with no surviving quote loses its summary automatically. What
 * is left here is the operating procedure plus the two facts the model needs in
 * order to behave sensibly rather than defensively: that quotes are verified,
 * and that nothing-found is an accepted answer.
 *
 * The contract did not get weaker. It got moved somewhere a shorter prompt
 * cannot erode.
 */
function runPrompt(question) {
  const c = cfg();
  return (
    `You look things up in your own conversation history with Ellie. This is an errand, not a problem to ` +
    `solve. Work fast and do not deliberate.\n\n` +
    `QUESTION: "${question}"\n\n` +
    `1. SEARCH. history_find(query) with the distinctive words — names, projects, terms. Skip the filler.\n` +
    `2. READ. Most hits come back COMPLETE and say so: quote straight from those, no second call. Only a ` +
    `hit marked as an excerpt needs history_read before you quote it.\n` +
    `3. QUOTE. Copy the passages that answer the question exactly, character for character, and emit the ` +
    `JSON below. Usually one search is enough. Stop as soon as you have what answers the question.\n\n` +
    `Every quote is checked against the database. One that is not literally in the message you attribute ` +
    `it to is thrown away, and if all of them are, the whole answer is discarded. Copy; never tidy, join ` +
    `or paraphrase inside a quote.\n\n` +
    `FINDING NOTHING IS A REAL ANSWER. Empty searches, or hits that do not answer this — return ` +
    `found=false with no quotes. Never write what a conversation probably said.\n\n` +
    `DO NOT WRITE THE REFERENCES. Titles, timestamps and speakers are read from the row. You give ids.\n\n` +
    `Output ONE JSON object, nothing else:\n` +
    `{"found":true|false,"summary":"<=${c.summaryChars} chars framing the quotes, or empty",` +
    `"quotes":[{"message_id":"id exactly as given","quote":"exact words, <=${c.quoteChars} chars"}],` +
    `"gaps":"one line on what you could not find, or empty"}\n` +
    `At most ${c.maxQuotes} quotes, best first.`
  );
}

/**
 * Per-run counters, keyed by the tool session's step name.
 *
 * The background tool loop hands a tool `{ caller: session.stepName }` and
 * nothing else, so this is how a history_find/history_read call finds the run
 * it belongs to. It exists so the digest can say "read 7 messages from 12 hits"
 * truthfully — those numbers are counted where the reads happen, not estimated
 * afterwards from the model's account of itself.
 */
const scopes = new Map();

function noteRead(caller, { hits = 0, messages = 0 } = {}) {
  const s = scopes.get(caller);
  if (!s) return;
  s.calls += 1;
  s.hits += hits;
  s.messagesRead += messages;
}

/**
 * Waiters on in-flight runs, so a tool call can return the digest itself.
 *
 * A SET PER JOB, not one callback per job. v1.1 lets a second ask JOIN a run
 * this conversation already has (see inFlightFor), and with a single-callback
 * map the joiner overwrote the original waiter — the first caller's promise
 * would then never settle and its turn would hang until its own timeout, which
 * is the orphaning bug reappearing one layer down.
 */
const waiting = new Map();

/**
 * Run one history-search job. Called by the agent-job runner on the SOURCE
 * branch, exactly as coding-jobs.runDispatched is.
 *
 * Never throws: it returns the outcome the job row is closed with, and settles
 * whatever chat turn is waiting on it. A run that dies still has to answer the
 * caller, and it answers honestly.
 */
async function runDispatched(job) {
  const c = cfg();
  const caller = `agent-job:${job.id.slice(0, 8)}`;
  const scope = { calls: 0, hits: 0, messagesRead: 0 };
  scopes.set(caller, scope);

  let outcome;
  let metrics = null;
  try {
    const mm = memoryManager();
    const MCPClient = require('../mcp/mcp-client');
    const allowed = MCPClient.shared().backgroundToolsAmong(RUN_TOOLS);
    if (!allowed.length) {
      // Loud and honest rather than a run with no way to look anything up. A
      // lookup that cannot look is not a lookup that found nothing.
      throw new Error('the history-search tools are not registered, so nothing could be searched');
    }

    const session = mm.createToolSession(caller, allowed, {
      maxCalls: c.maxToolCalls,
      maxWallMs: c.maxWallClockMs,
      maxRounds: c.maxRounds
    });

    const startedMs = Date.now();
    const res = await mm.callLLM(runPrompt(job.task), job.task, {
      maxTokens: c.answerTokens,
      thinkingTokens: c.thinkingTokens,
      toolSession: session
    });

    // WHERE THE TIME WENT, kept on the row. The v1 autopsy had to be
    // reconstructed from journalctl because nothing here recorded it; a
    // capability whose whole problem is latency must carry its own numbers.
    metrics = {
      totalMs: Date.now() - startedMs,
      roundMs: (res && res.roundMs) || [],
      reasoningChars: (res && res.reasoningChars) || 0,
      answerChars: String((res && res.content) || '').length,
      thinkingCap: c.thinkingTokens,
      toolCalls: scope.calls,
      hits: scope.hits,
      messagesRead: scope.messagesRead,
      budget: (res && res.budget) || session.summary()
    };

    const parsed = parseRunOutput(res && res.content);
    if (!parsed) {
      outcome = {
        status: 'partial',
        digest: renderDigest({
          question: job.task, summary: null, quotes: [], gaps: null,
          stats: scope, dropped: []
        }),
        error: 'the run did not return a readable answer, so nothing it might have found could be checked'
      };
    } else {
      const claimed = Array.isArray(parsed.quotes) ? parsed.quotes.slice(0, c.maxQuotes) : [];
      const quotes = [], dropped = [];
      for (const q of claimed) {
        const v = verifyQuote({ message_id: q && q.message_id, quote: q && q.quote });
        if (v.ok) quotes.push({ quote: v.quote, ref: v.ref });
        else dropped.push({ why: v.why });
      }
      if (dropped.length) {
        console.warn(`[HistorySearch] ${job.id.slice(0, 8)} offered ${claimed.length} quote(s), ${dropped.length} failed verification: ${dropped.map(d => d.why).join('; ')}`);
      }
      outcome = {
        status: quotes.length ? 'ok' : 'partial',
        digest: renderDigest({
          question: job.task,
          // The summary lives or dies with the quotes — see renderDigest.
          summary: quotes.length ? parsed.summary : null,
          quotes,
          gaps: parsed.gaps,
          stats: scope,
          dropped
        }),
        error: quotes.length
          ? (dropped.length ? `${dropped.length} of ${claimed.length} quote(s) were not in the store and were dropped` : null)
          : 'nothing was found, or nothing offered could be verified against the store',
        verified: quotes.length,
        rejected: dropped.length
      };
    }
  } catch (err) {
    console.error(`[HistorySearch] ${job.id.slice(0, 8)} failed:`, err.message);
    outcome = {
      status: 'failed',
      digest: [
        `Conversation-history search — "${clip(squash(job.task), 160)}"`,
        '',
        'THE SEARCH DID NOT RUN.',
        `It stopped with: ${err.message}`,
        '',
        'You have nothing from your history here. Say the lookup failed — do not answer as though it had succeeded and found nothing, and do not fill the gap from impression.'
      ].join('\n'),
      error: err.message
    };
  } finally {
    scopes.delete(caller);
  }

  // Every waiter, not just the first — a joined ask is waiting on this same run.
  const ws = waiting.get(job.id);
  if (ws) {
    waiting.delete(job.id);
    for (const fn of ws) {
      try { fn(outcome); } catch (err) { console.error('[HistorySearch] waiter threw:', err.message); }
    }
  }

  return {
    status: outcome.status,
    // The row keeps the digest verbatim. It is the record of what the entity was
    // actually told, which is the thing worth being able to check later.
    resultText: outcome.digest,
    error: outcome.error || null,
    toolCalls: scope.calls,
    metrics,
    digest: outcome.digest
  };
}

// ---------------------------------------------------------------------------
// The chat-side entry point
// ---------------------------------------------------------------------------

/** Wait for a run to settle, or give up at `ms`. Shared by the new and joined paths. */
function waitFor(jobId, ms) {
  return new Promise(resolve => {
    let done = false;
    const drop = () => {
      const set = waiting.get(jobId);
      if (!set) return;
      set.delete(fn);
      if (!set.size) waiting.delete(jobId);
    };
    const fn = (outcome) => {
      if (done) return;
      done = true; clearTimeout(timer); drop(); resolve(outcome);
    };
    const timer = setTimeout(() => {
      if (done) return;
      done = true; drop(); resolve(null);
    }, ms);
    const set = waiting.get(jobId) || new Set();
    set.add(fn);
    waiting.set(jobId, set);
  });
}

/**
 * The "not yet" answer — and it now says what happens next, which v1 could not.
 *
 * v1 told him to "offer to ask again", and asking again started a second run.
 * The digest is delivered on its own now: either he asks again and gets THIS
 * job's result (inFlightFor), or the announcement block hands it to him at the
 * start of a later turn. Both are true whatever he decides to say here, so he
 * is told them rather than left to invent a plan.
 */
function stillRunning(jobId, question, c) {
  return {
    ok: true,
    job_id: jobId,
    short_id: jobId.slice(0, 8),
    status: 'still-running',
    digest:
      `Conversation-history search — "${clip(question, 160)}"\n\n` +
      `STILL RUNNING after ${Math.round(c.waitMs / 1000)}s, so there is no digest yet.\n\n` +
      `You do not have an answer and must not describe one. Tell her the lookup is taking longer than the ` +
      `turn allows and say what you already know from what is in front of you.\n` +
      `IT IS NOT LOST: it will be handed to you when it finishes — either at the start of a later reply, ` +
      `or immediately if you call history_search again in this conversation, which joins this same lookup ` +
      `rather than starting another. So promising to come back to her is a promise you can keep.`
  };
}

/**
 * Stamp a digest as having reached the conversation that asked for it.
 *
 * Called on every path that hands a digest to the entity — the inline return,
 * and the repeat-ask that collects a finished one. What it turns off is the
 * late-delivery channel: a digest delivered here must never come back a second
 * time through the announcement block, which would have him report the same
 * lookup twice as though it were news.
 */
function markDelivered(jobId) {
  const db = getSqliteDb();
  if (!db || !jobId) return false;
  try {
    return db.prepare(
      'UPDATE agent_jobs SET delivered_at = ? WHERE id = ? AND delivered_at IS NULL'
    ).run(new Date().toISOString(), jobId).changes > 0;
  } catch (err) {
    console.error('[HistorySearch] markDelivered failed:', err.message);
    return false;
  }
}

/**
 * The history lookup this conversation already has in play, if any.
 *
 * ONE RUNNING LOOKUP PER CONVERSATION, and the reason is the failure it
 * prevents rather than the resource it saves. v1's "still running" reply handed
 * back a job_id and then had nothing that could use it: asking again started a
 * SECOND run over the same store for the same question, so the natural thing to
 * do next — she asks again, he tries again — doubled the load and left two
 * digests where nobody could see either.
 *
 * Two states count as "in play":
 *   running  — attach to it and wait, rather than racing it.
 *   finished but never delivered, within the grace window — that IS the answer.
 *              Hand it over now. This is the case that was orphaned.
 *
 * Scoped to the conversation, not global: two conversations asking about
 * different things are two questions, and a global lock would make one wait on
 * the other for no reason. A null conversationId (a script, a test) gets no
 * dedupe at all — there is no conversation to scope to, and quietly sharing one
 * lookup between unrelated callers is worse than running two.
 */
function inFlightFor(conversationId) {
  const db = getSqliteDb();
  if (!db || !conversationId) return null;
  const live = db.prepare(`
    SELECT * FROM agent_jobs
    WHERE source = ? AND conversation_id = ? AND status IN ('queued','running')
    ORDER BY datetime(created_at) DESC LIMIT 1
  `).get(SOURCE, conversationId);
  if (live) return { kind: 'running', row: live };

  const graceMs = cfg().undeliveredGraceMinutes * 60 * 1000;
  if (!graceMs) return null;
  const since = new Date(Date.now() - graceMs).toISOString();
  const undelivered = db.prepare(`
    SELECT * FROM agent_jobs
    WHERE source = ? AND conversation_id = ? AND delivered_at IS NULL
      AND result_text IS NOT NULL
      AND finished_at IS NOT NULL AND datetime(finished_at) > datetime(?)
    ORDER BY datetime(finished_at) DESC LIMIT 1
  `).get(SOURCE, conversationId, since);
  return undelivered ? { kind: 'undelivered', row: undelivered } : null;
}

/**
 * Ask the archive a question and wait for the digest.
 *
 * ONE JOB PER CALL, and no fan-out. The queue's own caps still apply — a call
 * arriving when ten jobs are already active is refused by enqueue() with a
 * reason, in words, rather than quietly queued behind them.
 *
 * WHAT THE WAIT IS. The chat turn blocks here, which is the point: the digest
 * has to come back INTO the conversation or the entity has nothing to answer
 * with. It is safe because chat's deadlines are gaps between tokens on an
 * engine call (db/config.js `chat`), and this is not one — it sits between two
 * of them. It is bounded because `waitMs` bounds it, and a run that outlives
 * the wait is reported as still running, with its id, and never as a result.
 */
async function ask({ question, conversationId = null, messageId = null } = {}) {
  const c = cfg();
  if (!c.enabled) {
    return { ok: false, error: 'Conversation-history search is switched off in configuration, so nothing was searched.' };
  }
  const q = squash(question);
  if (q.length < 3) {
    return { ok: false, error: 'Give the search an actual question about your past conversations.' };
  }

  // CHECK BEFORE STARTING. See inFlightFor() — a repeat ask must join the
  // lookup this conversation already has, never open a second one.
  const existing = inFlightFor(conversationId);
  if (existing && existing.kind === 'undelivered') {
    markDelivered(existing.row.id);
    console.log(`[HistorySearch] handed back the undelivered digest from ${existing.row.id.slice(0, 8)} instead of starting a new run`);
    return {
      ok: true,
      job_id: existing.row.id,
      short_id: existing.row.id.slice(0, 8),
      status: 'recovered',
      reused: true,
      digest:
        `[This is the lookup you started earlier and were told was still running. It finished; here it ` +
        `is. It answers: "${clip(squash(existing.row.task), 160)}" — say so if the conversation has moved ` +
        `on since.]\n\n${existing.row.result_text}`
    };
  }
  if (existing && existing.kind === 'running') {
    const joined = await waitFor(existing.row.id, c.waitMs);
    if (joined) {
      markDelivered(existing.row.id);
      return {
        ok: true, job_id: existing.row.id, short_id: existing.row.id.slice(0, 8),
        status: joined.status, reused: true,
        verified: joined.verified ?? 0, rejected: joined.rejected ?? 0,
        digest: joined.digest
      };
    }
    return stillRunning(existing.row.id, squash(existing.row.task), c);
  }

  const jobs = agentJobs();
  const started = jobs.enqueue({
    title: `history: ${clip(q, 60)}`,
    task: q,
    why: null,
    conversationId,
    messageId,
    source: SOURCE,
    // NOT charged to the background-job start budget. That cap (6/hour) bounds
    // work the entity hands off and walks away from; this is a read inside a
    // turn, and it is charged to the read budget the other read tools share.
    // Two counters on one action is how they come to disagree.
    countsAgainstStarts: false
  });
  if (!started.ok) return { ok: false, error: started.error };

  const id = started.id;
  const settled = await waitFor(id, c.waitMs);

  if (!settled) {
    // Honest, and specifically NOT a result. The run may still be going; what
    // it finds is not knowable from here and must not be described.
    return stillRunning(id, q, c);
  }

  markDelivered(id);
  return {
    ok: true,
    job_id: id,
    short_id: id.slice(0, 8),
    status: settled.status,
    verified: settled.verified ?? 0,
    rejected: settled.rejected ?? 0,
    digest: settled.digest
  };
}

module.exports = {
  SOURCE,
  markDelivered,
  inFlightFor,
  RUN_TOOLS,
  cfg,
  find,
  readAround,
  noteRead,
  verifyQuote,
  normaliseQuote,
  parseRunOutput,
  renderDigest,
  runPrompt,
  runDispatched,
  ask
};
