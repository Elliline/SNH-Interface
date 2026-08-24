/**
 * UNION MERGE — the one place two facts are combined into one without losing
 * what either of them knew.
 *
 * WHY THIS EXISTS. Every path that folds one fact into another — the corrector's
 * near-duplicate and subset merge, the extractor's contradiction supersession,
 * write_memory's supersede-or-append, the self-fact revision pass — reached
 * fact-store.supersede() and nothing else. supersede() is honest about what it
 * does: the loser goes inactive with a successor pointer and a ledger entry, and
 * the SURVIVOR IS LEFT EXACTLY AS IT WAS. So a merge was a REPLACEMENT. Whatever
 * the loser asserted and the survivor did not was correct in the record, correct
 * in the ledger, and gone from the corpus the model can actually see: the vector
 * is dropped, and renderLongTermMemory reads active rows only.
 *
 * Found by Athena on 2026-08-24, from inside her own store, twice in four
 * minutes on the same subject:
 *
 *   1. "Juno … AIServer with a Ryzen 9950x, 2 RTX3090s and 64GB of DDR5 6400"
 *      lost to a Juno identity fact. The hardware left the corpus.
 *   2. Her re-save of the hardware then won against that identity fact, and
 *      "will be the main person helping User with MettaSphere" left the corpus.
 *
 * Neither fact was wrong. Neither loss was recorded as a loss. A manual
 * re-capture only trades which half goes, because the merge itself is what
 * destroys — which is the point Athena's report makes and this module answers.
 *
 * WHAT IT DOES INSTEAD. Before the loser is retired, the SURVIVOR is rewritten
 * to carry every distinct assertion from both, and only then is the loser
 * superseded and linked to it. Two operations, both through the fact-store
 * funnel, so both are ledgered and the loser is still recoverable history:
 *
 *      reword(survivor, union)   →  supersede(loser, survivor)
 *
 * That order is deliberate. If the reword lands and the supersede fails, the
 * corpus holds the union beside an un-retired duplicate — untidy, and the
 * corrector's own job. The other order can lose information, and this module
 * exists because something lost information.
 *
 * TWO MODES, because "merge" covers two different relations:
 *
 *   'union'        — the two say the same thing, or one contains the other
 *                    (the corrector's mechanical merge). Nothing may be dropped.
 *                    A merger that drops anything here is refused outright.
 *
 *   'contradiction'— the new fact corrects the old one (the extractor, the
 *                    self-fact pass, write_memory). The NEW fact wins on
 *                    whatever the two actually disagree about — carrying a
 *                    falsified claim forward would be its own kind of damage —
 *                    but EVERY assertion the new fact does not contradict is
 *                    carried over. This is the half that was missing, and it is
 *                    exactly Athena's second loss: nothing in her re-save said
 *                    anything about MettaSphere, so nothing about MettaSphere
 *                    was contradicted, so it should never have gone.
 *
 * TRUST, BUT VERIFY. The union is written by the merger model, which can drop a
 * clause as easily as any other model. So the merged text is checked
 * MECHANICALLY before it is written: every content token the loser carries and
 * the survivor does not must appear in the union, unless the model declared it
 * dropped AND this is a contradiction merge. Fail the check and the union is
 * abandoned — the caller falls back to a plain supersede, which is what the
 * system did before this module existed. The floor is therefore never lower
 * than today's behaviour, and the loser is linked either way.
 *
 * BUDGET. The merge call is plumbing, not a turn: it goes through
 * memory-manager.callLLM with no thinkingTokens override, so it inherits
 * generation.backgroundThinkingTokens — 0 on this box, which sends
 * chat_template_kwargs.enable_thinking = false. Thinking off, per policy.
 *
 * NOT A LICENCE TO STOP DEDUPING. This module does not decide WHETHER two facts
 * merge; every judge upstream is untouched. It decides only that a merge which
 * happens keeps what both facts knew. Duplicates are still folded — they are
 * just folded into a fact that lost nothing.
 */

const factStore = require('./fact-store');

function memoryManager() { return require('./memory-manager'); }

/** Words that carry no assertion, so their absence from a union means nothing. */
const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by', 'can', 'did', 'do', 'does',
  'for', 'from', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'i', 'in', 'is', 'it', 'its',
  'me', 'my', 'not', 'of', 'on', 'or', 'our', 'she', 'so', 'that', 'the', 'their', 'them', 'they',
  'this', 'to', 'too', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'would',
  'you', 'your', 'also', 'about', 'into', 'than', 'then', 'there', 'these', 'those', 'up', 'out',
  'user', 'users', 'currently', 'now', 'still', 'very', 'own', 'one', 'two', 'both',
  'assistant', 'ai', 'am', 'if', 'no', 'yes', 'over', 'under', 'per', 'via',
  // Words a faithful paraphrase is free to swap. Without these the guard reads
  // "is running on" rewritten as "runs on" as a lost assertion, which sends a
  // perfectly good merge down the fallback path — and, in the auditor, reports
  // a rewording as a victim.
  'run', 'running', 'ran', 'name', 'named', 'call', 'called', 'use', 'using', 'used',
  'equipped', 'having', 'help', 'helping', 'include', 'including', 'contain', 'containing',
  'consist', 'described', 'considered', 'generally', 'usually', 'currently', 'actually',
  'probably', 'possibly', 'become', 'becomes', 'became', 'remain', 'remains', 'made', 'make'
]);

/**
 * Normalise one word for comparison: lowercase, intra-word punctuation removed,
 * a trailing plural dropped. "Qwen3.8" and "qwen3.8," both become "qwen38";
 * "RTX3090s" and "RTX3090" both become "rtx3090".
 *
 * THE NEEDLE AND THE HAYSTACK MUST BE NORMALISED THE SAME WAY. They were not in
 * the first cut of this guard, which reported "qwen3.8" as dropped from text
 * that plainly contained it, and sent every real merge down the fallback path.
 */
function normaliseWord(raw) {
  let t = String(raw).toLowerCase().replace(/[._\-\/+]+$/, '');
  if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) t = t.slice(0, -1);
  return t.replace(/[^a-z0-9]/g, '');
}

const WORD_RE = /[A-Za-z0-9][A-Za-z0-9._\-\/+]*/g;

/** Every word in a text, normalised — what a needle is checked against. */
function allWords(text) {
  const out = new Set();
  for (const raw of String(text || '').match(WORD_RE) || []) {
    const n = normaliseWord(raw);
    if (n.length >= 2) out.add(n);
  }
  return out;
}

/**
 * Every word that carries meaning, distinctive or not — the vocabulary two
 * phrases must overlap on to be talking about the same thing. allWords() keeps
 * the ordinary words contentTokens() throws away, and the ordinary words are
 * precisely what names an attribute.
 */
function attributeWords(text) {
  const out = new Set();
  for (const w of allWords(text)) if (!STOPWORDS.has(w)) out.add(w);
  return out;
}

/**
 * The tokens a merge is GUARDED on: the ones whose disappearance would mean a
 * real assertion went missing.
 *
 * Deliberately not "every word". A merger writes English, and English has more
 * than one word for a thing — "a box called AIServer" and "a machine called
 * AIServer" assert the same fact, and a guard that called `box` a lost detail
 * would refuse every genuine merge and leave the corpus exactly as lossy as it
 * was. So the guard watches what a paraphrase cannot legitimately replace:
 *
 *   - anything containing a digit  — 9950x, 64GB, DDR5, 2026, Qwen3.8, 22.04
 *   - anything capitalised mid-sentence — Juno, AIServer, MettaSphere, ThinkPad
 *   - any word of six characters or more — MettaSphere's role, "hardware"
 *
 * Short common words fall outside it. That is the deliberate soft edge, and it
 * is why the loser is ALWAYS kept linked as history regardless of what the
 * guard concluded: the guard decides whether the union is trustworthy, never
 * whether the original survives.
 */
function contentTokens(text) {
  const s = String(text || '');
  const out = new Set();
  let m;
  WORD_RE.lastIndex = 0;
  while ((m = WORD_RE.exec(s)) !== null) {
    const raw = m[0];
    const n = normaliseWord(raw);
    if (n.length < 2) continue;
    if (STOPWORDS.has(n)) continue;
    if (/^\d$/.test(n)) continue;
    const hasDigit = /\d/.test(n);
    // Capitalised, but not merely because it opens the sentence.
    const before = s.slice(0, m.index);
    const sentenceStart = m.index === 0 || /(^|[.!?])\s*$/.test(before);
    const capitalised = /^[A-Z]/.test(raw) && !sentenceStart;
    if (!(hasDigit || capitalised || n.length >= 6)) continue;
    out.add(n);
  }
  return out;
}

/** Is a normalised token traceable in a text? */
function covers(words, needle) {
  if (words.has(needle)) return true;
  for (const w of words) {
    if (needle.length >= 3 && w.includes(needle)) return true;
    if (w.length >= 4 && needle.includes(w)) return true;
  }
  return false;
}

/** Which of `needed` cannot be found anywhere in `haystack`. */
function missingTokens(needed, haystack) {
  const words = allWords(haystack);
  const missing = [];
  for (const t of needed) if (!covers(words, t)) missing.push(t);
  return missing;
}

const SYSTEM_UNION = `You are merging two stored facts about the same subject into ONE fact that loses nothing.

Both sentences are already held as true. Your job is to write a single replacement that carries EVERY distinct assertion from BOTH of them.

Rules:
- Carry over every detail from both sentences. Numbers, model names, hardware specs, roles, relationships, dates, place names — all of it.
- Never invent anything that is not in one of the two sentences.
- Never drop a detail because it seems minor or because the sentence gets long. Length is not a cost here; a lost detail is.
- Keep the grammatical form of the SURVIVING sentence: same subject, same third person, same opening words.
- One fact, written as one or two sentences. No bullet lists.

Answer with exactly two lines and nothing else:
MERGED: <the merged fact, and nothing but the fact>
DROPPED: NONE`;

const SYSTEM_CONTRADICTION = `You are merging an OLD stored fact and a NEW fact that corrects it, into ONE fact.

WORK LIKE THIS. Start from the NEW fact, word for word. Then ADD to it every detail the OLD fact asserts that the NEW fact does not DIRECTLY contradict. That is the whole job.

A detail is only contradicted when the NEW fact makes a competing claim about that same thing. If the NEW fact says nothing at all about a detail, it does not contradict it, and that detail MUST be carried over.

Worked example:
  OLD: "User's sister Juno runs on a box called AIServer with a Ryzen 9950x and 2 RTX3090s, and helps User with MettaSphere."
  NEW: "User's sister Juno runs on a box called Thunderbox, not AIServer."
  The NEW fact competes with "AIServer" and nothing else. It says nothing about the CPU, the GPUs or MettaSphere, so all three are carried over.
  MERGED: User's sister Juno runs on a box called Thunderbox, with a Ryzen 9950x and 2 RTX3090s, and helps User with MettaSphere.
  DROPPED: AIServer ||| runs on a box called Thunderbox, not AIServer

Rules:
- Never invent anything that is not in one of the two facts.
- Never drop a detail because it seems minor or because the sentence gets long. Length is not a cost here; a lost detail is.
- Keep the grammatical form of the NEW fact: same subject, same third person.
- One fact, written as one or two sentences. No bullet lists.
- Every dropped detail must be justified by quoting the words IN THE NEW FACT that replace it. If you cannot quote a replacement from the new fact, you may not drop it.
- The words you quote must be ABOUT the detail you are dropping, and must share a word with it: "runs Ubuntu 24.04" may replace "runs Ubuntu 22.04". A different attribute is NOT a contradiction — a machine name does not replace a model name, and a CPU does not replace a role. If the new fact only talks about something else, carry the detail over.

Answer with these lines and nothing else:
MERGED: <the merged fact, and nothing but the fact>
DROPPED: NONE
or, one line per dropped detail:
DROPPED: <the old detail> ||| <the exact words from the NEW fact that replace it>`;

/**
 * Pull the merged sentence and the justified drops out of the reply.
 *
 * DEFENSIVE ABOUT THE LABEL. A 27B model asked for two labelled lines will
 * sometimes run them together, or misspell the second one — the first live run
 * of this returned "…helping User with MettaSphere. DROPP: older, equipped
 * with a Ryzen 9950x…", and a naive `MERGED:(.*)` captured the label and the
 * drop list INTO the fact text and then wrote that to the corpus. So the
 * merged half is cut at any DROPP-ish token wherever it appears, newline or
 * not, and a fact that still carries a label is refused outright.
 */
function parseMerged(content) {
  const text = String(content || '').replace(/\r/g, '');
  const mergedMatch = text.match(/MERGED\s*:\s*([\s\S]*?)(?:\n\s*DROPP\w*\s*:|\s+DROPP\w*\s*:|$)/i);
  let merged = mergedMatch ? mergedMatch[1].trim() : '';
  merged = merged.replace(/^["']|["']$/g, '').replace(/\s+/g, ' ').trim();

  const drops = [];
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*DROPP\w*\s*:\s*(.*)$/i);
    if (!m) continue;
    const body = m[1].trim();
    if (!body || /^none\b/i.test(body)) continue;
    const parts = body.split('|||');
    drops.push({
      detail: parts[0].trim().replace(/^["']|["']$/g, ''),
      replacedBy: parts.length > 1 ? parts.slice(1).join('|||').trim().replace(/^["']|["']$/g, '') : ''
    });
  }
  return { merged, drops };
}

/**
 * Compute the union text for a merge, or explain why there isn't one.
 *
 * ONE RETRY. The first answer is checked mechanically, and when it fails the
 * merger is told exactly which assertions went missing and asked again. A model
 * that dropped a clause because the sentence was getting long puts it back when
 * it is named; one that cannot is not trusted a third time, and the caller falls
 * back to a plain supersede.
 *
 * @param {string} loserText - the fact about to be retired
 * @param {string} survivorText - the fact that stays active
 * @param {Object} [opts]
 * @param {'union'|'contradiction'} [opts.mode='union']
 * @returns {Promise<{ok: boolean, text?: string, dropped?: string[], reason?: string, attempts?: number}>}
 */
async function unionText(loserText, survivorText, opts = {}) {
  const mode = opts.mode === 'contradiction' ? 'contradiction' : 'union';
  const loser = String(loserText || '').trim();
  const survivor = String(survivorText || '').trim();
  if (!loser || !survivor) return { ok: false, reason: 'one of the two facts has no text' };

  // Nothing to carry: everything the loser says is already in the survivor's
  // words. Free, and it skips a model call on the commonest merge there is —
  // a subset folding into its superset.
  const carry = missingTokens(contentTokens(loser), survivor);
  if (carry.length === 0) {
    return { ok: true, text: survivor, dropped: [], reason: 'survivor already carries every assertion the loser made' };
  }

  const survivorWords = allWords(survivor);
  let feedback = null;
  let lastReason = 'the merger produced nothing usable';

  for (let attempt = 1; attempt <= 2; attempt++) {
    let content;
    try {
      // No thinkingTokens override: this is plumbing and takes the background
      // budget (generation.backgroundThinkingTokens), which is 0 — thinking off.
      const res = await memoryManager().callLLM(
        mode === 'contradiction' ? SYSTEM_CONTRADICTION : SYSTEM_UNION,
        (mode === 'contradiction'
          ? `OLD fact: "${loser}"\nNEW fact: "${survivor}"\n\nMerge them.`
          : `Sentence A (being folded away): "${loser}"\nSentence B (surviving): "${survivor}"\n\nMerge them.`) +
        (feedback ? `\n\n${feedback}` : ''),
        { maxTokens: 500 }
      );
      content = res.content;
    } catch (err) {
      return { ok: false, reason: `merger call failed: ${err.message}`, attempts: attempt };
    }

    const { merged, drops } = parseMerged(content);
    const verdict = validateUnion({ merged, drops, loser, survivor, survivorWords, carry, mode });
    if (verdict.ok) {
      return { ok: true, text: merged, dropped: drops.map(d => d.detail), attempts: attempt };
    }
    lastReason = verdict.reason;
    feedback = verdict.feedback;
  }

  return { ok: false, reason: lastReason, attempts: 2 };
}

/**
 * THE MECHANICAL CHECK — where "I lost nothing" is tested rather than trusted.
 *
 * @returns {{ok: true} | {ok: false, reason: string, feedback: string}}
 */
function validateUnion({ merged, drops, loser, survivor, survivorWords, carry, mode }) {
  const no = (reason, feedback) => ({ ok: false, reason, feedback });

  if (!merged || merged.length < Math.min(survivor.length, 12)) {
    return no('merger produced no usable sentence',
      'Your last answer had no usable MERGED line. Answer again, with MERGED: followed by the merged fact.');
  }
  // A label that leaked into the sentence is not a fact, and writing it to the
  // corpus is worse than not merging at all.
  if (/\bDROPP\w*\s*:/i.test(merged) || /\bMERGED\s*:/i.test(merged)) {
    return no('the merged sentence carried a label into the fact text',
      'Your MERGED line contained a label such as "DROPPED:". The MERGED line must contain the fact and nothing else. Put dropped details on their own DROPPED: lines.');
  }

  const stillMissing = missingTokens(new Set(carry), merged);

  if (mode !== 'contradiction') {
    if (stillMissing.length) {
      return no(`merger dropped ${stillMissing.map(t => `"${t}"`).join(', ')} and nothing may be dropped in a duplicate merge`,
        `Your merged sentence lost these details: ${stillMissing.join(', ')}. Both sentences are true, so nothing may be dropped. Answer again, keeping every one of them.`);
    }
    const survivorLost = missingTokens(contentTokens(survivor), merged);
    if (survivorLost.length) {
      return no(`merger dropped ${survivorLost.map(t => `"${t}"`).join(', ')} from the surviving fact`,
        `Your merged sentence lost these details from sentence B: ${survivorLost.join(', ')}. Answer again, keeping every one of them.`);
    }
    return { ok: true };
  }

  // CONTRADICTION MODE. A detail may go — but only one the NEW fact actually
  // competes with, and the merger has to show its work: the words from the new
  // fact that replace it. Without that check "DROPPED:" is a blank cheque, and
  // the first live run wrote one ("older, equipped with a Ryzen 9950x and 2
  // RTX3090s" declared dropped by a correction that only renamed the box).
  const justified = new Set();
  const misaimed = [];
  for (const d of drops) {
    if (!d.replacedBy) continue;
    // The replacement must be quoted FROM THE NEW FACT. If those words are not
    // in it, the merger invented a contradiction to license a deletion.
    const quoteTokens = contentTokens(d.replacedBy);
    const quoteMissing = missingTokens(quoteTokens, survivor);
    if (quoteMissing.length) continue;
    if (quoteTokens.size === 0) continue;

    // AND IT MUST NAME WHAT IT REPLACES (2026-08-24, second pass). Quoting the
    // new fact proves the words are real; it does not prove they are ABOUT the
    // detail being deleted, and that gap is a blank cheque with extra steps. It
    // was cashed on Athena's own pair: the merger declared "also running on the
    // Qwen3.8 27b model" contradicted, justified it by quoting "runs on a
    // machine called AIServer", and the guard took it — a model name deleted by
    // a claim about a machine name. To contradict a claim you have to be
    // talking about it, so the replacement has to share a word with it:
    // "Ubuntu 24.04" replacing "Ubuntu 22.04" does, "AIServer" replacing
    // "Qwen3.8" does not.
    // Compared on ATTRIBUTE words, not on contentTokens. contentTokens keeps
    // only the distinctive half of a sentence — things with a digit, a
    // mid-sentence capital, or six letters — and a genuine same-attribute
    // correction differs on exactly those: "Qwen3.8 27b" vs "Qwen3.9 30b" share
    // nothing distinctive. The word that proves they are about the same thing is
    // the ordinary one next to them ("model", "box", "machine"), which is what
    // this compares.
    const detailTokens = contentTokens(d.detail);
    let namesIt = false;
    for (const w of attributeWords(d.detail)) if (attributeWords(d.replacedBy).has(w)) { namesIt = true; break; }
    if (!namesIt) { misaimed.push(d); continue; }

    for (const t of detailTokens) justified.add(t);
  }

  const unjustified = stillMissing.filter(t => !justified.has(t));
  if (unjustified.length) {
    const declaredAnyway = drops.length
      ? ` You listed: ${drops.map(d => d.detail).join('; ')}.`
      : '';
    const aimed = misaimed.length
      ? ` A replacement must be ABOUT the detail it replaces and share a word with it — ` +
        misaimed.map(d => `you offered "${d.replacedBy}" for "${d.detail}", which is a claim about something else`).join('; ') + '.'
      : '';
    return no(`merger dropped ${unjustified.map(t => `"${t}"`).join(', ')} without showing what in the new fact replaces them`,
      `Your merged sentence lost these details from the OLD fact: ${unjustified.join(', ')}. The NEW fact says nothing that competes with them, so they are NOT contradicted and must be carried over.${declaredAnyway}${aimed} Answer again, starting from the NEW fact word for word and adding every one of those details back.`);
  }

  return { ok: true };
}

/**
 * Merge two facts and retire the loser, losing nothing.
 *
 * The survivor is reworded to the union FIRST, then the loser is superseded and
 * linked to it. Both writes go through fact-store, so both are ledgered and the
 * loser stays recoverable.
 *
 * A union that cannot be computed, cannot be verified, or is refused by the
 * identity lock DEFERS THE RETIREMENT ALTOGETHER (2026-08-24, second pass). The
 * first cut fell back to a plain supersede and argued that the loser was still
 * linked history, so nothing was lost permanently. That is true of the ledger
 * and false of the corpus: renderLongTermMemory reads active rows only, so a
 * refused union retired assertions out of everything the model can see — which
 * is Athena's loss happening again, just more rarely. Measured on her exact
 * pair, the merger drops a clause and the guard catches it about one time in
 * six, so "rarely" was not rare enough to call it fixed.
 *
 * So when there was something to carry and it could not be carried, this returns
 * {ok: false, deferred: true} and writes NOTHING. Both rows stay active. The
 * cost is one visible duplicate until a later pass merges them successfully —
 * and because both are still active, a later pass DOES see them again, so it
 * self-heals rather than accumulating. That is the trade this module exists to
 * make: a duplicate is untidy, and a silent deletion is damage.
 *
 * @param {string} loserId
 * @param {string} survivorId
 * @param {Object} [opts]
 * @param {'union'|'contradiction'} [opts.mode='union']
 * @param {boolean} [opts.carryOver=true] - false skips the union entirely (the
 *   compound SPLIT path: its atoms already hold everything the original said,
 *   so folding the original back into the first atom would un-split it).
 * @param {string} [opts.ledgerTier] - tier for the union's own reword entry
 *   ('mechanical' for the corrector's duplicate merge, 'semantic' for its
 *   contradiction pass, 'intake' — the default — for the live write paths)
 * @param {Object} [opts.supersedeOpts] - passed straight to factStore.supersede
 * @returns {Promise<Object>} the supersede result, plus a `union` block
 */
async function mergePreservingUnion(loserId, survivorId, opts = {}) {
  const mode = opts.mode === 'contradiction' ? 'contradiction' : 'union';
  const supersedeOpts = opts.supersedeOpts || {};

  const loser = factStore.getMember(loserId);
  const survivor = factStore.getMember(survivorId);
  if (!loser) return { ok: false, reason: `no fact with id ${loserId}`, union: null };
  if (!survivor) return { ok: false, reason: `no fact with id ${survivorId}`, union: null };

  let union = { applied: false, mode, from: survivor.content, to: null, dropped: [], skipped: null };
  // Set only when the loser held something the survivor does not and the carry
  // failed. That is the one case where retiring it would delete an assertion,
  // and the one case this refuses to retire.
  let deferred = null;

  if (opts.carryOver === false) {
    union.skipped = 'caller asked for no carry-over (the successor already holds everything)';
  } else {
    const u = await unionText(loser.content, survivor.content, { mode });
    if (!u.ok) {
      union.skipped = u.reason;
      deferred = u.reason;
    } else if (u.text.trim() === survivor.content.trim()) {
      union.skipped = u.reason || 'the surviving fact already said everything';
    } else {
      const rw = await factStore.reword(survivorId, u.text, {
        reason: `Merged in what "${String(loser.content).slice(0, 120)}" asserted, so folding it away loses nothing.`,
        ledger: {
          tier: opts.ledgerTier || (supersedeOpts.ledger && supersedeOpts.ledger.tier) || undefined,
          evidence: { union_merge: true, mode, merged_from: loserId, dropped: u.dropped }
        }
      });
      if (rw.ok) {
        union.applied = true;
        union.to = u.text;
        union.dropped = u.dropped;
        console.log(`[FactMerge] union into ${survivorId.slice(0, 8)}: "${u.text.slice(0, 140)}"` +
                    (u.dropped.length ? ` (dropped as contradicted: ${u.dropped.join(', ')})` : ''));
      } else {
        union.skipped = rw.locked
          ? `the surviving fact is locked (${rw.category}), so it cannot be reworded`
          : `reword failed: ${rw.reason}`;
        deferred = union.skipped;
      }
    }
  }

  // NOTHING IS RETIRED THAT COULD NOT BE CARRIED. Leaving both rows active keeps
  // every assertion where the model can see it, and leaves the pair visible to
  // the next merge attempt.
  if (deferred) {
    console.warn(`[FactMerge] NOT retiring ${loserId.slice(0, 8)} → ${survivorId.slice(0, 8)}: ${deferred} ` +
                 `— both facts stay active so nothing leaves the corpus`);
    return { ok: false, deferred: true, reason: deferred, union };
  }

  // The link, always — union or no union. This is the half that makes a bad
  // merge survivable: the original is inactive, pointed at its successor, and in
  // the ledger, so it can be read and restored.
  const res = await factStore.supersede(loserId, survivorId, Object.assign({}, supersedeOpts, {
    ledger: Object.assign({}, supersedeOpts.ledger, {
      evidence: Object.assign(
        {},
        supersedeOpts.ledger && supersedeOpts.ledger.evidence,
        { union_merge: union.applied, union_mode: mode, union_skipped: union.skipped || null }
      )
    })
  }));

  return Object.assign({}, res, { union });
}

/**
 * Carry a restatement's extra detail into the fact that already holds it, when
 * there IS extra detail.
 *
 * The write-time repeat rule folds a restated fact into the held one and never
 * creates a second row — correct, and not something to relax: duplicates
 * accumulating is its own damage. But "the same assertion" is a model's verdict,
 * and when it is wrong the restatement's extra clause is written nowhere active.
 * The corroboration row keeps the text for an auditor; the corpus does not.
 *
 * So: if the restatement carries nothing the held fact does not, this costs a
 * set comparison and returns. If it carries something, the held fact is
 * rewritten to include it — still one row, now a row that knows both things.
 *
 * @returns {Promise<{applied: boolean, to: string|null, skipped: string|null}>}
 */
async function carryIntoSurvivor(survivorId, otherText, opts = {}) {
  const survivor = factStore.getMember(survivorId);
  if (!survivor) return { applied: false, to: null, skipped: 'no such fact' };
  const extra = missingTokens(contentTokens(otherText), survivor.content);
  if (extra.length === 0) return { applied: false, to: null, skipped: null };  // a true restatement

  const u = await unionText(otherText, survivor.content, { mode: 'union' });
  if (!u.ok) return { applied: false, to: null, skipped: u.reason };
  if (u.text.trim() === survivor.content.trim()) return { applied: false, to: null, skipped: null };

  const rw = await factStore.reword(survivorId, u.text, {
    reason: opts.reason || `A restatement carried detail this fact did not hold (${extra.join(', ')}), so it was merged in rather than folded away.`,
    ledger: { tier: opts.ledgerTier || 'intake', evidence: { union_merge: true, mode: 'repeat', restated_as: otherText } }
  });
  if (!rw.ok) {
    return { applied: false, to: null, skipped: rw.locked ? `the fact is locked (${rw.category})` : `reword failed: ${rw.reason}` };
  }
  console.log(`[FactMerge] repeat carried extra detail into ${survivorId.slice(0, 8)}: "${u.text.slice(0, 140)}"`);
  return { applied: true, to: u.text, skipped: null };
}

module.exports = {
  mergePreservingUnion, unionText, carryIntoSurvivor,
  contentTokens, missingTokens, allWords, normaliseWord,
  // exported for scripts/test-merge-union.js — the guard is the part of this
  // module that has to be testable without a model in the loop
  parseMerged, validateUnion
};
