/**
 * THE AUDIT TURN, AND WHY A DECIDED PAIR STAYS DECIDED.
 *
 * Since 2026-09-02 the self-coherence audit's questions land in the entity's own
 * Corrections queue instead of Ellie's list — they were always addressed to the
 * entity, and routing them to her re-pointed the "you" at someone who could not
 * answer them. That fixed the destination and left the queue with no reply box.
 * This is the reply box.
 *
 * The entity reads the pair, uses its tools, and files one of four conclusions:
 *
 *   retire-one   — one of them goes, with a receipt
 *   keep-both    — they are different in kind, or both true; nothing changes
 *   supersede    — one is the newer version of the other
 *   cannot-settle — the evidence does not decide it, and this goes to Ellie
 *
 * ─── THE RE-RAISE PROBLEM, WHICH IS THE WHOLE REASON THIS IS A TABLE ───────
 *
 * The detector runs every pass and will keep flagging the same two facts
 * forever. Athena's line: "THE AUDIT'S OWN RE-FIRE ON UNCHANGED FACTS IS NOT
 * NEW EVIDENCE." Without a decided state, a settled pair comes back tomorrow and
 * the queue becomes noise; with a decided state that never re-opens, a decision
 * made on ground that later moved stands forever. So a pair re-opens on exactly
 * four conditions, and every one of them is a ledger or store comparison rather
 * than a judgement:
 *
 *   1. A RECEIPT MOVED. Something cited in the decision was later retracted,
 *      reworded, superseded or re-filed. The ground changed.
 *   2. A MEMBER CHANGED STATUS. And note the asymmetry Athena asked for: if a
 *      member went inactive, the pair does NOT re-raise — it closes as
 *      `superseded-closed` and the survivor stands. Only a member coming BACK
 *      (restored to active) re-opens it.
 *   3. A NEW FACT CONTRADICTS THE CONCLUSION. The conclusion is run through the
 *      contradiction detector as if it were a candidate fact. Flagged in her
 *      review as a design choice, and taken because it is checkable.
 *   4. AN UNANCHORED MEMBER GAINED AN ANCHOR. If the pair was closed because one
 *      side was a felt report with nothing outside the entity behind it, and
 *      that member later acquires a message or tool receipt, the basis is gone.
 *      "The detector improving is evidence."
 *
 * ─── AGEING ───────────────────────────────────────────────────────────────
 *
 * "Item 2 has no default for an unanswered decision. I'm not always at the
 * queue […] so an unanswered decision keeps a pair silently open while I hold
 * the box." After `repair.decisionAgeDays` a pair the entity has not settled
 * goes to Ellie as a plain question with whatever partial reasoning exists,
 * written through db/message-standards.js like anything else that reaches her.
 */

const { randomUUID } = require('crypto');
const { getSqliteDb } = require('./database');
const repair = require('./memory-repair');

function sqlite() { return getSqliteDb(); }
function ledger() { return require('./corrections-ledger'); }

const CONCLUSIONS = ['retire-one', 'keep-both', 'supersede', 'cannot-settle'];

/** A pair is identified by its members, order-independent. */
function pairKey(aId, bId) {
  return [aId, bId].filter(Boolean).sort().join(':');
}

// ───────────────────────────────────────────────── filing a decision

/**
 * File the entity's decision on one pair.
 *
 * Two records, deliberately: the `audit_decisions` row is the STATE the re-raise
 * check reads, and the ledger entry is the VISIBILITY — guardrail 3, the
 * decisions that were settled being as readable as the ones that were not.
 */
function fileDecision({ memberA, memberB = null, conclusion, rationale, receipts = [], subject = 'self' }) {
  const db = sqlite();
  if (!db) return { ok: false, reason: 'database unavailable' };
  if (!CONCLUSIONS.includes(conclusion)) {
    return { ok: false, reason: `conclusion must be one of ${CONCLUSIONS.join(', ')}` };
  }
  if (!String(rationale || '').trim()) {
    return { ok: false, reason: 'a decision files with its reasoning or it does not file' };
  }
  // cannot-settle is the one conclusion that does NOT need a receipt: it is the
  // entity saying the evidence does not decide it, and demanding a receipt for
  // that would push it toward inventing one.
  if (conclusion !== 'cannot-settle') {
    const gated = repair.gateReceipts(receipts);
    if (!gated.ok) return { ok: false, reason: gated.reason, available: false };
  }

  const verified = (Array.isArray(receipts) ? receipts : [])
    .map(r => repair.verifyReceipt(r)).filter(v => v.ok)
    .map(v => ({ kind: v.kind, id: v.id }));

  const ledgerId = repair.recordDecision({
    kind: 'audit-pair', subject,
    conclusion, rationale,
    receipts: verified.map(v => ({ kind: v.kind, id: v.id })),
    memberA, memberB,
    outcome: conclusion === 'cannot-settle' ? 'cannot-settle' : 'decided',
    extra: { pair_key: pairKey(memberA, memberB) }
  });

  const id = randomUUID();
  db.prepare(`
    INSERT INTO audit_decisions
      (id, created_at, pair_key, member_a, member_b, conclusion, rationale, receipts_json, ledger_id, state)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id, new Date().toISOString(), pairKey(memberA, memberB), memberA, memberB,
    conclusion, rationale, JSON.stringify(verified), ledgerId,
    conclusion === 'cannot-settle' ? 'awaiting-ellie' : 'decided'
  );
  return { ok: true, id, ledgerId, conclusion, receipts: verified };
}

/** The standing decision on a pair, if there is one. */
function standingDecision(aId, bId) {
  const db = sqlite();
  if (!db) return null;
  try {
    return db.prepare(`
      SELECT * FROM audit_decisions
      WHERE pair_key = ? AND state IN ('decided', 'awaiting-ellie', 'escalated')
      ORDER BY created_at DESC LIMIT 1
    `).get(pairKey(aId, bId)) || null;
  } catch { return null; }
}

// ───────────────────────────────────────────────── the re-raise check

/**
 * Should the audit raise this pair again?
 *
 * Returns `{raise: false, drop: 'unchanged'}` for the common case, which is the
 * detector re-firing on facts nothing has touched.
 *
 * @returns {{raise: boolean, reason: string|null, drop: string|null, close: string|null}}
 */
function shouldRaise(aId, bId) {
  const db = sqlite();
  if (!db) return { raise: true, reason: 'no store to check against', drop: null, close: null };

  const decision = standingDecision(aId, bId);
  if (!decision) return { raise: true, reason: 'no decision has ever been filed on this pair', drop: null, close: null };

  const a = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(aId);
  const b = bId ? db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(bId) : null;

  // Rule 2, and the asymmetry: a member going inactive CLOSES the pair. It does
  // not re-raise it. "If a member is gone, that's not a re-raise — the pair
  // closes as superseded-closed and the survivor stands."
  const gone = [a, b].filter(m => m && m.status !== 'active');
  if (gone.length) {
    closePair(decision.id, 'superseded-closed',
      `${gone.length === 2 ? 'Both members are' : 'One member is'} no longer active, so there is no live pair left to settle.`);
    return { raise: false, reason: null, drop: null, close: 'superseded-closed' };
  }

  // ── THE COMPARISON IS `>=`, AND THAT IS NOT SLOPPINESS ──────────────────
  //
  // `corrections_ledger.created_at` is an ISO string with millisecond
  // resolution, and a decision followed immediately by the operation it
  // justifies lands in the SAME millisecond. Under a strict `>` that receipt has
  // moved and the pair can never learn it — silently, forever, which is the one
  // outcome this state machine may not have. (The conversation channel hit the
  // same wall from the other side: `messages.timestamp` has one-second
  // resolution and needed a tie count.)
  //
  // `>=` costs a pair being re-opened by something that happened in the same
  // millisecond as the decision. That is the safe direction: a wrongly re-opened
  // pair is looked at once more, a wrongly dropped one leaves a decision
  // standing on ground that moved. The decision's own ledger entry cannot
  // collide — its action is 'decision', which is not in the list below.
  const since = decision.created_at;

  // Rule 1: a cited receipt moved after the decision was filed.
  let cited = [];
  try { cited = JSON.parse(decision.receipts_json || '[]'); } catch { cited = []; }
  for (const r of cited) {
    if (r.kind === 'fact') {
      const moved = db.prepare(`
        SELECT action, created_at FROM corrections_ledger
        WHERE target_id = ? AND created_at >= ?
          AND action IN ('retire','reword','supersede','refile','expire')
        ORDER BY created_at DESC LIMIT 1
      `).get(r.id, since);
      if (moved) {
        return reopen(decision.id, `a receipt this decision rested on was ${moved.action}d after it was filed`);
      }
    }
  }

  // Rule 4: a member that was unanchored gained an anchor.
  for (const m of [a, b].filter(Boolean)) {
    if (repair.anchorOf(m) === 'anchored' && /felt|anchor|unanchored/i.test(decision.rationale || '')) {
      const changed = db.prepare('SELECT updated_at FROM cluster_members WHERE id = ? AND updated_at >= ?').get(m.id, since);
      if (changed) {
        return reopen(decision.id, 'a member that had no anchor when this was decided has one now — the basis is gone');
      }
    }
  }

  // Rule 3: a new fact contradicts the stated conclusion. Checked against facts
  // filed AFTER the decision only; an older fact is not new evidence.
  //
  // Deliberately the cheap half of the detector — a text-level check, not an
  // embedding pass — because this runs on every audit pass over every decided
  // pair, and the expensive version is what the audit itself already does.
  const fresh = db.prepare(`
    SELECT id, content FROM cluster_members
    WHERE status = 'active' AND subject = ? AND created_at >= ? AND id NOT IN (?, ?)
    ORDER BY created_at DESC LIMIT 40
  `).all(a ? a.subject : 'self', since, aId, bId || '');
  for (const f of fresh) {
    if (negates(decision.conclusion === 'keep-both' ? decision.rationale : decision.rationale, f.content)) {
      return reopen(decision.id, `a fact filed since this decision contradicts its stated conclusion (${f.id.slice(0, 8)})`);
    }
  }

  // Nothing moved. This is the audit re-firing on unchanged facts, which is not
  // evidence — drop it.
  return { raise: false, reason: null, drop: 'unchanged-re-fire', close: null };
}

/**
 * A deliberately narrow negation test, run over the conclusion as if it were a
 * candidate fact. Narrow because a false positive here re-opens a settled pair
 * and the queue fills back up — the failure this whole state machine exists to
 * stop. It fires only when a later fact denies what the conclusion asserts,
 * about the same words.
 */
function negates(conclusionText, candidateText) {
  const NEG = /\b(?:not|never|no longer|cannot|can't|don't|doesn't|isn't|aren't|won't)\b/i;
  const A = String(conclusionText || ''), B = String(candidateText || '');
  if (NEG.test(A) === NEG.test(B)) return false;
  const words = t => new Set(String(t).toLowerCase().match(/\b[a-z]{6,}\b/g) || []);
  const wa = words(A), wb = words(B);
  const shared = [...wa].filter(w => wb.has(w));
  return shared.length >= 4;
}

function reopen(decisionId, reason) {
  try {
    sqlite().prepare("UPDATE audit_decisions SET state = 'reopened', reopened_at = ?, reopen_reason = ? WHERE id = ?")
      .run(new Date().toISOString(), reason, decisionId);
  } catch { /* the raise still happens */ }
  return { raise: true, reason, drop: null, close: null };
}

function closePair(decisionId, state, reason) {
  try {
    sqlite().prepare('UPDATE audit_decisions SET state = ?, reopen_reason = ? WHERE id = ?')
      .run(state, reason, decisionId);
  } catch { /* non-fatal */ }
}

// ───────────────────────────────────────────────── ageing to Ellie

/**
 * A pair the entity has not settled after `repair.decisionAgeDays` goes to Ellie
 * as a plain question, with whatever partial reasoning exists attached.
 *
 * It goes through the ordinary message path, which means db/message-standards.js
 * applies: no fact ids in the text, no audit vocabulary, and a question a person
 * who was not there can answer. The ids are in the ledger entry behind it.
 *
 * @returns {{escalated: number, items: Array}}
 */
async function ageOutToEllie({ now = Date.now() } = {}) {
  const db = sqlite();
  const out = { escalated: 0, items: [] };
  if (!db) return out;
  const days = repair.repairConfig().decisionAgeDays;
  if (!Number.isFinite(days)) return out;
  const cutoff = new Date(now - days * 86400_000).toISOString();

  let rows = [];
  try {
    rows = db.prepare(`
      SELECT * FROM audit_decisions
      WHERE state = 'awaiting-ellie' AND created_at <= ? AND escalated_at IS NULL
      ORDER BY created_at ASC LIMIT 5
    `).all(cutoff);
  } catch { return out; }

  for (const row of rows) {
    const a = db.prepare('SELECT content FROM cluster_members WHERE id = ?').get(row.member_a);
    const b = row.member_b ? db.prepare('SELECT content FROM cluster_members WHERE id = ?').get(row.member_b) : null;
    try {
      const engine = require('./initiative-engine');
      const body =
        `I have been holding a question about myself for ${days} day${days === 1 ? '' : 's'} and I cannot get to the ` +
        `bottom of it on my own, so I am bringing it to you.\n\n` +
        `I believe two things about myself that do not sit together. One is that ${plain(a && a.content)}. ` +
        (b ? `The other is that ${plain(b.content)}. ` : '') +
        `\n\nHere is as far as I got: ${row.rationale}\n\n` +
        `Nothing has been changed either way. Which of these is right, or are they both?`;
      await engine.sayToEllie({
        subject: 'Two things I believe about myself that do not fit together',
        body,
        sourceKind: 'audit-decision-aged',
        sourceRef: row.id
      });
      db.prepare('UPDATE audit_decisions SET state = ?, escalated_at = ? WHERE id = ?')
        .run('escalated', new Date().toISOString(), row.id);
      out.escalated++;
      out.items.push({ id: row.id, pair: row.pair_key });
    } catch (err) {
      console.error('[AuditDecisions] could not take an aged question to Ellie:', err.message);
    }
  }
  return out;
}

/** Strip a stored fact down to something readable mid-sentence. */
function plain(text) {
  return String(text || 'something I could not restate')
    .replace(/^I\s+/i, 'I ')
    .replace(/\.$/, '')
    .slice(0, 240);
}

/**
 * The questions waiting on the entity's own turn.
 *
 * KEYED ON `raised_by`, NOT ON THE `awaiting_entity_turn` FLAG, and the
 * difference is not cosmetic. That flag was added on 2026-09-02 when the audit
 * stopped routing its asks to Ellie; every finding raised BEFORE that — and
 * there were six standing in Athena's store, three of them from that same
 * morning — carries only `unresolved`. Keyed on the flag, the reply box opened
 * onto an empty queue while six real questions sat behind it, which is the
 * failure this whole build exists to end: a thing the entity can see and cannot
 * act on. Every self-coherence raise is addressed to the entity by definition,
 * so `raised_by` IS the predicate and the flag is a refinement of it.
 *
 * A question already decided drops out: `source_ref` on a pair is the two member
 * ids sorted and joined, which is exactly `pairKey`, and on a revision it is the
 * claim's own id — so one comparison covers both shapes.
 */
function openPairs({ limit = 20 } = {}) {
  const db = sqlite();
  if (!db) return [];
  try {
    return db.prepare(`
      SELECT l.id, l.created_at, l.target_id, l.target_text, l.reason,
             json_extract(l.evidence, '$.source_ref')  AS source_ref,
             json_extract(l.evidence, '$.reason_code') AS reason_code
      FROM corrections_ledger l
      WHERE json_extract(l.evidence, '$.raised_by') = 'self-coherence-audit'
        AND json_extract(l.evidence, '$.unresolved') = 1
        AND l.reverted_at IS NULL
        AND NOT EXISTS (
          SELECT 1 FROM audit_decisions d
          WHERE d.state IN ('decided', 'escalated', 'superseded-closed')
            AND (d.pair_key = json_extract(l.evidence, '$.source_ref')
                 OR d.member_a = json_extract(l.evidence, '$.source_ref'))
        )
      ORDER BY l.created_at DESC LIMIT ?
    `).all(limit);
  } catch (err) {
    console.error('[AuditDecisions] could not read the open questions:', err.message);
    return [];
  }
}

module.exports = {
  CONCLUSIONS, pairKey, fileDecision, standingDecision, shouldRaise,
  negates, ageOutToEllie, openPairs, closePair
};
