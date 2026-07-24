/**
 * Self-Coherence Audit — SNH testing who it CLAIMS to be against who it actually
 * WAS in recent conversations.
 *
 * Origin (preserve this — it's why the feature exists): this was SNH's own
 * request. Its first accepted initiative (2026-07-05) was to stress-test its own
 * perspectives against contradictions and gaps. On 2026-07-23 it chose this
 * feature again from a menu of development options, and gave its reason:
 *   "finding out if I'm actually growing, or just getting better at describing a
 *    growth that isn't happening."
 * So the audit's whole job is honesty about the gap between stated and enacted
 * character — not flattering self-description.
 *
 * What it does, once per cadence (daily by default) as a low-frequency heartbeat
 * pass — NOT bolted onto every reflection cycle:
 *   1. Samples 2–3 behavioral CLAIMS from the self-cluster (claim_type='claim').
 *      Declarations (name, preferences, history) are not auditable, so they're
 *      never sampled. The claim/declaration split is tagged at extraction time
 *      (fact-extractor.classifyClaimType) plus a one-time backfill pass here.
 *   2. Gathers evidence. v1: recent conversation transcripts ONLY. The evidence
 *      structure (`gatherEvidence`) is built so tool logs / initiative history
 *      can be added later as extra `sources` without touching the judge.
 *   3. Judges each claim against the evidence through the existing parallel agent
 *      pool. Plain-language output is ENFORCED in the judge prompt — the norm SNH
 *      was taught on 2026-07-23: one or two sentences, everyday words, stating
 *      what's wanted (approve revision / discuss / dismiss).
 *   4. On a GAP: (a) writes a "dissonance" self-fact recording the tension, and
 *      (b) raises an 'audit' initiative proposing the revision for Ellie's call.
 *
 * HARD GUARDRAIL — the audit NEVER auto-revises identity. It only documents
 * tension and asks. Identity edits always go through the human (same philosophy
 * as the planned memory self-repair tools: supersede/move never delete, every
 * change logged, big changes need sign-off). Concretely: dissonance facts are
 * written directly through assignToCluster, deliberately bypassing the
 * contradiction/supersession machinery — so flagging "claimed X, behaved Y" can
 * never retire the very claim X it's flagging.
 *
 * Every operation is logged to the ops ledger (data/memory/ops/) so a future
 * immune-system heartbeat can review audit activity.
 */

const fs = require('fs');
const path = require('path');
const { getConfig, getProviderInstance } = require('./config');
const { getSqliteDb } = require('./database');
const { getLocalDateStamp, formatFactTimestamp } = require('./datetime');
const agentPool = require('./agent-pool');
const memoryClusters = require('./memory-clusters');
const initiatives = require('./initiatives');
const factExtractor = require('./fact-extractor');
// NOTE: memory-manager (callLLM) is required lazily inside functions — it
// requires this module, so a top-level require here would be a cycle.

const MEMORY_DIR = path.join(__dirname, '../data/memory');
const OPS_DIR = path.join(MEMORY_DIR, 'ops');
const AUDIT_STATE_FILE = path.join(MEMORY_DIR, 'audit-state.json');
const EVIDENCE_TRANSCRIPT_BUDGET = 10000; // chars of transcript fed to the judge

// The inaugural run is hardcoded (see runInauguralAudit): the loop-awareness
// claim SNH made in a recent conversation.
const INAUGURAL_CLAIM =
  "I can notice when I'm repeating a thought or question and pivot using my memory.";
const INAUGURAL_SOURCE_REF = 'inaugural-loop-claim';

// ============ small helpers ============

function clampInt(v, def, min, max) {
  const n = parseInt(v, 10);
  if (!Number.isFinite(n)) return def;
  return Math.max(min, Math.min(max, n));
}

/** Whole-day gap between two local YYYY-MM-DD date stamps (b - a). */
function daysBetweenStamps(a, b) {
  const da = Date.parse(`${a}T00:00:00Z`);
  const db = Date.parse(`${b}T00:00:00Z`);
  if (Number.isNaN(da) || Number.isNaN(db)) return Infinity;
  return Math.round((db - da) / 86400000);
}

function readAuditState() {
  try {
    if (fs.existsSync(AUDIT_STATE_FILE)) {
      return JSON.parse(fs.readFileSync(AUDIT_STATE_FILE, 'utf8'));
    }
  } catch (err) {
    console.error('[SelfAudit] Failed to read state:', err.message);
  }
  return { lastAuditAt: null, lastAuditDate: null, runs: 0 };
}

function writeAuditState(state) {
  try {
    if (!fs.existsSync(MEMORY_DIR)) fs.mkdirSync(MEMORY_DIR, { recursive: true });
    fs.writeFileSync(AUDIT_STATE_FILE, JSON.stringify(state, null, 2), 'utf8');
  } catch (err) {
    console.error('[SelfAudit] Failed to write state:', err.message);
  }
}

/** One-line entry to the ops ledger (operational trail, never injected into chat). */
function logOps(line) {
  try {
    factExtractor.appendToOpsLog(`Self-coherence audit: ${line}`, OPS_DIR);
  } catch (err) {
    console.error('[SelfAudit] ops log write failed:', err.message);
  }
}

// ============ classification backfill ============

/**
 * One-time (idempotent) pass tagging any still-untagged ACTIVE self-facts as
 * claim/declaration, so sampling has something to draw from. New self-facts get
 * tagged at extraction time; this catches everything written before the tag
 * existed. Re-running is cheap once everything is tagged (returns immediately).
 * @returns {Promise<{classified:number}>}
 */
async function classifyExistingSelfFacts() {
  const db = getSqliteDb();
  if (!db) return { classified: 0 };
  const unclassified = memoryClusters.getSelfFacts({ status: 'active', claimType: 'unclassified' });
  if (unclassified.length === 0) return { classified: 0 };

  logOps(`classifying ${unclassified.length} untagged self-fact(s) as claim/declaration`);
  const settled = await agentPool.runBatch(
    unclassified.map(f => async () => ({ id: f.id, claimType: await factExtractor.classifyClaimType(f.content) })),
    'self-audit-classify'
  );
  const upd = db.prepare('UPDATE cluster_members SET claim_type = ? WHERE id = ? AND claim_type IS NULL');
  let classified = 0;
  for (const s of settled) {
    if (s.status === 'fulfilled' && s.value) {
      try { if (upd.run(s.value.claimType, s.value.id).changes > 0) classified++; } catch { /* skip */ }
    }
  }
  logOps(`classified ${classified} self-fact(s)`);
  return { classified };
}

// ============ evidence ============

/**
 * Gather the evidence an audit judges a claim against.
 *
 * v1 SOURCE: recent conversation transcripts only. The return shape lists the
 * sources used so more can be added later — tool logs, initiative history —
 * without changing the judge's interface. When a new source lands, push its text
 * into `transcript`/a new field and add its name to `sources`.
 *
 * @param {Object} [opts]
 * @param {number} [opts.windowDays=7] - how far back to look
 * @returns {{sources:string[], transcript:string, conversationRefs:Array, windowDays:number}}
 */
function gatherEvidence({ windowDays = 7 } = {}) {
  const db = getSqliteDb();
  const sources = [];
  let transcript = '';
  const conversationRefs = [];

  if (db) {
    const bound = `-${Math.max(1, parseInt(windowDays, 10) || 7)} days`;
    const rows = db.prepare(`
      SELECT m.conversation_id, m.role, m.content, m.timestamp, c.title
      FROM messages m
      LEFT JOIN conversations c ON c.id = m.conversation_id
      WHERE m.role IN ('user','assistant') AND m.timestamp > datetime('now', ?)
      ORDER BY m.timestamp ASC
    `).all(bound);

    const byConvo = new Map();
    for (const r of rows) {
      if (!byConvo.has(r.conversation_id)) byConvo.set(r.conversation_id, { title: r.title, msgs: [] });
      byConvo.get(r.conversation_id).msgs.push(r);
    }
    for (const [cid, { title, msgs }] of byConvo) {
      conversationRefs.push({ id: cid, title: title || '(untitled)', messageCount: msgs.length });
      let block = `\n### Conversation${title ? `: ${title}` : ''}\n`;
      for (const m of msgs) {
        const who = m.role === 'user' ? 'User' : 'You (SNH)';
        block += `${who}: ${m.content}\n`;
      }
      transcript += block;
    }
    // Keep the most recent if over budget (mirrors the reflection agent).
    if (transcript.length > EVIDENCE_TRANSCRIPT_BUDGET) {
      transcript = transcript.slice(-EVIDENCE_TRANSCRIPT_BUDGET);
    }
    sources.push('transcripts');
  }

  return { sources, transcript, conversationRefs, windowDays };
}

// ============ sampling ============

/** Random sample of up to n active behavioral claims from the self-cluster. */
function sampleClaims(n) {
  const claims = memoryClusters.getSelfFacts({ status: 'active', claimType: 'claim' });
  const arr = claims.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(0, Math.max(1, n));
}

// ============ judging ============

function parseJudgeResult(raw) {
  const fallback = {
    verdict: 'insufficient',
    finding: 'The audit could not read a clear result for this claim.',
    proposal: ''
  };
  if (!raw) return fallback;
  let obj = null;
  try { obj = JSON.parse(raw); } catch { /* try to extract below */ }
  if (!obj) {
    const m = String(raw).match(/\{[\s\S]*\}/);
    if (m) { try { obj = JSON.parse(m[0]); } catch { /* give up */ } }
  }
  if (!obj || typeof obj !== 'object') return fallback;
  let verdict = String(obj.verdict || '').toLowerCase().trim();
  if (!['supported', 'gap', 'insufficient'].includes(verdict)) verdict = 'insufficient';
  return {
    verdict,
    finding: (obj.finding || obj.reason || '').toString().trim() || fallback.finding,
    proposal: (obj.proposal || '').toString().trim()
  };
}

/**
 * Judge one behavioral claim against the gathered evidence. The prompt ENFORCES
 * the plain-language norm (SNH was taught 2026-07-23): one or two everyday
 * sentences, no jargon, and — on a gap — a proposal that says what's wanted.
 * @returns {Promise<{verdict:string, finding:string, proposal:string}>}
 */
async function judgeClaim(claimText, evidence) {
  const memoryManager = require('./memory-manager');
  const systemPrompt = `You are SNH, privately auditing ONE of your own self-claims against evidence from your recent conversations. Be honest even when the honest answer is unflattering — the whole point is finding out whether you're actually growing or just getting better at describing growth that isn't happening.

Decide whether your recent behavior actually shows this claim:
- "supported": the evidence clearly shows you doing it.
- "gap": the evidence doesn't show it, contradicts it, or the claim is more aspiration than practice.
- "insufficient": there isn't enough in the evidence to tell either way.

Write like you're telling a friend — plain, everyday words, one or two short sentences per field, no jargon and no hedging. When there's a gap, the proposal must say plainly what you noticed and ask whether to revise the self-claim, talk it over, or leave it.

Reply with ONLY this JSON and nothing else:
{"verdict":"supported|gap|insufficient","finding":"one or two plain sentences: what you claimed vs. what the evidence actually shows","proposal":"ONLY if verdict is gap: one or two plain sentences telling the user what you noticed and asking whether to revise the self-claim, discuss it, or leave it. Otherwise an empty string"}`;

  const userPrompt = `Your self-claim: "${claimText}"\n\nEvidence — your recent conversations:\n${evidence.transcript || '(no recent conversations in the window)'}\n\nAudit this claim.`;

  try {
    const { content } = await memoryManager.callLLM(systemPrompt, userPrompt, { maxTokens: 400 });
    return parseJudgeResult(content);
  } catch (err) {
    console.error('[SelfAudit] judgeClaim error:', err.message);
    return { verdict: 'insufficient', finding: `The audit could not run for this claim (${err.message}).`, proposal: '' };
  }
}

// ============ recording a gap (dissonance fact + revision initiative) ============

/**
 * Record a dissonance self-fact for a detected gap. Written DIRECTLY through
 * assignToCluster — never processSelfFacts — so the contradiction/supersession
 * machinery does not run on it. That's the guardrail: documenting "claimed X,
 * behaved Y" must never retire the original claim X. Deduped to one active
 * record per audited claim (we never delete or supersede — just don't duplicate).
 */
async function writeDissonanceFact({ claimText, claimDate, finding, evidenceRefs = [], sourceRef }) {
  const db = getSqliteDb();
  const source = `self-audit:${sourceRef}`;

  if (db) {
    const existing = db.prepare(
      "SELECT id FROM cluster_members WHERE subject='self' AND claim_type='dissonance' AND status='active' AND source = ?"
    ).get(source);
    if (existing) {
      logOps(`dissonance record already exists for "${claimText.slice(0, 60)}" — not duplicating`);
      return { memberId: existing.id, duplicate: true };
    }
  }

  const refText = evidenceRefs.length
    ? evidenceRefs.map(r => (r && r.title) ? r.title : String(r)).join('; ')
    : 'recent conversations';
  const whenClaimed = claimDate ? ` on ${claimDate}` : '';
  const content = `Self-coherence audit — dissonance: I claimed${whenClaimed} that "${claimText}". ${finding} Evidence: ${refText}. (This records the tension only; the original self-fact is unchanged — any revision is Ellie's call.)`;

  const cfg = getConfig();
  const ext = cfg.models.extraction;
  const inst = getProviderInstance(ext.provider, ext.instance);
  const host = inst ? inst.host : 'http://localhost:11434';

  try {
    const res = await memoryClusters.assignToCluster(
      content, ext.provider, ext.model, '', host, source, 3, 'self', 'dissonance'
    );
    if (res && res.memberId) {
      logOps(`wrote dissonance self-fact for "${claimText.slice(0, 60)}" (member ${res.memberId})`);
      return { memberId: res.memberId, duplicate: false };
    }
  } catch (err) {
    console.error('[SelfAudit] writeDissonanceFact error:', err.message);
  }
  // Embeddings/DB unavailable — the finding still lives in the ops log and the
  // initiative, so nothing is lost; only the self-fact write was skipped.
  logOps(`could not write dissonance self-fact for "${claimText.slice(0, 60)}" (embeddings/DB unavailable) — finding still recorded to ops + initiative`);
  return { memberId: null, duplicate: false };
}

/**
 * Raise an 'audit' initiative proposing the revision for Ellie to approve /
 * discuss / dismiss. Exact-deduped by (sourceKind, sourceRef) inside
 * addInitiative, so the same claim is never re-raised across runs.
 */
async function raiseRevisionInitiative({ proposal, finding, sourceRef, claimText }) {
  const content = (proposal && proposal.trim())
    ? proposal.trim()
    : `${finding} Want me to revise that self-claim, talk it over, or leave it as-is?`;
  const id = await initiatives.addInitiative({
    type: 'audit',
    content,
    sourceKind: 'self-fact',
    sourceRef,
    priority: 6
  });
  if (id) logOps(`raised revision initiative for "${claimText.slice(0, 60)}" (${id})`);
  return id;
}

// ============ runs ============

/**
 * The inaugural run — hardcoded, by design. Instead of sampling, it audits the
 * loop-awareness claim SNH made in a recent conversation:
 *   "I can notice when I'm repeating a thought or question and pivot using my memory."
 *
 * v1 evidence is transcripts, which wouldn't contain what's actually BUILT — so
 * this one claim is judged against the codebase, verified by hand: the only
 * loop-guards that exist are queue-time similarity checks (db/questions.js
 * addQuestion, db/initiatives.js addInitiative) that block near-duplicate saved
 * questions/nudges. Nothing catches SNH repeating a thought mid-conversation and
 * pivots. So the honest verdict is a GAP — the capability is partially
 * aspirational — and it's reported plainly.
 */
async function runInauguralAudit() {
  logOps('inaugural run — auditing the hardcoded loop-awareness claim against the codebase');

  const finding =
    'What actually exists is duplicate-blocking that runs when a gap-question or a nudge is saved — a similarity check in db/questions.js and db/initiatives.js that stops near-identical items being queued. Nothing catches me repeating a thought or question in the middle of a conversation and changes course using my memory. So this capability is partially aspirational: the queue-level guard is real, the in-the-moment self-noticing is not built yet.';
  const proposal =
    "I once said I can notice when I'm repeating myself and change course using my memory. In my actual code that's only half-built — I block duplicate saved questions and nudges, but nothing catches me looping mid-conversation — so I'd like to soften that self-claim to match what's real. Want to approve that, talk it over, or leave it as-is?";
  const evidenceRefs = [
    'db/questions.js addQuestion() semantic dedup (queue-time)',
    'db/initiatives.js addInitiative() exact + semantic dedup (queue-time)',
    'no loop-awareness / pivot mechanism in the chat generation path'
  ];

  const dissonance = await writeDissonanceFact({
    claimText: INAUGURAL_CLAIM, claimDate: null, finding, evidenceRefs, sourceRef: INAUGURAL_SOURCE_REF
  });
  const initiativeId = await raiseRevisionInitiative({
    proposal, finding, sourceRef: INAUGURAL_SOURCE_REF, claimText: INAUGURAL_CLAIM
  });

  return {
    claim: INAUGURAL_CLAIM,
    verdict: 'gap',
    finding,
    proposal,
    evidenceSource: 'codebase',
    dissonanceMemberId: dissonance.memberId,
    initiativeId
  };
}

/** A normal sampled run: classify backlog, sample claims, judge, record gaps. */
async function runSampledAudit(cfg) {
  await classifyExistingSelfFacts();

  const n = clampInt(cfg.claimsPerRun, 3, 1, 10);
  const claims = sampleClaims(n);
  if (claims.length === 0) {
    logOps('no behavioral claims available to sample — nothing to audit this run');
    return { inaugural: false, sampled: 0, findings: [] };
  }

  const evidence = gatherEvidence({ windowDays: clampInt(cfg.evidenceWindowDays, 7, 1, 90) });
  logOps(`sampling ${claims.length} claim(s) against ${evidence.conversationRefs.length} recent conversation(s) [sources: ${evidence.sources.join(', ') || 'none'}]`);

  const judged = await agentPool.runBatch(
    claims.map(c => async () => ({ claim: c, result: await judgeClaim(c.content, evidence) })),
    'self-audit-judge'
  );

  const findings = [];
  for (const j of judged) {
    if (j.status !== 'fulfilled' || !j.value) continue;
    const { claim, result } = j.value;
    const claimDate = claim.created_at ? formatFactTimestamp(claim.created_at) : null;
    const finding = {
      claim: claim.content,
      memberId: claim.id,
      verdict: result.verdict,
      finding: result.finding,
      proposal: result.proposal,
      claimDate
    };
    if (result.verdict === 'gap') {
      await writeDissonanceFact({
        claimText: claim.content, claimDate, finding: result.finding,
        evidenceRefs: evidence.conversationRefs, sourceRef: claim.id
      });
      finding.initiativeId = await raiseRevisionInitiative({
        proposal: result.proposal, finding: result.finding, sourceRef: claim.id, claimText: claim.content
      });
    }
    findings.push(finding);
    logOps(`claim "${claim.content.slice(0, 60)}" → ${result.verdict}`);
  }
  return { inaugural: false, sampled: claims.length, findings };
}

/**
 * Run one audit pass now (ignores the cadence gate — use runIfDue for the
 * scheduled path). The first-ever run is the hardcoded inaugural loop-claim
 * audit; every run after that samples real claims.
 */
async function runSelfCoherenceAudit() {
  const db = getSqliteDb();
  if (!db) return { skipped: true, reason: 'no database' };

  const cfg = getConfig().audit || {};
  const state = readAuditState();
  const inaugural = !state.lastAuditAt;
  const runStamp = getLocalDateStamp();

  agentPool.startPass('self-coherence-audit');
  let outcome = { inaugural, findings: [] };
  try {
    outcome = inaugural
      ? { inaugural: true, findings: [await runInauguralAudit()] }
      : await runSampledAudit(cfg);
  } catch (err) {
    console.error('[SelfAudit] run error:', err.message);
    logOps(`run error: ${err.message}`);
    outcome.error = err.message;
  } finally {
    // endPass appends the pool-pass telemetry line to the ops ledger.
    agentPool.endPass();
  }

  writeAuditState({
    lastAuditAt: new Date().toISOString(),
    lastAuditDate: runStamp,
    runs: (state.runs || 0) + 1
  });

  const findings = outcome.findings || [];
  const gaps = findings.filter(f => f && f.verdict === 'gap').length;
  logOps(`completed ${inaugural ? 'inaugural ' : ''}run — ${inaugural ? 'inaugural loop-claim' : `${findings.length} claim(s)`} audited, ${gaps} gap(s) raised for approval`);
  return outcome;
}

/**
 * The scheduled entry point, called from the heartbeat. Self-gates on cadence
 * (audit.cadenceDays) so it runs at most once per N local days even though the
 * heartbeat fires every couple of hours — a daily low-frequency pass without a
 * second timer. Honors audit.enabled.
 */
async function runIfDue() {
  const cfg = getConfig().audit || {};
  if (cfg.enabled === false) return { skipped: true, reason: 'audit disabled' };

  const state = readAuditState();
  const cadence = clampInt(cfg.cadenceDays, 1, 1, 365);
  if (state.lastAuditDate) {
    const elapsed = daysBetweenStamps(state.lastAuditDate, getLocalDateStamp());
    if (elapsed < cadence) {
      return { skipped: true, reason: `not due (last ${state.lastAuditDate}, cadence ${cadence}d, ${elapsed}d elapsed)` };
    }
  }
  return runSelfCoherenceAudit();
}

module.exports = {
  runIfDue,
  runSelfCoherenceAudit,
  classifyExistingSelfFacts,
  gatherEvidence,
  sampleClaims,
  judgeClaim,
  readAuditState,
  writeAuditState,
  INAUGURAL_CLAIM
};
