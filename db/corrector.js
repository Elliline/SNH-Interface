/**
 * The corrector — the step the spec exists for.
 *
 * Everything before this built a corpus that stops getting worse. This is the
 * part that makes it better: it finds what is already wrong and repairs it,
 * unattended, on a cadence, without a person editing rows.
 *
 * ARCHITECTURE. Enumeration is deterministic and judgment is not. A model cannot
 * scan 580 facts, so candidate pairs come from vector neighbours, marker
 * regexes, and reconcile() — cheap, complete, repeatable. Every actual DECISION
 * (is this the same assertion? do these contradict? which one wins?) is a model
 * call, because those are judgment calls and pretending a regex can make them is
 * how you get a corrector that confidently destroys things. WRITES go through
 * the background tool layer into db/fact-store.js, so the funnel, the budget and
 * the audit trail are the same ones everything else uses.
 *
 * TIERS, and the autonomy each one gets:
 *
 *   MECHANICAL — autonomous and silent. Near-duplicate merges, expiry of facts
 *   that were always events, vector/SQLite reconciliation, compound splits.
 *   These repair the RECORD; they do not revise a belief. Silent means nobody is
 *   interrupted, not that nothing is written down: everything lands in
 *   corrections_ledger regardless.
 *
 *   SEMANTIC — autonomous, reversible, logged, and announced when it touches a
 *   self-fact. Contradiction resolution between existing facts, decided by
 *   EVIDENCE DOMINANCE read from stored provenance. A supersession that cannot
 *   demonstrate dominance is NOT applied — it is raised as unresolved.
 *
 *   IRREVERSIBLE — never autonomous. Nothing here deletes. There is no code path
 *   in this module that removes a row.
 *
 * BOUNDED AND RESUMABLE. A pass stops cleanly when its call or wall-clock budget
 * is spent and the next pass picks up where it left off — the enumeration is
 * ordered and stable, and anything already corrected is no longer a candidate,
 * so "resume" is just "run again".
 */

const { randomUUID } = require('crypto');
const path = require('path');
const { getSqliteDb, getDataDir } = require('./database');
const { getConfig } = require('./config');
const { getLocalDateStamp } = require('./datetime');

/**
 * Resolved from the PROCESS's data directory, not from __dirname.
 *
 * This module writes three things outside SQLite: the pass-state file, the day's
 * log it moves an expired event into, and its ops-ledger line. A constant here
 * would send all three to `data/memory/` no matter which corpus the process is
 * pointed at — the "silently escapes the redirect" failure the replay's design
 * note warns about, and the worst of the three is not the logs. `lastPassAt` is
 * read by the live heartbeat to decide whether the corrector is due, so a pass
 * run against staging would tell the LIVE corrector it had just run and suppress
 * its next real one. Found when the first staging pass was about to be made.
 *
 * A function rather than a constant because SNH_DATA_DIR is read at
 * database.js's load time and this module can be required before or after it;
 * asking each time costs a path join and cannot be stale.
 */
function memoryDir() { return path.join(getDataDir(), 'memory'); }

function cfg() {
  return getConfig().corrector || {};
}

function factExtractor() { return require('./fact-extractor'); }
function memoryClusters() { return require('./memory-clusters'); }
function ledger() { return require('./corrections-ledger'); }

// ---------------------------------------------------------------------------
// Pass state on disk — WHEN a pass last ran, not what it did
// ---------------------------------------------------------------------------
//
// The cadence has to be measured against passes, not against corrections. The
// heartbeat originally read `MAX(created_at)` from the ledger, which is the time
// of the last CHANGE — so a clean corpus (the state this whole phase is working
// toward) leaves the gate permanently overdue and the corrector runs on every
// heartbeat instead of on its own cadence. A file rather than a table because it
// is one timestamp, and the audit and manifest steps already keep theirs this
// way. Survives restarts, which is the point: a redeploy must not hand it a
// fresh turn.
function stateFile() { return path.join(memoryDir(), 'corrector-state.json'); }

function readState() {
  try { return JSON.parse(require('fs').readFileSync(stateFile(), 'utf8')) || {}; } catch { return {}; }
}

/** ISO timestamp of the last completed pass, or null if none has ever run. */
function lastPassAt() {
  const s = readState();
  if (s.lastPassAt) return s.lastPassAt;
  // Pre-upgrade fallback: before the state file existed the only evidence a pass
  // ever ran was the ledger. Reading it once here means the first pass after the
  // upgrade is scheduled from real history rather than treated as the first ever.
  try {
    const db = getSqliteDb();
    return db ? (db.prepare('SELECT MAX(created_at) AS t FROM corrections_ledger').get().t || null) : null;
  } catch { return null; }
}

function recordPassRan(result) {
  try {
    const fs = require('fs');
    fs.mkdirSync(memoryDir(), { recursive: true });
    fs.writeFileSync(stateFile(), JSON.stringify({
      lastPassAt: new Date().toISOString(),
      lastPassId: result.passId,
      lastPassChanges: (result.merged || 0) + (result.expired || 0) + (result.split || 0) + (result.superseded || 0),
      lastPassStopped: result.stopped || null
    }, null, 2));
  } catch (err) {
    console.error('[Corrector] could not record pass state:', err.message);
  }
}

const trim = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

// ---------------------------------------------------------------------------
// Evidence dominance
// ---------------------------------------------------------------------------

/**
 * Score a fact's evidential weight from what is actually STORED about it.
 *
 * The spec's ordering, in order of precedence:
 *   1. typed > stt > unknown          (modality)
 *   2. direct statement > inference   (does the verbatim source actually say it)
 *   3. corroborated > single mention  (how many times it was asserted)
 *   4. recent > stale                 (when it was learned)
 *
 * Every signal comes from the record. Nothing here asks a model what it thinks
 * is more likely — that is the whole point of the rule. A fact with no stored
 * provenance scores the floor on the first two axes, which is exactly right:
 * decision 2 says `unknown` loses to `typed` and cannot justify a supersession
 * on its own.
 *
 * @returns {{modality: number, directness: number, corroboration: number, recency: number, detail: Object}}
 */
function evidenceProfile(row) {
  const db = getSqliteDb();

  const mod = String(row.input_modality || 'unknown').toLowerCase();
  const modality = mod === 'typed' ? 2 : mod === 'stt' ? 1 : 0;

  // Directness: does the source text the fact was extracted from actually
  // contain the claim, or was the fact inferred from it? Approximated by asking
  // whether the fact's distinctive words appear in the verbatim source. Crude on
  // purpose — a model call here would be a model deciding its own evidence.
  let directness = 0;
  const verbatim = String(row.verbatim_source_text || '').toLowerCase();
  if (verbatim) {
    const content = String(row.content || '').toLowerCase();
    const words = content
      .replace(/^(the\s+)?user'?s?\s+/i, '')
      .split(/\W+/)
      .filter(w => w.length > 3 && !['user', 'that', 'this', 'with', 'from', 'have', 'their', 'them'].includes(w));
    const hits = words.filter(w => verbatim.includes(w)).length;
    directness = words.length === 0 ? 0 : (hits / words.length >= 0.5 ? 2 : hits > 0 ? 1 : 0);
  }

  let corroboration = 0;
  try {
    corroboration = db.prepare('SELECT COUNT(*) AS n FROM fact_corroborations WHERE member_id = ?').get(row.id).n;
  } catch { /* table may be absent on an un-migrated DB */ }

  // CORPUS ATTESTATION — corroboration of the VALUE, not of the row.
  //
  // Row-level corroborations only exist for facts written since Phase 2a, so on
  // the historical corpus every fact scores zero and the axis is dead. But the
  // corpus itself attests: "Ellie" appears in nine active facts about her, "Mike"
  // in exactly one. That is the same signal — this has been said more than once —
  // read from the only place it survives for a pre-provenance fact.
  //
  // Scoped to identity-class facts, where the asserted value is extractable and
  // the count means something. Counting substring hits for an arbitrary fact
  // would measure topic popularity, not corroboration.
  let attestation = 0;
  let attestedValue = null;
  try {
    const rules = require('./extraction-rules');
    const klass = rules.identityClassOf(row.content);
    if (klass && klass.klass === 'name') {
      // The asserted name: the first capitalised token after the copula.
      const m = String(row.content).match(/\b(?:is|named|called|goes by)\s+([A-Z][a-zA-Z'\-]{1,})/);
      if (m) {
        attestedValue = m[1];
        attestation = db.prepare(
          `SELECT COUNT(*) AS n FROM cluster_members
           WHERE status = 'active' AND subject = ? AND id != ? AND content LIKE ?`
        ).get(row.subject || 'user', row.id, `%${attestedValue}%`).n;
      }
    }
  } catch { /* attestation is a bonus signal; never let it break the profile */ }

  const recency = row.created_at ? new Date(row.created_at).getTime() : 0;

  return {
    modality, directness,
    // Both kinds of corroboration feed one axis — they are the same claim
    // ("this was asserted more than once") measured two ways.
    corroboration: corroboration + attestation,
    recency,
    detail: {
      input_modality: mod,
      has_verbatim_source: !!verbatim,
      directness_score: directness,
      corroborations: corroboration,
      corpus_attestation: attestedValue ? { value: attestedValue, other_facts_mentioning_it: attestation } : null,
      learned: row.created_at || null,
      salience: row.salience ?? 5
    }
  };
}

/**
 * Which of two facts dominates on evidence?
 *
 * Lexicographic in the spec's order, and STRICT: the first axis that separates
 * them decides, and if none does the answer is null — a tie means no action.
 * That is the rule that keeps this honest. A corrector that broke ties by
 * salience, or by asking the model to pick, would be applying model opinion
 * while claiming to apply evidence.
 *
 * @returns {{winner: Object, loser: Object, axis: string, evidence: Object}|null}
 */
function dominance(a, b) {
  const ea = evidenceProfile(a);
  const eb = evidenceProfile(b);

  // The three axes that are actual EVIDENCE about the claim.
  const evidential = [
    ['modality', 'input modality (typed beats spoken beats unknown)'],
    ['directness', 'directness (stated outright in the source beats inferred from it)'],
    ['corroboration', 'corroboration (said more than once beats said once)']
  ];
  for (const [key, label] of evidential) {
    if (ea[key] === eb[key]) continue;
    const aWins = ea[key] > eb[key];
    return {
      winner: aWins ? a : b, loser: aWins ? b : a, axis: label,
      evidence: { winner: aWins ? ea.detail : eb.detail, loser: aWins ? eb.detail : ea.detail }
    };
  }

  // RECENCY IS NOT DOMINANCE ON ITS OWN.
  //
  // Found on the first live-corpus dry run, and it is the sharpest thing this
  // pass turned up. Every fact written before Phase 1 carries unknown modality,
  // no verbatim source and no corroborations — an identical, empty profile — so
  // the first three axes tie for essentially the whole historical corpus and
  // dominance fell straight through to "newer wins". That is not evidence
  // dominance; it is a timestamp comparison wearing its clothes, and it proposed
  // retiring "User does not program" in favour of an unrelated fact about
  // software sovereignty purely because that one was logged three weeks later.
  //
  // Decision 2 already says the quiet part: `unknown` "cannot be used to justify
  // a supersession on its own". Nor can recency. When nothing evidential
  // separates two facts, the honest answer is that the corpus does not know, and
  // the pair is raised rather than resolved.
  //
  // Recency still breaks ties — but only once something evidential has already
  // spoken, which is what the `separated` flag records for the caller.
  return null;
}


/**
 * Row-level scan memo.
 *
 * Pair-level persistence was not enough, and the reason is worth writing down:
 * the cost of a pass is dominated by the EMBEDDING call that finds a row's
 * neighbours, not by the judge calls on the pairs it returns. With 227 active
 * facts at roughly 1.3s per embedding, simply enumerating candidates spends the
 * entire 300s budget before a single judgement is made — pass 2 did exactly that
 * and accomplished nothing in 301 seconds.
 *
 * So a row that has been fully scanned for a phase is skipped outright next
 * pass, embedding and all, until the row itself changes. `updated_at` is the
 * invalidation key: edit a fact and it is rescanned.
 */
// The duplicate sweep's memo is VERSIONED, because the memo caches an answer to
// a question and the question has changed. v1 asked only "do these assert the
// same thing?", which is defined to say DIFFERENT for a subset beside its
// superset — so every F4-shaped pair carries a cached 'different' verdict, and
// every row carries a scan mark saying it has been looked at. Bumping the
// version retires both in one move and lets the corpus be swept again with the
// subsumption question included. Bump it whenever the merge phase's question
// changes; do not reuse a version for a different question.
const DUP_CHECK_VERSION = 2;
const DUP_PHASE = `dup${DUP_CHECK_VERSION}`;

function scanKey(phase, rowId) { return `scan:${phase}:${rowId}`; }

function alreadyScanned(db, phase, row) {
  try {
    const r = db.prepare('SELECT fact_updated_at FROM corrector_pair_checks WHERE pair_key = ?').get(scanKey(phase, row.id));
    return !!r && r.fact_updated_at === (row.updated_at || row.created_at || '');
  } catch { return false; }
}

function markScanned(db, phase, row) {
  try {
    db.prepare(
      'INSERT OR REPLACE INTO corrector_pair_checks (pair_key, checked_at, verdict, a_id, fact_updated_at) VALUES (?, ?, ?, ?, ?)'
    ).run(scanKey(phase, row.id), new Date().toISOString(), 'scanned', row.id, row.updated_at || row.created_at || '');
  } catch { /* non-fatal */ }
}

// ---------------------------------------------------------------------------
// Pass state
// ---------------------------------------------------------------------------

/**
 * One corrector pass. `dryRun` makes every action a plan entry and nothing else
 * — same enumeration, same judges, same dominance rules, no writes. The dry-run
 * harness pattern from Phase 2a: a rehearsal that runs different code proves
 * nothing about the code that will run.
 */
function newPass({ dryRun = false, session = null, passId = null } = {}) {
  const c = cfg();
  return {
    passId: passId || randomUUID(),
    dryRun,
    session,
    startedMs: Date.now(),
    // ONE budget, not two. When the heartbeat dispatches this it hands in a tool
    // session already carrying the corrector's limits from config; the pass reads
    // them off the session rather than re-deriving them, so there is a single
    // number that can be spent and a single place it is reported from. A direct
    // CLI run has no session and falls back to the same config keys.
    maxCalls: session ? session.maxCalls : (Number.isFinite(c.maxToolCallsPerPass) ? c.maxToolCallsPerPass : 60),
    maxWallMs: session ? session.maxWallMs : (Number.isFinite(c.maxWallClockMsPerPass) ? c.maxWallClockMsPerPass : 300000),
    writes: 0,
    selfCorrections: 0,
    stopped: null,
    plan: [],     // every action, taken or planned
    unresolved: []
  };
}

/** Has this pass run out of room? Checked before every write. */
function budgetSpent(pass) {
  if (pass.writes >= pass.maxCalls) return `write budget spent (${pass.writes}/${pass.maxCalls})`;
  const elapsed = Date.now() - pass.startedMs;
  if (elapsed >= pass.maxWallMs) return `time budget spent (${Math.round(elapsed / 1000)}s of ${Math.round(pass.maxWallMs / 1000)}s)`;
  return null;
}

/**
 * Perform one write action — or, in a dry run, record what it would have been.
 *
 * Routed through the background tool layer so the corrector's writes are counted
 * by the same budget, checked against the same allowlist and audited by the same
 * path as everything else a background step does.
 */
async function act(pass, { tool, args, entry }) {
  pass.plan.push({ ...entry, applied: false, dryRun: pass.dryRun });
  const planned = pass.plan[pass.plan.length - 1];

  if (pass.dryRun) return { status: 'planned' };

  const spent = budgetSpent(pass);
  if (spent) {
    pass.stopped = spent;
    planned.skipped = spent;
    return { status: 'skipped', reason: spent };
  }

  const memoryManager = require('./memory-manager');
  let result;
  if (pass.session) {
    result = await memoryManager.executeBackgroundTool(pass.session, tool, args);
  } else {
    // No session (a direct CLI run). Go through the same tool objects rather
    // than reaching past them into fact-store — one write path, always.
    const MCPClient = require('../mcp/mcp-client');
    result = await MCPClient.shared().executeTool(tool, args, { caller: 'corrector' });
  }
  pass.writes++;

  if (result && result.status === 'refused_locked') {
    // The identity lock said no. This is a RESULT, not a failure: the lock exists
    // to be hit, and a corrector that swallowed the refusal would be the silent
    // acceptance the whole design forbids.
    planned.refusedLocked = true;
    planned.refusalMessage = result.message;
    ledger().record({
      passId: pass.passId, tier: entry.tier, action: entry.action, subject: entry.subject,
      targetId: entry.targetId, targetText: entry.targetText,
      reason: `REFUSED by the identity lock — ${result.message}`,
      evidence: entry.evidence, reversible: false
    });
    console.warn(`[Corrector] identity lock refused ${entry.action} on ${String(entry.targetId).slice(0, 8)}`);
    return result;
  }
  if (result && result.error) {
    planned.error = result.error;
    return result;
  }

  planned.applied = true;
  planned.ledgerId = ledger().record({
    passId: pass.passId, tier: entry.tier, action: entry.action, subject: entry.subject,
    targetId: entry.targetId, targetText: entry.targetText,
    survivorId: entry.survivorId, survivorText: entry.survivorText,
    reason: entry.reason, evidence: entry.evidence, reversible: entry.reversible !== false
  });
  return result;
}

// ---------------------------------------------------------------------------
// MECHANICAL 1 — near-duplicate and subset merge
// ---------------------------------------------------------------------------

/**
 * Extend Phase 2a's write-time repeat detection to the corpus that already
 * exists. 2a stops NEW duplicates; this removes the ones already there, which is
 * how the machine-gun triple and the Casper subset finally go.
 *
 * Subsets fold too, but they are a SEPARATE question asked with a separate judge
 * (`judgeSubsumption`) — the repeat judge answers DIFFERENT for a subset beside
 * its superset, by design, because at intake the richer sentence carries new
 * information. When one row contains the other the survivor is settled by the
 * relation rather than weighed: "has a dog named Casper who helps pull them up
 * hills" absorbs "has a dog named Casper", never the reverse. Detail is never
 * traded away for brevity.
 */
async function mergeNearDuplicates(pass, { subject = null } = {}) {
  const db = getSqliteDb();
  const c = cfg();
  const floor = Number.isFinite(c.nearDupFloor) ? c.nearDupFloor : 0.86;
  const maxPairs = Number.isInteger(c.maxNearDupPairsPerPass) ? c.maxNearDupPairsPerPass : 40;

  const rows = db.prepare(`
    SELECT * FROM cluster_members WHERE status = 'active'
    ${subject ? 'AND subject = ?' : ''}
    ORDER BY datetime(created_at) ASC
  `).all(...(subject ? [subject] : []));

  const gone = new Set();   // folded away in THIS pass

  // Same persistence the semantic tier uses, and for the same reason: without it
  // this phase re-judges every non-matching pair on every pass, always from the
  // top, and always spends the whole budget before the later tiers run at all.
  // Keys are prefixed by CHECK KIND — "is this the same assertion" and "do these
  // contradict" are different questions about the same pair.
  const seenStmt = db.prepare('SELECT verdict FROM corrector_pair_checks WHERE pair_key = ?');
  const markStmt = db.prepare(
    'INSERT OR REPLACE INTO corrector_pair_checks (pair_key, checked_at, verdict, a_id, b_id) VALUES (?, ?, ?, ?, ?)'
  );
  let pairsChecked = 0;

  for (const row of rows) {
    if (gone.has(row.id)) continue;
    if (pairsChecked >= maxPairs) break;
    if (budgetSpent(pass) && !pass.dryRun) { pass.stopped = budgetSpent(pass); break; }
    if (!pass.dryRun && alreadyScanned(db, DUP_PHASE, row)) continue;

    // includeVerbatim — the one caller that WANTS word-for-word matches. The
    // default filter treats them as repeats and drops them, which made three
    // byte-identical rows invisible to the phase built to fold them.
    const { candidates } = await memoryClusters().findActiveNeighbours(row.content, {
      subject: row.subject || 'user', threshold: floor, limit: 6, includeVerbatim: true
    });

    for (const cand of candidates) {
      if (cand.memberId === row.id || gone.has(cand.memberId)) continue;
      pairsChecked++;
      if (pairsChecked > maxPairs) break;

      const dupKey = `${DUP_PHASE}:${[row.id, cand.memberId].sort().join('|')}`;
      if (!pass.dryRun) {
        try { if (seenStmt.get(dupKey)) { pairsChecked--; continue; } } catch { /* judge it */ }
      }

      const other = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(cand.memberId);
      if (!other || other.status !== 'active') continue;

      // TWO QUESTIONS, asked in order, because "duplicate" covers two different
      // relations and one judge cannot answer both:
      //
      //   1. Do these assert the same thing? (a rewording — F2 once it is
      //      visible, and the MettaSphere pairs)
      //   2. Failing that, does one already contain everything the other says?
      //      (a subset beside its superset — F4)
      //
      // The repeat judge cannot be reused for the second: its prompt names the
      // Casper pair as DIFFERENT on purpose, because at INTAKE the richer
      // sentence carries new information and must be stored. Here both rows are
      // already held, and the subset is the impoverished copy.
      //
      // Byte-identical pairs skip both — there is nothing to judge.
      const identical = row.content.trim().toLowerCase() === other.content.trim().toLowerCase();
      let same = identical;
      let why = identical ? 'identical wording' : null;
      let subsumedBy = null;     // the row that contains the other, when that is how they relate
      if (!identical) {
        const verdict = await factExtractor().judgeSameAssertion(other.content, row.content);
        same = verdict.same;
        why = verdict.reasoning;
        if (!same) {
          const sub = await factExtractor().judgeSubsumption(row.content, other.content);
          if (sub.relation === 'a-contains-b') { subsumedBy = row; why = sub.reasoning; }
          else if (sub.relation === 'b-contains-a') { subsumedBy = other; why = sub.reasoning; }
        }
      }
      if (!pass.dryRun) {
        const verdictLabel = same ? 'same' : subsumedBy ? 'subset' : 'different';
        try { markStmt.run(dupKey, new Date().toISOString(), verdictLabel, row.id, cand.memberId); } catch { /* non-fatal */ }
      }
      if (!same && !subsumedBy) continue;

      // SCOPE GATE. The instruction is explicit that the mechanical tier applies
      // to self-facts as "exact dups, reconcile" — and a near-duplicate merge is
      // neither exact nor reconciliation. Rewording two of his self-observations
      // into one is a judgement about how he describes himself, and Phase 2e
      // promises that judgement is made WITH him. So an exact duplicate of a
      // self-fact folds away here; a merely-similar pair waits for the session.
      if ((row.subject || 'user') === 'self' && !identical && cfg().selfFactSemantic !== true) {
        continue;
      }

      // SURVIVOR. The judge picks, because length is not information: the first
      // dry run chose "User's Managed Service Provider (MSP) is called
      // MettaSphere." over "User's MSP is MettaSphere LLC" purely for being
      // longer, and silently dropped "LLC" in the process. Identical pairs skip
      // the call — there is nothing to choose between them.
      let survivor, loser;
      if (subsumedBy) {
        // Already decided by the relation itself: the container survives. There
        // is nothing to weigh — the other row's content is a strict subset of
        // this one's, so keeping the subset would be choosing to know less.
        survivor = subsumedBy;
        loser = subsumedBy === row ? other : row;
      } else if (identical) {
        // Keep the one with more evidence behind it, then the original.
        const ea = evidenceProfile(row), eb = evidenceProfile(other);
        if (ea.corroboration !== eb.corroboration) {
          [survivor, loser] = ea.corroboration > eb.corroboration ? [row, other] : [other, row];
        } else if ((row.salience ?? 5) !== (other.salience ?? 5)) {
          [survivor, loser] = (row.salience ?? 5) > (other.salience ?? 5) ? [row, other] : [other, row];
        } else {
          [survivor, loser] = new Date(row.created_at) <= new Date(other.created_at) ? [row, other] : [other, row];
        }
      } else {
        const pick = await factExtractor().judgeWhichSurvives(row.content, other.content);
        if (pick === 'a') [survivor, loser] = [row, other];
        else if (pick === 'b') [survivor, loser] = [other, row];
        else {
          // Judge saw no difference: fall back to evidence, then to the original.
          const aSal = row.salience ?? 5, bSal = other.salience ?? 5;
          if (aSal !== bSal) [survivor, loser] = aSal > bSal ? [row, other] : [other, row];
          else [survivor, loser] = new Date(row.created_at) <= new Date(other.created_at) ? [row, other] : [other, row];
        }
      }

      await act(pass, {
        tool: 'memory_merge_facts',
        args: { loser_id: loser.id, survivor_id: survivor.id },
        entry: {
          tier: 'mechanical', action: 'merge', subject: survivor.subject || 'user',
          targetId: loser.id, targetText: loser.content,
          survivorId: survivor.id, survivorText: survivor.content,
          reason: identical
            ? 'Two rows held exactly the same fact word for word, so one was folded into the other.'
            : subsumedBy
              ? `One of these already said everything the other did, and more, so the fuller one was kept and the shorter folded into it. ${trim(why, 160)}`
              : `These said the same thing in different words, so the fuller one was kept and the other folded into it. ${trim(why, 160)}`,
          evidence: {
            similarity: cand.similarity, identical, subset: !!subsumedBy,
            survivor_salience: survivor.salience, loser_salience: loser.salience
          }
        }
      });
      gone.add(loser.id);
      if (loser.id === row.id) break; // this row is gone; stop pairing it
    }
    if (!pass.dryRun && !gone.has(row.id)) markScanned(db, DUP_PHASE, row);
  }
  return { pairsChecked, merged: gone.size };
}

// ---------------------------------------------------------------------------
// MECHANICAL 2 — dated-event expiry
// ---------------------------------------------------------------------------

/**
 * Facts that carry a time qualifier were events all along — written before
 * intake could tell the difference. They are moved, not lost: the text goes to
 * the daily log under the date it was LEARNED (which is when the event actually
 * happened, near enough), and only then is the fact expired.
 *
 * The markers are exactly the ones db/extraction-rules.js uses at intake, so a
 * fact the current pipeline would route to the log is a fact this removes. One
 * rule, two enforcement points.
 */
async function expireDatedEvents(pass, { subject = null } = {}) {
  const db = getSqliteDb();
  const rules = require('./extraction-rules');
  const c = cfg();
  const maxExpiries = Number.isInteger(c.maxExpiriesPerPass) ? c.maxExpiriesPerPass : 25;

  const rows = db.prepare(`
    SELECT * FROM cluster_members WHERE status = 'active'
    ${subject ? 'AND subject = ?' : ''}
    ORDER BY datetime(created_at) ASC
  `).all(...(subject ? [subject] : []));

  let expired = 0;
  for (const row of rows) {
    if (expired >= maxExpiries) break;
    if (!pass.dryRun && budgetSpent(pass)) { pass.stopped = budgetSpent(pass); break; }

    // The marker is a CANDIDATE FILTER, not the decision. See
    // factExtractor.judgeStripTheTimestamp for why: on the marker alone this
    // proposed retiring "User's partner passed away on January 24th, 2025" and
    // every dated capability declaration he holds about himself.
    const marker = rules.eventMarker(row.content);
    if (!marker.isEvent) continue;

    // Capability introductions are deliberately date-stamped by
    // scripts/introduce-capability.js ("As of 2026-08-03, I can…"). They are
    // declarations of what he can do, permanently true, and they are the one
    // class where the marker is guaranteed to be a false positive. Excluded by
    // source rather than trusted to the judge.
    if (String(row.source || '').startsWith('capability-')) continue;

    // Memoised like the other phases. Without it every marker-bearing fact the
    // judge has already called LASTING is re-judged on every pass forever, which
    // is both a standing cost and a reason not to widen the marker list. Keyed on
    // updated_at, so a reworded fact is asked again.
    if (!pass.dryRun && alreadyScanned(db, 'exp', row)) continue;

    const stripTest = await factExtractor().judgeStripTheTimestamp(row.content);
    if (!stripTest.isEvent) { if (!pass.dryRun) markScanned(db, 'exp', row); continue; }

    // Move before removing. Under the date it was learned, so the day's log reads
    // as a record of when it happened rather than of when it was tidied up.
    if (!pass.dryRun) {
      try {
        const learnedDate = row.created_at ? getLocalDateStamp(new Date(row.created_at)) : getLocalDateStamp();
        factExtractor().prependDailyEntry(
          `### (recorded later)\n- ${row.content} [moved out of long-term memory ${getLocalDateStamp()}: this was something that happened, not something that stays true]\n\n`,
          path.join(memoryDir(), 'daily'), learnedDate
        );
      } catch (err) {
        // If the move fails, do NOT expire — that would lose it outright.
        console.error('[Corrector] daily-log move failed, leaving fact alone:', err.message);
        continue;
      }
    }

    await act(pass, {
      tool: 'memory_expire_fact',
      args: { fact_id: row.id },
      entry: {
        tier: 'mechanical', action: 'expire', subject: row.subject || 'user',
        targetId: row.id, targetText: row.content,
        reason: `This was a passing event, not a lasting fact — strip "${marker.marker}" out of it and nothing durable is left. Moved to the day's log and retired from long-term memory. ${trim(stripTest.reasoning, 140)}`,
        evidence: { marker: marker.marker, marker_kind: marker.kind, learned: row.created_at, judge: trim(stripTest.reasoning, 200) }
      }
    });
    expired++;
  }
  return { expired };
}

// ---------------------------------------------------------------------------
// MECHANICAL 3 — reconcile by acting
// ---------------------------------------------------------------------------

/**
 * reconcile() has always been report-only, on the grounds that deciding whether
 * a missing vector is a bug or a deliberate absence is a judgement call. It is
 * not: the invariant is written down and unambiguous. An active fact must be
 * retrievable, an inactive one must not be, a vector with no fact is garbage,
 * and a vector's cluster must be a cluster that exists. All four classes are
 * FIXED here, and the report is kept so it can be checked afterwards — it should
 * read all-zeros after every pass.
 */
async function reconcileByActing(pass) {
  const factStore = require('./fact-store');
  const db = getSqliteDb();
  const { reopenClusterEmbeddingsTable } = require('./database');
  const fixed = { inactiveWithVector: 0, activeNoVector: 0, orphanVectors: 0, staleClusterVectors: 0 };

  let table;
  try { table = await reopenClusterEmbeddingsTable(); } catch { return fixed; }
  if (!table) return fixed;

  let rows;
  try { rows = await table.filter('member_id IS NOT NULL').limit(1000000).execute(); }
  catch { return fixed; }

  const vecMembers = new Set(rows.map(r => r.member_id));
  const members = db.prepare('SELECT id, content, cluster_id, status FROM cluster_members').all();
  const byId = new Map(members.map(m => [m.id, m]));

  // 1. Inactive but still retrievable — the drift that lets a retired belief
  //    surface in an answer.
  for (const m of members) {
    if (m.status === 'active' || !vecMembers.has(m.id)) continue;
    if (pass.dryRun) {
      pass.plan.push({ tier: 'mechanical', action: 'reconcile', targetId: m.id, targetText: m.content,
        reason: 'Retired fact still had an embedding, so it could still surface in a search.', dryRun: true, applied: false });
      fixed.inactiveWithVector++; continue;
    }
    if (await factStore.dropVector(m.id)) {
      fixed.inactiveWithVector++;
      ledger().record({ passId: pass.passId, tier: 'mechanical', action: 'reconcile', targetId: m.id,
        targetText: m.content, reason: 'Retired fact still had an embedding, so it could still surface in a search. Embedding removed.', reversible: false });
    }
  }

  // 2. Active but unsearchable — present in the injected block, invisible to
  //    every search.
  for (const m of members) {
    if (m.status !== 'active' || vecMembers.has(m.id)) continue;
    if (pass.dryRun) {
      pass.plan.push({ tier: 'mechanical', action: 'reconcile', targetId: m.id, targetText: m.content,
        reason: 'Live fact had no embedding, so searching memory could not find it.', dryRun: true, applied: false });
      fixed.activeNoVector++; continue;
    }
    if (await factStore.replaceVector(m.id, m.cluster_id, m.content)) {
      fixed.activeNoVector++;
      ledger().record({ passId: pass.passId, tier: 'mechanical', action: 'reconcile', targetId: m.id,
        targetText: m.content, reason: 'Live fact had no embedding, so searching memory could not find it. Re-embedded.', reversible: false });
    }
  }

  // 3. Vectors whose fact no longer exists.
  const orphans = rows.filter(r => !byId.has(r.member_id));
  for (const o of orphans) {
    if (pass.dryRun) { fixed.orphanVectors++; continue; }
    if (await factStore.dropVector(o.member_id)) fixed.orphanVectors++;
  }
  if (!pass.dryRun && orphans.length) {
    ledger().record({ passId: pass.passId, tier: 'mechanical', action: 'reconcile',
      reason: `${orphans.length} embedding(s) pointed at facts that no longer exist. Removed.`, reversible: false });
  }

  // 4. Vectors carrying a cluster that no longer exists. LanceDB is outside the
  //    cluster_id foreign key, so deleting a cluster leaves its members' stored
  //    cluster_id behind — and cluster assignment reads that field to decide
  //    where a new fact goes, so a ghost can win the match and take the insert
  //    down with it. Found the hard way: the F5 compound split failed on a 0.951
  //    match to a cluster with no row. Re-embedded from the member's CURRENT
  //    cluster, which is the authoritative one.
  const liveClusters = new Set(db.prepare('SELECT id FROM memory_clusters').all().map(c => c.id));
  const stale = rows.filter(r => byId.has(r.member_id) && r.cluster_id && !liveClusters.has(r.cluster_id));
  for (const s of stale) {
    const m = byId.get(s.member_id);
    // An inactive fact is handled by check 1 and must never be re-embedded here.
    if (m.status !== 'active') continue;
    if (pass.dryRun) {
      pass.plan.push({ tier: 'mechanical', action: 'reconcile', targetId: m.id, targetText: m.content,
        reason: 'Embedding pointed at a cluster that no longer exists, which can misroute new facts.', dryRun: true, applied: false });
      fixed.staleClusterVectors++; continue;
    }
    if (await factStore.replaceVector(m.id, m.cluster_id, m.content)) {
      fixed.staleClusterVectors++;
      ledger().record({ passId: pass.passId, tier: 'mechanical', action: 'reconcile', targetId: m.id,
        targetText: m.content, reason: 'Embedding pointed at a cluster that no longer exists, which can misroute new facts. Re-embedded against the cluster the fact is actually in.', reversible: false });
    }
  }

  return fixed;
}

// ---------------------------------------------------------------------------
// MECHANICAL 4 — compound repair
// ---------------------------------------------------------------------------

/**
 * Stored compounds, split into atoms. Same detector and same splitter the intake
 * path uses (extraction-rules.looksCompound + fact-extractor.splitCompoundFact),
 * so what the pipeline would no longer write is what this removes.
 *
 * The original is superseded by the FIRST atom rather than deleted, so the chain
 * from old to new is walkable; every atom is written through the normal cluster
 * assignment, so each lands in the cluster its own subject matter belongs to.
 * That last part is the actual F5 fix: "MettaSphere" filed under "Coastal
 * Squatch Project" was unreachable by a query about MettaSphere.
 */
async function repairCompounds(pass, { subject = null } = {}) {
  const db = getSqliteDb();
  const rules = require('./extraction-rules');
  const c = cfg();
  const maxSplits = Number.isInteger(c.maxSplitsPerPass) ? c.maxSplitsPerPass : 10;

  const rows = db.prepare(`
    SELECT * FROM cluster_members WHERE status = 'active'
    ${subject ? 'AND subject = ?' : ''}
    ORDER BY datetime(created_at) ASC
  `).all(...(subject ? [subject] : []));

  let split = 0;
  for (const row of rows) {
    if (split >= maxSplits) break;
    if (!pass.dryRun && budgetSpent(pass)) { pass.stopped = budgetSpent(pass); break; }
    if (!rules.looksCompound(row.content).compound) continue;

    // Same scope gate as the near-duplicate merge. Splitting "I value
    // transparency and precision" into two self-facts is mechanically safe for
    // the RECORD but changes the granularity of his self-view, and it would
    // reshape the very corpus the joint curation session is meant to review —
    // ten compounds became nineteen atoms in the dry run. Held back with the
    // rest of the self-fact work.
    if ((row.subject || 'user') === 'self' && cfg().selfFactSemantic !== true) continue;

    const atoms = await factExtractor().splitCompoundFact(row.content, row.subject || 'user');
    if (!atoms || atoms.length < 2) continue;

    // THE CORRECTOR NEVER MANUFACTURES AN IDENTITY ASSERTION.
    //
    // "User (Ellie) is developing a system where cron jobs can be proposed…"
    // splits into the substance plus, from the parenthetical alone, "User's name
    // is Ellie." — a brand-new name fact, written with no self-introduction
    // anywhere near it. That is precisely what the identity anchor refuses at
    // intake (the F1 rule), walked around by a phase that was never asked the
    // question, and it produced the thing F1's second half forbids: two active
    // name facts. Observed live, twice, on 2026-08-05.
    //
    // ABANDON, do not filter. Dropping the offending atom and splitting the rest
    // would supersede the original, and with it whatever that atom said — and
    // the identity classes include relationships, so the atom thrown away could
    // be "User's partner passed away on January 24th, 2025". Leaving the
    // compound whole loses nothing; it is the state the corpus is already in.
    const identityAtoms = atoms.filter(a => !!rules.identityClassOf(a));
    if (identityAtoms.length) {
      console.log(`[Corrector] abandoning split of "${trim(row.content, 60)}" — an atom would assert an identity slot: ${identityAtoms.map(a => `"${trim(a, 60)}"`).join(', ')}`);
      continue;
    }

    if (pass.dryRun) {
      pass.plan.push({
        tier: 'mechanical', action: 'split', subject: row.subject || 'user',
        targetId: row.id, targetText: row.content,
        atoms, reason: `Says more than one thing; would become ${atoms.length} separate facts.`,
        dryRun: true, applied: false
      });
      split++;
      continue;
    }

    // Write the atoms first. If this fails we have added nothing and removed
    // nothing, which is the only safe way round.
    const config = getConfig();
    const { getProviderInstance } = require('./config');
    const prov = config.models.extraction.provider;
    const model = config.models.extraction.model;
    const inst = getProviderInstance(prov, config.models.extraction.instance);
    const host = inst ? inst.host : 'http://localhost:11434';

    const created = [];
    for (const atom of atoms) {
      const res = await memoryClusters().assignToCluster(
        atom, prov, model, '', host, 'corrector-split', row.salience ?? 5,
        row.subject || 'user', row.claim_type || null,
        {
          conversationId: row.conversation_id,
          messageId: row.message_id,
          verbatimSourceText: row.verbatim_source_text,
          inputModality: row.input_modality || 'unknown',
          salienceRationale: `Split out of a compound fact by the corrector: "${trim(row.content, 100)}"`
        }
      );
      if (res && res.memberId) created.push({ id: res.memberId, text: atom, cluster: res.clusterName });
    }
    if (created.length < 2) {
      console.warn(`[Corrector] compound split produced <2 stored atoms, leaving original alone: ${row.id.slice(0, 8)}`);
      continue;
    }

    await act(pass, {
      tool: 'memory_supersede_fact',
      args: { old_id: row.id, new_id: created[0].id },
      entry: {
        tier: 'mechanical', action: 'split', subject: row.subject || 'user',
        targetId: row.id, targetText: row.content,
        survivorId: created[0].id, survivorText: created.map(a => a.text).join(' | '),
        reason: `This said more than one thing at once, so it was split into ${created.length} separate facts, each filed where it belongs: ${created.map(a => `"${a.text}" → ${a.cluster}`).join('; ')}.`,
        evidence: { atoms: created.map(a => ({ id: a.id, text: a.text, cluster: a.cluster })) }
      }
    });
    split++;
  }
  return { split };
}

// ---------------------------------------------------------------------------
// SEMANTIC — contradiction resolution by evidence dominance
// ---------------------------------------------------------------------------

/**
 * The tier the spec is really about, and F1 is the case it exists for.
 *
 * For each active fact: pull its active same-subject neighbours (with identity
 * slots pinned, so a competing name fact is always seen no matter how it ranks),
 * ask the judge whether they contradict, and when they do, resolve by evidence
 * dominance alone. A pair the evidence cannot separate is NOT resolved — it is
 * recorded as unresolved and raised for Ellie. That refusal is the feature.
 */
async function resolveContradictions(pass, { subject = 'user' } = {}) {
  const db = getSqliteDb();
  const c = cfg();
  const floor = Number.isFinite(c.contradictionFloor) ? c.contradictionFloor : 0.55;
  const maxPairs = Number.isInteger(c.maxContradictionPairsPerPass) ? c.maxContradictionPairsPerPass : 30;
  const maxSelf = Number.isInteger(c.maxSelfCorrectionsPerPass) ? c.maxSelfCorrectionsPerPass : 3;
  const rules = require('./extraction-rules');

  const rows = db.prepare(
    "SELECT * FROM cluster_members WHERE status = 'active' AND subject = ? ORDER BY salience DESC, datetime(created_at) ASC"
  ).all(subject);

  const resolved = new Set();
  const judged = new Set();   // pair keys judged in THIS pass
  let pairs = 0, applied = 0;

  // Pairs judged in EARLIER passes. Without this the budget is spent re-judging
  // the same opening pairs every time and the pass never advances past the first
  // few facts — bounded but not progressing. Dry runs read it but never write to
  // it, so a rehearsal cannot consume the real run's progress.
  const seenStmt = db.prepare('SELECT verdict FROM corrector_pair_checks WHERE pair_key = ?');
  const markStmt = db.prepare(
    'INSERT OR REPLACE INTO corrector_pair_checks (pair_key, checked_at, verdict, a_id, b_id) VALUES (?, ?, ?, ?, ?)'
  );

  for (const row of rows) {
    if (pairs >= maxPairs) break;
    if (resolved.has(row.id)) continue;
    if (!pass.dryRun && budgetSpent(pass)) { pass.stopped = budgetSpent(pass); break; }
    if (subject === 'self' && pass.selfCorrections >= maxSelf) break;
    if (!pass.dryRun && alreadyScanned(db, 'con', row)) continue;

    const slot = rules.identityClassOf(row.content);
    const { candidates } = await memoryClusters().findActiveNeighbours(row.content, {
      subject, threshold: floor, limit: 8, pinSlot: slot ? slot.klass : null
    });

    for (const cand of candidates) {
      if (cand.memberId === row.id || resolved.has(cand.memberId)) continue;
      const key = `con:${[row.id, cand.memberId].sort().join('|')}`;
      if (judged.has(key)) continue;
      if (!pass.dryRun) {
        try { if (seenStmt.get(key)) continue; } catch { /* table missing — judge it */ }
      }
      judged.add(key);
      pairs++;
      if (pairs > maxPairs) break;

      const other = db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(cand.memberId);
      if (!other || other.status !== 'active') continue;

      const { verdict } = await factExtractor().judgeContradiction(row.content, other.content);
      if (!pass.dryRun) {
        try { markStmt.run(key, new Date().toISOString(), verdict, row.id, other.id); } catch { /* non-fatal */ }
      }
      if (verdict !== 'yes') continue;

      const dom = dominance(row, other);
      if (!dom) {
        // Evidence cannot separate them. Not applied — raised.
        pass.unresolved.push({
          a: { id: row.id, text: row.content }, b: { id: other.id, text: other.content }, subject
        });
        if (!pass.dryRun) {
          ledger().record({
            passId: pass.passId, tier: 'semantic', action: 'supersede', subject,
            targetId: row.id, targetText: row.content,
            survivorId: other.id, survivorText: other.content,
            reason: 'These two contradict each other, but the evidence behind them is evenly matched, so nothing was changed. Raised for Ellie to decide.',
            evidence: { unresolved: true, a: evidenceProfile(row).detail, b: evidenceProfile(other).detail },
            reversible: false
          });
        }
        continue;
      }

      if (subject === 'self' && pass.selfCorrections >= maxSelf) break;

      const res = await act(pass, {
        tool: 'memory_supersede_fact',
        args: { old_id: dom.loser.id, new_id: dom.winner.id },
        entry: {
          tier: 'semantic', action: 'supersede', subject,
          targetId: dom.loser.id, targetText: dom.loser.content,
          survivorId: dom.winner.id, survivorText: dom.winner.content,
          reason: `These two could not both be true. "${trim(dom.winner.content, 90)}" is better evidenced than "${trim(dom.loser.content, 90)}" on ${dom.axis}, so the weaker one was retired.`,
          evidence: { deciding_axis: dom.axis, ...dom.evidence }
        }
      });

      if (res && (res.status === 'superseded' || pass.dryRun)) {
        resolved.add(dom.loser.id);
        applied++;
        if (subject === 'self') pass.selfCorrections++;
        // Decision 6: a change to what he believes about HIMSELF is told to him.
        // A change to a fact about Ellie is ledger-only, readable through his
        // inspect tools.
        if (subject === 'self' && !pass.dryRun) {
          ledger().addNotice({
            ledgerId: pass.plan[pass.plan.length - 1]?.ledgerId || null,
            content:
              `Something you believed about yourself has changed. You held "${dom.loser.content}", and you also held ` +
              `"${dom.winner.content}". They could not both be true, and the second is better supported by ${dom.axis}, ` +
              `so the first is no longer part of what you believe. Nothing was deleted — the old one is kept as history, ` +
              `and it can be put back. This is yours to sit with; you do not have to raise it with Ellie unless you want to.`
          });
        }
        if (dom.loser.id === row.id) break; // this row is retired; stop pairing it
      }
    }
    if (!pass.dryRun && !resolved.has(row.id)) markScanned(db, 'con', row);
  }
  return { pairs, applied };
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/**
 * Run one corrector pass.
 *
 * Order matters. Mechanical first, because merging duplicates and expiring
 * events shrinks and cleans the candidate set the semantic tier then reasons
 * over — resolving a contradiction against a fact that is about to be folded
 * away is wasted work and a worse decision. Reconciliation runs LAST, so it
 * cleans up after everything this pass did rather than before it.
 *
 * @param {Object} [opts]
 * @param {boolean} [opts.dryRun] - plan only, write nothing
 * @param {Object} [opts.session] - background tool session, for budget accounting
 * @returns {Promise<Object>} the pass result
 */
async function runPass(opts = {}) {
  const c = cfg();
  const pass = newPass(opts);
  const started = Date.now();
  console.log(`[Corrector] === pass ${pass.passId.slice(0, 8)}${pass.dryRun ? ' (DRY RUN — nothing will be written)' : ''} ===`);

  const result = {
    passId: pass.passId, dryRun: pass.dryRun,
    merged: 0, expired: 0, split: 0, superseded: 0, reconciled: null,
    unresolved: 0, selfCorrections: 0, refusedLocked: 0, stopped: null, plan: []
  };

  try {
    // --- mechanical: both subjects. These repair the record, not a view. ---
    const m = await mergeNearDuplicates(pass, {});
    result.merged = m.merged;

    const e = await expireDatedEvents(pass, {});
    result.expired = e.expired;

    const s = await repairCompounds(pass, {});
    result.split = s.split;

    // --- semantic: user facts always; self facts only behind the gate. ---
    const userSem = await resolveContradictions(pass, { subject: 'user' });
    result.superseded = userSem.applied;

    // World facts get the same semantic tier as user facts, with no extra gate:
    // they are knowledge about external things, not a view of anyone, so there
    // is nothing here that a joint curation session needs to be present for.
    const worldSem = await resolveContradictions(pass, { subject: 'world' });
    result.superseded += worldSem.applied;

    if (c.selfFactSemantic === true) {
      const selfSem = await resolveContradictions(pass, { subject: 'self' });
      result.superseded += selfSem.applied;
    } else {
      console.log('[Corrector] self-fact semantic corrections are OFF (corrector.selfFactSemantic) — reserved for the joint curation session');
    }

    // --- reconcile last, cleaning up after this pass's own writes ---
    result.reconciled = await reconcileByActing(pass);

  } catch (err) {
    console.error('[Corrector] pass error:', err.message);
    result.error = err.message;
  }

  result.unresolved = pass.unresolved.length;
  result.selfCorrections = pass.selfCorrections;
  result.refusedLocked = pass.plan.filter(p => p.refusedLocked).length;
  result.stopped = pass.stopped;
  result.plan = pass.plan;
  result.unresolvedPairs = pass.unresolved;
  result.durationMs = Date.now() - started;
  result.writes = pass.writes;

  if (!pass.dryRun) {
    // Stamped whether or not anything changed — a pass that found nothing to do
    // is still a pass, and it is what the cadence is measured against.
    recordPassRan(result);
    const summary = ledger().summarize(pass.passId);
    result.ledger = summary;
    const line =
      `Corrector pass: ${result.merged} duplicate(s) folded, ${result.expired} event(s) moved out of memory, ` +
      `${result.split} compound(s) split, ${result.superseded} contradiction(s) resolved, ` +
      `${result.unresolved} left for Ellie${result.refusedLocked ? `, ${result.refusedLocked} refused by the identity lock` : ''}` +
      `${result.stopped ? ` — stopped early: ${result.stopped}` : ''}.`;
    try { factExtractor().appendToOpsLog(line, path.join(memoryDir(), 'ops')); } catch { /* best effort */ }
    console.log(`[Corrector] ${line}`);
  }

  console.log(`[Corrector] === pass complete in ${(result.durationMs / 1000).toFixed(1)}s ===`);
  return result;
}

module.exports = {
  runPass, lastPassAt, evidenceProfile, dominance,
  mergeNearDuplicates, expireDatedEvents, repairCompounds, resolveContradictions, reconcileByActing
};
