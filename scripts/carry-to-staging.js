#!/usr/bin/env node
/**
 * The MERGE half of the staging gate — carry live facts the replay did not
 * rebuild into the staging corpus.
 *
 * The first gate answered "can the corpus be rebuilt from source?" and the
 * answer was mostly: 111 of 256 active user facts came back, 145 did not. Ellie's
 * decision on that report was MERGE, not replace — the rebuild is the trunk, and
 * what it could not re-derive is carried across rather than lost.
 *
 * This runs POINTED AT STAGING (SNH_DATA_DIR), like the replay and the gate
 * report, and opens live READONLY alongside. Same reason as
 * scripts/replay-to-staging.js: redirecting the whole process needs no special
 * case in fact-store, memory-clusters or the extractor, and there is no call site
 * that can forget a flag and write to the live corpus. It refuses to start if the
 * staging path resolves to live.
 *
 * WHAT "MISSING" MEANS — recomputed here, not read from the gate report. A live
 * active user fact is missing if staging holds no active user fact within 0.80
 * cosine of it. Identical definition to the report's coverage table, so the
 * number is reproducible rather than transcribed — and it needs to be, because it
 * MOVES. The report counted 145 on 2026-08-05; the same test on 2026-08-06 gave
 * 154, because live had gained facts overnight from its own corrector and the
 * daily-log archiver. A merge planned against a transcribed number would have
 * silently skipped every fact learned since the number was written down.
 *
 * THE THREE PILES. Every missing fact lands in exactly one, and the rule that put
 * it there is recorded beside it:
 *
 *   AUTO-CARRY      Its justification already exists somewhere a person can read.
 *                   A corrector-split fact whose parent correction is in the
 *                   ledger; an identity-adjacent fact (a name, pronouns, or a
 *                   core relationship — this is where the grief and partner facts
 *                   live, and Ellie has said they carry); anything at salience 8
 *                   or above. Carried without asking.
 *
 *   RECOMMEND-DROP  Its content is not being lost. Either it reads as an event
 *                   and the day's log the replay wrote already holds it, or
 *                   staging has a near-equivalent in the 0.70–0.80 band that
 *                   covers it. Both are checked, not assumed: the log claim is a
 *                   keyword-coverage test against the actual replayed logs, the
 *                   coverage claim is a model call.
 *
 *                   TWO judges are asked in that band, because "is it covered?"
 *                   is two questions. judgeSameAssertion answers "is this a
 *                   rewording?"; judgeSubsumption answers "does staging's version
 *                   already contain everything this one says?". Either one is
 *                   enough to drop. The second matters more than it looks:
 *                   measured over the band, the repeat judge said DIFFERENT to 39
 *                   of 43 pairs — correctly, because it is built to refuse when
 *                   unsure so intake never eats a fact — and most of those 39 are
 *                   cases where the LIVE fact is the richer one. That is not a
 *                   reason to drop it; it is the strongest reason to carry it.
 *
 *   ELLIE-DECIDES   Everything the rules cannot place. One line each, with a
 *                   lean. This pile is meant to be SMALL and HONEST — a rule that
 *                   guesses to shrink it is worse than a longer list.
 *
 * THE CARRY ITSELF goes through the same funnel every other write uses —
 * memoryClusters.assignToCluster, which is what applyExtraction and the
 * corrector's splitter call. So a carried fact is deduped at the write, embedded,
 * and placed in the cluster its subject matter belongs to, exactly like one the
 * pipeline produced. Nothing here inserts a row by hand.
 *
 * What is preserved, and how:
 *   - the ORIGINAL learned date. assignToCluster stamps created_at with now, so
 *     it is put back afterwards. A fact carried today did not become true today,
 *     and evidenceProfile() reads recency off this column to decide which of two
 *     contradicting facts wins — a carried fact stamped 2026-08-06 would beat
 *     everything the replay rebuilt, on nothing but the carry.
 *   - the original provenance (conversation, message, verbatim text, modality),
 *     copied across. The corrector's dominance rules are written in terms of
 *     these, so a carried fact with them blanked is a fact the semantic tier
 *     cannot adjudicate.
 *   - every corroboration, re-pointed at the new row. A restatement is evidence
 *     ABOUT the fact and travels with it.
 *   - the carry itself, as a `carry` entry in staging's corrections ledger,
 *     naming the live row it came from and the rule that carried it.
 *
 * `source` becomes 'carried_from_live' — the producing path, same slot that holds
 * 'fact-extraction' and 'corrector-split'. The original producer is not lost: it
 * is in the ledger entry's evidence, alongside the review date.
 *
 * Usage:
 *   node scripts/carry-to-staging.js --plan
 *       Classify and write docs/carry-review.md + data-staging/carry-plan.json.
 *       Writes NOTHING to either corpus.
 *
 *   node scripts/carry-to-staging.js --rewrite
 *       Re-render docs/carry-review.md from the frozen plan. No judge calls, no
 *       reclassification — editing a sentence in the report cannot move a fact
 *       between piles.
 *
 *   node scripts/carry-to-staging.js --apply auto
 *       Carry the AUTO-CARRY pile from the frozen plan.
 *
 *   node scripts/carry-to-staging.js --apply decided
 *       Carry the ELLIE-DECIDES entries she has marked CARRY in the review doc.
 *       Reads her marks back out of docs/carry-review.md.
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

const LIVE_DATA = path.join(ROOT, 'data');
const STAGING_DATA = process.env.SNH_STAGING_DIR || path.join(ROOT, 'data-staging');

// Before db/database.js is required — the paths resolve at module load.
process.env.SNH_DATA_DIR = STAGING_DATA;

const REVIEW_DOC = path.join(ROOT, 'docs', 'carry-review.md');
const PLAN_JSON = path.join(STAGING_DATA, 'carry-plan.json');

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const MODE = args.includes('--plan') ? 'plan'
  : args.includes('--rewrite') ? 'rewrite'
    : args.includes('--withdraw') ? 'withdraw'
      : (args.includes('--apply') ? `apply:${argVal('--apply', 'auto')}` : null);

const TWIN_FLOOR = 0.80;        // the gate report's definition of "represented"
const BAND_LO = 0.70;           // near-equivalent band, lower edge
const HIGH_SALIENCE = 8;        // Ellie's floor for automatic carry

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

// Words distinctive enough that finding them in a log line means something.
// Same filter the gate report uses, so "covered by the log" means there what it
// means here.
const keyWords = (text) => String(text).toLowerCase()
  .replace(/[^a-z0-9 ]/g, ' ')
  .split(/\s+/)
  .filter(w => w.length > 4 && !['user', 'their', 'there', 'about', 'which', 'would', 'these', 'those'].includes(w));

function abortIfLive() {
  if (path.resolve(STAGING_DATA) === path.resolve(LIVE_DATA)) {
    console.error('ABORT: staging dir resolves to the live data dir.');
    process.exit(2);
  }
  if (!fs.existsSync(path.join(STAGING_DATA, 'chat.db'))) {
    console.error(`ABORT: no staging store at ${STAGING_DATA} — run scripts/replay-to-staging.js first.`);
    process.exit(2);
  }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/**
 * Identity-adjacent, deliberately WIDER than extraction-rules.identityClassOf.
 *
 * That classifier is anchored to the sentence subject because it guards a
 * refusal — it has to be narrow or it eats facts. This is the opposite job:
 * deciding what carries without asking, where a false positive costs one extra
 * carried fact and a false negative loses something about who Ellie is. "User is
 * grieving the loss of their partner." asserts no identity SLOT and would fail
 * the narrow test entirely.
 *
 * So: the slot classifier, OR a locked row, OR a core relationship term
 * anywhere in the sentence.
 */
function identityAdjacent(row, rules) {
  if (row.locked) return 'locked row';
  const klass = rules.identityClassOf(row.content);
  if (klass) return `asserts the ${klass.klass} slot`;
  const rel = String(row.content).match(
    new RegExp(`\\b(${CARRY_RELATIONSHIP_TERMS.join('|')})\\b`, 'i')
  );
  if (rel) return `names a core relationship ("${rel[1].toLowerCase()}")`;
  return null;
}

/**
 * The relationship vocabulary this script carries on, which is NOT
 * extraction-rules.RELATIONSHIP_TERMS and must not be conflated with it.
 *
 * That list is load-bearing at intake: the identity anchor refuses on it and the
 * corrector abandons a split on it. Widening it there changes what the pipeline
 * accepts, and this is not the place to make that decision.
 *
 * Here the list only decides what gets carried without being asked about, so it
 * takes the step-relations too. `\bfather\b` does not match "stepfather", which
 * is how "User is not a fan of their stepfather" was about to be leant DROP at
 * salience 5 — a real fact about a person she lives with, filed as residue.
 */
const CARRY_RELATIONSHIP_TERMS = [
  'wife', 'husband', 'spouse', 'partner', 'fiancé', 'fiancee', 'fiancée',
  'mother', 'mom', 'father', 'dad', 'son', 'daughter', 'brother', 'sister',
  'grandmother', 'grandfather', 'grandma', 'grandpa',
  'stepfather', 'stepmother', 'stepdad', 'stepmom', 'stepson', 'stepdaughter',
  'stepbrother', 'stepsister', 'parents', 'siblings'
];

/**
 * Is this fact's parent correction in the ledger?
 *
 * A corrector split records its atoms in the entry's evidence, by id and text,
 * and supersedes the compound with the first of them. So a split-produced fact
 * has a readable justification — "this said more than one thing at once" — sitting
 * in a ledger row that carried over into staging untouched. That is what makes it
 * safe to carry without asking: the reason it exists is already written down.
 */
function parentCorrection(row, ledgerRows) {
  for (const e of ledgerRows) {
    if (e.survivor_id === row.id) return e;
    if (!e.evidence) continue;
    // Cheap containment test before the parse — the evidence column holds the
    // atom ids verbatim.
    if (!e.evidence.includes(row.id)) continue;
    try {
      const ev = JSON.parse(e.evidence);
      if (Array.isArray(ev.atoms) && ev.atoms.some(a => a.id === row.id)) return e;
    } catch { /* malformed evidence is not a match */ }
  }
  return null;
}

/**
 * Which of two facts contains the other — asked BOTH WAYS ROUND, and believed
 * only when the two answers agree.
 *
 * judgeSubsumption answers with a letter naming one of its two arguments, and
 * on this corpus that letter is positionally unstable: measured over the band,
 * every one of its three "B contains A" verdicts came back with reasoning that
 * described A containing B. "User owns a 2026 Tundra" was reported as containing
 * "User traded the GR86 and the 2015 Tundra for a new 2026 Tundra", which it
 * plainly does not. Trusting that would have dropped the trade.
 *
 * Swapping the arguments turns a positional bias into a disagreement, and a
 * disagreement is resolved the way everything else here is resolved: NEITHER.
 * The judge's own prompt sets that default ("if you are unsure, answer NEITHER")
 * for the same reason — a fact folded away wrongly is worse than one kept.
 *
 * @returns {Promise<{relation: 'staging-contains-live'|'live-contains-staging'|'neither', reasoning: string}>}
 */
async function subsumption(factExtractor, liveText, stagingText) {
  const fwd = await factExtractor.judgeSubsumption(liveText, stagingText);   // A=live,    B=staging
  const rev = await factExtractor.judgeSubsumption(stagingText, liveText);   // A=staging, B=live

  const fwdSays = fwd.relation === 'a-contains-b' ? 'live' : fwd.relation === 'b-contains-a' ? 'staging' : null;
  const revSays = rev.relation === 'a-contains-b' ? 'staging' : rev.relation === 'b-contains-a' ? 'live' : null;

  if (fwdSays && fwdSays === revSays) {
    return {
      relation: fwdSays === 'staging' ? 'staging-contains-live' : 'live-contains-staging',
      reasoning: trunc(fwd.reasoning, 200)
    };
  }
  return {
    relation: 'neither',
    reasoning: fwdSays || revSays
      ? `the judge did not agree with itself when the two were swapped (${fwdSays || 'neither'} / ${revSays || 'neither'}), so neither is treated as containing the other`
      : trunc(fwd.reasoning, 200)
  };
}

/** How much of a fact's substance appears in the day's logs the replay wrote. */
function logCoverage(content, dailyText) {
  const kw = keyWords(content);
  if (!kw.length) return { ratio: 0, hits: 0, total: 0 };
  const hits = kw.filter(w => dailyText.includes(w)).length;
  return { ratio: hits / kw.length, hits, total: kw.length };
}

/**
 * The lean offered on an ELLIE-DECIDES entry.
 *
 * Signals a person can check, not a score. Whichever fires first is what is
 * shown, and it is shown with its reason attached — a lean you can argue with is
 * worth something; a number is not.
 */
function lean(f) {
  if (f.notEvent) {
    return { call: 'carry', why: `it looks like an event on the marker alone, but the strip-the-timestamp judge — the corrector's own expiry gate — says something lasting remains when the time reference is removed. ${trunc(f.strip.reasoning, 110)}` };
  }
  if (f.event.isEvent) {
    return { call: 'drop', why: `needs a time reference to be true ("${f.event.marker}"), so intake would route it to the day's log now rather than store it — but the replayed logs do not hold it (${f.log.hits}/${f.log.total} of its distinctive words), so dropping does lose the wording` };
  }
  if (f.judge && f.judge.relation === 'live-contains-staging') {
    // The corrector's own subset rule, applied ahead of time: when a fact and a
    // shorter version of itself are both held, the DETAIL survives. Staging is
    // holding the impoverished copy.
    return { call: 'carry', why: `staging's nearest ("${trunc(f.near.content, 55)}") is a strictly poorer version of this — the subsumption judge says this one contains it. Carry it and the corrector folds the short one in on its next pass, which is exactly what its subset rule is for` };
  }
  if (f.corroborations > 0) {
    return { call: 'carry', why: `said more than once — ${f.corroborations} corroboration(s) recorded against it in live` };
  }
  if (f.near && f.near.similarity >= BAND_LO) {
    return { call: 'carry', why: `staging's closest is only ${f.near.similarity.toFixed(3)}, and both judges say it does not cover this — nothing already there holds it` };
  }
  if (f.salience >= 6) {
    return { call: 'carry', why: `salience ${f.salience} — scored as mattering when it was learned, and staging has nothing within ${TWIN_FLOOR}` };
  }
  if (f.salience <= 4) {
    return { call: 'drop', why: `salience ${f.salience}, no corroboration, and nothing in staging within ${TWIN_FLOOR} — scored as minor when it was learned and never restated since` };
  }
  // Salience 5 is the DEFAULT the scorer assigns, the middle of the scale, and
  // most of it means "nobody decided". There is no signal here, and inventing one
  // is worse than saying so: an earlier draft of this leant DROP on salience 5
  // alone, which put "User is not a fan of their stepfather" and "User has 26
  // years of experience cleaning up vendor software" in the discard column on
  // the strength of a number that was never a judgement in the first place.
  return { call: 'none', why: 'no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule' };
}

/**
 * When was the staging store seeded from live?
 *
 * The replay's own bookkeeping answers it: the stats file is written when the
 * run finishes and records how long the run took, so the seed is the difference.
 * This matters because a fact created in live AFTER that moment was never a
 * candidate for rebuild — the replay could not have produced it and did not fail
 * to. Reporting those beside genuine rebuild misses would overstate the gap.
 */
function seedTime() {
  try {
    const p = path.join(STAGING_DATA, 'replay-stats.json');
    const stats = JSON.parse(fs.readFileSync(p, 'utf8'));
    const finished = fs.statSync(p).mtimeMs;
    return new Date(finished - (stats.durationMs || 0)).toISOString();
  } catch { return null; }
}

async function classify() {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  if (path.resolve(db.getDataDir()) !== path.resolve(STAGING_DATA)) {
    console.error(`ABORT: data dir is ${db.getDataDir()}, expected ${STAGING_DATA}`);
    process.exit(2);
  }

  const Database = require('better-sqlite3');
  const live = new Database(path.join(LIVE_DATA, 'chat.db'), { readonly: true });
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const rules = require(path.join(ROOT, 'db/extraction-rules'));

  const liveFacts = live.prepare(`
    SELECT m.*, c.name AS cluster_name
    FROM cluster_members m
    LEFT JOIN memory_clusters c ON c.id = m.cluster_id
    WHERE m.status = 'active' AND COALESCE(m.subject,'user') = 'user'
    ORDER BY m.salience DESC, datetime(m.created_at) ASC
  `).all();

  const ledgerRows = live.prepare(
    "SELECT id, action, target_id, target_text, survivor_id, reason, evidence FROM corrections_ledger WHERE action = 'split'"
  ).all();

  const corrCount = live.prepare('SELECT COUNT(*) n FROM fact_corroborations WHERE member_id = ?');

  let dailyText = '';
  try {
    const dir = path.join(STAGING_DATA, 'memory', 'daily');
    for (const f of fs.readdirSync(dir)) dailyText += fs.readFileSync(path.join(dir, f), 'utf8').toLowerCase();
  } catch { /* no logs written */ }

  const seededAt = seedTime();
  console.log(`[Carry] ${liveFacts.length} active live user fact(s) — finding which staging does not hold`);
  console.log(`[Carry] staging was seeded from live at ${seededAt || '(unknown)'}`);

  // Facts a previous run already brought across. Re-planning after the corrector
  // has run finds some of them "missing" again — not because the carry failed but
  // because the corrector then retired or folded what was carried. Re-proposing
  // those would read as a carry that did not happen, so they are separated out
  // and reported as what they are: carried, and then acted on.
  const staging = db.getSqliteDb();
  const carriedAlready = new Map(
    staging.prepare("SELECT target_id, survivor_id FROM corrections_ledger WHERE action = 'carry'").all()
      .map(r => [r.target_id, r.survivor_id])
  );
  const stagingRow = staging.prepare('SELECT status, inactive_reason, content FROM cluster_members WHERE id = ?');

  const missing = [];
  const carriedThenChanged = [];
  for (const row of liveFacts) {
    // Neighbours down to the band's lower edge in one call: above TWIN_FLOOR
    // means represented, between BAND_LO and TWIN_FLOOR is the near-equivalent
    // question, below is nothing.
    const { candidates } = await memoryClusters.findActiveNeighbours(row.content, {
      subject: 'user', threshold: BAND_LO, limit: 3, includeVerbatim: true
    });
    const best = candidates[0] || null;
    if (best && best.similarity >= TWIN_FLOOR) continue;   // represented

    if (carriedAlready.has(row.id)) {
      const landed = stagingRow.get(carriedAlready.get(row.id));
      carriedThenChanged.push({
        id: row.id, content: row.content, salience: row.salience ?? 5,
        stagingId: carriedAlready.get(row.id),
        stagingStatus: landed ? landed.status : 'gone',
        stagingReason: landed ? landed.inactive_reason : null
      });
      continue;
    }

    missing.push({
      id: row.id,
      content: row.content,
      salience: row.salience ?? 5,
      source: row.source || '(null)',
      cluster: row.cluster_name || null,
      claimType: row.claim_type || null,
      locked: !!row.locked,
      createdAt: row.created_at,
      conversationId: row.conversation_id,
      messageId: row.message_id,
      verbatimSourceText: row.verbatim_source_text,
      inputModality: row.input_modality,
      salienceRationale: row.salience_rationale,
      corroborations: corrCount.get(row.id).n,
      event: rules.eventMarker(row.content),
      log: logCoverage(row.content, dailyText),
      near: best ? { id: best.memberId, content: best.content, similarity: best.similarity } : null,
      // Never a candidate for rebuild — it did not exist when staging was seeded.
      postSnapshot: !!(seededAt && row.created_at && row.created_at > seededAt)
    });
  }
  const post = missing.filter(f => f.postSnapshot).length;
  console.log(`[Carry] ${missing.length} missing (${post} of them created in live AFTER the staging seed)`);
  if (carriedThenChanged.length) {
    console.log(`[Carry] ${carriedThenChanged.length} more were carried by an earlier run and then changed by the corrector — reported, not re-proposed`);
  }

  // ---- pile assignment ----------------------------------------------------
  const piles = { auto: [], drop: [], decide: [] };
  let judged = 0;

  for (const f of missing) {
    // AUTO-CARRY, in the order Ellie named the rules.
    const parent = f.source === 'corrector-split' ? parentCorrection({ id: f.id }, ledgerRows) : null;
    if (parent) {
      f.rule = 'parent-correction-in-ledger';
      f.ruleDetail = `split out by the corrector; the correction that produced it is ledger ${parent.id.slice(0, 8)} — "${trunc(parent.reason, 140)}"`;
      piles.auto.push(f); continue;
    }
    const ident = identityAdjacent({ content: f.content, locked: f.locked }, rules);
    if (ident) {
      f.rule = 'identity-adjacent';
      f.ruleDetail = ident;
      piles.auto.push(f); continue;
    }
    if (f.salience >= HIGH_SALIENCE) {
      f.rule = 'high-salience';
      f.ruleDetail = `salience ${f.salience} — at or above the floor Ellie set for automatic carry`;
      piles.auto.push(f); continue;
    }

    // RECOMMEND-DROP. Every test has to actually pass; none is assumed.
    //
    // A marker alone is NOT enough to call something an event, and the corrector
    // already knows it: expireDatedEvents puts every marker-matched row to
    // judgeStripTheTimestamp, because "her partner died on 24 January 2025"
    // carries a date and is permanent. The same gate is used here, for the same
    // reason and with the same judge. Measured without it, this rule dropped
    // "User (Ellie) has blue eyes and her favorite color is green (last updated
    // 2026-07-27)" — where the timestamp is bookkeeping about the RECORD, not a
    // qualifier on the claim, and the claim contradicts something staging holds.
    if (f.event.isEvent && f.log.ratio >= 0.6) {
      const strip = await factExtractor.judgeStripTheTimestamp(f.content);
      judged++;
      f.strip = { isEvent: strip.isEvent, reasoning: trunc(strip.reasoning, 200) };
      if (strip.isEvent) {
        f.rule = 'covered-by-log';
        f.ruleDetail = `reads as an event ("${f.event.marker}", ${f.event.kind}), the strip-the-timestamp judge agrees nothing lasting remains once the time reference is removed, and the replayed day's logs hold ${f.log.hits}/${f.log.total} of its distinctive words`;
        piles.drop.push(f); continue;
      }
      // Marker-matched but LASTING. Not an event, so not droppable as one — and
      // the marker is still worth showing her.
      f.notEvent = `carries "${f.event.marker}" but the strip-the-timestamp judge calls it lasting: ${trunc(strip.reasoning, 120)}`;
    }
    if (f.near && f.near.similarity >= BAND_LO && f.near.similarity < TWIN_FLOOR) {
      // The rules picked the band; the models answer the two questions inside it.
      const same = await factExtractor.judgeSameAssertion(f.content, f.near.content);
      judged++;
      if (same.same) {
        f.judge = { relation: 'same-assertion', reasoning: trunc(same.reasoning, 200) };
        f.rule = 'staging-says-it';
        f.ruleDetail = `staging holds "${trunc(f.near.content, 90)}" at ${f.near.similarity.toFixed(3)}, and the repeat judge calls it the same assertion — ${trunc(same.reasoning, 120)}`;
        piles.drop.push(f); continue;
      }

      // Not a rewording. The other half of "is it covered": does staging's
      // version already contain everything this one says?
      const sub = await subsumption(factExtractor, f.content, f.near.content);
      judged += 2;
      f.judge = { relation: sub.relation, reasoning: sub.reasoning };
      if (sub.relation === 'staging-contains-live') {
        f.rule = 'staging-says-it';
        f.ruleDetail = `staging holds "${trunc(f.near.content, 90)}" at ${f.near.similarity.toFixed(3)}, and the subsumption judge — asked both ways round and agreeing with itself — says it already contains everything this one asserts: ${trunc(sub.reasoning, 120)}`;
        piles.drop.push(f); continue;
      }
    }

    f.lean = lean(f);
    piles.decide.push(f);
  }

  live.close();
  console.log(`[Carry] piles — auto ${piles.auto.length}, drop ${piles.drop.length}, decide ${piles.decide.length} (${judged} judge call(s))`);
  return {
    missing, piles, seededAt, liveTotal: liveFacts.length, carriedThenChanged,
    carriedTotal: staging.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'carry'").get().n
  };
}

// ---------------------------------------------------------------------------
// The review document
// ---------------------------------------------------------------------------

function writeReview({ missing, piles, seededAt, liveTotal, carriedThenChanged = [], carriedTotal = 0 }) {
  const today = new Date().toISOString().slice(0, 10);
  const post = missing.filter(f => f.postSnapshot);
  const L = [];
  const p = (s = '') => L.push(s);

  p(`# Carry review — the ${missing.length} live facts staging does not hold`);
  p('');
  p(`Generated ${today} by \`scripts/carry-to-staging.js --plan\`, against the staging`);
  p('corpus at `data-staging/`. Nothing was written to either store by the run that');
  p('produced this file.');
  p('');
  p('A live fact counts as **missing** if staging holds no active user fact within');
  p(`${TWIN_FLOOR} cosine of it — the same test the gate report's coverage table uses, re-run`);
  p('here rather than transcribed.');
  p('');
  p(`## It is ${missing.length}, not 145`);
  p('');
  p('The gate report counted 145 on 2026-08-05. Re-running the same test today gives');
  p(`${missing.length}, because **live has not stood still**: it now holds ${liveTotal} active user facts`);
  p('against the 256 the report measured. The corrector ran overnight and the daily-log');
  p('archiver added facts this morning.');
  p('');
  if (seededAt) {
    p(`Staging was seeded from live at \`${seededAt}\`. **${post.length}** of the ${missing.length} were created`);
    p('in live *after* that moment, so the replay never had them to rebuild — they are');
    p('missing from staging the way a letter posted yesterday is missing from a box');
    p('emptied the day before. They are marked `new since the seed` below. Counting them');
    p('as rebuild failures would overstate what the pipeline got wrong.');
    p('');
  }
  p(piles.auto.length
    ? `- **AUTO-CARRY** — ${piles.auto.length}. Carried without asking. Runs as soon as this doc exists.`
    : `- **AUTO-CARRY** — 0. Empty because it has already run: ${carriedTotal} facts are in staging`
      + ' already, listed under the rule that carried each one below.');
  p(`- **RECOMMEND-DROP** — ${piles.drop.length}. The content survives somewhere else, and that was checked.`);
  p(`- **ELLIE-DECIDES** — ${piles.decide.length}. Waiting on your mark.`);
  if (carriedThenChanged.length) {
    p(`- **already carried, then changed** — ${carriedThenChanged.length}. Listed at the end, not re-proposed.`);
  }
  p('');
  p('---');
  p('');

  // ---- AUTO-CARRY ----
  p(`## AUTO-CARRY — ${piles.auto.length} still to carry`);
  p('');
  if (!piles.auto.length) {
    p('**This pile is empty because it has already run.** The facts it held are in');
    p('staging now, written through the funnel with their original learned dates,');
    p('provenance and corroborations, and each one has a `carry` row in the staging');
    p('corrections ledger naming the live fact it came from and the rule that carried');
    p('it. `SELECT * FROM corrections_ledger WHERE action = \'carry\'` is the record.');
    p('');
    p('Re-planning after the carry finds these facts represented, so they are no longer');
    p('proposed. The rules that put them there were:');
    p('');
  } else {
    p('Three rules put a fact here, and the one that fired is named beside it:');
  }
  p('');
  p('1. **parent-correction-in-ledger** — the corrector split a compound into this,');
  p('   and the correction that did it is in the ledger. Its justification already');
  p('   exists in a form a person can read.');
  p('2. **identity-adjacent** — it asserts a name, pronouns, or a core relationship,');
  p('   or the row is identity-locked. The grief and partner facts are here.');
  p(`3. **high-salience** — salience ${HIGH_SALIENCE} or above.`);
  p('');
  const byRule = (pile, rule) => pile.filter(f => f.rule === rule);
  for (const [rule, title] of [
    ['parent-correction-in-ledger', 'Parent correction is in the ledger'],
    ['identity-adjacent', 'Identity-adjacent'],
    ['high-salience', `Salience ${HIGH_SALIENCE}+`]
  ]) {
    const rows = byRule(piles.auto, rule);
    if (!rows.length) continue;
    p(`### ${title} (${rows.length})`);
    p('');
    for (const f of rows) {
      p(`- **"${f.content}"**`);
      p(`  <br>salience ${f.salience} · ${f.source} · learned ${String(f.createdAt).slice(0, 10)}` +
        `${f.corroborations ? ` · ${f.corroborations} corroboration(s)` : ''}`);
      p(`  <br>*${f.ruleDetail}*`);
    }
    p('');
  }

  // ---- RECOMMEND-DROP ----
  p('---');
  p('');
  p(`## RECOMMEND-DROP — ${piles.drop.length}`);
  p('');
  p('Nothing in this pile is deleted from live. It is the set the merge would not');
  p('carry into staging, because the content is already in the rebuilt corpus by');
  p('another route. Two rules can put a fact here, and both are verified rather than');
  p('asserted:');
  p('');
  p('- **covered-by-log** — the fact carries a time marker; the strip-the-timestamp');
  p("  judge agrees nothing lasting remains once the time reference is removed; and the");
  p("  day's logs the replay actually wrote hold at least 60% of its distinctive words.");
  p('- **staging-says-it** — staging has a near-equivalent between');
  p(`  ${BAND_LO} and ${TWIN_FLOOR}, and either the repeat judge calls it the same assertion or the`);
  p('  subsumption judge says staging already contains everything it asserts.');
  p('');

  const lastingHeld = missing.filter(x => x.notEvent);
  const swapHeld = missing.filter(x => x.judge && /did not agree with itself/.test(x.judge.reasoning || ''));

  if (!piles.drop.length) {
    p('### It came out empty, and that is a finding');
    p('');
    p(`${lastingHeld.length + swapHeld.length} facts reached one of the two rules and none survived it. Both rules were`);
    p("built with a second gate, and both gates fired — this is what they caught:");
    p('');
    p(`**${lastingHeld.length} were marker-matched but lasting.** A time marker alone does not make a`);
    p('sentence an event; the corrector already knows this, which is why');
    p('`expireDatedEvents` puts every marker-matched row to `judgeStripTheTimestamp`');
    p(`before retiring it. The same judge is asked here, and it called all ${lastingHeld.length} LASTING:`);
    p('');
    for (const f of lastingHeld) {
      p(`- **"${f.content}"** — *${trunc(f.strip.reasoning, 150)}*`);
    }
    p('');
    p('The first is the one that shows why the gate is needed. Its timestamp is');
    p('bookkeeping about the *record* ("last updated 2026-07-27"), not a qualifier on the');
    p('*claim* — eye colour is not an event. Worse, it says her favourite colour is green');
    p('and staging holds `User\'s favorite color is blue` at salience 10. Dropping it as');
    p('an event would have silently resolved a live contradiction by discarding one side,');
    p('which is exactly the thing the corrector is built to refuse to do.');
    p('');
    p(`**${swapHeld.length} were subsumption verdicts the judge would not repeat.** \`judgeSubsumption\``);
    p('answers with a letter naming one of its two arguments, and on this corpus that');
    p('letter is positionally unstable. The run that found this asked once, in one');
    p('direction, and got three "staging contains live" verdicts — all three with');
    p('reasoning that described the opposite. It reported `User owns a 2026 Tundra` as');
    p('containing `User traded the GR86 and the 2015 Tundra for a new 2026 Tundra`, which');
    p('it plainly does not, and on that verdict the trade would have been dropped.');
    p('');
    p('So it is asked both ways round now, and believed only when the two answers agree.');
    p('Swapping turns the positional bias into a disagreement, and a disagreement resolves');
    p('to NEITHER — the default its own prompt sets, for the same reason a failed check');
    p(`never folds a fact away. In this run ${swapHeld.length} pair(s) disagreed under the swap, and no`);
    p('subsumption verdict survived to produce a drop. They are all in ELLIE-DECIDES below.');
    p('');
    p('So: **no fact in the missing set is safely droppable by rule.** Everything not in');
    p('AUTO-CARRY is a judgment call, which is why the pile below is longer than either');
    p(`of us wanted. I would rather hand you ${piles.decide.length} real decisions than ${piles.decide.length - (lastingHeld.length + swapHeld.length)} real ones and`);
    p(`${lastingHeld.length + swapHeld.length} quiet mistakes.`);
    p('');
  } else {
    p('Mark any of these `KEEP` to have it carried after all.');
    p('');
    for (const [rule, title] of [
      ['covered-by-log', "Covered by the replayed day's log"],
      ['staging-says-it', 'Staging already says it']
    ]) {
      const rows = byRule(piles.drop, rule);
      if (!rows.length) continue;
      p(`### ${title} (${rows.length})`);
      p('');
      for (const f of rows) {
        p(`- [ ] KEEP \`${f.id.slice(0, 8)}\` — **"${f.content}"**`);
        p(`  <br>salience ${f.salience} · ${f.source}`);
        p(`  <br>*${f.ruleDetail}*`);
      }
      p('');
    }
  }

  // ---- ELLIE-DECIDES ----
  p('---');
  p('');
  p('## ELLIE-DECIDES');
  p('');
  p('Put a mark in the box on each row, then run:');
  p('');
  p('```');
  p('node scripts/carry-to-staging.js --apply decided');
  p('```');
  p('');
  p('Any of these is read the same way — mark it however is quickest:');
  p('');
  p('| you write | it means |');
  p('|---|---|');
  p('| `[x]` | carry |');
  p('| `[CARRY]` | carry |');
  p('| `[ ] CARRY` | carry |');
  p('| `[DROP]` | drop |');
  p('| `[ ]` | no decision — left alone |');
  p('');
  p('Case does not matter. Anything unmarked is left alone: not carried, and not');
  p('lost — the live corpus is untouched by all of this, so an unmarked fact is still');
  p('sitting there to decide on later. If a mark cannot be read, the run says so');
  p('rather than skipping it quietly.');
  p('');
  p('**Split by the lean, not by producer**, so you can agree with a whole block at');
  p('once rather than adjudicate every line. The leans are reasons, not scores —');
  p('each one names the signal it came from so you can disagree with it cheaply.');
  p('');

  // ---- the thing that came out of building this pile ----------------------
  const archived = piles.decide.filter(f => f.source === 'daily-log-archive');
  if (archived.length) {
    const TELL = /\b(non-judgmental|sounding board|questioning tone|as an interface|presence|inquiry|probe|probing|validating|renaming|re-gendering|locked|springboard)\b/i;
    const illustrative = archived.filter(f => TELL.test(f.content));
    p('> ### ⚠ Read this before you mark the `daily-log-archive` rows');
    p('>');
    p(`> ${archived.length} of the ${piles.decide.length} below came from the daily-log archiver, and about ${illustrative.length} of those`);
    p('> do not read as facts about you at all. They read as Aurelius describing');
    p('> himself, written in the third person and filed as your preferences:');
    p('>');
    for (const f of illustrative.slice(0, 6)) p(`> - "${f.content}"`);
    p('>');
    p('> "aims to be a steady, non-judgmental presence that respects boundaries and');
    p('> cognitive load" is a description of an assistant. "responds to attempts at');
    p('> renaming or re-gendering with a highly structured, defensive protocol to protect');
    p('> core facts" is a description of the identity lock. These are self-facts wearing');
    p('> the third person, which is the 2026-07-27 defect `db/memory-write.js` was built');
    p('> to refuse — reached by a path that was never asked the question.');
    p('>');
    p('> **This is not a staging problem.** These rows are in the LIVE corpus now, and');
    p(`> the dates run 2026-08-03 to 2026-08-06 — the archiver is still producing them,`);
    p('> roughly ten a day. Merging or not merging them into staging does not touch that.');
    p('>');
    p('> I have not filed them separately or dropped them, because I cannot tell them');
    p('> apart from the genuine ones mechanically — the grammar is third-person in both');
    p('> cases, so `verifySubjectAgreement` passes. Reading them is the check. Some of');
    p('> this group really are yours: the blue eyes, the pet named Roscoe, being told');
    p('> directly when you have made a mistake.');
    p('');
  }

  const cell = (s) => String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ');
  const whyMissing = (f) => {
    if (f.postSnapshot) return `**new since the seed** — created in live ${String(f.createdAt).slice(0, 10)}, after staging was copied, so the replay never saw it`;
    if (f.notEvent) return f.notEvent;
    if (f.event.isEvent) return `carries "${f.event.marker}" — intake routes that to the day's log now; the logs hold ${f.log.hits}/${f.log.total} of its words`;
    if (f.near) {
      const verdict = f.judge && f.judge.relation === 'live-contains-staging'
        ? ' — and it is the poorer of the two'
        : f.judge ? ' — judged not to cover it' : '';
      return `nearest in staging is ${f.near.similarity.toFixed(3)}: "${trunc(f.near.content, 55)}"${verdict}`;
    }
    if (f.source !== 'fact-extraction') return `written by ${f.source}, not by conversation intake — a replay of conversations has nothing to rebuild it from`;
    return 'the pipeline did not produce it from the surviving source';
  };

  for (const [call, title, blurb, groupByCluster] of [
    ['carry', 'Leaning CARRY', 'Nothing in staging covers these, and each has a reason to be kept that is stated beside it.', false],
    ['none', 'No lean — your read', 'These are the ones I will not pretend to have an opinion about. All sit at salience 5, which is the value the scorer assigns when nothing pushed it either way; none is corroborated; staging holds nothing near any of them. There is no signal in the record to lean on, so rather than dress a number up as a reason, here they are grouped by the cluster they live in — you can take a topic at a time.', true],
    ['drop', 'Leaning DROP', 'Salience 4 or below, uncorroborated, nothing near them in staging — or an event whose wording is already on disk in the daily logs. Dropping one of these loses a sentence, not a fact about you.', false]
  ]) {
    const rows = piles.decide.filter(f => f.lean.call === call);
    if (!rows.length) continue;
    p(`### ${title} (${rows.length})`);
    p('');
    p(blurb);
    p('');

    const table = (items) => {
      p('| mark | fact | sal | producer | why it is missing | lean |');
      p('|---|---|---|---|---|---|');
      for (const f of items) {
        const leanText = f.lean.call === 'none' ? `*${cell(f.lean.why)}*` : `**${f.lean.call}** — ${cell(f.lean.why)}`;
        p(`| \`${f.id.slice(0, 8)}\` [ ] | ${cell(f.content)} | ${f.salience} | ${f.source} | ${cell(whyMissing(f))} | ${leanText} |`);
      }
      p('');
    };

    if (groupByCluster) {
      const groups = new Map();
      for (const f of rows) {
        const k = f.cluster || '(no cluster)';
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(f);
      }
      for (const [name, items] of [...groups.entries()].sort((a, b) => b[1].length - a[1].length)) {
        p(`**${name}** (${items.length})`);
        p('');
        table(items.sort((a, b) => b.salience - a.salience));
      }
    } else {
      table(rows.sort((a, b) => b.salience - a.salience));
    }
  }
  if (carriedThenChanged.length) {
    p('---');
    p('');
    p(`## Already carried, then changed — ${carriedThenChanged.length}`);
    p('');
    p('These are live facts an earlier run of this script did carry into staging, and');
    p('which the corrector then acted on. The test at the top of this document — "does');
    p(`staging hold an active fact within ${TWIN_FLOOR} of it?" — says no for each of them, but the`);
    p('reason is not that the carry failed. It is that the carry succeeded and the');
    p('corrector then retired or folded what arrived.');
    p('');
    p('They are listed rather than re-proposed, because carrying them again would');
    p('write a second copy of a row that is already there.');
    p('');
    for (const c of carriedThenChanged) {
      const fate = c.stagingStatus === 'active'
        ? 'still active in staging — reworded or moved far enough that the similarity test no longer matches it'
        : c.stagingStatus === 'gone'
          ? 'the row it was carried into is no longer in staging'
          : `carried in, then **${c.stagingReason || c.stagingStatus}** by the corrector`;
      p(`- **"${c.content}"**`);
      p(`  <br>salience ${c.salience} · ${fate}`);
    }
    p('');
  }

  p('---');
  p('');
  p('## What carrying does');
  p('');
  p('A carried fact is written into staging through `assignToCluster` — the same');
  p('funnel intake and the corrector\'s splitter use — so it is deduped at the write,');
  p('embedded, and placed in the cluster its subject matter belongs to. Nothing is');
  p('inserted by hand.');
  p('');
  p('It keeps its **original learned date**, not today\'s. The corrector decides');
  p('contradictions partly on recency, and a fact stamped with the carry date would');
  p('beat everything the replay rebuilt on nothing but having been carried.');
  p('');
  p('It keeps its **provenance** — conversation, message, verbatim text, modality —');
  p('because the evidence-dominance rules are written in terms of those fields, and');
  p('it keeps its **corroborations**, re-pointed at the new row.');
  p('');
  p('Its `source` becomes `carried_from_live`. The original producer and the review');
  p('date are recorded in the staging corrections ledger, alongside the rule that');
  p('carried it and the live id it came from.');
  p('');
  p('The live corpus is not modified. No cutover happens here.');

  fs.mkdirSync(path.dirname(REVIEW_DOC), { recursive: true });
  fs.writeFileSync(REVIEW_DOC, L.join('\n'));
  console.log(`[Carry] wrote ${REVIEW_DOC}`);
}

// ---------------------------------------------------------------------------
// The carry
// ---------------------------------------------------------------------------

/**
 * Write one live fact into staging, through the funnel, with its history intact.
 *
 * @returns {Promise<{ok: boolean, memberId?: string, cluster?: string, folded?: boolean, reason?: string}>}
 */
async function carryOne(f, { reviewDate, passId, ruleLabel }) {
  const db = require(path.join(ROOT, 'db/database'));
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));
  const d = db.getSqliteDb();

  const config = getConfig();
  const prov = config.models.extraction.provider;
  const model = config.models.extraction.model;
  const inst = getProviderInstance(prov, config.models.extraction.instance);
  const host = inst ? inst.host : 'http://localhost:11434';

  const res = await memoryClusters.assignToCluster(
    f.content, prov, model, '', host,
    'carried_from_live',
    f.salience,
    'user',
    f.claimType,
    {
      conversationId: f.conversationId,
      messageId: f.messageId,
      verbatimSourceText: f.verbatimSourceText,
      inputModality: f.inputModality,
      salienceRationale: f.salienceRationale
    }
  );
  if (!res || !res.memberId) return { ok: false, reason: 'assignToCluster returned no member' };

  // Folded into a row staging already held word-for-word. Nothing was created,
  // so there is no date to restore and no corroboration to move — but it IS a
  // carry outcome and is recorded as one.
  if (res.duplicateOf) {
    return { ok: true, memberId: res.memberId, cluster: res.clusterName, folded: true };
  }

  // The learned date, put back. assignToCluster stamps created_at with now
  // because every other caller is writing something learned now; this one is
  // not. updated_at stays at the carry time, which is true and is what makes the
  // carry visible in the row itself.
  if (f.createdAt) {
    d.prepare('UPDATE cluster_members SET created_at = ? WHERE id = ?').run(f.createdAt, res.memberId);
  }

  // Corroborations, re-pointed. A restatement is evidence about the fact and
  // cannot be left behind pointing at a row in another database.
  let carriedCorroborations = 0;
  if (f.corroborations > 0) {
    const Database = require('better-sqlite3');
    const live = new Database(path.join(LIVE_DATA, 'chat.db'), { readonly: true });
    const rows = live.prepare('SELECT * FROM fact_corroborations WHERE member_id = ?').all(f.id);
    live.close();
    const { randomUUID } = require('crypto');
    const ins = d.prepare(`
      INSERT INTO fact_corroborations
        (id, member_id, created_at, conversation_id, message_id,
         verbatim_source_text, input_modality, restated_as, similarity, detected_by)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    for (const c of rows) {
      ins.run(randomUUID(), res.memberId, c.created_at, c.conversation_id, c.message_id,
        c.verbatim_source_text, c.input_modality, c.restated_as, c.similarity, c.detected_by);
      carriedCorroborations++;
    }
  }

  ledger.record({
    passId,
    tier: 'mechanical',
    action: 'carry',
    subject: 'user',
    targetId: f.id,
    targetText: f.content,
    survivorId: res.memberId,
    survivorText: f.content,
    reason: `The replay did not rebuild this fact from source, and it was carried across from the live corpus instead. ${ruleLabel} Its original learned date (${String(f.createdAt).slice(0, 10)}), provenance and ${carriedCorroborations} corroboration(s) came with it.`,
    evidence: {
      carried_from_live: f.id,
      review_date: reviewDate,
      // A fact from the ELLIE-DECIDES pile has no rule — she is the rule, and
      // the record should say that rather than leave the field absent.
      rule: f.rule || 'ellie-decided',
      rule_detail: f.ruleDetail || (f.lean ? `marked for carry in the review; my lean had been ${f.lean.call}` : null),
      original_source: f.source,
      original_created_at: f.createdAt,
      original_cluster: f.cluster,
      salience: f.salience,
      corroborations_carried: carriedCorroborations,
      cluster: res.clusterName
    },
    // A carry ADDS a row and removes nothing, so "revert" means retiring the
    // row it added — which revert-correction.js cannot do: it restores losers,
    // it does not retire winners. Recorded honestly rather than claimed.
    reversible: false
  });

  return { ok: true, memberId: res.memberId, cluster: res.clusterName, corroborations: carriedCorroborations };
}

/**
 * Ellie's marks, read back out of the review doc.
 *
 * Deliberately tolerant about HOW she marks it, because the alternative is a
 * document that silently ignores a decision she made. All of these are read the
 * same way:
 *
 *     | `abc12345` [x]    | …        an x in the box means carry
 *     | `abc12345` [CARRY]| …        the word in the box
 *     | `abc12345` [ ] CARRY | …     the word after the box
 *     | `abc12345` [drop] | …        case does not matter
 *     - [x] KEEP `abc12345`         the RECOMMEND-DROP override form
 *
 * An empty box is not a decision and is not treated as one. Anything it cannot
 * read is reported rather than skipped quietly — an unparsed mark is a decision
 * that would vanish.
 */
function readMarks() {
  if (!fs.existsSync(REVIEW_DOC)) {
    console.error(`ABORT: no review doc at ${REVIEW_DOC}`);
    process.exit(2);
  }
  const text = fs.readFileSync(REVIEW_DOC, 'utf8');
  const marks = new Map();      // short id -> 'carry' | 'drop' | 'keep'
  const unreadable = [];

  // The RECOMMEND-DROP override form first — its id comes AFTER the box, so the
  // general pattern below would not see it.
  for (const m of text.matchAll(/\[\s*([xX]?)\s*\]\s*KEEP\s+`([0-9a-f]{8})`/g)) {
    if (m[1]) marks.set(m[2], 'keep');
  }

  // The table form: an id, then a box, then optionally a word after it.
  for (const m of text.matchAll(/`([0-9a-f]{8})`[^|\n]{0,4}\[([^\]\n]{0,12})\]\s*([A-Za-z]{4,5})?/g)) {
    const [, id, box, after] = m;
    const token = String(box).trim() || String(after || '').trim();
    if (!token) continue;                       // empty box — not a decision
    const t = token.toLowerCase();
    if (t === 'x') marks.set(id, 'carry');
    else if (t === 'carry' || t === 'keep' || t === 'drop') marks.set(id, t);
    else unreadable.push({ id, token });
  }

  if (unreadable.length) {
    console.warn(`[Carry] ${unreadable.length} mark(s) could not be read and were IGNORED — fix these and run again:`);
    for (const u of unreadable) console.warn(`  ${u.id}: "${u.token}" (expected x, CARRY, DROP or KEEP)`);
  }
  return marks;
}

async function apply(which) {
  if (!fs.existsSync(PLAN_JSON)) {
    console.error(`ABORT: no frozen plan at ${PLAN_JSON} — run --plan first.`);
    process.exit(2);
  }
  const plan = JSON.parse(fs.readFileSync(PLAN_JSON, 'utf8'));

  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  if (path.resolve(db.getDataDir()) !== path.resolve(STAGING_DATA)) {
    console.error(`ABORT: data dir is ${db.getDataDir()}, expected ${STAGING_DATA}`);
    process.exit(2);
  }

  let batch, label;
  if (which === 'auto') {
    batch = plan.piles.auto;
    label = 'AUTO-CARRY';
  } else if (which === 'decided') {
    const marks = readMarks();
    const wanted = new Set([...marks.entries()].filter(([, v]) => v === 'carry' || v === 'keep').map(([k]) => k));
    batch = [...plan.piles.decide, ...plan.piles.drop].filter(f => wanted.has(f.id.slice(0, 8)));
    label = `ELLIE-DECIDES (${wanted.size} marked)`;
    if (!batch.length) {
      console.log('[Carry] nothing marked to carry — the review doc has no CARRY or KEEP marks.');
      process.exit(0);
    }
  } else {
    console.error(`ABORT: unknown --apply target "${which}" (expected auto | decided)`);
    process.exit(2);
  }

  // Already carried? The ledger is the record — re-running must not double-write.
  const d = db.getSqliteDb();
  const already = new Set(
    d.prepare("SELECT target_id FROM corrections_ledger WHERE action = 'carry'").all().map(r => r.target_id)
  );
  const todo = batch.filter(f => !already.has(f.id));
  if (todo.length < batch.length) {
    console.log(`[Carry] ${batch.length - todo.length} of ${batch.length} already carried (ledger) — skipping those`);
  }

  const passId = `carry-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  const reviewDate = plan.generatedAt.slice(0, 10);
  console.log(`[Carry] ${label}: carrying ${todo.length} fact(s), pass ${passId}`);

  const out = { carried: [], folded: [], failed: [] };
  for (let i = 0; i < todo.length; i++) {
    const f = todo[i];
    const ruleLabel = f.ruleDetail
      ? `It was carried because it is ${f.rule.replace(/-/g, ' ')}: ${f.ruleDetail}.`
      : `It was carried on Ellie's mark in the carry review.`;
    let res;
    try {
      res = await carryOne(f, { reviewDate, passId, ruleLabel });
    } catch (err) {
      res = { ok: false, reason: err.message };
    }
    if (!res.ok) {
      out.failed.push({ id: f.id, content: f.content, reason: res.reason });
      console.error(`[Carry] FAILED ${f.id.slice(0, 8)} "${trunc(f.content, 60)}": ${res.reason}`);
    } else if (res.folded) {
      out.folded.push({ id: f.id, content: f.content, into: res.memberId });
      console.log(`[Carry] ${i + 1}/${todo.length} folded (already held word-for-word) "${trunc(f.content, 60)}"`);
    } else {
      out.carried.push({ id: f.id, newId: res.memberId, content: f.content, cluster: res.cluster });
      console.log(`[Carry] ${i + 1}/${todo.length} → ${res.cluster} "${trunc(f.content, 60)}"`);
    }
  }

  console.log(`\n${'='.repeat(74)}`);
  console.log(`CARRY COMPLETE — ${label}`);
  console.log(`  carried : ${out.carried.length}`);
  console.log(`  folded  : ${out.folded.length} (staging already held the exact text)`);
  console.log(`  failed  : ${out.failed.length}`);
  console.log(`  pass    : ${passId}`);
  console.log(`${'='.repeat(74)}\n`);

  // ACCUMULATE, do not overwrite. A second `--apply auto` after the plan was
  // regenerated carries only what newly qualifies, and writing that over the file
  // would drop the first run's rows from it — which is how the finalize step's
  // live-id → staging-id map lost 63 of its 66 entries and would have stopped
  // repairing successor chains it had already repaired. The ledger is the record
  // of truth either way; this file just has to stop contradicting it.
  const resultPath = path.join(STAGING_DATA, `carry-result-${which}.json`);
  let prior = { runs: [], carried: [], folded: [], failed: [] };
  if (fs.existsSync(resultPath)) {
    try {
      const old = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
      prior = {
        runs: old.runs || [{ passId: old.passId, reviewDate: old.reviewDate, label: old.label }],
        carried: old.carried || [], folded: old.folded || [], failed: old.failed || []
      };
    } catch { /* unreadable — start fresh rather than lose this run */ }
  }
  const seenNew = new Set([...out.carried, ...out.folded].map(r => r.id));
  fs.writeFileSync(resultPath, JSON.stringify({
    runs: [...prior.runs, { passId, reviewDate, label, carried: out.carried.length, folded: out.folded.length, failed: out.failed.length }],
    carried: [...prior.carried.filter(r => !seenNew.has(r.id)), ...out.carried],
    folded: [...prior.folded.filter(r => !seenNew.has(r.id)), ...out.folded],
    failed: out.failed
  }, null, 2));
  console.log(`[Carry] result → ${resultPath}`);
  return out;
}

// ---------------------------------------------------------------------------

(async () => {
  abortIfLive();
  if (!MODE) {
    console.error('Usage: carry-to-staging.js --plan | --rewrite | --apply auto | --apply decided');
    process.exit(2);
  }

  // The document is a RENDERING of the frozen plan, so re-rendering it must not
  // cost another 148 judge calls — and must not silently produce a different
  // classification either. Editing a sentence in the report should not be able to
  // change which pile a fact is in.
  if (MODE === 'rewrite') {
    if (!fs.existsSync(PLAN_JSON)) {
      console.error(`ABORT: no frozen plan at ${PLAN_JSON} — run --plan first.`);
      process.exit(2);
    }
    const plan = JSON.parse(fs.readFileSync(PLAN_JSON, 'utf8'));
    const missing = [...plan.piles.auto, ...plan.piles.drop, ...plan.piles.decide];
    writeReview({
      missing, piles: plan.piles,
      seededAt: plan.seededAt, liveTotal: plan.liveActiveUserFacts,
      carriedThenChanged: plan.carriedThenChanged || [],
      carriedTotal: plan.carriedTotal || 0
    });
    console.log(`[Carry] re-rendered from the plan frozen at ${plan.generatedAt} — no facts were re-classified`);
    process.exit(0);
  }

  /**
   * Take back a carry.
   *
   * A carry ADDS a row, so revert-correction.js cannot undo one — it restores
   * losers, it does not retire winners, and its ledger entry is recorded
   * reversible = 0 for exactly that reason. This is the other half, and it is a
   * deliberate path: a person names the fact and gives a reason.
   *
   * It retires rather than deletes, like everything else in the funnel. The row
   * stays as history with inactive_reason = 'retracted' and the vector goes, so
   * it can no longer reach an answer by either route. The withdrawal is its own
   * ledger entry, pointing at the carry it undoes.
   */
  if (MODE === 'withdraw') {
    const target = argVal('--withdraw', '');
    const reason = argVal('--reason', '');
    if (!target || !reason) {
      console.error('Usage: carry-to-staging.js --withdraw <staging-id-or-prefix> --reason "why"');
      process.exit(2);
    }
    const db = require(path.join(ROOT, 'db/database'));
    db.initDatabase();
    await db.initVectorStore();
    const d = db.getSqliteDb();
    const factStore = require(path.join(ROOT, 'db/fact-store'));
    const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

    const row = d.prepare(
      "SELECT * FROM cluster_members WHERE id LIKE ? AND source = 'carried_from_live' AND status = 'active'"
    ).get(`${target}%`);
    if (!row) {
      console.error(`ABORT: no active carried fact matching "${target}".`);
      process.exit(2);
    }
    const carryEntry = d.prepare(
      "SELECT id, target_id, evidence FROM corrections_ledger WHERE action = 'carry' AND survivor_id = ?"
    ).get(row.id);

    const res = await factStore.retire(row.id, { reason, deliberate: true });
    if (!res.ok) { console.error(`ABORT: retire failed — ${res.reason}`); process.exit(1); }

    ledger.record({
      passId: `withdraw-${new Date().toISOString().replace(/[:.]/g, '-')}`,
      tier: 'mechanical', action: 'carry-withdrawn', subject: row.subject || 'user',
      targetId: row.id, targetText: row.content,
      reason: `This fact was carried across from the live corpus during the merge and then taken back out. ${reason} It is retired in staging, not deleted, and the live corpus still holds it unchanged.`,
      evidence: {
        undoes_carry: carryEntry ? carryEntry.id : null,
        live_id: carryEntry ? carryEntry.target_id : null,
        withdrawn_reason: reason
      },
      // Retiring is reversible through the normal path.
      reversible: true
    });

    console.log(`[Carry] withdrawn ${row.id.slice(0, 8)} "${trunc(row.content, 70)}"`);
    console.log(`[Carry] reason: ${reason}`);
    console.log(`[Carry] the live row is untouched; staging keeps this as inactive history.`);
    process.exit(0);
  }

  if (MODE === 'plan') {
    const t0 = Date.now();
    const result = await classify();
    writeReview(result);
    fs.writeFileSync(PLAN_JSON, JSON.stringify({
      generatedAt: new Date().toISOString(),
      stagingDir: STAGING_DATA,
      seededAt: result.seededAt,
      liveActiveUserFacts: result.liveTotal,
      twinFloor: TWIN_FLOOR, bandLo: BAND_LO, highSalience: HIGH_SALIENCE,
      carriedThenChanged: result.carriedThenChanged,
      carriedTotal: result.carriedTotal,
      counts: {
        missing: result.missing.length,
        postSnapshot: result.missing.filter(f => f.postSnapshot).length,
        auto: result.piles.auto.length, drop: result.piles.drop.length, decide: result.piles.decide.length
      },
      piles: result.piles
    }, null, 2));
    console.log(`[Carry] froze the plan → ${PLAN_JSON} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    process.exit(0);
  }

  await apply(MODE.split(':')[1]);
  process.exit(0);
})().catch(err => { console.error('carry failed:', err); process.exit(1); });
