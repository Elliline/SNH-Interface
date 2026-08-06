#!/usr/bin/env node
/**
 * The SECOND staging gate report — the merged corpus, and what is left to decide.
 *
 * The first gate asked whether the corpus could be rebuilt from source. It could,
 * partly, and Ellie's decision on that answer was MERGE rather than replace. This
 * reports the result of the merge: what was carried, what the corrector then did
 * to the combined corpus, what the two ambiguities came to, and the one list that
 * still needs her.
 *
 * It assembles, it does not re-derive. Every step of the merge wrote its own
 * result file, and this reads those alongside a fresh count of both stores, so
 * the numbers in the report and the numbers the steps acted on cannot disagree.
 * The one thing it does run is the fixture check, because a fixture result is
 * only worth anything if it was taken against the corpus as it stands right now.
 *
 * READ-ONLY against both stores. NO CUTOVER — nothing here promotes staging to
 * live, and there is no code in this file that could.
 *
 * Usage:
 *   SNH_DATA_DIR is set internally; just run it.
 *   node scripts/staging-gate2-report.js [--json] [--out docs/staging-gate2-report.txt]
 */
const path = require('path');
const fs = require('fs');
const { execFileSync } = require('child_process');
const ROOT = path.join(__dirname, '..');

const LIVE_DATA = path.join(ROOT, 'data');
const STAGING_DATA = process.env.SNH_STAGING_DIR || path.join(ROOT, 'data-staging');
process.env.SNH_DATA_DIR = STAGING_DATA;

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const JSON_OUT = args.includes('--json');
const OUT_PATH = argVal('--out', path.join(ROOT, 'docs', 'staging-gate2-report.txt'));

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

const readJson = (p, dflt = null) => {
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return dflt; }
};

(async () => {
  if (path.resolve(STAGING_DATA) === path.resolve(LIVE_DATA)) {
    console.error('ABORT: staging dir resolves to the live data dir.');
    process.exit(2);
  }

  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  const staging = db.getSqliteDb();
  const Database = require('better-sqlite3');
  const live = new Database(path.join(LIVE_DATA, 'chat.db'), { readonly: true });

  // ---- counts, from both stores, counted now ------------------------------
  const countBy = (conn) => {
    const rows = conn.prepare(
      "SELECT COALESCE(subject,'user') s, status, COUNT(*) n FROM cluster_members GROUP BY s, status"
    ).all();
    const get = (s, st) => rows.find(r => r.s === s && r.status === st)?.n ?? 0;
    return {
      user_active: get('user', 'active'), user_inactive: get('user', 'inactive'),
      self_active: get('self', 'active'), self_inactive: get('self', 'inactive'),
      world_active: get('world', 'active'), world_inactive: get('world', 'inactive'),
      clusters: conn.prepare('SELECT COUNT(*) n FROM memory_clusters').get().n,
      user_clusters: conn.prepare("SELECT COUNT(*) n FROM memory_clusters WHERE COALESCE(subject,'user')='user'").get().n,
      self_clusters: conn.prepare("SELECT COUNT(*) n FROM memory_clusters WHERE subject='self'").get().n,
      corroborations: conn.prepare('SELECT COUNT(*) n FROM fact_corroborations').get().n,
      ledger: conn.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n
    };
  };
  const liveStats = countBy(live);
  const stagingStats = countBy(staging);

  const bySource = staging.prepare(
    "SELECT COALESCE(source,'(null)') src, COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user' GROUP BY src ORDER BY n DESC"
  ).all();

  const dangling = staging.prepare(`
    SELECT COUNT(*) n FROM cluster_members m
    WHERE m.successor_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM cluster_members s WHERE s.id = m.successor_id)
  `).get().n;

  const emptyClusters = staging.prepare(`
    SELECT COUNT(*) n FROM memory_clusters c
    WHERE NOT EXISTS (SELECT 1 FROM cluster_members m WHERE m.cluster_id = c.id)
  `).get().n;
  const ghostClusters = staging.prepare(`
    SELECT COUNT(*) n FROM memory_clusters c
    WHERE NOT EXISTS (SELECT 1 FROM cluster_members m WHERE m.cluster_id = c.id AND m.status='active')
      AND EXISTS (SELECT 1 FROM cluster_members m WHERE m.cluster_id = c.id)
  `).get().n;

  // ---- the artefacts each step left behind --------------------------------
  const replay = readJson(path.join(STAGING_DATA, 'replay-stats.json'), {});
  const plan = readJson(path.join(STAGING_DATA, 'carry-plan.json'), null);
  // Counted from the LEDGER, not from the carry runs' result files. Every carry
  // writes its ledger row in the same step that writes the fact, so the two
  // cannot disagree; a result file can be — and was — overwritten by a later run
  // that carried only what newly qualified, which reported 3 carries where 66
  // had happened. Grouped by the rule that carried each one, which is the thing
  // worth reading anyway.
  const carryRows = staging.prepare(`
    SELECT target_id, survivor_id, evidence FROM corrections_ledger WHERE action = 'carry'
  `).all();
  const carried = { total: carryRows.length, byRule: {}, stillActive: 0, changedSince: [] };
  const statusOf = staging.prepare('SELECT status, inactive_reason, content FROM cluster_members WHERE id = ?');
  for (const r of carryRows) {
    let rule = '(unrecorded)';
    try { rule = JSON.parse(r.evidence).rule || rule; } catch { /* keep default */ }
    carried.byRule[rule] = (carried.byRule[rule] || 0) + 1;
    const s = r.survivor_id ? statusOf.get(r.survivor_id) : null;
    if (s && s.status === 'active') carried.stillActive++;
    else if (s) carried.changedSince.push({ text: s.content, reason: s.inactive_reason });
  }
  const corr = readJson(path.join(STAGING_DATA, 'corrector-staging-result.json'), null);
  const finalizeFile = readJson(path.join(STAGING_DATA, 'finalize-result.json'), null);
  // `latest` is the last run and reports 0 for an idempotent re-run; `totals` is
  // what the merge actually did to the corpus across every run. Read both.
  const finalize = finalizeFile && (finalizeFile.latest || finalizeFile);
  const finalizeTotals = finalizeFile && finalizeFile.totals;

  live.close();

  // ---- fixtures, run against the corpus as it stands right now ------------
  // db/database.js logs "SQLite database initialized successfully" to stdout on
  // load, so --json output is preceded by chatter and JSON.parse of the whole
  // stream fails. Take the JSON document rather than the stream.
  const parseJsonBlock = (s) => {
    const i = String(s || '').indexOf('{');
    if (i < 0) throw new Error('no JSON in output');
    return JSON.parse(String(s).slice(i));
  };

  let fixtures = null;
  try {
    const out = execFileSync(process.execPath, [path.join(ROOT, 'scripts/check-fixtures.js'), '--staging', '--json'], {
      env: { ...process.env, SNH_DATA_DIR: STAGING_DATA },
      encoding: 'utf8'
    });
    fixtures = parseJsonBlock(out);
  } catch (err) {
    // Exit code 1 means a fixture survived — that is a RESULT, not a crash, and
    // its JSON is on stdout. Anything else is a genuine failure to measure.
    try { fixtures = parseJsonBlock(err.stdout); }
    catch { fixtures = { pass: false, error: err.message, results: [] }; }
  }

  const report = {
    generatedAt: new Date().toISOString(),
    stagingDir: STAGING_DATA,
    fixtures, stats: { live: liveStats, staging: stagingStats },
    bySource, dangling, emptyClusters, ghostClusters,
    replay, plan: plan && { counts: plan.counts, seededAt: plan.seededAt },
    carried,
    corrector: corr && { totals: corr.totals, rounds: corr.rounds.length, stopReason: corr.stopReason, activeUserFacts: corr.activeUserFacts },
    finalize: finalize && { clusters: finalize.clusters, successors: finalize.successors },
    finalizeTotals,
    ellieDecides: plan ? plan.piles.decide.map(f => ({
      id: f.id.slice(0, 8), text: f.content, salience: f.salience, source: f.source, lean: f.lean.call
    })) : []
  };

  if (JSON_OUT) { console.log(JSON.stringify(report, null, 2)); process.exit(0); }

  const L = [];
  const say = (s = '') => L.push(s);
  const line = (c = '=') => say(c.repeat(78));

  line();
  say('SECOND STAGING GATE REPORT — the merged corpus');
  say(`staging: ${STAGING_DATA}`);
  say(`generated: ${report.generatedAt}`);
  line();
  say('');
  say('NO CUTOVER HAS HAPPENED. Live is untouched by every step below; staging is a');
  say('separate store and nothing in this pipeline promotes it.');

  // ---- 1. fixtures --------------------------------------------------------
  say('');
  say('--- FIXTURES ---');
  if (fixtures && fixtures.results && fixtures.results.length) {
    const passed = fixtures.results.filter(r => r.pass).length;
    say(`  ${passed}/${fixtures.results.length} pass against the merged staging corpus.`);
    say('');
    for (const r of fixtures.results) {
      say(`  ${r.pass ? 'PASS' : 'FAIL'}  ${r.id}  ${r.name}`);
      if (r.synthetic) say(`        SYNTHETIC: ${trunc(r.synthetic, 200)}`);
      say(`        want: ${r.want}`);
      for (const n of (r.notes || [])) say(`        ${n}`);
    }
    if (!fixtures.pass) say(`\n  SURVIVING: ${(fixtures.failed || []).join(', ')}`);

    // Most of these fixtures assert an ABSENCE, and a zero is the pass — F1's
    // "0 facts mentioning Mike" and F3's zeroes are the result, not a gap, and F1
    // additionally checks something positive (exactly one name fact, saying
    // Ellie).
    //
    // F4 is the exception, and it is worth saying out loud. It asserts a
    // RELATIONSHIP between two facts — no subset sitting beside its superset —
    // and a corpus holding no fact about Casper at all satisfies it without
    // testing anything. The checker cannot tell "the defect was repaired" from
    // "the subject never arrived", so this does.
    const casper = fixtures.results.find(r => r.id === 'F4');
    const casperCount = casper && (casper.notes || []).find(n => /active facts naming Casper/.test(n));
    if (casper && casper.pass && /:\s*0$/.test(String(casperCount || '').trim())) {
      say('');
      say('  F4 passes VACUOUSLY. It asks whether a fact about Casper is sitting beside a');
      say('  poorer copy of itself, and staging holds no fact about Casper at all — there');
      say('  is nothing for the rule to be true or false about. The reason is below: the');
      say('  Casper fact is in ELLIE-DECIDES, unmarked. Carry it and F4 becomes a real');
      say('  test again. The other four are checking what they are meant to check.');
    }
  } else {
    say(`  could not run the fixture check: ${fixtures && fixtures.error}`);
  }

  // ---- 2. counts ----------------------------------------------------------
  say('');
  say('--- FINAL COUNTS ---');
  const row = (label, a, b) => say(`  ${label.padEnd(26)} live ${String(a).padStart(6)}    staging ${String(b).padStart(6)}   ${b - a >= 0 ? '+' : ''}${b - a}`);
  row('user facts (active)', liveStats.user_active, stagingStats.user_active);
  row('user facts (inactive)', liveStats.user_inactive, stagingStats.user_inactive);
  row('self facts (active)', liveStats.self_active, stagingStats.self_active);
  row('self facts (inactive)', liveStats.self_inactive, stagingStats.self_inactive);
  row('world facts (active)', liveStats.world_active, stagingStats.world_active);
  row('clusters (all)', liveStats.clusters, stagingStats.clusters);
  row('clusters (user)', liveStats.user_clusters, stagingStats.user_clusters);
  row('clusters (self)', liveStats.self_clusters, stagingStats.self_clusters);
  row('corroborations', liveStats.corroborations, stagingStats.corroborations);
  row('corrections ledger', liveStats.ledger, stagingStats.ledger);
  say('');
  say('  staging active user facts, by what produced them:');
  for (const s of bySource) say(`    ${s.src.padEnd(22)} ${String(s.n).padStart(5)}`);

  // ---- 3. how it got here -------------------------------------------------
  say('');
  say('--- HOW THE MERGE WENT ---');
  if (replay && replay.stored != null) {
    say(`  replay          : ${replay.stored} fact(s) rebuilt from ${replay.applied}/${replay.exchanges} exchanges,`);
    say(`                    ${replay.repeats} repeats folded, ${replay.events} events to the day's log, ${replay.refusals} refused`);
  }
  if (plan) {
    say(`  carry review    : ${plan.counts.missing} live fact(s) staging did not hold` +
      (plan.counts.postSnapshot ? ` (${plan.counts.postSnapshot} created after the seed)` : ''));
    say(`                    AUTO-CARRY ${plan.counts.auto} · RECOMMEND-DROP ${plan.counts.drop} · ELLIE-DECIDES ${plan.counts.decide}`);
  }
  say(`  carried in      : ${carried.total} fact(s), counted from the ledger`);
  for (const [rule, n] of Object.entries(carried.byRule).sort((x, y) => y[1] - x[1])) {
    say(`                    ${String(n).padStart(4)}  ${rule}`);
  }
  say(`                    ${carried.stillActive} of them are still active; ${carried.changedSince.length} were changed by the corrector afterwards`);
  for (const c of carried.changedSince) say(`                      ${c.reason || 'inactive'}: "${trunc(c.text, 68)}"`);
  const decidedRun = carried.byRule['ellie-decided'] || 0;
  if (!decidedRun) say('  carry (decided) : not run — waiting on the marks in docs/carry-review.md');

  // ---- 4. the corrector ---------------------------------------------------
  say('');
  say('--- CORRECTOR PASS AGAINST THE MERGED CORPUS ---');
  if (!corr) {
    say('  not run.');
  } else {
    say(`  ${corr.rounds.length} round(s), ${(corr.durationMs / 60000).toFixed(1)} min — ${corr.stopReason}`);
    say(`  duplicates folded       : ${corr.totals.merged}`);
    say(`  events moved to a log   : ${corr.totals.expired}`);
    say(`  compounds split         : ${corr.totals.split}`);
    say(`  contradictions resolved : ${corr.totals.superseded}`);
    say(`  left unresolved for you : ${corr.totals.unresolved}`);
    say(`  refused by the lock     : ${corr.totals.refusedLocked}`);
    say(`  active user facts       : ${corr.activeUserFacts.before} → ${corr.activeUserFacts.after} (${corr.activeUserFacts.delta >= 0 ? '+' : ''}${corr.activeUserFacts.delta})`);
    say('');
    say(`  Its ledger for this run is ${corr.ledgerEntries} entr(y/ies), in full, in`);
    say('  data-staging/corrector-staging-result.json. The gate report for round one');
    say('  counted 26 near-duplicate pairs at 0.86 that the corrector had never been');
    say('  allowed to see; this is what it did when it was.');
    // The ledger records an unresolved raise as `semantic/supersede` with
    // reversible = 0, in the same column as a supersession that happened. The
    // rule is that anything rendering the ledger has to say NOTHING CHANGED for
    // those, or it claims edits that were never made — so they are split here
    // rather than counted together.
    const entries = corr.entries || [];
    const retired = entries.filter(e => e.action === 'supersede' && e.reversible);
    const raised = entries.filter(e => e.action === 'supersede' && !e.reversible);
    const merged = entries.filter(e => e.action === 'merge');

    say('');
    say(`  The ledger holds ${entries.length} entr(y/ies) for this run: ${merged.length} merge(s), ${retired.length} supersession(s)`);
    say(`  that RETIRED a fact, and ${raised.length} contradiction(s) recorded with reversible = 0 because`);
    say('  the evidence could not separate them and nothing was changed. Those last are');
    say('  refusals, not corrections, and are not counted as work done.');

    if (merged.length) {
      say('');
      say(`  MERGED — one fact held twice, folded into one  (${merged.length})`);
      for (const e of merged) {
        say(`    "${trunc(e.target_text, 74)}"`);
        say(`      into: "${trunc(e.survivor_text, 74)}"`);
      }
    }

    if (retired.length) {
      say('');
      say(`  RETIRED — the ${retired.length} change(s) with consequences. LOOK AT THESE.`);
      say('');
      for (const e of retired) {
        say(`    ledger ${e.id}`);
        say(`      retired  : "${trunc(e.target_text, 82)}"`);
        say(`      in favour: "${trunc(e.survivor_text, 82)}"`);
        try { say(`      on       : ${JSON.parse(e.evidence).deciding_axis}`); } catch { /* none */ }
        say('');
      }
      say('    Any of them can be put back:');
      say('      SNH_DATA_DIR=data-staging node scripts/revert-correction.js <ledger-id>');

      // HISTORY IS NOT A CONTRADICTION, and this is where that shows up.
      //
      // A past-tense sentence retired in favour of a present-tense one is not a
      // contradiction resolved — it is a record of what was, deleted because of
      // what is. "User had to get rid of the RAV and the Tacoma due to painful
      // memories associated with a death" and "User has a Rav4" are both true;
      // they are about different vehicles at different times. The contradiction
      // judge said YES and evidence dominance then picked the newer one.
      //
      // The test below is deliberately crude — a past-tense verb in the retired
      // sentence and a present-tense one in the survivor. It is a prompt to go
      // and read them, not a verdict.
      const PAST = /\b(had|kept|traded|preferred|used to|purchased|acquired|got rid of|was|were|sold|lost)\b/i;
      const PRESENT = /\b(has|have|is|are|owns|prefers|likes|runs)\b/i;
      const historical = retired.filter(e => PAST.test(e.target_text || '') && PRESENT.test(e.survivor_text || ''));
      if (historical.length) {
        say('');
        say(`    ${historical.length} of the ${retired.length} retire a PAST-TENSE sentence in favour of a PRESENT-TENSE one:`);
        for (const e of historical) say(`      "${trunc(e.target_text, 62)}"  →  "${trunc(e.survivor_text, 40)}"`);
        say('');
        say('    History is not a contradiction. "User had to get rid of the RAV" and "User');
        say('    has a Rav4" are both true — different vehicles, different times — and');
        say('    retiring the first loses why the first one went. The contradiction judge');
        say('    answered YES and evidence dominance then preferred the newer, better-');
        say('    evidenced sentence, which is the rule working exactly as written on a pair');
        say('    it should never have been given. Worth deciding whether that is a corpus');
        say('    problem here or a judge problem that will keep happening on live.');
        say('');
        say('    One of them has already cost something concrete: the chain from "As of late');
        say('    June 2026, User purchased a Toyota Tundra" was about to be repaired onto the');
        say('    carried Highlander fact, and repoint refused because that fact had just been');
        say('    retired. It is one of the 66 still dangling below.');
      }
    }

    if (raised.length) {
      say('');
      say(`  RAISED FOR YOU, NOTHING CHANGED  (${raised.length})`);
      say('  Pairs that contradict where the evidence is evenly matched. The corrector');
      say('  refusing to pick is the feature, not a gap.');
      const seen = new Set();
      for (const e of raised) {
        const k = `${e.target_id}|${e.survivor_id}`;
        if (seen.has(k)) continue;
        seen.add(k);
        say(`    "${trunc(e.target_text, 70)}"`);
        say(`      vs "${trunc(e.survivor_text, 70)}"`);
      }
    }
  }

  // ---- 5. the two ambiguities --------------------------------------------
  say('');
  say('--- THE TWO THINGS THE FIRST GATE LEFT UNDECIDED ---');
  if (finalize) {
    // Repoints and dangling-successor records are counted from the LEDGER, which
    // is authoritative and cannot be overwritten by a later idempotent run. Only
    // the cluster prune has no ledger row to count, so it comes from the run
    // record.
    const T = {
      clustersPruned: (finalizeTotals && finalizeTotals.clustersPruned) ?? finalize.clusters.pruned,
      linksRemoved: (finalizeTotals && finalizeTotals.linksRemoved) ?? finalize.clusters.linksRemoved,
      // Scoped to the merge's own passes. staging inherited live's ledger whole,
      // and live already had one repoint in it — the mis-subjected name chain
      // repaired on 2026-08-05. Counting that as work this merge did made 82
      // dangling minus 17 repaired come to 66 instead of 65.
      successorsRepaired: staging.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'repoint' AND pass_id LIKE 'finalize-%'").get().n,
      danglingLedgerEntries: staging.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'dangling-successor' AND pass_id LIKE 'finalize-%'").get().n
    };
    say('  CLUSTERS');
    say(`    ${T.clustersPruned} cluster(s) held nothing at all and were pruned, taking ${T.linksRemoved} dead`);
    say('    cluster link(s) with them — a link naming a cluster that is gone cannot be');
    say('    rendered or traversed, and the foreign key refuses the delete without it.');
    say(`    ${emptyClusters} empty cluster(s) remain.`);
    say(`    ${finalize.clusters.ghostsOnly} more hold only retired facts. Those were KEPT: getCluster returns`);
    say('    inactive members on purpose so the Map can draw them as ghosts, and deleting');
    say('    the cluster would orphan those rows against the foreign key.');
    say(`    embeddings naming a deleted cluster afterwards: ${finalize.clusters.staleClusterVectorsAfter}`);
    say('');
    say('  DANGLING SUCCESSORS');
    say(`    left by the replay       : ${replay.seed ? replay.seed.dangling : '?'}`);
    say(`    repaired by the carry    : ${T.successorsRepaired}`);
    say(`    still dangling           : ${finalize.successors.stillDangling}`);
    say(`    ledger entries recording : ${T.danglingLedgerEntries}`);
    say(`    inactive_reason on those left: ${JSON.stringify(finalize.successors.inactiveReasons)}`);
    if (finalize.successors.notSuperseded && finalize.successors.notSuperseded.length) {
      say(`    ${finalize.successors.notSuperseded.length} of them are NOT marked superseded — listed in finalize-result.json`);
    } else {
      say('    all of them keep inactive_reason = superseded, as instructed');
    }
    say('');
    say('    A pointer still dangling means: this fact was replaced, and the thing that');
    say('    replaced it is not in the rebuilt corpus. That is recorded per fact in the');
    say("    ledger as 'dangling-successor' with reversible = 0, because nothing changed.");
  } else {
    say('  finalize step not run.');
  }
  say('');
  say(`  counted again just now: ${dangling} dangling pointer(s), ${emptyClusters} empty cluster(s), ${ghostClusters} ghost-only cluster(s)`);

  // ---- 6. what is still hers ---------------------------------------------
  say('');
  say('--- ELLIE-DECIDES — still waiting on you ---');
  if (!plan) {
    say('  no carry plan on disk.');
  } else {
    const decide = plan.piles.decide;
    const byLean = { carry: [], none: [], drop: [] };
    for (const f of decide) (byLean[f.lean.call] || (byLean[f.lean.call] = [])).push(f);
    say(`  ${decide.length} fact(s). Full text, evidence and marking instructions are in`);
    say('  docs/carry-review.md — this is the index.');
    say('');
    for (const [call, title] of [['carry', 'leaning CARRY'], ['none', 'no lean, your read'], ['drop', 'leaning DROP']]) {
      const rows = byLean[call] || [];
      if (!rows.length) continue;
      say(`  ${title} (${rows.length}):`);
      for (const f of rows.sort((a, b) => b.salience - a.salience)) {
        say(`    ${f.id.slice(0, 8)}  [sal ${String(f.salience).padStart(2)}]  ${trunc(f.content, 92)}`);
      }
      say('');
    }
    say('  Mark them in docs/carry-review.md, then:');
    say('    node scripts/carry-to-staging.js --apply decided');
    say('  Unmarked means untouched — the live corpus still holds every one of them.');

    // Found while building that pile, and it is not about staging.
    const archived = decide.filter(f => f.source === 'daily-log-archive');
    if (archived.length) {
      say('');
      say('--- SOMETHING THE REVIEW TURNED UP, WHICH IS NOT A STAGING PROBLEM ---');
      say('');
      say(`  ${archived.length} of those ${decide.length} came from the daily-log archiver, and read together most of`);
      say('  them are not facts about Ellie. They are Aurelius describing himself, in the');
      say('  third person, filed as her preferences:');
      say('');
      // Show the ones that make the case, not the first eight — the batch also
      // holds genuine facts about her, and leading with those undercuts the point
      // while looking like the evidence for it.
      const TELL = /\b(non-judgmental|sounding board|questioning tone|as an interface|presence|inquiry|probe|probing|validating|renaming|re-gendering|locked|springboard)\b/i;
      const illustrative = archived.filter(f => TELL.test(f.content));
      for (const f of illustrative.slice(0, 8)) say(`    "${trunc(f.content, 88)}"`);
      say('');
      say(`  (${illustrative.length} of the ${archived.length} read that way to me. The rest of the batch includes real ones —`);
      say('   the blue eyes, the pet named Roscoe, wanting to be told directly about a mistake.)');
      say('');
      say('  "aims to be a steady, non-judgmental presence that respects boundaries and');
      say('  cognitive load" describes an assistant. "responds to attempts at renaming or');
      say('  re-gendering with a highly structured, defensive protocol to protect core');
      say('  facts" describes the identity lock. This is the 2026-07-27 misattribution that');
      say('  db/memory-write.js was written to refuse — self-observations rewritten into the');
      say('  third person and stored as beliefs about her — reached through the archiver,');
      say('  which was never asked the subject question.');
      say('');
      say('  THESE ROWS ARE IN LIVE, NOW. Their created_at runs 2026-08-03 to 2026-08-06 and');
      say('  the recent days carry about ten each. Whatever is decided about the merge does');
      say('  not touch that; the archiver is still writing them.');
      say('');
      say('  Not acted on here. The grammar is third-person either way, so');
      say('  verifySubjectAgreement passes and there is no mechanical test that separates');
      say('  these from the genuine ones — "User has a pet named Roscoe" and "User has blue');
      say('  eyes" are in the same batch and are really hers. Reading them is the check,');
      say('  and deciding what to do about the archiver is a change of its own.');
    }
  }

  say('');
  line();
  say('STOP. This is the gate. The next step is a cutover and it has not been taken.');
  line();

  const text = L.join('\n');
  console.log(text);
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, `${text}\n`);
  console.error(`\n[gate2] written to ${OUT_PATH}`);
  process.exit(0);
})().catch(err => { console.error('gate report failed:', err); process.exit(1); });
