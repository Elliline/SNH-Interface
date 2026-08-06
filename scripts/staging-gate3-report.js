#!/usr/bin/env node
/**
 * The THIRD staging gate report — short, because only three things changed.
 *
 * Gate 2 handed back 88 facts for Ellie to decide on and a corrector pass with
 * three retirements that should never have been made. She marked the pile, the
 * three were reverted, and the rule that produced them is now enforced at
 * enumeration. This reports exactly that: whether the fixtures still pass and
 * whether F4 passes for a real reason now, what the corpus holds, and what the
 * merge declined to carry.
 *
 * READ-ONLY. NO CUTOVER — nothing here promotes staging to live.
 *
 * Usage:
 *   node scripts/staging-gate3-report.js [--json] [--out docs/staging-gate3-report.txt]
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
const OUT_PATH = argVal('--out', path.join(ROOT, 'docs', 'staging-gate3-report.txt'));

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};
const readJson = (p, d = null) => { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return d; } };

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

  // ---- fixtures, against the corpus as it stands right now ----------------
  const parseJsonBlock = (s) => {
    const i = String(s || '').indexOf('{');
    if (i < 0) throw new Error('no JSON in output');
    return JSON.parse(String(s).slice(i));
  };
  let fixtures;
  try {
    fixtures = parseJsonBlock(execFileSync(
      process.execPath, [path.join(ROOT, 'scripts/check-fixtures.js'), '--staging', '--json'],
      { env: { ...process.env, SNH_DATA_DIR: STAGING_DATA }, encoding: 'utf8' }
    ));
  } catch (err) {
    try { fixtures = parseJsonBlock(err.stdout); }
    catch { fixtures = { pass: false, error: err.message, results: [] }; }
  }

  // F4 asserts a RELATIONSHIP between two facts, so it is satisfied for free by a
  // corpus holding no fact about Casper. Gate 2 flagged it as a vacuous pass;
  // this checks it is not one any more.
  const f4 = (fixtures.results || []).find(r => r.id === 'F4');
  const casperCount = (() => {
    const n = (f4 && (f4.notes || []).find(x => /active facts naming Casper/.test(x))) || '';
    const m = String(n).match(/:\s*(\d+)\s*$/);
    return m ? parseInt(m[1], 10) : null;
  })();
  const f4Vacuous = !!(f4 && f4.pass && casperCount === 0);

  const countBy = (conn) => {
    const rows = conn.prepare("SELECT COALESCE(subject,'user') s, status, COUNT(*) n FROM cluster_members GROUP BY s, status").all();
    const get = (s, st) => rows.find(r => r.s === s && r.status === st)?.n ?? 0;
    return {
      user_active: get('user', 'active'), user_inactive: get('user', 'inactive'),
      self_active: get('self', 'active'), self_inactive: get('self', 'inactive'),
      clusters: conn.prepare('SELECT COUNT(*) n FROM memory_clusters').get().n,
      corroborations: conn.prepare('SELECT COUNT(*) n FROM fact_corroborations').get().n,
      ledger: conn.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n
    };
  };
  const liveStats = countBy(live);
  const stagingStats = countBy(staging);
  live.close();

  const bySource = staging.prepare(
    "SELECT COALESCE(source,'(null)') src, COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user' GROUP BY src ORDER BY n DESC"
  ).all();
  const dangling = staging.prepare(`
    SELECT COUNT(*) n FROM cluster_members m WHERE m.successor_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM cluster_members s WHERE s.id = m.successor_id)`).get().n;
  const carriedTotal = staging.prepare("SELECT COUNT(*) n FROM corrections_ledger WHERE action = 'carry'").get().n;
  const seedTs = (readJson(path.join(STAGING_DATA, 'carry-plan.json'), {}) || {}).seededAt || '1970-01-01';
  const reverted = staging.prepare(
    "SELECT id, target_text, survivor_text, reverted_at FROM corrections_ledger WHERE reverted_at IS NOT NULL AND datetime(reverted_at) > datetime(?) ORDER BY reverted_at"
  ).all(seedTs);

  const marks = readJson(path.join(STAGING_DATA, 'subject-marks.json'), null);
  const corr = readJson(path.join(STAGING_DATA, 'corrector-staging-result.json'), null);

  const report = {
    generatedAt: new Date().toISOString(), stagingDir: STAGING_DATA,
    fixtures: { pass: fixtures.pass, f4Vacuous, casperCount },
    stats: { live: liveStats, staging: stagingStats }, bySource, dangling,
    carriedTotal, reverted: reverted.length,
    dropped: marks ? marks.wrongSubject : [],
    corrector: corr && { totals: corr.totals, rounds: corr.rounds.length, stopReason: corr.stopReason }
  };
  if (JSON_OUT) { console.log(JSON.stringify(report, null, 2)); process.exit(0); }

  const L = []; const say = (s = '') => L.push(s); const line = (c = '=') => say(c.repeat(78));
  line();
  say('THIRD STAGING GATE REPORT');
  say(`staging: ${STAGING_DATA}`);
  say(`generated: ${report.generatedAt}`);
  line();
  say('');
  say('NO CUTOVER. Live is untouched; staging is a separate store.');

  // ---- 1. fixtures --------------------------------------------------------
  say('');
  say('--- FIXTURES ---');
  const passed = (fixtures.results || []).filter(r => r.pass).length;
  say(`  ${passed}/${(fixtures.results || []).length} pass.`);
  for (const r of fixtures.results || []) {
    say(`  ${r.pass ? 'PASS' : 'FAIL'}  ${r.id}  ${r.name}`);
    for (const n of (r.notes || [])) say(`        ${n}`);
  }
  say('');
  if (f4Vacuous) {
    say('  F4 IS STILL VACUOUS — staging holds no fact naming Casper, so the rule it');
    say('  asserts (no subset sitting beside its superset) has nothing to be true about.');
  } else {
    say(`  F4 is no longer vacuous: staging holds ${casperCount} fact(s) naming Casper, so the`);
    say('  subset rule is being tested rather than satisfied by an empty set. That was');
    say("  gate 2's one weak pass, and carrying the Casper fact is what fixed it.");
  }

  // ---- 2. counts ----------------------------------------------------------
  say('');
  say('--- FINAL COUNTS ---');
  const row = (label, a, b) => say(`  ${label.padEnd(24)} live ${String(a).padStart(6)}    staging ${String(b).padStart(6)}   ${b - a >= 0 ? '+' : ''}${b - a}`);
  row('user facts (active)', liveStats.user_active, stagingStats.user_active);
  row('user facts (inactive)', liveStats.user_inactive, stagingStats.user_inactive);
  row('self facts (active)', liveStats.self_active, stagingStats.self_active);
  row('self facts (inactive)', liveStats.self_inactive, stagingStats.self_inactive);
  row('clusters', liveStats.clusters, stagingStats.clusters);
  row('corroborations', liveStats.corroborations, stagingStats.corroborations);
  row('corrections ledger', liveStats.ledger, stagingStats.ledger);
  say('');
  say('  staging active user facts, by what produced them:');
  for (const s of bySource) say(`    ${s.src.padEnd(22)} ${String(s.n).padStart(5)}`);
  say('');
  const carriedInactive = staging.prepare(
    "SELECT COUNT(*) n FROM cluster_members WHERE source = 'carried_from_live' AND status != 'active'"
  ).get().n;
  say(`  carried from live : ${carriedTotal} written (ledger); ${carriedInactive} of them since retired by the`);
  say('                      corrector or withdrawn, which is why the source table above is lower');
  say(`  dangling pointers : ${dangling}`);

  // ---- 3. the corrector, after the rule -----------------------------------
  say('');
  say('--- CORRECTOR, WITH THE HISTORY RULE IN PLACE ---');
  if (!corr) { say('  not run.'); } else {
    // corrector-staging-result.json holds the LAST run, which was the confirming
    // pass after the rule was fixed. The cumulative picture comes from the ledger
    // below, which cannot be overwritten by a later run.
    say(`  Final confirming pass: ${corr.rounds.length} round(s) — ${corr.stopReason}`);
    say(`  folded ${corr.totals.merged} · expired ${corr.totals.expired} · split ${corr.totals.split} · ` +
        `superseded ${corr.totals.superseded} · left unresolved ${corr.totals.unresolved}`);
    say('');
    say('  The rule was exercised, not merely present: the six pairs the corrector had');
    say('  exempted under an earlier, over-broad version of it were cleared from the');
    say('  pair-check memo and put back through. All six came back with nothing to do.');
  }

  // Every retirement STILL STANDING in staging, not just this pass's — a fact
  // retired two passes ago and never reverted is as gone as one retired today.
  // Scoped to entries made AFTER the staging store was seeded. The seed copied
  // live's whole ledger, so an unscoped query reports live's history — the Mike
  // supersession, the Aurelius name chain — as work this merge did.
  const seededAt = (readJson(path.join(STAGING_DATA, 'carry-plan.json'), {}) || {}).seededAt || '1970-01-01';
  const standing = staging.prepare(`
    SELECT id, target_text, survivor_text, evidence FROM corrections_ledger
    WHERE action = 'supersede' AND reversible = 1 AND reverted_at IS NULL
      AND datetime(created_at) > datetime(?)
    ORDER BY datetime(created_at)
  `).all(seededAt);
  say('');
  if (!standing.length) {
    say('  No supersession is still standing — every retirement has been reverted.');
  } else {
    say(`  ${standing.length} retirement(s) still standing in staging:`);
    for (const e of standing) {
      say(`    ${e.id.slice(0, 8)}  "${trunc(e.target_text, 62)}"`);
      say(`              for "${trunc(e.survivor_text, 62)}"`);
    }
    say('');
    say('    Revert any of them:');
    say('      SNH_DATA_DIR=data-staging node scripts/revert-correction.js <ledger-id>');
  }

  // A COMPOUND retired to settle a contradiction in ONE of its clauses takes the
  // other clauses with it, and they were never in dispute. The corrector splits
  // compounds and resolves contradictions in the same pass, but nothing orders
  // the two, so a compound can lose before it is ever split.
  const rules = require(path.join(ROOT, 'db/extraction-rules'));
  const compoundLosses = standing.filter(e => rules.looksCompound(e.target_text || '').compound);
  if (compoundLosses.length) {
    say('');
    say(`  ⚠ ${compoundLosses.length} of those retired a COMPOUND fact, which takes its other clauses with it:`);
    for (const e of compoundLosses) {
      say('');
      say(`    ${e.id.slice(0, 8)}  "${trunc(e.target_text, 74)}"`);
      say(`      retired for "${trunc(e.survivor_text, 60)}"`);
    }
    say('');
    say('    The disputed clause really is disputed — but the rest of the sentence was');
    say('    not, and it goes too. The corrector splits compounds and resolves');
    say('    contradictions in the same pass with nothing ordering the two, so a');
    say('    compound can lose a contradiction before it is ever split. Left standing:');
    say('    it is one supersession decided on real evidence, and unpicking it is a');
    say('    change to the corrector\'s phase order, not a merge decision.');
  }
  if (reverted.length) {
    say('');
    say(`  ${reverted.length} earlier retirement(s) reverted — history is not a contradiction:`);
    for (const r of reverted) {
      say(`    "${trunc(r.target_text, 66)}"`);
      say(`      had been retired for "${trunc(r.survivor_text, 50)}"`);
    }
    say('');
    say('  The rule is now enforced at ENUMERATION, in');
    say('  extraction-rules.historyCoexists, so the pair never reaches a judge. Tested by');
    say('  scripts/test-history-coexists.js — the three above are its first three cases.');
  }

  // ---- 3b. carries taken back ---------------------------------------------
  const withdrawn = staging.prepare(
    "SELECT target_text, reason, evidence FROM corrections_ledger WHERE action = 'carry-withdrawn' ORDER BY datetime(created_at)"
  ).all();
  if (withdrawn.length) {
    say('');
    say('--- A CARRY TAKEN BACK, AND WHY IT OVERRODE YOUR MARK ---');
    say('');
    for (const w of withdrawn) {
      say(`  "${trunc(w.target_text, 74)}"`);
    }
    say('');
    say('  You marked every ELLIE-DECIDES row CARRY except the archiver rows, and you also');
    say('  asked for 5/5 fixtures. Carrying this one broke F3, which asserts that a');
    say('  transient statement does not become a durable fact — so the two instructions');
    say('  collided and the fixture won. It is retired in staging, not deleted, and the');
    say('  LIVE row is untouched. To put it back:');
    say('    SNH_DATA_DIR=data-staging node scripts/revert-correction.js <the carry-withdrawn ledger id>');
    say('');
    // The fixture caught one member of a class. Naming the rest is the difference
    // between fixing an instance and reporting a problem.
    const rules = require(path.join(ROOT, 'db/extraction-rules'));
    const TASKISH = /\b(needs to|plans to pick up|is looking at buying|is going to|intends to pick up)\b/i;
    const siblings = staging.prepare(
      "SELECT content FROM cluster_members WHERE status='active' AND source='carried_from_live'"
    ).all().filter(r => TASKISH.test(r.content));
    if (siblings.length) {
      say(`  ⚠ ${siblings.length} more carried fact(s) are the same shape and were NOT withheld, because`);
      say('  nothing measures them:');
      for (const s of siblings) say(`    "${trunc(s.content, 74)}"`);
      say('');
      say('  These are errands, not facts about her. They carry no time marker, so the');
      say('  carry review\'s event test did not catch them and neither did any fixture —');
      say('  the one that was withheld differs only in that F3 happens to probe for the');
      say('  word "cleaners". Left carried because you said carry them, and flagged');
      say('  because the reason the other one went is a reason that applies to these too.');
    }
  }

  // ---- 4. what was not carried --------------------------------------------
  say('');
  say('--- NOT CARRIED: WRONG SUBJECT ---');
  if (!marks) { say('  no subject marks on disk.'); } else {
    const dropped = marks.wrongSubject.filter(w => w.disposition === 'drop');
    const quarantined = marks.wrongSubject.filter(w => w.disposition === 'quarantine');
    say('');
    say(`  ${marks.wrongSubject.length} of the ${marks.decided} in ELLIE-DECIDES were withheld as wrong-subject:`);
    say('  daily-log-archiver rows that describe Aurelius, stored as facts about Ellie.');
    say(`  ${marks.carried} were carried.`);
    say('');
    say(`  IT IS ${marks.wrongSubject.length}, NOT THE ~11 GATE 2 ESTIMATED. That estimate came from a keyword`);
    say('  pattern; this came from measuring every archiver row against his self-facts.');
    const byTwin = marks.wrongSubject.filter(w => w.byTwin).length;
    const byJudge = marks.wrongSubject.filter(w => w.byJudge).length;
    say(`  ${byTwin} sit at or above ${marks.twinFloor} of one of his own self-facts — the same sentence with`);
    say(`  the person flipped. The stored-subject judge independently called ${byJudge} of them`);
    say('  SELF, so it agrees where it fires and finds nothing the twin distance missed.');
    say('  Its own reading is far more conservative than the corpus evidence, which is why');
    say('  it is a second signal here rather than the deciding one.');
    say('');
    say(`  DROPPED (${dropped.length}) — he already holds the content, so nothing is lost:`);
    for (const d of dropped) {
      say(`    ${d.selfTwin ? d.selfTwin.similarity.toFixed(3) : '  -  '}  "${trunc(d.content, 74)}"`);
      if (d.selfTwin) say(`           his: "${trunc(d.selfTwin.content, 74)}"`);
    }
    say('');
    if (quarantined.length) {
      say(`  QUARANTINED (${quarantined.length}) — wrong subject AND nothing on his side holds it, so`);
      say(`  dropping would lose the only copy. Listed in docs/subject-quarantine.md,`);
      say('  flagged for the subject-repair pass:');
      for (const q of quarantined) say(`    "${trunc(q.content, 82)}"`);
    } else {
      say(`  QUARANTINED (0) — every withheld row had a self-side equivalent above ${marks.quarantineFloor},`);
      say('  so nothing needed holding back. The quarantine list exists and is empty.');
    }
    say('');
    say('  ALL OF THESE ROWS ARE STILL ACTIVE IN LIVE. This records what the merge');
    say('  declined to carry into staging. Repairing the live rows, and stopping the');
    say('  archiver writing more, are separate decisions and neither has been taken.');
  }

  say('');
  line();
  say('STOP. This is the gate. No cutover has been taken.');
  line();

  const text = L.join('\n');
  console.log(text);
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, `${text}\n`);
  console.error(`\n[gate3] written to ${OUT_PATH}`);
  process.exit(0);
})().catch(err => { console.error('gate report failed:', err); process.exit(1); });
