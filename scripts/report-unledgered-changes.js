#!/usr/bin/env node
/**
 * What was changed before the ledger became part of the write — a report, and
 * only a report.
 *
 * WHY IT EXISTS. Until 2026-08-18 the corrections-ledger entry was filed by each
 * CALLER, and most callers never filed one. Measured on the live corpus that
 * day: 68 self-facts stood superseded and not one carried a `supersede` entry.
 * Every one of those is a change `correctionsLedger.revert()` cannot touch,
 * because revert works by reading an entry and there is nothing to read. The
 * funnel fixes that going forward (see db/fact-store.js). This says what the past
 * looks like.
 *
 * THE TWO COUNTS ARE DIFFERENT QUESTIONS, and this report answers the one that
 * matters. "68 with no supersede entry" counts one action. This counts rows with
 * no entry AT ALL, by any action — which is what Revert actually looks for — and
 * comes out lower (61 self supersessions on the same corpus), because some of
 * those rows were later touched by a merge or an expiry that did file one. If
 * the two numbers ever disagree in a report you are reading, that is why.
 *
 * ⚠ IT CHANGES NOTHING, AND THAT IS THE DESIGN RATHER THAN A LIMITATION.
 *
 * The obvious next thought — "re-run the judge over the 68 and put back the ones
 * that were wrong" — is the defect proposing to audit itself. The mechanism that
 * made those decisions is the same one that retired a salience-9 declaration on
 * a 0.741 cosine match and, on identical input at 0.857, answered "no" about half
 * the time. Handing it the whole history would produce a fresh set of confident
 * wrong answers on top of the old ones, and the wrong ones would this time be
 * ledgered, which makes them look considered.
 *
 * So: a READONLY connection, no writes anywhere in this file, and the only
 * action it offers is a command a person can choose to run, one row at a time.
 *
 * WHAT IT FLAGS. Every inactive row with no ledger entry, and within them the
 * ones the bars added on 2026-08-18 would refuse today — a `declaration`, or
 * salience at or above identity.protectSelfFactSalience. Those are the changes
 * that, made now, would have been raised for a person instead of applied.
 *
 * Bar 2 (evidence dominance as a veto) is deliberately NOT evaluated
 * retrospectively. It reads provenance the historical corpus mostly does not
 * carry, so a retrospective verdict from it would be an artifact of missing data
 * rather than a finding — and the point of this report is to be trustworthy
 * about a corpus that has already been decided about carelessly once.
 *
 * Usage:
 *   node scripts/report-unledgered-changes.js
 *   node scripts/report-unledgered-changes.js --all       # every row, not just flagged
 *   node scripts/report-unledgered-changes.js --subject self
 *   node scripts/report-unledgered-changes.js --json
 */

const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
const { getConfig } = require(path.join(ROOT, 'db/config'));

const ARGS = process.argv.slice(2);
const SHOW_ALL = ARGS.includes('--all');
const AS_JSON = ARGS.includes('--json');
const SUBJECT = (() => {
  const i = ARGS.indexOf('--subject');
  return i > -1 ? ARGS[i + 1] : null;
})();

const DB_PATH = path.join(database.getDataDir(), 'chat.db');

function trim(s, n) {
  const t = String(s || '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

function main() {
  if (!fs.existsSync(DB_PATH)) {
    console.error(`No database at ${DB_PATH}`);
    process.exit(1);
  }
  // READONLY. This file must not be able to change anything even by accident.
  const db = new Database(DB_PATH, { readonly: true });

  const protectSalience = Number.isFinite(getConfig().identity?.protectSelfFactSalience)
    ? getConfig().identity.protectSelfFactSalience : 8;

  const where = ["m.status = 'inactive'"];
  const params = [];
  if (SUBJECT) { where.push('m.subject = ?'); params.push(SUBJECT); }

  // "Unledgered" = no corrections_ledger row points at this fact, by any action.
  // A supersession the corrector made has one; the ones the intake paths made do
  // not, and those are what this finds.
  const rows = db.prepare(`
    SELECT m.id, m.content, m.subject, m.salience, m.claim_type, m.inactive_reason,
           m.successor_id, m.source, m.created_at, m.updated_at,
           c.name AS cluster_name,
           s.content AS successor_content, s.status AS successor_status, s.salience AS successor_salience
    FROM cluster_members m
    LEFT JOIN memory_clusters c ON c.id = m.cluster_id
    LEFT JOIN cluster_members s ON s.id = m.successor_id
    WHERE ${where.join(' AND ')}
      AND NOT EXISTS (SELECT 1 FROM corrections_ledger l WHERE l.target_id = m.id)
    ORDER BY datetime(m.updated_at) DESC
  `).all(...params);

  const ledgeredTotal = db.prepare(`
    SELECT COUNT(*) n FROM cluster_members m
    WHERE m.status = 'inactive'
      AND EXISTS (SELECT 1 FROM corrections_ledger l WHERE l.target_id = m.id)
  `).get().n;

  const flagged = rows.filter(r =>
    r.subject === 'self' && (r.claim_type === 'declaration' || (r.salience ?? 5) >= protectSalience));

  if (AS_JSON) {
    console.log(JSON.stringify({
      generated_at: new Date().toISOString(),
      unledgered: rows.length,
      ledgered: ledgeredTotal,
      protect_salience: protectSalience,
      flagged: flagged.map(r => ({
        id: r.id, content: r.content, subject: r.subject, salience: r.salience,
        claim_type: r.claim_type, reason: r.inactive_reason, at: r.updated_at,
        successor_id: r.successor_id, successor_content: r.successor_content
      })),
      rows: SHOW_ALL ? rows : undefined
    }, null, 2));
    db.close();
    return;
  }

  // ---- the report --------------------------------------------------------
  console.log('\n════════════════════════════════════════════════════════════════════');
  console.log(' Changes made before the ledger was part of the write');
  console.log('════════════════════════════════════════════════════════════════════\n');
  console.log(`Inactive facts with NO ledger entry : ${rows.length}`);
  console.log(`Inactive facts that do have one     : ${ledgeredTotal}`);
  console.log('\n"No entry" means no corrections_ledger row points at the fact by ANY action —');
  console.log('which is the question that matters, because that is what Revert looks for. A');
  console.log('count of one action alone (say, supersede) reads higher: some of these rows');
  console.log('were later touched by a merge or an expiry that did file one.');
  console.log('\nThe ones with no entry cannot be undone through the Self tab\'s Revert or');
  console.log('scripts/revert-correction.js — both work by reading an entry, and there is');
  console.log('nothing for them to read. Everything written since 2026-08-18 has one.\n');

  // By subject and reason.
  const by = {};
  for (const r of rows) {
    const k = `${r.subject || '(none)'} / ${r.inactive_reason || '(none)'}`;
    by[k] = (by[k] || 0) + 1;
  }
  console.log('Breakdown:');
  for (const [k, v] of Object.entries(by).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${k.padEnd(28)} ${v}`);
  }

  if (rows.length) {
    const dates = rows.map(r => r.updated_at).filter(Boolean).sort();
    console.log(`\nSpanning ${String(dates[0]).slice(0, 10)} to ${String(dates[dates.length - 1]).slice(0, 10)}.`);
  }

  // ---- the flagged ones --------------------------------------------------
  console.log('\n────────────────────────────────────────────────────────────────────');
  console.log(` FLAGGED: self-facts that today's bars would NOT have taken (${flagged.length})`);
  console.log('────────────────────────────────────────────────────────────────────');
  const decl = flagged.filter(r => r.claim_type === 'declaration').length;
  const salient = flagged.filter(r => (r.salience ?? 5) >= protectSalience).length;
  const both = flagged.filter(r => r.claim_type === 'declaration' && (r.salience ?? 5) >= protectSalience).length;
  const top = flagged.filter(r => (r.salience ?? 5) >= 9 && r.claim_type === 'declaration');
  console.log(`A declaration, or salience ${protectSalience}+. Made now, each of these would have been`);
  console.log('raised for you to decide instead of applied. Listed newest first.\n');
  console.log(`  declarations: ${decl}    salience ${protectSalience}+: ${salient}    both: ${both}`);
  if (top.length) console.log(`  of those, ${top.length} are salience-9-or-higher declarations — the sharpest cases.`);
  console.log('');

  if (!flagged.length) {
    console.log('  (none)\n');
  }
  for (const r of flagged) {
    const why = [];
    if (r.claim_type === 'declaration') why.push('a declaration — something he said about himself');
    if ((r.salience ?? 5) >= protectSalience) why.push(`salience ${r.salience}`);
    console.log(`  ${r.id.slice(0, 8)}  ${String(r.updated_at).slice(0, 10)}  ${r.inactive_reason}  [${why.join(', ')}]`);
    console.log(`      cluster : ${r.cluster_name || '(none)'}   source: ${r.source || '(none)'}`);
    console.log(`      RETIRED : "${trim(r.content, 150)}"`);
    if (r.successor_id) {
      console.log(`      REPLACED BY (${String(r.successor_id).slice(0, 8)}, ${r.successor_status}, salience ${r.successor_salience}):`);
      console.log(`                "${trim(r.successor_content, 150)}"`);
    } else {
      console.log('      REPLACED BY: nothing — it was retired or expired outright');
    }
    console.log(`      put it back: node scripts/restore-self-fact.js ${r.id} --confirm`);
    console.log('');
  }

  // ---- everything else ---------------------------------------------------
  const rest = rows.filter(r => !flagged.includes(r));
  if (SHOW_ALL) {
    console.log('────────────────────────────────────────────────────────────────────');
    console.log(` EVERYTHING ELSE (${rest.length})`);
    console.log('────────────────────────────────────────────────────────────────────\n');
    for (const r of rest) {
      console.log(`  ${r.id.slice(0, 8)}  ${String(r.updated_at).slice(0, 10)}  ${r.subject}/${r.inactive_reason}  sal ${r.salience}  ${r.claim_type || '-'}`);
      console.log(`      "${trim(r.content, 130)}"`);
      if (r.successor_content) console.log(`      → "${trim(r.successor_content, 130)}"`);
    }
    console.log('');
  } else if (rest.length) {
    console.log(`(${rest.length} more unledgered rows are not flagged — ordinary observations and`);
    console.log(' user facts. Run with --all to see them.)\n');
  }

  // The sharpest cases, again, at the bottom — because the flagged list is long
  // enough to scroll past and these are the ones worth a person's attention.
  const sharpest = flagged.filter(r => (r.salience ?? 5) >= 9 && r.claim_type === 'declaration');
  if (sharpest.length) {
    console.log('────────────────────────────────────────────────────────────────────');
    console.log(` THE SHARPEST CASES (${sharpest.length}) — salience-9+ declarations, no way back except by hand`);
    console.log('────────────────────────────────────────────────────────────────────');
    for (const r of sharpest) {
      console.log(`  ${String(r.updated_at).slice(0, 10)}  "${trim(r.content, 120)}"`);
      console.log(`      node scripts/restore-self-fact.js ${r.id} --confirm`);
    }
    console.log('');
  }

  console.log('────────────────────────────────────────────────────────────────────');
  console.log(' This report changed nothing, and nothing here runs by itself.');
  console.log('────────────────────────────────────────────────────────────────────');
  console.log(' Re-judging these automatically was considered and rejected: the mechanism');
  console.log(' that made them is the one that retired a salience-9 declaration on a 0.741');
  console.log(' match and, at 0.857, said "no" about half the time on identical input.');
  console.log(' Pointing it at the whole history would produce fresh confident mistakes —');
  console.log(' and this time they would be ledgered, which makes them look considered.');
  console.log(' Restoring is one row at a time, by a person, with the command above.\n');

  db.close();
}

main();
