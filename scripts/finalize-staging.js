#!/usr/bin/env node
/**
 * The two cosmetic-looking loose ends the replay left, settled — in staging.
 *
 * The gate report raised both as AMBIGUOUS, deliberately undecided by the replay:
 * 72 clusters with no active member, and 82 inactive facts pointing at a
 * successor that no longer exists. Ellie's decision at the gate was prune the
 * clusters and, for pointers still dangling after the merge, leave the retirement
 * reason alone and record that the pointer cannot be resolved.
 *
 * Runs POINTED AT STAGING and refuses to run against live.
 *
 * ---------------------------------------------------------------------------
 * 1. EMPTY CLUSTERS — and "empty" means empty, not "no active member"
 *
 * The report counted clusters holding no ACTIVE member. That is the right thing
 * to report and the wrong thing to delete on. `memoryClusters.getCluster` returns
 * inactive members on purpose — the Memory Map draws them as ghosts — so a
 * cluster whose members are all retired is still holding history that a person
 * can open. Deleting it would orphan those rows against the cluster_id foreign
 * key and take the ghosts off the map.
 *
 * So the prune is on clusters with NO members at all. Both counts are reported,
 * because the difference between them is exactly the set this refuses to touch.
 *
 * ---------------------------------------------------------------------------
 * 2. DANGLING SUCCESSORS — repaired where the merge made repair possible
 *
 * A dangling pointer says "this was replaced by something that is no longer
 * here". After the replay that was simply true: every active user fact had been
 * discarded and rebuilt, so 82 chains ended at a deleted row.
 *
 * The carry changes that for some of them. A fact carried in from live is the
 * SAME assertion as the discarded row a pointer was aiming at — it is the row,
 * brought across — so the pointer has somewhere real to land again. The carry
 * result records live-id → staging-id for every fact it wrote, which is the map.
 *
 * Repointing goes through `factStore.repoint`, the narrowest write in the funnel
 * and the one built for exactly this shape of defect: a supersession that was
 * correct to make, pointing at the wrong (or a vanished) winner. It touches
 * successor_id and superseded_by and nothing else — no status change, no vector,
 * so an inactive fact cannot be made briefly active and retrievable on the way
 * through. It is a deliberate path, which is why it is invoked from a script a
 * person ran and never from anything automatic.
 *
 * What is left after that is genuinely unresolvable: the pointer named a fact the
 * replay did not rebuild and the merge did not carry. Those keep
 * inactive_reason = 'superseded' — the retirement happened and was correct, and
 * rewriting it to 'retracted' would claim she withdrew a fact she never touched —
 * and each one gets a ledger entry saying the pointer leads nowhere and why, with
 * reversible = 0 because nothing changed.
 *
 * Usage:
 *   node scripts/finalize-staging.js [--dry-run] [--json]
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

const LIVE_DATA = path.join(ROOT, 'data');
const STAGING_DATA = process.env.SNH_STAGING_DIR || path.join(ROOT, 'data-staging');
process.env.SNH_DATA_DIR = STAGING_DATA;

const args = process.argv.slice(2);
const DRY_RUN = args.includes('--dry-run');
const JSON_OUT = args.includes('--json');

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

/**
 * live member id -> staging member id, for every fact any carry run brought over.
 *
 * Read from the LEDGER, not from the carry runs' JSON result files. Every carry
 * writes a ledger row naming the live id it came from (`target_id`) and the
 * staging row it became (`survivor_id`), and that row is written inside the same
 * step that did the write — so the ledger cannot disagree with the corpus. The
 * result files are a convenience, and one of them has already been overwritten
 * by a second run once; a map built from those silently loses whatever the last
 * run did not repeat, and a successor chain does not get repaired because a file
 * was rewritten.
 */
function carryMap(d) {
  const rows = d.prepare(
    "SELECT target_id, survivor_id FROM corrections_ledger WHERE action = 'carry' AND survivor_id IS NOT NULL"
  ).all();
  return new Map(rows.map(r => [r.target_id, r.survivor_id]));
}

(async () => {
  if (path.resolve(STAGING_DATA) === path.resolve(LIVE_DATA)) {
    console.error('ABORT: staging dir resolves to the live data dir.');
    process.exit(2);
  }
  if (!fs.existsSync(path.join(STAGING_DATA, 'chat.db'))) {
    console.error(`ABORT: no staging store at ${STAGING_DATA}.`);
    process.exit(2);
  }

  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  if (path.resolve(db.getDataDir()) !== path.resolve(STAGING_DATA)) {
    console.error(`ABORT: data dir is ${db.getDataDir()}, expected ${STAGING_DATA}`);
    process.exit(2);
  }
  const d = db.getSqliteDb();
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

  const passId = `finalize-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  const out = { passId, dryRun: DRY_RUN, clusters: {}, successors: {} };

  // -------------------------------------------------------------------------
  // 1. Clusters
  // -------------------------------------------------------------------------
  const noActive = d.prepare(`
    SELECT c.id, c.name, c.subject,
           (SELECT COUNT(*) FROM cluster_members m WHERE m.cluster_id = c.id) AS members
    FROM memory_clusters c
    WHERE NOT EXISTS (SELECT 1 FROM cluster_members m WHERE m.cluster_id = c.id AND m.status = 'active')
    ORDER BY c.name
  `).all();
  const trulyEmpty = noActive.filter(c => c.members === 0);
  const ghostsOnly = noActive.filter(c => c.members > 0);

  // memory_clusters is referenced by cluster_links as well as by cluster_members,
  // and a cluster with no members can still have links — 25 of them did, and the
  // first prune died on the foreign key. A link naming a cluster that is being
  // removed is dead weight: nothing can render it and nothing can traverse it. It
  // goes with the cluster, in the same transaction, and the count is reported
  // rather than absorbed. cluster_link_judgments is a memo table with no foreign
  // key; its rows for a removed cluster are stale but harmless, and clearing them
  // would throw away judgements about clusters that still exist on the other side.
  const emptyIds = new Set(trulyEmpty.map(c => c.id));
  const doomedLinks = emptyIds.size
    ? d.prepare('SELECT id, cluster_a, cluster_b FROM cluster_links').all()
      .filter(l => emptyIds.has(l.cluster_a) || emptyIds.has(l.cluster_b))
    : [];

  out.clusters = {
    total: d.prepare('SELECT COUNT(*) n FROM memory_clusters').get().n,
    noActiveMember: noActive.length,
    trulyEmpty: trulyEmpty.length,
    ghostsOnly: ghostsOnly.length,
    pruned: 0,
    linksRemoved: 0,
    linksToRemove: doomedLinks.length,
    ghostsKept: ghostsOnly.map(c => ({ id: c.id, name: c.name, inactiveMembers: c.members }))
  };

  if (!DRY_RUN && trulyEmpty.length) {
    const delLink = d.prepare('DELETE FROM cluster_links WHERE id = ?');
    const delCluster = d.prepare('DELETE FROM memory_clusters WHERE id = ?');
    const tx = d.transaction((links, rows) => {
      for (const l of links) delLink.run(l.id);
      for (const c of rows) delCluster.run(c.id);
    });
    tx(doomedLinks, trulyEmpty);
    out.clusters.pruned = trulyEmpty.length;
    out.clusters.linksRemoved = doomedLinks.length;
    console.log(`[Finalize] pruned ${trulyEmpty.length} cluster(s) with no members at all, and ${doomedLinks.length} link(s) that named one`);
  } else if (doomedLinks.length) {
    console.log(`[Finalize] ${doomedLinks.length} cluster link(s) name a cluster that would be pruned — they would go with it`);
  }
  if (ghostsOnly.length) {
    console.log(`[Finalize] KEPT ${ghostsOnly.length} cluster(s) that hold only inactive members — the Map draws those as ghosts`);
  }

  // A cluster row is gone, but a vector's cluster_id is not covered by the
  // foreign key. reconcile() is the detector for embeddings left naming a
  // cluster that no longer exists; report it rather than assume.
  const { counts } = await factStore.reconcile();
  out.clusters.staleClusterVectorsAfter = counts.staleClusterVectors;

  // -------------------------------------------------------------------------
  // 2. Dangling successors
  // -------------------------------------------------------------------------
  const danglingSql = `
    SELECT m.id, m.content, m.successor_id, m.superseded_by, m.inactive_reason, m.subject
    FROM cluster_members m
    WHERE m.successor_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM cluster_members s WHERE s.id = m.successor_id)
    ORDER BY m.subject, datetime(m.created_at)
  `;
  const danglingBefore = d.prepare(danglingSql).all();
  const map = carryMap(d);
  console.log(`[Finalize] ${danglingBefore.length} dangling successor pointer(s); carry map holds ${map.size} live→staging id(s)`);

  const repaired = [];
  const stillDangling = [];

  for (const row of danglingBefore) {
    const target = map.get(row.successor_id);
    if (!target) { stillDangling.push(row); continue; }
    if (DRY_RUN) { repaired.push({ id: row.id, from: row.successor_id, to: target, dryRun: true }); continue; }

    const res = await factStore.repoint(row.id, target, { deliberate: true });
    if (!res.ok) {
      console.warn(`[Finalize] repoint refused for ${row.id.slice(0, 8)}: ${res.reason}`);
      stillDangling.push(row);
      continue;
    }
    repaired.push({ id: row.id, from: row.successor_id, to: target });
    // An earlier run may have recorded this pointer as unresolvable — it was, at
    // the time, and a later carry made it resolvable. Leaving that entry behind
    // would put a ledger row saying "this leads nowhere" beside a pointer that
    // now leads somewhere, which is a false statement about the corpus in the one
    // place that is supposed to be reliable about it.
    try {
      const stale = d.prepare(
        "DELETE FROM corrections_ledger WHERE action = 'dangling-successor' AND target_id = ?"
      ).run(row.id);
      if (stale.changes) console.log(`[Finalize] cleared ${stale.changes} stale dangling-successor record(s) for ${row.id.slice(0, 8)} — its pointer resolves again`);
    } catch (err) { console.warn(`[Finalize] could not clear stale record: ${err.message}`); }
    // The repoint filed its own entry (fact-store funnel, same transaction —
    // 2026-08-18). Enriched, not filed again: one change, one entry.
    ledger.enrich(res.ledgerId, {
      passId, tier: 'mechanical', action: 'repoint', subject: row.subject || 'user',
      survivorId: target, survivorText: null,
      reason: 'This fact had been replaced by a newer one, and the rebuild discarded the newer one before writing the corpus again — so the record said "replaced by something that is no longer here". The fact it pointed at was carried back in from the live corpus, so the pointer now leads to it again. The retirement itself is unchanged.',
      evidence: { previous_successor: row.successor_id, new_successor: target, resolved_by: 'carried_from_live' }
    });
  }

  // What is left cannot be repaired. Record it, once each, and change nothing.
  const alreadyRecorded = new Set(
    d.prepare("SELECT target_id FROM corrections_ledger WHERE action = 'dangling-successor'").all().map(r => r.target_id)
  );
  let recorded = 0;
  const reasons = {};
  for (const row of stillDangling) {
    reasons[row.inactive_reason || '(null)'] = (reasons[row.inactive_reason || '(null)'] || 0) + 1;
    if (DRY_RUN || alreadyRecorded.has(row.id)) continue;
    ledger.record({
      passId, tier: 'mechanical', action: 'dangling-successor', subject: row.subject || 'user',
      targetId: row.id, targetText: row.content,
      reason: `This fact was replaced by a newer one, and the newer one is not in the rebuilt corpus — the replay did not re-derive it from source and the merge did not carry it. The record of the replacement is kept as it is: the retirement was correct and still stands, the pointer just leads nowhere. Following the chain from here stops at this fact.`,
      evidence: { successor_id: row.successor_id, inactive_reason: row.inactive_reason, unresolved_by: 'replay' },
      // Nothing changed. A refusal is not a correction.
      reversible: false
    });
    recorded++;
  }

  const danglingAfter = d.prepare(danglingSql).all().length;
  out.successors = {
    before: danglingBefore.length,
    repaired: repaired.length,
    stillDangling: stillDangling.length,
    after: danglingAfter,
    ledgerEntriesWritten: recorded,
    inactiveReasons: reasons
  };

  // The instruction is that a still-dangling pointer keeps inactive_reason =
  // 'superseded'. Nothing above writes that column — this asserts it rather than
  // trusting it, because a row that drifted to some other reason would be a
  // different claim about what happened to it.
  const wrongReason = stillDangling.filter(r => r.inactive_reason && r.inactive_reason !== 'superseded');
  out.successors.notSuperseded = wrongReason.map(r => ({ id: r.id, reason: r.inactive_reason, text: trunc(r.content, 80) }));

  // Accumulate across runs, same reason as the carry's result file: this script
  // is idempotent, so running it again reports 0 pruned and 0 repaired — which is
  // true of THAT run and false about the corpus. The totals are what a reader
  // wants, and overwriting hides them.
  const RESULT = path.join(STAGING_DATA, 'finalize-result.json');
  const prior = (() => {
    try {
      const old = JSON.parse(fs.readFileSync(RESULT, 'utf8'));
      return old.runs ? old : { runs: [old], totals: null };
    } catch { return { runs: [], totals: null }; }
  })();
  const runs = [...prior.runs, out];
  const sum = (pick) => runs.reduce((a, r) => a + (pick(r) || 0), 0);
  fs.writeFileSync(RESULT, JSON.stringify({
    latest: out,
    totals: {
      clustersPruned: sum(r => r.clusters && r.clusters.pruned),
      linksRemoved: sum(r => r.clusters && r.clusters.linksRemoved),
      successorsRepaired: sum(r => r.successors && r.successors.repaired),
      danglingLedgerEntries: sum(r => r.successors && r.successors.ledgerEntriesWritten)
    },
    runs
  }, null, 2));

  if (JSON_OUT) { console.log(JSON.stringify(out, null, 2)); process.exit(0); }

  const line = (c = '=') => console.log(c.repeat(78));
  console.log('');
  line();
  console.log(`FINALIZE STAGING${DRY_RUN ? ' — DRY RUN, nothing written' : ''}`);
  line();

  console.log('\n--- CLUSTERS ---');
  console.log(`  clusters in staging          : ${out.clusters.total}`);
  console.log(`  holding no ACTIVE member     : ${out.clusters.noActiveMember}`);
  console.log(`    of those, holding nothing  : ${out.clusters.trulyEmpty}   → pruned ${out.clusters.pruned}`);
  console.log(`    of those, holding ghosts   : ${out.clusters.ghostsOnly}   → kept, the Map draws these`);
  console.log(`  cluster links naming a pruned cluster: ${out.clusters.linksToRemove}   → removed ${out.clusters.linksRemoved}`);
  console.log(`  embeddings naming a deleted cluster afterwards: ${out.clusters.staleClusterVectorsAfter}`);
  if (ghostsOnly.length) {
    console.log('\n  kept (inactive members only):');
    for (const c of out.clusters.ghostsKept.slice(0, 15)) {
      console.log(`    "${c.name}" — ${c.inactiveMembers} retired fact(s)`);
    }
    if (out.clusters.ghostsKept.length > 15) console.log(`    … and ${out.clusters.ghostsKept.length - 15} more`);
  }

  console.log('\n--- DANGLING SUCCESSOR POINTERS ---');
  console.log(`  before this run              : ${out.successors.before}`);
  console.log(`  repaired by the carry map    : ${out.successors.repaired}`);
  console.log(`  still dangling               : ${out.successors.stillDangling}`);
  console.log(`  counted again afterwards     : ${out.successors.after}`);
  console.log(`  ledger entries written       : ${out.successors.ledgerEntriesWritten}`);
  console.log(`  inactive_reason on those left: ${JSON.stringify(out.successors.inactiveReasons)}`);
  if (wrongReason.length) {
    console.log(`\n  ${wrongReason.length} still-dangling row(s) are NOT marked superseded — look at these:`);
    for (const r of out.successors.notSuperseded.slice(0, 10)) console.log(`    [${r.reason}] "${r.text}"`);
  } else {
    console.log('  every still-dangling row is marked superseded, as intended');
  }

  console.log('');
  line();
  process.exit(0);
})().catch(err => { console.error('finalize failed:', err); process.exit(1); });
