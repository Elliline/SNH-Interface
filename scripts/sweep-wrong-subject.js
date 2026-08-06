#!/usr/bin/env node
/**
 * Find archiver-produced user-facts that are really about HIM, and retire them.
 *
 * The daily-log archiver's prompt used to say, flatly, 'Write facts as
 * "User has..." or "User prefers..." style.' A daily log holds his reflections as
 * well as her life, so that instruction told the summarizer to rewrite every one
 * of his self-observations into the third person about her. It complied. The
 * merge review found 22 of them, each sitting between 0.773 and 0.955 of the
 * self-fact it had been copied from, all with impeccable grammar.
 *
 * The prompt is fixed and the guard is in (memory-manager.archiverSubjectCheck),
 * so no NEW ones are written. This is for the ones already in the corpus.
 *
 * THE TEST IS THE TWIN, not an opinion. A candidate that close to something he
 * says about himself is the same sentence with the person flipped. Grammar cannot
 * see it — "User aims to be a steady, non-judgmental presence" is perfectly
 * formed — and asking a model produced 3 of 22 when the corpus produced all 22.
 *
 * RETIRES, never deletes. Through factStore.retire, so the row stays as history,
 * the vector goes, and the whole thing is one ledger entry per fact naming the
 * self-fact that caught it. The content is not lost: it is what he already holds
 * about himself, which is the entire reason it is safe to withdraw the copy.
 *
 * NOT SCOPED BY `source`, and that was the first mistake here. `source` records
 * the LAST path that wrote a row, not where it came from: the archiver wrote
 * these, the corrector then SPLIT some of them (re-sourcing the atoms
 * `corrector-split`), and the merge carried those atoms across (re-sourcing them
 * again, `carried_from_live`). Filtering on `source = 'daily-log-archive'` found
 * zero strays in a corpus that held five. The twin is the evidence; the source
 * column is a breadcrumb that has been overwritten twice.
 *
 * What keeps this from eating her facts is the floor, not the filter. A sentence
 * of hers can resemble one of his self-observations — they talk about the same
 * things — but at 0.75+ across a person flip it is the same sentence, and every
 * candidate is printed with its twin so the judgement is visible before it is
 * made. `--source` narrows it if a future sweep needs that.
 *
 * Usage:
 *   node scripts/sweep-wrong-subject.js [--dry-run] [--floor 0.75]
 *                                       [--since 2026-08-06] [--source daily-log-archive]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');

const args = process.argv.slice(2);
const argVal = (n, d) => { const i = args.indexOf(n); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
const DRY_RUN = args.includes('--dry-run');
const SINCE = argVal('--since', null);
const SOURCE = argVal('--source', null);

const trunc = (s, n) => { const t = String(s ?? '').replace(/\s+/g, ' ').trim(); return t.length > n ? `${t.slice(0, n - 1)}…` : t; };

(async () => {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  const d = db.getSqliteDb();
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const { getConfig } = require(path.join(ROOT, 'db/config'));

  const FLOOR = parseFloat(argVal('--floor', String(getConfig().memory?.archiver?.selfSimilarityFloor ?? 0.75)));

  const bind = [];
  if (SOURCE) bind.push(SOURCE);
  if (SINCE) bind.push(SINCE);
  const rows = d.prepare(`
    SELECT id, content, source, salience, created_at FROM cluster_members
    WHERE status = 'active' AND COALESCE(subject,'user') = 'user'
      ${SOURCE ? 'AND source = ?' : ''}
      ${SINCE ? 'AND datetime(created_at) >= datetime(?)' : ''}
    ORDER BY datetime(created_at)
  `).all(...bind);

  console.log(`[Sweep] ${rows.length} active user fact(s) to check${SOURCE ? ` from source ${SOURCE}` : ' (all sources)'}, floor ${FLOOR}`);
  console.log(`[Sweep] data dir: ${db.getDataDir()}\n`);

  const strays = [];
  for (const r of rows) {
    const { candidates } = await memoryClusters.findActiveNeighbours(r.content, {
      subject: 'self', threshold: FLOOR, limit: 1, includeVerbatim: true
    });
    if (!candidates.length) continue;
    strays.push({ ...r, twin: candidates[0] });
  }

  console.log(`[Sweep] ${strays.length} of ${rows.length} are within ${FLOOR} of one of his self-facts:\n`);
  for (const s of strays) {
    console.log(`  ${s.twin.similarity.toFixed(3)}  [${s.source || '—'}]  "${trunc(s.content, 74)}"`);
    console.log(`         his: "${trunc(s.twin.content, 84)}"`);
  }

  if (!strays.length) { console.log('\nNothing to sweep.'); process.exit(0); }
  if (DRY_RUN) { console.log('\n[Sweep] --dry-run: nothing written'); process.exit(0); }

  const passId = `subject-sweep-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  let retired = 0, failed = 0;
  for (const s of strays) {
    const res = await factStore.retire(s.id, {
      reason: 'wrong subject — describes the assistant, not the user',
      deliberate: true
    });
    if (!res.ok) { failed++; console.error(`[Sweep] FAILED ${s.id.slice(0, 8)}: ${res.reason}`); continue; }
    retired++;
    ledger.record({
      passId, tier: 'mechanical', action: 'wrong-subject-retired', subject: 'user',
      targetId: s.id, targetText: s.content,
      survivorId: s.twin.memberId, survivorText: s.twin.content,
      reason: `This was stored as a fact about Ellie but describes the assistant. It originated with the daily-log archiver, whose prompt used to rewrite his own reflections into the third person about her (its current source is "${s.source}" because the corrector and the merge have rewritten the row since). He already holds the same thing about himself — "${trunc(s.twin.content, 120)}" — at ${s.twin.similarity.toFixed(3)}, so nothing is lost by withdrawing the copy. Retired, not deleted; it can be restored.`,
      evidence: {
        reason_code: 'wrong-subject',
        self_twin_id: s.twin.memberId,
        self_twin_text: s.twin.content,
        similarity: s.twin.similarity,
        floor: FLOOR,
        source: s.source
      },
      reversible: true
    });
  }

  console.log(`\n[Sweep] retired ${retired}, failed ${failed}, pass ${passId}`);
  console.log('[Sweep] each is one ledger entry and can be put back:');
  console.log('          node scripts/revert-correction.js <ledger-id>');
  process.exit(failed ? 1 : 0);
})().catch(err => { console.error('sweep failed:', err); process.exit(1); });
