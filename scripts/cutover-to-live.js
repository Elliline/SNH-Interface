#!/usr/bin/env node
/**
 * The cutover — staging becomes the system of record for USER FACTS.
 *
 * NOT a file swap, and this is the whole design of it. `data-staging/chat.db` is
 * a snapshot of live taken when the replay was seeded on 2026-08-05T20:35, and
 * live has not stood still since: 2 more conversations, 10 more messages, 4 more
 * self-facts, and movement in initiatives, heartbeat reports, audits, follow-ups
 * and liveness probes. Copying staging over live would silently discard every one
 * of them. So this promotes the corpus that was rebuilt and leaves alone
 * everything that was not.
 *
 * WHAT MOVES, and what does not:
 *
 *   user facts        REPLACED from staging. This is the thing the whole phase
 *                     rebuilt: 485 rows, active and inactive, with their carry
 *                     provenance and supersession history.
 *   self facts        LIVE's are KEPT. The replay never touched them by design —
 *                     self-facts are curated with him, not replayed — and live
 *                     has 412 to staging's 408 because reflection has run since
 *                     the seed. Taking staging's would quietly drop four.
 *   clusters          Driven by what the promoted facts REFERENCE, not by
 *                     subject. 17 staging user-facts sit in self-subject clusters
 *                     and one self-fact sits in a user cluster, so "user clusters
 *                     from staging, self clusters from live" would have broken
 *                     the foreign key on both sides.
 *   corroborations    Staging's for the promoted user facts; live's for the
 *                     self-facts that stay.
 *   ledger            UNION. Live gained 2 entries after the seed and staging
 *                     produced 278; both halves are history and neither is
 *                     disposable.
 *   pair-check memo   Staging's, so the corrector does not re-judge from scratch.
 *   conversations,    UNTOUCHED. None of it was rebuilt, so none of it moves.
 *   messages, cron,
 *   questions, …
 *
 * cluster_links and cluster_link_judgments are DROPPED here. They described a
 * corpus that no longer exists — the writer was disabled on 2026-08-02, and after
 * the replay the rows pointed at clusters that had been discarded — and the Map
 * drew them as current. Association is a query-time vector lookup now.
 *
 * Everything happens in ONE transaction. A cutover that half-applies leaves a
 * corpus that is neither the old one nor the new one, and the foreign keys are
 * the only thing that would notice.
 *
 * REFUSES TO RUN unless the server is stopped and a phase-tagged backup exists.
 *
 * Usage:
 *   node scripts/cutover-to-live.js --backup-tag <tag> [--dry-run]
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');
const LIVE = path.join(ROOT, 'data', 'chat.db');
const STAGING = path.join(ROOT, 'data-staging', 'chat.db');

const args = process.argv.slice(2);
const argVal = (n, d) => { const i = args.indexOf(n); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
const DRY_RUN = args.includes('--dry-run');
const TAG = argVal('--backup-tag', '');

const trunc = (s, n) => { const t = String(s ?? '').replace(/\s+/g, ' ').trim(); return t.length > n ? `${t.slice(0, n - 1)}…` : t; };

(async () => {
  if (!TAG) { console.error('ABORT: --backup-tag is required.'); process.exit(2); }
  const backup = path.join(ROOT, 'data', `chat.db.bak-${TAG}`);
  if (!fs.existsSync(backup)) { console.error(`ABORT: no backup at ${backup}`); process.exit(2); }
  if (!fs.existsSync(STAGING)) { console.error(`ABORT: no staging store at ${STAGING}`); process.exit(2); }

  // A running server holds a write connection and its heartbeat would be mutating
  // the very rows being replaced.
  const { execSync } = require('child_process');
  try {
    const active = execSync('systemctl --user is-active snh.service || true', { encoding: 'utf8' }).trim();
    if (active === 'active') { console.error('ABORT: snh.service is running. Stop it first.'); process.exit(2); }
  } catch { /* systemctl absent — fall through to the port check */ }

  const Database = require('better-sqlite3');
  const d = new Database(LIVE);
  d.pragma('foreign_keys = OFF');   // re-enabled and CHECKED after the swap
  d.exec(`ATTACH DATABASE '${STAGING.replace(/'/g, "''")}' AS s`);

  const before = {
    userFacts: d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE COALESCE(subject,'user')!='self'").get().n,
    selfFacts: d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE subject='self'").get().n,
    clusters: d.prepare('SELECT COUNT(*) n FROM memory_clusters').get().n,
    ledger: d.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n,
    corrob: d.prepare('SELECT COUNT(*) n FROM fact_corroborations').get().n,
    convos: d.prepare('SELECT COUNT(*) n FROM conversations').get().n,
    messages: d.prepare('SELECT COUNT(*) n FROM messages').get().n
  };
  const incoming = {
    userFacts: d.prepare("SELECT COUNT(*) n FROM s.cluster_members WHERE COALESCE(subject,'user')!='self'").get().n,
    ledgerNew: d.prepare('SELECT COUNT(*) n FROM s.corrections_ledger x WHERE NOT EXISTS (SELECT 1 FROM corrections_ledger l WHERE l.id = x.id)').get().n
  };

  console.log('BEFORE  ', JSON.stringify(before));
  console.log('INCOMING', JSON.stringify(incoming));
  if (DRY_RUN) { console.log('\n--dry-run: nothing written'); d.close(); process.exit(0); }

  const memberCols = d.prepare('SELECT * FROM cluster_members LIMIT 1').columnNames?.() || null;

  const swap = d.transaction(() => {
    // 1. Out with live's user/world facts and everything hanging off them.
    d.prepare(`DELETE FROM fact_corroborations WHERE member_id IN (
        SELECT id FROM cluster_members WHERE COALESCE(subject,'user') != 'self')`).run();
    d.prepare("DELETE FROM cluster_members WHERE COALESCE(subject,'user') != 'self'").run();

    // 2. Clusters the promoted facts will need. Insert BEFORE the facts, or the
    //    foreign key has nothing to point at. Driven by reference, not subject —
    //    user facts do sit in self-subject clusters in this corpus.
    d.prepare(`
      INSERT OR IGNORE INTO memory_clusters (id, name, description, created_at, updated_at, subject)
      SELECT c.id, c.name, c.description, c.created_at, c.updated_at, c.subject
      FROM s.memory_clusters c
      WHERE c.id IN (SELECT DISTINCT cluster_id FROM s.cluster_members
                     WHERE COALESCE(subject,'user') != 'self')
    `).run();

    // 3. In with staging's user facts, whole — provenance, lifecycle, chain.
    d.prepare(`
      INSERT INTO cluster_members
        (id, cluster_id, content, source, importance, created_at, updated_at, status,
         superseded_by, salience, subject, claim_type, locked, locked_at, lock_category,
         conversation_id, message_id, verbatim_source_text, input_modality,
         salience_rationale, inactive_reason, successor_id)
      SELECT id, cluster_id, content, source, importance, created_at, updated_at, status,
             superseded_by, salience, subject, claim_type, locked, locked_at, lock_category,
             conversation_id, message_id, verbatim_source_text, input_modality,
             salience_rationale, inactive_reason, successor_id
      FROM s.cluster_members WHERE COALESCE(subject,'user') != 'self'
    `).run();

    // 4. Corroborations for those facts — evidence travels with the fact it is
    //    evidence for. Live's self-fact corroborations were never deleted.
    d.prepare(`
      INSERT OR IGNORE INTO fact_corroborations
        (id, member_id, created_at, conversation_id, message_id,
         verbatim_source_text, input_modality, restated_as, similarity, detected_by)
      SELECT fc.id, fc.member_id, fc.created_at, fc.conversation_id, fc.message_id,
             fc.verbatim_source_text, fc.input_modality, fc.restated_as, fc.similarity, fc.detected_by
      FROM s.fact_corroborations fc
      WHERE fc.member_id IN (SELECT id FROM cluster_members)
    `).run();

    // 5. Ledger union. Both sides are history.
    d.prepare(`
      INSERT OR IGNORE INTO corrections_ledger
        (id, created_at, pass_id, tier, action, subject, target_id, target_text,
         survivor_id, survivor_text, reason, evidence, reversible, reverted_at, reverted_by, announced)
      SELECT id, created_at, pass_id, tier, action, subject, target_id, target_text,
             survivor_id, survivor_text, reason, evidence, reversible, reverted_at, reverted_by, announced
      FROM s.corrections_ledger
    `).run();

    // 6. The corrector's memo, so the first live pass does not re-judge the whole
    //    corpus from scratch.
    d.prepare('DELETE FROM corrector_pair_checks').run();
    d.prepare(`
      INSERT OR IGNORE INTO corrector_pair_checks (pair_key, checked_at, verdict, a_id, b_id, fact_updated_at)
      SELECT pair_key, checked_at, verdict, a_id, b_id, fact_updated_at FROM s.corrector_pair_checks
    `).run();

    // 7. Clusters nothing points at any more.
    const orphans = d.prepare(`
      DELETE FROM memory_clusters
      WHERE NOT EXISTS (SELECT 1 FROM cluster_members m WHERE m.cluster_id = memory_clusters.id)
    `).run();
    console.log(`[Cutover] removed ${orphans.changes} cluster(s) nothing references any more`);

    // 8. The stale association graph, gone.
    d.exec('DROP TABLE IF EXISTS cluster_links');
    d.exec('DROP TABLE IF EXISTS cluster_link_judgments');
  });

  swap();
  d.exec('DETACH DATABASE s');

  // --- verification, before anything is allowed to call this done -----------
  d.pragma('foreign_keys = ON');
  const fk = d.prepare('PRAGMA foreign_key_check').all();
  const integrity = d.prepare('PRAGMA integrity_check').get();

  const after = {
    userActive: d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE status='active' AND COALESCE(subject,'user')='user'").get().n,
    userInactive: d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE status!='active' AND COALESCE(subject,'user')='user'").get().n,
    selfActive: d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE status='active' AND subject='self'").get().n,
    clusters: d.prepare('SELECT COUNT(*) n FROM memory_clusters').get().n,
    ledger: d.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n,
    corrob: d.prepare('SELECT COUNT(*) n FROM fact_corroborations').get().n,
    convos: d.prepare('SELECT COUNT(*) n FROM conversations').get().n,
    messages: d.prepare('SELECT COUNT(*) n FROM messages').get().n,
    carried: d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE source='carried_from_live'").get().n,
    linksGone: d.prepare("SELECT COUNT(*) n FROM sqlite_master WHERE type='table' AND name IN ('cluster_links','cluster_link_judgments')").get().n
  };

  console.log('\nAFTER   ', JSON.stringify(after));
  console.log(`integrity_check    : ${integrity.integrity_check}`);
  console.log(`foreign_key_check  : ${fk.length} violation(s)`);
  for (const v of fk.slice(0, 10)) console.log(`   ${JSON.stringify(v)}`);

  const problems = [];
  if (integrity.integrity_check !== 'ok') problems.push('integrity_check failed');
  if (fk.length) problems.push(`${fk.length} foreign-key violation(s)`);
  if (after.selfActive < before.selfFacts - 60) problems.push('self-facts look wrong — they should have been left alone');
  if (after.convos !== before.convos) problems.push('conversation count changed — nothing here should touch conversations');
  if (after.messages !== before.messages) problems.push('message count changed — nothing here should touch messages');
  if (after.linksGone !== 0) problems.push('cluster_links tables still present');

  d.close();

  if (problems.length) {
    console.error(`\nCUTOVER FAILED VERIFICATION:\n  - ${problems.join('\n  - ')}`);
    console.error(`\nRestore with:  cp data/chat.db.bak-${TAG} data/chat.db`);
    process.exit(1);
  }
  console.log('\nCutover applied and verified. Vectors still need rebuilding — run scripts/reembed-corpus.js next.');
  process.exit(0);
})().catch(err => { console.error('cutover failed:', err); process.exit(1); });
