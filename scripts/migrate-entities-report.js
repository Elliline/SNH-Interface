#!/usr/bin/env node
/**
 * Run the entities migration against THIS instance's store and report what it
 * did — counts before and after, anything that failed to repoint, and a
 * READ-ONLY audit of what the write-time subject check would say about facts
 * that are already here.
 *
 * WHY A REPORT AND NOT JUST A RESTART. The migration runs inside
 * initDatabase(), so restarting the service would perform it silently and the
 * only account of what happened to a live corpus would be a log line. A
 * migration on a real store is reported, per *A restore is not done until the
 * ENTITY can see it* — the numbers are the evidence, and they are taken before
 * and after rather than inferred.
 *
 * THE AUDIT HALF CHANGES NOTHING. It runs the check over facts that were
 * written long before the gate existed and prints what it would have said.
 * Re-judging them automatically is the defect auditing itself — the same
 * reasoning as scripts/report-unledgered-changes.js. Ellie decides.
 *
 * Usage:
 *   node scripts/migrate-entities-report.js            # migrate + report
 *   node scripts/migrate-entities-report.js --audit-only
 */
const fs = require('fs');
const path = require('path');
const Database = require(path.join(__dirname, '..', 'node_modules/better-sqlite3'));

const ROOT = path.join(__dirname, '..');
const AUDIT_ONLY = process.argv.includes('--audit-only');

function counts(db) {
  const q = (sql, ...a) => { try { return db.prepare(sql).get(...a).n; } catch { return null; } };
  return {
    facts: q('SELECT COUNT(*) n FROM cluster_members'),
    active: q("SELECT COUNT(*) n FROM cluster_members WHERE status = 'active'"),
    subjectUser: q("SELECT COUNT(*) n FROM cluster_members WHERE subject = 'user'"),
    subjectSelf: q("SELECT COUNT(*) n FROM cluster_members WHERE subject = 'self'"),
    subjectOther: q("SELECT COUNT(*) n FROM cluster_members WHERE subject NOT IN ('user','self') OR subject IS NULL"),
    withEntity: q('SELECT COUNT(*) n FROM cluster_members WHERE subject_entity_id IS NOT NULL'),
    clusters: q('SELECT COUNT(*) n FROM memory_clusters'),
    entities: q('SELECT COUNT(*) n FROM entities'),
    ledger: q('SELECT COUNT(*) n FROM corrections_ledger'),
    lockedMembers: q('SELECT COUNT(*) n FROM cluster_members WHERE locked = 1'),
    entityLocks: q('SELECT COUNT(*) n FROM entity_locks'),
    flagged: q("SELECT COUNT(*) n FROM cluster_members WHERE status = 'flagged-unverified-subject'")
  };
}

function show(label, c) {
  console.log(`\n${label}`);
  for (const [k, v] of Object.entries(c)) console.log(`  ${k.padEnd(14)} ${v === null ? '(table not present)' : v}`);
}

(async () => {
  const dataDir = process.env.SNH_DATA_DIR
    ? path.resolve(process.env.SNH_DATA_DIR) : path.join(ROOT, 'data');
  const dbPath = path.join(dataDir, 'chat.db');
  console.log(`Store: ${dbPath}`);

  // ---- 1. BACKUP, from a readonly connection, no checkpoint. ----
  if (!AUDIT_ONLY) {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, 'Z');
    const backup = path.join(dataDir, `chat.db.bak-entities-${stamp}`);
    if (fs.existsSync(backup)) { console.error('backup already exists, refusing'); process.exit(1); }
    const ro = new Database(dbPath, { readonly: true, fileMustExist: true });
    ro.prepare('VACUUM INTO ?').run(backup);
    ro.close();
    const size = fs.statSync(backup).size;
    console.log(`\nBACKUP: ${path.basename(backup)} (${(size / 1048576).toFixed(1)} MB)`);
    console.log(`ROLLBACK: cp ${backup} ${dbPath}   # with the service stopped`);
  }

  const db = new Database(dbPath, { fileMustExist: true });
  db.pragma('busy_timeout = 15000');

  const before = counts(db);
  show('BEFORE', before);

  let summary = null;
  if (!AUDIT_ONLY) {
    // ---- 2. MIGRATE. ----
    console.log('\n--- running entities.initSchema ---');
    summary = require(path.join(ROOT, 'db/entities')).initSchema(db);
    const after = counts(db);
    show('AFTER', after);

    console.log('\nWHAT THE MIGRATION DID');
    console.log(`  founding entities : ${summary.foundingEntities.map(e => `${e.kind}="${e.name}"`).join(', ')}`);
    console.log(`  columns added     : ${summary.createdColumns.length ? summary.createdColumns.join(', ') : '(already present)'}`);
    console.log(`  repointed         : user=${summary.repointed.user ?? 0}, self=${summary.repointed.self ?? 0}`);
    console.log(`  UNREPOINTED       : ${summary.unrepointed}`);
    console.log(`  locks carried     : ${summary.locksMigrated}`);
    console.log(`  self name synced  : ${summary.selfNameSync ? `${summary.selfNameSync.from} -> ${summary.selfNameSync.to}` : '(already in step)'}`);

    if (summary.unrepointed > 0) {
      console.log('\n  Rows that could not be repointed mechanically:');
      for (const r of db.prepare(
        "SELECT id, subject, status, substr(content,1,80) c FROM cluster_members WHERE subject_entity_id IS NULL LIMIT 25").all()) {
        console.log(`    ${r.id.slice(0, 8)} subject=${JSON.stringify(r.subject)} [${r.status}] "${r.c}"`);
      }
    }
  }

  // ---- 3. READ-ONLY AUDIT of the existing corpus against the new gate. ----
  const subjectCheck = require(path.join(ROOT, 'db/subject-check'));
  // Read the registry through THIS script's handle. db/entities.js resolves its
  // own connection from db/database.js, which this script never initialised —
  // going through the module here would read a null handle.
  const hydrate = (row) => {
    let aliasList = [];
    try { const a = JSON.parse(row.aliases || '[]'); if (Array.isArray(a)) aliasList = a; } catch {}
    return { id: row.id, name: row.name, type: row.type, aliasList };
  };
  const known = db.prepare("SELECT * FROM entities WHERE status = 'active'").all().map(hydrate);
  const ptr = (kind) => {
    const p = db.prepare('SELECT entity_id FROM entity_pointers WHERE kind = ?').get(kind);
    if (!p) return null;
    const row = db.prepare('SELECT * FROM entities WHERE id = ?').get(p.entity_id);
    return row ? hydrate(row) : null;
  };
  const userEnt = ptr('user');
  const selfEnt = ptr('self');

  const rows = db.prepare(
    "SELECT id, content, subject, salience, status, verbatim_source_text FROM cluster_members WHERE status = 'active'"
  ).all();

  let inScope = 0, wouldFlag = 0;
  const flagged = [];
  for (const r of rows) {
    const ent = r.subject === 'self' ? selfEnt : userEnt;
    if (!ent) continue;
    const v = subjectCheck.check(r.content, ent, r.verbatim_source_text || '',
      { knownEntities: known, sourceIsUserMessage: r.subject !== 'self' });
    if (v.reason === 'no-specifics') continue;
    inScope++;
    if (!v.verified) { wouldFlag++; flagged.push({ r, v }); }
  }

  console.log('\n--- READ-ONLY AUDIT: what the gate says about facts already here ---');
  console.log('    (nothing below was changed; existing facts keep the status they have)');
  console.log(`  active facts            : ${rows.length}`);
  console.log(`  carrying specifics      : ${inScope}`);
  console.log(`  would NOT verify today  : ${wouldFlag}`);
  const byReason = {};
  for (const f of flagged) byReason[f.v.reason] = (byReason[f.v.reason] || 0) + 1;
  console.log(`  by reason               : ${Object.entries(byReason).map(([k, v]) => `${k}=${v}`).join(', ') || '(none)'}`);

  const withSource = flagged.filter(f => (f.r.verbatim_source_text || '').trim());
  console.log(`\n  Of those, ${withSource.length} HAVE a source to check against — these are the real signal:`);
  for (const f of withSource.slice(0, 15)) {
    console.log(`    ${f.r.id.slice(0, 8)} sal=${f.r.salience} [${f.v.reason}]`);
    console.log(`        "${f.r.content.slice(0, 110)}"`);
    console.log(`        ${f.v.detail}`);
  }
  if (withSource.length > 15) console.log(`    … and ${withSource.length - 15} more`);

  db.close();
  console.log('\nDone. Nothing was restored, retired or re-judged.');
})().catch(e => { console.error('FAILED:', e.stack || e.message); process.exit(1); });
