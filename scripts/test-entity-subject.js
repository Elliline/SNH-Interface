#!/usr/bin/env node
/**
 * ENTITY SUBJECT + WRITE-TIME SUBJECT CHECK — against a THROWAWAY store.
 *
 * Same isolation as scripts/test-identity-lock.js: redirect SNH_DATA_DIR before
 * anything resolves a store, seed a corpus, throw the directory away. Live data
 * is never opened.
 *
 * THE TWO REGRESSIONS THIS EXISTS FOR are sections 3a and 3b — the two facts
 * that were live in Juno's store on 2026-09-01, both produced by the same
 * Lincoln City Animal Clinic email, neither catchable by any audit because
 * they contradicted nothing. They must die at the door.
 *
 * No engine calls: every assertion here is deterministic, so a red result is a
 * real regression and never a slow GPU.
 *
 * Usage: node scripts/test-entity-subject.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const ROOT = path.join(__dirname, '..');
// The runner passes a fresh temp dir (baseline dataDir: 'required'); standalone
// runs make their own. Only the directory this process created is removed.
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-entity-subject-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  if (INHERITED) return;
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {}
});

const db = require(path.join(ROOT, 'db/database'));

let passed = 0, failed = 0;
function ok(name, cond, detail = '') {
  if (cond) { passed++; console.log(`  PASS  ${name}`); }
  else { failed++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}
function section(t) { console.log(`\n=== ${t} ===`); }

// The source email, verbatim from Juno's store (cluster_members
// .verbatim_source_text on 16ee64d8 / 620c841f). Every specific in the two bad
// facts traces to the signature block at the bottom.
const CLINIC_EMAIL_SOURCE = `I got this email today and I need a script written so i can give it to elevenlabs to give me a audio file and i can upload it to there phone system. Here is the email:

Good morning Ellie,

I forgot to send a reminder email Monday about our phones needing to be shut off at 1:00pm today for our open house! I am so sorry for the short notice!!

Thank you so much!
Adrienne
PS hope to see you at the open house. Its between 3-7pm. We have goodies and a raffle!

4090 NE HWY 101
Lincoln City, OR 97367
Phone: (541) 994-8181
email: lcac2009@live.com`;

(async () => {
  db.initDatabase();

  const entities = require(path.join(ROOT, 'db/entities'));
  const subjectCheck = require(path.join(ROOT, 'db/subject-check'));
  const sql = db.getSqliteDb();

  // ---------------------------------------------------------------- 1. schema
  section('1. Schema and the handle-not-a-profile invariant');

  ok('entities table exists', !!sql.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='entities'").get());
  ok('entity_pointers table exists', !!sql.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='entity_pointers'").get());
  ok('entity_locks table exists', !!sql.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='entity_locks'").get());

  let indexOnly = false;
  try { indexOnly = entities.assertIndexOnly(); } catch (e) { indexOnly = e.message; }
  ok('entities table is INDEX ONLY (no knowledge columns)', indexOnly === true,
     typeof indexOnly === 'string' ? indexOnly : '');

  // The invariant has teeth: adding a knowledge column must fail the assertion.
  sql.exec('ALTER TABLE entities ADD COLUMN address TEXT');
  let caught = null;
  try { entities.assertIndexOnly(); } catch (e) { caught = e.message; }
  ok('a knowledge column makes assertIndexOnly throw', !!caught && /address/.test(caught), caught || 'did not throw');
  // Put it back the way it was; sqlite can drop a column since 3.35.
  try { sql.exec('ALTER TABLE entities DROP COLUMN address'); } catch { /* older sqlite: leave it */ }

  ok('cluster_members.subject_entity_id exists',
     sql.prepare('PRAGMA table_info(cluster_members)').all().some(c => c.name === 'subject_entity_id'));

  // -------------------------------------------------------------- 2. founding
  section('2. Founding entities and mechanical repoint');

  const self = entities.selfEntity();
  const user = entities.userEntity();
  ok('self entity exists', !!self && self.type === 'self');
  ok('user entity exists', !!user && user.type === 'user');
  ok('self and user are different entities', self && user && self.id !== user.id);

  // Seed legacy facts carrying only the old subject strings, then re-run the
  // migration and assert it repoints them mechanically.
  // cluster_members.cluster_id is a real foreign key, so the seed makes real
  // clusters. Named ones are created on demand and reused.
  const ensureCluster = (name) => {
    const existing = sql.prepare('SELECT id FROM memory_clusters WHERE name = ?').get(name);
    if (existing) return existing.id;
    const id = randomUUID();
    sql.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)')
      .run(id, name, 'seeded by test-entity-subject', new Date().toISOString(), new Date().toISOString());
    return id;
  };
  const seed = (content, subject, clusterName = 'test-cluster') => {
    const id = randomUUID();
    sql.prepare(`INSERT INTO cluster_members (id, cluster_id, content, subject, status, salience, created_at)
                 VALUES (?, ?, ?, ?, 'active', 5, ?)`)
      .run(id, ensureCluster(clusterName), content, subject, new Date().toISOString());
    return id;
  };
  const legacyUser = seed('User keeps a spare key under the mat', 'user');
  const legacySelf = seed('I prefer concrete examples', 'self');
  sql.prepare('UPDATE cluster_members SET subject_entity_id = NULL WHERE id IN (?, ?)').run(legacyUser, legacySelf);

  const summary = entities.initSchema(sql);
  const uRow = sql.prepare('SELECT subject_entity_id FROM cluster_members WHERE id = ?').get(legacyUser);
  const sRow = sql.prepare('SELECT subject_entity_id FROM cluster_members WHERE id = ?').get(legacySelf);
  ok('a legacy user fact repoints to the user entity', uRow.subject_entity_id === user.id);
  ok('a legacy self fact repoints to the self entity', sRow.subject_entity_id === self.id);
  ok('migration reports what it repointed', summary.repointed.user >= 1 && summary.repointed.self >= 1,
     JSON.stringify(summary.repointed));
  ok('nothing is left unrepointed', summary.unrepointed === 0, `unrepointed=${summary.unrepointed}`);

  // ------------------------------------------------- 3. THE TWO JUNO CASES
  section('3. Regressions: the two facts that were live in Juno\'s store');

  const ish = entities.create({ name: 'Inn at Spanish Head', type: 'org', aliases: ['ISH'], relationship: 'client' });
  const lcac = entities.create({
    name: 'Lincoln City Animal Clinic', type: 'org',
    aliases: ['LCAC', 'the clinic'], relationship: 'client'
  });
  const known = [ish, lcac, user, self].map(subjectCheck.forEntity);

  // 3a. The address fact. Subject was the user; the claim names ISH and carries
  // the clinic's street address. ISH appears NOWHERE in the source.
  const badAddress =
    'User works at ISH (Inn At Spanish Head), a business located at 4090 NE Hwy 101, Lincoln City, OR 97367, ' +
    'and her role is effectively IT manager, involving leading teams and mediating between vendors like IMB and Shift4.';
  const r3a = subjectCheck.check(badAddress, subjectCheck.forEntity(user), CLINIC_EMAIL_SOURCE, { knownEntities: known });
  ok('JUNO CASE 1 — the ISH street address does NOT verify', r3a.verified === false, JSON.stringify(r3a.detail));
  ok('JUNO CASE 1 — names the unattributable specific', /4090|Spanish Head|ISH/i.test(r3a.detail || ''), r3a.detail);

  // The same claim filed against ISH itself must also fail: the source never
  // mentions Inn at Spanish Head at all.
  const r3aIsh = subjectCheck.check(
    'Inn at Spanish Head is located at 4090 NE Hwy 101, Lincoln City, OR 97367',
    subjectCheck.forEntity(ish), CLINIC_EMAIL_SOURCE, { knownEntities: known });
  ok('JUNO CASE 1b — the address does not verify against ISH either', r3aIsh.verified === false, r3aIsh.detail);

  // 3b. The business-email fact. The address is in the source, but the source
  // never connects it to Ellie — it is the sender's signature block.
  const badEmail = "User's business email is lcac2009@live.com";
  const r3b = subjectCheck.check(badEmail, subjectCheck.forEntity(user), CLINIC_EMAIL_SOURCE, { knownEntities: known });
  ok('JUNO CASE 2 — the clinic email as HER business email does NOT verify', r3b.verified === false, r3b.detail);
  ok('JUNO CASE 2 — reason is an attribution failure, not a missing source',
     ['no-attribution', 'competing-attribution'].includes(r3b.reason), r3b.reason);

  // And the same email filed against the clinic is fine — it IS the clinic's.
  const goodEmail = subjectCheck.check(
    "Lincoln City Animal Clinic's email is lcac2009@live.com",
    subjectCheck.forEntity(lcac), CLINIC_EMAIL_SOURCE, { knownEntities: known });
  ok('the clinic email DOES verify against the clinic', goodEmail.verified === true, goodEmail.detail);

  // ------------------------------------------------------ 4. no false alarms
  section('4. The gate must not flag ordinary true facts');

  const c1 = subjectCheck.check(
    'User works at ISH (Inn At Spanish Head)', subjectCheck.forEntity(user),
    'I plan on going upstairs to get a Chicken and Salad. Im working at ISH (Inn At Spanish Head) today and they have a restraunt named Fathoms.',
    { knownEntities: known });
  ok('"works at ISH" verifies from the message where she said it', c1.verified === true, c1.detail);

  const c2 = subjectCheck.check(
    'User is based in Lincoln City, Oregon', subjectCheck.forEntity(user),
    'Yeah, we are based in Lincoln City Oregon. and thank you for not saying im so sorry',
    { knownEntities: known });
  ok('"based in Lincoln City" verifies from first person', c2.verified === true, c2.detail);

  const c3 = subjectCheck.check(
    'User prefers to be told directly when she has made a mistake',
    subjectCheck.forEntity(user), 'just tell me straight when i get it wrong', { knownEntities: known });
  ok('a preference with no attributable specifics is out of scope', c3.verified === true, c3.reason);

  const c4 = subjectCheck.check(
    'I tend to over-explain when I am unsure', subjectCheck.forEntity(self),
    'you over-explain when you are not sure of something', { knownEntities: known });
  ok('a self-observation from second-person address is in scope and passes', c4.verified === true, c4.detail);

  const c5 = subjectCheck.check(
    "User's phone is (541) 994-8181", subjectCheck.forEntity(user), '', { knownEntities: known });
  ok('a specific with NO source at all does not verify', c5.verified === false && c5.reason === 'no-source', c5.reason);

  // ------------------------------------------------------- 5. retrieval by id
  section('5. Entity retrieval does not depend on cluster state');

  const f1 = seed('Inn at Spanish Head runs Shift4 for payments', 'user', 'hotel-cluster');
  sql.prepare('UPDATE cluster_members SET subject_entity_id = ? WHERE id = ?').run(ish.id, f1);
  ok('getFactsForEntity finds it', entities.getFactsForEntity(ish.id).some(f => f.id === f1));

  // Juno's visibility failure was a fact filed under the wrong cluster, which
  // made it unreachable by every cluster-shaped path. cluster_id is NOT NULL,
  // so the faithful reproduction is a MISFILED cluster, not a missing one.
  sql.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?')
    .run(ensureCluster('somewhere-completely-unrelated'), f1);
  ok('still found when filed under the WRONG cluster (Juno\'s visibility failure)',
     entities.getFactsForEntity(ish.id).some(f => f.id === f1));
  ok('and the wrong cluster is genuinely unrelated',
     sql.prepare('SELECT cluster_id FROM cluster_members WHERE id = ?').get(f1).cluster_id
       === ensureCluster('somewhere-completely-unrelated'));

  // ------------------------------------------------------------ 6. resolution
  section('6. Tiered resolution, no approval queue');

  ok('exact alias is a clear match', entities.resolve('ISH').tier === 'clear');
  ok('containment is a clear match', entities.resolve('Inn at Spanish Head').tier === 'clear');
  ok('an unknown name is new', entities.resolve('Newport Dental').tier === 'new');

  entities.create({ name: 'Bob', type: 'person', relationship: 'client contact' });
  entities.create({ name: 'Bobby Chen', type: 'person', relationship: 'vendor' });
  const bob = entities.resolve('Bob');
  ok('a bare first name with other people known is AMBIGUOUS (blast radius)',
     bob.tier === 'ambiguous', `${bob.tier}: ${bob.why}`);

  // ----------------------------------------------------------- 7. entity ops
  section('7. Merge is union-preserving and ledgered');

  const dupe = entities.create({ name: 'Spanish Head Inn', type: 'org', aliases: ['SHI'] });
  const dFact = seed('Spanish Head Inn replaced its APs in August', 'user');
  sql.prepare('UPDATE cluster_members SET subject_entity_id = ? WHERE id = ?').run(dupe.id, dFact);
  entities.lockEntity(dupe.id, 'name', dFact);

  const before = entities.getFactsForEntity(ish.id).length;
  const merged = entities.merge(dupe.id, ish.id, { reason: 'the same hotel under two names' });
  ok('the absorbed entity keeps no facts', entities.getFactsForEntity(dupe.id).length === 0);
  ok('every fact moved, none retired', entities.getFactsForEntity(ish.id).length === before + 1);
  ok('the loser name survives as an alias', entities.namesOf(merged.survivor).includes('spanish head inn'));
  ok('the loser alias survives too', entities.namesOf(merged.survivor).includes('shi'));
  ok('the lock carried over rather than being re-derived', entities.isEntityLocked(ish.id, 'name'));
  ok('the merge is in the corrections ledger',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='entity-merge' AND target_id=?").get(dupe.id));
  ok('the merged entity is marked, not deleted',
     entities.get(dupe.id).status === 'merged' && entities.get(dupe.id).merged_into === ish.id);

  // ------------------------------------------------------------- 8. identity
  section('8. Per-entity locks and the self pointer');

  let ptrErr = null;
  try { entities.setPointer('self', ish.id, { actor: 'test' }); } catch (e) { ptrErr = e.message; }
  ok('the self pointer refuses to move outside the identity-lock path', !!ptrErr, ptrErr || 'it moved');
  ok('the self pointer still points at the self entity', entities.selfEntity().id === self.id);

  const lockedMember = seed('I am named Athena, a name Ellie gave me.', 'self');
  sql.prepare("UPDATE cluster_members SET locked=1, lock_category='name', subject_entity_id=? WHERE id=?")
    .run(self.id, lockedMember);
  entities.lockEntity(self.id, 'name', lockedMember);
  let repointErr = null;
  try { entities.repointFact(lockedMember, ish.id); } catch (e) { repointErr = e.message; }
  ok('a lock-holding fact cannot be repointed to another entity', !!repointErr, repointErr || 'it repointed');

  // A repoint of an ordinary fact is allowed, and ledgered.
  const ordinary = seed('User mentioned the lobby wifi is slow', 'user');
  sql.prepare('UPDATE cluster_members SET subject_entity_id = ? WHERE id = ?').run(user.id, ordinary);
  entities.repointFact(ordinary, ish.id, { reason: 'this was about the hotel, not about her' });
  ok('an ordinary fact repoints', sql.prepare('SELECT subject_entity_id FROM cluster_members WHERE id=?').get(ordinary).subject_entity_id === ish.id);
  ok('the repoint is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='entity-repoint' AND target_id=?").get(ordinary));

  // Before the sync, pointer and locked name disagree — and the check must SAY
  // so. A check that only validated the name fact would miss exactly this.
  const beforeSync = entities.identityAgrees();
  ok('the identity check DETECTS a pointer/name disagreement', beforeSync.ok === false, beforeSync.why);

  // The entity name is derived from the locked fact, so syncing settles it.
  entities.syncSelfName(sql);
  const agree = entities.identityAgrees();
  ok('after deriving the name from the locked fact, pointer and name agree', agree.ok === true, agree.why);
  ok('the self entity is now findable by its locked name', entities.resolve('Athena').tier === 'clear');
  ok('the previous entity name survives as an alias, never dropped',
     entities.namesOf(entities.selfEntity()).length >= 1);

  // ---------------------------------------------------------------- results
  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => {
  console.error('\nCRASH:', e.stack || e.message);
  process.exit(1);
});
