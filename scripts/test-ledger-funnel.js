#!/usr/bin/env node
/**
 * The invariant: if the row changed, an entry exists — and there is exactly one.
 *
 * WHY THESE CHECKS. Every one of them is a rule whose failure is invisible until
 * someone tries to undo something and finds they cannot. Measured on the live
 * corpus on 2026-08-18: 68 self-fact supersessions, 0 ledger entries, so 68
 * changes that `revert()` could not touch — because the ledger call lived in
 * each CALLER and most callers never made it. "Every caller remembers" is not an
 * invariant; it is a hope, and it had failed every single time.
 *
 * So the entry is filed by the write, in the same transaction as the write, and
 * these tests hold that line from both sides: no change without an entry, and no
 * second entry for one change.
 *
 * Two of them are regression tests for the FIRST attempt at this fix, which was
 * reverted: `reword` and `repoint` referenced an `opts` their signatures did not
 * bind, so they threw a ReferenceError AFTER the row had been written — a
 * written row with no entry, which is precisely the defect being fixed, arriving
 * through the fix itself.
 *
 * No model calls, no network: every decision under test is deterministic, so
 * these tests are too. Runs against a throwaway SNH_DATA_DIR.
 *
 * Usage: node scripts/test-ledger-funnel.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-ledger-funnel-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail !== undefined ? ` — ${detail}` : ''}`); }
}

(async () => {
  database.initDatabase();
  // The vector store is opened so the write paths take their real branches —
  // dropVector/replaceVector are part of what they do, and a null handle would
  // quietly test a different function than the one that runs in production.
  await database.initVectorStore();

  const db = database.getSqliteDb();
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

  const clusterId = randomUUID();
  db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
    .run(clusterId, 'Test', '', new Date().toISOString(), new Date().toISOString(), 'self');

  let n = 0;
  function seed(content, { subject = 'self', status = 'active', salience = 5, claimType = 'claim', successor = null } = {}) {
    const id = randomUUID();
    const now = new Date().toISOString();
    db.prepare(`
      INSERT INTO cluster_members
        (id, cluster_id, content, source, created_at, updated_at, status, subject, salience, claim_type,
         inactive_reason, successor_id, verbatim_source_text, input_modality)
      VALUES (?,?,?,'test',?,?,?,?,?,?,?,?,?,'typed')
    `).run(id, clusterId, content, now, now, status, subject, salience, claimType,
      status === 'active' ? null : 'superseded', successor, content);
    n++;
    return id;
  }
  const row = (id) => db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
  const entriesFor = (id) => db.prepare('SELECT * FROM corrections_ledger WHERE target_id = ? ORDER BY created_at').all(id);
  const allEntries = () => db.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n;

  console.log(`\nLedger-funnel tests (throwaway data dir: ${TMP})\n`);

  // =========================================================================
  console.log('── Every write that changes a row files exactly one entry ──');

  const oldFact = seed('the old belief');
  const newFact = seed('the new belief');
  const sup = await factStore.supersede(oldFact, newFact);
  check('supersede reports ok and hands back a ledger id', sup.ok && !!sup.ledgerId, JSON.stringify(sup));
  let e = entriesFor(oldFact);
  check('…exactly one entry exists for it', e.length === 1, `${e.length} entries`);
  check('…recording the action, the subject and both sides',
    e[0].action === 'supersede' && e[0].subject === 'self' && e[0].survivor_id === newFact && e[0].target_text === 'the old belief',
    JSON.stringify({ action: e[0].action, subject: e[0].subject, survivor: !!e[0].survivor_id }));
  check('…marked reversible, because restore() can undo it', e[0].reversible === 1);
  check('…and its id is the one the caller was handed', e[0].id === sup.ledgerId);

  const toRetire = seed('a fact to retire');
  const ret = await factStore.retire(toRetire, { reason: 'no longer true of me' });
  check('retire files one entry', ret.ok && entriesFor(toRetire).length === 1);
  check('…and keeps the caller\'s reason as evidence',
    /no longer true of me/.test(entriesFor(toRetire)[0].evidence || ''), entriesFor(toRetire)[0].evidence);

  const toExpire = seed('a fact that was really an event');
  const exp = await factStore.expire(toExpire);
  check('expire files one entry', exp.ok && entriesFor(toExpire).length === 1);
  check('…as the expire action', entriesFor(toExpire)[0].action === 'expire');

  // =========================================================================
  console.log('\n── reword and repoint: the regression that was reverted ──');
  // Both took a destructured `{ deliberate }` and then referenced `opts`, so
  // they threw AFTER writing the row. The row changed and nothing recorded it.

  const toReword = seed('the wording before');
  let threw = null;
  let rw;
  try { rw = await factStore.reword(toReword, 'the wording after'); } catch (err) { threw = err; }
  check('reword does not throw', !threw, threw && threw.message);
  check('…the row really changed', row(toReword).content === 'the wording after', row(toReword).content);
  check('…and one entry exists for it', entriesFor(toReword).length === 1, `${entriesFor(toReword).length}`);
  const rwEntry = entriesFor(toReword)[0];
  check('…holding the wording before AND after',
    rwEntry.target_text === 'the wording before' && rwEntry.survivor_text === 'the wording after',
    JSON.stringify({ before: rwEntry.target_text, after: rwEntry.survivor_text }));
  check('…and marked NOT reversible, because revert() cannot undo a reword',
    rwEntry.reversible === 0, String(rwEntry.reversible));

  const survivorA = seed('successor A');
  const survivorB = seed('successor B');
  const pointed = seed('a retired fact', { status: 'inactive', successor: survivorA });
  threw = null;
  let rp;
  try { rp = await factStore.repoint(pointed, survivorB, { deliberate: true }); } catch (err) { threw = err; }
  check('repoint does not throw', !threw, threw && threw.message);
  check('…the pointer really moved', row(pointed).successor_id === survivorB);
  check('…and one entry exists for it', entriesFor(pointed).length === 1, `${entriesFor(pointed).length}`);
  check('…naming both the old successor and the new',
    /successor/.test(entriesFor(pointed)[0].evidence || '') && entriesFor(pointed)[0].evidence.includes(survivorA),
    entriesFor(pointed)[0].evidence);
  check('…and marked NOT reversible', entriesFor(pointed)[0].reversible === 0);

  // =========================================================================
  console.log('\n── restore is a change too, so it is recorded like one ──');
  const restored = await factStore.restore(oldFact, { deliberate: true });
  check('restore files an entry of its own', restored.ok && !!restored.ledgerId);
  check('…and it is not reversible (the fact is already active — Revert would be a no-op)',
    ledger.get(restored.ledgerId).reversible === 0);
  check('…and the supersede entry it undid is still there, untouched',
    entriesFor(oldFact).filter(x => x.action === 'supersede').length === 1);

  // =========================================================================
  console.log('\n── The undo actually works, through the one shared path ──');
  const gone = seed('superseded and then reverted');
  const replacement = seed('the replacement');
  const s2 = await factStore.supersede(gone, replacement);
  check('the fact is inactive after the supersede', row(gone).status === 'inactive');
  const rev = await ledger.revert(s2.ledgerId, { by: 'test' });
  check('revert() finds the entry and undoes it', rev && rev.ok !== false, JSON.stringify(rev).slice(0, 160));
  check('…the fact is active again', row(gone).status === 'active', row(gone).status);
  check('…and the entry is marked reverted rather than deleted',
    !!ledger.get(s2.ledgerId).reverted_at);

  // =========================================================================
  console.log('\n── One change, one entry: a caller that knows more ENRICHES ──');
  const loser = seed('the weaker fact');
  const winner = seed('the better-evidenced fact');
  const before = allEntries();
  const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
  const res = await MCPClient.shared().executeTool('memory_supersede_fact', { old_id: loser, new_id: winner }, {});
  check('the corrector\'s write tool reports the supersession', res.status === 'superseded', JSON.stringify(res));
  check('…and hands back the ledger id the write filed', !!res.ledger_id);
  check('…which is exactly one new entry, not two', allEntries() === before + 1, `${allEntries() - before} new`);

  const enriched = ledger.enrich(res.ledger_id, {
    passId: 'test-pass', tier: 'semantic',
    reason: 'These two could not both be true, and the survivor is better evidenced on provenance.',
    evidence: { deciding_axis: 'typed-over-stt' }
  });
  check('enrich() updates the entry in place', enriched === true);
  check('…still exactly one entry for that change', allEntries() === before + 1);
  const en = ledger.get(res.ledger_id);
  check('…carrying the caller\'s reason', /better evidenced on provenance/.test(en.reason));
  check('…and its pass and tier', en.pass_id === 'test-pass' && en.tier === 'semantic');
  check('…with the caller\'s evidence merged into the funnel\'s, not replacing it',
    /deciding_axis/.test(en.evidence) && /filed_by/.test(en.evidence), en.evidence);
  check('…and what it did NOT supply is left alone', en.target_text === 'the weaker fact');
  check('enrich() on an unknown id changes nothing and says so',
    ledger.enrich(randomUUID(), { reason: 'x' }) === false);

  // =========================================================================
  console.log('\n── A change that cannot be recorded does not happen ──');
  // The whole point of the transaction. If the ledger insert fails, the row must
  // be rolled back — a written row with no entry is the thing being prevented.
  const protectedFact = seed('must not change unrecorded');
  const other = seed('the would-be successor');
  const realRecord = ledger.record;
  ledger.record = () => null;                     // the ledger is broken
  const blocked = await factStore.supersede(protectedFact, other);
  ledger.record = realRecord;
  check('the write reports failure rather than pretending', blocked.ok === false, JSON.stringify(blocked));
  check('…and says the ledger is why', /ledger/i.test(blocked.reason || ''), blocked.reason);
  check('…the row is UNCHANGED — rolled back, not written-and-unlogged',
    row(protectedFact).status === 'active' && !row(protectedFact).successor_id,
    JSON.stringify({ status: row(protectedFact).status, successor: row(protectedFact).successor_id }));

  const rewordBlocked = seed('wording that must survive a broken ledger');
  ledger.record = () => null;
  const rwBlocked = await factStore.reword(rewordBlocked, 'this must not stick');
  ledger.record = realRecord;
  check('reword rolls back too', rwBlocked.ok === false && row(rewordBlocked).content === 'wording that must survive a broken ledger',
    row(rewordBlocked).content);

  // =========================================================================
  console.log('\n── A refused write files nothing at all ──');
  const identityLock = require(path.join(ROOT, 'db/identity-lock'));
  const named = seed('My name is Aurelius.', { claimType: 'declaration', salience: 10 });
  identityLock.autoLock(named, 'My name is Aurelius.', 'self');
  const isLocked = row(named).locked === 1;
  const entriesBefore = allEntries();
  const usurper = seed('My name is Bob.');
  const refused = await factStore.supersede(named, usurper);
  check('the identity lock still refuses the write', isLocked ? refused.ok === false : true,
    isLocked ? JSON.stringify(refused) : '(fact did not lock — check autoLock categories)');
  check('…and nothing was written to the ledger for a change that did not happen',
    allEntries() === entriesBefore, `${allEntries() - entriesBefore} entries appeared`);
  check('…and the locked fact is untouched', row(named).status === 'active');

  // =========================================================================
  console.log('\n── Nothing was deleted anywhere in this run ──');
  check('every seeded row still exists',
    db.prepare('SELECT COUNT(*) n FROM cluster_members').get().n === n,
    `${db.prepare('SELECT COUNT(*) n FROM cluster_members').get().n} of ${n}`);

  console.log(`\n${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('\nTest harness crashed:', err);
  process.exit(1);
});
