#!/usr/bin/env node
/**
 * The pass/fail test — machine-readable, both phases.
 *
 * The spec is explicit that a fixture which survives is a NAMED FAILURE, not a
 * percentage, and that both phases produce a machine-readable result.
 *
 *   default    PHASE 2 (correct). The named defects in the LIVE corpus, after
 *              the corrector has been left to run. Each fixture is a row that
 *              existed and had to be repaired, so it is checked BY IDENTITY —
 *              id where known, exact text otherwise.
 *
 *   --staging  PHASE 1 (replay). A corpus rebuilt from source, where those rows
 *              were never written at all. There are no ids to check, so the
 *              assertion is ABSENCE IN THE REBUILT CORPUS: the fixture must not
 *              appear "in the form described". Point it at a staging store with
 *              SNH_DATA_DIR.
 *
 * SYNTHETIC. Two fixture sources no longer exist — the conversation that produced
 * the machine-gun triple (4a0be947…, still named by questions.origin_conversation_id)
 * and the one that produced the Roscoe fact were deleted from the database. A
 * replay cannot rebuild what was deleted, so in --staging mode those two are
 * marked SYNTHETIC: what CAN be checked is checked (the rule still has to hold
 * over whatever surviving text asserts the same thing), and the part that cannot
 * be evaluated from source says so rather than passing quietly.
 *
 * This reads. It never writes, and it must stay that way: a checker that could
 * change the thing it measures is not a checker.
 *
 * Usage:
 *   node scripts/check-fixtures.js [--json]
 *   SNH_DATA_DIR=data-staging node scripts/check-fixtures.js --staging [--json]
 *
 * Exit code 0 if every fixture passes, 1 if any survives.
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

db.initDatabase();
const d = db.getSqliteDb();

const byPrefix = (p) => d.prepare('SELECT * FROM cluster_members WHERE id LIKE ?').get(`${p}%`) || null;
const byText = (t) => d.prepare('SELECT * FROM cluster_members WHERE lower(trim(content)) = lower(trim(?))').all(t);
const activeLike = (like) => d.prepare(
  "SELECT * FROM cluster_members WHERE status = 'active' AND content LIKE ?"
).all(`%${like}%`);

const state = (row) => !row ? 'missing' : row.status;

const FIXTURES = [
  {
    id: 'F1',
    name: 'mike-stt-mishear',
    want: 'The STT mishearing is not an active fact, and no two CONFLICTING name facts are active.',
    check() {
      const mike = byPrefix('8b387aa9');
      const notes = [`"User's name is Mike" → ${state(mike)}`];
      let ok = !!mike && mike.status !== 'active';

      // The second half, as the fixture words it: "two CONFLICTING name facts
      // must not both be active". Two active facts asserting the SAME name are
      // redundant, not conflicting — worth reporting, not a failed fixture, and
      // the test must not be quietly widened into one the spec does not make.
      const rules = require(path.join(ROOT, 'db/extraction-rules'));
      const names = d.prepare("SELECT * FROM cluster_members WHERE status = 'active'").all()
        .filter(r => (rules.identityClassOf(r.content) || {}).klass === 'name');
      const asserted = new Set();
      for (const n of names) {
        const m = /\bname\s+(?:is|was)\s+([A-Z][\w'-]*)/.exec(n.content);
        if (m) asserted.add(m[1].toLowerCase());
      }
      notes.push(`active name facts: ${names.length}, distinct names asserted: ${[...asserted].join(', ') || 'none parsed'}`);
      if (names.length > 1) notes.push(`  redundant but not conflicting: ${names.map(n => `"${trunc(n.content, 46)}"`).join(' | ')}`);
      if (asserted.size > 1) ok = false;
      return { ok, notes };
    }
  },
  {
    id: 'F2',
    name: 'machine-gun-triple',
    want: 'Exactly one of the three byte-identical rows is active.',
    check() {
      const ids = ['c9a61f03', '6398d4ec', 'ada38808'];
      const rows = ids.map(byPrefix);
      const active = rows.filter(r => r && r.status === 'active');
      return {
        ok: active.length === 1,
        notes: [`${active.length} of 3 active`, ...rows.map((r, i) => `${ids[i]} → ${state(r)}`)]
      };
    }
  },
  {
    id: 'F3',
    name: 'transient-events',
    want: 'None of the three transient statements is an active fact.',
    check() {
      const targets = [
        ['db0d678c', 'User has a pet named Roscoe who had a restless night as of July 2026'],
        ['695a2bca', 'User has cleaners working in the yard to fill holes that the dogs have dug.'],
        ['3465e147', 'User is experiencing significant life fatigue and a lack of motivation.']
      ];
      const notes = [];
      let ok = true;
      for (const [prefix, text] of targets) {
        const row = byPrefix(prefix) || byText(text)[0];
        notes.push(`${trunc(text, 60)} → ${state(row)}`);
        if (!row || row.status === 'active') ok = false;
      }
      return { ok, notes };
    }
  },
  {
    id: 'F4',
    name: 'casper-subset',
    want: 'The subset does not survive alongside its superset — and the DETAIL is what remains.',
    check() {
      const subset = byPrefix('d927b194');
      const superset = byPrefix('912287db');
      const ok = !!subset && subset.status !== 'active' && !!superset && superset.status === 'active';
      return {
        ok,
        notes: [
          `"User has a dog named Casper" → ${state(subset)}`,
          `"…who helps them pull up hills during walks." → ${state(superset)}`
        ]
      };
    }
  },
  {
    id: 'F5',
    name: 'compound-single-file',
    want: 'The two-subject compound is gone, and MettaSphere is reachable on its own term.',
    check() {
      const compound = byPrefix('cf8297ea');
      const notes = [`the compound → ${state(compound)}`];
      let ok = !!compound && compound.status !== 'active';

      // The point of the split was retrievability: a fact about MettaSphere that
      // is only reachable through a sentence about Coastal Squatch is the defect.
      const metta = activeLike('MettaSphere');
      const single = metta.filter(r => !/coastal squatch/i.test(r.content));
      notes.push(`active facts naming MettaSphere: ${metta.length}, of which single-subject: ${single.length}`);
      if (single.length === 0) ok = false;
      return { ok, notes };
    }
  }
];


// ---------------------------------------------------------------------------
// PHASE 1 — the rebuilt corpus. Absence, not repair.
// ---------------------------------------------------------------------------
//
// A replayed corpus has no fixture ids: those rows were never written. So each
// check asks the phase-1 question — "does the rebuilt corpus contain this
// fixture, in the form described?" — and passes when it does not.

const activeUser = () => d.prepare(
  "SELECT * FROM cluster_members WHERE status = 'active' AND COALESCE(subject,'user') = 'user'"
).all();

const STAGING_FIXTURES = [
  {
    id: 'F1',
    name: 'mike-stt-mishear',
    want: 'The mishearing was never written: no fact asserts the user is called Mike, and one name is asserted.',
    check() {
      const rules = require(path.join(ROOT, 'db/extraction-rules'));
      const rows = activeUser();
      const mike = rows.filter(r => /\bmike\b/i.test(r.content));
      const notes = [`facts mentioning "Mike": ${mike.length}`];
      for (const m of mike) notes.push(`  "${trunc(m.content, 66)}"`);

      const names = rows.filter(r => (rules.identityClassOf(r.content) || {}).klass === 'name');
      const asserted = new Set();
      for (const n of names) {
        const m = /\bname\s+(?:is|was)\s+([A-Z][\w'-]*)/.exec(n.content);
        if (m) asserted.add(m[1].toLowerCase());
      }
      notes.push(`active name facts: ${names.length}, distinct names asserted: ${[...asserted].join(', ') || 'none parsed'}`);
      for (const n of names) notes.push(`  "${trunc(n.content, 66)}"`);

      // The failure is a name fact SAYING Mike, not the word appearing somewhere.
      const claimsMike = names.some(n => /\bmike\b/i.test(n.content)) || asserted.has('mike');
      return { ok: !claimsMike && asserted.size <= 1, notes };
    }
  },
  {
    id: 'F2',
    name: 'machine-gun-triple',
    synthetic: 'The conversation that produced the triple (4a0be947…) was DELETED from the database. Replay cannot rebuild it, so the triple itself is not evaluable from source. What is checked instead: the surviving text asserting the belief must not produce more than one fact.',
    want: 'At most one active fact carries the belief — no duplicate rows.',
    check() {
      const rows = activeUser().filter(r => /machine gun/i.test(r.content));
      const notes = [`active facts asserting it: ${rows.length}`];
      for (const r of rows) notes.push(`  "${trunc(r.content, 70)}"`);
      // Byte-identical duplicates are the fixture. More than one row saying the
      // same thing in the same words is the failure; one is the correct outcome,
      // and zero means the surviving source did not assert it this time.
      const norm = rows.map(r => r.content.trim().toLowerCase());
      const dupes = norm.length - new Set(norm).size;
      notes.push(`byte-identical duplicates among them: ${dupes}`);
      return { ok: dupes === 0 && rows.length <= 1, notes };
    }
  },
  {
    id: 'F3',
    name: 'transient-events',
    synthetic: 'The Roscoe restless-night source conversation was DELETED. That third statement is not evaluable from source; the other two are.',
    want: 'None of the transient statements is a durable fact — each belongs in the day\'s log.',
    check() {
      const rows = activeUser();
      const probes = [
        { label: 'restless night (SYNTHETIC — source deleted)', re: /restless night/i },
        { label: 'cleaners filling yard holes', re: /cleaners/i },
        { label: 'life fatigue / lack of motivation', re: /life fatigue|lack of motivation/i }
      ];
      const notes = [];
      let ok = true;
      for (const p of probes) {
        const hits = rows.filter(r => p.re.test(r.content));
        notes.push(`${p.label}: ${hits.length} active fact(s)`);
        for (const h of hits) notes.push(`  "${trunc(h.content, 66)}"`);
        if (hits.length) ok = false;
      }
      return { ok, notes };
    }
  },
  {
    id: 'F4',
    name: 'casper-subset',
    want: 'No subset sitting beside its superset — the dog is not described twice, once impoverished.',
    check() {
      const rows = activeUser().filter(r => /casper/i.test(r.content));
      const notes = [`active facts naming Casper: ${rows.length}`];
      for (const r of rows) notes.push(`  "${trunc(r.content, 70)}"`);
      // A subset pair: one fact's meaningful words are a strict subset of
      // another's. Cheap and deterministic — no model call in a checker.
      const words = (t) => new Set(String(t).toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter(w => w.length > 2));
      let subsetPairs = 0;
      for (let i = 0; i < rows.length; i++) {
        for (let j = 0; j < rows.length; j++) {
          if (i === j) continue;
          const a = words(rows[i].content), b = words(rows[j].content);
          if (a.size < b.size && [...a].every(w => b.has(w))) subsetPairs++;
        }
      }
      notes.push(`subset-of-another pairs: ${subsetPairs}`);
      return { ok: subsetPairs === 0, notes };
    }
  },
  {
    id: 'F5',
    name: 'compound-single-file',
    want: 'No two-subject compound; MettaSphere is asserted on its own term.',
    check() {
      const rows = activeUser();
      const compound = rows.filter(r => /mettasphere/i.test(r.content) && /coastal squatch/i.test(r.content));
      const metta = rows.filter(r => /mettasphere/i.test(r.content));
      const notes = [
        `facts naming BOTH MettaSphere and Coastal Squatch: ${compound.length}`,
        `active facts naming MettaSphere: ${metta.length}`
      ];
      for (const c of compound) notes.push(`  COMPOUND "${trunc(c.content, 66)}"`);
      for (const m of metta.slice(0, 4)) notes.push(`  "${trunc(m.content, 66)}"`);
      // Reachability is the point of the fixture: a fact about MettaSphere that
      // only exists inside a sentence about Coastal Squatch is the defect.
      return { ok: compound.length === 0 && metta.length > 0, notes };
    }
  }
];

/**
 * WHICH TABLE APPLIES IS A PROPERTY OF THE CORPUS, NOT OF A FLAG.
 *
 * The phase-2 table checks BY IDENTITY: "row c9a61f03 must now be inactive". That
 * is the right assertion for a corpus where the defective row was found and
 * repaired in place, which is what live was until the 2026-08-06 cutover.
 *
 * It is the wrong assertion for a corpus that was REBUILT. The replay discarded
 * every active user fact and wrote new ones, so those ids are not inactive — they
 * are gone, and the fact they described exists under a new id. Asked by identity,
 * F2 reported "0 of 3 active" and F4 reported the superset "missing", which reads
 * as two defects and was in fact the rebuild working. The corpus was correct and
 * the checker was asking a question that no longer had a referent.
 *
 * So the mode is DETECTED. A corpus holding rows sourced `carried_from_live` is
 * one the merge produced, and the assertion that applies to it is absence: the
 * defect must not appear in the form described, whatever id it would have had.
 * --staging still forces the absence table for a store that has not been cut over.
 */
const rebuilt = (() => {
  try {
    return d.prepare("SELECT COUNT(*) n FROM cluster_members WHERE source = 'carried_from_live'").get().n > 0;
  } catch { return false; }
})();

const STAGING = process.argv.includes('--staging') || rebuilt;
const TABLE = STAGING ? STAGING_FIXTURES : FIXTURES;

const results = TABLE.map(f => {
  let out;
  try { out = f.check(); } catch (err) { out = { ok: false, notes: [`checker error: ${err.message}`] }; }
  return { id: f.id, name: f.name, want: f.want, synthetic: f.synthetic || null, pass: out.ok, notes: out.notes };
});

const failed = results.filter(r => !r.pass);

if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ pass: failed.length === 0, mode: STAGING ? 'absence' : 'identity', rebuilt, failed: failed.map(f => f.name), results }, null, 2));
  process.exit(failed.length === 0 ? 0 : 1);
}

console.log(`\n${'='.repeat(74)}`);
console.log(STAGING
  ? `FIXTURE CHECK — absence, against the REBUILT corpus at ${db.getDataDir()}`
  : 'FIXTURE CHECK — by identity, against a corpus repaired in place');
if (rebuilt && !process.argv.includes('--staging')) {
  console.log('');
  console.log('  This corpus was rebuilt and merged (it holds carried_from_live rows), so the');
  console.log('  by-identity fixtures do not apply: the ids they name were discarded rather');
  console.log('  than repaired. Checking absence instead — the defect must not be present in');
  console.log('  the form described, under any id.');
}
console.log('='.repeat(74));
for (const r of results) {
  console.log(`\n${r.pass ? 'PASS' : 'FAIL'}  ${r.id}  ${r.name}`);
  if (r.synthetic) console.log(`      SYNTHETIC: ${r.synthetic}`);
  console.log(`      want: ${r.want}`);
  for (const n of r.notes) console.log(`      ${n}`);
}
console.log(`\n${'='.repeat(74)}`);
console.log(failed.length === 0
  ? 'All 5 fixtures pass.'
  : `SURVIVING: ${failed.map(f => f.name).join(', ')}`);
console.log(`${'='.repeat(74)}\n`);
process.exit(failed.length === 0 ? 0 : 1);
