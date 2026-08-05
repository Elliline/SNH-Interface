#!/usr/bin/env node
/**
 * The pass/fail test, phase 2 — machine-readable.
 *
 * The spec is explicit that a fixture which survives is a NAMED FAILURE, not a
 * percentage, and that both phases of the test produce a machine-readable
 * result. Phase 1 (replay through intake) is scripts/dryrun-extract.js. This is
 * phase 2: the named defects, checked against the LIVE corpus after the
 * corrector has been left to run.
 *
 * Checked by IDENTITY — id where the row is known, exact text otherwise — never
 * by count, because "there are fewer facts than there were" is not the claim
 * being made.
 *
 * This reads. It never writes, and it must stay that way: a checker that could
 * change the thing it measures is not a checker.
 *
 * Usage:
 *   node scripts/check-fixtures.js
 *   node scripts/check-fixtures.js --json
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

const results = FIXTURES.map(f => {
  let out;
  try { out = f.check(); } catch (err) { out = { ok: false, notes: [`checker error: ${err.message}`] }; }
  return { id: f.id, name: f.name, want: f.want, pass: out.ok, notes: out.notes };
});

const failed = results.filter(r => !r.pass);

if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ pass: failed.length === 0, failed: failed.map(f => f.name), results }, null, 2));
  process.exit(failed.length === 0 ? 0 : 1);
}

console.log(`\n${'='.repeat(74)}`);
console.log('FIXTURE CHECK — the named defects, against the live corpus');
console.log('='.repeat(74));
for (const r of results) {
  console.log(`\n${r.pass ? 'PASS' : 'FAIL'}  ${r.id}  ${r.name}`);
  console.log(`      want: ${r.want}`);
  for (const n of r.notes) console.log(`      ${n}`);
}
console.log(`\n${'='.repeat(74)}`);
console.log(failed.length === 0
  ? 'All 5 fixtures pass.'
  : `SURVIVING: ${failed.map(f => f.name).join(', ')}`);
console.log(`${'='.repeat(74)}\n`);
process.exit(failed.length === 0 ? 0 : 1);
