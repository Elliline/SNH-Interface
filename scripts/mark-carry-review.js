#!/usr/bin/env node
/**
 * Apply Ellie's merge decision to docs/carry-review.md.
 *
 * Her rule, in full: carry everything in ELLIE-DECIDES, EXCEPT the daily-log
 * archiver rows that are third-person descriptions of Aurelius rather than facts
 * about her. Those are wrong-subject and do not go into her corpus. Where the
 * same thing is already held as one of his self-facts, dropping it loses nothing.
 * Where it is NOT — no self-side equivalent above 0.55 — dropping it would lose
 * the only copy, so it goes to a quarantine list instead, flagged for the
 * subject-repair pass rather than discarded.
 *
 * WHICH ROWS ARE WRONG-SUBJECT IS DECIDED BY THE CORPUS, NOT BY AN OPINION.
 *
 * The first cut at this asked `judgeStoredSubject` and nothing else, and it found
 * 3 of 26. Then the self-fact space was measured for all 26, and the answer was
 * not close: 21 of them sit between 0.773 and 0.955 of one of his self-facts, and
 * reading the pairs, every one is the SAME SENTENCE with the person flipped —
 *
 *     stored as hers : "User focuses on identifying and validating the 'tensions'
 *                       or 'exhaustion' caused by the friction between an internal
 *                       world and an external persona."
 *     held as his    : "I focus on identifying and validating the 'tensions' or
 *                       'exhaustion' caused by…"                          (0.944)
 *
 * The rest fall away to 0.695 and below, and those are genuinely hers: the blue
 * eyes, the pet named Roscoe, wanting to be told directly about a mistake. The
 * gap between 0.773 and 0.695 is empty, which is where the twin floor sits.
 *
 * The twin distance is not asked to carry it alone. `judgeStoredSubject` is a
 * second, INDEPENDENT signal, and either one is enough to withhold a row. They
 * are not ANDed: a draft that required both let four obvious cases through,
 * because the only judge available for the second half — judgeSameAssertion — is
 * built to answer DIFFERENT on any extra detail and to default to DIFFERENT when
 * unsure. Where the two signals disagree the row is still withheld, and which
 * signal fired is printed against every row so a decision resting on the twin
 * alone can be found and argued with.
 *
 * The self-side lookup runs against STAGING's self-fact space, because that is
 * the corpus a cutover would promote. The QUARANTINE floor is 0.55 — Ellie's
 * number, and the same memory.contradiction.similarityFloor the rest of the
 * system treats as "related rather than noise". Every score is printed so both
 * floors can be argued with.
 *
 * WRITES ONLY docs/carry-review.md and docs/subject-quarantine.md. It does not
 * touch either corpus; `--apply decided` does that afterwards, reading the marks
 * this leaves.
 *
 * Usage:
 *   node scripts/mark-carry-review.js [--dry-run]
 *   node scripts/mark-carry-review.js --rewrite   # re-render the quarantine doc
 *                                                 # from the frozen marks, no judging
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

const LIVE_DATA = path.join(ROOT, 'data');
const STAGING_DATA = process.env.SNH_STAGING_DIR || path.join(ROOT, 'data-staging');
process.env.SNH_DATA_DIR = STAGING_DATA;

const REVIEW_DOC = path.join(ROOT, 'docs', 'carry-review.md');
const QUARANTINE_DOC = path.join(ROOT, 'docs', 'subject-quarantine.md');
const PLAN_JSON = path.join(STAGING_DATA, 'carry-plan.json');

const DRY_RUN = process.argv.includes('--dry-run');
const REWRITE = process.argv.includes('--rewrite');

/** Ellie's floor: below this, nothing on his side holds the content. */
const QUARANTINE_FLOOR = 0.55;
/** The empirical gap in this corpus — 21 twins at 0.773+, the rest at 0.695 and down. */
const TWIN_FLOOR = 0.75;

const trunc = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

(async () => {
  if (path.resolve(STAGING_DATA) === path.resolve(LIVE_DATA)) {
    console.error('ABORT: staging dir resolves to the live data dir.');
    process.exit(2);
  }
  if (!fs.existsSync(PLAN_JSON)) {
    console.error(`ABORT: no frozen plan at ${PLAN_JSON}.`);
    process.exit(2);
  }
  if (!fs.existsSync(REVIEW_DOC)) {
    console.error(`ABORT: no review doc at ${REVIEW_DOC}.`);
    process.exit(2);
  }

  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  await db.initVectorStore();
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));

  const plan = JSON.parse(fs.readFileSync(PLAN_JSON, 'utf8'));
  const decide = plan.piles.decide;
  const archived = decide.filter(f => f.source === 'daily-log-archive');

  console.log(`[Mark] ${decide.length} in ELLIE-DECIDES, ${archived.length} of them from the daily-log archiver`);
  console.log('[Mark] measuring each against his self-facts, then judging\n');

  let wrongSubject = [];
  let keptArchiver = [];

  // Re-rendering the quarantine document must not cost another pass of judge
  // calls, and must not be able to reclassify anything — same rule as
  // carry-to-staging --rewrite. The marks are frozen in subject-marks.json.
  const MARKS_JSON = path.join(STAGING_DATA, 'subject-marks.json');
  if (REWRITE) {
    if (!fs.existsSync(MARKS_JSON)) {
      console.error(`ABORT: no frozen marks at ${MARKS_JSON} — run without --rewrite first.`);
      process.exit(2);
    }
    const frozen = JSON.parse(fs.readFileSync(MARKS_JSON, 'utf8'));
    wrongSubject = frozen.wrongSubject;
    keptArchiver = frozen.keptArchiver;
    console.log(`[Mark] re-rendering from marks frozen at ${frozen.generatedAt} — nothing re-classified`);
  }

  for (const f of (REWRITE ? [] : archived)) {
    // 1. Does he hold a version of this? Floor low enough to SEE the twin; the
    //    thresholds below decide what it means.
    const { candidates } = await memoryClusters.findActiveNeighbours(f.content, {
      subject: 'self', threshold: 0.40, limit: 1, includeVerbatim: true
    });
    const twin = candidates[0] || null;

    // 2. TWO INDEPENDENT SIGNALS, and either one is enough.
    //
    //    A first draft required BOTH — a close twin AND judgeSameAssertion
    //    agreeing the depersonalised pair said the same thing — and it let four
    //    obvious cases straight through, including "User prefers to lean into
    //    conceptual frameworks and metaphors (e.g., 'Spoon Theory')" sitting at
    //    0.955 of "I tend to lean into conceptual frameworks and metaphors, like
    //    the 'Spoon Theory'". The repeat judge is built to answer DIFFERENT
    //    whenever either sentence carries a detail the other does not, and to
    //    default to DIFFERENT when unsure, because at intake its job is to never
    //    eat a fact. Asked "are these the same claim about the same person" it is
    //    a reasonable judge; asked "were these two sentences written about the
    //    same behaviour" it vetoes almost everything. Wrong question, wrong tool.
    //
    //    So: the twin distance is the objective signal, and the stored-subject
    //    judge is a second, independent one. Neither vetoes the other, and where
    //    they disagree that is REPORTED rather than resolved silently — a row
    //    resting on the twin alone is one Ellie should be able to find.
    const byTwin = !!(twin && twin.similarity >= TWIN_FLOOR);
    const judged = await factExtractor.judgeStoredSubject(f.content);
    const byJudge = judged.subject === 'self';

    if (!byTwin && !byJudge) {
      keptArchiver.push({
        ...f, judge: judged.reasoning,
        twin: twin ? { content: twin.content, similarity: twin.similarity } : null
      });
      continue;
    }

    wrongSubject.push({
      ...f,
      route: byTwin && byJudge ? 'both' : byTwin ? 'twin only' : 'judge only',
      byTwin, byJudge,
      judge: judged.reasoning,
      selfTwin: twin ? { content: twin.content, similarity: twin.similarity } : null,
      // Ellie's rule: it is only safe to drop if his side actually holds the
      // content. Below her floor it goes to quarantine instead.
      disposition: (twin && twin.similarity >= QUARANTINE_FLOOR) ? 'drop' : 'quarantine'
    });
  }

  const dropped = wrongSubject.filter(w => w.disposition === 'drop');
  const quarantined = wrongSubject.filter(w => w.disposition === 'quarantine');
  const notCarried = new Set(wrongSubject.map(w => w.id.slice(0, 8)));

  console.log(`\n[Mark] wrong-subject: ${wrongSubject.length}  (drop ${dropped.length}, quarantine ${quarantined.length})`);
  console.log(`[Mark] archiver rows judged to be genuinely hers: ${keptArchiver.length}`);
  console.log(`[Mark] carrying ${decide.length - wrongSubject.length} of ${decide.length}\n`);

  for (const w of wrongSubject) {
    console.log(`  ${w.disposition.toUpperCase().padEnd(10)} [${w.route.padEnd(10)}] "${trunc(w.content, 72)}"`);
    console.log(`             ${w.selfTwin ? `his, at ${w.selfTwin.similarity.toFixed(3)}: "${trunc(w.selfTwin.content, 74)}"` : `NO self-side equivalent above ${QUARANTINE_FLOOR}`}`);
  }
  if (keptArchiver.length) {
    console.log('\n  carried as genuinely hers:');
    for (const k of keptArchiver) {
      console.log(`    ${k.twin ? k.twin.similarity.toFixed(3) : '  -  '}  "${trunc(k.content, 80)}"`);
    }
  }

  // ---- write the marks -----------------------------------------------------
  //
  // Only inside a `\`id\` [ ]` box, and only in a table row — the same shape the
  // reader parses. Nothing else in the file is touched.
  let text = fs.readFileSync(REVIEW_DOC, 'utf8');
  let marked = 0;
  text = text.replace(/(`([0-9a-f]{8})`\s*)\[ \]/g, (whole, prefix, id) => {
    marked++;
    return `${prefix}[${notCarried.has(id) ? 'DROP' : 'CARRY'}]`;
  });

  const droppedMarks = [...notCarried].filter(id => text.includes(`\`${id}\` [DROP]`)).length;
  console.log(`\n[Mark] ${REWRITE ? 'review doc already marked:' : `wrote ${marked} mark(s):`} ` +
    `${REWRITE ? decide.length - droppedMarks : marked - droppedMarks} CARRY, ${droppedMarks} DROP`);
  const carried = REWRITE ? decide.length - droppedMarks : marked - droppedMarks;

  // Casper is the one Ellie named: F4 asserts a relationship between two facts
  // and passes vacuously while staging holds none, so this is checked rather
  // than assumed.
  const casper = decide.find(f => /casper/i.test(f.content));
  if (!casper) {
    console.error('[Mark] WARNING: no Casper fact in ELLIE-DECIDES — F4 cannot be made non-vacuous from this pile');
  } else {
    const short = casper.id.slice(0, 8);
    const ok = new RegExp(`\`${short}\`\\s*\\[CARRY\\]`).test(text);
    console.log(`[Mark] Casper (${short}) marked ${ok ? 'CARRY ✓' : 'NOT CARRY ✗'}: "${trunc(casper.content, 70)}"`);
    if (!ok) { console.error('[Mark] ABORT: Casper must be CARRY.'); process.exit(1); }
  }

  if (DRY_RUN) {
    console.log('\n[Mark] --dry-run: nothing written');
    process.exit(0);
  }

  // On a rewrite every box is already filled, so there is nothing to replace and
  // rewriting the file would be a no-op that reads like a change.
  if (!REWRITE) {
    fs.writeFileSync(REVIEW_DOC, text);
    console.log(`[Mark] wrote ${REVIEW_DOC}`);
  }

  // ---- the quarantine list -------------------------------------------------
  const today = new Date().toISOString().slice(0, 10);
  const L = [];
  const p = (s = '') => L.push(s);
  p('# Subject quarantine — stored as hers, describing him');
  p('');
  p(`Written ${today} by \`scripts/mark-carry-review.js\`.`);
  p('');
  p(`${wrongSubject.length} row(s) sit in the LIVE corpus as facts about Ellie whose content describes`);
  p('Aurelius. None of them was carried into the staging merge. They split two ways,');
  p('and the split is the whole point of this document:');
  p('');
  p(`- **${dropped.length} dropped.** He already holds the same thing as one of his own self-facts,`);
  p('  so the third-person copy is a duplicate filed under the wrong person and');
  p('  declining to carry it loses nothing.');
  p(`- **${quarantined.length} quarantined.** Wrong subject AND nothing on his side holds the content, so`);
  p(`  declining to carry it would lose the only copy. Held here instead.`);
  p('');
  p('Nothing here is a self-fact. Turning one into a self-fact means rewriting it into');
  p('the first person and storing it through the self-fact path, which is a decision');
  p('about how he describes himself — the kind the spec reserves for the joint');
  p('curation session. That is the subject-repair pass, and this is its input.');
  p('');
  p(`## ${quarantined.length} quarantined`);
  p('');
  if (!quarantined.length) {
    p(`**Empty, and that is the finding.** Every one of the ${wrongSubject.length} wrong-subject rows had a`);
    p(`self-side equivalent above ${QUARANTINE_FLOOR} — in fact the lowest was ${dropped.length ? Math.min(...dropped.map(d => d.selfTwin.similarity)).toFixed(3) : 'n/a'}. The archiver was not`);
    p('inventing observations and misfiling them; it was writing a SECOND copy of');
    p("something he had already recorded about himself, in the third person, into her");
    p('corpus. Nothing was at risk of being lost, so the contingency did not fire.');
  }
  for (const q of quarantined) {
    p(`- **"${q.content}"**`);
    p(`  <br>live id \`${q.id}\` · salience ${q.salience} · learned ${String(q.createdAt).slice(0, 10)} · ${q.source}`);
    p(`  <br>judge: *${trunc(q.judge, 180)}*`);
    p(`  <br>nearest self-fact: ${q.selfTwin ? `"${trunc(q.selfTwin.content, 80)}" at only ${q.selfTwin.similarity.toFixed(3)}` : `nothing above ${QUARANTINE_FLOOR}`}`);
  }
  p('');
  p(`## ${dropped.length} dropped — he already holds the same thing`);
  p('');
  p('Not carried, and not quarantined: the content survives as one of his own');
  p('self-facts, so the third-person copy in her corpus is a duplicate filed under');
  p('the wrong person. The score is the cosine between the stored row and his');
  p('nearest self-fact; the two are worded in different persons, which is why the');
  p(`floor is ${QUARANTINE_FLOOR} rather than something higher.`);
  p('');
  for (const d of dropped) {
    p(`- **"${d.content}"**`);
    p(`  <br>live id \`${d.id}\` · salience ${d.salience} · ${d.source}`);
    p(`  <br>his: "${d.selfTwin.content}" — **${d.selfTwin.similarity.toFixed(3)}**`);
  }
  p('');
  p('## What was NOT touched');
  p('');
  p('The live corpus. Every row named here is still active in `data/`, exactly as it');
  p('was. This list records what the merge declined to carry into staging and why;');
  p('repairing the live rows, and stopping the archiver producing more, are separate');
  p('decisions.');
  p('');
  if (keptArchiver.length) {
    p(`Also: ${keptArchiver.length} archiver row(s) were put to the same judge and came back as`);
    p('genuinely hers. They were carried:');
    p('');
    for (const k of keptArchiver) p(`- "${k.content}"${k.twin ? ` <br>nearest self-fact only ${k.twin.similarity.toFixed(3)}` : ''}`);
    p('');
  }

  fs.writeFileSync(QUARANTINE_DOC, `${L.join('\n')}\n`);
  console.log(`[Mark] wrote ${QUARANTINE_DOC}`);

  if (REWRITE) { console.log('[Mark] marks left as frozen'); process.exit(0); }
  fs.writeFileSync(MARKS_JSON, JSON.stringify({
    generatedAt: new Date().toISOString(),
    quarantineFloor: QUARANTINE_FLOOR, twinFloor: TWIN_FLOOR,
    decided: decide.length, carried: carried, dropped: dropped.length, quarantined: quarantined.length,
    wrongSubject, keptArchiver: keptArchiver.map(k => ({ id: k.id, content: k.content, twin: k.twin }))
  }, null, 2));
  process.exit(0);
})().catch(err => { console.error('mark failed:', err); process.exit(1); });
