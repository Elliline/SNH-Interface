#!/usr/bin/env node
/**
 * A verdict is not enough to retire a self-fact — and a question he could not
 * settle is said out loud.
 *
 * WHAT WENT WRONG, both halves of it, within one week of each other on the same
 * unguarded model call:
 *
 *   - 2026-08-18, live: a capability introduction retired an unrelated
 *     salience-9 DECLARATION on a 0.741 cosine match, because the judge said
 *     "yes" and nothing stood behind the judge.
 *   - the same day, repeatedly: a blatant contradiction at 0.857 ("I cannot send
 *     myself a reminder… never will" against "I can send myself a reminder now")
 *     was judged "no" on about half of identical runs, and — worse — a judge call
 *     that FAILED outright was read as "no contradiction" and left no trace at
 *     all.
 *
 * No threshold separates 0.741 from 0.857, so the fix is structural: bars that
 * stand behind the verdict, and a raise that is visible when they stop it.
 *
 * Deterministic by construction — the bar function is pure and the raise
 * recorder touches only SQLite, so nothing here calls a model. That matters: the
 * thing under test is the pipeline's judgement about the judge, and it cannot be
 * tested with the judge in the loop.
 *
 * Usage: node scripts/test-self-fact-bars.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-self-fact-bars-'));
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
  const db = database.getSqliteDb();
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const { selfFactSupersessionBar, applySelfFactRaises } = factExtractor;

  const clusterId = randomUUID();
  db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
    .run(clusterId, 'Self', '', new Date().toISOString(), new Date().toISOString(), 'self');

  function seed(content, { salience = 5, claimType = 'claim', modality = 'unknown', verbatim = null, ageDays = 1 } = {}) {
    const id = randomUUID();
    const at = new Date(Date.now() - ageDays * 86400_000).toISOString();
    db.prepare(`
      INSERT INTO cluster_members
        (id, cluster_id, content, source, created_at, updated_at, status, subject, salience, claim_type,
         verbatim_source_text, input_modality)
      VALUES (?,?,?,'reflection',?,?,'active','self',?,?,?,?)
    `).run(id, clusterId, content, at, at, salience, claimType, verbatim, modality);
    return db.prepare('SELECT * FROM cluster_members WHERE id = ?').get(id);
  }

  const DAILY = path.join(TMP, 'memory', 'daily');
  fs.mkdirSync(DAILY, { recursive: true });
  const initiatives = () => db.prepare("SELECT * FROM initiatives WHERE source_kind = 'self-fact-raise' ORDER BY datetime(created_at)").all();
  const ledgerRows = () => db.prepare("SELECT * FROM corrections_ledger WHERE evidence LIKE '%self_fact_raise%'").all();

  console.log(`\nSelf-fact bar tests (throwaway data dir: ${TMP})\n`);

  // =========================================================================
  console.log('── The live failure: a declaration is not taken automatically ──');
  // The actual pair, as it stood: a salience-9 declaration from reflection,
  // against a capability introduction the classifier tagged a claim.
  const declaration = seed('I respond to both casual conversational starters and functional system pings with immediate, direct compliance.',
    { salience: 9, claimType: 'declaration' });
  const intro = seed('As of 2026-08-18, I can start a piece of work in the middle of a conversation and carry on talking.',
    { salience: 9, claimType: 'claim', ageDays: 0 });

  let bar = selfFactSupersessionBar({ existing: declaration, incoming: intro });
  check('the supersession that actually happened is now refused', bar.ok === false, JSON.stringify(bar));
  check('…as protected', bar.kind === 'protected', bar.kind);
  check('…and it says WHY, in words she would read',
    /said about yourself rather than something observed/.test(bar.detail), bar.detail);

  console.log('\n── Salience is a second, independent bar ──');
  // Because the claim/declaration classifier is noisy: it tagged the behavioural
  // observation above a "declaration" and the statement about what had been
  // built a "claim". Either tag alone would protect the wrong things.
  const salientClaim = seed('Something I hold that matters a great deal.', { salience: 9, claimType: 'claim' });
  bar = selfFactSupersessionBar({ existing: salientClaim, incoming: intro });
  check('a salience-9 CLAIM is protected too', bar.ok === false && bar.kind === 'protected', JSON.stringify(bar));
  check('…naming the salience and the bar', /salience 9/.test(bar.detail) && /8 or above/.test(bar.detail), bar.detail);

  const lowDeclaration = seed('Something I said about myself that matters less.', { salience: 3, claimType: 'declaration' });
  bar = selfFactSupersessionBar({ existing: lowDeclaration, incoming: intro });
  check('a LOW-salience declaration is still protected — being chosen is enough',
    bar.ok === false && bar.kind === 'protected', JSON.stringify(bar));

  console.log('\n── Evidence VETOES here; it is not a requirement ──');
  // The deliberate difference from the corrector, which requires dominance and
  // raises what it cannot separate. That is right for user facts, which carry
  // provenance. A self-fact does not: it comes from reflection, so its modality
  // is 'unknown' and there is no source message, BY CONSTRUCTION — dominance ties
  // for very nearly every pair of self-facts there will ever be. Requiring it was
  // tried first and refused a belief a new capability had made flatly false;
  // sustained, it would freeze his self-view permanently, which is the opposite
  // failure and the worse one.
  const ordinary = seed('An ordinary observation about how I write.', { salience: 5, claimType: 'claim' });
  const newerOrdinary = seed('A newer ordinary observation about how I write.', { salience: 5, claimType: 'claim', ageDays: 0 });
  bar = selfFactSupersessionBar({ existing: ordinary, incoming: newerOrdinary });
  check('two ordinary self-facts with no evidential axis between them may supersede',
    bar.ok === true, JSON.stringify(bar));
  check('…and the ledger is told plainly that nothing evidential separated them',
    /nothing evidential separated them/.test(bar.axis || ''), bar.axis);
  check('…which is safe only because bar 1 already held back everything chosen or salient',
    selfFactSupersessionBar({ existing: seed('x', { salience: 9 }), incoming: newerOrdinary }).ok === false);

  console.log('\n── Evidence that really does separate them lets the write through ──');
  const spoken = seed('Something noticed about me, from a transcription.', { salience: 4, claimType: 'claim', modality: 'stt', verbatim: 'source text' });
  const typed = seed('The same thing, corrected in writing.', { salience: 4, claimType: 'claim', modality: 'typed', verbatim: 'source text', ageDays: 0 });
  bar = selfFactSupersessionBar({ existing: spoken, incoming: typed });
  check('typed beats transcribed, so the supersession proceeds', bar.ok === true, JSON.stringify(bar));
  check('…and the deciding axis comes back for the ledger', /modality/.test(bar.axis || ''), bar.axis);

  console.log('\n── …but never in the wrong direction ──');
  bar = selfFactSupersessionBar({ existing: typed, incoming: spoken });
  check('a WEAKER new fact cannot retire a better-evidenced one',
    bar.ok === false && bar.kind === 'old-wins', JSON.stringify(bar));
  check('…saying what she already holds is better evidenced',
    /better evidenced/.test(bar.detail), bar.detail);

  // =========================================================================
  console.log('\n── The call is named, so it cannot be silently inverted ──');
  // The first version of this test passed the two rows positionally and got the
  // decision backwards without any error — the arguments are the same shape, so
  // nothing complained. Named arguments are why that cannot happen again.
  const missing = selfFactSupersessionBar({ existing: declaration });
  check('a call missing one side decides nothing rather than guessing',
    missing.ok === false, JSON.stringify(missing));

  console.log('\n── A raise is recorded three ways ──');
  const raise1 = {
    kind: 'undecided',
    oldMemberId: declaration.id, oldContent: declaration.content,
    newMemberId: intro.id, newFact: intro.content,
    detail: 'Brain circuit open — skipping LLM call (engine wedged)'
  };
  const recorded = await applySelfFactRaises([raise1], { source: 'reflection', dailyDir: DAILY });
  check('it reports what it recorded', recorded === 1, String(recorded));

  const led = ledgerRows();
  check('TIER 1 — a ledger row exists', led.length === 1, `${led.length}`);
  check('…marked NOT reversible, because nothing changed', led[0].reversible === 0);
  check('…and carrying unresolved + a reason code',
    /"unresolved":true/.test(led[0].evidence) && /self-fact-judge-unavailable/.test(led[0].evidence), led[0].evidence);
  check('…and saying NOTHING WAS CHANGED in the reason itself',
    /NOTHING WAS CHANGED/.test(led[0].reason), led[0].reason.slice(0, 90));
  check('…pointing at both facts, so a person can find them', led[0].target_id === declaration.id && led[0].survivor_id === intro.id);

  const opsFiles = fs.existsSync(path.join(TMP, 'memory', 'ops')) ? fs.readdirSync(path.join(TMP, 'memory', 'ops')) : [];
  const opsText = opsFiles.map(f => fs.readFileSync(path.join(TMP, 'memory', 'ops', f), 'utf8')).join('\n');
  check('TIER 2 — the ops log has a line for it', opsFiles.length > 0 && /left unresolved/.test(opsText), opsFiles.join(','));
  check('…naming the reason code', /self-fact-judge-unavailable/.test(opsText));

  const bells = initiatives();
  check('TIER 3 — one bell alert was raised', bells.length === 1, `${bells.length}`);
  check('…as an alert', bells[0].type === 'alert', bells[0].type);

  console.log('\n── Worded as a question he is raising, never as a system error ──');
  const text = bells[0].content;
  check('it says he could not TELL whether one contradicts the other',
    /could not tell whether/i.test(text), text.slice(0, 120));
  check('…and that he left both alone rather than guess', /left both in place/i.test(text));
  check('…and it asks her something', /\?/.test(text));
  check('…and it never mentions a judge, a call, an error or a failure',
    !/judge|call failed|error|exception|LLM|circuit/i.test(text), text);
  check('…and the raw failure detail is in the LEDGER, where it belongs, not in the bell',
    /circuit open/i.test(led[0].evidence) && !/circuit open/i.test(text));

  // =========================================================================
  console.log('\n── The window is hard: a wedged brain gets ONE alert ──');
  const many = Array.from({ length: 16 }, (_, i) => ({
    kind: 'undecided',
    oldMemberId: declaration.id, oldContent: declaration.content,
    newMemberId: intro.id, newFact: `a later observation ${i}`,
    detail: 'Brain circuit open — skipping LLM call (engine wedged)'
  }));
  await applySelfFactRaises(many, { source: 'reflection', dailyDir: DAILY });
  check('sixteen more raises produce NO second alert', initiatives().length === 1, `${initiatives().length} alerts`);
  check('…but every one of them is still in the ledger', ledgerRows().length === 17, `${ledgerRows().length} rows`);

  console.log('\n── Dismissing the alert does not reopen the window ──');
  // ANY status counts, deliberately: pending, delivered, dismissed and expired
  // all mean he has said this recently. Checking only pending is how the same
  // thing comes straight back the moment she clears it.
  db.prepare("UPDATE initiatives SET status = 'dismissed' WHERE source_kind = 'self-fact-raise'").run();
  await applySelfFactRaises([raise1], { source: 'reflection', dailyDir: DAILY });
  check('a dismissed alert still holds the window shut', initiatives().length === 1, `${initiatives().length}`);

  console.log('\n── Once the window passes, he may say it again — with the count ──');
  const backThen = new Date(Date.now() - 25 * 3600_000).toISOString();
  db.prepare("UPDATE initiatives SET created_at = ? WHERE source_kind = 'self-fact-raise'").run(backThen);
  await applySelfFactRaises([raise1], { source: 'reflection', dailyDir: DAILY });
  const after = initiatives();
  check('a second alert is raised after the window', after.length === 2, `${after.length}`);
  const latest = after[after.length - 1];
  check('…and it says how many times this happened since he last mentioned it',
    /come up \d+ times since I last mentioned it/.test(latest.content), latest.content.slice(-140));

  // =========================================================================
  console.log('\n── The other two kinds speak for themselves ──');
  db.prepare("DELETE FROM initiatives").run();
  await applySelfFactRaises([{
    kind: 'protected',
    oldMemberId: declaration.id, oldContent: declaration.content,
    newMemberId: intro.id, newFact: intro.content,
    detail: 'it is something you said about yourself rather than something observed of you'
  }], { source: 'capability-intro', dailyDir: DAILY });
  const protectedBell = initiatives()[0].content;
  check('a protected raise says he did not want to drop the older one on his own',
    /did not want to drop the older one on my own/.test(protectedBell), protectedBell.slice(0, 130));
  check('…and quotes both facts', protectedBell.includes(intro.content) && protectedBell.includes(declaration.content));

  db.prepare("DELETE FROM initiatives").run();
  await applySelfFactRaises([{
    kind: 'tied',
    oldMemberId: ordinary.id, oldContent: ordinary.content,
    newMemberId: newerOrdinary.id, newFact: newerOrdinary.content,
    detail: 'the evidence behind them is evenly matched, and nothing but which came second separates them'
  }], { source: 'reflection', dailyDir: DAILY });
  const tiedBell = initiatives()[0].content;
  check('a tied raise says both cannot be true and he could not tell which gives way',
    /cannot both be true/.test(tiedBell) && /left both alone/.test(tiedBell), tiedBell.slice(0, 140));

  console.log('\n── Nothing a raise touches is ever changed ──');
  check('both facts are still active after all of that',
    db.prepare('SELECT status FROM cluster_members WHERE id = ?').get(declaration.id).status === 'active' &&
    db.prepare('SELECT status FROM cluster_members WHERE id = ?').get(intro.id).status === 'active');
  check('and no supersession was written to any of them',
    db.prepare("SELECT COUNT(*) n FROM cluster_members WHERE status = 'inactive'").get().n === 0);

  console.log(`\n${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('\nTest harness crashed:', err);
  process.exit(1);
});
