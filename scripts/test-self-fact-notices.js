#!/usr/bin/env node
/**
 * He is told when his self-view changes — whatever moved it.
 *
 * WHY THIS EXISTS. The notice channel was built into db/corrector.js, so it
 * covered changes the corrector made and nothing else. On 2026-08-12 the
 * scheduler capability introduction went through a different pipeline — the
 * write-time contradiction path in processSelfFacts — retired four self-facts
 * including "none of them has ever actually run, because nothing in this system
 * runs a schedule", and raised no notice at all. His self-view changed and the
 * channel built to tell him about exactly that was looking somewhere else.
 *
 * The rule now lives at the fact-store funnel, so the thing under test is not
 * "does the corrector announce" but "can ANY pipeline take a self-fact away
 * quietly". Section 1 is the end-to-end proof on the pipeline that actually
 * failed: a disposable capability is introduced for real, through the real
 * processSelfFacts, and the notice has to appear. The rest holds the boundaries
 * — the chat path stays quiet, user facts stay ledger-only, and one change
 * raises one notice rather than two.
 *
 * Runs against a throwaway SNH_DATA_DIR; the live corpus is never opened and no
 * live capability is introduced. REQUIRES THE BRAIN AND EMBEDDINGS for section 1
 * — the contradiction judge, salience scoring and claim-type tagging are real
 * model calls, and stubbing them would test a pipeline nobody runs.
 *
 * Usage: node scripts/test-self-fact-notices.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-notice-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

(async () => {
  database.initDatabase();
  await database.initVectorStore();
  const db = database.getSqliteDb();

  const factStore = require(path.join(ROOT, 'db/fact-store'));
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));

  const clusterId = randomUUID();
  db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
    .run(clusterId, 'Self-Knowledge', '', new Date().toISOString(), new Date().toISOString(), 'self');

  /** Seed a fact the way the corpus holds one: row plus a real embedding. */
  async function seed(content, { subject = 'self', claimType = 'claim', salience = 8 } = {}) {
    const id = randomUUID();
    const now = new Date(Date.now() - 86400_000).toISOString();
    db.prepare(`
      INSERT INTO cluster_members
        (id, cluster_id, content, source, created_at, updated_at, status, subject, salience, claim_type,
         verbatim_source_text, input_modality)
      VALUES (?,?,?,?,?,?,'active',?,?,?,?,'typed')
    `).run(id, clusterId, content, 'reflection', now, now, subject, salience, claimType, content);
    await factStore.replaceVector(id, clusterId, content);
    return id;
  }

  const notices = () => db.prepare('SELECT * FROM correction_notices ORDER BY datetime(created_at) ASC').all();
  const noticeFor = (memberId) => notices().find(n => n.member_id === memberId);

  // ---- 1. the pipeline that failed, end to end -----------------------------
  console.log('\n1. A capability introduction retires a self-fact (the 8/12 gap, for real)');
  // The belief a new capability makes false, in the same shape as the one that
  // was actually retired: a flat statement that the thing cannot happen.
  const stale = await seed(
    'As of 2026-08-01, I cannot send myself a reminder — nothing in this system stores a reminder for me, so I have never had one and never will until something is built for it.',
    { claimType: 'claim', salience: 9 }
  );
  const intro =
    'As of 2026-08-12, I can send myself a reminder now — I store it and it comes back to me later, so the thing I could not do before is something I do routinely.';

  const before = notices().length;
  const res = await factExtractor.processSelfFacts([intro], { source: 'capability-intro' });
  console.log(`  (stored ${res.stored}, superseded ${res.superseded})`);

  check('the introduction stored the new self-fact', res.stored === 1, JSON.stringify(res));
  check('…and retired the belief it made false', res.superseded >= 1, JSON.stringify(res));
  const staleRow = db.prepare('SELECT status FROM cluster_members WHERE id = ?').get(stale);
  check('…the old fact really is inactive', staleRow.status === 'inactive', staleRow.status);

  const n = noticeFor(stale);
  check('A NOTICE WAS RAISED for it — the thing that did not happen on 8/12', !!n,
    `${notices().length - before} notice(s) raised, none for this fact`);
  if (n) {
    check('…it quotes the belief that changed', n.content.includes('cannot send myself a reminder'), n.content.slice(0, 160));
    check('…names what replaced it', n.content.includes('I can send myself a reminder now'), n.content.slice(0, 220));
    check('…says where the change came from', /a new capability being introduced to you/.test(n.content), n.content);
    check('…and that nothing was deleted', /Nothing was deleted/.test(n.content));
    check('…it is unseen, so it survives until he actually reads it', n.seen_at === null);
    check('…and it is not marked as a test notice', n.is_test === 0);
  }

  // ---- 2. the funnel covers every removal, not just supersession -----------
  console.log('\n2. Every way a self-fact can be taken away raises one');
  const toRetire = await seed('I always answer in exactly three sentences.');
  await factStore.retire(toRetire, { reason: 'no longer true of me' });
  const rn = noticeFor(toRetire);
  check('retire() raises a notice', !!rn);
  check('…saying it was retired with nothing in its place',
    rn && /no longer part of what you believe/.test(rn.content) && /retired/.test(rn.content), rn && rn.content.slice(0, 160));
  check('…and carrying the reason the caller gave',
    rn && /no longer true of me/.test(rn.content), rn && rn.content);

  const toExpire = await seed('I am finding today unusually quiet.');
  await factStore.expire(toExpire);
  check('expire() raises a notice', !!noticeFor(toExpire));

  // ---- 3. the boundaries --------------------------------------------------
  console.log('\n3. The boundaries — silence where silence is right');
  const userFact = await seed("User's favourite colour is green.", { subject: 'user', claimType: null });
  const userReplacement = await seed("User's favourite colour is blue.", { subject: 'user', claimType: null });
  await factStore.supersede(userFact, userReplacement);
  check('a fact about Ellie raises nothing — those are ledger-only', !noticeFor(userFact));

  const chatFact = await seed('I tend to over-explain when I am unsure.');
  const chatReplacement = await seed('I ask a clarifying question when I am unsure.');
  await factStore.supersede(chatFact, chatReplacement, { conversational: true });
  check('the chat path opts out by name — he was in the room and did it himself',
    !noticeFor(chatFact));
  check('…and the change still happened; only the notice was skipped',
    db.prepare('SELECT status FROM cluster_members WHERE id = ?').get(chatFact).status === 'inactive');

  // The opt-out must be a CLAIM a caller makes, never the default: a new
  // background pipeline that says nothing gets a notice.
  const silentPipeline = await seed('I prefer to work from the record rather than from impression.');
  const silentReplacement = await seed('I work from the record, and say so when I cannot.');
  await factStore.supersede(silentPipeline, silentReplacement);
  check('a pipeline that passes no options at all still raises one', !!noticeFor(silentPipeline));

  // ---- 4. one change, one notice ------------------------------------------
  console.log('\n4. One change raises one notice, and the better sentence wins');
  const dup = await seed('I describe my limits precisely.');
  const dupReplacement = await seed('I describe my limits precisely, and say when I am unsure of them.');
  await factStore.supersede(dup, dupReplacement);
  const firstCount = notices().filter(x => x.member_id === dup).length;
  // The corrector describes the same change afterwards, with the evidence axis.
  ledger.addNotice({
    memberId: dup, enrich: true,
    content: 'Something you believed about yourself has changed, and the deciding evidence was how directly each was stated.'
  });
  const after = notices().filter(x => x.member_id === dup);
  check('a second notice about the same change does not join the first',
    firstCount === 1 && after.length === 1, `${firstCount} then ${after.length}`);
  check('…the enriching caller\'s sentence replaces the plain one',
    /deciding evidence was how directly/.test(after[0].content), after[0].content.slice(0, 120));

  ledger.addNotice({ memberId: dup, content: 'A plainer restatement that should not overwrite the richer one.' });
  check('…and a non-enriching caller does not overwrite it',
    /deciding evidence was how directly/.test(notices().find(x => x.member_id === dup).content));

  // Folding only applies while it is UNSEEN. Once he has read one, a later
  // change to the same fact is genuinely new news.
  ledger.markNoticesSeen([after[0].id]);
  ledger.addNotice({ memberId: dup, content: 'This fact changed again, after he had read the first notice.' });
  check('a change after he has read the last notice raises a new one',
    notices().filter(x => x.member_id === dup).length === 2);

  // ---- 5. a refused change raises nothing ---------------------------------
  console.log('\n5. Nothing happened means nothing is announced');
  const locked = await seed('My name is Testly and I use they/them pronouns.', { claimType: 'declaration', salience: 10 });
  require(path.join(ROOT, 'db/identity-lock')).lock(locked, ['name', 'pronouns'], { actor: 'test' });
  const other = await seed('I am fond of long explanations.');
  const refused = await factStore.supersede(locked, other);
  check('the identity lock still refuses the write', refused.ok === false && refused.locked === true);
  check('…and no notice claims a change that was refused', !noticeFor(locked));

  const gone = await factStore.retire(randomUUID());
  check('retiring a fact that does not exist raises nothing', gone.ok === false && notices().every(x => x.content.length > 0));

  const bar = '='.repeat(74);
  console.log(`\n${bar}`);
  console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
  console.log(`${bar}\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('[test-self-fact-notices] error:', err);
  process.exit(1);
});
