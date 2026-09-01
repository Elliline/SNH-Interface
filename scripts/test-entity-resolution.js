#!/usr/bin/env node
/**
 * TIERED ENTITY RESOLUTION — against a THROWAWAY store.
 *
 * The half of the entity-subject redesign that CREATES and ASSIGNS. Same
 * isolation as scripts/test-entity-subject.js: redirect SNH_DATA_DIR before
 * anything resolves a store, and throw the directory away at the end. Live data
 * is never opened, and no chat turn is sent anywhere.
 *
 * Deterministic — no engine. The resolution path is rules plus registry
 * lookups by design, precisely so a red result here is a real regression and
 * never a slow GPU.
 *
 * Usage: node scripts/test-entity-resolution.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-entity-resolution-test-'));
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
  const rules = require(path.join(ROOT, 'db/extraction-rules'));
  const subjectCheck = require(path.join(ROOT, 'db/subject-check'));
  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const sql = db.getSqliteDb();

  const notices = () => ledger.unseenNotices(100).map(n => n.content);
  const bellCount = () => {
    try { return sql.prepare('SELECT COUNT(*) n FROM initiatives').get().n; } catch { return 0; }
  };
  const bellBefore = bellCount();

  // ------------------------------------------------------------ 1. new client
  section('1. A first mention of a client creates an organization, and says so');

  const r1 = entities.resolveMentions('I picked up a new client, Newport Dental Clinic.');
  const clinic = entities.resolve('Newport Dental Clinic');
  ok('an entity was created', r1.created.length === 1, JSON.stringify(r1.created.map(c => c.entity.name)));
  ok('it is an organization', clinic.tier === 'clear' && clinic.entity.type === 'organization',
     `${clinic.tier}/${clinic.entity && clinic.entity.type}`);
  ok('no question was raised for it', r1.questions.length === 0, JSON.stringify(r1.questions.map(q => q.mention.name)));
  ok('the reply is told to mention it', notices().some(c => /Newport Dental Clinic/.test(c)),
     JSON.stringify(notices()));
  ok('the creation is in the corrections ledger',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='entity-create' AND target_text=?")
       .get('Newport Dental Clinic'));
  ok('nothing went to the bell', bellCount() === bellBefore, `${bellCount()} vs ${bellBefore}`);

  // -------------------------------------------------------------- 2. by alias
  section('2. A second mention by alias attaches — no duplicate');

  const before2 = entities.list().length;
  const r2 = entities.resolveMentions('Newport Dental called about the invoice.');
  ok('nothing new was created', r2.created.length === 0, JSON.stringify(r2.created.map(c => c.entity.name)));
  ok('it attached to the same entity',
     r2.assignments.length === 1 && r2.assignments[0].tier === 'clear' &&
     r2.assignments[0].entity.id === clinic.entity.id,
     JSON.stringify(r2.assignments.map(a => `${a.tier}:${a.entity && a.entity.name}`)));
  ok('the registry did not grow', entities.list().length === before2);

  // the acronym alias minted at creation is what makes the short form resolve
  const r2b = entities.resolveMentions('NDC emailed me twice.');
  ok('the acronym alias resolves to the same entity too',
     r2b.created.length === 0 && r2b.assignments[0] && r2b.assignments[0].entity.id === clinic.entity.id,
     JSON.stringify(entities.namesOf(clinic.entity)));

  // ------------------------------------------------------------- 3. a contact
  section('3. A contact at that client becomes a person, linked to the org');

  const r3 = entities.resolveMentions('Sarah Whitfield is my contact at Newport Dental Clinic.');
  const sarah = entities.resolve('Sarah Whitfield');
  ok('a person was created', sarah.tier === 'clear' && sarah.entity.type === 'person',
     `${sarah.tier}/${sarah.entity && sarah.entity.type}`);
  ok('she is linked to the organization',
     sarah.entity && sarah.entity.org_id === clinic.entity.id,
     `org_id=${sarah.entity && sarah.entity.org_id}`);
  ok('the clinic was recognised, not created again',
     r3.created.filter(c => c.entity.type === 'organization').length === 0);

  // ------------------------------------------------------------------ 4. a dog
  section('4. One of her dogs by name becomes an animal');

  entities.resolveMentions('My dog Cece is a Rottweiler and she needs her shots.');
  const cece = entities.resolve('Cece');
  ok('an animal was created', cece.tier === 'clear' && cece.entity.type === 'animal',
     `${cece.tier}/${cece.entity && cece.entity.type}`);

  // ------------------------------------------- 5. an attribute is not a subject
  section("5. \"Bob's house\" is a fact about Bob, not an entity called house");

  entities.resolveMentions('My friend Bob Chen fixed the fence yesterday.');
  const bob = entities.resolve('Bob Chen');
  ok('Bob himself is a person entity', bob.tier === 'clear' && bob.entity.type === 'person');

  const before5 = entities.list().length;
  const r5 = entities.resolveMentions("Bob Chen's house has bad wifi.");
  ok('no entity was created for the house', entities.list().length === before5,
     JSON.stringify(entities.list().map(e => e.name)));
  ok('nothing named house exists at all',
     !entities.list().some(e => /house/i.test(e.name)));
  ok('the mention extractor never proposed the house',
     !rules.entityMentions("Bob Chen's house has bad wifi.").some(m => /house/i.test(m.name)),
     JSON.stringify(rules.entityMentions("Bob Chen's house has bad wifi.").map(m => m.name)));
  ok('and the fact files against Bob',
     entities.entityForFact("Bob Chen's house has bad wifi.", r5.assignments) === bob.entity.id);

  // A mention that is merely mentioned is NOT what the fact is about.
  ok('a fact that only mentions an entity stays with the default subject',
     entities.entityForFact('User works at Newport Dental Clinic on Tuesdays', r1.assignments) === null);

  // ------------------------------------------------------------- 6. ambiguous
  section('6. An ambiguous mention asks in the reply — not a silent create, not the bell');

  entities.resolveMentions('My neighbour Bob Marley brought the mail in.');
  const before6 = entities.list().length;
  const noticesBefore = notices().length;
  const r6 = entities.resolveMentions('Bob called about the thing.');
  ok('a question was raised', r6.questions.length === 1, JSON.stringify(r6.questions));
  ok('nothing was created', r6.created.length === 0 && entities.list().length === before6);
  ok('the question names both candidates',
     r6.questions[0] && /Bob Chen/.test(r6.questions[0].ask) && /Bob Marley/.test(r6.questions[0].ask),
     r6.questions[0] && r6.questions[0].ask);
  ok('nothing went to the bell for it', bellCount() === bellBefore, `${bellCount()}`);

  // A cue-proven mention of an unknown KIND asks rather than guessing.
  const r6b = entities.resolveMentions('Fathoms called to reschedule.');
  ok('an unknown kind asks rather than guessing a type',
     r6b.created.length === 0 && r6b.questions.length === 1,
     JSON.stringify(r6b.questions.map(q => q.mention.name)));

  // -------------------------------------- 6c. junk found on REAL messages
  section('6c. Junk the cue rule produced on her real messages (2026-09-01)');

  // Verbatim excerpts from Athena's own store. Each one produced a junk entity
  // in the extraction verification runs, and each is here because a synthetic
  // sentence would not have found it — these are the shapes real writing has.
  const REAL = {
    fromMsp: "From me, for your store:\n\n1. I own MettaSphere. It's an MSP on the Oregon coast, helping around 20 businesses with their IT.",
    junoOrg: "The merge-loss audit did verifiably restore two genuine losses to Juno's store later that day, and their visibility was confirmed.",
    junoPerson: "This box is to be used for you for testing. You have an older sister named Juno and she is also running on the Qwen3.8 27b model."
  };
  const namesIn = (t) => rules.entityMentions(t).map(m => m.name);
  const typeOf = (t, n) => (rules.entityMentions(t).find(m => m.name === n) || {}).type;

  // "From me, for your store" — the after-scan reached past a comma and a
  // preposition to "store" and made an organisation out of an email header.
  ok('"From" is not an entity', !namesIn(REAL.fromMsp).includes('From'),
     JSON.stringify(namesIn(REAL.fromMsp)));

  // "MSP" is a TYPE CUE. A word that says what kind of thing something is can
  // never be the thing's name.
  ok('"MSP" is not an entity', !namesIn(REAL.fromMsp).includes('MSP'),
     JSON.stringify(namesIn(REAL.fromMsp)));

  // …and the real subject in that sentence still survives, because the fix is
  // precision, not a retreat from the cue requirement.
  ok('MettaSphere is still found in the same sentence', namesIn(REAL.fromMsp).includes('MettaSphere'),
     JSON.stringify(namesIn(REAL.fromMsp)));

  // "Juno's store" — the possessed noun is not Juno's type. Reading across the
  // apostrophe turned a person into an organisation.
  ok('"Juno\'s store" does not type Juno as an organization', typeOf(REAL.junoOrg, 'Juno') !== 'organization',
     `got ${typeOf(REAL.junoOrg, 'Juno')}`);

  // The same name in a sentence that DOES say what she is still types her.
  ok('"an older sister named Juno" still types her as a person', typeOf(REAL.junoPerson, 'Juno') === 'person',
     `got ${typeOf(REAL.junoPerson, 'Juno')}`);

  // And once she is known, a later mention resolves rather than re-creating
  // under whatever cue happens to be nearby.
  const junoEnt = entities.create({ name: 'Juno', type: 'person', relationship: 'sister' });
  const beforeJ = entities.list().length;
  const rJuno = entities.resolveMentions(REAL.junoOrg);
  ok('a known name is not re-created under a different type',
     rJuno.created.length === 0 && entities.list().length === beforeJ,
     JSON.stringify(rJuno.created.map(c => `${c.entity.name}:${c.entity.type}`)));
  ok('…and she is still a person', entities.get(junoEnt.id).type === 'person');

  // The silence targets, on real phrasing.
  for (const [label, text] of [
    ['Monday', 'I will look at it Monday when the office opens.'],
    ['Oregon', 'We are based in Oregon and it rains a lot.'],
    ['header block', 'From: adrienne@example.com\nTo: Ellie\nSubject: phones\nRe: the open house']
  ]) {
    ok(`no entity from "${label}"`, rules.entityMentions(text).length === 0,
       JSON.stringify(rules.entityMentions(text).map(m => m.name)));
  }

  // ------------------------------------------- 6d. products, makers and users
  section('6d. Products: made-by is a column, uses is a table');

  const rP = entities.resolveMentions('ISH is getting Opera from Oracle next month.');
  const opera = entities.resolve('Opera').entity;
  const oracle = entities.resolve('Oracle').entity;
  ok('a maker cue creates the product', opera && opera.type === 'product', opera && opera.type);
  ok('…and creates the maker as an organization', oracle && oracle.type === 'organization',
     oracle && oracle.type);
  ok('…and points the product at it (made by)', opera && opera.org_id === oracle.id);
  ok('the org link on a product reads as "made by"', entities.orgLinkLabel('product') === 'made by');
  ok('the maker link is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='entity-link' AND target_id=?").get(opera.id));

  // A usage cue makes a LINK, and the product is not created twice.
  const beforeUse = entities.list().length;
  const ishOrg = entities.create({ name: 'Inn at Spanish Head', type: 'organization', aliases: ['ISH'] });
  entities.resolveMentions('Inn at Spanish Head runs Opera for the front desk.');
  ok('usage does not create a second Opera', entities.list().filter(e => e.name === 'Opera').length === 1,
     JSON.stringify(entities.list().filter(e => /opera/i.test(e.name)).map(e => e.name)));
  ok('the uses link exists', entities.usesOf(ishOrg.id).some(e => e.id === opera.id),
     JSON.stringify(entities.usesOf(ishOrg.id).map(e => e.name)));
  ok('…and reads from the other side too', entities.usersOf(opera.id).some(e => e.id === ishOrg.id));
  ok('the uses link is ledgered',
     !!sql.prepare("SELECT 1 FROM corrections_ledger WHERE action='entity-link' AND target_id=? AND survivor_id=?")
       .get(ishOrg.id, opera.id));

  // A SECOND client on the same product links to the same row.
  const clinic2 = entities.create({ name: 'Lincoln City Animal Clinic', type: 'organization', aliases: ['LCAC'] });
  entities.resolveMentions('Lincoln City Animal Clinic runs Opera as well.');
  ok('a second client links to the SAME product row',
     entities.usersOf(opera.id).length === 2 && entities.list().filter(e => e.name === 'Opera').length === 1,
     JSON.stringify(entities.usersOf(opera.id).map(e => e.name)));
  ok('and that client uses it', entities.usesOf(clinic2.id).some(e => e.id === opera.id));

  // Repeating the sentence does not mint a second link.
  const linkCount = () => sql.prepare("SELECT COUNT(*) n FROM entity_links WHERE status='active'").get().n;
  const beforeRepeat = linkCount();
  entities.resolveMentions('Lincoln City Animal Clinic runs Opera as well.');
  ok('a repeated usage mention does not duplicate the link', linkCount() === beforeRepeat,
     `${linkCount()} vs ${beforeRepeat}`);

  // A product with NO maker cue gets no maker, and none is invented.
  entities.resolveMentions('The clinic is on Covetrus Pulse now.');
  const covetrus = entities.resolve('Covetrus Pulse').entity;
  ok('a product with no maker cue is still created', covetrus && covetrus.type === 'product',
     covetrus && covetrus.type);
  ok('…with NO maker invented for it', covetrus && !covetrus.org_id, covetrus && covetrus.org_id);

  // The link table is an index, like the entity table.
  ok('entity_links is index-only', entities.assertIndexOnly() === true);
  ok('relationsOf answers both edges',
     entities.relationsOf(opera.id).usedBy.length === 2 &&
     entities.relationsOf(opera.id).orgLink.entity.id === oracle.id &&
     entities.relationsOf(oracle.id).products.some(pp => pp.id === opera.id));

  // ----------------------------------------------- 7. the door is still shut
  section('7. The two Juno cases still die at the door, after resolution has run');

  // Resolution runs over the clinic email first, exactly as intake would.
  const rJ = entities.resolveMentions(CLINIC_EMAIL_SOURCE);
  // These two already exist from 6d; reuse them rather than minting duplicates.
  const ish = ishOrg;
  const lcac = clinic2;
  const known = entities.list().map(subjectCheck.forEntity);
  const user = entities.userEntity();

  const badAddress =
    'User works at ISH (Inn At Spanish Head), a business located at 4090 NE Hwy 101, Lincoln City, OR 97367, ' +
    'and her role is effectively IT manager, involving leading teams and mediating between vendors like IMB and Shift4.';
  const vA = subjectCheck.check(badAddress, subjectCheck.forEntity(user), CLINIC_EMAIL_SOURCE, { knownEntities: known });
  ok('JUNO CASE 1 — the ISH street address still does NOT verify', vA.verified === false, vA.detail);

  const vB = subjectCheck.check("User's business email is lcac2009@live.com",
    subjectCheck.forEntity(user), CLINIC_EMAIL_SOURCE, { knownEntities: known });
  ok('JUNO CASE 2 — the clinic email as hers still does NOT verify', vB.verified === false, vB.detail);

  ok('creating an entity did not open a bypass — the gate reads the source, not the registry',
     vA.reason !== 'no-specifics' && vB.reason !== 'no-specifics', `${vA.reason}/${vB.reason}`);

  // ------------------------------------------------------- registry integrity
  section('8. The registry is still an index, and locks are untouched');

  ok('entities table is still index-only', entities.assertIndexOnly() === true);
  ok('the self pointer still points at the self entity',
     entities.selfEntity() && entities.pointer('self').id === entities.selfEntity().id);
  ok('the identity check still agrees', entities.identityAgrees().ok === true, entities.identityAgrees().why);
  ok('every created entity has one of the four starting types',
     entities.list().every(e => entities.TYPES.includes(e.type) || ['self', 'user'].includes(e.type)),
     JSON.stringify(entities.list().map(e => `${e.name}:${e.type}`)));

  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => {
  console.error('\nCRASH:', e.stack || e.message);
  process.exit(1);
});
