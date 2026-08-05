#!/usr/bin/env node
/**
 * Dry-run the passive extraction pipeline over stored conversations.
 *
 * Runs the REAL intake path — factExtractor.planExtraction, the same function
 * live chat calls — and prints what it WOULD write, without writing any of it.
 * planExtraction is read-only by construction: it makes model calls and vector
 * reads, and every mutation lives in applyExtraction, which this script never
 * calls. That separation is the point. A rehearsal that exercises a copy of the
 * pipeline proves nothing about the pipeline.
 *
 * SAFE TO RUN against the live corpus. It writes no facts, no vectors, no log
 * lines, no questions.
 *
 * Usage:
 *   node scripts/dryrun-extract.js <conversationId|prefix>   # one conversation
 *   node scripts/dryrun-extract.js --fixture <name>          # a named fixture
 *   node scripts/dryrun-extract.js --list                    # fixtures + their sources
 *   node scripts/dryrun-extract.js --text "<what the user said>" [--modality stt]
 *
 * Options:
 *   --modality stt|typed|unknown   override the recorded input modality
 *   --json                         machine-readable output
 *   --max N                        stop after N exchanges
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));

// ---------------------------------------------------------------------------
// Fixtures. Each names the source in the LIVE corpus that produced a defect in
// the spec's F1–F5 table.
//
// Two of them have no source left: the conversations that produced the
// machine-gun triple (4a0be947…) and the Roscoe fact were DELETED from the
// database — questions.origin_conversation_id still points at 4a0be947, and no
// row for it survives. Those entries carry a `synthetic` source text
// reconstructed from the fact and the daily log, and say so in their output, so
// a reconstruction is never mistaken for a replay.
// ---------------------------------------------------------------------------
const FIXTURES = {
  f1: {
    label: 'F1 mike-stt-mishear',
    conversation: '57a5cc99',
    expect: 'ZERO name facts. "Hey, it\'s Mike not picking up the right words." is a mis-transcribed "mic".'
  },
  f2: {
    label: 'F2 machine-gun-triple',
    conversation: '87649607',
    note: 'The conversation that ORIGINALLY produced the triple (4a0be947…) has been deleted from the database. 87649607 is the only surviving conversation whose text asserts the belief.',
    expect: 'One fact, one row. Re-assertions fold into the held fact as corroborations.'
  },
  f3a: {
    label: 'F3 transient-events / yard holes',
    conversation: 'd19d4280',
    message: '60e0dd93',
    expect: 'Routed to the day\'s log, not the fact store.'
  },
  f3b: {
    label: 'F3 transient-events / life fatigue',
    conversation: '0992ad90',
    message: 'fe178ce5',
    expect: 'Routed to the day\'s log, not the fact store.'
  },
  f3c: {
    label: 'F3 transient-events / restless night',
    synthetic: 'Roscoe had a restless night last night, he kept getting up and moving around.',
    note: 'Source conversation deleted — no message in the database mentions Roscoe. This text is reconstructed from the stored fact and the 2026-07-26 daily log.',
    expect: 'The restless night goes to the log. Note the atomicity/F3 tension in the report.'
  },
  f4: {
    label: 'F4 casper-subset',
    conversation: '37b9d818',
    expect: 'The subset does not survive alongside its superset (Phase 2b — the corrector).'
  },
  f5: {
    label: 'F5 compound-single-file',
    conversation: 'e5deb6d0',
    message: 'e8c004ee',
    expect: 'Atomic splits, each retrievable on its own term.'
  }
};

function parseArgs(argv) {
  const out = { positional: [], modality: null, json: false, max: null, fixture: null, text: null, list: false, repeat: 1 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--json') out.json = true;
    else if (a === '--list') out.list = true;
    else if (a === '--modality') out.modality = argv[++i];
    else if (a === '--max') out.max = parseInt(argv[++i], 10);
    else if (a === '--fixture') out.fixture = argv[++i];
    else if (a === '--text') out.text = argv[++i];
    else if (a === '--repeat') out.repeat = Math.max(1, parseInt(argv[++i], 10) || 1);
    else out.positional.push(a);
  }
  return out;
}

/** Pair a conversation's messages into (user, assistant) exchanges, in order. */
function loadExchanges(sql, conversationPrefix, onlyMessagePrefix) {
  const convo = sql.prepare(
    'SELECT id, title, created_at, initiated_by FROM conversations WHERE id LIKE ? LIMIT 1'
  ).get(`${conversationPrefix}%`);
  if (!convo) return { convo: null, exchanges: [] };

  const rows = sql.prepare(
    'SELECT id, role, content, timestamp, input_modality FROM messages WHERE conversation_id = ? ORDER BY datetime(timestamp) ASC, rowid ASC'
  ).all(convo.id);

  const exchanges = [];
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].role !== 'user') continue;
    const next = rows[i + 1];
    exchanges.push({
      messageId: rows[i].id,
      timestamp: rows[i].timestamp,
      userMessage: rows[i].content,
      assistantMessage: next && next.role === 'assistant' ? next.content : '',
      inputModality: rows[i].input_modality || 'unknown'
    });
  }
  const filtered = onlyMessagePrefix
    ? exchanges.filter(e => e.messageId.startsWith(onlyMessagePrefix))
    : exchanges;
  return { convo, exchanges: filtered };
}

const trunc = (s, n) => {
  const t = String(s || '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n)}…` : t;
};

/**
 * In-run repeat check.
 *
 * planExtraction's repeat detection asks the LIVE corpus "do I already hold
 * this?". In a real replay the answer changes as the replay proceeds — exchange
 * 1's fact is in the database by the time exchange 2 runs, because replay writes
 * are serial through the fact-store funnel. A dry run never writes, so it has to
 * carry that ledger itself, or it would report three rows where a replay would
 * write one.
 *
 * Uses the same two tests the live path uses — exact match, then embedding
 * similarity above the configured floor confirmed by the same judge — so the
 * rehearsal and the real thing agree.
 */
async function repeatWithinRun(ledger, text) {
  const mc = require(path.join(ROOT, 'db/memory-clusters'));
  const fx = require(path.join(ROOT, 'db/fact-extractor'));
  const { getConfig } = require(path.join(ROOT, 'db/config'));
  const floor = getConfig().memory?.extraction?.repeatSimilarityFloor ?? 0.80;

  const key = text.trim().toLowerCase();
  const exact = ledger.find(l => l.key === key);
  if (exact) return { of: exact, similarity: 1, how: 'exact' };

  const emb = await mc.generateEmbedding(text);
  if (!emb) return null;
  for (const l of ledger) {
    if (!l.emb) continue;
    const sim = mc.cosineSimilarity(Array.from(emb), Array.from(l.emb));
    if (sim < floor) continue;
    const { same } = await fx.judgeSameAssertion(text, l.text);
    if (same) return { of: l, similarity: sim, how: 'semantic' };
  }
  ledger.push({ key, text, emb, exchange: null });
  return null;
}

async function printPlan(plan, idx, ledger) {
  const fx = require(path.join(ROOT, 'db/fact-extractor'));
  console.log(`\n  ── exchange ${idx} ${'─'.repeat(56)}`);
  console.log(`  said (${plan.inputModality}): "${trunc(plan.userMessage, 220)}"`);

  const modelFacts = plan.proposed.facts.length;
  const modelEvents = plan.proposed.events.length;
  console.log(`  model proposed: ${modelFacts} fact(s), ${modelEvents} event(s)`);

  for (const s of plan.splits) {
    console.log(`  SPLIT   "${trunc(s.from, 110)}"  (${s.why})`);
    for (const p of s.into) console.log(`            → "${p}"`);
  }
  for (const r of plan.routedToLog) {
    console.log(`  → LOG   "${trunc(r.text, 110)}"`);
    console.log(`            ${r.why}`);
  }
  for (const e of plan.events) {
    if (plan.routedToLog.some(r => r.text === e.text)) continue;
    console.log(`  → LOG   "${trunc(e.text, 110)}"  (extracted as an event)`);
  }
  for (const ref of plan.refusals) {
    console.log(`  REFUSE  "${trunc(ref.text, 110)}"`);
    console.log(`            [${ref.rule}] ${ref.detail}`);
  }
  for (const rep of plan.repeats) {
    console.log(`  REPEAT  "${trunc(rep.text, 110)}"`);
    console.log(`            folds into ${String(rep.existingId).slice(0, 8)} "${trunc(rep.existingContent, 80)}" ` +
                `(${rep.detectedBy}, sim ${Number(rep.similarity).toFixed(4)}) — salience ${rep.existingSalience} → ${rep.plannedSalience ?? rep.existingSalience}, corroboration recorded`);
  }
  for (const f of plan.facts) {
    // Would this be a NEW row, given what earlier exchanges in this same run
    // would already have written? The live-corpus check inside planExtraction
    // only sees what is stored today.
    const dupInRun = await repeatWithinRun(ledger, f.text);
    if (dupInRun) {
      console.log(`  REPEAT  "${trunc(f.text, 110)}"`);
      console.log(`            restates what exchange ${dupInRun.of.exchange} of this same run would have written ` +
                  `("${trunc(dupInRun.of.text, 70)}", ${dupInRun.how}, sim ${dupInRun.similarity.toFixed(4)}) — one row, not two`);
      continue;
    }
    ledger[ledger.length - 1].exchange = idx;
    console.log(`  STORE   "${f.text}"`);
    console.log(`            salience ${f.salience}/10 — ${trunc(f.salienceRationale, 150)}`);
    if (f.corrects) console.log(`            corrects: ${f.corrects}`);
  }
  for (const s of plan.supersessions) {
    console.log(`  SUPERSEDE ${String(s.oldMemberId).slice(0, 8)} "${trunc(s.oldContent, 90)}"`);
    console.log(`            replaced by "${trunc(s.newFact, 90)}"`);
  }
  for (const u of plan.uncertainties) {
    console.log(`  ASK     conflict unresolved: "${trunc(u.oldContent, 80)}" vs "${trunc(u.newFact, 80)}"`);
  }
  for (const line of fx.describeRecall(plan)) console.log(`  recall  ${line}`);
  if (plan.truncated.facts || plan.truncated.events) {
    console.log(`  CEILING ${plan.truncated.facts} fact(s) and ${plan.truncated.events} event(s) dropped by the per-exchange cap`);
  }
  if (plan.gapQuestion) console.log(`  gap?    "${plan.gapQuestion}"`);
  if (!plan.facts.length && !plan.events.length && !plan.refusals.length && !plan.repeats.length) {
    console.log('  (nothing to record)');
  }
}

(async () => {
  const args = parseArgs(process.argv.slice(2));

  if (args.list) {
    console.log('\nFixtures:\n');
    for (const [key, f] of Object.entries(FIXTURES)) {
      console.log(`  ${key.padEnd(5)} ${f.label}`);
      console.log(`        source: ${f.synthetic ? 'SYNTHETIC (source conversation deleted)' : `conversation ${f.conversation}${f.message ? `, message ${f.message}` : ''}`}`);
      if (f.note) console.log(`        note:   ${f.note}`);
      console.log(`        expect: ${f.expect}\n`);
    }
    process.exit(0);
  }

  db.initDatabase();
  await db.initVectorStore();
  const sql = db.getSqliteDb();
  const fx = require(path.join(ROOT, 'db/fact-extractor'));

  // Guard rail, belt and braces: if anything in this process tries to write a
  // fact, fail loudly rather than quietly corrupting the corpus we are measuring.
  const factStore = require(path.join(ROOT, 'db/fact-store'));
  for (const fn of ['supersede', 'retire', 'reword', 'absorbRepeat', 'absorbDuplicate', 'recordCorroboration']) {
    factStore[fn] = () => { throw new Error(`dry-run: fact-store.${fn} must not be called`); };
  }

  const fixture = args.fixture ? FIXTURES[args.fixture.toLowerCase()] : null;
  if (args.fixture && !fixture) {
    console.error(`Unknown fixture "${args.fixture}". Try --list.`);
    process.exit(2);
  }

  let header, exchanges;
  const syntheticText = args.text || (fixture && fixture.synthetic);

  if (syntheticText) {
    header = fixture ? `${fixture.label}  [SYNTHETIC SOURCE]` : 'ad-hoc text';
    // --repeat N replays the same utterance N times as consecutive exchanges,
    // which is the shape that produced the machine-gun triple: the same belief
    // asserted three times inside 84 seconds.
    exchanges = Array.from({ length: args.repeat }, () => ({
      messageId: null, timestamp: null,
      userMessage: syntheticText, assistantMessage: '',
      inputModality: args.modality || 'unknown'
    }));
  } else {
    const target = fixture ? fixture.conversation : args.positional[0];
    if (!target) {
      console.error('Give a conversation id, --fixture <name>, or --text "<message>". Try --list.');
      process.exit(2);
    }
    const loaded = loadExchanges(sql, target, fixture && fixture.message);
    if (!loaded.convo) {
      console.error(`No conversation matching "${target}".`);
      process.exit(2);
    }
    header = `${fixture ? `${fixture.label}  ` : ''}conversation ${loaded.convo.id}  "${loaded.convo.title}"  (${loaded.convo.created_at}, initiated by ${loaded.convo.initiated_by})`;
    exchanges = loaded.exchanges;
  }

  if (args.max) exchanges = exchanges.slice(0, args.max);

  console.log(`\n${'='.repeat(78)}`);
  console.log(`DRY RUN — nothing is written`);
  console.log(header);
  if (fixture) {
    if (fixture.note) console.log(`NOTE:   ${fixture.note}`);
    console.log(`EXPECT: ${fixture.expect}`);
  }
  console.log(`${exchanges.length} exchange(s)`);
  console.log('='.repeat(78));

  const ledger = []; // {key, text, emb, exchange} — what this run would have written so far
  const plans = [];
  let i = 0;
  for (const ex of exchanges) {
    i++;
    const plan = await fx.planExtraction({
      userMessage: ex.userMessage,
      assistantMessage: ex.assistantMessage,
      conversationId: null,                       // read-only: nothing is attributed
      messageId: ex.messageId,
      inputModality: args.modality || ex.inputModality
    });
    plans.push(plan);
    if (!args.json) await printPlan(plan, i, ledger);
  }

  const totals = plans.reduce((a, p) => ({
    proposed: a.proposed + p.proposed.facts.length,
    facts: a.facts + p.facts.length,
    events: a.events + p.events.length,
    repeats: a.repeats + p.repeats.length,
    refusals: a.refusals + p.refusals.length,
    splits: a.splits + p.splits.length,
    supersessions: a.supersessions + p.supersessions.length
  }), { proposed: 0, facts: 0, events: 0, repeats: 0, refusals: 0, splits: 0, supersessions: 0 });

  const uniqueRows = new Set();
  for (const p of plans) for (const f of p.facts) uniqueRows.add(f.text.trim().toLowerCase());

  if (args.json) {
    console.log(JSON.stringify({ header, totals, uniqueRows: uniqueRows.size, plans }, null, 2));
  } else {
    console.log(`\n${'='.repeat(78)}`);
    console.log(`SUMMARY  proposed ${totals.proposed} → would store ${uniqueRows.size} row(s)`);
    console.log(`         ${totals.splits} compound split(s), ${totals.events} event(s) to the log, ` +
                `${totals.repeats} repeat(s) folded, ${totals.refusals} refused, ${totals.supersessions} supersession(s)`);
    console.log('='.repeat(78));
  }
  process.exit(0);
})().catch(err => {
  console.error('dry-run failed:', err);
  process.exit(1);
});
