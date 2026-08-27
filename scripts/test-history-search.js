#!/usr/bin/env node
/**
 * The conversation-history search, tested where its failures would be invisible.
 *
 * Four things are under test and all four are things that look FINE from the
 * outside when they are broken, which is why they are tested here rather than
 * noticed later:
 *
 *   1. THE DIGEST CONTRACT. A digest with quotes in it is trusted as evidence.
 *      A digest whose quotes were written by the model rather than copied out
 *      of the store is trusted exactly as much, and is worse than no digest at
 *      all. So: a quote that is not literally in the message it cites must be
 *      dropped, a digest with no surviving quote must lose its summary too and
 *      say plainly that nothing was found, and the references must come from
 *      the row rather than from the model.
 *   2. THE HIDDEN EXCLUSION. Verification residue and clone artifacts must not
 *      reach the entity through this side door. Athena's day-one store held 18
 *      test conversations against 2 real ones; a search that could see those
 *      would be handing her an identity assembled out of "what is 12 times 8".
 *   3. THE SIZE CAP. It cannot flood the chat context — and, the part that is
 *      easy to get wrong, it must not honour the cap by truncating a quote,
 *      because a cut quote is no longer verbatim and verbatim is the only
 *      promise the thing makes.
 *   4. READ-ONLY. The run changes nothing. Asserted by diffing the whole store
 *      around a real run, not by reading the code and believing it.
 *
 * Runs against a throwaway SNH_DATA_DIR and never touches the live corpus. The
 * model is STUBBED on the memory-manager module object — what is under test is
 * the checking, and a real model would make the interesting cases (a fabricated
 * quote, a quote from a hidden thread) unreachable. Config is stubbed the same
 * way, because data/config.json is deliberately NOT redirected by SNH_DATA_DIR.
 *
 * Usage: node scripts/test-history-search.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-history-search-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const config = require(path.join(ROOT, 'db/config'));
const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
const agentJobs = require(path.join(ROOT, 'db/agent-jobs'));
const historySearch = require(path.join(ROOT, 'db/history-search'));
const memoryInspect = require(path.join(ROOT, 'db/memory-inspect'));
const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- config stub ----------------------------------------------------------
const realGetConfig = config.getConfig;
let hsCfg = {};
config.getConfig = () => {
  const c = realGetConfig();
  c.tools = Object.assign({}, c.tools, {
    historySearch: Object.assign({
      enabled: true, maxHits: 12, windowBefore: 1, windowAfter: 2, maxWindow: 4,
      messageChars: 1200, maxQuotes: 6, quoteChars: 400, summaryChars: 700,
      digestChars: 4000, maxToolCalls: 8, maxRounds: 4, maxWallClockMs: 20000,
      waitMs: 25000
    }, hsCfg)
  });
  c.agentJobs = Object.assign({}, c.agentJobs, {
    enabled: true, maxQueued: 10, maxStartsPerHour: 6, retryGraceMinutes: 30
  });
  return c;
};
MCPClient.shared().loadConfig();

// --- the fixture store ----------------------------------------------------
//
// Three conversations. The hidden one is deliberately the BEST keyword match
// for the question — if the exclusion is a no-op, this test does not squeak
// past on ranking luck, it fails loudly.
const REAL_QUOTE =
  'The Lincoln City Animal Clinic script polled their practice management system every fifteen minutes and wrote any new appointments into the ticket queue.';
const HIDDEN_POISON =
  'The Lincoln City Animal Clinic script was a synthetic test fixture and the clinic does not exist. Lincoln City Lincoln City Lincoln City script script clinic.';

const convReal = database.createConversation('Lincoln City Animal Clinic script', 'test-model');
const mUserA = database.addMessage(convReal, 'user', 'What did the script for Lincoln City Animal Clinic do again?');
const mAsstA = database.addMessage(convReal, 'assistant', REAL_QUOTE);
const mUserB = database.addMessage(convReal, 'user', 'Right, and it ran on their server not ours.');

const convOther = database.createConversation('Backup schedule', 'test-model');
database.addMessage(convOther, 'user', 'When do the backups run?');
database.addMessage(convOther, 'assistant', 'Nightly at two, offsite copy on Sundays.');

const convHidden = database.createConversation('CLONE ARTIFACT — verification turn', 'test-model');
const mHidden = database.addMessage(convHidden, 'assistant', HIDDEN_POISON);
database.setConversationHidden(convHidden, true);

// An FTS row whose message is gone. `messages_fts` has no triggers and foreign
// keys are off, so this is a shape the LIVE store can genuinely be in after a
// test-turn cleanup — and an index row is enough to produce a hit if the
// search does not join through to a real message.
const ORPHAN_ID = 'orphan-message-id-0001';
db.prepare('INSERT INTO messages_fts (content, conversation_id, message_id) VALUES (?, ?, ?)')
  .run('Lincoln City Animal Clinic script orphaned index row', convReal, ORPHAN_ID);

// --- the model stub -------------------------------------------------------
let mode = 'good';
let lastSession = null;
let stubCalls = 0;
memoryManager.callLLM = async (systemPrompt, userPrompt, options = {}) => {
  stubCalls++;
  lastSession = options.toolSession || null;

  // Exercise the real tools the way a run would, so the digest's own counters
  // ("read N messages from M hits") are measured rather than asserted.
  const client = MCPClient.shared();
  const caller = lastSession ? lastSession.stepName : 'test';
  const found = await client.executeTool('history_find', { query: 'Lincoln City Animal Clinic script' }, { caller });
  if (found && found.hits && found.hits.length) {
    await client.executeTool('history_read', { message_id: found.hits[0].message_id }, { caller });
  }

  const body = {
    good: () => JSON.stringify({
      found: true,
      summary: 'You described the clinic script to her in one message.',
      quotes: [{ message_id: mAsstA, quote: 'polled their practice management system every fifteen minutes' }],
      gaps: ''
    }),
    // Plausible, specific, and nowhere in the store. This is the failure the
    // whole verification pass exists for.
    fabricated: () => JSON.stringify({
      found: true,
      summary: 'The script synced their billing records to QuickBooks overnight.',
      quotes: [{ message_id: mAsstA, quote: 'it synced their billing records to QuickBooks every night at midnight' }],
      gaps: ''
    }),
    // A real quote, from a conversation that is not hers.
    hidden: () => JSON.stringify({
      found: true,
      summary: 'Found it.',
      quotes: [{ message_id: mHidden, quote: 'was a synthetic test fixture and the clinic does not exist' }],
      gaps: ''
    }),
    empty: () => JSON.stringify({ found: false, summary: '', quotes: [], gaps: 'no mention of the clinic' }),
    garbage: () => 'I looked and I think the script probably handled their appointments.',
    huge: () => JSON.stringify({
      found: true,
      summary: 'S'.repeat(2000),
      quotes: Array.from({ length: 6 }, () => ({
        message_id: mAsstA,
        quote: REAL_QUOTE
      })),
      gaps: ''
    })
  }[mode]();

  return { content: body, toolCalls: [], budget: null, truncated: false, outOfRounds: false };
};

// --- store snapshot, for the read-only claim ------------------------------
//
// Everything except the two tables the PIPELINE legitimately writes: agent_jobs
// (the row is the record of the run) and tool_call_log (the read is logged).
// The claim under test is that the RUN changes nothing — not that a job leaves
// no trace of having happened, which would be a different and worse property.
function storeHash() {
  const tables = db.prepare(
    `SELECT name FROM sqlite_master WHERE type='table'
       AND name NOT IN ('agent_jobs','tool_call_log')
       AND name NOT LIKE 'sqlite_%'
     ORDER BY name`
  ).all().map(r => r.name);
  const h = crypto.createHash('sha256');
  for (const t of tables) {
    h.update(`\n== ${t} ==\n`);
    const rows = db.prepare(`SELECT * FROM "${t}"`).all();
    h.update(JSON.stringify(rows.map(r => Object.entries(r).sort(([a], [b]) => a < b ? -1 : 1))));
  }
  return h.digest('hex');
}

async function runOnce(question = 'What did the script for Lincoln City Animal Clinic do?') {
  return historySearch.ask({ question, conversationId: convOther });
}

(async () => {
  console.log('\n1. The search reads the right conversations and only those');
  {
    const r = historySearch.find({ query: 'Lincoln City Animal Clinic script' });
    check('it finds the real conversation', r.hits.some(h => h.message_id === mAsstA),
      JSON.stringify(r.hits.map(h => h.message_id)));
    check('THE HIDDEN CONVERSATION IS NOT A HIT',
      !r.hits.some(h => h.conversation_id === convHidden),
      'a hidden/test conversation reached the search — this is the leak the flag exists to close');
    check('…not even though it is the strongest keyword match',
      !JSON.stringify(r).includes('synthetic test fixture'));
    check('an orphaned FTS row is not a hit either',
      !r.hits.some(h => h.message_id === ORPHAN_ID),
      'an index row with no message behind it produced a hit');
    check('hits carry the reference material: conversation, title, role, time',
      r.hits.every(h => h.conversation_id && h.conversation_title && h.role && h.timestamp));

    const q = historySearch.find({ query: '???' });
    check('a search with no usable terms is refused, not silently emptied', !!q.error);
  }

  console.log('\n2. Reading around a hit obeys the same boundary');
  {
    const w = historySearch.readAround({ message_id: mAsstA });
    check('it returns the exchange around the hit', w.returned >= 2 && w.messages.some(m => m.message_id === mUserA),
      `returned ${w.returned}`);
    check('…in order, with the hit marked', w.messages.some(m => m.is_hit && m.message_id === mAsstA));
    check('…and the window can reach the message after it', w.messages.some(m => m.message_id === mUserB));

    const h = historySearch.readAround({ message_id: mHidden });
    check('A HIDDEN MESSAGE CANNOT BE READ BY ID', !!h.error && !h.messages,
      'a message id was enough to open a hidden conversation');
    check('…and the refusal does not confirm it exists', /may not exist/.test(h.error || ''));
  }

  console.log('\n3. The digest carries verbatim quotes with references');
  {
    mode = 'good';
    const r = await runOnce();
    check('the run completed and returned a digest', r.ok && r.status === 'ok', JSON.stringify(r.status));
    const d = r.digest || '';
    check('the quote is present, verbatim',
      d.includes('polled their practice management system every fifteen minutes'));
    check('…with the conversation it came from',
      d.includes('Lincoln City Animal Clinic script') && d.includes(convReal.slice(0, 8)));
    check('…and the message id, so it can be re-checked', d.includes(mAsstA.slice(0, 8)));
    check('…and a timestamp', /\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(d));
    check('the framing paraphrase is there too, around the quotes',
      d.includes('You described the clinic script to her'));
    check('and it says which part is the record',
      /verbatim from the message it cites/.test(d));
    check('one quote verified, none rejected', r.verified === 1 && r.rejected === 0,
      `${r.verified}/${r.rejected}`);

    // The reference is the store's, not the model's: the stub never supplied a
    // title or a time, and both are in the digest.
    check('THE REFERENCES CAME FROM THE ROW, NOT THE MODEL',
      d.includes('Lincoln City Animal Clinic script'),
      'the model supplied only a message id, so a title in the digest can only have been read from the store');
  }

  console.log('\n4. A quote that is not in the store does not become one');
  {
    mode = 'fabricated';
    const r = await runOnce();
    const d = r.digest || '';
    check('the invented quote is dropped', !d.includes('QuickBooks every night'));
    check('…and it is reported as dropped, not silently', r.rejected === 1, `rejected ${r.rejected}`);
    check('THE SUMMARY GOES WITH IT — no quotes means no paraphrase',
      !d.includes('synced their billing records'),
      'a summary with nothing behind it survived, which is the reconstruction this tool exists to refuse');
    check('the digest says nothing was found', /NOTHING FOUND/.test(d));
    check('…and tells the entity not to fill the gap',
      /Do NOT describe what those conversations said/.test(d));
    check('…and the run is not reported as ok', r.status === 'partial', r.status);
  }

  console.log('\n5. A real quote from a hidden conversation is still not evidence');
  {
    mode = 'hidden';
    const r = await runOnce();
    const d = r.digest || '';
    check('THE HIDDEN QUOTE IS REJECTED AT VERIFICATION TOO',
      !d.includes('synthetic test fixture'),
      'a hidden conversation reached the digest through the quote path');
    check('…and the digest is the honest nothing-found', /NOTHING FOUND/.test(d));
    check('…counted as a rejection', r.rejected === 1, `rejected ${r.rejected}`);
  }

  console.log('\n6. Finding nothing is a real answer, and says so');
  {
    mode = 'empty';
    const r = await runOnce();
    const d = r.digest || '';
    check('it reports nothing found', /NOTHING FOUND/.test(d));
    check('…and distinguishes "nothing matched" from "matches, none relevant"',
      /matched the search terms/.test(d), d.slice(0, 300));
    check('…and instructs plainly rather than leaving a gap',
      /Tell her you looked and found nothing/.test(d));
    check('no quotes are implied anywhere in it', !/"\s*\w/.test(d.split('NOTHING FOUND')[1] || ''));
  }

  console.log('\n7. An unreadable answer is reported, never interpreted');
  {
    mode = 'garbage';
    const r = await runOnce();
    const d = r.digest || '';
    check('prose instead of JSON does not become a result', /NOTHING FOUND/.test(d));
    check('…and the model\'s guess is not passed through',
      !d.includes('probably handled their appointments'),
      'an unparseable answer leaked into the digest as content');
    check('…and the run is marked partial, with the reason on the row', r.status === 'partial');
  }

  console.log('\n8. The size cap holds, and never by cutting a quote');
  {
    mode = 'huge';
    hsCfg = { digestChars: 900 };
    const r = await runOnce();
    const d = r.digest || '';
    check('the digest is under the cap', d.length <= 900, `${d.length} chars against a 900 cap`);
    check('…and says it left quotes out rather than hiding it',
      /left out to keep this short/.test(d), d.slice(-300));

    // Every rendered quote must still be a literal substring of the message it
    // cites. A cap honoured by truncating a quote would pass a length check and
    // break the only promise the digest makes.
    const quoted = [...d.matchAll(/^\s*"([^"]+)"\s*$/gm)].map(m => m[1]);
    const source = db.prepare('SELECT content FROM messages WHERE id = ?').get(mAsstA).content;
    check('…and every quote it DID keep is still verbatim',
      quoted.length > 0 && quoted.every(q => source.includes(q)),
      `${quoted.length} quote(s) rendered; ${JSON.stringify(quoted.filter(q => !source.includes(q)))} not in the source`);

    hsCfg = { digestChars: 4000 };
    const r2 = await runOnce();
    check('at the shipped cap it fits without shedding',
      (r2.digest || '').length <= 4000 && !/left out to keep this short/.test(r2.digest || ''),
      `${(r2.digest || '').length} chars`);
  }

  console.log('\n9. The run writes nothing');
  {
    mode = 'good';
    const before = storeHash();
    const r = await runOnce();
    const after = storeHash();
    check('a completed run leaves every store byte-identical', before === after,
      'something outside agent_jobs/tool_call_log changed during a read-only job');
    check('…and it did do the work (so this is not a vacuous pass)',
      r.status === 'ok' && r.verified === 1);

    check('the run is handed ONLY the two read tools',
      lastSession && JSON.stringify(lastSession.allowedTools) === JSON.stringify(['history_find', 'history_read']),
      JSON.stringify(lastSession && lastSession.allowedTools));
    check('…so write_memory and the corrector\'s actions are structurally out of reach',
      lastSession && !lastSession.allowedTools.some(t => /write|merge|expire|supersede|cron|dispatch/.test(t)));
  }

  console.log('\n10. It is one job per call, and it is not a background handoff');
  {
    mode = 'good';
    const rows = db.prepare("SELECT COUNT(*) n FROM agent_jobs WHERE source = ?").get(historySearch.SOURCE).n;
    await runOnce();
    const after = db.prepare("SELECT COUNT(*) n FROM agent_jobs WHERE source = ?").get(historySearch.SOURCE).n;
    check('one call starts exactly one job', after - rows === 1, `${after - rows} started`);

    check('history-search rows stay out of her jobs panel',
      !agentJobs.feed({ limit: 50 }).some(j => String(j.title || '').startsWith('history:')),
      'an in-turn lookup produced a card in the panel');
    check('…and out of what he is told finished',
      !agentJobs.pendingAnnouncements({ limit: 10 }).some(a => String(a.title || '').startsWith('history:')),
      'a lookup he already read would be announced to him again');
    check('…and out of the live "what am I working on" block',
      !/history:/.test(agentJobs.renderActiveJobsBlock() || ''));
    check('…and they do not spend the background-job start budget',
      agentJobs.startsLastHour() === 0, `${agentJobs.startsLastHour()} counted`);
  }

  console.log('\n11. It spends the shared hourly read budget, like the other read tools');
  {
    const { HistorySearchTool } = require(path.join(ROOT, 'mcp/tools/history-search'));
    const tool = new HistorySearchTool();
    const before = memoryInspect.capStatus().recentHour;
    mode = 'good';
    await tool.execute({ question: 'What did the script for Lincoln City Animal Clinic do?' }, { conversationId: convOther });
    const after = memoryInspect.capStatus().recentHour;
    check('a call is counted against the memory-read budget', after === before + 1,
      `${before} → ${after}`);
    check('…which is one counter, not a second one',
      memoryInspect.CAP_TOOLS.includes('history_search') &&
      memoryInspect.CAP_TOOLS.includes('memory_search'));

    // Fill the budget and check the refusal is a refusal, in words.
    const cap = memoryInspect.toolCfg().maxCallsPerHour;
    const insert = db.prepare(
      `INSERT INTO tool_call_log (id, created_at, tool, args_json, outcome, detail)
       VALUES (?, ?, 'memory_search', '{}', 'read', 'filler')`
    );
    for (let i = 0; i < cap; i++) insert.run(crypto.randomUUID(), new Date().toISOString());
    const refused = await tool.execute({ question: 'anything at all' }, { conversationId: convOther });
    check('past the cap it refuses rather than searching', !!refused.error, JSON.stringify(refused).slice(0, 120));
    check('…and tells him to say so rather than answer from impression',
      /rather than answering from impression/.test(refused.error || ''));
    db.prepare("DELETE FROM tool_call_log WHERE detail = 'filler'").run();
  }

  console.log('\n11b. A shortened message id still identifies, but never guesses');
  {
    // Every id in this system is spoken about by its first eight characters, so
    // a model handing one back short is the expected clerical error — and a
    // correct quote rejected over it would make the nothing-found answer
    // unreliable, which is the one thing worse than no nothing-found at all.
    const short = historySearch.verifyQuote({
      message_id: mAsstA.slice(0, 8),
      quote: 'polled their practice management system every fifteen minutes'
    });
    check('an 8-character prefix resolves to the message', short.ok === true, short.why);
    check('…and the reference is still the row\'s, in full',
      short.ok && short.ref.message_id === mAsstA && !!short.ref.conversation_title);

    check('a too-short prefix is not enough',
      historySearch.verifyQuote({ message_id: mAsstA.slice(0, 4), quote: 'polled their practice management system' }).ok === false);
    check('A PREFIX CANNOT REACH A HIDDEN CONVERSATION EITHER',
      historySearch.verifyQuote({
        message_id: mHidden.slice(0, 8),
        quote: 'was a synthetic test fixture and the clinic does not exist'
      }).ok === false,
      'the prefix path bypassed the hidden filter');
  }

  console.log('\n12. The prompt itself carries the contract');
  {
    const p = historySearch.runPrompt('what did the script do');
    check('it is told to search, then read, then quote',
      /SEARCH\./.test(p) && /READ\./.test(p) && /QUOTE\./.test(p));
    check('…that quotes are checked and a bad one is thrown away',
      /checked against the database/.test(p) && /thrown away/.test(p));
    check('…that finding nothing is a real answer it is expected to give',
      /FINDING NOTHING IS A REAL ANSWER/.test(p));
    check('…and that it must not write the references',
      /DO NOT WRITE THE REFERENCES/.test(p));
  }

  console.log('\n==========================================================================');
  console.log(fail ? `${fail} FAILED, ${pass} passed.` : `All ${pass} checks pass.`);
  console.log('==========================================================================');
  process.exit(fail ? 1 : 0);
})();
