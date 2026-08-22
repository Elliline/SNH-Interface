#!/usr/bin/env node
/**
 * SHE CAN TALK LIKE A PERSON.
 *
 * The requirement behind this whole path: no phrase, keyword or wording may
 * gate a dispatch. Dispatch ran at 2 real out of 7 claimed, and every previous
 * fix was gated on a phrase list, so a wording she had not used before killed
 * both the pin and the backstop at once.
 *
 * Usage: SNH_DATA_DIR=$(mktemp -d) node scripts/test-approval-trigger.js
 */
const path = require('path');
const assert = require('assert');
const ROOT = path.join(__dirname, '..');

if (!process.env.SNH_DATA_DIR) {
  console.error('refusing to run without SNH_DATA_DIR (this writes rows)');
  process.exit(1);
}

const db = require(path.join(ROOT, 'db/database'));
db.initDatabase();
const sql = db.getSqliteDb();
const approval = require(path.join(ROOT, 'db/approval-classifier'));
const codingJobs = require(path.join(ROOT, 'db/coding-jobs'));
const briefShown = require(path.join(ROOT, 'db/brief-shown'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// A brief long enough and clean enough to be dispatchable.
const BRIEF = [
  'Coding Brief: Squatch Crawler fog of war',
  'Objective: fix the two-tier fog rendering so explored tiles stay dimly visible',
  'while unexplored tiles remain black. Currently everything re-darkens when the',
  'player moves away, which loses the map the player has already earned.',
  'Requirements: keep a persistent explored set; render explored-but-not-visible',
  'at reduced alpha; keep the single-file structure and vanilla JavaScript with no',
  'external libraries; make sure performance does not drop as the maze grows.',
  'Success criteria: walking away from a corridor leaves it dimly drawn, and the',
  'frame rate holds steady at the largest maze size.',
].join(' ');

function seedConversation(cid, replyText, when = '2026-08-22T18:00:00.000Z') {
  sql.prepare('INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)')
    .run(cid, 'test', when);
  sql.prepare('INSERT INTO messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)')
    .run(`${cid}-u`, cid, 'user', 'can you write the brief', when);
  sql.prepare('INSERT INTO messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)')
    .run(`${cid}-a`, cid, 'assistant', replyText, when);
}

// ── The structural half: is there something to approve? ───────────────────
console.log('\n── A pending brief is a STATE, not a phrase ──');

seedConversation('c-pending', BRIEF);
const pending = approval.pendingBrief({ conversationId: 'c-pending' });
check('a brief in the last reply is pending', !!pending);
check('  and it is the brief text itself', pending && pending.text.includes('fog of war'));

seedConversation('c-chatty', 'Sure, I can help with that. What would you like it to do?');
check('a short chatty reply is not a pending brief',
  !approval.pendingBrief({ conversationId: 'c-chatty' }));

seedConversation('c-pathy', BRIEF + ' Also create the directory Projects/squatch_crawler first.');
check('a brief the tool would refuse for naming a path is not pending',
  !approval.pendingBrief({ conversationId: 'c-pathy' }),
  'pinning on this would force a guaranteed refusal');

seedConversation('c-done', BRIEF);
sql.prepare(`INSERT INTO coding_jobs (id, project, brief, status, conversation_id, created_at)
             VALUES ('j1','squatch_crawler',?, 'dispatched', 'c-done', '2026-08-22T18:05:00.000Z')`).run(BRIEF);
check('a brief already dispatched is no longer pending',
  !approval.pendingBrief({ conversationId: 'c-done' }));

// ── The classifier: her real approvals, verbatim ──────────────────────────
console.log('\n── Her three real approvals must all read YES ──');

// A stub standing in for the engine, so this suite needs no GPU. It is given
// the SAME prompt the server builds, and answers the way the real classifier
// is expected to: one word.
function stubEngine(answers) {
  return async (system, user) => {
    const m = /HER MESSAGE:\n"""\n([\s\S]*?)\n"""/.exec(user);
    const msg = (m ? m[1] : '').trim();
    if (!(msg in answers)) throw new Error(`stub has no answer for ${JSON.stringify(msg)}`);
    // callLLM resolves to an OBJECT, not a string. The first version of this
    // stub returned a bare string, so the suite passed while the live path
    // read "[object Object]" and failed closed on every turn. The stub now
    // returns the real shape.
    return { content: answers[msg], reasoning: '', provider: 'stub', truncated: false };
  };
}

const REAL_APPROVALS = [
  'Send away',
  'Go ahead and send it. Thank you.',
  'Please try sending the brief again. Something did not work the last time and it should be fixed.',
];
const REAL_NEGATIVES = [
  'Can you show me the whole brief before you send it',
  'I will read it and let you know if i aprove it',
  'witch ones did you change',
];

(async () => {
  const answers = {};
  for (const a of REAL_APPROVALS) answers[a] = 'YES';
  for (const n of REAL_NEGATIVES) answers[n] = 'NO';
  const callLLM = stubEngine(answers);

  for (const msg of REAL_APPROVALS) {
    const v = await approval.isApproval({ brief: BRIEF, message: msg, callLLM });
    check(`YES: ${JSON.stringify(msg.slice(0, 52))}`, v.approved === true);
  }
  for (const msg of REAL_NEGATIVES) {
    const v = await approval.isApproval({ brief: BRIEF, message: msg, callLLM });
    check(`NO:  ${JSON.stringify(msg.slice(0, 52))}`, v.approved === false);
  }

  console.log('\n── The phrase list does NOT gate anything ──');
  const dc = require(path.join(ROOT, 'db/dispatch-claims'));
  const RESEND = REAL_APPROVALS[2];
  check('the 8/22 resend still misses the phrase list',
    dc.classifyCodingGoAhead(RESEND).goAhead === false,
    'if this now passes, the test below stops proving anything');
  const v = await approval.isApproval({ brief: BRIEF, message: RESEND, callLLM });
  check('  and the classifier approves it anyway', v.approved === true,
    'this is the exact 8/22 failure — a real approval that matched no phrase');

  console.log('\n── It fails CLOSED ──');
  const boom = async () => { throw new Error('brain circuit open'); };
  check('an engine error is NO, not YES',
    (await approval.isApproval({ brief: BRIEF, message: 'send it', callLLM: boom })).approved === false);
  const waffle = async () => 'Well, it depends on what she means by send.';
  check('an unparseable answer is NO',
    (await approval.isApproval({ brief: BRIEF, message: 'send it', callLLM: waffle })).approved === false);
  check('a plain string answer still works',
    (await approval.isApproval({ brief: BRIEF, message: 'x', callLLM: async () => 'YES' })).approved === true);
  check('a bare YES parses', approval.parseVerdict('YES') === true);
  check('"yes." parses', approval.parseVerdict('yes.') === true);
  check('"NO" parses', approval.parseVerdict('NO') === false);
  check('waffle does not parse', approval.parseVerdict('I think probably') === null);

  console.log('\n── Temperature is pinned, and the budget is tiny ──');
  let seen = null;
  await approval.isApproval({
    brief: BRIEF, message: 'send it',
    callLLM: async (_s, _u, opts) => { seen = opts; return 'YES'; },
  });
  check('temperature 0', seen && seen.temperature === 0);
  check('a handful of tokens', seen && seen.maxTokens <= 8, `maxTokens=${seen && seen.maxTokens}`);
  check('no thinking budget', seen && seen.thinkingTokens === 0);

  // ── Refusals leave a record ─────────────────────────────────────────────
  console.log('\n── A refused dispatch is recorded, with the brief ──');
  const before = sql.prepare("SELECT COUNT(*) n FROM tool_call_log WHERE tool='dispatch_coding_job'").get().n;
  const refused = codingJobs.dispatch({
    project: 'squatch_crawler',
    brief: 'Create the folder Projects/whatever and then build a thing in it that does something.',
    conversationId: 'c-pending',
  });
  check('the path-naming brief is refused', refused.ok === false);
  const rows = sql.prepare(
    "SELECT outcome, detail, args_json FROM tool_call_log WHERE tool='dispatch_coding_job' ORDER BY created_at DESC"
  ).all();
  check('  a row was written', rows.length > before, `${before} -> ${rows.length}`);
  const row = rows[0];
  check('  with the refusal reason', /path|director/i.test(row.detail || ''), row.detail);
  const args = JSON.parse(row.args_json || '{}');
  check('  with the brief hash', typeof args.brief_sha256 === 'string' && args.brief_sha256.length === 16);
  check('  with the brief length', args.brief_chars > 0);
  check('  and enough of the text to recognise it', /Create the folder/.test(args.brief_head || ''));

  // ── The resend path ─────────────────────────────────────────────────────
  console.log('\n── A re-pasted brief is not an unseen brief ──');
  const exact = briefShown.check(BRIEF, { conversationId: 'c-pending' });
  check('the original brief matches its own earlier reply', exact.ok === true);
  check('  and is marked exact', exact.exact === true);
  const trimmed = BRIEF.replace('Success criteria: walking away from a corridor leaves it dimly drawn, and the frame rate holds steady at the largest maze size.', '').trim();
  const close = briefShown.check(trimmed, { conversationId: 'c-pending' });
  check('a re-paste missing a trailing line still matches', close.ok === true,
    `ratio ${close.ratio}`);
  const invented = briefShown.check(
    'Build a completely different thing: a REST API for tracking bird sightings with a SQLite backend and three endpoints.',
    { conversationId: 'c-pending' });
  check('but an invented brief is still refused', invented.ok === false);

  // ── The re-run path ─────────────────────────────────────────────────────
  //
  // THE GAP THIS CLOSES: a brief dispatches for real, the job fails, and she
  // says "try that again". Before this the brief was no longer pending so the
  // approval classifier never ran, and the claim-keyed backstop skips
  // dispatched briefs on purpose. The request hit nothing at all — not a
  // refusal, not a correction, silence. The 2026-08-22 resend only worked
  // through the new path because that first dispatch had been FAKE, which left
  // the brief pending.
  console.log('\n── After a dispatch, the brief is still ACTIONABLE ──');

  function seedDispatched(cid, { jobStatus = 'failed', jobError = 'spawn squatch-job ENOENT' } = {}) {
    seedConversation(cid, BRIEF, '2026-08-22T19:00:00.000Z');
    const aj = `${cid}-aj`;
    sql.prepare(`INSERT INTO agent_jobs (id, title, task, status, source, created_at, error)
                 VALUES (?,?,?,?,'squatch-code','2026-08-22T19:01:00.000Z',?)`)
      .run(aj, 'squatch-code: squatch_crawler', BRIEF, jobStatus, jobError);
    sql.prepare(`INSERT INTO coding_jobs (id, project, brief, status, conversation_id, created_at, agent_job_id)
                 VALUES (?,?,?,'dispatched',?, '2026-08-22T19:01:00.000Z', ?)`)
      .run(`${cid}-cj`, 'squatch_crawler', BRIEF, cid, aj);
    return aj;
  }

  seedDispatched('c-ran');
  const after = approval.actionableBrief({ conversationId: 'c-ran' });
  check('a dispatched brief is still actionable', !!after);
  check('  and is marked dispatched', after && after.dispatched === true);
  check('  carrying the project', after && after.project === 'squatch_crawler');
  check('  and how the last run ended', after && after.lastJob && after.lastJob.status === 'failed');
  check('pendingBrief still returns null for it (the backstop skip is unchanged)',
    approval.pendingBrief({ conversationId: 'c-ran' }) === null,
    'loosening this would let a false claim re-run real work');

  console.log('\n── The re-run question is stricter than the approval one ──');
  const rerunAnswers = {
    'Try that again': 'YES',
    'run it one more time': 'YES',
    'Please try sending the brief again. Something did not work the last time and it should be fixed.': 'YES',
    'resend it': 'YES',
    'How did the job go?': 'NO',
    'that result looks wrong': 'NO',
    'the fog still re-darkens when I walk away': 'NO',
    'can you change the scoring and then send it': 'NO',
  };
  const rerunLLM = stubEngine(rerunAnswers);
  for (const [msg, want] of Object.entries(rerunAnswers)) {
    const v = await approval.isRerunRequest({
      brief: BRIEF, message: msg, lastJob: after.lastJob, callLLM: rerunLLM,
    });
    check(`${want === 'YES' ? 'RE-RUN' : 'no    '}: ${JSON.stringify(msg.slice(0, 48))}`,
      v.rerun === (want === 'YES'));
  }

  console.log('\n── The re-run classifier fails closed too ──');
  check('an engine error is NO', (await approval.isRerunRequest({
    brief: BRIEF, message: 'try again', lastJob: null,
    callLLM: async () => { throw new Error('brain circuit open'); } })).rerun === false);
  check('waffle is NO', (await approval.isRerunRequest({
    brief: BRIEF, message: 'try again', lastJob: null,
    callLLM: async () => ({ content: 'probably, if she meant that' }) })).rerun === false);
  let rerunOpts = null;
  await approval.isRerunRequest({ brief: BRIEF, message: 'try again', lastJob: null,
    callLLM: async (_s, _u, o) => { rerunOpts = o; return { content: 'YES' }; } });
  check('temperature 0 and a tiny budget',
    rerunOpts && rerunOpts.temperature === 0 && rerunOpts.maxTokens <= 8);
  check('the prompt states the bias toward NO', /When in doubt, answer NO/.test(approval.RERUN_SYSTEM));
  check('and names complaining as NOT a re-run',
    /Complaining about a result is not a request to repeat it/.test(approval.RERUN_SYSTEM));

  console.log('\n── Same-project concurrency is refused, not queued ──');
  seedDispatched('c-busy', { jobStatus: 'running', jobError: null });
  const busy = codingJobs.activeForProject('squatch_crawler');
  check('an in-flight job for the project is visible', busy.length >= 1);
  check('  and only queued/running count',
    busy.every(j => ['queued', 'running'].includes(j.status)));
  const refusalsBefore = sql.prepare("SELECT COUNT(*) n FROM tool_call_log WHERE outcome='rejected-busy'").get().n;
  codingJobs.logRefusal({
    reason: 're-run refused: 1 job(s) still in flight for squatch_crawler',
    project: 'squatch_crawler', brief: BRIEF, conversationId: 'c-busy', kind: 'rejected-busy',
  });
  const refusalsAfter = sql.prepare("SELECT COUNT(*) n FROM tool_call_log WHERE outcome='rejected-busy'").get().n;
  check('  a busy refusal is recorded like every other refusal', refusalsAfter === refusalsBefore + 1);
  const busyRow = sql.prepare("SELECT detail, args_json FROM tool_call_log WHERE outcome='rejected-busy' ORDER BY created_at DESC LIMIT 1").get();
  check('  with the reason and the brief', /still in flight/.test(busyRow.detail)
    && JSON.parse(busyRow.args_json).brief_sha256.length === 16);

  console.log('\n── A re-run makes NEW rows, never reuses old ones ──');
  // Reusing a row would destroy the first run's restore-point reference, which
  // is the only thing that makes `reversible: true` true.
  const shaOf = t => require('crypto').createHash('sha256').update(t).digest('hex').slice(0, 16);
  // Only real dispatches: an earlier case in this suite hand-seeds a row with
  // no agent_job_id to test "already dispatched", and that is not a run.
  const jobRows = sql.prepare(
    "SELECT c.id, c.agent_job_id, c.brief FROM coding_jobs c "
    + "WHERE c.project = 'squatch_crawler' AND c.agent_job_id IS NOT NULL"
  ).all();
  check('two coding_jobs rows exist for the same brief',
    jobRows.length >= 2 && new Set(jobRows.map(r => shaOf(r.brief))).size === 1,
    `${jobRows.length} rows, ${new Set(jobRows.map(r => shaOf(r.brief))).size} distinct briefs`);
  check('  with distinct ids', new Set(jobRows.map(r => r.id)).size === jobRows.length);
  check('  each joined to its own agent_jobs row',
    new Set(jobRows.map(r => r.agent_job_id)).size === jobRows.length
      && jobRows.every(r => sql.prepare('SELECT 1 FROM agent_jobs WHERE id = ?').get(r.agent_job_id)));

  // ── The wiring, asserted against the source ─────────────────────────────
  console.log('\n── The pin is set BEFORE generation, and by the classifier ──');
  const fs = require('fs');
  const src = fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8');

  const decideAt = src.indexOf('approvalClassifier.isApproval');
  // The provider log lines are template literals, so match the loop itself.
  const firstRoundAt = src.indexOf('for (let round = 0; round < MAX_TOOL_ROUNDS; round++)');
  check('the classifier runs before the tool loop',
    decideAt > 0 && firstRoundAt > 0 && decideAt < firstRoundAt,
    `classifier at ${decideAt}, first tool round at ${firstRoundAt} — a pin decided after generation is not a pin`);
  check('forceCodingCall is set from the classifier verdict',
    /forceCodingCall = verdict\.approved/.test(src),
    'something other than the classifier is deciding the pin');
  check('the phrase list does not gate the pin',
    !/forceCodingCall\s*=\s*codingGoAhead\.goAhead/.test(src),
    'the phrase list is gating again — this is the 8/22 defect');
  check('and it is still consulted, for the free early hit',
    /classifyCodingGoAhead\(userMessage\.content\)/.test(src));
  check('the pin names dispatch_coding_job',
    /forceCodingCall \? 'dispatch_coding_job'/.test(src));

  console.log('\n── The backstop keys off HIS CLAIM, not her wording ──');
  const backstopAt = src.indexOf('=== THE CLAIM-KEYED BACKSTOP ===');
  check('the backstop exists', backstopAt > 0);
  const branch = src.slice(src.lastIndexOf('} else if', backstopAt), backstopAt);
  check('  its condition tests the claim', /claimsDispatch/.test(branch), branch.trim());
  check('  and NOT forceCodingCall', !/forceCodingCall/.test(branch),
    'the shared signal is back — a phrase miss will kill the backstop again');
  check('  it re-runs a pinned round', /forcedDispatchRound\(/.test(src));
  check('  the retry pins the tool',
    /tool_choice: \{ type: 'function', function: \{ name: 'dispatch_coding_job' \} \}/.test(src));

  console.log('\n── Corrections never enter the transcript ──');
  check('notice() does not touch fullResponse',
    /const notice = \(kind, text\) => \{\s*const frame[\s\S]{0,320}?\};/.test(src)
      && !/const notice = \(kind, text\) => \{[\s\S]{0,320}?fullResponse \+=/.test(src),
    'a notice that appends is just an append with extra steps');
  check('there is no append() left on the chat path',
    !/^\s*append\(/m.test(src),
    'a server-authored correction is being written into his message again');
  check('every correction goes out as a notice', (src.match(/notice\('/g) || []).length >= 6);
  check('the client renders the frame as chrome',
    /parsed\.snh_notice/.test(fs.readFileSync(path.join(ROOT, 'public/script.js'), 'utf8')));
  check('and the frame carries no `content`, so the content path ignores it',
    !/snh_notice[\s\S]{0,120}choices/.test(src));

  console.log(`\n=== ${pass} passed, ${fail} failed ===`);
  process.exit(fail ? 1 : 0);
})();
