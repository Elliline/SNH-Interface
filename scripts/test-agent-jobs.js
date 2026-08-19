#!/usr/bin/env node
/**
 * The agent-job queue's rules, tested where they can actually be observed.
 *
 * Everything asserted here is a rule whose failure is INVISIBLE in production
 * until it has already cost something: a job that vanishes because the server
 * restarted, a failure that leaves no row, a result announced to him by a block
 * that never shipped, a queue that silently stops starting work — and, above all
 * of them, a job result that finds its way into the initiative table and gets a
 * channel that can open a conversation.
 *
 * Runs against a throwaway SNH_DATA_DIR and never touches the live corpus. The
 * model is STUBBED — memory-manager.callLLM is replaced on the module object,
 * which is the object db/agent-jobs.js resolves at call time. That is
 * deliberate: what is under test is the queue's decision-making, and a real
 * model call would make the failure cases (throws, empty output) unreachable.
 *
 * Config is stubbed the same way, on db/config's module object, because
 * data/config.json is deliberately NOT redirected by SNH_DATA_DIR and a test
 * must never write to the live one.
 *
 * Usage: node scripts/test-agent-jobs.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-agent-jobs-test-'));
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

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

// --- the config stub ------------------------------------------------------
const realGetConfig = config.getConfig;
let jobCfg = {};
config.getConfig = () => {
  const c = realGetConfig();
  c.agentJobs = Object.assign({
    enabled: true, maxConcurrent: 2, maxQueued: 10, maxStartsPerHour: 6,
    maxToolCallsPerJob: 40, maxWallClockMs: 900000, maxRoundsPerJob: 16,
    maxOutputTokens: 2000, retryGraceMinutes: 30, retentionDays: 90
  }, jobCfg);
  return c;
};

// --- the model stub -------------------------------------------------------
let mode = 'ok';
let lastPrompt = null;
let callCount = 0;
let releaseSlow = null;
// Every prompt the stub was handed, so the SALVAGE call can be told apart from
// the run call — they are two calls with two different jobs and the second one
// is the thing under test.
const prompts = [];
memoryManager.callLLM = async (systemPrompt, userPrompt, options) => {
  callCount++;
  lastPrompt = { systemPrompt, userPrompt, options };
  prompts.push({ systemPrompt, userPrompt, hadSession: !!(options && options.toolSession) });
  if (mode === 'throw') throw new Error('Brain circuit open — skipping LLM call (engine wedged)');
  // Empty run, then a salvage call that CAN write: this is the ordinary
  // "budget ran out before it wrote anything" path.
  if (mode === 'empty') {
    if (options && options.toolSession) {
      return { content: '  ', provider: 'stub', toolCalls: [{ name: 'web_search', args: { query: 'cars' }, ok: true, productive: false, note: 'the search returned no results' }], budget: { calls: 4, maxCalls: 40, billed: 1, failedCalls: 4, exhausted: 'attempt ceiling reached (80/80 calls, 80 of them empty or failed)' } };
    }
    return { content: 'Every search came back empty, so I have nothing on the cars themselves. What I did establish first: you have three vehicle facts in memory, all from July.', provider: 'stub', toolCalls: [], budget: {} };
  }
  // Nothing can write — the brain is up for the run and gone for the salvage.
  // This is the floor: the mechanical account has to carry the card alone.
  if (mode === 'empty-and-mute') {
    if (options && options.toolSession) {
      return { content: '', provider: 'stub', toolCalls: [{ name: 'web_search', args: { query: 'x' }, ok: false, productive: false, note: 'the call returned an error' }], budget: { calls: 6, maxCalls: 40, billed: 1.5, failedCalls: 6, exhausted: 'time budget spent (900s of 900s)' } };
    }
    throw new Error('brain unreachable for the salvage call too');
  }
  // A run that wrote something AND was cut short — partial with real text.
  if (mode === 'cut-short') {
    return { content: 'Here is what I got to before I ran out of rounds: two of the three suppliers list a price.', provider: 'stub', toolCalls: [{ name: 'web_search', args: {}, ok: true, productive: true }], budget: { calls: 30, maxCalls: 40, billed: 30, failedCalls: 0, exhausted: null }, outOfRounds: true };
  }
  if (mode === 'hold') {
    // Blocks until the test lets it go — the only way to observe a job while it
    // is genuinely in flight.
    await new Promise(r => { releaseSlow = r; });
    return { content: 'held, then finished', provider: 'stub', toolCalls: [], budget: {} };
  }
  return {
    content: 'Four corrections since Monday: three near-duplicate folds and one supersession.',
    provider: 'stub', toolCalls: [{ name: 'memory_corrections', args: {} }],
    budget: { calls: 1, maxCalls: 12 }
  };
};

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const job = (id) => db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id);

/** Wait for a job to reach a terminal state (or give up loudly). */
async function settle(id, ms = 3000) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    const j = job(id);
    if (j && agentJobs.TERMINAL.includes(j.status)) return j;
    await sleep(20);
  }
  return job(id);
}

(async () => {
  console.log(`\nAgent-job queue tests (throwaway data dir: ${TMP})\n`);

  // =========================================================================
  console.log('── A job starts, the caller does not wait, and it finishes ──');
  mode = 'ok';
  const t0 = Date.now();
  const started = agentJobs.enqueue({ title: 'sweep the ledger', task: 'Look at the corrections ledger since Monday.', why: 'it needs several lookups', conversationId: 'convo-1' });
  const enqueueMs = Date.now() - t0;
  check('enqueue returned ok with an id', started.ok && !!started.id, JSON.stringify(started));
  check('enqueue returned immediately (< 100ms, no model call awaited)', enqueueMs < 100, `${enqueueMs}ms`);
  check('the row exists the instant it returns — "I started it" is checkable', !!job(started.id));

  const done = await settle(started.id);
  check('it reached ok on the pool, outside the caller', done.status === 'ok', done.status);
  check('the result text is stored', /corrections since Monday/.test(done.result_text || ''));
  check('the tool call count is recorded', done.tool_calls === 1, String(done.tool_calls));
  check('it recorded one attempt', done.attempts === 1, String(done.attempts));
  check('the run prompt told it nobody is in the room', /Nobody is in the room/.test(lastPrompt.systemPrompt));
  check('the run prompt told it this is not a message to her',
    /THIS IS NOT A MESSAGE TO HER/.test(lastPrompt.systemPrompt));
  check('the task itself was the prompt', lastPrompt.userPrompt === 'Look at the corrections ledger since Monday.');

  // =========================================================================
  console.log('\n── THE CHANNEL RULE: a finished job writes nothing to the bell ──');
  // The rule the whole design is bent around. A job result is not an initiative
  // and must never get a channel that can open a conversation.
  const initiativeRows = db.prepare('SELECT COUNT(*) n FROM initiatives').get().n;
  check('zero initiative rows exist after a completed job', initiativeRows === 0, String(initiativeRows));
  const src = fs.readFileSync(path.join(ROOT, 'db/agent-jobs.js'), 'utf8');
  check("db/agent-jobs.js does not require db/initiatives at all",
    !/require\(['"]\.\/initiatives['"]\)/.test(src));

  // =========================================================================
  console.log('\n── A job that dies lands in the panel as a failure, with the reason ──');
  mode = 'throw';
  const threw = agentJobs.enqueue({ title: 'ask a wedged brain', task: 'anything' });
  const threwDone = await settle(threw.id);
  check('a throw is recorded as failed', threwDone.status === 'failed', threwDone.status);
  check('the reason is the real one, not a placeholder', /circuit open/i.test(threwDone.error || ''), threwDone.error);
  check('a failed job still has a finish time', !!threwDone.finished_at);
  check('and even a thrown run writes an account rather than leaving an empty card',
    !!(threwDone.result_text || '').trim(), JSON.stringify(threwDone.result_text));
  check('which says what it does not know rather than claiming it looked nothing up',
    /no record of what it managed to look up/.test(threwDone.result_text || ''), threwDone.result_text);

  // =========================================================================
  console.log('\n── AN EMPTY RESULT CARD IS IMPOSSIBLE ──');
  // 2026-08-18: a job spent every tool call on searches that failed, returned no
  // text, and was closed as `failed` with result_text NULL. The panel card was
  // blank — and the memory work it had finished BEFORE it ever reached a search
  // went in the bin with it. The work was done; only the writeup was missing.
  mode = 'empty';
  prompts.length = 0;
  const silent = agentJobs.enqueue({ title: 'a job that says nothing', task: 'What are the cars worth?' });
  const silentDone = await settle(silent.id);
  check('it is PARTIAL — not ok, and not a failure that throws the work away',
    silentDone.status === 'partial', silentDone.status);
  check('and it WROTE something', !!(silentDone.result_text || '').trim(), JSON.stringify(silentDone.result_text));
  check('the text is the salvaged writeup, not a placeholder',
    /nothing on the cars themselves/.test(silentDone.result_text || ''), silentDone.result_text);
  check('the reason it stopped is recorded beside it', /attempt ceiling|empty or failed/.test(silentDone.error || ''), silentDone.error);
  check('the salvage call was made WITHOUT tools — it cannot spend more budget',
    prompts.length === 2 && prompts[0].hadSession === true && prompts[1].hadSession === false,
    JSON.stringify(prompts.map(p => p.hadSession)));
  check('the salvage prompt carried the record of what the run actually did',
    /web_search/.test(prompts[1].systemPrompt) && /no results/.test(prompts[1].systemPrompt));
  check('and told it that failed lookups are not a finding of nothing',
    /do NOT report that as having found nothing/i.test(prompts[1].systemPrompt));

  // The floor: nothing can write, and the card is STILL not empty.
  mode = 'empty-and-mute';
  const mute = agentJobs.enqueue({ title: 'nothing can write this', task: 'Check the supplier prices.' });
  const muteDone = await settle(mute.id);
  check('a job whose salvage call also fails still writes a result',
    !!(muteDone.result_text || '').trim(), JSON.stringify(muteDone.result_text));
  check('the mechanical account says what was asked', /Check the supplier prices/.test(muteDone.result_text || ''));
  check('…what ran', /tool call\(s\)/.test(muteDone.result_text || ''));
  check('…that the lookups failing is not a finding',
    /not a finding that there is nothing to find/.test(muteDone.result_text || ''), muteDone.result_text);
  check('…and where it stopped', /Why it stopped/.test(muteDone.result_text || ''));
  check('it is still partial rather than a silent failure', muteDone.status === 'partial', muteDone.status);

  // The mechanical account is pure — provable with no model anywhere near it.
  const floor = agentJobs.mechanicalAccount(
    { task: 'find the thing' },
    [{ name: 'web_search', productive: false, note: 'the search returned no results' },
     { name: 'web_search', productive: false, note: 'the search returned no results' }],
    { exhausted: 'call budget spent (40.00/40 billed over 43 call(s), 40 of them empty or failed)' },
    { maxRoundsPerJob: 16 }
  );
  check('the floor is never empty, with no model involved at all', floor.trim().length > 80);
  check('and it never claims a finding it does not have',
    /failure of my lookups/.test(floor), floor);

  // A run that DID write but was cut short is partial too — the text is hers to
  // read and the reason sits beside it, not instead of it.
  mode = 'cut-short';
  const cut = agentJobs.enqueue({ title: 'stopped early', task: 'Compare the three suppliers.' });
  const cutDone = await settle(cut.id);
  check('a cut-short run with real text is partial', cutDone.status === 'partial', cutDone.status);
  check('its text is kept whole', /two of the three suppliers/.test(cutDone.result_text || ''));
  check('and it says it ran out of rounds', /ran out of tool rounds/.test(cutDone.error || ''), cutDone.error);

  // What HE is told has to match what is on her card.
  const partialBlock = agentJobs.renderAnnouncementBlock({ limit: 5 });
  check('a partial job is announced with its text, not as "produced no result"',
    /two of the three suppliers|nothing on the cars themselves/.test(partialBlock.text)
    && !/^It did not produce a result/m.test(partialBlock.text), partialBlock.text.slice(0, 300));
  check('and it is marked as having stopped short', /stopped short of finishing/.test(partialBlock.text));

  // =========================================================================
  console.log('\n── Concurrency: past the cap a job waits, and the wait is visible ──');
  mode = 'hold';
  // The trailing-hour start cap is tested on its own below; here it must not be
  // what refuses the second job, or this section would pass for the wrong reason.
  jobCfg = { maxConcurrent: 1, maxStartsPerHour: 1000 };
  const first = agentJobs.enqueue({ title: 'the one in flight', task: 'hold' });
  await sleep(50);
  const second = agentJobs.enqueue({ title: 'the one waiting', task: 'hold too' });
  check('the first is running', job(first.id).status === 'running', job(first.id).status);
  check('the second stays queued rather than being dropped', job(second.id).status === 'queued', job(second.id).status);

  console.log('\n── Cancel is honest: queued yes, running no, and it says why ──');
  const cancelRunning = agentJobs.cancel(first.id);
  check('a running job refuses to be cancelled', !cancelRunning.ok);
  check('and the refusal explains that it cannot be stopped cleanly',
    /cannot be stopped cleanly/.test(cancelRunning.error || ''), cancelRunning.error);
  const cancelQueued = agentJobs.cancel(second.id);
  check('a queued job cancels', cancelQueued.ok);
  check('and lands as cancelled with a reason', job(second.id).status === 'cancelled' && !!job(second.id).error);

  if (releaseSlow) releaseSlow();
  await settle(first.id);
  check('the held job finished normally once released', job(first.id).status === 'ok');
  const cancelFinished = agentJobs.cancel(first.id);
  check('a finished job cannot be cancelled, and says so', !cancelFinished.ok && /already finished/.test(cancelFinished.error));

  // =========================================================================
  console.log('\n── A restart: nothing is lost silently ──');
  jobCfg = { maxConcurrent: 2, retryGraceMinutes: 30, maxStartsPerHour: 1000 };
  mode = 'ok';

  // Three rows left `running` by a process that died, in the three states the
  // sweep has to tell apart.
  const young = randomUUID(), old = randomUUID(), retried = randomUUID();
  const ins = db.prepare(`INSERT INTO agent_jobs (id, title, task, status, started_at, attempts, created_at)
                          VALUES (?, ?, 'x', 'running', ?, ?, ?)`);
  const now = Date.now();
  ins.run(young, 'killed two minutes in', new Date(now - 2 * 60000).toISOString(), 1, new Date(now - 3 * 60000).toISOString());
  ins.run(old, 'killed hours ago', new Date(now - 90 * 60000).toISOString(), 1, new Date(now - 91 * 60000).toISOString());
  ins.run(retried, 'killed twice', new Date(now - 60000).toISOString(), 2, new Date(now - 2 * 60000).toISOString());

  const swept = agentJobs.sweepInterrupted();
  check('every open row was closed out', swept.closed === 3, String(swept.closed));
  check('the young one is re-queued for one more go', job(young).status === 'queued', job(young).status);
  check('the old one is left interrupted rather than redone', job(old).status === 'interrupted', job(old).status);
  check('and it says why, in words', /too old to be worth redoing/.test(job(old).error || ''), job(old).error);
  check('the already-retried one is not retried again', job(retried).status === 'interrupted', job(retried).status);
  check('and it says that it had already been retried', /already been retried/.test(job(retried).error || ''), job(retried).error);
  check('nothing was deleted — all three rows are still there',
    db.prepare('SELECT COUNT(*) n FROM agent_jobs WHERE id IN (?,?,?)').get(young, old, retried).n === 3);

  const resumed = agentJobs.drain();
  check('the re-queued job is launched again by the resume', resumed >= 1, String(resumed));
  await settle(young);
  check('and it completes on its second attempt', job(young).status === 'ok', job(young).status);
  check('with the attempt counted', job(young).attempts === 2, String(job(young).attempts));

  // =========================================================================
  console.log('\n── The caps refuse OUT LOUD, never silently ──');
  jobCfg = { maxStartsPerHour: 1000, maxQueued: 1, maxConcurrent: 1 };
  mode = 'hold';
  const filler = agentJobs.enqueue({ title: 'occupies the queue', task: 'hold' });
  await sleep(30);
  const refusedQueue = agentJobs.enqueue({ title: 'one too many', task: 'x' });
  check('past maxQueued the enqueue is refused', !refusedQueue.ok);
  check('and the refusal is a sentence he can repeat to her',
    /Nothing was started/.test(refusedQueue.error || ''), refusedQueue.error);
  if (releaseSlow) releaseSlow();
  await settle(filler.id);

  jobCfg = { maxStartsPerHour: 1, maxQueued: 10, maxConcurrent: 2 };
  const refusedRate = agentJobs.enqueue({ title: 'too soon', task: 'x' });
  check('past maxStartsPerHour the enqueue is refused', !refusedRate.ok);
  check('and it warns him not to imply it is running',
    /rather than implying it is running/.test(refusedRate.error || ''), refusedRate.error);

  jobCfg = { enabled: false };
  const refusedOff = agentJobs.enqueue({ title: 'switched off', task: 'x' });
  check('a disabled queue refuses rather than pretending', !refusedOff.ok && /switched off/.test(refusedOff.error));
  jobCfg = { maxStartsPerHour: 1000 };

  const noTask = agentJobs.enqueue({ title: 'no task', task: '   ' });
  check('a job with no task is refused with a reason', !noTask.ok && /nothing was started/i.test(noTask.error));

  // =========================================================================
  console.log('\n── What he is told at the start of his next turn ──');
  const pendingBefore = agentJobs.pendingAnnouncements({ limit: 10 });
  check('finished jobs are waiting to be announced', pendingBefore.length > 0, String(pendingBefore.length));

  const block = agentJobs.renderAnnouncementBlock({ limit: 3, tokenCap: 400 });
  check('the block renders', !!block && block.text.includes('=== Background Work That Finished ==='));
  check('it is capped to three items', block.items.length <= 3, String(block.items.length));
  check('it fits the token cap', block.tokens <= 400 + 120, String(block.tokens));
  check('it tells him she has probably NOT read them', /assume she has NOT/.test(block.text));
  check('it tells him he is not obliged to report them', /not\s+obliged to report them/.test(block.text));
  check('a failed job is announced as a failure, with the reason',
    /did not produce a result/.test(block.text) || block.items.every(i => i.status === 'ok'));

  check('rendering alone does NOT stamp them (the block might still be trimmed)',
    agentJobs.pendingAnnouncements({ limit: 10 }).length === pendingBefore.length);

  const stamped = agentJobs.markAnnounced(block.items);
  check('stamping marks exactly the items that shipped', stamped === block.items.length, String(stamped));
  const afterStamp = agentJobs.pendingAnnouncements({ limit: 10 }).map(i => i.id);
  check('a stamped job is never announced twice',
    block.items.every(i => !afterStamp.includes(i.id)));
  check('the ones that did not fit are still waiting, not lost',
    pendingBefore.length > block.items.length ? afterStamp.length > 0 : true);

  // =========================================================================
  console.log('\n── Scheduled runs share the panel and the announcement ──');
  const cronId = randomUUID(), runId = randomUUID();
  db.prepare(`INSERT INTO cron_jobs (id, schedule, description, enabled, source, status, created_at)
              VALUES (?, '0 9 * * *', 'Daily memory digest', 1, 'kid-proposed', 'approved', ?)`)
    .run(cronId, new Date().toISOString());
  db.prepare(`INSERT INTO job_runs (id, job_id, started_at, finished_at, status, duration_ms, trigger, output_text, tool_calls)
              VALUES (?, ?, ?, ?, 'ok', 4200, 'schedule', 'Two facts merged and one event moved.', 2)`)
    .run(runId, cronId, new Date().toISOString(), new Date().toISOString());

  const feed = agentJobs.feed({ limit: 100 });
  check('the scheduled run appears in the jobs panel feed',
    feed.some(f => f.kind === 'scheduled' && f.id === runId));
  check('handed-off jobs appear in the same feed', feed.some(f => f.kind === 'handoff'));
  check('the feed carries the run text without copying it into another table',
    feed.find(f => f.id === runId).result_text === 'Two facts merged and one event moved.');
  check('deferred/skipped runs are not in the feed (they are not results)',
    feed.every(f => !['deferred', 'skipped'].includes(f.status)));

  const schedAnnounce = agentJobs.pendingAnnouncements({ limit: 5 });
  check('the scheduled result is announced to him too',
    schedAnnounce.some(a => a.kind === 'scheduled' && a.id === runId));
  agentJobs.markAnnounced(schedAnnounce);
  check('and stamping it works on the run row',
    !!db.prepare('SELECT announced_at FROM job_runs WHERE id = ?').get(runId).announced_at);

  console.log('\n── The badge counts unread RESULTS, not work in progress ──');
  const beforeCounts = agentJobs.counts();
  check('unread results are counted', beforeCounts.unseen > 0, JSON.stringify(beforeCounts));
  agentJobs.markRunSeen(runId);
  check('marking a scheduled run read reduces the count',
    agentJobs.counts().unseen === beforeCounts.unseen - 1);
  const anyJob = db.prepare("SELECT id FROM agent_jobs WHERE status = 'ok' AND seen_at IS NULL LIMIT 1").get();
  if (anyJob) {
    agentJobs.markSeen(anyJob.id);
    check('marking a handed-off job read reduces it too',
      agentJobs.counts().unseen === beforeCounts.unseen - 2);
  }

  console.log('\n── Retention prunes the tail and never the live work ──');
  const ancient = randomUUID();
  db.prepare(`INSERT INTO agent_jobs (id, title, task, status, created_at, finished_at)
              VALUES (?, 'from last year', 'x', 'ok', ?, ?)`)
    .run(ancient, '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z');
  const queuedSurvivor = randomUUID();
  db.prepare(`INSERT INTO agent_jobs (id, title, task, status, created_at)
              VALUES (?, 'old but never started', 'x', 'queued', ?)`)
    .run(queuedSurvivor, '2025-01-01T00:00:00.000Z');
  agentJobs.prune();
  check('a long-finished job is pruned', !job(ancient));
  check('an old QUEUED job is never pruned — that is a bug to see, not a row to sweep',
    !!job(queuedSurvivor));

  // =========================================================================
  console.log('\n── The worker reaches the SAME search stack the chat path does ──');
  // 2026-08-18: it did not. web_search's execute() was (args, endpointOverride) —
  // a positional STRING — while every other tool takes (args, context), and
  // executeTool only passed the string when the caller supplied `searxngHost`.
  // The chat path does; the background tool loop passes { caller }, so the whole
  // CONTEXT OBJECT arrived where a URL base was expected:
  //
  //   Search failed: Failed to parse URL from [object Object]/search?q=…
  //
  // Seven of those inside one job in 11 seconds, reported to Ellie as "an issue
  // with the search tool" because that is all the model could see.
  //
  // The signature is now uniform, so what this asserts has changed with it: the
  // second argument reaches the tool AS THE CONTEXT OBJECT, unreshaped, whoever
  // called. The provider chain behind it is tested in test-search-providers.js;
  // what matters here is that a worker context is not mangled on the way in.
  const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
  const client = MCPClient.shared();
  const searchTool = client.tools.get('web_search');
  if (!searchTool) {
    check('web_search is registered (a search provider is available)', false, 'not registered — cannot check the worker path');
  } else {
    const realExecute = searchTool.execute.bind(searchTool);
    let sawSecondArg = 'never called';
    searchTool.execute = async (args, context) => { sawSecondArg = context; return { results: [] }; };

    await client.executeTool('web_search', { query: 'x' }, { caller: 'agent-job:test' });
    check('a worker context arrives as an object, with its caller intact',
      sawSecondArg && typeof sawSecondArg === 'object' && sawSecondArg.caller === 'agent-job:test',
      JSON.stringify(sawSecondArg));

    await client.executeTool('web_search', { query: 'x' }, {});
    check('…and a call with no context at all gets an object, not a string',
      sawSecondArg && typeof sawSecondArg === 'object', JSON.stringify(sawSecondArg));

    await client.executeTool('web_search', { query: 'x' }, { caller: 'chat', searxngHost: 'http://example.test:9999' });
    check('a searxngHost override rides along as a named field',
      sawSecondArg.searxngHost === 'http://example.test:9999', JSON.stringify(sawSecondArg));

    check('nothing is passed positionally as a URL any more',
      typeof sawSecondArg !== 'string', typeof sawSecondArg);
    searchTool.execute = realExecute;
  }

  // =========================================================================
  console.log('\n── A job cannot start a job ──');
  check('start_background_job is not among the tools a background step may hold',
    !MCPClient.BACKGROUND_TOOLS.includes('start_background_job'));
  check('and it is not in the job allowlist either',
    !agentJobs.JOB_TOOLS.includes('start_background_job'));
  check('nor is write_memory or any corrector write action',
    !agentJobs.JOB_TOOLS.some(t => ['write_memory', 'memory_merge_facts', 'memory_expire_fact', 'memory_supersede_fact'].includes(t)));

  console.log(`\n${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('\nTest harness crashed:', err);
  process.exit(1);
});
