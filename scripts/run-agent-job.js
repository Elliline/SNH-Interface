#!/usr/bin/env node
/**
 * RUN ONE BACKGROUND JOB IN A THROWAWAY STORE, AGAINST THE REAL ENGINE, AND
 * REPORT HOW IT LIVED IN ITS WINDOW.
 *
 * The verification for 2026-09-17: re-run the shape that died on Juno (six
 * research questions, forty tool calls) and see it complete — with the tool
 * call count, the peak prompt size, how many rounds were compacted and how
 * many checkpoints were written, all from the row the runner wrote.
 *
 * It requires SNH_DATA_DIR: the job's row, checkpoint, findings document and
 * result file all land in the throwaway, never in the live store or her
 * documents folder (db/job-artifacts.outputDir puts them inside the data dir
 * when the redirect is set). data/config.json is NOT redirected, so the real
 * engine, the real search provider and the real budgets are used — and any
 * override is applied IN PROCESS, never written:
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/run-agent-job.js --task-file brief.txt \
 *       --title "Qwen3.8-Flash-Next substrate research" --window 131072
 *
 *   --window N        pin agentJobs.context.windowTokens (rehearse a smaller engine)
 *   --set a.b.c=VALUE any other config override (JSON value, or a bare string)
 *   --no-phases / --no-compaction  turn a part off, to compare
 *   --keep            leave the throwaway directory in place for reading
 *
 * Nothing here touches a live instance. The one thing it shares with one is
 * the engine, and that is the point of the exercise.
 */
const fs = require('fs');
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory. Use: SNH_DATA_DIR=$(mktemp -d) node scripts/run-agent-job.js …');
  process.exit(1);
}
const ROOT = path.join(__dirname, '..');
const argv = process.argv.slice(2);
const opt = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : null; };
const flag = (name) => argv.includes(name);

const taskFile = opt('--task-file');
const task = taskFile ? fs.readFileSync(taskFile, 'utf8').trim() : (opt('--task') || '');
if (!task) { console.error('Give the brief with --task-file <file> or --task "<text>".'); process.exit(1); }
const title = opt('--title') || task.slice(0, 60);

// In-process overrides, the same seam the suites use.
const sets = [];
for (let i = 0; i < argv.length; i++) if (argv[i] === '--set' && argv[i + 1]) sets.push(argv[++i]);
const setDeep = (obj, dotted, value) => { const ks = dotted.split('.'); let o = obj; for (const k of ks.slice(0, -1)) { o[k] = o[k] && typeof o[k] === 'object' ? o[k] : {}; o = o[k]; } o[ks[ks.length - 1]] = value; };
const parseVal = (v) => { try { return JSON.parse(v); } catch { return v; } };

const config = require(path.join(ROOT, 'db/config'));
const realGetConfig = config.getConfig;
config.getConfig = () => {
  const c = realGetConfig();
  const win = opt('--window');
  if (win) setDeep(c, 'agentJobs.context.windowTokens', Number(win));
  if (flag('--no-phases')) setDeep(c, 'agentJobs.phases.enabled', false);
  if (flag('--no-compaction')) setDeep(c, 'agentJobs.compaction.enabled', false);
  // No budget ask: nobody is in a conversation to answer it. The hard stop
  // (writeup, partial) applies instead, exactly as with asking switched off.
  setDeep(c, 'agentJobs.askBeforeCeiling', false);
  // The brain watchdog is the live instance's; a throwaway must never restart
  // the shared engine (it is disabled under SNH_DATA_DIR anyway — belt and braces).
  setDeep(c, 'watchdog.enabled', false);
  for (const s of sets) { const eq = s.indexOf('='); if (eq > 0) setDeep(c, s.slice(0, eq), parseVal(s.slice(eq + 1))); }
  return c;
};

const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();
const agentJobs = require(path.join(ROOT, 'db/agent-jobs'));
const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
MCPClient.shared();

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const fmt = (n) => Number.isFinite(n) ? n.toLocaleString('en-US') : '?';

(async () => {
  const c = config.getConfig();
  const win = await require(path.join(ROOT, 'db/job-window')).engineWindow();
  console.log(`store: ${process.env.SNH_DATA_DIR}`);
  console.log(`engine: ${c.models.heartbeat.provider}/${c.models.heartbeat.model}; window ${fmt(win.window)} (${win.source})`);
  console.log(`budgets: ${c.agentJobs.maxToolCallsPerJob} calls, ${c.agentJobs.maxRoundsPerJob} rounds/sitting, ${Math.round(c.agentJobs.maxWallClockMs / 60000)} min; answer ${c.generation.agentJobResponseTokens}, thinking ${c.generation.agentJobThinkingTokens}`);
  console.log(`phases ${c.agentJobs.phases.enabled ? 'on' : 'off'}, compaction ${c.agentJobs.compaction.enabled ? 'on' : 'off'} (digest ${c.agentJobs.compaction.digestFetches ? 'on' : 'off'}, recent ${c.agentJobs.compaction.recentRounds})`);
  console.log(`tools: ${MCPClient.shared().backgroundToolsAmong(agentJobs.JOB_TOOLS || ['memory_search', 'memory_list', 'memory_count', 'memory_get', 'memory_corrections', 'memory_jobs', 'web_search', 'web_fetch']).join(', ')}\n`);

  const started = agentJobs.enqueue({ title, task, conversationId: null, source: 'chat-handoff' });
  if (!started || !started.id) { console.error('enqueue refused:', JSON.stringify(started)); process.exit(1); }
  const id = started.id;
  console.log(`job ${id} queued at ${new Date().toISOString()}\n`);
  const t0 = Date.now();
  let last = '';
  for (;;) {
    const j = db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id);
    if (j && agentJobs.TERMINAL.includes(j.status)) break;
    const ck = agentJobs.readCheckpoint(id);
    const line = ck ? `${Math.round((Date.now() - t0) / 1000)}s — phase ${ck.phase || 1}${ck.phases ? `/${ck.phases.length}` : ''}, ${(ck.toolCalls || []).length} call(s), peak prompt ${fmt(ck.stats && ck.stats.peakPromptTokens)}` : `${Math.round((Date.now() - t0) / 1000)}s — starting`;
    if (line !== last) { console.log(line); last = line; }
    await sleep(5000);
  }
  // THE FILE COMES AFTER THE STATUS. attachArtifact runs after finish() and
  // cannot fail the job, so the row is terminal before the file exists — an
  // exit on the status alone kills the artifact step under it. Wait for it.
  const until = Date.now() + 120000;
  while (Date.now() < until) {
    const r = db.prepare('SELECT artifact_kind, artifact_error, source FROM agent_jobs WHERE id = ?').get(id);
    if (r && (r.artifact_kind || r.artifact_error)) break;
    await sleep(500);
  }
  const j = db.prepare('SELECT * FROM agent_jobs WHERE id = ?').get(id);
  const p = (() => { try { return JSON.parse(j.progress_json); } catch { return null; } })();
  console.log('\n=== RESULT ===');
  console.log(`status: ${j.status}${j.error ? ` — ${j.error}` : ''}`);
  console.log(`duration: ${Math.round((j.duration_ms || 0) / 1000)}s; tool calls: ${j.tool_calls}`);
  if (p) {
    console.log(`window: ${fmt(p.window && p.window.tokens)} (${p.window && p.window.source}); reservation ${fmt(p.reservation)}`);
    console.log(`peak prompt: ${fmt(p.peakPromptTokens)} tokens; peak completion: ${fmt(p.peakCompletionTokens)}; squeezed turns: ${p.squeezedRounds}; refit retries: ${p.refitRetries}`);
    const cp = p.compaction;
    console.log(`compaction: ${cp && cp.entries ? `${cp.roundsCompacted} round(s), ${cp.entries} result(s), ${fmt(cp.charsBefore)} → ${fmt(cp.charsAfter)} chars (${fmt(cp.droppedChars)} dropped); ${cp.digested} digested, ${cp.trimmed} trimmed, ${cp.truncated} truncated${cp.digestFailed ? `, ${cp.digestFailed} digest failure(s)` : ''}; ${cp.passes} pass(es), ${cp.pressurePasses} under pressure` : 'none'}`);
    console.log(`phases: ${p.multi ? `${p.phases.length} (${p.plan}); checkpoints written: ${p.checkpoints}` : 'one (no document)'}`);
    for (const ph of p.phases || []) console.log(`  ${ph.n}. [${ph.status}] ${ph.goal} — ${ph.calls} call(s), ${ph.rounds} round(s), ${ph.sources} source(s)${ph.parts ? `, ${ph.parts + 1} sittings` : ''}${ph.stop ? ` — ${ph.stop}` : ''}`);
  }
  console.log(`findings document: ${j.findings_path || '(none)'}`);
  console.log(`result file: ${j.artifact_path || '(none)'}${j.artifact_error ? ` — ${j.artifact_error}` : ''}`);
  console.log(`result text: ${(j.result_text || '').length} chars`);
  const budget = (() => { try { return JSON.parse(j.budget_json); } catch { return null; } })();
  if (budget) console.log(`budget: ${budget.calls} calls (${budget.billed} billed, ${budget.failedCalls} failed), ${budget.rounds ?? budget.roundsUsed} rounds, ${Math.round((budget.elapsedMs || 0) / 1000)}s`);
  console.log(`\nresult head:\n${(j.result_text || '').slice(0, 1500)}\n…`);
  if (!flag('--keep')) console.log(`\n(the throwaway store is left for you to remove: rm -rf ${process.env.SNH_DATA_DIR})`);
  process.exit(0);
})().catch(err => { console.error('run-job crashed:', err); process.exit(1); });
