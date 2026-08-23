#!/usr/bin/env node
/**
 * A CODING JOB IS KILLED FOR STALLING, NOT FOR TAKING TIME.
 *
 * On 2026-08-22 a flat 20-minute wall clock killed a job that was working: the
 * engine logged 33.7 tok/s continuously for the whole twenty minutes, 40,439
 * output tokens, zero samples at 0 tok/s, one request in flight. It was
 * rewriting a 20KB file each iteration — 160-170s per iteration, ~71 minutes
 * for the 25 it was allowed. The ceiling was doing the pacing. It died
 * mid-refactor with the phase/hunter logic torn out and its replacement
 * unwritten, and the card told her to "check git status" without telling her
 * the sha to go back to.
 *
 * Usage: SNH_DATA_DIR=$(mktemp -d) node scripts/test-coding-job-limits.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

if (!process.env.SNH_DATA_DIR) {
  console.error('refusing to run without SNH_DATA_DIR');
  process.exit(1);
}
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
db.initDatabase();

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'cjlimits-'));
const PROJECTS = path.join(TMP, 'Projects');
fs.mkdirSync(path.join(PROJECTS, 'demo'), { recursive: true });

// A real git project with a real restore point, so the restore command in the
// card is checked against a sha that actually exists.
const git = (...a) => execFileSync('git', ['-C', path.join(PROJECTS, 'demo'), ...a], { encoding: 'utf8' });
git('init', '-q');
git('config', 'user.email', 't@t');
git('config', 'user.name', 'T');
fs.writeFileSync(path.join(PROJECTS, 'demo', 'game.html'), '<html>v1</html>\n');
git('add', '-A');
git('commit', '-q', '-m', 'Restore point before dispatched job');
const BASELINE = git('rev-parse', 'HEAD').trim();

/**
 * A stand-in for squatch-job.
 *
 * IT WRITES THE REAL PROGRESS FILE, in the real shape and the real place — the
 * run derives progress.json from the --report-json directory, and the stall
 * watcher reads `updated_at` out of it. A stub that skipped that would be
 * testing a timer against nothing. (Two stub bugs this week came from a stub
 * not matching the real dependency; this one writes what squatch-code writes.)
 *
 *   MODE=steps   — completes a step every STEP_MS, forever: slow but working
 *   MODE=stall   — one step, then silence: stuck
 *   MODE=dirty   — edits the file, then goes silent: stuck mid-edit
 */
const STUB = path.join(TMP, 'fake-squatch-job');
fs.writeFileSync(STUB, `#!/usr/bin/env node
const fs = require('fs'), path = require('path');
const argv = process.argv.slice(2);
const reportPath = argv[argv.indexOf('--report-json') + 1];
const progress = path.join(path.dirname(reportPath), 'progress.json');
const project = argv[argv.indexOf('--project') + 1];
const root = argv[argv.indexOf('--projects-root') + 1];
const mode = process.env.MODE || 'steps';
const stepMs = Number(process.env.STEP_MS || 200);
let n = 0;
const started = Date.now() / 1000;
function write() {
  n++;
  fs.writeFileSync(progress, JSON.stringify({
    schema: 1, project, iteration: n, max_iterations: 25, tool_calls: n,
    last_action: 'write_file game.html', started_at: started,
    elapsed_seconds: Math.round(Date.now() / 1000 - started),
    updated_at: Date.now() / 1000, finished: false,
  }));
}
write();
if (mode === 'dirty') {
  fs.writeFileSync(path.join(root, project, 'game.html'), '<html>half-edited');
}
if (mode === 'steps') setInterval(write, stepMs);
process.stdin.resume();
setTimeout(() => {}, 1 << 30);
`);
fs.chmodSync(STUB, 0o755);

const codingJobs = require(path.join(ROOT, 'db/coding-jobs'));
const { getConfig } = require(path.join(ROOT, 'db/config'));

// NEVER updateConfig() HERE. data/config.json is not redirected by
// SNH_DATA_DIR, so writing config in a test writes the REAL file — during this
// suite's own development it pointed the live projects root at a temp dir and
// the live binary at a stub. runDispatched takes a config instead.
function configure(extra = {}) {
  return { enabled: true, projectsRoot: PROJECTS, binary: STUB, ...extra };
}

function runWith(mode, { stallTimeoutMs, maxRuntimeMinutes, stepMs = 200, extra = {} }) {
  const config = configure({ stallTimeoutMs, maxRuntimeMinutes, ...extra });
  process.env.MODE = mode;
  process.env.STEP_MS = String(stepMs);
  const t0 = Date.now();
  return codingJobs.runDispatched(
    { id: `job-${mode}-${t0}`, task: 'do the thing', title: 'squatch-code: demo' },
    { config })
    .then(r => ({ ...r, tookMs: Date.now() - t0 }));
}

(async () => {
  console.log('\n── A stall is killed, and the card says so ──');
  // 1.5s of silence allowed; the stub goes quiet immediately.
  const stalled = await runWith('stall', { stallTimeoutMs: 1500, maxRuntimeMinutes: 60 });
  check('a stalled job is killed', stalled.status === 'partial', `status ${stalled.status}`);
  check('  quickly, not at the ceiling', stalled.tookMs < 10000, `${stalled.tookMs}ms`);
  check('  the card says NO ACTIVITY', /no activity for/i.test(stalled.resultText), stalled.resultText.slice(0, 90));
  check('  and not "exceeded maximum runtime"', !/exceeded maximum runtime/i.test(stalled.resultText));
  check('  the error names the stall and the limit',
    /stalled — no progress for \d+s \(limit \d+s\)/.test(stalled.error), stalled.error);

  console.log('\n── A ceiling kill is a DIFFERENT card ──');
  // Steps every 200ms so nothing stalls; ceiling is what binds.
  const ceilinged = await runWith('steps', { stallTimeoutMs: 600000, maxRuntimeMinutes: 0.05, stepMs: 200 });
  check('a runaway is killed', ceilinged.status === 'partial');
  check('  the card says EXCEEDED MAXIMUM RUNTIME',
    /exceeded maximum runtime/i.test(ceilinged.resultText), ceilinged.resultText.slice(0, 90));
  check('  and NOT "no activity"', !/no activity for/i.test(ceilinged.resultText));
  check('  it says the job was still working', /still working/i.test(ceilinged.resultText));
  check('  the error names the ceiling',
    /exceeded maximum runtime of/.test(ceilinged.error), ceilinged.error);

  console.log('\n── Slow but flowing SURVIVES past the old flat limit ──');
  // The regression that matters: steps 600ms apart, a stall window of 1.5s, and
  // a run that outlives what a flat clock scaled to this test would have been.
  const survived = await runWith('steps', { stallTimeoutMs: 1500, maxRuntimeMinutes: 0.08, stepMs: 600 });
  check('it was not killed for stalling',
    !/no activity for/i.test(survived.resultText), survived.resultText.slice(0, 80));
  check('  it ran past the point a flat clock would have cut it',
    survived.tookMs > 3000, `${survived.tookMs}ms — steps 600ms apart, stall window 1.5s`);
  check('  and ended on the ceiling instead', /exceeded maximum runtime/i.test(survived.resultText));

  console.log('\n── Killed mid-edit: the card hands her the way back ──');
  const dirty = await runWith('dirty', { stallTimeoutMs: 1500, maxRuntimeMinutes: 60 });
  check('the tree really is dirty', git('status', '--porcelain').trim().length > 0);
  check('the card says the project is modified',
    /modified, possibly broken state/i.test(dirty.resultText), dirty.resultText.slice(-260));
  check('  it gives a reset command', /git -C .* reset --hard/.test(dirty.resultText));
  check('  with the REAL baseline sha', dirty.resultText.includes(BASELINE),
    `expected ${BASELINE.slice(0, 12)} in the card`);
  check('  and the clean step, or the untracked files stay', /clean -fd/.test(dirty.resultText));
  check('  it does NOT just say "check git status"',
    !/check the project's git status/i.test(dirty.resultText),
    'that was the old card, and it left her to find the sha herself');

  // Undo it, and confirm the command in the card is the one that works.
  const cmd = /git -C (\S+) reset --hard ([0-9a-f]{7,40})/.exec(dirty.resultText);
  execFileSync('git', ['-C', cmd[1], 'reset', '--hard', cmd[2]], { encoding: 'utf8' });
  execFileSync('git', ['-C', cmd[1], 'clean', '-fd'], { encoding: 'utf8' });
  check('  and running it restores the file',
    fs.readFileSync(path.join(PROJECTS, 'demo', 'game.html'), 'utf8') === '<html>v1</html>\n');

  console.log('\n── A clean kill says there is nothing to undo ──');
  const clean = await runWith('stall', { stallTimeoutMs: 1500, maxRuntimeMinutes: 60 });
  check('no false alarm when nothing was written',
    /nothing to undo/i.test(clean.resultText) && !/reset --hard/.test(clean.resultText),
    clean.resultText.slice(-140));

  console.log('\n── The retired key warns once ──');
  const warned = [];
  const realWarn = console.warn;
  // Warn-ONCE per process, and earlier cases in this suite already ran the
  // resolver — without clearing the latch this asserts on a warning that fired
  // before the capture started.
  codingJobs._resetRetiredWarning();
  console.warn = (...a) => { warned.push(a.join(' ')); };
  codingJobs.runLimits(configure({ timeoutMinutes: 20 }));
  const secondCall = warned.length;
  codingJobs.runLimits(configure({ timeoutMinutes: 20 }));
  console.warn = realWarn;
  check('setting timeoutMinutes produces a warning',
    warned.some(w => /timeoutMinutes.*NO LONGER READ/i.test(w)),
    warned.join(' | ').slice(0, 120) || '(no warning)');
  check('  and it names the replacements',
    warned.some(w => /stallTimeoutMs/.test(w) && /maxRuntimeMinutes/.test(w)));
  check('  and it warns ONCE, not on every job',
    warned.length === secondCall, `${warned.length} warnings from two calls`);

  console.log('\n── The defaults are the ones the evidence supports ──');
  const d = getConfig().tools.codingJobs;
  check('the live config is untouched by this suite',
    d.projectsRoot === require('path').join(require('os').homedir(), 'Projects')
      || !String(d.projectsRoot).startsWith('/tmp/cjlimits-'),
    `live projectsRoot is ${d.projectsRoot}`);
  // DEFAULTS is not exported, so read what the file ships — same technique as
  // test-run-limits-exposed, and it checks the shipped value rather than
  // whatever this test just wrote into config.
  const cfgSrc = fs.readFileSync(path.join(ROOT, 'db/config.js'), 'utf8');
  const shippedStall = /stallTimeoutMs:\s*(\d+)/.exec(cfgSrc.slice(cfgSrc.indexOf('codingJobs: {')));
  const shippedCeil = /maxRuntimeMinutes:\s*(\d+)/.exec(cfgSrc);
  check('shipped stall window clears a measured 170s step by a wide margin',
    shippedStall && Number(shippedStall[1]) >= 300000,
    `${shippedStall && shippedStall[1]}ms — a 120s window would have killed the 2026-08-22 job`);
  check('shipped ceiling is 60 minutes, not 20',
    shippedCeil && Number(shippedCeil[1]) === 60, shippedCeil && shippedCeil[1]);
  check('and timeoutMinutes is no longer shipped as a default',
    !/^\s*timeoutMinutes:\s*\d+,/m.test(cfgSrc),
    'we would be warning about a key we set ourselves');

  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
  console.log(`\n=== ${pass} passed, ${fail} failed ===`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
