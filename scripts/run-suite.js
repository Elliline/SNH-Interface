#!/usr/bin/env node
/**
 * THE KNOWN-FAILING BASELINE IS A COMPUTED CLAIM, NOT A REMEMBERED ONE.
 *
 * "That suite was already failing" was hand-maintained prose in session notes,
 * and it was wrong twice in one week:
 *
 *   - test-brain-watchdog was recorded as a pre-existing CRASH. It was not
 *     pre-existing; the crash was caused by the invocation (SNH_DATA_DIR set,
 *     which disables the watchdog under test). A remembered baseline cannot
 *     tell "this suite fails" from "I ran it wrong".
 *   - test-history-search was recorded GREEN across the timezone layer. The
 *     assertion it was green on only held between 10am and noon. A remembered
 *     baseline cannot tell "verified" from "verified at a lucky hour".
 *
 * So the baseline moved into scripts/test-baseline.json, and this runner diffs
 * against it. Three rules, each one a case the prose version got wrong:
 *
 *   1. AN EXIT CODE OUTSIDE {0,1} IS NEVER "MATCHES BASELINE". A suite that
 *      crashes has not reported a failure count — it reported nothing. Its
 *      output is UNKNOWN no matter what the baseline says. This is the
 *      watchdog case: a TypeError read as "3 expected failures".
 *
 *      AND THE EXIT CODE ALONE IS NOT ENOUGH, which this runner found the hard
 *      way: node exits 1 on an uncaught exception, so a crashed suite is
 *      indistinguishable BY CODE from one that failed three assertions.
 *      test-dispatch-guards had been dying on a TypeError for weeks and never
 *      appeared on the known-failing list for exactly that reason. What
 *      actually catches it is the tally: a suite that crashed never printed
 *      one, and no tally is UNKNOWN. Both halves are load-bearing.
 *   2. AN ENTRY OLDER THAN THE CODE IT DESCRIBES IS STALE. If the last commit
 *      touching the suite file or any of its subject modules is newer than the
 *      entry's verifiedAt, the entry describes a tree that no longer exists.
 *      STALE is reported as UNKNOWN, never as expected. This is the
 *      history-search case: the layer shipped, the note did not move.
 *   3. UNKNOWN IS LOUD. It is a non-zero exit, its own block in the summary,
 *      and it is never folded into the "as expected" count.
 *
 * Usage:
 *   node scripts/run-suite.js                 # every suite in the baseline
 *   node scripts/run-suite.js test-cron-eval  # named suites only
 *   node scripts/run-suite.js --record        # print observed counts as JSON,
 *                                             # to curate into the baseline
 *   node scripts/run-suite.js --include-flaky      # opt in to LLM-dependent suites
 *   node scripts/run-suite.js --include-instance   # opt in to suites needing a server
 *   node scripts/run-suite.js --include-live-corpus # opt in to live-corpus reads
 *
 * Exit: 0 only when every suite ran and every one matched its entry.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const BASELINE_PATH = path.join(__dirname, 'test-baseline.json');

// ---- Result kinds. UNKNOWN is a kind, not a flag on another kind. -----------
const OK = 'MATCHES BASELINE';
const REGRESSION = 'REGRESSION';
const IMPROVED = 'IMPROVED';
const UNKNOWN = 'UNKNOWN';
const SKIPPED = 'SKIPPED';

const argv = process.argv.slice(2);
const flags = new Set(argv.filter(a => a.startsWith('--')));
const named = argv.filter(a => !a.startsWith('--')).map(s => path.basename(s, '.js'));
const RECORD = flags.has('--record');
const INCLUDE_FLAKY = flags.has('--include-flaky');
const INCLUDE_INSTANCE = flags.has('--include-instance');
const INCLUDE_LIVE_CORPUS = flags.has('--include-live-corpus');

const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8'));

function git(args) {
  try {
    return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
  } catch { return ''; }
}

/**
 * RULE 2. The entry claims a commit it was verified at. Find the newest commit
 * that touched the suite file or any subject module; if the entry was verified
 * before it, the entry is describing code that has since changed.
 *
 * Fails CLOSED: an unresolvable commit, a path git does not know, or no git at
 * all all report stale. An entry we cannot check is exactly as unproven as one
 * we checked and found old — and this whole file exists because "probably still
 * fine" was the wrong default twice.
 */
function stalenessOf(entry, suiteFile) {
  const verifiedAt = entry.verifiedAt;
  if (!verifiedAt) return { stale: true, why: 'entry has no verifiedAt commit' };
  const full = git(['rev-parse', '--verify', `${verifiedAt}^{commit}`]);
  if (!full) return { stale: true, why: `verifiedAt ${verifiedAt} is not a commit in this repo` };

  const paths = [path.relative(ROOT, suiteFile), ...(entry.subjects || [])];
  const missing = paths.filter(p => !fs.existsSync(path.join(ROOT, p)));
  if (missing.length) return { stale: true, why: `subject path(s) gone: ${missing.join(', ')}` };

  const lastTouch = git(['log', '-1', '--format=%H', '--', ...paths]);
  if (!lastTouch) return { stale: true, why: `git knows no commit touching ${paths.join(', ')}` };
  if (lastTouch === full) return { stale: false };

  // verifiedAt older than lastTouch == verifiedAt is an ancestor of it.
  const older = spawnSync('git', ['merge-base', '--is-ancestor', full, lastTouch],
    { cwd: ROOT }).status === 0;
  if (older) {
    const subj = git(['log', '-1', '--format=%h %s', lastTouch]);
    return { stale: true, why: `${verifiedAt} predates ${subj} which touched ${paths.join(', ')}` };
  }
  return { stale: false };
}

/**
 * Suites report their tally in one of two house styles. THE EXIT CODE IS THE
 * AUTHORITY on pass/fail; this count only feeds the baseline diff.
 *
 * The loose `(\d+) failed` fallback this started with matched "3 failed checks
 * 60s apart" out of the watchdog's own policy line and reported a green suite
 * as three failures — a reader of a rendered string mistaking it for a field,
 * which is the same mistake the suites themselves are being cured of. So only
 * the two explicit summary forms are read as a tally, and where there is no
 * tally at all the exit code answers instead.
 */
function parseFailures(stdout, exitCode) {
  // Three house styles, and they are matched exactly. Anything looser reads a
  // number out of the suite's own prose — see the comment above.
  const explicit = [
    /(\d+)\s+CHECK\(S\)\s+FAILED/gi,
    /(\d+)\s+passed[,.]?\s+(\d+)\s+failed/gi,
    /(\d+)\s+FAILED,\s+\d+\s+passed/gi,
  ];
  for (const re of explicit) {
    const hits = [...stdout.matchAll(re)];
    if (hits.length) {
      const last = hits[hits.length - 1];
      return { count: Number(last[last.length - 1]), source: 'tally' };
    }
  }
  if (/✅\s*ALL\s*PASS(ED)?\b|^\s*ALL PASS\b|^All \d+ checks pass\./mi.test(stdout)) {
    return { count: 0, source: 'banner' };
  }
  // No tally printed. A suite that exited 0 has said it passed; that is a
  // claim it made itself, not one inferred from its prose.
  if (exitCode === 0) return { count: 0, source: 'exit-0' };
  const marks = (stdout.match(/❌/g) || []).length;
  if (marks > 0) return { count: marks, source: 'marks' };
  return null;
}

function runSuite(entry) {
  const suiteFile = path.join(__dirname, `${entry.suite}.js`);
  if (!fs.existsSync(suiteFile)) {
    return { entry, kind: UNKNOWN, why: `no such suite file: scripts/${entry.suite}.js` };
  }

  // A manual tool, not a suite: it has no assertions and it acts on a real
  // container. It is listed so it is accounted for, never run by the runner.
  if (entry.manual) {
    return { entry, kind: SKIPPED, why: 'manual tool, not an assertion suite — run it by hand' };
  }
  if (entry.needsInstance && !INCLUDE_INSTANCE) {
    return { entry, kind: SKIPPED, why: 'needs a running instance (--include-instance, and point PORT at a throwaway)' };
  }
  // A suite whose fixtures ARE the live corpus. It reads rows the redirect does
  // not have, so SNH_DATA_DIR makes it fail for the wrong reason — but running
  // it means reading the live store, which the runner will not do unasked.
  if (entry.needsLiveCorpus && !INCLUDE_LIVE_CORPUS) {
    return { entry, kind: SKIPPED, why: 'its fixtures are the live corpus (--include-live-corpus; read-only)' };
  }
  if (entry.flaky && !INCLUDE_FLAKY) {
    return { entry, kind: SKIPPED, why: 'engine-dependent, varies run to run (--include-flaky)' };
  }

  // The data-dir contract is per-suite and part of the entry, because getting
  // it wrong is not a test failure — it is a different test.
  const env = { ...process.env };
  let tmp = null;
  if (entry.dataDir === 'required') {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), `snh-suite-${entry.suite}-`));
    env.SNH_DATA_DIR = tmp;
  } else if (entry.dataDir === 'forbidden') {
    delete env.SNH_DATA_DIR;
  }

  const started = Date.now();
  const r = spawnSync(process.execPath, [suiteFile], {
    cwd: ROOT, env, encoding: 'utf8',
    timeout: (entry.timeoutSeconds || 300) * 1000,
    maxBuffer: 64 * 1024 * 1024,
  });
  const secs = ((Date.now() - started) / 1000).toFixed(1);
  const out = (r.stdout || '') + (r.stderr || '');
  if (tmp) fs.rmSync(tmp, { recursive: true, force: true });

  const base = { entry, secs, code: r.status, out, tail: out.trim().split('\n').slice(-6).join('\n') };

  // RULE 1, and it is checked before anything else: a suite that did not exit
  // 0 or 1 did not report a failure count, so there is nothing to diff.
  if (r.error && r.error.code === 'ETIMEDOUT') {
    return { ...base, kind: UNKNOWN, why: `timed out after ${entry.timeoutSeconds || 300}s` };
  }
  if (r.signal) return { ...base, kind: UNKNOWN, why: `killed by signal ${r.signal}` };
  if (r.status !== 0 && r.status !== 1) {
    return { ...base, kind: UNKNOWN, why: `exit ${r.status} — crashed rather than failed` };
  }

  const parsed = parseFailures(out, r.status);
  if (parsed === null) {
    return { ...base, kind: UNKNOWN, why: `exit ${r.status} with no readable failure count in its output` };
  }
  const observed = parsed.count;
  // A suite that exits 0 while printing a non-zero tally is disagreeing with
  // itself, and neither half can be trusted over the other.
  if (r.status === 0 && parsed.source === 'tally' && observed !== 0) {
    return { ...base, observed, kind: UNKNOWN, why: `exit 0 but its own tally says ${observed} failure(s)` };
  }

  // An entry with no verified count is not a claim, and running the suite does
  // not turn it into one. Two shapes reach here: a suite this runner has never
  // been able to verify (it needs a live instance), and one whose count is not a
  // number at all (the engine-dependent pair, seen at 0, 3, 4, 6 and 20 failures
  // on identical code). Writing a figure into either would be inventing the
  // thing this file exists to stop being invented.
  if (entry.expectedFailures === null || entry.expectedFailures === undefined) {
    return { ...base, observed, kind: UNKNOWN,
      why: `no verified count in the baseline — observed ${observed}. ${entry.reason || ''}`.trim() };
  }

  const stale = stalenessOf(entry, suiteFile);
  if (stale.stale && !RECORD) {
    return { ...base, observed, kind: UNKNOWN, why: `STALE entry — ${stale.why}` };
  }

  if (observed === entry.expectedFailures) return { ...base, observed, kind: OK };
  if (observed > entry.expectedFailures) {
    return { ...base, observed, kind: REGRESSION, why: `expected ${entry.expectedFailures}, got ${observed}` };
  }
  return { ...base, observed, kind: IMPROVED, why: `expected ${entry.expectedFailures}, got ${observed} — fix the baseline` };
}

(async () => {
  let entries = baseline.suites;
  if (named.length) entries = entries.filter(e => named.includes(e.suite));

  // A suite file with no entry is not "fine", it is unaccounted for.
  const onDisk = fs.readdirSync(__dirname)
    .filter(f => /^test-.*\.js$/.test(f)).map(f => path.basename(f, '.js'));
  const unlisted = onDisk.filter(s => !baseline.suites.some(e => e.suite === s));

  const results = [];
  for (const entry of entries) {
    process.stdout.write(`… ${entry.suite}`);
    const r = runSuite(entry);
    results.push(r);
    process.stdout.write(`\r${r.kind === OK ? '  ok  ' : r.kind === SKIPPED ? ' skip ' : ' !!!! '}` +
      `${entry.suite.padEnd(32)} ${r.kind}${r.why ? ' — ' + r.why : ''}` +
      `${r.secs ? ` (${r.secs}s)` : ''}\n`);
  }

  if (RECORD) {
    console.log('\n--- observed, for curating into scripts/test-baseline.json ---');
    console.log(JSON.stringify(results.map(r => ({
      suite: r.entry.suite, exit: r.code, observedFailures: r.observed ?? null, kind: r.kind,
    })), null, 2));
  }

  const by = k => results.filter(r => r.kind === k);
  const bar = '─'.repeat(72);
  console.log(`\n${bar}`);
  console.log(`matches baseline ${by(OK).length}   regression ${by(REGRESSION).length}   ` +
    `improved ${by(IMPROVED).length}   UNKNOWN ${by(UNKNOWN).length}   skipped ${by(SKIPPED).length}`);
  console.log(bar);

  // RULE 3. Unknown is loud: it gets its own block, above everything else, and
  // it never merges into the "as expected" line.
  for (const [label, kind] of [['UNKNOWN — no claim can be made about these', UNKNOWN],
                               ['REGRESSION', REGRESSION], ['IMPROVED', IMPROVED]]) {
    const rs = by(kind);
    if (!rs.length) continue;
    console.log(`\n${label}:`);
    for (const r of rs) {
      console.log(`  ${r.entry.suite}: ${r.why}`);
      if (kind === UNKNOWN && r.tail) console.log(r.tail.split('\n').map(l => '      | ' + l).join('\n'));
    }
  }
  if (unlisted.length) {
    console.log(`\nUNKNOWN — suite files with no baseline entry (add one or delete them):`);
    for (const s of unlisted) console.log(`  ${s}`);
  }

  const bad = by(UNKNOWN).length + by(REGRESSION).length + by(IMPROVED).length + unlisted.length;
  if (bad === 0 && !by(SKIPPED).length) console.log('\n✅ every suite matched its baseline entry.');
  process.exit(bad === 0 ? 0 : 1);
})();
