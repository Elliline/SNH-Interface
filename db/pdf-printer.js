/**
 * HTML → PDF, by handing it to a headless chromium.
 *
 * NO PDF LIBRARY, DELIBERATELY. A PDF writer is a large dependency that would
 * then own how every report looks, and it would still not lay out a table, wrap
 * a paragraph or draw an SVG as well as the engine already on the machine. The
 * browser does typesetting; we write HTML and ask it to print.
 *
 * WHAT "PRINT" MEANS HERE. `chromium --headless --print-to-pdf` is chromium's own
 * `Page.printToPDF` — the DevTools command — exposed as a command-line flag; it
 * is the same code inside the browser either way. Driving the DevTools socket
 * directly would mean hand-writing a WebSocket client (there is no `ws` here and
 * no dependency is being added for this), and it would buy exactly one thing we
 * do not otherwise have: page numbers in a footer, via headerTemplate. That is
 * the whole difference, and it is written down in db/report-html.js as well so
 * nobody goes looking for the feature twice. Everything else — paper size,
 * margins, backgrounds, page-break control — is CSS, and CSS is where paged
 * media puts it.
 *
 * ⚠ CHROMIUM IS NOT ASSUMED TO EXIST. On this box it did not, and on Ubuntu 24.04
 * there is no apt package for it — `chromium-browser` is a transitional stub and
 * the real install is `sudo snap install chromium`, which needs a human. So the
 * absence is a FIRST-CLASS OUTCOME, not an error path: probe() says so plainly,
 * the caller writes a formatted text file instead, and the card carries one line
 * explaining why the report is not a PDF. A missing browser downgrades what the
 * file looks like. It never costs the work.
 *
 * ⚠ SNAP CONFINEMENT SHAPES TWO THINGS HERE, and both are non-obvious:
 *   - A snap gets a PRIVATE /tmp. Chromium cannot read a file the server wrote
 *     to the system temp directory, so the intermediate HTML is written under
 *     the data directory instead — which is under $HOME on any real install.
 *   - The snap `home` interface does not cross HIDDEN directories. Nothing this
 *     writes may live in a dot-directory, which is why the work directory is
 *     `print-work` and not `.print-work`.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { getDataDir } = require('./database');

/** Names to try on PATH, plus the place snap puts its wrapper. */
const CANDIDATES = [
  'chromium', 'chromium-browser', 'google-chrome-stable', 'google-chrome',
  'chrome', 'chrome-headless-shell'
];
const EXTRA_PATHS = [
  '/snap/bin/chromium',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/usr/bin/google-chrome',
  '/opt/google/chrome/chrome'
];

/**
 * Probe cache.
 *
 * A POSITIVE result is cached for the life of the process — the binary does not
 * move. A NEGATIVE is cached for a minute and no longer, because the expected
 * way this box goes from "no chromium" to "chromium" is Ellie running one snap
 * command, and it would be absurd to make her restart the server to be believed.
 *
 * Keyed by the configured path, because the answer is only about the binary it
 * was asked about. Unkeyed, changing Settings -> Job output left the old answer
 * standing for the life of the process — invisible while only printToPdf read
 * this, and no longer invisible now that the capability manifest reads it on
 * every chat turn to decide what to tell him he produces.
 */
let probeCache = null;      // the verified answer, from probe()
let probeCacheAt = 0;
let syncCache = null;       // the no-spawn answer, from probeSync()
let syncCacheAt = 0;
let cacheKey = null;        // the configured path both of the above are about
const NEGATIVE_TTL_MS = 60_000;

/**
 * Said once, because two callers now report it: printToPdf puts it on the card,
 * and probeSync hands it to the capability manifest. It is written for a person
 * and names the one command that fixes it.
 */
const NO_BROWSER_REASON =
  'no chromium on this machine, so the report was written as a text file instead. '
  + 'Install one with "sudo snap install chromium" (Ubuntu has no apt package for it), '
  + 'or set the path in Settings → Job output, and the next report will be a PDF.';

/** Is a cached answer still good for the path we are being asked about? */
function cacheHit(cache, at, key) {
  // A different path invalidates both caches at once: neither of them is an
  // answer about the binary being asked about now.
  if (!cache || key !== cacheKey) return false;
  return cache.ok || Date.now() - at < NEGATIVE_TTL_MS;
}

/** Run a command with a hard timeout. Never throws; reports instead. */
function run(cmd, args, { timeoutMs = 60_000, cwd } = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (err) {
      resolve({ ok: false, code: null, stdout: '', stderr: String(err && err.message || err) });
      return;
    }
    let stdout = '', stderr = '', done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      try { child.kill('SIGKILL'); } catch { /* already gone */ }
      resolve({ ok: false, code: null, stdout, stderr, timedOut: true });
    }, timeoutMs);

    child.stdout.on('data', d => { stdout += d.toString(); });
    child.stderr.on('data', d => { stderr += d.toString(); });
    child.on('error', (err) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ ok: false, code: null, stdout, stderr: String(err && err.message || err) });
    });
    child.on('close', (code) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ ok: code === 0, code, stdout, stderr });
    });
  });
}

/** Absolute path to a chromium we can actually execute, or null. */
function findBinary(configured) {
  const tried = [];
  if (configured) {
    // A configured path is used as given and NOT silently fallen back from — a
    // setting that is quietly ignored is worse than one that fails.
    try {
      fs.accessSync(configured, fs.constants.X_OK);
      return { path: configured, tried };
    } catch {
      return { path: null, tried: [`${configured} (from settings — not executable)`] };
    }
  }
  for (const p of EXTRA_PATHS) {
    tried.push(p);
    try { fs.accessSync(p, fs.constants.X_OK); return { path: p, tried }; } catch { /* next */ }
  }
  const dirs = String(process.env.PATH || '').split(path.delimiter).filter(Boolean);
  for (const name of CANDIDATES) {
    for (const dir of dirs) {
      const p = path.join(dir, name);
      try { fs.accessSync(p, fs.constants.X_OK); return { path: p, tried }; } catch { /* next */ }
    }
    tried.push(`${name} (on PATH)`);
  }
  return { path: null, tried };
}

/**
 * Is there a browser to print with?
 *
 * @returns {Promise<{ok: boolean, path?: string, version?: string, reason?: string}>}
 *          `reason` is written for a person: it goes on the card verbatim.
 */
async function probe({ chromiumPath = '' } = {}) {
  const now = Date.now();
  const key = chromiumPath ? String(chromiumPath).trim() : '';
  if (cacheHit(probeCache, probeCacheAt, key)) return probeCache;
  cacheKey = key;

  const found = findBinary(key);
  if (!found.path) {
    probeCache = { ok: false, verified: true, reason: NO_BROWSER_REASON };
    probeCacheAt = now;
    return probeCache;
  }

  const res = await run(found.path, ['--version'], { timeoutMs: 15_000 });
  if (!res.ok) {
    probeCache = {
      ok: false,
      verified: true,
      path: found.path,
      reason: `chromium at ${found.path} would not run (${(res.stderr || res.stdout || 'no output').trim().slice(0, 200)}), `
        + 'so the report was written as a text file instead.'
    };
    probeCacheAt = now;
    return probeCache;
  }

  probeCache = { ok: true, verified: true, path: found.path, version: res.stdout.trim() };
  probeCacheAt = now;
  return probeCache;
}

/**
 * The same question probe() answers, WITHOUT spawning anything.
 *
 * The capability manifest asks this on the chat injection path — which runs on
 * every message and may not spawn a process or touch the network — so that the
 * list he is handed can say what he actually produces on THIS box instead of
 * hedging about what a browser would do if there were one. It stops at "is
 * there a binary here we could execute", which is a handful of stat() calls,
 * cached the same way as above.
 *
 * IT IS ONE NOTCH WEAKER THAN probe(), and deliberately so: a binary that is
 * present but will not run reads as ok here, because verifying that means
 * running it. That is the right way round for this caller — the alternative is
 * spawning chromium on every chat turn — and the window is small: a VERIFIED
 * answer always wins over this one, checkDrift() takes one on the heartbeat,
 * and the first real print takes one too. Both write the shared cache below.
 *
 * @returns {{ok: boolean, path?: string, verified?: boolean, reason?: string}}
 */
function probeSync({ chromiumPath = '' } = {}) {
  const now = Date.now();
  const key = chromiumPath ? String(chromiumPath).trim() : '';
  if (cacheHit(probeCache, probeCacheAt, key)) return probeCache;
  if (cacheHit(syncCache, syncCacheAt, key)) return syncCache;

  const found = findBinary(key);
  // Kept in its own cache rather than written to probeCache: an unverified yes
  // must not stand in for the version check the next real probe() would run.
  syncCache = found.path
    ? { ok: true, path: found.path, verified: false }
    : { ok: false, verified: false, reason: NO_BROWSER_REASON };
  syncCacheAt = now;
  cacheKey = key;
  return syncCache;
}

/** Forget what we learned — for tests, and for anyone who just installed one. */
function resetProbe() {
  probeCache = null; probeCacheAt = 0; cacheKey = null;
  syncCache = null; syncCacheAt = 0;
}

/**
 * Where the intermediate HTML goes.
 *
 * Under the data directory, NOT the system temp directory: a snap-confined
 * chromium has a private /tmp and cannot see anything the server puts there.
 * Non-hidden, because the snap home interface does not cross dot-directories.
 */
function workDir() {
  const dir = path.join(getDataDir(), 'print-work');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/** file:// URL for a path, with the characters that matter percent-encoded. */
function fileUrl(p) {
  return 'file://' + path.resolve(p).split('/').map(encodeURIComponent).join('/');
}

/** A chromium failure that means "the sandbox could not start", not "bad input". */
function isSandboxFailure(text) {
  return /namespace|sandbox|SUID|clone\(\)|Operation not permitted/i.test(String(text || ''));
}

/** Is this binary running under snap confinement? */
function isConfined(binPath) {
  return /^\/snap\//.test(String(binPath || ''));
}

/**
 * The one failure a confined browser gives no useful error for.
 *
 * A snap can only read NON-HIDDEN paths under $HOME. Point one at a file
 * anywhere else — /tmp, /var, a hidden directory — and it does not report a
 * permission error and does not exit non-zero. **It exits 0 and writes no PDF at
 * all.** Measured on Chromium 151 snap: exit code 0, empty stderr, no output
 * file. The empty-PDF guard downstream catches it, but on its own it reports
 * "chromium reported success but produced an empty PDF", which sends whoever
 * reads it looking at the HTML — the one place the fault is not.
 *
 * So the condition is checked BEFORE spawning, and named. This is the normal
 * state of affairs for any instance whose data directory has been redirected
 * outside $HOME, which is every throwaway test instance.
 *
 * @returns {string|null} a sentence for the card, or null if there is no problem
 */
function confinementProblem(binPath, htmlPath) {
  if (!isConfined(binPath)) return null;
  const home = os.homedir();
  const p = path.resolve(htmlPath);
  const underHome = p === home || p.startsWith(home + path.sep);
  // A hidden directory anywhere in the path is refused by the snap home
  // interface just as firmly as a path outside $HOME.
  const hidden = p.split(path.sep).some(seg => seg.startsWith('.') && seg.length > 1);
  if (underHome && !hidden) return null;
  return `chromium is installed as a snap, which can only read files under ${home}, `
    + `and this instance builds its report in ${path.dirname(p)}. `
    + (underHome
      ? 'A hidden directory in that path is enough to block it. '
      : 'That is normal for a test instance with a redirected data directory. ')
    + 'The report was written as a text file instead.';
}

/**
 * Print HTML to a PDF at `outPath`.
 *
 * @param {string} html      a complete, self-contained document
 * @param {string} outPath   absolute path to write
 * @param {Object} [opts]
 * @param {string} [opts.chromiumPath]
 * @param {boolean} [opts.keepHtml] leave the intermediate HTML beside the PDF
 * @param {number} [opts.timeoutMs]
 * @returns {Promise<{ok: boolean, bytes?: number, reason?: string, htmlPath?: string, warning?: string}>}
 *          Never throws. `reason` is a sentence for the card.
 */
async function printToPdf(html, outPath, opts = {}) {
  const { chromiumPath = '', keepHtml = false, timeoutMs = 120_000 } = opts;

  const found = await probe({ chromiumPath });
  if (!found.ok) return { ok: false, reason: found.reason };

  const dir = workDir();
  const stem = path.basename(outPath).replace(/\.pdf$/i, '') || 'report';
  const htmlPath = path.join(dir, `${stem}-${process.pid}-${Date.now()}.html`);
  // Chromium writes a profile whether we want one or not; giving it a directory
  // of ours keeps it out of the real browser profile Ellie may be using.
  const profileDir = path.join(dir, 'profile');

  let warning = '';
  try {
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(htmlPath, html, 'utf8');

    const baseArgs = [
      '--headless=new',
      '--disable-gpu',
      '--disable-dev-shm-usage',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-extensions',
      // The page is local and has no scripts, but there is no reason for the
      // browser to be able to reach the network at all while it renders it.
      '--disable-background-networking',
      '--disable-component-update',
      `--user-data-dir=${profileDir}`,
      // Chromium's own header/footer is a file:// URL and a date, which looks
      // worse than nothing. The document draws its own running footer.
      '--no-pdf-header-footer',
      // Nothing here is async, but a small budget guarantees layout has settled
      // before the snapshot rather than relying on it.
      '--virtual-time-budget=3000',
      `--print-to-pdf=${outPath}`,
      fileUrl(htmlPath)
    ];

    const confined = confinementProblem(found.path, htmlPath);
    if (confined) return { ok: false, reason: confined };

    let res = await run(found.path, baseArgs, { timeoutMs, cwd: dir });

    // --headless=new is rejected by older builds; the bare flag is the same
    // thing there. Retried once rather than version-sniffed, because the
    // version string format is its own moving target.
    if (!res.ok && /headless/i.test(res.stderr || '')) {
      res = await run(found.path, ['--headless', ...baseArgs.slice(1)], { timeoutMs, cwd: dir });
    }

    // THE SANDBOX IS ONLY DROPPED AS A LAST RESORT, and it is said out loud when
    // it happens. Under some service managers chromium cannot create its user
    // namespace and refuses to start at all. The page being rendered is our own
    // HTML with no scripts and no network, so the residual risk is a parser bug
    // rather than hostile code — but it is still a weakened sandbox and that
    // belongs in the record rather than in a silent retry.
    if (!res.ok && isSandboxFailure(res.stderr)) {
      warning = 'chromium could not start its sandbox on this machine, so the PDF was printed with it disabled';
      res = await run(found.path, ['--no-sandbox', ...baseArgs], { timeoutMs, cwd: dir });
    }

    if (!res.ok) {
      const why = res.timedOut
        ? `chromium did not finish within ${Math.round(timeoutMs / 1000)}s`
        : (String(res.stderr || res.stdout || '').trim().split('\n').filter(Boolean).pop() || `exit code ${res.code}`);
      return { ok: false, reason: `the PDF could not be printed (${why.slice(0, 220)}), so the report was written as a text file instead.` };
    }

    // A zero-exit chromium that wrote nothing is a real outcome, and it is the
    // one that would otherwise present as a finished report that will not open.
    let bytes = 0;
    try { bytes = fs.statSync(outPath).size; } catch { bytes = 0; }
    if (!bytes) {
      return {
        ok: false,
        reason: 'chromium reported success but produced an empty PDF, so the report was written as a text file instead.'
          + (isConfined(found.path)
            ? ' A snap-packaged chromium does this — exit code 0, no error, no file — when it cannot read the page it was given.'
            : '')
      };
    }

    return { ok: true, bytes, warning, htmlPath: keepHtml ? htmlPath : undefined };
  } catch (err) {
    return { ok: false, reason: `the PDF could not be printed (${String(err && err.message || err).slice(0, 200)}), so the report was written as a text file instead.` };
  } finally {
    if (!keepHtml) {
      try { fs.unlinkSync(htmlPath); } catch { /* it may never have been written */ }
    }
  }
}

module.exports = { probe, probeSync, resetProbe, printToPdf, findBinary };
