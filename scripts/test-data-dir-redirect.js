#!/usr/bin/env node
/**
 * No module may resolve a store or a log from __dirname.
 *
 * SNH_DATA_DIR redirects a whole PROCESS at a different corpus — that is the
 * mechanism the replay, the merge and every staging tool rest on, and it works
 * precisely because no call site has to remember a flag. A path built from
 * __dirname is a hole in it, and every hole found so far was found by reading
 * output rather than by anything failing:
 *
 *   - db/corrector.js wrote its pass-state into the LIVE data dir during a
 *     staging pass, which tells the live heartbeat the corrector had just run
 *     and suppresses its next real one.
 *   - db/corrections-ledger.js wrote three "is back in memory" notes into the
 *     LIVE daily log — a log that is injected into his context — about facts
 *     that had never moved in the corpus those notes described.
 *   - db/identity-lock.js would have filed a staging lock refusal in the live
 *     ops ledger.
 *
 * So the audit is a test rather than a sweep somebody remembers to redo. It is a
 * grep with an allowlist, and the allowlist has exactly two entries, each of
 * which is the reason it is allowed.
 *
 * PURE. Reads source files; touches no database and no model.
 *
 * Usage: node scripts/test-data-dir-redirect.js
 */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

/**
 * Paths allowed to build a data path from __dirname, and why.
 *
 * Adding to this list is a decision, not a convenience. If a new entry appears
 * here, the question to answer first is "what happens when a staging run writes
 * this?" — and if the answer is "it lands in live", it does not belong here.
 */
const ALLOWED = {
  'db/database.js':
    'defines DATA_DIR — this is the redirect itself, and the __dirname form is its fallback when SNH_DATA_DIR is unset',
  'db/config.js':
    'configuration is not corpus. A staging run is the same system pointed at a different store and must use the same models and thresholds; redirecting it would silently fall back to bare DEFAULTS because data-staging holds no config.json'
};

// A data path built from the module's own location rather than the process's.
const OFFENDER = /__dirname\s*,\s*['"`]\.\.?\/data/;

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, out);
    else if (entry.name.endsWith('.js')) out.push(p);
  }
  return out;
}

const roots = ['db', 'routes', 'lib'].map(d => path.join(ROOT, d)).filter(fs.existsSync);
const files = roots.flatMap(d => walk(d));
const serverJs = path.join(ROOT, 'server.js');
if (fs.existsSync(serverJs)) files.push(serverJs);

const findings = [];
for (const file of files) {
  const rel = path.relative(ROOT, file);
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  lines.forEach((line, i) => {
    if (!OFFENDER.test(line)) return;
    // A mention inside a comment is documentation, not a path being built.
    if (/^\s*(\*|\/\/)/.test(line)) return;
    findings.push({ rel, line: i + 1, text: line.trim(), allowed: ALLOWED[rel] || null });
  });
}

const violations = findings.filter(f => !f.allowed);
const allowed = findings.filter(f => f.allowed);

const bar = '='.repeat(74);
console.log(`\n${bar}\nDATA-DIR REDIRECT AUDIT — no module resolves a store from __dirname\n${bar}\n`);
console.log(`scanned ${files.length} file(s) under db/, routes/, lib/, server.js\n`);

for (const a of allowed) {
  console.log(`ALLOWED  ${a.rel}:${a.line}`);
  console.log(`         ${a.text}`);
  console.log(`         why: ${a.allowed}\n`);
}

if (violations.length) {
  console.log(`${violations.length} VIOLATION(S) — these escape SNH_DATA_DIR:\n`);
  for (const v of violations) {
    console.log(`FAIL     ${v.rel}:${v.line}`);
    console.log(`         ${v.text}`);
    console.log('         fix: resolve it from database.getMemoryDir() / getDailyDir() / getOpsDir()\n');
  }
}

console.log(bar);
console.log(violations.length === 0
  ? `Clean. ${allowed.length} documented exception(s), 0 violations.`
  : `${violations.length} module(s) can still write to live from a staging run.`);
console.log(`${bar}\n`);
process.exit(violations.length === 0 ? 0 : 1);
