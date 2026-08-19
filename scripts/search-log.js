#!/usr/bin/env node
/**
 * "Was it Exa or SearXNG, and did it return anything?"
 *
 * The whole reason db/search-log.js exists, in the form a person actually asks
 * it. On 2026-08-18 that question was answerable only by reading a journal by
 * hand, and it cost hours.
 *
 * Read-only. No writes anywhere in this file.
 *
 * Usage:
 *   node scripts/search-log.js                 # summary (24h) + last 25 attempts
 *   node scripts/search-log.js --hours 168     # summary window
 *   node scripts/search-log.js --limit 100     # how many attempts to list
 *   node scripts/search-log.js --provider exa  # only one provider
 *   node scripts/search-log.js --failures      # only empty/error/skipped
 */

const path = require('path');
const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
const { providerSummary, recentSearchCalls } = require(path.join(ROOT, 'db/search-log'));

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 && process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[i + 1] : dflt;
}
const has = (name) => process.argv.includes(`--${name}`);

database.initDatabase();

const hours = parseInt(arg('hours', '24'), 10) || 24;
const limit = parseInt(arg('limit', '25'), 10) || 25;
const onlyProvider = arg('provider', null);

console.log(`\n=== Search providers, last ${hours}h ===\n`);
const summary = providerSummary({ hours });
if (!summary.length) {
  console.log('  No searches in the window.');
} else {
  console.log('  provider  attempts  served  empty  errors  skipped   avg ms   cost');
  for (const r of summary) {
    if (onlyProvider && r.provider !== onlyProvider) continue;
    console.log(
      `  ${String(r.provider).padEnd(9)} ${String(r.attempts).padStart(8)} ${String(r.with_results).padStart(7)}` +
      ` ${String(r.empty).padStart(6)} ${String(r.errors).padStart(7)} ${String(r.skipped).padStart(8)}` +
      ` ${String(r.avg_latency_ms ?? '-').padStart(8)}   ${r.cost_usd ? `$${r.cost_usd}` : '-'}`
    );
  }
}

console.log(`\n=== Last ${limit} attempts ===\n`);
let rows = recentSearchCalls({ limit: Math.max(limit, 200) });
if (onlyProvider) rows = rows.filter(r => r.provider === onlyProvider);
if (has('failures')) rows = rows.filter(r => r.outcome !== 'results');
rows = rows.slice(0, limit);

if (!rows.length) {
  console.log('  Nothing matched.');
} else {
  for (const r of rows) {
    const when = String(r.created_at || '').replace('T', ' ').slice(0, 19);
    const served = r.served ? ' *served*' : '';
    console.log(`  ${when}  ${String(r.provider).padEnd(8)} ${String(r.outcome).padEnd(8)} ${String(r.num_results).padStart(2)} result(s)` +
      `  [${r.caller || 'unknown'}]${served}`);
    console.log(`      "${r.query}"`);
    if (r.detail) console.log(`      → ${r.detail}`);
  }
}
console.log('');
