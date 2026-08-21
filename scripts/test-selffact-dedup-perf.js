#!/usr/bin/env node
/**
 * test-selffact-dedup-perf — the dedup sweep reads vectors, it does not
 * regenerate them, and a dedup that could not run says so.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-selffact-dedup-perf.js
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  process.exit(1);
}

const database = require('../db/database');
const memoryClusters = require('../db/memory-clusters');

let passed = 0, failed = 0;
async function test(name, fn) {
  try { await fn(); passed++; console.log(`  ok   ${name}`); }
  catch (err) { failed++; console.log(`  FAIL ${name}\n       ${err.message}`); }
}

(async () => {
  database.initDatabase();
  await database.initVectorStore();

  console.log('\nstored vectors are read, not regenerated\n');

  await test('getStoredEmbeddings returns a Map keyed by member id', async () => {
    const m = await memoryClusters.getStoredEmbeddings([]);
    assert.ok(m instanceof Map);
    assert.strictEqual(m.size, 0);
  });

  await test('an unknown id is simply absent, not an error', async () => {
    const m = await memoryClusters.getStoredEmbeddings(['no-such-member-id']);
    assert.strictEqual(m.size, 0);
  });

  await test('nulls and undefined in the id list are ignored', async () => {
    const m = await memoryClusters.getStoredEmbeddings([null, undefined, '']);
    assert.strictEqual(m.size, 0);
  });

  await test('the dedup sweep no longer calls generateEmbedding per existing fact', () => {
    // Structural: the loop must consult stored vectors first. A future
    // edit that reinstates the unconditional call reintroduces 6 minutes.
    const src = fs.readFileSync(path.join(__dirname, '../db/fact-extractor.js'), 'utf8');
    const sweep = src.slice(src.indexOf('Semantic dedup'), src.indexOf('Identity lock: drop'));
    assert.ok(/getStoredEmbeddings/.test(sweep),
      'the sweep does not read stored vectors');
    assert.ok(/storedEmbs\.get\(/.test(sweep),
      'the sweep does not look up per fact before embedding');
  });

  await test('dedupe-self-facts.js reads stored vectors too', () => {
    const src = fs.readFileSync(path.join(__dirname, 'dedupe-self-facts.js'), 'utf8');
    assert.ok(/getStoredEmbeddings/.test(src),
      'the dedupe script still regenerates every embedding');
  });

  await test('dedupe-self-facts.js REFUSES to run while blind to part of the corpus', () => {
    const src = fs.readFileSync(path.join(__dirname, 'dedupe-self-facts.js'), 'utf8');
    assert.ok(/REFUSING TO RUN/.test(src),
      'it would supersede facts without comparing against all of them');
  });

  console.log('\na dedup that could not run is loud\n');

  await test('a skipped dedup is reported in the result', async () => {
    const factExtractor = require('../db/fact-extractor');
    const original = memoryClusters.getSelfFacts;
    memoryClusters.getSelfFacts = () => { throw new Error('embedding provider down'); };
    try {
      const res = await factExtractor.processSelfFacts(
        ['I notice I explain things at length when I am unsure.'],
        { source: 'capability-intro' });
      assert.ok(res.dedupSkipped, 'the result did not say dedup was skipped');
      assert.match(res.dedupSkipped.reason, /embedding provider down/);
    } finally {
      memoryClusters.getSelfFacts = original;
    }
  });

  await test('...and it reaches the bell, once', () => {
    const db = database.getSqliteDb();
    const n = db.prepare(
      "SELECT COUNT(*) n FROM initiatives WHERE source_kind = 'self-fact-dedup-skipped'"
    ).get().n;
    assert.ok(n >= 1, 'no bell alert was raised for a skipped dedup');
    assert.ok(n <= 1, `the window did not hold: ${n} alerts`);
  });

  await test('...in his voice, never as a system error', () => {
    const db = database.getSqliteDb();
    const row = db.prepare(
      "SELECT content FROM initiatives WHERE source_kind = 'self-fact-dedup-skipped' LIMIT 1"
    ).get();
    assert.ok(row, 'no alert to inspect');
    assert.match(row.content, /my own memory/i);
    assert.ok(!/error|exception|stack/i.test(row.content.split('(')[0]),
      'the alert reads as a system error rather than something he is saying');
  });

  await test('...and it says what it means for her, not just that it happened', () => {
    const db = database.getSqliteDb();
    const row = db.prepare(
      "SELECT content FROM initiatives WHERE source_kind = 'self-fact-dedup-skipped' LIMIT 1"
    ).get();
    assert.match(row.content, /repeating myself/i);
    assert.match(row.content, /nothing was lost/i);
  });

  console.log(`\n${passed} passed, ${failed} failed\n`);
  process.exit(failed ? 1 : 0);
})();
