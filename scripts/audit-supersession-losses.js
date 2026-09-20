#!/usr/bin/env node
/**
 * WHAT DID A MERGE TAKE WITH IT?
 *
 * READ-ONLY audit for the silent data loss Athena reported on 2026-08-24: until
 * db/fact-merge.js landed, folding one fact into another kept the survivor's
 * wording and dropped whatever the loser asserted and the survivor did not. The
 * loser stayed as an inactive row with a successor pointer, so every one of
 * those losses is still recoverable — this walks them and says which ones took
 * something with them.
 *
 * For each fact that is inactive/superseded, it follows the successor CHAIN to
 * whatever is active today and asks whether the assertions the retired fact
 * carried are still somewhere in the active corpus. Anything that is not is a
 * victim, listed with the exact detail that went missing.
 *
 * It writes NOTHING. Restoration is a separate, deliberate act — see the
 * --restore-hint output, which prints the write_memory statement that would put
 * a lost assertion back.
 *
 * Run it against any SNH store, including a live one:
 *   node scripts/audit-supersession-losses.js
 *   SNH_DATA_DIR=/path/to/other/store node scripts/audit-supersession-losses.js
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const Database = require(path.join(ROOT, 'node_modules/better-sqlite3'));
const factMerge = require(path.join(ROOT, 'db/fact-merge'));

const DATA_DIR = process.env.SNH_DATA_DIR ? path.resolve(process.env.SNH_DATA_DIR) : path.join(ROOT, 'data');
const DB_PATH = path.join(DATA_DIR, 'chat.db');

const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
console.log(`Auditing ${DB_PATH} (read-only)\n`);

const rows = db.prepare('SELECT * FROM cluster_members').all();
const byId = new Map(rows.map(r => [r.id, r]));
const active = rows.filter(r => r.status === 'active');
const activeBySubject = new Map();
for (const r of active) {
  const s = r.subject || 'user';
  if (!activeBySubject.has(s)) activeBySubject.set(s, []);
  activeBySubject.get(s).push(r);
}

const retired = rows.filter(r => r.status !== 'active' && (r.inactive_reason === 'superseded' || r.successor_id));
console.log(`${rows.length} fact(s) total — ${active.length} active, ${retired.length} superseded/linked.\n`);

/** Follow successor pointers to whatever is active now (or the end of the chain). */
function chainEnd(row) {
  const seen = new Set([row.id]);
  let cur = row;
  while (cur && cur.status !== 'active') {
    const next = cur.successor_id || cur.superseded_by;
    if (!next || seen.has(next)) break;
    seen.add(next);
    cur = byId.get(next);
  }
  return cur || null;
}

const victims = [];
for (const r of retired) {
  const end = chainEnd(r);
  const subject = r.subject || 'user';
  // The whole ACTIVE corpus of that subject is the haystack — a detail that
  // moved into a different row (a split, a separate re-save) is not lost.
  const haystack = (activeBySubject.get(subject) || []).map(a => a.content).join(' \n ');
  const needed = factMerge.contentTokens(r.content);
  const lost = factMerge.missingTokens(needed, haystack);
  if (!lost.length) continue;
  victims.push({ row: r, end, lost, subject });
}

if (!victims.length) {
  console.log('No supersession dropped an assertion that is not still somewhere in the active corpus.\n');
} else {
  console.log(`${victims.length} supersession(s) took something with them:\n`);
  for (const v of victims) {
    console.log(`  [${v.subject}] ${v.row.id.slice(0, 8)}  superseded ${v.row.updated_at || v.row.created_at}`);
    console.log(`    retired : "${v.row.content}"`);
    console.log(`    now     : ${v.end && v.end.status === 'active' ? `"${v.end.content}"` : '(the chain ends with nothing active)'}`);
    console.log(`    MISSING : ${v.lost.join(', ')}`);
    console.log(`    restore : write_memory "${v.row.content}"`);
    console.log('');
  }
}

// The ledger's own view, for cross-checking that every retirement was recorded.
try {
  const led = db.prepare("SELECT action, COUNT(*) n FROM corrections_ledger GROUP BY action").all();
  console.log('Ledger entries by action: ' + (led.length ? led.map(l => `${l.action}=${l.n}`).join(', ') : '(none)'));
  const unledgered = retired.filter(r =>
    !db.prepare('SELECT 1 FROM corrections_ledger WHERE target_id = ?').get(r.id));
  if (unledgered.length) {
    console.log(`\nWARNING: ${unledgered.length} retired fact(s) have NO ledger entry — their retirement was not recorded:`);
    for (const u of unledgered) console.log(`  ${u.id.slice(0, 8)} "${u.content.slice(0, 90)}"`);
  } else {
    console.log('Every retired fact has a ledger entry.');
  }
} catch (e) {
  console.log(`(no corrections_ledger in this store: ${e.message})`);
}

// The other silent fold: a repeat absorbed into a held fact, which never becomes
// a row at all. Its only record is the corroboration.
try {
  const cor = db.prepare('SELECT member_id, restated_as FROM fact_corroborations WHERE restated_as IS NOT NULL').all();
  const repeatVictims = [];
  for (const c of cor) {
    const held = byId.get(c.member_id);
    if (!held) continue;
    const lost = factMerge.missingTokens(factMerge.contentTokens(c.restated_as), held.content);
    if (lost.length) repeatVictims.push({ c, held, lost });
  }
  console.log(`\n${cor.length} corroboration(s) recorded; ${repeatVictims.length} restatement(s) carried detail the held fact does not.`);
  for (const v of repeatVictims) {
    console.log(`  held    : "${v.held.content}"`);
    console.log(`  restated: "${v.c.restated_as}"`);
    console.log(`  MISSING : ${v.lost.join(', ')}\n`);
  }
} catch (e) {
  console.log(`(no fact_corroborations in this store: ${e.message})`);
}

db.close();
