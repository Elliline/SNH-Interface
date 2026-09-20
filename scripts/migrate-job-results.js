#!/usr/bin/env node
/**
 * Move job results out of the initiative queue.
 *
 * WHY. Until 2026-08-18 a scheduled job's output was delivered as a `job-result`
 * row in the initiatives table — the bell. It sat there exempted, by name, from
 * every rule that table has: no dedup, no re-scoring, no stale sweep, no cap. A
 * queue whose machinery has to be switched off for one of its types is telling
 * you that type belongs somewhere else, and results now go to the jobs panel,
 * which reads job_runs directly and never opens a conversation.
 *
 * WHAT THIS DOES, and does not do:
 *
 *  - It does NOT copy any text. Every one of these bell items was rendered from
 *    a job_runs row that still holds the same output in output_text, so the
 *    panel already shows it. Copying would create a second copy of one result
 *    and let the copies disagree.
 *  - It does NOT delete. Nothing here is deleted, ever. Each row's status
 *    becomes `relocated`, which keeps it in the bell's History exactly as it
 *    happened, labelled as moved. `dismissed` was the tempting shortcut and it
 *    would have been a lie — it claims Ellie read them.
 *  - Every row is CHECKED before it moves. If the run behind it is missing or
 *    has lost its output, the row is left alone and reported, because the whole
 *    point of the move is that the result is safe somewhere else. A migration
 *    that would drop evidence stops instead.
 *  - It stamps `announced_at` on every pre-existing run. Without that, the first
 *    message Ellie sent after deploying would hand him eight old digests at once
 *    as though they had just landed. The channel starts from now.
 *
 * Backs the database up first, from a READONLY handle via VACUUM INTO, the same
 * way the replay seeds its staging copy.
 *
 * Usage:
 *   node scripts/migrate-job-results.js            # report only, changes nothing
 *   node scripts/migrate-job-results.js --apply
 */

const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

const APPLY = process.argv.includes('--apply');
const ROOT = path.join(__dirname, '..');

const database = require(path.join(ROOT, 'db/database'));
const DATA_DIR = database.getDataDir();
const DB_PATH = path.join(DATA_DIR, 'chat.db');

function main() {
  if (!fs.existsSync(DB_PATH)) {
    console.error(`No database at ${DB_PATH}`);
    process.exit(1);
  }

  // --- 1. Look, from a readonly handle -------------------------------------
  const ro = new Database(DB_PATH, { readonly: true });
  const rows = ro.prepare(`
    SELECT i.id, i.status, i.created_at, i.source_ref, i.content,
           r.id AS run_id, r.status AS run_status, r.output_text
    FROM initiatives i
    LEFT JOIN job_runs r ON r.id = i.source_ref
    WHERE i.type = 'job-result'
    ORDER BY datetime(i.created_at) ASC
  `).all();

  const movable = [];
  const stuck = [];
  for (const row of rows) {
    const safeElsewhere = !!(row.run_id && row.output_text && row.output_text.trim()
      && row.content && row.content.includes(row.output_text.trim().slice(0, 60)));
    (safeElsewhere ? movable : stuck).push(row);
  }

  // The announcement column arrives with the server's own schema migration, on
  // the boot that runs the new code. This script is a DEPLOY step, not a schema
  // step: if the column is not there yet, say which order to do this in rather
  // than quietly creating it — two things creating the same column is how a
  // schema drifts from the one file that is supposed to own it.
  const hasAnnounced = ro.prepare('PRAGMA table_info(job_runs)').all().some(c => c.name === 'announced_at');
  const unannouncedRuns = hasAnnounced
    ? ro.prepare("SELECT COUNT(*) n FROM job_runs WHERE announced_at IS NULL AND status IN ('ok','failed')").get().n
    : null;
  ro.close();

  console.log(`\nJob-result rows in the initiative queue: ${rows.length}`);
  for (const r of rows) {
    const mark = movable.includes(r) ? 'move' : 'HOLD';
    console.log(`  [${mark}] ${r.id.slice(0, 8)} ${r.status.padEnd(9)} ${r.created_at}  run ${r.run_id ? r.run_id.slice(0, 8) : '(missing)'} ${r.run_status || ''}`);
  }
  if (stuck.length) {
    console.log(`\n  ${stuck.length} row(s) HELD: the run behind them is missing or its output no longer matches.`);
    console.log('  They are NOT being moved — the result would only exist in the bell item, and moving it');
    console.log('  out of the queue would be the one thing this migration must never do.');
  }
  if (hasAnnounced) {
    console.log(`\nPre-existing runs never announced to him: ${unannouncedRuns} (these get stamped, so the first`);
    console.log('message after deploy does not hand him a backlog of old digests as though they just landed).');
  } else {
    console.log('\nThe job_runs.announced_at column does not exist yet, which means this database has not been');
    console.log('opened by the new code. Restart the server first (systemctl --user restart snh.service) —');
    console.log('db/database.js adds the column on boot — then run this with --apply.');
  }

  if (APPLY && !hasAnnounced) {
    console.log('\nRefusing to apply against the old schema. Nothing changed.\n');
    process.exit(1);
  }

  if (!APPLY) {
    console.log('\nDry run. Nothing changed. Re-run with --apply to make the move.\n');
    return;
  }

  // --- 2. Back up ----------------------------------------------------------
  const stamp = new Date().toISOString().replace(/[:.]/g, '').replace(/-/g, '').slice(0, 15) + 'Z';
  const backup = path.join(DATA_DIR, `chat.db.bak-jobqueue-${stamp}`);
  const src = new Database(DB_PATH, { readonly: true });
  // Deliberately no wal_checkpoint first: checkpointing is a write, and this
  // handle is readonly on purpose (the replay's rule).
  src.prepare('VACUUM INTO ?').run(backup);
  src.close();
  console.log(`\nBacked up to ${backup}`);

  // --- 3. Move -------------------------------------------------------------
  const db = new Database(DB_PATH);
  const now = new Date().toISOString();
  const relocate = db.prepare("UPDATE initiatives SET status = 'relocated' WHERE id = ? AND status <> 'relocated'");
  let moved = 0;
  const tx = db.transaction(() => {
    for (const r of movable) moved += relocate.run(r.id).changes;
  });
  tx();

  const stamped = db.prepare(
    "UPDATE job_runs SET announced_at = ? WHERE announced_at IS NULL AND status IN ('ok','failed')"
  ).run(now).changes;

  db.close();

  console.log(`Relocated ${moved} bell item(s) out of the pending pool (kept as history, labelled "moved to the jobs panel").`);
  console.log(`Stamped ${stamped} pre-existing run(s) as already announced.`);
  console.log('Their results are unchanged and readable in the jobs panel, which renders them from job_runs.\n');

  // --- 4. Say what the queue looks like now --------------------------------
  const after = new Database(DB_PATH, { readonly: true });
  const pending = after.prepare("SELECT type, COUNT(*) n FROM initiatives WHERE status = 'pending' GROUP BY type").all();
  after.close();
  console.log('Bell queue now pending:');
  if (!pending.length) console.log('  (nothing)');
  for (const p of pending) console.log(`  ${p.type.padEnd(20)} ${p.n}`);
  console.log('');
}

main();
