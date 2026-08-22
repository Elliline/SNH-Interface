/**
 * Database layer for chat history and semantic memory
 * Combines SQLite (structured data) and LanceDB (vector embeddings)
 */

const Database = require('better-sqlite3');
const lancedb = require('vectordb');
const path = require('path');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { getConfig, getProviderInstance } = require('./config');
const { getLocalDateStamp } = require('./datetime');

// Database paths.
//
// SNH_DATA_DIR redirects the WHOLE store — SQLite and the vector index together
// — and exists for one reason: the replay. The spec's rule is that replay runs
// the same code path as live intake, and "if replay needs a special case, the
// pipeline is wrong". A staging flag threaded through fact-store, memory-clusters
// and the extractor would be exactly that special case, in a dozen places, with
// the failure mode that one of them forgets and writes to the live corpus.
// Pointing a whole PROCESS at a different data directory needs no special case
// anywhere: every module below reads the same handle it always did.
//
// The server never sets it. Only scripts/replay-to-staging.js does, and it
// refuses to run against the live directory.
const DATA_DIR = process.env.SNH_DATA_DIR
  ? path.resolve(process.env.SNH_DATA_DIR)
  : path.join(__dirname, '../data');
const SQLITE_PATH = path.join(DATA_DIR, 'chat.db');
const LANCEDB_PATH = path.join(DATA_DIR, 'lancedb');

/** Where this process's store lives. The replay uses it to find its memory dir. */
function getDataDir() { return DATA_DIR; }

/**
 * The memory tree, and its two halves, resolved from THIS process's data dir.
 *
 * These exist so no other module has to write `path.join(__dirname, '../data')`
 * again. Every time one did, it became a path SNH_DATA_DIR could not redirect,
 * and the failure was never loud: a corrector pass against staging wrote its
 * pass-state into live and told the live corrector it had just run; a revert
 * against staging wrote three "is back in memory" notes into the live daily log,
 * which is injected, about facts that had never moved. Both were found by
 * reading output, not by anything failing.
 *
 * Functions rather than constants because DATA_DIR is fixed at load time but
 * these modules are required in any order; a call costs a path join and cannot
 * be stale.
 */
function getMemoryDir() { return path.join(DATA_DIR, 'memory'); }
function getDailyDir() { return path.join(getMemoryDir(), 'daily'); }
function getOpsDir() { return path.join(getMemoryDir(), 'ops'); }

// Ensure data directories exist
function ensureDirectories() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  if (!fs.existsSync(LANCEDB_PATH)) {
    fs.mkdirSync(LANCEDB_PATH, { recursive: true });
  }
}

// Global database instances
let sqliteDb = null;
let vectorDb = null;
let embeddingsTable = null;
let clusterEmbeddingsTable = null;

// ============ SQLite Setup ============

/**
 * Initialize SQLite database with schema
 * @returns {Database} SQLite database instance
 */
function initDatabase() {
  try {
    ensureDirectories();

    // Initialize SQLite
    sqliteDb = new Database(SQLITE_PATH);
    sqliteDb.pragma('journal_mode = WAL'); // Write-Ahead Logging for better concurrency
    sqliteDb.pragma('foreign_keys = ON'); // Enable foreign key constraints

    // Create conversations table
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        model_used TEXT
      )
    `);

    // Create messages table
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model TEXT,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
      )
    `);

    // Create indexes for performance
    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
      ON messages(conversation_id)
    `);

    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
      ON conversations(updated_at DESC)
    `);

    // Migration: sources — provenance for search-backed answers. When the
    // assistant answers after a web search, the source titles+urls it drew from
    // are stored here as JSON, so a later "cite your sources" reads retained links
    // instead of reconstructing them (7/23: it confabulated attributions when
    // asked to cite after the fact). Null for messages with no sources.
    const msgCols = sqliteDb.prepare('PRAGMA table_info(messages)').all();
    if (!msgCols.some(c => c.name === 'sources')) {
      sqliteDb.exec('ALTER TABLE messages ADD COLUMN sources TEXT');
      console.log('Migration: added sources to messages (search provenance)');
    }

    // Migration: input_modality (2026-08-02) — how a message arrived.
    // 'stt' (spoken, transcribed by whisper), 'typed' (keyboard), or 'unknown'.
    //
    // This is upstream of the memory system but the memory system is what needs
    // it: a fact extracted from a transcription is weaker evidence than one Ellie
    // typed, and without this column that distinction cannot be made at all. The
    // 830 pre-existing messages are backfilled to 'unknown' rather than guessed —
    // 'unknown' loses to 'typed' and cannot on its own justify a supersession.
    if (!msgCols.some(c => c.name === 'input_modality')) {
      sqliteDb.exec("ALTER TABLE messages ADD COLUMN input_modality TEXT");
      const n = sqliteDb.prepare(
        "UPDATE messages SET input_modality = 'unknown' WHERE input_modality IS NULL"
      ).run().changes;
      console.log(`Migration: added input_modality to messages (${n} historical messages set to 'unknown')`);
    }

    // Create FTS5 virtual table for full-text search
    sqliteDb.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        conversation_id UNINDEXED,
        message_id UNINDEXED
      )
    `);

    // === UPGRADE 4: Memory Clustering tables ===
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS memory_clusters (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS cluster_members (
        id TEXT PRIMARY KEY,
        cluster_id TEXT NOT NULL,
        content TEXT NOT NULL,
        source TEXT,
        importance REAL DEFAULT 0.5,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'active',
        superseded_by TEXT,
        salience INTEGER DEFAULT 5,
        FOREIGN KEY (cluster_id) REFERENCES memory_clusters(id)
      )
    `);

    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster_id
      ON cluster_members(cluster_id)
    `);

    // cluster_links and cluster_link_judgments were DROPPED at the 2026-08-06
    // cutover. They held a weighted cluster<->cluster graph and the memo that fed
    // it; the writer was disabled on 2026-08-02 and everything after could only
    // delete rows, so what remained described a corpus that no longer existed —
    // 2225 links, 459 of them naming clusters that had been deleted, still drawn
    // on the Map as if current. Association is answered at query time now, from
    // the vector index, for the one cluster a person has selected:
    // GET /api/memory/graph/neighbours/:clusterId. A computed answer cannot go
    // stale. Do not re-create these tables.

    // Question queue: gaps/oddities in the facts become questions the model
    // may ask the user at a natural moment. cluster_id/member_id are the
    // fact/cluster the question is about; reason is why it was raised.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS questions (
        id TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        cluster_id TEXT,
        member_id TEXT,
        reason TEXT,
        status TEXT DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        asked_at DATETIME,
        origin_conversation_id TEXT,
        asked_conversation_id TEXT
      )
    `);

    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status)
    `);

    // Migration: question dedup + answer auditing.
    //  - embedding: cached vector of the question text, so dedup and topic-match
    //    answer-detection don't re-embed every existing question each pass.
    //  - answered_at: audit trail for when a question was retired (was missing,
    //    which made re-ask bugs hard to trace).
    const questionCols = sqliteDb.prepare('PRAGMA table_info(questions)').all();
    if (!questionCols.some(c => c.name === 'embedding')) {
      sqliteDb.exec('ALTER TABLE questions ADD COLUMN embedding TEXT');
      console.log('Migration: added embedding to questions (backfilled lazily on next dedup/answer pass)');
    }
    if (!questionCols.some(c => c.name === 'answered_at')) {
      sqliteDb.exec('ALTER TABLE questions ADD COLUMN answered_at DATETIME');
      console.log('Migration: added answered_at to questions');
    }

    // Migration: ensure facts (cluster_members) carry an updated_at timestamp.
    // Existing rows are backfilled from created_at (the closest recoverable
    // origin date), falling back to now() if created_at is somehow missing.
    const memberCols = sqliteDb.prepare('PRAGMA table_info(cluster_members)').all();
    if (!memberCols.some(c => c.name === 'updated_at')) {
      // SQLite can't add a column with a non-constant DEFAULT, so add it NULL
      // then backfill in a follow-up UPDATE.
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN updated_at DATETIME');
      sqliteDb.prepare(
        "UPDATE cluster_members SET updated_at = COALESCE(created_at, ?) WHERE updated_at IS NULL"
      ).run(new Date().toISOString());
      console.log('Migration: added updated_at to cluster_members and backfilled from created_at');
    }

    // Migration: fact supersession — status (active/superseded) + superseded_by.
    // Existing rows are all active (no contradiction history recorded yet).
    if (!memberCols.some(c => c.name === 'status')) {
      sqliteDb.exec("ALTER TABLE cluster_members ADD COLUMN status TEXT DEFAULT 'active'");
      sqliteDb.exec("UPDATE cluster_members SET status = 'active' WHERE status IS NULL");
      console.log('Migration: added status to cluster_members (all existing rows active)');
    }
    if (!memberCols.some(c => c.name === 'superseded_by')) {
      // Nullable pointer to the fact that replaced this one.
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN superseded_by TEXT');
      console.log('Migration: added superseded_by to cluster_members');
    }

    // Migration: salience (1–10, how much a fact matters). Existing rows
    // backfill to the neutral midpoint 5.
    if (!memberCols.some(c => c.name === 'salience')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN salience INTEGER DEFAULT 5');
      sqliteDb.exec('UPDATE cluster_members SET salience = 5 WHERE salience IS NULL');
      console.log('Migration: added salience to cluster_members (existing rows backfilled to 5)');
    }

    // Migration: subject ('user' | 'self') distinguishes facts ABOUT the user
    // from self-observations SNH makes about itself. Self-facts cluster and
    // inject separately. Existing rows are all facts about the user.
    if (!memberCols.some(c => c.name === 'subject')) {
      sqliteDb.exec("ALTER TABLE cluster_members ADD COLUMN subject TEXT DEFAULT 'user'");
      sqliteDb.exec("UPDATE cluster_members SET subject = 'user' WHERE subject IS NULL");
      console.log("Migration: added subject to cluster_members (existing rows set to 'user')");
    }

    // Migration: claim_type — the auditability tag for self-facts, added for the
    // self-coherence audit (SNH's first accepted initiative, 2026-07-05: stress-
    // testing its own perspectives against contradictions and gaps). Self-facts
    // split into behavioral CLAIMS ("I value precision", "I'm an analytical
    // partner") — testable against how SNH actually behaved, so the audit samples
    // these — and DECLARATIONS (name, preferences, history) — not testable, so
    // never sampled. DISSONANCE rows are the audit's OWN records ("claimed X,
    // behaved Y"); they are self-facts but are excluded from identity injection
    // and never re-sampled. User facts stay NULL (not applicable). Existing self-
    // facts stay NULL until the one-time classification pass tags them.
    if (!memberCols.some(c => c.name === 'claim_type')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN claim_type TEXT');
      console.log('Migration: added claim_type to cluster_members (self-fact auditability tag)');
    }

    // Migration: identity lock — the flag that makes a fact untouchable by every
    // AUTOMATIC path (contradiction judge, write_memory, passive extraction,
    // reflection). Added 2026-07-28, the day after the entity chose its own name:
    // a single sentence in chat ("your name is Bob") could route through the
    // contradiction judge and supersede that choice, and nothing in the system
    // treated a chosen thing differently from an observed one.
    //
    // The claim/declaration split already drew the right line — a DECLARATION is
    // something the entity chose, a CLAIM is something observed about it, and the
    // self-coherence audit only ever samples claims. This is that distinction
    // given teeth for the narrow set of declarations that are identity-critical.
    //
    //  locked        - 1 = no automatic path may supersede, retire, or reword it
    //  locked_at     - when the lock was set
    //  lock_category - which identity slot it holds ('name' | 'pronouns'). A
    //                  category with a locked fact also REFUSES new competing
    //                  facts, so the lock can't be walked around by appending a
    //                  second name instead of replacing the first.
    if (!memberCols.some(c => c.name === 'locked')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN locked INTEGER DEFAULT 0');
      sqliteDb.exec('UPDATE cluster_members SET locked = 0 WHERE locked IS NULL');
      console.log('Migration: added locked to cluster_members (identity lock; existing rows unlocked)');
    }
    if (!memberCols.some(c => c.name === 'locked_at')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN locked_at DATETIME');
      console.log('Migration: added locked_at to cluster_members');
    }
    if (!memberCols.some(c => c.name === 'lock_category')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN lock_category TEXT');
      console.log('Migration: added lock_category to cluster_members (which identity slot a locked fact holds)');
    }

    // Migration: PROVENANCE (2026-08-02). Until now a fact carried no trace of
    // where it came from — only a coarse `source` string like 'fact-extraction'.
    // That made two things impossible: explaining a fact ("when did I tell you
    // that?") and adjudicating a correction, because the evidence rules the
    // corrector needs (typed beats stt, direct beats inferred) have nothing to
    // read. The canonical case is "User's name is Mike", extracted at salience 10
    // from an STT mishearing of "mic" — indistinguishable, without provenance,
    // from a fact Ellie actually typed.
    //
    // All five are NULLABLE and stay null on the 658 pre-existing rows. Every
    // NEW fact populates them; nothing backfills, because the information does
    // not exist for old rows and a guessed provenance is worse than an absent one.
    //
    //  conversation_id       - which conversation the fact came from
    //  message_id            - which message in it
    //  verbatim_source_text  - what was actually said, unparaphrased
    //  input_modality        - 'stt' | 'typed' | 'unknown' — how it arrived
    //  salience_rationale    - why it scored the salience it did (was logged as
    //                          prose to the daily log and then discarded)
    for (const [col, type, note] of [
      ['conversation_id', 'TEXT', 'which conversation a fact came from'],
      ['message_id', 'TEXT', 'which message a fact came from'],
      ['verbatim_source_text', 'TEXT', 'what was actually said'],
      ['input_modality', 'TEXT', "how it arrived ('stt' | 'typed' | 'unknown')"],
      ['salience_rationale', 'TEXT', 'why this fact scored the salience it did']
    ]) {
      if (!memberCols.some(c => c.name === col)) {
        sqliteDb.exec(`ALTER TABLE cluster_members ADD COLUMN ${col} ${type}`);
        console.log(`Migration: added ${col} to cluster_members (${note})`);
      }
    }

    // Corroborations (2026-08-03, Phase 2a). When the user says a thing she has
    // already said, the REPEAT rule folds it into the fact that already holds it
    // rather than creating a second row — but "she has told me this three times"
    // is itself evidence, and the corrector's evidence-dominance bar reads
    // "corroborated > single mention". Bumping salience alone would record the
    // conclusion and throw away the reason.
    //
    // One row per re-assertion, carrying the same provenance shape a fact does,
    // so EXPLAIN can show every occasion a fact was stated, not just the first.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS fact_corroborations (
        id TEXT PRIMARY KEY,
        member_id TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        conversation_id TEXT,
        message_id TEXT,
        verbatim_source_text TEXT,
        input_modality TEXT,
        restated_as TEXT,
        similarity REAL,
        detected_by TEXT,
        FOREIGN KEY (member_id) REFERENCES cluster_members(id)
      )
    `);
    sqliteDb.exec(
      'CREATE INDEX IF NOT EXISTS idx_fact_corroborations_member ON fact_corroborations(member_id)'
    );

    // Corrections ledger (2026-08-03, Phase 2c). Every action the corrector
    // takes, mechanical and semantic alike.
    //
    // The mechanical tier is specified as "autonomous and SILENT", and silent is
    // about not interrupting anyone — it is not about leaving no trace. An
    // unrecorded automatic edit to the substrate an identity is built from is
    // indistinguishable from corruption after the fact, so everything lands here
    // whether or not it is announced.
    //
    // `revert_of` / `reverted_at` make the semantic tier's reversibility a
    // property of the record rather than a promise: the entry carries what it
    // takes to undo it (which fact to restore, which successor to unset), and
    // scripts/revert-correction.js reads exactly that.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS corrections_ledger (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        pass_id TEXT,                  -- groups one corrector pass
        tier TEXT NOT NULL,            -- mechanical | semantic
        action TEXT NOT NULL,          -- merge | expire | supersede | split | reconcile
        subject TEXT,                  -- user | self
        target_id TEXT,                -- the fact acted ON (the loser / the expired / the split original)
        target_text TEXT,
        survivor_id TEXT,              -- the fact that won, when there is one
        survivor_text TEXT,
        reason TEXT NOT NULL,          -- plain language: why this happened
        evidence TEXT,                 -- JSON: the dominance signals that decided it
        reversible INTEGER DEFAULT 1,
        reverted_at DATETIME,
        reverted_by TEXT,
        announced INTEGER DEFAULT 0
      )
    `);
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_corrections_pass ON corrections_ledger(pass_id)');
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_corrections_target ON corrections_ledger(target_id)');

    // Pairs the corrector has already put to the contradiction judge.
    //
    // Without this the semantic tier is not actually resumable. It is budgeted
    // per pass, and its enumeration is stable — so every pass re-judged the same
    // first N pairs, spent the whole budget on them, and never advanced. The
    // corpus has ~227 active user facts and roughly nine candidates each, which
    // is around two thousand pairs; a pass that always starts from the beginning
    // reaches the first three facts forever. Recording what has been checked is
    // what turns "bounded" into "bounded and progressing".
    //
    // fact_updated_at is stored so a pair can be re-judged if either side later
    // changes, rather than being written off permanently on one verdict.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS corrector_pair_checks (
        pair_key TEXT PRIMARY KEY,
        checked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        verdict TEXT,
        a_id TEXT,
        b_id TEXT,
        fact_updated_at TEXT
      )
    `);

    // Heartbeat anomalies the ops log has already reported.
    //
    // Same idea as corrector_pair_checks, for the same reason: a pass that
    // re-states an unchanged condition is not telemetry, it is wallpaper. The
    // cluster audit emitted the SAME 29 anomaly lines every two hours from
    // 2026-08-07 to 2026-08-10 — 350 lines of ops log in which nothing had
    // happened — and real entries (the corrector's) were buried under them.
    //
    // So an anomaly is reported in full the first time, then counted. The ops
    // log gets the new ones plus one line saying how many old ones are still
    // true. `first_seen_at` is what makes the count meaningful: "unchanged
    // since Friday" is a different fact from "started this pass".
    //
    // Rows are pruned once an anomaly has gone quiet for a while (see
    // ANOMALY_STATE_TTL_DAYS), so a condition that clears and later comes back
    // is reported fresh rather than silently folded into a months-old count.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS heartbeat_anomaly_state (
        anomaly_key TEXT PRIMARY KEY,
        first_seen_at DATETIME,
        last_seen_at DATETIME,
        seen_count INTEGER DEFAULT 1,
        anomaly_text TEXT
      )
    `);
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_anomaly_state_last_seen ON heartbeat_anomaly_state(last_seen_at)');

    // Correction notices — the private channel to Aurelius (decision 6).
    //
    // Deliberately NOT rows in `initiatives`. Every property these need is a way
    // of not participating in the initiative machinery — exempt from caps, from
    // freshness scoring, from dedup, undroppable — and the audience is different:
    // initiatives are surfaced to Ellie in the bell, these are for him and nobody
    // else. Sharing the table would mean every Ellie-facing selector needed a
    // filter, and the failure mode of missing one is a private notice about his
    // own memory appearing in her notification panel.
    //
    // `seen_at` is set when the notice is actually injected into his context, so
    // an unread one persists across restarts and sessions until he has had it.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS correction_notices (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        ledger_id TEXT,
        content TEXT NOT NULL,
        seen_at DATETIME,
        seen_conversation_id TEXT,
        is_test INTEGER DEFAULT 0
      )
    `);
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_correction_notices_seen ON correction_notices(seen_at)');

    // Migration: member_id (2026-08-12) — WHICH self-fact a notice is about.
    //
    // Added when notices moved from the corrector to the fact-store funnel, so
    // every non-conversational change to a self-fact raises one whatever
    // pipeline made it. Two emitters can now describe the same change — the
    // funnel generically, the corrector with its dominance reasoning — and two
    // notices about one change reads as two changes. This column is what lets
    // addNotice fold them into one instead.
    const noticeCols = sqliteDb.prepare('PRAGMA table_info(correction_notices)').all();
    if (!noticeCols.some(c => c.name === 'member_id')) {
      sqliteDb.exec('ALTER TABLE correction_notices ADD COLUMN member_id TEXT');
      console.log('Migration: added member_id to correction_notices (which fact a notice is about)');
    }

    // Migration: LIFECYCLE (2026-08-02). Replaces the active/superseded/retired
    // triple with status + inactive_reason + successor_id, so "why is this fact
    // not live" is answerable from one column instead of inferred from which of
    // three status strings it holds.
    //
    //  status          - 'active' | 'inactive'
    //  inactive_reason - 'superseded' | 'expired' | 'retracted' (null when active)
    //  successor_id    - the fact that replaced this one; REQUIRED when the
    //                    reason is 'superseded', null otherwise
    //
    // superseded_by is kept and mirrored into successor_id rather than dropped:
    // the Memory Map's supersede edges read it, and losing the pointer would lose
    // provenance (decision 3 — supersession history is never discarded).
    if (!memberCols.some(c => c.name === 'inactive_reason')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN inactive_reason TEXT');
      console.log('Migration: added inactive_reason to cluster_members');
    }
    if (!memberCols.some(c => c.name === 'successor_id')) {
      sqliteDb.exec('ALTER TABLE cluster_members ADD COLUMN successor_id TEXT');
      console.log('Migration: added successor_id to cluster_members');
    }
    // One-time value migration, idempotent: any row still carrying an old status
    // string is folded into the new two-column form.
    const legacy = sqliteDb.prepare(
      "SELECT COUNT(*) AS n FROM cluster_members WHERE status IN ('superseded','retired')"
    ).get().n;
    if (legacy > 0) {
      const supersededN = sqliteDb.prepare(
        "UPDATE cluster_members SET status='inactive', inactive_reason='superseded', successor_id=superseded_by WHERE status='superseded'"
      ).run().changes;
      const retiredN = sqliteDb.prepare(
        "UPDATE cluster_members SET status='inactive', inactive_reason='retracted' WHERE status='retired'"
      ).run().changes;
      console.log(`Migration: lifecycle — ${supersededN} superseded → inactive/superseded, ${retiredN} retired → inactive/retracted`);
    }

    // Self-coherence audit records (2026-08-02). These used to be stored as
    // cluster_members — multi-kilobyte narratives ("I claimed X, the evidence
    // shows Y") with an inline dump of 30+ truncated message openings, sitting in
    // the fact corpus at salience 3 with claim_type='dissonance'.
    //
    // They were excluded from identity injection, but they were still facts
    // everywhere else: clustered, embedded, and handed to the contradiction judge
    // as candidates — a 3KB blob costing a model call to compare against a
    // one-line observation. They are audit output, not things the entity believes
    // about itself, so they live in their own table now (decision 4). A later
    // curation pass with Aurelius may distill one-line self-facts from them; the
    // narratives themselves never return to the corpus.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS self_audit_records (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        claim_text TEXT,
        claim_date TEXT,
        finding TEXT,
        evidence TEXT,
        content TEXT NOT NULL,
        origin_member_id TEXT
      )
    `);
    sqliteDb.exec(`
      CREATE INDEX IF NOT EXISTS idx_self_audit_source ON self_audit_records(source)
    `);

    // One-time move of any dissonance rows still living in cluster_members.
    // Idempotent: rows already moved are gone from cluster_members, so a re-run
    // finds nothing. The cluster_members row is DELETED rather than marked
    // inactive — it was never a fact, so leaving a ghost would misrepresent the
    // history. The full narrative is preserved verbatim in the new table, and
    // origin_member_id keeps the old id so vectors can be cleaned up and any
    // outside reference still resolves.
    const strayDissonance = sqliteDb.prepare(
      "SELECT id, content, source, created_at FROM cluster_members WHERE claim_type = 'dissonance'"
    ).all();
    if (strayDissonance.length > 0) {
      const ins = sqliteDb.prepare(`
        INSERT INTO self_audit_records (id, created_at, source, claim_text, finding, evidence, content, origin_member_id)
        VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)
      `);
      const del = sqliteDb.prepare('DELETE FROM cluster_members WHERE id = ?');
      const move = sqliteDb.transaction((rows) => {
        for (const r of rows) {
          ins.run(randomUUID(), r.created_at, r.source, r.content, r.id);
          del.run(r.id);
        }
      });
      move(strayDissonance);
      console.log(`Migration: moved ${strayDissonance.length} self-audit dissonance record(s) out of cluster_members into self_audit_records`);
      console.log('  (their embeddings are cleared on the next heartbeat reconciliation — they now read as orphan vectors)');
    }

    const clusterCols = sqliteDb.prepare('PRAGMA table_info(memory_clusters)').all();
    if (!clusterCols.some(c => c.name === 'subject')) {
      sqliteDb.exec("ALTER TABLE memory_clusters ADD COLUMN subject TEXT DEFAULT 'user'");
      sqliteDb.exec("UPDATE memory_clusters SET subject = 'user' WHERE subject IS NULL");
      console.log("Migration: added subject to memory_clusters (existing clusters set to 'user')");
    }

    // Initiative layer: things SNH notices and may raise unprompted. Candidates
    // are written by heartbeat tasks, re-scored by a prioritizer, and delivered
    // via a conversation-open greeting or an unprompted SNH-initiated message.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS initiatives (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,                -- question | observation | alert | reflection-insight | followup
        content TEXT NOT NULL,
        source_kind TEXT,                  -- question | fact | cluster | reflection
        source_ref TEXT,                   -- id of the source (question/fact/cluster)
        priority INTEGER DEFAULT 5,        -- 1..10
        status TEXT DEFAULT 'pending',     -- pending | delivered | dismissed | expired
        channel TEXT,                      -- greeting | unprompted | panel | nudge (set on delivery)
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        delivered_at DATETIME,
        delivered_conversation_id TEXT
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_initiatives_status ON initiatives(status)`);

    // Conversation-followup traces: every reflection cycle records what it
    // reviewed, which older memory clusters it pulled in, the follow-up
    // candidates it weighed, and what it generated or skipped (with reasoning).
    // Stored queryably so the UI can surface the reasoning behind each follow-up.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS followup_traces (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        conversations_reviewed INTEGER DEFAULT 0,  -- count of conversations reviewed
        message_count INTEGER DEFAULT 0,
        reviewed_json TEXT,                 -- JSON [{id,title,messageCount}]
        related_clusters_json TEXT,         -- JSON [{name,members:[...]}] pulled by similarity
        candidates_json TEXT,               -- JSON [string] follow-up candidates weighed
        generated TEXT,                     -- the follow-up message produced, or NULL
        skipped INTEGER DEFAULT 1,          -- 1 = no follow-up produced this cycle
        reasoning TEXT,                     -- why it generated or skipped
        initiative_id TEXT                  -- the queued initiative id, if any
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_followup_traces_created ON followup_traces(created_at)`);

    // Daily-log follow-up traces. The sibling of followup_traces, for the source
    // that reads the day's LOG rather than the conversation transcript.
    //
    // A separate table rather than a `source` column on followup_traces because
    // the two describe different things: that one records conversations and
    // message counts, this one records log entries and the window they were
    // drawn from. Folding them together would leave half the columns null on
    // every row and make neither readable.
    //
    // EVERY PASS WRITES A ROW, INCLUDING THE ONES THAT RAISE NOTHING. Choosing
    // not to ask is the common outcome and it is a decision, not an absence —
    // without a row, "it read the log and judged nothing worth raising" and "it
    // never ran" are the same silence, and that is not a thing anyone should
    // have to debug. `skipped` and `reasoning` are how they are told apart.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS log_followup_traces (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        window_days INTEGER DEFAULT 0,      -- staleness window used for this pass
        files_read TEXT,                    -- JSON [string] daily files actually present
        entries_considered INTEGER DEFAULT 0, -- event entries in the window
        entries_json TEXT,                  -- JSON [{id,date,time,text}] what it looked at
        candidates_json TEXT,               -- JSON [string] follow-ups weighed
        generated TEXT,                     -- the follow-up raised, or NULL
        source_entry_id TEXT,               -- the log entry it followed up on, if any
        skipped INTEGER DEFAULT 1,          -- 1 = raised nothing this pass
        reasoning TEXT,                     -- why it raised, or why it declined
        initiative_id TEXT                  -- the queued initiative id, if any
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_log_followup_traces_created ON log_followup_traces(created_at)`);

    // Heartbeat pass stats: one row per maintenance/rebuild cycle so the Thinking
    // view can show what each background pass actually did (clusters audited/split,
    // links touched, duration, anomalies) alongside the reflection reasoning.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS heartbeat_reports (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        clusters_audited INTEGER DEFAULT 0,
        clusters_split INTEGER DEFAULT 0,
        links_added INTEGER DEFAULT 0,
        links_updated INTEGER DEFAULT 0,
        links_removed INTEGER DEFAULT 0,
        duration TEXT,
        anomaly_count INTEGER DEFAULT 0,
        report_json TEXT                    -- full report object (splits, timing, anomalies)
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_heartbeat_reports_created ON heartbeat_reports(created_at)`);

    // Migration: heartbeat outcome. The table used to be written ONLY by
    // generateReport, which runs only on the success path — so during a brain
    // outage the record looked clean while nothing was actually completing.
    // Aborted/failed passes now write a row too, and `status` says which.
    // Existing rows are backfilled to 'ok' because that is what they were: a
    // row existing at all used to mean the pass reached the end.
    const hbCols = sqliteDb.prepare('PRAGMA table_info(heartbeat_reports)').all();
    if (!hbCols.some(c => c.name === 'status')) {
      sqliteDb.exec("ALTER TABLE heartbeat_reports ADD COLUMN status TEXT DEFAULT 'ok'");
      sqliteDb.exec("UPDATE heartbeat_reports SET status = 'ok' WHERE status IS NULL");
      console.log("Migration: added status to heartbeat_reports (existing rows set to 'ok')");
    }
    if (!hbCols.some(c => c.name === 'status_reason')) {
      sqliteDb.exec('ALTER TABLE heartbeat_reports ADD COLUMN status_reason TEXT');
      console.log('Migration: added status_reason to heartbeat_reports');
    }
    if (!hbCols.some(c => c.name === 'duration_ms')) {
      // duration was only ever stored as a display string ("1062.2s"), which the
      // Activity view cannot sort or trend on. Numeric ms alongside it.
      sqliteDb.exec('ALTER TABLE heartbeat_reports ADD COLUMN duration_ms INTEGER');
      console.log('Migration: added duration_ms to heartbeat_reports');
    }

    // Liveness probe results — one row per probe, every few minutes. Previously
    // the probe only wrote to the ops log on a STATE TRANSITION (ok→fail,
    // fail→ok), so a probe that ran and passed left no trace at all and "when
    // did it last run" was unanswerable. Pruned on a retention window
    // (config.livenessProbe.retentionDays) so it cannot grow without bound.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS liveness_probes (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        ok INTEGER NOT NULL,        -- 1 = brain answered, 0 = did not
        latency_ms INTEGER,         -- round-trip for the probe completion
        error TEXT                  -- failure reason when ok = 0
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_liveness_probes_created ON liveness_probes(created_at)`);

    // Cron jobs proposed by the entity through the create_cron_job tool. PROPOSE
    // ONLY: a tool call lands here as status='proposed' and raises an initiative;
    // nothing becomes 'approved' without Ellie deciding in the bell panel.
    //
    // `source` is the provenance tag, mirroring cluster_members.source — every
    // row the entity proposed carries 'kid-proposed', so kid-created jobs can be
    // listed and reverted in bulk without guessing.
    //
    // APPROVING NOW ARMS IT (2026-08-12). Until then this note said the opposite
    // — "approving records the job; nothing executes it" — and it was true for
    // six days after Ellie approved one. db/scheduler.js reads approved+enabled
    // rows and runs them; approve() computes the first next_run_at. A row that is
    // proposed, rejected, reverted or disabled is still inert, and next_run_at is
    // NULL for exactly those.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS cron_jobs (
        id TEXT PRIMARY KEY,
        schedule TEXT NOT NULL,             -- 5-field cron expression as proposed
        description TEXT NOT NULL,
        enabled INTEGER DEFAULT 0,          -- the 'enabled' the entity asked for
        source TEXT DEFAULT 'kid-proposed', -- provenance (mirrors cluster_members.source)
        status TEXT DEFAULT 'proposed',     -- proposed | approved | rejected | reverted
        initiative_id TEXT,                 -- the bell item raised for this proposal
        conversation_id TEXT,               -- where the entity proposed it
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        decided_at DATETIME,
        decided_note TEXT
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_cron_jobs_status ON cron_jobs(status)`);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_cron_jobs_source ON cron_jobs(source)`);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_cron_jobs_created ON cron_jobs(created_at)`);

    // Migration: THE SCHEDULER (2026-08-12). The note above stood for six days
    // longer than it should have — an approved job was a row nothing read.
    // db/scheduler.js now reads it, so these columns hold the state that makes a
    // schedule answerable without guessing from the cron expression:
    //
    //  next_run_at          - the computed local firing this job is waiting for.
    //                         Set at approval and after every run. NULL means the
    //                         job is not armed (proposed, rejected, or disabled),
    //                         which is a different thing from "no schedule".
    //  last_run_at          - when it last actually ran (not when it was due)
    //  last_status          - ok | failed | skipped, mirroring the newest job_runs row
    //  run_count            - how many times it has run, ever
    //  consecutive_failures - reset to 0 by any successful run
    //  disabled_reason      - why the scheduler turned `enabled` off by itself.
    //                         Non-null is the difference between a job Ellie left
    //                         disabled and one that disabled itself after failing.
    const cronCols = sqliteDb.prepare('PRAGMA table_info(cron_jobs)').all();
    for (const [col, type, note] of [
      ['next_run_at', 'DATETIME', 'the next firing this job is armed for'],
      ['last_run_at', 'DATETIME', 'when it last actually ran'],
      ['last_status', 'TEXT', 'outcome of the most recent run'],
      ['run_count', 'INTEGER DEFAULT 0', 'how many times it has run, ever'],
      ['consecutive_failures', 'INTEGER DEFAULT 0', 'reset by any successful run'],
      ['disabled_reason', 'TEXT', 'why the scheduler disabled it by itself']
    ]) {
      if (!cronCols.some(c => c.name === col)) {
        sqliteDb.exec(`ALTER TABLE cron_jobs ADD COLUMN ${col} ${type}`);
        console.log(`Migration: added ${col} to cron_jobs (${note})`);
      }
    }

    // Every attempt to run a job, and what came of it.
    //
    // Written for the attempts that did NOT execute as much as the ones that
    // did. A scheduler whose log holds only successes cannot answer the question
    // people actually ask it — "why didn't it run this morning" — and the honest
    // answers to that are all statuses of their own:
    //
    //  ok       - it ran and produced output
    //  failed   - it ran and threw, or produced nothing. error holds why.
    //  skipped  - deliberately not run: too late to catch up after a restart, or
    //             interrupted by one. Not a failure, and not counted as one.
    //  deferred - it came due while another job was still running. The next tick
    //             picks it up; the row exists so the delay is visible.
    //
    // scheduled_for is the firing this row is about, which is not started_at when
    // a run is late — that gap is the whole story of a catch-up. output_initiative_id
    // points at the bell item carrying the result, so "it ran" and "she was told"
    // are separately checkable.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS job_runs (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        scheduled_for DATETIME,
        started_at DATETIME,
        finished_at DATETIME,
        status TEXT NOT NULL,               -- ok | failed | skipped | deferred
        duration_ms INTEGER,
        trigger TEXT,                       -- schedule | catchup | manual
        output_initiative_id TEXT,          -- the bell item carrying the result
        output_text TEXT,
        tool_calls INTEGER DEFAULT 0,
        budget_json TEXT,                   -- the tool budget as it stood at the end
        error TEXT,
        FOREIGN KEY (job_id) REFERENCES cron_jobs(id)
      )
    `);
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_job_runs_job ON job_runs(job_id)');
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_job_runs_started ON job_runs(started_at)');

    // Panel state for a scheduled run. The jobs panel shows scheduled results
    // beside handed-off ones, so it needs the same two stamps agent_jobs carries
    // — and they are two different readers, which is why they are two columns:
    //   seen_at      - ELLIE has looked at it in the jobs panel
    //   announced_at - HE has been told about it at the top of a chat turn
    // Neither implies the other. A result she read on her phone is still news to
    // him, and one he led with is not thereby marked read for her.
    const runCols = sqliteDb.prepare('PRAGMA table_info(job_runs)').all();
    for (const [col, type, note] of [
      ['seen_at', 'DATETIME', 'when Ellie read it in the jobs panel'],
      ['announced_at', 'DATETIME', 'when it was handed to the entity in a chat turn']
    ]) {
      if (!runCols.some(c => c.name === col)) {
        sqliteDb.exec(`ALTER TABLE job_runs ADD COLUMN ${col} ${type}`);
        console.log(`Migration: added ${col} to job_runs (${note})`);
      }
    }

    // ============ The agent-job queue — ROBOT, not BELL ============
    //
    // Work the entity starts mid-conversation and does NOT finish in the turn.
    // The tool call writes a row and returns; the run happens on the agent pool,
    // outside the request; the turn ends normally.
    //
    // ⚠ THIS TABLE IS NOT THE INITIATIVE TABLE, AND THE DIFFERENCE IS THE POINT.
    //
    //   ROBOT (here)      = results. A record of work that ran. It NEVER opens a
    //                       conversation. It is a panel Ellie reads when she is
    //                       ready, and nothing in this table's write path calls
    //                       into db/initiatives.js — not "must not", DOES NOT.
    //   BELL (initiatives)= things the entity WANTS TO SAY. That channel can
    //                       still open a conversation, unchanged.
    //
    // A finding can LEAD TO a bell item — by him deciding, in an ordinary turn,
    // that it is worth raising, subject to the same judgement as anything else
    // he might say. Job completion is the most mechanical trigger there is, and
    // giving it a channel of its own would let it bypass that judgement.
    //
    // STATUSES. Every one of them is visible in the panel; the whole reason this
    // table exists rather than a promise held in memory is that a job must never
    // be able to vanish:
    //   queued      - written, not yet started. Survives a restart (re-enqueued).
    //   running     - on the pool. The one transient status: it never survives a
    //                 process, either finish() replaces it or the startup sweep
    //                 does (see db/agent-jobs.js sweepInterrupted).
    //   ok          - it ran and produced text.
    //   failed      - it threw, timed out, or produced nothing. `error` says which.
    //                 An empty answer is a failure with a cause, not a silence.
    //   interrupted - a restart killed it mid-run. Re-queued once inside the
    //                 grace window; past that it stays here, with the reason.
    //   cancelled   - Ellie cancelled it. Only reachable while queued: an
    //                 in-flight LLM call cannot be cleanly cancelled, and a
    //                 button that claims otherwise is a lie.
    //
    // source: 'chat-handoff' today. The column exists so a later starter (a
    // heartbeat step, a scheduled job spawning one) is a value rather than a
    // schema change — it does not mean any other starter exists yet.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS agent_jobs (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,                -- short label, his words, for the panel
        task TEXT NOT NULL,                 -- the instruction the run follows; it IS the prompt
        why TEXT,                           -- one line: why it was worth handing off
        status TEXT NOT NULL DEFAULT 'queued',
        source TEXT DEFAULT 'chat-handoff',
        conversation_id TEXT,               -- where he started it
        message_id TEXT,                    -- the message that asked
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        started_at DATETIME,
        finished_at DATETIME,
        duration_ms INTEGER,
        result_text TEXT,
        error TEXT,
        tool_calls INTEGER DEFAULT 0,
        budget_json TEXT,                   -- the tool budget as it stood at the end
        attempts INTEGER DEFAULT 0,         -- runs begun, including ones a restart killed
        seen_at DATETIME,                   -- Ellie read it in the jobs panel
        announced_at DATETIME               -- he was told about it in a chat turn
      )
    `);
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_agent_jobs_status ON agent_jobs(status)');
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_agent_jobs_created ON agent_jobs(created_at)');
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_agent_jobs_finished ON agent_jobs(finished_at)');

    // A record of coding work sent to squatch-code. Written at DISPATCH,
    // not before it: the approval happened in conversation, so there is no
    // pending state to hold. Each row has a matching agent_jobs row with
    // source 'squatch-code', which is what actually runs — so the result
    // lands in the same panel as every other job rather than in a second
    // one that could disagree with the first.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS coding_jobs (
        id TEXT PRIMARY KEY,
        project TEXT NOT NULL,             -- name under Projects/
        brief TEXT NOT NULL,               -- exactly what was sent
        status TEXT NOT NULL DEFAULT 'dispatched',
        agent_job_id TEXT,                 -- the run
        conversation_id TEXT,              -- where she gave the go-ahead
        message_id TEXT,                   -- the message that gave it
        -- How closely the sent brief matched what she was actually shown.
        -- 1.0 with match_exact means she read these words; anything less
        -- is a paraphrase, dispatched but recorded as one.
        match_ratio REAL,
        match_exact INTEGER DEFAULT 0,
        -- Where the running job writes its live position. Set when the
        -- process starts, so a status line can be rendered while the job
        -- is still working rather than only from the report afterwards.
        progress_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);
    sqliteDb.exec('CREATE INDEX IF NOT EXISTS idx_coding_jobs_status ON coding_jobs(status)');

    // Migration: coding_jobs rows written before live progress existed
    // have no place to look for it.
    const codingCols = sqliteDb.prepare('PRAGMA table_info(coding_jobs)').all();
    if (!codingCols.some(c => c.name === 'progress_path')) {
      sqliteDb.exec('ALTER TABLE coding_jobs ADD COLUMN progress_path TEXT');
      console.log('Migration: added progress_path to coding_jobs (live status line)');
    }

    // ============ What the run PRODUCED, as a file ============
    //
    // A result used to be one column: result_text, shown whole on the card. A
    // report and a Python module were the same shape as "nothing new since
    // Monday", which is how a 4,000-word report came to be rendered as raw
    // markdown in a narrow panel.
    //
    // These five columns say what form the result took. They are NOT a second
    // copy of the result — result_text stays the truth and the file is made
    // from it, so a lost or deleted file costs the formatting and never the
    // work.
    //
    //   artifact_kind  - pdf | code | text | null. Null means it stayed on the
    //                    card, which is the right answer for a short result and
    //                    is not a failure.
    //   artifact_path  - absolute path on the server. The download route reads
    //                    the file by JOB ID and gets the path from here, so no
    //                    client-supplied path is ever opened.
    //   artifact_name  - the basename, which is what the card shows and what the
    //                    browser saves it as.
    //   artifact_bytes - size on disk, for the card. A 0-byte file that reads as
    //                    finished is the failure this makes visible.
    //   artifact_error - why there is no file, or why it is a text file and not
    //                    a PDF. Written even on success paths that DOWNGRADED,
    //                    because "chromium is not installed" must reach her as a
    //                    sentence on the card rather than as a missing feature
    //                    she is left to notice.
    //
    //   summary_text   - a few lines for the card, so a document is announced
    //                    rather than pasted. Null for a result that stayed
    //                    inline, where the text IS the summary.
    const jobCols = sqliteDb.prepare('PRAGMA table_info(agent_jobs)').all();
    for (const [col, type, note] of [
      ['artifact_kind', 'TEXT', 'pdf | code | text, or null when it stayed on the card'],
      ['artifact_path', 'TEXT', 'absolute path to the file this run produced'],
      ['artifact_name', 'TEXT', 'basename, for the card and the download'],
      ['artifact_bytes', 'INTEGER', 'size on disk'],
      ['artifact_error', 'TEXT', 'why there is no file, or why it was downgraded'],
      ['summary_text', 'TEXT', 'the few lines the card shows instead of the document']
    ]) {
      if (!jobCols.some(c => c.name === col)) {
        sqliteDb.exec(`ALTER TABLE agent_jobs ADD COLUMN ${col} ${type}`);
        console.log(`Migration: added ${col} to agent_jobs (${note})`);
      }
    }

    // Every tool call the entity makes and what came of it. Operational telemetry
    // → surfaced in the Thinking tab, never injected into chat. Rejected/capped
    // calls are logged too: a tool call that was refused is exactly the thing
    // worth being able to look back at.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS tool_call_log (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        tool TEXT NOT NULL,
        args_json TEXT,                     -- arguments as the model emitted them
        outcome TEXT NOT NULL,              -- proposed | rejected-cap | error | approved | rejected
        detail TEXT,                        -- human-readable one-liner
        ref_id TEXT,                        -- e.g. the cron_jobs.id this produced
        conversation_id TEXT
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_tool_call_log_created ON tool_call_log(created_at)`);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_tool_call_log_tool ON tool_call_log(tool)`);

    // Every SEARCH provider attempt — see db/search-log.js for why this is its
    // own table rather than a `tool` value in tool_call_log. One tool call can
    // try Exa and then SearXNG, and the two attempts have different outcomes;
    // squeezing both into one row would lose exactly the fact worth keeping.
    sqliteDb.exec(`
      CREATE TABLE IF NOT EXISTS search_call_log (
        id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        provider TEXT NOT NULL,             -- exa | searxng
        query TEXT NOT NULL,
        num_results INTEGER NOT NULL DEFAULT 0,
        outcome TEXT NOT NULL,              -- results | empty | error | skipped
        detail TEXT,                        -- the error, or why it was skipped
        caller TEXT,                        -- chat | agent-job:3f9a1c2b | heartbeat:corrector
        latency_ms INTEGER,
        attempt_id TEXT,                    -- groups the providers tried for ONE tool call
        served INTEGER NOT NULL DEFAULT 0,  -- 1 for the attempt whose results were used
        cost_usd REAL                       -- when the provider reports one
      )
    `);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_search_call_log_created ON search_call_log(created_at)`);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_search_call_log_provider ON search_call_log(provider)`);
    sqliteDb.exec(`CREATE INDEX IF NOT EXISTS idx_search_call_log_attempt ON search_call_log(attempt_id)`);

    // Conversations may be started by SNH itself (unprompted initiatives).
    const convCols = sqliteDb.prepare('PRAGMA table_info(conversations)').all();
    if (!convCols.some(c => c.name === 'initiated_by')) {
      sqliteDb.exec("ALTER TABLE conversations ADD COLUMN initiated_by TEXT DEFAULT 'user'");
      sqliteDb.exec("UPDATE conversations SET initiated_by = 'user' WHERE initiated_by IS NULL");
      console.log("Migration: added initiated_by to conversations (existing rows set to 'user')");
    }

    console.log('SQLite database initialized successfully');

    // Backfill FTS table with existing messages
    backfillFTS();

    return sqliteDb;
  } catch (error) {
    console.error('Failed to initialize SQLite database:', error.message);
    throw error;
  }
}

/**
 * Get list of all conversations with preview
 * @returns {Array} Array of conversation objects
 */
function getConversations() {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare(`
      SELECT
        c.id,
        c.title,
        c.created_at,
        c.updated_at,
        c.model_used,
        c.initiated_by,
        (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY timestamp ASC LIMIT 1) as preview
      FROM conversations c
      ORDER BY c.updated_at DESC
    `);

    return stmt.all();
  } catch (error) {
    console.error('Error fetching conversations:', error.message);
    throw error;
  }
}

/** Parse a message's stored sources JSON to an array (null/[] safe). */
function parseSourcesJson(json) {
  if (!json) return null;
  try {
    const arr = JSON.parse(json);
    return Array.isArray(arr) && arr.length ? arr : null;
  } catch { return null; }
}

/**
 * The saved sources from the most recent search-backed assistant messages in a
 * conversation, newest first. Used to give the entity the links it found earlier
 * so a follow-up "give me the link for [S3]" reads the stored URL. Returns an
 * array of source-arrays (one per message).
 * @param {string} conversationId
 * @param {number} [limit=1] - how many recent source-bearing messages to return
 * @returns {Array<Array<{n:number,title:string,url:string}>>}
 */
function getRecentSources(conversationId, limit = 1) {
  try {
    if (!sqliteDb) return [];
    const rows = sqliteDb.prepare(`
      SELECT sources FROM messages
      WHERE conversation_id = ? AND role = 'assistant' AND sources IS NOT NULL
      ORDER BY timestamp DESC
      LIMIT ?
    `).all(conversationId, limit);
    return rows.map(r => parseSourcesJson(r.sources)).filter(Boolean);
  } catch (error) {
    console.error('Error fetching recent sources:', error.message);
    return [];
  }
}

/**
 * Get full conversation with all messages
 * @param {string} id - Conversation ID
 * @returns {Object|null} Conversation object with messages array
 */
function getConversation(id) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    // Get conversation metadata
    const conversationStmt = sqliteDb.prepare(`
      SELECT id, title, created_at, updated_at, model_used, initiated_by
      FROM conversations
      WHERE id = ?
    `);
    const conversation = conversationStmt.get(id);

    if (!conversation) {
      return null;
    }

    // Get all messages for this conversation. `sources` (search provenance) is
    // parsed from JSON so the UI can render the clickable [S#] link list.
    const messagesStmt = sqliteDb.prepare(`
      SELECT id, conversation_id, role, content, timestamp, model, sources
      FROM messages
      WHERE conversation_id = ?
      ORDER BY timestamp ASC
    `);
    const messages = messagesStmt.all(id).map(m => ({
      ...m,
      sources: parseSourcesJson(m.sources)
    }));

    return {
      ...conversation,
      messages
    };
  } catch (error) {
    console.error('Error fetching conversation:', error.message);
    throw error;
  }
}

/**
 * Create a new conversation
 * @param {string} title - Conversation title
 * @param {string} model_used - Model identifier
 * @returns {string} New conversation ID
 */
function createConversation(title, model_used, initiatedBy = 'user') {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const id = randomUUID();
    const stmt = sqliteDb.prepare(`
      INSERT INTO conversations (id, title, model_used, initiated_by)
      VALUES (?, ?, ?, ?)
    `);

    stmt.run(id, title, model_used, initiatedBy);
    return id;
  } catch (error) {
    console.error('Error creating conversation:', error.message);
    throw error;
  }
}

/**
 * Delete a conversation and all its messages
 * @param {string} id - Conversation ID
 */
function deleteConversation(id) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare('DELETE FROM conversations WHERE id = ?');
    stmt.run(id);

    // Also delete vector embeddings
    deleteConversationEmbeddings(id).catch(err => {
      console.error('Error deleting embeddings:', err.message);
    });
  } catch (error) {
    console.error('Error deleting conversation:', error.message);
    throw error;
  }
}

/**
 * Update conversation title
 * @param {string} id - Conversation ID
 * @param {string} title - New title
 */
function updateConversationTitle(id, title) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare(`
      UPDATE conversations
      SET title = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `);

    stmt.run(title, id);
  } catch (error) {
    console.error('Error updating conversation title:', error.message);
    throw error;
  }
}

/**
 * Add a message to a conversation
 * @param {string} conversation_id - Conversation ID
 * @param {string} role - Message role (user/assistant/system)
 * @param {string} content - Message content
 * @param {string} model - Model used (optional)
 * @returns {string} New message ID
 */
function addMessage(conversation_id, role, content, model = null, sources = null, inputModality = null) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const id = randomUUID();
    // sources: array of {n,title,url} the answer drew from, stored as JSON so
    // "cite your sources" later reads retained links. Null when nothing was searched.
    const sourcesJson = (Array.isArray(sources) && sources.length) ? JSON.stringify(sources) : null;
    // Only a user message has a meaningful modality — an assistant message was
    // not "input" at all. Anything unrecognised records as 'unknown' rather than
    // defaulting to 'typed', so a missing client flag never masquerades as
    // stronger evidence than it is.
    const modality = role === 'user'
      ? (['stt', 'typed'].includes(inputModality) ? inputModality : 'unknown')
      : null;
    const stmt = sqliteDb.prepare(`
      INSERT INTO messages (id, conversation_id, role, content, model, sources, input_modality)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(id, conversation_id, role, content, model, sourcesJson, modality);

    // Also insert into FTS5 table for full-text search
    try {
      const ftsStmt = sqliteDb.prepare(`
        INSERT INTO messages_fts (content, conversation_id, message_id)
        VALUES (?, ?, ?)
      `);
      ftsStmt.run(content, conversation_id, id);
    } catch (ftsError) {
      console.error('[FTS] Error adding message to FTS:', ftsError.message);
    }

    // Update conversation's updated_at timestamp
    updateConversationTimestamp(conversation_id);

    return id;
  } catch (error) {
    console.error('Error adding message:', error.message);
    throw error;
  }
}

/**
 * Update conversation's updated_at timestamp
 * @param {string} id - Conversation ID
 */
function updateConversationTimestamp(id) {
  try {
    if (!sqliteDb) {
      throw new Error('Database not initialized. Call initDatabase() first.');
    }

    const stmt = sqliteDb.prepare(`
      UPDATE conversations
      SET updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `);

    stmt.run(id);
  } catch (error) {
    console.error('Error updating conversation timestamp:', error.message);
    throw error;
  }
}

// ============ LanceDB (Vector Store) Setup ============

/**
 * Initialize LanceDB for vector embeddings
 * @returns {Promise<Object>} LanceDB table instance
 */
async function initVectorStore() {
  try {
    ensureDirectories();

    // Connect to LanceDB
    vectorDb = await lancedb.connect(LANCEDB_PATH);

    // Check if table exists
    const tableNames = await vectorDb.tableNames();

    if (tableNames.includes('message_embeddings')) {
      // Open existing table
      embeddingsTable = await vectorDb.openTable('message_embeddings');
      console.log('LanceDB: Opened existing message_embeddings table');
    } else {
      // Create new table with schema
      // Note: We need at least one record to create the table with schema
      // Use regular array for LanceDB compatibility
      const sampleData = [{
        id: randomUUID(),
        message_id: 'sample',
        conversation_id: 'sample',
        text: 'Sample initialization text',
        role: 'system',
        vector: Array(768).fill(0) // nomic-embed-text produces 768-dim vectors
      }];

      embeddingsTable = await vectorDb.createTable('message_embeddings', sampleData);

      // Delete the sample record
      await embeddingsTable.delete('id = "' + sampleData[0].id + '"');

      console.log('LanceDB: Created new message_embeddings table');
    }

    // === UPGRADE 4: Cluster embeddings table ===
    if (tableNames.includes('cluster_embeddings')) {
      clusterEmbeddingsTable = await vectorDb.openTable('cluster_embeddings');
      console.log('LanceDB: Opened existing cluster_embeddings table');
    } else {
      const sampleCluster = [{
        id: randomUUID(),
        member_id: 'sample',
        cluster_id: 'sample',
        content: 'Sample cluster text',
        vector: Array(768).fill(0)
      }];
      clusterEmbeddingsTable = await vectorDb.createTable('cluster_embeddings', sampleCluster);
      await clusterEmbeddingsTable.delete('id = "' + sampleCluster[0].id + '"');
      console.log('LanceDB: Created new cluster_embeddings table');
    }

    return embeddingsTable;
  } catch (error) {
    console.error('Failed to initialize LanceDB:', error.message);
    throw error;
  }
}

/**
 * Generate embedding using Ollama's nomic-embed-text model
 * @param {string} text - Text to embed
 * @returns {Promise<Float32Array>} Embedding vector
 */
async function generateEmbedding(text) {
  try {
    const config = getConfig();
    const embInst = getProviderInstance(config.models.embedding.provider, config.models.embedding.instance);
    const embeddingHost = embInst ? embInst.host : 'http://localhost:11434';
    const embeddingModel = config.models.embedding.model;
    const response = await fetch(`${embeddingHost}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: embeddingModel,
        prompt: text
      })
    });

    if (!response.ok) {
      throw new Error(`Embedding API returned ${response.status}`);
    }

    const data = await response.json();

    if (!data.embedding || !Array.isArray(data.embedding)) {
      throw new Error('Invalid embedding response format');
    }

    return new Float32Array(data.embedding);
  } catch (error) {
    console.error('Error generating embedding:', error.message);
    throw error;
  }
}

/**
 * Add embedding to vector store
 * @param {string} message_id - Message ID from SQLite
 * @param {string} conversation_id - Conversation ID
 * @param {string} text - Message text
 * @param {string} role - Message role
 * @param {Float32Array} vector - Embedding vector
 * @returns {Promise<void>}
 */
async function addEmbedding(message_id, conversation_id, text, role, vector) {
  try {
    if (!embeddingsTable) {
      throw new Error('Vector store not initialized. Call initVectorStore() first.');
    }

    // Convert Float32Array to regular array for LanceDB compatibility
    const vectorArray = Array.from(vector);

    const record = {
      id: randomUUID(),
      message_id,
      conversation_id,
      text,
      role,
      vector: vectorArray
    };

    await embeddingsTable.add([record]);
  } catch (error) {
    console.error('Error adding embedding:', error.message);
    throw error;
  }
}

/**
 * Search for similar messages across conversations
 * @param {Float32Array} vector - Query vector
 * @param {string} excludeConversationId - Conversation ID to exclude from results
 * @param {number} limit - Maximum number of results
 * @param {number} threshold - Similarity threshold (0-1, higher is more similar)
 * @returns {Promise<Array>} Array of similar messages
 */
async function searchSimilar(vector, excludeConversationId, limit = 5, threshold) {
  if (threshold === undefined) {
    threshold = getConfig().memory.similarityThreshold;
  }
  try {
    if (!embeddingsTable) {
      throw new Error('Vector store not initialized. Call initVectorStore() first.');
    }

    // Convert Float32Array to regular array for LanceDB compatibility
    const vectorArray = Array.from(vector);

    // Perform vector search with cosine distance metric
    // Cosine distance returns values 0-1 where 0 = identical, 1 = opposite
    const results = await embeddingsTable
      .search(vectorArray)
      .metricType('cosine')
      .limit(limit * 2) // Get more results to filter
      .execute();

    // Filter out current conversation and apply threshold
    const filtered = results
      .filter(result => {
        // Exclude current conversation
        if (result.conversation_id === excludeConversationId) {
          return false;
        }
        // Apply similarity threshold
        // Convert cosine distance to similarity: similarity = 1 - distance
        const similarity = 1 - result._distance;
        return similarity >= threshold;
      })
      .slice(0, limit)
      .map(result => ({
        message_id: result.message_id,
        conversation_id: result.conversation_id,
        text: result.text,
        role: result.role,
        similarity: 1 - result._distance
      }));

    return filtered;
  } catch (error) {
    console.error('Error searching similar messages:', error.message);
    // Return empty array on error to not break the application
    return [];
  }
}

/**
 * Delete all embeddings for a conversation
 * @param {string} conversation_id - Conversation ID
 * @returns {Promise<void>}
 */
async function deleteConversationEmbeddings(conversation_id) {
  try {
    if (!embeddingsTable) {
      // If table not initialized, silently skip
      return;
    }

    await embeddingsTable.delete(`conversation_id = "${conversation_id}"`);
  } catch (error) {
    console.error('Error deleting conversation embeddings:', error.message);
    // Don't throw - embedding deletion is not critical
  }
}

// ============ Full-Text Search (FTS5) Functions ============

/**
 * Backfill FTS table with existing messages
 * Called during database initialization
 */
function backfillFTS() {
  try {
    if (!sqliteDb) {
      console.error('[FTS] Cannot backfill: database not initialized');
      return;
    }

    // Count rows in each table
    const ftsCount = sqliteDb.prepare('SELECT COUNT(*) as count FROM messages_fts').get().count;
    const messagesCount = sqliteDb.prepare('SELECT COUNT(*) as count FROM messages').get().count;

    console.log(`[FTS] Current state: ${ftsCount} FTS rows, ${messagesCount} total messages`);

    if (ftsCount >= messagesCount) {
      console.log('[FTS] FTS table is up to date, no backfill needed');
      return;
    }

    // Find messages not yet in FTS
    const missingMessages = sqliteDb.prepare(`
      SELECT m.id, m.conversation_id, m.content
      FROM messages m
      WHERE m.id NOT IN (SELECT message_id FROM messages_fts)
    `).all();

    if (missingMessages.length === 0) {
      console.log('[FTS] No missing messages to backfill');
      return;
    }

    console.log(`[FTS] Backfilling ${missingMessages.length} messages into FTS table`);

    const insertStmt = sqliteDb.prepare(`
      INSERT INTO messages_fts (content, conversation_id, message_id)
      VALUES (?, ?, ?)
    `);

    const insertMany = sqliteDb.transaction((messages) => {
      for (const msg of messages) {
        insertStmt.run(msg.content, msg.conversation_id, msg.id);
      }
    });

    insertMany(missingMessages);

    console.log(`[FTS] Successfully backfilled ${missingMessages.length} messages`);
  } catch (error) {
    console.error('[FTS] Error during backfill:', error.message);
    // Don't throw - backfill failure should not prevent app from starting
  }
}

/**
 * Sanitize FTS5 query to prevent MATCH syntax errors and column-name injection.
 * Wraps each term in double quotes so FTS5 treats them as literal strings
 * instead of interpreting them as column names or operators.
 * @param {string} query - Raw query string
 * @returns {string} Sanitized query safe for FTS5 MATCH
 */
function sanitizeFTSQuery(query) {
  if (!query || typeof query !== 'string') {
    return '"search"';
  }

  // Strip everything except word chars, spaces, and hyphens
  const cleaned = query.replace(/[^\w\s\-]/g, ' ');

  // Split into individual terms, quote each one for FTS5 safety
  const terms = cleaned
    .split(/\s+/)
    .filter(w => w.length > 0)
    .map(w => '"' + w.replace(/"/g, '') + '"');

  if (terms.length === 0) {
    return '"search"';
  }

  return terms.join(' ');
}

/**
 * Search messages using BM25 ranking (FTS5)
 * @param {string} query - Search query
 * @param {string} excludeConversationId - Conversation ID to exclude from results
 * @param {number} limit - Maximum number of results
 * @returns {Array} Array of search results with BM25 scores
 */
function searchBM25(query, excludeConversationId, limit = 10) {
  try {
    if (!sqliteDb) {
      console.error('[BM25] Database not initialized');
      return [];
    }

    const sanitized = sanitizeFTSQuery(query);
    console.log(`[BM25] Searching for: "${sanitized}" (original: "${query}")`);

    const stmt = sqliteDb.prepare(`
      SELECT
        message_id,
        conversation_id,
        content,
        bm25(messages_fts) as score
      FROM messages_fts
      WHERE messages_fts MATCH ? AND conversation_id != ?
      ORDER BY bm25(messages_fts)
      LIMIT ?
    `);

    const results = stmt.all(sanitized, excludeConversationId, limit);
    console.log(`[BM25] Found ${results.length} results`);

    // BM25 scores are negative (more negative = better match)
    // Normalize to 0-1 range
    if (results.length === 0) {
      return [];
    }

    // Find the best (most negative) score
    const maxAbsScore = Math.max(...results.map(r => Math.abs(r.score)));

    return results.map(result => ({
      message_id: result.message_id,
      conversation_id: result.conversation_id,
      content: result.content,
      score: maxAbsScore > 0 ? Math.abs(result.score) / maxAbsScore : 1.0
    }));
  } catch (error) {
    console.error('[BM25] Search error:', error.message);
    return [];
  }
}

/**
 * Hybrid search combining vector similarity and BM25 keyword search
 * @param {string} query - Search query
 * @param {string} excludeConversationId - Conversation ID to exclude from results
 * @param {number} limit - Maximum number of results to return
 * @param {number} threshold - Similarity threshold for vector search (0-1)
 * @returns {Promise<Array>} Array of search results with combined scores
 */
async function hybridSearch(query, excludeConversationId, limit = 5, threshold) {
  if (threshold === undefined) {
    threshold = getConfig().memory.similarityThreshold;
  }
  try {
    console.log(`[HybridSearch] Query: "${query}", limit: ${limit}, threshold: ${threshold}`);

    // Step 1: Generate embedding for query
    let vectorResults = [];
    try {
      const embedding = await generateEmbedding(query);
      console.log('[HybridSearch] Generated query embedding');

      // Step 2: Run vector search
      vectorResults = await searchSimilar(embedding, excludeConversationId, limit * 2, threshold);
      console.log(`[HybridSearch] Vector search found ${vectorResults.length} results`);
    } catch (embeddingError) {
      console.error('[HybridSearch] Vector search failed:', embeddingError.message);
      // Continue with BM25 only
    }

    // Step 3: Run BM25 search
    const bm25Results = searchBM25(query, excludeConversationId, limit * 2);
    console.log(`[HybridSearch] BM25 search found ${bm25Results.length} results`);

    // Step 4: Fuse results by message_id
    const resultsMap = new Map();

    // Add vector results
    for (const result of vectorResults) {
      resultsMap.set(result.message_id || result.text, {
        text: result.text,
        role: result.role,
        conversation_id: result.conversation_id,
        message_id: result.message_id,
        vectorScore: result.similarity,
        bm25Score: 0
      });
    }

    // Merge BM25 results
    for (const result of bm25Results) {
      const key = result.message_id || result.content;
      if (resultsMap.has(key)) {
        // Result appears in both searches
        resultsMap.get(key).bm25Score = result.score;
      } else {
        // BM25-only result - get full message details from SQLite
        try {
          const msgStmt = sqliteDb.prepare(`
            SELECT id, conversation_id, role, content
            FROM messages
            WHERE id = ?
          `);
          const msg = msgStmt.get(result.message_id);

          if (msg) {
            resultsMap.set(key, {
              text: msg.content,
              role: msg.role,
              conversation_id: msg.conversation_id,
              message_id: msg.id,
              vectorScore: 0,
              bm25Score: result.score
            });
          }
        } catch (dbError) {
          console.error('[HybridSearch] Error fetching message details:', dbError.message);
        }
      }
    }

    // Step 5: Calculate combined scores and sort
    const config = getConfig();
    const weightVector = config.memory.hybridSearchWeights.vector;
    const weightBM25 = config.memory.hybridSearchWeights.bm25;

    const combinedResults = Array.from(resultsMap.values()).map(result => {
      const finalScore = (weightVector * result.vectorScore) + (weightBM25 * result.bm25Score);

      // Determine source
      let source = 'hybrid';
      if (result.vectorScore > 0 && result.bm25Score === 0) {
        source = 'vector';
      } else if (result.bm25Score > 0 && result.vectorScore === 0) {
        source = 'bm25';
      }

      return {
        text: result.text,
        role: result.role,
        conversation_id: result.conversation_id,
        similarity: finalScore,
        source,
        // Include component scores for debugging
        _vectorScore: result.vectorScore,
        _bm25Score: result.bm25Score
      };
    });

    // Sort by combined score descending
    combinedResults.sort((a, b) => b.similarity - a.similarity);

    // Return top N results
    const topResults = combinedResults.slice(0, limit);
    console.log(`[HybridSearch] Returning ${topResults.length} results (${topResults.filter(r => r.source === 'hybrid').length} hybrid, ${topResults.filter(r => r.source === 'vector').length} vector-only, ${topResults.filter(r => r.source === 'bm25').length} bm25-only)`);

    return topResults;
  } catch (error) {
    console.error('[HybridSearch] Error:', error.message);
    return [];
  }
}

// ============ Memory Files Functions ============

/**
 * Load memory files for system prompt injection
 * @param {string} memoryDir - Directory containing memory files
 * @returns {Object} Object containing memory file contents
 */
function loadMemoryFiles(memoryDir = getMemoryDir()) {
  try {
    console.log(`[MemoryFiles] Loading from: ${memoryDir}`);

    const result = {
      memory: null,
      user: null,
      dailyToday: null,
      dailyYesterday: null
    };

    // Helper to safely read a file
    const readFileSafe = (filePath, label) => {
      try {
        if (fs.existsSync(filePath)) {
          const content = fs.readFileSync(filePath, 'utf-8');
          console.log(`[MemoryFiles] Loaded ${label}: ${content.length} chars`);
          return content;
        } else {
          console.log(`[MemoryFiles] ${label} not found: ${filePath}`);
          return null;
        }
      } catch (error) {
        console.error(`[MemoryFiles] Error reading ${label}:`, error.message);
        return null;
      }
    };

    // MEMORY.md is NOT read here. Long-term memory is rendered from SQLite by
    // memoryClusters.renderLongTermMemory(); the caller fills result.memory in.
    // Reading it from a file is what let the injected block drift from the
    // database — see docs/memory-mvp-spec.md (decision 1).

    // Load USER.md
    result.user = readFileSafe(path.join(memoryDir, 'USER.md'), 'USER.md');

    // Load today's daily note (bucketed by local Pacific date, matching writers)
    const todayFilename = `${getLocalDateStamp()}.md`;
    result.dailyToday = readFileSafe(path.join(memoryDir, 'daily', todayFilename), `daily/${todayFilename}`);

    // Load yesterday's daily note
    const yesterdayFilename = `${getLocalDateStamp(new Date(Date.now() - 86400000))}.md`;
    result.dailyYesterday = readFileSafe(path.join(memoryDir, 'daily', yesterdayFilename), `daily/${yesterdayFilename}`);

    return result;
  } catch (error) {
    console.error('[MemoryFiles] Error loading memory files:', error.message);
    return {
      memory: null,
      user: null,
      dailyToday: null,
      dailyYesterday: null
    };
  }
}

// ============ Accessors for sub-modules ============

function getSqliteDb() { return sqliteDb; }
function getClusterEmbeddingsTable() { return clusterEmbeddingsTable; }

/**
 * Re-open cluster_embeddings and swap in the fresh handle.
 *
 * WHY THIS EXISTS. A LanceDB table handle pins the dataset version it was opened
 * at. The server opens this one once at boot and holds it for its whole life, so
 * the moment ANY other process commits — a CLI script like
 * scripts/introduce-capability.js or scripts/identity-lock.js, a repair script,
 * a test — the server's handle is behind, and every write it attempts is
 * rejected with "Commit conflict … Please rerun the operation off the latest
 * version of the table."
 *
 * The consequence was measured twice (13 stale vectors on 2026-07-27, 3 more on
 * 2026-07-28): fact-store's delete retried four times, each attempt re-running
 * off the SAME pinned version, so all four failed identically. Retiring a fact
 * left its embedding live and semantically retrievable — the store that
 * actually reaches answers, since retrieval matches on similarity and never
 * consults status.
 *
 * A handle only advances its own read view when IT writes, so refreshing cannot
 * be left to chance: it has to be explicit, and it has to happen between retries
 * or the retry is inert.
 *
 * @returns {Promise<Object|null>} the refreshed table, or null if unavailable
 */
async function reopenClusterEmbeddingsTable() {
  if (!vectorDb) return clusterEmbeddingsTable;
  try {
    clusterEmbeddingsTable = await vectorDb.openTable('cluster_embeddings');
    return clusterEmbeddingsTable;
  } catch (err) {
    console.error('[LanceDB] Could not re-open cluster_embeddings:', err.message);
    return clusterEmbeddingsTable;   // keep the old handle rather than losing it
  }
}

/**
 * Drop and recreate the cluster_embeddings LanceDB table
 * Used by rebuild scripts to get a clean table with a fresh index
 * @returns {Promise<Object>} New cluster embeddings table
 */
async function resetClusterEmbeddingsTable() {
  if (!vectorDb) {
    throw new Error('Vector store not initialized. Call initVectorStore() first.');
  }

  // Drop existing table
  const tableNames = await vectorDb.tableNames();
  if (tableNames.includes('cluster_embeddings')) {
    await vectorDb.dropTable('cluster_embeddings');
    console.log('LanceDB: Dropped cluster_embeddings table');
  }

  // Recreate with schema
  const sampleCluster = [{
    id: randomUUID(),
    member_id: 'sample',
    cluster_id: 'sample',
    content: 'Sample cluster text',
    vector: Array(768).fill(0)
  }];
  clusterEmbeddingsTable = await vectorDb.createTable('cluster_embeddings', sampleCluster);
  await clusterEmbeddingsTable.delete('id = "' + sampleCluster[0].id + '"');
  console.log('LanceDB: Recreated cluster_embeddings table');

  return clusterEmbeddingsTable;
}

// ============ Exports ============

module.exports = {
  // Initialization
  initDatabase,
  initVectorStore,
  getDataDir,
  getMemoryDir,
  getDailyDir,
  getOpsDir,

  // SQLite functions
  getConversations,
  getConversation,
  getRecentSources,
  createConversation,
  deleteConversation,
  updateConversationTitle,
  addMessage,
  updateConversationTimestamp,

  // Vector store functions
  generateEmbedding,
  addEmbedding,
  searchSimilar,
  deleteConversationEmbeddings,

  // Full-text search functions
  searchBM25,
  hybridSearch,
  backfillFTS,

  // Memory files
  loadMemoryFiles,

  // Accessors for sub-modules
  getSqliteDb,
  getClusterEmbeddingsTable,
  reopenClusterEmbeddingsTable,
  resetClusterEmbeddingsTable
};
