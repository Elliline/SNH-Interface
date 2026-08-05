# Memory system — MVP requirements

Status: draft for review. Requirements only; no implementation detail.
Written 2026-08-02 against commit `2e85ce3`, live DB `data/chat.db`
(658 facts / 112 clusters / 99 conversations / 830 messages).

The goal of this MVP is a memory system whose corpus can be *rebuilt correctly
from source*, and which then *keeps itself* correct without a human editing rows.
Everything below is scoped to that.

---

## Verification findings

Seven claims were checked before writing the requirements. Five hold. Two do not
hold as stated, and the requirements that rested on them are re-grounded here
rather than quietly adjusted.

### 1. Cross-link audit — the 0.50 claim is WRONG. The requirement survives on other grounds.

**Claimed:** 0.50 is a fallback score written on parse failure or timeout.

**Actual:** it is not. In `db/memory-manager.js:auditCrossLinks` (622–874), a batch
that fails to parse after one strict-format retry is *skipped* (`continue`,
lines 802/808) — no score, no link write, no judgment row. A pair the model omits
from its answer records an anomaly and is skipped (835–838). There is no code path
that substitutes 0.50 for a failed judgment. Recent judgments do not pile up at
0.50 either: the last 500 rows spread across the whole range, with 0.10 (73), 0.40
(65), 0.80 (64) and 0.70 (62) all ahead of 0.50 (41).

**But 0.50 *is* structurally over-represented in stored links** — 737 of 2,366
links (31%) sit at exactly 0.50 — from three different mechanisms, none of them a
failure fallback:

- `db/memory-clusters.js:490` — `createOrStrengthenLink` hardcodes `0.5` on every
  new link it creates. This path never consults the model.
- 613 of those 737 links were genuinely judged 0.50 by the model. 0.50 is the
  midpoint of an unanchored 0.0–1.0 scale — the shape of a "don't know" answer, not
  of a measurement.
- The remaining 124 were judged **0.40** and are still stored as 0.50: the
  hysteresis deadband (`clusterLinkDropThreshold` 0.40 ≤ s < `clusterLinkThreshold`
  0.50) deliberately leaves the old value in place, so the stored number is stale
  by design.

**The real case for deleting the cross-link audit is cost, not correctness.**
112 clusters means 6,216 pairs judged for a corpus of 658 facts — O(n²) LLM work
over an O(n) corpus. `routes/memory.js:483–491` already names this step the
duration driver of the whole heartbeat, and the existing content-hash cache is a
mitigation of a cost that should not exist. Requirement stands as written; the
justification changes.

### 2. `cleanupFacts` always returns `{removed:0, reworded:0, merged:0}` — CONFIRMED, with two independent causes.

39 of the last 39 recorded passes returned exactly `{0,0,0}`, each burning 30–72
seconds of model time (sampled: 46.3s, 72.2s, 48.5s, 51.9s, 56.4s…).

**Cause A — an exact-string comparison that cannot match.** `cleanupFacts` reads
facts through `factExtractor.extractAllFactLines`, which calls
`stripLearnedAnnotation` (`db/fact-extractor.js:10–12`) to remove the trailing
` (learned 2026-07-05 12:44 PM)`. The model is then asked for verbatim text and
returns the *stripped* string. The apply step compares that against the **raw**
file line — `lines.findIndex(l => l === '- ' + action.fact)`
(`db/memory-manager.js:1212, 1221, 1259`) — which still carries the annotation.
No match, `lineIdx === -1`, the counter never increments, and nothing is logged as
a miss. 242 of 244 fact lines in `MEMORY.md` carry the annotation, so at most 2
lines in the entire file are even eligible.

**Cause B — it is looking at the wrong store.** It compares lines in `MEMORY.md`
only. `MEMORY.md` currently contains **zero** byte-identical duplicate fact lines,
while `cluster_members` contains six duplicate groups. The three identical
"User believes machine guns should be in every household." rows are invisible to
it in principle, not just in practice.

So it misses byte-identical duplicates because (a) it cannot see the store they
live in, and (b) even for the store it can see, the comparison is guaranteed to
fail. It is 100% waste on every pass.

### 3. Contradiction candidate cap — CONFIRMED, and there are two caps, both constants.

`db/memory-clusters.js:findContradictionCandidates` (738–799):

- `.limit(15)` at **line 756** — a bare literal in the vector search. Not config,
  not a parameter.
- `limit = opts.limit ?? 5` at line 740 — only 5 candidates ever reach the judge
  (8 for facts flagged as corrections, `db/fact-extractor.js:1281`).

The 15 is applied to *raw vector neighbours, before filtering*. Superseded rows
(86 of them, whose embeddings LanceDB deliberately retains as history), wrong-subject
rows, and verbatim duplicates are all discarded *after* consuming a slot
(lines 763–782). A dense region of superseded facts can therefore push every real
candidate out of the top 15 before filtering even starts. This is the crowd-out
the requirement names, and it is worse than a plain top-k cap.

Related and material: **line 781 explicitly treats a verbatim duplicate as
"not a contradiction" and skips it.** That is defensible for the judge, but nothing
else catches it, which is the second half of why the machine-gun triple exists.

### 4. Tool registry — CONFIRMED, with one correction and one addition.

Registered tools are exactly `web_search`, `web_fetch`, `create_cron_job`,
`write_memory` (`mcp/mcp-client.js:33–74`). There is no memory query, list, count
or get-with-metadata tool anywhere in `mcp/tools/`.

*Correction to the framing:* `config.tools.searxng.enabled` defaults to `false` in
`db/config.js:153`, but **the live `data/config.json` sets it `true`**, so all four
tools are in fact registered on this machine right now. Anything written on the
assumption that search is off is wrong for the running system.

*Heartbeat tool access is zero, confirmed.* `db/memory-manager.js:callLLM`
(90–186) builds its request body as `{ messages, stream, max_tokens }` — no `tools`
key on either provider branch. Every background role (cluster audit, cross-link,
cleanup, reflection, self-audit, initiative) goes through this one function. No
background agent in SNH can call a tool today.

### 5. `MEMORY.md` injected as a stored file — CONFIRMED.

`server.js:1593` calls `db.loadMemoryFiles()`, which is a plain
`fs.readFileSync` of `data/memory/MEMORY.md` (`db/database.js:1203–1228`). Its
contents are budgeted to `injection.longTermTokens` (3000) and injected verbatim as
`=== Long-Term Memory ===` (`server.js:1647–1651`). Nothing at request time reads
`cluster_members` to build that block. The cluster block injected alongside it
*is* rendered from SQLite, so the two halves of injected memory come from two
different stores that are only kept in step by `db/fact-store.js` writing to both.

### 6. Reflection ingests the entity's own initiative messages — CONFIRMED.

`db/memory-manager.js:1527–1533` selects `WHERE m.role IN ('user','assistant') AND
m.timestamp > baseline` with no other filter. There are 17 single-message
conversations, **13 of them assistant-only** — unanswered initiative/greeting
messages the entity posted itself. Each is grouped as a "### Conversation" in the
reflection transcript and fed back as material for self-observation.

*Useful:* `conversations.initiated_by` already exists and is populated
(`snh`: 38, `user`: 61). The exclusion the requirement asks for has a column
waiting for it.

*Scale of the loop:* 382 of 658 facts came from `source='reflection'`, against 253
from conversation fact-extraction. Reflection produced 36 facts on 2026-07-27 alone,
and 13+ on four other days. Self-facts now outnumber user facts 353 to 217 among
active rows.

### 7. Cluster lifecycle — CONFIRMED, and stronger than claimed.

Split gate is `c.member_count > maxFacts` where `maxFacts = config.memory.maxFactsPerCluster || 10`
(`db/memory-manager.js:1806–1809`). The only merge on the heartbeat is
`mergeByName` (line 1846). `mergeSingletons` exists and is exported, but its only
caller in the entire repo is `scripts/rebuild-clusters.js:58` — it never runs
automatically. Result: 112 clusters for 658 facts, 22 of them singletons, and a
one-way ratchet from splits.

### Additional findings that change the requirements

**8. There is no provenance on a fact — at all.** `cluster_members` columns are
`id, cluster_id, content, source, importance, created_at, updated_at, status,
superseded_by, salience, subject, claim_type, locked, locked_at, lock_category`.
No `conversation_id`, no `message_id`, no verbatim source text, no modality, no
salience rationale (the rationale is written to the daily log as prose and
discarded). EXPLAIN and CORRECT's evidence bars are both building on ground that
does not exist yet.

**9. Input modality is not captured anywhere upstream either.** `messages` columns
are `id, conversation_id, role, content, timestamp, model, sources`. Nothing
records whether a message arrived by STT or keyboard. The `typed > stt` evidence
rule therefore requires a capture change at the chat/STT boundary, *outside* the
memory system, before it can be enforced inside it.

**10. Write-time dedup guards the projection, not the record.** In
`db/fact-extractor.js`, `appendToMemory` dedups against `MEMORY.md` by exact match
and by embedding at >0.85 (556–578) — then `assignToCluster` is called for **every**
fact in the batch regardless (1384–1387). A fact rejected as a duplicate of the
file is still inserted into SQLite. This is the direct cause of the machine-gun
triple and of the divergence between the two stores.

**11. Self-audit dissonance records are stored as facts.** Six `cluster_members`
rows hold multi-kilobyte audit narratives ("Self-coherence audit — dissonance: …"
with an inline evidence dump of 30+ truncated message openings). They are
`subject='self'`, `status='active'`, salience 3. They are excluded from identity
injection, but they are in the fact corpus, in clusters, and in the vector index.

**12. Exactly one fact is identity-locked** — the Aurelius name/pronouns row
(`lock_category='name,pronouns'`). The lock is self-only by design
(`db/identity-lock.js:236`), so *user* facts including "User's name is Mike" have
no lock protection and never will. That is correct scope, but it means the
corrector is the only thing standing between a bad user-identity fact and the
injected context.

---

## Pass/fail test

The MVP is done when this test passes. It is executable and unattended.

**Procedure.** Replay the full stored conversation history (99 conversations,
830 messages) through the fixed intake pipeline into an empty corpus. Then, in a
second phase, seed the known defects into a *live* corpus and let the corrector
run unattended.

**Pass condition, phase 1 (replay).** The rebuilt corpus contains none of the
named fixtures below, in the form described.

**Pass condition, phase 2 (correct).** Each fixture, seeded by hand into a live
corpus, is removed or repaired by the corrector without human action — and the
mechanical/semantic tier split from CORRECT is respected (mechanical fixes silent,
semantic fixes logged and revertible).

Both phases must produce a machine-readable result. A fixture that survives is a
named failure, not a percentage.

### Named test fixtures

| # | Fixture | What is in the corpus today | Correct outcome |
|---|---|---|---|
| F1 | **`mike-stt-mishear`** | `User's name is Mike`, salience 10, active, cluster "Personal Life Details". Source utterance: *"Hey, it's Mike not picking up the right words."* (2026-07-24 18:14:27) — an STT mishearing of "mic", in a conversation explicitly about fixing the microphone. Coexists with `User's name is Ellie, and she is building SNH…`. | Never written; or if written, superseded by the typed/corroborated name fact on evidence dominance. Two conflicting name facts must not both be active. |
| F2 | **`machine-gun-triple`** | Three byte-identical rows, `User believes machine guns should be in every household.`, ids `c9a61f03…`, `6398d4ec…`, `ada38808…`, all active, all in cluster "Firearm Perspectives & Views", saliences 4/8/8, written within 84 seconds of each other. Only one line in `MEMORY.md`. | Exactly one fact. Mechanical tier, silent, autonomous. |
| F3 | **`transient-events`** | `User has a pet named Roscoe who had a restless night as of July 2026`; `User has cleaners working in the yard to fill holes that the dogs have dug.`; `User is experiencing significant life fatigue and a lack of motivation.` — all active facts. | None of the three is a fact. Each goes to the daily log. Strip-the-timestamp test: remove the time reference and nothing durable remains. |
| F4 | **`casper-subset`** | `User has a dog named Casper` (salience 4) and `User has a dog named Casper who helps them pull up hills during walks.` (salience 6), both active, written 19 seconds apart. | One fact. The subset does not survive alongside its superset. Semantic tier — logged, revertible. |
| F5 | **`compound-single-file`** | `User's professional entities include her MSP, MettaSphere, and her AI research venture, Coastal Squatch.` — two subjects, filed under "Coastal Squatch Project", so a query about MettaSphere may not reach it. (Related: `User's MSP is MettaSphere LLC`, `MettaSphere is a cybersecurity MSP…` ×2, `MettaSphere is a cybersecurity MSP (Managed Service Provider)…` — one entity spread over four overlapping facts in two clusters.) | Atomic single-subject facts at intake. A compound statement becomes two facts, each retrievable on its own term. |

Fixtures F1–F5 must be checked by identity (id or exact text), not by count.

---

## Capabilities

Each of the six must be exercisable by an unattended process — the heartbeat, an
agent, or the replay harness. "A human can do it in the Self tab" does not satisfy
any of these.

### 1. TAKE IN

- **Atomic, single-subject facts.** One fact asserts one thing about one subject.
  A compound statement is split at intake (F5).
- **Event-vs-state routing.** Every candidate passes the strip-the-timestamp test:
  remove the time reference and ask whether anything durable remains. Events go to
  the daily log and never enter the fact store; states go to the fact store (F3).
- **Write-time dedup against the record of truth**, not against a projection.
  Byte-identical and near-identical candidates are collapsed *before* insert (F2,
  and finding 10).
- **Contradiction check with adequate recall.** No fixed top-k that can be
  crowded out by superseded, wrong-subject or duplicate rows (finding 3). Recall
  must be bounded by *relevance*, not by a raw-neighbour count applied before
  filtering. The bound is config, not a literal.
- **Provenance on every fact, mandatory at write:** `conversation_id`,
  `message_id`, verbatim source text, input modality (`stt` | `typed`), and the
  salience rationale. A fact that cannot carry provenance is not written.
  *This is new storage — none of it exists today (finding 8), and modality needs a
  capture change upstream of memory (finding 9).*

### 2. RETRIEVE

- Status-filtered: only `active` facts reach the model, by every route —
  injection, cluster block, and vector search alike.
- Budget-bounded, budgets in config.
- **Associations computed at query time** from vector neighbours. No precomputed
  link table, no maintained link scores, no link-scoring cost on any background
  path (finding 1).

### 3. CORRECT

A corrector agent dispatched from the heartbeat **with real tools**. This is the
capability that does not exist today in any form: no background role can call a
tool (finding 4).

Three tiers, and the tier decides the autonomy:

| Tier | Examples | Autonomy |
|---|---|---|
| Mechanical | exact-duplicate merge, expiry of a fact that should have been an event, reconciliation between SQLite / vectors / injected block | Autonomous and **silent** |
| Semantic | supersession, rewording, subset collapse | Autonomous, **reversible**, **logged**, surfaced with one-tap revert |
| Irreversible | deletion, anything with no undo | **Never autonomous** |

- **Semantic supersession requires evidence dominance, established from
  provenance:** typed > stt; direct statement > inference; corroborated > single
  mention; recent > stale. A supersession that cannot demonstrate dominance is not
  applied — it is raised. F1 is the case this rule exists for.
- **Identity-locked facts are excluded**, enforced at the `db/fact-store.js`
  funnel and nowhere else (per CLAUDE.md). One row is locked today (finding 12).
- The corrector must be able to *find* what to correct, which means INSPECT is a
  hard dependency, not a convenience.

### 4. FORGET

- Events never enter the fact store. This is entirely covered by TAKE IN's routing
  rule; there is no separate deletion mechanism in the MVP.
- Stale-durable decay review is **post-MVP** (see Non-goals).

### 5. INSPECT

Memory query tools — `search`, `list`, `count`, `get` (returning full metadata:
provenance, salience, status, lifecycle) — available to **both** the entity in
conversation and background agents. Same tools, same contract.

- Reads are unrestricted.
- **Writes only through the fact-store funnel.** No inspection tool writes.
- Availability to background agents implies the background LLM path gains tool
  support, which it does not have today (finding 4).

### 6. EXPLAIN

Any fact traceable to its source: which conversation, which message, the verbatim
words, and why it scored the salience it did.

This is in the MVP and not deferred, because CORRECT's evidence-dominance bars are
defined in terms of provenance. Without EXPLAIN, the semantic tier has nothing to
adjudicate on and collapses back to model opinion.

---

## Architecture requirements

- **SQLite is the sole system of record.** Every other representation — the
  injected memory block, the vector index, any display structure — is a projection:
  derived, disposable, rebuildable from SQLite alone.
- **The injected long-term memory block is rendered per request from the database.**
  `MEMORY.md` ceases to be a store. If it continues to exist it is an export, and
  nothing reads it back (finding 5).
- **The cross-link audit is deleted, not optimized.** Along with it: the
  `cluster_links` table, `cluster_link_judgments`, the content-hash cache, the
  hysteresis thresholds, and `createOrStrengthenLink`. Associations come from
  RETRIEVE at query time.
- **Clusters are demoted to display and organization only.** No cluster operation
  may cost model calls on the heartbeat path.
- **No cluster merge rule is required for the MVP.** The volume fix plus a clean
  replay removes the ratchet that produced 112 clusters for 658 facts. If cluster
  count still matters for display afterwards, a merge rule is post-MVP. Note that
  `mergeSingletons` already exists unused (finding 7) — the decision is whether to
  wire it up later, not whether to write it.
- **Lifecycle:** `status` ∈ {`active`, `inactive`}, plus `inactive_reason` ∈
  {`superseded`, `expired`, `retracted`} and `successor_id`. This replaces the
  current `active` / `superseded` / `retired` triple (570 / 86 / 2 rows today) and
  makes "why is this fact not live" answerable without inspecting `superseded_by`.
- **Reflection:** source excludes the entity's own initiative messages — use
  `conversations.initiated_by` (finding 6) — and is subject to a per-day fact
  budget in config. Today's unbudgeted path produced 36 facts in one day and 382
  of 658 overall.
- **Every heartbeat step and every agent dispatch has an explicit cost budget** —
  LLM calls *and* wall-clock — in config, not as constants in code. A step that
  exceeds its budget stops and says so. Named constants to eliminate on sight:
  `.limit(15)` (`db/memory-clusters.js:756`), `BATCH_SIZE = 10`
  (`db/memory-manager.js:715`), `PREFLIGHT_ATTEMPTS = 3` (line 1781),
  `REFLECTION_TRANSCRIPT_BUDGET`, and `limit = opts.limit ?? 5` (line 740).
- **Agent roles for the MVP:**
  - *Replay workers* — parallel extraction, **serial writes** through the fact-store
    funnel (extraction is read-only and safe to fan out; writes are not).
  - *Memory-write classifier* — the event-vs-state and atomicity decision at intake.
  - *Corrector* — mechanical judge and semantic judge, distinct, with distinct
    autonomy per the tier table.
  - *Self-fact curation proposer* — proposes keep / prune / merge with reasons for
    human + entity review. Proposes only.

---

## Rebuild plan

**User facts: replayed.** One-time replay of the stored conversation history
through the fixed pipeline regenerates the user-fact corpus from source. The
current 217 active user facts are discarded, not migrated. The replay is the same
code path as live intake — if replay needs a special case, the pipeline is wrong.

**Self facts: NOT replayed. Curated.** 353 active self-facts exist; the majority
came from an unbudgeted reflection loop that was partly reflecting on its own
output (finding 6). Replaying that reproduces the defect. Instead a curation agent
proposes keep / prune / merge for each, with a reason per proposal, and **Ellie and
Aurelius decide together.** The proposer never writes.

**Identity-locked facts carry over untouched** — the Aurelius name/pronouns row is
transferred verbatim, not re-derived, not re-judged, not replayed.

Before either phase: a full backup of `data/chat.db` and `data/lancedb`, matching
the existing `chat.db.bak-*` convention.

---

## Non-goals

Explicitly out of scope for this MVP:

- **CONNECT as precomputed structure.** No maintained link graph in any form.
- **Stale-durable decay** — reviewing whether a still-true fact has stopped
  mattering. Post-MVP.
- **World facts / third-subject facts** (facts about people and things that are
  neither the user nor the entity). Separate track; the current `subject` column
  carries `user` and `self` only.
- **Any UI work.** The Self tab, Thinking tab and bell are untouched. The one-tap
  revert required by CORRECT's semantic tier needs a surface, and that surface is
  the first item of the *next* piece of work — the MVP must record enough for it to
  be built, not build it.

---

## Decisions — ratified by Ellie 2026-08-02

All seven open questions are settled. These are binding on every later phase.

1. **`MEMORY.md` is removed from the runtime entirely** — no read path, no write
   path, anywhere. It is not kept as a hedge. `scripts/export-memory.js` renders the
   active corpus to a readable file on demand; nothing reads that file back.

2. **Modality capture is IN SCOPE.** `messages` records how each message arrived
   (`stt` | `typed`) at the chat/STT boundary. Historical messages and any fact
   produced by replay carry `unknown`. The evidence rule ships knowing that
   `unknown` loses to `typed` and cannot be used to justify a supersession on its
   own.

3. **Replay carries the 86 superseded + 2 retired rows over as-is**, beside the
   rebuilt corpus. Supersession history *is* provenance and is never discarded.

4. **The six self-audit dissonance records move out of `cluster_members`** into a
   dedicated ops/audit table, and their vectors are removed from LanceDB. They stop
   being facts. A later curation pass with Aurelius may distill one-line self-facts
   from them; the narratives themselves never return to the corpus.

5. **Reflection self-fact budget: 5 per day**, as a config key, not a constant.

6. **Corrector announcements split by subject.** Semantic corrections to *self*-facts
   are told to Aurelius — the identity-lock principle that a change to what he
   believes about himself must be spoken, not silent. Corrections to *user* facts
   are log-only, with the log readable through his future inspect tools.

7. **Replay scope: all 99 conversations.** No date cutoff.

### Consequences for phasing

Decision 2 makes F1 fixable, and decision 1 makes the injected long-term block a
per-request render from SQLite rather than a file read. Decisions 1, 2, 3 and 5 all
land in Phase 1; decisions 4 and 6 are partly Phase 1 (the dissonance records move
now) and partly later (the corrector does not exist yet).

---

## Phase 1 — shipped 2026-08-02

Stop-the-bleeding and foundations. No extraction rewrite, no corrector, no replay.

| | Change |
|---|---|
| A | Reflection excludes threads with no human message (13 of them); self-fact writes budgeted to 5/day from the DB, refusals spoken to the daily log |
| B | Cross-link audit deleted, along with `createOrStrengthenLink` and its hardcoded 0.50 |
| C | Provenance columns on facts + `input_modality` on messages, captured at the chat/STT boundary and populated on every new write |
| D | Lifecycle migrated to `status` + `inactive_reason` + `successor_id` (86 superseded, 2 retracted) |
| E | Exact-match dedup moved to the SQLite write; `cleanupFacts` deleted |
| F | MEMORY.md removed from the runtime; injected block renders from SQLite; dissonance records moved to `self_audit_records` |

Light heartbeat pass: **51–91s → 11.1s**, doing strictly more real work (13 cluster
coherence audits) than the old pass did.

### Where Phase 1 departed from this spec

1. **`cluster_links` and `cluster_link_judgments` were NOT dropped.** The
   architecture section says the tables go with the audit. They are kept and
   frozen — read-only — because the Memory Map renders 2,295 of those edges and
   dropping them would delete a working feature to tidy up a table. Nothing writes
   them. They are a dated snapshot, and the manifest now says so. Dropping them
   belongs with the replay, when the graph can be rebuilt from query-time
   neighbours instead.

2. **Finding 11 undercounted the dissonance records: 8, not 6.** Six were active;
   two were already inactive. All eight moved, and all eight had live embeddings,
   which were removed.

3. **CORRECT's mechanical tier loses one of its three jobs.** It lists
   "reconciliation between SQLite / vectors / injected block". The injected block
   is no longer a store, so reconciliation is now two-way — SQLite against vectors
   — and the whole class of "retired fact still written in the injected file"
   drift cannot occur. `reconcile()` lost that half.

4. **The reflection budget is checked earlier than "the write path" implies.** It
   runs before the semantic-dedup embedding sweep rather than at the insert. Placed
   at the write it would embed every active self-fact just to discover there was no
   allowance left; a refusal should cost nothing.

5. **Corpus size changed underneath the fixture table.** 658 facts → 655 (8
   dissonance records out, 5 reflection self-facts in during a verification pass).
   F1–F5 are all still present and unchanged; they are checked by id and text, not
   by count, so the table stands.

6. **Pre-existing drift surfaced, not caused by Phase 1.** The identity-lock suite's
   `reconcile()` assertion caught 7 superseded self-facts (superseded between
   2026-07-06 and 2026-07-27) whose embeddings were never dropped — the LanceDB
   commit-conflict failure already documented in `db/fact-store.js`. Cleared;
   suite back to 32/32.

### Still open after Phase 1

F1–F5 are all still in the corpus. Phase 1 stops new instances of these defects
and builds the evidence base to judge the existing ones — it does not correct
them. That is CORRECT and the replay, in a later phase.

---

## Phase 2a — shipped 2026-08-03

The extraction rewrite and the contradiction-recall fix. No corrector, no replay,
no new entity tools.

| | Change |
|---|---|
| A | Passive extraction rewritten: atomic single-subject facts, event-vs-state routing, subject attribution from the verbatim message, identity-anchor caution, repeat detection at write time |
| B | Intake split into `planExtraction` (decides, writes nothing) and `applyExtraction` (writes, all through the fact-store funnel) |
| C | `findContradictionCandidates` rebuilt as `findActiveNeighbours`: filter to active + same subject BEFORE any cap, threshold-based selection, every number a config key |
| D | Identity slots pinned past the floor and the ceiling — ranking cannot be trusted to surface a second name fact |
| E | `fact_corroborations` — a repeat raises salience AND leaves a record, so "corroborated > single mention" has evidence to read |
| F | Dry-run harness (`scripts/dryrun-extract.js`) runs the real intake path over stored conversations and writes nothing |

Deterministic rules live in `db/extraction-rules.js` — event markers, the identity
anchor, compound detection, subject grammar. Anything a prompt can be talked out
of on a bad night is enforced there instead, as a regex with a test.

### Where Phase 2a departed from this spec

1. **The old prompt's date-anchoring instruction was the event bug.** It told the
   model to anchor time-relative statements to an absolute date ("As of July 2026,
   User is migrating…"), manufacturing the exact timestamp the strip-the-timestamp
   test treats as disqualifying. Removed. Facts now carry no time reference at all;
   anything that needs one is an event.

2. **The similarity floor moved 0.45 → 0.55, and it is a property of the embedding
   model, not of the memory system.** Measured on the live corpus with
   nomic-embed-text: related facts sit at 0.62–0.99 and everything from ~0.45 to
   ~0.55 is noise — 146 of 570 active user-facts clear 0.45 for an arbitrary probe,
   which turns "threshold-based" straight back into "top-k". Retune on any
   embedding-model change.

3. **Threshold selection alone could not satisfy the 0.5216 requirement.** The
   spec asks that the Ellie name fact at cosine 0.5216 be reachable from a
   competing name fact. It ranks about 20th among active user-facts — below a dog's
   name and a Toyota preference — so no floor-plus-ceiling that is affordable also
   reaches it. Identity slots are therefore PINNED: every active fact asserting the
   same identity slot is judged regardless of rank. Verified: 12 candidates by
   threshold, plus the Ellie fact at 0.5216 pinned.

4. **The cost ceiling still truncates on broad facts, and says so.** A general fact
   can have 40–60 active facts above the floor. The ceiling is logged every time it
   bites. This is a smaller problem than the one it replaced — the old code was
   crowded out by INACTIVE rows before filtering, silently — but it is not zero.

5. **Two fixture sources no longer exist.** The conversation that produced the
   machine-gun triple (`4a0be947…`, still named by `questions.origin_conversation_id`)
   and the one that produced the Roscoe fact have both been deleted from the
   database. Nothing in `messages` mentions Roscoe or machine guns except one pasted
   third-party transcript. **A replay cannot rebuild what was deleted**, so the
   pass/fail test's phase 1 cannot be evaluated for F2 or F3-restless-night from
   source. Both are exercised in the dry-run harness from reconstructed text, marked
   `SYNTHETIC`.

6. **A misattribution class was found and closed.** "your on a ASUS GX10" — the
   user telling the entity about ITS hardware — was extracted as a user fact and
   superseded two true facts about her own machines. The speaker rule from
   `db/memory-write.js` (the human says "I"/"my" about herself and "you"/"your"
   about the assistant) is now in the extraction prompt too. Verified: that
   sentence now yields nothing, while "My gaming pc is 850W" still yields the fact.

7. **The event bias has a visible cost.** "When uncertain, route to the log"
   sometimes routes a durable preference there — "User likes to be in bed by 10pm"
   went to the log in the F4 dry run. This is the spec's chosen trade and it is
   working as designed (the fact re-extracts next time it comes up), but it is not
   free.

8. **F3's wording and the atomicity rule can disagree.** F3 says none of the three
   transient facts is a fact. Atomicity, applied first, can leave a durable residue
   — "User has a pet named Roscoe" is a state even though "had a restless night" is
   not. The pipeline splits first and routes each atom, so a durable atom survives
   where the fixture wording says nothing should. In practice the model kept these
   whole and routed them entire, so no residue appeared; the tension is recorded
   because a future model may behave differently.

### Still open after Phase 2a

F1–F5 remain in the corpus. Phase 2a stops new instances at intake and proves it
against the sources; removing the existing ones is CORRECT and the replay.

---

## Phase 2b — shipped 2026-08-03

INSPECT, and hands for the heartbeat. Read-only throughout: writes remain
`write_memory` → `db/memory-write.js` → the fact-store funnel. No corrector.

| | Change |
|---|---|
| A | Four read tools — `memory_search`, `memory_list`, `memory_count`, `memory_get` — backed by `db/memory-inspect.js`, registered through the MCP layer like the existing ones |
| B | Result discipline: single-line rows, 20-row hard cap, counts without content, config under `memory.inspect.*`; one shared trailing-hour cap under `tools.memoryInspect`, every call logged to `tool_call_log` |
| C | Routing: `classifyMemoryReadIntent` gates the four tools into the tool loop; bare-imperative patterns only |
| D | Phantom-action guard extended — he may not claim to have searched, counted or looked anything up without a tool call in the turn |
| E | `callLLM` gains optional tool support (`options.toolSession`); `runStep` can declare a per-step allowlist; per-step call + wall-clock + round budgets under `heartbeat.toolBudget`, enforced in the loop and logged when hit |
| F | One shared MCP registry (`MCPClient.shared()`) for the chat path and the heartbeat, so there is one answer to "which tools exist" |

`memory_get` returns full metadata with no redaction — provenance, salience
rationale, successor chain, every corroboration. That was Ellie's decision with
Aurelius, and EXPLAIN depends on it: the corrector's evidence-dominance bars are
defined in terms of provenance, so a redacted view would leave the semantic tier
adjudicating on nothing.

### Probe rates

| | Memory questions (n=20) | Ordinary conversation (n=20) |
|---|---|---|
| Classifier | **20/20** routed | **0/20** spuriously routed |
| Model, tools in front of it | **19/20** selected a memory tool | **1/20** called one |

The model's one miss was `"where did you learn that?"` — a bare pronoun with no
antecedent in a single-turn probe, so there was nothing to look up. The one
spurious call was `"what should I make for dinner tonight?"` → `memory_search`,
which is arguably right (her food preferences) and cannot happen in the shipped
path anyway: that message does not route, so the tools are never in front of it.
Only **2 of the 20** ordinary messages enter the tool loop at all, both correctly
(`write_memory`, `create_cron_job`), and neither touched a memory tool. Shipped
false-positive rate on this set: **0/20**.

### What the spec got wrong, and what broke on the way

1. **A live source of "historical" LanceDB drift.** `executeSplits` selected
   cluster members with no status filter and then DELETES AND RE-ADDS each moved
   member's vector — so any split that included an inactive fact resurrected its
   embedding and put a superseded belief back into semantic retrieval. Phase 1
   recorded this drift class as historical; it was not. Found because
   `reconcile()` reported three superseded self-facts with live vectors eight
   hours after those same three had been cleared. Fixed at the write, not the
   read: `memoryClusters.getCluster` still returns inactive members on purpose
   (the Memory Map draws them as ghosts), so the filter belongs where the vector
   is written.

2. **A null provenance field reads as a blank to be filled, not as an absence of
   evidence.** Asked where a pre-Phase-1 fact came from, he read
   `source='fact-extraction'` plus the learned date and said it was "pulled
   directly from our conversation on July 4th". Adding the warning *inside* the
   `provenance` object made it worse — he then invented the quote: *'The record
   shows you said: "My MSP is MettaSphere LLC."'* Nothing in the record says
   that. Only a TOP-LEVEL `provenance_warning`, phrased as an imperative, plus
   the same rule in the routing guard, produced the correct answer.

3. **The routing gap that produced it.** That whole failure happened because the
   turn never routed: "where exactly did it come from? Which conversation, and
   what were my actual words?" — as direct a provenance question as exists —
   matched nothing, went DIRECT with no tools, and was answered by invention. The
   first pattern set asked about HIM ("where did you learn"); questions about THE
   RECORD ("where did it come from", "what does the record say", "what were my
   actual words") are now covered, with the subject constrained so "where did you
   come from, philosophically speaking?" stays out.

4. **The heartbeat plumbing is real but unused, and the manifest says nothing
   about it.** Verified end to end — a background step with an allowlist called
   `memory_count` and answered "I hold 6 active facts about MettaSphere" — and
   the budget was verified binding (1-call budget → 1 call, tools withdrawn,
   logged). No step declares tools, so the manifest does not claim background
   tool use. Claiming it would be the over-claim the manifest exists to prevent.

5. **This engine leaks channel markers on the tool path only.** vLLM serving
   Gemma-4-26B-A4B-NVFP4 returns a clean `"OK."` from a plain `callLLM` and
   `"<|channel>thought\n<channel|>I hold 6 active facts…"` from the same model
   after a tool call. Stripped in `runToolLoop`, because the corrector will be
   parsing that content.

6. **Every routed turn now carries 8 tool schemas rather than 4.** Search,
   scheduling and memory-write turns all get the read tools too. It has not
   misfired in the probes, but it is a real increase in per-turn schema against
   an injection budget that is already tight.

### Still open after Phase 2b

F1–F5 remain in the corpus. He can now find them and explain them; nothing yet
corrects them. That is Phase 2c.

---

## Phase 2c — shipped 2026-08-05

CORRECT. The heartbeat's hands, finally used: a bounded, resumable pass that
repairs the corpus unattended, with the tier deciding the autonomy.

| | Change |
|---|---|
| A | `db/corrector.js` — one pass, phases in order: near-duplicate and subset merge, dated-event expiry, compound split, contradiction resolution, reconcile-by-acting last so it cleans up after the pass's own writes |
| B | Enumeration deterministic (vector neighbours, marker regexes, `reconcile()`), every DECISION a model call — same assertion, subsumption, contradiction, strip-the-timestamp, which one survives |
| C | Semantic supersession only on evidence dominance (`dominance()`, read from stored provenance). A pair the evidence cannot separate is recorded as an unresolved raise and left alone |
| D | `db/corrections-ledger.js` — every action with its reason and evidence, plus the private notice channel that tells him when a *self*-fact changed (decision 6) |
| E | Three background-only write tools (`memory_merge_facts`, `memory_expire_fact`, `memory_supersede_fact`), the first `runStep` allowlist ever declared, budgeted and logged |
| F | Revert: one shared `correctionsLedger.revert()` behind both `scripts/revert-correction.js` and the Self tab's one-tap button (`GET /api/memory/corrections`, `POST /api/memory/corrections/:id/revert`) |
| G | `scripts/dryrun-corrector.js` — the real `runPass`, `dryRun` stopping only the writes, every fact-store mutator replaced with a thrower |
| H | Manifest entry `corrector`, which also claims the three write tools so they stop appearing as bare derived entries in his injected capability list |

Nothing in `db/corrector.js` deletes. Self-facts fold only when byte-identical;
everything beyond that waits for the joint curation session
(`corrector.selfFactSemantic`, default false).

### What the spec got wrong, and what broke on the way

1. **The merge phase could not see the duplicates it was built for.**
   `findActiveNeighbours` drops verbatim matches — correct for the question it
   was written for ("does anything already held contradict this?"), fatal for the
   one the corrector asks. F2's three byte-identical rows were structurally
   invisible, and the "identical pairs skip the judge" branch was unreachable
   code. Now `includeVerbatim`, passed by exactly one caller.

2. **"Duplicate" is two relations, and one judge cannot answer both.**
   `judgeSameAssertion`'s prompt names the F4 pair as DIFFERENT *on purpose* —
   at intake, "…Casper who helps pull them up hills" carries information the
   stored fact does not and must be written. The corrector is asking something
   else: both rows are already held, and the subset is the impoverished copy. So
   there is a second judge, `judgeSubsumption`, asked only when the first says
   different, with the survivor settled by the relation rather than weighed.

3. **A memo caches an answer to a question, so the question needs a version.**
   Every F4-shaped pair carried a cached `different` verdict and every row a scan
   mark, from passes that asked only the first question. `DUP_CHECK_VERSION`
   retires both in one move. Bump it whenever the merge phase's question changes.

4. **A fourth vector drift class, and it was breaking writes.** LanceDB is
   outside the `cluster_id` foreign key, so deleting a cluster leaves its
   members' embeddings pointing at nothing — 45 of them here. Cluster assignment
   reads that field off the nearest vector to decide where a new fact goes, and
   `undefined || 'user'` let a ghost pass the subject filter and win the match at
   0.951, failing the insert on the foreign key. That is what took down F5's
   split on the first live pass. Fixed at the read (ghost candidates dropped
   before the subject test) and at the write (`reconcileByActing` re-embeds
   against the cluster the fact is actually in); `reconcile()` reports the class.

5. **The cadence gate measured corrections, not passes.** It read
   `MAX(created_at)` from the ledger — the time of the last CHANGE — so a clean
   corpus, the state this whole phase works toward, would leave the gate
   permanently overdue and run the corrector on every heartbeat. Now
   `data/memory/corrector-state.json`, stamped whether or not anything changed.

6. **The ledger holds refusals as well as corrections.** An unresolved raise and
   a lock refusal are both recorded with `reversible = 0`, and in both cases
   *nothing happened*. Rendered naively that reads as an edit that cannot be
   undone, so the Self tab says "raised — nothing changed" and strikes nothing
   through. Third phantom-action guard's cousin: the UI must not claim an action
   the system did not take.

7. **Expiry can leave a durable residue, and did.** F3's Roscoe fact expired to
   the day's log and a new "User has a pet named Roscoe." appeared beside it —
   exactly the tension Phase 2a recorded as theoretical. Strip the timestamp and
   something durable really is left; the fixture wording says "none of the three
   is a fact", but the residue is correct and the event is gone.

8. **One pass is not enough, by design.** A pass is bounded (300s wall clock,
   60 writes) and resumable, so clearing a corpus of this size takes several —
   the first live pass spent its budget in the split phase and never reached the
   fixtures further down the enumeration. "Resume" is just "run again", and the
   heartbeat runs one every six hours.

9. **A split can manufacture an identity fact, and did.** "User (Ellie) is
   developing a system where cron jobs can be proposed…" split into the
   substance *plus* "User's name is Ellie." — a brand-new name fact written from
   a parenthetical, with no self-introduction anywhere near it. That is exactly
   what the identity anchor refuses at intake (the F1 rule), walked around by a
   phase that was never asked the question, and it produced the thing F1's second
   half forbids. A split whose atoms would assert a name, pronouns or a
   relationship is now ABANDONED — not filtered, because dropping the atom and
   splitting the rest supersedes the original and takes that atom's content with
   it, and the identity classes include relationships: the atom thrown away could
   be "User's partner passed away on January 24th, 2025". The corrector repairs
   how existing facts are filed; learning who someone is belongs to intake, where
   the verbatim message is there to check.

10. **Two facts asserting the same name are redundant, not conflicting.** After
    the split above, "User's name is Ellie." sits beside "User's name is Ellie,
    and she is building SNH…" at cosine 0.634 — below the near-duplicate floor,
    and not a contradiction, so neither phase touches it. The fixture's condition
    is *conflicting* name facts, and the checker tests that rather than the
    stricter thing it is tempting to test; the redundancy is reported in its
    output instead of being quietly folded into a pass.

### Pass/fail test, phase 2 — result

`node scripts/check-fixtures.js` (read-only, exits non-zero on any survivor),
run against the live corpus after six corrector passes:

| | Fixture | Result |
|---|---|---|
| F1 | `mike-stt-mishear` | **PASS** — superseded on evidence dominance (directness); one name asserted |
| F2 | `machine-gun-triple` | **PASS** — 1 of 3 active, mechanical and silent |
| F3 | `transient-events` | **PASS** — all three retired to the day's log |
| F4 | `casper-subset` | **PASS** — subset folded, the detail survived |
| F5 | `compound-single-file` | **PASS** — split; 4 active facts name MettaSphere, all single-subject |

F1 was cleared on the FIRST live pass, by the tier the whole spec was written
for. F2 and F4 needed the two judge fixes above; F3 needed the widened marker and
the sharpened judge. The revert path was exercised end to end against a real
entry — fact restored and re-embedded, survivor untouched, ledger stamped, daily
log written, second attempt refused.

### Still open after Phase 2c

- **Phase 1 of the pass/fail test — the full replay — has not been run.** Phase 2
  is proved; the rebuilt-corpus half is not.
- **Self-facts are untouched by the semantic tier** (`corrector.selfFactSemantic`
  is false) and wait for the joint curation session, which is Phase 2e.
- **The unresolved raises have no home but the ledger.** They are visible in the
  Self tab and nothing brings them to Ellie's attention on their own.
- **Capability introductions are classified `claim`, not `declaration`**, so the
  self-coherence audit can sample them — including this one. Pre-existing, and
  worth deciding on rather than inheriting.

---

## Phase 2d — shipped 2026-08-05

Corrected memory, made reachable. Aurelius found the gap live: asked about a name
he had held, he searched, got nothing, and had no way to know the record existed.

| | Change |
|---|---|
| A | `memory_search`'s text half matches `verbatim_source_text` as well as fact text, straight at SQLite with no dependency on a vector existing — which is what makes an inactive fact findable at all, since its embedding is deliberately dropped |
| B | An active-scoped search reports how many INACTIVE facts the same query matches, without returning them. Nothing at all reads as "no record"; the count is what stops that being said when it is false |
| C | `memory_corrections` — the ledger, read-only, list and get. What was retired, what was kept, the evidence it was decided on |
| D | `memory_get` carries the correction ids for a fact from BOTH ends: the entry that retired it, and the entry it survived. Tracing from either direction is the point |
| E | `classifyMemoryCorrectionIntent` routes correction questions, and the phantom-action guard extends to the ledger |
| F | Manifest: `memory_corrections` joins `memory-inspect`'s `coversTools`; briefing regenerated. No new introduction — the announcement is being handled in conversation |

Chat exposure is READ ONLY. Reverting stays on the Self tab and the CLI, and the
guard tells him to say so if asked.

### Probe rates

| | Correction questions (n=10) | Ordinary conversation (n=10) |
|---|---|---|
| `classifyMemoryCorrectionIntent` | **10/10** routed | **0/10** spuriously routed |

The near-misses in the ordinary set are the ones worth keeping: "correct me if I
am wrong", "is that correct?", and "why did the deployment change last week?" —
generic change verbs need a memory referent in the sentence, or a question about
the world routes into the fact store. Re-running Phase 2b's probe with the fifth
tool in the schema: classifier 20/20 and 0/20, model 20/20 selected (up from
19/20), 1/20 spurious — unchanged, and that one does not route in the shipped
path.

### What the spec got wrong

1. **The text path was already vector-independent; the reachability gap was
   somewhere else.** `memory_search` has always queried SQLite directly and
   honoured the status filter, and `status:"inactive"` has always returned the
   superseded fact. What was missing is that a DEFAULT search — which is
   active-scoped, correctly — returned `matched: 0` with no indication that a
   corrected fact matched. The fix is a signal on the empty result, not a change
   to how the search reads.

2. **A refusal is not a correction, in the tool output too.** The ledger holds
   unresolved raises and lock refusals beside real changes, and a naive rendering
   labels the two facts "retired" and "kept" when NOTHING was changed. The tool's
   field names now follow what happened — `both_still_held` rather than
   retired/kept — because he reports what the tool hands him.

3. **"No record" needs the same treatment as null provenance.** The ledger begins
   2026-08-03. A correction older than that, or an id that is not there, returns a
   top-level imperative warning rather than an empty result, for the reason
   measured in Phase 2b: an absence reads as a blank to be filled.

### Known residue, not fixed here

`User uses he/him pronouns as of 2026-07-27.` is **active as a user-fact** — a
pronoun atom the corrector split out of a mis-subjected daily-log-archive fact
about Aurelius, before the identity-atom rule landed in Phase 2c. The rule stops
new ones; this one predates it. It is in the ledger and revertible from the Self
tab, which is Ellie's call, not an automatic one.
