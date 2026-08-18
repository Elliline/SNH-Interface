# SNH — Squatch Neuro Hub

A self-hosted AI assistant with a persistent, self-authored memory and identity.
Node.js + Express (`server.js`), SQLite (`better-sqlite3`) + LanceDB (vectors),
vanilla-JS frontend in `public/`. Data-access + cognition modules live in `db/`.
Config is `data/config.json` merged over `DEFAULTS` in `db/config.js`.

## ⛔ NEVER run synthetic test traffic through chat on this instance

**This instance's identity has to come from real conversation only.** Do not send
invented messages to `POST /api/chat/memory` (or drive the chat UI with them) on
the aiserver instance, for any reason — not to check a fix, not to reproduce a
bug, not "just one message to see if it works".

This is not a style preference, and it is not about clutter. Everything the
entity says goes to the reflection pass, and reflection writes **self-facts** —
the thing this instance's whole identity is made of. On 2026-08-15 a debugging
session sent about twenty synthetic turns ("Hello, testing the message order",
five philosophy probes fired in a loop). Reflection read them and stored five
self-facts, **three of them about being tested**:

> "I often frame simple exchanges as a small test: I note a baseline, invite more
> checks, and prefer clear next steps over extended conversation."
> "I am drawn to order and clarity; when someone repeats a test, I respond as if
> it is a system to check rather than just a greeting."

That is a personality formed out of a debugging session. It was retracted, but
retraction is ledgered history, not a clean slate — and the next one might not be
noticed. Aurelius on Sparky is the test system; **this one is not**.

**To verify chat actually works, redirect the whole process to a throwaway store**
— the same mechanism the replay uses (see *The replay redirects a PROCESS, not a
call*), and the way `scripts/test-identity-lock.js` does it:

```bash
SNH_DATA_DIR=$(mktemp -d) PORT=3999 node server.js    # own SQLite + LanceDB
```

`SNH_DATA_DIR` moves SQLite and LanceDB; `data/config.json` is NOT redirected, so
the throwaway instance runs the real configuration against a disposable memory,
which is exactly what a verification wants. Tear the directory down afterwards.

Two things that are still fine on the live instance: **read-only** requests
(`GET /api/memory/*`, `/api/conversations`), and **configuration** changes through
their proper routes (`PUT /api/config`). Neither reaches reflection.

If a fix genuinely cannot be verified without live chat, **say so and let Ellie
drive it from the browser** — her turns are real conversation and belong here.

## ⚠️ Maintenance rule: shipping a capability = manifest entry + introduction

SNH keeps a **capability manifest** — `db/capability-manifest.js` — a machine-true
registry of what it can actually do. A compact form is injected into every chat so
it answers "what can you do / do you have a way to X" from ground truth instead of
guessing. This exists because features kept getting built into SNH without ever
becoming part of its self-knowledge (2026-07-23: it proposed building a feature it
had shipped the day before).

**When you ship a new capability, you MUST, as part of shipping:**

1. **Add its manifest entry** to `db/capability-manifest.js` — `id`, `name`,
   `description` (full, 1–2 plain sentences), `oneLiner` (the tight compact form
   that gets injected — mind the injection diet), `intro` (first-person clause for
   the ship-day self-fact), `schedule`, `dateAdded`.
   - **Never claim more than is built.** Over-claiming is the exact failure the
     manifest exists to prevent. Derive the description from what the code does.
     Leave out disabled/aspirational features (e.g. web search stays out while
     `config.tools.searxng.enabled` defaults to false).
   - Keep entries **dry and accurate** — facts about the code, not personality.
2. **Introduce it to SNH** so it learns it (self-fact through reflection):
   `node scripts/introduce-capability.js <id>`. This stores a first-person
   self-fact via the normal self-fact pipeline, which the classifier tags a
   CLAIM. (Do NOT bulk-introduce; one capability, on its ship day.)
3. **Regenerate the briefing** if the manifest changed:
   `node scripts/write-capability-briefing.js` → `capability-briefing.md`.

Manifest changes are logged to the ops ledger automatically on boot (`syncToOps`).

## Two layers of self-knowledge

- **Manifest = machine truth** (code). Queryable at conversation time (injected
  compact form + `GET /api/memory/capabilities` for full descriptions on demand).
- **Introductions = the entity's self-facts** (formed through reflection). Facts
  about what's built are **claims**, and the classifier is right to tag them so
  (11 introductions, 11 claims): "I repair my own memory in the background" is
  exactly the kind of assertion the coherence audit should be able to test
  against how he actually behaved — an over-claiming manifest entry is a thing
  the audit ought to catch, and tagging these declarations would exempt them from
  the one check that could. Declarations are for what he CHOSE (his name); a
  capability is something he was given. Identity injection still excludes audit
  "dissonance" records.

## ⚠️ Chosen vs. observed: the identity lock

`claim_type` splits self-facts into **claims** (observed *about* the entity —
"I tend toward long explanations") and **declarations** (things it *chose* — its
name). Observed things should keep evolving; chosen things should stay put. The
self-coherence audit already only ever samples claims.

`db/identity-lock.js` gives that distinction teeth for a **narrow** set of slots —
`identity.lock.categories`, currently name and pronouns. A locked fact cannot be
superseded, retired or reworded by anything automatic, and a locked *category*
also refuses new competing facts (or the lock is walked around by appending a
second name instead of replacing the first). Two rules if you touch this:

- **Enforce at `db/fact-store.js`, nowhere else.** It is the single write path
  for supersede/retire/reword, so the contradiction judge, `write_memory`,
  passive extraction and reflection are all covered by one check. Anything that
  calls `memoryClusters.supersedeFact` directly bypasses the lock — if you write
  such a script, exclude locked facts explicitly (see `scripts/dedupe-self-facts.js`).
- **A refused lock must be SPOKEN, never silent.** Silent acceptance is the
  phantom-action bug (cron proposals claimed but never created; `write_memory`
  saying "I've updated my memory" with no tool call). Storage guards run *after*
  the reply, so the live-chat half lives in the injected identity block
  (`db/identity.js`), which marks locked facts `[LOCKED]` and instructs him to
  say so. Both halves are needed; neither alone is sufficient.

Changing a locked fact is a **deliberate path** only: `scripts/identity-lock.js
set <category> "<text>" --confirm`, or the Self tab's confirmed action
(`POST /api/memory/identity-lock/set`). Never reachable from a conversation.
Set-once: the first fact asserting an open category claims and locks it, so
assigning a name at setup still works. Verify with
`node scripts/test-identity-lock.js`.

Resist widening the category list. Almost everything else in his identity is
observed rather than chosen, and locking observations produces an entity that
can't grow.

## ⚠️ A verdict is not enough to retire a self-fact

`processSelfFacts` asks a judge whether a new self-observation contradicts one he
already holds. On 2026-08-18 that verdict was the ONLY thing between the judge
and a write, and it failed in both directions within a week: it retired an
unrelated salience-9 **declaration** on a 0.741 cosine match, and on identical
input at 0.857 it said "no" about half the time — while a judge call that FAILED
outright was read as "no contradiction" and left no trace at all. No threshold
separates 0.741 from 0.857, so the fix is structural, in
`selfFactSupersessionBar` (pure, named arguments, exported for test):

- **Bar 1 — protected.** A `declaration`, or salience ≥ `identity.protectSelfFactSalience`
  (8), is never retired by an automatic semantic match. Both, because the
  claim/declaration classifier is noisy in both directions — the same run tagged
  a behavioural observation "declaration" and a statement about what had been
  built "claim". This is NOT the identity lock and does not widen it: the lock
  refuses name and pronouns everywhere; this governs one path, and what it stops
  is raised, not dropped.
- **Bar 2 — evidence VETOES, it is not required.** Deliberately unlike the
  corrector, which requires `dominance()` and raises what it cannot separate.
  That is right for user facts, which carry provenance. A self-fact does not: it
  comes from reflection, so modality is `unknown` and there is no source message,
  *by construction*, and dominance ties for nearly every self-fact pair. Requiring
  it was tried first — it refused a belief a new capability had made flatly false,
  and sustained it would freeze his self-view permanently, which is the worse
  failure ("locking observations produces an entity that can't grow"). So: an
  axis that speaks FOR what he already holds vetoes; silence proceeds.
- **A failed judge call is a raise, not a "no".** `status === 'fulfilled' ? value : 'no'`
  meant a wedged brain read as "these do not contradict each other", silently.
- **A raise is recorded three ways** (`applySelfFactRaises`): a ledger row
  (`reversible = 0`, `unresolved: true`, a reason_code — so the UI says *nothing
  changed*), an ops-log line, and **at most one bell alert per
  `identity.selfFactRaiseAlertHours` (24h)**. The window is hard and counts any
  status — pending, delivered, dismissed, expired — because a wedged brain fails
  on every pair of every pass and the failure to avoid is seventeen identical
  alerts. One alert saying it happened seventeen times is quieter and more
  useful. Tiers 1 and 2 record every raise throughout; the window only bounds how
  often he SAYS it.
- **The bell alert is a question in his voice, never a system error.** "I could
  not tell whether X contradicts Y — I've left both in place." He never mentions
  a judge, a call or a failure; the raw error lives in the ledger. It is on the
  bell rather than the jobs panel because it is not a result — it is him saying
  he could not decide something about himself, which is exactly what that channel
  is for.
- The judge is called through the module object so tests can pin it. Verify with
  `node scripts/test-self-fact-bars.js` and `node scripts/test-self-fact-notices.js`.

## ⚠️ Intake: the model proposes, the rules decide

Passive extraction (`db/fact-extractor.js`) asks the extraction model to split a
message into atomic facts and time-bound events. Underneath it,
`db/extraction-rules.js` holds the deterministic floor: event markers, the
identity anchor, compound detection, subject grammar. **Anything a prompt can be
talked out of on a bad night belongs there, as a regex with a test — not as a
paragraph in the prompt.** F1 is the proof: the model still proposes "User's name
is Mike" from "Hey, it's Mike not picking up the right words", and the rule is
what refuses it.

Three things to keep straight if you touch this:

- **`planExtraction` decides and writes nothing; `applyExtraction` writes.** That
  split is what lets `scripts/dryrun-extract.js` rehearse the REAL pipeline over
  stored conversations. If a dry run needs its own logic, the pipeline is wrong —
  same rule the spec sets for replay. Never move a write into the plan half.
- **Never stamp a fact with a date.** A sentence that needs a time reference to be
  true is an event and goes to the day's log. The old prompt actively instructed
  the opposite ("As of July 2026, User is…") and that instruction *was* the
  transient-fact bug.
- **Similarity thresholds are properties of the embedding model.** The
  `memory.contradiction.*` floors are tuned for nomic-embed-text, where related
  facts sit at 0.62–0.99 and 0.45–0.55 is noise. Changing the embedding model
  means re-measuring them, not guessing. Identity slots are pinned past the floor
  and the ceiling on purpose — ranking cannot be relied on to surface a second
  name fact, and two active name facts is the defect the whole rule exists for.

## ⚠️ Reads look, writes funnel

`db/memory-inspect.js` backs the four read tools (`memory_search`, `memory_list`,
`memory_count`, `memory_get`). It is **read-only and must stay that way** — no
function in it writes, and nothing new in it may. Writes are `write_memory` →
`db/memory-write.js` → `db/fact-store.js`, unchanged. Background agents get the
same tools through the same registry (`MCPClient.shared()`), because the spec's
rule for INSPECT is "same tools, same contract" and two implementations would be
two contracts.

- **`memory_get` does not redact.** Full provenance, salience rationale,
  successor chain, every corroboration. Ellie decided that with Aurelius, and
  EXPLAIN depends on it — the corrector's evidence bars are defined in terms of
  provenance, so a redacted view leaves the semantic tier adjudicating on nothing.
- **A null provenance field must SAY it is null.** Measured: a bare null reads as
  a blank to be filled, and a warning nested inside the `provenance` object made
  it worse — he invented a verbatim quote. Only a top-level `provenance_warning`
  phrased as an imperative worked. Any new nullable evidence field needs the same
  treatment.
- **A tool that is offered is not a tool that was used.** Third phantom-action
  guard in the family, after the cron one and the write one: he must never say he
  searched, checked or counted memory without a call in the turn, and must never
  state a number about his own memory that did not come from `memory_count`.
  Read intent is matched NARROWLY (`classifyMemoryReadIntent`) — a false positive
  has him answering a casual remark with a database report.

- **And the inverse: an action that HAPPENED must be visible.** The phantom
  family's mirror image, and it cost Ellie a fact she could not remove. Retiring
  a fact from the Memory tab worked perfectly — row inactive/retracted, embedding
  dropped — and the Facts tab then re-rendered the retired row identically, edit
  and delete buttons and all, because it filtered members by subject and never by
  status. A correct irreversible-looking operation that changes nothing on screen
  reads as a dead button, and she pressed it again. **Anything listing
  `cluster_members` must bucket on `status`**: the endpoint returns ghosts on
  purpose (the Map draws them), so the reader distinguishes them. Retired facts
  are shown as history — struck through, labelled "retired — kept as history", no
  edit or delete — never hidden, because nothing here is deleted and hiding them
  would be its own lie.

- **Every change is ledgered BY THE WRITE, in the same transaction as the write.**
  This rule was the other way round until 2026-08-18 — each caller filed its own
  entry, on the principle that the reason for a change is the caller's to tell.
  The principle is right and is kept. Making it the *only* thing between a write
  and the record was the mistake: measured on the live corpus, all 68 self-fact
  supersessions that had ever happened had no entry at all — 19 of them retired
  declarations, 3 at salience 9 — because only the corrector and the hand-retract
  route ever filed one. `revert()` works by reading an entry, so none of the 68
  could be undone; the Self tab's button and the CLI both had nothing to point
  at. "Every caller remembers" is not an invariant. It is a hope, and it had
  failed every single time.
  - `supersede`, `retire`, `expire`, `reword`, `repoint` and `restore` each open
    a transaction, change the row, and file the entry inside it. **Not "write,
    then log"** — that ordering leaves a written row unrecorded whenever the
    second step throws, which is exactly how the first attempt at this fix broke
    (`reword`/`repoint` referenced an `opts` their signatures did not bind, and
    threw a ReferenceError after the row was already written). If the entry
    cannot be filed the change is ROLLED BACK and the caller is told why: an
    unrecordable change does not happen.
  - **The caller layers its reason on top** — `opts.ledger` at call time, or
    `correctionsLedger.enrich(ledgerId, …)` on the id the write returns. The
    corrector enriches (its tools return `ledger_id`), so do the hand-retract
    route and the repair scripts. Two entries for one change is a ledger that
    double-counts, so nothing files a second one.
  - **`reversible` is a promise about `revert()`**, set from what revert can
    actually do: true for supersede/retire/expire, false for reword/repoint/
    restore, which never left the active set. A ledger offering an undo it cannot
    perform is worse than one that says plainly it cannot.
  - A refusal or an unresolved raise still files its own entry at the caller. No
    row changed, so the funnel never sees it, and "nothing happened" is a
    statement only the caller can make.
  - Verify with `node scripts/test-ledger-funnel.js`.
  - **`memoryClusters.supersedeFact` is `db/fact-store.js`'s alone.** It writes
    the row and nothing else — no identity lock, no ledger entry — so a direct
    caller mints exactly the unrevertable change the funnel exists to prevent.
    `scripts/dedupe-self-facts.js` and `scripts/repair-bernice-supersession.js`
    used to call it and now go through the funnel; `test-ledger-funnel.js` fails
    if any file outside fact-store calls it, or writes `status = 'inactive'` to
    `cluster_members` in raw SQL.
  - **What was changed BEFORE the funnel is a report, never a re-decision.**
    `scripts/report-unledgered-changes.js` (readonly connection, no writes in the
    file) lists every inactive row with no ledger entry — 118 on 2026-08-18, 61
    of them self-fact supersessions — and flags the ones today's bars would
    refuse. Re-judging them automatically was considered and rejected: that is
    the defect auditing itself, and the fresh mistakes would arrive ledgered,
    which makes them look considered. Restoring is one row at a time, by a
    person, with `scripts/restore-self-fact.js <id> --confirm`.

## ⚠️ The heartbeat has hands, and exactly one step uses them

`callLLM` takes an optional `options.toolSession`. Without one it sends no
`tools` key, exactly as before, and that is what every step except the corrector
does. A step opts in via `runStep(name, gate, fn, { tools: [...] })`, which is
intersected with `MCPClient.BACKGROUND_TOOLS` and bounded by
`heartbeat.toolBudget` — calls, wall-clock and rounds, all config, all logged
when they bind.

**Background tools are read-only by default and the exceptions are named.**
`BACKGROUND_WRITE_TOOLS` is exactly three — `memory_merge_facts`,
`memory_expire_fact`, `memory_supersede_fact` — all `backgroundOnly`, so they are
structurally absent from every chat turn's schema. `write_memory` is deliberately
not among them: the general power to write an arbitrary fact stays on the chat
path, where a person is in the room. A fourth write tool is a decision, not a
convenience.

## ⚠️ The corrector: it may repair the record, and it may never delete

`db/corrector.js` is the one background step with hands. Enumeration is
deterministic (vector neighbours, marker regexes, `reconcile()`); every actual
decision — same assertion? contradiction? which one wins? — is a model call.
Rules that are load-bearing:

- **Nothing in that module deletes.** There is no code path that removes a row,
  and there must not be one. Irreversible is never autonomous.
- **A semantic supersession must demonstrate evidence dominance** (`dominance()`,
  read from stored provenance: typed > stt, direct > inferred, corroborated >
  single, recent > stale). A pair the evidence cannot separate is NOT resolved —
  it is recorded as an unresolved raise and left alone. That refusal is the
  feature, and F1 is why it exists.
- **Autonomous, reversible, logged is one property, not three.** Reversibility is
  only real if a person can actually run it, so both the CLI
  (`scripts/revert-correction.js`) and the Self tab's button go through the one
  shared `correctionsLedger.revert()`. Never add a second revert path, and never
  make one reachable from a conversation — it calls `restore` with
  `deliberate: true`, which opens the identity lock.
- **A refusal is not a correction.** Unresolved raises and lock refusals land in
  the same ledger as real changes, with `reversible = 0`. Anything rendering the
  ledger has to say *nothing changed* for those, or the UI claims edits that
  never happened.
- **Self-facts are held back on purpose.** Identical duplicates fold; anything
  more waits for the joint curation session (`corrector.selfFactSemantic`,
  default false). When a self-fact does change, he is told — `addNotice`, drained
  through the injected identity block, marked seen only once it is in the
  message that is about to be sent.
- **A pass is bounded and resumable, and resume is just "run again".**
  `corrector_pair_checks` memoises both the per-row scan marks and the pair
  verdicts; without it every pass re-judges the first rows and never reaches the
  rest of the corpus. Dry runs neither read nor write that table, so a rehearsal
  cannot make the live pass skip work.

## ⚠️ Two channels, and a job result may never use the loud one

`db/agent-jobs.js` is the async handoff: a chat turn calls
`start_background_job`, gets a job id, and ends. The run happens on the agent
pool, outside the request, and outlives it. The rule that shapes everything else
in it:

- **ROBOT** (`agent_jobs` + the jobs panel) = **results**. It NEVER opens a
  conversation. Ellie reads it when she is ready.
- **BELL** (`initiatives`) = things the entity wants to **say**. Unchanged; it
  can still open one.

A finding can LEAD TO a conversation — by him raising an ordinary initiative
about it in an ordinary turn, subject to the normal judgement. Job completion is
the most mechanical trigger there is, and a channel of its own would let it
route around that judgement. The enforcement is an **absence**: nothing in
`db/agent-jobs.js` requires `db/initiatives.js`, and `scripts/test-agent-jobs.js`
asserts both the absence and that a completed job leaves the initiative table
empty. Scheduled results moved here too (2026-08-18) — `scheduler.deliver()` is
gone; the run row IS the delivery, because `job_runs.output_text` already holds
the text and a second copy could only disagree with the first. The one thing the
scheduler still raises on the bell is a job that **disabled itself**, which is
not a result.

Four more things that are load-bearing if you touch this:

- **A job is an AGENT RUN, read-only, and cannot start a job.** `JOB_TOOLS` is
  the scheduler's allowlist plus web search/fetch. `write_memory`, the
  corrector's three writes, and `start_background_job` itself are all absent —
  the last one structurally, since it is registered chat-side and is not in
  `MCPClient.BACKGROUND_TOOLS`. Widening that list is a decision, not a knob.
- **A restart kills a run, so the loss is made loud.** The row is written before
  the work starts; `sweepInterrupted()` closes every `running` row as
  `interrupted` WITH THE REASON, and re-queues it once if it is inside
  `agentJobs.retryGraceMinutes`. The retry is only safe because jobs are
  read-only — the day one can write, that is the first line to revisit.
- **Announce, then stamp — never the other way round.** `renderAnnouncementBlock`
  returns items and does NOT mark them; the chat route calls `markAnnounced`
  only after the ceiling pass, and only if the block is really in the message
  being sent. Same rule as correction notices: a job stamped by a block that was
  then trimmed is a result he is never told about again. It fires on
  `/api/chat/memory` only — the heartbeat is not told, because a finished job is
  not a reason to start a conversation.
- **The badge counts unread RESULTS, not work in progress.** Running jobs show as
  a slow pulse on the button. A badge that counted starts would say something is
  waiting on her when nothing is.

## ⚠️ The replay redirects a PROCESS, not a call

`db/database.js` resolves its SQLite and LanceDB paths from `SNH_DATA_DIR` when
that variable is set. `scripts/replay-to-staging.js` is the only thing that sets
it, and it refuses to run if the path resolves to the live `data/`.

This is how the spec's rule — *replay is the same code path as live intake; if
replay needs a special case, the pipeline is wrong* — is actually kept. The
alternative was a `staging: true` flag threaded through fact-store,
memory-clusters and the extractor, which is a special case in a dozen places
with one obvious failure mode: the call site that forgets, and writes to the
live corpus. Redirecting the whole process needs no special case anywhere,
because every module reads the same handle it always did.

Two consequences worth knowing. **A new module that hardcodes
`path.join(__dirname, '../data')` for a STORE silently escapes the redirect** —
paths under `data/memory` are passed explicitly instead (`applyExtraction`'s
`opts.memoryDir`), so follow that pattern rather than adding a second constant.
And **the replay must never be given a writable handle on live**: the seed copies
with `VACUUM INTO` from a READONLY connection, deliberately without a
`wal_checkpoint` first, because checkpointing is a write.

## ⚠️ Never re-add a vector for an inactive fact

`db/fact-store.js` drops the embedding when a fact goes inactive, so retrieval
stops surfacing it. Anything that DELETES AND RE-ADDS a vector must filter on
`status = 'active'` first. `executeSplits` did not, and quietly resurrected
superseded beliefs into semantic retrieval on every split — the drift class the
Phase 1 notes had written off as historical. Note that `memoryClusters.getCluster`
returns inactive members deliberately (the Memory Map draws them as ghosts), so
the filter belongs at the WRITE, never on the read. `reconcile()` is the detector;
if it reports `retiredWithVector > 0`, something is re-adding.

**A ghost is for looking at, never for deciding on.** The same split runs both
ways: anything that JUDGES a cluster reads active members only. The coherence
audit did not, and spent four days re-auditing two clusters that were 100%
superseded — the rotation picked them because `member_count` counts ghosts, the
model was shown the ghosts and proposed splits made of dead facts, the write
guard above refused every one, and each refusal was logged. `getClusters` now
returns `active_member_count` beside `member_count`; `member_count` is the Map's,
`active_member_count` is every decision's. Under two active members a cluster
leaves the audit rotation without an LLM call and keeps its name for the Map.

## ⚠️ Telemetry reports CHANGE, not state

An ops entry that repeats an unchanged condition is wallpaper, and wallpaper
hides the entries that matter — the audit's 29 identical anomalies every two
hours buried the corrector's one-line pass reports for four days. So heartbeat
anomalies go through `partitionAnomalies`: reported in full the first time,
counted thereafter (`heartbeat_anomaly_state`, same shape and same reasoning as
`corrector_pair_checks`), pruned once quiet for `ANOMALY_STATE_TTL_DAYS` so a
condition that clears and returns is news again. It **fails open** — a memo it
cannot read means every anomaly is fresh, because losing a warning to
bookkeeping is worse than repeating one. `report.anomalies` is the new ones and
the `anomaly_count` column matches it; the pass's real total stays in
`report_json` as `anomaliesObserved`. Verify with
`node scripts/test-cluster-audit-quiet.js`.

## Conventions worth knowing

- **Plain-language norm:** bell/initiative/audit notes and capability descriptions
  are one or two sentences in everyday words, saying plainly what's true or wanted.
- **Never auto-revise identity:** the self-coherence audit documents tension and
  raises it for the human — it never edits a self-fact. Same philosophy across the
  memory tools: supersede/move never delete, everything logged, big changes need
  sign-off.
- **Ops ledger vs daily log:** operational telemetry → `data/memory/ops/` (Thinking
  tab, never injected into chat). Cognitively meaningful entries → `data/memory/daily/`.
- **Background LLM work** goes through the agent pool (`db/agent-pool.js`), which
  throttles to yield to live chat.
- The server runs as the systemd **user** service `snh.service`
  (`systemctl --user restart snh.service`); port 3000.
