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
   - **Never write a condition into the text, either.** If what the entry claims
     depends on the box (`job-documents` and whether a chromium is installed),
     make `oneLiner`/`description` a **function of config** and resolve it, the
     way `when` resolves. An injected "a PDF, or text where no browser is
     installed" is a question he cannot evaluate, so he hedges at Ellie instead
     of answering — an over-claim and an under-claim stapled together.
   - **Check it still fits.** The whole list is budgeted
     (`memory.injection.manifestTokens`, 700). Over budget, the render keeps
     every name and sheds one-liners **from the newest backwards** — i.e. from
     the entry you just added. Boot warns and the heartbeat rings the bell when
     that happens, but the fix is tighter one-liners, not a bigger budget.
     `node scripts/test-injection-budget.js` asserts it fits with headroom.
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
- **No pre-generation tool gate.** Tools are attached to EVERY chat turn; the
  model decides. The classifiers still run and still drive guidance blocks, but
  they are a safety net, not the mechanism. They gated generation until
  2026-08-18, when Ellie typed "Use an agent and write up any thing you know
  about my clients", every classifier returned false, the turn routed DIRECT with
  an empty tools array, and the model wrote three paragraphs about the job it had
  started. No job existed; none could have. The two dispatches that worked that
  day worked because their messages happened to mention a company and a year,
  tripping the SEARCH classifier and dragging the registry along — working by
  coincidence of shape. Cost measured: **3,432 tokens of tool schemas per turn**
  (11 tools), in the `tools` field, so outside the 6,000-token message ceiling.
- **The triggers are built from HER messages, not ours.** Measured over 590 real
  user messages: "take your time" appeared twice and both were Claude's own test
  prompts typed that afternoon; "while I'm out" never. Tier 1 (she names the
  mechanism) matches her actual typing including `use and agent`, which is the
  only real instance in the corpus. Tier 2 (build asks) is behind
  `tools.agentJobs.dispatchBuildRequests`, off until code execution ships. Tier 4
  (time granted) is flagged in code as unattested and never dispatches alone.
- **A claimed dispatch is checked against the queue, in the turn.** If a reply
  asserts a started job and no `agent_jobs` row was created in that conversation
  in that turn, it is logged as PHANTOM DISPATCH and a correction is appended to
  the reply she is reading. Same doctrine as the ledger funnel: not "the model
  should not claim this" but "a false claim does not reach her unmarked".
- **He can see running jobs only when the block is there.** `renderActiveJobsBlock`
  injects live queue state per turn (null, and zero tokens, when nothing is
  active), paired with a standing rule in the epistemic block: absent block means
  nothing is running and he must say so; present block means he knows THAT a job
  runs and for how long, never how far along. Asked "are you still working on
  this?" with no such view, he had produced a detailed progress report on two
  jobs — one already finished, one that never existed.
- **The description says WHEN, and a guidance block says "now".** Shipped, the
  tool described mechanics and prohibitions only: every reason to delegate was
  self-assessed, while the reason not to was concrete and always satisfiable —
  it *can* answer now, `web_search` is in the same list. Two research prompts
  that explicitly granted time were answered inline and no job was queued. The
  triggers are observable now (she granted time; more than ~2 searches or
  several sources; over a minute of work), the trade is stated as a gain, and
  `classifyHandoffIntent` firing also pushes a `guidance` block saying she is
  not waiting. A tool the model can use immediately will always win an argument
  conducted only inside a tool description.
- **Handing off never means answering nothing.** Both halves failed live within a
  day of each other. The guard is BOTH/AND in the description and again in the
  tool's return message: say what you already know this turn, hand off the
  digging. A turn that only promises to come back has given her nothing.
- **A time grant outranks "right now".** `classifyHandoffIntent` keeps immediacy
  negatives (`quick`, `right now`) separate from hard ones (`never mind`): a
  message asking for current facts *and* granting time is not asking to be
  rushed. "…as inference backends as of right now" had been cancelling exactly
  that kind of request.
- **Tier 1 is a decision she already made, so the call is FORCED and then
  backstopped.** Tier 1 was never gated — `classifyHandoffSignal` returns it
  before it reads `allowBuild` — and on 2026-08-18 that was not enough: "Use and
  agent and write me a python script for a calculator" fired tier 1, fired the
  guidance block, had the tool fourth of eleven in the payload, and came back
  `tool_calls: []` with "I have started a background job to write a Python
  calculator script." Nothing blocked it; the suggestion was declined. So there
  are now three layers, and the third is the only one that cannot be argued with:
  the description says her naming an agent settles it; the first tool round pins
  `tool_choice` to `start_background_job` (retried unforced if the engine refuses,
  and Ollama ignores the field); and if the turn still ends with no row, **the
  server enqueues one from her message verbatim and says so in the reply**. Tier 1
  only — tiers 2–4 are inferences about the shape of the work, and forcing one
  would dispatch a job she never asked for.
- **A job may PRODUCE, not only report.** The run prompt said "everything you
  report must come from a tool result", full stop — under which a job asked to
  write a calculator has no legal move, because no tool returns one. The rule is
  kept and scoped: anything asserted **as fact** must come from a tool result in
  that run; anything she asked to be **produced** — a script, a draft, a plan — he
  writes himself, from what he knows, using tools to check what it depends on. A
  job still cannot execute what it writes.
- **A job that started has a result she can read. Always.** The car job spent all
  twelve of its calls on searches failing with the same broken-URL error, returned
  no text, and closed as `failed` with `result_text` NULL — an empty card, and the
  memory work it had finished before it ever reached a search went in the bin with
  it. The work was done; only the writing-up was missing, and nothing asked for
  it. Three layers now: `runToolLoop` spends its last round on a **no-tools
  writeup turn** instead of returning `content: ''`; `salvageWriteup()` asks once
  more, without tools, carrying the run's own tool record; and
  `mechanicalAccount()` is the floor — deterministic, no model, so it cannot come
  back empty either. A cut-short run is **`partial`**, the honest third status:
  `ok` over-claims and `failed` throws away a real result. The panel shows the
  text with the reason it stopped underneath — never instead of it — and the
  announcement block tells him the same thing her card says.
- **A failed call is not priced like progress.** The budget bills in units:
  a usable result costs 1, an error (any tool) or an empty `web_search` /
  `memory_search` costs `heartbeat.toolBudget.failedCallCost` (0.25). The rule is
  `toolCallCost()` — pure, exported, tested — and it is deliberately narrow:
  `memory_count` returning 0 bills in full, because there zero **is** the answer,
  and an empty `memory_search` that surfaced inactive facts bills in full too. A
  discount alone would let an everything-fails loop run four times as long, so a
  raw **attempt ceiling** of 2× the budget sits underneath it and is what binds
  when a provider is down. `session.calls` stays the raw count — it is what the
  logs and the panel mean by "tool calls" — and `session.billed` is the budget.
  Job budgets were raised with it: 40 calls, **16 rounds** (6 rounds at 2–3 calls
  each made the 12-call budget decorative — rounds were what actually bound), 15
  minutes, 2000 output tokens because 700 cannot hold a script.
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

## ⚠ A result has a FORM, and the form is derived, not chosen

A job result was one column of text on a card and that was the whole of it. Fine
for three sentences; wrong for everything else. A research report arrived as raw
markdown in a narrow panel — pipes and asterisks, the source of a table with the
table nowhere — and a Python module arrived as a code block to be selected out of
a scrolling box. Neither is a thing you can keep, open later, or send to anyone.

`db/job-artifacts.js` reads what the run actually produced and picks one of three
forms. **Derived from the output, never asked for and never chosen by the model**
— which is why the run prompt tells him the rule exists and forbids him to
announce it: a model that could pick would pick "PDF" for a two-line answer.

- **code** — one fenced block that IS the result → a source file with the right
  extension. Two or more substantial blocks is a document ABOUT code, not one
  file, and flattening it would throw the others away.
- **document** — long prose → a PDF, or a formatted text file where there is no
  browser. `documents.inlineMaxChars` is the line.
- **inline** — short → the card, rendered as markdown.

Load-bearing pieces, in the order they bite:

- **Length is measured with code discounted, and that is not sufficient on its
  own.** `toPlainText` counts a fence as the token `[code]`, which is right for
  "is the WRITING long" — a short note wrapped around a long script is a short
  note. It is wrong as the only test: a run returning two scripts under sixty
  words measured as 60 characters and stayed on a card, 140 lines of code in a
  panel the width of a phone. Substantial code is a file whatever the prose does.
- **ONE renderer, served from `db/`.** `db/markdown.js` runs under Node and in the
  browser, and `server.js` serves it at `/markdown.js` rather than a copy living
  in `public/`. The card and the printed report have to agree on what a document
  looks like; two renderers would be two answers that drift.
- **Escaping is structural, and the ordering IS the security property.** Every
  construct that produces markup is stashed behind a NUL placeholder as it is
  recognised, and only plain text is ever handed to `esc()`. The first version
  built `<a>`/`<img>` in place and split on a tag-shaped regex to decide what to
  escape — which cannot tell our markup from the author's. A literal
  `<img src=x onerror=…>` in a result matched the split and reached the browser
  unescaped. **Job text is written by a model that has just been reading
  arbitrary web pages; it is untrusted input.**
- **No PDF library, and the DevTools socket is deliberately not opened.**
  `chromium --headless --print-to-pdf` IS `Page.printToPDF` behind a flag — the
  same code inside the browser. Driving the socket would mean hand-writing a
  WebSocket client (no `ws` here, and no dependency is being added for this) and
  would buy exactly one thing: page numbers via `headerTemplate`. Everything else
  — paper, margins, backgrounds, page-break control — is CSS, which is where
  paged media puts it. **So there are no page numbers, and there is no way to add
  them without that socket.** Chrome does not implement the `@page` margin boxes
  that would generate them.
- **A missing chromium is a DOWNGRADE, never a failure.** On this box there is
  none, and Ubuntu 24.04 has no apt package — `chromium-browser` is a
  transitional stub and the real install is `sudo snap install chromium`. So the
  text-fallback path is the ordinary path here, not an edge case, and it has to
  produce something worth opening: structure kept, prose wrapped to 78 columns,
  **tables aligned into columns**, because an unaligned table is the exact
  unreadable thing this work started from.
- **Snap confinement shapes two things**, and both are invisible until they
  break: a snap gets a PRIVATE `/tmp`, so the intermediate HTML goes under the
  data directory instead; and the snap `home` interface does not cross HIDDEN
  directories, so nothing this writes may live in a dot-directory (`print-work`,
  not `.print-work`).
- **A confined chromium given a page it cannot read EXITS 0 AND WRITES NOTHING.**
  Measured on Chromium 151 snap: exit code 0, empty stderr, no output file. It
  does not fail, so only the empty-output guard notices — and on its own that
  guard reports "produced an empty PDF", which sends you to inspect the HTML, the
  one place the fault is not. `confinementProblem()` therefore checks the path
  BEFORE spawning and names the cause. The practical consequence:
  **any instance whose data directory is outside `$HOME` cannot print PDFs**,
  which is every test instance, and is why `scripts/test-job-artifacts.js` asserts
  the fallback rather than the PDF when the browser is a snap.
- **There is NO running header or footer, and adding one is a trap.** The
  `position: fixed` trick that every search result recommends does repeat on
  every printed page and does NOT stay in the margin: on a real three-page print
  it painted at a fixed offset over a table row on page 2 and over a blockquote
  on page 3. **Page one was perfect**, which is the whole problem — nothing short
  of printing past one page and looking at it would catch it. The test suite
  asserts the absence of `position: fixed` for that reason. A real running footer
  needs `footerTemplate`, which needs the DevTools socket, which this path does
  not open.
- **Axes are snapped to round numbers** (`niceScale`). Dividing a data range into
  four equal parts is arithmetically right and reads as noise — a real print came
  out labelled 2.61 / 4.38 / 6.15 / 7.92 / 9.69. Gridlines are a ruler, and a
  ruler is marked in round units. Related: a line chart gets a **padded** range,
  not a forced zero. Zero-anchoring is a BAR rule, because a bar's length IS the
  value; forcing it onto lead times that never approach zero spent a quarter of
  the figure on empty space and flattened the crossing the chart existed to show.
- **A pie labels only slices at or above 8%.** At 4% the labels on a 5.2% and a
  6.3% slice collided against the edge. Every slice's exact value and share is in
  the key beside the figure, so a label on a thin slice costs legibility and buys
  nothing.
- **The report must be self-contained.** The printer opens a `file://` URL with
  no network. A stylesheet link, a webfont or a CDN script would fail silently
  and the PDF would print unstyled — **a failure that still produces a file**, so
  nothing errors and nobody finds out until she opens it. Fonts are named
  explicitly (Liberation, DejaVu, Nimbus) rather than left to `system-ui`, which
  under a headless browser with no desktop resolves to whatever fontconfig
  fancies.
- **Charts are our own SVG** (`db/charts.js`), for the same reason: a charting
  library behind a `file://` URL is a blank rectangle. The categorical palette is
  a fixed eight-slot order validated for colour-vision deficiency **in that
  order**; slots are assigned by position and never cycled, and a seventh
  category folds into "Other" rather than getting a colour nobody can tell from
  another. Every mark carries a visible label, which is what makes three of the
  slots legal below 3:1 against white. A chart that cannot be drawn falls back to
  a table of its own numbers — never to a hole where a figure was promised.
- **The file is never the only copy, and never the only way to reach it.**
  `result_text` stays in the row and the file is made from it, so a full disk or
  a deleted file costs the formatting and never the work. And the card carries a
  **download link** as well as the folder: the folder is on the server and she is
  usually on a laptop, so a path alone is a fact about a machine she is not
  sitting at. Both halves are the feature.
- **The download route takes a JOB ID and never a path.** `GET /api/jobs/:id/file`
  looks the location up in the row; there is no parameter that names a file and
  therefore nothing to traverse out of. A `?path=` version — "so it can link to
  older files too" — would be a directory traversal with a rate limit on it.
- **`attachArtifact` runs after `finish()`, and cannot fail the job.** `finish()`
  is the one write that ENDS a job and the invariant is that every exit through
  it writes exactly one terminal row. Making a file is neither terminal nor
  required, so the status is settled first and the file is a follow-up write that
  only adds columns. Same doctrine as the empty-card guard, applied to a new way
  of losing something.
- **A redirected process gets its own documents folder.** `SNH_DATA_DIR` does not
  move `data/config.json`, so a throwaway instance reads the real `outputDir` —
  and would write into her real documents folder. `outputDir()` therefore puts
  the folder inside the data directory whenever the redirect is set. A file on
  disk is live state, and that redirect exists so a disposable process cannot
  touch live state.
- **Scheduled runs produce no file, deliberately.** They share the panel with
  handed-off jobs, but a digest arrives on a cadence and one PDF per firing would
  silt the folder up with a hundred near-identical reports nobody asked for.

## ⚠️ One search tool, a provider chain, and every call on the record

`web_search` is one tool with two providers behind it (2026-08-18). **Exa's
`/search` first, SearXNG as the fallback**, order from `config.tools.search.order`,
tried in turn until one returns results — the same chain for chat and for agent
jobs, because one of them being quietly stale is exactly what nobody would notice.
The model never picks a provider; the routing is code (`mcp/tools/web-search.js`),
and offering the choice would double the schema for a question with one answer.

- **The key is a SECRET, not config.** `EXA_API_KEY` comes from the encrypted
  store or the environment (`db/secrets.js` — see *The Tools tab is generated, and
  a secret goes one way*), never from `data/config.json`, which is served by
  routes, written by the settings UI and copied into staging seeds. It is settable
  from the browser, so a fresh install needs no shell.
  `db/config.js` calls `dotenv` itself so scripts and cron entry points see `.env`
  too, not just `server.js`. `getSearchConfig()` is the single answer to "how does a
  search run right now": order, plus per-provider availability, where availability
  means the prerequisite as well as the flag. `web_search` registers when ANY
  provider is available, and with none it is absent exactly as before.
- **Search-endpoint only, enforced in code.** `type` is pinned to
  `auto`/`fast`/`instant` and any `deep*` value is refused by `resolveExaType()`
  and downgraded, with the refusal logged. Deep Search and the Agent endpoint do
  the multi-step research SNH does itself, with its own tools and its own memory,
  on its own GPU; buying that from an API moves the thinking off the machine. The
  free tier has no payment method, so it stops with **402** rather than billing —
  that 402 is surfaced in words ("the monthly credit is spent… nothing was
  billed"), never left as a status code to be rediscovered in six weeks.
- **Empty and broken are different facts.** A provider that worked and found
  nothing (`ok`, zero results) is a real answer; a provider that failed is not.
  Both fall through to the next provider, and both are logged distinctly. When
  **every** provider is empty the result says which were tried and that the answer
  is nothing — never a bare empty array, which is what gets filled in from memory.
- **Every provider ATTEMPT writes a row** to `search_call_log` (`db/search-log.js`)
  with provider, query, count, outcome, caller, latency and reported cost, sharing
  one `attempt_id` per tool call — so Exa-then-SearXNG reads as one search with two
  steps. A fallback that worked also writes an ops line, because a working fallback
  means the provider before it is failing and nothing else would say so. Read it
  with `node scripts/search-log.js`. This exists because on 2026-08-18 "was it Exa
  or SearXNG, and did it return anything" was answerable only by reading a journal
  by hand, and it cost hours.
- **`web_search` no longer has its own signature.** It is `(args, context)` like
  every other tool, and the special case in `MCPClient.executeTool` is **gone**.
  The old positional-string shape worked on the chat path and handed the context
  object to a URL base on every other path (`Failed to parse URL from [object
  Object]/search?q=…`, seven times in one job). Resolving the endpoint at the call
  site fixed the symptom and kept the defect: a contract one tool alone breaks is
  a contract the next call site forgets. Anything added to the registry whose
  `execute()` is not `(args, context)` is a bug in the tool.
- Verify with `node scripts/test-search-providers.js` (stubbed `fetch` — no
  credit is spent).

## ⚠️ The Tools tab is generated, and a secret goes one way

Two rules, and both are the same rule twice: **one fact, one place.**

**The page is derived from the registry.** `mcp/mcp-client.js` holds
`TOOL_CATALOGUE` — one row per tool, carrying its class, its gate, the config path
its switch writes, and its other settings as dotted paths. `loadConfig()` registers
from it and `describeCatalogue()` renders the settings page from it, so a tool
cannot exist without appearing in the UI. Before this, registration was a
hand-written if-chain and the Tools tab carried a list of its own: fourteen tools
were registered and **three** were on the page, and every tool shipped after the
page was written was invisible there. Three things to keep:

- **The page lists tools that are OFF.** A page built from what is *registered*
  loses the row for anything you switch off, which leaves no way to switch it back
  on. Each row says whether it is registered right now and, when it is not, why —
  "off because: turned off here", never a missing row.
- **A row's description is the tool's own.** `describeCatalogue()` reads
  `tool.description`, the text the model is given. A second human-facing copy would
  be a second thing to maintain, and the two would drift.
- **A row with no switch of its own says what decides it instead.** `web_search` is
  gated on a provider being available, `web_fetch` rides on `web_search`, and the
  corrector's three writes are always registered but reachable only by a step
  `corrector.enabled` governs. An unexplained missing control reads as a broken
  page. Verify with `node scripts/test-tools-settings.js`, which adds a dummy tool
  to the catalogue and asserts a fifteenth row appears with no page edit.

**Secrets are write-only, encrypted, and not in config.** `db/secrets.js` holds
them in `data/secrets.json` — AES-256-GCM per secret, random IV, and the secret's
NAME as additional authenticated data, so a ciphertext moved to another slot fails
rather than becoming that other secret. The key is 32 bytes in `data/.secret-key`
(mode 0600) or from `SNH_SECRET_KEY` where a platform injects one. Both files sit
beside `config.json` and are NOT redirected by `SNH_DATA_DIR`, for the reason
config is not: a throwaway instance that could not see the key would search
differently from live, and a verification against it would measure the wrong thing.
`SNH_SECRETS_PATH` / `SNH_SECRET_KEY_PATH` move them where a deployment or a test
needs that (`scripts/test-tools-settings.js` does, so it never opens the live ones).

- **Never in `data/config.json`.** `GET /api/config` returns that file whole, so a
  key in it is served to the browser on every settings load. A separate file means
  there is no redaction step to forget.
- **The server never puts any part of a value in a response.** No preview, no
  last-four, no length — status only: set or not, from where, when. "Write-only"
  survives the next convenience feature only if there is nothing to trim back to.
  The test asserts the key is absent from the real `GET /api/tools` payload,
  including any prefix of it.
- **Env still wins, and the UI SAYS so.** `process.env.EXA_API_KEY` overrides the
  store, so everything reading `.env` keeps working — and `status()` reports
  `envOverrides`, because a key typed into the UI while a stale one sits in `.env`
  is stored and then ignored, which is an afternoon nobody should lose.
- **What the encryption is worth, stated plainly:** it protects a copied data
  directory, a backup, a synced folder, a committed file. It does not protect
  against anything running as this user, since the key sits beside the data. Fixing
  that means a passphrase at every boot (no unattended restart, and systemd
  restarts this service) or an external KMS (not self-hosted). Do not claim more
  than that in the UI.
- **Only a DECLARED secret name is writable.** The route is not a general-purpose
  writer of environment-shaped keys into a file: a name has to come from a tool's or
  a provider's own declaration, which is also what makes its field appear.

**Off and second are different states.** `tools.<provider>.enabled` is the switch;
`tools.search.order` is the ordering. A provider switched OFF is dropped from the
chain entirely — not tried, and no "skipped" row on every search, because you
decided and there is nothing to report. A provider that is ON but cannot run (Exa
with no key) STAYS in the chain and is skipped **loudly**, because that is a
misconfiguration and the log is how anyone finds out. `getSearchConfig()` returns
`providers` (the live chain) and `allProviders` (every known provider with its
switch state) — the page needs the second one, or it cannot show you what is off.
It also takes an optional config argument, the same test seam
`selfFactSupersessionBar` uses: the chain logic cannot be tested by writing to the
live `data/config.json`.

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
