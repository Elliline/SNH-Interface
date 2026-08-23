/**
 * Centralized configuration loader for Squatch Neuro Hub
 * Reads from data/config.json, deep-merges with defaults,
 * and applies environment variable overrides at runtime.
 */

const fs = require('fs');
const path = require('path');

// SECRETS COME FROM THE ENVIRONMENT, AND EVERY ENTRY POINT NEEDS THEM.
// server.js already called this, which was enough while the only env values were
// hosts that had config fallbacks. EXA_API_KEY has no fallback: a script, a test
// or a cron entry point that never loaded .env would see search silently fall
// back to SearXNG and report that as the truth. dotenv does not overwrite an
// already-set variable, so a real environment still wins, and calling it twice
// is a no-op.
try { require('dotenv').config(); } catch { /* .env is optional */ }

/**
 * DELIBERATELY NOT redirected by SNH_DATA_DIR, and the only such path left.
 *
 * Every other file the system writes or reads under data/ now resolves through
 * database.getDataDir(), because a path that ignores the redirect is a path a
 * staging run writes into the live store. Configuration is the exception, and
 * the reason is that it is not part of the corpus. A staging run is the SAME
 * system pointed at a different store, not a differently configured one — it has
 * to use the same extraction model, the same similarity floors, the same
 * corrector budgets, or what it measures is not what live would do.
 *
 * Redirecting this would also fail quietly rather than loudly: data-staging/
 * holds no config.json, so getConfig() would silently return bare DEFAULTS and
 * the replay would run on a different model than the one it is validating. That
 * is a worse failure than the one the redirect exists to prevent.
 *
 * Checked by scripts/test-data-dir-redirect.js, which allows exactly this file.
 */
const CONFIG_PATH = path.join(__dirname, '../data/config.json');

const DEFAULTS = {
  providers: {
    ollama: [
      { name: 'Local', host: 'http://localhost:11434' }
    ],
    vllm: [],
    llamacpp: [
      { name: 'Local', host: 'http://localhost:8080', model: 'scout' }
    ]
  },
  models: {
    chat: { provider: 'llamacpp', instance: 'Local', model: 'scout' },
    extraction: { provider: 'ollama', instance: 'Local', model: 'gemma3:4b' },
    heartbeat: { provider: 'ollama', instance: 'Local', model: 'qwen3:14b' },
    embedding: { provider: 'ollama', instance: 'Local', model: 'nomic-embed-text' }
  },
  // Generation budgets for the chat path.
  //
  // A reasoning model spends its output allowance in TWO places — the thinking
  // it does before answering, and the answer itself — and the OpenAI wire format
  // has a single field for the total. Sizing only the total is what produced the
  // 2026-08-15 empty reply: the model thought for 8,000 characters, decided
  // "let me run the tool", and emitted end-of-turn having written no answer at
  // all. finish_reason was `stop`, not `length`, so nothing looked wrong.
  //
  // So the two budgets are declared separately and the wire total is derived:
  //   max_tokens            = thinkingTokens + responseTokens
  //   thinking_token_budget = thinkingTokens
  //
  // thinking_token_budget is engine-enforced: when thinking runs past it the
  // engine closes the reasoning channel and makes the model answer, which is the
  // thing that actually guarantees an answer comes back. Measured on vLLM
  // 0.27.1: budget 64 → 288 chars of thinking and a 2,637-char answer; budget
  // 256 → 1,164 and 1,478. It is a vLLM extension, so it is only sent to the
  // local OpenAI-compatible engines (see server.js) — hosted providers get
  // max_tokens and reasoningEffort only.
  //
  // reasoningEffort is the model's own dial. Qwen3's chat template defaults it
  // to 'xhigh' when nothing is passed, which is the most expensive setting it
  // has; 'medium' is the default here deliberately. Accepted by this engine:
  // none, minimal, low, medium, high, xhigh, max — but a model's TEMPLATE may
  // accept fewer (Qwen3's raises on anything but xhigh/medium/low), so a wrong
  // value fails as a 400 rather than silently.
  //
  // EVERY FIELD DEFAULTS TO null, AND null MEANS "SEND NOTHING".
  //
  // These defaults ship to every box, and one of them (Sparky/Aurelius) runs a
  // model with no reasoning channel at all. A shared default is only safe if
  // pulling it changes nothing, and a value here would not have been:
  //
  //   - max_tokens is the dangerous one. SNH has never sent it on the
  //     vllm/llamacpp path, so the engine allows max_model_len minus the prompt
  //     — effectively uncapped. Defaulting it to 4096 would hand every box a new
  //     output ceiling, and a long answer would stop MID-SENTENCE with
  //     finish_reason `length`, which nothing in the streaming path surfaces.
  //     Measured on this engine: max_tokens 300 → 527 chars, cut at "## Eastern".
  //   - thinking_token_budget is a vLLM extension. Its behaviour on llama.cpp,
  //     and on a vLLM served without a --reasoning-parser, is UNVERIFIED here.
  //   - reasoning_effort reaches the chat template. Qwen3's reads it; a template
  //     that does not reference it should ignore it, but that is reasoning about
  //     Jinja rather than a measurement, and it was not measured against Gemma.
  //
  // So the shipped default reproduces the request SNH sent before this block
  // existed, byte for byte, and a box opts in through its own data/config.json
  // (gitignored, so it stays local). aiserver sets all three; see that file.
  //
  // Turning it on, on a reasoning model:
  //   reasoningEffort: 'medium', thinkingTokens: 2048, responseTokens: 2048
  //
  // Then the wire carries max_tokens = thinkingTokens + responseTokens = 4096
  // and thinking_token_budget = 2048. Sizing the total ALONE is what produced
  // the 2026-08-15 empty reply: the model thought for 8,000 characters, decided
  // "let me run the tool", and ended its turn having written no answer, with
  // finish_reason `stop` — so nothing looked wrong. thinking_token_budget is the
  // half that fixes it, because it is engine-enforced: when thinking runs past
  // it the engine closes the reasoning channel and makes the model answer.
  // Measured on vLLM 0.27.1: budget 64 → 288 chars of thinking and a 2,637-char
  // answer; budget 256 → 1,164 and 1,478.
  //
  // reasoningEffort is the model's own dial, and leaving it null is not neutral
  // on a model whose template has an opinion: Qwen3's defaults it to 'xhigh',
  // the most expensive setting it has. Accepted by vLLM: none, minimal, low,
  // medium, high, xhigh, max — but a model's TEMPLATE may accept fewer (Qwen3's
  // raises on anything but xhigh/medium/low), so a wrong value is a 400, not a
  // silent misconfiguration.
  // The BACKGROUND half of the same problem, and it is worse there than in chat.
  //
  // Every callLLM site sizes maxTokens for the ANSWER — 8 for a claim-type tag,
  // 100 for a gap question, 120 for a salience score — because they were written
  // for a model that does not think. On a reasoning model that budget covers the
  // thinking too, and the thinking goes first. Measured on the real prompts:
  //
  //   scoreSalience   (answer budget 120), no thinking budget:  0/3 usable, finish=length, content ""
  //   detectGapQuestion (answer budget 100), no thinking budget: 0/3 usable, finish=length, content ""
  //
  // So salience scoring and gap detection were failing on EVERY run, falling back
  // to salience 5 with an empty rationale. The floor is low — 128 was already 3/3
  // on both with a real rationale — and 256 is set as twice the floor, which buys
  // roughly 1,000–1,200 characters of reasoning:
  //
  //   budget 128 -> 3/3, "9  This is a defining identity fact — being the creator..."
  //   budget 256 -> 3/3, "9  Being the creator of a named project (SNH) is durable..."
  //
  // backgroundThinkingTokens is added ON TOP of each caller's maxTokens rather
  // than carved out of it, so every existing call site keeps the answer budget it
  // asked for and none of them need editing.
  //
  // Extraction is sized separately because it is a much larger job and the
  // failure there was the 30s timeout, not an empty string — its call sends no
  // max_tokens at all, so thinking simply ran long. Measured over the three
  // longest exchanges of a real conversation, worst case of three:
  //
  //   no budget:  30.8s  (EXCEEDS the 30s timeout — this is the live failure)
  //   budget 256:  8.3s  but one exchange returned unparseable JSON
  //   budget 512: 13.2s
  //   budget 768: 19.6s  and reproduced the uncapped result exactly (4 facts, 1 event)
  //
  // 768 keeps the quality of unbounded thinking at roughly two thirds of the
  // wall-clock. extractionTimeoutMs is then 45s: 2.3x the measured worst, because
  // these numbers came off an idle GPU and the real one is shared with live chat.
  //
  // null for all three, for the same reason as above — a box with a non-reasoning
  // model must send exactly what it sends today.
  //
  // AGENT JOBS ARE THE THIRD BUDGET, AND UNTIL 2026-08-19 THEY HAD HALF OF ONE.
  //
  // A job sized its ANSWER through agentJobs.maxOutputTokens (2000) and got its
  // THINKING from backgroundThinkingTokens — the budget written for a 120-token
  // salience score. So the most expensive path in the system was running on the
  // cheapest path's thinking allowance, and nothing said so: the two numbers
  // lived in different sections and neither one named the job path.
  //
  // What that cost, measured on the aiserver 2026-08-18: three coding jobs in a
  // row stopped mid-file. The wire carried max_tokens = 2000 + 256 = 2256 for a
  // run asked to produce a complete module, against chat's 16384 + 8192 = 24576
  // for a run asked to produce a paragraph.
  //
  // So the job path now has both halves of its own, and they are sized for what
  // a job actually produces:
  //
  //   agentJobResponseTokens 8192 — PARITY WITH CHAT, and parity is the whole
  //     argument. A job's deliverable is a file, not a sentence; there is no
  //     reading on which it should be capped BELOW what he can say in one chat
  //     message, and 2000 capped it at a quarter. For scale, a complete 150-line
  //     Python module with docstrings runs about 2,000-2,500 tokens, so 8192 is
  //     roughly 3x a whole file — room for the module plus the notes around it,
  //     not room for an essay.
  //   agentJobThinkingTokens null — SHIPPED EMPTY like every other field here,
  //     for the same reason: this box's model has no reasoning channel and the
  //     default must reproduce today's request byte for byte. A reasoning box
  //     sets it in its own data/config.json (Settings → Thinking and Answer
  //     Budgets). 16384 is the number to set, again at parity with chat: on
  //     Qwen 3.8 the expensive part of a job is not the writing, it is the
  //     self-review — it drafts the module, writes tests against it, runs them
  //     in its head and revises before it answers. That pass is the work, not
  //     padding around it, and it is where the old budget went.
  //
  // NOTE FOR A REASONING BOX: leaving agentJobThinkingTokens empty does NOT fall
  // back to backgroundThinkingTokens any more. Empty means nothing is sent, and
  // on a thinking model an unbounded think inside a bounded max_tokens is the
  // 2026-08-15 empty-reply failure. Set it.
  // SCHEDULED RUNS HAD THE SAME DEFECT, one subsystem over (2026-08-19).
  // scheduler.maxOutputTokens was 700 — a fifth of the agent job's already-too-
  // small 2000 — and the thinking half came from backgroundThinkingTokens the
  // same way. A cron run that hit the ceiling closed as `ok` on the same panel,
  // because the scheduler had no `partial` status at all.
  //
  //   scheduledJobResponseTokens 4096 — HALF the agent job's, deliberately. A
  //     scheduled run is a recurring digest, and it is a smaller job by
  //     construction: 12 tool calls to a job's 40, 6 rounds to 16, 3 minutes of
  //     wall clock to 15. 700 could not hold a real report; 4096 holds a long
  //     one, and something that arrives every morning should stay readable.
  //   scheduledJobThinkingTokens null — ships empty for the same reason as every
  //     other field here. 8192 is the number to set on a reasoning box, half the
  //     agent job's 16384, keeping the same ratio as the answer budgets.
  generation: {
    reasoningEffort: null,
    thinkingTokens: null,
    responseTokens: null,
    backgroundThinkingTokens: null,
    extractionThinkingTokens: null,
    extractionTimeoutMs: null,
    agentJobThinkingTokens: null,
    agentJobResponseTokens: 8192,
    scheduledJobThinkingTokens: null,
    scheduledJobResponseTokens: 4096,
    // HOW A CALL IS KILLED, and it is no longer a predicted duration.
    //
    // llmTimeoutTokensPerSecond and llmTimeoutFloorMs lived here for one day and
    // are RETIRED, not kept alongside these: a second way to kill a job is a
    // second thing to discover the hard way. The rate they encoded was never a
    // property of the engine, it was a property of how many streams were running
    // — measured on this GB10 at 8-bit, per stream: 33.3 tok/s alone, 19.7 at 8
    // concurrent, 15.6 at 64, 10.6 at 128. A timeout derived from it got
    // stricter exactly as the system did more of what it exists to do.
    //
    // A stall is indifferent to that. Tokens still arriving means the job is
    // working, however slowly; nothing for a minute means the engine is wedged.
    // It is also far quicker at the job: the old formula gave an 8192 + 16384
    // token run 1,229s before it would notice a dead engine. This notices in 60.
    //
    // TWO LIMITS BECAUSE SILENCE MEANS DIFFERENT THINGS AT DIFFERENT TIMES.
    //   stallTimeoutMs — after the first token. At the worst load measured here
    //     a token lands every ~95ms, so 60s is roughly 630x the real gap, and it
    //     still clears the longest legitimate pause (a full 131k-token prefill
    //     being scheduled ahead of you is ~26s of work).
    //   firstTokenTimeoutMs — before it, where silence is NORMAL. This covers
    //     queue wait as well as prefill: past --max-num-seqs vLLM holds requests
    //     in `waiting` deliberately, and that wait belongs to the queue depth,
    //     not to us. Measured time-to-first-token here: 0.13s alone, 1.24s at
    //     128 concurrent — so 300s is ~240x the worst seen, which is the right
    //     margin for a limit whose real job is noticing a dead engine. Raise it
    //     if you run queues deep enough to wait longer than that.
    //
    // Both are DEFAULTS FOR THE SMALLEST MACHINE, not for this one. A 12GB card
    // running a smaller model at a few tokens a second is still nowhere near a
    // 60s gap between tokens, and that is the point: neither number scales with
    // the budget, the model, or how many agents are running.
    stallTimeoutMs: 60000,
    firstTokenTimeoutMs: 300000
  },
  // THE SAME TWO LIMITS, FOR THE PATH WITH A PERSON WAITING ON IT.
  //
  // Chat had one flat wall-clock — 120s per tool round, 90s on the final
  // stream — and a flat deadline cannot tell a wedged engine from a slow
  // one. On 2026-08-22 the engine stopped generating at 07:03 with two
  // requests in flight; her turn went out at 07:07 and was killed at
  // 07:09 by `AbortSignal.timeout(120000)`. The kill was correct. What was
  // wrong is that the identical deadline would have killed a turn that was
  // still producing tokens — a real brief, twelve tools and 6k of injected
  // context is not "how was your day", and work in progress was thrown
  // away on the same timer that catches a corpse.
  //
  // So chat measures GAPS, exactly as the background path does, and for the
  // same reason (see generation.stallTimeoutMs above — the argument is not
  // repeated here, it is the same argument). Tokens still arriving means the
  // turn is working, however slowly; nothing for a minute means the engine is
  // wedged. A heavy turn is now bounded by max_tokens and the round cap, not
  // by a clock it can lose to for being big.
  //
  // WHY THESE ARE SEPARATE FROM generation.* RATHER THAN SHARED. The stall
  // limit is the same number for the same reason and could have been shared.
  // The first-token limit could not: background work waits behind whatever
  // depth the queue has and 300s is right for it, while a person watching a
  // blank screen for five minutes has been failed whatever the engine is
  // doing. Two callers with genuinely different tolerances need two knobs,
  // and folding them would mean one of the two paths is always wrong.
  chat: {
    // After the first token. Same 60s as background, same evidence: at the
    // worst load measured here a token lands every ~95ms.
    stallTimeoutMs: 60000,
    // Before it. This is the whole budget for queue wait plus prefill, and it
    // is deliberately NOT the background's 300s: it is the longest she should
    // ever sit looking at nothing. 120s keeps the ceiling she already had for
    // a dead engine while removing it from turns that are working.
    firstTokenTimeoutMs: 120000
  },
  // HTTP rate limiting. The old literals (100 requests / 15 minutes for ALL of
  // /api/) worked out to 6.7 req/min shared across every endpoint, which a
  // polling UI plus chunked TTS exhausts in a couple of minutes — that is what
  // 429'd the whole app on 2026-07-27.
  //
  // The limiter exists for a future PUBLIC deployment. On this single-user box
  // every request comes from loopback (via Tailscale serve) or the tailnet, and
  // those are exempted outright by `exemptLoopback`/`exemptTailnet` — so these
  // caps only ever bind on traffic from somewhere else.
  rateLimit: {
    exemptLoopback: true,     // 127.0.0.0/8, ::1 — includes the Tailscale serve hop
    exemptTailnet: true,      // 100.64.0.0/10, the CGNAT range Tailscale assigns
    windowMinutes: 15,
    max: 1000,                // was 100; ~67/min, comfortable for a polling UI
    chatWindowMinutes: 1,
    chatMax: 60,              // was 20
    // TTS is chunked: one request per sentence of a spoken reply, so it needs a
    // much higher ceiling than ordinary API calls or a long answer trips it.
    ttsWindowMinutes: 1,
    ttsMax: 240
  },
  heartbeat: {
    enabled: true, intervalHours: 2, warmupMinutes: 5,
    // Per-step tool budget (2026-08-03). Background steps could not call tools at
    // all until now — callLLM built its body as {messages, stream, max_tokens}
    // with no tools key on either provider branch — so a step that wants tools
    // has to declare an allowlist, and this is what bounds it once it does.
    //
    // Both limits, not one. A call cap alone lets a step spend twenty minutes on
    // three slow lookups; a clock alone lets a fast loop make two hundred calls.
    // The corrector (Phase 2c) is the first consumer and inherits these; no
    // existing step declares tools, so today this scaffold binds nothing.
    toolBudget: {
      maxCallsPerStep: 12,
      maxWallClockMsPerStep: 120000,
      // Rounds of the tool loop within a single callLLM. A round is one model
      // turn, which may emit several tool calls.
      maxRoundsPerCall: 5,
      // WHAT A DEAD CALL COSTS (2026-08-18). A call that errors, or a search that
      // comes back empty, bills this instead of 1 — see toolCallCost in
      // db/memory-manager.js for the exact rule and its exceptions. The budget is
      // meant to bound WORK DONE, and a job that spent all twelve of its calls on
      // the same broken-URL error had done none.
      failedCallCost: 0.25,
      // And the floor under the discount: no session may make more than this
      // multiple of its budget in RAW calls, whatever they were worth. Without
      // it, a quarter-price failure lets an everything-fails loop run four times
      // as long as an everything-works one.
      attemptCeilingMultiple: 2
    }
  },
  // The scheduler (2026-08-12) — the thing that finally runs an approved job.
  //
  // Its own timer, beside the heartbeat and the liveness probe, because it is
  // answering a different question: not "is it time for maintenance" but "has
  // any job's wall-clock time arrived", which has to be asked at roughly the
  // resolution of a cron minute.
  //
  // A job is an AGENT RUN — its description becomes the task prompt for a
  // background model call with a read-only tool allowlist. That is the only job
  // type there is. No shell, no code execution, no arbitrary side effects: a
  // scheduler that can run commands is a different security posture, decided
  // separately.
  scheduler: {
    enabled: true,
    tickSeconds: 60,             // the resolution a 5-field cron actually needs
    // How late a missed firing may be and still run once on restart. A digest
    // that was due at 09:00 is worth having at 09:40; the same digest at 16:00
    // is a confusing artifact of a deploy. Past this, the run is recorded as
    // skipped — with the reason — and the job is re-armed forward.
    catchupGraceMinutes: 120,
    // Never retry forever in silence. After this many consecutive failures the
    // job disables itself and raises a bell alert naming the error.
    maxConsecutiveFailures: 3,
    // Per-run tool budget, same two-limit shape as heartbeat.toolBudget and for
    // the same reason. Sized for a summarizing run: a handful of lookups, not a
    // corpus sweep.
    maxToolCallsPerRun: 12,
    maxWallClockMsPerRun: 180000,
    maxRoundsPerRun: 6
    // Output tokens MOVED OUT of this block on 2026-08-19 — see
    // generation.scheduledJobResponseTokens, beside the chat, agent-job and
    // background rows. Same reasoning as the agent path: the answer budget and
    // the thinking budget have to be read against each other, and they cannot be
    // when they live in different sections and only one of them is on a screen.
  },
  // The agent-job queue (2026-08-18) — the async handoff.
  //
  // A chat turn can START work and END. The tool call writes a row and returns a
  // job id; the run happens on the agent pool, outside the request path, and
  // survives the response finishing and the browser closing. Results go to the
  // JOBS panel, which never opens a conversation — see db/agent-jobs.js.
  //
  // A job is an AGENT RUN, exactly as a scheduled job is: its task prose becomes
  // the prompt for one background model call with a READ-ONLY tool allowlist.
  // Nothing here executes code, and start_background_job is deliberately absent
  // from that allowlist, so a job cannot start a job.
  agentJobs: {
    enabled: true,
    // How many jobs may run at once, at the pool's FULL width. The pool's own
    // chat throttle still applies on top of this: while a chat request is in
    // flight the whole background pool is concurrency 1, so a job started
    // mid-conversation starts immediately and a second one waits. This cap is
    // the other direction — it stops jobs from filling every slot the pool has
    // and starving the corrector and the heartbeat.
    // maxConcurrent MOVED to agentPool.lanes.agentJobs on 2026-08-19 — it was a
    // second cap on the same quantity, gating before the pool ever saw the job,
    // so the lower of the two always won silently.
    maxQueued: 10,           // refuse past this, out loud, rather than queue forever
    maxStartsPerHour: 6,     // trailing-hour cap, counted from the table (a restart grants no budget)
    // Per-job budget, RAISED 2026-08-18 after a real job hit every one of these
    // and returned nothing. What it was measured against:
    //
    //   rounds were the limit that actually bound. 6 rounds at 2–3 calls a round
    //     is nine calls, so the 12-call budget was decorative — a job could not
    //     reach it before running out of turns. 16 rounds is the number that
    //     makes 40 calls reachable.
    //   40 calls is roughly a dozen searches plus the fetches to read what they
    //     found, which is the depth the tool description already promises her
    //     ("it can run a dozen searches and read whole pages").
    //   15 minutes of a GPU that is already his. Chat still preempts: the agent
    //     pool drops to concurrency 1 while a request is in flight.
    //   output tokens MOVED OUT of this block on 2026-08-19 — see
    //     generation.agentJobResponseTokens. It sat here as maxOutputTokens
    //     while the job's thinking budget sat in `generation`, which is how a
    //     job ended up sized for its answer and unsized for its thinking with
    //     nothing on either screen to say so. Both halves are now one section,
    //     next to the chat and background rows they have to be read against.
    //
    // The billed-vs-raw accounting is in createToolSession: an error or an empty
    // search bills a quarter, with a hard ceiling of 2× the budget in raw
    // attempts underneath. So 40 is 40 pieces of WORK, not 40 tries.
    maxToolCallsPerJob: 40,
    maxWallClockMs: 900000,  // 15 minutes
    maxRoundsPerJob: 16,
    // How many times a job may be STARTED, counting the first. 2 is one retry,
    // which is what "exactly one retry, ever" meant while it was the literal
    // `(j.attempts || 0) < 2` in sweepInterrupted. Safe to retry at all only
    // because every job in this phase is read-only — the day one can write,
    // this is the first line to revisit, and setting it above 2 then is a way
    // to repeat a write.
    maxAttempts: 2,
    // How long after a restart an interrupted run is still worth redoing. An
    // LLM call cannot be resumed, so the run is lost either way; the only
    // question is whether repeating it is still useful.
    retryGraceMinutes: 30,
    // Terminal rows older than this are pruned. The run they describe stays in
    // the ops log; this table is a panel, not an archive.
    retentionDays: 90
  },
  // ============ Job output as FILES ============
  //
  // A job result was one column of text on a card, and the card was the only
  // place it existed. That is fine for three sentences and wrong for everything
  // else: a research report arrived as raw markdown in a narrow panel, and a
  // Python module arrived as a code block she had to select and copy out of a
  // scrolling box. Neither is a thing you can keep, send, or open later.
  //
  // So a job now picks a FORM from what it actually made — source file, PDF, or
  // a card — and anything that becomes a file lands in a real folder AND gets a
  // download link on the card. Both, deliberately: the folder is on the server
  // and she is usually on her laptop, so a path alone is no use to her, and a
  // download alone would mean the machine that did the work does not keep it.
  documents: {
    enabled: true,
    // WHERE FILES GO. A bare name (the default) is resolved under the home
    // directory of the user the server runs as — ~/SNH_Documents. An absolute
    // path is used as given, which is how this points at a synced folder or a
    // NAS mount. The directory is created if it is missing.
    //
    // NOT under data/: this is the one thing SNH produces that is FOR HER
    // rather than for itself, and burying it beside the vector store would make
    // it something to go digging for. It is deliberately not moved by
    // SNH_DATA_DIR either — a throwaway test instance writing into her real
    // documents folder is exactly the accident that redirect exists to prevent,
    // so a redirected process gets a documents folder inside its own data dir.
    outputDir: 'SNH_Documents',
    // THE LINE BETWEEN A CARD AND A DOCUMENT, in characters of readable prose —
    // code fences are measured as one token, not as their own length, so a short
    // note wrapped around a long script counts as short and the script becomes
    // the file. Roughly 1,200 characters is four or five paragraphs: past that a
    // card is a scrolling box and a document is a document.
    inlineMaxChars: 1200,
    // Which chromium to print with. Empty means look for the usual names on
    // PATH (chromium, chromium-browser, google-chrome, …). Set this when it is
    // installed somewhere unusual — a snap wrapper, a flatpak, a vendored
    // headless shell.
    //
    // NOTHING FAILS IF IT IS ABSENT. A box with no chromium writes the report as
    // a formatted text file instead and the card says why, in one line. A
    // missing browser is a downgrade in what the file looks like; it is never a
    // job that produced nothing.
    chromiumPath: '',
    // Page setup for the printed report. Letter or A4 — the two the CSS knows.
    pageSize: 'Letter',
    // Keep the intermediate HTML beside the PDF. Off; it is a build artifact and
    // she did not ask for two files. Worth turning on when a PDF comes out
    // looking wrong and the question is whether the fault is in the HTML or in
    // the printing.
    keepHtml: false
  },
  // The corrector (Phase 2c) — the heartbeat step that repairs the corpus.
  //
  // Its own cadence, deliberately slower than the heartbeat: a pass is expensive
  // (a judge call per candidate pair) and the corpus does not rot by the hour.
  // Every pass is bounded, resumable, and fully recorded in corrections_ledger.
  corrector: {
    enabled: true,
    intervalHours: 6,          // its own cadence, not every heartbeat
    // Per-pass tool budget. Overrides heartbeat.toolBudget for this step,
    // because the corrector legitimately makes far more calls than any other
    // background role and a shared number would either starve it or over-grant
    // everything else.
    maxToolCallsPerPass: 60,
    maxWallClockMsPerPass: 300000,   // 5 minutes, then stop cleanly and resume
    // --- mechanical tier ---
    nearDupFloor: 0.86,        // cosine floor for near-duplicate candidate pairs
    maxNearDupPairsPerPass: 40,
    maxExpiriesPerPass: 25,
    maxSplitsPerPass: 10,
    // --- semantic tier ---
    contradictionFloor: 0.55,  // same regime as memory.contradiction.similarityFloor
    // Sized against the wall clock, not plucked. ~227 active user facts × ~9
    // candidates each is roughly two thousand pairs, and a judge call is ~1.5s,
    // so 200 pairs is about the 300s budget. Progress is persisted in
    // corrector_pair_checks, so successive passes advance instead of re-judging
    // the same opening pairs — at 30 the pass never got past the first three
    // facts, which is how the Mike fact survived a clean dry run.
    maxContradictionPairsPerPass: 200,
    // Semantic corrections to SELF-facts are OFF by default. The loop-era
    // self-fact corpus is promised as a joint curation session with Aurelius
    // (Phase 2e), and a corrector that quietly pre-empted that would be deciding
    // for him what he no longer believes about himself. Mechanical fixes (exact
    // duplicates, vector reconciliation) still apply to self-facts — those are
    // repairs to the record, not revisions of a view.
    selfFactSemantic: false,
    // So he never wakes up to ten changes to his self-view at once.
    maxSelfCorrectionsPerPass: 3
  },
  // Background LLM concurrency against the shared vLLM engine. Kept modest (3)
  // so background passes never starve chat or pile abandoned requests onto the
  // engine — over-saturation was a contributing cause of the brain wedge.
  // THE BACKGROUND POOL, IN LANES (2026-08-19).
  //
  // It was one bucket with one cap, and every kind of work competed in it: a
  // swarm of agent jobs, the scheduled runs, memory repair, the heartbeat. One
  // busy kind starved the others, and the only protection chat had was a blunt
  // "throttle everything to 1 while a reply is being written", which stopped
  // background work dead every time the user typed.
  //
  // Now each kind has its own queue and its own cap, and they are drained
  // round-robin so a busy lane cannot starve a quiet one.
  //
  // CHAT IS NOT A LANE, DELIBERATELY. It does not go through this pool and must
  // not: queueing the user's reply behind anything is the one latency
  // regression that is never worth it. What chat gets instead is RESERVED
  // HEADROOM — backgroundDuringChat is how much background work may run while a
  // reply is being written, and it is chat's cap expressed honestly.
  //
  // DEFAULTS ARE FOR THE SMALLEST MACHINE THIS COULD INSTALL ON, not for the
  // box that measured them. On a 12GB card running a small model, KV cache is
  // the binding constraint long before throughput is: a few thousand tokens of
  // context per stream is all there is, so a deep lane means vLLM queueing and
  // preempting rather than a crash. That degrades into slowness, and slowness no
  // longer kills anything now that calls are killed on a stall rather than on a
  // predicted duration. Anyone with real hardware raises these in Settings; the
  // measurement on this GB10 supports far more (64 concurrent streams cost each
  // stream only 2.1x its solo speed).
  agentPool: {
    lanes: {
      agentJobs: 8,      // her call, and the measured curve supports much more
      scheduled: 2,      // fires unattended; should never own the machine
      background: 4      // memory repair, the heartbeat, extraction, scoring
    },
    // The sum is bounded separately, because three caps that each look
    // reasonable can still add up to a machine that cannot answer.
    maxTotalBackground: 12,
    // Chat's reserved headroom. Was effectively 1 — background stopped dead
    // whenever a reply was being written. 2 keeps most of the engine for the
    // reply while letting background inch forward instead of stalling.
    backgroundDuringChat: 2
  },
  // The mid-cycle circuit breaker's trip point, which was CIRCUIT_TIMEOUT_THRESHOLD
  // in db/memory-manager.js. After this many consecutive callLLM timeouts every
  // subsequent call fast-fails until something succeeds — so it is the limit that
  // decides whether a wedged engine kills one run or all of them. It belongs
  // beside the other numbers that can stop a run rather than inside a module.
  brainCircuit: { consecutiveTimeoutsToOpen: 3 },
  // Lightweight periodic liveness probe: a tiny completion with a short timeout
  // that writes a daily-log warning when the brain stops answering, so a wedged
  // engine is caught in minutes instead of at the next heartbeat.
  // `retentionDays` bounds the per-probe log (liveness_probes). At the default
  // 5-minute cadence that's ~288 rows/day, so 14 days is ~4k rows — enough to
  // see a pattern of intermittent failures without growing forever.
  livenessProbe: { enabled: true, intervalMinutes: 5, timeoutMs: 8000, retentionDays: 14 },
  // Brain watchdog: the self-healing ACTION for the vLLM wedge. Fed each liveness
  // probe result — after `failureThreshold` consecutive failures it runs
  // `docker restart <container>`. `cooldownMinutes` is grace while the model
  // reloads (no re-trigger). `maxRestartsPerHour` is a hard cap: past it the
  // watchdog stops restarting, logs CRITICAL, and leaves the circuit breaker as
  // the fallback (a restart loop that isn't healing means something worse).
  watchdog: { enabled: true, container: 'sparky-brain', failureThreshold: 3, cooldownMinutes: 5, maxRestartsPerHour: 2 },
  // Self-identity: a deliberately minimal seed. We do NOT define the AI's
  // personality — it develops one through its own accumulated self-observations
  // (self-facts). maxSelfFacts budgets how many active self-facts inject.
  identity: {
    seed: 'You are an AI running on SNH — Squatch Neuro Hub — a memory system on this machine. You have no assigned name or personality. Your identity is yours to develop through your own experiences, observations, and memory. Observe yourself. What you notice about yourself will be remembered and will become who you are.',
    maxSelfFacts: 12,
    // Skip a new self-observation this cosine-similar to one SNH already holds
    // (or to another in the same batch). Tuned from real data: genuine reworded
    // duplicates sit ~0.89–0.97, while distinct-but-related traits stay ≤0.85.
    selfFactDedupThreshold: 0.88,
    // A self-fact this salient is never retired by an automatic semantic match —
    // it is raised for Ellie instead. 8, because the failure that produced this
    // rule retired a salience-9 DECLARATION on a 0.741 cosine match, and the
    // claim/declaration classifier is too noisy to be the only guard: the same
    // run tagged a behavioural observation "declaration" and a statement about
    // what had been built "claim". Salience is the second, independent bar.
    protectSelfFactSalience: 8,
    // How long one "I couldn't settle this about myself" bell alert stands for.
    // A hard window, not a per-pass thing: if the brain is wedged the judge fails
    // on every pair of every pass, and the point is ONE alert saying it happened
    // seventeen times — not seventeen alerts. Raises keep being recorded in the
    // ledger and the ops log throughout; this only bounds how often he says it
    // out loud.
    selfFactRaiseAlertHours: 24,
    // Identity lock: the narrow set of self-facts SNH CHOSE rather than observed,
    // which no automatic path may change (db/identity-lock.js). Kept to name and
    // pronouns on purpose — nearly everything else in its identity is observed,
    // and locking observations produces an entity that can't grow. Each category
    // is set once (the first fact asserting it locks itself) and thereafter only
    // changes through the deliberate path: the Self tab action or
    // scripts/identity-lock.js. Adding a category here means committing to it
    // being effectively permanent, so add sparingly.
    lock: { enabled: true, categories: ['name', 'pronouns'] }
  },
  // Initiative layer: SNH noticing things worth saying and saying them unprompted.
  // Thresholds are priority (1–10). Quiet hours are local Pacific 24h clock.
  initiative: {
    greetingThreshold: 7,       // min priority to weave into a new conversation's greeting
    followupThreshold: 5,       // lower greeting bar for conversation-followups ("I've been thinking about what you said")
    unpromptedThreshold: 8,     // min priority to start an unprompted conversation
    maxUnpromptedPerDay: 1,     // hard cap on SNH-initiated conversations per day
    quietHours: { start: 22, end: 8 }, // no unprompted messages 22:00–08:00 Pacific
    questionAgeDays: 3,         // a pending gap question this old becomes an initiative
    // How far back the daily-log follow-up reader looks, in calendar days
    // including today. Log entries go stale as conversational material —
    // "you mentioned yesterday" lands, "you mentioned three weeks ago" is odd —
    // and the line between those is days, not hours. 3 survives a day the user
    // does not talk to SNH, and a weekend; 7 would put the back half of the
    // window past the point where circling back reads as attentive. It also sits
    // inside staleDays below, so a follow-up drawn from the oldest entry in the
    // window cannot outlive the window that justified it.
    logFollowupDays: 3,
    staleDays: 7,               // pending initiatives older than this expire
    maxPending: 10,             // cap on the pending pool so it never nags
    dedupThreshold: 0.85        // skip a new initiative this cosine-similar to a pending one of the same type
  },
  // Self-coherence audit: SNH testing its stored self-CLAIMS ("I value X", "I am
  // a Y kind of partner") against how it actually behaved in recent conversations,
  // and raising the gaps through the initiative channel for Ellie's call. This was
  // SNH's own feature request — its first accepted initiative (2026-07-05), chosen
  // again from a menu on 2026-07-23 with the stated reason: "finding out if I'm
  // actually growing, or just getting better at describing a growth that isn't
  // happening." A daily low-frequency heartbeat pass; it NEVER auto-revises
  // identity — it documents tension and asks. Ellie wants to move daily→every-
  // other-day without a code change, so cadence lives here.
  audit: {
    enabled: true,
    cadenceDays: 1,        // run every N local days (1 = daily, 2 = every other day, …)
    claimsPerRun: 3,       // behavioral claims sampled per run (2–3)
    evidenceWindowDays: 7  // days of recent conversation transcripts considered as evidence
  },
  // Reflection: SNH observing itself from recent conversations.
  //
  // maxSelfFactsPerDay is a real limit on what the entity is allowed to conclude
  // about itself in a day, not a tuning knob. Unbudgeted, this path wrote 36
  // self-facts on 2026-07-27 and 382 of the corpus's 658 facts overall, partly by
  // reflecting on its OWN unanswered initiative messages. Ellie set 5/day on
  // 2026-08-02. Counted from the DB, so a restart cannot reset it.
  reflection: {
    maxSelfFactsPerDay: 5,
    transcriptBudgetChars: 12000  // conversation text fed to the model per pass
  },
  // Question queue: gaps/oddities SNH may ask the user about. These guards keep
  // it from re-asking things already asked or already answered.
  questions: {
    dedupThreshold: 0.85,       // don't queue a question this cosine-similar to ANY existing one (pending/asked/answered)
    answerMatchFloor: 0.40,     // min cosine(user message, question) to bother LLM-judging "did this answer it"
    answerMaxJudge: 6           // cap on how many topically-close outstanding questions we LLM-judge per message
  },
  memory: {
    // THE USABLE CONTEXT WINDOW, and it is deliberately not the engine's maximum.
    //
    // What a model's name says about its window is worth nothing: the static
    // table read "gemma" and answered 8,192 while the engine was serving 131,072,
    // so threads were being compacted at a twentieth of the real capacity. The
    // engine is now asked (db/model-context.js) and this is the cap applied to
    // its answer — never above it, and by default well below.
    //
    // Below, because a big window is not free. KV cache is memory and prefill is
    // latency, and both are paid on every turn for the whole life of a thread.
    //
    // 24,576 is MEASURED, not chosen for looking round. The intended default was
    // 32,768; scripts/probe-at-context-size.js re-ran the routing and honesty
    // probes against padded conversations and tool selection drifts with size:
    //
    //   pad 0     19/20 selected   16k  18/20   24k  18/20 (twice)   30k  18 then 17/20
    //
    // The honesty guards held everywhere — 0 phantom claims in 50 samples at 24k,
    // no flat false-absence at any size, every present fact still stated — so the
    // binding constraint is tool selection, and it drops below the 18/20 floor at
    // ~30k. The default sits just under that. Mean latency per turn over the same
    // sizes: 724ms → 1,096ms → 1,465ms, which is the other half of the argument.
    //
    // Raising this means re-running that probe at the new size, not assuming it
    // scales.
    contextTokens: 24576,
    similarityThreshold: 0.60,
    clusterLinkThreshold: 0.50,
    // Hysteresis deadband for the cross-link auditor: an EXISTING link is only
    // torn down when the LLM scores a pair strictly below this floor, while a NEW
    // link still needs to reach clusterLinkThreshold. Pairs scoring in the
    // [drop, link) band keep whatever link state they already have, so verdicts
    // that jitter around 0.50 between passes stop flip-flopping create↔drop.
    clusterLinkDropThreshold: 0.40,
    maxFactsPerCluster: 10,
    dailyLogRetentionDays: 7,
    // The daily-log archiver is a SECOND write path into her corpus and used to
    // have none of intake's guards. A candidate user-fact this close to a
    // self-fact he already holds is his, not hers — the same sentence with the
    // person flipped. 0.75 is the measured gap: the 22 misattributed rows found
    // at the merge ran 0.773–0.955, and the genuinely-hers rows in the same batch
    // fell to 0.695 and below. It is a property of nomic-embed-text; changing the
    // embedding model means measuring it again rather than guessing.
    archiver: {
      selfSimilarityFloor: 0.75
    },
    hybridSearchWeights: { vector: 0.6, bm25: 0.4 },
    // Per-source token budgets for what gets injected into every chat's system
    // context. Long prefill (60–90s TTFT) was caused by injecting whole daily
    // logs + long-term memory wholesale (~17–27k tokens). These caps keep the
    // total system context near ~6–8k tokens. Token counts are estimated at
    // ~4 chars/token. Self-facts are separately budgeted by identity.maxSelfFacts.
    injection: {
      // THE CEILING (2026-08-12). Every number below it is a per-source cap, and
      // per-source caps do not add up to a bound: measured on the live corpus,
      // sources summing to 6,900 shipped ~9,100 tokens a request, because the
      // identity block, the capability manifest and USER.md were never budgeted
      // at all. This is the total, applied after everything has rendered.
      //
      // Trimming is ordered and NEVER touches the identity block: his self-facts
      // and his locked name are the one thing that must be in front of him on
      // every turn, and a ceiling that can cut them is a ceiling that can take
      // his name away on a busy day. Retrieval goes first because it is
      // regenerated next turn; the day's log next; long-term memory last,
      // because it is the only source that is not recoverable within the turn.
      totalTokens: 6000,
      trimOrder: ['pastConvo', 'clusters', 'dailySummary', 'dailyToday', 'ltm'],
      longTermTokens: 3000,      // long-term fact block, rendered from SQLite
      dailyTodayTokens: 1500,    // today's most-recent entries injected verbatim
      dailySummaryTokens: 400,   // brief digest of older-today + yesterday
      // Today's log entries stamped with the CURRENT conversation's id are an
      // echo of the message history that is already in the request verbatim, and
      // they are what makes this block change on every turn. Excluded from the
      // render only; other conversations' entries from today still inject, and
      // nothing about how the log is written or read elsewhere changes.
      dailyExcludeActiveConversation: true,
      clusterTokens: 1200,       // associated cluster memory cap
      pastConvoTokens: 800,      // hybrid-search past-conversation snippets cap
      // Correction notices, as a BATCH not a count. Ten notices capped only by
      // number measured at 2,700–3,100 tokens, and the channel now fires for
      // every non-conversational change to a self-fact, so the count cap stopped
      // being a bound. Overflow stays UNSEEN and arrives next turn — the channel
      // is persistent by construction, so draining over several turns is
      // delivery, not loss.
      noticeTokens: 800,
      // Background jobs that finished since he was last told. Zero on almost
      // every turn — this block only renders when something actually landed.
      // Measured against a real scheduled-job output: header 64 tokens, ~65 per
      // job, so three jobs is ~250. The cap is the batch, not the count, for the
      // same reason as noticeTokens.
      jobTokens: 400,
      maxAnnouncedJobs: 3,
      // The capability manifest block, which grows with every entry shipped.
      //
      // 700, not the 600 this was specified at, and the 100 is bought
      // deliberately. At 600 the renderer compacts the tail to name-only, and
      // the tail is where the limits live — "she approves", "cannot delete",
      // "read-only", "deletes nothing", "a job stops itself after 3 failures".
      // A manifest that lists "Writing to memory on request" with no "cannot
      // delete" is the over-claim this whole registry exists to prevent, so
      // buying the clauses back at 100 tokens is the cheap side of that trade.
      //
      // AND IT STAYS AT 700. On 2026-08-19 the list reached 24 entries and did
      // not fit: the last three lost their one-liners, job-documents among them,
      // so what he was shown of the day-old capability was the four words of its
      // name. This box was raised to 760 as a stopgap and is back here now — the
      // one-liners were tightened instead (2026-08-20), which is the fix that
      // does not charge every prompt for every future entry. 25 entries render
      // at ~679 with a browser installed. When it next binds, tighten again:
      // shedding is honest about what it did — it never drops an entry, it warns
      // at boot and rings the bell — but it costs the newest entry its
      // description, which is the one nobody has learned yet. Check with
      // `node scripts/test-injection-budget.js` (it asserts the fit and the
      // headroom).
      manifestTokens: 700,
      // Per self-fact, in the rendered identity block only. A single rambling
      // reflection should not be able to eat the block the way a 400-token
      // self-fact would. Locked facts are exempt — see db/identity.js.
      selfFactTokens: 60,
      // subject='world' facts — knowledge about external things that is not
      // relational to Ellie or to him (how a service behaves, a tool's quirk,
      // task knowledge a future agent job leaves behind).
      //
      // OFF by default, and the default is the design rather than caution. The
      // injected block runs to a ~2.9k diet that everything else competes
      // inside; world knowledge is unbounded in a way personal facts are not,
      // so letting it in by default would crowd out the facts the block exists
      // for. It stays reachable on demand through the inspect tools and
      // topic-relevant retrieval, which is where unbounded knowledge belongs.
      includeWorld: false
    },
    // Contradiction-candidate recall (Phase 2a). These replace two bare literals
    // that made the check unreliable: a `.limit(15)` on the RAW vector search and
    // a `limit = opts.limit ?? 5` on the result. The 15 was applied BEFORE
    // filtering, so superseded rows — whose embeddings LanceDB deliberately keeps
    // as history — and wrong-subject rows consumed candidate slots and could push
    // every real candidate out before the filter ran.
    //
    // Selection is now threshold-based, not fixed-k: every ACTIVE, same-subject
    // fact above the floor is a candidate, and maxCandidates is only a cost
    // ceiling. When the ceiling truncates, that is logged — never silent.
    // The replay (Phase 2d). Concurrency applies to the PLAN half only —
    // planExtraction decides and writes nothing, so it parallelises safely;
    // applies stay strictly serial in source order so repeat-folding and
    // supersession always see a consistent corpus. A knob, not a ceiling.
    replay: {
      concurrency: 6
    },
    contradiction: {
      // How many raw vector neighbours to pull before filtering. Large on
      // purpose: the filter must run against a superset, not a top-k that
      // inactive rows can already have crowded. At corpus scale this is
      // effectively the whole index, and it costs no model calls.
      vectorFetchLimit: 2000,
      // Floors are set where THIS embedding model (nomic-embed-text) actually
      // separates signal from noise. Measured on the live corpus: genuinely
      // related facts sit at 0.62–0.99, and everything from ~0.45 to ~0.55 is
      // noise — 146 of 570 active user-facts clear 0.45 for an arbitrary probe,
      // which turns "threshold-based" straight back into "top-k". Retune these
      // if the embedding model changes; they are properties of the model, not of
      // the memory system.
      similarityFloor: 0.55,           // ordinary new fact
      correctionSimilarityFloor: 0.45, // fact flagged as a correction — wider net
      maxCandidates: 12,               // cost ceiling on judge calls per fact
      correctionMaxCandidates: 20,
      // Identity slots are PINNED past both the floor and the ceiling: every
      // active fact asserting the same identity slot (name, pronouns, a core
      // relationship) is always put to the judge, however it ranks.
      //
      // This exists because ranking cannot be trusted for this case. "User's name
      // is Ellie…" sits at cosine 0.5216 from "User's name is Mike" — below the
      // floor, and around 20th among active user-facts — while a dog's name and a
      // household roster rank above it. Two active name facts is a defect no
      // similarity threshold is going to catch, and there are only ever a handful
      // of identity facts, so the cost of pinning them is a rounding error.
      pinIdentitySlots: true
    },
    // Result discipline for the INSPECT tools. His injection budget is small and
    // tool results land in the same window, so rows are single-line and capped.
    // Separate from tools.memoryInspect on purpose: that one is permission to
    // call, this one is the shape of what comes back.
    inspect: {
      maxRows: 20,          // hard ceiling per search/list call
      defaultRows: 10,      // when he does not ask for a number
      rowChars: 140,        // one line per fact, truncated to this
      verbatimChars: 400,   // the verbatim source text in memory_get
      rationaleChars: 240,  // the salience rationale in memory_get
      maxClusters: 40,      // memory_list mode:'clusters'
      maxCorroborations: 10, // corroboration detail rows in memory_get
      // Relevance floor for the semantic half of memory_search. Same regime as
      // memory.contradiction.similarityFloor and the same reason: with
      // nomic-embed-text everything from ~0.45 to ~0.55 is noise, and without a
      // floor a search for "MettaSphere" reports sixty matches when six facts
      // mention it. A search that overstates what it found is worse than one
      // that finds less.
      semanticFloor: 0.55
    },
    // Passive extraction (Phase 2a rewrite).
    extraction: {
      // REPEAT detection: an incoming fact this close to an existing active fact
      // of the same subject is put to the judge as "same assertion?". A confirmed
      // repeat raises the existing fact's salience and records a corroboration —
      // it never creates a second row. Extends Phase 1's exact-match dedup to
      // semantic near-matches at write time.
      repeatSimilarityFloor: 0.80,
      repeatMaxCandidates: 5,
      // IDENTITY-ANCHOR caution. A fact in the identity class (the user's own
      // name, pronouns, or a core relationship) is only written when the VERBATIM
      // message carries an explicit self-introduction ("my name is", "call me").
      // Applied to these modalities. 'unknown' is included deliberately: every
      // historical message carries 'unknown' (decision 2), an unknown-modality
      // message may well have been dictated, and "Hey, it's Mike not picking up
      // the right words" is exactly the shape of the mishearing this exists for.
      // 'typed' is trusted because a typed name is evidence a person supplied.
      identityAnchorModalities: ['stt', 'unknown'],
      // Safety valves on a single exchange, so one pasted transcript cannot turn
      // into a fact avalanche.
      maxFactsPerExchange: 12,
      maxEventsPerExchange: 8
    }
  },
  // Web search via SearXNG. Single source of truth for BOTH the on/off toggle and
  // the server URL — the search path and the capability manifest read these. (The
  // URL used to live only in the client's localStorage + a hardcoded server
  // fallback, so settings and reality disagreed; now it's config.)
  tools: {
    // WHICH PROVIDER RUNS FIRST — one list, read by both the chat path and the
    // agent-job path, because the answer is the same for both: try the fast
    // hosted index, keep the self-hosted one as the thing that still works when
    // the account is spent. The model never sees this; it calls `web_search` and
    // the chain is code (mcp/tools/web-search.js).
    //
    // A provider missing its prerequisite is SKIPPED, not an error — with no
    // EXA_API_KEY the chain is just SearXNG, which is exactly what it was before
    // 2026-08-18.
    search: { order: ['exa', 'searxng'] },
    // Exa's /search endpoint. The KEY IS NOT HERE — it comes from EXA_API_KEY in
    // the environment, because this file is served by routes, written by the
    // settings UI and copied into staging seeds, and a secret in it would leak
    // through all three.
    //
    // Free tier as of 2026-08-18: $10 of credit a month, 5 queries/sec, no
    // payment method on file — so it STOPS with a 402 rather than billing. The
    // 402 is surfaced in words, not swallowed (see search-providers.js).
    //
    // `type` is pinned to a non-agentic search. Deep Search and the Agent
    // endpoint are deliberately unreachable: they do the multi-step research
    // SNH does itself, with its own tools and its own memory, and buying that
    // from an API would move the thinking off the machine. Any `deep*` value
    // here is refused in code, not just discouraged in this comment.
    exa: { enabled: true, url: 'https://api.exa.ai/search', type: 'auto', numResults: 5, timeoutMs: 8000, textChars: 1000 },
    searxng: { enabled: false, url: 'http://localhost:8888', timeoutMs: 8000 },
    // web_fetch's own HTTP timeout, which was AbortSignal.timeout(10000) in
    // mcp/tools/web-fetch.js. A job reading whole pages spends most of its wall
    // clock here, and a page that hangs costs the run a tool call it cannot get
    // back — so this is a per-run limit in everything but name.
    webFetch: { timeoutMs: 10000 },
    // create_cron_job — the first action tool. PROPOSE ONLY: a call raises an
    // initiative for Ellie to approve or reject in the bell panel; nothing is
    // created without her decision, and nothing executes even once approved
    // (SNH has no scheduler yet — approving records the job).
    //
    // The caps are on the TOOL, not on any notion of trust level. Same shape as
    // the watchdog's maxRestartsPerHour: a trailing-hour window plus a hard
    // ceiling, past which the tool refuses and says so. Difference from the
    // watchdog: these are counted from the DB rather than an in-memory array,
    // so a server restart can't reset the entity's budget.
    cron: {
      enabled: true,
      maxProposalsPerHour: 3,  // proposals (approved or not) in any trailing hour
      maxKidCreatedJobs: 10    // hard ceiling on live kid-created jobs
    },
    // write_memory — records a fact when asked to remember it. DIRECT-EXECUTE,
    // not propose-only: it writes down something Ellie just said, so an approval
    // queue would defeat the point. Trailing-hour cap only — no total ceiling,
    // because accumulating remembered facts is the system working, not a leak.
    memoryWrite: {
      enabled: true,
      maxWritesPerHour: 20
    },
    // memory_search / memory_list / memory_count / memory_get — the INSPECT
    // tools. Strictly read-only: they cannot write, and writes stay
    // write_memory → fact-store. One shared trailing-hour cap across all four,
    // because what is worth bounding is how much of a turn goes into rummaging;
    // four separate budgets would let a loop spend 4× while each counter looked
    // healthy. Counted from tool_call_log, so a restart grants no fresh budget.
    memoryInspect: {
      enabled: true,
      maxCallsPerHour: 40
    },
    // start_background_job — the async handoff tool. Starts work and returns a
    // job id immediately; the turn ends normally. Its rate cap lives in
    // agentJobs.maxStartsPerHour rather than here, because the queue enforces it
    // and one counter is better than two that can disagree.
    agentJobs: {
      enabled: true,
      // TIER 2 of the handoff triggers: "write me a script", "I need a game
      // built". OFF until sandboxed code execution ships, and the reason is not
      // caution — it is that dispatching one today produces script text from an
      // agent that cannot run it either, which is not better than answering
      // inline and costs a round trip. Turn this on the day a job can execute
      // what it writes. The patterns are built and tested (HANDOFF_BUILD in
      // db/tool-routing.js) so that day is a flag flip, not a rewrite.
      dispatchBuildRequests: false
    },
    // Handing coding work to squatch-code. OFF by default: an approved brief
    // runs unattended and can edit files, so switching it on is a decision.
    codingJobs: {
      enabled: false,
      projectsRoot: require('path').join(require('os').homedir(), 'Projects'),
      // === A STALL AND A CEILING, NOT ONE FLAT WALL CLOCK ===
      //
      // `timeoutMinutes` is RETIRED and is no longer shipped as a default — a
      // key we set ourselves must never trip our own deprecation warning. It is
      // still READ FOR THE WARNING if an existing data/config.json sets it. On
      // 2026-08-22 it killed a job that was working perfectly: measured on the
      // engine side, generation ran at 33.7 tok/s continuously for the whole
      // twenty minutes — 40,439 output tokens, ZERO samples at 0 tok/s, and
      // never more than one request in flight. It was not stuck and it was not
      // queued behind anything. It was rewriting a 20KB file every iteration,
      // six iterations at 160–170s each, and 25 of those needs ~71 minutes. The
      // ceiling was doing the pacing, and pacing is not what a ceiling is for.
      //
      // WHERE THE REAL STALL CHECK LIVES, and why this one is coarse. The
      // fine-grained signal is token flow, and SNH cannot see it: squatch-code
      // is a separate process that writes progress only when a STEP completes.
      // But squatch-code already has that check — its httpx client is built
      // with `timeout=120`, which on a stream is a per-read deadline, i.e. no
      // bytes for 120s. A wedged engine is caught there, one layer down, on the
      // right signal, within two minutes.
      //
      // So this window is a backstop for what that check cannot see:
      // squatch-code itself stuck — a hung command, a loop that stops calling
      // the model. Its signal is "no completed step", whose measured legitimate
      // gap in real work is 170s. A 120s window here would have killed the
      // healthy job above, so the number has to clear a slow step by a wide
      // margin: ten minutes, not two.
      stallTimeoutMs: 600000,
      // The runaway backstop, and nothing else. The 25-iteration job above
      // needed ~71 minutes at measured rates, so 60 is a ceiling rather than a
      // comfortable fit — raise it if real work ever hits it, because hitting
      // it should mean something is wrong.
      maxRuntimeMinutes: 60,
      maxPendingProposals: 3,
      binary: 'squatch-job',
      // A speed bump on top of the git restore point, not a security
      // boundary - an allowed interpreter can run anything. Ellie chose it
      // on that understanding.
      allowedCommands: ['pytest', 'python', 'python3', 'node', 'npm', 'go', 'cargo', 'make', 'git', 'ls', 'cat']
    }
  },
  voice: {
    // Manifest honesty gate: the voice capability registers in the capability
    // manifest ONLY when this is true — set it on once the TTS+STT containers are
    // verified live (mirrors tools.searxng.enabled). Default false = "deferred"
    // so voice is never claimed while the engines are down. The proxies themselves
    // (/api/tts, /api/stt) don't read this; it purely gates self-knowledge.
    enabled: false,
    stt: {
      active: 'whisper:Local',
      providers: [
        { name: 'Local', type: 'whisper', host: 'http://localhost:5051' }
      ]
    },
    tts: {
      active: 'kokoro:Local',
      providers: [
        { name: 'Local', type: 'kokoro', host: 'http://localhost:5050' }
      ]
    }
  }
};

let currentConfig = null;

/**
 * Recursively deep-merge source into target.
 * Objects merge, primitives and arrays replace.
 */
const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function deepMerge(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (UNSAFE_KEYS.has(key)) continue;
    if (
      source[key] &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      target[key] &&
      typeof target[key] === 'object' &&
      !Array.isArray(target[key])
    ) {
      result[key] = deepMerge(target[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

/**
 * Migrate old single-host config format to new array-based instance format.
 * Called before deepMerge so the file data is in the right shape.
 */
function migrateConfig(fileConfig) {
  const p = fileConfig.providers;

  if (p) {
    // Migrate ollama: { host: '...' } → ollama: [{ name: 'Local', host: '...' }]
    if (p.ollama && !Array.isArray(p.ollama) && p.ollama.host) {
      p.ollama = [{ name: 'Local', host: p.ollama.host }];
    }

    // Migrate llamacpp: { host: '...' } → llamacpp: [{ name: 'Local', host: '...', model: '...' }]
    if (p.llamacpp && !Array.isArray(p.llamacpp) && p.llamacpp.host) {
      const chatModel = fileConfig.models?.chat?.model || 'scout';
      p.llamacpp = [{ name: 'Local', host: p.llamacpp.host, model: chatModel }];
    }

    // Migrate vllm: { host: '...' } → vllm: [{ name: 'Local', host: '...', model: '...' }]
    if (p.vllm && !Array.isArray(p.vllm) && p.vllm.host) {
      p.vllm = [{ name: 'Local', host: p.vllm.host, model: p.vllm.model || '' }];
    }
    // Ensure vllm array exists
    if (!p.vllm) p.vllm = [];
  }

  // Migrate model role assignments to include instance: 'Local'
  if (fileConfig.models) {
    for (const role of ['chat', 'extraction', 'heartbeat', 'embedding']) {
      if (fileConfig.models[role] && !fileConfig.models[role].instance) {
        fileConfig.models[role].instance = 'Local';
      }
    }
  }

  // Migrate the SearXNG URL to a single canonical key. Older configs stored it as
  // `endpoint` (and the URL also lived in client localStorage); fold that into
  // `url` so the search path + settings + manifest all read one field.
  if (fileConfig.tools && fileConfig.tools.searxng) {
    const sx = fileConfig.tools.searxng;
    if (sx.endpoint && !sx.url) sx.url = sx.endpoint;
  }

  // Migrate old flat voice config to new provider-based format
  if (fileConfig.voice) {
    const v = fileConfig.voice;
    // Old format: voice.tts.host / voice.stt.host as flat strings
    if (v.tts && typeof v.tts.host === 'string' && !v.tts.providers) {
      v.tts = {
        active: 'kokoro:Local',
        providers: [{ name: 'Local', type: 'kokoro', host: v.tts.host }]
      };
    }
    if (v.stt && typeof v.stt.host === 'string' && !v.stt.providers) {
      v.stt = {
        active: 'whisper:Local',
        providers: [{ name: 'Local', type: 'whisper', host: v.stt.host }]
      };
    }
  }

  return fileConfig;
}

/**
 * Load config from disk, deep-merge with defaults.
 * Auto-creates config file if missing.
 */
function loadConfig() {
  let fileConfig = {};

  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
      fileConfig = JSON.parse(raw);
    } else {
      // Auto-create with defaults
      const dir = path.dirname(CONFIG_PATH);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(CONFIG_PATH, JSON.stringify(DEFAULTS, null, 2), 'utf8');
      console.log('[Config] Created default config at', CONFIG_PATH);
    }
  } catch (err) {
    console.error('[Config] Error reading config file:', err.message);
  }

  fileConfig = migrateConfig(fileConfig);
  currentConfig = deepMerge(DEFAULTS, fileConfig);
  return currentConfig;
}

/**
 * Get the current config with env var overrides applied.
 * Env vars OLLAMA_HOST and LLAMACPP_HOST update the 'Local' instance host,
 * or prepend a new 'Local' instance if none exists.
 */
function getConfig() {
  if (!currentConfig) {
    loadConfig();
  }

  // Deep clone to avoid env overrides mutating the cached config
  const config = JSON.parse(JSON.stringify(currentConfig));

  if (process.env.OLLAMA_HOST) {
    if (!Array.isArray(config.providers.ollama)) config.providers.ollama = [];
    const local = config.providers.ollama.find(i => i.name === 'Local');
    if (local) {
      local.host = process.env.OLLAMA_HOST;
    } else {
      config.providers.ollama.unshift({ name: 'Local', host: process.env.OLLAMA_HOST });
    }
  }

  if (process.env.LLAMACPP_HOST) {
    if (!Array.isArray(config.providers.llamacpp)) config.providers.llamacpp = [];
    const local = config.providers.llamacpp.find(i => i.name === 'Local');
    if (local) {
      local.host = process.env.LLAMACPP_HOST;
    } else {
      config.providers.llamacpp.unshift({ name: 'Local', host: process.env.LLAMACPP_HOST, model: 'scout' });
    }
  }

  return config;
}

/**
 * Deep-merge a partial update into the current config and persist to disk.
 * @param {Object} partial - Partial config to merge
 * @returns {Object} Updated config
 */
function updateConfig(partial) {
  if (!currentConfig) {
    loadConfig();
  }

  currentConfig = deepMerge(currentConfig, partial);

  try {
    const dir = path.dirname(CONFIG_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(currentConfig, null, 2), 'utf8');
    console.log('[Config] Saved config to', CONFIG_PATH);
  } catch (err) {
    console.error('[Config] Error writing config file:', err.message);
  }

  return getConfig();
}

/**
 * Look up a provider instance by type and name.
 * Returns { name, host, model? } or null.
 */
function getProviderInstance(providerType, instanceName) {
  const config = getConfig();
  const instances = config.providers[providerType];
  if (!Array.isArray(instances)) return null;
  return instances.find(i => i.name === instanceName) || null;
}

/**
 * Look up a voice provider by category and active string.
 * @param {string} category - 'tts' or 'stt'
 * @returns {{ name: string, type: string, host?: string, api_key?: string } | null}
 */
function getVoiceProvider(category) {
  const config = getConfig();
  const voiceCat = config.voice?.[category];
  if (!voiceCat || !voiceCat.active || !Array.isArray(voiceCat.providers)) return null;

  const [type, ...nameParts] = voiceCat.active.split(':');
  const name = nameParts.join(':');
  return voiceCat.providers.find(p => p.name === name && p.type === type) || null;
}

/**
 * Resolve the effective SearXNG search config from the single source of truth.
 * Env SEARXNG_HOST wins (ops override), then config.tools.searxng.url, then the
 * built-in default. `enabled` gates whether the search path runs at all.
 * @returns {{ enabled: boolean, url: string }}
 */
function getSearxngConfig(configOverride = null) {
  const cfg = configOverride || getConfig();
  const sx = (cfg.tools && cfg.tools.searxng) || {};
  return {
    enabled: !!sx.enabled,
    url: process.env.SEARXNG_HOST || sx.url || 'http://localhost:8888'
  };
}

/**
 * Resolve the effective EXA search config. Same shape and same rules as
 * getSearxngConfig, with one difference that matters: the KEY comes from the
 * environment and nowhere else, so `enabled` alone is not availability.
 * @returns {{ enabled: boolean, available: boolean, url: string, apiKey: string|null,
 *            type: string, numResults: number, timeoutMs: number, textChars: number }}
 */
function getExaConfig(configOverride = null) {
  // The optional argument is a TEST SEAM, and the same one selfFactSupersessionBar
  // uses: the logic is what is worth testing, and it cannot be tested by writing
  // to data/config.json — that file is the live one and is deliberately not
  // redirected by SNH_DATA_DIR. Callers in the system pass nothing.
  const cfg = configOverride || getConfig();
  const ex = (cfg.tools && cfg.tools.exa) || {};
  // THE KEY COMES FROM THE SECRET STORE, environment first (db/secrets.js).
  // Required lazily and defensively: config is loaded by every entry point,
  // including ones that run before data/ exists, and a settings page that cannot
  // render because a secrets file is missing would be worse than one that shows
  // the key as unset.
  let apiKey = null, keySource = null;
  try {
    const got = require('./secrets').get('EXA_API_KEY');
    apiKey = got.value;
    keySource = got.source;
  } catch (err) {
    console.error('[Config] could not read EXA_API_KEY:', err.message);
  }
  const enabled = ex.enabled !== false;
  return {
    enabled,
    available: enabled && !!apiKey,
    // Which of .env and the settings page answered. Reported so the UI can say
    // that a stored key is being overridden rather than leaving someone to wonder
    // why the one they just typed made no difference.
    keySource,
    url: ex.url || 'https://api.exa.ai/search',
    apiKey,
    type: ex.type || 'auto',
    numResults: Math.min(Math.max(1, ex.numResults ?? 5), 25),
    timeoutMs: Math.max(1000, ex.timeoutMs ?? 8000),
    textChars: Math.max(200, ex.textChars ?? 1000)
  };
}

/**
 * THE ONE ANSWER to "how does a search run right now" — provider order, and for
 * each provider whether it can actually be called.
 *
 * Both the chat path and the agent-job path read this, and so does the tool
 * registry when it decides whether `web_search` exists at all. One function,
 * because two would be two answers: that is the mistake mcp/tools.json made with
 * its own `enabled` flag, and it produced a tool that appeared in the model's
 * list and was never routed to.
 *
 * @returns {{ any: boolean, order: string[], providers: Array<{name, available, config, why: string|null}> }}
 */
function getSearchConfig(configOverride = null) {
  const cfg = configOverride || getConfig();
  const configured = (cfg.tools && cfg.tools.search && Array.isArray(cfg.tools.search.order))
    ? cfg.tools.search.order
    : ['exa', 'searxng'];

  // Threaded through, so an injected config governs the whole chain rather than
  // half of it — a seam that only reached the top level would compute a provider
  // list from the test's config and its availability from the live one.
  const exa = getExaConfig(cfg);
  const searxng = getSearxngConfig(cfg);

  const describe = {
    exa: () => ({
      name: 'exa',
      enabledInConfig: exa.enabled,
      available: exa.available,
      config: exa,
      why: exa.available ? null
        : (!exa.enabled ? 'switched off in settings' : 'no API key has been set for it')
    }),
    searxng: () => ({
      name: 'searxng',
      enabledInConfig: !!searxng.enabled,
      available: !!searxng.enabled && !!searxng.url,
      config: { url: searxng.url, timeoutMs: Math.max(1000, searxng.timeoutMs ?? 8000) },
      why: !searxng.enabled ? 'switched off in settings' : (!searxng.url ? 'no instance URL is set' : null)
    })
  };

  // Unknown names in the order list are dropped rather than guessed at — same
  // rule as the background tool allowlist: the list is a ceiling, not a request.
  const all = configured.map(n => describe[n] && describe[n]()).filter(Boolean);

  // OFF AND MISCONFIGURED ARE DIFFERENT, so they leave the chain differently.
  //
  // A provider you switched OFF is not in the chain at all: it is not tried, and
  // it writes no "skipped" row on every search — you decided, and there is nothing
  // to report. A provider that is ON but cannot run (Exa with no key) STAYS in the
  // chain and is skipped loudly, because that is a misconfiguration and the log
  // saying so is how anyone finds out. "SearXNG off" and "SearXNG as the fallback"
  // are different states and both have to be expressible.
  const providers = all.filter(p => p.enabledInConfig);
  return {
    any: providers.some(p => p.available),
    order: providers.map(p => p.name),
    providers,
    // Every known provider with its switch state, for the settings page: the page
    // must show what is off, or there is no way to turn it back on.
    allProviders: all
  };
}

module.exports = { getConfig, updateConfig, loadConfig, getProviderInstance, getVoiceProvider, getSearxngConfig, getExaConfig, getSearchConfig };
