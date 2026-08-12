/**
 * Centralized configuration loader for Squatch Neuro Hub
 * Reads from data/config.json, deep-merges with defaults,
 * and applies environment variable overrides at runtime.
 */

const fs = require('fs');
const path = require('path');

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
      maxRoundsPerCall: 5
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
    maxRoundsPerRun: 6,
    // Output ceiling. A bell item is read in a panel, so a job that writes an
    // essay is a job nobody reads.
    maxOutputTokens: 700
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
  agentPool: { concurrency: 3 },
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
      clusterTokens: 1200,       // associated cluster memory cap
      pastConvoTokens: 800,      // hybrid-search past-conversation snippets cap
      // Correction notices, as a BATCH not a count. Ten notices capped only by
      // number measured at 2,700–3,100 tokens, and the channel now fires for
      // every non-conversational change to a self-fact, so the count cap stopped
      // being a bound. Overflow stays UNSEEN and arrives next turn — the channel
      // is persistent by construction, so draining over several turns is
      // delivery, not loss.
      noticeTokens: 800,
      // The capability manifest block, which grows with every entry shipped.
      //
      // 700, not the 600 this was specified at, and the 100 is bought
      // deliberately. Measured 2026-08-12 with the one-liners tightened as far
      // as they go with their scope clauses intact: 23 entries render at 670
      // tokens. At 600 the renderer compacts the last 8 to name-only, and those
      // 8 are where the limits live — "she approves", "cannot delete",
      // "read-only", "deletes nothing", "stops itself after 3 failures". A
      // manifest that lists "Writing to memory on request" with no "cannot
      // delete" is the over-claim this whole registry exists to prevent, so
      // buying the clauses back at 100 tokens is the cheap side of that trade.
      // (1,064 → ~670 is still a 37% cut.) Lower it and the compaction is
      // honest about what it did — it never drops an entry — but it will cost
      // qualifiers, so lower it on purpose or not at all.
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
    searxng: { enabled: false, url: 'http://localhost:8888' },
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
function getSearxngConfig() {
  const cfg = getConfig();
  const sx = (cfg.tools && cfg.tools.searxng) || {};
  return {
    enabled: !!sx.enabled,
    url: process.env.SEARXNG_HOST || sx.url || 'http://localhost:8888'
  };
}

module.exports = { getConfig, updateConfig, loadConfig, getProviderInstance, getVoiceProvider, getSearxngConfig };
