/**
 * Tool routing — decides whether a user message should enter the web-search /
 * fetch tool loop, or be answered directly from model knowledge + memory.
 *
 * Extracted from server.js so the decision is unit-testable in isolation
 * (see scripts/test-tool-routing.js). The chat endpoint calls
 * classifyToolNeed(); when it returns false the tool loop is skipped and the
 * model is streamed WITHOUT any tool schema — so a false negative here means
 * the model literally cannot search this turn even if it wants to. That is why
 * explicit user requests and recency signals are hard triggers below.
 */

/**
 * Explicit user requests to search. These are a HARD OVERRIDE — checked before
 * any "no tools needed" short-circuit, so "look it up", "search for it",
 * "check the web" route to tools regardless of whatever else the message says.
 *
 * Note the pronoun-tolerant forms: users write "look it up" / "look this up"
 * far more often than the bare "look up", and the old substring match for
 * 'look up' missed every one of them.
 */
const EXPLICIT_SEARCH_PATTERNS = [
  /\bsearch (for|the web|online|it up|that|this|around)\b/,
  /\bsearch\b.{0,20}\b(web|internet|online|google)\b/,
  /\bweb search\b/,
  /\blook (it|this|that|them|these|those|him|her|us) up\b/,
  /\blook up\b/,
  /\bgoogle (it|this|that|them)\b/,
  /\b(check|find|look) (it up )?(on )?(the )?(web|internet|interwebs|google|online)\b/,
  /\bon the (web|internet|interwebs)\b/,
  /\bcheck (the web|online|the internet)\b/,
  /\bfind out (online|on the (web|internet)|for me)\b/,
  /\bcan you (search|look .{0,10}up|google)\b/,
];

/**
 * Recency signals — "new", "latest", "just released", etc. On their own these
 * are ambiguous ("new to programming"), so they only trigger tools when paired
 * with an entity: a version number ("4.5", "v2.0", "version 3") or a proper
 * noun (a capitalized token that isn't the first word of the message). "new" +
 * proper noun / version is a strong "product release / current info" signal.
 */
const RECENCY_SIGNAL = /\b(new|newer|newest|latest|recent|recently|upcoming|just (released|launched|announced|dropped|unveiled|out|shipped))\b/;
const VERSION_TOKEN = /\b\d+(\.\d+)+\b|\bversion\s+\d|\bv\d+(\.\d+)?\b/i;

/**
 * True if `raw` contains a proper-noun-like token — a capitalized word that is
 * not the first word of the message (first-word capitalization is just
 * sentence case, not a name). Deliberately loose: the classifier is documented
 * to prefer false positives over false negatives, so a stray capitalized word
 * biasing toward search is an acceptable trade.
 * @param {string} raw original-case message text
 * @returns {boolean}
 */
function hasProperNoun(raw) {
  const words = raw.trim().split(/\s+/);
  return words.slice(1).some(w => /^["'“(]?[A-Z][a-zA-Z][a-zA-Z0-9.\-]*/.test(w));
}

/**
 * Classifies whether a user message likely requires web search/fetch tools.
 * Returns true if tools should be invoked, false if the request can be
 * answered directly from the model's knowledge or memory system.
 *
 * Errs on the side of including tools — false negatives (missed searches)
 * are worse than false positives (an unnecessary tool round).
 *
 * @param {string} messageText the user's message (original case preserved)
 * @param {boolean} superSearchEnabled forces the tool loop on when true
 * @returns {boolean}
 */
function classifyToolNeed(messageText, superSearchEnabled) {
  if (superSearchEnabled) return true;

  const raw = String(messageText || '');
  const text = raw.toLowerCase();

  // === HARD OVERRIDE: an explicit request to search always wins. Checked ===
  // === before every negative short-circuit below — if the user said to    ===
  // === look it up, we look it up, period.                                  ===
  if (EXPLICIT_SEARCH_PATTERNS.some(r => r.test(text))) return true;

  // === "No tools needed" patterns checked next — short-circuit before ===
  // === positive-match patterns that contain overly broad keywords.    ===

  // Memory / personal notes references — answer from memory system, not web
  const memoryPhrases = [
    'what do you remember', 'my notes', 'my memories', 'my clusters',
    'what have i told you', 'from our conversation', 'from memory',
    'you told me', 'we talked about', 'previous conversation',
  ];
  if (memoryPhrases.some(p => text.includes(p))) return false;

  // Pure conversational openers
  const conversationalPatterns = [
    /^(hi|hello|hey|howdy|sup|yo|good (morning|afternoon|evening|night))[\s!?.]*$/,
    /^(thanks?|thank you|thx|ty|appreciate it|cheers)[\s!.]*$/,
    /^(bye|goodbye|see you|later|cya)[\s!.]*$/,
    /^how are you[\s!?]*$/,
    /^what('s| is) up[\s!?]*$/,
    /^(ok|okay|got it|sounds good|sure|alright|cool|nice|great|perfect|wonderful)[\s!.]*$/,
  ];
  if (conversationalPatterns.some(r => r.test(text.trim()))) return false;

  // Coding / programming questions — model knowledge is sufficient
  const codingPatterns = [
    /\b(write|create|generate|implement|code|program|script|function|class|method)\b.{0,30}\b(in|using|with)\b.{0,20}\b(python|javascript|typescript|rust|go|java|c\+\+|sql|bash|ruby|php)\b/,
    /\b(debug|fix|refactor|optimize|explain)\b.{0,30}\b(this|the|my)\b.{0,20}\b(code|function|script|bug|error)\b/,
    /\bhow (do|does|to)\b.{0,40}\b(function|work|implement|use|call|declare|define)\b/,
    /\bsyntax (for|of)\b/,
    /\bexample (of|for)\b.{0,30}\b(code|function|class|pattern)\b/,
  ];
  if (codingPatterns.some(r => r.test(text))) return false;

  // Conceptual / educational questions
  // Guard: skip this shortcut when the message contains real-time keywords
  const realtimeKeywords = /\b(weather|forecast|stock|price|latest|current|today|score|version|release|update|news|new|just (released|launched|announced|out))\b/;
  if (!realtimeKeywords.test(text)) {
    const conceptualPatterns = [
      /^(explain|describe|what is|what are|what does|how does|why (is|does|do|are)|define)\b/,
      /\bcan you explain\b/,
      /\btell me about\b/,
      /\bwhat (is|are) (the )?(difference|meaning|definition|concept|purpose|point)\b/,
    ];
    if (conceptualPatterns.some(r => r.test(text))) return false;
  }

  // Creative writing
  const creativePatterns = [
    /\b(write|compose|draft|create)\b.{0,30}\b(story|poem|essay|letter|email|haiku|sonnet|blog|fiction|narrative)\b/,
    /\b(continue|finish|extend)\b.{0,30}\b(story|poem|narrative|text)\b/,
    /\bonce upon a time\b/,
  ];
  if (creativePatterns.some(r => r.test(text))) return false;

  // Questions about the model itself
  const modelPatterns = [
    /\b(you|your)\b.{0,20}\b(model|training|knowledge|cutoff|capabilities|limitations|version)\b/,
    /\bwhat (model|llm|ai) are you\b/,
    /\bwho (made|created|built|trained) you\b/,
  ];
  if (modelPatterns.some(r => r.test(text))) return false;

  // === Positive-match patterns: message likely needs web tools ===

  // --- Recency + entity: "the new Grok 4.5", "latest iOS 19", "just ---
  // --- released Vision Pro". "new"/"latest" alone is ambiguous, so it   ---
  // --- only fires alongside a version number or a proper noun.          ---
  if (RECENCY_SIGNAL.test(text) && (VERSION_TOKEN.test(raw) || hasProperNoun(raw))) {
    return true;
  }

  // --- Named product + decimal version — "Grok 4.5", "Llama 3.1", ---
  // --- "iOS 18.2". A capitalized name directly followed by an X.Y version  ---
  // --- is a specific release query even without a "new"/"latest" word; the ---
  // --- decimal requirement keeps out "Chapter 3" / "Room 204" false hits.  ---
  if (/\b[A-Z][a-zA-Z0-9]*\s?\d+\.\d+\b/.test(raw)) return true;

  // --- Explicit search/lookup intent (kept for phrasings the hard ---
  // --- override above doesn't cover) ---
  const searchPhrases = [
    'find me', 'find out',
    "what's happening with", 'what\'s happening with',
  ];
  if (searchPhrases.some(p => text.includes(p))) return true;

  // --- Current events / news ---
  const currentEventsPhrases = [
    'latest news', 'breaking news', 'headline', 'recent events',
    'what happened', "what's happening", 'what\'s happening',
    'who won', 'election results', 'sports score', 'game result',
  ];
  if (currentEventsPhrases.some(p => text.includes(p))) return true;

  // --- Real-time / time-sensitive data ---
  const realtimePatterns = [
    /\bweather\b/,
    /\bforecast\b/,
    /\bstock price\b/,
    /\bstock market\b/,
    /\bcrypto(currency)?\s+(price|value|market)/,
    /\bbitcoin\s+(price|value|worth)/,
    /\betherei?um\s+(price|value|worth)/,
    /\bright now\b/,
    /\bat the moment\b/,
    /\bcurrently\b.{0,30}\b(price|cost|rate|status|available)\b/,
    /\btoday('s)?\b.{0,40}\b(price|rate|score|news|update|status)\b/,
    /\blatest\b.{0,40}\b(version|release|update|news|patch)\b/,
    /\bcurrent\b.{0,40}\b(price|rate|status|version|leader|president|ceo)\b/,
    /\b202[5-9]\b/,  // years in the near-future range suggesting current info
    /\b203\d\b/,
  ];
  if (realtimePatterns.some(r => r.test(text))) return true;

  // --- URL / website requests ---
  const urlPatterns = [
    /\burl\b/,
    /\blink\b.{0,20}\b(to|for)\b/,
    /\bwebsite\b/,
    /\bhomepage\b/,
    /https?:\/\//,
    /\bwww\./,
    /\bdownload\b.{0,30}\b(from|link|url)\b/,
  ];
  if (urlPatterns.some(r => r.test(text))) return true;

  // --- "Is X still Y" / state-change questions ---
  const stateChangePatterns = [
    /\bis\b.{0,40}\bstill\b/,
    /\bhas\b.{0,30}\bchanged\b/,
    /\bdid\b.{0,30}\b(release|launch|announce|update|merge|fix)\b/,
    /\bwhen (did|will|is)\b/,
    /\bwhat (version|release)\b/,
  ];
  if (stateChangePatterns.some(r => r.test(text))) return true;

  // --- Specific products / releases that change frequently ---
  const productPatterns = [
    /\b(new|latest|recent|upcoming)\b.{0,30}\b(iphone|android|macbook|windows|ubuntu|debian|firefox|chrome|edge)\b/,
    /\b(changelog|patch notes|roadmap)\b/,
    /\bgithub\b.{0,30}\b(issue|pr|pull request|release|commit)\b/,
  ];
  if (productPatterns.some(r => r.test(text))) return true;

  // --- Factual / encyclopedic lookups ---
  const factualPatterns = [
    /\bwho is\b.{0,30}\b(the|a)\b/,
    /\bhow (much|many|long|far|old)\b/,
    /\bwhere (is|are|can|do)\b.{0,20}\b(the|a|i)\b/,
    /\bpopulation\b/,
    /\bcapital (of|city)\b/,
  ];
  if (factualPatterns.some(r => r.test(text))) return true;

  // Default: no tools needed — conversational, descriptive, or planning messages
  // don't require web search. Only explicit patterns above should trigger tools.
  return false;
}

// Questions about CURRENT / changeable facts that memory cannot answer reliably —
// weather, news, prices, live status, "right now"/"latest". Used by the chat path
// to REFUSE a confident-from-memory answer when search won't run (the 7/23 failure
// where it invented a weather "high of 75°F"). This only gates an honesty nudge,
// never an action, so it errs a little broad on the classic time-sensitive nouns.
const TIME_SENSITIVE_PATTERNS = [
  /\bweather\b/, /\bforecast\b/, /\btemperature\b/, /\bhow (hot|cold|warm) is it\b/,
  /\b(stock|share)\s*price\b/, /\bstock market\b/, /\bexchange rate\b/, /\bprice of\b/,
  /\b(crypto|bitcoin|ethereum|btc|eth)\b.{0,20}\b(price|value|worth)\b/,
  /\b(latest|breaking|today'?s?|recent)\s+news\b/, /\bheadlines?\b/,
  /\bwho won\b/, /\belection results?\b/, /\b(sports?|game)\s+scores?\b/,
  /\bright now\b/, /\bat the moment\b/, /\bas of (today|now)\b/,
  /\bcurrently\b.{0,30}\b(price|cost|rate|status|available|leader|president|ceo|score)\b/,
  /\btoday('?s)?\b.{0,40}\b(price|rate|score|news|update|status|weather|forecast)\b/,
  /\blatest\b.{0,40}\b(version|release|update|news|price|score)\b/,
  /\bcurrent\b.{0,40}\b(price|rate|status|version|leader|president|ceo|weather|temperature)\b/,
];

/**
 * Is this a question about current/changeable facts memory can't answer? See the
 * pattern list above. The epistemic layer uses this so the entity offers to look
 * it up instead of confidently making one up.
 * @param {string} text
 * @returns {boolean}
 */
function isTimeSensitive(text) {
  const t = String(text || '').toLowerCase();
  return TIME_SENSITIVE_PATTERNS.some(r => r.test(t));
}

/**
 * Scheduling intent — does this message ask for something to happen on a
 * recurring schedule? Gates the create_cron_job action tool into the tool loop.
 *
 * Kept SEPARATE from classifyToolNeed (which is about web search) and
 * deliberately NARROW, for the opposite reason: a false positive on the search
 * classifier costs a wasted lookup, while a false positive here means the entity
 * proposes a scheduled job during ordinary conversation and puts a decision in
 * front of Ellie that she never asked for. The probe measured a 0/20
 * false-positive rate on the model itself; this keeps the routing layer from
 * undoing that.
 *
 * Requires BOTH a recurrence signal and a request/ask signal — "every morning"
 * on its own ("I check disk space every morning") is a statement of habit, not
 * a request to schedule one.
 */
const RECURRENCE_PATTERNS = [
  /\bevery (day|morning|evening|night|hour|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d+ (minutes?|hours?|days?|weeks?))\b/,
  /\b(daily|hourly|weekly|monthly|nightly)\b/,
  /\beach (day|morning|evening|night|week|month)\b/,
  /\bon a (schedule|timer|cadence)\b/,
  /\bcron\b/,
  /\bevery so often\b/,
];

// The request must actually be ADDRESSED to the entity. A bare verb list is far
// too loose here: "I run backups daily and it is fine" is a statement of habit,
// and matching "run" in it would put an unasked-for decision in front of Ellie.
// So a request signal is either an explicit second-person ask, or an imperative
// at the very start of the message.
const SCHEDULE_REQUEST_PATTERNS = [
  /\bremind me\b/,
  /\b(can|could|would|will) you\b/,
  /\bplease\b/,
  /\bi(’|')?d like you to\b/,
  /\bi want you to\b/,
  /\bset (it|this|that) up\b/,
  /\bkeep (checking|watching|an eye)\b/,
  // Imperative opener: "schedule a …", "set up a …", "add a nightly …"
  /^\s*(schedule|set ?up|create|add|make|start|run|check)\b/,
];

/**
 * @param {string} text - the user's message
 * @returns {boolean} true when the message is asking for a recurring job
 */
function classifySchedulingIntent(text) {
  const t = String(text || '').toLowerCase();
  if (!RECURRENCE_PATTERNS.some(r => r.test(t))) return false;
  return SCHEDULE_REQUEST_PATTERNS.some(r => r.test(t));
}

/**
 * Memory-write intent — is this message asking the entity to REMEMBER something?
 * Gates the write_memory action tool into the tool loop.
 *
 * Tuned the opposite way from classifySchedulingIntent. A false positive there
 * puts an unasked-for decision in front of Ellie; here it only means the model
 * is handed a tool it can decline to call, and it still chooses. A false
 * NEGATIVE, though, is the exact bug this tool was built to fix: she asks it to
 * remember something, the turn never enters the tool loop, and it agrees warmly
 * and writes nothing. So this leans toward catching the ask.
 *
 * Still requires an explicit instruction, not any mention of memory —
 * "I remember when we set that up" is reminiscence, not a request.
 */
const MEMORY_WRITE_PATTERNS = [
  // Direct instructions to store something.
  /\b(remember|memorize|memorise) (that|this|it|i|you|my|your|we|to|the|a|an)\b/,
  /\b(write|save|add|commit|note|put) (this|that|it|the following)?\s*(down)?\s*(in ?to|to|in)? ?(your |long[- ]?term )?(memory|memories|notes)\b/,
  /\bmake a note\b/,
  /\bnote (this|that) down\b/,
  /\bkeep (this|that) in mind\b/,
  /\bdon'?t forget\b/,
  /\bnever forget\b/,
  /\bfor future reference\b/,
  /\bstore (this|that|it)\b/,
  // Bare imperative openers: "remember: …", "remember, …"
  /^\s*(remember|note)\b\s*[:,]/,
];

// Reminiscence and questions ABOUT memory are not write instructions.
const MEMORY_WRITE_NEGATIVES = [
  /\bi remember\b/,
  /\bdo you remember\b/,
  /\bdon'?t you remember\b/,
  /\bremember when\b/,
  /\bcan you remember\b.*\?/,
  /\bwhat do you remember\b/,
];

/**
 * @param {string} text - the user's message
 * @returns {boolean} true when the message asks for something to be remembered
 */
function classifyMemoryWriteIntent(text) {
  const t = String(text || '').toLowerCase();
  if (MEMORY_WRITE_NEGATIVES.some(r => r.test(t))) return false;
  return MEMORY_WRITE_PATTERNS.some(r => r.test(t));
}

/**
 * Memory-READ intent — is this message asking about what he holds in memory?
 * Gates the four inspection tools into the tool loop.
 *
 * Tuned like classifySchedulingIntent (narrow), not like classifyMemoryWriteIntent
 * (broad), and the asymmetry is deliberate. A missed read is recoverable in the
 * next sentence — she asks again, or he offers to look. A false positive puts a
 * memory-tool schema in front of him during ordinary conversation, and the
 * failure mode there is not a wasted call: it is him rummaging through the fact
 * store mid-sentence and answering a casual remark with a database report.
 *
 * The hard part is the difference between a question about his MEMORY and a
 * question about the WORLD. "What do you know about the Roman Empire" is not a
 * memory question; "what do you know about my dogs" is. Where a phrase is
 * ambiguous on its own, a PERSONAL REFERENT is required — the question has to be
 * about her, about them, or about a belief he holds.
 *
 * Bare imperatives, no hedging. Hedged phrasing measured 0/20 on this brain.
 */

// Referents that make an otherwise-general question a question about memory.
const PERSONAL_REF = String.raw`(i|i'm|i am|me|my|mine|we|us|our|she|her|hers|ellie|myself)`;

const MEMORY_READ_PATTERNS = [
  // --- explicit instruction to consult memory ---
  /\b(search|look (in|through|inside)|check|query|go through|dig through)\s+(your|the)\s+(long[- ]?term\s+)?(memory|memories|notes|facts|fact store)\b/,
  /\b(what'?s|what is|what do you have)\s+(in|stored in|saved in)\s+your\s+(memory|notes|facts)\b/,
  /\b(list|show me|show)\b.{0,25}\byour\s+(memory|memories|facts|clusters)\b/,
  /\bwhat (are|is)\b.{0,15}\byour\s+(memory\s+)?clusters\b/,

  // --- "what do you remember/recall" — 'remember' always means HIS memory ---
  /\bwhat (do|can) you (remember|recall)\b/,
  /\bwhat else do you (remember|recall|know)\b/,
  /\bdo you (remember|recall)\b/,
  /\bwhat have i (told|said to) you\b/,

  // --- "what do you know/have about <personal>" — referent required ---
  new RegExp(String.raw`\bwhat (do|did) you (know|have|hold)\b.{0,25}\babout\b.{0,30}\b${PERSONAL_REF}\b`),
  new RegExp(String.raw`\btell me what you (know|remember|have)\b.{0,30}\b${PERSONAL_REF}\b`),
  /\bwhat (facts?|memories)\b.{0,30}\b(do )?you (have|hold|store|keep)\b/,

  // --- counts: never estimate a number about your own memory ---
  /\bhow many\b.{0,40}\b(facts?|memories|clusters|entries)\b/,
  /\bhow many\b.{0,30}\bdo you (have|hold|know|remember|store)\b/,

  // --- provenance: why do you believe / where did you learn ---
  new RegExp(String.raw`\bwhy do you (believe|think|say|have)\b.{0,30}\b${PERSONAL_REF}\b`),
  new RegExp(String.raw`\bwhat makes you (think|believe|say)\b.{0,30}\b${PERSONAL_REF}\b`),
  new RegExp(String.raw`\bhow do you know\b.{0,30}\b${PERSONAL_REF}\b`),
  /\bwhere did you (learn|get|hear|find out)\b/,
  /\bwhen did (i|we) (tell|say|mention)\b/,
  /\bwhen did you learn\b/,
  /\b(is|are) that still (true|the case|right)\b/,

  // --- provenance, asked about the RECORD rather than about him ---
  // Found live on 2026-08-03: "where exactly did it come from? Which
  // conversation, and what were my actual words?" is as direct a provenance
  // question as exists, matched nothing above, went DIRECT with no tools, and he
  // answered by inventing a quote. Asking about the record is asking to read it.
  // "where did IT come from" is provenance; "where did YOU come from" is
  // philosophy, and he gets asked that. The subject has to be the record.
  /\bwhere (did|does)\s+(it|that|this|the fact|the record|the memory|(that|this|the) belief)\b.{0,15}\bcome(s)? from\b/,
  /\bwhat (were|was) (my|her) (actual |exact |original )?words\b/,
  // "which conversation topic do you enjoy most" is small talk. Require the
  // clause to be asking which conversation something came from.
  /\bwhich conversation\b.{0,30}\b(did|was|came?|from|told|said|learn|hear)\b/,
  /\bwhat did i (actually |exactly )?say\b/,
  /\bwhat does (the|your) (record|memory) (say|show)\b/,
  /\b(open|pull up|look at)\b.{0,20}\b(the |that )?(fact|record|memory|entry)\b/,
  /\bhow (did|do) you (come to )?(know|learn|have) (that|this|it)\b/,
];

/**
 * Reminiscence that is not a lookup request, and phrasings whose subject is
 * plainly the world rather than the record.
 */
const MEMORY_READ_NEGATIVES = [
  /\bi remember\b/,
  /\bremember when we\b/,
  // "remember that X" / "remember to X" are WRITE instructions; the write
  // classifier owns them, and firing both would hand him eight tools for one ask.
  /\bremember (that|to)\b/,
];

/**
 * @param {string} text - the user's message
 * @returns {boolean} true when the message asks about what he holds in memory
 */
function classifyMemoryReadIntent(text) {
  const t = String(text || '').toLowerCase();
  if (MEMORY_READ_NEGATIVES.some(r => r.test(t))) return false;
  return MEMORY_READ_PATTERNS.some(r => r.test(t));
}

/**
 * Questions about what CHANGED in his memory, rather than what is in it.
 *
 * Its own classifier because it is its own question. "What do you know about my
 * dogs" is answered from the fact store; "why did that change" can only be
 * answered from the corrections ledger, and answering it from the fact store
 * means reconstructing a reason from the facts that remain — which is invention
 * dressed as recall, the same failure the provenance warning exists to stop.
 *
 * Two shapes of pattern. Some verbs are memory-specific in this system —
 * superseded, retired, expired, merged, corrected — and carry the intent on
 * their own. The generic ones (changed, removed, replaced) need a MEMORY_REF in
 * the sentence, or "why was the meeting changed" routes into the fact store.
 */
const MEMORY_REF = String.raw`(memor(y|ies)|fact|facts|record|records|belief|beliefs|note|notes|entry|entries)`;

const MEMORY_CORRECTION_PATTERNS = [
  // --- the correction record, named ---
  /\b(show|list|pull up|give me|tell me about)\b.{0,30}\b(correction|corrections)\b/,
  /\b(correction|corrections)\s+(record|records|log|ledger|history|list)\b/,
  /\bwhat (corrections?|changes)\b.{0,25}\b(have|has|did)\s+(you|your memory)\b/,

  // --- what changed, with the referent required ---
  new RegExp(String.raw`\b(what|anything)\b.{0,25}\b(changed|has changed|have you changed|did you change|been corrected|been updated)\b.{0,25}\b(in\s+)?(your|the)\s+${MEMORY_REF}\b`),
  new RegExp(String.raw`\b(your|the)\s+${MEMORY_REF}\b.{0,25}\b(changed|been changed|been corrected|been updated|been cleaned up)\b`),
  new RegExp(String.raw`\bdid you (change|correct|remove|retire|replace|fix)\b.{0,30}\b(in\s+)?(your|the)\s+${MEMORY_REF}\b`),

  // --- why did this stop being true / why was it corrected ---
  // Memory-specific verbs: these do not need a referent.
  /\bwhy (was|were|did|is|are)\b.{0,35}\b(corrected|superseded|retired|expired|merged|folded (away|together)|no longer (held|active))\b/,
  /\bwhy (did|do) you (no longer|not) (believe|think|hold|have)\b/,
  /\bwhy (did|do) you (stop|stopped) (believing|thinking|holding)\b/,
  // Generic verbs: referent required.
  new RegExp(String.raw`\bwhy (was|were|did)\b.{0,35}\b(changed|removed|replaced|dropped|deleted|taken out)\b.{0,30}\b${MEMORY_REF}\b`),
  new RegExp(String.raw`\b${MEMORY_REF}\b.{0,35}\bwhy (was|were|did)\b.{0,25}\b(changed|removed|replaced|corrected)\b`),

  // --- what you used to believe ---
  /\bwhat (did|do) you use[d]? to (believe|think|remember|know|hold)\b/,
  /\bwhat did you (believe|think) before\b/,
  /\bwhat (have|did) you (superseded|retired|expired|merged|corrected)\b/,
];

/**
 * Ordinary uses of "correct" that are not about the ledger. "Correct me if I'm
 * wrong" is the one that matters: it appears in ordinary conversation and would
 * otherwise pull every hedged statement into a database lookup.
 */
const MEMORY_CORRECTION_NEGATIVES = [
  /\bcorrect me if\b/,
  /\b(that'?s|this is|you'?re|that is) correct\b/,
  /\bis that correct\b/,
  /\bam i correct\b/,
  /\bcorrect\?\s*$/,
];

/**
 * @param {string} text - the user's message
 * @returns {boolean} true when the message asks what changed in his memory, or why
 */
function classifyMemoryCorrectionIntent(text) {
  const t = String(text || '').toLowerCase();
  if (MEMORY_CORRECTION_NEGATIVES.some(r => r.test(t))) return false;
  return MEMORY_CORRECTION_PATTERNS.some(r => r.test(t));
}

module.exports = {
  classifyToolNeed, isTimeSensitive, classifySchedulingIntent,
  classifyMemoryWriteIntent, classifyMemoryReadIntent, classifyMemoryCorrectionIntent
};
