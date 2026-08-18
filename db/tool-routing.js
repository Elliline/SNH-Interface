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

/**
 * Questions about SCHEDULED JOBS — what he proposed, what she decided, whether it
 * ran. Routes to memory_jobs.
 *
 * Distinct from classifySchedulingIntent, which is the other direction: that one
 * catches "remind me every Monday" and routes to create_cron_job to PROPOSE. This
 * catches asking about jobs that already exist, and the two must not collide —
 * a question is not a request to make another one.
 *
 * The failure this exists for: asked "which approved job never ran", he had no
 * tool that could answer, reached for one that did not exist, and the malformed
 * call rendered into the chat. The answer he finally gave was invented, because
 * inventing was the only option left open to him.
 *
 * A JOB REFERENT IS REQUIRED for the generic verbs. "Did it run?" and "what's
 * scheduled" would otherwise pull ordinary conversation into a job lookup — "did
 * the deployment run", "what's scheduled for Tuesday" are about her week, not
 * about his cron rows.
 */
const JOB_REF = String.raw`(cron|cron job|cron jobs|job|jobs|task|tasks|schedule|scheduled (job|jobs|task|tasks)|digest|reminder|reminders)`;

const JOBS_PATTERNS = [
  // --- what exists ---
  new RegExp(String.raw`\b(what|which|any|list|show|show me|tell me)\b.{0,25}\b${JOB_REF}\b.{0,25}\b(do you have|have you|are there|exist|running|set up|scheduled|pending)\b`),
  new RegExp(String.raw`\bwhat'?s\s+(scheduled|on the schedule|queued|pending)\b`),
  new RegExp(String.raw`\b(list|show me)\b.{0,20}\b(your|the|my)\s+${JOB_REF}\b`),
  new RegExp(String.raw`\b(your|the)\s+${JOB_REF}\b.{0,20}\b(status|state)\b`),

  // --- what she decided ---
  /\bwhat (did|have) (i|ellie)\s+(approve|approved|reject|rejected|turn(ed)? down)\b/,
  new RegExp(String.raw`\b(did|has)\s+(i|ellie|she)\s+(approve|approved|reject|rejected)\b.{0,30}\b${JOB_REF}\b`),
  new RegExp(String.raw`\b(approved|rejected|pending|proposed)\b.{0,20}\b${JOB_REF}\b`),
  new RegExp(String.raw`\b${JOB_REF}\b.{0,25}\b(approved|rejected|proposed|waiting on me|waiting for me)\b`),

  // --- did it run / when does it run ---
  new RegExp(String.raw`\b(did|has|have)\b.{0,30}\b${JOB_REF}\b.{0,25}\b(run|ran|fired|executed|gone off|happened)\b`),
  new RegExp(String.raw`\b${JOB_REF}\b.{0,25}\b(never (ran|run)|hasn'?t run|didn'?t run|failed to run)\b`),
  new RegExp(String.raw`\bwh(en|at time)\b.{0,25}\b${JOB_REF}\b.{0,20}\b(run|runs|next|due|fire)\b`),
  new RegExp(String.raw`\bnext run\b`),
  // The exact shape of the morning failure.
  new RegExp(String.raw`\bwhich\b.{0,20}\b(approved|scheduled)\b.{0,20}\b${JOB_REF}\b.{0,25}\b(never|not|hasn'?t|didn'?t)\b`),
  new RegExp(String.raw`\bwhich\b.{0,25}\b${JOB_REF}\b.{0,25}\bnever ran\b`),
];

/**
 * Not about his cron rows. "Are you scheduled to…" is conversational, and
 * "schedule a call" is her diary.
 */
const JOBS_NEGATIVES = [
  /\b(schedule|set up|create|make|add)\s+(a|an|another)\s+(call|meeting|appointment)\b/,
  /\bscheduled to (meet|see|call|visit)\b/,
];

/**
 * @param {string} text - the user's message
 * @returns {boolean} true when the message asks about his scheduled jobs
 */
function classifyJobsIntent(text) {
  const t = String(text || '').toLowerCase();
  if (JOBS_NEGATIVES.some(r => r.test(t))) return false;
  return JOBS_PATTERNS.some(r => r.test(t));
}

/**
 * HANDOFF INTENT — she is asking for work, not for an answer.
 *
 * The routing question this answers is only "may he be OFFERED
 * start_background_job in this turn", never "should he use it". The judgement of
 * whether a thing is worth handing off is his, in the tool description, and it
 * has to be: the same sentence ("look into X") is a two-second lookup or a
 * twenty-minute sweep depending entirely on what X is, which a regex cannot see.
 *
 * Matched in the MIDDLE of the range, deliberately, and the two neighbours show
 * why. Memory-write is matched loosely because missing the ask is the failure it
 * exists to fix; memory-read is matched narrowly because a false positive has
 * him answering a casual remark with a database report. A false positive here is
 * cheap — the tool is merely present and he ignores it — and a false negative is
 * only a turn spent making her wait. So: an explicit request to go and do
 * something, or an explicit grant of time, and nothing looser.
 *
 * The turn is ALSO given the tool whenever it is already in the tool loop for
 * search or memory reading (server.js), which is where the interesting case
 * lives: a research question that turns out to be bigger than a turn can be
 * handed off rather than half-answered.
 */
/**
 * HANDOFF SIGNALS — built from HER messages, not from mine.
 *
 * The first version of this list was assembled from my own test prompts and
 * then, inevitably, matched them. Measured against 590 real user messages on
 * 2026-08-18: "take your time" appeared twice and BOTH were mine, typed that
 * same afternoon; "take as long as you need" once, mine; "I'll keep chatting"
 * once, mine; "while I'm out" never. The one time she used background-work
 * language herself she was describing the feature she wanted built, not asking
 * for work: "down the road i want you to be able to be given a task and you run
 * it in the background with an agent" (2026-07-23).
 *
 * So these lists come from how she actually writes, including how she actually
 * types. Four tiers, highest first — see classifyHandoffSignal.
 */

/**
 * TIER 1 — SHE NAMES THE MECHANISM. Dispatch, no judgement call.
 *
 * `and` is not a typo to be tolerated grudgingly; it is her spelling, and the
 * only real instance of this tier in the whole corpus is "Use and agent and
 * write me a write up on Paradox Interactive" (2026-08-18). A pattern requiring
 * "an agent" would have missed the one message that ever used this tier. She
 * also writes "witch" for which, "aproved", "scedual", "acurate" — matching her
 * typing is the job, not correcting it.
 */
const HANDOFF_MECHANISM = [
  /\b(use|send|have|get|give|spin ?up|spool ?up|fire ?up|kick off|start)\b[^.!?]{0,20}\b(an?|and|the|another|one)\s+agents?\b/,
  /\b(an?|and)\s+agent\s+(can|could|should|to)\b/,
  /\bagent work\b/,
  /\bhave (the|an?|and) agents? (do|handle|take|run|write|research)\b/,
  /\b(start|queue|kick off) (a|an|the)?\s*(background )?job\b/,
  /\bbackground job\b/,
];

/**
 * TIER 2 — SHE ASKS FOR SOMETHING BUILT. Behind a config flag; see
 * tools.agentJobs.dispatchBuildRequests and the note there.
 *
 * Her real phrasings: "write me a write up", "I need a game built and working",
 * "can you make games". Thin in the corpus for a plain reason — she has been
 * told he cannot build yet ("well you are not a programer so how are you going
 * to build stuff?").
 */
const HANDOFF_BUILD = [
  /\b(write|make|build|create|draft|put together)\b[^.!?]{0,15}\b(me\s+)?(a|an|the)?\s*(write[- ]?up|writeup|report|script|app|application|tool|program|game|page|site|website|dashboard|spreadsheet|document)\b/,
  /\bi need (a|an|the)?\s*\w+\s+(built|made|written|created)\b/,
];

/**
 * TIER 3 — THE SHAPE OF THE WORK. More than a couple of lookups, several
 * sources, more than one subject, or a writeup.
 *
 * Drawn from how she actually asks for research:
 *   "Can you search the web for reviews and discussions of that book and tell me
 *    what others have said about it — criticisms as well as praise?"
 *   "Can you look up what new stuff has been happening in the AI world over the
 *    last week?"
 *   "Any ways look again for the latest AI news with in the last 7 days."
 *   "Can you please verify these news items are actually news from the last 7
 *    days please. also if you find some that are not let me know witch ones and
 *    find what really is going on in the ai world"
 *   "I dont know, why dont you look online and see if any one else knows."
 *
 * The recurring shape is a LOOKING VERB plus either a SCOPE marker or a SECOND
 * CLAUSE. "Can you" on its own is worthless as a signal — thirteen occurrences,
 * split between "can you look up X" and "can you explain this more" / "can you
 * see why im getting brain dead" — so it never counts without a verb behind it.
 */
const LOOKING_VERB = /\b(look up|look online|look into|look again|search the web|search online|find out|go find|research|dig into|compare|verify|catalogue|catalog|categorize|catagorize)\b/;
const SCOPE_MARKER = /\b(over the (last|past)|last \d+ days?|past \d+ days?|latest|current|everything|all of|what you can find|anything you know|any thing you know|each (one|client|item)|as of (right )?now)\b/;
const SECOND_CLAUSE = /\b(also|as well as|and tell me|and let me know|and find|criticisms|plus)\b/;
const COMPARISON = /\b(compare|comparing|comparison|versus|vs\.?)\b|\bwrite[- ]?up\b|\bwriteup\b|\breport on\b/;

/**
 * TIER 4 — SHE GRANTED TIME. A WEAK signal, and flagged here as what it is.
 *
 * ⚠ UNATTESTED IN HER HISTORY. Not one of the 590 messages grants time in these
 * words; every apparent instance traced back to my own test prompts. It is kept
 * so the signal exists if she ever does use it, and it is deliberately NOT
 * sufficient on its own — a single-fact lookup with "take your time" on the end
 * is still a single-fact lookup. It only lifts something that already has work
 * shape.
 */
const HANDOFF_TIME_GRANTED = [
  /\b(take your time|take as long as you need|take all the time you need)\b/,
  /\bno (rush|hurry)\b/,
  /\b(whenever|when) you (get a chance|can|have time|have a moment|get to it)\b/,
  /\bin the background\b/,
];

/**
 * NEGATIVES — from her actual brevity and answer asks, not invented.
 *
 *   "In one sentence: what do you know about my projects?"
 *   "Ok can you break this down barnie style, Your big words get me confused"
 *   "can you explain this more?"
 *   "A few questions: … (Don't search…)"
 *   "Check again, what all can you do?"
 */
const HANDOFF_NEGATIVES = [
  /\bin one sentence\b/,
  /\b(explain|break) (this|it) (more|down)\b/,
  /\bdon'?t (search|look it up|bother|worry about it)\b/,
  /\bwhat (all )?can you do\b/,
  /\bnever ?mind\b/,
  /\bforget (it|about it|that)\b/,
  /\b(quick|quickly|real quick|off the top of your head|just tell me|briefly)\b/,
];

/**
 * Which tier fired, and why — so the guidance block can say the true thing
 * rather than a generic one.
 *
 * @param {string} text
 * @returns {{dispatch: boolean, tier: number|null, reason: string|null}}
 */
function classifyHandoffSignal(text, { allowBuild = false } = {}) {
  const t = String(text || '').toLowerCase();
  const none = { dispatch: false, tier: null, reason: null };
  if (!t) return none;

  // Tier 1 outranks the negatives: "use an agent, quick" is still "use an agent".
  if (HANDOFF_MECHANISM.some(r => r.test(t))) {
    return { dispatch: true, tier: 1, reason: 'she asked for an agent by name' };
  }
  if (HANDOFF_NEGATIVES.some(r => r.test(t))) return none;

  if (allowBuild && HANDOFF_BUILD.some(r => r.test(t))) {
    return { dispatch: true, tier: 2, reason: 'she asked for something to be produced, not explained' };
  }

  const looking = LOOKING_VERB.test(t);
  const shaped = looking && (SCOPE_MARKER.test(t) || SECOND_CLAUSE.test(t));
  if (shaped || COMPARISON.test(t)) {
    return { dispatch: true, tier: 3, reason: 'it needs more than a lookup — several sources or several parts' };
  }

  // Tier 4 never carries a turn alone. It lifts a looking verb that did not
  // otherwise reach tier 3; it does nothing for a bare question.
  if (looking && HANDOFF_TIME_GRANTED.some(r => r.test(t))) {
    return { dispatch: true, tier: 4, reason: 'she said she is not waiting on it' };
  }
  return none;
}

/**
 * @param {string} text - the user's message
 * @returns {boolean} true when the message asks for work that may outlive the turn
 */
function classifyHandoffIntent(text, opts) {
  return classifyHandoffSignal(text, opts).dispatch;
}

module.exports = {
  classifyToolNeed, isTimeSensitive, classifySchedulingIntent,
  classifyMemoryWriteIntent, classifyMemoryReadIntent, classifyMemoryCorrectionIntent,
  classifyJobsIntent, classifyHandoffIntent, classifyHandoffSignal
};
