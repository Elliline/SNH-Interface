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

module.exports = { classifyToolNeed, isTimeSensitive };
