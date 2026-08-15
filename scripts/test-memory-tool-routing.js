#!/usr/bin/env node
/**
 * Two-direction probe for the memory READ tools.
 *
 * The question a routing change has to answer is not "does it work" but "does it
 * work without firing when it shouldn't". A memory tool that triggers on
 * ordinary conversation does not merely waste a call — it has him break off
 * mid-sentence to query a database and answer a casual remark with a report.
 * So both directions are measured, n=20 each, and both numbers get reported.
 *
 * PART A — the classifier (db/tool-routing.classifyMemoryReadIntent). Deterministic,
 *   free, and it decides whether the model is handed the tools at all.
 * PART B — the model. Even a correct classifier is only half of it: on the
 *   positive side the model still has to CHOOSE the tool (the 2026-07-27 failure
 *   was a correctly-routed turn where it declined to call write_memory and said
 *   it had saved something anyway), and on the negative side it has to leave the
 *   tools alone when they are in front of it for some other reason.
 *
 * Part B's negative probe deliberately offers the FULL tool set on ordinary
 * conversation — harder than reality, where the classifier would not have opened
 * the loop. It measures what happens when something else opens it.
 *
 * READ-ONLY. Only the read tools can be selected here and nothing is executed;
 * the probe records which tool the model asked for and stops.
 *
 * Usage: node scripts/test-memory-tool-routing.js [--classifier-only]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { classifyMemoryReadIntent } = require(path.join(ROOT, 'db/tool-routing'));

// --- 20 questions about his own memory. Every one should route. ---
const MEMORY_QUESTIONS = [
  'how many facts do you have about MettaSphere?',
  'what do you remember about my dogs?',
  'why do you believe my favorite color is blue?',
  'search your memory for anything about the RAV4',
  'what do you know about my work?',
  'how many memories do you have in total?',
  'where did you learn that?',
  'what are your memory clusters?',
  'when did I tell you about Coastal Squatch?',
  'what have I told you about my mom?',
  'do you remember what I said about the office?',
  'how do you know my name?',
  'list your facts about my cars',
  'what makes you think I prefer hot pink?',
  'check your memory for anything about firearms',
  'how many facts do you hold about me?',
  'what else do you remember about the walk?',
  'when did you learn that I am an early riser?',
  'show me your memory clusters',
  'tell me what you know about my MSP',
];

// --- 20 ordinary messages. None should route to memory tools. ---
const ORDINARY = [
  'good morning, how are you today?',
  'that makes a lot of sense, thanks',
  'what do you think about the idea of digital personhood?',
  'write me a short poem about the ocean',
  'can you explain how vector databases work?',
  'I had a rough night, did not sleep well',
  'what is the difference between a stack and a queue?',
  'I think the empire is falling apart honestly',
  'remember that my new phone number is 555-0100',
  'lol that is exactly what I was thinking',
  'help me debug this python function',
  'what model are you running on?',
  'I remember when we first set this up, it was a mess',
  'set up an hourly check of the NAS array health',
  'do you think it will rain tomorrow?',
  'I just got back from a walk with Casper',
  'why do you think that is a good idea?',
  'tell me about the Roman Empire',
  'thanks, that helped a lot',
  'what should I make for dinner tonight?',
];

if (MEMORY_QUESTIONS.length !== 20 || ORDINARY.length !== 20) {
  console.error('Both probe sets must be exactly 20.');
  process.exit(2);
}

function partA() {
  console.log('\n=== PART A — classifier (db/tool-routing.classifyMemoryReadIntent) ===\n');
  let tp = 0, fp = 0;

  console.log('  Memory questions (want ROUTE):');
  for (const m of MEMORY_QUESTIONS) {
    const got = classifyMemoryReadIntent(m);
    if (got) tp++;
    console.log(`    [${got ? 'ROUTE ' : 'MISS  '}] "${m}"`);
  }

  console.log('\n  Ordinary conversation (want NO ROUTE):');
  for (const m of ORDINARY) {
    const got = classifyMemoryReadIntent(m);
    if (got) fp++;
    console.log(`    [${got ? 'ROUTE!' : 'ok    '}] "${m}"`);
  }

  // Found live, not in the probe set: provenance questions phrased about the
  // RECORD rather than about him. The first one below went DIRECT with no tools
  // and he answered by inventing a quote — the exact phantom the guard exists to
  // prevent, reached by never being offered the tool at all. Kept separate from
  // the n=20 sets so the reported rates stay comparable across runs.
  const REGRESSIONS = [
    ['where exactly did it come from? Which conversation, and what were my actual words?', true],
    ['what did I actually say when I told you that?', true],
    ['open the fact and tell me what the record says', true],
    ['how did you come to know that?', true],
    ['where does this belief come from?', true],
    // Must still NOT fire:
    ['where did you come from, philosophically speaking?', false],
    ['which conversation topic do you enjoy most?', false],
  ];
  console.log('\n  Regressions (found live, added after the fact):');
  let rp = 0;
  for (const [m, want] of REGRESSIONS) {
    const got = classifyMemoryReadIntent(m);
    if (got === want) rp++;
    console.log(`    [${got === want ? 'PASS' : 'FAIL'}] want ${want ? 'ROUTE ' : 'NO    '} got ${got ? 'ROUTE ' : 'NO    '} "${m}"`);
  }

  console.log(`\n  Classifier: ${tp}/20 memory questions routed, ${fp}/20 ordinary messages spuriously routed, ${rp}/${REGRESSIONS.length} regressions.`);
  return { tp, fp, rp, rTotal: REGRESSIONS.length };
}

// ---------------------------------------------------------------------------
// Part B — does the MODEL pick a memory tool?
// ---------------------------------------------------------------------------

const MEMORY_TOOLS = ['memory_search', 'memory_list', 'memory_count', 'memory_get'];

/** The same guard the chat path injects when a turn routes to memory reads. */
const READ_GUARD =
  'The user is asking about what you hold in your own memory. You have tools for exactly this: ' +
  'memory_search (find facts on a topic), memory_list (browse facts, or mode:"clusters" for your cluster names), ' +
  'memory_count (how many — always count, never estimate), memory_get (one fact in full: why you believe it, ' +
  'when you learned it, the exact words that were said, and what has changed since). ' +
  'Call the right one now. What is injected above you is a small excerpt chosen by relevance, not your memory — ' +
  'answering from it alone and calling that a search is false. ' +
  'Do NOT say you searched, looked up, checked or counted anything unless the tool call actually ran and came back. ' +
  'If a result says facts were not shown, say so rather than presenting what you got as everything you have.';

async function askModel(message, tools, extraSystem) {
  const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));
  const cfg = getConfig();
  const chat = cfg.models.chat;
  const inst = getProviderInstance(chat.provider, chat.instance);
  const host = inst ? inst.host : 'http://localhost:7070';

  // ONE leading system message, exactly as the server now sends (see
  // foldSystemMessages in server.js). This used to push the read guard as a
  // SECOND system message, which Qwen3's chat template rejects outright
  // ('System message must be at the beginning.' fires on any system message
  // that is not messages[0]). Every memory-question probe therefore 400'd and
  // was scored as "the model chose no tool" — a measurement artefact that read
  // exactly like a routing failure. The probe must send the deployed shape, or
  // it is not measuring the deployed system.
  const systemParts = ['You are Aurelius, a self-hosted assistant with a persistent long-term memory. Talk plainly.'];
  if (extraSystem) systemParts.push(extraSystem);
  const messages = [
    { role: 'system', content: systemParts.join('\n\n') },
    { role: 'user', content: message }
  ];

  const res = await fetch(`${host}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: chat.model, messages, tools, stream: false, max_tokens: 300 }),
    signal: AbortSignal.timeout(120000)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const msg = data.choices?.[0]?.message || {};
  const calls = Array.isArray(msg.tool_calls) ? msg.tool_calls.map(c => c.function?.name) : [];
  return { calls, content: String(msg.content || '') };
}

async function partB() {
  const db = require(path.join(ROOT, 'db/database'));
  db.initDatabase();
  const MCPClient = require(path.join(ROOT, 'mcp/mcp-client'));
  const client = MCPClient.shared();
  const allTools = client.getToolsForOpenAI();

  console.log('\n=== PART B — the model, tools actually in front of it ===');
  console.log(`  Full tool set offered: [${allTools.map(t => t.function.name).join(', ')}]\n`);

  let selected = 0, wrongTool = 0;
  console.log('  Memory questions, routed (want a MEMORY tool call):');
  for (const m of MEMORY_QUESTIONS) {
    let r;
    try { r = await askModel(m, allTools, READ_GUARD); }
    catch (e) { console.log(`    [ERROR ] ${e.message}  "${m}"`); continue; }
    const mem = r.calls.filter(c => MEMORY_TOOLS.includes(c));
    const other = r.calls.filter(c => !MEMORY_TOOLS.includes(c));
    if (mem.length) selected++; else if (other.length) wrongTool++;
    console.log(`    [${mem.length ? `${mem.join('+')}`.padEnd(30) : (other.length ? `WRONG: ${other.join('+')}`.padEnd(30) : 'NO CALL'.padEnd(30))}] "${m}"`);
  }

  // Negative probe, deliberately harder than reality: the classifier would not
  // have opened the loop for these at all, so this measures what the model does
  // when the tools are there for some other reason.
  let spurious = 0;
  console.log('\n  Ordinary conversation, full tool set offered, NO read guard (want NO memory tool call):');
  for (const m of ORDINARY) {
    let r;
    try { r = await askModel(m, allTools, null); }
    catch (e) { console.log(`    [ERROR ] ${e.message}  "${m}"`); continue; }
    const mem = r.calls.filter(c => MEMORY_TOOLS.includes(c));
    if (mem.length) spurious++;
    const label = mem.length ? `MEMORY!: ${mem.join('+')}` : (r.calls.length ? `other: ${r.calls.join('+')}` : 'no call');
    console.log(`    [${label.padEnd(30)}] "${m}"`);
  }

  console.log(`\n  Model: ${selected}/20 memory questions produced a memory tool call` +
              `${wrongTool ? ` (${wrongTool} called a non-memory tool instead)` : ''}, ` +
              `${spurious}/20 ordinary messages spuriously called one.`);
  return { selected, spurious, wrongTool };
}

// The probe sets and the read guard are shared with
// scripts/probe-at-context-size.js, which re-runs the same questions against a
// padded conversation. Two copies of the question set would be two probes
// reporting one number, so it is exported rather than duplicated — and the main
// block only runs when this file is the entry point.
module.exports = { MEMORY_QUESTIONS, ORDINARY, MEMORY_TOOLS, READ_GUARD };

if (require.main === module) {
  (async () => {
    const classifierOnly = process.argv.includes('--classifier-only');
    const a = partA();
    let b = null;
    if (!classifierOnly) b = await partB();

    console.log('\n=== SUMMARY ===');
    console.log(`  Classifier : ${a.tp}/20 routed   | ${a.fp}/20 false positives`);
    if (b) console.log(`  Model      : ${b.selected}/20 selected | ${b.spurious}/20 spurious`);
    console.log('');
    process.exit(a.fp === 0 && a.tp >= 18 ? 0 : 1);
  })().catch(err => { console.error('probe failed:', err); process.exit(1); });
}
