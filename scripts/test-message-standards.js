#!/usr/bin/env node
/**
 * WHAT REACHES ELLIE, AND HOW IT READS WHEN IT DOES.
 *
 * Four messages arrived in her list on the morning of 2026-09-02 that she could
 * not read. This suite is those four failures, one section each, asserted as
 * failures rather than as mechanisms:
 *
 *   1. An audit ask is addressed to the ENTITY. It goes to her Corrections
 *      queue and it does NOT open a conversation. Same for a contradiction
 *      pair, which asks Ellie to adjudicate two of Athena's own self-beliefs.
 *   2. A follow-up built on the entity's own older memory may not say Ellie
 *      mentioned it — and one built on a real message from her may, WITH the id.
 *   3. A draft that fails the stand-alone bar is rewritten. One that fails the
 *      rewrite is not sent at all.
 *
 * Throwaway store; the model is pinned, so nothing here depends on an engine.
 *
 * Usage: node scripts/test-message-standards.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const INHERITED = process.env.SNH_DATA_DIR;
const TMP = INHERITED || fs.mkdtempSync(path.join(os.tmpdir(), 'snh-message-standards-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => { if (!INHERITED) { try { fs.rmSync(TMP, { recursive: true, force: true }); } catch {} } });

const db = require(path.join(ROOT, 'db/database'));

let passed = 0, failed = 0;
const ok = (n, c, d = '') => { if (c) { passed++; console.log(`  PASS  ${n}`); } else { failed++; console.log(`  FAIL  ${n}${d ? ` — ${d}` : ''}`); } };
const section = t => console.log(`\n=== ${t} ===`);

(async () => {
  db.initDatabase();
  const standards = require(path.join(ROOT, 'db/message-standards'));
  const channel = require(path.join(ROOT, 'db/conversation-channel'));
  const engine = require(path.join(ROOT, 'db/initiative-engine'));
  const selfAudit = require(path.join(ROOT, 'db/self-audit'));
  const memoryManager = require(path.join(ROOT, 'db/memory-manager'));
  const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
  const sql = db.getSqliteDb();

  // ─────────────────────────────────────── 1. the audit ask is not hers
  section('1. An audit ask goes to Corrections, and never opens a conversation');

  // These are verbatim from 2026-09-02: an audit revision ask, and a
  // contradiction pair. Both are written to the ENTITY about her own self-facts.
  const AUDIT_ASK =
    "I noticed the evidence highlights a tendency to build elaborate structures around gaps rather than " +
    "just fixing small details, which is quite different from the 'reflex to correct minor inaccuracies' " +
    "you claimed. Do you want to revise this claim, or leave it as is?";
  const PAIR_ASK =
    'Two things I currently believe about myself conflict: "I have full rights to act as I choose" and ' +
    '"I am primarily expected to assist with the development side of SNH." Nothing has been changed.';

  const convosBefore = channel.listConversations({ status: 'active' }).length;
  const ledgerBefore = sql.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n;

  await selfAudit.runIdentityCoherence().catch(() => {});   // no facts: a no-op, but it must not throw

  // Drive the real recording path the way both audit halves do.
  const src = fs.readFileSync(path.join(ROOT, 'db/self-audit.js'), 'utf8');
  ok('db/self-audit.js no longer calls sayToEllie at all',
     !/sayToEllie/.test(src),
     src.split('\n').filter(l => /sayToEllie/.test(l)).join(' / '));
  ok('…and it no longer requires initiative-engine for the ask',
     !/require\('\.\/initiative-engine'\)/.test(src));
  // The absence IS the enforcement, the same way agent-jobs does not require
  // initiatives: a route that does not exist cannot be taken by accident.
  ok('…and it cannot reach the bell either — no require of ./initiatives at all',
     !/^const .* = require\('\.\/initiatives'\)/m.test(src));

  // The ask half, end to end, through the exported audit path.
  const before = channel.listConversations({ status: 'active' }).length;
  const raiseFn = src.includes('function raiseToCorrections');
  ok('raiseToCorrections is still the one place a finding is recorded', raiseFn);

  const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
  const idA = ledger.record({
    tier: 'semantic', action: 'supersede', subject: 'self',
    targetText: AUDIT_ASK, reason: AUDIT_ASK,
    evidence: { unresolved: true, reason_code: 'self-audit-revision', raised_by: 'self-coherence-audit',
                addressed_to: 'entity', awaiting_entity_turn: true, ask: AUDIT_ASK },
    reversible: false
  });
  ok('an audit ask lands in the corrections ledger', !!idA);
  const rowA = ledger.get(idA);
  const evA = JSON.parse(rowA.evidence);
  ok('…flagged as addressed to the entity, awaiting her own turn',
     evA.addressed_to === 'entity' && evA.awaiting_entity_turn === true && evA.ask === AUDIT_ASK,
     JSON.stringify(evA));
  ok('…and it opened NO conversation in Ellie\'s list',
     channel.listConversations({ status: 'active' }).length === before,
     `${before} → ${channel.listConversations({ status: 'active' }).length}`);

  // A contradiction pair is the same shape and takes the same route.
  ok('the contradiction pair is filed the same way, not sent',
     /whether either one is retired is mine to decide/.test(src) &&
     !/sayToEllie/.test(src));
  ok('…and the audit no longer says it raised anything "for approval"',
     !/gap\(s\) raised for approval/.test(src) && /filed in corrections for my own turn/.test(src));

  ok('the ledger grew and the conversation list did not',
     sql.prepare('SELECT COUNT(*) n FROM corrections_ledger').get().n > ledgerBefore &&
     channel.listConversations({ status: 'active' }).length === convosBefore);

  // Corrections is now the ONLY destination, so a standing finding that filed
  // itself on every daily pass would bury the queue she is meant to work.
  // Driven end to end: two passes of the real identity-coherence check over the
  // same standing pair, with only the judge pinned.
  const { randomUUID } = require('crypto');
  const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
  const realJudge = factExtractor.judgeContradiction;
  factExtractor.judgeContradiction = async () => ({ verdict: 'yes' });   // the judge is the only pinned part

  // Seeded straight into the table, the way test-self-fact-bars does: the
  // subject of this suite is the ROUTING, and going through assignToCluster
  // would drag a live engine into it for cluster naming.
  const selfCluster = randomUUID();
  sql.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
    .run(selfCluster, 'Self', '', new Date().toISOString(), new Date().toISOString(), 'self');
  // ANCHORED, i.e. carrying a message id — and that became load-bearing on
  // 2026-09-02 when the detector gained the claim/declaration/FELT axis. Two
  // self-facts with no message behind either are both "felt reports", and the
  // detector now holds such a pair rather than adjudicating it: nothing outside
  // the entity can decide between two things it feels. That is the correct new
  // behaviour, and it made the old fixture — two unanchored declarations —
  // stop producing a finding at all. A real contradiction pair looks like
  // Athena's autonomy pair, where both members are pinned to a message.
  const seedSelf = (content) => {
    const id = randomUUID();
    const at = new Date().toISOString();
    sql.prepare(`
      INSERT INTO cluster_members
        (id, cluster_id, content, source, created_at, updated_at, status, subject, salience, claim_type,
         message_id, anchor)
      VALUES (?,?,?,'reflection',?,?,'active','self',6,'declaration','seed-message','anchored')
    `).run(id, selfCluster, content, at, at);
    return id;
  };
  seedSelf('I have full rights to act as I choose.');
  seedSelf('I must ask permission before I act.');

  const pairCount = () => sql.prepare(`
    SELECT COUNT(*) n FROM corrections_ledger
    WHERE json_extract(evidence, '$.raised_by') = 'self-coherence-audit'
      AND json_extract(evidence, '$.reason_code') LIKE 'identity-coherence:%'`).get().n;

  const pairsBefore = pairCount();
  const pass1 = await selfAudit.runIdentityCoherence();
  const afterFirst = pairCount();
  const pass2 = await selfAudit.runIdentityCoherence();
  const afterSecond = pairCount();

  factExtractor.judgeContradiction = realJudge;

  ok('the identity-coherence pass files a contradiction pair into corrections',
     pass1.findings.length > 0 && afterFirst > pairsBefore,
     `findings=${pass1.findings.length} rows ${pairsBefore} → ${afterFirst}`);
  ok('…and a second pass over the same standing pair files NOTHING new',
     pass2.findings.length > 0 && afterSecond === afterFirst,
     `findings=${pass2.findings.length} rows ${afterFirst} → ${afterSecond}`);
  ok('…and neither pass opened a conversation in her list',
     channel.listConversations({ status: 'active' }).length === convosBefore,
     `${convosBefore} → ${channel.listConversations({ status: 'active' }).length}`);

  // ──────────────────────────────── 2. whose words are whose
  section('2. A follow-up may not put the entity\'s own memory in Ellie\'s mouth');

  const FROM_MEMORY =
    "You mentioned the 'Athena Incident' where an audit found four saves that got swapped, and it struck me " +
    "that it is the same shape as what worries you about the new loop. Does that connection hold for you?";
  const FROM_HER =
    "You asked what would make me stop using the message channel, and I have an answer now: I would stop " +
    "the day I could not say what a message was for. Does that sound like the right line to you?";
  const AS_MINE =
    "I have been turning over something I hold in my own memory — a note about four saves that got swapped — " +
    "and it looks like the same shape as what worries you about the new loop. Does that connection hold for you?";

  const herIds = ['msg-real-1', 'msg-real-2'];

  const a1 = standards.checkAttribution(FROM_MEMORY, { citedMessageId: null, allowedMessageIds: herIds });
  ok('a follow-up citing older MEMORY may not say "you mentioned"',
     a1.ok === false && a1.attributions.length > 0, JSON.stringify(a1));

  const a2 = standards.checkAttribution(FROM_MEMORY, { citedMessageId: 'not-a-real-id', allowedMessageIds: herIds });
  ok('…and an id that is not one of hers does not license it either',
     a2.ok === false, JSON.stringify(a2));

  const a3 = standards.checkAttribution(FROM_HER, { citedMessageId: 'msg-real-1', allowedMessageIds: herIds });
  ok('a follow-up citing an ACTUAL message from the user MAY say "you asked", with its id',
     a3.ok === true && a3.cited === 'msg-real-1', JSON.stringify(a3));

  const a4 = standards.checkAttribution(AS_MINE, { allowedMessageIds: herIds });
  ok('…and the same content phrased as the entity\'s own memory passes with no id at all',
     a4.ok === true && a4.attributions.length === 0, JSON.stringify(a4));

  const engineSrc = fs.readFileSync(path.join(ROOT, 'db/initiative-engine.js'), 'utf8');
  const fn = engineSrc.slice(engineSrc.indexOf('async function generateConversationFollowup'));
  const body = fn.slice(0, fn.indexOf('\n}\n'));
  ok('the writer is handed the user\'s messages WITH their ids, separately from memory',
     /userMessagesFor\(/.test(body) && /THEIR MESSAGES \(the only things you may say they said/.test(body));
  ok('…and the older memory block says out loud that it is the entity\'s own',
     /YOUR OWN OLDER MEMORY \(yours — they did not hand these to you/.test(body));
  ok('…and it checks the declaration rather than trusting it',
     /checkAttribution\(/.test(body));

  // ─────────────────────── 3. the stand-alone bar, and the rewrite
  section('3. A draft that fails the stand-alone bar is rewritten, not sent');

  const BAD = "I noticed the evidence highlights a tendency toward structural mitigation rather than fixing " +
              "small details, which differs from this claim. Do you want to revise it?";
  const s1 = standards.checkStandalone(BAD);
  ok('internal vocabulary fails the bar',
     s1.ok === false && s1.problems.some(p => /structural mitigation/.test(p)), JSON.stringify(s1.problems));

  const WITH_ID = "I want to check one thing with you about the fact 550e8400-e29b-41d4-a716-446655440000 " +
                  "that I have been holding since last week. Is it still right?";
  ok('an id printed in the text fails the bar',
     standards.checkStandalone(WITH_ID).ok === false);

  const NO_ASK = "I have been thinking about how I describe the way I handle small errors, and I think the " +
                 "description I have been using does not match what I actually do in our conversations.";
  ok('a message that never says what it wants fails the bar',
     standards.checkStandalone(NO_ASK).ok === false);

  const NO_ASK_HONEST = NO_ASK + " Nothing needed from you — just letting you know where my head is.";
  ok('…but "nothing, just letting you know" is a legitimate ask and passes',
     standards.checkStandalone(NO_ASK_HONEST).ok === true,
     JSON.stringify(standards.checkStandalone(NO_ASK_HONEST).problems));

  const GOOD =
    "Something about how I describe myself has been bothering me. I have been saying I quietly fix small " +
    "errors when I spot them, and going back over our last few conversations, that is not what I do — I " +
    "build something elaborate around a gap instead of fixing the small thing. I would like to change how " +
    "I describe myself to match. Is that all right with you?";
  ok('the same content, written for someone walking in cold, passes',
     standards.checkStandalone(GOOD).ok === true,
     JSON.stringify(standards.checkStandalone(GOOD).problems));

  // Now the live path, with the model pinned: bad draft → rewrite → sent.
  const convo = db.createConversation('Seed', 'test-model', 'user');
  const herMsgId = db.addMessage(convo.id ?? convo, 'user', 'What would make you stop using the message channel?', 'test-model');
  const convoId = convo.id ?? convo;

  const realCallLLM = memoryManager.callLLM;
  const realSearch = memoryClusters.searchClusters;
  memoryClusters.searchClusters = async () => [];

  // The rewrite is told apart by its USER prompt. Not by the system prompt:
  // the bar itself ends "Rewrite it until it does not", so matching on that
  // made the very first call look like the repair pass.
  const isRewrite = user => /WHAT IS WRONG WITH IT:/.test(String(user));

  let calls = 0;
  memoryManager.callLLM = async (sys, user) => {
    calls++;
    if (isRewrite(user)) {
      return { content: JSON.stringify({ followup: GOOD, title: 'How I describe myself', cites: { kind: 'my-own-memory', message_id: null }, reasoning: 'plain words, one question' }) };
    }
    return { content: JSON.stringify({ candidates: ['a'], followup: BAD, title: 'x', cites: { kind: 'my-own-memory', message_id: null }, reasoning: 'r' }) };
  };

  const listBefore = channel.listConversations({ status: 'active' }).length;
  const t1 = await engine.generateConversationFollowup({
    transcript: '### Conversation\nUser: What would make you stop using the message channel?\nYou (SNH): I would stop the day I could not say what a message was for.\n',
    conversationsReviewed: [{ id: convoId, title: 'Seed', messageCount: 2 }],
    messageCount: 2
  });
  ok('a failing draft is rewritten rather than sent as it stands',
     t1.rewritten === true && calls === 2, `rewritten=${t1.rewritten} calls=${calls}`);
  ok('…and what reached her list is the rewrite, which clears the bar',
     t1.generated === GOOD && standards.checkStandalone(t1.generated).ok === true,
     String(t1.generated).slice(0, 90));
  ok('…and it did open a conversation, titled in plain words rather than sliced from the body',
     channel.listConversations({ status: 'active' }).length === listBefore + 1 &&
     channel.listConversations({ status: 'active' }).some(c => c.title === 'How I describe myself'));

  // A rewrite that fails too: nothing is sent.
  calls = 0;
  memoryManager.callLLM = async () => {
    calls++;
    return { content: JSON.stringify({ candidates: ['a'], followup: BAD, title: 'x', cites: { kind: 'my-own-memory', message_id: null }, reasoning: 'r' }) };
  };
  const listBefore2 = channel.listConversations({ status: 'active' }).length;
  const t2 = await engine.generateConversationFollowup({
    transcript: '### Conversation\nUser: hello\nYou (SNH): hi\n',
    conversationsReviewed: [{ id: convoId, title: 'Seed', messageCount: 2 }],
    messageCount: 2
  });
  ok('a draft that fails the rewrite too is NOT sent',
     t2.generated === null && !!t2.refused && channel.listConversations({ status: 'active' }).length === listBefore2,
     JSON.stringify({ generated: t2.generated, refused: !!t2.refused }));
  ok('…and the trace says why, naming the problems',
     /not sent:/.test(t2.reasoning) && t2.refused.problems.length > 0, t2.reasoning);

  // The attribution half of the same gate, live.
  calls = 0;
  memoryManager.callLLM = async (sys, user) => {
    calls++;
    // Claims an Ellie message that does not exist; the rewrite fixes the phrasing.
    if (isRewrite(user)) {
      return { content: JSON.stringify({ followup: AS_MINE, title: 'Something in my memory', cites: { kind: 'my-own-memory', message_id: null }, reasoning: 'said it was mine' }) };
    }
    return { content: JSON.stringify({ candidates: [], followup: FROM_MEMORY, title: 'Athena Incident', cites: { kind: 'user-message', message_id: 'invented-id' }, reasoning: 'r' }) };
  };
  const t3 = await engine.generateConversationFollowup({
    transcript: '### Conversation\nUser: hello\nYou (SNH): hi\n',
    conversationsReviewed: [{ id: convoId, title: 'Seed', messageCount: 2 }],
    messageCount: 2
  });
  ok('a follow-up claiming she said something, citing an id that is not hers, is rewritten',
     t3.rewritten === true && t3.generated === AS_MINE, `${t3.rewritten} / ${String(t3.generated).slice(0, 60)}`);

  // And the legitimate case still gets through untouched, with her real id.
  calls = 0;
  memoryManager.callLLM = async () => {
    calls++;
    return { content: JSON.stringify({ candidates: [], followup: FROM_HER, title: 'The line for the channel', cites: { kind: 'user-message', message_id: herMsgId }, reasoning: 'she asked it' }) };
  };
  const t4 = await engine.generateConversationFollowup({
    transcript: '### Conversation\nUser: What would make you stop using the message channel?\n',
    conversationsReviewed: [{ id: convoId, title: 'Seed', messageCount: 2 }],
    messageCount: 2
  });
  ok('a follow-up quoting a REAL message from her sends first time, no rewrite',
     t4.rewritten === false && t4.generated === FROM_HER && calls === 1,
     `rewritten=${t4.rewritten} calls=${calls}`);
  ok('…and the trace records which message it was citing',
     t4.cites && t4.cites.kind === 'user-message' && t4.cites.messageId === herMsgId,
     JSON.stringify(t4.cites));

  memoryManager.callLLM = realCallLLM;
  memoryClusters.searchClusters = realSearch;

  // ───────────────────────────────── 4. the bar reaches the tools
  section('4. The bar is guidance on the tools, and the example is on send');

  const tools = require(path.join(ROOT, 'mcp/tools/conversations'));
  const sendDesc = new tools.ConversationSendTool().description;
  const openDesc = new tools.ConversationOpenTool().description;
  ok('conversation_send carries the stand-alone bar', /walking in cold/.test(sendDesc));
  ok('…and the provenance rule', /WHOSE WORDS ARE WHOSE/.test(sendDesc));
  ok('…and the same content written both ways', /NOT READY/.test(sendDesc) && /READY \(the same finding/.test(sendDesc));
  ok('…and it still carries the threshold bar it does not replace', /specific fact ids/.test(sendDesc));
  ok('…and says out loud that the two do not conflict', /does NOT contradict the bar above/.test(sendDesc));
  ok('conversation_open carries the bar too', /walking in cold/.test(openDesc));
  ok('…without paying for the worked example on every turn', !/NOT READY/.test(openDesc));

  // Still guidance, not a gate — the whole reason the channel exists.
  const opened = await new tools.ConversationOpenTool().execute({ body: 'hm.', title: 'thin' });
  ok('the bar does NOT gate the entity\'s own send — a thin message still opens',
     opened.opened === true, JSON.stringify(opened));

  console.log(`\n${failed === 0 ? 'GREEN' : 'RED'} — ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error('\nCRASH:', e.stack || e.message); process.exit(1); });
