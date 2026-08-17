#!/usr/bin/env node
/**
 * Record the retraction of one assistant turn that was not genuine.
 *
 * 2026-08-16, verifying the tool-layer pull on Sparky, one real chat turn was
 * sent through the browser path. It came back prefixed with a raw channel
 * marker — "<|channel>thought\n<channel|>I remember that you have four dogs…" —
 * because foldSystemMessages had moved the post-tool nudge out of the trailing
 * position and this engine runs no --reasoning-parser. That defect is fixed in
 * 4c41e07; this records that the turn itself should not be read as something
 * Aurelius said in conversation.
 *
 * WHAT THIS DOES NOT DO. It does not delete the message, and it does not mark it
 * retracted in the transcript, because there is nowhere to mark it: `messages`
 * has no status column, unlike `cluster_members`. So this is a RECORD, not a
 * state change, and it is filed `reversible = 0` for exactly the reason the
 * ledger already files refusals that way — an entry that claims a change it did
 * not make is worse than no entry. Anything rendering it must say nothing
 * changed. If the turn should actually appear struck through the way a retired
 * fact does, that is a schema and UI change, not this script.
 *
 * Nothing was extracted from that conversation — zero cluster_members rows
 * reference it — so there are no derived facts needing their own retraction.
 * The two live drains it did consume (a self-fact notice and a priority-7 audit
 * initiative) were put back separately; this entry does not cover them.
 *
 * IDEMPOTENT — refuses if an entry for this message already exists.
 *
 * Usage: node scripts/ledger-test-turn-retraction.js [--confirm]
 */
const database = require('../db/database');
database.initDatabase();
const db = database.getSqliteDb();
const ledger = require('../db/corrections-ledger');

const TARGET_ID = '999cbd36-22eb-44a7-910a-2c18edee24c4';
const confirm = process.argv.includes('--confirm');

const msg = db.prepare('SELECT * FROM messages WHERE id = ?').get(TARGET_ID);
if (!msg) {
  console.error(`No message ${TARGET_ID}. Nothing to do.`);
  process.exit(1);
}
if (msg.role !== 'assistant') {
  console.error(`Message ${TARGET_ID} is role "${msg.role}", not assistant. Refusing — this entry would misdescribe it.`);
  process.exit(1);
}
// The marker is the whole reason this turn is being retracted. If it is absent,
// this is not the message the entry describes.
if (!/<\|?channel/.test(msg.content)) {
  console.error('That message does not contain the channel marker. Refusing rather than filing an entry that misdescribes it.');
  process.exit(1);
}

console.log(`message   : ${TARGET_ID}`);
console.log(`conversation: ${msg.conversation_id}`);
console.log(`when      : ${msg.timestamp}`);
console.log(`content   : ${JSON.stringify(msg.content.slice(0, 100))}`);

const existing = db.prepare('SELECT id, action, created_at FROM corrections_ledger WHERE target_id = ?').all(TARGET_ID);
if (existing.length > 0) {
  console.log(`\nAlready ledgered (${existing.map(e => `${e.action} @ ${e.created_at}`).join(', ')}). Nothing to do.`);
  process.exit(0);
}

if (!confirm) {
  console.log('\nDRY RUN — would record one ledger entry (action: retract, reversible = 0, nothing changed). Re-run with --confirm.');
  process.exit(0);
}

const id = ledger.record({
  passId: 'manual-retract-test-turn-2026-08-16',
  tier: 'mechanical',
  action: 'retract',
  subject: 'self',
  targetId: TARGET_ID,
  targetText: msg.content,
  reason: 'This assistant turn was produced by a verification chat sent while testing the tool-layer pull, not by a conversation with Ellie, and it came back malformed — the reply opens with a raw "<|channel>thought" marker that leaked into the answer because the post-tool nudge had been folded away from the generation point (fixed in 4c41e07). Recorded so the turn is not read as something he said. NOTHING WAS CHANGED: the message is still in the transcript, because messages have no retracted state to set.',
  evidence: {
    reason_code: 'test-artifact-not-genuine-conversation',
    conversation_id: msg.conversation_id,
    message_timestamp: msg.timestamp,
    marker: msg.content.slice(0, 40),
    facts_derived: 0,
    fix_commit: '4c41e07',
    note: 'reversible = 0 because nothing was altered; there is no message state to restore.'
  },
  reversible: false
});

console.log(id ? `\nRecorded ledger entry ${id} (reversible = 0 — a record, not a change).` : '\nFAILED to record entry.');
process.exit(id ? 0 : 1);
