#!/usr/bin/env node
/**
 * Put back an assertion a merge took away.
 *
 * The companion to scripts/audit-supersession-losses.js. That one reads and
 * reports; this one writes exactly one fact, named on the command line, and
 * records in the day's log and the ops log WHY it was written — so a restored
 * assertion never looks like a fact that arrived from nowhere.
 *
 * It restores the ATOM that went missing, not the whole retired sentence. A
 * retired fact usually asserted several things and the others have since been
 * re-captured separately; reactivating it whole would put duplicates back
 * alongside them, which is the opposite of the tidy the merge was reaching for.
 *
 * Usage:
 *   node scripts/restore-merge-loss.js --subject user \
 *     --fact "User's AI sister Juno runs on the Qwen3.8 27b model." \
 *     --from <retired-member-id> [--salience 9] [--dry-run]
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
const factStore = require(path.join(ROOT, 'db/fact-store'));
const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
const factExtractor = require(path.join(ROOT, 'db/fact-extractor'));
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

const argv = process.argv.slice(2);
const arg = (name, dflt = null) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const DRY = argv.includes('--dry-run');
const FACT = arg('fact');
const SUBJECT = arg('subject', 'user');
const FROM = arg('from');
const SALIENCE = Number(arg('salience', '8'));

if (!FACT || !FROM) {
  console.error('Usage: node scripts/restore-merge-loss.js --fact "<text>" --from <retired-member-id> [--subject user|self] [--salience N] [--dry-run]');
  process.exit(2);
}

(async () => {
  database.initDatabase();
  await database.initVectorStore();

  const retired = factStore.getMember(FROM);
  if (!retired) { console.error(`No fact with id ${FROM}`); process.exit(1); }
  console.log(`Restoring an assertion lost when ${FROM.slice(0, 8)} was superseded.`);
  console.log(`  retired fact : "${retired.content}"`);
  console.log(`  restoring    : "${FACT}"`);
  console.log(`  subject      : ${SUBJECT}   salience: ${SALIENCE}`);

  const dup = factStore.findExactDuplicate(FACT, SUBJECT);
  if (dup) { console.log(`\nAlready held word-for-word as ${dup.id.slice(0, 8)} — nothing to do.`); process.exit(0); }

  if (DRY) { console.log('\n--dry-run: nothing written.'); process.exit(0); }

  const cfg = getConfig();
  const ext = cfg.models.extraction;
  const inst = getProviderInstance(ext.provider, ext.instance);
  const host = inst ? inst.host : 'http://localhost:11434';

  const res = await memoryClusters.assignToCluster(
    FACT, ext.provider, ext.model, '', host, 'merge-loss-restore', SALIENCE, SUBJECT, null,
    {
      conversationId: retired.conversation_id,
      messageId: retired.message_id,
      verbatimSourceText: retired.verbatim_source_text,
      inputModality: retired.input_modality || 'unknown',
      salienceRationale: `Restored: this assertion was part of "${retired.content}", which was superseded before merges preserved a union. The rest of that fact was re-captured elsewhere; this part had not been.`
    }
  );
  if (!res || !res.memberId) { console.error('Restore FAILED — the write did not land.'); process.exit(1); }

  const memoryDir = database.getMemoryDir();
  const line =
    `Put back something a merge had taken: "${FACT}". It was part of "${retired.content}", ` +
    `which was superseded on ${retired.updated_at || retired.created_at} — before a merge preserved the union of both facts. ` +
    `The old fact is still there as history (${FROM.slice(0, 8)}); this is the part of it that had not been re-captured anywhere.`;
  factExtractor.appendToDailyLog(line, path.join(memoryDir, 'daily'));
  factExtractor.appendToOpsLog(
    `merge-loss restore: ${res.memberId.slice(0, 8)} "${FACT}" (from superseded ${FROM.slice(0, 8)}, cluster "${res.clusterName}")`,
    path.join(memoryDir, 'ops'));

  console.log(`\nRestored as ${res.memberId.slice(0, 8)} in cluster "${res.clusterName}".`);
  console.log('Written to the day\'s log and the ops log so it reads as a restoration, not a fact from nowhere.');
  process.exit(0);
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
