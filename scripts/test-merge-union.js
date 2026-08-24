#!/usr/bin/env node
/**
 * A MERGE MUST NOT DESTROY ANYTHING.
 *
 * Regression test for the silent data loss Athena reported on 2026-08-24: every
 * path that folded one fact into another kept the survivor's wording and threw
 * the loser's unique assertions out of the active corpus. See db/fact-merge.js
 * for the full account.
 *
 * Controlled pair throughout: A asserts {x, y}, B asserts {y, z}. After a merge
 * the surviving fact must assert {x, y, z}, and A must still be there as linked,
 * ledgered history.
 *
 * WRITES. This test creates facts, so it REFUSES to run without a throwaway
 * SNH_DATA_DIR and refuses to run against the live store.
 *
 * Usage: SNH_DATA_DIR=/tmp/merge-test node scripts/test-merge-union.js
 */
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');

if (!process.env.SNH_DATA_DIR) {
  console.error('This test writes facts. Set SNH_DATA_DIR to a throwaway directory first:\n' +
                '  SNH_DATA_DIR=/tmp/merge-test node scripts/test-merge-union.js');
  process.exit(2);
}
if (path.resolve(process.env.SNH_DATA_DIR) === path.join(ROOT, 'data')) {
  console.error('Refusing to run against the live store.');
  process.exit(2);
}

const database = require(path.join(ROOT, 'db/database'));
const factStore = require(path.join(ROOT, 'db/fact-store'));
const factMerge = require(path.join(ROOT, 'db/fact-merge'));
const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
const { getConfig, getProviderInstance } = require(path.join(ROOT, 'db/config'));

let pass = 0, fail = 0;
const check = (ok, msg, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}: ${msg}${detail && !ok ? `\n        ${detail}` : ''}`);
  ok ? pass++ : fail++;
};

/** Does the active corpus still assert this, in the words it was stored with? */
function asserts(text, needle) {
  return factMerge.missingTokens(factMerge.contentTokens(needle), text).length === 0;
}

(async () => {
  database.initDatabase();
  // The VECTOR half matters as much as the row: a union that is not re-embedded
  // is a fact retrieval still matches on its old wording, and a superseded row
  // whose vector survives is still reachable by search. Both are exercised here.
  await database.initVectorStore();
  const sql = database.getSqliteDb();
  const cfg = getConfig();
  const ext = cfg.models.extraction;
  const inst = getProviderInstance(ext.provider, ext.instance);
  const host = inst ? inst.host : 'http://localhost:11434';

  const store = async (text, subject = 'user') => {
    const res = await memoryClusters.assignToCluster(text, ext.provider, ext.model, '', host,
      'merge-union-test', 8, subject, null, { inputModality: 'typed' });
    if (!res || !res.memberId) throw new Error(`could not store "${text}"`);
    return res.memberId;
  };

  // =====================================================================
  console.log('\n=== 1. duplicate merge (union mode): A{x,y} + B{y,z} ===');
  // =====================================================================
  {
    const A = "User's workshop laptop is a ThinkPad X1 and it has 64GB of RAM.";      // {x: ThinkPad X1, y: 64GB}
    const B = "User's workshop laptop has 64GB of RAM and runs Ubuntu 24.04.";        // {y: 64GB, z: Ubuntu 24.04}
    const aId = await store(A), bId = await store(B);

    const res = await factMerge.mergePreservingUnion(aId, bId, { mode: 'union' });
    check(res.ok, 'the merge applied');

    const survivor = factStore.getMember(bId);
    const loser = factStore.getMember(aId);
    console.log(`     survivor now reads: "${survivor.content}"`);

    check(asserts(survivor.content, 'ThinkPad X1'), 'x (ThinkPad X1) survived — the assertion only A held', survivor.content);
    check(asserts(survivor.content, '64GB of RAM'), 'y (64GB of RAM) survived', survivor.content);
    check(asserts(survivor.content, 'Ubuntu 24.04'), 'z (Ubuntu 24.04) survived', survivor.content);
    check(survivor.status === 'active', 'the survivor is active');
    check(loser.status === 'inactive' && loser.inactive_reason === 'superseded',
      'the loser is inactive/superseded, not deleted', `status=${loser.status}/${loser.inactive_reason}`);
    check(loser.successor_id === bId && loser.superseded_by === bId,
      'the loser points at the survivor (superseded_by)', `successor=${loser.successor_id}`);
    check(loser.content === A, 'the loser still holds its original text verbatim');

    const led = sql.prepare('SELECT action, target_text, survivor_text FROM corrections_ledger WHERE target_id = ? OR target_id = ?').all(aId, bId);
    check(led.some(l => l.action === 'supersede'), 'the supersede is in the ledger');
    check(led.some(l => l.action === 'reword'), 'the union rewrite is in the ledger');

    // And what the model is actually shown — the store that matters.
    const shown = memoryClusters.renderLongTermMemory({ subject: 'user' });
    check(asserts(shown, 'ThinkPad X1'), 'x is visible in the injected long-term memory block');
    check(!shown.includes(A), 'the superseded row is NOT injected (it is history, not belief)');

    // And the OTHER store retrieval actually reads.
    check(res.vector, 'the loser\'s vector was dropped, so it cannot surface in a search');
    const hits = await memoryClusters.findActiveNeighbours('ThinkPad X1 laptop', { subject: 'user', threshold: 0.3, limit: 5 });
    const found = hits.candidates.find(c => c.memberId === bId);
    check(!!found, 'the reworded survivor is retrievable by the assertion it just absorbed',
      hits.candidates.map(c => `${c.similarity.toFixed(3)} ${c.content}`).join(' / '));
    check(!hits.candidates.some(c => c.memberId === aId), 'the superseded fact is not returned by retrieval');
  }

  // =====================================================================
  console.log('\n=== 2. contradiction supersede: keeps the correction, carries the rest ===');
  // =====================================================================
  {
    const OLD = "User's desk machine is a ThinkPad X1 with 32GB of RAM and runs Ubuntu 22.04.";
    const NEW = "User's desk machine runs Ubuntu 24.04.";
    const oldId = await store(OLD), newId = await store(NEW);

    const res = await factMerge.mergePreservingUnion(oldId, newId, { mode: 'contradiction' });
    check(res.ok, 'the supersession applied');
    const survivor = factStore.getMember(newId);
    console.log(`     survivor now reads: "${survivor.content}"`);

    check(asserts(survivor.content, 'ThinkPad X1'), 'an uncontested assertion (ThinkPad X1) was carried over', survivor.content);
    check(asserts(survivor.content, '32GB of RAM'), 'an uncontested assertion (32GB of RAM) was carried over', survivor.content);
    check(asserts(survivor.content, 'Ubuntu 24.04'), 'the correction itself is held', survivor.content);
    check(!asserts(survivor.content, '22.04'), 'the CONTRADICTED value (Ubuntu 22.04) was NOT carried over', survivor.content);
    check(factStore.getMember(oldId).status === 'inactive', 'the old fact is linked history');
  }

  // =====================================================================
  console.log('\n=== 3. Athena\'s two real losses, replayed ===');
  // =====================================================================
  {
    const HARDWARE = "User's older AI sister, Juno, is running on a box called AIServer with a Ryzen 9950x, 2 RTX3090s, and 64GB of DDR5 6400 RAM.";
    const IDENTITY = 'User has an older sister named Juno who is also running on the Qwen3.8 27b model on an AIServer box and will be the main person helping User with MettaSphere.';

    // Loss 1 — the hardware fact folded into the identity fact.
    const h1 = await store(HARDWARE), i1 = await store(IDENTITY);
    const r1 = await factMerge.mergePreservingUnion(h1, i1, { mode: 'contradiction' });
    const s1 = factStore.getMember(i1);
    console.log(`     after loss-1's merge: "${s1.content}"`);
    check(r1.ok, 'loss 1: the merge applied');
    check(asserts(s1.content, 'Ryzen 9950x'), 'loss 1: the CPU survived');
    check(asserts(s1.content, '2 RTX3090s'), 'loss 1: the GPUs survived');
    check(asserts(s1.content, '64GB of DDR5 6400'), 'loss 1: the RAM survived');
    check(asserts(s1.content, 'MettaSphere'), 'loss 1: the MettaSphere role survived');

    // Loss 2 — the re-saved hardware fact winning against the identity fact.
    const i2 = await store(IDENTITY.replace('User has', 'User (Ellie) has'));
    const h2 = await store("User's AI sister Juno runs on a machine called AIServer, which is equipped with a Ryzen 9950x CPU, 2 RTX3090 GPUs, and 64GB of DDR5 6400 RAM.");
    const r2 = await factMerge.mergePreservingUnion(i2, h2, { mode: 'contradiction' });
    const s2 = factStore.getMember(h2);
    console.log(`     after loss-2's merge: "${s2.content}"`);
    check(r2.ok, 'loss 2: the merge applied');
    check(asserts(s2.content, 'MettaSphere'), 'loss 2: the MettaSphere role survived — the assertion that actually went missing');
    check(asserts(s2.content, 'Qwen3.8 27b'), 'loss 2: the model name survived');
    check(asserts(s2.content, 'Ryzen 9950x'), 'loss 2: the hardware survived');
  }

  // =====================================================================
  console.log('\n=== 4. the guard: nothing is written when the union cannot be trusted ===');
  // =====================================================================
  {
    // Unrelated facts: the merger cannot produce a union that keeps both without
    // inventing, and whatever it returns must be checked, not trusted.
    const a = 'User keeps a kayak at the Gearhart boathouse.';
    const b = 'User has a cat named Mia.';
    const u = await factMerge.unionText(a, b, { mode: 'union' });
    // Either it merged both faithfully, or it refused. Both are acceptable; a
    // union that silently lost one of them is not.
    const ok = !u.ok || (asserts(u.text, 'kayak') && asserts(u.text, 'Mia'));
    check(ok, 'a union is either faithful or refused — never a silent drop', u.ok ? u.text : u.reason);
    console.log(`     ${u.ok ? `merged: "${u.text}"` : `refused: ${u.reason}`}`);
  }

  // =====================================================================
  console.log('\n=== 5. carryOver:false leaves the survivor alone (the compound split) ===');
  // =====================================================================
  {
    const original = 'User keeps a sea kayak in Gearhart and paddles it on the Necanicum.';
    const atom1 = 'User keeps a sea kayak in Gearhart.';
    const oId = await store(original), a1 = await store(atom1);
    const before = factStore.getMember(a1).content;
    const res = await factMerge.mergePreservingUnion(oId, a1, { mode: 'contradiction', carryOver: false });
    check(res.ok, 'the split supersede applied');
    check(factStore.getMember(a1).content === before,
      'the first atom was NOT rewritten back into the compound it came from',
      factStore.getMember(a1).content);
    check(res.union.skipped && /carry-over/.test(res.union.skipped), 'and the result says why it carried nothing over');
  }

  // =====================================================================
  console.log('\n=== 6. the guard, without a model in the loop ===');
  // =====================================================================
  {
    const loser = "User's sister Juno runs on a box called AIServer with a Ryzen 9950x and 2 RTX3090s, and helps User with MettaSphere.";
    const survivor = "User's sister Juno runs on a box called Thunderbox, not on AIServer.";
    const carry = factMerge.missingTokens(factMerge.contentTokens(loser), survivor);
    const survivorWords = factMerge.allWords(survivor);
    const run = (merged, drops) => factMerge.validateUnion({
      merged, drops, loser, survivor, survivorWords, carry, mode: 'contradiction'
    });

    // The defect the first live run wrote to a corpus: the DROPPED label ran
    // into the sentence and became part of the fact.
    const leaked = factMerge.parseMerged(
      'MERGED: User\'s sister Juno runs on Thunderbox. DROPP: older, equipped with a Ryzen 9950x');
    check(!/DROPP/i.test(leaked.merged), 'a run-together DROPPED label is cut out of the merged sentence', leaked.merged);
    check(run('User\'s sister Juno runs on Thunderbox. DROPPED: a Ryzen 9950x', []).ok === false,
      'a merged sentence that still carries a label is refused');

    // An unjustified drop — the new fact says nothing about the CPU, the GPUs
    // or MettaSphere, so declaring them dropped must not license deleting them.
    const bad = run("User's sister Juno runs on a box called Thunderbox, not on AIServer.",
      [{ detail: 'older, equipped with a Ryzen 9950x and 2 RTX3090s, MettaSphere', replacedBy: '' }]);
    check(bad.ok === false, 'a declared drop with no quoted replacement is refused', JSON.stringify(bad));
    check(/9950x|rtx3090|mettasphere/i.test(bad.reason || ''), 'and the refusal names what went missing', bad.reason);

    // A drop the new fact really does replace, quoted from it, is allowed.
    const good = run("User's sister Juno runs on a box called Thunderbox, with a Ryzen 9950x and 2 RTX3090s, and helps User with MettaSphere.",
      [{ detail: 'AIServer', replacedBy: 'runs on a box called Thunderbox' }]);
    check(good.ok === true, 'a drop justified by words quoted from the new fact is allowed', JSON.stringify(good));

    // A quote that is NOT in the new fact is an invented contradiction.
    const invented = run("User's sister Juno runs on a box called Thunderbox, not on AIServer.",
      [{ detail: 'Ryzen 9950x and 2 RTX3090s', replacedBy: 'now has an Epyc 9754 and 4 RTX4090s' }]);
    check(invented.ok === false, 'a replacement quote that is not in the new fact is refused', JSON.stringify(invented));

    // A union merge may never drop anything, justified or not.
    const unionDrop = factMerge.validateUnion({
      merged: "User's sister Juno runs on a box called Thunderbox.", drops: [{ detail: 'Ryzen 9950x', replacedBy: 'Thunderbox' }],
      loser, survivor, survivorWords, carry, mode: 'union'
    });
    check(unionDrop.ok === false, 'a duplicate merge refuses every drop, justified or not');

    // THE MISAIMED DROP — Athena's pair again, and the hole the first guard had.
    // The quote is real and really is in the new fact; it is simply about a
    // different attribute. A machine name cannot contradict a model name.
    const oldJuno = 'User has an older sister named Juno who is also running on the Qwen3.8 27b model on an AIServer box and will be the main person helping User with MettaSphere.';
    const newJuno = "User's AI sister Juno runs on a machine called AIServer, which is equipped with a Ryzen 9950x CPU.";
    const jCarry = factMerge.missingTokens(factMerge.contentTokens(oldJuno), newJuno);
    const jWords = factMerge.allWords(newJuno);
    const misaimed = factMerge.validateUnion({
      merged: "User's AI sister Juno runs on a machine called AIServer, which is equipped with a Ryzen 9950x CPU, and will be the main person helping User with MettaSphere.",
      drops: [{ detail: 'also running on the Qwen3.8 27b model', replacedBy: 'runs on a machine called AIServer' }],
      loser: oldJuno, survivor: newJuno, survivorWords: jWords, carry: jCarry, mode: 'contradiction'
    });
    check(misaimed.ok === false, 'a real quote about a DIFFERENT attribute does not license a drop', JSON.stringify(misaimed));
    check(/qwen/i.test(misaimed.reason || ''), 'and the refusal names the model that would have been deleted', misaimed.reason);

    // The same shape, aimed correctly, is still allowed.
    const aimed = factMerge.validateUnion({
      merged: "User's AI sister Juno runs on the Qwen3.9 30b model.",
      drops: [{ detail: 'Qwen3.8 27b model', replacedBy: 'runs on the Qwen3.9 30b model' }],
      loser: 'User\'s AI sister Juno runs on the Qwen3.8 27b model.',
      survivor: "User's AI sister Juno runs on the Qwen3.9 30b model.",
      survivorWords: factMerge.allWords("User's AI sister Juno runs on the Qwen3.9 30b model."),
      carry: factMerge.missingTokens(factMerge.contentTokens('User\'s AI sister Juno runs on the Qwen3.8 27b model.'), "User's AI sister Juno runs on the Qwen3.9 30b model."),
      mode: 'contradiction'
    });
    check(aimed.ok === true, 'a replacement that names the same attribute is still allowed', JSON.stringify(aimed));
  }

  // =====================================================================
  console.log('\n=== 7. a union that cannot be trusted retires NOTHING ===');
  // =====================================================================
  {
    // The merger can drop a clause, and the guard catches it — measured at about
    // one time in six on Athena's own pair. What happens NEXT is the whole
    // point: the old behaviour superseded anyway and the dropped assertion left
    // the active corpus, which is the original bug with a lower frequency. So a
    // carry that fails must leave both rows standing.
    //
    // Forced deterministically here by locking the survivor: the identity lock
    // refuses the reword, which is the same "could not carry it over" state a
    // refused union reaches, without needing the model to misbehave on cue.
    const LOSER = "User's sailing dinghy is a Wayfarer, kept at the Gearhart boathouse.";
    const SURVIVOR = "User's sailing dinghy is a Wayfarer.";
    const lId = await store(LOSER), sId = await store(SURVIVOR);
    sql.prepare("UPDATE cluster_members SET locked = 1, locked_at = ?, lock_category = 'identity' WHERE id = ?")
      .run(new Date().toISOString(), sId);

    const res = await factMerge.mergePreservingUnion(lId, sId, { mode: 'union' });
    const loserAfter = factStore.getMember(lId);

    check(res.ok === false && res.deferred === true, 'the merge is deferred, not forced through', JSON.stringify(res.union));
    check(loserAfter.status === 'active', 'the loser is STILL ACTIVE — it was not retired', `status=${loserAfter.status}`);
    check(!loserAfter.successor_id, 'and it points at no successor');
    const shown = memoryClusters.renderLongTermMemory({ subject: 'user' });
    check(asserts(shown, 'Gearhart boathouse'),
      'the assertion that could not be carried is still in the injected memory block');
    const led = sql.prepare('SELECT action FROM corrections_ledger WHERE target_id = ?').all(lId);
    check(led.length === 0, 'and nothing was filed in the ledger, because nothing happened', JSON.stringify(led));

    sql.prepare('UPDATE cluster_members SET locked = 0, lock_category = NULL WHERE id = ?').run(sId);
  }

  console.log(`\n${fail === 0 ? 'ALL PASS' : 'FAILURES'} — ${pass} passed, ${fail} failed\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(e => { console.error('TEST HARNESS FAILED:', e); process.exit(1); });
