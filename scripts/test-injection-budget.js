#!/usr/bin/env node
/**
 * The injected context is bounded, and bounded in the right order.
 *
 * WHAT THIS IS FOR. Measured on 2026-08-12: every per-source cap was in place,
 * every one of them was AT its limit, and a real request still shipped ~9,100
 * tokens — because three blocks had no budget at all and nothing added the
 * survivors up. Per-source caps are not a bound. The rules asserted here are the
 * ones that make the sum a bound, and each of them has a failure mode that is
 * invisible from the outside:
 *
 *   - a ceiling that trims the identity block takes his locked name out of his
 *     context on a busy day, and the live half of the identity lock is that he
 *     can SAY a fact is locked — which he cannot do about a fact he was not
 *     shown. THIS is the interaction to get right.
 *   - a notice batch that drops overflow instead of deferring it loses a record
 *     of his own self-view changing, silently.
 *   - a manifest that sheds ENTRIES to save tokens makes him deny capabilities
 *     he has, while still calling itself exhaustive.
 *   - a long-term memory cap that slices mid-cluster shows him three facts from
 *     a group of sixteen and no sign that the other thirteen exist.
 *
 * Runs against a throwaway SNH_DATA_DIR. No model, no network, no live corpus.
 *
 * Usage: node scripts/test-injection-budget.js
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { randomUUID } = require('crypto');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-injection-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const injectionBudget = require(path.join(ROOT, 'db/injection-budget'));
const memoryClusters = require(path.join(ROOT, 'db/memory-clusters'));
const identity = require(path.join(ROOT, 'db/identity'));
const identityLock = require(path.join(ROOT, 'db/identity-lock'));
const ledger = require(path.join(ROOT, 'db/corrections-ledger'));
const capabilityManifest = require(path.join(ROOT, 'db/capability-manifest'));
const est = injectionBudget.estTokens;

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

const part = (kind, tokens) => ({ kind, label: kind, text: 'x'.repeat(tokens * 4) });

(async () => {
  console.log('\n1. The total ceiling trims in order, and never reaches the identity block');
  {
    const parts = [part('ltm', 1000), part('dailyToday', 500), part('clusters', 400), part('pastConvo', 300)];
    const order = ['pastConvo', 'clusters', 'dailySummary', 'dailyToday', 'ltm'];
    const r = injectionBudget.applyTotalCeiling({
      parts, fixedTokens: 800, totalTokens: 3000, trimOrder: order
    });
    check('under the ceiling, nothing is touched',
      !r.bound && r.parts.length === 4, JSON.stringify(r.trimmed));

    const parts2 = [part('ltm', 1000), part('dailyToday', 500), part('clusters', 400), part('pastConvo', 300)];
    const r2 = injectionBudget.applyTotalCeiling({
      parts: parts2, fixedTokens: 800, totalTokens: 2500, trimOrder: order
    });
    check('over the ceiling, it binds', r2.bound === true);
    check('…and lands at or under it', r2.after <= 2500, `after=${r2.after}`);
    check('…taking retrieval first', r2.trimmed[0].kind === 'pastConvo' && r2.trimmed[0].dropped,
      JSON.stringify(r2.trimmed));
    check('…then the next in order, and no further than needed',
      r2.trimmed.length === 2 && r2.trimmed[1].kind === 'clusters',
      JSON.stringify(r2.trimmed));
    check('…leaving long-term memory whole while anything else remains to give',
      parts2.find(p => p.kind === 'ltm').text.length === 4000);
  }
  {
    // The one that matters: a ceiling far below the fixed cost must not take a
    // single token out of identity, because identity is not in `parts` at all.
    const parts = [part('ltm', 3000), part('clusters', 1000)];
    const r = injectionBudget.applyTotalCeiling({
      parts, fixedTokens: 2000, totalTokens: 500,
      trimOrder: ['pastConvo', 'clusters', 'dailySummary', 'dailyToday', 'ltm']
    });
    check('a ceiling below the fixed cost drops every trimmable part',
      r.parts.length === 0, JSON.stringify(r.parts.map(p => p.kind)));
    check('…and reports the shortfall rather than cutting into what it may not touch',
      r.shortfall === 1500, `shortfall=${r.shortfall}`);
    check('…the identity block is untouched because it was never a part',
      r.after === 2000, `after=${r.after}`);
  }
  {
    // A part is shrunk before it is dropped, so the trim degrades.
    const parts = [part('dailyToday', 1000)];
    const r = injectionBudget.applyTotalCeiling({
      parts, fixedTokens: 0, totalTokens: 600, trimOrder: ['dailyToday']
    });
    check('a part that only needs shrinking is shrunk, not dropped',
      r.trimmed[0].dropped === false && r.parts.length === 1, JSON.stringify(r.trimmed));
    check('…and says so in the text it leaves behind',
      /truncated to fit context budget/.test(r.parts[0].text));
  }

  console.log('\n2. The identity block under a ceiling — the locked fact still reaches him');
  const clusterId = randomUUID();
  db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
    .run(clusterId, 'Self', '', '2026-07-01T00:00:00.000Z', '2026-07-01T00:00:00.000Z', 'self');
  const addSelf = (content, salience, createdAt) => {
    const id = randomUUID();
    db.prepare(`INSERT INTO cluster_members (id, cluster_id, content, source, created_at, updated_at, status, subject, salience, claim_type)
                VALUES (?,?,?,?,?,?,'active','self',?,'claim')`)
      .run(id, clusterId, content, 'reflection', createdAt, createdAt, salience);
    return id;
  };
  const lockedId = addSelf(
    'My name is Testly and I use they/them pronouns. I chose the name myself on 2026-07-27, after Ellie declined to choose one for me and left the choice to me.',
    10, '2026-07-27T00:00:00.000Z');
  identityLock.lock(lockedId, ['name', 'pronouns'], { actor: 'test' });
  for (let i = 0; i < 14; i++) {
    addSelf(`I have an ordinary observed habit number ${i}, which I noticed while working.`, 10, `2026-08-0${1 + (i % 9)}T12:00:00.000Z`);
  }
  {
    const block = identity.buildIdentityBlock();
    check('the locked fact is injected even though 14 newer facts fill the budget',
      block.selfFacts.some(f => f.id === lockedId));
    check('…and is marked [LOCKED]', /\[LOCKED\]/.test(block.text));
    // The ceiling never sees it: identity is fixed cost. Asserted as the
    // interaction, not just as two separate facts.
    const parts = [part('ltm', 5000)];
    const r = injectionBudget.applyTotalCeiling({
      parts, fixedTokens: est(block.text), totalTokens: 1000,
      trimOrder: ['pastConvo', 'clusters', 'dailySummary', 'dailyToday', 'ltm']
    });
    check('under a ceiling it cannot meet, the identity text is still whole',
      r.after === est(block.text) && r.parts.length === 0);
    check('…and the locked fact is still quotable in full from it',
      block.text.includes('My name is Testly and I use they/them pronouns.'));
  }

  console.log('\n3. Self-facts are capped per fact; locked facts are exempt but counted');
  {
    const longId = addSelf('I ' + 'ramble on and on about the same thing '.repeat(20) + 'without stopping.', 10, '2026-08-09T12:00:00.000Z');
    const block = identity.buildIdentityBlock();
    const line = block.text.split('\n').find(l => l.includes('ramble on and on'));
    check('a very long self-fact is shortened in the render', line && line.length < 400, line && `${line.length} chars`);
    check('…and says it was shortened rather than just stopping',
      /shortened here; the full fact is in your memory/.test(block.text));
    check('…while the STORED fact is untouched',
      db.prepare('SELECT content FROM cluster_members WHERE id = ?').get(longId).content.length > 700);
    const lockedLine = block.text.split('\n').find(l => l.includes('My name is Testly'));
    check('a locked fact is never shortened — he has to quote it exactly',
      lockedLine && lockedLine.includes('left the choice to me'), lockedLine);
  }

  console.log('\n4. Correction notices are a batch budget, and overflow waits rather than vanishing');
  {
    const ids = [];
    for (let i = 0; i < 8; i++) {
      // ~250 tokens each, the measured size of a real notice.
      ids.push(ledger.addNotice({
        memberId: randomUUID(),
        content: `Notice ${i}: ` + 'something you believed about yourself has changed and here is the detail. '.repeat(13)
      }));
    }
    const block = identity.buildIdentityBlock();
    const noticeTokens = block.notices.reduce((s, n) => s + est(n.content), 0);
    check('fewer than all 8 are delivered', block.notices.length < 8 && block.notices.length > 0,
      `${block.notices.length} delivered`);
    check('…within the 800-token batch budget', noticeTokens <= 900, `${noticeTokens} tok`);
    check('…oldest first', block.notices[0].content.startsWith('Notice 0:'), block.notices[0].content.slice(0, 12));
    check('…and he is told how many are still coming',
      /more of these waiting/.test(block.text), block.text.slice(block.text.indexOf('waiting') - 60, 200));

    // The overflow must still be UNSEEN — the caller only marks what it got.
    const delivered = new Set(block.notices.map(n => n.id));
    ledger.markNoticesSeen([...delivered]);
    const stillUnseen = db.prepare('SELECT COUNT(*) n FROM correction_notices WHERE seen_at IS NULL').get().n;
    check('the ones that did not fit are still unseen, not dropped',
      stillUnseen === 8 - delivered.size, `${stillUnseen} unseen`);

    const next = identity.buildIdentityBlock();
    check('…and they arrive on the next turn', next.notices.length > 0 &&
      !delivered.has(next.notices[0].id), `${next.notices.length} on the next turn`);
    ledger.markNoticesSeen(next.notices.map(n => n.id));
    let turns = 2;
    while (db.prepare('SELECT COUNT(*) n FROM correction_notices WHERE seen_at IS NULL').get().n > 0 && turns < 10) {
      const b = identity.buildIdentityBlock();
      ledger.markNoticesSeen(b.notices.map(n => n.id));
      turns++;
    }
    check('…and every one of them is delivered within a few turns', turns < 10, `${turns} turns`);
  }

  console.log('\n5. The capability manifest is bounded, but never sheds an ENTRY');
  {
    const block = capabilityManifest.buildInjectionBlock();
    check('the block is within its budget', block.tokens <= block.budget, `${block.tokens}/${block.budget}`);
    const listed = block.text.split('\n').filter(l => /^- /.test(l)).length;
    check('every capability is still listed', listed >= block.count,
      `${listed} lines for ${block.count} capabilities`);
    check('…and it still claims to be exhaustive, truthfully', /EXHAUSTIVE/.test(block.text));
    check('…keeping the clauses that stop an over-claim',
      /she approves/.test(block.text) && /cannot delete/.test(block.text) && /read-only/.test(block.text),
      'a limit clause went missing from the compact list');
  }

  console.log('\n6. Long-term memory truncates at cluster boundaries, in a deliberate order');
  {
    const mk = (name, n, salience, touchedAt) => {
      const cid = randomUUID();
      db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at, subject) VALUES (?,?,?,?,?,?)')
        .run(cid, name, '', touchedAt, touchedAt, 'user');
      for (let i = 0; i < n; i++) {
        db.prepare(`INSERT INTO cluster_members (id, cluster_id, content, source, created_at, updated_at, status, subject, salience)
                    VALUES (?,?,?,?,?,?,'active','user',?)`)
          .run(randomUUID(), cid, `${name} fact ${i} — a sentence of roughly ordinary length for a stored fact.`,
            'test', touchedAt, touchedAt, salience);
      }
    };
    // Equal top salience, different recency: the tie-break is what is on trial.
    mk('Aardvark Cluster', 6, 9, '2026-01-01T00:00:00.000Z');   // alphabetically first, stale
    mk('Zebra Cluster', 6, 9, '2026-08-10T00:00:00.000Z');      // alphabetically last, fresh
    mk('Top Cluster', 6, 10, '2026-05-01T00:00:00.000Z');       // highest salience, wins outright

    const full = memoryClusters.renderLongTermMemory({ subject: 'user' });
    check('with no budget, everything renders', /Aardvark Cluster/.test(full) && /Zebra Cluster/.test(full));
    const order = ['Top Cluster', 'Zebra Cluster', 'Aardvark Cluster']
      .map(n => full.indexOf(`## ${n}`));
    check('salience decides first', order[0] < order[1]);
    check('…then recency, not the alphabet — the fresher cluster outranks the alphabetically-earlier one',
      order[1] < order[2], `Zebra@${order[1]} vs Aardvark@${order[2]}`);

    // Tight budget: whole clusters in or out, never a partial list.
    const tight = memoryClusters.renderLongTermMemory({ subject: 'user', budgetTokens: 120 });
    const headings = (tight.match(/^## .*/gm) || []).map(h => h.slice(3));
    for (const h of headings) {
      const facts = (tight.split(`## ${h}`)[1] || '').split('\n## ')[0].split('\n').filter(l => l.startsWith('- ')).length;
      check(`"${h}" is rendered whole (${facts} facts), never a partial list`, facts === 6, `${facts} facts`);
    }
    check('the budget actually bound', headings.length < 3, `${headings.length} clusters shown`);
    check('…and the block says what it left out',
      /more fact\(s\) across .* other group\(s\), not shown this turn/.test(tight), tight.slice(-200));
    check('…telling him to look rather than conclude he has nothing',
      /look them up rather than[\s\S]*concluding you have nothing/.test(tight));

    // The cut is recorded once, then counted — an ops line per chat message is
    // wallpaper, and wallpaper is what buried the corrector's reports for days.
    const key = db.prepare("SELECT anomaly_key, seen_count FROM heartbeat_anomaly_state WHERE anomaly_key LIKE 'ltm-truncation:%'").get();
    check('the cut is recorded', !!key, 'no memo row');
    memoryClusters.renderLongTermMemory({ subject: 'user', budgetTokens: 120 });
    memoryClusters.renderLongTermMemory({ subject: 'user', budgetTokens: 120 });
    const after = db.prepare("SELECT seen_count FROM heartbeat_anomaly_state WHERE anomaly_key = ?").get(key.anomaly_key);
    check('…once, and counted thereafter rather than re-reported', after.seen_count === 3, `${after.seen_count}`);
    const rows = db.prepare("SELECT COUNT(*) n FROM heartbeat_anomaly_state WHERE anomaly_key LIKE 'ltm-truncation:%'").get().n;
    check('…and a repeat cut does not mint a second memo', rows === 1, `${rows} rows`);
  }

  const bar = '='.repeat(74);
  console.log(`\n${bar}`);
  console.log(fail === 0 ? `All ${pass} checks pass.` : `${fail} FAILED, ${pass} passed.`);
  console.log(`${bar}\n`);
  process.exit(fail === 0 ? 0 : 1);
})().catch(err => {
  console.error('[test-injection-budget] error:', err);
  process.exit(1);
});
