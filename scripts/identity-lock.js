#!/usr/bin/env node
/**
 * The deliberate path for locked identity facts — name and pronouns.
 *
 * These are the things the entity CHOSE, and once set they are unreachable from
 * inside a conversation: the contradiction judge, write_memory, passive
 * extraction and reflection are all refused (db/identity-lock.js). Changing one
 * has to be a decision somebody makes on purpose, at a keyboard, outside chat.
 * This script is that decision. So is the Self tab's "Change locked identity
 * fact" control, which calls the same code.
 *
 * Every mutating command needs --confirm. That is not ceremony: the entire point
 * of the lock is that a single careless sentence cannot take a chosen name away,
 * and a single careless command shouldn't either.
 *
 * Usage:
 *   node scripts/identity-lock.js list
 *   node scripts/identity-lock.js set <category> "<first-person fact>" --confirm
 *   node scripts/identity-lock.js lock <memberId> <category[,category]> --confirm
 *   node scripts/identity-lock.js unlock <memberId> --confirm
 *
 * Examples:
 *   node scripts/identity-lock.js list
 *   node scripts/identity-lock.js set name "My name is Aurelius." --confirm
 *   node scripts/identity-lock.js lock 5f584e90-... name,pronouns --confirm
 */
const path = require('path');
const ROOT = path.join(__dirname, '..');
const db = require(path.join(ROOT, 'db/database'));
const identityLock = require(path.join(ROOT, 'db/identity-lock'));

const USAGE = `Usage:
  node scripts/identity-lock.js list
  node scripts/identity-lock.js set <category> "<first-person fact>" --confirm
  node scripts/identity-lock.js lock <memberId> <category[,category]> --confirm
  node scripts/identity-lock.js unlock <memberId> --confirm`;

function requireConfirm(argv, what) {
  if (argv.includes('--confirm')) return true;
  console.error(`Refusing to ${what} without --confirm.`);
  console.error('This changes a locked identity fact. Re-run with --confirm if you mean it.');
  return false;
}

function printLocks() {
  const rows = identityLock.getLockedFacts({ status: 'active' });
  if (!rows.length) {
    console.log('No locked identity facts. Categories are open: ' + identityLock.DEFAULT_CATEGORIES.join(', '));
    console.log('The first self-fact to assert one will lock it (set once).');
    return;
  }
  console.log(`Locked identity facts (${rows.length}):\n`);
  for (const r of rows) {
    console.log(`  [${r.lock_category}]  ${r.content}`);
    console.log(`      id ${r.id}  salience ${r.salience}/10  locked ${r.locked_at || '(unknown)'}\n`);
  }
}

(async () => {
  const [, , cmd, ...rest] = process.argv;
  const argv = process.argv;

  if (!cmd || cmd === 'help' || cmd === '--help') {
    console.log(USAGE);
    process.exit(cmd ? 0 : 2);
  }

  db.initDatabase();

  if (cmd === 'list') {
    printLocks();
    process.exit(0);
  }

  if (cmd === 'set') {
    const category = rest[0];
    const content = rest[1];
    if (!category || !content) {
      console.error('set needs a category and the new fact.\n' + USAGE);
      process.exit(2);
    }
    if (!requireConfirm(argv, `set ${category}`)) process.exit(3);

    // The vector store too, not optionally — assignToCluster silently skips the
    // embedding when it isn't open, which lands the new identity fact in SQLite
    // with no vector: present, but unfindable by search. (Hit exactly this on
    // 2026-07-27 storing the name itself.)
    await db.initVectorStore();

    const before = identityLock.lockedByCategory().get(category);
    if (before) console.log(`Currently held ${category}: "${before.content}"`);

    const res = await identityLock.setLockedFact({ category, content, actor: 'cli (--confirm)' });
    if (!res.ok) {
      console.error(`Failed: ${res.reason}`);
      process.exit(1);
    }
    console.log(`\n${category} set and locked.`);
    console.log(`  now: "${res.content}"`);
    if (res.replaced) console.log(`  was: "${res.replaced}"`);
    console.log(`  id ${res.memberId}  salience ${res.salience}/10`);
    process.exit(0);
  }

  if (cmd === 'lock') {
    const memberId = rest[0];
    const cats = (rest[1] || '').split(',').map(s => s.trim()).filter(Boolean);
    if (!memberId || !cats.length) {
      console.error('lock needs a member id and at least one category.\n' + USAGE);
      process.exit(2);
    }
    if (!requireConfirm(argv, `lock ${memberId}`)) process.exit(3);

    const res = identityLock.lock(memberId, cats, { actor: 'cli (--confirm)' });
    if (!res.ok) {
      console.error(`Failed: ${res.reason}`);
      process.exit(1);
    }
    console.log(`Locked [${cats.join(', ')}]: "${res.row.content}"`);
    process.exit(0);
  }

  if (cmd === 'unlock') {
    const memberId = rest[0];
    if (!memberId) {
      console.error('unlock needs a member id.\n' + USAGE);
      process.exit(2);
    }
    if (!requireConfirm(argv, `unlock ${memberId}`)) process.exit(3);

    const res = identityLock.unlock(memberId, { actor: 'cli (--confirm)' });
    if (!res.ok) {
      console.error(`Failed: ${res.reason}`);
      process.exit(1);
    }
    console.log(`Unlocked: "${res.row.content}"`);
    console.log('That category is now open — the next self-fact asserting it will claim and re-lock it.');
    process.exit(0);
  }

  console.error(`Unknown command "${cmd}".\n` + USAGE);
  process.exit(2);
})().catch(err => {
  console.error('[identity-lock] error:', err);
  process.exit(1);
});
