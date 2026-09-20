#!/usr/bin/env node
/**
 * Export the active memory corpus to a readable file.
 *
 * MEMORY.md stopped being a store on 2026-08-02 (docs/memory-mvp-spec.md,
 * decision 1). The injected long-term block is rendered from SQLite on every
 * request, so there is no file to drift out of step — but a file you can open in
 * an editor is still worth having, so this writes one ON DEMAND.
 *
 * The output is a DEAD END by design: nothing reads it back, no code path parses
 * it, and editing it changes nothing. That is the entire difference between an
 * export and a store.
 *
 * Usage:
 *   node scripts/export-memory.js                    # → data/memory/export/memory-<date>.md
 *   node scripts/export-memory.js --out /tmp/mem.md  # explicit path
 *   node scripts/export-memory.js --subject self     # self-facts instead of user facts
 *   node scripts/export-memory.js --all              # both corpora in one file
 *   node scripts/export-memory.js --stdout           # print instead of writing
 */

const fs = require('fs');
const path = require('path');

const database = require('../db/database');
const { getLocalDateStamp } = require('../db/datetime');

function parseArgs(argv) {
  const args = { subject: 'user', out: null, stdout: false, all: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') args.out = argv[++i];
    else if (a === '--subject') args.subject = argv[++i];
    else if (a === '--stdout') args.stdout = true;
    else if (a === '--all') args.all = true;
    else if (a === '--help' || a === '-h') args.help = true;
    else {
      console.error(`Unknown argument: ${a}`);
      args.help = true;
    }
  }
  return args;
}

function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(fs.readFileSync(__filename, 'utf8').split('*/')[0].split('/**')[1]
      .split('\n').map(l => l.replace(/^\s*\*ceci?/, '').replace(/^\s*\* ?/, '')).join('\n'));
    process.exit(0);
  }
  if (!['user', 'self'].includes(args.subject)) {
    console.error(`--subject must be 'user' or 'self' (got '${args.subject}')`);
    process.exit(1);
  }

  database.initDatabase();
  const memoryClusters = require('../db/memory-clusters');
  const db = database.getSqliteDb();

  const subjects = args.all ? ['user', 'self'] : [args.subject];
  const stamp = new Date().toISOString();

  const parts = [
    `<!-- Exported ${stamp} by scripts/export-memory.js.`,
    '     This is a SNAPSHOT, not a store. Nothing reads this file back —',
    '     the injected memory block is rendered from SQLite per request.',
    '     Editing this file changes nothing. -->',
    ''
  ];

  let total = 0;
  for (const subject of subjects) {
    const rendered = memoryClusters.renderLongTermMemory({ subject });
    const count = db.prepare(
      "SELECT COUNT(*) AS n FROM cluster_members WHERE subject = ? AND status = 'active'"
    ).get(subject).n;
    total += count;

    if (subjects.length > 1) {
      parts.push(`<!-- ${subject} corpus: ${count} active fact(s) -->`, '');
    }
    parts.push(rendered || `# Long-Term Memory\n\n(no active ${subject} facts)`, '');
  }

  const content = parts.join('\n').trimEnd() + '\n';

  if (args.stdout) {
    process.stdout.write(content);
    return;
  }

  let outPath = args.out;
  if (!outPath) {
    const dir = path.join(__dirname, '../data/memory/export');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const suffix = args.all ? 'all' : args.subject;
    outPath = path.join(dir, `memory-${getLocalDateStamp()}-${suffix}.md`);
  }

  fs.writeFileSync(outPath, content, 'utf8');
  console.log(`Exported ${total} active fact(s) (${subjects.join(' + ')}) → ${outPath}`);
  console.log('Nothing reads this file back; it is a snapshot for you, not a store.');
}

main();
