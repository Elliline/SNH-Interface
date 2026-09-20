#!/usr/bin/env node
/**
 * File output for job results, tested where it can actually be observed.
 *
 * Everything asserted here is a rule whose failure is QUIET in production. A
 * result that should have become a file and stayed on a card looks like a card;
 * a report that came out as text because chromium is missing looks like a
 * report; a summary that is really the whole document looks like a summary until
 * you scroll. None of them throw, and none of them appear in a log — which is
 * exactly the class of bug the rest of this system is written against.
 *
 * The three that matter most, and why each is here:
 *
 *   THE MODEL IS NEVER CALLED. Every case below is a pure function of a string,
 *   because the classifier's decisions are the thing under test and a real run
 *   would make half of them unreachable.
 *
 *   NOTHING TOUCHES HER DOCUMENTS FOLDER. SNH_DATA_DIR is redirected, and
 *   job-artifacts.outputDir() honours that by putting the folder inside the
 *   throwaway data directory. That is asserted here rather than assumed: the
 *   whole redirect exists so a disposable process cannot write live state, and a
 *   file in ~/SNH_Documents is live state.
 *
 *   A MISSING CHROMIUM IS A DOWNGRADE, NEVER A FAILURE. On this box there is no
 *   chromium at all, so the text-fallback path is the one that actually runs —
 *   which makes it the ordinary path and not an edge case.
 *
 * Usage: node scripts/test-job-artifacts.js
 */
process.env.TZ = 'America/Los_Angeles';

const fs = require('fs');
const os = require('os');
const path = require('path');

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'snh-job-artifacts-test-'));
process.env.SNH_DATA_DIR = TMP;
process.on('exit', () => {
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* best effort */ }
});

const ROOT = path.join(__dirname, '..');
const database = require(path.join(ROOT, 'db/database'));
database.initDatabase();
const db = database.getSqliteDb();

const config = require(path.join(ROOT, 'db/config'));
const md = require(path.join(ROOT, 'db/markdown'));
const charts = require(path.join(ROOT, 'db/charts'));
const reportHtml = require(path.join(ROOT, 'db/report-html'));
const printer = require(path.join(ROOT, 'db/pdf-printer'));
const artifacts = require(path.join(ROOT, 'db/job-artifacts'));
const agentJobs = require(path.join(ROOT, 'db/agent-jobs'));

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}
function section(title) { console.log(`\n=== ${title} ===`); }

// Config stub on the module object — data/config.json is deliberately NOT
// redirected by SNH_DATA_DIR, and a test must never write to the live one.
const realGetConfig = config.getConfig;
let docCfg = {};
config.getConfig = () => {
  const c = realGetConfig();
  c.documents = Object.assign({
    enabled: true, outputDir: 'SNH_Documents', inlineMaxChars: 1200,
    chromiumPath: '', pageSize: 'Letter', keepHtml: false
  }, docCfg);
  return c;
};

const LONG = 'This is a sentence about the quarter that carries actual content. '.repeat(30);
const SCRIPT = Array.from({ length: 40 }, (_, i) => `def step_${i}(x):\n    return x + ${i}`).join('\n');

(async () => {
  // ---------------------------------------------------------------------------
  section('The renderer: markdown becomes markup, and stays escaped');
  // ---------------------------------------------------------------------------

  const table = md.renderMarkdown('| Client | Rev |\n|---|--:|\n| Acme | 4000 |');
  check('a table renders as a table, not as pipes',
    /<table class="md-table">/.test(table) && /<td style="text-align:right">4000<\/td>/.test(table), table.slice(0, 120));

  check('a fenced block keeps its language',
    /<pre class="md-code" data-lang="python">/.test(md.renderMarkdown('```python\nx = 1\n```')));

  check('markdown inside a code span is not interpreted',
    md.renderMarkdown('`**not bold**`').includes('<code>**not bold**</code>'));

  // THE ONE THAT MATTERS. Job text is written by a model that has just been
  // reading arbitrary web pages, so a literal tag in the result is an XSS vector
  // aimed straight at the panel. The first version of the renderer built <a> and
  // <img> tags in place and then split on a tag-shaped regex to decide what to
  // escape — which cannot tell our markup from the author's, and let this through.
  const hostile = md.renderMarkdown('<img src=x onerror=alert(1)> and <script>alert(2)</script>');
  check('a literal <img> in the result is escaped, not rendered',
    hostile.includes('&lt;img src=x onerror=alert(1)&gt;') && !/<img /.test(hostile), hostile);
  check('a literal <script> in the result is escaped',
    hostile.includes('&lt;script&gt;') && !/<script>/.test(hostile), hostile);
  check('a javascript: link loses its href and keeps its text',
    !/href="javascript/i.test(md.renderMarkdown('[click](javascript:alert(1))')),
    md.renderMarkdown('[click](javascript:alert(1))'));
  check('an ordinary link survives intact',
    md.renderMarkdown('[x](https://example.com/a_(b))').includes('href="https://example.com/a_(b)"'),
    md.renderMarkdown('[x](https://example.com/a_(b))'));

  check('toPlainText strips table syntax rather than counting it as words',
    !md.toPlainText('| a | b |\n|---|---|\n| 1 | 2 |').includes('|'),
    JSON.stringify(md.toPlainText('| a | b |\n|---|---|\n| 1 | 2 |')));

  // ---------------------------------------------------------------------------
  section('Classification: the form comes from what was made');
  // ---------------------------------------------------------------------------

  check('a short answer stays on the card',
    artifacts.classify('Nothing new since Monday.').kind === 'inline');

  check('a long report becomes a document',
    artifacts.classify(LONG).kind === 'document');

  const oneScript = `Here is the script.\n\n\`\`\`python\n${SCRIPT}\n\`\`\`\n\nIt handles the four operations.`;
  const codeDecision = artifacts.classify(oneScript);
  check('one dominant code block becomes a source file',
    codeDecision.kind === 'code' && codeDecision.ext === 'py', JSON.stringify(codeDecision.reason));
  check('…and the code is extracted without its fence',
    codeDecision.code.startsWith('def step_0') && !codeDecision.code.includes('```'));

  // The fence scanner used to be one regex with `$` under the m flag, which
  // matched at the first newline: every block came back empty, nothing failed, and
  // every script the entity wrote stayed on a card.
  check('the fence scanner reads the whole block, not the empty string',
    artifacts.fences('```python\na\nb\nc\n```')[0].code === 'a\nb\nc',
    JSON.stringify(artifacts.fences('```python\na\nb\nc\n```')));

  check('an unclosed fence still yields its code',
    artifacts.fences('```python\nimport os\ndef f():').length === 1);

  // Length is measured with code discounted, which is right for "is the WRITING
  // long" and wrong on its own for "does this belong on a card".
  const twoScripts = `Two scripts.\n\n\`\`\`python\n${SCRIPT}\n\`\`\`\n\n\`\`\`bash\n${'echo hi\n'.repeat(40)}\`\`\``;
  check('two substantial code blocks become a document, not a card',
    artifacts.classify(twoScripts).kind === 'document', JSON.stringify(artifacts.classify(twoScripts)));

  check('a three-line snippet inside a short note stays inline',
    artifacts.classify('Try this:\n\n```python\nprint(1)\n```').kind === 'inline');

  check('a ```text block is not mistaken for source',
    artifacts.classify(`Log output:\n\n\`\`\`text\n${'line\n'.repeat(80)}\`\`\``).kind !== 'code');

  check('the inline limit is the setting, not a constant',
    artifacts.classify(LONG, { inlineMaxChars: 100000 }).kind === 'inline');

  // ---------------------------------------------------------------------------
  section('Where files go');
  // ---------------------------------------------------------------------------

  check('a redirected process never writes to the real documents folder',
    artifacts.outputDir().startsWith(TMP), artifacts.outputDir());

  check('a filename is a slug of the title with the date on it',
    /^\d{4}-\d{2}-\d{2}-a-report-about-things$/.test(
      `${new Date().toISOString().slice(0, 10)}-${artifacts.slug('A Report: about "things"!')}`),
    artifacts.slug('A Report: about "things"!'));

  check('a title of pure punctuation still yields a usable name',
    artifacts.slug('!!! ???') === 'job-result', artifacts.slug('!!! ???'));

  const collideDir = fs.mkdtempSync(path.join(TMP, 'collide-'));
  fs.writeFileSync(path.join(collideDir, 'thing.pdf'), 'x');
  check('a second file on the same day does not overwrite the first',
    path.basename(artifacts.uniquePath(collideDir, 'thing', 'pdf')) === 'thing-2.pdf',
    artifacts.uniquePath(collideDir, 'thing', 'pdf'));

  // ---------------------------------------------------------------------------
  section('Charts');
  // ---------------------------------------------------------------------------

  const pieSvg = charts.renderChart('{"type":"pie","title":"T","data":[{"label":"Acme","value":42},{"label":"Beta","value":17}]}');
  check('a pie chart renders as SVG', /<svg/.test(pieSvg) && /<path d="M /.test(pieSvg));
  check('…with the values written out, so colour is never the only channel',
    pieSvg.includes('Acme') && pieSvg.includes('42'), 'labels missing');

  const many = charts.renderChart(JSON.stringify({
    type: 'pie',
    data: Array.from({ length: 12 }, (_, i) => ({ label: `c${i}`, value: 12 - i }))
  }));
  check('past six slices the tail folds into "Other" rather than cycling colours',
    many.includes('Other ('), 'no fold happened');

  check('a pie of negative values is refused rather than drawn wrong',
    charts.renderChart('{"type":"pie","data":[{"label":"a","value":-5},{"label":"b","value":10}]}') === null);
  check('a malformed spec is refused rather than throwing',
    charts.renderChart('{not json') === null);
  check('a refused chart still offers its numbers as a table',
    (charts.chartFallbackTable('{"type":"nope","data":[{"label":"a","value":1}]}') || '').includes('<table'));
  check('a bar chart renders', /<rect /.test(charts.renderChart('{"type":"bar","data":{"a":1,"b":2}}') || ''));
  check('a line chart renders', /<path d="M /.test(
    charts.renderChart('{"type":"line","series":[{"name":"s","data":[["a",1],["b",5],["c",3]]}]}') || ''));

  // ---------------------------------------------------------------------------
  section('The printable report');
  // ---------------------------------------------------------------------------

  const html = reportHtml.buildReportHtml({
    title: 'Q3 Review',
    body: '# Q3 Review\n\nOpening line.\n\n```chart\n{"type":"pie","data":[{"label":"a","value":1},{"label":"b","value":2}]}\n```\n',
    task: 'review the quarter',
    note: 'it ran out of rounds'
  });
  check('the report is a complete document', html.startsWith('<!DOCTYPE html>') && html.includes('</html>'));
  check('a chart fence becomes a figure in the report', html.includes('<svg') && html.includes('chart-figure'));
  // A stylesheet link or a CDN script would fail silently behind a file:// URL and
  // the PDF would print unstyled — a failure that still produces a file.
  check('nothing in the report is loaded from outside',
    !/<(link|script)\b/i.test(html) && !/(src|href)="https?:/i.test(html.replace(/<a [^>]*>/g, '')));
  check('the paper size comes from settings',
    reportHtml.buildReportHtml({ title: 't', body: 'x', pageSize: 'A4' }).includes('size: A4'));
  check('the title is not printed twice when the body opens with it',
    (html.match(/Q3 Review/g) || []).length <= 3, 'title repeated');
  check('why a run stopped short is carried into the document', html.includes('it ran out of rounds'));

  // ⚠ A REGRESSION GUARD FOR A BUG THAT IS INVISIBLE ON ONE PAGE.
  //
  // The report carried a running footer done the way the internet recommends —
  // position: fixed, nudged into the bottom margin. It does repeat on every
  // printed page, and it does not stay in the margin: on a real three-page print
  // it painted at the same offset on pages 2 and 3, over a table row and then
  // over a blockquote. Page one was perfect, so nothing short of printing a long
  // document and looking at it would have caught it. Since that is exactly what
  // nobody does on a routine change, the rule is asserted here instead.
  const cssOnly = html.replace(/\/\*[\s\S]*?\*\//g, '');
  check('nothing in the report is fixed-positioned — it prints over the content',
    !/position\s*:\s*fixed/.test(cssOnly));
  check('…and there is no running footer element to be positioned',
    !/<footer/.test(cssOnly));

  // A body heading that is not word-for-word the job title does not get removed
  // by the dedupe, so it must at least not be set at the title's own size.
  check('a body h1 sits below the document title in the hierarchy',
    /main h1 \{ font-size: 15pt/.test(cssOnly), 'main h1 not demoted');

  // ---------------------------------------------------------------------------
  section('The printer, on a machine that may have no browser');
  // ---------------------------------------------------------------------------

  printer.resetProbe();
  const probe = await printer.probe({ chromiumPath: '' });

  if (probe.ok) {
    console.log(`  (chromium found: ${probe.version})`);
    check('the probe reports a real binary', !!probe.path && fs.existsSync(probe.path));
  } else {
    console.log('  (no chromium on this machine — the fallback path is the live one)');
    check('the absence is reported as a sentence, not an error code',
      /chromium/i.test(probe.reason) && /text file/i.test(probe.reason), probe.reason);
    check('…and it says how to fix it', /snap install chromium/.test(probe.reason), probe.reason);
  }

  printer.resetProbe();
  const badPath = await printer.probe({ chromiumPath: '/nonexistent/chromium' });
  check('a configured path that does not work is reported, never silently ignored',
    !badPath.ok, JSON.stringify(badPath));
  printer.resetProbe();

  // ---------------------------------------------------------------------------
  section('Producing the file');
  // ---------------------------------------------------------------------------

  const codeOut = await artifacts.produce({
    id: 'a', title: 'Log parser', task: 'write a log parser', result_text: oneScript
  });
  check('a code result is written with the right extension',
    codeOut.kind === 'code' && codeOut.name.endsWith('.py'), JSON.stringify(codeOut));
  check('…the file holds the code and not the prose around it',
    fs.readFileSync(codeOut.path, 'utf8').startsWith('def step_0')
    && !fs.readFileSync(codeOut.path, 'utf8').includes('Here is the script'));
  check('…it ends with a newline, as every tool that reads source expects',
    fs.readFileSync(codeOut.path, 'utf8').endsWith('\n'));
  check('…and the card gets a summary rather than the file',
    codeOut.summary && codeOut.summary.length < 200 && !codeOut.summary.includes('def step_0'),
    codeOut.summary);

  const docOut = await artifacts.produce({
    id: 'b', title: 'Q3 Review', task: 'review the quarter', result_text: `# Q3 Review\n\n${LONG}`
  });
  check('a long result becomes a file', !!docOut.kind && !!docOut.path);

  // WHETHER A PDF IS POSSIBLE HERE IS NOT THE SAME QUESTION AS WHETHER CHROMIUM
  // EXISTS, and conflating the two is how this assertion first failed.
  //
  // A snap-packaged chromium can only read non-hidden paths under $HOME. This
  // suite runs with its data directory in the system temp directory — correctly,
  // that is what keeps it away from live state — and the report is built inside
  // that directory. So on a box where chromium is a snap, the PDF path is
  // genuinely unavailable TO THIS TEST while being perfectly available to the
  // live server, whose data directory is under $HOME.
  //
  // The nastiest part, and the reason this is spelled out: a confined chromium
  // given an unreadable page exits 0, prints nothing to stderr, and writes no
  // file. It does not fail. Only the empty-output guard notices.
  const printable = probe.ok && !/^\/snap\//.test(probe.path || '');
  const reachable = printable || (probe.ok && TMP.startsWith(os.homedir()));
  check('…of the kind this machine can actually make here',
    reachable ? docOut.kind === 'pdf' : docOut.kind === 'text',
    `${docOut.kind} (chromium ${probe.ok ? probe.path : 'absent'})`);
  check('…that is not empty', docOut.bytes > 0);

  if (probe.ok && !reachable) {
    check('a confined chromium that cannot reach the page says SO, by name',
      /snap/i.test(docOut.error || '') && /read files under/.test(docOut.error || ''), docOut.error);
  }

  if (docOut.kind === 'text') {
    check('…and the card is told why it is not a PDF',
      /chromium/i.test(docOut.error || ''), docOut.error);
    const txt = fs.readFileSync(docOut.path, 'utf8');
    check('the text fallback is formatted, not dumped',
      txt.includes('Q3 Review') && txt.includes('====') && txt.includes('The job:'), txt.slice(0, 200));
    const tableTxt = artifacts.formatAsText('| Client | Tickets |\n|---|---:|\n| Acme | 42 |\n| Beta | 7 |', { title: 'T' });
    check('…and its tables are aligned into columns, which is the whole point',
      /Client\s+Tickets/.test(tableTxt) && /Acme\s+42/.test(tableTxt), tableTxt);
    check('…with prose wrapped to a readable measure',
      artifacts.formatAsText(LONG, { title: 'T' }).split('\n').every(l => l.length <= 80));
  }

  const shortOut = await artifacts.produce({ id: 'c', title: 'Quick check', result_text: 'Nothing new since Monday.' });
  check('a short result makes no file at all', shortOut.kind === null && !shortOut.path);
  check('…and that is reported as a reason, not as an error',
    !!shortOut.reason && !shortOut.error, JSON.stringify(shortOut));

  docCfg = { enabled: false };
  const offOut = await artifacts.produce({ id: 'd', title: 'Q3 Review', result_text: LONG });
  check('the setting switches the whole thing off', offOut.kind === null, JSON.stringify(offOut));
  docCfg = {};

  // A DISK THAT WILL NOT TAKE THE FILE MUST NOT FAIL THE JOB. This is the rule the
  // whole module is bent around, so it is provoked rather than reasoned about.
  //
  // Provoked by putting a FILE where the folder should go, rather than by naming
  // an unwritable path: under a redirected data directory an absolute outputDir
  // is reduced to its basename (that redirect is what keeps a test out of her
  // real documents folder), so /proc/nope/not/writable would have quietly become
  // a perfectly writable directory inside the sandbox and the case would never
  // have been reached. ENOTDIR is the same class of failure as ENOSPC or EACCES
  // and takes the same branch.
  fs.writeFileSync(path.join(TMP, 'blocked'), 'not a directory');
  docCfg = { outputDir: 'blocked' };
  const brokenOut = await artifacts.produce({ id: 'e', title: 'Q3 Review', result_text: LONG });
  check('an unwritable folder yields an explanation, never a throw',
    brokenOut.kind === null && !!brokenOut.error, JSON.stringify(brokenOut));
  check('…and the explanation points her back at the text she still has',
    /above/.test(brokenOut.error || ''), brokenOut.error);
  docCfg = {};

  // ---------------------------------------------------------------------------
  section('The row, the feed, and what he is told');
  // ---------------------------------------------------------------------------

  const id = require('crypto').randomUUID();
  db.prepare(`
    INSERT INTO agent_jobs (id, title, task, status, result_text, created_at, finished_at, duration_ms, tool_calls)
    VALUES (?, ?, ?, 'ok', ?, ?, ?, 1200, 2)
  `).run(id, 'Supplier report', 'check the suppliers', `# Supplier report\n\n${LONG}`,
    new Date().toISOString(), new Date().toISOString());

  const attached = await agentJobs.attachArtifact(id);
  check('attachArtifact stores what was made', !!attached.artifact_kind && !!attached.artifact_path,
    JSON.stringify({ kind: attached.artifact_kind, path: attached.artifact_path }));
  check('…and a summary for the card', !!attached.summary_text);
  check('…without disturbing the status or the result text',
    attached.status === 'ok' && attached.result_text.includes(LONG.slice(0, 40)));

  const feedRow = agentJobs.feed({ limit: 50 }).find(j => j.id === id);
  check('the feed carries the file', feedRow && feedRow.artifact_kind === attached.artifact_kind);
  check('the feed carries the folder, so she knows where it also lives', !!feedRow.artifact_location);
  // The panel has no use for a server path it cannot open, and the download route
  // looks the path up by job id rather than being handed one.
  check('the feed does NOT carry the full path', feedRow.artifact_path === undefined);
  check('a scheduled run reports no file rather than undefined',
    agentJobs.feed({ limit: 50 }).every(j => 'artifact_kind' in j));

  const block = agentJobs.renderAnnouncementBlock({ limit: 3 });
  check('he is told the result became a file, by name',
    block && block.text.includes(attached.artifact_name), block ? block.text.slice(0, 300) : 'no block');
  check('…and told that a file existing is not her having read it',
    block && /has a link to it|assume she has NOT/.test(block.text));

  // ---------------------------------------------------------------------------
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);

})();
