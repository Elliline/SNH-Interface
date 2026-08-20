/**
 * What a job produced, as a file.
 *
 * WHY. A result was one column of text on a card, and the card was where it
 * lived and died. That is the right shape for three sentences and the wrong
 * shape for everything else: a research report arrived as raw markdown in a
 * narrow panel — pipes and asterisks, no table — and a Python module arrived as
 * a code block to be selected out of a scrolling box. Neither is a thing you can
 * keep, open later, or send to anyone.
 *
 * THE FORM COMES FROM WHAT WAS MADE, not from what was asked. classify() reads
 * the text the run actually produced:
 *
 *   code      one fenced block that IS the result → a source file with the right
 *             extension, and the prose around it becomes the card's summary.
 *   document  long prose → a PDF, printed from HTML by db/pdf-printer.js, or a
 *             formatted text file on a machine with no chromium.
 *   inline    short → stays on the card, rendered as markdown, exactly as now.
 *
 * ⚠ THE FILE IS NEVER THE ONLY COPY. result_text stays in the database whatever
 * happens here, and the file is made FROM it. Deleting the file, a full disk, a
 * chromium that will not start — none of them can cost the work, only the
 * formatting. Every failure path in this module ends with a job that still has
 * its result and a card that says in one sentence what happened to the file.
 * That is the same rule as the empty-card guard in db/agent-jobs.js, applied to
 * a new way of losing something.
 *
 * ⚠ AND IT IS NEVER THE ONLY WAY TO REACH IT. A path on the server is no use to
 * someone reading the panel on a laptop, so the card carries a download link as
 * well as the location — the route in routes/jobs.js serves it BY JOB ID, and
 * looks the path up here. No path from a browser is ever opened.
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const md = require('./markdown');
const { getDataDir } = require('./database');

function getConfig() { return require('./config').getConfig(); }

/** The documents settings, with the defaults spelled out for a config that predates them. */
function cfg() {
  const c = getConfig().documents || {};
  return {
    enabled: c.enabled !== false,
    outputDir: c.outputDir || 'SNH_Documents',
    inlineMaxChars: Number.isFinite(c.inlineMaxChars) ? c.inlineMaxChars : 1200,
    chromiumPath: c.chromiumPath || '',
    pageSize: c.pageSize === 'A4' ? 'A4' : 'Letter',
    keepHtml: !!c.keepHtml
  };
}

/**
 * Fence languages that mean "this is a source file", and what to call it.
 *
 * A language NOT in this list does not become a code file. That is deliberate:
 * ```text and ```output and ```log are all common, and a run's captured console
 * output saved as `result.text` would be a file that claims to be a deliverable
 * and is not.
 */
const CODE_EXT = {
  python: 'py', py: 'py', python3: 'py',
  javascript: 'js', js: 'js', node: 'js', jsx: 'jsx',
  typescript: 'ts', ts: 'ts', tsx: 'tsx',
  bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh',
  powershell: 'ps1', ps1: 'ps1',
  ruby: 'rb', rb: 'rb',
  go: 'go', golang: 'go',
  rust: 'rs', rs: 'rs',
  java: 'java', kotlin: 'kt', scala: 'scala', swift: 'swift',
  c: 'c', h: 'h', cpp: 'cpp', 'c++': 'cpp', cxx: 'cpp',
  csharp: 'cs', cs: 'cs',
  php: 'php', perl: 'pl', lua: 'lua', r: 'R', julia: 'jl',
  haskell: 'hs', elixir: 'ex', clojure: 'clj',
  sql: 'sql',
  html: 'html', css: 'css', scss: 'scss',
  json: 'json', yaml: 'yml', yml: 'yml', toml: 'toml', xml: 'xml', ini: 'ini',
  dockerfile: 'Dockerfile', makefile: 'Makefile',
  vim: 'vim', diff: 'diff', patch: 'patch'
};

/**
 * Every fenced block in the text, with its language and body.
 *
 * Scanned line by line rather than by one regex over the whole string. The
 * regex version — `([\s\S]*?)(?:^ {0,3}\1[ \t]*$|$)` under the m flag — looks
 * right and is not: `$` matches at every line ending, so the lazy body matched
 * the empty string at the first newline and every block came back with no code
 * in it. Nothing failed; the classifier simply never saw a code block and every
 * script it was given stayed on the card.
 *
 * An UNCLOSED fence counts, and takes the rest of the text. A run that hit its
 * output budget mid-file is exactly when the file matters most.
 */
function fences(text) {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
  const open = /^ {0,3}(`{3,}|~{3,})[ \t]*([\w+#.-]*)[ \t]*$/;
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(open);
    if (!m) continue;
    const marker = m[1][0], minLen = m[1].length;
    const lang = (m[2] || '').toLowerCase();
    const body = [];
    i++;
    while (i < lines.length) {
      const close = lines[i].match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/);
      if (close && close[1][0] === marker && close[1].length >= minLen) break;
      body.push(lines[i]);
      i++;
    }
    out.push({ lang, code: body.join('\n') });
  }
  return out;
}

/**
 * Decide the form.
 *
 * @returns {{kind: 'code'|'document'|'inline', lang?: string, ext?: string,
 *            code?: string, prose?: string, reason: string}}
 *          `reason` is for the ops log — the decision has to be inspectable when
 *          a result lands in a form that looks wrong.
 */
function classify(text, { inlineMaxChars = 1200 } = {}) {
  const src = String(text || '');
  const blocks = fences(src);
  const codeBlocks = blocks.filter(b => CODE_EXT[b.lang] && b.code.trim().length >= 200);

  // ONE dominant code block is a file. Two or more substantial ones are a
  // document ABOUT code — a report with three scripts in it is not three files
  // and is certainly not one, and flattening it would throw two of them away.
  if (codeBlocks.length === 1) {
    const block = codeBlocks[0];
    // Does the code dominate, or is it an illustration inside an argument? The
    // prose is measured with code already discounted (toPlainText counts a fence
    // as one token), so this compares the deliverable against the writing around
    // it rather than against itself.
    const prose = md.toPlainText(src);
    if (block.code.length >= prose.length) {
      return {
        kind: 'code',
        lang: block.lang,
        ext: CODE_EXT[block.lang],
        code: block.code,
        prose,
        reason: `one ${block.lang} block of ${block.code.length} chars against ${prose.length} chars of prose`
      };
    }
  }

  const plain = md.toPlainText(src);
  if (plain.length > inlineMaxChars) {
    return {
      kind: 'document',
      reason: `${plain.length} chars of prose, over the ${inlineMaxChars} inline limit`
        + (codeBlocks.length > 1 ? ` (and ${codeBlocks.length} code blocks, so not one file)` : '')
    };
  }

  // CODE COUNTS EVEN WHEN THE PROSE DOES NOT. The length above is measured with
  // fences discounted to one token, which is right for deciding whether the
  // WRITING is long — and wrong on its own for deciding whether the result
  // belongs on a card. A run that returned two scripts under sixty words of
  // explanation measured as 60 characters and stayed inline: 140 lines of code
  // in a panel the width of a phone. Substantial code is a file whatever the
  // prose around it does; which file, the branch above already decided.
  if (codeBlocks.length) {
    return {
      kind: 'document',
      reason: `${codeBlocks.length} code block(s) of ${codeBlocks.reduce((n, b) => n + b.code.length, 0)} chars`
        + `, which do not belong on a card even though the prose is short (${plain.length} chars)`
    };
  }

  return { kind: 'inline', reason: `${plain.length} chars, inside the ${inlineMaxChars} inline limit` };
}

/**
 * Where files go.
 *
 * A bare name resolves under the home directory; an absolute path is taken as
 * given. A process whose data directory has been redirected (SNH_DATA_DIR — the
 * replay, the tests, any throwaway instance) gets a documents folder INSIDE that
 * directory instead, so a test run can never write into her real documents
 * folder. That redirect exists precisely so a disposable process cannot touch
 * live state, and a file on disk is live state.
 */
function outputDir() {
  const configured = cfg().outputDir;
  if (process.env.SNH_DATA_DIR) return path.join(getDataDir(), path.basename(configured) || 'SNH_Documents');
  if (path.isAbsolute(configured)) return configured;
  return path.join(os.homedir(), configured);
}

/** A filename component: lowercase, hyphens, nothing a shell or a filesystem minds. */
function slug(s, max = 48) {
  const base = String(s || '')
    .normalize('NFKD')
    .replace(/[^\w\s-]/g, ' ')
    .trim()
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .toLowerCase()
    .slice(0, max)
    .replace(/^-|-$/g, '');
  return base || 'job-result';
}

/** YYYY-MM-DD in local time — the day she would call it. */
function datePart(d) {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/**
 * A path that does not exist yet.
 *
 * Two reports on the same subject on the same day is a normal Tuesday, and the
 * second one silently replacing the first would be the worst possible way to
 * find that out.
 */
function uniquePath(dir, base, ext) {
  // Dockerfile and Makefile are whole NAMES rather than extensions, and they are
  // the only two — an earlier version tested for a leading capital instead,
  // which caught them and also caught `R`, so an R script was written as
  // `2026-08-19-analysisR` with no extension at all.
  const wholeName = ext === 'Dockerfile' || ext === 'Makefile';
  const dotted = ext ? (ext.startsWith('.') ? ext : `.${ext}`) : '';
  const name = wholeName ? ext : `${base}${dotted}`;
  let candidate = path.join(dir, name);
  let n = 2;
  while (fs.existsSync(candidate)) {
    const stem = wholeName ? `${ext}-${n}` : `${base}-${n}${dotted}`;
    candidate = path.join(dir, stem);
    n++;
    if (n > 200) break;   // something is very wrong; stop rather than spin
  }
  return candidate;
}

/**
 * A few lines for the card, so a document is ANNOUNCED rather than pasted.
 *
 * The whole point of a file is that the card stops being the document. What she
 * needs on the card is enough to know whether to open it: the opening of what
 * was written, and what the file is.
 */
function summarise(text, { kind, lang, lines } = {}) {
  const plain = md.toPlainText(text).replace(/\s*\[code\]\s*/g, ' ');
  const paragraphs = plain.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);

  let summary = '';
  for (const p of paragraphs) {
    if (summary.length >= 240) break;
    // A heading on its own line is a title, not a summary — it tells her nothing
    // the card's own title does not.
    if (p.length < 40 && !/[.!?]$/.test(p) && paragraphs.length > 1) continue;
    summary += (summary ? ' ' : '') + p;
  }
  if (!summary) summary = paragraphs[0] || plain;
  summary = summary.replace(/\s+/g, ' ').trim();
  if (summary.length > 400) summary = summary.slice(0, 397).replace(/\s\S*$/, '') + '…';

  if (kind === 'code' && !summary) {
    summary = `A ${lang || 'source'} file${Number.isFinite(lines) ? `, ${lines} lines` : ''}.`;
  }
  return summary;
}

/**
 * Markdown as readable plain text — the fallback when there is no chromium.
 *
 * NOT toPlainText(), which throws structure away to measure length. This keeps
 * the structure and renders it in the one medium every machine has: headings
 * underlined, lists bulleted, code indented, and TABLES ALIGNED INTO COLUMNS,
 * which is the whole point — an unaligned table is the exact unreadable thing
 * this work started from, and a fallback that reproduced it would be no fallback
 * at all.
 */
function formatAsText(source, { title, task, date = new Date(), note = '' } = {}) {
  const lines = String(source || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];

  if (title) {
    out.push(String(title));
    out.push('='.repeat(Math.min(78, String(title).length)));
  }
  out.push(date.toLocaleString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: '2-digit'
  }));
  if (task) { out.push(''); out.push(`The job: ${task}`); }
  out.push('');

  let i = 0;
  // The document already carries its title above. A body that opens with the
  // same title as a heading would print it twice — only skipped when it is the
  // very first thing and it matches, never further down where a repeated
  // heading is a real section.
  while (i < lines.length && !lines[i].trim()) i++;
  const opener = lines[i] && lines[i].match(/^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$/);
  if (opener && title && opener[1].trim().toLowerCase() === String(title).trim().toLowerCase()) i++;

  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^ {0,3}(`{3,}|~{3,})[ \t]*([\w+#.-]*)[ \t]*$/);
    if (fence) {
      const marker = fence[1][0];
      i++;
      if (fence[2]) out.push(`    [${fence[2]}]`);
      while (i < lines.length && !new RegExp(`^ {0,3}${marker === '`' ? '`' : '~'}{3,}[ \\t]*$`).test(lines[i])) {
        out.push('    ' + lines[i]);
        i++;
      }
      i++;
      out.push('');
      continue;
    }

    const h = line.match(/^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (h) {
      const text = h[2];
      out.push('');
      out.push(text.toUpperCase());
      out.push((h[1].length <= 2 ? '=' : '-').repeat(Math.min(78, text.length)));
      i++;
      continue;
    }

    // A table: gather the whole thing, then lay it out in columns.
    if (line.includes('|') && i + 1 < lines.length && /^[\s|:-]+$/.test(lines[i + 1]) && lines[i + 1].includes('-')) {
      const rows = [splitRow(line)];
      i += 2;
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      out.push(...alignTable(rows));
      out.push('');
      continue;
    }

    if (/^ {0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) { out.push('-'.repeat(60)); i++; continue; }

    const plainLine = line
      .replace(/(\*\*\*|\*\*|__|~~)/g, '')
      .replace(/`([^`]*)`/g, '$1')
      .replace(/!\[([^\]]*)\]\(([^)]*)\)/g, '[image: $1]')
      .replace(/\[([^\]]*)\]\(([^)]*)\)/g, '$1 <$2>');

    // Bullets and quotes keep their marker on the first line and hang under it.
    const bullet = plainLine.match(/^(\s*)[-*+]\s+(.*)$/);
    const quote = plainLine.match(/^ {0,3}>\s?(.*)$/);
    if (bullet) out.push(...wrap(bullet[2], `${bullet[1]}• `, `${bullet[1]}  `));
    else if (quote) out.push(...wrap(quote[1], '  | ', '  | '));
    else out.push(...wrap(plainLine, '', ''));
    i++;
  }

  if (note) { out.push(''); out.push('-'.repeat(60)); out.push(note); }
  return out.join('\n').replace(/\n{4,}/g, '\n\n\n') + '\n';
}

/**
 * Wrap prose to a readable measure.
 *
 * 78 columns, because that is what a terminal, an email client and a printed
 * page all show without folding. Only prose is wrapped — code keeps its own
 * line breaks (they are semantic) and a table has already been laid out in
 * columns that a wrap would destroy.
 */
function wrap(text, firstPrefix = '', restPrefix = '', width = 78) {
  const body = String(text == null ? '' : text);
  if (!body.trim()) return [body ? firstPrefix + body : ''];
  const words = body.split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = firstPrefix;
  let prefix = firstPrefix;
  for (const word of words) {
    if (cur.length > prefix.length && cur.length + 1 + word.length > width) {
      lines.push(cur);
      prefix = restPrefix;
      cur = restPrefix + word;
    } else {
      cur = cur.length > prefix.length ? `${cur} ${word}` : cur + word;
    }
  }
  if (cur.trim()) lines.push(cur);
  return lines.length ? lines : [''];
}

/** A markdown table row, split on pipes with the outer pair dropped. */
function splitRow(line) {
  const cells = line.split('|').map(c => c.trim());
  if (cells.length && cells[0] === '') cells.shift();
  if (cells.length && cells[cells.length - 1] === '') cells.pop();
  return cells;
}

/** Rows to aligned monospace columns, with a rule under the header. */
function alignTable(rows) {
  const width = Math.max(...rows.map(r => r.length));
  const padded = rows.map(r => Array.from({ length: width }, (_, i) => (r[i] || '')));
  const widths = Array.from({ length: width }, (_, i) =>
    Math.min(34, Math.max(...padded.map(r => r[i].length))));
  const render = (r) => r.map((c, i) => c.length > widths[i]
    ? c.slice(0, widths[i] - 1) + '…'
    : c.padEnd(widths[i])).join('  ').trimEnd();
  const out = [render(padded[0]), widths.map(w => '-'.repeat(w)).join('  ')];
  for (const r of padded.slice(1)) out.push(render(r));
  return out;
}

/**
 * Produce the file for a finished job, if it should have one.
 *
 * NEVER THROWS, and never rejects. Every failure comes back as a shape the
 * caller can store: `kind: null` with an `error` sentence for the card. A job
 * that produced a real result must not be turned into a failed one because a
 * disk was full or a browser would not start.
 *
 * @param {Object} job    the agent_jobs row (title, task, result_text, id)
 * @param {Object} [opts] {date, note} — note is why a partial run stopped
 * @returns {Promise<{kind: string|null, path?: string, name?: string,
 *                    bytes?: number, summary?: string, error?: string,
 *                    reason?: string}>}
 */
async function produce(job, opts = {}) {
  const c = cfg();
  const text = String(job && job.result_text || '');
  if (!c.enabled) return { kind: null, reason: 'file output is switched off in settings' };
  if (!text.trim()) return { kind: null, reason: 'the run produced no text' };

  const decision = classify(text, { inlineMaxChars: c.inlineMaxChars });
  if (decision.kind === 'inline') return { kind: null, reason: decision.reason };

  const date = opts.date instanceof Date ? opts.date : new Date();
  const note = opts.note || '';
  const base = `${datePart(date)}-${slug(job.title)}`;

  let dir;
  try {
    dir = outputDir();
    fs.mkdirSync(dir, { recursive: true });
  } catch (err) {
    return {
      kind: null,
      error: `the result could not be saved to ${dir || 'the documents folder'} (${String(err && err.message || err).slice(0, 160)}). The full text is above.`,
      reason: decision.reason
    };
  }

  // --- a source file ------------------------------------------------------
  if (decision.kind === 'code') {
    const target = uniquePath(dir, base, decision.ext);
    try {
      // A trailing newline, because every tool that reads source expects one.
      fs.writeFileSync(target, decision.code.replace(/\n*$/, '\n'), 'utf8');
      const bytes = fs.statSync(target).size;
      const lineCount = decision.code.split('\n').length;
      const summary = summarise(
        decision.prose && decision.prose.replace(/\[code\]/g, '').trim() ? text : '',
        { kind: 'code', lang: decision.lang, lines: lineCount }
      ) || `A ${decision.lang} file, ${lineCount} lines.`;
      return {
        kind: 'code',
        path: target,
        name: path.basename(target),
        bytes,
        summary,
        reason: decision.reason
      };
    } catch (err) {
      return {
        kind: null,
        error: `the ${decision.lang} file could not be written (${String(err && err.message || err).slice(0, 160)}). The code is above.`,
        reason: decision.reason
      };
    }
  }

  // --- a document ---------------------------------------------------------
  const { buildReportHtml } = require('./report-html');
  const printer = require('./pdf-printer');
  const summary = summarise(text, { kind: 'document' });

  const html = buildReportHtml({
    title: job.title, body: text, task: job.task, date, note, pageSize: c.pageSize
  });
  const pdfPath = uniquePath(dir, base, 'pdf');
  const printed = await printer.printToPdf(html, pdfPath, {
    chromiumPath: c.chromiumPath,
    keepHtml: c.keepHtml
  });

  if (printed.ok) {
    return {
      kind: 'pdf',
      path: pdfPath,
      name: path.basename(pdfPath),
      bytes: printed.bytes,
      summary,
      // A weakened sandbox is not an error, and it is not nothing either.
      error: printed.warning || null,
      reason: decision.reason
    };
  }

  // NO CHROMIUM, OR IT WOULD NOT PRINT → a formatted text file, and the reason
  // said plainly. This is the branch that runs on a box with no browser, so it
  // is the ordinary path rather than an exceptional one, and it must produce
  // something worth opening: structure kept, tables aligned into columns.
  const txtPath = uniquePath(dir, base, 'txt');
  try {
    fs.writeFileSync(txtPath, formatAsText(text, { title: job.title, task: job.task, date, note }), 'utf8');
    return {
      kind: 'text',
      path: txtPath,
      name: path.basename(txtPath),
      bytes: fs.statSync(txtPath).size,
      summary,
      error: printed.reason,
      reason: decision.reason
    };
  } catch (err) {
    return {
      kind: null,
      error: `${printed.reason} Writing the text file failed too (${String(err && err.message || err).slice(0, 120)}). The full result is above.`,
      reason: decision.reason
    };
  }
}

module.exports = {
  produce, classify, summarise, formatAsText, outputDir, slug, uniquePath, fences, CODE_EXT
};
