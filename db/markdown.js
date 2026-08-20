/**
 * Markdown → HTML. One renderer, used on both sides of the wire.
 *
 * WHY THIS EXISTS. A job result arrived as a string and the panel card printed
 * it through escapeHtml(), so a research report rendered as one long thin column
 * of pipes and asterisks — the raw source of a table, in a narrow card, with the
 * table nowhere. The chat pane was no better off: formatMessageContent() does
 * bold, italic, code and <br> with four regexes and knows nothing about a
 * heading, a list or a table.
 *
 * WHY IT IS SHARED, AND WHY IT LIVES IN db/. The same text has to be rendered in
 * two places that could not be further apart — the panel card in the browser,
 * and the printable HTML the PDF is made from on the server. Two renderers would
 * be two answers to "what does this document look like", and the one Ellie reads
 * on screen would drift from the one she gets in the file. So this file is the
 * only answer, it is written to run under both Node and a browser, and server.js
 * serves it verbatim at /markdown.js rather than a copy being kept in public/.
 *
 * ⚠ ESCAPING IS STRUCTURAL, NOT A PASS. The old chat formatter escaped the whole
 * string and then ran markdown regexes over the escaped text. That is safe, and
 * it is also why it could never grow: once everything is escaped there is no way
 * to tell a `<` the author typed from one the renderer wants to emit. Here, text
 * is escaped at the moment it becomes output and never before, so this file must
 * hold to one rule without exception:
 *
 *     EVERY path that puts author-supplied text into the HTML runs it through
 *     esc() (or, for a URL, safeUrl()) first. There are no exceptions and there
 *     is no "this one came from us".
 *
 * The job text this renders is written by an LLM, which is to say by whatever
 * the web pages it read told it to write. It is untrusted input.
 *
 * PURE and synchronous. No DOM, no database, no network — it has to run inside a
 * background job with no window object.
 */

/** The five that matter in an HTML text node or a double-quoted attribute. */
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * A URL we are willing to put in an href or a src.
 *
 * An allowlist of schemes rather than a blocklist of `javascript:`, because the
 * blocklist has to anticipate every spelling — tab-separated, mixed case,
 * entity-encoded — and the allowlist has to anticipate nothing. Anything that is
 * not plainly one of these, or plainly relative, comes back '' and the link
 * renders as its label with no href at all.
 */
function safeUrl(raw) {
  const url = String(raw == null ? '' : raw).trim();
  if (!url) return '';
  // Strip whitespace and control characters before testing: they are the whole
  // trick behind `java&#09;script:`, and they mean nothing in a real URL.
  const flat = url.replace(/[\u0000-\u0020]/g, '');
  if (/^(https?:|mailto:|tel:)/i.test(flat)) return flat;
  if (/^data:image\/(png|jpe?g|gif|svg\+xml|webp);base64,[A-Za-z0-9+/=]+$/i.test(flat)) return flat;
  // Relative: no scheme at all, and not protocol-relative (//evil.example).
  if (!/^[a-z][a-z0-9+.-]*:/i.test(flat) && !flat.startsWith('//')) return flat;
  return '';
}

// ---------------------------------------------------------------------------
// Inline
// ---------------------------------------------------------------------------

/**
 * Sentinel around a stashed piece of finished markup.
 *
 * NUL, because the block reader strips every control character from the input
 * before this runs — so it is the one byte that cannot arrive from outside and
 * cannot collide with anything the author wrote.
 */
const NUL = '\u0000';

/**
 * Inline markdown inside one block of text.
 *
 * ⚠ THE ORDER HERE IS THE SECURITY PROPERTY, so it is worth saying why.
 *
 * Every construct that PRODUCES MARKUP — a code span, an image, a link — is cut
 * out of the string as it is recognised, rendered immediately, and replaced by a
 * NUL-delimited placeholder. Only once nothing but plain author text is left
 * does the emphasis pass run, and it escapes the whole of what remains. The
 * finished markup is spliced back in at the very end.
 *
 * The first version did the obvious thing instead: it built the <a> and <img>
 * tags in place, then had the emphasis pass split the string on a tag-shaped
 * regex so it could escape only the gaps between them. That regex cannot tell
 * OUR <img …> from one the author typed — and this text is written by a model
 * that has just been reading arbitrary web pages. A literal
 * `<img src=x onerror=alert(1)>` in a job result matched the split, was taken
 * for generated markup, and went to the browser unescaped. Placeholders close
 * that by construction: at the moment esc() runs there is no markup in the
 * string to mistake anything for, and no regex is being asked to tell
 * provenance apart.
 *
 * Code spans have to come out first for a second, older reason: everything
 * between backticks is literal by definition, so a `**` inside one must never
 * reach the emphasis pass — otherwise the one construct whose entire job is to
 * show markdown verbatim is the one that cannot.
 */
function inline(src) {
  const spans = [];
  const stash = (html) => {
    spans.push(html);
    return `${NUL}${spans.length - 1}${NUL}`;
  };

  let text = String(src == null ? '' : src)
    .replace(/(`+)([\s\S]*?)\1/g, (m, ticks, code) => stash(`<code>${esc(code.trim())}</code>`));

  // Images before links: ![alt](src) shares its tail with [text](href), so a
  // link pass running first would eat the `[alt](src)` and leave a stray `!`.
  text = text.replace(/!\[([^\]]*)\]\(([^()\s]*(?:\([^()\s]*\)[^()\s]*)*)(?:\s+"([^"]*)")?\)/g, (m, alt, imgSrc, title) => {
    const u = safeUrl(imgSrc);
    if (!u) return esc(alt);
    return stash(`<img src="${esc(u)}" alt="${esc(alt)}"${title ? ` title="${esc(title)}"` : ''}>`);
  });

  text = text.replace(/\[([^\]]*)\]\(([^()\s]*(?:\([^()\s]*\)[^()\s]*)*)(?:\s+"([^"]*)")?\)/g, (m, label, href, title) => {
    const u = safeUrl(href);
    // A refused URL still shows its label. Dropping the whole thing would make
    // the renderer silently delete content, which is worse than a dead link.
    if (!u) return stash(inlineEmphasis(label));
    return stash(`<a href="${esc(u)}"${title ? ` title="${esc(title)}"` : ''} rel="noopener noreferrer">${inlineEmphasis(label)}</a>`);
  });

  // Nothing in `text` is markup now. Escape all of it, then put the markup back.
  text = inlineEmphasis(text);

  return text.replace(new RegExp(`${NUL}(\\d+)${NUL}`, 'g'), (m, i) => spans[Number(i)] || '');
}

/**
 * Escape, then emphasis. Never the other way round, and never on a string that
 * already holds markup — see the placeholder rule in inline() above.
 */
function inlineEmphasis(text) {
  return esc(text)
    .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    // Single-underscore italics only at word boundaries, so snake_case_names and
    // file_names.js survive intact. Asterisk italics need no such guard.
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,;:!?])/g, '$1<em>$2</em>')
    .replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s).,;:!?])/g, '$1<em>$2</em>')
    .replace(/~~([^~]+)~~/g, '<del>$1</del>');
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const BULLET_RE = /^(\s*)([-*+])\s+(.*)$/;
const ORDERED_RE = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;
const HEADING_RE = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const FENCE_RE = /^ {0,3}(`{3,}|~{3,})\s*([\w+-]*)\s*$/;
const QUOTE_RE = /^ {0,3}>\s?(.*)$/;
const TABLE_SEP_RE = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

/** Is this line the head of a table, given the line after it? */
function isTableHead(lines, i) {
  return lines[i].includes('|')
    && i + 1 < lines.length
    && lines[i + 1].includes('-')
    && TABLE_SEP_RE.test(lines[i + 1]);
}

/** A table row split on unescaped pipes, with the outer pair dropped. */
function splitRow(line) {
  const cells = [];
  let cur = '';
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '\\' && line[i + 1] === '|') { cur += '|'; i++; continue; }
    if (ch === '|') { cells.push(cur); cur = ''; continue; }
    cur += ch;
  }
  cells.push(cur);
  if (cells.length && cells[0].trim() === '') cells.shift();
  if (cells.length && cells[cells.length - 1].trim() === '') cells.pop();
  return cells.map(c => c.trim());
}

/** Column alignments from the |:---|---:|:--:| separator row. */
function alignments(sep) {
  return splitRow(sep).map(c => {
    const left = c.startsWith(':'), right = c.endsWith(':');
    if (left && right) return 'center';
    if (right) return 'right';
    if (left) return 'left';
    return '';
  });
}

/**
 * Render markdown to an HTML fragment.
 *
 * @param {string} src
 * @param {Object} [opts]
 * @param {(lang: string, code: string) => (string|null)} [opts.fence]
 *        Handles a fenced block whose info string is `lang`. Return HTML to
 *        replace the block, or null to let it render as ordinary code. This is
 *        how a ```chart block becomes an SVG in the printable report and stays a
 *        code block everywhere else — the renderer itself knows nothing about
 *        charts, which is the only reason it can be shared.
 * @param {boolean} [opts.softBreaks=true]
 *        A single newline inside a paragraph becomes <br>. True by default
 *        because this renders text written for a person to read rather than
 *        markdown authored against a spec, and there hard-wrapped prose that
 *        reflows into one run-on paragraph reads as a bug.
 * @returns {string} HTML. Safe to insert with innerHTML — see the escaping rule
 *        at the top of this file.
 */
function renderMarkdown(src, opts = {}) {
  const fence = typeof opts.fence === 'function' ? opts.fence : null;
  const softBreaks = opts.softBreaks !== false;

  // Normalise line endings, then strip control characters (tab and newline
  // kept). NUL in particular is the code-span placeholder and must never arrive
  // from outside.
  const lines = String(src == null ? '' : src)
    .replace(/\r\n?/g, '\n')
    .replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, '')
    .split('\n');

  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i++; continue; }

    // --- fenced code -------------------------------------------------------
    const fenceOpen = line.match(FENCE_RE);
    if (fenceOpen) {
      const marker = fenceOpen[1][0];
      const minLen = fenceOpen[1].length;
      const lang = (fenceOpen[2] || '').toLowerCase();
      const body = [];
      i++;
      while (i < lines.length) {
        const close = lines[i].match(FENCE_RE);
        if (close && close[1][0] === marker && close[1].length >= minLen) { i++; break; }
        body.push(lines[i]);
        i++;
      }
      const code = body.join('\n');
      // An UNCLOSED fence still renders: the model runs out of output budget
      // mid-file often enough that dropping the block would throw away the exact
      // thing the job was asked to produce.
      const custom = fence ? fence(lang, code) : null;
      if (custom) { out.push(custom); continue; }
      out.push(`<pre class="md-code"${lang ? ` data-lang="${esc(lang)}"` : ''}><code>${esc(code)}</code></pre>`);
      continue;
    }

    // --- heading -----------------------------------------------------------
    const h = line.match(HEADING_RE);
    if (h) {
      const level = h[1].length;
      out.push(`<h${level} class="md-h md-h${level}">${inline(h[2])}</h${level}>`);
      i++;
      continue;
    }

    // --- horizontal rule ---------------------------------------------------
    if (HR_RE.test(line)) { out.push('<hr class="md-hr">'); i++; continue; }

    // --- table -------------------------------------------------------------
    // Requires the separator row. A single line with pipes in it is prose about
    // a pipe far more often than it is a one-row table.
    if (isTableHead(lines, i)) {
      const head = splitRow(line);
      const align = alignments(lines[i + 1]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      const th = head.map((c, n) => `<th${align[n] ? ` style="text-align:${align[n]}"` : ''}>${inline(c)}</th>`).join('');
      const tb = rows.map(r => {
        // Padded to the header width so a ragged row cannot shear the table —
        // a model writing a wide table gets the cell count wrong often enough.
        const cells = head.map((_, n) => (r[n] == null ? '' : r[n]));
        return `<tr>${cells.map((c, n) => `<td${align[n] ? ` style="text-align:${align[n]}"` : ''}>${inline(c)}</td>`).join('')}</tr>`;
      }).join('');
      out.push(`<div class="md-table-wrap"><table class="md-table"><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`);
      continue;
    }

    // --- blockquote --------------------------------------------------------
    if (QUOTE_RE.test(line)) {
      const body = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        body.push(lines[i].match(QUOTE_RE)[1]);
        i++;
      }
      out.push(`<blockquote class="md-quote">${renderMarkdown(body.join('\n'), opts)}</blockquote>`);
      continue;
    }

    // --- list --------------------------------------------------------------
    if (BULLET_RE.test(line) || ORDERED_RE.test(line)) {
      const consumed = renderList(lines, i, opts);
      out.push(consumed.html);
      i = consumed.next;
      continue;
    }

    // --- paragraph ---------------------------------------------------------
    const para = [];
    while (i < lines.length && lines[i].trim()
      && !HEADING_RE.test(lines[i]) && !FENCE_RE.test(lines[i]) && !HR_RE.test(lines[i])
      && !QUOTE_RE.test(lines[i]) && !BULLET_RE.test(lines[i]) && !ORDERED_RE.test(lines[i])
      && !isTableHead(lines, i)) {
      para.push(lines[i]);
      i++;
    }
    if (para.length) {
      const joined = para.map(l => l.trim()).join('\n');
      const html = inline(joined).replace(/\n/g, softBreaks ? '<br>' : ' ');
      out.push(`<p class="md-p">${html}</p>`);
    }
  }

  return out.join('\n');
}

/**
 * One list, however deep, starting at `start`.
 *
 * Nesting is by indentation against the FIRST item's indent: anything indented
 * further opens a sublist, rendered by recursion rather than by carrying a
 * stack. Returns where the caller should resume.
 */
function renderList(lines, start, opts) {
  const first = lines[start].match(BULLET_RE) || lines[start].match(ORDERED_RE);
  const baseIndent = first[1].length;
  const ordered = !BULLET_RE.test(lines[start]);
  const startNum = ordered ? parseInt(first[2], 10) : 1;

  const items = [];
  let i = start;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      // A blank line ends the list unless the next line continues it — that is
      // a "loose" list, and its items are still items.
      const next = lines[i + 1];
      if (!next || !(BULLET_RE.test(next) || ORDERED_RE.test(next))) break;
      i++;
      continue;
    }
    const m = line.match(BULLET_RE) || line.match(ORDERED_RE);
    if (!m) {
      // A plain continuation line indented under the current item belongs to it.
      if (items.length && line.search(/\S/) > baseIndent) {
        items[items.length - 1].lines.push(line.trim());
        i++;
        continue;
      }
      break;
    }
    const indent = m[1].length;
    if (indent < baseIndent) break;
    if (indent > baseIndent) {
      // Sublist: hand the whole run to a recursive call and attach the result to
      // the item above it.
      const sub = renderList(lines, i, opts);
      if (items.length) items[items.length - 1].sub += sub.html;
      i = sub.next;
      continue;
    }
    // Same-level item. A bullet where an ordered list started (or the reverse)
    // ends this list rather than being silently renumbered.
    if (!BULLET_RE.test(line) !== ordered) break;
    items.push({ lines: [m[3]], sub: '' });
    i++;
  }

  const body = items.map(it => {
    const text = it.lines.join('\n');
    // A task-list checkbox renders as a real (disabled) checkbox — plans and
    // checklists are a common job output, and "[ ]" as literal text reads as a
    // typo.
    const task = text.match(/^\[([ xX])\]\s+([\s\S]*)$/);
    const inner = task
      ? `<input type="checkbox" disabled${task[1] === ' ' ? '' : ' checked'}> ${inline(task[2]).replace(/\n/g, '<br>')}`
      : inline(text).replace(/\n/g, '<br>');
    return `<li${task ? ' class="md-task"' : ''}>${inner}${it.sub}</li>`;
  }).join('');

  const tag = ordered ? 'ol' : 'ul';
  const attr = ordered && startNum !== 1 ? ` start="${startNum}"` : '';
  return { html: `<${tag} class="md-list"${attr}>${body}</${tag}>`, next: i };
}

/**
 * Markdown reduced to the words in it.
 *
 * For the card summary, and anywhere else a length has to be measured against
 * what a person would actually read. Deliberately lossy: a code block counts as
 * the single token "[code]" rather than as its own length, because a result that
 * is 90% source is a short note ABOUT a long file, and measuring it by raw
 * character count would call it a long document and route it wrong.
 */
function toPlainText(src) {
  return String(src == null ? '' : src)
    .replace(/\r\n?/g, '\n')
    .replace(/```[\s\S]*?(?:```|$)/g, ' [code] ')
    .replace(/^ {0,3}#{1,6}\s+/gm, '')
    .replace(/^ {0,3}>\s?/gm, '')
    .replace(/^(\s*)[-*+]\s+/gm, '$1')
    .replace(/^(\s*)\d{1,9}[.)]\s+/gm, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/(\*\*\*|\*\*|__|~~)/g, '')
    .replace(/(^|[\s(])[*_]([^*_\n]+)[*_](?=$|[\s).,;:!?])/g, '$1$2')
    .replace(/^ {0,3}([-*_])(?:\s*\1){2,}\s*$/gm, '')
    // A table's separator row is pure syntax; its data rows are words with pipes
    // between them. Left alone, "|---|---|" was the single most common thing in
    // a card summary of a report, which is how the thin-column bug looked even
    // after the card itself started rendering properly.
    .replace(/^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$/gm, '')
    .replace(/^[ \t]*\|(.*)\|[ \t]*$/gm, (m, row) => row.split('|').map(c => c.trim()).filter(Boolean).join(' · '))
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

const api = { renderMarkdown, toPlainText, esc, safeUrl, inline };

// Node and the browser, from one file. server.js serves this path verbatim at
// /markdown.js; there is no second copy in public/ to fall out of step.
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else if (typeof window !== 'undefined') window.SNHMarkdown = api;
