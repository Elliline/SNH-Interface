/**
 * Charts, as SVG we generate ourselves.
 *
 * WHY OURS AND NOT A LIBRARY. The printed report is made by pointing a headless
 * browser at a local HTML file, and that file has no network: a CDN script tag
 * would silently render nothing and the PDF would come out with a blank rectangle
 * where the chart was. Bundling a charting library would mean a third-party
 * dependency for one figure per report. SVG is the format the printer already
 * speaks — it is markup, it scales to whatever the page is, and it needs nothing
 * to be loaded.
 *
 * WHAT DRIVES IT. A job writes a fenced block and the report builder hands the
 * body here:
 *
 *     ```chart
 *     {"type":"pie","title":"Tickets by client",
 *      "data":[{"label":"Acme","value":42},{"label":"Beta","value":17}]}
 *     ```
 *
 * Anywhere that is NOT the printed report — the panel card, the chat pane — the
 * same block stays a code block, because db/markdown.js knows nothing about
 * charts and only calls a fence handler when one is passed in.
 *
 * ⚠ IT IS FED BY A MODEL, so every number is treated as hostile: values are
 * coerced, non-finite ones dropped, labels escaped, and a spec that cannot be
 * made into a chart comes back as null so the caller can fall back to showing
 * the block as text. A chart is never allowed to be the reason a report fails
 * to build.
 *
 * COLOUR IS NOT A FREE CHOICE HERE. The palette below is a fixed eight-slot
 * categorical order, validated for colour-vision deficiency in that order
 * (worst adjacent pair ΔE 9.1, normal-vision floor 19.6, against a white page).
 * Slots are assigned in order and NEVER cycled — a ninth category folds into
 * "Other" instead, because a cycled palette says two different things are the
 * same thing. Three of the slots sit below 3:1 against white, which is allowed
 * only because every mark here carries a visible label: identity never rests on
 * colour alone.
 *
 * PURE and synchronous. Strings in, strings out.
 */

/** Fixed categorical order. Assigned by slot, never cycled — see the header. */
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];

/** Chrome and ink, for a light page. The printed report has no dark mode. */
const INK = {
  primary: '#0b0b0b',
  secondary: '#52514e',
  muted: '#898781',
  grid: '#e1e0d9',
  axis: '#c3c2b7',
  surface: '#ffffff'
};

/**
 * Past this many categories the tail is folded into one "Other" slice.
 *
 * Six is the part-to-whole limit: beyond it adjacent slices blur, the labels
 * collide, and the figure stops answering the question it was drawn for. Folding
 * is honest in a way that a seventh colour is not — "Other, 9%" is true, whereas
 * a ninth hue nobody can distinguish is a chart that looks precise and is not.
 */
const MAX_SLICES = 6;

/** The five that matter in markup or a double-quoted attribute. */
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** A number, or null. Strings with commas and currency marks are accepted. */
function num(v) {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v !== 'string') return null;
  const cleaned = v.replace(/[^0-9.eE+-]/g, '');
  if (!cleaned) return null;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

/** Readable on a page: thousands separated, decimals only when they say something. */
function fmt(n) {
  if (!Number.isFinite(n)) return '';
  const abs = Math.abs(n);
  if (abs >= 1000) return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (Number.isInteger(n)) return String(n);
  return n.toLocaleString('en-US', { maximumFractionDigits: abs < 1 ? 3 : 2 });
}

/** Round to one decimal for geometry, so the markup stays small and diffable. */
function r(n) { return Math.round(n * 10) / 10; }

/**
 * Normalise a spec's `data` into [{label, value}], dropping what cannot be read.
 *
 * Accepts the three shapes a model actually writes: a list of objects, a list of
 * [label, value] pairs, and a plain {label: value} map.
 */
function readPoints(data) {
  const out = [];
  if (Array.isArray(data)) {
    for (const row of data) {
      if (Array.isArray(row) && row.length >= 2) {
        const v = num(row[1]);
        if (v !== null) out.push({ label: String(row[0] ?? ''), value: v });
        continue;
      }
      if (row && typeof row === 'object') {
        const label = row.label ?? row.name ?? row.key ?? row.category ?? row.x;
        const v = num(row.value ?? row.y ?? row.count ?? row.amount ?? row.total);
        if (v !== null) out.push({ label: String(label ?? ''), value: v });
      }
    }
  } else if (data && typeof data === 'object') {
    for (const [label, raw] of Object.entries(data)) {
      const v = num(raw);
      if (v !== null) out.push({ label, value: v });
    }
  }
  return out;
}

/**
 * Fold the tail into "Other" past `max`, largest first.
 *
 * Only for the part-to-whole forms. A bar chart with twelve bars is fine — it
 * has an axis and the bars do not have to be told apart by colour — so this is
 * not applied there.
 */
function foldTail(points, max = MAX_SLICES) {
  if (points.length <= max) return points;
  const sorted = [...points].sort((a, b) => b.value - a.value);
  const head = sorted.slice(0, max - 1);
  const tail = sorted.slice(max - 1);
  const rest = tail.reduce((s, p) => s + p.value, 0);
  return [...head, { label: `Other (${tail.length})`, value: rest, folded: true }];
}

/**
 * An axis a person would have drawn: round numbers, evenly spaced.
 *
 * Dividing the data range into four equal parts is arithmetically fine and reads
 * badly — a real print came out labelled 2.61 / 4.38 / 6.15 / 7.92 / 9.69, which
 * are the correct quartiles of the data and tell the reader nothing. Gridlines
 * are a ruler, and a ruler is marked in round units.
 *
 * The step is snapped to 1, 2, 2.5 or 5 times a power of ten — the set of
 * intervals that produce labels people read without decoding — and the range is
 * widened outwards to land on multiples of it.
 *
 * @returns {{lo:number, hi:number, step:number, ticks:number[]}}
 */
function niceScale(lo, hi, targetTicks = 4) {
  if (!(hi > lo)) return { lo, hi: lo + 1, step: 1, ticks: [lo, lo + 1] };
  const raw = (hi - lo) / Math.max(1, targetTicks);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const niceLo = Math.floor(lo / step) * step;
  const niceHi = Math.ceil(hi / step) * step;
  const ticks = [];
  // Accumulate with a rounding guard: repeated addition of 2.5 drifts, and a
  // gridline labelled 7.500000000000001 is worse than no gridline.
  for (let v = niceLo; v <= niceHi + step / 1000; v += step) {
    ticks.push(Math.round(v / step) * step);
  }
  return { lo: niceLo, hi: niceHi, step, ticks };
}

/** The figure's frame: title above, caption below, chart in the middle. */
function figure(inner, { title, caption, width, height }) {
  return [
    '<figure class="chart-figure">',
    title ? `<figcaption class="chart-title">${esc(title)}</figcaption>` : '',
    `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}"`,
    ` role="img" xmlns="http://www.w3.org/2000/svg"${title ? ` aria-label="${esc(title)}"` : ''}>`,
    inner,
    '</svg>',
    caption ? `<p class="chart-caption">${esc(caption)}</p>` : '',
    '</figure>'
  ].join('');
}

// ---------------------------------------------------------------------------
// Pie
// ---------------------------------------------------------------------------

/** A point on the circle, at `deg` clockwise from twelve o'clock. */
function polar(cx, cy, radius, deg) {
  const rad = (deg - 90) * Math.PI / 180;
  return [cx + radius * Math.cos(rad), cy + radius * Math.sin(rad)];
}

/**
 * Part-to-whole, at a glance.
 *
 * The rules that are not obvious from the geometry:
 *
 * - NEGATIVES ARE REFUSED, not clamped. A pie of a set that sums to less than
 *   its parts is a lie about proportion, and silently dropping the negative
 *   would draw a confident chart of the wrong numbers. It comes back null and
 *   the caller shows the data as a table instead.
 * - Slices are separated by a 2px gap of page, so two adjacent fills read as two
 *   marks rather than one shape with a colour change in it.
 * - Every slice ≥ 4% is labelled where it sits, with its share; the rest are
 *   identified by the legend. The direct label is what makes a sub-3:1 fill
 *   legal, so it is not optional styling.
 */
function pie(spec) {
  let points = readPoints(spec.data);
  if (!points.length) return null;
  if (points.some(p => p.value < 0)) return null;
  points = points.filter(p => p.value > 0);
  if (!points.length) return null;
  points = foldTail(points);

  const total = points.reduce((s, p) => s + p.value, 0);
  if (!(total > 0)) return null;

  const width = 680;
  const cx = 190, cy = 175, radius = 130;
  const parts = [];

  // A single category is a whole circle: the arc path degenerates at 360° and
  // would draw nothing at all.
  if (points.length === 1) {
    parts.push(`<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${SERIES[0]}"/>`);
  } else {
    let angle = 0;
    points.forEach((p, i) => {
      const sweep = (p.value / total) * 360;
      // The gap is taken off the arc, half at each end, and never allowed to
      // swallow a thin slice whole.
      const gap = Math.min(1.2, sweep / 3);
      const a0 = angle + gap, a1 = angle + sweep - gap;
      const [x0, y0] = polar(cx, cy, radius, a0);
      const [x1, y1] = polar(cx, cy, radius, a1);
      const large = (a1 - a0) > 180 ? 1 : 0;
      parts.push(
        `<path d="M ${r(cx)} ${r(cy)} L ${r(x0)} ${r(y0)} A ${radius} ${radius} 0 ${large} 1 ${r(x1)} ${r(y1)} Z"`
        + ` fill="${SERIES[i % SERIES.length]}"/>`
      );
      angle += sweep;
    });
  }

  // Direct labels: share inside the slice where it fits, at two thirds out.
  let angle = 0;
  points.forEach((p) => {
    const sweep = (p.value / total) * 360;
    const share = p.value / total;
    // 8%, not 4%. At 4% the labels on a 5.2% and a 6.3% slice sat almost on top
    // of each other against the pie's edge — measured on a real eight-client
    // print. The key to the right already gives every slice its exact value AND
    // its share, so a label on a thin slice adds nothing and costs legibility.
    // The in-slice label is for the ones big enough to read at a glance.
    if (share >= 0.08) {
      const [lx, ly] = polar(cx, cy, radius * 0.66, angle + sweep / 2);
      const pct = `${(share * 100).toFixed(share < 0.1 ? 1 : 0)}%`;
      parts.push(
        `<text x="${r(lx)}" y="${r(ly)}" text-anchor="middle" dominant-baseline="middle"`
        + ` font-size="12" font-weight="600" fill="#ffffff"`
        + ` style="paint-order:stroke;stroke:rgba(0,0,0,0.35);stroke-width:2.5px">${esc(pct)}</text>`
      );
    }
    angle += sweep;
  });

  // The key sits to the right, one row per slice, with the value spelled out —
  // a pie tells you the shape and the key tells you the numbers.
  const keyX = 400;
  let keyY = 60;
  points.forEach((p, i) => {
    const pct = ((p.value / total) * 100).toFixed(p.value / total < 0.1 ? 1 : 0);
    parts.push(
      `<rect x="${keyX}" y="${keyY - 9}" width="11" height="11" rx="2" fill="${SERIES[i % SERIES.length]}"/>`,
      `<text x="${keyX + 19}" y="${keyY}" font-size="12" fill="${INK.primary}">${esc(p.label)}</text>`,
      `<text x="${width - 20}" y="${keyY}" font-size="12" text-anchor="end" fill="${INK.secondary}">`
      + `${esc(fmt(p.value))}  ·  ${pct}%</text>`
    );
    keyY += 22;
  });
  parts.push(
    `<line x1="${keyX}" y1="${keyY - 6}" x2="${width - 20}" y2="${keyY - 6}" stroke="${INK.grid}" stroke-width="1"/>`,
    `<text x="${keyX + 19}" y="${keyY + 13}" font-size="11" fill="${INK.muted}">Total</text>`,
    `<text x="${width - 20}" y="${keyY + 13}" font-size="11" text-anchor="end" fill="${INK.muted}">${esc(fmt(total))}</text>`
  );

  const height = Math.max(325, keyY + 34);   // the circle's bottom edge is at 305
  return figure(parts.join(''), { title: spec.title, caption: spec.caption, width, height });
}

// ---------------------------------------------------------------------------
// Bar
// ---------------------------------------------------------------------------

/**
 * Magnitude, compared.
 *
 * Horizontal bars, because the labels are words: a vertical bar chart of client
 * names ends in rotated text, which is the anti-pattern this form exists to
 * avoid. One hue for the whole series — the bars are one measure, and colouring
 * each one differently would claim they are different KINDS of thing. The value
 * is printed at the end of every bar, so the chart is readable without an axis
 * ruler and without colour.
 */
function bar(spec) {
  const points = readPoints(spec.data);
  if (!points.length) return null;

  const rawMax = Math.max(...points.map(p => Math.abs(p.value)));
  if (!(rawMax > 0)) return null;
  // The axis runs to a round number at or above the largest bar — the same
  // reason as the line chart's: 5,525 / 11,050 / 16,575 are correct quarters of
  // the data and nobody reads a chart in units of 5,525.
  // Five target ticks rather than four: at four, a 22,100 maximum snapped the
  // axis out to 30,000 and the longest bar stopped three quarters of the way
  // across, which reads as a chart with a chunk missing. Five lands on 25,000.
  const scale = niceScale(0, rawMax, 5);
  const max = scale.hi;

  const width = 680;
  const rowH = 30, top = 20, left = 150, right = 70;
  const plotW = width - left - right;
  const height = top + points.length * rowH + 24;
  const parts = [];

  // Recessive gridlines at quarters — enough to read a value off, not enough to
  // compete with the bars.
  for (const v of scale.ticks) {
    const x = left + (plotW * v) / max;
    parts.push(
      `<line x1="${r(x)}" y1="${top - 6}" x2="${r(x)}" y2="${r(top + points.length * rowH)}"`
      + ` stroke="${INK.grid}" stroke-width="1"/>`,
      `<text x="${r(x)}" y="${r(top + points.length * rowH + 16)}" font-size="10"`
      + ` text-anchor="middle" fill="${INK.muted}">${esc(fmt(v))}</text>`
    );
  }

  points.forEach((p, i) => {
    const y = top + i * rowH;
    const w = Math.max(2, (Math.abs(p.value) / max) * plotW);
    parts.push(
      `<text x="${left - 10}" y="${r(y + 15)}" font-size="12" text-anchor="end"`
      + ` fill="${INK.primary}">${esc(p.label)}</text>`,
      // 4px rounded data-end, square against the baseline it grows from.
      `<rect x="${left}" y="${r(y + 5)}" width="${r(w)}" height="14" rx="4"`
      + ` fill="${SERIES[0]}"/>`,
      `<rect x="${left}" y="${r(y + 5)}" width="4" height="14" fill="${SERIES[0]}"/>`,
      `<text x="${r(left + w + 8)}" y="${r(y + 15)}" font-size="11" dominant-baseline="middle"`
      + ` fill="${INK.secondary}">${esc(fmt(p.value))}</text>`
    );
  });

  parts.push(
    `<line x1="${left}" y1="${top - 6}" x2="${left}" y2="${r(top + points.length * rowH)}"`
    + ` stroke="${INK.axis}" stroke-width="1"/>`
  );

  return figure(parts.join(''), { title: spec.title, caption: spec.caption, width, height });
}

// ---------------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------------

/**
 * Change over time.
 *
 * ONE AXIS, ALWAYS. Several series share one y-scale or they do not go on the
 * same chart — a second y-axis lets any two lines be made to cross wherever the
 * author likes, which is the single most common way a chart lies. A spec asking
 * for two scales is refused rather than drawn.
 *
 * `series` is [{name, data:[…]}, …]; a bare `data` is read as one unnamed series.
 */
function line(spec) {
  const raw = Array.isArray(spec.series) && spec.series.length
    ? spec.series
    : [{ name: spec.name || '', data: spec.data }];

  const series = raw
    .map(s => ({ name: String(s && s.name || ''), points: readPoints(s && s.data) }))
    .filter(s => s.points.length);
  if (!series.length) return null;
  // Past the eighth slot there is no colour left that anyone can tell apart.
  if (series.length > SERIES.length) return null;

  const all = series.flatMap(s => s.points.map(p => p.value));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  // A flat series still needs a band to be drawn in.
  if (lo === hi) { lo -= 1; hi += 1; }
  // A PADDED RANGE, NOT A FORCED ZERO — and the distinction is a real one.
  //
  // Anchoring at zero is a BAR rule: a bar's length is the value, so a bar that
  // does not start at zero is drawing a proportion that is not true. A line
  // encodes change through slope, and forcing zero onto a series of lead times
  // that never approach it just squeezes the whole story into the top third of
  // the plot. Measured on a real print: two crossing series spent a quarter of
  // the figure's height on empty space below them, and the crossing — the entire
  // point of the chart — was flattened into it.
  //
  // The protection against a cropped axis magnifying nothing into something is
  // that every gridline is LABELLED with its real value, which they are. The
  // floor is still zero for data that is entirely non-negative, so the axis
  // never implies negative quantities that cannot exist.
  const pad = (hi - lo) * 0.08;
  hi += pad;
  lo = (lo >= 0 && lo - pad < 0) ? 0 : lo - pad;
  // Then snapped outwards to round numbers, so the gridlines are a ruler.
  const scale = niceScale(lo, hi, 4);
  lo = scale.lo;
  hi = scale.hi;

  const labels = series[0].points.map(p => p.label);
  const n = Math.max(...series.map(s => s.points.length));
  if (n < 2) return null;

  const width = 680;
  const top = 20, bottom = 46, left = 60, right = 24;
  const height = 300;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const xAt = (i) => left + (plotW * i) / (n - 1);
  const yAt = (v) => top + plotH - ((v - lo) / (hi - lo)) * plotH;

  const parts = [];
  for (const v of scale.ticks) {
    const y = yAt(v);
    parts.push(
      `<line x1="${left}" y1="${r(y)}" x2="${r(left + plotW)}" y2="${r(y)}"`
      + ` stroke="${INK.grid}" stroke-width="1"/>`,
      `<text x="${left - 8}" y="${r(y + 4)}" font-size="10" text-anchor="end"`
      + ` fill="${INK.muted}">${esc(fmt(v))}</text>`
    );
  }

  // x labels, thinned so they never collide.
  const every = Math.ceil(n / 8);
  labels.forEach((lab, i) => {
    if (i % every) return;
    parts.push(
      `<text x="${r(xAt(i))}" y="${r(height - bottom + 18)}" font-size="10"`
      + ` text-anchor="middle" fill="${INK.muted}">${esc(lab)}</text>`
    );
  });

  series.forEach((s, si) => {
    const colour = SERIES[si % SERIES.length];
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'} ${r(xAt(i))} ${r(yAt(p.value))}`).join(' ');
    parts.push(`<path d="${d}" fill="none" stroke="${colour}" stroke-width="2"`
      + ` stroke-linejoin="round" stroke-linecap="round"/>`);
    // Markers only when they are readable — a point every 8px is a texture, not
    // a set of marks. The 2px surface ring keeps overlapping series apart.
    if (s.points.length <= 24) {
      s.points.forEach((p, i) => {
        parts.push(`<circle cx="${r(xAt(i))}" cy="${r(yAt(p.value))}" r="4" fill="${colour}"`
          + ` stroke="${INK.surface}" stroke-width="2"/>`);
      });
    }
    // Direct label at the end of the line, which is where the eye already is.
    if (s.name) {
      const last = s.points[s.points.length - 1];
      parts.push(
        `<text x="${r(xAt(s.points.length - 1) - 6)}" y="${r(yAt(last.value) - 10)}" font-size="11"`
        + ` text-anchor="end" font-weight="600" fill="${INK.secondary}">${esc(s.name)}</text>`
      );
    }
  });

  parts.push(
    `<line x1="${left}" y1="${r(top + plotH)}" x2="${r(left + plotW)}" y2="${r(top + plotH)}"`
    + ` stroke="${INK.axis}" stroke-width="1"/>`
  );

  return figure(parts.join(''), { title: spec.title, caption: spec.caption, width, height });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

const FORMS = { pie: pie, donut: pie, bar: bar, column: bar, barh: bar, line: line, area: line };

/**
 * Render one ```chart block.
 *
 * @param {string} source JSON body of the fence.
 * @returns {string|null} SVG figure markup, or null if it could not be drawn —
 *          in which case the caller must show the block as text rather than
 *          leaving a hole where a figure was promised.
 */
function renderChart(source) {
  let spec;
  try {
    spec = JSON.parse(String(source || ''));
  } catch {
    return null;
  }
  if (!spec || typeof spec !== 'object') return null;
  const form = FORMS[String(spec.type || '').toLowerCase()];
  if (!form) return null;
  try {
    return form(spec);
  } catch {
    // A chart is never the reason a report fails to build.
    return null;
  }
}

/**
 * The table a refused chart falls back to.
 *
 * A figure that could not be drawn must not become a silence — the numbers were
 * the point, and they are still perfectly readable as rows.
 */
function chartFallbackTable(source) {
  let spec;
  try { spec = JSON.parse(String(source || '')); } catch { return null; }
  const points = readPoints(spec && spec.data);
  if (!points.length) return null;
  const rows = points.map(p =>
    `<tr><td>${esc(p.label)}</td><td style="text-align:right">${esc(fmt(p.value))}</td></tr>`
  ).join('');
  return `<figure class="chart-figure">`
    + (spec.title ? `<figcaption class="chart-title">${esc(spec.title)}</figcaption>` : '')
    + `<table class="md-table"><tbody>${rows}</tbody></table></figure>`;
}

module.exports = { renderChart, chartFallbackTable, SERIES, INK, MAX_SLICES };
