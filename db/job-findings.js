/**
 * A LONG JOB IS NOT ONE CONTEXT WINDOW.
 *
 * The 9/17 job tried to hold six research questions, forty tool results and
 * the answer in a single transcript, and the transcript did not fit the model.
 * Compaction (db/job-compaction.js) makes a transcript smaller; this makes the
 * job not depend on one transcript at all. A job runs in PHASES. Each phase is
 * a fresh context — the brief, one phase's goal, and the findings document so
 * far — and ends when the model writes that phase up. The RUNNER appends the
 * writeup to the document on disk, with the sources the phase actually drew on
 * (from the tool record, so a writeup that forgot a URL still has it), and
 * starts the next phase clean.
 *
 * THE RUNNER DOES THE CHECKPOINT, NOT THE MODEL. Deliberate. A job that dies
 * in phase 3 has phases 1 and 2 on disk with nothing depending on the entity
 * having remembered to save — the document is written the moment a phase
 * ends, from the runner's own record. The entity writing its own documents is
 * separate work (todo #240); this is built so that can sit beside it: the
 * document here is a plain markdown file in her documents folder, named for
 * the job, and nothing in it is a format the entity would have to learn.
 *
 * WHAT IS IN HERE: the phase plan (one model call, or one phase when the
 * brief is one question), the document's path and rendering, what a phase
 * carries into the next, and how a dead job's result is assembled from what
 * its phases had. No queue, no status writes — db/agent-jobs.js owns those.
 */

const fs = require('fs');
const path = require('path');

function artifacts() { return require('./job-artifacts'); }

/** A phase record, as the checkpoint and the row carry it. */
function newPhase(n, goal) {
  return { n, goal: String(goal || '').trim(), status: 'pending', parts: 0, rounds: 0, calls: 0, writeup: null, sources: [], startedAt: null, endedAt: null, stop: null };
}

/**
 * Split the brief into phases. One model call, no tools, JSON out. A brief
 * with one question comes back as one phase, which makes the job behave
 * exactly as it did before phases existed — one transcript, one answer.
 *
 * @param {Object} job            {title, task}
 * @param {Object} opts
 * @param {Function} opts.callLLM the model call (injected so a test can pin it)
 * @param {number} [opts.maxPhases=6]
 * @returns {Promise<{phases: Array, planned: boolean, why: string}>}
 */
/** Does the brief look like more than one thing? A list, numbered items, or simple length. */
function looksMultipart(task, { minChars = 400 } = {}) {
  const t = String(task || '');
  if (t.length >= minChars) return true;
  return /(^|\n)\s*(?:\d+[).:]|[-*•])\s+\S/.test(t) || /\b(?:first|second|third|then|finally|and also|as well as)\b/i.test(t) && t.length > 200;
}

async function planPhases(job, { callLLM, maxPhases = 6, thinkingTokens = 0 } = {}) {
  const one = () => ({ phases: [newPhase(1, 'the whole job')], planned: false, why: 'one phase' });
  if (typeof callLLM !== 'function') return { ...one(), why: 'no planner' };
  // A short, single-question brief is one phase without asking — the planning
  // call is a real model call, and spending one to learn that "count my facts"
  // is one thing would be the cost with none of the point.
  if (!looksMultipart(job.task)) return { ...one(), why: 'one phase (a short brief)' };
  const system =
    `You are about to run one of your own background jobs, and it may be too large for one sitting. Split THE JOB ` +
    `into ordered phases you will work through one at a time, each ending with a written set of findings that the ` +
    `next phase can build on. A phase is a coherent piece of the work — one question, one section, one area to ` +
    `research — described in one plain line. Between 1 and ${maxPhases} phases. A job that is one question, one ` +
    `lookup or one thing to write is ONE phase; do not pad. WHEN THE JOB ALREADY NUMBERS ITS PARTS, the phases ARE ` +
    `those parts, in that order, one each (grouped only if there are more than ${maxPhases}) — do not invent a ` +
    `preliminary "confirm the basics" phase in front of them; the first part covers what it covers. Do not add a ` +
    `"write the final report" phase — that happens after the last one on its own.\n` +
    `Return ONLY a JSON object: {"phases":["goal of phase 1","goal of phase 2"]}`;
  const user = `THE JOB:\n${String(job.task || '').slice(0, 8000)}`;
  try {
    const res = await callLLM(system, user, { maxTokens: 600, thinkingTokens });
    const text = String(res && res.content || '');
    const m = text.match(/\{[\s\S]*\}/);
    const parsed = m ? JSON.parse(m[0]) : null;
    const goals = parsed && Array.isArray(parsed.phases)
      ? parsed.phases.filter(g => typeof g === 'string' && g.trim()).map(g => g.trim().slice(0, 240)).slice(0, Math.max(1, maxPhases))
      : [];
    if (goals.length <= 1) return { phases: [newPhase(1, goals[0] || 'the whole job')], planned: true, why: 'the plan came back as one phase' };
    return { phases: goals.map((g, i) => newPhase(i + 1, g)), planned: true, why: `${goals.length} phases` };
  } catch (err) {
    console.warn(`[JobFindings] phase plan failed, running as one phase: ${err && err.message}`);
    return { ...one(), why: `plan failed: ${err && err.message}` };
  }
}

/** Where the document lives: her documents folder (or the throwaway's), named for the job. */
function findingsPath(job, { now = new Date() } = {}) {
  const a = artifacts();
  const dir = a.outputDir();
  const p = (n) => String(n).padStart(2, '0');
  const date = `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}`;
  return a.uniquePath(dir, `${date}-${a.slug(job.title || job.task, 40)}-findings`, '.md');
}

function sayStatus(ph) {
  switch (ph.status) {
    case 'done': return 'done';
    case 'running': return 'in progress when the job stopped';
    case 'stopped': return `stopped — ${ph.stop || 'no reason recorded'}`;
    case 'skipped': return 'not reached';
    default: return 'not started';
  }
}

/** The document, rendered whole from the phase records. */
function renderDocument({ job, phases = [], stop = null, now = new Date() }) {
  const lines = [];
  lines.push(`# ${job.title || 'Background job'} — findings by phase`);
  lines.push('');
  lines.push(`Written by SNH's job runner as the job ran, one section per phase, updated ${now.toISOString()}. ` +
    `Each section is the phase's own writeup, followed by the sources that phase read or found — kept by the runner from the tool record.`);
  lines.push('');
  lines.push('## The job');
  lines.push('');
  lines.push(String(job.task || '').trim());
  lines.push('');
  lines.push('## Phases');
  lines.push('');
  for (const ph of phases) lines.push(`${ph.n}. ${ph.goal} — ${sayStatus(ph)}${ph.calls ? ` (${ph.calls} tool call${ph.calls === 1 ? '' : 's'}, ${ph.rounds} round${ph.rounds === 1 ? '' : 's'})` : ''}`);
  lines.push('');
  if (stop) { lines.push(`**The job stopped:** ${stop}`); lines.push(''); }
  for (const ph of phases) {
    if (!ph.writeup && !(ph.sources && ph.sources.length)) continue;
    lines.push(`## Phase ${ph.n}: ${ph.goal}`);
    lines.push('');
    if (ph.status !== 'done') { lines.push(`_${sayStatus(ph)}._`); lines.push(''); }
    if (ph.writeup) { lines.push(String(ph.writeup).trim()); lines.push(''); }
    if (ph.sources && ph.sources.length) {
      lines.push(`### Sources (phase ${ph.n})`);
      lines.push('');
      for (const s of ph.sources) {
        const bits = [s.title ? `**${s.title}**` : null, s.url ? `<${s.url}>` : null, s.date ? String(s.date).slice(0, 10) : null, s.how === 'read' ? 'read' : 'found in search'].filter(Boolean);
        lines.push(`- ${bits.join(' — ')}`);
      }
      lines.push('');
    }
  }
  return lines.join('\n');
}

/** Write the document atomically. Never throws — a document that cannot be written is logged, not fatal. */
function writeDocument(file, doc) {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, doc, 'utf8');
    fs.renameSync(tmp, file);
    return true;
  } catch (err) {
    console.warn(`[JobFindings] could not write ${file}: ${err.message}`);
    return false;
  }
}

/**
 * What the next phase is told about the ones before it: the document, capped.
 * The most recent phases go in verbatim; when the whole thing is over the cap,
 * earlier phases are cut to their opening. The transcript is never carried —
 * that is the point.
 */
function carryBlock(phases = [], { carryChars = 12000 } = {}) {
  const done = phases.filter(p => p.writeup);
  if (!done.length) return '';
  const sections = done.map(p => `### Phase ${p.n}: ${p.goal}\n${String(p.writeup).trim()}`);
  let total = sections.reduce((n, s) => n + s.length, 0);
  // Cut from the oldest until it fits, never below 1,200 chars of any section.
  for (let i = 0; i < sections.length - 1 && total > carryChars; i++) {
    const keep = Math.max(1200, sections[i].length - (total - carryChars));
    if (keep < sections[i].length) {
      total -= sections[i].length - keep;
      sections[i] = `${sections[i].slice(0, keep)}\n…(cut here — the full text is in the findings document)`;
    }
  }
  return sections.join('\n\n');
}

/**
 * The result text for a job that ended WITHOUT a synthesis — died mid-phase,
 * ran out of budget, was declined — assembled from the phases it finished
 * plus whatever the current one salvaged. What she reads is what it found,
 * with the stop underneath, never the stop alone.
 */
function assembleResult({ phases = [], partial = null, stop = null }) {
  const out = [];
  const done = phases.filter(p => p.writeup && p.status === 'done');
  const current = phases.find(p => p.status === 'running' || p.status === 'stopped');
  if (done.length) {
    out.push(`Findings from ${done.length} of ${phases.length} phase${phases.length === 1 ? '' : 's'}${current ? ` — it stopped in phase ${current.n} (${current.goal})` : ''}.`);
    out.push('');
    for (const p of done) { out.push(`## Phase ${p.n}: ${p.goal}`); out.push(''); out.push(String(p.writeup).trim()); out.push(''); }
  }
  if (partial && String(partial).trim()) {
    out.push(`## ${current ? `Phase ${current.n}: ${current.goal}` : 'What it had'} — partial`);
    out.push('');
    out.push(String(partial).trim());
    out.push('');
  }
  const notReached = phases.filter(p => p.status === 'pending' || p.status === 'skipped');
  if (notReached.length) {
    out.push(`## Not reached`);
    out.push('');
    for (const p of notReached) out.push(`- Phase ${p.n}: ${p.goal}`);
    out.push('');
  }
  if (stop) { out.push(`_Why it stopped: ${stop}_`); }
  return out.join('\n').trim();
}

/** The summary a row and a card carry — small, no writeups. */
function phaseSummary(phases = []) {
  return phases.map(p => ({ n: p.n, goal: p.goal, status: p.status, parts: p.parts || 0, rounds: p.rounds || 0, calls: p.calls || 0, sources: (p.sources || []).length, stop: p.stop || null }));
}

module.exports = { newPhase, planPhases, looksMultipart, findingsPath, renderDocument, writeDocument, carryBlock, assembleResult, phaseSummary, sayStatus };
