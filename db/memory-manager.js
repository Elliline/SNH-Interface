/**
 * Memory Manager Heartbeat
 * Scheduled background job that maintains and optimizes the memory system.
 * Runs every 2 hours (configurable) with a 4-step audit pipeline + cleanup tasks:
 *   Step 1: auditClusterCoherence — per-cluster LLM coherence check, flags splits
 *   Step 2: executeSplits         — apply flagged splits, re-embed moved facts
 *   Step 3: auditCrossLinks       — LLM-driven batch link scoring across all cluster pairs
 *   Step 4: generateReport        — build report object, log to console + daily file
 *   Task B: cleanupFacts          — LLM-driven fact dedup/reword/merge in MEMORY.md
 *   Task C: summarizeDailyLogs    — archive daily logs older than retention window
 */

const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const { getConfig, getProviderInstance } = require('./config');

const { getSqliteDb, getClusterEmbeddingsTable } = require('./database');
const memoryClusters = require('./memory-clusters');
const factExtractor = require('./fact-extractor');

const MEMORY_DIR = path.join(__dirname, '../data/memory');
const DAILY_DIR = path.join(MEMORY_DIR, 'daily');
const ARCHIVE_DIR = path.join(DAILY_DIR, 'archive');

let heartbeatTimer = null;
let warmupTimer = null;
let isRunning = false;

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Safely build a LanceDB delete filter for member_id, validating UUID format first */
function memberIdFilter(id) {
  if (!UUID_RE.test(id)) throw new Error(`Invalid member_id format: ${id}`);
  return `member_id = "${id}"`;
}

// ============ LLM Helper ============

/**
 * Call an LLM with system + user prompts.
 * Uses the heartbeat model/provider from config.
 * @param {string} systemPrompt
 * @param {string} userPrompt
 * @returns {Promise<{content: string, provider: string}>}
 */
async function callLLM(systemPrompt, userPrompt, options = {}) {
  const config = getConfig();
  const heartbeatModel = config.models.heartbeat;
  const inst = getProviderInstance(heartbeatModel.provider, heartbeatModel.instance);
  const host = inst ? inst.host : 'http://localhost:11434';
  const maxTokens = options.maxTokens ?? 1024;
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt }
  ];

  // Build provider call based on config
  let url, body, extract;
  if (['llamacpp', 'vllm'].includes(heartbeatModel.provider)) {
    url = `${host}/v1/chat/completions`;
    body = { messages, stream: false, max_tokens: maxTokens };
    extract = (data) => data.choices?.[0]?.message?.content || '';
  } else {
    url = `${host}/api/chat`;
    body = { model: heartbeatModel.model, messages, stream: false, options: { num_predict: maxTokens } };
    extract = (data) => data.message?.content || '';
  }

  const providers = [
    {
      name: `${heartbeatModel.provider}/${heartbeatModel.model}`,
      url,
      body,
      extract
    }
  ];

  let lastError = null;

  for (const provider of providers) {
    try {
      console.log(`[Heartbeat] Trying ${provider.name} → ${provider.url}`);
      const response = await fetch(provider.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(provider.body),
        signal: AbortSignal.timeout(60000)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const content = provider.extract(data);
      if (content) {
        console.log(`[Heartbeat] ${provider.name} responded (${content.length} chars)`);
        return { content, provider: provider.name };
      }
      throw new Error('Empty response');
    } catch (err) {
      console.log(`[Heartbeat] ${provider.name} failed: ${err.message}`);
      lastError = err;
    }
  }

  throw new Error(`All LLM providers failed. Last error: ${lastError?.message}`);
}

/**
 * Parse a JSON object from LLM response text (handles markdown code blocks)
 * @param {string} text - LLM response
 * @returns {Object|null}
 */
function parseJSON(text) {
  try {
    // Try to find JSON object or array in the response
    const match = text.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);

    const arrMatch = text.match(/\[[\s\S]*\]/);
    if (arrMatch) return JSON.parse(arrMatch[0]);

    return null;
  } catch {
    return null;
  }
}

// ============ Step 1: Audit Cluster Coherence ============

/**
 * Audit a single cluster for internal coherence using the LLM.
 * Returns an audit result object for that cluster, including any suggested splits.
 *
 * Designed as a self-contained unit so that future parallelization is trivial —
 * just swap the sequential loop for Promise.all().
 *
 * @param {Object} cluster - Cluster row from getClusters() (has id, name, member_count)
 * @returns {Promise<{clusterId: string, clusterName: string, coherent: boolean, splits: Array, durationMs: number, error?: string}>}
 */
async function auditClusterCoherence(cluster) {
  const startMs = Date.now();
  const base = { clusterId: cluster.id, clusterName: cluster.name, coherent: true, splits: [], durationMs: 0 };

  try {
    const detail = memoryClusters.getCluster(cluster.id);
    if (!detail || !Array.isArray(detail.members) || detail.members.length === 0) {
      base.durationMs = Date.now() - startMs;
      return base;
    }

    const factLines = detail.members.map(m => {
      const ts = m.created_at ? m.created_at.split('T')[0] : 'unknown';
      const src = m.source || 'unknown';
      return `[id:${m.id}] [date:${ts}] [source:${src}] ${m.content}`;
    }).join('\n');

    const systemPrompt = `You are a memory cluster coherence auditor. Your job is to decide whether all the facts in a named cluster genuinely belong together.

Cluster name: "${cluster.name}"

Return ONLY valid JSON in this exact format:
{
  "coherent": true,
  "splits": []
}

If the cluster contains clearly distinct categories that do NOT belong under a single name, set "coherent" to false and list the splits:
{
  "coherent": false,
  "splits": [
    {
      "newClusterName": "Descriptive Category Name",
      "factIds": ["id1", "id2"]
    }
  ]
}

Rules:
- Only flag a split when the facts fall into genuinely different topics. Do NOT split if facts are loosely related to the same theme.
- Every fact id that appears in the cluster must appear in exactly one split group if you flag incoherence. Do not drop any.
- Split names should be concise noun phrases (2-4 words).
- If in doubt, return coherent: true.`;

    const { content } = await callLLM(systemPrompt, `Facts in cluster "${cluster.name}":\n${factLines}`);
    const parsed = parseJSON(content);

    if (!parsed) {
      base.durationMs = Date.now() - startMs;
      return { ...base, error: 'LLM returned unparseable JSON' };
    }

    base.coherent = parsed.coherent !== false;
    base.splits = Array.isArray(parsed.splits) ? parsed.splits : [];
    base.durationMs = Date.now() - startMs;
    return base;

  } catch (err) {
    base.durationMs = Date.now() - startMs;
    return { ...base, error: err.message };
  }
}

// ============ Step 2: Execute Splits ============

/**
 * Apply all cluster splits flagged by auditClusterCoherence.
 * For each split: creates new clusters, moves facts in SQLite, re-embeds in LanceDB.
 * Runs renameAllClusters() once if any splits were applied.
 *
 * @param {Array} auditResults - Array of results from auditClusterCoherence
 * @returns {Promise<{clustersSplit: number, splitDetails: Array, anomalies: Array}>}
 */
async function executeSplits(auditResults) {
  console.log('[Heartbeat] Step 2: Executing cluster splits...');

  const results = { clustersSplit: 0, splitDetails: [], anomalies: [] };
  const db = getSqliteDb();
  if (!db) {
    results.anomalies.push('SQLite not available — skipping splits');
    return results;
  }

  const clusterTable = await getClusterEmbeddingsTable();
  const incoherentResults = auditResults.filter(r => !r.coherent && r.splits && r.splits.length > 0);

  if (incoherentResults.length === 0) {
    console.log('[Heartbeat] No splits to execute');
    return results;
  }

  console.log(`[Heartbeat] Applying splits for ${incoherentResults.length} incoherent cluster(s)`);

  for (const auditResult of incoherentResults) {
    const { clusterId, clusterName, splits } = auditResult;

    try {
      // Verify source cluster still exists
      const sourceCluster = db.prepare('SELECT id FROM memory_clusters WHERE id = ?').get(clusterId);
      if (!sourceCluster) {
        results.anomalies.push(`Source cluster ${clusterId} (${clusterName}) no longer exists, skipping splits`);
        continue;
      }

      const now = new Date().toISOString();
      const movedMemberIds = new Set();
      const splitDetail = { originalCluster: clusterName, newClusters: [] };

      for (const split of splits) {
        if (!split.newClusterName || !Array.isArray(split.factIds) || split.factIds.length === 0) {
          results.anomalies.push(`Invalid split spec in cluster "${clusterName}": missing name or factIds`);
          continue;
        }

        // Resolve members that actually exist in this cluster
        const membersToMove = [];
        for (const rawFactId of split.factIds) {
          // Strip "id:" prefix the LLM echoes back from the audit prompt
          const factId = rawFactId.replace(/^id:/, '');
          const member = db.prepare('SELECT * FROM cluster_members WHERE id = ? AND cluster_id = ?')
            .get(factId, clusterId);
          if (member) {
            membersToMove.push(member);
          } else {
            results.anomalies.push(`Fact id "${factId}" not found in cluster "${clusterName}"`);
          }
        }

        if (membersToMove.length === 0) {
          results.anomalies.push(`No valid facts found for split "${split.newClusterName}" in cluster "${clusterName}"`);
          continue;
        }

        // Create new cluster
        const newClusterId = randomUUID();
        db.prepare('INSERT INTO memory_clusters (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)')
          .run(newClusterId, split.newClusterName, '', now, now);

        console.log(`[Heartbeat] Created cluster "${split.newClusterName}" (${newClusterId}), moving ${membersToMove.length} facts`);

        // Move facts
        for (const member of membersToMove) {
          db.prepare('UPDATE cluster_members SET cluster_id = ? WHERE id = ?')
            .run(newClusterId, member.id);
          movedMemberIds.add(member.id);

          if (clusterTable) {
            try {
              await clusterTable.delete(memberIdFilter(member.id));
              const embedding = await memoryClusters.generateEmbedding(member.content);
              if (embedding) {
                await clusterTable.add([{
                  id: randomUUID(),
                  member_id: member.id,
                  cluster_id: newClusterId,
                  content: member.content,
                  vector: Array.from(embedding)
                }]);
              }
            } catch (e) {
              console.error('[Heartbeat] LanceDB re-embed error:', e.message);
              results.anomalies.push(`LanceDB re-embed failed for member ${member.id}: ${e.message}`);
            }
          }
        }

        splitDetail.newClusters.push({ name: split.newClusterName, factsCount: membersToMove.length });
      }

      // Check how many facts remain in the original cluster
      const remaining = db.prepare('SELECT COUNT(*) as cnt FROM cluster_members WHERE cluster_id = ?')
        .get(clusterId);
      const remainingCount = remaining ? remaining.cnt : 0;

      if (remainingCount === 0) {
        // All facts moved out — delete original cluster and its links
        db.prepare('DELETE FROM cluster_links WHERE cluster_a = ? OR cluster_b = ?')
          .run(clusterId, clusterId);
        db.prepare('DELETE FROM memory_clusters WHERE id = ?').run(clusterId);
        console.log(`[Heartbeat] Deleted empty original cluster "${clusterName}"`);
        splitDetail.originalDeleted = true;
      } else {
        db.prepare('UPDATE memory_clusters SET updated_at = ? WHERE id = ?').run(now, clusterId);
        splitDetail.originalRetained = true;
        splitDetail.originalRemainingFacts = remainingCount;
      }

      if (splitDetail.newClusters.length > 0) {
        results.clustersSplit++;
        results.splitDetails.push(splitDetail);
      }

    } catch (err) {
      console.error(`[Heartbeat] Error executing splits for cluster "${clusterName}":`, err.message);
      results.anomalies.push(`Split execution failed for "${clusterName}": ${err.message}`);
    }
  }

  if (results.clustersSplit > 0) {
    console.log(`[Heartbeat] Renaming all clusters after ${results.clustersSplit} split(s)`);
    try {
      await memoryClusters.renameAllClusters();
    } catch (err) {
      console.error('[Heartbeat] renameAllClusters error:', err.message);
      results.anomalies.push(`renameAllClusters failed: ${err.message}`);
    }
  }

  console.log(`[Heartbeat] Split execution complete: ${results.clustersSplit} cluster(s) split`);
  return results;
}

// ============ Step 3: Audit Cross-Links ============

/**
 * Re-evaluate ALL cluster pair link strengths using batched LLM calls.
 * Processes up to 10 pairs per LLM call to avoid O(n²) individual calls.
 * Creates, updates, or removes links based on LLM scores vs. config threshold.
 *
 * @returns {Promise<{linksUpdated: number, linksAdded: number, linksRemoved: number, anomalies: Array}>}
 */
async function auditCrossLinks() {
  console.log('[Heartbeat] Step 3: Auditing cross-cluster links...');

  const results = { linksUpdated: 0, linksAdded: 0, linksRemoved: 0, anomalies: [] };
  const db = getSqliteDb();
  if (!db) {
    results.anomalies.push('SQLite not available — skipping cross-link audit');
    return results;
  }

  const config = getConfig();
  const linkThreshold = config.memory.clusterLinkThreshold || 0.50;

  const clusters = memoryClusters.getClusters();
  if (clusters.length < 2) {
    console.log('[Heartbeat] Fewer than 2 clusters — no links to audit');
    return results;
  }

  // Build brief summaries for each cluster (first 5 facts, truncated)
  // label disambiguates duplicate cluster names by appending the cluster's index
  const clusterSummaries = {};
  for (let i = 0; i < clusters.length; i++) {
    const cluster = clusters[i];
    try {
      const detail = memoryClusters.getCluster(cluster.id);
      const label = cluster.name + '#' + i;
      if (!detail || !Array.isArray(detail.members)) {
        clusterSummaries[cluster.id] = { name: cluster.name, label, summary: '(no members)' };
        continue;
      }
      const snippets = detail.members.slice(0, 5).map(m => m.content.slice(0, 120)).join('; ');
      clusterSummaries[cluster.id] = { name: cluster.name, label, summary: snippets || '(empty)' };
    } catch (err) {
      clusterSummaries[cluster.id] = { name: cluster.name, label: cluster.name + '#' + i, summary: '(error loading)' };
    }
  }

  // Generate all unique pairs
  const pairs = [];
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      pairs.push([clusters[i], clusters[j]]);
    }
  }

  console.log(`[Heartbeat] Evaluating ${pairs.length} cluster pair(s) in batches of 10`);

  const BATCH_SIZE = 10;

  const systemPrompt = `You are a memory cluster link evaluator. For each cluster pair listed, decide how strongly related they are on a scale from 0.0 (completely unrelated) to 1.0 (extremely closely related).

Return ONLY valid JSON in this exact format:
{
  "links": [
    {"pair": "ClusterA|ClusterB", "strength": 0.7, "reason": "brief reason"},
    ...
  ]
}

Rules:
- Use the pipe character | to separate cluster labels in the "pair" field, exactly as given.
- strength 0.0–0.3: unrelated or barely connected.
- strength 0.4–0.6: loosely related, share some context.
- strength 0.7–1.0: closely related, topics naturally co-occur.
- Return one entry per pair, in the same order as input.
- Keep "reason" under 15 words.`;

  for (let batchStart = 0; batchStart < pairs.length; batchStart += BATCH_SIZE) {
    const batch = pairs.slice(batchStart, batchStart + BATCH_SIZE);

    const pairDescriptions = batch.map(([a, b]) => {
      const sa = clusterSummaries[a.id];
      const sb = clusterSummaries[b.id];
      return `Pair "${sa.label}|${sb.label}":\n  ${sa.label}: ${sa.summary}\n  ${sb.label}: ${sb.summary}`;
    }).join('\n\n');

    let parsed = null;
    try {
      const { content } = await callLLM(systemPrompt, pairDescriptions);
      parsed = parseJSON(content);
    } catch (err) {
      console.error(`[Heartbeat] Cross-link batch ${batchStart / BATCH_SIZE + 1} LLM call failed:`, err.message);
      results.anomalies.push(`Cross-link batch LLM failed (pairs ${batchStart}–${batchStart + batch.length - 1}): ${err.message}`);
      continue;
    }

    if (!parsed || !Array.isArray(parsed.links)) {
      results.anomalies.push(`Cross-link batch ${batchStart / BATCH_SIZE + 1} returned unparseable JSON`);
      continue;
    }

    // Index the LLM results by pair key (both orderings)
    const linkMap = {};
    for (const entry of parsed.links) {
      if (typeof entry.pair === 'string') {
        linkMap[entry.pair] = entry;
        // Also store reversed key so lookup is order-independent
        const pipeIdx = entry.pair.lastIndexOf('|');
        if (pipeIdx > 0) {
          const left = entry.pair.slice(0, pipeIdx);
          const right = entry.pair.slice(pipeIdx + 1);
          linkMap[`${right}|${left}`] = entry;
        }
      }
    }

    for (const [a, b] of batch) {
      const nameA = clusterSummaries[a.id].name;
      const nameB = clusterSummaries[b.id].name;
      const labelA = clusterSummaries[a.id].label;
      const labelB = clusterSummaries[b.id].label;
      const key = `${labelA}|${labelB}`;
      const entry = linkMap[key];

      if (!entry || typeof entry.strength !== 'number') {
        results.anomalies.push(`No LLM rating for pair "${key}"`);
        continue;
      }

      const strength = Math.min(1.0, Math.max(0.0, entry.strength));

      // Look up existing link (either direction)
      const existing = db.prepare(`
        SELECT id, strength FROM cluster_links
        WHERE (cluster_a = ? AND cluster_b = ?)
           OR (cluster_a = ? AND cluster_b = ?)
      `).get(a.id, b.id, b.id, a.id);

      if (strength < linkThreshold) {
        if (existing) {
          db.prepare('DELETE FROM cluster_links WHERE id = ?').run(existing.id);
          results.linksRemoved++;
          console.log(`[Heartbeat] Removed link "${nameA}" ↔ "${nameB}" (strength ${strength.toFixed(2)} below threshold)`);
        }
        // No existing link and below threshold — nothing to do
      } else {
        if (existing) {
          if (Math.abs(existing.strength - strength) >= 0.01) {
            db.prepare('UPDATE cluster_links SET strength = ? WHERE id = ?')
              .run(strength, existing.id);
            results.linksUpdated++;
          }
        } else {
          const [orderedA, orderedB] = a.id < b.id ? [a.id, b.id] : [b.id, a.id];
          db.prepare('INSERT INTO cluster_links (id, cluster_a, cluster_b, strength) VALUES (?, ?, ?, ?)')
            .run(randomUUID(), orderedA, orderedB, strength);
          results.linksAdded++;
          console.log(`[Heartbeat] Added link "${nameA}" ↔ "${nameB}" (strength ${strength.toFixed(2)})`);
        }
      }
    }
  }

  console.log(`[Heartbeat] Cross-link audit complete: ${results.linksAdded} added, ${results.linksUpdated} updated, ${results.linksRemoved} removed`);
  return results;
}

// ============ Step 4: Generate Report ============

/**
 * Build a structured heartbeat report, log it to console and append to today's daily log file.
 *
 * @param {Object} opts
 * @param {number}  opts.cycleStartMs          - Date.now() at cycle start
 * @param {Array}   opts.auditResults          - Per-cluster audit result objects
 * @param {Object}  opts.splitResults          - Result from executeSplits
 * @param {Object}  opts.crossLinkResults      - Result from auditCrossLinks
 * @returns {Object} The report object
 */
function generateReport({ cycleStartMs, auditResults, splitResults, crossLinkResults }) {
  const totalDurationMs = Date.now() - cycleStartMs;
  const totalDuration = (totalDurationMs / 1000).toFixed(1) + 's';

  const clustersAudited = auditResults.length;
  const clustersSplit = splitResults.clustersSplit || 0;

  const perClusterTiming = auditResults.map(r => ({
    clusterName: r.clusterName,
    durationMs: r.durationMs || 0
  }));

  const anomalies = [
    ...auditResults.filter(r => r.error).map(r => `Audit error for "${r.clusterName}": ${r.error}`),
    ...(splitResults.anomalies || []),
    ...(crossLinkResults.anomalies || [])
  ];

  const report = {
    clustersAudited,
    clustersSplit,
    splitDetails: splitResults.splitDetails || [],
    linksUpdated: crossLinkResults.linksUpdated || 0,
    linksRemoved: crossLinkResults.linksRemoved || 0,
    linksAdded: crossLinkResults.linksAdded || 0,
    totalDuration,
    perClusterTiming,
    anomalies
  };

  // Console summary
  console.log('[Heartbeat] === Heartbeat Report ===');
  console.log(`[Heartbeat]   Clusters audited : ${clustersAudited}`);
  console.log(`[Heartbeat]   Clusters split   : ${clustersSplit}`);
  console.log(`[Heartbeat]   Links added      : ${report.linksAdded}`);
  console.log(`[Heartbeat]   Links updated    : ${report.linksUpdated}`);
  console.log(`[Heartbeat]   Links removed    : ${report.linksRemoved}`);
  console.log(`[Heartbeat]   Total duration   : ${totalDuration}`);
  if (anomalies.length > 0) {
    console.log(`[Heartbeat]   Anomalies (${anomalies.length}):`);
    for (const a of anomalies) console.log(`[Heartbeat]     - ${a}`);
  }
  console.log('[Heartbeat] === End Report ===');

  // Append to daily log
  try {
    const dailyDir = path.join(__dirname, '../data/memory/daily');
    if (!fs.existsSync(dailyDir)) fs.mkdirSync(dailyDir, { recursive: true });
    const today = new Date().toISOString().split('T')[0];
    const dailyFile = path.join(dailyDir, `${today}.md`);

    let splitSummary = '';
    if (report.splitDetails.length > 0) {
      splitSummary = '\n### Splits\n' + report.splitDetails.map(d => {
        const newNames = d.newClusters.map(c => `"${c.name}" (${c.factsCount} facts)`).join(', ');
        const fate = d.originalDeleted ? 'original deleted' : `original retained (${d.originalRemainingFacts} facts remaining)`;
        return `- "${d.originalCluster}" → ${newNames}; ${fate}`;
      }).join('\n');
    }

    let anomalySection = '';
    if (anomalies.length > 0) {
      anomalySection = '\n### Anomalies\n' + anomalies.map(a => `- ${a}`).join('\n');
    }

    const timingRows = perClusterTiming
      .sort((a, b) => b.durationMs - a.durationMs)
      .slice(0, 10)
      .map(t => `| ${t.clusterName} | ${t.durationMs}ms |`)
      .join('\n');

    const reportText = [
      `\n## Heartbeat Report — ${new Date().toISOString()}`,
      '',
      `| Metric | Value |`,
      `|--------|-------|`,
      `| Clusters audited | ${clustersAudited} |`,
      `| Clusters split | ${clustersSplit} |`,
      `| Links added | ${report.linksAdded} |`,
      `| Links updated | ${report.linksUpdated} |`,
      `| Links removed | ${report.linksRemoved} |`,
      `| Total duration | ${totalDuration} |`,
      splitSummary,
      timingRows.length > 0 ? `\n### Per-cluster audit timing (top 10)\n| Cluster | Duration |\n|---------|----------|\n${timingRows}` : '',
      anomalySection,
      ''
    ].join('\n');

    fs.appendFileSync(dailyFile, reportText, 'utf8');
    console.log(`[Heartbeat] Report appended to ${dailyFile}`);
  } catch (err) {
    console.error('[Heartbeat] Failed to write daily report:', err.message);
  }

  return report;
}

// ============ Task B: Cleanup Facts ============

async function cleanupFacts() {
  console.log('[Heartbeat] Task B: Cleaning up facts...');
  const results = { removed: 0, reworded: 0, merged: 0 };

  try {
    const memoryFile = path.join(MEMORY_DIR, 'MEMORY.md');
    if (!fs.existsSync(memoryFile)) {
      console.log('[Heartbeat] No MEMORY.md found');
      return results;
    }

    let content = fs.readFileSync(memoryFile, 'utf8');
    const facts = factExtractor.extractAllFactLines(content);

    if (facts.length < 3) {
      console.log('[Heartbeat] Too few facts to clean up');
      return results;
    }

    const systemPrompt = `You are a memory maintenance system. Review the facts below and suggest cleanup actions. Return ONLY valid JSON:
{"actions":[]}

Action types:
- remove: {"type":"remove","fact":"exact fact text","reason":"..."}
  Use for outdated, trivial, or clearly wrong facts.
- reword: {"type":"reword","original":"exact original text","replacement":"improved text","reason":"..."}
  Use for awkward phrasing, typos, or facts that could be clearer.
- merge: {"type":"merge","originals":["fact1","fact2"],"replacement":"merged fact","reason":"..."}
  Use when two or more facts say essentially the same thing.

Rules:
- Only suggest confident actions. When in doubt, leave facts alone.
- The "fact" and "original" fields must match the input EXACTLY (verbatim).
- Prefer merging duplicates over removing them.
- Do NOT remove facts just because they seem mundane — the user chose to remember them.
- If the facts look clean, return {"actions":[]}.`;

    const numberedFacts = facts.map((f, i) => `${i + 1}. ${f}`).join('\n');
    const { content: llmResponse } = await callLLM(systemPrompt, numberedFacts);
    const parsed = parseJSON(llmResponse);

    if (!parsed || !Array.isArray(parsed.actions) || parsed.actions.length === 0) {
      console.log('[Heartbeat] No fact cleanup actions suggested');
      return results;
    }

    console.log(`[Heartbeat] LLM suggested ${parsed.actions.length} fact cleanup actions`);

    const db = getSqliteDb();
    const clusterTable = await getClusterEmbeddingsTable();
    const lines = content.split('\n');

    // Process actions in reverse order of line position to preserve indices
    // Build a list of line operations first
    const lineOps = []; // { lineIndex, op: 'delete' | 'replace', newText? }

    for (const action of parsed.actions) {
      try {
        if (action.type === 'remove' && action.fact) {
          const lineIdx = lines.findIndex(l => l === `- ${action.fact}`);
          if (lineIdx >= 0) {
            lineOps.push({ lineIndex: lineIdx, op: 'delete' });
            console.log(`[Heartbeat] Removing fact: "${action.fact}" — ${action.reason}`);
            results.removed++;
          }

        } else if (action.type === 'reword' && action.original && action.replacement) {
          const lineIdx = lines.findIndex(l => l === `- ${action.original}`);
          if (lineIdx >= 0) {
            lineOps.push({ lineIndex: lineIdx, op: 'replace', newText: `- ${action.replacement}` });
            console.log(`[Heartbeat] Rewording: "${action.original}" → "${action.replacement}"`);
            results.reworded++;

            // Update cluster_members content + re-embed
            if (db) {
              const member = db.prepare('SELECT id, cluster_id FROM cluster_members WHERE content = ?')
                .get(action.original);
              if (member) {
                db.prepare('UPDATE cluster_members SET content = ? WHERE id = ?')
                  .run(action.replacement, member.id);
                if (clusterTable) {
                  try {
                    await clusterTable.delete(memberIdFilter(member.id));
                    const embedding = await memoryClusters.generateEmbedding(action.replacement);
                    if (embedding) {
                      await clusterTable.add([{
                        id: randomUUID(),
                        member_id: member.id,
                        cluster_id: member.cluster_id,
                        content: action.replacement,
                        vector: Array.from(embedding)
                      }]);
                    }
                  } catch (e) {
                    console.error('[Heartbeat] LanceDB re-embed error:', e.message);
                  }
                }
              }
            }
          }

        } else if (action.type === 'merge' && Array.isArray(action.originals) && action.replacement) {
          const lineIndices = [];
          for (const orig of action.originals) {
            const idx = lines.findIndex(l => l === `- ${orig}`);
            if (idx >= 0) lineIndices.push({ idx, text: orig });
          }
          if (lineIndices.length < 2) continue;

          // Replace first occurrence, delete the rest
          lineOps.push({ lineIndex: lineIndices[0].idx, op: 'replace', newText: `- ${action.replacement}` });
          for (let i = 1; i < lineIndices.length; i++) {
            lineOps.push({ lineIndex: lineIndices[i].idx, op: 'delete' });
          }

          console.log(`[Heartbeat] Merging ${lineIndices.length} facts into: "${action.replacement}"`);
          results.merged++;

          // Update first cluster member, delete others
          if (db) {
            let keptMember = null;
            for (const { text } of lineIndices) {
              const member = db.prepare('SELECT id, cluster_id FROM cluster_members WHERE content = ?').get(text);
              if (!member) continue;

              if (!keptMember) {
                keptMember = member;
                db.prepare('UPDATE cluster_members SET content = ? WHERE id = ?')
                  .run(action.replacement, member.id);
                if (clusterTable) {
                  try {
                    await clusterTable.delete(memberIdFilter(member.id));
                    const embedding = await memoryClusters.generateEmbedding(action.replacement);
                    if (embedding) {
                      await clusterTable.add([{
                        id: randomUUID(),
                        member_id: member.id,
                        cluster_id: member.cluster_id,
                        content: action.replacement,
                        vector: Array.from(embedding)
                      }]);
                    }
                  } catch (e) {
                    console.error('[Heartbeat] LanceDB re-embed error:', e.message);
                  }
                }
              } else {
                // Delete duplicate member
                db.prepare('DELETE FROM cluster_members WHERE id = ?').run(member.id);
                if (clusterTable) {
                  try {
                    await clusterTable.delete(memberIdFilter(member.id));
                  } catch (e) {
                    console.error('[Heartbeat] LanceDB delete error:', e.message);
                  }
                }
              }
            }
          }
        }
      } catch (actionErr) {
        console.error(`[Heartbeat] Error executing ${action.type} fact action:`, actionErr.message);
      }
    }

    // Apply line operations (sort by line index descending to preserve positions)
    lineOps.sort((a, b) => b.lineIndex - a.lineIndex);
    for (const op of lineOps) {
      if (op.op === 'delete') {
        lines.splice(op.lineIndex, 1);
      } else if (op.op === 'replace') {
        lines[op.lineIndex] = op.newText;
      }
    }

    // Write back
    fs.writeFileSync(memoryFile, lines.join('\n'), 'utf8');

  } catch (error) {
    console.error('[Heartbeat] cleanupFacts error:', error.message);
  }

  console.log(`[Heartbeat] Fact cleanup complete: ${results.removed} removed, ${results.reworded} reworded, ${results.merged} merged`);
  return results;
}

// ============ Task C: Summarize Daily Logs ============

async function summarizeDailyLogs() {
  console.log('[Heartbeat] Task C: Summarizing old daily logs...');
  const results = { archived: 0, factsExtracted: 0 };

  try {
    if (!fs.existsSync(DAILY_DIR)) {
      console.log('[Heartbeat] No daily log directory');
      return results;
    }

    const files = fs.readdirSync(DAILY_DIR).filter(f => f.endsWith('.md'));
    const config = getConfig();
    const retentionDays = config.memory.dailyLogRetentionDays;
    const now = new Date();
    const cutoff = new Date(now.getTime() - retentionDays * 24 * 60 * 60 * 1000);

    const oldFiles = files.filter(f => {
      const dateStr = f.replace('.md', '');
      const fileDate = new Date(dateStr);
      return !isNaN(fileDate.getTime()) && fileDate < cutoff;
    });

    if (oldFiles.length === 0) {
      console.log(`[Heartbeat] No daily logs older than ${retentionDays} days`);
      return results;
    }

    console.log(`[Heartbeat] Found ${oldFiles.length} daily logs to archive`);

    const memoryFile = path.join(MEMORY_DIR, 'MEMORY.md');

    for (const file of oldFiles) {
      try {
        const filePath = path.join(DAILY_DIR, file);
        const content = fs.readFileSync(filePath, 'utf8');

        if (content.trim().length < 20) {
          // Too short to summarize, just archive
          if (!fs.existsSync(ARCHIVE_DIR)) {
            fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
          }
          fs.renameSync(filePath, path.join(ARCHIVE_DIR, file));
          results.archived++;
          continue;
        }

        const systemPrompt = `You are a memory log summarizer. Review the daily log below and extract any important facts that should be preserved long-term. Return ONLY valid JSON:
{"summary":"one-line summary of the day","remainingFacts":["fact1","fact2"]}

Rules:
- remainingFacts should only contain facts worth preserving permanently (user preferences, project decisions, personal info).
- Write facts as "User has..." or "User prefers..." style.
- Skip routine entries like "Chat exchange with model - 0 facts extracted".
- If nothing is worth keeping, return {"summary":"...","remainingFacts":[]}.`;

        const { content: llmResponse } = await callLLM(systemPrompt, content);
        const parsed = parseJSON(llmResponse);

        if (parsed && Array.isArray(parsed.remainingFacts) && parsed.remainingFacts.length > 0) {
          const validFacts = parsed.remainingFacts.filter(f => typeof f === 'string' && f.trim().length > 0);
          if (validFacts.length > 0) {
            await factExtractor.appendToMemory(validFacts, memoryFile);
            results.factsExtracted += validFacts.length;
            console.log(`[Heartbeat] Extracted ${validFacts.length} facts from ${file}`);
          }
        }

        // Archive the file
        if (!fs.existsSync(ARCHIVE_DIR)) {
          fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
        }
        fs.renameSync(filePath, path.join(ARCHIVE_DIR, file));
        results.archived++;
        console.log(`[Heartbeat] Archived ${file}`);

      } catch (fileErr) {
        console.error(`[Heartbeat] Error processing daily log ${file}:`, fileErr.message);
      }
    }

  } catch (error) {
    console.error('[Heartbeat] summarizeDailyLogs error:', error.message);
  }

  console.log(`[Heartbeat] Daily log archival complete: ${results.archived} archived, ${results.factsExtracted} facts extracted`);
  return results;
}

// ============ Core Audit Pipeline ============

/**
 * Run the full cluster audit pipeline (steps 1–3) on the given cluster list.
 * Returns { auditResults, splitResults, crossLinkResults }.
 *
 * @param {Array} clusters - Array of cluster rows from getClusters(). Pass all for rebuildClusters, filtered for runMaintenance.
 * @returns {Promise<{auditResults: Array, splitResults: Object, crossLinkResults: Object}>}
 */
async function runAuditPipeline(clusters) {
  // Step 1: per-cluster coherence audit (sequential, self-contained per cluster)
  console.log(`[Heartbeat] Step 1: Auditing coherence of ${clusters.length} cluster(s)...`);
  const auditResults = [];
  for (const cluster of clusters) {
    console.log(`[Heartbeat] Auditing cluster "${cluster.name}" (${cluster.member_count} members)`);
    const result = await auditClusterCoherence(cluster);
    auditResults.push(result);
    if (!result.coherent) {
      console.log(`[Heartbeat] Cluster "${cluster.name}" flagged for ${result.splits.length} split(s)`);
    }
  }

  // Step 2: execute splits
  const splitResults = await executeSplits(auditResults);

  // Step 3: cross-link audit (always runs across ALL clusters, not just audited subset)
  const crossLinkResults = await auditCrossLinks();

  return { auditResults, splitResults, crossLinkResults };
}

// ============ Orchestration ============

/**
 * Run the full maintenance cycle.
 * Only audits clusters that exceed config.memory.maxFactsPerCluster (default 10).
 * @returns {Promise<Object>} Combined results
 */
async function runMaintenance() {
  if (isRunning) {
    console.log('[Heartbeat] Maintenance already in progress, skipping');
    return { skipped: true };
  }

  isRunning = true;
  const cycleStartMs = Date.now();
  console.log('[Heartbeat] === Starting maintenance cycle ===');

  try {
    const config = getConfig();
    const maxFacts = config.memory.maxFactsPerCluster || 10;

    const allClusters = memoryClusters.getClusters();
    const oversizedClusters = allClusters.filter(c => c.member_count > maxFacts);

    console.log(`[Heartbeat] ${allClusters.length} total cluster(s), ${oversizedClusters.length} exceed maxFactsPerCluster (${maxFacts})`);

    let auditResults = [];
    let splitResults = { clustersSplit: 0, splitDetails: [], anomalies: [] };
    let crossLinkResults = { linksUpdated: 0, linksAdded: 0, linksRemoved: 0, anomalies: [] };

    if (oversizedClusters.length > 0) {
      ({ auditResults, splitResults, crossLinkResults } = await runAuditPipeline(oversizedClusters));
    } else {
      console.log('[Heartbeat] No oversized clusters to audit — skipping steps 1–3');
      // Still run cross-link audit to keep link table healthy
      crossLinkResults = await auditCrossLinks();
    }

    // Task B & C unchanged
    const cleanup = await cleanupFacts();
    const archive = await summarizeDailyLogs();

    // Step 4: report
    const report = generateReport({ cycleStartMs, auditResults, splitResults, crossLinkResults });

    const elapsed = ((Date.now() - cycleStartMs) / 1000).toFixed(1) + 's';
    console.log(`[Heartbeat] === Maintenance complete in ${elapsed} ===`);

    return { report, cleanup, archive };
  } catch (error) {
    console.error('[Heartbeat] Maintenance cycle error:', error.message);
    return { error: error.message };
  } finally {
    isRunning = false;
  }
}

/**
 * Run the full cluster audit pipeline on ALL clusters regardless of size.
 * Skips cleanup and archival tasks. Useful for manual cluster reorganization.
 * @returns {Promise<Object>} Report object, or { skipped: true } if already running
 */
async function rebuildClusters() {
  if (isRunning) {
    console.log('[Heartbeat] Maintenance already in progress, skipping rebuildClusters');
    return { skipped: true };
  }

  isRunning = true;
  const cycleStartMs = Date.now();
  console.log('[Heartbeat] === Starting full cluster rebuild ===');

  try {
    const allClusters = memoryClusters.getClusters();
    console.log(`[Heartbeat] Rebuilding across all ${allClusters.length} cluster(s)`);

    if (allClusters.length === 0) {
      console.log('[Heartbeat] No clusters found — nothing to rebuild');
      const report = generateReport({
        cycleStartMs,
        auditResults: [],
        splitResults: { clustersSplit: 0, splitDetails: [], anomalies: [] },
        crossLinkResults: { linksUpdated: 0, linksAdded: 0, linksRemoved: 0, anomalies: [] }
      });
      return { report };
    }

    const { auditResults, splitResults, crossLinkResults } = await runAuditPipeline(allClusters);
    const report = generateReport({ cycleStartMs, auditResults, splitResults, crossLinkResults });

    const elapsed = ((Date.now() - cycleStartMs) / 1000).toFixed(1) + 's';
    console.log(`[Heartbeat] === Cluster rebuild complete in ${elapsed} ===`);

    return { report };
  } catch (error) {
    console.error('[Heartbeat] rebuildClusters error:', error.message);
    return { error: error.message };
  } finally {
    isRunning = false;
  }
}

// ============ Timer Controls ============

/**
 * Start the heartbeat timer using config values for interval and warmup.
 */
function startHeartbeat() {
  const config = getConfig();

  if (!config.heartbeat.enabled) {
    console.log('[Heartbeat] Disabled by config, skipping startup');
    return;
  }

  if (heartbeatTimer) {
    console.log('[Heartbeat] Already running, ignoring start');
    return;
  }

  const intervalMs = config.heartbeat.intervalHours * 60 * 60 * 1000;
  const warmupMs = config.heartbeat.warmupMinutes * 60 * 1000;

  console.log(`[Heartbeat] Scheduled every ${config.heartbeat.intervalHours}h (first run in ${config.heartbeat.warmupMinutes}min)`);

  // Warmup delay, then first run + interval
  warmupTimer = setTimeout(() => {
    warmupTimer = null;
    runMaintenance().catch(err => {
      console.error('[Heartbeat] Initial run error:', err.message);
    });

    heartbeatTimer = setInterval(() => {
      runMaintenance().catch(err => {
        console.error('[Heartbeat] Scheduled run error:', err.message);
      });
    }, intervalMs);
  }, warmupMs);
}

/**
 * Stop the heartbeat timer
 */
function stopHeartbeat() {
  if (warmupTimer) {
    clearTimeout(warmupTimer);
    warmupTimer = null;
  }
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
  console.log('[Heartbeat] Stopped');
}

module.exports = { runMaintenance, startHeartbeat, stopHeartbeat, rebuildClusters };
