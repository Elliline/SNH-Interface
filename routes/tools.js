/**
 * The tools API — what the Tools tab is built from.
 *
 * GENERATED, NOT LISTED. Every row here comes from MCPClient's catalogue and the
 * search providers' own specs, so the page cannot fall behind the registry the way
 * it did: on 2026-08-18 fourteen tools were registered and three appeared in
 * settings, because the page carried a hand-written list of its own. There is no
 * list in this file either — it reads the same table registration reads.
 *
 * SECRETS GO ONE WAY. GET returns whether a key is set, where it came from and
 * when it changed. It does not return the key, a prefix of it, or its length; the
 * server never puts any part of the value in a response, which is the easiest way
 * to keep "write-only" true. PUT accepts values, stores them encrypted
 * (db/secrets.js), and answers with the same status shape — so a save is
 * confirmable without anything being echoed back.
 *
 * A secret name is only accepted if some tool or provider DECLARED it. The route
 * is not a general-purpose writer of environment-shaped keys into a file.
 */

const express = require('express');
const rateLimit = require('express-rate-limit');
const router = express.Router();

const MCPClient = require('../mcp/mcp-client');
const secrets = require('../db/secrets');
// Through the module object, not destructured — so a test can pin the config this
// route describes without writing to the live data/config.json.
function getConfig() { return require('../db/config').getConfig(); }
function getSearchConfig() { return require('../db/config').getSearchConfig(); }
const { SEARCH_PROVIDER_SPECS } = require('../mcp/tools/search-providers');

/** Writing a key is a deliberate human action, and not one to script in bulk. */
const writeLimiter = rateLimit({
  windowMs: 5 * 60 * 1000,
  max: 30,
  message: { error: 'Too many requests — slow down' }
});

/**
 * The cards the page lays out, in order. Titles and blurbs live here because they
 * are about how the page reads, not about what a tool does — the tool's own
 * description comes from the tool. A card with no rows is simply not rendered.
 */
const CARDS = [
  { id: 'search', title: 'Web search', blurb: 'One tool, two providers. They are tried in the order below and the first one with results wins, so a provider that fails or finds nothing falls through to the next.' },
  { id: 'jobs', title: 'Background jobs', blurb: 'Work handed to an agent that keeps running after the conversation ends. Results land in the robot panel and never open a conversation.' },
  { id: 'memoryInspect', title: 'Reading his own memory', blurb: 'Read-only lookups into the fact store. None of these can change, add or remove a memory.' },
  { id: 'memoryWrite', title: 'Writing to memory', blurb: 'Recording something you asked to be remembered. Direct-execute, on the chat path, where a person is in the room.' },
  { id: 'cron', title: 'Scheduled jobs', blurb: 'Proposals only: a call raises an item in the bell panel for you to approve or reject. Nothing is scheduled without your decision.' },
  { id: 'correctorWrites', title: 'Repairing the record', blurb: 'The three narrow writes the background corrector may make. Never offered in a conversation, and every change is reversible from the Self tab.' }
];

/** Resolve a dotted path against live config, for a field's current value. */
function valueAt(cfg, path) {
  return String(path).split('.').reduce((o, k) => (o == null ? undefined : o[k]), cfg);
}

/** A field descriptor plus what it currently holds. */
function withValue(cfg, field) {
  return { ...field, value: valueAt(cfg, field.path) ?? null };
}

/** Every secret name any tool or provider declared. The write allowlist. */
function declaredSecretNames() {
  const names = new Set();
  for (const p of SEARCH_PROVIDER_SPECS) if (p.secret) names.add(p.secret.env);
  for (const entry of MCPClient.TOOL_CATALOGUE) {
    for (const s of entry.secrets || []) names.add(s.env);
  }
  return names;
}

/**
 * GET /api/tools
 * Everything the tab renders: cards, one row per registered-or-not tool, the search
 * provider chain with switches and positions, and secret STATUS (never values).
 */
router.get('/', (req, res) => {
  try {
    const cfg = getConfig();
    const client = MCPClient.shared();
    const rows = client.describeCatalogue();
    const chain = getSearchConfig();

    const tools = rows.map(r => ({
      ...r,
      fields: (r.fields || []).map(f => withValue(cfg, f)),
      toggleValue: r.toggle ? valueAt(cfg, r.toggle) !== false : null,
      secrets: (r.secrets || []).map(s => ({ ...s, status: secrets.status(s.env) }))
    }));

    // Providers in the order they are actually tried, with the switched-off ones
    // after them — the page has to show what is off or there is no way back on.
    const known = chain.allProviders || [];
    const positionOf = (id) => {
      const i = chain.order.indexOf(id);
      return i === -1 ? null : i + 1;
    };
    const providers = SEARCH_PROVIDER_SPECS.map(spec => {
      const live = known.find(p => p.name === spec.id) || {};
      return {
        id: spec.id,
        label: spec.label,
        blurb: spec.blurb,
        toggle: spec.toggle,
        enabled: !!live.enabledInConfig,
        position: positionOf(spec.id),
        available: !!live.available,
        why: live.why || null,
        fields: (spec.fields || []).map(f => withValue(cfg, f)),
        secret: spec.secret
          ? { ...spec.secret, status: secrets.status(spec.secret.env) }
          : null
      };
    }).sort((a, b) => {
      if (a.position && b.position) return a.position - b.position;
      if (a.position) return -1;
      if (b.position) return 1;
      return a.label.localeCompare(b.label);
    });

    res.json({
      cards: CARDS,
      tools,
      registeredCount: client.getToolNames().length,
      catalogueCount: rows.length,
      search: {
        order: chain.order,
        any: chain.any,
        providers
      },
      // So the page can state plainly how secrets are held, rather than implying it.
      secretStore: {
        encrypted: true,
        algorithm: secrets.health().algorithm,
        keySource: secrets.health().keySource,
        stored: secrets.health().stored
      }
    });
  } catch (error) {
    console.error('[Tools] describe failed:', error.message);
    res.status(500).json({ error: 'Failed to describe tools' });
  }
});

/**
 * PUT /api/tools/secrets
 * Body: { secrets: { EXA_API_KEY: "value" | "" | null } }
 *
 * An empty string or null CLEARS that secret — that is what clearing the field in
 * the UI means, and a save that silently kept the old value would be a field that
 * cannot be emptied.
 *
 * Returns statuses only. Re-registers the tool set afterwards, because a key is
 * exactly the kind of change that makes a tool available: without it, search would
 * stay unregistered in this process until the next restart, and the page would say
 * "available" while the registry disagreed.
 */
router.put('/secrets', writeLimiter, (req, res) => {
  try {
    const body = req.body && req.body.secrets;
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      return res.status(400).json({ error: 'Body must be { secrets: { NAME: value } }' });
    }

    const allowed = declaredSecretNames();
    const results = {};
    const changed = [];

    for (const [name, value] of Object.entries(body)) {
      if (!allowed.has(name)) {
        return res.status(400).json({ error: `No tool declares a secret called "${name}".` });
      }
      if (value != null && typeof value !== 'string') {
        return res.status(400).json({ error: `The value for ${name} must be a string, or null to clear it.` });
      }
      if (String(value || '').length > 4096) {
        return res.status(400).json({ error: `The value for ${name} is too long.` });
      }
      // NOTHING IS LOGGED BUT THE NAME. db/secrets.js logs a length, never a value.
      results[name] = secrets.set(name, value);
      changed.push(name);
    }

    // A key can turn a tool on. Re-register so the registry, the page and the
    // model's tool list agree inside this process, without a restart.
    let registered = null;
    try {
      const client = MCPClient.shared();
      client.loadConfig();
      registered = client.getToolNames();
    } catch (e) {
      console.error('[Tools] re-registration after a secret change failed:', e.message);
    }

    // The capability manifest's web-search entry is true only while a provider can
    // actually be called, and a key is half of that — so the ledger records the
    // capability coming or going when it happens rather than at the next boot.
    try {
      const { added, removed } = require('../db/capability-manifest').syncToOps();
      if (added.length || removed.length) {
        console.log(`[Capabilities] secret change synced: +${added.length} -${removed.length}`);
      }
    } catch (e) {
      console.error('[Capabilities] syncToOps after a secret change failed:', e.message);
    }

    console.log(`[Tools] secrets updated: ${changed.join(', ')}`);
    res.json({ ok: true, secrets: results, registered });
  } catch (error) {
    console.error('[Tools] secret write failed:', error.message);
    res.status(500).json({ error: `Failed to save: ${error.message}` });
  }
});

module.exports = router;
