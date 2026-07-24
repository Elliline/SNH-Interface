# SNH — Squatch Neuro Hub

A self-hosted AI assistant with a persistent, self-authored memory and identity.
Node.js + Express (`server.js`), SQLite (`better-sqlite3`) + LanceDB (vectors),
vanilla-JS frontend in `public/`. Data-access + cognition modules live in `db/`.
Config is `data/config.json` merged over `DEFAULTS` in `db/config.js`.

## ⚠️ Maintenance rule: shipping a capability = manifest entry + introduction

SNH keeps a **capability manifest** — `db/capability-manifest.js` — a machine-true
registry of what it can actually do. A compact form is injected into every chat so
it answers "what can you do / do you have a way to X" from ground truth instead of
guessing. This exists because features kept getting built into SNH without ever
becoming part of its self-knowledge (2026-07-23: it proposed building a feature it
had shipped the day before).

**When you ship a new capability, you MUST, as part of shipping:**

1. **Add its manifest entry** to `db/capability-manifest.js` — `id`, `name`,
   `description` (full, 1–2 plain sentences), `oneLiner` (the tight compact form
   that gets injected — mind the injection diet), `intro` (first-person clause for
   the ship-day self-fact), `schedule`, `dateAdded`.
   - **Never claim more than is built.** Over-claiming is the exact failure the
     manifest exists to prevent. Derive the description from what the code does.
     Leave out disabled/aspirational features (e.g. web search stays out while
     `config.tools.searxng.enabled` defaults to false).
   - Keep entries **dry and accurate** — facts about the code, not personality.
2. **Introduce it to SNH** so it learns it (self-fact through reflection):
   `node scripts/introduce-capability.js <id>`. This stores a first-person
   DECLARATION via the normal self-fact pipeline. (Do NOT bulk-introduce; one
   capability, on its ship day.)
3. **Regenerate the briefing** if the manifest changed:
   `node scripts/write-capability-briefing.js` → `capability-briefing.md`.

Manifest changes are logged to the ops ledger automatically on boot (`syncToOps`).

## Two layers of self-knowledge

- **Manifest = machine truth** (code). Queryable at conversation time (injected
  compact form + `GET /api/memory/capabilities` for full descriptions on demand).
- **Introductions = the entity's self-facts** (formed through reflection). Facts
  about what's built are **declarations**, not auditable claims — the existing
  claim/declaration classifier tags them accordingly, and identity injection still
  excludes audit "dissonance" records.

## Conventions worth knowing

- **Plain-language norm:** bell/initiative/audit notes and capability descriptions
  are one or two sentences in everyday words, saying plainly what's true or wanted.
- **Never auto-revise identity:** the self-coherence audit documents tension and
  raises it for the human — it never edits a self-fact. Same philosophy across the
  memory tools: supersede/move never delete, everything logged, big changes need
  sign-off.
- **Ops ledger vs daily log:** operational telemetry → `data/memory/ops/` (Thinking
  tab, never injected into chat). Cognitively meaningful entries → `data/memory/daily/`.
- **Background LLM work** goes through the agent pool (`db/agent-pool.js`), which
  throttles to yield to live chat.
- The server runs as the systemd **user** service `snh.service`
  (`systemctl --user restart snh.service`); port 3000.
