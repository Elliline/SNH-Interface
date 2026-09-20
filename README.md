# Squatch Neuro Hub (SNH)

Squatch Neuro Hub is a local AI with a persistent, self-authored memory and identity. The memory is the entity; the language model is just the voice. Each instance starts as an empty seed with no name or personality assigned, forms its identity through real conversation, and keeps it across sessions. It repairs its own memory, hands long work to background agents mid-conversation and leads with the result next time, and can run entirely on your own hardware. Part of the **Coastal Squatch AI** ecosystem by MettaSphere LLC.

## How it works

SNH is a Node/Express server with a vanilla-JS front end that talks to a language model you run yourself (Ollama, vLLM or llama.cpp) or to a hosted API (Claude, OpenAI, Grok). The memory layer is what sets it apart: after each exchange, facts are extracted from the conversation into a SQLite store, clustered by topic, indexed for hybrid retrieval (LanceDB vectors + SQLite FTS5 keyword search), and injected back into the system prompt on later turns. The entity's own observations about itself are stored the same way and become its identity. A corrector pass runs on a schedule to retire contradicted facts, corrections are written to a ledger, and the model has tools to inspect, search and repair its own memory.

Beyond chat, the server runs background jobs (scheduled cron jobs, long agent jobs, web research), routes tool calls through an MCP-style client, and can speak and listen through local TTS/STT services.

## Requirements

- Node.js 20 (the version it is developed and run on; `better-sqlite3` builds a native module on `npm install`)
- At least one model backend: an Ollama, vLLM or llama.cpp server on your network, or an API key for Claude, OpenAI or Grok
- Optional: a SearXNG instance or an Exa API key for web search; Kokoro (TTS) and a Whisper/Parakeet-compatible service (STT) for voice

## Quick start

```bash
git clone https://github.com/Elliline/SNH-Interface.git
cd SNH-Interface
npm install
cp .env.example .env    # PORT, HOST, OLLAMA_HOST and optional API keys
npm start               # http://localhost:3000
```

The server binds to `127.0.0.1` unless `HOST` is set in `.env`. Everything it stores lives under `data/` (created on first run, ignored by git): `data/chat.db` (SQLite), `data/lancedb/` (vectors), `data/config.json` (settings edited from the UI) and `data/secrets.json` (API keys, encrypted at rest). Set `SNH_DATA_DIR` to point the whole store somewhere else, for example at a throwaway directory for testing.

API keys can be entered from the Settings page in the browser instead of `.env`; they are stored write-only and never returned to the client.

## Layout

```
server.js               Express app: chat, providers, streaming, voice proxies
db/                     Memory, extraction, clustering, corrector, jobs, config
mcp/mcp-client.js       Tool-calling client
mcp/tools/              The tools the model can call (memory, search, jobs, ...)
routes/                 REST routes: conversations, memory, jobs, tools
public/                 Front end (vanilla JS, no build step)
scripts/                Tests, migrations and one-off maintenance scripts
deploy/snh.service      systemd user unit for running it as a service
```

## Tests

Test suites live in `scripts/test-*.js`. The runner compares each suite against a recorded baseline and uses a temporary data directory, so it never touches your live store.

Two things to know on a fresh clone:

- Run it with `--record`. The baseline's `verifiedAt` entries are commit hashes from the development history, which this repository does not carry, so without `--record` every suite is reported UNKNOWN. `--record` runs every suite and prints the measured failure counts next to the expected ones.
- Point the config at your model first. `data/config.json` is created with defaults on the first start (or edit it from the Settings page); the defaults expect Ollama at `http://localhost:11434`. Suites that call the model fail with connection errors until a provider there is reachable.

```bash
node scripts/run-suite.js --record                  # every suite
node scripts/run-suite.js --record test-cron-eval   # one suite
```

Suites that need a running SNH instance or make many model calls are skipped unless you opt in; see the header of `scripts/run-suite.js`.

## Running as a service

`deploy/snh.service` is a systemd user unit that runs `node server.js` from the checkout and reads `.env`. Fill in the placeholder paths before installing it.

## License

`package.json` declares MIT, but this repository does not yet contain a LICENSE file.
