/**
 * Centralized configuration loader for Squatch Neuro Hub
 * Reads from data/config.json, deep-merges with defaults,
 * and applies environment variable overrides at runtime.
 */

const fs = require('fs');
const path = require('path');

const CONFIG_PATH = path.join(__dirname, '../data/config.json');

const DEFAULTS = {
  providers: {
    ollama: [
      { name: 'Local', host: 'http://localhost:11434' }
    ],
    vllm: [],
    llamacpp: [
      { name: 'Local', host: 'http://localhost:8080', model: 'scout' }
    ]
  },
  models: {
    chat: { provider: 'llamacpp', instance: 'Local', model: 'scout' },
    extraction: { provider: 'ollama', instance: 'Local', model: 'gemma3:4b' },
    heartbeat: { provider: 'ollama', instance: 'Local', model: 'qwen3:14b' },
    embedding: { provider: 'ollama', instance: 'Local', model: 'nomic-embed-text' }
  },
  heartbeat: { enabled: true, intervalHours: 2, warmupMinutes: 5 },
  memory: {
    similarityThreshold: 0.55,
    clusterLinkThreshold: 0.50,
    maxFactsPerCluster: 10,
    dailyLogRetentionDays: 7,
    hybridSearchWeights: { vector: 0.6, bm25: 0.4 }
  },
  tools: {
    searxng: { enabled: false }
  }
};

let currentConfig = null;

/**
 * Recursively deep-merge source into target.
 * Objects merge, primitives and arrays replace.
 */
const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function deepMerge(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (UNSAFE_KEYS.has(key)) continue;
    if (
      source[key] &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      target[key] &&
      typeof target[key] === 'object' &&
      !Array.isArray(target[key])
    ) {
      result[key] = deepMerge(target[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

/**
 * Migrate old single-host config format to new array-based instance format.
 * Called before deepMerge so the file data is in the right shape.
 */
function migrateConfig(fileConfig) {
  if (!fileConfig.providers) return fileConfig;

  const p = fileConfig.providers;

  // Migrate ollama: { host: '...' } → ollama: [{ name: 'Local', host: '...' }]
  if (p.ollama && !Array.isArray(p.ollama) && p.ollama.host) {
    p.ollama = [{ name: 'Local', host: p.ollama.host }];
  }

  // Migrate llamacpp: { host: '...' } → llamacpp: [{ name: 'Local', host: '...', model: '...' }]
  if (p.llamacpp && !Array.isArray(p.llamacpp) && p.llamacpp.host) {
    const chatModel = fileConfig.models?.chat?.model || 'scout';
    p.llamacpp = [{ name: 'Local', host: p.llamacpp.host, model: chatModel }];
  }

  // Migrate vllm: { host: '...' } → vllm: [{ name: 'Local', host: '...', model: '...' }]
  if (p.vllm && !Array.isArray(p.vllm) && p.vllm.host) {
    p.vllm = [{ name: 'Local', host: p.vllm.host, model: p.vllm.model || '' }];
  }
  // Ensure vllm array exists
  if (!p.vllm) p.vllm = [];

  // Migrate model role assignments to include instance: 'Local'
  if (fileConfig.models) {
    for (const role of ['chat', 'extraction', 'heartbeat', 'embedding']) {
      if (fileConfig.models[role] && !fileConfig.models[role].instance) {
        fileConfig.models[role].instance = 'Local';
      }
    }
  }

  return fileConfig;
}

/**
 * Load config from disk, deep-merge with defaults.
 * Auto-creates config file if missing.
 */
function loadConfig() {
  let fileConfig = {};

  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
      fileConfig = JSON.parse(raw);
    } else {
      // Auto-create with defaults
      const dir = path.dirname(CONFIG_PATH);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(CONFIG_PATH, JSON.stringify(DEFAULTS, null, 2), 'utf8');
      console.log('[Config] Created default config at', CONFIG_PATH);
    }
  } catch (err) {
    console.error('[Config] Error reading config file:', err.message);
  }

  fileConfig = migrateConfig(fileConfig);
  currentConfig = deepMerge(DEFAULTS, fileConfig);
  return currentConfig;
}

/**
 * Get the current config with env var overrides applied.
 * Env vars OLLAMA_HOST and LLAMACPP_HOST update the 'Local' instance host,
 * or prepend a new 'Local' instance if none exists.
 */
function getConfig() {
  if (!currentConfig) {
    loadConfig();
  }

  // Deep clone to avoid env overrides mutating the cached config
  const config = JSON.parse(JSON.stringify(currentConfig));

  if (process.env.OLLAMA_HOST) {
    if (!Array.isArray(config.providers.ollama)) config.providers.ollama = [];
    const local = config.providers.ollama.find(i => i.name === 'Local');
    if (local) {
      local.host = process.env.OLLAMA_HOST;
    } else {
      config.providers.ollama.unshift({ name: 'Local', host: process.env.OLLAMA_HOST });
    }
  }

  if (process.env.LLAMACPP_HOST) {
    if (!Array.isArray(config.providers.llamacpp)) config.providers.llamacpp = [];
    const local = config.providers.llamacpp.find(i => i.name === 'Local');
    if (local) {
      local.host = process.env.LLAMACPP_HOST;
    } else {
      config.providers.llamacpp.unshift({ name: 'Local', host: process.env.LLAMACPP_HOST, model: 'scout' });
    }
  }

  return config;
}

/**
 * Deep-merge a partial update into the current config and persist to disk.
 * @param {Object} partial - Partial config to merge
 * @returns {Object} Updated config
 */
function updateConfig(partial) {
  if (!currentConfig) {
    loadConfig();
  }

  currentConfig = deepMerge(currentConfig, partial);

  try {
    const dir = path.dirname(CONFIG_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(currentConfig, null, 2), 'utf8');
    console.log('[Config] Saved config to', CONFIG_PATH);
  } catch (err) {
    console.error('[Config] Error writing config file:', err.message);
  }

  return getConfig();
}

/**
 * Look up a provider instance by type and name.
 * Returns { name, host, model? } or null.
 */
function getProviderInstance(providerType, instanceName) {
  const config = getConfig();
  const instances = config.providers[providerType];
  if (!Array.isArray(instances)) return null;
  return instances.find(i => i.name === instanceName) || null;
}

module.exports = { getConfig, updateConfig, loadConfig, getProviderInstance };
