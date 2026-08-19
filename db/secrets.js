/**
 * Secrets: encrypted at rest, write-only from the browser, env-overridable.
 *
 * WHY THIS EXISTS. Until 2026-08-18 the only way to give SNH an API key was to
 * hand-edit `.env` over SSH. That is not a setup step a person testing this will
 * do, and it makes a managed instance — where there is no shell — impossible to
 * configure at all. The browser has to be sufficient on its own.
 *
 * WHY NOT data/config.json. `GET /api/config` returns that file whole. A key in
 * it would be served to the browser on every settings load, which is the exact
 * opposite of write-only, and it would also ride along in anything that copies
 * config. So secrets get their own file, and nothing in the config path ever has
 * to remember to redact.
 *
 * WHERE IT LIVES, and why SNH_DATA_DIR does not move it. Beside data/config.json,
 * for exactly the reason config is not redirected either: a staging or replay run
 * is the SAME system pointed at a different store, not a differently configured
 * one. A throwaway instance that could not see the key would search differently
 * from live, and a verification run against it would then be measuring something
 * other than live behaviour.
 *
 * `SNH_SECRETS_PATH` and `SNH_SECRET_KEY_PATH` move the two files deliberately, for
 * a deployment that needs them elsewhere — a mounted volume on a managed instance —
 * and for tests, which set them before requiring this module so the live file and
 * the live key are never opened at all, not even for reading.
 *
 * THE ENCRYPTION, and what it is honestly worth. AES-256-GCM per secret, random
 * 12-byte IV, and the secret's NAME as additional authenticated data — so a
 * ciphertext cannot be copied into a different slot and decrypt. The key is 32
 * random bytes in `data/.secret-key` (mode 0600), created on first write, or
 * supplied by `SNH_SECRET_KEY` for a managed instance where the platform injects
 * it.
 *
 * What that protects against: a copied data directory, a backup, a synced
 * folder, an accidentally-committed file, someone reading the disk image. What it
 * does NOT protect against: anything running as this user, which can read the
 * key file next to the data. Fixing THAT means a passphrase typed at every boot
 * (no unattended restart, and systemd restarts this service) or an external KMS
 * (not self-hosted). Neither is worth what it costs here, so the honest statement
 * is: encrypted at rest, keyed to this machine.
 *
 * ENV STILL WINS. `process.env.EXA_API_KEY` overrides the stored value, so
 * everything that reads `.env` today keeps working unchanged — and `source` says
 * which one answered, because a stale `.env` silently beating a key just typed
 * into the UI is a debugging session nobody should have to have.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

/**
 * Beside data/config.json, NOT under getDataDir() — see the header. Along with
 * config.json these are the paths that deliberately ignore the SNH_DATA_DIR
 * redirect, and they ignore it for the same reason. The env overrides exist for
 * deployments and tests, not for ordinary operation.
 */
const SECRETS_PATH = process.env.SNH_SECRETS_PATH || path.join(__dirname, '../data/secrets.json');
const KEY_PATH = process.env.SNH_SECRET_KEY_PATH || path.join(__dirname, '../data/.secret-key');

const ALGO = 'aes-256-gcm';
const FILE_MODE = 0o600;

/** Parsed store + the mtime it was read at, so a write from elsewhere is noticed. */
let cache = null;
let cacheMtimeMs = 0;

// ---------------------------------------------------------------------------
// The key
// ---------------------------------------------------------------------------

/**
 * The 32-byte encryption key: from the environment if a platform injected one,
 * otherwise from the key file, which is created on first use.
 *
 * `SNH_SECRET_KEY` accepts base64 or hex. A malformed one THROWS rather than
 * falling back to the file: silently encrypting with a different key than the
 * operator supplied would make every stored secret unreadable the next time the
 * variable was set correctly, and it would look like corruption.
 */
function loadKey({ create = true } = {}) {
  const fromEnv = (process.env.SNH_SECRET_KEY || '').trim();
  if (fromEnv) {
    const buf = /^[0-9a-f]{64}$/i.test(fromEnv)
      ? Buffer.from(fromEnv, 'hex')
      : Buffer.from(fromEnv, 'base64');
    if (buf.length !== 32) {
      throw new Error('SNH_SECRET_KEY must be 32 bytes, as base64 or hex');
    }
    return buf;
  }

  try {
    if (fs.existsSync(KEY_PATH)) {
      const buf = Buffer.from(fs.readFileSync(KEY_PATH, 'utf8').trim(), 'base64');
      if (buf.length === 32) return buf;
      throw new Error(`${KEY_PATH} does not contain a 32-byte key`);
    }
  } catch (err) {
    // A key file that exists but cannot be read is NOT a reason to mint a new
    // one — that would abandon every secret already encrypted with it.
    if (err && /does not contain/.test(err.message)) throw err;
    throw new Error(`could not read the secret key at ${KEY_PATH}: ${err.message}`);
  }

  if (!create) return null;

  const key = crypto.randomBytes(32);
  const dir = path.dirname(KEY_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(KEY_PATH, key.toString('base64'), { encoding: 'utf8', mode: FILE_MODE });
  try { fs.chmodSync(KEY_PATH, FILE_MODE); } catch { /* best effort on odd filesystems */ }
  console.log(`[Secrets] created a new encryption key at ${KEY_PATH} (mode 600). Back it up with the secrets file or neither is any use.`);
  return key;
}

/** Is a key available at all, without creating one? Used by the status endpoint. */
function hasKey() {
  try { return !!loadKey({ create: false }); } catch { return false; }
}

/** Where the key came from — reported to the UI, never the key itself. */
function keySource() {
  if ((process.env.SNH_SECRET_KEY || '').trim()) return 'env';
  try { return fs.existsSync(KEY_PATH) ? 'file' : 'none'; } catch { return 'none'; }
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

function emptyStore() { return { version: 1, secrets: {} }; }

function readStore() {
  try {
    if (!fs.existsSync(SECRETS_PATH)) { cache = emptyStore(); cacheMtimeMs = 0; return cache; }
    const st = fs.statSync(SECRETS_PATH);
    if (cache && st.mtimeMs === cacheMtimeMs) return cache;
    const parsed = JSON.parse(fs.readFileSync(SECRETS_PATH, 'utf8'));
    cache = (parsed && typeof parsed === 'object' && parsed.secrets) ? parsed : emptyStore();
    cacheMtimeMs = st.mtimeMs;
    return cache;
  } catch (err) {
    // A corrupt store must not take the process down, and must not silently read
    // as "no secrets set" either — that would look like the key was never saved.
    console.error(`[Secrets] could not read ${SECRETS_PATH}: ${err.message}`);
    return emptyStore();
  }
}

function writeStore(store) {
  const dir = path.dirname(SECRETS_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const tmp = `${SECRETS_PATH}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(store, null, 2), { encoding: 'utf8', mode: FILE_MODE });
  fs.renameSync(tmp, SECRETS_PATH);   // atomic: a half-written store loses every key
  try { fs.chmodSync(SECRETS_PATH, FILE_MODE); } catch { /* best effort */ }
  cache = store;
  try { cacheMtimeMs = fs.statSync(SECRETS_PATH).mtimeMs; } catch { cacheMtimeMs = 0; }
}

// ---------------------------------------------------------------------------
// Encrypt / decrypt
// ---------------------------------------------------------------------------

function encrypt(name, plaintext) {
  const key = loadKey();
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv(ALGO, key, iv);
  // The NAME is authenticated data: a blob lifted from one slot and pasted into
  // another fails to decrypt rather than quietly becoming that other secret.
  cipher.setAAD(Buffer.from(name, 'utf8'));
  const data = Buffer.concat([cipher.update(String(plaintext), 'utf8'), cipher.final()]);
  return {
    cipher: ALGO,
    iv: iv.toString('base64'),
    tag: cipher.getAuthTag().toString('base64'),
    data: data.toString('base64'),
    updatedAt: new Date().toISOString()
  };
}

function decrypt(name, entry) {
  if (!entry || !entry.data) return null;
  const key = loadKey({ create: false });
  if (!key) throw new Error('no encryption key is available');
  const decipher = crypto.createDecipheriv(ALGO, key, Buffer.from(entry.iv, 'base64'));
  decipher.setAAD(Buffer.from(name, 'utf8'));
  decipher.setAuthTag(Buffer.from(entry.tag, 'base64'));
  return Buffer.concat([
    decipher.update(Buffer.from(entry.data, 'base64')),
    decipher.final()
  ]).toString('utf8');
}

// ---------------------------------------------------------------------------
// The API everything else uses
// ---------------------------------------------------------------------------

/**
 * Resolve a secret, environment first.
 *
 * @param {string} name - the env-style name, e.g. 'EXA_API_KEY'
 * @returns {{value: string|null, source: 'env'|'store'|null, error: string|null}}
 */
function get(name) {
  const fromEnv = (process.env[name] || '').trim();
  if (fromEnv) return { value: fromEnv, source: 'env', error: null };

  const entry = readStore().secrets[name];
  if (!entry) return { value: null, source: null, error: null };

  try {
    const value = decrypt(name, entry);
    return value ? { value, source: 'store', error: null } : { value: null, source: null, error: null };
  } catch (err) {
    // Loud, and WITHOUT the ciphertext or the key in the message. The usual cause
    // is a key file replaced or lost, and that is worth saying in those words.
    const error = `the stored ${name} could not be decrypted (${err.message}). If the key file was replaced or lost, the secret has to be entered again.`;
    console.error(`[Secrets] ${error}`);
    return { value: null, source: null, error };
  }
}

/** Just the value, for callers that only need the string. */
function resolve(name) { return get(name).value; }

/**
 * What the UI is allowed to know: THAT it is set, where from, and when.
 *
 * No preview, no last-four, no length. "Masked after save, never sent to the
 * browser again" is easiest to keep true when the server never puts any part of
 * the value in a response at all.
 */
function status(name) {
  const fromEnv = (process.env[name] || '').trim();
  const entry = readStore().secrets[name];
  const g = entry ? get(name) : { error: null };
  return {
    name,
    set: !!fromEnv || !!entry,
    source: fromEnv ? 'env' : (entry ? 'store' : null),
    // Both halves matter to the reader: a key typed into the UI while .env holds
    // an old one is stored and IGNORED, and the UI has to be able to say so.
    storedToo: !!entry,
    envOverrides: !!fromEnv && !!entry,
    updatedAt: entry ? entry.updatedAt || null : null,
    error: g.error || null
  };
}

/**
 * Store a secret, encrypted. Returns its status — never the value.
 * An empty or whitespace-only value is a DELETE, because that is what clearing
 * the field in the UI means.
 */
function set(name, value) {
  if (!/^[A-Z][A-Z0-9_]{2,63}$/.test(String(name || ''))) {
    throw new Error('a secret name must be an env-style name: A-Z, 0-9 and underscores');
  }
  const v = String(value == null ? '' : value).trim();
  const store = { ...readStore(), secrets: { ...readStore().secrets } };

  if (!v) {
    delete store.secrets[name];
    writeStore(store);
    console.log(`[Secrets] cleared ${name}`);
    return status(name);
  }

  store.secrets[name] = encrypt(name, v);
  writeStore(store);
  // Length is logged, never the value: "did my paste land" is answerable without
  // the key appearing in a log file that gets shipped around.
  console.log(`[Secrets] stored ${name} (${v.length} chars, encrypted with ${ALGO})`);
  return status(name);
}

/** Names currently in the store — names only. */
function names() { return Object.keys(readStore().secrets); }

/** Ops view for the boot banner and the settings page. Nothing sensitive. */
function health() {
  return {
    path: SECRETS_PATH,
    keySource: keySource(),
    hasKey: hasKey(),
    algorithm: ALGO,
    stored: names().length
  };
}

module.exports = { get, resolve, status, set, names, health, hasKey, keySource, SECRETS_PATH, KEY_PATH };
