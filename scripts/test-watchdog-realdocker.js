/**
 * Real-docker integration test: proves the watchdog's `docker restart` actually
 * executes from node with the service user's real permissions. Targets a
 * throwaway container (NOT the real brain). Only config + initiatives + opsLog
 * are stubbed; child_process.execFile is REAL.
 *   node scripts/test-watchdog-realdocker.js <container-name>
 */
const CONTAINER = process.argv[2] || 'watchdog-selftest';

const config = require('../db/config');
config.getConfig = () => ({ watchdog: { enabled: true, container: CONTAINER, failureThreshold: 3, cooldownMinutes: 5, maxRestartsPerHour: 2 } });
require('../db/initiatives').addInitiative = async () => 'stub';
require('../db/fact-extractor').appendToOpsLog = (m) => console.log('  [ops] ' + m);

const wd = require('../db/brain-watchdog');

(async () => {
  console.log(`Driving 3 failing probes → real docker restart of "${CONTAINER}"...`);
  await wd.onProbeResult({ ok: false, ms: 8000, error: 'timeout after 8000ms' });
  await wd.onProbeResult({ ok: false, ms: 8000, error: 'timeout after 8000ms' });
  await wd.onProbeResult({ ok: false, ms: 8000, error: 'timeout after 8000ms' }); // fires real restart
  // Give the async execFile callback time to complete.
  await new Promise(r => setTimeout(r, 4000));
  console.log('state:', JSON.stringify(wd._getState()));
  process.exit(0);
})();
