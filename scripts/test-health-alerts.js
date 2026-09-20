#!/usr/bin/env node
/**
 * THE SYSTEM REPORTING ON ITSELF IS NOT SCORED ON HOW INTERESTING IT IS.
 *
 * On 2026-08-27 the watchdog restarted the engine and queued its account of it
 * at priority 7 — the bar for reaching a greeting. The prioritiser re-scored it
 * to 1 ("trivial; probably not worth interrupting for") and it sat pending for
 * 25 hours. Athena could not say what had happened until the next day, by which
 * time the container logs had nearly rotated out.
 *
 * The scorer was not wrong by its own lights. Judged as conversational interest,
 * a container restart IS dull. It is not queued for its interest, and that is
 * the distinction a bare priority number cannot carry.
 *
 * The floor is enforced at updatePriority — the one place priority is written
 * after creation — rather than inside the prioritiser, for the same reason the
 * identity lock lives in fact-store rather than in each pipeline that calls it:
 * a guard inside one caller protects against one caller.
 *
 *   SNH_DATA_DIR=$(mktemp -d) node scripts/test-health-alerts.js
 */
const path = require('path');

if (!process.env.SNH_DATA_DIR) {
  console.error('Refusing to run against the live data directory.');
  console.error('Use: SNH_DATA_DIR=$(mktemp -d) node scripts/test-health-alerts.js');
  process.exit(1);
}

const ROOT = path.join(__dirname, '..');
let pass = 0, fail = 0;
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`); }
}

(async () => {
  const config = require(path.join(ROOT, 'db/config'));
  const database = require(path.join(ROOT, 'db/database'));
  database.initDatabase();

  console.log('\n1. A health item cannot be scored below the delivery bar\n');
  const initiatives = require(path.join(ROOT, 'db/initiatives'));
  const floor = initiatives.healthFloor();
  check('the floor is the greeting bar, not a second number that can drift from it',
    floor === (config.getConfig().initiative.greetingThreshold), String(floor));

  const healthId = await initiatives.addInitiative({
    type: 'alert', content: 'Engine restarted itself after a stall.',
    sourceKind: 'watchdog', sourceRef: `test-${Date.now()}`, priority: 7,
    healthClass: 'engine', dedupe: false
  });
  const plainId = await initiatives.addInitiative({
    type: 'observation', content: 'A mildly interesting thing was noticed.',
    sourceKind: 'test', sourceRef: `test-plain-${Date.now()}`, priority: 7, dedupe: false
  });
  check('the health item stored its class', initiatives.get(healthId).health_class === 'engine',
    String(initiatives.get(healthId).health_class));

  // This is the 8/27 failure, reproduced: the prioritiser scored it 1.
  initiatives.updatePriority(healthId, 1);
  check('a re-score to 1 leaves it AT the delivery bar',
    initiatives.get(healthId).priority === floor, String(initiatives.get(healthId).priority));

  initiatives.updatePriority(plainId, 1);
  check('…while an ordinary item is scored down as before — the floor is not global',
    initiatives.get(plainId).priority === 1, String(initiatives.get(plainId).priority));

  initiatives.updatePriority(healthId, 9);
  check('and a health item can still be raised ABOVE the floor',
    initiatives.get(healthId).priority === 9, String(initiatives.get(healthId).priority));

  const top = initiatives.getTopForGreeting({ greetingThreshold: floor, followupThreshold: 5 });
  check('so it is still eligible for a greeting after the pass that used to bury it',
    !!top, top ? top.id : 'nothing eligible');


  console.log('\n2. The alert says what happened, in fields a reader can check\n');
  const wd = require(path.join(ROOT, 'db/brain-watchdog'));
  const stalled = wd.recoveryAlertContent({
    unavailableMs: 342000, verdict: 'stalled', clock: '8:24 AM',
    engine: { reachable: true, running: 16, waiting: 10, generating: false }
  });
  check('the queue depth is in the sentence Ellie reads',
    stalled.includes('16 request(s) running, 10 queued'), stalled);
  check('…and the two words the 8/27 alert got wrong are gone',
    !/locked up|wedged/i.test(stalled), stalled);
  check('…and unavailability is a measured duration',
    /unavailable for 6 minutes/.test(stalled), stalled);

  const unreachable = wd.recoveryAlertContent({
    unavailableMs: 20000, verdict: 'unreachable', engine: null, clock: '8:24 AM'
  });
  check('an unreachable engine is described differently from a stalled one',
    unreachable !== stalled && /stopped answering entirely/.test(unreachable), unreachable);
  check('…and a short outage is reported in seconds rather than rounded to a minute',
    /unavailable for 20 seconds/.test(unreachable), unreachable);

  console.log(`\n=== ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})().catch(err => { console.error('Test harness crashed:', err); process.exit(1); });
