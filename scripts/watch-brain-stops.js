#!/usr/bin/env node
/**
 * Who is stopping the brain?
 *
 * On 2026-08-16 sparky-brain bounced four times — 10:24:51, ~17:10, 17:59:53 and
 * 18:00:10 — every one with dockerd logging `hasBeenManuallyStopped=true` and a
 * SIGTERM→SIGKILL, which is the signature of an explicit `docker stop`/`restart`
 * rather than the restart policy or a crash. The SNH brain-watchdog logged
 * nothing on any of them, there is no cron entry, no systemd timer and no
 * autoheal container, so the cause was never attributed. Each bounce costs
 * ~3.5 minutes of model load, during which Aurelius answers "fetch failed".
 *
 * dockerd records THAT a container was stopped, never BY WHOM. This closes that
 * gap the only way available without auditd: `docker restart` holds its socket
 * open for the full 10-second grace period, so a process snapshot taken the
 * instant the kill event fires will still contain the command that asked for it.
 *
 * Writes one line per lifecycle event plus, on kill/die/stop, a snapshot of every
 * docker client process and every ellie-owned process whose command mentions
 * docker. `ellie` is the only member of the docker group, so the caller is
 * running as her — a shell, a script, or an agent session.
 *
 * READ-ONLY. It observes the docker event stream and reads /proc. It never
 * starts, stops or configures anything.
 *
 * Usage:
 *   node scripts/watch-brain-stops.js [--container sparky-brain] [--out FILE]
 *
 * Run it detached so it survives the shell:
 *   systemd-run --user --unit=brain-stop-watch \
 *     node /home/ellie/snh/scripts/watch-brain-stops.js
 *   journalctl --user -u brain-stop-watch -f      # watch it
 *   systemctl --user stop brain-stop-watch        # stop it
 */
const { spawn, execSync } = require('child_process');
const fs = require('fs');

const argv = process.argv.slice(2);
const argOf = (flag, dflt) => {
  const i = argv.indexOf(flag);
  return i > -1 ? argv[i + 1] : dflt;
};
const CONTAINER = argOf('--container', 'sparky-brain');
const OUT = argOf('--out', null);

function emit(line) {
  const stamped = `${new Date().toISOString()} ${line}`;
  console.log(stamped);
  if (OUT) { try { fs.appendFileSync(OUT, stamped + '\n'); } catch { /* non-fatal */ } }
}

/**
 * Everything that could be the caller, captured while `docker restart` is still
 * blocked on its grace period. lstart is included because a process that started
 * a second ago is a far better suspect than one that has been up for days.
 */
function snapshot(reason) {
  emit(`  ── process snapshot (${reason}) ──`);
  let out = '';
  try {
    out = execSync('ps -eo pid,ppid,user,lstart,args', { encoding: 'utf8', timeout: 5000 });
  } catch (e) {
    emit(`  (ps failed: ${e.message})`);
    return;
  }
  const hits = out.split('\n').filter(l =>
    /\bdocker\b/.test(l) && !/docker-proxy|dockerd|containerd|watch-brain-stops/.test(l)
  );
  if (!hits.length) {
    emit('  no docker client process visible — caller had already returned, or acted through the socket directly');
  }
  for (const h of hits) emit(`  SUSPECT ${h.trim().slice(0, 200)}`);

  // Anything holding the docker socket open right now, which a blocked
  // `docker restart` will be.
  try {
    const lsof = execSync('ls -l /proc/*/fd 2>/dev/null | grep -l docker.sock || true', { encoding: 'utf8', timeout: 5000, shell: '/bin/bash' });
    if (lsof.trim()) emit(`  socket holders: ${lsof.trim().slice(0, 300)}`);
  } catch { /* best effort */ }
}

emit(`watching docker events for "${CONTAINER}" (attach/detach, kill, die, stop, start)`);
emit(`only members of the docker group can do this without sudo: ${(() => {
  try { return execSync('getent group docker', { encoding: 'utf8' }).trim(); } catch { return 'unknown'; }
})()}`);

const ev = spawn('docker', [
  'events',
  '--filter', `container=${CONTAINER}`,
  '--format', '{{.Time}} {{.Action}} {{.Actor.Attributes.name}} {{.Actor.Attributes.signal}}',
]);

ev.stdout.on('data', chunk => {
  for (const line of String(chunk).split('\n')) {
    if (!line.trim()) continue;
    emit(`EVENT ${line.trim()}`);
    if (/\b(kill|die|stop)\b/.test(line)) snapshot(line.trim().split(' ')[1] || 'event');
  }
});
ev.stderr.on('data', d => emit(`docker events stderr: ${String(d).trim().slice(0, 200)}`));
ev.on('exit', code => { emit(`docker events exited (${code}) — watcher stopping`); process.exit(code || 0); });

process.on('SIGTERM', () => { emit('watcher stopped'); ev.kill(); process.exit(0); });
process.on('SIGINT', () => { emit('watcher stopped'); ev.kill(); process.exit(0); });
