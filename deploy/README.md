# deploy/ — records of what is actually running

These are COPIES. The live instances are the deployed ones, at the paths named
below; editing a file here changes nothing until it is copied out. They live in
git because all of them were edited during the 2026-08-23/24 engine swap and
none of them is inside this repo, so the changes were otherwise unversioned.

| file | live path | notes |
|---|---|---|
| `snh-backup` | `/usr/local/bin/snh-backup` | nightly NAS backup |
| `snh-backup.conf` | `/etc/snh-backup.conf` | paths only — no credentials |
| `snh-backup.service` / `.timer` | `/etc/systemd/system/` | 03:15 nightly |
| `snh.service` | `~/.config/systemd/user/` | points at `~/snh-prod` |
| `serve-command-baseline.json` | `~/serve-command-baseline.json` | vLLM serve cmd, Qwen3.8-27B NVFP4 |
| `gemma-serve-command-baseline.json` | `~/gemma-serve-command-baseline.json` | the retired Gemma command, kept for rollback |
| `qwen-engine-versions.txt` | `~/qwen-engine-versions.txt` | container/driver/model hashes |
| `config.prod.json` | `~/snh-prod/data/config.json` | **record only** — `data/` is gitignored |

`config.prod.json` is the one that needs care: the live file is under `data/`,
which is excluded from git on purpose, so this copy exists to make the thinking
budgets and retuned timeouts reviewable. It carries no credentials (those are in
`data/secrets.json`, which is never copied here). If you change the live config,
this copy goes stale — it is a record, not a source.
