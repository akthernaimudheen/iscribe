# iScribe Production Trial Runbook

Operator procedures for the controlled real-world consultation trial on the
single Windows host. No secrets appear in this document: credentials live only
in the host `.env` (never printed, never committed) and in the operator's
password manager.

## Topology

| Component | Where | Notes |
|---|---|---|
| App (uvicorn, single worker) | `127.0.0.1:8123`, native process (no Docker) | `python -m service` via `start-service.cmd` |
| Malayalam sidecar | `127.0.0.1:8131` | optional; separate launcher |
| Named tunnel | `iscribe-trial` → `scribe.prompttoshort.online` | `start-named-tunnel.cmd`; config `C:\Users\akthe\.cloudflared\config.yml` |
| Watchdog | runs forever, 60 s cycle | launched at logon via Startup `iscribe-watchdog.vbs` |
| Database | `<repo>\data\iscribe.db` | one SQLite file: users, consultations, notes, transcripts, jobs, audit |
| Audio uploads | `<repo>\data\uploads\` | staged `.part` → final; retention sweep deletes per policy |
| App logs | `<repo>\data\logs\iscribe.log` (+ watchdog/proc logs) | `C:\Users\akthe\.iscribe-logs\` |
| Backups | `<repo>\iscribe-backups\` | timestamped, verified, rolling 14; gitignored; OneDrive-synced |

## A. Start application

```bat
C:\Users\akthe\.iscribe-logs\start-service.cmd
```
Runs in the foreground of that window; for normal operation the watchdog
starts it hidden. Verify: `curl http://127.0.0.1:8123/api/health` → `"status":"ok"`.

## B. Stop application

```powershell
Get-NetTCPConnection -LocalPort 8123 -State Listen | Select-Object -First 1 -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```
Warning: the watchdog restarts the service within 60 s. To keep it stopped,
stop the watchdog first (see L).

## C. Check health (process alive)

```powershell
Invoke-RestMethod http://127.0.0.1:8123/api/health
```
Healthy response: `status=ok`, `engine_build=engine-<git-sha>`. No secrets are
returned (covered by regression tests).

## D. Check readiness (transcription possible)

```powershell
Invoke-RestMethod http://127.0.0.1:8123/api/ready
```
`status=ready` + HTTP 200 means models/provider are usable. HTTP 503 with
`status=loading` during warmup is normal for ~1 min after start.

## E. Check Cloudflare tunnel

```powershell
Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | Select-Object ProcessId, CommandLine
Invoke-RestMethod https://scribe.prompttoshort.online/api/health
```
If no `cloudflared` process: `C:\Users\akthe\.iscribe-logs\start-named-tunnel.cmd`.
The watchdog restarts the tunnel automatically within ~75 s if the process dies.

## F. Create backup

```bat
cd /d "C:\Users\akthe\OneDrive\Documents\New folder"
.venv\Scripts\python.exe scripts\backup_db.py
```
Online (no downtime), integrity-checked, pruned to 14 copies, writes a
`.manifest.txt` beside each backup. Exit code 0 + `backup ok:` = success; any
failure also leaves `LAST_BACKUP_FAILED.txt` in the repo root, which the
watchdog logs. The watchdog also takes a backup automatically every 24 h.

## G. Verify backup

```bat
.venv\Scripts\python.exe scripts\restore_db.py --verify iscribe-backups\<file>.db
```
`RESULT: PASS` requires integrity ok. Check `newest_updated_at` is recent and
row counts look sane (no content is printed).

## H. Restore backup safely

**Never restore over the live database without stopping the app first.**

1. Stop the service (section B) and the watchdog (section L).
2. Verify the chosen backup (section G).
3. Remove stale sidecars and copy the backup over the live file:
   ```bat
   del "C:\Users\akthe\OneDrive\Documents\New folder\data\iscribe.db-wal"
   del "C:\Users\akthe\OneDrive\Documents\New folder\data\iscribe.db-shm"
   copy /Y iscribe-backups\<file>.db "C:\Users\akthe\OneDrive\Documents\New folder\data\iscribe.db"
   ```
4. Start the service (section A) and re-run the restore drill to confirm:
   `python scripts\restore_drill.py` → `DRILL RESULT: PASS`.

The drill script (`scripts\restore_drill.py`) performs exactly this procedure
against a THROWAWAY copy — run it any time to prove restorability without
touching production.

## I. Rollback: recover from a bad Git deployment

Deployment = the watchdog sees `git HEAD` change → restarts the service.
Before any deploy: run `python scripts\predeploy_check.py` (backup + tests).

To roll back code (the database is never touched — `data/` is untracked):

```bat
cd /d "C:\Users\akthe\OneDrive\Documents\New folder"
git log --oneline -5
git reset --hard <last-known-good-sha>
```
The watchdog restarts onto the old build within 60 s. If the new build fails
to even start, the watchdog keeps restarting it — apply the reset above, then
`git push -f` only if the bad commit reached the shared remote and no one else
has built on it. A restore point from *before* the deploy exists because
`predeploy_check.py` backs up first.

## J. Check logs

| Log | Path | Look for |
|---|---|---|
| App events | `<repo>\data\logs\iscribe.log` | `event=` lines, errors |
| Proc stdout/stderr | `C:\Users\akthe\.iscribe-logs\iscribe-prod.log`, `iscribe.err.log` | tracebacks |
| Watchdog | `C:\Users\akthe\.iscribe-logs\watchdog.log` | restarts, backup results |
| Tunnel | `C:\Users\akthe\.iscribe-logs\cloudflared-named.log` | connection errors |

All app logs pass the PHI redaction filter and the access-log query scrubber.

## K. Recover after Windows reboot

Logon runs `iscribe-watchdog.vbs` (Startup folder) → watchdog starts service,
sidecar and tunnel within one cycle. If it did not start:
`powershell -File C:\Users\akthe\.iscribe-logs\watchdog.ps1` (or run section A/E
manually). Check `C:\Users\akthe\...\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
still contains the .vbs.

## L. Recover after application crash

The watchdog detects the closed port within 60 s and restarts the service.
Interrupted documentation jobs are auto-marked failed at startup (retry from
the dashboard — no data is lost; consultations persist). If the watchdog
itself is dead: `powershell -File C:\Users\akthe\.iscribe-logs\watchdog.ps1`.

To stop the watchdog deliberately (e.g. for maintenance):
```powershell
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'watchdog\.ps1' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## M. Disable access if a security incident occurs

Fastest lever first:

1. **Stop the tunnel** (cuts all public access, keeps the app local):
   ```powershell
   Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
   ```
   The watchdog will restart it — stop the watchdog first if the outage
   should persist.
2. **Disable a user account** (admin required):
   `PATCH /api/users/<email>` with `{"disabled": true}` — sessions die on
   their next request (12 h max cookie life).
3. **Rotate the break-glass key**: replace `ISCRIBE_ACCESS_TOKEN` in `.env`
   and restart the service (invalidates all sessions + download tokens).
4. **Freeze evidence**: copy `data\logs\iscribe.log` and
   `C:\Users\akthe\.iscribe-logs\watchdog.log` aside before restarting anything.
5. **Worst case — stop everything**: stop watchdog (L), then service (B),
   then tunnel (above). Data remains on disk; restore later per section H.

## Data persistence summary (what survives what)

| Event | Consultations/notes/users | Audio | In-flight job |
|---|---|---|---|
| App restart / crash / watchdog restart | survive (SQLite WAL) | survive | marked failed, retryable |
| Windows reboot | survive | survive | marked failed at startup |
| Tunnel restart | n/a (no data path) | n/a | n/a |
| Git deploy (watchdog restart) | survive (data/ untracked) | survive | marked failed, retryable |
| Repo `git reset --hard` | survive | survive | n/a |
