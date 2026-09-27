# iScribe — Deployment Guide

Operational guide for the 30-day clinical demonstration trial
(`https://scribe.prompttoshort.online`). Everything here has been run and
measured on the trial host; the numbers in [Performance](#performance) are
actual, not estimates.

---

## Architecture

```
  Clinician's browser
        │  HTTPS
        ▼
  Cloudflare Tunnel  ──── outbound-only connection, no inbound firewall rule
        │                  (DNS: scribe.prompttoshort.online → tunnel)
        │  HTTP (loopback)
        ▼
  ┌──────────────────────────────────────────────┐
  │  iScribe (single process, single host)       │
  │                                              │
  │   FastAPI ── static UI (same origin)         │
  │      │                                       │
  │      ├─ accounts (email/password, roles,     │
  │      │   clinic isolation) + break-glass key │
  │      │                                       │
  │      └─ ScribeEngine                         │
  │            ├─ faster-whisper  (local CPU)    │
  │            ├─ speaker labelling              │
  │            └─ spaCy clinical extraction      │
  │                                              │
  │   SQLite  ·  uploaded audio  ·  logs         │
  │   (all under ISCRIBE_DATA_DIR)               │
  └──────────────────────────────────────────────┘
```

**No audio, transcript, note or prescription ever leaves this host.** All
inference is local and open-source. No Deepgram, OpenAI, Anthropic or other
paid AI API is contacted at runtime — the only outbound connection is the
Cloudflare Tunnel carrying the clinician's HTTPS session, and (unless
`HF_HUB_OFFLINE=1`) a model-metadata check to huggingface.co at startup.

---

## Install

One-time, on the host:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Requires **ffmpeg/ffprobe on PATH** (`winget install Gyan.FFmpeg`). The Whisper
weights download once on first start and are cached under `~/.cache/huggingface`;
set `HF_HUB_OFFLINE=1` afterwards so startup never waits on the network.

---

## Configuration

Copy `.env.example` to `.env` and fill it in. **`.env` is git-ignored and must
never be committed.** Variable names only — values live solely in `.env`:

| Variable | Purpose |
|---|---|
| `ISCRIBE_ENV` | `production` or `development`. Production requires an access key. |
| `ISCRIBE_HOST` | Bind address. Keep `127.0.0.1` when fronted by the tunnel. |
| `ISCRIBE_PORT` | Bind port. Default `8123`. |
| `ISCRIBE_ACCESS_TOKEN` | **Secret.** Shared key clinicians sign in with. Required in production. |
| `ISCRIBE_SESSION_HOURS` | Session lifetime before re-sign-in. |
| `ISCRIBE_COOKIE_SECURE` | `true` when served over HTTPS (the tunnel). |
| `ISCRIBE_CORS_ORIGINS` | Empty = same-origin only. Leave empty for this deployment. |
| `ISCRIBE_DATA_DIR` | Database, uploaded audio and logs. **Back this up.** |
| `ISCRIBE_UPLOAD_DIR` | Override audio location (defaults inside the data dir). |
| `ISCRIBE_DB_PATH` | Override database location. |
| `ISCRIBE_LOG_DIR` | Override log location. |
| `ISCRIBE_MAX_UPLOAD_MB` | Upload size cap. |
| `ISCRIBE_AUDIO_RETENTION_HOURS` | Audio is deleted this long after the consultation. `0` disables. |
| `ISCRIBE_MAX_CONCURRENT_JOBS` | Simultaneous transcriptions. Keep at `1` below 8 GB RAM. |
| `ISCRIBE_STT_PROVIDER` | Production default `current_faster_whisper`. |
| `ISCRIBE_BOOTSTRAP_ADMIN_CODE` | **Secret.** One-time code to create the first ADMIN account. |
| `ISCRIBE_AUTH_RATE_LIMIT` / `_WINDOW_SECONDS` | Login attempts per IP per window (default 5 / 300s). |
| `ISCRIBE_PROCESSING_RATE_LIMIT` / `_WINDOW_SECONDS` | Job submissions per user per window (default 20 / 3600s). |
| `ISCRIBE_DEMO_MODE` | Show the guided demo walkthrough banner. |
| `REAL_CONSULTATION_TRIAL` | Trial-mode banners for demonstrations on real recordings. |
| `ISCRIBE_WHISPER_MODEL` | `tiny` / `base` / `small` / `medium`. |
| `ISCRIBE_WHISPER_DEVICE` | `cpu` or `cuda`. |
| `ISCRIBE_WHISPER_COMPUTE_TYPE` | `int8` on CPU. |
| `ISCRIBE_WHISPER_CPU_THREADS` | Thread cap so transcription cannot starve the web server. |
| `ISCRIBE_PRELOAD_MODELS` | Warm models at startup instead of on first request. |
| `ISCRIBE_LOG_LEVEL` | `INFO` normally. |
| `HF_HUB_OFFLINE` | `1` after the first model download. |

Generate or rotate the access key (writes to `.env`, prints once):

```bash
powershell -File scripts/new-access-key.ps1
```

---

## Start

```bash
powershell -File scripts/start-iscribe.ps1
```

Or directly (reads `.env` the same way):

```bash
.venv/Scripts/python.exe -m service
```

Then expose it over HTTPS:

```bash
powershell -File scripts/start-tunnel.ps1 -Quick
```

`-Quick` gives an ephemeral `*.trycloudflare.com` URL that changes on every
restart — fine for same-day testing. For the real trial use a named tunnel with
a stable hostname:

```bash
powershell -File scripts/start-tunnel.ps1 -Named iscribe-trial
```

### Container deployment (trial path)

The image bakes in the models, so it starts fully offline:

```bash
docker compose up -d --build
```

---

## Public URL: scribe.prompttoshort.online

One-time setup (Cloudflare account holding `prompttoshort.online`):

```bash
cloudflared tunnel login
cloudflared tunnel create iscribe-trial
cloudflared tunnel route dns iscribe-trial scribe.prompttoshort.online
```

`scripts/start-tunnel.ps1 -Named iscribe-trial` then serves the stable URL.
HTTPS is terminated by Cloudflare; the app sits behind the tunnel on
loopback only (`docker-compose.yml` binds `127.0.0.1:8123`), so no inbound
port is ever open on the host.

If the subdomain cannot be routed yet, the temporary fallback is
`https://prompttoshort.online/scribe` via a Cloudflare Workers/redirect rule
pointing at the same tunnel — the app itself needs no changes.

---

## CI/CD: Git is the source of truth

```
git push → GitHub Actions CI (tests + PHI/secret guard + docker build)
         → deploy (docker compose pull/build on the host)
         → health check (/api/health + /api/ready)
```

- **CI** (`.github/workflows/ci.yml`): full test suite on Python 3.12, a
  PHI/secret guard that fails the build on real-consultation content
  markers or tracked secrets/binaries, and a Docker build gate on `main`.
- **Deploy from Git only** — no manual edits on the host. To update:

  ```bash
  git pull && docker compose up -d --build
  ```

  or the pull-and-restart one-liner in `scripts/`. A deployment is only
  ever a function of a commit; rollback is a function of a tag.

---

## Rollback

```bash
# List releases
git tag -l

# Roll the deployment back to the known-good clinical baseline
git checkout v0.1.0-real-consultation-trial && docker compose up -d --build

# Or roll back to the previous main commit
git checkout main~1 && docker compose up -d --build

git checkout main   # resume development
```

The tag `v0.1.0-real-consultation-trial` is the frozen known-good baseline —
never delete or re-point it. Consultation data lives in the
`iscribe-data` volume / `ISCRIBE_DATA_DIR` and is untouched by any code
rollback; restoring it means restoring that directory from backup.

---

## Cost envelope (30-day trial)

| Item | Cost | Notes |
|---|---|---|
| Host (existing trial machine) | ₹0 | Docker + tunnel on current hardware |
| Cloudflare Tunnel | ₹0 | Free plan; DNS + TLS included |
| Domain | ₹0 | `prompttoshort.online` already owned |
| GitHub Actions CI | ₹0 | Free tier covers this repo's usage |
| Deepgram STT | usage-based | `nova-3-medical`; trial budget the clinic controls. Zero-retention mode (`mip_opt_out=true`) is ON by default. |

Total fixed infrastructure: **₹0/month**. The only variable is STT usage,
protected by: upload caps (`ISCRIBE_MAX_UPLOAD_MB`), per-user submission
limits (`ISCRIBE_PROCESSING_RATE_LIMIT`), login rate limiting, job
concurrency caps, and audio-duration limits in the recorder.

If the trial later needs a rented VM (absence/sleep of the demo host
becomes a problem): Fly.io shared-cpu-2x (2 GB) ≈ US$6/mo — **requires the
developer's explicit approval before enabling**.

---

## Backup posture (honest statement)

- Consultation data (SQLite DB + audio + logs) lives under
  `ISCRIBE_DATA_DIR`. **The trial deployment's backup is the operator's
  scheduled copy of that directory** — there is no automated off-site
  backup in this ₹0 configuration, and none is claimed.
- Recommended minimum: a nightly copy to a second disk/cloud folder
  (`robocopy`/`rclone`), retained ≥ 30 days.
- Audio is short-lived by design (approval-gated deletion); the durable
  clinical content is the approved note, which clinicians export into the
  clinic's own records workflow.

---

## Health checks

```bash
curl -s http://127.0.0.1:8123/api/health
```

Liveness — the process is serving. Always public, never requires a key.

```bash
curl -s http://127.0.0.1:8123/api/ready
```

Readiness — returns **200** only when transcription is actually possible, and
**503** with `"status":"loading"` during the ~5–20 s model warm-up after a
restart. Route clinicians only once this returns 200.

Full end-to-end verification against a running deployment:

```bash
.venv/Scripts/python.exe scripts/smoke_test.py --audio uploads/939dcfab957e.mp3
```

Add `--base-url https://<your-tunnel-host>` to test through the public URL.
**Use synthetic audio only** — anything passed here is processed and stored.

---

## Performance

Measured on the trial host (AMD Ryzen 5 5600H, 6 cores / 12 threads, 5.9 GB RAM,
no CUDA, Whisper `base` int8 on CPU, 4-thread cap) using a 104-second synthetic
consultation, with the machine also running other workloads:

| Metric | Measured |
|---|---|
| Process startup to listening | ~2 s |
| Model warm-up (cached weights) | 4–16 s |
| Audio upload (816 KB) | 0.07 s local / 0.48 s via tunnel |
| Full pipeline, 104 s audio | **41–59 s** across four runs |
| Realtime factor | **1.8× – 2.5× faster than realtime** |
| Text-only flow (no STT) | 0.5–1.1 s |
| Transcription CPU use | ~2 cores average |
| Server memory (committed) | ~830 MB |
| Server memory (peak working set) | ~345 MB |

**Extrapolation:** at ~2× realtime a 10-minute consultation takes roughly
**5 minutes** to process. That is slower than the 40–60 s target. Options, in
order of effort: switch `ISCRIBE_WHISPER_MODEL` to `tiny` (faster, less
accurate), move to a host with more RAM and cores, or enable CUDA on a machine
with a working GPU stack.

---

## Rollback

The pre-deployment state is preserved on the `main` branch.

Revert the code and restart:

```bash
git checkout main
```

Return to the deployed version:

```bash
git checkout deploy/hospital-trial
```

Roll back one commit while staying on the branch:

```bash
git revert --no-edit HEAD
```

Consultation data is **not** touched by any of these — it lives in
`ISCRIBE_DATA_DIR`, outside version control. To roll back data as well, stop the
service and restore that directory from backup.

Stop the service and the tunnel:

```bash
powershell -Command "Get-NetTCPConnection -State Listen -LocalPort 8123 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
```

---

## Data protection

- **Audio, database and logs** live under `ISCRIBE_DATA_DIR`, which is outside
  the static web root and git-ignored. No route serves files from it.
- **Uploaded audio** is deleted automatically after
  `ISCRIBE_AUDIO_RETENTION_HOURS` (default 24). Notes and transcripts remain in
  the database until the trial's own retention decision is applied.
- **Logs contain operational data only** — consultation ids, stage names,
  durations, byte counts, error types. Patient identifiers, transcripts, notes
  and prescriptions are never logged, and a truncation filter backstops
  third-party libraries. This is asserted by the test suite.
- **Server filesystem paths** are stripped from every API response.
- **The access key** is never written to a log, never returned to the browser,
  and never placed in a URL. The browser holds only an HMAC-signed, HttpOnly,
  SameSite=Strict session cookie.
- **Back up** `ISCRIBE_DATA_DIR` on the hospital's own schedule. It contains
  clinical data and must be treated accordingly.

---

## Known limitations

1. **Single clinic per deployment.** Accounts, roles and clinic isolation
   are enforced per deployment (`ISCRIBE_HOSPITAL_ID`); multi-clinic SaaS
   tenancy (many clinics, one server) is a future architecture, not this
   trial. Per-clinician identity, ADMIN/DOCTOR/STAFF roles and the clinic
   boundary ARE active in this release.
2. **No email-password recovery flow.** A forgotten password is reset by an
   ADMIN (`users.py` CLI/`set_password`) — there is no email service in the
   ₹0 configuration. Magic-link/OAuth were deferred by design; the session
   layer accepts named identities from any future issuer without endpoint
   rewrites.
3. **Rate limiting is per-process.** The sliding-window limiter is in-memory
   (single-process by design); a multi-worker deployment would need a shared
   store. This deployment is single-worker.
4. **Quick tunnel URLs are ephemeral** and change on every restart. Use a named
   tunnel for the actual trial.
5. **Single host, no redundancy.** If the machine sleeps, reboots or loses
   network, iScribe is unavailable until restarted. Disable sleep on the host.
6. **Speaker labelling is a heuristic.** Without pyannote (gated model, needs
   `HF_TOKEN`), speakers are assigned by alternating turns and are wrong a
   significant fraction of the time. The UI reports `confidence: low`. Every
   note must be read against the transcript.
7. **Template text.** Fields marked `[template]` are fixed boilerplate, not
   extracted from the consultation. They appear in exports and must be corrected
   or removed during review.
8. **Clinical extraction is keyword- and regex-based**, inherited from the
   upstream project, with known weaknesses documented in `ANALYSIS.md` §4.
   It supports review; it does not replace it.
9. **Malayalam STT is experimental.** English production path: Deepgram
   (zero-retention) or local faster-whisper. Malayalam runs through the
   verified ASR-correction + semantic pipeline and is validated for
   test/decision data only — not for unsupervised clinical use; the UI
   labels it accordingly.
10. **Upstream licence.** The upstream repository declares no licence
   (`ANALYSIS.md` §5). Resolve this before any use beyond a trial.
11. **Real consultation fixtures are not in the repository** (patient data).
    CI and fresh clones run with those tests skipped; the recording owner's
    machine runs the full suite. The baseline tag was cut from a tree whose
    git history never contained the recordings — but the OLD local history
    (branch `local/phi-history-do-not-push`) DOES contain them and must
    never be pushed.
