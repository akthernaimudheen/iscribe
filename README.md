# iScribe — AI Medical Scribe

AI-generated clinical documentation from consultation audio: speech-to-text,
speaker-labelled transcripts, a typed clinical fact graph, and a
deterministic, validator-gated draft note that a clinician reviews and
approves before it becomes a document.

**Every note is an AI-generated DRAFT until a clinician reviews it.** The
product never represents an AI note as an independently verified clinical
record.

## What it does

- **English STT** via Deepgram (`nova-3-medical`, zero-retention mode) or
  fully local faster-whisper — selected per deployment, never silently
  switched at runtime (fail-closed).
- **Malayalam pipeline**: verified ASR-correction layer, clinical semantic
  normalization, Malayalam lexicon; heavy inference runs as a separate local
  sidecar so the cloud deployment never needs a GPU.
- **Clinical Intelligence V2**: PRESENT / ABSENT / RESOLVED / UNCERTAIN /
  QUESTIONED states with verbatim evidence spans; suspected diagnoses stay
  suspected; resolved episodes stay separate from residual problems.
- **Note generation**: section-aware draft (HPI, exam, assessment, plan…)
  rendered from the fact graph and gated by a fail-closed validator that
  rejects any sentence without transcript evidence.
- **Privacy boundary**: pre-flight policy check before any audio leaves the
  host — provider identity, region, zero-retention and off-host rules are
  enforced and audited; a refusal is explicit (`STTPolicyBlocked`), never a
  silent fallback.
- **Retention lifecycle**: approval-gated audio deletion, consent withdrawal
  propagation, training-vault separation from clinical data, append-only
  PHI-free audit trail.

## Quick start (development)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
copy .env.example .env        # then fill values
python -m service             # http://127.0.0.1:8123
```

Requires ffmpeg/ffprobe on PATH. See `DEPLOYMENT.md` for production.

## Accounts and roles

Day-to-day sign-in is per-user (email/password, PBKDF2-hashed). Roles:

| Role | Can |
|---|---|
| `ADMIN` | everything in its clinic + user management |
| `DOCTOR` | create/edit consultations, review, approve, export |
| `STAFF` | create/edit consultations, review; **no** approve/export |

Clinic isolation is mandatory and server-enforced: every consultation is
stamped with the signed session's clinic; a consultation from another clinic
is indistinguishable from a missing one (404), on detail reads, lists and
actions alike. The shared `ISCRIBE_ACCESS_TOKEN` remains as break-glass
access: it opens the door but cannot approve, finalize or export notes.

First admin bootstrap:

```bash
curl -X POST https://<host>/api/auth/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@clinic.test","password":"<long-password>",
       "display_name":"Trial Admin","bootstrap_code":"<ISCRIBE_BOOTSTRAP_ADMIN_CODE>"}'
```

## Tests

```bash
python -m pytest -q
```

The suite includes clinical-logic regressions (normalization semantics,
note validation, Malayalam parity), privacy/retention guarantees
(fail-closed STT, audit allowlist, audio lifecycle) and deployment checks
(auth, tenant isolation, rate limits, health endpoints).

Real consultation recordings/transcripts used to develop the clinical logic
are **deliberately not in this repository** (patient data); the tests that
need them skip cleanly when the fixtures are absent.

## Deployment

See `DEPLOYMENT.md` — Docker Compose + Cloudflare Tunnel,
`https://scribe.prompttoshort.online`, rollback procedure, backup posture,
cost envelope, and known limitations.
