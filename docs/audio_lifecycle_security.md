# Audio Lifecycle & Security Architecture

Scope: hospital-safe lifecycle for clinical consultation audio and the strictly
separate training/quality-improvement dataset. Status quo is described as
implemented; every "not done" is an honest limitation, not a plan. No legal
compliance (HIPAA/GDPR/DPDP) is claimed anywhere — technical controls only,
for review by the hospital's privacy/security team.

## 1. Clinical audio state machine (implemented)

```
UPLOADED → PROCESSING → NOTE_READY → AWAITING_DOCTOR_REVIEW
        → DOCTOR_APPROVED → SCHEDULED_FOR_DELETION → DELETED
                                            ↘ DELETION_FAILED → (retry) ↰
```

Note gate: `note_status DRAFT → REVIEWED (doctor edit) → APPROVED (explicit
approval event, approval_version)`. "Note generated" is never "approved".
Columns: `consultations.audio_state, note_status, approved_at/by,
audio_deletion_due_ts, deletion_attempts, last_deletion_error,
next_deletion_retry_ts, deleted_at, retention_policy_id,
retention_policy_version`.

## 2. Retention policy (implemented)

Per-hospital row (`hospital_config`) over env defaults:
`audio_retention_after_note_approval_hours` (0 = delete at approval), plus a
hospital-configured `audio_retention_unreviewed_hours` ceiling anchored to
upload. Every decision snapshots `retention_policy_id/version`
(`after_approval=24h`) so later policy edits never rewrite history (tested).
Deletion is executed only by the sweep, never inline, with retry/backoff
(max 5 attempts), `deleted_at` stamped only on verified success, and audited.

Review safety (all enforced + tested): no approval ⇒ no deletion; failed
transcription or note generation leaves audio untouched; pending jobs block
deletion; the deadline and state survive restarts (SQLite columns; the sweep
and backfill re-derive missing deadlines after crashes).

## 3. Training lifecycle (implemented; de-identification honestly scoped)

```
DOCTOR_APPROVED ── explicit selection (fail-closed authz) ──▶ TRAINING_PENDING_REVIEW
        (candidate = SEPARATE record; clinical audio MOVED into vault)
TRAINING_PENDING_REVIEW ── authorized reviewer ──▶ TRAINING_DATASET (approved)
                         └──────────── reviewer ──▶ TRAINING_REJECTED → deleted
TRAINING_DATASET ── retention expiry / withdrawal ──▶ TRAINING_DATASET_DELETED
Withdrawal before dataset inclusion: candidate deleted.
```

`doctor_approved` is never `training_allowed`. Entry requires: deployment
switch AND hospital row enabled AND a current, versioned, expiring
authorization AND (default) explicit per-recording selection with a reason.
Any ambiguity denies. The clinical encounter is never mutated into a training
artifact: the training candidate is its own row (training_record_id,
source_encounter reference, authorization_reference, own retention deadline)
and the vault file is a move, so exactly one copy exists under authorized
storage.

De-identification: a deterministic transcript redaction layer removes direct
identifiers (patient/doctor/hospital names, MRN/patient id, phone, email,
dates, long digit runs) on a COPY stored with the candidate, recorded as
`deidentification_status=TRANSCRIPT_DEIDENTIFIED`. **The audio is NOT claimed
de-identified** — voices/names may persist acoustically. Default policy:
audio excluded from training use while the de-identified transcript is
retained; `AUDIO_DEIDENTIFIED` requires a future, separately validated
artifact pipeline that does not exist today.

## 4. Authorization model

Single-deployment, single hospital (`ISCRIBE_HOSPITAL_ID`, server-assigned —
client data can never set it). Opaque random ids (`uuid4().hex`) for
consultations, jobs, training records; no name+DOB identifiers. Audio access
requires a valid session AND a short-lived (15 min), HMAC-signed,
cid-bound token issued only by `POST /api/consultations/{cid}/audio-link`.
Training administration requires a separate credential
(`ISCRIBE_TRAINING_ADMIN_TOKEN`); unset ⇒ disabled. **Not done:** per-doctor
accounts/roles (shared token), so doctor-to-doctor isolation inside one
deployment is NOT guaranteed; `doctor_id`/`approved_by` are session-claimed.

## 5. Deletion model

Sweep finds due clinical audio and training records; per-object attempt
accounting; filesystem `unlink`; success ⇒ state DELETED + `deleted_at` +
`AUDIO_DELETED`/`TRAINING_DATASET_DELETED` audit; failure ⇒ DELETION_FAILED +
retry time; missing file counts as a visible deletion (audited), never
silent. **Documented limitation:** local filesystem unlinks are not
cryptographic erasure; backups taken before deletion outlive the deletion
until the hospital's backup policy expires them; provider-side copies are
governed by the provider's own terms (unknown without a DPA).

## 6. Provider boundary

STT lives behind `scribe_engine.stt` (`STTProvider`; registry; language-map
resolution; Deepgram nova-3-medical → faster-whisper fallback). The lifecycle
belongs to this system: audio is stored locally first, POSTed whole to
Deepgram's `/v1/listen`, and `retention_configuration()` returns the honest
surface (`retention_known=False`, `audio_sent_off_host=True`,
`deletion_supported=False`). Per job we record provider, model, request_id,
timing (audit `transcription_completed`); we do NOT invent provider retention
guarantees. Provider failure never deletes source audio (tested); retry and
failover re-run the pipeline on the preserved file.

## 7. Audit model

Append-only `audit_events` (no update/delete code path): event_id, ts,
action, consultation_id, hospital_id, actor, detail-JSON. Ids, enums, byte
counts, sha256 hashes and policy versions only — never transcripts, notes,
audio bytes, or patient identifiers (allowlist-tested). Events cover the full
lifecycle: upload, transcription requested/completed, note generated/edited/
approved/exported, audio accessed, retention evaluated, deletion
scheduled/executed/failed, training authorization granted/withdrawn, candidate
created/reviewed/approved/rejected, de-identification completed, training
deletion executed/failed.

## 8. Threat model

| Threat | Exposure | Control | Residual risk |
|---|---|---|---|
| Unauthorized doctor (shared token) | Any authenticated session reads all encounters in the deployment | Token + session cookie, audited access | NOT mitigated for cross-doctor isolation; needs per-user identity |
| Unauthorized staff | Same as above | Same | Same |
| Leaked audio URL | URL contains cid + expiry + MAC | cid-bound 15-min HMAC token; wrong cid/expiry ⇒ 403; no static mount | Window of validity; rotation of access token invalidates all |
| Stolen session cookie | Full access until 12h expiry | HttpOnly, SameSite=Strict, Secure(prod), HMAC-signed | No per-user revocation; rotate token to kill all sessions |
| Database compromise | Plaintext SQLite: transcripts, notes, training table | Local-file deployment; disk encryption is deployment's duty | HIGH: no at-rest encryption in code; audit log is forgeable by disk attacker |
| Storage compromise (uploads/vault) | Raw audio files readable | OS permissions, separate vault namespace, sha256 anchors | Same as above |
| Provider compromise | Full audio POSTed to Deepgram | TLS; env-only key; no key in URLs/logs | Provider-side retention UNKNOWN (no verified DPA); use local faster-whisper for sensitive deployments |
| Accidental deletion | Doctor still reviewing | Approval-gated deadlines; pending-job guard; retry/backoff; audit | Only the configured unreviewed ceiling deletes unapproved audio (a policy choice) |
| Training consent error | Clinical audio silently trained on | Fail-closed multi-guard entry; explicit selection; separate admin credential; candidate review staging; audited | Human error in granting authorization itself |
| Cross-hospital access | hospital_id forged by client | Server-assigned stamps; mismatch refuses training moves | Single-hospital-per-process deployment model |
| Insider access | Operator reads DB/files | Audit trail of actions | Insider with disk access defeats in-code audit |
| Browser/device compromise | Session cookie on device | HttpOnly/SameSite; no tokens in localStorage; no keys in frontend | Device-level malware out of scope |
| Withdrawal after dataset use | Derived artifacts | Candidate flagged + deleted on withdrawal | Cannot un-train an already-trained model — documented, not claimed |
