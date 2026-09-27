# Audio Retention, Consent, Security and Training-Data Separation

Reference for the retention layer introduced on `deploy/hospital-trial`.
Evidence-based claims only: wherever this document says "unknown", that is a
deliberate, honest statement, not an omission.

## 1. Provider data flow (browser → note)

```
Browser (MediaRecorder)
  → HTTPS multipart POST /api/consultations/{cid}/audio   (session-gated)
  → application server streams to data/uploads/{cid}.{ext}   (OUR storage;
    size-capped; .part atomicity; sha256 recorded; never on a static mount)
  → on Process: ScribeEngine.process_audio resolves an STT provider
  → DeepgramSTTProvider POSTs the full audio file to
      https://api.deepgram.com/v1/listen
      (model nova-3-medical; diarize/utterances/smart_format/numerals on;
       Authorization header from DEEPGRAM_API_KEY — env-only, never logged)
  → transcript + utterances + request_id returned
  → validation → diarization labelling → clinical fact graph (V2)
  → deterministic note (clinical_note_v2, fail-closed validator)
  → result JSON persisted in SQLite (data/iscribe.db)
```

- Audio is uploaded to the application server first; it is never uploaded
  directly from the browser to a provider.
- What is sent to the provider: the audio bytes plus the query parameters
  listed in `DeepgramSTTProvider.retention_configuration()` (model, language,
  punctuate, smart_format, numerals, diarize, utterances, profanity_filter).
  No API key travels in a URL; the key is never logged.
- **OUR STORAGE RETENTION** is owned by the lifecycle in `service/retention.py`
  (see §2): the local file is deleted on a deadline with audit events.
- **THIRD-PARTY PROCESSOR RETENTION** (what Deepgram keeps, for how long,
  whether it trains on customer audio) is documented per provider
  documentation in `docs/stt_provider_privacy_and_security.md`: Deepgram's
  default (Model Improvement Program) retains audio AND transcripts for
  training; `mip_opt_out=true` (sent by default on every request since this
  phase) yields zero data retention — process-only. A full in-region
  guarantee requires a regional endpoint (`ISCRIBE_DEEPGRAM_REGION`:
  eu/au/in) plus the opt-out. No DPA is verified for this deployment — that
  remains a contractual task. `supports_delete_after_processing()` returns
  `False` (no per-request deletion API exists).
- Claiming "we deleted the audio" refers ONLY to our own storage. It is never
  a claim about the provider's copies.

## 2. Audio lifecycle (state machine)

```
UPLOADED → PROCESSING → NOTE_READY → AWAITING_DOCTOR_REVIEW
        → DOCTOR_APPROVED → SCHEDULED_FOR_DELETION → DELETED
                                            ↘ DELETION_FAILED → (retry) ↰
```

Note approval gate (`note_status`): `DRAFT → REVIEWED (doctor edit) →
APPROVED (explicit approval event)`. "Note generated" is never "approved";
approval and completion endpoints refuse unreviewed drafts, and the export
endpoint serves only APPROVED notes.

- The deletion deadline is computed at policy evaluation and snapshots the
  policy that produced it (`retention_policy_id`, `retention_policy_version`,
  e.g. `after_approval=24h`), so a later global policy change never silently
  rewrites a historical decision:
  - after explicit approval: `approved_at + audio_retention_after_note_approval_hours`
    (configurable; 0 = delete immediately),
  - never-approved audio: `upload + audio_retention_unreviewed_hours`
    (a configured ceiling the hospital sets, never an internal default),
  - records with a pending documentation job are never scheduled away while
    the job runs (invariant: audio must not disappear while review needs it).
- A rejected/edited note keeps the audio; a REVIEWED note still carries no
  deletion deadline until approval or the configured ceiling applies.
- Deletion failures are recorded (`DELETION_FAILED`, attempt count, error
  type, next retry) and retried with backoff, max 5 attempts, remaining
  visible in the audit trail. A deletion is never pretended to have happened
  (`deleted_at` is stamped only on success).

## 2b. Training consent revocation

`POST /api/hospital/training-authorization {authorized: false}` records the
revocation (timestamp + actor), keeps the full audit trail, flags every
retained training record (`revoked = 1`) for the dataset/version policy, and
blocks all future ingestion with the explicit reason
`training_consent_revoked` — even if a stale authorization row remains.
Consent rows are never silently deleted.

## 2c. Training pipeline status (honest)

The required training pipeline stages (de-identification, human review,
dataset approval) require a validated de-identification capability that this
codebase does not have (see §5). No stage is faked: ingestion stops at the
authorized vault, records remain identified clinical audio, and no
`DE_IDENTIFICATION` / `TRAINING_DATASET_APPROVED` state exists in code. A
validated de-identification pipeline is a prerequisite for adding them.

## 3. Retention policy model (Part 1 of the spec)

`service/retention.py::RetentionPolicy` resolves, per hospital
(row in `hospital_config` overriding environment defaults):

| Field | Env var | Default |
|---|---|---|
| audio_retention_after_note_approval | ISCRIBE_AUDIO_RETENTION_AFTER_APPROVAL_HOURS | 24 |
| training_retention_enabled | ISCRIBE_TRAINING_RETENTION_ENABLED | false |
| training_retention_period | ISCRIBE_TRAINING_RETENTION_PERIOD_HOURS | 8760 |
| training_data_requires_explicit_selection | ISCRIBE_TRAINING_REQUIRES_EXPLICIT_SELECTION | true |
| training_data_requires_hospital_authorization | ISCRIBE_TRAINING_REQUIRES_HOSPITAL_AUTHORIZATION | true |

Plus `audio_retention_unreviewed_hours` (default 168) and the server-assigned
`hospital_id` (`ISCRIBE_HOSPITAL_ID`). Every default favours deletion and
denial. No retention duration is hard-coded in logic.

## 4. Training-data separation

- The training dataset is its own table (`training_records`) and its own
  storage namespace (`ISCRIBE_TRAINING_VAULT_DIR`, default
  `data/training_vault/`). A clinical record is NEVER flagged "training";
  a training copy is its own record with:
  `training_record_id, source_consultation_id, hospital_id, audio_path,
  sha256, authorization_reference, retention_deadline, created_at,
  approved_for_training_at, approved_by, selection_reason, deletion_status,
  deletion_attempts, last_deletion_error, next_deletion_retry_ts`.
- Entry is only via explicit selection of an APPROVED consultation; every
  guard fails closed (disabled → 403; no/expired/ambiguous authorization →
  403; explicit selection required and missing → 403).
- The authorization event (`POST /api/hospital/training-authorization`)
  requires a SEPARATE credential (`ISCRIBE_TRAINING_ADMIN_TOKEN`; unset =
  endpoint disabled), a version string, a finite expiry, and optionally a
  policy reference. "Uses the product" never implies "allows training".
- The training copy is a MOVE: exactly one copy exists, in the vault. The
  clinical row is marked purged with `training_record_id` linking to the
  independent record. Training deletion (deadline, retry, audit) is fully
  independent of clinical deletion.
- Nothing here is an export path: training audio stays inside the authorized
  environment.

## 5. De-identification: honest status

The system has **no automated audio de-identification**. There is no safe,
validated transformation of IDENTIFIED CLINICAL AUDIO into DE-IDENTIFIED
TRAINING DATA in this codebase, and no anonymity claim is made anywhere.
Training audio therefore remains identified clinical audio held inside the
hospital-authorized vault until a validated de-identification pipeline exists
(direct identifiers: names, phone numbers, addresses, DOB, MRNs, hospital
identifiers, emails, exact dates — none are removed today).

## 6. Audit trail

`audit_events` is append-only (no update/delete path exists in code) and
stores: event_id, ts, action, cid, hospital_id, actor, detail (JSON of ids,
enums, byte counts, hashes — never transcripts, notes, audio bytes, or patient
identifiers). Events include: consultation_created, audio_uploaded,
transcription_requested, note_generated, note_reviewed, note_approved,
note_exported, audio_accessed, retention_policy_evaluated, audio_deleted,
audio_deletion_failed, training_authorization_granted/revoked,
training_copy_created, training_selection_rejected, training_record_deleted,
training_deletion_failed. The rotating operational log remains the second,
PHI-redacted channel.

## 7. Security properties — guaranteed and not

Guaranteed by code + tests (tests/test_retention_security.py):

- Unauthenticated requests cannot reach any clinical endpoint (session or
  token required; production refuses to start without a token).
- Raw audio has no static/public URL: it is served only via
  `/api/consultations/{cid}/audio` with a valid session AND a short-lived
  (15 min), HMAC-signed, cid-bound token.
- hospital_id is server-assigned; client request data cannot set it.
- Deletion requires the approval event (or the configured unreviewed
  ceiling); pending jobs block deletion; failures are recorded and retried.
- Training entry is fail-closed on every axis and auditable end-to-end.
- Audit events carry ids/hashes only.

NOT yet guaranteed (documented honestly):

- **Per-doctor identity and doctor A vs doctor B isolation.** One shared
  access token means any authenticated session can access every consultation
  in this deployment. `doctor_id` columns exist for the future user model;
  they are not populated from a verified identity. (See
  experiments/hospital_readiness/HOSPITAL_REJECTION_AUDIT.md, Phase A/C.)
- **Cross-hospital isolation in one process.** Tenancy is stamped and
  queryable, but this deployment serves ONE hospital per instance; there is
  no request-time tenant selection to enforce.
- **Encryption at rest** (SQLite and audio files are plaintext; use OS disk
  encryption), **backup lifecycle** (backups can outlive deletion deadlines),
  **Deepgram-side retention** (unknown without a DPA).
- Automated **de-identification** (none exists — see §5).

## 8. Operations

- Migration is additive and idempotent (columns + 3 new tables); legacy rows
  are backfilled to this server's hospital and safe lifecycle states. No new
  database engine; no commit of data files.
- Restart: use the existing relaunch helper
  (`C:/Users/akthe/.iscribe-logs/_relaunch_service.py`, port 8123). Rollback:
  stop the service, `git checkout` the previous revision of the service
  package, restart — the new tables/columns are ignored by old code, so the
  rollback is code-level only.
- Manual verification: upload audio → process → confirm `audio_state`
  progresses and `note_generated` appears in audit → approve → confirm
  `SCHEDULED_FOR_DELETION` with a deadline → expire the deadline → confirm
  file removal + `audio_deleted` event → attempt training selection without
  authorization (expect 403) → grant authorization with the admin token →
  select → confirm vault file + `training_record_deleted` at expiry.
