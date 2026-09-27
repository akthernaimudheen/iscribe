# GLM Implementation Specification -- iScribe Hospital Pilot
## Phases A through H -- Ordered by pilot priority

**Generated:** 2026-09-26
**Prerequisite reading:** HOSPITAL_REJECTION_AUDIT.md (same directory)
**Regression baseline:** 507 tests passing, 1 skipped, 0 failures -- must remain intact after every phase

---

## MANDATORY RULES FOR GLM

1. **Do NOT modify these files under any circumstances:**
   - scribe_engine/clinical_facts.py
   - scribe_engine/note_v2.py
   - scribe_engine/fact_graph.py
   - Any file under tests/ -- do not delete, rename, or break existing tests

2. **Do NOT rewrite the clinical intelligence engine.** It is correct. Add only alongside it.

3. **Every phase must be individually testable.** Write new tests for new behaviour. Do not alter existing tests.

4. **auth_required=False must remain functional** for local development and for the test suite.
   Per-user auth runs only when auth_required=True.

5. **PHI-safe logging rules do NOT change.** Never add transcript content, patient names,
   or clinical note text to any log line.

6. **Session cookie security attributes do NOT change:**
   HttpOnly=True, SameSite=strict, Secure=True (production), Path=/

7. **Implement phases strictly in order.** Phase B requires Phase A. Phase C requires Phase B.

---

## PHASE A -- Per-User Authentication

### Objective
Replace the single shared token with per-doctor user accounts.
Every session must know which user is signed in.
This is the prerequisite for every other security feature.

### New file: service/users.py

Functions to implement:
- create_user(store, username, role, password) -> dict
- verify_password(store, username, password) -> dict | None
- deactivate_user(store, user_id) -> bool
- list_users(store) -> list[dict]
- get_user(store, user_id) -> dict | None

Passwords are hashed with bcrypt. The hash is stored in the users table.
Plaintext passwords never touch the database or logs.

### Schema addition to service/store.py _SCHEMA

```sql
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    role       TEXT NOT NULL DEFAULT 'doctor',
    pw_hash    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active  INTEGER NOT NULL DEFAULT 1,
    last_login TEXT
);
```

### Changes to service/security.py

1. Replace issue_session(access_token, ttl_seconds) with
   issue_session(user_id, role, access_token, ttl_seconds).
   New payload: {user_id}:{role}:{expiry} -- HMAC signed as before.

2. Replace verify_session(cookie_value, access_token) with
   verify_session(cookie_value, access_token) -> dict | None.
   Returns {user_id, role} on success, None on failure.

3. Remove token_matches and TOKEN_HEADER (X-Access-Token bearer auth path is removed).

### Changes to service/app.py

Login endpoint: accepts {username, password} instead of {access_token}.
On success: issue_session with user_id and role, set cookie, redirect to /.
On failure: login_page('Incorrect username or password.') + audit log login_failed.

Middleware: AccessControlMiddleware.dispatch must:
1. Verify session cookie -> gets {user_id, role}
2. Set request.state.user_id = user_id
3. Set request.state.role = role
4. If no valid session: redirect to /login (non-API) or return 401 JSON (API)

New admin-only endpoints:
- POST /api/users -- create user: {username, role, password}
- PATCH /api/users/{uid}/deactivate
- GET /api/users -- list users (no pw_hash)

Admin bootstrap: On startup, if users table is empty AND ISCRIBE_ADMIN_USERNAME is set,
create initial admin from ISCRIBE_ADMIN_USERNAME + ISCRIBE_ADMIN_PASSWORD.
Log: event=admin_user_created hint=change_password_immediately
Do NOT log the password.

### Config additions (service/config.py)

```
admin_username: str = ''   # ISCRIBE_ADMIN_USERNAME
admin_password: str = ''   # ISCRIBE_ADMIN_PASSWORD -- used once for bootstrap then ignored
```

### Tests to write (tests/test_auth.py)

- POST /api/login with correct credentials returns 303 + session cookie
- POST /api/login with wrong password returns 401
- With auth_required=True, accessing /api/consultations without session returns 401
- With auth_required=False (default), existing tests continue to pass unchanged
- Deactivated user cookie is rejected
- Admin can create a user; doctor role cannot
- Admin can list users; doctor role cannot

---

## PHASE B -- Audit Log

### Objective
Append-only record of every consultation state change, attributed to a user.
A hospital IT committee must be able to answer: who viewed/approved/exported this note?

### New file: service/audit.py

Function to implement:
- record(store, user_id, action, cid=None, **detail_kwargs) -> None
  Writes one audit event. detail_kwargs must be PHI-free scalars only.
  No UPDATE or DELETE operations ever. INSERT only.
  This file must never import from scribe_engine.

### Schema addition to service/store.py _SCHEMA

```sql
CREATE TABLE IF NOT EXISTS audit_events (
    event_id  TEXT PRIMARY KEY,
    ts        TEXT NOT NULL,
    ts_epoch  REAL NOT NULL,
    user_id   TEXT,
    action    TEXT NOT NULL,
    cid       TEXT,
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events (user_id, ts_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_audit_cid  ON audit_events (cid, ts_epoch DESC);
```

IMPORTANT: Add a verification that ConsultationStore has NO update or delete method for audit_events.

### Audit calls to add in service/app.py

| Endpoint | Action string |
|---|---|
| POST /api/login success | login_ok |
| POST /api/login failure | login_failed |
| POST /api/consultations | consultation_created |
| POST /api/consultations/{cid}/audio | audio_uploaded |
| POST /api/consultations/{cid}/process | process_queued |
| _execute_doc_job completion | process_completed |
| _execute_doc_job failure | process_failed |
| PATCH /api/consultations/{cid}/review | consultation_reviewed |
| POST /api/consultations/{cid}/complete | consultation_completed |
| GET /api/consultations/{cid}/export | consultation_exported |
| _purge_expired_audio per file | audio_purged |

### New API endpoints in service/app.py

- GET /api/audit?cid={cid} (admin only)
- GET /api/audit/user/{uid} (admin only)
Both return list of audit_event records in chronological order.

### Tests to write (tests/test_audit.py)

- Every state-changing endpoint generates an audit_event row
- Audit events are in chronological order
- Audit events for a cid are queryable by admin
- Non-admin cannot query audit log (403)
- Detail field contains only PHI-free scalars (assert no long strings > 200 chars)

---

## PHASE C -- Doctor-Owned Consultations

### Objective
A doctor session can only access consultations they created.
Admin role can access all consultations.

### Schema change in service/store.py

Add column to consultations table:
```sql
created_by_user_id  TEXT   -- NULL for legacy rows created before Phase A
```
ConsultationStore.create() must accept and persist created_by_user_id.

### Access control helper in service/app.py

```python
def _require_consultation_access(record: dict, request: Request) -> None:
    if not settings.auth_required:
        return  # dev mode: no restriction
    role = getattr(request.state, 'role', 'doctor')
    if role == 'admin':
        return
    user_id = getattr(request.state, 'user_id', None)
    if record.get('created_by_user_id') != user_id:
        raise HTTPException(403, 'Access denied')
```

Apply _require_consultation_access to:
- GET /api/consultations/{cid}
- POST /api/consultations/{cid}/audio
- POST /api/consultations/{cid}/process
- PATCH /api/consultations/{cid}/review
- POST /api/consultations/{cid}/complete
- GET /api/consultations/{cid}/export

List endpoint: doctors see only their own; admin sees all.

Consultation creation: set created_by_user_id from request.state.user_id.

### Tests to write (tests/test_ownership.py)

- Doctor A cannot GET, review, complete, or export Doctor B's consultation (403)
- Admin can GET, review, complete, and export any consultation
- list_consultations for doctor returns only their own consultations
- list_consultations for admin returns all consultations

---

## PHASE D -- Training Consent Data Model (Inert by Default)

### Objective
Build the data model and the consent endpoint.
Keep TRAINING_ENABLED=false (default). The path exists but is inert.

### Config addition (service/config.py)

```
training_enabled: bool = False  # TRAINING_ENABLED env var
```

### Schema additions to consultations table

```sql
training_consent_given  INTEGER NOT NULL DEFAULT 0,
training_consent_by     TEXT,
training_consent_at     TEXT,
training_consent_basis  TEXT,
training_category       TEXT NOT NULL DEFAULT 'clinical',
training_vault_path     TEXT,
training_vault_moved_at TEXT
```

### New endpoint in service/app.py

POST /api/consultations/{cid}/training-consent
  Body: {consent_basis: string}
  Auth: doctor role, own consultation only
  Conditions: TRAINING_ENABLED=true AND consultation status=completed
  Effect: sets training_consent_given=1, training_category=authorized_training
  Audit: training_consent_granted event
  Returns 403 if TRAINING_ENABLED=false
  Returns 400 if consultation not completed
  Returns 409 if consent already granted

### Cleanup timer change

In _purge_expired_audio, before Path.unlink:
  If record.get('training_category') == 'authorized_training': skip (continue).

### Tests to write

- With TRAINING_ENABLED=false, endpoint returns 403
- With TRAINING_ENABLED=true, consent granted for completed consultation succeeds
- With TRAINING_ENABLED=true, consent cannot be granted for non-completed consultation
- Cleanup timer does not delete training-category audio
- Cleanup timer does delete clinical-category audio past retention window

---

## PHASE E -- PDF Export

### Objective
A professional, printable PDF that a doctor can hand to a patient or file in a HIS.
Plain text export must continue to work unchanged.

### New dependency

Add to requirements.txt: reportlab>=4.0

### New file: service/export_pdf.py

Implement: generate_pdf(record: dict, doctor_name: str, hospital_name: str) -> bytes
Use ReportLab. Return PDF bytes.

PDF content requirements:
1. Hospital name (configurable via ISCRIBE_HOSPITAL_NAME env)
2. Consultation metadata header (id, doctor, department, date)
3. Prominent warning at top if reviewed=False
4. Disclaimer: 'Trial software. Clinician review and sign-off required before clinical use.'
5. Full transcript text
6. Clinical note section (V2 note when valid; legacy note with label otherwise)
7. Prescription section
8. Doctor name from authenticated session user (not from the free-text doctor field)
9. Approval timestamp from completed_at
10. Footer: 'Fields reading Not mentioned were not stated in this consultation.'

### Changes in service/app.py

Add format=pdf and format=json query parameters to GET /api/consultations/{cid}/export.
Default is format=txt (existing behaviour unchanged).

Config addition: hospital_name: str = 'Hospital'  # ISCRIBE_HOSPITAL_NAME env var

### Tests to write

- GET /export?format=txt returns plain text (existing behaviour unchanged)
- GET /export?format=pdf returns application/pdf with non-zero content
- GET /export?format=json returns valid JSON with clinical_facts_v2 present
- Export creates audit event with format field
- Doctor cannot export another doctor's consultation

---

## PHASE F -- STT Provider Hardening

### Objective
Add local-only mode. Standardise provider capabilities.
Protect against Deepgram being the only path for hospitals that require local processing.

### Config addition (service/config.py)

```
stt_local_only: bool = False  # ISCRIBE_STT_LOCAL_ONLY env var
```

### Changes in scribe_engine/stt/base.py

Add to STTProvider:
- provider_capabilities: dict with keys: streaming, languages, medical_tuned, local, diarization
- is_configured: property, returns True by default (override in hosted providers)

### Changes in scribe_engine/stt/__init__.py

In resolve_production_provider: if local_only=True, skip Deepgram and return faster-whisper.

### Changes in service/app.py build_engine()

Pass local_only=settings.stt_local_only into resolve_production_provider.

### Changes in scribe_engine/stt/deepgram_provider.py

Add provider_capabilities dict with local=False, medical_tuned=True, diarization=True.

### Tests to write

- With ISCRIBE_STT_LOCAL_ONLY=true, resolve_production_provider returns faster-whisper
- With ISCRIBE_STT_LOCAL_ONLY=true, DeepgramSTTProvider.transcribe is never called
- /api/ready reports the correct stt_provider in local-only mode

---

## PHASE G -- Backup and Deletion Verification

### New file: scripts/Backup-iScribe.ps1

PowerShell script that:
1. Runs PRAGMA wal_checkpoint(FULL) via Python one-liner
2. Copies the .db file to ISCRIBE_BACKUP_DIR with datestamp
3. Prints confirmation with file size

### New file: scripts/verify-deletion.py

Python script that:
1. Queries consultations where audio_purged=1
2. For each, checks that audio_path does not exist anywhere under backup-dir
3. Reports any found as violations

### Deletion audit event

In app.py _purge_expired_audio, after Path.unlink:
Compute SHA-256 hash of the deleted path string (not content).
Write audit event: audio_purged_verified with path_hash (first 16 hex chars).

---

## PHASE H -- Hospital Pilot Hardening (Process Checklist -- Not Code)

These are verification steps to complete before handing to a hospital. Not code changes.

```
[ ] OS-level disk encryption verified active on host
    Evidence: Screenshot of encryption status + date

[ ] Deepgram DPA
    Option A: Obtain and sign Deepgram Enterprise DPA covering clinical audio
    Option B: Set ISCRIBE_STT_LOCAL_ONLY=true
    Evidence: Signed DPA OR confirmation of local-only mode

[ ] Upstream repository licence resolved
    Evidence: Written confirmation on file

[ ] Admin bootstrap completed
    Action: Set ISCRIBE_ADMIN_USERNAME + ISCRIBE_ADMIN_PASSWORD on first deployment
    Action: Create named doctor accounts, then rotate or remove admin bootstrap credentials
    Evidence: Named accounts in users table

[ ] Port 8123 not accessible from outside localhost
    Evidence: Firewall configuration screenshot

[ ] Named Cloudflare Tunnel with stable hostname
    Evidence: Stable hostname + DNS record

[ ] Retention hours agreed with hospital
    Evidence: Hospital sign-off on retention policy

[ ] Backup tested
    Action: Run Backup-iScribe.ps1, then restore and verify consultations present
    Evidence: Restore confirmation log

[ ] TRAINING_ENABLED=false confirmed
    Evidence: Screenshot or config export

[ ] Smoke test passes end-to-end
    Evidence: scripts/smoke_test.py --base-url <tunnel-url> passes

[ ] Pilot doctors trained and signed off
    Evidence: Each participating doctor has reviewed and signed an acknowledgement
    that notes require their review before clinical use
```

---

## ACCEPTANCE CRITERIA -- ALL PHASES

After all phases are implemented:

1. Full regression suite still passes: 507+ tests, 0 failures
2. New test suites cover auth, audit, ownership, training consent, PDF export, local-only STT
3. With auth_required=False, all existing behaviour is unchanged (dev/test mode)
4. With auth_required=True:
   - Per-doctor login works
   - Doctor A cannot access Doctor B's data
   - Admin can access all data
   - Every state change generates an audit_event row with user_id
   - PDF export is available and correct
   - ISCRIBE_STT_LOCAL_ONLY=true routes all audio to faster-whisper
5. TRAINING_ENABLED=false is the default and is enforced at the endpoint

## PHASE DEPENDENCIES

```
Phase A (auth) must complete before:
  Phase B (audit needs user_id from session)
  Phase C (ownership needs user_id from session)
  Phase D (consent needs user_id from session)
  Phase E (PDF export needs doctor_name from session)

Phase B (audit) must complete before:
  Phase G (deletion audit events use audit.record)

Phase C (ownership) must complete before:
  Phase E (export gating uses ownership check)
  Phase D (consent gating uses ownership check)
```

---

*This specification is for implementation only. Do not modify the clinical safety layer.*
*All test changes must be additive -- existing tests must continue to pass.*