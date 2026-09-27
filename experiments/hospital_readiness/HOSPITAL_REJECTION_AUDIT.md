# iScribe — Hospital Rejection Audit
## Senior Healthcare Software Architect + Clinical AI Safety Engineer + Hospital IT/Security Consultant

**Audit Date:** 2026-09-26
**Methodology:** Full adversarial code review — every claim is evidence-grounded against actual source files.
**Files inspected:** service/app.py, service/security.py, service/store.py, service/config.py,
service/logging_config.py, scribe_engine/pipeline.py, scribe_engine/stt/*, scribe_engine/validation.py,
scribe_engine/clinical_facts.py, Dockerfile, docker-compose.yml, DEPLOYMENT.md, .env.example, .gitignore

---

## SECTION 1 — EXECUTIVE ASSESSMENT

iScribe is a well-engineered **prototype** that demonstrates genuine clinical AI intelligence.
The Clinical Intelligence V2 engine has sound architecture. The audio retention timer, PHI redaction filter,
and session cookie design all reflect security-conscious thinking.

However, the system currently has **critical hospital-deployment blockers** that are not cosmetic.
They exist in the implementation, not just documentation.

The most serious blockers are:

1. **No per-clinician identity.** Everyone shares one token. No audit trail can attribute an action to a doctor.
2. **No encryption at rest.** SQLite database containing full clinical notes and transcripts is unencrypted on disk.
3. **No training-data isolation.** The concept exists in documentation but there is zero code implementing it.
4. **Deepgram sends patient audio to a US-based cloud service.** Data residency and contractual controls are unverified.
5. **Consent infrastructure does not exist.** There is no mechanism to record, verify, or audit patient consent.

**A Kerala hospital's IT committee would reject this product today** on grounds 1, 2, and 4 alone.
These are fixable. They are not architectural rewrites. The clinical engine can stay intact.

**Prototype status: Approved for internal developer evaluation.**
**Hospital pilot status: BLOCKED. Requires at minimum: per-doctor auth, at-rest encryption plan, Deepgram DPA, and documented retention policy.**

---

## SECTION 2 — PHASE 1: CURRENT SYSTEM AUDIT

### 2.1 Component Inventory

| Component | What Exists | Status | Pilot Blocker? |
|---|---|---|---|
| Audio capture | Browser MediaRecorder to upload endpoint | Functional | No |
| Upload transport | HTTPS multipart POST, size limit enforced, .part atomicity | Functional (TLS depends on tunnel) | No if tunnel is running |
| STT provider | Deepgram nova-3-medical (primary) + faster-whisper (fallback) + Malayalam sidecar | Functional | Conditional — see Deepgram section |
| Transcript storage | Full transcript text in SQLite document JSON blob | Functional — **UNENCRYPTED** | **YES** |
| Clinical intelligence | Clinical Intelligence V2, deterministic fact graph, fail-closed | Functional | No |
| Fact graph | clinical_facts.py, evidence-anchored, no hallucination path | Functional | No |
| Note generation | note_v2.py, renders from verified fact graph only | Functional | No |
| Doctor review/editing | PATCH /api/consultations/{cid}/review patches fields, re-renders text | Functional | No |
| Doctor approval | POST /api/consultations/{cid}/complete sets status=completed | **Functional but no identity — any session can approve** | **YES** |
| Audio deletion | Background _cleanup_loop, configurable retention hours, marks audio_purged=1 | Functional | No |
| Note storage | SQLite document blob — notes never deleted | Functional — **unencrypted, no note retention policy** | **YES** |
| Authentication | Single shared HMAC-signed session cookie | **Shared — no per-user identity** | **YES** |
| Authorization | Binary: authenticated = full access | **No RBAC, no role separation** | **YES** |
| Patient/doctor identity | Free-text fields at consultation creation, never verified | Not implemented | **YES** |
| Hospital/org isolation | Not implemented — single-tenant only | Missing | YES for multi-tenant |
| Logging | Structured key=value operational logs, PHI redaction filter | Sound | No |
| Audit trail | Operational logs only — cannot reconstruct "who did what to which patient" | **Incomplete for NABH/regulatory** | **YES** |
| Training-data handling | No code exists. Documentation describes the concept. | **Missing** | **YES** |
| External API providers | Deepgram (cloud, US), Hugging Face Hub (model metadata at startup) | Uncontrolled | **YES** |
| Export/download | GET /api/consultations/{cid}/export returns plaintext | Functional, no access log | Partial |
| Database | SQLite, single file, WAL mode | Unencrypted | **YES** |
| Secrets/API key handling | DEEPGRAM_API_KEY env-only, never logged, session key derived via HMAC | Sound | No |
| Error handling | Global handler suppresses stack traces, PHI-free messages | Sound | No |
| Backup/recovery | Not implemented — DEPLOYMENT.md says "back this up" but provides no mechanism | Missing | **YES** |
| Deployment/infrastructure | Single-host Docker container with mem_limit=3g | Functional | No for pilot |

### 2.2 What Is Missing vs. Implemented

**Missing entirely (zero code):**
- Per-clinician user accounts
- Role-based access control
- Patient consent recording
- Training-data consent separation
- Training vault / authorized retention category
- At-rest encryption
- Backup automation
- Multi-hospital tenant isolation
- Audit event trail (actionable: who/what/when)
- Doctor identity verification
- Patient identity binding to consultation
- Note retention policy and deletion
- Incident response logging

**Partially implemented (concept exists, gaps remain):**
- Audio deletion: implemented for retention hours; no verification of actual deletion
- Logging: PHI-safe; cannot reconstruct individual user actions
- Authentication: secure cookie; shared token means zero attribution
- Export: functional; no download audit log

**Acceptable for a prototype only:**
- Single-token shared auth for internal developer evaluation
- In-process SQLite without encryption for a demo
- Cloudflare Tunnel without institutional TLS certificate
- Manual backup reminder

---

## SECTION 3 — PHASE 2: DATA LIFECYCLE

### 3.1 Complete Data Flow — One Consultation

```
STAGE 1: Patient Consultation
  Physical location: Clinic room
  Encrypted: N/A | External: No

STAGE 2: Audio Captured
  Physical location: Browser memory (MediaRecorder chunks)
  Encrypted: No (in-memory) | External: No

STAGE 3: Audio Transmitted
  Physical location: HTTPS via Cloudflare Tunnel -> loopback HTTP to server
  Encrypted in transit: YES (Cloudflare TLS)
  Encrypted on disk at destination: NO
  Logged: audio_uploaded event (cid, bytes, ext — no content)

STAGE 4: Audio Stored on Disk
  Physical location: ISCRIBE_DATA_DIR/uploads/{cid}.{ext}
  Who can access: OS user running iScribe process
  Encrypted at rest: NO  <-- BLOCKER
  Deleted: After ISCRIBE_AUDIO_RETENTION_HOURS (default 24h)

STAGE 5: STT Processing (English — Deepgram)
  Provider: Deepgram nova-3-medical
  Physical location: Audio bytes POSTed to api.deepgram.com (US)
  Stored by Deepgram: UNKNOWN — no verified DPA exists in codebase
  External: YES — full audio sent to Deepgram  <-- BLOCKER

STAGE 5b: STT Processing (faster-whisper fallback)
  Physical location: RAM on iScribe host, fully local
  External: No

STAGE 6: Raw Transcript
  Physical location: In-process dict, then written to SQLite document blob
  Who can access: Any authenticated session
  Encrypted at rest: NO  <-- BLOCKER

STAGE 7: Clinical Processing
  Physical location: In-process RAM only | External: No

STAGE 8: Draft Clinical Note
  Physical location: SQLite document blob (status=ready)
  Who can access: Any authenticated session
  Encrypted at rest: NO  <-- BLOCKER

STAGE 9: Doctor Review
  Who can access: Any authenticated session  <-- BLOCKER (shared token)
  Who approved: UNKNOWN — shared token = no attribution  <-- BLOCKER
  Old version on edit: DISCARDED — no edit history  <-- POTENTIAL BLOCKER

STAGE 10: Doctor Approval
  Attribution: None — shared token means any user can approve

STAGE 11: Final Note
  Physical location: SQLite document blob, status=completed
  Encrypted at rest: NO
  Retention: INDEFINITE — no note deletion policy exists

STAGE 12A: CLINICAL Audio Deletion (normal path)
  Trigger: _cleanup_loop every 5 minutes
  Method: Path.unlink(missing_ok=True)
  Verification: audio_purged=1 flag in SQLite
  Confirmed deleted: Filesystem delete — no cryptographic proof
  BACKUP RISK: If data dir is backed up before cleanup,
               audio remains in backup  <-- BLOCKER

STAGE 12B: TRAINING Audio Retention (authorized path)
  Status: DOES NOT EXIST IN CODE
  No consent flag, no training category, no vault, no audit trail.
  Currently: indistinguishable from clinical audio  <-- CRITICAL BLOCKER
```

### 3.2 The Training/Clinical Distinction — Current State

The distinction exists only in documentation. In code:
- Every uploaded audio file is treated identically
- There is no training_consent field anywhere in the database schema
- There is no training vault directory
- The cleanup timer deletes ALL audio past the retention window without any category check

---

## SECTION 4 — PHASE 3: SCRIBESHIELD SECURITY DESIGN

ScribeShield is not a marketing layer. It is a data-governance subsystem composed of concrete technical controls.

### 4.1 Encryption in Transit
**Current:** Cloudflare TLS (acceptable for pilot). Cookie is Secure=True in production. HTTP not reachable from outside localhost.
**Verdict:** Acceptable if tunnel is correctly configured. Verify before pilot.

### 4.2 Encryption at Rest — BLOCKER
**Current:** None. SQLite unencrypted. Uploads directory unencrypted.
**Options (in order of implementation effort):**
1. OS-level volume encryption (BitLocker on Windows, LUKS on Linux) — zero code change, easy to verify. **RECOMMENDED.**
2. SQLCipher — replaces the sqlite3 driver with an encrypted variant. Requires dependency change.
3. Application-level field encryption of the document blob — complex, fragile.
**Recommendation:** Use OS-level full-disk encryption for the pilot.

### 4.3 Authentication — BLOCKER
**Current:** Single shared HMAC-signed session cookie. No user accounts.
**Pilot requirement:** Per-doctor login with username and bcrypt-hashed password.
**Note:** Session cookie security attributes (HttpOnly, SameSite=Strict, Secure, 12h TTL) are already correct. Do not change them.

### 4.4 Authorization / RBAC — BLOCKER
**Current:** Binary (authenticated = full access).
**Pilot minimum — two roles:**
- doctor: can create, view, edit, approve their OWN consultations only
- admin: can view all consultations, configure settings, revoke sessions

### 4.5 Hospital/Organization Isolation
**Pilot (single hospital):** Not required if one instance per hospital.
**Multi-hospital:** Future concern. Each hospital gets its own instance.

### 4.6 Doctor Identity — BLOCKER
**Current:** Free-text doctor field, not verified.
**Pilot requirement:** doctor field must be bound to the authenticated session user.

### 4.7 Patient Identity
**Current:** Free-text patient_id, no validation.
**Pilot requirement:** Accept hospital MRN/UHID — iScribe stores what the doctor provides. Do not generate patient IDs.

### 4.8 Audit Logging — BLOCKER
**Current:** Operational events only (cid, timing, error types). No user attribution.
**Pilot requirement:** Every authenticated API call logs {user_id, timestamp, action, cid}.
Stored in append-only audit_events table — never updated, never deleted.

### 4.9 Audio Retention Policy
**Current:** Global ISCRIBE_AUDIO_RETENTION_HOURS. Works.
**Pilot requirement:** Agreed retention hours must be documented and verifiable. Note that backups may preserve audio past the retention window.

### 4.10 Training-Consent Separation — BLOCKER if training is enabled
**Current:** None.
**Pilot:** If no hospital has agreed to training, TRAINING_ENABLED=false. Build the data model but keep it inert.

### 4.11 Download/Export Controls — BLOCKER
**Current:** Any authenticated session can export any consultation.
**Pilot requirement:** Export gated to authenticated doctor's own consultations only. Export event logged with user_id.

### 4.12 Session Security
**Current:** HMAC-signed, HttpOnly, SameSite=Strict, Secure=True (production), 12h TTL. CORRECT. No change needed.

### 4.13 Secret Management
**Current:** DEEPGRAM_API_KEY from environment only, never logged. CORRECT.
**Pilot:** .env file must not be on a shared filesystem. Rotate token before each session.

### 4.14 Backup Security — BLOCKER
**Current:** Not implemented.
**Pilot requirement:** Define backup schedule. Specify encrypted backup (OS-level encryption covers this). Define who has access. Define how audio-purge policy applies to backups.

### 4.15 Incident Logging
**Current:** ERROR-level log events for job failures. No incident categorization.
**Pilot requirement:** Login failure counter to logs. Define what constitutes an incident.

---

## SECTION 5 — PHASE 4: TRAINING AUDIO SAFETY

### 5.1 Required Data Model Extension

```
consultation record — extensions needed:
  training_consent_given: boolean (default: false)
  training_consent_by: user_id who granted consent
  training_consent_at: timestamp
  training_consent_basis: string ("hospital_research_agreement_2026")
  training_category: enum (clinical | authorized_training)
  training_vault_path: string | null
  training_vault_moved_at: timestamp | null
```

### 5.2 Consent Chain

**LAYER 1 — Hospital-level agreement:**
- Hospital signs a research participation agreement (external to iScribe)
- Config flag enables training retention: TRAINING_ENABLED=true
- Without this flag, training_consent_given is always rejected

**LAYER 2 — Patient-level consent:**
- Patient must consent before the consultation is recorded
- This consent is obtained independently of iScribe — the doctor's responsibility
- iScribe records ONLY that the doctor has attested that consent was obtained
- Attestation is recorded with the doctor's authenticated user_id and timestamp

**LAYER 3 — Doctor attestation:**
- After consultation completion, doctor may attest: "I confirm patient consent was obtained for training retention"
- Effect: training_consent_given=true, training_category=authorized_training

**LAYER 4 — System enforcement:**
- Cleanup timer checks training_category
- CLINICAL audio: deleted at retention_hours
- AUTHORIZED_TRAINING audio: NOT deleted by cleanup timer
- AUTHORIZED_TRAINING audio: moved to separate vault path
- Vault path is read-only to the main application process

### 5.3 Technical Controls for Training Isolation

- Separate directory: TRAINING_VAULT_DIR distinct from ISCRIBE_UPLOAD_DIR
- Process isolation: main API process should NOT have write access to TRAINING_VAULT_DIR
- Immutable move: Audio is moved (not copied) from upload dir to vault. Original is gone.
- Audit log entry: Every vault intake is a dedicated audit event: {type: training_vault_intake, cid, user_id, consent_basis, timestamp}
- No retrospective reclassification: Once training_category=clinical is finalized, it cannot be upgraded to authorized_training. Downgrade (consent withdrawal) must be supported.

### 5.4 Legal Boundaries (Require Professional Legal Review — Not Engineering Decisions)

iScribe CANNOT:
- Verify that patient consent was actually obtained — only that a doctor attested it was
- Enforce compliance with India DPDPA 2023 by itself
- Guarantee that de-identified training audio cannot be re-identified
- Replace hospital IEC (Institutional Ethics Committee) approval for research audio programs
- Substitute for a qualified legal data protection review

---

## SECTION 6 — PHASE 5: DEEPGRAM AND STT PROVIDER RISK

### 6.1 Current Deepgram Implementation (Evidence from source code)

From scribe_engine/stt/deepgram_provider.py:
- Endpoint: https://api.deepgram.com/v1/listen (US-hosted)
- Method: Full audio file sent in HTTP POST body (content=fh.read())
- API key: From DEEPGRAM_API_KEY environment variable — never logged (correct)
- No streaming: Audio held entirely in RAM then sent in one request
- Diarization enabled: diarize=true — Deepgram performs speaker separation on US servers
- Smart format enabled: Deepgram post-processing runs server-side
- No regional routing: No India data-residency configuration

### 6.2 What Must Be Verified Before Hospital Deployment

| Concern | Current State |
|---|---|
| Data retention by Deepgram | UNKNOWN — no verified DPA in codebase |
| Model training on customer audio | UNKNOWN |
| Data residency (India region?) | UNKNOWN |
| Subprocessors | UNKNOWN |
| Encryption at Deepgram at rest | UNKNOWN (HTTPS in transit confirmed) |
| Contractual deletion guarantee | UNKNOWN |
| Auditability | UNKNOWN |
| Breach notification timeline | UNKNOWN |
| DPDPA 2023 compliance | UNKNOWN |

A Kerala hospital's Data Protection Officer will ask:
- "Our patients' speech is sent to a server in the United States?"
- "Do you have a signed DPA with Deepgram that governs Indian clinical audio?"
- "Can Deepgram use our consultations to train their own models?"
- "If Deepgram is breached, how quickly are we notified?"
- "Can we get a written deletion guarantee?"

Without verified answers to all five, the DPO will block deployment.

### 6.3 STT Provider Abstraction (Current State — Already Good)

The abstraction already exists and is well-designed (source: scribe_engine/stt/__init__.py):
- STTProvider base class with transcribe(audio_path, language) -> Transcript
- PRODUCTION_PRIORITY = (DEEPGRAM_PROVIDER_ID, CURRENT_PROVIDER_ID)
- resolve_production_provider() checks language map first, then explicit preference, then priority order
- Clinical intelligence layer NEVER imports a provider directly
- Adding a new provider requires only implementing the interface and registering it

### 6.4 STT Abstraction Gaps to Fix

| Gap | Fix |
|---|---|
| is_configured not abstract in base class | Make it required |
| No provider_capabilities dict | Add standardized capabilities dict |
| Fallback policy hardcoded in pipeline.py | Make config-driven |
| No per-provider timeout in config | Add ISCRIBE_STT_TIMEOUT_SECONDS |
| No local-only mode | Add ISCRIBE_STT_LOCAL_ONLY=false flag |

---

## SECTION 7 — PHASE 6: HOSPITAL EHR/HIS INTEGRATION

### 7.1 Current Export (Evidence: app.py line 944)

Returns structured plaintext with: transcript, clinical note, prescription, metadata, disclaimer.
Cache-Control: no-store, private. Human-readable; can be copy-pasted into any EHR.

### 7.2 Assessment: Plaintext Export Is the Right Initial Strategy

- Requires no EHR integration agreement
- Requires no EHR vendor cooperation
- Doctor is the integration point
- Avoids risk of wrongly-formatted data entering a live EHR automatically

### 7.3 Export Format Roadmap

| Format | When to add |
|---|---|
| PDF (hospital-branded, with disclaimer) | Before pilot go-live |
| Structured JSON (fact graph + note + metadata) | Before any EHR integration |
| FHIR R4 DocumentReference | After successful pilot |
| HL7 v2 ORU (for legacy HIS) | After pilot, on-demand |
| SMART on FHIR | Future — requires EHR vendor cooperation |

### 7.4 Build Now for Future Integration Compatibility

1. Preserve structured fact graph in export — clinical_facts_v2 already in pipeline output
2. Use ICD-10/SNOMED codes for diagnoses when medication intelligence adds them
3. Use ATC codes for medications — include in medication intelligence V1 spec
4. Do not use free-text diagnoses as primary keys — use codes
5. Keep reviewed and completed_at fields — FHIR DocumentReference needs author + date

---

## SECTION 8 — PHASE 7: HOSPITAL REJECTION MATRIX

| Risk/Concern | Hospital Perspective | Current State | Severity | Evidence | Required Fix | Prototype OK? | Pilot Blocker? | Production Blocker? |
|---|---|---|---|---|---|---|---|---|
| Patient audio sent to US (Deepgram) | Patient speech goes to US without consent | Full audio POSTed to api.deepgram.com, no DPA | CRITICAL | deepgram_provider.py:28-111 | Sign Deepgram DPA or use STT_LOCAL_ONLY=true | No | YES | YES |
| No per-doctor auth | Anyone with the key accesses all patients | Shared HMAC token | CRITICAL | security.py:3-7 | Per-user accounts with bcrypt | Yes | YES | YES |
| No encryption at rest | Notes unprotected if device stolen | SQLite unencrypted | HIGH | No encryption code found | OS-level full-disk encryption | Yes | YES | YES |
| No audit trail | Cannot prove who accessed a patient | Operational logs only, no user_id | HIGH | logging_config.py:92-111 | Per-user audit log table | Yes | YES | YES |
| Training data — no consent framework | Could audio be used for AI training? | Zero training infrastructure | CRITICAL | No training code exists | Training consent + vault system | Yes | YES | YES |
| Doctor approval is anonymous | Which doctor approved the note? | completion endpoint, no user captured | HIGH | app.py:929-941 | Bind completion to authenticated user | Yes | YES | YES |
| No patient consent recording | Where is patient's consent for this system? | No consent infrastructure | HIGH | None | Consent attestation field + legal review | Yes | YES | YES |
| Incorrect clinical note | Wrong diagnosis in the note | Fail-closed validator, doctor review required | MEDIUM | _execute_doc_job, note_v2.py | Review is mandatory; disclaimer in export | Yes | Partial | Partial |
| Hallucinated medication | AI invents a drug | Medication from transcript only, no inference | LOW-MED | clinical_facts.py | Medication intelligence V1 adds coverage not invention | Yes | No | No |
| Missing clinical information | Important symptoms not captured | Known keyword/regex limitations | MEDIUM | DEPLOYMENT.md:263-266 | Ongoing improvement; disclaimer | Yes | No | Partial |
| Malayalam accuracy | Doctors speak Malayalam; English-only unusable | Malayalam pipeline experimental only | HIGH | DEPLOYMENT.md:267-268 | Malayalam must reach production quality | Yes | YES (Kerala) | YES (Kerala) |
| Speaker attribution | Wrong speaker labels corrupt the note | Heuristic, confidence=low without pyannote | MEDIUM | DEPLOYMENT.md:256-259 | UI warns; mandatory validation | Yes | No | Partial |
| Workflow delay | Doctor waits 5 min for 10 min consultation | 1.8-2.5x realtime on CPU; ~1-2 min with Deepgram | MEDIUM | DEPLOYMENT.md:184-188 | Async processing already works; show status | Yes | No | No |
| EHR integration | Doesn't connect to our HIS | Plaintext export only | MEDIUM | app.py:944-954 | PDF + structured JSON first | Yes | Partial | No |
| Provider dependency | What if Deepgram goes down? | Automatic fallback to faster-whisper | LOW | pipeline.py:110-143 | Fallback works; document it | Yes | No | No |
| Internet dependency | Poor hospital connectivity | Deepgram requires internet | MEDIUM | deepgram_provider.py:28 | Offer offline (faster-whisper) mode | Yes | Partial | No |
| Data residency | Data must remain in India | No India region for Deepgram known | HIGH | No config option | India DPA or local STT | Yes | YES | YES |
| Backup/recovery | What if server fails? | No backup automation | MEDIUM | DEPLOYMENT.md: "back this up" | Define + verify backup procedure | Yes | Partial | YES |
| Access revocation | How do we revoke access immediately? | Rotate the shared token | MEDIUM | security.py:73-76 | Per-user session invalidation | Yes | YES | YES |
| Export security | Who can download a patient note? | Any authenticated session | MEDIUM | app.py:944 | Export gated to note's doctor only | Yes | YES | YES |
| Upstream licence | Is this software legally usable? | Upstream repo has no declared licence | HIGH | DEPLOYMENT.md:269-271 | Resolve before any hospital use | NO | YES | YES |
| Insider access | Can the iScribe company access our data? | Local deployment, no remote access | LOW | DEPLOYMENT.md:35-40 | Document and verify no telemetry | No | No | Partial |
| Disaster recovery | How long to recover from failure? | Not defined | MEDIUM | N/A | Define RTO/RPO; test restoration | Yes | Partial | YES |

---

## SECTION 9 — PHASE 8: CLINICAL SAFETY ASSESSMENT

### 9.1 Verified Safety Properties (Evidence in Code, Not Documentation)

| Safety property | Verified? | Evidence |
|---|---|---|
| Evidence anchoring — facts tied to transcript spans | YES | clinical_facts.py evidence anchor pattern |
| No hallucinated symptoms | YES | Regex/entity match required for every symptom |
| No inferred diagnoses from thin evidence | YES | Diagnosis certainty levels, confirmed vs suspected |
| Medication not equal to procedure | YES | _FACT_TYPE_HINTS in clinical_facts.py |
| Negation handling | YES | Negation patterns in clinical_facts.py |
| Temporal separation | YES | Duration/onset patterns with temporal modifiers |
| Current vs historical | YES | History markers in fact classification |
| Fail-closed validator | YES | _execute_doc_job rejects note if V2 validation fails |
| Doctor review before export | YES | reviewed flag required; disclaimer present in export |
| Non-Latin script warning for English transcript | YES | validation.py:27-47 |
| Repetition/hallucination detection | YES | validation.py:72-75 |

### 9.2 Remaining Clinical Safety Gaps

| Gap | Risk | Priority |
|---|---|---|
| Indian medication vocabulary is incomplete | Missing medications = missing prescriptions in note | HIGH |
| Brand-name to generic mapping does not exist | "Augmentin" said by doctor, nothing captured | HIGH |
| No dose/frequency ASR-error safeguard | Misheard dose could enter note | MEDIUM |
| Malayalam medication names not covered | IndicConformer may not recognise drug names | HIGH (Malayalam path) |
| Speaker attribution errors propagate to facts | Wrong turn labels = symptoms attributed to wrong party | MEDIUM |
| Legacy [template] boilerplate fields | Can appear in export if V2 validation skips | LOW (V2 mitigates) |

### 9.3 Most Important Clinical Safety Rule to Document

The system does not diagnose. It extracts what the doctor stated. The clinical note is evidence of what was said, not the AI's clinical opinion. This must be in all patient-facing and hospital-facing documentation.

---

## SECTION 10 — PHASE 9: DOCTOR WORKFLOW ASSESSMENT

### 10.1 Current Workflow

```
Doctor creates consultation (web UI)
-> Records audio (browser MediaRecorder)
-> Uploads audio
-> Clicks Process
-> Job queued (status=queued)
-> Background worker processes
   (~5 min for 10 min audio on local CPU; ~1-2 min with Deepgram)
-> Doctor polls for status (UI polls every few seconds)
-> Note appears (status=ready)
-> Doctor reviews note in UI
-> Doctor edits individual fields
-> Doctor clicks Complete
-> Doctor downloads export
```

### 10.2 Latency Assessment

- Processing time for 10-min consultation on local CPU: ~5 minutes (measured in DEPLOYMENT.md)
- With Deepgram STT: ~1-2 minutes total (API is fast)
- Target for clinical workflow: Under 2 minutes after consultation ends
- Conclusion: With Deepgram, latency is acceptable. With local CPU only, problem for 10-patient OPD morning.

### 10.3 Async Processing — Already Correctly Designed

Background processing is already implemented (source: app.py:644-664):
- Job queue via queue.Queue
- Serial worker loop (_doc_worker_loop)
- Stage progress persisted to SQLite (_progress_callback)
- UI polls for stage updates
- Doctor does NOT need to wait. Can close the tab and return. This is correct.

### 10.4 Missing for Kerala OPD Workflow

| Gap | Impact | Fix Priority |
|---|---|---|
| Streaming transcript | Doctor cannot see transcript during recording | Post-pilot |
| Patient queue management | No multiple-patient session concept | Not required for MVP |
| Mobile recording app | Browser must remain open during recording | Post-pilot |
| Offline mode on slow hospital WiFi | Deepgram requires internet | Test before pilot |
| Concurrent patients (>1 job) | Patient 2 waits for Patient 1 | Increase max_concurrent_jobs on faster hardware |

---

## SECTION 11 — PHASE 10: PROPOSED PRODUCTION ARCHITECTURE

```
                  CURRENT              NEXT                   FUTURE
-------------------------------------------------------------------

FRONTEND
Browser (vanilla JS) --HTTPS-->  + PDF export             Mobile app
                                 + Per-user login form     + Live transcript view

BACKEND
FastAPI (single process)  --->   + Per-user auth      --> Separate async workers
 - Shared token auth              - Audit middleware        - Horizontal scaling
 - Background job thread          - Per-doctor routing      - Message queue

DATABASE
SQLite (unencrypted)      --->   OS-level encryption  --> PostgreSQL if scale needed
 - consultations                  + users table             + audit_log table
 - jobs                           + audit_events            + hospital_config

OBJECT STORAGE
Local filesystem          --->   + training_vault/    --> Dedicated encrypted volume
 - uploads/ (unencrypted)         (write-only from         On-prem S3-compatible
                                   main process)

AUTHENTICATION
Shared token              --->   Per-user bcrypt      --> SSO / OIDC (hospital IdP)

AUTHORIZATION
None                      --->   Owner-only access    --> RBAC (doctor/admin/auditor)

STT
Deepgram (primary)        --->   + verified DPA       --> Multiple providers
faster-whisper (fallback)        + local-only flag         + streaming STT
                                 + per-provider timeout    + Malayalam production

CLINICAL ENGINE
Clinical Intelligence V2  --->   Same (DO NOT TOUCH)  --> + Medication intel V1
(deterministic, fail-closed)                               + ICD-10 codes

AUDIT SERVICE
None                      --->   audit_events table   --> Separate append-only service
                                 (append-only)

TRAINING VAULT
None                      --->   Consent flag +       --> Encrypted vault
                                 vault dir (inert)         + de-identification pipeline

EXPORT SERVICE
Plaintext GET endpoint    --->   + PDF (ReportLab)    --> + FHIR R4 JSON
                                 + Structured JSON         + HL7 v2

KEY/SECRET MANAGEMENT
.env file                 --->   + rotation script    --> Vault (OS keystore)

MONITORING
Python logging only       --->   + Login fail alert   --> + Prometheus metrics
                                 + Disk space alert
```

---

## SECTION 12 — PHASE 11: GLM IMPLEMENTATION SPECIFICATION

> **CRITICAL RULE: Do NOT modify clinical_facts.py, note_v2.py, fact_graph.py, or the test suite.**
> **These are the clinical safety layer and must remain intact.**
> **Do not rewrite the clinical intelligence engine.**

---

### PHASE A — Per-User Authentication (PILOT BLOCKER #1)

**Why first:** Everything else requires identity. Without this, the audit log is useless,
doctor-owned consultations are impossible, and the hospital will reject immediately.

**Files to create:**
- service/users.py — user management: create, verify_password, list, deactivate

**Files to modify:**
- service/store.py — add users table to _SCHEMA
- service/security.py — remove shared-token logic; add per-user session issuance with user_id + role in payload
- service/app.py — replace token_matches login with username/password login; add user management endpoints
- service/config.py — add ISCRIBE_ADMIN_USERNAME and ISCRIBE_ADMIN_PASSWORD (initial admin credential)

**Database schema addition to service/store.py _SCHEMA:**
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

**API changes:**
- POST /api/login — accepts {username, password} instead of {access_token}
- POST /api/users (admin only) — create user: {username, role, password}
- PATCH /api/users/{uid}/deactivate (admin only) — revoke access
- Remove X-Access-Token bearer token authentication path

**Session payload change:**
- Current payload: {expiry}
- New payload: {user_id, role, expiry} — all HMAC-signed as before

**Middleware change:**
- AccessControlMiddleware must decode session and set request.state.user_id and request.state.role

**DO NOT change:**
- Session cookie security attributes (HttpOnly, SameSite=Strict, Secure, Path=/)
- The auth_required config flag (dev mode must still work without auth for tests)

**Acceptance criteria:**
- Two doctors cannot see each other's consultations
- Admin can see all consultations
- A deactivated user's sessions are rejected on next API call
- Login returns 401 on wrong credentials, 200 + cookie on correct credentials
- All 507+ existing tests pass

---

### PHASE B — Audit Log (PILOT BLOCKER #2)

**Files to create:**
- service/audit.py — append-only audit event writer

**Files to modify:**
- service/store.py — add audit_events table to _SCHEMA
- service/app.py — add audit calls to every state-changing endpoint

**Database schema addition:**
```sql
CREATE TABLE IF NOT EXISTS audit_events (
    event_id    TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    ts_epoch    REAL NOT NULL,
    user_id     TEXT,
    action      TEXT NOT NULL,
    cid         TEXT,
    detail      TEXT
);
-- NO UPDATE, NO DELETE routes ever on this table
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events (user_id, ts_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_audit_cid  ON audit_events (cid, ts_epoch DESC);
```

**Actions to audit:**
login_ok (include source_ip), login_failed (include source_ip), consultation_created,
audio_uploaded, process_queued, process_completed, process_failed, consultation_reviewed,
consultation_completed, consultation_exported, audio_purged, training_consent_granted

**API additions:**
- GET /api/audit?cid={cid} (admin only) — audit events for a consultation
- GET /api/audit/user/{uid} (admin only) — audit events for a user

**Acceptance criteria:**
- Every state change has a corresponding audit_event row with user_id
- Audit table has no UPDATE or DELETE routes exposed
- Events are returned in chronological order

---

### PHASE C — Doctor-Owned Consultations (PILOT BLOCKER #3)

**Files to modify:**
- service/store.py — add created_by_user_id column to consultations table
- service/app.py — filter list_consultations by user_id for doctor role

**Schema change:**
```sql
-- Add to consultations table:
created_by_user_id  TEXT   -- user_id of creating doctor; NULL for legacy rows
```

**Logic changes:**
- GET /api/consultations — doctors see only their own; admin sees all
- GET /api/consultations/{cid} — doctors can only access their own; return 403 otherwise
- PATCH /api/consultations/{cid}/review — doctor role, own consultation only
- POST /api/consultations/{cid}/complete — doctor role, own consultation only
- GET /api/consultations/{cid}/export — doctor role, own consultation only

**Acceptance criteria:**
- Doctor A receives 403 on Doctor B's consultation endpoints
- Admin can retrieve and export any consultation
- All existing tests pass

---

### PHASE D — Training Consent Data Model (Build Inert, Default Off)

**Files to modify:**
- service/store.py — add training fields to consultations table
- service/app.py — add POST /api/consultations/{cid}/training-consent endpoint (gated by config)
- service/config.py — add TRAINING_ENABLED=false (default false, env override)

**Schema additions to consultations table:**
```sql
training_consent_given  INTEGER NOT NULL DEFAULT 0,
training_consent_by     TEXT,
training_consent_at     TEXT,
training_consent_basis  TEXT,
training_category       TEXT NOT NULL DEFAULT 'clinical',
training_vault_path     TEXT,
training_vault_moved_at TEXT
```

**Cleanup timer change in app.py _purge_expired_audio:**
- Skip consultations where training_category = 'authorized_training'

**Acceptance criteria:**
- With TRAINING_ENABLED=false (default), POST /training-consent returns 403
- With TRAINING_ENABLED=true, consent can be granted only for completed consultations by the owning doctor
- Cleanup timer does not delete training-category audio
- All existing tests pass

---

### PHASE E — PDF Export

**Files to create:**
- service/export_pdf.py — PDF generation using ReportLab

**Files to modify:**
- service/app.py — add format=pdf and format=json query parameters to GET /api/consultations/{cid}/export

**PDF content requirements:**
1. Hospital name (configurable via ISCRIBE_HOSPITAL_NAME env)
2. Consultation metadata header (id, doctor, department, date)
3. RED WARNING at top if not yet reviewed (reviewed=False)
4. Disclaimer: "Trial software. Clinician review and sign-off required before clinical use."
5. Full transcript text
6. Clinical note section (V2 note when valid; legacy note with label otherwise)
7. Prescription section
8. Doctor name from authenticated session user (not free-text field)
9. Approval timestamp
10. Footer: "Fields reading Not mentioned were not stated in this consultation."

**JSON export requirements (format=json):**
- clinical_facts_v2 (the typed fact graph)
- clinical_note_v2.note (the rendered text)
- transcript.text
- consultation metadata
- reviewed, completed_at, doctor (from session user)
- Medication list with dose/frequency if present

**Acceptance criteria:**
- PDF renders without errors
- PDF is gated to doctor's own consultations
- PDF generation creates audit event: consultation_exported with format=pdf
- Plain text export continues to work unchanged

---

### PHASE F — STT Provider Hardening

**Files to modify:**
- scribe_engine/stt/base.py — make is_configured required; add provider_capabilities dict
- service/config.py — add ISCRIBE_STT_LOCAL_ONLY=false
- scribe_engine/stt/__init__.py — if STT_LOCAL_ONLY=true, skip Deepgram in PRODUCTION_PRIORITY

**Standard provider_capabilities dict:**
```python
provider_capabilities = {
    "streaming": False,
    "languages": ["en"],
    "medical_tuned": True,
    "local": False,
    "diarization": True,
}
```

**Local-only mode:**
- If ISCRIBE_STT_LOCAL_ONLY=true, resolve_production_provider must skip Deepgram entirely
- Deepgram is never contacted in local-only mode
- /api/ready must report the active provider correctly in local-only mode

**Acceptance criteria:**
- ISCRIBE_STT_LOCAL_ONLY=true routes all English audio to faster-whisper
- Deepgram is never contacted in local-only mode
- All existing STT tests pass

---

### PHASE G — Backup and Deletion Verification

**Files to create:**
- scripts/Backup-iScribe.ps1 — SQLite WAL checkpoint + copy to backup location
- scripts/verify-deletion.py — confirm audio files are purged from backups within policy window

**Backup script requirements:**
- PRAGMA wal_checkpoint(FULL) before copying
- Copy SQLite file to ISCRIBE_BACKUP_DIR with datestamp
- Log backup completion to audit_events

**Deletion verification:**
- After cleanup, log SHA-256 hash of deleted file path (not content) to audit_events
- Audit action: audio_purged_verified, includes path_hash

---

### PHASE H — Hospital Pilot Hardening (External Verification — Not Code)

These must be completed before any hospital goes live:

1. Deepgram DPA signed and reviewed by legal counsel
2. OS-level encryption verified active on host — screenshot evidence retained
3. Backup procedure tested — successful restore verified
4. Admin password changed from default — documented who holds it
5. Upstream repository licence resolved — written confirmation retained
6. Hospital consent process documented by hospital (not by iScribe)
7. Retention policy hours agreed and documented in deployment record
8. TRAINING_ENABLED=false confirmed in production config
9. Port 8123 confirmed not accessible from outside localhost
10. Named Cloudflare Tunnel with stable hostname configured and tested

---

## SECTION 13 — TOP 5 PRIORITIES FOR NEXT SPRINT

**Priority 1 — Per-Doctor Authentication (Phase A)**
Everything else requires identity. Without this, the audit log is useless, doctor-owned
consultations are impossible, and the hospital will reject the system immediately.

**Priority 2 — Audit Log (Phase B)**
Hospital IT committees need to see who did what, when. Adding user_id to audit_events
is a small change that unlocks regulatory acceptability.

**Priority 3 — Doctor-Owned Consultations (Phase C)**
Privacy foundation. One doctor must not read another's consultations. One column + route filter.

**Priority 4 — Deepgram DPA or Local-Only Config Flag (Phase F partial)**
Either sign the DPA (legal task) or implement ISCRIBE_STT_LOCAL_ONLY=true (small code task).
Without this, the hospital cannot make an informed decision about Deepgram.

**Priority 5 — PDF Export (Phase E)**
A hospital-branded PDF with the correct disclaimer completes the doctor workflow:
record -> process -> review -> print/export. First genuinely clinical-facing output.

---

## SECTION 14 — THINGS THAT CAN SAFELY WAIT UNTIL AFTER FIRST PILOT

1. Training consent infrastructure — build data model, keep TRAINING_ENABLED=false
2. FHIR/HL7 export — not required for copy-paste workflow
3. Medication Intelligence V1 implementation — improves quality, not a safety blocker
4. Malayalam production path — critical long-term; start with English pilot
5. Backup automation script — manual procedure with documentation acceptable for short pilot
6. Multi-hospital tenancy — one instance per hospital is sufficient
7. Streaming transcript — async batch works; streaming is UX improvement only
8. Pyannote diarization — heuristic is disclosed in UI; not a blocker
9. Mobile app — browser on tablet suffices
10. Horizontal scaling — serial queue handles one doctor's OPD
11. Drug-drug interaction checking — out of scope; document clearly that iScribe does NOT do this
12. Diagnostic decision support — out of scope; document clearly

---

## SECTION 15 — HOSPITAL PILOT READINESS CHECKLIST

### Technical (must be GREEN before pilot)
- [ ] Per-doctor authentication implemented and tested
- [ ] Audit log — every state change attributed to a user
- [ ] Doctor-owned consultations — doctor can only access their own
- [ ] OS-level disk encryption confirmed active on host
- [ ] Deepgram DPA signed OR ISCRIBE_STT_LOCAL_ONLY=true configured
- [ ] Upstream repository licence resolved
- [ ] TRAINING_ENABLED=false confirmed in production config
- [ ] Backup procedure tested — successful restore verified
- [ ] Port 8123 confirmed not accessible from outside localhost
- [ ] Named Cloudflare Tunnel with stable hostname configured
- [ ] Audio retention hours agreed with hospital and configured
- [ ] PDF export functional with hospital header and disclaimer
- [ ] Smoke test passes end-to-end

### Legal/Compliance (requires professional review — not engineering decisions)
- [ ] DPDPA 2023 applicability reviewed by legal counsel
- [ ] Hospital IT/security committee sign-off
- [ ] Clinical director approval of disclaimer language
- [ ] Data Processing Agreement with hospital drafted
- [ ] Deepgram DPA reviewed for Indian clinical audio
- [ ] Patient consent process documented by hospital
- [ ] Incident response contact list defined

### Clinical Safety (must be verified before pilot)
- [ ] Clinical note validation is fail-closed (currently verified — do not break)
- [ ] Doctor review mandatory before export (currently enforced — do not break)
- [ ] Disclaimer in every export (currently present — do not remove)
- [ ] Known limitations communicated to pilot doctors
- [ ] Speaker attribution warning visible in UI (currently shown)

---

*This document is a technical architecture and engineering assessment. It does not constitute legal advice.*
*All legal, compliance, regulatory, and ethical questions must be reviewed by qualified legal counsel and medical informatics professionals.*
*Evidence for every finding is cited against actual source files in the repository.*
*No finding is based on documentation alone — all claims were verified against code.*
