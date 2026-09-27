# STT Provider Privacy & Security — Evidence-Based Audit

Retrieved and verified against official provider documentation on **2026-09-26**.
Every claim is labelled:

- **[CONFIRMED-CODE]** — verified in this repository's source.
- **[PROVIDER-DOC]** — stated in the provider's official documentation (link given); a documentation claim, not a contract we hold.
- **[CONTRACT-REQUIRED]** — must be obtained in writing; not verified.
- **[UNKNOWN]** — not established; do not assume.
- **[OUR-BEHAVIOR]** — what this application itself does.

Language principle: this application never says "audio is deleted after
transcription". It says **[OUR-BEHAVIOR]** "our application deletes its
retained copy according to the configured hospital retention policy", and
describes provider-side retention separately, only per provider documentation.

---

## 1. Executive summary

1. The current production path sends **the full consultation audio** to
   Deepgram's cloud [CONFIRMED-CODE]. No patient identifiers or clinical
   metadata accompany it — only audio bytes and documented query parameters
   [CONFIRMED-CODE].
2. **Deepgram's default (without opt-out) retains audio AND transcripts to
   train its models** via the Model Improvement Program [PROVIDER-DOC].
   This is unacceptable for clinical audio as a default.
3. Setting `mip_opt_out=true` yields **zero data retention**: audio and
   transcript are "retained only for the duration needed to process the
   request" [PROVIDER-DOC]. **This application now sends that flag on every
   request by default** [OUR-BEHAVIOR, implemented this phase].
4. Deepgram publishes **regional endpoints — including India
   (`api.in.deepgram.com`)** — where requests are processed and stored
   in-region with no cross-region fallback; full in-region guarantee requires
   regional endpoint **plus** MIP opt-out [PROVIDER-DOC]. This application now
   supports region selection via `ISCRIBE_DEEPGRAM_REGION`
   [OUR-BEHAVIOR, implemented].
5. Deleting our local copy has **no effect** on any provider-side data. With
   ZDR there is nothing retained to delete; without it, deletion is a
   contractual/account matter — no per-request deletion API exists
   [PROVIDER-DOC].
6. No DPA/BAA is verified for this deployment [CONTRACT-REQUIRED].
7. For **Malayalam**, Deepgram offers no support [PROVIDER-DOC]; the project's
   local sidecar (IndicConformer) is the Malayalam path and keeps no audio
   [CONFIRMED-CODE].

## 2. Current Deepgram data flow

```
Doctor browser (MediaRecorder)
  → HTTPS multipart POST /api/consultations/{cid}/audio        (session-gated)
  → OUR application server: streamed to data/uploads/{cid}.ext  [OUR STORAGE]
  → on Process: pipeline resolves provider (deepgram → faster-whisper fallback)
  → DeepgramSTTProvider.transcribe():
      POST https://{region endpoint}/v1/listen?<documented params>
      Headers: Authorization: Token <env key>; Content-Type: audio/<ext>
      Body: FULL audio file bytes
  → JSON response: transcript, utterances (speaker + timings),
      metadata (request_id, model_info, duration, detected_language)
  → validation → diarization labelling → clinical fact graph (V2)
  → deterministic note (fail-closed validator) → result persisted in SQLite
  → audit: transcription_requested / transcription_completed
      (provider, model, request_id when returned, elapsed, off_host)
  → local audio deleted ONLY by the hospital retention lifecycle (approval-gated)
```

Audio is uploaded to our server first; the browser never talks to the provider
[CONFIRMED-CODE].

## 3. Data sent externally

Per request [CONFIRMED-CODE]: audio bytes (whole file), `model=nova-3-medical`,
`language` (forced, never auto-detect), `punctuate`, `smart_format`,
`numerals`, `diarize`, `utterances`, `profanity_filter=false`,
`mip_opt_out=true` (now default), `Authorization` header. **No patient name,
no MRN, no doctor name, no hospital id, no consultation id** is sent
[CONFIRMED-CODE]. Deepgram's docs state request metadata/usage logs record
usage, not content [PROVIDER-DOC].

What we cannot exclude: the audio **content itself** contains spoken
identifiers (names, numbers). That is inherent to STT and is the reason the
retention flags above matter.

## 4. Data returned

Transcript text, alternatives, utterances with speaker indices and timestamps,
`metadata.request_id`, `model_info`, `duration`, `detected_language`
[CONFIRMED-CODE]. The `request_id` is an opaque correlation id — it cannot
identify a patient by itself [CONFIRMED-CODE, and it is stored only in our
audit trail and logs]. No confidence PII fields are returned.

## 5. Provider retention (Deepgram, provider-documented)

Source: developers.deepgram.com — "Your Data at Deepgram" [PROVIDER-DOC]:

| Data | Default (MIP) | With `mip_opt_out=true` |
|---|---|---|
| STT request audio | **Retained to improve models** | Retained only for the duration needed to process the request |
| STT transcript | **Retained to improve models** | Same — process-only retention |
| Request metadata & usage logs | Retrievable 90 days (no audio/transcripts in logs) | Same 90 days |

The precise duration of "needed to process the request" is not numerically
specified [UNKNOWN]. Whether our account's terms match the published defaults
is [CONTRACT-REQUIRED].

## 6. Provider training / model-improvement policy

MIP participation is the **default** and voluntary; opted-out requests are
excluded from training; training data is never sold, used for advertising, or
redistributed without permission; Deepgram states the trained model retains no
rote memory of training data [PROVIDER-DOC]. Account-level opt-out
enforcement is available on request [PROVIDER-DOC]. This deployment defaults
to per-request opt-out [OUR-BEHAVIOR].

## 7. Geographic processing / storage

- `api.deepgram.com` (current default): **no residency guarantee**
  [PROVIDER-DOC].
- Regional endpoints `api.{eu,au,in}.deepgram.com`: request processed and
  stored in-region; no cross-region fallback (fail, not reroute); Whisper
  models excluded from regions (Nova/Flux available) [PROVIDER-DOC].
- Full in-region guarantee = regional endpoint **and** `mip_opt_out=true`
  (MIP training may process retained data outside the region) [PROVIDER-DOC].
- Regional residency "is not sovereignty" (foreign legal process caveat);
  Deepgram Dedicated / self-hosted for stricter pinning [PROVIDER-DOC].
- `nova-3-medical` supports `en-IN` (Indian English) among its locales
  [PROVIDER-DOC]; region × model availability should be confirmed with the
  vendor [UNKNOWN].

## 8. DPA / security controls

- SOC 2 Type 1 + Type 2 (report on request); HIPAA **BAA available on
  request**; GDPR-ready; PCI; CCPA; APP8 [PROVIDER-DOC].
- Encryption TLS 1.3 in transit, AES-256 at rest [PROVIDER-DOC].
- Sub-processors published at deepgram.com/privacy/subprocessors
  [PROVIDER-DOC]; the current list must be reviewed and annexed to any DPA
  [CONTRACT-REQUIRED].
- **No DPA or BAA is currently held or verified for this deployment**
  [CONTRACT-REQUIRED]. Nothing in this document claims HIPAA/GDPR/DPDP
  compliance for iScribe; Deepgram's own certifications are provider-side
  attestations, not ours.

## 9. Deletion semantics

- **Our copy:** deleted only by the hospital lifecycle (approval-gated,
  audited, verified `unlink`, failure-retried) [OUR-BEHAVIOR]. Filesytem
  deletion is not cryptographic erasure; backups taken earlier outlive it
  until backup expiry [OUR-BEHAVIOR, documented limitation].
- **Provider side:** no per-request deletion API exists
  [PROVIDER-DOC]. With ZDR there is nothing retained to delete (process-only)
  [PROVIDER-DOC]. With MIP enabled, deletion/withdrawal is a contractual and
  account-level matter [CONTRACT-REQUIRED]. **Deleting our local audio does
  not and cannot delete provider-side data** — the only lever is ZDR by
  configuration.

## 10. Threat model (provider boundary)

| Threat | Analysis |
|---|---|
| Audio interception in transit | TLS to provider [PROVIDER-DOC]; our leg is HTTPS/TLS via tunnel [OUR-BEHAVIOR]. |
| Provider-side retention of clinical audio | Default MIP retains it [PROVIDER-DOC] → mitigated by default ZDR flag [OUR-BEHAVIOR]; residual: misconfiguration — surfaced in audit `mip_opt_out` field and Deepgram Console [PROVIDER-DOC]. |
| Provider training on clinical audio | Same control; opt-out is per request [PROVIDER-DOC]. |
| Provider breach | Provider controls (SOC 2, AES-256) are attestations [PROVIDER-DOC]; incident timelines are contractual [CONTRACT-REQUIRED]. With ZDR, exposure window is process-duration only [PROVIDER-DOC]. |
| Subprocessor chain | Published list [PROVIDER-DOC]; contractual flow-down needed [CONTRACT-REQUIRED]. |
| Wrong-region processing | Region selected per deployment [OUR-BEHAVIOR]; verify endpoint config per hospital [CONTRACT-REQUIRED/OPERATIONS]. |
| Fallback silently changes processing location | Today a Deepgram failure falls back to local faster-whisper — which actually keeps audio on-host, but changes model/quality silently. A hospital that requires deterministic processing should disable auto-fallback (flagged as next-phase config; NOT implemented). |
| Key exposure | Env-only, never in URLs/logs/tests [CONFIRMED-CODE]. |

## 11. Provider comparison (researched; no winner selected)

Costs are approximate published/list rates at research time and vary by plan;
**[VERIFY]** marks items to confirm with the vendor before any decision.

| | Deepgram | Google Cloud STT | Azure Speech | AWS Transcribe | OpenAI speech APIs | faster-whisper (self-hosted) | India sidecar (IndicConformer) |
|---|---|---|---|---|---|---|---|
| Malayalam | **No** | Yes (ml-IN) [VERIFY] | Yes (ml-IN) [VERIFY] | Batch/stream ml-IN [VERIFY] | Whisper models list ml; API support [VERIFY] | Weak (measured: confident nonsense risk on ml) [CONFIRMED-CODE/benchmarks] | **Yes — purpose-built** [CONFIRMED-CODE] |
| English (Indian) medical | **nova-3-medical en-IN** [PROVIDER-DOC] | No medical model; word-boost phrases | No dedicated medical model; Custom Speech adaptation | Transcribe **Medical is en-US only**; general Transcribe has no medical tuning | Not medical-tuned [VERIFY] | Good general English, not medical-tuned | en limited |
| Medical vocabulary | Yes, dedicated model | Boost/custom classes | Custom Speech training | HealthScribe (en-US) separate | No | Fine-tuning possible (research-grade) | No |
| Streaming | Yes | Yes | Yes | Yes | Yes | Yes | No (batch only, current) |
| Latency | Low | Moderate | Low | Moderate (batch) | Low | Hardware-dependent | Moderate (CPU/GPU local) |
| ~Cost (pre-recorded) | ~$0.0043–0.0078/min | ~$0.016–0.024/min | ~$1/hr | ~$0.024/min (medical ~$0.0375) | ~$0.006/min [VERIFY] | Infrastructure only | Infrastructure only |
| Training/model-improvement | **Default ON; opt-out per request (ZDR)** | **Opt-in** data logging; default not used for training; sync/streaming in-memory, async transcript ~5 days | **No training on customer audio; no data at rest (real-time)** | **Default ON; org-level opt-out policy** | Default retention for abuse monitoring; ZDR by agreement [VERIFY] | None — local | None — local, no audio persisted [CONFIRMED-CODE] |
| DPA | BAA/DPA on request [PROVIDER-DOC] | Google Cloud DPA | Microsoft DPA + HIPAA BAA | AWS BAA (HIPAA-eligible) | Enterprise agreements [VERIFY] | n/a | n/a |
| India residency | **api.in.deepgram.com** [PROVIDER-DOC] | Limited (eu/us endpoint pinning only; STT processed globally) [PROVIDER-DOC] | **Central India region** | **Mumbai (ap-south-1)** | No | n/a (on-prem) | n/a (on-prem) |
| Deletion controls | None per request; ZDR instead | Async ops deletable; sync nothing stored | Batch artifacts customer-controlled (TTL/delete API) | Outputs in customer S3 | ZDR agreement [VERIFY] | Full control | Full control |
| Audio leaves our infra | **Yes** | Yes | Yes (or container offline) | Yes | Yes | **No** | **No** |
| Self-host option | Deepgram self-hosted/Dedicated [PROVIDER-DOC] | No | Speech containers (offline) [PROVIDER-DOC] | No | No | **Yes — current fallback** | **Yes — current Malayalam path** |
| Operational complexity | Low | Low–moderate | Low–moderate (+containers if offline) | Low | Low | High (GPU/CPU ops) | High (separate process/env) |

Key structural facts: only the self-hosted options guarantee audio never
leaves the hospital host; only Deepgram offers a medical-tuned `en-IN` model
among hosted providers; Google/Azure/AWS never train on customer STT data by
default (Google opt-in, Azure no-training, AWS org opt-out), whereas Deepgram
and AWS default to using content for service improvement until opted out.

## 12. Recommended architecture options (for hospital choice; not a winner)

- **Option A — Deepgram global + ZDR (current default):** fastest, cheapest,
  audited; no residency guarantee.
- **Option B — Deepgram India endpoint + ZDR + DPA:** in-country processing
  for Indian hospitals [PROVIDER-DOC]; confirm regional availability of
  `nova-3-medical` [VERIFY].
- **Option C — Azure (Central India) / AWS (Mumbai):** strong in-country
  residency + enterprise DPAs + no-training defaults; weaker medical tuning;
  integration work via the existing provider interface.
- **Option D — Fully local (faster-whisper + sidecar):** zero off-host audio;
  accepts measured quality trade-offs; highest operational complexity.
- **Any option must set:** explicit provider selection per hospital (env),
  ZDR/no-training flags, DPA on file, audit of every boundary crossing.
- **Recommended next-phase flag (not implemented):** disable automatic
  provider fallback for deployments that require deterministic processing
  location/model.

## 13. Hospital onboarding checklist (provider boundary)

- [ ] Choose provider option A/B/C/D with the hospital's DPO.
- [ ] Set `ISCRIBE_DEEPGRAM_REGION` (or pin a non-Deepgram provider) per the residency decision.
- [ ] Keep `ISCRIBE_DEEPGRAM_MIP_OPT_OUT=true` unless the hospital explicitly accepts MIP.
- [ ] Obtain and file the provider DPA (and BAA where applicable); annex subprocessor list.
- [ ] Record contractual retention/deletion terms in the hospital's deployment record.
- [ ] Verify in Deepgram Console (Usage > Logs) that requests show `mip_opt_out=true` [PROVIDER-DOC].
- [ ] Confirm audit trail shows `transcription_completed` with provider/model/endpoint for a test consultation.
- [ ] Decide auto-fallback policy (allow local fallback vs fail-closed).
- [ ] Document backup expiry vs audio deletion deadlines.

## 14. Questions that must be answered contractually by the provider

1. Exact maximum retention window for opted-out (ZDR) requests; is it verifiable?
2. Is `mip_opt_out=true` contractually binding, or documentation-only?
3. Full subprocessor list with data-processing roles and locations.
4. Breach notification timeline and liability terms.
5. DPA terms for Indian clinical audio; DPDP Act 2023 posture (provider docs are silent on DPDP) [UNKNOWN].
6. BAA availability and scope for this customer.
7. Region × model availability: is `nova-3-medical` (en-IN) served from `api.in.deepgram.com`?
8. Deletion/withdrawal process for any content retained outside ZDR (e.g., support artifacts).
9. Employee/administrative access controls to customer content; audit rights.
10. Data residency commitments in contract form (regional endpoint SLA).

## 15. Explicit unknowns

- The precise numeric duration of Deepgram's ZDR "needed to process the
  request" window [UNKNOWN — provider docs give no number].
- Whether this deployment's account terms match published defaults — **no DPA
  is signed or verified** [CONTRACT-REQUIRED].
- Current subprocessor list contents (page exists; not reproduced here) [VERIFY].
- Region-specific model availability for `nova-3-medical` [VERIFY].
- Whether Deepgram's "model has no rote memory of training data" claim has
  independent verification [PROVIDER-DOC claim only].
- Third-party blog summaries of MIP pricing effects (e.g., "the list price
  opts you in") are consistent with the docs but pricing specifics change —
  confirm with the vendor [VERIFY].
- Malayalam support on Google/Azure/AWS current API surfaces and quality for
  clinical speech [VERIFY — no reliable clinical-quality benchmark exists
  publicly; benchmark on our own fixtures before any decision].

**Bottom line:** with the changes implemented this phase, our application
truthfully reports, per consultation, what left the host, to which endpoint,
under which retention flag — and clinical audio defaults to zero provider-side
retention. Residency, contractual deletion, and breach terms remain
contractual tasks for the hospital's privacy team, not code claims.
