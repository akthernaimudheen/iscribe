# Malayalam Extraction Failure — Production Recheck Report

**Date:** 2026-09-29
**Failure consultation:** `0f2eb47e731a` (production, language=ml, STT provider `malayalam_indicconformer`)
**Production build at time of report:** `engine-a19f900` (fix NOT deployed; verified in working tree only)
**Transcript handling:** exact production STT transcript processed internally; **no PHI printed, committed, or embedded** (verified: no digits, no latin-script names, no phone/DOB/address/email, no CI PHI-guard markers; CI guard clean).
**Baseline before fix:** 774 tests passing. **After fix:** 801 passing.

---

## EXECUTIVE SUMMARY

A production Malayalam consultation produced a transcript containing substantially
more clinical information than the fact graph extracted: only `pain` (PRESENT,
"5 days") and `medicine` (PRESENT, "5 days") survived. Root cause was a **semantic
boundary failure**, not STT: one 229-character clause fused the doctor's
wh-questions with the patient's answers, so question scope and the duration
answer spread across every concept in the clause; the observed vomiting spelling
variant was never matched; the "nothing else" negation was unrecognized.

All defects were fixed at the architecture's existing layers (no keyword
stuffing, no per-consultation hardcoding), pinned by 27 new regression tests
including a sanitized fixture built from the exact production transcript, and
verified by re-running the fixed pipeline against that transcript.

**PRODUCTION RECHECK: PASS.** Deployment recommendation: **WAIT** (see §J, §Limitations).

---

## SANITIZED DIAGNOSTIC COMPARISON

### 1. Facts present BEFORE (stored production graph)

| concept | status | duration | temporal |
|---|---|---|---|
| pain | PRESENT | 5 days | CURRENT |
| medicine | PRESENT | 5 days (incorrect) | CURRENT |

### 2. Facts present AFTER (post-fix pipeline, same transcript)

| concept | type | status | when | duration | speaker | certainty |
|---|---|---|---|---|---|---|
| pain | SYMPTOM | PRESENT | CURRENT | 5 days | patient | CONFIRMED |
| vomiting | SYMPTOM | QUESTIONED | CURRENT | — | patient | UNCERTAIN |
| medicine | MEDICATION | QUESTIONED | CURRENT | — | patient | UNCERTAIN |

### 3. Facts recovered

- `vomiting` — the observed spelling variant ഛർദി now maps to the same concept
  as ഛർദ്ദി. Evidence-grounded: concept surface verified inside its own clause span.

### 4. Facts still missing

- **None.** Audit of every lexicon concept occurring in the transcript
  (`medicine` ×1, `pain` ×2, `vomiting` ×2): all represented in the graph.
  No diagnosis/assessment/plan facts exist because the transcript genuinely
  contains none. Classification: **intentionally not extracted — evidence
  insufficient** (the "No diagnosis documented" warning is correct behavior,
  not a miss).

### 5. Facts that changed status

- `medicine`: PRESENT → **QUESTIONED**. The transcript asks about medicine; the
  patient never confirms taking any. PRESENT was a false assertion.

### 6–7. Temporal links before / after

| concept | before | after |
|---|---|---|
| pain | 5 days | **5 days** (only fact the speech actually modifies) |
| medicine | 5 days ❌ | **none** ✅ |
| vomiting | — | none |

### 8. New false positives

- **None.** Per-fact evidence-grounding audit passed (concept surface present in
  its own clause); ghost-concept audit (concepts in graph but never in the
  transcript): none; duration audit: no non-pain fact carries a duration.

---

## STAGE-BY-STAGE VERIFICATION

### A. Clause segmentation — PASS

- BEFORE: 1 fact-spanning clause of **229 chars**.
- AFTER: **7 clauses**, lengths 13 / 17 / 25 / 30 / 58 / 68 / 101.
- Mega-clause eliminated via semantic boundaries (MOVE_WH opener split +
  wh-duration Q→A boundary), not character-length splitting.

### B. Temporal attachment — PASS

- `pain = 5 days` retained; `medicine = 5 days` eliminated.
- Mechanism: the duration answer lives in a concept-free clause after the Q→A
  split; `extract_mentions` now carries it **one hop** to the next clause's
  finding and the hop is consumed by the first mention — it can never travel
  further or cross a topic change.
- Separate-durations safety is pinned by test (symptom A → 5 days, medication
  B → 2 days remain distinct; unrelated later concepts inherit nothing).

### C. Vomiting variant — PASS

- ഛർദി and ഛർദ്ദി (plus inflected forms) map to the single normalized concept
  `vomiting`; no duplicate facts; unrelated words unaffected (tested).

### D. WH boundaries — PASS, no regression

- `_MOVE_WH` (ഏത്, എവിടെ, എപ്പോൾ, …) starts a new clause when it follows an
  answer; `_QUANT_WH` (എത്ര\*, എന്ത\*) keeps the original split-after behaviour
  so "X എത്രയാണ്" amount questions stay intact.
- Regression proof: the blood-pressure amount-question case
  (BP → QUESTION, diabetes stays PRESENT) passes; wh tokens are never
  clause-final verbs; ordinary sentences are not split by wh substrings.
- All 416 pre-existing semantic tests pass unchanged.

### E. Negation — PASS

- ഒന്നുമല്ല ("nothing else") recognized; **existential-negation scope rule**
  added: it denies *additional* findings, never the findings named before it.
- This also fixed a latent bug: the pre-existing ഒന്നുമില്ല marker could flip a
  stated finding to (false) RESOLVED via the ellipsis path — now covered.
- Concepts *after* the existential negation are still negated (tested both
  directions; no false global negation).

### F. Fact graph — COMPLETE for this transcript

Nothing hidden; see §4 for the miss classification (none).

### G. Canonical note — PASS

- Fail-closed validation: valid; 3 facts checked.
- Note contains pain + five days + vomiting (vomiting in the ROS/questioned
  list); contains **no** "taking medicine" assertion; nothing hallucinated.

### H. Structured review fields — PASS (single source, deterministic projection)

| field | value |
|---|---|
| symptoms_reported | Pain |
| duration | 5 days |
| ros | Vomiting |
| denies / medications / physical_findings / impression / plan / current_treatment_discussed | Not documented (honest) |
| prescription_fields.medications | [] (medicine question is not a drug) |
| prescription_fields.duration_of_treatment | Not documented. |
| warnings | "No diagnosis documented in this consultation" |
| confidence | None (per-fact confidence lives in `clinical_facts_v2.facts`) |

### I. Regression suite — PASS

- 27 new tests in `tests/test_malayalam_wh_boundary_regression.py`
  (per-marker boundary behaviour, false-positive guards, vomiting variants,
  temporal safety, negation context, production-transcript fixture).
- Targeted semantic suites: 416 passed. Combined targeted: 514 passed.
- **Full suite: 801 passed / 0 failed** (baseline 774 + 27 new; count change
  fully accounted for).

### J. Production smoke — NOT RUN (by design)

The existing smoke test uploads audio and creates real consultations on the
live service, mutating production state; it was intentionally not executed.
Read-only check only: `GET /api/health` → 200 OK, `engine-a19f900`,
uptime ≈ 20.9 h. **Production was not modified.**

---

## ROOT CAUSES

1. **WH-boundary failure** — `question_markers` lacked the observed wh-words
   (എന്താ, എത്ര, ഏത്, എവിടെ, …), and the splitter treated all question markers
   as clause-*final* verbs, so Q→A dialogue never segmented.
2. **Vomiting spelling gap** — lexicon knew only ഛർദ്ദി; this STT emits ഛർദി.
3. **Mega-clause fusion** — the 229-char clause fused the duration answer,
   pain, the negation, and the medicine question; interrogative scope and the
   "5 days" answer spread across every concept.
4. **ഒന്നുമല്ല unrecognized** as a negative-existential ("nothing else").
5. **Discovered en route:**
   - `_extract_duration` scanned only the *first* occurrence of a duration-unit
     word, silently dropping durations positioned after a Q→A split.
   - The polarity-ellipsis path let an existential negation flip previously
     stated findings to ABSENT→(false) RESOLVED (latent for ഒന്നുമില്ല too).

## FIXES (exact files / architectural layers)

| file | layer | change |
|---|---|---|
| `scribe_engine/data/malayalam_clinical_semantics.json` | semantics/lexicon merge | observed wh markers; ഒന്നുമല്ല; ഛർദി variants (evidence-gated merge, base lexicon untouched) |
| `scribe_engine/normalization.py` | clause splitter + status + duration | `_QUANT_WH`/`_MOVE_WH` classification; MOVE_WH verb-split exclusion; before-wh clause pass; wh-duration Q→A boundary pass; all-occurrence unit scan in `_extract_duration`; existential-negation scope rules in `_status_for` and the polarity-ellipsis path |
| `scribe_engine/clinical_facts.py` | fact graph | one-hop Q→A duration handoff in `extract_mentions` (consumed at first mention) |
| `tests/test_malayalam_wh_boundary_regression.py` | regression | 27 tests; embeds the exact production transcript as the sanitized fixture |

Note: the working tree also contains earlier, unrelated in-flight changes
(family-history attribution FP-7, allergy projection, medication dedup, pipeline
cleanup) that were present before this investigation began.

## TEST EVIDENCE

| suite | result |
|---|---|
| new regression tests | 27 passed |
| targeted semantic suites | 416 passed |
| combined targeted | 514 passed |
| full suite | **801 passed / 0 failed** (baseline 774 + 27 new) |

## VERIFICATION CHAIN (same clinical meaning at every stage)

EXACT PRODUCTION TRANSCRIPT → normalized transcript (ASR-corrected NFC copy,
0 corrections applied — NFC composition only) → 7 semantic clauses → 3 concept
mention groups → fact graph (3 facts, evidence spans verified) → canonical
clinical note (validation valid) → structured review fields (deterministic
projection, `clinical_facts_v2` single source).

## KNOWN LIMITATIONS

- **vomiting = QUESTIONED, not PRESENT.** The question and its answer share one
  ASR turn with unknown speaker roles; QUESTIONED is the conservative, honest
  status. The finding is visible to the clinician in the ROS — this is not
  silent information loss. Speaker-attribution work could upgrade it later; it
  was deliberately not keyword-forced.
- എന്താ (quant-wh) fused with a finding in one clause keeps conservative
  QUESTION status — pinned by test as documented behaviour.
- Malayalam coverage is demonstrated **only for this transcript's vocabulary and
  structures** plus the pre-existing suites. No broad Malayalam accuracy claim
  is made or implied.
- Malayalam→English translation quality (garbled MT on the earlier back-pain
  case) remains a separate, open issue.
- Stored consultations keep their frozen pre-fix graphs until reprocessed; the
  fix applies to new/reprocessed jobs only.
- **Production still runs `engine-a19f900`.** The fix exists only in this
  working tree, protected by the 801-test suite.

## DEPLOYMENT RECOMMENDATION: **WAIT**

Pipeline fix verified against the exact failure; full suite green. Do **not**
deploy until explicitly instructed. On approval, the path is:

1. Commit (excluding the unrelated foreign `README.md` edit).
2. `scripts/predeploy_check.py` (mind the known exit-code-swallow pipe bug).
3. Watchdog auto-deploy on HEAD change.
4. Reproduce consultation `0f2eb47e731a` on production (reprocess) and verify
   the post-fix fact graph + projection checks against this report.
5. Only then resume pilot-readiness work.
