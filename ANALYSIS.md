# AI Medical Scribe — Upstream Analysis & Reuse Map

Source: https://github.com/AAC-Open-Source-Pool/AI-MEDICAL-SCRIBE
Upstream commit at time of analysis: **`faf97d5c17c1f397f842f6a9367deec64d91c50a`** (2025-10-24, "Fix indentation for medications deduplication")
Cloned into `upstream-scribe/` (reference only — nothing is imported from it at runtime).

## 1. What the repository actually contains (24 files)

The repo is a student-team iteration log: five generations of the same idea, each saved as a new top-level script, plus scraped data, a sample audio file, and screenshots. There is no package structure, no requirements.txt, no tests, and **no LICENSE file** (GitHub API confirms `license: null`).

The real pipeline exists in `final1.py` (Flask API) and `automatic labeling.py` / `speech to text transcription.py` (CLI variants of the same flow): pyannote diarization → Whisper transcription → segment-to-speaker merge → regex/NLP extraction → templated notes.

## 2. Traced execution path (actual, not README)

1. Audio enters as an uploaded file (Streamlit uploader) or POST multipart (Flask `final1.py /process_audio`).
2. Whisper `base` transcribes it (`model.transcribe(path)` → `result["segments"]` with start/end/text).
3. Speaker identity: `pyannote.audio` Pipeline `pyannote/speaker-diarization` (HuggingFace token required, gated model) produces SPEAKER_00/01 turns; segments are mapped to speakers by timestamp containment.
4. Speaker→role labeling: `final1.py` hard-codes `{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}`. The Streamlit apps (`front.py`, `finalfront.py`, `label.py`) instead just alternate DOCTOR/PATIENT per segment index — no diarization.
5. Clinical NLP: `nlp.py::extract_medical_info` — spaCy `en_core_web_sm` PhraseMatcher over a 20-item symptom list + per-token fuzzy matching (rapidfuzz, threshold 85) + regexes for diagnosis ("it looks like X", "diagnosed with X"), duration, name, age. Returns dict with symptoms/duration/diagnoses/warnings/confidence.
6. Notes/prescription: `front.py` and `new.py` — keyword scan of patient lines for ~60 symptoms, regex scan of doctor lines for ~15 hard-coded medication names, then f-string templates producing numbered "Clinical Notes" and "Prescription / Plan" blocks.
7. Output: text blocks in the browser; optional .txt download (Streamlit button).

## 3. Reuse map

| Component | Existing file(s) | Reuse? | Why |
|---|---|---|---|
| Audio decode | implicit via Whisper/ffmpeg | YES | ffmpeg is required by Whisper anyway; no upstream code needed |
| STT (Whisper base) | `transcribe.py`, `streamlit.py`, `final1.py` (all same `whisper.load_model("base")`) | YES (wrapped) | Works; this run uses faster-whisper (same base weights, CT2 runtime) because openaipublic.azureedge.net is unreachable from this machine |
| Diarization (pyannote) | `automatic labeling.py`, `final1.py` | NO (demo) | Gated HF model + HF token + torch/pyannote stack; upstream's own token in the repo is leaked and invalid to reuse. Engine exposes a pluggable interface; runs without it |
| Speaker→role labeling | `final1.py` (map), `front.py`/`label.py` (alternating) | YES (adapted) | Both approaches kept: diarization map when available, heuristic otherwise; alternating mode marked low-confidence in output |
| Clinical NLP | `nlp.py` | YES (verbatim functions) | The most robust upstream module (has guards, logging, warnings); `end.py`/`final1.py` copies are worse duplicates of it |
| Symptom/med keyword lists | `new.py`, `front.py` | YES (verbatim lists) | They power note generation; naive but functional |
| Note + prescription generation | `front.py`, `new.py` | YES (adapted into functions) | Hard-coded UI code converted to pure functions returning the same text blocks |
| Flask API | `final1.py` | NO (replaced) | We need consultations model + stages; superseded by our FastAPI service |
| Streamlit UIs | `streamlit.py`, `front.py`, `finalfront.py` | NO | Replaced by our UI per requirements |
| Static HTML mockup | `front_end.html` | NO | No backend wiring; dead mockup |
| CLI experiments | `transcribe.py`, `label.py`, `new.py`, `end.py`, `diarization_code.py` | NO | Duplicate/dead; `diarization_code.py` is actually a transcribe.py copy (no diarization at all) |
| Scraper | `Scrapping.py` | NO | Data-collection tool, unrelated to runtime |
| Sample audio | `Allergy final.mp3` | DEMO ONLY | Our end-to-end test fixture; no PHI |
| Scraped transcripts | `transcriptions.zip`, `Conversations.txt` | DEMO ONLY | Potential future NLP test fixtures; not wired in |

## 4. Defects found (documented, not silently fixed)

1. `nlp.py` duration regex matches noise on real transcripts (e.g. "started using Allegra again two weeks ago" → duration "using allegra").
2. Alternating DOCTOR/PATIENT labels are wrong ~50% of the time by construction; on the sample audio the patient's opening line is labeled DOCTOR, which suppresses patient-only symptom detection.
3. The sample audio is very noisy; Whisper base garbles key words ("allergies"→"energies", "nasal spray"→"natural space"), so notes from it are unreliable — a demo-data-quality limitation, not an engine bug.
4. `final1.py` contains a hard-coded (now revoked-looking) HuggingFace token — never reuse; also `speech to text transcription.py` has another.
5. No license anywhere in the repo.

## 5. Upstream / license

- Repo: https://github.com/AAC-Open-Source-Pool/AI-MEDICAL-SCRIBE (public, created 2025-03-21)
- Commit: `faf97d5c17c1f397f842f6a9367deec64d91c50a`
- License: **NONE declared.** All rights reserved by default. For a hospital trial demo this is a legal risk to flag: we extracted the smallest functional ideas (lists, regexes, templates — re-typed/adapted with attribution) and kept the clone for reference. Recommendation: contact the team (README lists members) or replace upstream-derived lists before any production/pilot use. ATTRIBUTION.md documents provenance.

## 6. Environment facts (this machine)

- Windows, Python 3.14.4 (only version installed) — whisper/spacy/rapidfuzz install and run fine on 3.14.
- ffmpeg 9.0.1 present (required).
- openaipublic.azureedge.net unreachable → openai-whisper model download hangs; faster-whisper (HF-hosted CT2 models) works.
- pyannote/speaker-diarization: gated on HF + requires HF token; not configured for this demo.
- GPU: none required; faster-whisper base int8 on CPU transcribes the 104 s sample in ~29 s.
