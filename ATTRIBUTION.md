# Attribution & Provenance

## Upstream source

- Repository: https://github.com/AAC-Open-Source-Pool/AI-MEDICAL-SCRIBE
- Commit analyzed: `faf97d5c17c1f397f842f6a9367deec64d91c50a` (2025-10-24)
- A reference clone is kept at `upstream-scribe/` and is **not** imported at runtime.

## License status

The upstream repository declares **no license** (`LICENSE` absent; GitHub API reports
`license: null`). Under default copyright rules the original authors retain all rights.

What was taken from upstream and how:

| Our file | Upstream origin | Relationship |
|---|---|---|
| `scribe_engine/clinical.py` (extract) | `nlp.py` | Function structure, symptom list, regex patterns adapted from upstream; re-implemented with attribution and light hardening (input guards kept) |
| `scribe_engine/clinical.py` (notes) | `front.py`, `new.py` | Note template text and keyword lists reproduced; converted from inline UI code into pure functions |
| `scribe_engine/prescription.py` | `front.py` | Prescription template text and medication regex reproduced; converted into a pure function |
| `scribe_engine/diarization.py` (role map) | `final1.py` | SPEAKER_00→Doctor / SPEAKER_01→Patient mapping and timestamp-merge idea reused; leak-prone hard-coded token NOT carried over |
| `demo audio` | `Allergy final.mp3` | Used unmodified as demo/test fixture only |

## Risk note (action required before any pilot use)

Because upstream has no license, formally we operate on informal goodwill of the
student team (README lists: Shivani Rao, Surya Teja, mentors Siddharth Mahesh,
Bhuvan Sai; org: AAC-Open-Source-Pool). Before any use beyond this demo:

1. Contact the team / org and request an explicit MIT/Apache-2.0 license, **or**
2. Replace the upstream-derived lists/templates with our own (small, mechanical work).

The heavier components (Whisper weights: MIT; faster-whisper: MIT; spaCy models:
CC-BY-SA / MIT per model; pyannote: MIT) are independently licensed.
