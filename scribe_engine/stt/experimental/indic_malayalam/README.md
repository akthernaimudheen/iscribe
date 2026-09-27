# Experimental: Malayalam → English pipeline (ISOLATED)

Status: **code-complete, model downloads blocked on HF terms acceptance** (see below).
Nothing here is wired into the demo flow. Run it only from `.venv-ml` (Python 3.12).

## Pipeline B

```
Malayalam audio (16 kHz wav)
   → IndicConformer ml  (ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large, MIT, ~120 MB)
       → Malayalam transcript            (preserved verbatim in all outputs)
   → IndicTrans2 dist-200M ml→en  (ai4bharat/indictrans2-indic-en-dist-200M, MIT)
       → English transcript              (working text for downstream clinical NLP)
```

Both transcripts are kept (`source_transcript` + `translated_transcript` on the
STTProvider `Transcript` object) for clinical traceability.

## Environment (isolated, per hardware-constraint policy)

- `.venv-ml` — Python 3.12.14 (uv-managed), separate from the demo's Python 3.14 `.venv`
- `nemo_toolkit[asr]` + torch 2.14 CPU — installed, imports verified
- `faster-whisper` — installed for Pipeline A benchmarking
- Pending: `IndicTransToolkit` + `transformers<4.53` + `sentencepiece` (install after unblock)

## BLOCKER: gated models

Both AI4Bharat repos are `gated: auto` on HuggingFace. With a valid token but without
terms acceptance, downloads fail with `GatedRepoError 403` (verified empirically).

**To unblock (one-time, ~1 minute):** while signed in as `akthernaimudheen`:

1. https://huggingface.co/ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large → click **Agree and access repository**
2. https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M → click **Agree and access repository**

Then re-run the benchmark (below). Everything else is ready.

## Usage

```bash
# benchmark Pipeline A (current faster-whisper baseline) vs Pipeline B (Malayalam)
HF_TOKEN=... python -X utf8 -m scribe_engine.stt.experimental.indic_malayalam.benchmark \
    scribe_engine/stt/experimental/indic_malayalam/test_audio/*.wav \
    --device cpu --out benchmark_results.json
```

Test fixtures (synthetic, non-PHI, generated with edge-tts — see make_fixtures.py):
`ml_fever_cough`, `ml_english_medical_terms`, `ml_chest_pain`, `manglish_codeswitch`,
`en_only_control` — covering Malayalam-only, English-medical-terms-in-Malayalam,
code-switching, and an English control.

## What was already measured (Pipeline A baseline, CPU, whisper base)

faster-whisper `base` with auto language detection on these fixtures:
- Malayalam audio: wrong-script output (Tamil script for 2/4 files, romanized
  noise for others); detected languages: `si`, `te`, `ml` — i.e. the current
  baseline does NOT handle Malayalam. (Full numbers in `benchmark_pipelineA_partial.json`.)
- Even the `en_only_control` was garbled with auto-detect; re-run with `language="en"`
  before drawing conclusions about English quality of the baseline.

These are honest observations on synthetic TTS audio; not a claim about real patients.
