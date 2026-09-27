# Third-Party Notices

Verified from actual model/repository metadata (HF API `gated` + license tags, LICENSE files)
on 2026-09-22. Not inferred from popularity or secondary sources.

## Models — demo (Pipeline A / baseline)

| Component | Version/commit | Source | License | Commercial use | Notes |
|---|---|---|---|---|---|
| OpenAI Whisper `base` weights (as run by faster-whisper) | base checkpoint | github.com/openai/whisper (weights mirrored on HF `Systran/faster-whisper-base`) | MIT | Permitted | Same weights the upstream repo used |
| faster-whisper (CTranslate2 runtime) | 1.2.1 | pypi / github.com/SYSTRAN/faster-whisper | MIT | Permitted | CPU int8 verified |
| spaCy `en_core_web_sm` | 3.8.0 | github.com/explosion/spacy-models | MIT (code); model redistribution terms per model card — verify before redistribution | Demo use OK | Used by reused clinical NLP |
| rapidfuzz | 3.x | pypi | MIT | Permitted | |
| FastAPI / Uvicorn / pydantic | current | pypi | MIT / BSD / MIT | Permitted | |

## Models — Malayalam experiment (Pipeline B, ISOLATED, not integrated)

| Component | Version/revision | Source | License | Commercial use | Gating | Status on this machine |
|---|---|---|---|---|---|---|
| AI4Bharat IndicConformer Malayalam `indicconformer_stt_ml_hybrid_ctc_rnnt_large` | HF rev `e96d81e42e5fe73f282b0322d422a48b461a4ca6`; `.nemo` SHA256 `9d7a21ad…d9cc18` | huggingface.co/ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large | **MIT** (verified via HF API tags + model card) | Permitted | Metadata public; anonymous `model_info` OK. Already cached, so no further fetch needed | **Cached locally (523,192,320 B = 499 MiB), inference verified on CPU *and* CUDA, tokenless/offline verified 2026-09-22 (identical transcript on both devices)** |
| AI4Bharat IndicTrans2 `indictrans2-indic-en-dist-200M` | rev `eb9e49d81077cfc5311e82ff36d8c1fc11557b5d` (pinned) | huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M | **MIT** (LICENSE file present in the local copy, verified 2026-09-22) | Permitted | `gated: auto` — **access granted** after accepting terms on the `indic-en` page | **Cached locally and verified.** `C:/Users/akthe/scribe-models/indictrans2-indic-en-dist-200M`, 921,516,494 B across 13 files. `model.safetensors` (913,353,672 B) downloaded; `pytorch_model.bin` deliberately skipped (identical weights, second format). Inference verified on CUDA and CPU, tokenless/offline verified. Earlier block was terms accepted on the wrong variant (`en-indic`); the `Raghavan/…` mirror was never used |
| NeMo (AI4Bharat fork, branch `nemo-v2`) | 1.23.0rc0, commit `8dce88cf8e94` | github.com/AI4Bharat/NeMo | Apache-2.0 (per fork repo) | Permitted | — | Installed `--no-deps`; required for the checkpoint's `multisoftmax` decoder |
| IndicProcessor (vendored, pure-Python) | v1.0.2 era, commit `a95da3a008` of VarunGumma/IndicTransToolkit | github.com/VarunGumma/IndicTransToolkit | MIT | Permitted | — | Vendored verbatim at `scribe_engine/stt/experimental/indic_malayalam/_vendor/processor.py` (see `_vendor/PROVENANCE.md`); current toolkit release has no Windows wheels |
| torch (CPU) | 2.14.0 | pypi | BSD-3 | Permitted | — | `.venv-ml`-lineage experiment env |
| faster-whisper | 1.2.1 | pypi | MIT | Permitted | — | Pipeline A in benchmark |
| transformers / tokenizers / sentencepiece / sacremoses / indic-nlp-library-itt | 4.57.6 / 0.22.2 / 0.2.2 / 0.2.0 / 0.1.1 | pypi | Apache-2.0 / Apache-2.0 / (Apache-2.0/MIT docs) / MIT / MIT | Permitted | — | Translation + vendored processor deps |

## Fixture generation (non-PHI synthetic audio only)

| Component | Source | License | Usage boundary |
|---|---|---|---|
| edge-tts (Microsoft cloud TTS, free/keyless) | pypi `edge-tts` 7.2.8 | MIT (client) | **Fixture generation only.** Synthetic scripted sentences, no PHI, no patient audio ever. Not part of the scribe runtime. |

## Upstream repository — REFERENCE ONLY / NO DECLARED LICENSE / NOT A PRODUCT DEPENDENCY

`upstream-scribe/` — AI-MEDICAL-SCRIBE @ `faf97d5c17c1f397f842f6a9367deec64d91c50a`.
**NO license declared.** Treat as all-rights-reserved; keep the clone reference-only and
out of any build/packaging. See ATTRIBUTION.md for what was adapted and required actions
before any use beyond this demo. Its leaked HuggingFace tokens were identified and
deliberately NOT used; scans verify no token patterns exist outside the reference clone.

## Environment separation

- Demo/production: `.venv` (Python 3.14.4, project dir) — faster-whisper, spacy, fastapi.
  Untouched by the Malayalam experiment.
- Experiment (CPU): `C:/Users/akthe/scribe-ml-venv` (Python 3.12.14, **outside OneDrive** —
  site-packages inside OneDrive suffered file locks). torch 2.14.0 CPU.
- Experiment (GPU): `C:/Users/akthe/scribe-gpu-venv` (Python 3.12.14, torch 2.14.0+cu126,
  CUDA 12.6). **Complete and verified 2026-09-22.** Uses the same AI4Bharat NeMo fork
  and the same numpy 1.26.4 / scipy 1.13.1 pins as the CPU env. The stock
  `nemo_toolkit 2.7.3` package it shipped with is retained as `nemo.stock-2.7.3.bak`.
- No credentials stored in the repository. No HF token is present in any of these
  environments; the cached ASR model needs none. A token is required only to
  unblock the IndicTrans2 download, and must be supplied via environment variable.
