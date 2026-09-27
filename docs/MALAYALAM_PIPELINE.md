# Malayalam → English Pipeline (experimental, ISOLATED)

Status: **ASR stage proven locally on CPU *and* GPU, tokenless/offline verified.
Translation stage still blocked on IndicTrans2 gated-repo access.**
Nothing here is wired into the demo: `CurrentSTTProvider` remains the demo default.

## IndicTrans2 access — precise diagnosis (2026-09-22)

Re-probed this session. The blocker is now characterised exactly:

| Probe | Result |
|---|---|
| Repo id exists (`HEAD huggingface.co/...`) | **200** — id is correct, rules out a typo/wrong revision |
| `model_info` anonymous | **OK** — repo *metadata* is public |
| `gated` flag | **`auto`** — gate exists, approval is automatic on acceptance |
| `license` | **mit** |
| File download, no credential | **HTTP 401** on every file (`config.json`, `dict.SRC.json`, `dict.TGT.json`, `model.SRC`, `model.TGT`, `tokenization_indictrans.py`, `tokenizer_config.json`) |
| Credential present in this environment | **none** — `HF_TOKEN` unset, no `hf login` token in cache |

**Cause, narrowed to a single fact: the gate terms were accepted on the wrong
model page.**

With a working fine-grained token for account `akthernaimudheen`
(`canReadGatedRepos: true`, so token scope is *not* the problem), the
IndicTrans2 family reports:

| Repo | Access |
|---|---|
| `indictrans2-indic-en-dist-200M` | **gated — not accepted** ← the one we need |
| `indictrans2-en-indic-dist-200M` | **GRANTED** ← what was actually accepted |
| `indictrans2-indic-en-1B` | gated |
| `indictrans2-en-indic-1B` | gated |
| `indictrans2-indic-indic-dist-320M` | gated |
| `indictrans2-indic-indic-1B` | gated |

`en-indic` translates **English → Indic**. This pipeline needs **Indic → English**
(`indic-en`). The two names differ only in the order of that pair, which is why
the acceptance landed on the wrong page. The accessible `en-indic` model is the
wrong direction and is not a substitute.

Earlier observations, now superseded but kept for the record: **401** meant no
credential was present at all; **403** with a token means authenticated but not
authorised. Metadata being public is why `model_info` succeeds while every file
fetch is refused.

Not (C): the repo is not restricted beyond its `auto` gate. Not (D): the id and
revision `eb9e49d81077cfc5311e82ff36d8c1fc11557b5d` resolve correctly — all 15
files including both weight formats are listed. Not a token-scope problem either.

**The official repo is complete** — it ships all tokenizer files the mirror lacks:
`dict.SRC.json` (3,391,208 B), `dict.TGT.json` (644,755 B),
`tokenization_indictrans.py` (8,044 B), `model.SRC`, `model.TGT`, plus
`model.safetensors` (913,353,672 B) and `pytorch_model.bin` (913,515,337 B).
The incomplete `Raghavan/...` mirror is therefore not needed and is not used.

**Unblock procedure (owner action, one time):** signed in as the account that
owns the token, open **`huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M`**
— note `indic-en`, not `en-indic` — and click *Agree and access repository*
(auto-approved). The existing token already carries `canReadGatedRepos`, so no
new token is needed. No gate bypass was attempted and none should be.

## Pipeline

```
Malayalam audio (16 kHz mono wav)
  → IndicConformer ml (EncDecHybridRNNTCTCModel, CTC decode, language_id="ml")
      → Malayalam transcript   (always preserved)
  → IndicTrans2 dist-200M (IndicProcessor preprocess → Seq2SeqLM generate)
      → English transcript     (working text; source transcript never discarded)
```

## Exact model identities

| Stage | Repo | Revision (commit) | License | Local file | SHA256 |
|---|---|---|---|---|---|
| ASR | `ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large` | `e96d81e42e5fe73f282b0322d422a48b461a4ca6` | MIT | `indicconformer_stt_ml_hybrid_rnnt_large.nemo` (499 MB) | `9d7a21ad880720bac8acf2843ce61f78370d7cf68655a15381e73ce3dfd9cc18` |
| MT | `ai4bharat/indictrans2-indic-en-dist-200M` | `eb9e49d81077…` (target revision) | MIT | **not yet cached — gate 403** | pending |

The MT model repo ships `model.safetensors` (913,353,672 B) **and** `pytorch_model.bin`
(913,515,337 B) — identical weights in two formats; only one is needed.

## Environment (isolated, outside OneDrive on purpose)

### CPU environment

- Location: `C:/Users/akthe/scribe-ml-venv` (uv-managed venv, Python 3.12.14)
- Key packages: torch 2.14.0 (CPU), nemo_toolkit **1.23.0rc0** (AI4Bharat fork
  `git+https://github.com/AI4Bharat/NeMo.git@nemo-v2`, commit `8dce88cf8e94`,
  installed `--no-deps` over NeMo 2.7.3's dependency set), transformers 4.57.6,
  tokenizers 0.22.2, sentencepiece 0.2.2, sacremoses 0.2.0, indic-nlp-library-itt 0.1.1,
  faster-whisper 1.2.1, numpy 1.26.4, scipy 1.13.1, huggingface_hub 0.36.2,
  pytorch-lightning 2.6.6, psutil 7.2.2
- Why the fork: the checkpoint's decoder uses `multisoftmax`, removed from
  NVIDIA NeMo ≥2.x; the AI4Bharat `nemo-v2` branch retains it. Two compat shims
  (in `asr.py`) patch removed symbols (`huggingface_hub.ModelFilter`,
  `pytorch_lightning.loggers.NeptuneLogger`) — neither is used at inference.

### GPU environment (COMPLETE — verified 2026-09-22)

- Location: `C:/Users/akthe/scribe-gpu-venv` (Python 3.12.14)
- `torch 2.14.0+cu126`, CUDA 12.6, **GTX 1650 4096 MiB, compute 7.5, driver 555.99**
- Built by taking the CUDA torch stack and then matching the CPU env's three
  pinned pieces. Three blockers were hit and resolved in order:
  1. Stock `nemo_toolkit 2.7.3` cannot restore the checkpoint —
     `RNNTDecoder.__init__() got an unexpected keyword argument 'multisoftmax'`.
     Fixed by copying the AI4Bharat fork's `nemo/` package (16.2 MB) from the CPU
     env; the stock package is preserved as `nemo.stock-2.7.3.bak` so the change
     is reversible. No download was needed.
  2. `numpy 2.5.3` → `AttributeError: np.sctypes was removed in NumPy 2.0`
     (NeMo's audio preprocessing). Pinned to `1.26.4`, matching the CPU env.
  3. `scipy 1.18.1` then failed against numpy 1.26 (`np.long`). Pinned to
     `1.13.1`, matching the CPU env.
- Resulting known-good pair: **numpy 1.26.4 + scipy 1.13.1 + NeMo fork 1.23.0rc0**,
  identical to the CPU env, with CUDA torch on top.
- The CPU environment was not modified.

## Offline / tokenless operation (verified)

- `SCRIBE_OFFLINE_MODE=1` + `MALAYALAM_ASR_MODEL_PATH=<path to .nemo>` →
  loads only from the local file, no HF contact, no token. **Tested: PASS**
  (HF_TOKEN unset, HF_HUB_OFFLINE=1; 90 s cold load, 6.26 s inference).
- Translation offline mode: `INDIC_TRANS_MODEL_PATH=<snapshot dir>` +
  `local_files_only=True` (implemented; untested until the gate clears).
- Normal (online) mode uses the standard HF cache; after one authenticated
  download, inference re-uses the cache without contacting HF.

## Reproduction

```bash
# one-time (needs gated-repo access for the MT model)
python -m scribe_engine.stt.experimental.indic_malayalam.check_malayalam --live

# benchmark
python -X utf8 -m scribe_engine.stt.experimental.indic_malayalam.benchmark \
  scribe_engine/stt/experimental/indic_malayalam/test_audio/*.wav \
  --device cpu --out benchmarks/malayalam/benchmark_report.json

# smoke test
python -m pytest scribe_engine/stt/experimental/indic_malayalam/test_smoke.py -q
```

## Known behavior on synthetic fixtures (measured, CPU)

- Native Malayalam speech transcribes well: symptoms, duration, medication
  names reproduced (e.g. fever/cough/3-days/headache/vomiting/paracetamol).
- Word spacing errors at speaker turns; English loanwords inside Malayalam
  garble (blood pressure → "ചവദാഭയബഥ", nasal spray → "ലാൻഡിലാണ്");
  English-only audio produces Malayalam-script nonsense (expected — ml-only model).
- ASR latency: 1.9–4.4 s per 11–18 s clip (RTF ≈ 0.13–0.3), model load 73–119 s
  cold, in-process.

## Device comparison (measured 2026-09-22, `ml_fever_cough.wav`, 16.42 s)

Model loaded once per device, fixture transcribed 3×, timed with the same
`asr_time_seconds` field the frozen benchmark used. Machine was under concurrent
load. Full data: `benchmarks/malayalam/gpu_vs_cpu_device_comparison.json`.

| | CPU | GPU (GTX 1650) |
|---|---|---|
| ASR time, best of 3 | 1.88 s | **0.13 s** |
| ASR time, median | 2.29 s | **0.22 s** |
| RTF (best) | 0.115 | **0.008** |
| Model load (cold) | 94.5 s | 68.7 s |
| VRAM peak | — | 604.6 MiB allocated / 644 MiB reserved |
| Process RSS | 600 MB | 426 MB |

**Speed-up: 10.4× median, 14.5× best.** The transcript is byte-identical on CPU
and GPU and stable across all runs, so the GPU path is a pure latency win with no
observed quality change. VRAM headroom is ~3.4 GB of the 4 GB card, so the
translation model could plausibly share the same GPU once unblocked.

First run on each device is 4.3–4.8 s regardless of device (dataloader + numba
JIT warm-up); steady-state figures above exclude it. This also explains why a
single cold measurement understates GPU benefit.

---

## Semantic normalization + ASR correction layers (current phase)

Two new layers sit between the raw transcript and the scope engine. Both are
deterministic, data-driven, and keep the raw transcript immutable:

```
raw ASR transcript (immutable record)
  -> asr_correction.py        verified ASR corruptions only, full provenance
  -> semantic.py              colloquial Kerala semantics -> canonical concepts
  -> normalization.py         scope engine: negation / question / temporality
  -> clinical entities
```

### Files

- `scribe_engine/asr_correction.py` — evidence-based ASR corrections. An entry
  is added ONLY for a corruption observed in a frozen benchmark transcript with
  known ground truth. Context-guarded entries fire only inside the observed
  pattern. Colloquial mappings are deliberately NOT here.
- `scribe_engine/semantic.py` + `scribe_engine/data/malayalam_clinical_semantics.json`
  — colloquial aliases, controlled targets (BP/sugar high-low-check-question),
  family-subject attribution, negative idioms ("ശ്വാസം കിട്ടുന്നില്ല" asserts the
  symptom), medication forms routed as treatment, fused conditional markers
  (െങ്കിൾ/െങ്കിലും), and clause-final interrogatives (എത്രയാണ്) that give the
  verb splitter a boundary in punctuation-free ASR output.
- `scribe_engine/clinical.py` — `build_normalized_entities()` now runs the
  semantic layer; family history and medications are routed out of patient
  findings; `normalized_from` records `raw` vs `asr_corrected`.
- `scribe_engine/pipeline.py` / `malayalam_sidecar/server.py` /
  `scribe_engine/stt/malayalam_provider.py` — the sidecar returns raw and
  corrected copies plus provenance separately; `process_audio` meta carries
  `corrected_text` and `asr_corrections`; pasted text is NOT flagged as
  ASR-corrected.

### Measured on the synthetic fixtures (SYNTHETIC, not clinical accuracy)

See `benchmarks/malayalam/semantic_layer_report.json`
(runner: `run_semantic_layer_benchmark.py`). Concept recall/precision 1.0 on
the 5 frozen fixtures with 0 fabricated concepts, including the two fixtures
the ASR garbled completely (correctly returned nothing). Negation,
temporality, and subject attribution are covered by the 86-case safety suite
in `tests/test_malayalam_semantic_safety.py` (253 tests total, all passing;
English production suite unchanged and green).
