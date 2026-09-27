# Malayalam Pipeline — Decision Document

Date: 2026-09-22. Evidence:
`benchmarks/malayalam/benchmark_report.json` (frozen five-fixture CPU benchmark),
`benchmarks/malayalam/gpu_vs_cpu_device_comparison.json` (device comparison),
`benchmarks/malayalam/pipelineB_full_gpu.json` (**complete five-fixture
ASR + translation run on GPU**),
`benchmarks/malayalam/offline_tokenless_gpu.json` (offline verification),
`docs/MALAYALAM_PIPELINE.md`, `docs/THIRD_PARTY_NOTICES.md`.

All fixtures are **synthetic, non-PHI** TTS audio. No claim here is based on real
patient recordings.

> **Status change (2026-09-22): the IndicTrans2 blocker is cleared.** Gate terms
> were accepted for the correct `indic-en` variant, the official model is cached
> locally, and the complete Malayalam → English pipeline now runs end to end,
> offline and tokenless. Sections 2, 3, 4 and 7 are new or rewritten; section 1
> (ASR) stands unchanged from the verified checkpoint.

---

## 1. ASR — can IndicConformer replace faster-whisper for Malayalam?

**Yes, for Malayalam audio specifically. It is not a general replacement.**

### Quality (measured, five fixtures)

Pipeline A — current STT (`whisper base`, auto-detect) **fails on Malayalam**:

- Wrong language detected on 3 of 5 fixtures (`si` Sinhala, `te` Telugu)
- Wrong *script* output (Tamil/Urdu characters) on 2 of 5
- Hallucinated content, e.g. `"Android 12 Enica squash"`
- Best case was still garbled on the fixtures it labelled `ml`

Pipeline B — IndicConformer **produces clean, correct-script Malayalam** for
native-language content. Clinical content survives: fever/cough, three-day
duration (`മൂന്ന് ദിവസമായി`), headache, vomiting, and the medication
`പാരസിറ്റമോൾ ടാബ്ലറ്റ്` (paracetamol tablet) were all captured.

Known ASR defects, measured not assumed:

- Word-spacing errors at speaker turns (`ഉണ്ട്ഛർദ്ദിയോ`)
- English loanwords inside Malayalam garble: *blood pressure* → `ചവദാഭയബഥ`,
  *nasal spray* → `ലാൻഡിലാണ്`
- English-only audio → Malayalam-script nonsense. Expected: it is an ml-locked
  model. **This is why it cannot be the default provider.**

No WER/CER is quoted. There is no human-verified reference transcript set, so a
numeric accuracy score would be fabricated.

### Speed

| | RTF | Notes |
|---|---|---|
| Pipeline A (whisper base, CPU) | 1.57 – 5.06 | *Slower than realtime* |
| Pipeline B ASR (CPU) | 0.115 – 0.3 | Frozen benchmark + re-verified today |
| Pipeline B ASR (GPU, GTX 1650) | **0.008 – 0.013** | 10.4× median / 14.5× best vs CPU |

IndicConformer is both **more accurate and faster** than the current STT on
Malayalam audio.

### Offline and tokenless capability — VERIFIED

Re-verified this session on **both** CPU and GPU with `HF_TOKEN` unset,
`HF_HUB_OFFLINE=1`, `SCRIBE_OFFLINE_MODE=1`, and the model loaded from a local
`.nemo` path. Both produced a byte-identical 176-character transcript, 89%
Malayalam script. After the one-time model download, the ASR stage needs **no
token, no network and no HuggingFace access**.

### Resource requirements

| | CPU | GPU |
|---|---|---|
| Model on disk | 499 MiB (`.nemo`) | same |
| Process RSS | ~600 MB | ~426 MB |
| VRAM | — | 605 MiB peak (3.4 GB headroom on a 4 GB card) |
| Cold model load | 69 – 119 s | 69 s |

The 69–119 s cold load is significant and must be paid at service startup, not on
the first clinician request — the same warm-up pattern iScribe already uses.

### Licensing

MIT, verified from HF API tags and the model card. Revision
`e96d81e42e5fe73f282b0322d422a48b461a4ca6`. Commercial use permitted.

---

## 2. Translation — can IndicTrans2 be used?

**Yes. It is downloaded, reproducible, and runs offline.**

| | |
|---|---|
| Model | `ai4bharat/indictrans2-indic-en-dist-200M` |
| Revision | `eb9e49d81077cfc5311e82ff36d8c1fc11557b5d` (verified, pinned) |
| Licence | MIT (LICENSE file present in the local copy) |
| Local path | `C:/Users/akthe/scribe-models/indictrans2-indic-en-dist-200M` |
| Size on disk | 921,516,494 B (878.8 MiB), 13 files |
| Weights | `model.safetensors` (913,353,672 B). `pytorch_model.bin` deliberately **not** fetched — identical weights, second format |
| Tokenizer files | `dict.SRC.json`, `dict.TGT.json`, `model.SRC`, `model.TGT`, `tokenization_indictrans.py`, `tokenizer_config.json` — all present |

### What the block actually was

The account had accepted `indictrans2-`**`en-indic`**`-dist-200M` (English →
Indic) instead of `indictrans2-`**`indic-en`**`-dist-200M` (Indic → English).
The token was never at fault — it reported `canReadGatedRepos: true` throughout.
One acceptance click on the correct page cleared it. The `Raghavan/…` mirror was
never needed and never used.

### Two code defects found once it could actually run

Neither was visible while the model was unavailable:

1. **`use_cache=True` crashed generation.** transformers ≥ ~4.44 passes
   `generate()` an `EncoderDecoderCache` object where this model's 2023-era
   `modeling_indictrans.py` expects `None` or a legacy tuple; it evaluates
   `past_key_values[0][0].shape` and raises
   `AttributeError: 'NoneType' object has no attribute 'shape'`. Fixed by
   generating with `use_cache=False`, which avoids that path entirely. Revisit
   if upstream updates for the current cache API.
2. **`postprocess_batch()` was never called.** IndicProcessor masks numbers and
   named entities during preprocessing and restores them in postprocessing.
   Skipping it leaves placeholder tokens in the output — for clinical text that
   means **dosages and durations can surface as placeholders instead of values**.
   It is now called on every translation.

### Download note

Fetched into a flat directory rather than the HF blob cache: that cache's
symlinks need Developer Mode or admin on Windows and otherwise fail with
`WinError 1314`. A flat directory is also exactly what `INDIC_TRANS_MODEL_PATH`
consumes.

---

## 3. Complete pipeline — does Malayalam audio → English work, locally and offline?

**Yes, on both counts.**

Five-fixture run, GPU, `benchmarks/malayalam/pipelineB_full_gpu.json`:

| Fixture | Dur (s) | ASR (s) | MT (s) | Total (s) | RTF | Terms |
|---|---|---|---|---|---|---|
| ml_fever_cough | 16.42 | 5.96 | 2.31 | 111.77¹ | 6.81¹ | 9/15 |
| ml_chest_pain | 11.57 | 0.53 | 1.13 | 1.66 | 0.14 | 3/15 |
| ml_english_medical_terms | 15.24 | 0.41 | 1.50 | 1.91 | 0.13 | 3/15 |
| manglish_codeswitch | 11.74 | 0.34 | 0.66 | 0.99 | 0.08 | 0/15 |
| en_only_control | 18.67 | 0.66 | 2.14 | 2.80 | 0.15 | 0/15 |

¹ First fixture carries the one-time cold load of **both** models. Steady state
is the other four rows: **RTF 0.08–0.15**, i.e. 7–12× faster than realtime.

"Terms" counts how many of 15 expected English medical terms appear in the
output. It is a **presence count, not an accuracy score**, and section 4 shows
why a high count can still accompany a dangerous translation.

### GPU: both models coexist comfortably

Task 4 asked whether IndicTrans2 can share the GTX 1650 without destabilising
IndicConformer. Measured with both loaded:

| | |
|---|---|
| VRAM peak (ASR + MT together) | **1427.8 MiB allocated / 1564.0 MiB reserved** of 4095.8 MiB |
| Headroom | ~2.5 GiB |
| MT weights alone in VRAM | 810.1 MiB |
| Host RSS | 241–651 MiB |
| ASR quality change vs CPU | none — transcripts byte-identical |

No fallback was needed. CPU translation was measured anyway as the documented
fallback: **model load 23.03 s, translate 15.51 s** for the fixture that takes
**2.31 s on GPU** (~6.7× slower), **producing character-identical English**. CPU
is viable but noticeably slower, and on this 5.9 GB host it also thrashes.

### Offline and tokenless — verified, not assumed

`benchmarks/malayalam/offline_tokenless_gpu.json`, produced by
`verify_offline.py`. The harness is deliberately stronger than `HF_HUB_OFFLINE=1`:
it removes every HF credential from the process and blocks outbound sockets at
the socket layer, permitting only loopback.

`HF_TOKEN` **was set in the parent shell** and was stripped by the harness. Result:

- Complete pipeline ran: ASR 2.06 s, translation 4.42 s, correct English out
- **Blocked network attempts: 0** — nothing even tried to reach the network
- Both models loaded purely from `MALAYALAM_ASR_MODEL_PATH` and
  `INDIC_TRANS_MODEL_PATH`

After one-time acquisition, the pipeline needs **no token, no network and no
HuggingFace access**.

---

## 4. Clinical-language evaluation

Five synthetic TTS fixtures. **No accuracy figure is claimed or computed** —
there is no human-verified reference set. What follows is inspection of the
actual output, which is where the real risk shows.

### Where it works

`ml_fever_cough` is the clean case — native Malayalam, no English loanwords:

> **ML:** …എനിക്ക് മൂന്ന് ദിവസമായി പനിയും ചുമയും ഉണ്ട്… തലവേദയും ഉണ്ട്… പാരസിറ്റമോൾ ടാബ്ലറ്റ് രാവിലെ വൈകുന്നേരം കഴിക്കുക
> **EN:** "what is the problem i have fever and cough for three days, vomiting
> or headache, yesterday i had vomiting also take paracetamol tablet in the
> morning and evening"

Correct: symptom set (fever, cough, headache, vomiting), **duration** ("three
days"), **medication** ("paracetamol tablet"), **dosage timing** ("morning and
evening"). This is clinically usable text.

### Failure 1 — the MT layer launders ASR garbage into confident clinical terms

This is the most serious finding. In `ml_chest_pain`, the ASR garbles the
English loanword *blood pressure* into `ചവദാഭയബഥ`. IndicTrans2 does not pass the
garbage through and does not flag it — it **invents a plausible diagnosis**:

> **EN:** "Chest pain and shortness of breath should be cursed by a person who
> has had **asthma** for two years."

There is no asthma anywhere in the source. A reviewer reading only the English
sees a confident, plausible, entirely fabricated diagnosis. The same source term,
garbled differently in `ml_english_medical_terms` (`ചവതാഭയപട`), became
**"a garbage can"** — the same input produced two unrelated inventions.

Chest pain, shortness of breath and "two years" *were* carried correctly, so the
sentence reads as mostly-right with one fatal substitution. That is worse than
obvious nonsense.

### Failure 2 — fabricated numbers and units

`ml_english_medical_terms` opens with what should be "what is the BP?":

> **EN:** "curse how much is **five pesos** i've had **a garbage can** for two
> years. i'm taking a tablet. use it in the morning **in the land** ok doctor"

"five pesos" is a fabricated number *and* a fabricated currency unit, from audio
containing neither. Numbers and units are exactly the category clinical text
cannot afford to have invented. ("in the land" is garbled *nasal spray*.)

### Failure 3 — code-switching collapses

`manglish_codeswitch` scored 0/15:

> **EN:** "I don't have an A or B for two days."

Real Kerala clinical speech is heavily code-switched — English drug names,
"BP", "sugar", "tablet" inside Malayalam sentences. This fixture is the closest
proxy for it and the pipeline fails completely, including an apparent **negation
that is not in the source**.

### Failure 4 — speaker turns are lost

ASR emits no turn boundaries and drops the space between them
(`ഉണ്ട്ഛർദ്ദിയോ`). In fixture 1 the doctor's *question* "ഛർദ്ദിയോ ഉണ്ടോ?"
("do you have vomiting?") merges into the patient's narrative as "vomiting or
headache". **A clinician's question can be recorded as a patient's reported
symptom.**

### Failure 5 — English audio produces confident gibberish

`en_only_control` yields transliterated nonsense ("Nadathama Babapana and
Thosan…"). This one at least **fails loudly** — it cannot be mistaken for a
clinical record. It also confirms the model must never be auto-selected by
language detection from its own output.

### Summary of the risk

Accuracy degrades **silently and plausibly**, not visibly. The ASR's weakness
(English loanwords) feeds precisely the MT's weakness (fluent completion of
garbled input), and the combination manufactures clinical content — a diagnosis,
a number, a unit — that was never spoken.

---

## 5. English / default provider

**Unchanged: `current_faster_whisper` remains the production default.**

Verified this session — the provider registry contains exactly one entry
(`['current_faster_whisper']`) and the Malayalam provider is **not registered**.
The iScribe engine suite passes **9/9** and the full suite **38/38**.

---

## 6. Recommended eventual architecture

```
Malayalam audio
  → IndicConformer (ml)        → Malayalam transcript  ─┐  both preserved
  → IndicTrans2 (ml→en)        → English transcript    ─┘
  → clinical pipeline

English audio
  → faster-whisper             → English transcript
  → clinical pipeline
```

The `Transcript` dataclass already carries `source_language`,
`source_transcript`, `target_language` and `translated_transcript`, so the
original Malayalam is never discarded — required for clinical traceability. The
UI already renders the original-plus-translation view when `source_language`
is not `en`.

Routing between providers is a later decision: it needs either an explicit
clinician language selection or a language-identification step. An ml-locked
model given English audio returns confident nonsense, so **auto-routing must not
be inferred from IndicConformer's own output.**

---

## 7. Deployment recommendation

**Move from `experimental` to `candidate for controlled clinical evaluation`.
Do NOT register it as a production provider, and do not make it any default.**

Gate checklist:

| # | Condition | Status |
|---|---|---|
| 1 | Translation works end-to-end | ✅ five fixtures, GPU and CPU |
| 2 | Complete model files available locally | ✅ 878.8 MiB, all 13 files, MIT |
| 3 | Offline/tokenless reproducibility, **complete** pipeline | ✅ verified with sockets blocked, 0 network attempts |
| 4 | Medical terminology / code-switching quality evaluated | ⚠️ **evaluated — and it fails in a dangerous way** (section 4) |
| 5 | Resource requirements acceptable | ✅ 1.43 GiB VRAM for both models, 2.5 GiB headroom, RTF 0.08–0.15 |

Four of five are met, and condition 4 is now *evaluated* rather than unknown —
which is itself the change that justifies moving status. The evaluation result
is negative for unsupervised use but is exactly the kind of finding a controlled
evaluation exists to characterise on real audio.

### Why "candidate", not "experimental"

Every technical blocker is gone: the pipeline is reproducible, pinned, offline,
fast, and fits the existing hardware. Staying at `experimental` would imply
unresolved engineering, and there is none.

### Why not production, on any timeline yet

Section 4 documents a failure mode that is disqualifying for autonomous use: the
system **fabricates clinical content that was never spoken** — a diagnosis
("asthma"), a number and unit ("five pesos") — and does so in fluent, plausible
English that carries no signal of failure.

### Mandatory conditions for any controlled evaluation

1. **Always display the Malayalam source alongside the English.** The data model
   already supports this (`source_transcript` is never discarded) and the UI
   already renders both when `source_language != "en"`. A reviewer who reads
   only the English cannot catch failure 1.
2. **Never auto-populate diagnosis, medication or dosage fields** from Malayalam
   pipeline output. Extraction may suggest; a clinician must enter.
3. **Do not route by language automatically.** An ml-locked model returns
   confident nonsense for English audio (failure 5), so routing must come from
   an explicit clinician selection, never from the model's own output.
4. **The evaluation's primary objective is real code-switched audio** with a
   human-verified reference set. That is the missing evidence, and synthetic TTS
   fixtures cannot supply it.
5. Remains behind the provider abstraction, unregistered, opt-in only.

---

## 8. Next single action

**Build a human-verified reference set from real code-switched Kerala clinical
audio, and measure against it.**

Everything else is done: models pinned and local, pipeline offline and fast,
hardware sufficient. The one thing blocking a real clinical judgement is that all
current evidence comes from five synthetic TTS fixtures. The specific question to
answer first is how often the ASR garbles an English loanword and the MT then
launders it into a plausible clinical term — that rate, on real audio, decides
whether this pipeline is viable at all.
