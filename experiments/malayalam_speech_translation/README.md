# Malayalam Speech → English Translation Experiment (isolated)

**Question:** can an open-source Malayalam speech → English translation path replace
the current Deepgram/English-transcription front end while preserving clinical meaning?

**Audio:** `East Tippu Sulthan Road 4.m4a` (35.6 s, user upload, NOT committed —
may contain real clinical content; see data policy below). During the experiment the
decisive discovery was that **this file is English speech** — three independent
detectors agree — so the real Malayalam hypothesis test uses the previous
consultation recording (`previous_consultation.m4a`, also NOT committed), which is
the audio the production Malayalam path was validated on.

No production code was modified. `CurrentSTTProvider`/Deepgram routing and the ml
sidecar are untouched.

## Pipelines

| ID | Path | Implementation |
|----|------|----------------|
| A  | Current baseline: Deepgram nova-3-medical, `language=en`, diarized | project's `DeepgramSTTProvider` (production code path) |
| A2 | Same API, explicit `language=ml` probe | same provider |
| B  | IndicConformer ml (CTC) → IndicTrans2 (indic-en-dist-200M) | live sidecar (CPU) via HTTP, as the earlier frozen-benchmark runners |
| C  | faster-whisper `small`, `task=translate` (direct speech→EN), CPU int8 | local, ~470 MB model |

## Results on `East Tippu Sulthan Road 4.m4a` (the requested file)

**Discovery: the audio is English speech, not Malayalam.**
- Deepgram (en): coherent English transcript.
- faster-whisper transcribe pass: `detected_language=en` (p=0.694), coherent English transcript agreeing with Deepgram on content.
- IndicConformer (ml-locked): produced non-Malayalam garbage (`ദിഫലീതികായ ആ ഇ കൈനീ…`) — the expected failure of an ml-locked model on English audio.

**Clinical extraction** (`build_normalized_entities` over each English transcript):
- Deepgram A1 → fever PRESENT (confidence 0.75, clause "I have fever since last 6 to 7 days."), pain PRESENT (chest, clause preserved); `normalized_summary`: "Reports fever, pain."
- Whisper C → same two concepts; longer source clauses (no diarization sentence breaks). Breathing difficulty and fatigue appear in the transcript but are **not** extractor concepts — extraction-layer gap, not STT.

## Results on `previous_consultation.m4a` (real Malayalam, 33.3 s)

| Metric | A: Deepgram (en) | B: Indic→IndicTrans2 | C: Whisper translate |
|---|---|---|---|
| Transcript | "Good morning. Okay. Yeah. Okay. Okay. Okay. Okay. Okay." | Real Malayalam transcript with clinical content (പനി, ക്ഷീണം, തലവേദന, കഫക്കെട്ട് variant, ഫുഡ് കഴിക്കാൻ പറ്റണില്ല) | Fluent English **hallucination** ("I have to clean my hair") around a few real fragments |
| Clinical entities from English rendering | none | n/a (translation leg unusable, see below) | none (hallucinated text contains no real symptom assertions) |
| ml-native semantic layer (production reference) | — | fever RESOLVED, fatigue RESOLVED (status-scope bug reproduced: fatigue is actually still present per speaker) | — |
| Latency | ~5 s | ASR 7.7 s + **MT 230.5 s (CPU)** | translate 72 s (RTF ≈ 2.2), transcribe-only 4.3 s |
| Speaker attribution | diarized turns (production `speaker_turns`) | SPEAKER_UNKNOWN | none (single stream, no diarization) |

**Pipeline B translation leg findings (unchanged from the frozen benchmark, reconfirmed on CPU):**
- IndicTrans2 dist-200M on CPU took **230 s for a 33 s clip** (RTF ≈ 7) and emitted a degenerate
  "Ahh…" filler — **the Malayalam→English MT leg is not clinically viable on this hardware.**
- The ASR leg itself (IndicConformer) remains the only local path producing real Malayalam text,
  which is why the production sidecar uses it **without translation**, feeding the ml semantic layer.

## Clinical-meaning answers (acceptance criteria)

1. **Can Malayalam speech become English locally?** Technically yes (C), but not faithfully: Whisper-translate produces fluent hallucinations on this audio; B's MT leg degenerates.
2–5. **Clinical faithfulness / negation / resolved-vs-current / duration:** NOT preserved by any English-rendering path on real Malayalam audio. The only path that preserves them is the ml-native semantic layer — and it currently has the status-scope bug (fatigue RESOLVED) that is being fixed separately.
6. **Code-switching:** B's ASR captures code-switched English tokens inside Malayalam (ഫുഡ്, റെഡി); C garbles them; A cannot hear Malayalam at all.
7. **Latency:** cheapest viable ml path = sidecar ASR-only (~8 s for 33 s). Any English-translation leg adds 1–4 minutes on this CPU.
8. **GTX 1650 4 GB:** all local paths run on CPU; GPU is not required and showed instability (earlier segfaults).
9. **vs Deepgram:** materially different — Deepgram cannot process Malayalam at all (HTTP 400 on `language=ml`, and its English model hears only "Okay. Yeah." in real Malayalam audio).
10. **Realistic front end?** Not as speech→English translation on this hardware. The ml-native ASR + ml semantic layer is the viable architecture.

## Speaker attribution

Only Deepgram (A) provides diarization. IndicConformer and whisper-translate do not — any
pipeline built on them needs a separate diarization stage; until then outputs must stay
`SPEAKER_UNKNOWN`. The experiment stores raw transcripts only and never assigns roles.

## Data policy

Real consultation audio and this README's sibling result JSONs live under `results/` and the
audio files under this directory are treated as potentially containing PHI: audio files are
gitignored and never committed; result JSONs contain transcript text and are likewise not
committed. Runners are deterministic and re-runnable against your own copies.

## Reproduce

```bash
# A (Deepgram; needs DEEPGRAM_API_KEY in .env)
python experiments/malayalam_speech_translation/run_pipeline_a_deepgram.py <audio> <tag>
# B (needs the ml sidecar running on :8131)
python experiments/malayalam_speech_translation/run_pipeline_b_indic.py
# C
python experiments/malayalam_speech_translation/run_pipeline_c_whisper.py <audio> <tag>
# Clinical-extraction comparison
python experiments/malayalam_speech_translation/extract_clinical.py
# Audio conditioning (why B's garbage on file 4 was a model failure, not audio)
python experiments/malayalam_speech_translation/measure_audio.py
```

## Recommendation for the NEXT ENGINEERING STEP

**Do not adopt speech→English translation.** Keep the production architecture:
Deepgram for English; IndicConformer (sidecar) + the ml semantic layer for Malayalam.
The evidence says clinical meaning is preserved only when extraction runs on
Malayalam text. The highest-value next step is therefore the **status-scope fix in the
Malayalam semantic layer** (fatigue RESOLVED on real audio is the live example), plus
extraction-layer concept gaps surfaced here (breathing difficulty, fatigue — present in
English transcripts but not extractor concepts).
