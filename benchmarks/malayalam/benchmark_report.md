# Malayalam Pipeline A/B Benchmark — CPU (frozen)

Generated 2026-09-22 from `benchmark_report.json` (same directory). Machine: Windows,
CPU-only torch 2.14.0, Python 3.12.14 (`C:/Users/akthe/scribe-ml-venv`).
Fixtures are **synthetic, non-PHI** (edge-tts, ml-IN voices; see make_fixtures.py).

Terminology check = case-insensitive presence of 15 expected English medical terms
in the English output. It is a count, not an accuracy score.

## Summary table

| Fixture | Dur (s) | A: lang detected | A: RTF | A transcript quality | B: ASR | B: MT | B status |
|---|---|---|---|---|---|---|---|
| en_only_control (18.7s) | — | `si` (wrong) | 2.01 | garbled English | 4.4 s | — | MT blocked (403) |
| manglish_codeswitch (11.7s) | — | `si` (wrong) | 1.57 | romanized Malayalam-ish | 2.1 s | — | MT blocked (403) |
| ml_chest_pain (11.6s) | — | `te` (wrong) | 5.06 | mixed Tamil/Urdu script soup | 1.9 s | — | MT blocked (403) |
| ml_english_medical_terms (15.2s) | — | `ml` | 2.30 | Tamil script, some content | 2.9 s | — | MT blocked (403) |
| ml_fever_cough (16.4s) | — | `ml` | 2.74 | Tamil-script + hallucination ("Android 12") | 2.8 s | — | MT blocked (403) |

Pipeline B ASR latencies above are from the frozen run; standalone ASR run
(`logs/asr_run.json`) measured 1.9–4.4 s per clip (RTF ≈ 0.13–0.3) with model load
73–119 s (in-process, one-time).

## Key measured findings

1. **Pipeline A (current demo STT, whisper base, auto-detect) does NOT handle Malayalam.**
   Wrong language detection on 3/5 fixtures (`si`, `te`), wrong script output
   (Tamil/Urdu characters) on 2/5, hallucinated content. It is also SLOWER than
   Pipeline B's ASR on this machine (RTF 1.57–5.06 vs ≈0.13–0.3).
2. **Pipeline B ASR (IndicConformer) produces clean, correct-script Malayalam** for
   native-language content: symptoms, durations ("മൂന്ന് ദിവസമായി"), medication
   ("പാരസിറ്റമോൾ ടാബ്ലറ്റ്") captured. Defects: missing word spacing at speaker
   turns; English loanwords garbled (blood pressure → "ചവദാഭയബഥ"); English-only
   audio → Malayalam-script nonsense (expected for an ml-locked model).
3. **Pipeline B translation stage: NOT RUN — blocked.** `ai4bharat/indictrans2-indic-en-dist-200M`
   returned HTTP 403 (gated repo) for this account throughout the session, despite the
   ASR repo being accessible with the same token. No translation numbers exist; none are
   fabricated. `translations/` is therefore empty by design.

## Full per-fixture data

Machine-readable: `benchmark_report.json`. Raw ASR output: `logs/asr_run.json`,
`transcripts/*.ml.txt` (actual Malayalam, UTF-8). Benchmark stdout/stderr:
`logs/bench_stdout.log`.

## Honest limitations

- Synthetic TTS voices, not real patients; results don't predict clinical audio quality.
- whisper-base was evaluated with auto-detect (the demo's actual configuration);
  forced-`en` runs also garbled the TTS control earlier — baseline English quality
  on TTS audio is itself weak.
- No GPU numbers in this report (GPU run pending in separate env at freeze time).
