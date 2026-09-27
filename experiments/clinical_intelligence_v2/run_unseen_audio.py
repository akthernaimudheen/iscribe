# -*- coding: utf-8 -*-
"""Unseen Clinical Audio Generalization Test — batch runner.

Runs NEW, unseen real clinical recordings through the UNCHANGED production
pipeline (ScribeEngine.process_audio) and records:

  * the STT provider actually used (hard assertion: deepgram nova-3-medical,
    never a Whisper fallback)
  * the raw transcript
  * clinical facts V2 (typed fact graph)
  * deterministic note V2 + validator verdict
  * full result payload for later defect tracing

Usage:
    python experiments/clinical_intelligence_v2/run_unseen_audio.py <audio> [<audio> ...]

Artifacts -> experiments/clinical_intelligence_v2/unseen_audio/artifacts/<stem>/
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "unseen_audio" / "artifacts"


def provider_is_deepgram() -> bool:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return bool(os.environ.get("DEEPGRAM_API_KEY"))


def run_one(audio: Path) -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from scribe_engine import ScribeEngine

    engine = ScribeEngine()
    t0 = time.time()
    result = engine.process_audio(str(audio), language="en")
    elapsed = round(time.time() - t0, 1)

    meta = result.get("meta") or {}
    provider = meta.get("stt_provider")
    selection = meta.get("provider_selection") or ""
    fallback = meta.get("provider_fallback")

    # PROVIDER GUARD: the run is only valid when Deepgram produced the text.
    if os.environ.get("DEEPGRAM_API_KEY") and provider != "deepgram":
        print(f"!! PROVIDER GUARD [{audio.name}]: expected deepgram, got {provider!r}")
        print(f"   selection: {selection}")
        raise SystemExit(2)

    dest = OUT / audio.stem
    dest.mkdir(parents=True, exist_ok=True)
    transcript = (result.get("transcript") or {}).get("text") or ""
    v2 = result.get("clinical_note_v2") or {}
    facts = result.get("clinical_facts_v2") or {}

    (dest / "transcript.txt").write_text(transcript, encoding="utf-8")
    (dest / "note_v2.txt").write_text(v2.get("note", ""), encoding="utf-8")
    (dest / "facts_v2.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    turns = (result.get("speakers") or {}).get("turns") or []
    summary = {
        "audio": audio.name,
        "bytes": audio.stat().st_size,
        "provider": provider,
        "provider_selection": selection,
        "provider_fallback": fallback,
        "pipeline_seconds": elapsed,
        "transcript_chars": len(transcript),
        "speaker_method": (result.get("speakers") or {}).get("method"),
        "speaker_turns": len(turns),
        "roles_known": (result.get("speakers") or {}).get("roles_known"),
        "document_type": v2.get("document_type"),
        "fact_count": len(facts.get("facts") or []),
        "fact_validation": facts.get("validation"),
        "note_validation": v2.get("validation"),
        "transcript_validation_severity": (result.get("validation") or {}).get("severity"),
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not provider_is_deepgram():
        print("!! PROVIDER GUARD: DEEPGRAM_API_KEY not configured — refusing to run")
        return 2
    audios = [Path(a) for a in sys.argv[1:]]
    if not audios:
        print("usage: run_unseen_audio.py <audio> [<audio> ...]")
        return 1
    summaries = []
    for a in audios:
        if not a.exists():
            print(f"!! missing file: {a}")
            continue
        print(f"\n=== processing {a.name} ({a.stat().st_size/1e6:.1f} MB) ===", flush=True)
        s = run_one(a)
        print(f"    provider={s['provider']} selection={s['provider_selection'][:80]}")
        print(f"    turns={s['speaker_turns']} method={s['speaker_method']} "
              f"roles_known={s['roles_known']}")
        print(f"    document={s['document_type']} facts={s['fact_count']} "
              f"facts_valid={(s['fact_validation'] or {}).get('valid')} "
              f"note_valid={(s['note_validation'] or {}).get('valid')}")
        if not (s["note_validation"] or {}).get("valid", False):
            for v in (s["note_validation"] or {}).get("violations", [])[:8]:
                print(f"    NOTE VIOLATION: {v}")
        summaries.append(s)
    (OUT / "batch_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nbatch complete -> {OUT / 'batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
