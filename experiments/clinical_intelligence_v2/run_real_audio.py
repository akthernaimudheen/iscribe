# -*- coding: utf-8 -*-
"""Task 17: the newly uploaded English audio through the real production path.

Uses ScribeEngine.process_audio() unchanged — the same entry point the service
calls — so the provider is whatever resolve_production_provider() picks for
English (Deepgram nova-3-medical when DEEPGRAM_API_KEY is configured). No ASR
substitution, no transcript hand-editing.

    python experiments/clinical_intelligence_v2/run_real_audio.py <audio> [lang]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from scribe_engine import ScribeEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
DEFAULT_AUDIO = ROOT / "data" / "uploads" / "884764a5d6e9.mp3"


def main() -> int:
    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    OUT.mkdir(parents=True, exist_ok=True)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")          # so provider resolution sees DEEPGRAM_API_KEY
    engine = ScribeEngine()
    result = engine.process_audio(str(audio), language=language)

    meta = result.get("meta") or {}
    provider = meta.get("stt_provider")
    print(f"audio       : {audio.name} ({audio.stat().st_size} bytes)")
    print(f"provider    : {provider} ({meta.get('provider_reason') or meta.get('stt_reason')})")
    # Phase-2 assertion: the production English path must be Deepgram when the
    # key is configured. A silent fallback to Whisper would invalidate the run.
    from dotenv import load_dotenv as _ld
    _ld(ROOT / ".env")
    import os as _os
    deepgram_expected = bool(_os.environ.get("DEEPGRAM_API_KEY"))
    if deepgram_expected and provider != "deepgram":
        print(f"!! PROVIDER GUARD: expected deepgram, got {provider!r}")
        return 2
    print(f"language    : {meta.get('language')}")
    transcript = (result.get("transcript") or {}).get("text") or ""
    print(f"transcript  : {len(transcript)} chars")

    v2 = result.get("clinical_note_v2") or {}
    facts = result.get("clinical_facts_v2") or {}
    print(f"document    : {v2.get('document_type')}")
    print(f"facts       : {len(facts.get('facts') or [])}")
    print(f"validation  : {v2.get('validation')}")

    (OUT / f"{audio.stem}.transcript.txt").write_text(transcript, encoding="utf-8")
    (OUT / f"{audio.stem}.note_v2.txt").write_text(v2.get("note", ""), encoding="utf-8")
    (OUT / f"{audio.stem}.facts_v2.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"{audio.stem}.result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print(v2.get("note", ""))
    print("=" * 72)
    print("\nFACTS")
    for f in facts.get("facts") or []:
        print(f"  {f['fact_id']:>4} {f['fact_type']:<12} {f['status']:<10} "
              f"{f['certainty']:<10} {(f['temporal_context'] or ''):<12} "
              f"{f['section']:<20} {f['english']!r}  <- {f['source_text'][:60]!r}")
    print(f"\nartifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
