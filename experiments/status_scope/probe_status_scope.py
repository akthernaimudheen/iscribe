# -*- coding: utf-8 -*-
"""Before-fix probe: the 8 required cases + real-consultation transcript.

Documents current behavior so the fix is measured against a recorded baseline.
Run: python experiments/status_scope/probe_status_scope.py
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scribe_engine.semantic import semantic_normalize  # noqa: E402
from scribe_engine.normalization import load_lexicon, split_clauses  # noqa: E402

CASES = [
    ("main", "പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി. പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്."),
    ("denied", "പനി ഇല്ല."),
    ("past", "പനി ഉണ്ടായിരുന്നു."),
    ("now_absent", "ഇപ്പോൾ പനി ഇല്ല."),
    ("resolved", "പനി മാറി."),
    ("still_present", "ക്ഷീണം ഇപ്പോഴും ഉണ്ട്."),
    ("two_concepts_shared", "ചുമയും കഫക്കെട്ടും ഉണ്ടായിരുന്നു, ഇപ്പോൾ ചുമ മാറി പക്ഷേ കഫക്കെട്ട് ഉണ്ട്."),
    ("denied_but_present", "പനി ഇല്ല, പക്ഷേ ക്ഷീണം ഉണ്ട്."),
]

# Constructed multi-clause consultation exercising the same semantic rules as
# the (untracked, real) frozen transcript: no real patient content in git.
SYNTHETIC_CONSULTATION = (
    "ഗുഡ് മോണിം ഡോക്ടർ. പനി മാറിട്ടും ഇപ്പോഴും ഭയങ്കര ക്ഷീണവും ഉണ്ട്. "
    "ഇടക്ക് തലവേദനയും ഉണ്ട്. കഫക്കെട്ട് ഇല്ല. "
    "പക്ഷേ ഫുഡ് കഴിക്കാൻ ബുദ്ധിമുട്ട് ഉണ്ട്.")


def show(text: str, label: str) -> None:
    lex = load_lexicon()
    print(f"### {label}: {text}")
    print("  clauses:", " || ".join(split_clauses(text, lex)))
    for e in semantic_normalize(text):
        print(f"  -> {e.concept:24s} {e.status:10s} conf={e.confidence} "
              f"clause='{e.source_clause[:60]}'")
    print()


if __name__ == "__main__":
    for label, text in CASES:
        show(text, label)
    show(SYNTHETIC_CONSULTATION, "SYNTHETIC multi-clause consultation")
