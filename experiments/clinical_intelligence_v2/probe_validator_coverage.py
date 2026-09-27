# -*- coding: utf-8 -*-
"""Probe: could a strict 'every note bullet must trace to a fact' check pass?"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scribe_engine.clinical_facts import build_clinical_facts
from scribe_engine.note_v2 import render_clinical_note, _norm_label

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
CASES = {
    "referral": FIX / "english_real_cases" / "referral_letter_orthopaedic.txt",
    "jones": FIX / "english_real_cases" / "medicolegal_jones.txt",
    "finton": FIX / "english_real_cases" / "medicolegal_finton.txt",
    "malayalam": FIX / "malayalam_real_consultation.txt",
}

STOP = set("""the a an of and or to in on for with at by from as is are was were be been
being has have had does do did not no this that these those it its his her their
patient patients he she they mr mrs ms doctor clinician notes note reported reports
reports per approximately about since then than there here also some little well
""".split())

for name, path in CASES.items():
    text = path.read_text(encoding="utf-8")
    doc = build_clinical_facts(text, [], roles_known=False)
    out = render_clinical_note(doc)
    note = out["note"]
    labels = set()
    for f in doc["facts"]:
        for key in ("english", "concept", "evidence_text", "source_text"):
            if f.get(key):
                labels |= {w for w in _norm_label(str(f[key])).split() if w not in STOP}
        for v in (f.get("attributes") or {}).values():
            if v:
                labels |= {w for w in _norm_label(str(v)).split() if w not in STOP}
    unresolved = []
    for line in note.splitlines():
        s = line.strip()
        if not s.startswith("-"):
            continue
        words = {w for w in _norm_label(s.lstrip("- ")).split() if w not in STOP}
        if not words & labels:
            unresolved.append(s)
    print(f"--- {name}: facts={len(doc['facts'])} unresolved_bullets={len(unresolved)}")
    for u in unresolved:
        print("    ", u)
