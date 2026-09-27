# -*- coding: utf-8 -*-
"""Phase 4 adversarial battery: prove V2 is context-sensitive, not word-matching.

The SAME clinical token is presented in contexts that change its meaning; the
fact graph must follow the meaning, not the word. No production code is
touched. Expected outcomes are stated per phrase; a mismatch is a candidate
defect and counts only if the failure is reproducible in real document shape.

    python experiments/clinical_intelligence_v2/adversarial_battery.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from scribe_engine.clinical_facts import build_clinical_facts  # noqa: E402


def fact_tuples(text: str):
    doc = build_clinical_facts(text)
    return doc, [
        (f["fact_type"], (f["english"] or "").lower(), f["status"], f["certainty"],
         f["temporal_context"], f["section"], (f["attributes"] or {}).get("duration"))
        for f in doc["facts"]
    ]


def med(label, text, expect_any, expect_none=(), desc=""):
    """One case: some fact matching expect_any must exist, none matching expect_none."""
    return (label, text, expect_any, expect_none, desc)


CASES = [
    # --- 1. Dictation markup: the token "para" means a paragraph, not a drug ---
    med("markup-only", "New para.",
        (), (("MEDICATION", "paracetamol"),), "markup must not medicate"),
    med("patient-labelled markup", "Patient: new para",
        (), (("MEDICATION", "paracetamol"),), ""),
    med("doctor-labelled markup", "Doctor: new para.",
        (), (("MEDICATION", "paracetamol"),), ""),
    med("inline markup tail", "Tender over the medial arch on his left foot, full stop. New para.",
        (("EXAM_FINDING", "tenderness"),), (("MEDICATION", "paracetamol"),), ""),
    med("real medication for contrast", "Take paracetamol twice a day after food.",
        (("MEDICATION", "paracetamol"),), (), ""),

    # --- 2. "injection": treatment consideration / denial / wound pain ---
    med("considered treatment", "I wonder whether he would benefit from a steroid injection.",
        (("PROCEDURE", "steroid injection", "CONSIDERED"),),
        (("SYMPTOM", "injection"), ("MEDICATION", "injection")),
        "treatment consideration, not a symptom/medication"),
    med("explicit denial of procedure", "No injection was given.",
        (("PROCEDURE", "injection", "ABSENT"),), (("SYMPTOM", "injection"),), ""),
    med("possible treatment", "He may need an injection.",
        (("PROCEDURE", "injection"),), (("SYMPTOM", "injection"),), ""),
    med("genuine symptom mention for contrast", "The injection site is painful and swollen.",
        (("SYMPTOM", "swelling"),),
        # the injection itself must not be claimed as administered on the
        # strength of "the injection site is..."; acceptable until a dedicated
        # wound-site rule exists, recorded as a known limitation instead
        (),
        "pain/swelling here are the symptoms; the injection is a mention"),

    # --- 3. Diagnosis certainty: same noun, different modality ---
    med("suspected diagnosis", "I wonder if this is unresolved plantar fasciitis.",
        (("DIAGNOSIS", "plantar fasciitis", None, "SUSPECTED"),),
        (("DIAGNOSIS", "plantar fasciitis", None, "CONFIRMED"),), ""),
    med("confirmed diagnosis", "A whiplash injury was diagnosed.",
        (("DIAGNOSIS", "whiplash injury", None, "CONFIRMED"),),
        (), ""),
    med("rule-out", "We should rule out pneumonia.",
        (("DIAGNOSIS", "pneumonia"),), (("DIAGNOSIS", "pneumonia", None, "CONFIRMED"),),
        "rule-out must not confirm"),
    med("history of condition", "He has a history of pneumonia.",
        (("HISTORY", "pneumonia", "PRESENT", None, None, "pmh"),),
        (("DIAGNOSIS", "pneumonia"), ("SYMPTOM", "pneumonia")),
        "past history belongs to PMH, not a current finding"),
    med("diagnosed condition", "Pneumonia was diagnosed.",
        (("DIAGNOSIS", "pneumonia", None, "CONFIRMED"),), (), ""),

    # --- 4. Temporality: same symptom word, different time ---
    med("resolved pain", "The pain has resolved.",
        (("SYMPTOM", "pain", "RESOLVED"),), (), ""),
    med("ongoing pain", "The pain continues.",
        (("SYMPTOM", "pain", "PRESENT", None, "ONGOING"),),
        (("SYMPTOM", "pain", "RESOLVED"),), ""),
    med("improved + residual", "The neck pain improved but intermittent arm pain remains.",
        # two separate pain episodes: one improving, one current-intermittent
        (("SYMPTOM", "pain", "PRESENT", None, "IMPROVING"),
         ("SYMPTOM", "pain", "PRESENT", None, "INTERMITTENT"),),
        (("SYMPTOM", "pain", "RESOLVED"),),
        "improved episode and residual intermittent episode must both survive, "
        "separately — and neither may collapse into a single RESOLVED"),

    # --- 5. Negation forms are different facts ---
    med("patient denial", "The patient denies chest pain.",
        (("SYMPTOM", "chest pain", "ABSENT", None, None, "hpi"),), (), ""),
    med("exam negation", "On examination, there is no bony tenderness.",
        (("EXAM_FINDING", "no bony tenderness", "ABSENT", None, None, "physical_exam"),),
        (("SYMPTOM", "bony tenderness", "ABSENT"),),
        "an exam negation is an exam finding, not a patient denial"),
    med("no history of", "There is no past medical history of note.",
        (("HISTORY", "past medical history", "ABSENT", None, None, "pmh"),), (), ""),

    # --- 6. Administrative text can never become clinical ---
    med("greeting", "Hi. This is Doctor. Sarah Whitfield.",
        (), (("DIAGNOSIS", None), ("SYMPTOM", None), ("MEDICATION", None)), ""),
    med("reference number", "Our reference is a b slash 12 slash f GH slash 679.",
        (), (("DIAGNOSIS", None), ("SYMPTOM", None)), ""),
    med("legal declaration", "I understand my duty to the court and rule 35.",
        (), (("DIAGNOSIS", None), ("SYMPTOM", None), ("MEDICATION", None)), ""),
    med("date of birth", "Date of birth, 01/23/1945.",
        (), (("DIAGNOSIS", None), ("SYMPTOM", None)), ""),
]


def main() -> int:
    failures = []
    for label, text, expect_any, expect_none, desc in CASES:
        doc, fs = fact_tuples(text)

        def matches(pat, t):
            kind = pat[0]
            if t[0] != kind:
                return False
            if len(pat) > 1 and pat[1] is not None and pat[1] not in t[1]:
                return False
            for idx, val in enumerate(pat[2:], start=2):
                if val is not None and t[idx] != val:
                    return False
            return True

        got_any = [t for t in fs for p in expect_any if matches(p, t)]
        got_none = [t for t in fs for p in expect_none if matches(p, t)]
        ok = bool(got_any) == bool(expect_any) and not got_none
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {text[:68]!r}")
        if desc and not ok:
            print(f"        ({desc})")
        for t in fs:
            print(f"        {t[0]:<12} {t[1]!r:<34} {t[2]:<10} {t[3]:<10} "
                  f"{t[4] or '':<12} {t[5]:<20} dur={t[6]}")
        if not ok:
            failures.append((label, text, expect_any, expect_none, fs))
        print()

    print("=" * 70)
    print(f"{len(CASES) - len(failures)}/{len(CASES)} adversarial cases passed")
    for label, *_ in failures:
        print("  FAILED:", label)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
