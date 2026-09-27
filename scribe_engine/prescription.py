"""Prescription / plan generation.

Adapted from upstream front.py: the prescription was an f-string template fed by
the doctor-line medication regex. Reproduced here as a pure function returning
structured data plus the upstream-style text block. Upstream hard-coded boilerplate
(dosage/follow-up/lifestyle sentences) is kept verbatim but each field is marked
"[template]" in the structured output so the UI can visually distinguish extracted
content from fixed boilerplate — the demo must not present template text as
extracted fact.
"""

import re

# Upstream front.py medication regex (verbatim list)
MEDICATION_PATTERN = (
    r"\b(allegra|claritin|zyrtec|zytech|spray|anti-histamine|inhaler|paracetamol|"
    r"cetirizine|ibuprofen|antibiotic|nasal drop|metformin|vitamin-b complex|amlodipine)\b"
)

# CLINICAL SAFETY RULE: these fields used to carry hard-coded upstream
# boilerplate — "Continue medication for 5-7 days", "Review patient response in
# 1-2 weeks", dietary and lifestyle advice. That is a treatment plan, a duration
# and a follow-up interval no clinician prescribed, and marking it [template] did
# not stop it reaching the exported record. Every field now defaults to
# "Not mentioned" and is only populated from what was actually said.
NOT_MENTIONED = "Not mentioned"

TEMPLATE = {
    "dosage_instructions": NOT_MENTIONED,
    "duration_of_treatment": NOT_MENTIONED,
    "follow_up": NOT_MENTIONED,
    "lifestyle_advice": NOT_MENTIONED,
    "dietary_guidance": NOT_MENTIONED,
    "precautions": NOT_MENTIONED,
}


def build_prescription(doctor_lines: list) -> dict:
    text = " ".join(doctor_lines).lower()  # upstream lowercases before matching
    medications = sorted(set(re.findall(MEDICATION_PATTERN, text, re.IGNORECASE)))

    # Per-medication dosage context (upstream front.py med_dosage_map)
    med_details = {}
    for med in medications:
        ctx = [ln.strip() for ln in doctor_lines if med.lower() in ln.lower()]
        # No dosage was spoken -> say so. Never supply a default dose.
        med_details[med] = ctx or ["Dosage/frequency not mentioned"]

    # Verbatim quote of the doctor's closing lines. Kept because it is traceable
    # to the transcript, and marked as a quote so it is never mistaken for a
    # structured instruction the system derived.
    advice_tail = " ".join(doctor_lines[-2:]).strip() if doctor_lines else ""

    fields = {
        "medications": [
            {"name": m, "instructions": med_details[m]} for m in medications
        ],
        **TEMPLATE,
        "additional_advice": (
            f'Doctor said (verbatim): "{advice_tail}"' if advice_tail else NOT_MENTIONED
        ),
    }

    return {"fields": fields, "text": render_prescription_text(fields)}


def empty_prescription(reason: str) -> dict:
    """A prescription block for a transcript that failed validation."""
    fields = {key: NOT_MENTIONED for key, _ in RX_LAYOUT}
    fields.update({
        "medications": [],
        "not_generated": True,
        "not_generated_reason": reason,
    })
    text = ("Prescription / Plan:\n"
            "No prescription or plan was generated because the transcript did not "
            f"pass quality validation.\nReason: {reason}\n")
    return {"fields": fields, "text": text}


# Field key -> numbered heading, in the order upstream front.py printed them.
RX_LAYOUT = (
    ("dosage_instructions", "Dosage instructions"),
    ("duration_of_treatment", "Duration of treatment"),
    ("follow_up", "Follow-up"),
    ("lifestyle_advice", "Lifestyle advice"),
    ("dietary_guidance", "Dietary guidance"),
    ("additional_advice", "Additional advice"),
    ("precautions", "Precautions"),
)


def _render_medications(medications) -> str:
    """Render the medications block, tolerating doctor-edited shapes.

    Review hands back whatever the textarea contained, so this accepts the
    generated list-of-dicts, a plain list, or free text.
    """
    if not medications:
        return "   - Not prescribed"
    if isinstance(medications, str):
        return f"   - {medications}"
    lines = []
    for med in medications:
        if isinstance(med, dict):
            name = med.get("name", "unnamed")
            detail = med.get("instructions") or ["Dosage/frequency not specified"]
            if isinstance(detail, str):
                detail = [detail]
            lines.append(f"   - {name}: {' | '.join(str(d) for d in detail)}")
        else:
            lines.append(f"   - {med}")
    return "\n".join(lines)


def render_prescription_text(fields: dict) -> str:
    """Render the prescription text block from its fields.

    Fields are the single source of truth so that edits made by the doctor
    during review appear in the exported document.
    """
    lines = [
        "Prescription / Plan:",
        "1. Medications / interventions:",
        _render_medications(fields.get("medications")),
    ]
    for index, (key, heading) in enumerate(RX_LAYOUT, start=2):
        value = fields.get(key)
        lines.append(f"{index}. {heading}: {value if value not in (None, '') else 'Not specified'}")
    return "\n".join(lines) + "\n"
