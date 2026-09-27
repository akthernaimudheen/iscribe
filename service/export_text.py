"""Export renderer with the approved-only clinical document path.

Split out of service/app.py so the export contract is testable in isolation:
the doctor-facing EHR export carries the APPROVED, doctor-edited note only —
no raw transcript, no draft AI reasoning, no internal validation/debug
sections.
"""

from __future__ import annotations


def _export_text(record: dict, approved_only: bool = False) -> str:
    result = record.get("result") or {}
    note = result.get("clinical_note", {})
    rx = result.get("prescription", {})
    v2 = result.get("clinical_note_v2") or {}
    v2_valid = bool(v2.get("validation", {}).get("valid"))
    speakers = result.get("speakers", {})
    meta = result.get("meta", {})
    validation = result.get("validation", {})
    reviewed = "yes" if record.get("reviewed") else "NO - not yet reviewed"

    if approved_only:
        # The EHR document carries the DOCTOR-EDITED text: the review PATCH
        # re-rendered note['text'] from the edited fields, so it wins over the
        # AI-generated V2 draft whenever the doctor touched the note. The
        # validated V2 note is used when there is no edited legacy text.
        note_body = note.get("text") if note.get("text") else \
            (v2.get("note") if v2_valid else "(no clinical note)")
        # The EHR document: the doctor-edited fields re-rendered, plus the
        # validated V2 note when present. No transcript, no quality banner,
        # no legacy/debug sections, no internal model metadata beyond the
        # factual STT-engine line the trial requires.
        lines = [
            "iSCRIBE - CONSULTATION RECORD",
            "Approved clinical note. Exported after clinician review and sign-off.",
            "",
            f"Consultation : {record['id']}",
            f"Patient      : {record['patient_id']}",
            f"Doctor       : {record['doctor']}",
            f"Department   : {record['department']}",
            f"Type         : {record['consultation_type']}",
            f"Approved at  : {record.get('completed_at') or record.get('created_at')}",
            f"Reviewed     : yes",
            f"Language     : {meta.get('reported_language') or 'en'} "
            f"(requested: {meta.get('requested_language') or 'en'})",
            "",
            "=" * 60,
            "CLINICAL NOTE (approved)",
            "=" * 60,
            note_body,
            "",
            "=" * 60,
            rx.get("text", "(no prescription)"),
            "",
            "-" * 60,
            'Fields reading "Not mentioned" were not stated in this consultation.',
            "Nothing in this note is inferred or filled in from a template.",
        ]
        return "\n".join(lines)

    lines = [
        "iSCRIBE - CONSULTATION RECORD",
        "Trial software. Clinician review and sign-off required before clinical use.",
        "",
    ]
    # A quality warning belongs at the top of the document, not buried.
    if validation and validation.get("severity") != "ok":
        lines += [
            "*" * 60,
            f"TRANSCRIPT QUALITY: {str(validation.get('severity', '?')).upper()}",
            validation.get("message", ""),
            "*" * 60,
            "",
        ]
    lines += [
        f"Consultation : {record['id']}",
        f"Patient      : {record['patient_id']}",
        f"Doctor       : {record['doctor']}",
        f"Department   : {record['department']}",
        f"Type         : {record['consultation_type']}",
        f"Generated    : {record.get('completed_at') or record.get('created_at')}",
        f"Reviewed     : {reviewed}",
        f"Language     : {meta.get('reported_language') or 'en'} "
        f"(requested: {meta.get('requested_language') or 'en'})",
        f"STT engine   : {meta.get('stt_provider', 'n/a')}",
        f"Speakers     : {speakers.get('method', 'n/a')} "
        f"(confidence: {speakers.get('confidence', 'n/a')}; "
        f"roles {'identified' if speakers.get('roles_known') else 'NOT identified'})",
        "",
        "=" * 60,
        "TRANSCRIPT",
        "=" * 60,
        result.get("transcript", {}).get("text", "(no transcript)"),
        "",
        "=" * 60,
        # ONE authoritative clinical note: the validated fact-graph note when
        # it validated; otherwise the legacy template text.
        (v2.get("note") if v2_valid and v2.get("note")
         else note.get("text", "(no clinical note)")),
        "",
        "=" * 60,
        rx.get("text", "(no prescription)"),
        "",
        "-" * 60,
        'Fields reading "Not mentioned" were not stated in this consultation.',
        "Nothing in this note is inferred or filled in from a template.",
        "Verify every field against the transcript before signing.",
    ]
    if v2_valid and note.get("text"):
        lines += [
            "",
            "-" * 60,
            "LEGACY TEMPLATE NOTE — superseded; reconciliation/debug only, "
            "not for clinical use",
            "-" * 60,
            note["text"],
        ]
    return "\n".join(lines)
