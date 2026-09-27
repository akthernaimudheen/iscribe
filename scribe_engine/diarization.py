"""Speaker handling.

Upstream has two conflicting approaches, both reused in adapted form:

1. `final1.py` / `automatic labeling.py`: pyannote diarization + timestamp merge,
   then a hard-coded map SPEAKER_00→Doctor, SPEAKER_01→Patient.
   NOT enabled by default: pyannote requires a HuggingFace token and acceptance of
   the gated pyannote/speaker-diarization-3.1 model (upstream's committed token is
   leaked and must not be reused). The pyannote backend is implemented and activates
   automatically when (a) the package is installed, (b) HF_TOKEN is set.
2. `front.py` / `label.py` / `finalfront.py`: alternate DOCTOR/PATIENT labels per
   segment index. Kept as the fallback labeling heuristic; the result is marked
   low-confidence because correctness is coin-flip by construction. One minimal
   demo fix is added: if the opening segment reads patient-like, the label phase
   is swapped (see _flip_alternating).
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# Adapted from upstream final1.py speaker_map.
DEFAULT_SPEAKER_MAP = {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}

_ROLE_RE = re.compile(r"^\s*(doctor|dr\.?|patient|parent)\s*:", re.IGNORECASE)

_PATIENT_HINTS = (
    "i have been having", "i've been having", "i have", "i've",
    "i am", "i'm", "my ", "doctor,",
)
_DOCTOR_HINTS = (
    "how can i help", "how may i help", "what brings you in",
    "what brings you here", "good morning, i am", "good evening, i am",
)


class DiarizationError(RuntimeError):
    pass


def _pyannote_diarize(audio_path):
    """Run pyannote if available. Returns list of speaker turns, or None when unavailable."""
    try:
        from pyannote.audio import Pipeline  # optional dependency
    except ImportError:
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        diarization = pipe(str(audio_path))
    except Exception as e:
        raise DiarizationError(f"pyannote diarization failed: {e}") from e
    return [
        {"speaker": s, "start": float(t.start), "end": float(t.end)}
        for t, _, s in diarization.itertracks(yield_label=True)
    ]


def _has_explicit_role_labels(transcript_text: str) -> bool:
    """True if transcript lines carry explicit Doctor:/Patient: prefixes."""
    lines = [ln for ln in transcript_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    hits = sum(1 for ln in lines if _ROLE_RE.match(ln))
    return hits / len(lines) >= 0.8


def _parse_explicit_roles(transcript_text: str):
    """Parse 'Doctor: ...' / 'Patient: ...' lines into segments; parent→Patient."""
    segments = []
    for ln in transcript_text.splitlines():
        if not ln.strip():
            continue
        m = _ROLE_RE.match(ln)
        if not m:
            continue
        role = m.group(1).lower()
        if role.startswith("dr"):
            role = "doctor"
        if role == "parent":
            role = "patient"
        segments.append({
            "start": float(len(segments)),
            "end": float(len(segments) + 1),
            "text": ln[m.end():].strip(),
            "speaker": role.capitalize(),
        })
    return segments


def _speaker_dominance(speaker_turns, seg_start, seg_end):
    """Overlap-weighted vote of diarization turns within a segment window."""
    best_speaker, best_overlap = None, 0.0
    for turn in speaker_turns:
        overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
    return best_speaker, best_overlap


def _heuristic_role_map(speaker_turns):
    """Decide Doctor/Patient for pyannote speaker IDs.

    Upstream hard-codes SPEAKER_00→Doctor. Kept as the tie-break default, but the
    more talkative speaker is assumed to be the patient (patients explain symptoms
    at length; doctors steer). Confidence stays 'low' — heuristic, not ground truth.
    """
    totals: dict = {}
    for t in speaker_turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    if not totals:
        return {}, "low"
    speakers = sorted(totals, key=lambda s: totals[s], reverse=True)
    if len(speakers) == 1:
        return {speakers[0]: "Patient"}, "low"
    patient, doctor = speakers[0], speakers[1]
    return {patient: "Patient", doctor: "Doctor"}, "low"


def _flip_alternating(segments) -> bool:
    """True when the alternating pattern should start with Patient.

    The patient usually opens the consultation. When the FIRST segment reads
    patient-like (symptom statement) the even/odd label phases are swapped.
    """
    if not segments:
        return False
    first = (segments[0].get("text") or "").lower()
    # Doctor-opening cues win first: 'I am Dr. Rao, how can I help' also contains
    # the ambiguous patient hint 'i am'. Only when the line does not read like a
    # doctor greeting do patient-side cues decide the phase.
    if any(h in first for h in _DOCTOR_HINTS):
        return False
    return any(h in first for h in _PATIENT_HINTS)


def build_labeled_transcript(
    segments,
    audio_path=None,
    raw_transcript_text: str | None = None,
    provider_turns: list | None = None,
) -> dict:
    """Return {"turns": [...], "method": str, "confidence": str, "roles_known": bool}.

    method: "explicit_roles" | "provider_diarization" | "diarized" | "alternating"

    ``roles_known`` says whether Doctor/Patient identities are trustworthy. When
    it is False the speakers were separated but NOT identified, and callers must
    not treat a turn as the doctor's or the patient's words.
    """
    # 1) Explicit role prefixes in the transcript itself (demo fixtures, plain-text flow)
    if raw_transcript_text and _has_explicit_role_labels(raw_transcript_text):
        return {
            "turns": _parse_explicit_roles(raw_transcript_text),
            "method": "explicit_roles",
            "confidence": "high",
            "roles_known": True,
        }

    # 1b) Raw text with no role prefixes and no audio segments — a pasted
    #     transcript. Previously this fell through to the alternating branch
    #     with an empty segment list and produced ZERO turns, silently
    #     discarding the entire transcript. Keep the text as one unlabelled
    #     turn: we genuinely do not know who spoke, and inventing speakers
    #     would be worse than admitting that.
    if raw_transcript_text and not segments:
        text = raw_transcript_text.strip()
        if not text:
            return {"turns": [], "method": "empty", "confidence": "low",
                    "roles_known": False}
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return {
            "turns": [{"speaker": "Speaker", "start": float(i), "end": float(i + 1),
                       "text": ln} for i, ln in enumerate(lines)],
            "method": "unlabelled_text",
            "confidence": "low",
            "roles_known": False,
        }

    # 2) Speaker turns supplied by the STT provider (e.g. Deepgram diarization).
    #    These separate speakers reliably but carry no identity. Upstream mapped
    #    SPEAKER_00 -> Doctor by position; that is a guess, and a guess that
    #    attributes a patient's words to a clinician is not acceptable here, so
    #    the neutral labels are preserved for the clinician to assign.
    if provider_turns:
        return {
            "turns": [
                {"speaker": t.get("speaker", "Speaker"), "start": t.get("start", 0.0),
                 "end": t.get("end", 0.0), "text": t.get("text", "")}
                for t in provider_turns if (t.get("text") or "").strip()
            ],
            "method": "provider_diarization",
            "confidence": "medium",
            "roles_known": False,
        }

    # 2) pyannote diarization (optional backend)
    if audio_path is not None:
        speaker_turns = _pyannote_diarize(audio_path)
        if speaker_turns:
            role_map, conf = _heuristic_role_map(speaker_turns)
            turns = []
            for s in segments:
                spk, _ = _speaker_dominance(speaker_turns, s["start"], s["end"])
                role = role_map.get(spk, "Unknown")
                turns.append({
                    "speaker": role, "start": s["start"], "end": s["end"], "text": s["text"],
                })
            return {"turns": turns, "method": "diarized", "confidence": conf,
                    "roles_known": False}

    # 4) Alternating heuristic (upstream front.py behavior), low confidence,
    #    with the patient-first phase swap fix. roles_known is False because the
    #    labels are a coin flip by construction.
    flip = _flip_alternating(segments)
    turns = []
    for i, s in enumerate(segments):
        if flip:
            role = "Patient" if i % 2 == 0 else "Doctor"
        else:
            role = "Doctor" if i % 2 == 0 else "Patient"
        turns.append({"speaker": role, "start": s["start"], "end": s["end"], "text": s["text"]})
    return {"turns": turns, "method": "alternating", "confidence": "low",
            "roles_known": False}
