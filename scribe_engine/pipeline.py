"""ScribeEngine — the single entry point.

process_audio(audio_path) ->
    {
      "transcript": {text, language, segments},
      "speakers":   {method, confidence, turns},
      "clinical_note": {fields, text},
      "prescription":  {fields, text},
    }

This schema mirrors what the upstream apps produced (full transcript text,
speaker-labeled lines, medical info dict, notes/prescription text blocks),
restructured for an API/UI consumer. STT runs faster-whisper with the same
OpenAI base weights as upstream; the swap is documented in ANALYSIS.md.

An optional on_stage callback reports each real pipeline stage as it starts
and finishes, so a UI can show true progress instead of a fake animation.
"""

import logging
import re
import time
from pathlib import Path

from . import audio as audio_mod
from . import clinical, clinical_documentation, clinical_facts, diarization, prescription as prescription_mod
from . import note_v2
from .field_projection import project_structured_fields
from .stt import CURRENT_PROVIDER_ID, get_provider, resolve_production_provider
from .stt.base import SpeechLanguageUnavailable
from .stt.policy import STTPolicyBlocked, check_policy
from .validation import validate_transcript

logger = logging.getLogger(__name__)


# Kept as an alias because service code and operator docs have referred to the
# Malayalam failure mode by this name; it IS the language-mismatch failure.
MalayalamSpeechUnavailable = SpeechLanguageUnavailable


class ScribeEngine:
    def __init__(self, whisper_model_size: str = "base", device: str = "cpu",
                 stt_provider_id: str | None = None,
                 compute_type: str = "int8", cpu_threads: int = 0,
                 language: str = "en"):
        # The production path is English-only and never relies on automatic
        # language detection. Anything else the provider reports is treated as
        # a failed transcription by the validation layer, not translated.
        self.language = language
        # The engine depends on the STTProvider interface, not a concrete
        # implementation. Default: current_faster_whisper (baseline provider).
        self.whisper_model_size = whisper_model_size
        self.device = device
        self.stt_provider_id = stt_provider_id
        # Deployment knobs, passed through to the provider. cpu_threads=0 keeps
        # the library default (all cores) for standalone/CLI use; the service
        # caps it so serving the UI stays responsive during transcription.
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        # Hospital STT policy. Enforced BEFORE audio leaves the infrastructure
        # (pre-flight) and at fallback time (fail-closed mode). ``None`` means
        # "no policy" — the historical default for standalone/CLI use.
        self.stt_policy: dict | None = None
        self.stt_fail_closed: bool = False
        self.on_stage = None  # callable(dict) for live stage reporting
        self._stages: list = []

    # -- stage tracking -----------------------------------------------------
    def _stage(self, name: str):
        entry = {"stage": name, "status": "running", "ts": time.time()}
        self._stages.append(entry)
        if self.on_stage:
            self.on_stage(entry)

    def _stage_done(self, name: str, detail: str = ""):
        entry = {"stage": name, "status": "done", "ts": time.time(), "detail": detail}
        self._stages.append(entry)
        if self.on_stage:
            self.on_stage(entry)

    @property
    def last_stages(self) -> list:
        return self._stages

    # -- main entry ----------------------------------------------------------
    def process_audio(
        self,
        audio_path,
        language: str | None = None,
        model_size: str | None = None,
    ) -> dict:
        """Run the full pipeline on an audio file. Raises AudioError on bad input."""
        path = audio_mod.validate_audio(audio_path)
        self._stages = []
        t0 = time.time()

        self._stage("prepare")
        model_size = model_size or self.whisper_model_size


        # English is forced, never auto-detected, on the production path.
        requested_language = language or self.language

        self._stage("transcribe")
        provider, selection_reason = resolve_production_provider(
            self.stt_provider_id,
            language=requested_language,
            model_size=model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        # Pre-flight policy check: the provider's DECLARED boundary must
        # satisfy the hospital's policy before any audio leaves. Refusal here
        # means the audio was never sent — nothing to fall back, nothing to
        # delete, nothing leaked.
        if self.stt_policy:
            check_policy(provider, self.stt_policy)

        # The provider's DECLARED boundary, read from its capability surface
        # (the same facts the pre-flight check just validated). Travels on the
        # result meta so the audit trail can answer "why was this audio allowed
        # to leave the hospital environment?" from configuration facts alone.
        # Configuration only — no keys, no audio, no transcript content.
        _declared_rc = (provider.retention_configuration()
                        if hasattr(provider, "retention_configuration") else {})
        _declared_endpoint = (_declared_rc.get("endpoint")
                              if _declared_rc.get("audio_sent_off_host") else None)
        _declared_location = (provider.data_location()
                              if _declared_endpoint and
                              hasattr(provider, "data_location") else None)

        fallback_note = None
        try:
            stt = provider.transcribe(path, language=requested_language)
        except STTPolicyBlocked:
            # Policy refusals must never be swallowed by the language checks
            # or fallback logic below.
            raise
        except Exception as exc:
            # A hosted provider can fail transiently (network, rate limit, API
            # incident). Losing the consultation for that reason is worse than
            # transcribing locally, so fall back rather than fail — and record
            # which engine actually produced the text, since the two differ in
            # accuracy and the clinician must be able to see which was used.
            #
            # LANGUAGE MATCH, though, is not a preference but a requirement: a
            # local English engine handed Malayalam audio produces confident
            # nonsense (measured on real consultation audio, see
            # benchmarks/malayalam/real_audio_asr/). A failed language provider
            # therefore fails the consultation with a clear message instead of
            # silently degrading to an engine that cannot speak the language.
            if provider.provider_id == CURRENT_PROVIDER_ID:
                raise
            if self.stt_fail_closed:
                # Fail-closed deployment: an unexpected provider switch is a
                # privacy/residency/auditability risk. The job fails safely;
                # the source audio remains under the normal retention
                # lifecycle; another provider may only be selected explicitly
                # via deployment configuration.
                raise STTPolicyBlocked(
                    "provider_failed_fail_closed",
                    f"Transcription failed and automatic fallback is disabled "
                    f"by policy (ISCRIBE_STT_FAIL_CLOSED). The provider error "
                    f"was {type(exc).__name__}; the audio is preserved and the "
                    f"job can be retried.",
                    provider_id=provider.provider_id,
                    required={"fail_closed": True},
                )
            provider_language = getattr(provider, "language", None) or ""
            if str(requested_language or "en").split("-")[0].lower() != "en" and (
                    provider_language.split("-")[0].lower() !=
                    str(requested_language or "en").split("-")[0].lower()):
                raise MalayalamSpeechUnavailable(
                    requested_language,
                    f"{type(exc).__name__}: {exc}") from exc
            fallback_note = f"{provider.provider_id} failed ({exc}); used local fallback"
            logger.warning("STT provider %s failed, falling back to %s: %s",
                           provider.provider_id, CURRENT_PROVIDER_ID, exc)
            provider = get_provider(
                CURRENT_PROVIDER_ID,
                model_size=model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            selection_reason = fallback_note
            stt = provider.transcribe(path, language=requested_language)

        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in stt.segments]
        self._stage_done(
            "transcribe",
            f"{len(segs)} segments, lang={stt.language}, provider={provider.provider_id}")

        # -- quality guardrail: runs BEFORE any clinical interpretation ------
        self._stage("validate")
        validation = validate_transcript(
            stt.text,
            expected_language=requested_language,
            reported_language=stt.language,
            segments=segs,
        )
        self._stage_done("validate", f"severity={validation.severity}")

        self._stage("speakers")
        provider_turns = (stt.provider_meta or {}).get("speaker_turns") or []
        labeled = diarization.build_labeled_transcript(
            segs, audio_path=path, raw_transcript_text=None,
            provider_turns=provider_turns,
        )
        self._stage_done("speakers", f"method={labeled['method']}")

        def _line(t: dict) -> str:
            # Anonymous diarization labels ("Speaker 0") carry no identity, so
            # prefixing them onto the text only pollutes downstream evidence
            # spans (HPI verbatim quotes, metadata values) with turn structure.
            # Named roles (Doctor/Patient) and family roles stay prefixed.
            spk = t.get("speaker", "")
            if re.match(r"^Speaker\s*\d+$", spk, re.I):
                return t.get("text", "")
            return f"{spk}: {t.get('text', '')}"

        transcript_text = "\n".join(_line(t) for t in labeled["turns"])

        # An invalid transcript is not interpreted. Generating a clinical draft
        # from text we have evidence is wrong would manufacture findings, which
        # is the exact failure this guardrail exists to prevent.
        normalized: dict = {}
        # The Malayalam sidecar applies the verified ASR-correction pass and
        # returns the corrected COPY plus provenance; the raw transcript above
        # is untouched. Other providers have no correction layer, so
        # corrected_text stays None and the raw text is used unchanged.
        corrections_applied = (stt.provider_meta or {}).get("asr_corrections") or []
        corrected_text = (stt.provider_meta or {}).get("corrected_text") or None
        if corrected_text == stt.text:
            corrected_text = None  # nothing actually changed
        if validation.is_invalid:
            self._stage_done("prepare", f"{time.time() - t0:.1f}s total (clinical skipped)")
            note = clinical.empty_clinical_note(validation.message)
            rx = prescription_mod.empty_prescription(validation.message)
        else:
            self._stage("clinical")
            roles_known = labeled.get("roles_known", True)
            doctor_lines = [t["text"] for t in labeled["turns"] if t["speaker"] == "Doctor"]
            patient_lines = [t["text"] for t in labeled["turns"] if t["speaker"] == "Patient"]
            if not roles_known:
                # Speakers were separated but not identified. Scanning every line
                # for both sides is honest; guessing who is the doctor is not.
                all_lines = [t["text"] for t in labeled["turns"]]
                doctor_lines = patient_lines = all_lines

            structured = clinical.extract_medical_info(transcript_text)
            # Two-layer normalization (verified ASR corrections + colloquial
            # semantics) runs on a corrected COPY; the transcript itself is
            # never rewritten. With no correction layer output (English path,
            # pasted text), the raw text is used unchanged.
            normalized = clinical.build_normalized_entities(
                transcript_text, corrected_text=corrected_text)
            note = clinical.build_clinical_note(
                structured, doctor_lines, patient_lines, normalized=normalized)
            rx = prescription_mod.build_prescription(doctor_lines)
            # Structured clinical documentation layer (ABOVE the semantic layer;
            # entities stay the source of truth). Additive: clinical_note and
            # prescription are unchanged.
            documentation = clinical_documentation.build_structured_documentation(
                normalized.get("entities", []), labeled["turns"],
                roles_known=roles_known)
            # Clinical Intelligence V2 (document context -> section-aware,
            # evidence-grounded typed facts -> deterministic note + fail-closed
            # validator), mirroring the text flow.
            clinical_facts_v2 = clinical_facts.build_clinical_facts(
                transcript_text, labeled["turns"], roles_known=roles_known,
                entities=normalized.get("entities", []),
                corrected_text=corrected_text)
            clinical_note_v2 = note_v2.render_clinical_note(clinical_facts_v2)
            # SINGLE-SOURCE FIELDS: the editable review fields are a
            # deterministic projection of the SAME fact graph that produced
            # the canonical note — not an independent extraction. The legacy
            # extractor's keyword/regex output (its 'duration' regex matched
            # "buying tight" on the referral-letter case) is discarded from
            # the clinical dataflow; the extractor itself is retained below
            # only as an audit/debug artifact.
            projection = project_structured_fields(clinical_facts_v2)
            note["fields"] = projection["clinical_note_fields"]
            note["fields_source"] = "clinical_facts_v2"
            note["text"] = clinical.render_clinical_note_text(note["fields"])
            rx["fields"] = projection["prescription_fields"]
            rx["fields_source"] = "clinical_facts_v2"
            rx["text"] = prescription_mod.render_prescription_text(rx["fields"])
            note["warnings"] = projection["warnings"]
            note["confidence"] = projection["confidence"]
            note["confidence_note"] = projection["confidence_note"]
            # Clinical Intelligence V2 (document context -> section-aware,
            # evidence-grounded typed facts -> deterministic note + fail-closed
            # validator), mirroring the text flow.
            clinical_facts_v2 = clinical_facts.build_clinical_facts(
                transcript_text, labeled["turns"], roles_known=roles_known,
                entities=normalized.get("entities", []),
                corrected_text=corrected_text)
            clinical_note_v2 = note_v2.render_clinical_note(clinical_facts_v2)
            self._stage_done(
                "clinical",
                f"symptoms={len(structured['symptoms'])}, "
                f"normalized={len(normalized.get('entities', []))}")
            self._stage_done("prepare", f"{time.time() - t0:.1f}s total")

        return {
            "transcript": {
                "text": stt.text,
                "plain_text": " ".join(t["text"] for t in labeled["turns"]),
                "language": stt.language,
                "requested_language": requested_language,
                "segments": segs,
                "source_language": stt.source_language,
                "source_transcript": stt.source_transcript,
                "target_language": stt.target_language,
                "translated_transcript": stt.translated_transcript,
                "stt_provider": stt.provider_id,
                "asr_corrections_applied": corrections_applied,
            },
            "validation": validation.as_dict(),
            # Parallel to the transcript, never a substitute for it.
            "normalized_clinical_entities": normalized,
            "speakers": {
                "method": labeled["method"],
                "confidence": labeled["confidence"],
                "roles_known": labeled.get("roles_known", True),
                "turns": labeled["turns"],
            },
            "clinical_note": note,
            "clinical_documentation": documentation,
            "clinical_note_v2": clinical_note_v2,
            # The typed, evidence-grounded fact graph the note was rendered
            # from (raw_text omitted: the transcript is already above).
            "clinical_facts_v2": {k: v for k, v in clinical_facts_v2.items()
                                  if k != "raw_text"},
            "prescription": rx,
            "meta": {
                # Filename only. The full path is server-side detail and this
                # dict is nested inside the result the browser receives, so a
                # path here bypasses the top-level key stripping in the service.
                "audio_filename": path.name,
                "audio_duration_s": audio_mod.probe_duration(path),
                "whisper_model": model_size,
                "stt_provider": stt.provider_id,
                "provider_selection": selection_reason,
                "provider_fallback": fallback_note,
                "requested_language": requested_language,
                "reported_language": stt.language,
                # Provider boundary metadata: the provider's own request id
                # when it returns one (Deepgram does; local providers do not).
                # Carried for the audit trail; implies no retention behaviour.
                "provider_request_id": (stt.provider_meta or {}).get("request_id"),
                "policy_decision": "allowed",
                "policy_reason": "declared_boundary_satisfies_policy",
                "stt_endpoint": _declared_endpoint,
                "stt_region": _declared_location,
                "stt_zero_retention": (
                    True if not _declared_rc.get("audio_sent_off_host")
                    else bool(_declared_rc.get("mip_opt_out"))),
                "elapsed_s": round(time.time() - t0, 2),
                "stages": self.last_stages,
            },
        }

    def process_text(self, transcript_text: str, language: str | None = None) -> dict:
        """Run speaker parsing + clinical generation on a text transcript
        (demo/plain-text flow; no STT)."""
        lang = (language or self.language or "en").lower()
        labeled = diarization.build_labeled_transcript([], raw_transcript_text=transcript_text)
        lines = [f"{t['speaker']}: {t['text']}" for t in labeled["turns"]]
        joined = "\n".join(lines)
        roles_known = labeled.get("roles_known", True)
        doctor_lines = [t["text"] for t in labeled["turns"] if t["speaker"] == "Doctor"]
        patient_lines = [t["text"] for t in labeled["turns"] if t["speaker"] == "Patient"]
        if not roles_known:
            all_lines = [t["text"] for t in labeled["turns"]]
            doctor_lines = patient_lines = all_lines

        structured = clinical.extract_medical_info(joined)
        # Pasted text is not ASR output: run semantic normalization on it as-is
        # (no ASR-correction layer - there is nothing ASR-produced to correct).
        normalized = clinical.build_normalized_entities(joined, corrected_text=None)
        note = clinical.build_clinical_note(
            structured, doctor_lines, patient_lines, normalized=normalized)
        rx = prescription_mod.build_prescription(doctor_lines)
        documentation = clinical_documentation.build_structured_documentation(
            normalized.get("entities", []), labeled["turns"],
            roles_known=roles_known)
        # Clinical Intelligence V2: document context -> section-aware, evidence-
        # grounded typed facts -> deterministic note + fail-closed validator.
        # Pasted text is not ASR output (there is nothing ASR-produced to
        # correct), so the verified-correction copy stays None here.
        clinical_facts_v2 = clinical_facts.build_clinical_facts(
            transcript_text, labeled["turns"], roles_known=roles_known,
            entities=normalized.get("entities", []), corrected_text=None)
        clinical_note_v2 = note_v2.render_clinical_note(clinical_facts_v2)
        # SINGLE-SOURCE FIELDS (text flow): same projection as the audio
        # path — the review fields are the fact graph, not extractor B.
        projection = project_structured_fields(clinical_facts_v2)
        note["fields"] = projection["clinical_note_fields"]
        note["fields_source"] = "clinical_facts_v2"
        note["text"] = clinical.render_clinical_note_text(note["fields"])
        rx["fields"] = projection["prescription_fields"]
        rx["fields_source"] = "clinical_facts_v2"
        rx["text"] = prescription_mod.render_prescription_text(rx["fields"])
        note["warnings"] = projection["warnings"]
        note["confidence"] = projection["confidence"]
        note["confidence_note"] = projection["confidence_note"]

        return {
            "transcript": {
                "text": joined,
                "plain_text": " ".join(t["text"] for t in labeled["turns"]),
                "language": lang,
                "segments": [],
                "source_language": None,
                "source_transcript": None,
                "target_language": None,
                "translated_transcript": None,
                "stt_provider": "text-input (no STT)",
                "requested_language": lang,
            },
            "validation": validate_transcript(
                joined, expected_language=lang, reported_language=lang,
            ).as_dict(),
            "normalized_clinical_entities": normalized,
            "speakers": {
                "method": labeled["method"],
                "confidence": labeled["confidence"],
                "roles_known": roles_known,
                "turns": labeled["turns"],
            },
            "clinical_note": note,
            "clinical_documentation": documentation,
            "clinical_note_v2": clinical_note_v2,
            "clinical_facts_v2": {k: v for k, v in clinical_facts_v2.items()
                                  if k != "raw_text"},
            "prescription": rx,
        }
