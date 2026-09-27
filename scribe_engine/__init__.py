"""scribe_engine — clean extraction of the AI-MEDICAL-SCRIBE processing pipeline.

The engine reuses the working parts of the upstream repository
(https://github.com/AAC-Open-Source-Pool/AI-MEDICAL-SCRIBE, commit faf97d5)
behind a single entry point. See ATTRIBUTION.md and ANALYSIS.md for provenance.

`ScribeEngine` is exported lazily (PEP 562) so that STT-only environments
(e.g. the isolated Malayalam experiment) can import scribe_engine.stt.*
without pulling in spaCy/clinical dependencies.
"""

__all__ = ["ScribeEngine"]


def __getattr__(name):
    if name == "ScribeEngine":
        from .pipeline import ScribeEngine
        return ScribeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
