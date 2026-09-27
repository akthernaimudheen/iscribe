"""Speech-to-text.

Upstream uses `whisper.load_model("base")` (openai-whisper) in every script.
We wrap faster-whisper, which runs the SAME OpenAI base weights converted to
CTranslate2, on CPU, with models fetched from HuggingFace. Reason for the swap
(documented in ANALYSIS.md §6): upstream's model CDN (openaipublic.azureedge.net)
is unreachable from this machine, while HuggingFace is reachable. The output
schema (segments with start/end/text, language detection) mirrors openai-whisper's
so the underlying implementation can be swapped back without touching callers.
"""

from dataclasses import dataclass

_MODEL_CACHE: dict = {}


@dataclass
class Segment:
    start: float
    end: float
    text: str


class TranscriptionError(RuntimeError):
    pass


def _get_model(size: str, device: str, compute_type: str, cpu_threads: int = 0):
    """Load (and cache) a Whisper model.

    ``cpu_threads`` caps CTranslate2's intra-op thread pool. 0 means "use the
    library default", which is every core. Capping it matters when the process
    also has to serve HTTP: with all cores saturated the asyncio event loop gets
    starved and the UI's progress polling drops its connection mid-job.
    """
    from faster_whisper import WhisperModel  # deferred: heavy import

    key = (size, device, compute_type, cpu_threads)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(
                size, device=device, compute_type=compute_type, cpu_threads=cpu_threads
            )
        except Exception as e:  # e.g. model download blocked
            raise TranscriptionError(
                f"Could not load Whisper model '{size}': {e}"
            ) from e
    return _MODEL_CACHE[key]


def transcribe(
    audio_path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,  # None = auto-detect, like upstream
    cpu_threads: int = 0,
) -> dict:
    """Transcribe audio. Returns {"segments": [Segment], "language": str, "duration": float}."""
    model = _get_model(model_size, device, compute_type, cpu_threads)
    raw, info = model.transcribe(str(audio_path), language=language, vad_filter=False)
    segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in raw]
    return {
        "segments": segments,
        "language": info.language,
        "duration": info.duration,
    }
