"""CurrentSTTProvider — the demo's baseline STT.

Wraps the previously-proven transcription.transcribe() (faster-whisper, same
OpenAI base weights as upstream) WITHOUT changing its behavior. This is the
benchmark Pipeline A and the default for the demo.
"""

from ..transcription import transcribe as _fw_transcribe
from .base import STTProvider, Transcript


class CurrentSTTProvider(STTProvider):
    provider_id = "current_faster_whisper"

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 0):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads

    def transcribe(self, audio_path, language=None) -> Transcript:
        out = _fw_transcribe(
            audio_path,
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=language,
            cpu_threads=self.cpu_threads,
        )
        return Transcript(
            text=" ".join(s.text for s in out["segments"]),
            segments=out["segments"],
            language=out["language"],
            duration=out.get("duration"),
            provider_id=self.provider_id,
            provider_meta={"model_size": self.model_size, "device": self.device,
                           "cpu_threads": self.cpu_threads},
        )
