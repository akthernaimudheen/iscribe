"""Malayalam ASR sidecar.

Runs the AI4Bharat Malayalam stack in its own process because it cannot share
one with production: IndicConformer needs Python 3.12, the AI4Bharat NeMo fork
(1.23.0rc0, for the checkpoint's multisoftmax decoder), numpy 1.26 and
scipy 1.13, while the iScribe service runs Python 3.14 with faster-whisper and
numpy 2.x. Those dependency sets are mutually exclusive, so the boundary is a
process boundary and this is the HTTP seam across it.

Binds to loopback only. It holds no credentials, serves no clinical storage, and
never writes audio to disk — the caller streams bytes in, it streams text back.

Start it with the GPU environment:

    C:/Users/akthe/scribe-gpu-venv/Scripts/python.exe -m uvicorn \\
        malayalam_sidecar.server:app --host 127.0.0.1 --port 8131

Environment:
    MALAYALAM_ASR_MODEL_PATH   local .nemo checkpoint (required offline)
    INDIC_TRANS_MODEL_PATH     local IndicTrans2 snapshot (optional)
    MALAYALAM_SIDECAR_DEVICE   cuda | cpu   (default: cuda when available)
    MALAYALAM_ASR_DECODER      ctc | rnnt   (default: ctc - measured better)
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# The engine package lives one level up; the sidecar reuses its ASR/MT modules
# rather than duplicating the NeMo loading logic.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("malayalam-sidecar")

DECODER = os.environ.get("MALAYALAM_ASR_DECODER", "ctc").lower()
MAX_UPLOAD_BYTES = int(os.environ.get("MALAYALAM_SIDECAR_MAX_MB", "100")) * 1024 * 1024


def _compute_engine_build() -> str:
    """Deployed build identity, matching service.app (stale-process detection)."""
    try:
        import subprocess
        import scribe_engine
        pkg_dir = Path(scribe_engine.__file__).parent
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=pkg_dir,
            capture_output=True, text=True, timeout=5)
        if rev.returncode == 0 and rev.stdout.strip():
            return f"engine-{rev.stdout.strip()}"
    except Exception:
        pass
    return "engine-unknown"


# Resolved ONCE at import: the health endpoint must report the build this
# PROCESS was started from, not whatever is currently on disk. Recomputing per
# request would let a stale sidecar masquerade as the deployed build and defeat
# the watchdog's stale-process restart.
ENGINE_BUILD = _compute_engine_build()

_state: dict = {"device": None, "asr_ready": False, "mt_ready": False,
                "asr_error": None, "mt_error": None, "load_seconds": None}

app = FastAPI(title="iScribe Malayalam sidecar", version="1.0.0")


def _device() -> str:
    explicit = os.environ.get("MALAYALAM_SIDECAR_DEVICE")
    if explicit:
        return explicit
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@app.on_event("startup")
def warm() -> None:
    """Load models at startup so the first clinician never waits ~70 s for them.

    Order matters on small GPUs: loading IndicTrans2 AFTER the CUDA context +
    IndicConformer exist segfaulted reproducibly on a 4 GB GTX 1650 (native
    crash inside from_pretrained, uncatchable from Python). Loading the
    CPU-resident translation model first, then the CUDA ASR model, is stable.
    Translation therefore defaults to CPU (override with MALAYALAM_MT_DEVICE);
    IndicConformer keeps the GPU for ASR latency.
    """
    device = _device()
    _state["device"] = device
    t0 = time.time()

    # Translation is optional: the Malayalam transcript is the clinical record,
    # and the English rendering is a convenience. A missing MT model must not
    # take the ASR service down.
    if os.environ.get("INDIC_TRANS_MODEL_PATH"):
        try:
            from scribe_engine.stt.experimental.indic_malayalam.translation import (
                _get_model_and_tokenizer,
            )

            mt_device = os.environ.get("MALAYALAM_MT_DEVICE", "cpu").lower()
            _get_model_and_tokenizer(mt_device)
            _state["mt_ready"] = True
            logger.info("IndicTrans2 ready on %s", mt_device)
        except Exception as exc:
            _state["mt_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("IndicTrans2 unavailable: %s", type(exc).__name__)

    try:
        from scribe_engine.stt.experimental.indic_malayalam.asr import _get_model

        _get_model(device)
        _state["asr_ready"] = True
        logger.info("IndicConformer ready on %s (decoder=%s)", device, DECODER)
    except Exception as exc:
        _state["asr_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("IndicConformer failed to load")

    _state["load_seconds"] = round(time.time() - t0, 2)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "malayalam-sidecar",
            "engine_build": ENGINE_BUILD}


@app.get("/ready")
def ready():
    payload = {
        "asr_ready": _state["asr_ready"],
        "mt_ready": _state["mt_ready"],
        "device": _state["device"],
        "decoder": DECODER,
        "load_seconds": _state["load_seconds"],
        "asr_error": _state["asr_error"],
        "mt_error": _state["mt_error"],
        # Same identity the main service exposes via /api/health: a stale
        # sidecar process is detectable instead of silently dropping the
        # verified ASR-correction table.
        "engine_build": ENGINE_BUILD,
    }
    return JSONResponse(payload, status_code=200 if _state["asr_ready"] else 503)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), translate: str = Form("false")):
    """Malayalam audio in, Malayalam transcript (plus optional English) out.

    The Malayalam transcript is always returned and is the clinical record. The
    English rendering is clearly separate and never replaces it.
    """
    if not _state["asr_ready"]:
        raise HTTPException(503, f"ASR not ready: {_state['asr_error'] or 'loading'}")

    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio exceeds the sidecar upload limit")

    # Written to a temp file because NeMo transcribes from a path, then removed
    # immediately: this service keeps no audio.
    tmp = Path(tempfile.gettempdir()) / f"ml_sidecar_{os.getpid()}_{time.time_ns()}{suffix}"
    try:
        tmp.write_bytes(data)
        from scribe_engine.stt.experimental.indic_malayalam.asr import (
            transcribe_malayalam,
        )

        t0 = time.time()
        asr = transcribe_malayalam(tmp, device=_state["device"])
        asr_seconds = round(time.time() - t0, 2)
        malayalam = (asr.get("malayalam_text") or "").strip()

        # Layer 1: verified ASR corrections, run on a COPY. The raw transcript
        # is returned untouched as the clinical record; the corrected copy and
        # the provenance of every change travel separately so a clinician can
        # audit exactly what the ASR produced and what was changed.
        correction_payload: list[dict] = []
        corrected = malayalam
        if malayalam:
            try:
                from scribe_engine.asr_correction import apply_corrections

                cr = apply_corrections(malayalam)
                corrected = cr.corrected_text
                correction_payload = cr.corrections
            except Exception as exc:  # correction must never fail transcription
                logger.warning("asr correction unavailable: %s", type(exc).__name__)
                corrected = malayalam

        english = None
        mt_seconds = None
        mt_error = None
        if translate.lower() in ("1", "true", "yes") and malayalam:
            if not _state["mt_ready"]:
                mt_error = _state["mt_error"] or "translation model not loaded"
            else:
                try:
                    from scribe_engine.stt.experimental.indic_malayalam.translation import (
                        translate_ml_to_en,
                    )

                    t0 = time.time()
                    mt = translate_ml_to_en(malayalam, device=_state["device"])
                    english = (mt.get("english_text") or "").strip()
                    mt_seconds = round(time.time() - t0, 2)
                except Exception as exc:
                    mt_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("translation failed: %s", type(exc).__name__)

        logger.info("transcribed bytes=%d asr_s=%s chars=%d translated=%s",
                    len(data), asr_seconds, len(malayalam), bool(english))
        return {
            "malayalam_text": malayalam,
            "corrected_text": corrected,
            "asr_corrections": correction_payload,
            "english_text": english,
            "translation_error": mt_error,
            "asr_model": asr.get("model"),
            "asr_revision": asr.get("revision"),
            "decoder": DECODER,
            "device": _state["device"],
            "asr_seconds": asr_seconds,
            "translation_seconds": mt_seconds,
        }
    finally:
        tmp.unlink(missing_ok=True)
