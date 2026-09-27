"""Malayalam ASR — AI4Bharat IndicConformer (NeMo checkpoint).

Model: ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large (MIT license,
gated="auto" on HuggingFace — needs a free HF account, model terms acceptance,
and HF_TOKEN for download). The checkpoint is a NeMo `.nemo` archive.

This module lazily imports nemo_toolkit so the rest of the codebase never
depends on it. Runs on CPU by default; CUDA if available and device="cuda".
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

MODEL_ID = "ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large"

_model_cache: dict = {}   # device -> dict(model, load_seconds)
MODEL_REVISION = None     # HF commit sha, recorded when online metadata is available


def _record_revision():
    global MODEL_REVISION
    if MODEL_REVISION is not None:
        return
    try:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            MODEL_REVISION = "unavailable (offline mode)"
            return
        from huggingface_hub import HfApi
        MODEL_REVISION = HfApi().model_info(MODEL_ID).sha
    except Exception:
        MODEL_REVISION = "unavailable"


def _resolve_nemo_path() -> str:
    """Resolve a local .nemo file path.

    NeMo 3.x from_pretrained() expects an unpacked HF layout (model_config.yaml)
    which the AI4Bharat repos do not provide — they ship a .nemo archive. So we
    resolve the archive ourselves and call restore_from(), which works both for
    the HF cache (after one-time download) and for a user-supplied path.
    """
    env_path = os.environ.get("MALAYALAM_ASR_MODEL_PATH")
    if env_path:
        if not os.path.exists(env_path):
            raise RuntimeError(f"MALAYALAM_ASR_MODEL_PATH does not exist: {env_path}")
        return env_path
    if os.environ.get("SCRIBE_OFFLINE_MODE") == "1":
        raise RuntimeError(
            "SCRIBE_OFFLINE_MODE=1 requires MALAYALAM_ASR_MODEL_PATH pointing at "
            "the local indicconformer_stt_ml_hybrid_rnnt_large.nemo file."
        )
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")  # None is fine once terms accepted + cached
    path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="indicconformer_stt_ml_hybrid_rnnt_large.nemo",
        token=token,
    )
    return path


def _ensure_nemo_hub_compat():
    """Compat shims for the AI4Bharat NeMo fork (nemo-v2, 2024 era).

    1) nemo/core/classes/mixins/hf_io_mixin.py imports huggingface_hub.ModelFilter,
       removed from huggingface_hub in 0.26. Not used on our local restore path.
    2) nemo/utils/exp_manager.py imports pytorch_lightning.loggers.NeptuneLogger,
       removed from pytorch-lightning 2.4+. Not instantiated on our inference path.
    """
    import huggingface_hub
    if not hasattr(huggingface_hub, "ModelFilter"):
        huggingface_hub.ModelFilter = object  # noqa: shadowing removed symbol

    import pytorch_lightning.loggers as _pl_loggers
    if not hasattr(_pl_loggers, "NeptuneLogger"):
        class _NeptuneLoggerStub:  # pragma: no cover - never instantiated here
            """Stub for a removed PL logger; NeMo only imports the name."""

        _pl_loggers.NeptuneLogger = _NeptuneLoggerStub


def _get_model(device: str = "cpu"):
    if device in _model_cache:
        return _model_cache[device]
    _ensure_nemo_hub_compat()
    try:
        import nemo.collections.asr as nemo_asr  # heavy, optional dep
    except ImportError as e:
        raise RuntimeError(
            "nemo_toolkit (AI4Bharat fork, nemo-v2) is not installed in this "
            "environment. See scribe_engine/stt/experimental/indic_malayalam/README.md"
        ) from e
    nemo_path = _resolve_nemo_path()
    t0 = time.time()
    # The IndicConformer checkpoints are EncDecHybridRNNTCTCModel archives;
    # restoring via the abstract ASRModel class fails with abstract-method errors.
    model = nemo_asr.models.EncDecHybridRNNTCTCModel.restore_from(
        nemo_path, map_location=device
    )
    load_s = time.time() - t0
    if not os.environ.get("SCRIBE_OFFLINE_MODE"):
        _record_revision()
    model.eval()
    if device == "cpu":
        model = model.to("cpu")
    logger.info("IndicConformer loaded from %s in %.1fs", nemo_path, load_s)
    _model_cache[device] = {"model": model, "load_seconds": round(load_s, 2)}
    return _model_cache[device]


def _to_mono_16k_wav(audio_path) -> str:
    """Normalize any decodable input to 16 kHz mono WAV; return a temp path.

    NeMo's AudioToBPEDataset expects (batch, time) — mono. Real consultation
    audio arrives as stereo m4a/webm/mp3 from browsers and phones, which made
    inference crash with an output-shape mismatch (stereo [1, T, 2]). Every
    input is therefore decoded through ffmpeg before inference. The caller
    deletes the temp file.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to decode consultation audio but was not "
            "found on PATH"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
         "-ac", "1", "-ar", "16000", tmp.name],
        check=True,
    )
    return tmp.name


def transcribe_malayalam(audio_path, device: str = "cpu") -> dict:
    """Return malayalam text + model identity + load/infer timings."""
    entry = _get_model(device)
    model = entry["model"]
    wav_path = _to_mono_16k_wav(audio_path)
    try:
        t0 = time.time()
        # Multilingual hybrid model: per model card, CTC decoding with an explicit
        # language_id is the recommended inference mode.
        model.cur_decoder = "ctc"
        out = model.transcribe([wav_path], batch_size=1, logprobs=False, language_id="ml")
    finally:
        import os as _os
        _os.unlink(wav_path)
    elapsed = time.time() - t0
    elapsed = time.time() - t0
    texts = []
    for item in out:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, (list, tuple)):
            texts.extend(x for x in item if isinstance(x, str))
        else:
            texts.append(getattr(item, "text", str(item)))
    # The fork returns the text once per decoder; dedupe identical repeats.
    seen, unique = set(), []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return {
        "malayalam_text": " ".join(unique).strip(),
        "model": MODEL_ID,
        "revision": MODEL_REVISION or "unavailable",
        "device": device,
        "asr_time_seconds": round(elapsed, 2),
        "model_load_seconds": entry["load_seconds"] if len(_model_cache) == 1 and elapsed > 0 and entry["load_seconds"] > 0 else 0.0,
    }
