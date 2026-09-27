"""IndicTrans2 Malayalam → English translation.

Model: ai4bharat/indictrans2-indic-en-dist-200M (MIT, gated="auto" on HF).
Uses IndicTransToolkit's IndicProcessor for preprocessing and HF transformers
for inference.

Offline mode: set SCRIBE_OFFLINE_MODE=1 and INDIC_TRANS_MODEL_PATH to a local
directory containing the snapshot (config.json, pytorch_model.bin / safetensors,
dict.SRC.json, dict.TGT.json, tokenizer files). In offline mode nothing contacts
HuggingFace and no HF_TOKEN is required.
"""

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Vendored pure-Python IndicProcessor (v1.0.2-era) from VarunGumma/IndicTransToolkit
# (MIT), commit a95da3a008. The current toolkit release ships only a Cython extension
# with no Windows wheels and requires MSVC to build, which this machine lacks.
# See _vendor/PROVENANCE.md.
_VENDOR_DIR = Path(__file__).parent / "_vendor"


def _load_vendored_processor():
    spec = importlib.util.spec_from_file_location(
        "indic_processor_vendored", _VENDOR_DIR / "processor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["indic_processor_vendored"] = module
    spec.loader.exec_module(module)
    return module.IndicProcessor

MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
SRC_LANG = "mal_Mlym"
TGT_LANG = "eng_Latn"

_cache: dict = {}
MODEL_REVISION = None


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


def _get_model_and_tokenizer(device: str = "cpu"):
    if device in _cache:
        return _cache[device]
    try:
        IndicProcessor = _load_vendored_processor()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "transformers/sentencepiece/sacremoses/indicnlp not installed in this "
            "environment. See scribe_engine/stt/experimental/indic_malayalam/README.md"
        ) from e

    offline = os.environ.get("SCRIBE_OFFLINE_MODE") == "1"
    if offline:
        path = os.environ.get("INDIC_TRANS_MODEL_PATH")
        if not path:
            raise RuntimeError(
                "SCRIBE_OFFLINE_MODE=1 requires INDIC_TRANS_MODEL_PATH pointing at "
                "the local indictrans2-indic-en-dist-200M snapshot directory."
            )
        if not os.path.isdir(path):
            raise RuntimeError(f"Local translation model directory not found: {path}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True, torch_dtype="auto"
        )
        load_s = time.time() - t0
        revision = f"local:{path}"
    else:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError(
                "HF_TOKEN is not set. IndicTrans2 distilled models are gated on "
                "HuggingFace (or use SCRIBE_OFFLINE_MODE=1 with INDIC_TRANS_MODEL_PATH)."
            )
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype="auto"
        )
        load_s = time.time() - t0
        _record_revision()
        revision = MODEL_REVISION or "unavailable"

    model.eval().to(device)
    logger.info("IndicTrans2 loaded in %.1fs", load_s)
    _cache[device] = {
        "tokenizer": tokenizer,
        "model": model,
        "ip": IndicProcessor(inference=True),
        "load_seconds": round(load_s, 2),
        "revision": revision,
    }
    return _cache[device]


def translate_ml_to_en(text: str, device: str = "cpu") -> dict:
    """Translate Malayalam text to English. Returns translation + model identity + timings."""
    import torch

    entry = _get_model_and_tokenizer(device)
    tokenizer, model, ip = entry["tokenizer"], entry["model"], entry["ip"]

    t0 = time.time()
    batch = ip.preprocess_batch([text], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    inputs = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        # use_cache=False is required, not an optimisation choice. transformers
        # >= ~4.44 hands generate() an EncoderDecoderCache object where this
        # model's 2023-era custom modeling_indictrans.py expects None or a
        # legacy tuple; it then does past_key_values[0][0].shape and raises
        # AttributeError: 'NoneType' object has no attribute 'shape'.
        # Disabling the cache avoids that code path. Revisit if the upstream
        # repo is updated for the current transformers cache API.
        generated = model.generate(
            **inputs, num_beams=4, max_new_tokens=256, use_cache=False
        )
    decoded = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # IndicProcessor masks numbers and named entities during preprocessing and
    # restores them here. Skipping this step leaves placeholder tokens in the
    # output, which for clinical text means dosages and durations can come back
    # as placeholders instead of values.
    english = ip.postprocess_batch(decoded, lang=TGT_LANG)
    elapsed = time.time() - t0
    return {
        "english_text": english[0],
        "model": MODEL_ID,
        "revision": entry["revision"],
        "device": device,
        "translation_time_seconds": round(elapsed, 2),
        "model_load_seconds": entry["load_seconds"],
    }
