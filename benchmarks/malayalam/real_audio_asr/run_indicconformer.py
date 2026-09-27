# -*- coding: utf-8 -*-
"""Real-audio ASR benchmark: IndicConformer CTC on the real Malayalam consultation.

Same restore path as the sidecar engine (local .nemo, offline, no HF token).
Saves the RAW transcript verbatim - no correction, no normalization.
"""
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO_ROOT)
AUDIO = os.path.join(HERE, "real_consultation_16k.wav")
OUT_JSON = os.path.join(HERE, "indicconformer_ctc.json")

# Compat shims (documented in scribe_engine/stt/experimental/indic_malayalam/asr.py):
# the 2024 NeMo fork predates huggingface_hub>=0.26 and pytorch-lightning 2.6.
import huggingface_hub  # noqa: E402

if not hasattr(huggingface_hub, "ModelFilter"):
    huggingface_hub.ModelFilter = object
try:
    import pytorch_lightning as _pl  # noqa: E402

    if not hasattr(_pl.loggers, "NeptuneLogger"):
        _pl.loggers.NeptuneLogger = object
except Exception:
    pass
import numpy as np  # noqa: E402

if not hasattr(np, "sctypes"):
    _info = np.iinfo
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "bool": [bool],
    }

from scribe_engine.stt.experimental.indic_malayalam.asr import (  # noqa: E402
    MODEL_ID,
    transcribe_malayalam,
)

t0 = time.time()
result = transcribe_malayalam(AUDIO)
elapsed = time.time() - t0

record = {
    "audio": os.path.basename(AUDIO),
    "provider": "indicconformer_ctc",
    "model": result["model"],
    "model_revision": result["revision"],
    "device": result["device"],
    "asr_time_seconds": result["asr_time_seconds"],
    "model_load_seconds": result["model_load_seconds"],
    "total_elapsed_s": round(elapsed, 3),
    "language": "ml",
    "raw_transcript": result["malayalam_text"],
    "note": "raw IndicConformer CTC output, verbatim, no correction/normalization",
}
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(record, fh, ensure_ascii=False, indent=2)
print(json.dumps(record, ensure_ascii=False, indent=2))
