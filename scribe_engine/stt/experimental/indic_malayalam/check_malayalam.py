"""Environment check for the Malayalam pipeline.

Usage (run with the isolated experiment interpreter, not the service venv;
its location is machine-specific — see docs/MALAYALAM_PIPELINE.md):
    python -m scribe_engine.stt.experimental.indic_malayalam.check_malayalam

Verifies: python/packages, local model files (env-var path or HF cache),
HF-authentication requirement, and (optionally) live ASR/translation inference.
Never prints credentials. Exit code 0 = READY, 1 = NOT READY.
"""

import importlib.metadata as im
import os
import sys


def _ok(flag):
    return "FOUND" if flag else "MISSING"


def _cached_nemo_path():
    """Locate the .nemo file: env var first, then the HF cache."""
    env = os.environ.get("MALAYALAM_ASR_MODEL_PATH")
    if env and os.path.exists(env):
        return env, "MALAYALAM_ASR_MODEL_PATH"
    home = os.path.expanduser("~")
    cache = os.path.join(home, ".cache", "huggingface", "hub",
                         "models--ai4bharat--indicconformer_stt_ml_hybrid_ctc_rnnt_large")
    if os.path.isdir(cache):
        snaps = os.path.join(cache, "snapshots")
        for d in os.listdir(snaps):
            for f in os.listdir(os.path.join(snaps, d)):
                if f.endswith(".nemo"):
                    return os.path.join(snaps, d, f), "hf cache"
    return None, None


def _cached_mt_dir():
    env = os.environ.get("INDIC_TRANS_MODEL_PATH")
    if env and os.path.isdir(env):
        return env, "INDIC_TRANS_MODEL_PATH"
    home = os.path.expanduser("~")
    cache = os.path.join(home, ".cache", "huggingface", "hub",
                         "models--ai4bharat--indictrans2-indic-en-dist-200M")
    if os.path.isdir(cache):
        snaps = os.path.join(cache, "snapshots")
        if os.path.isdir(snaps):
            for d in os.listdir(snaps):
                if os.path.exists(os.path.join(snaps, d, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(snaps, d, "model.safetensors")):
                    return os.path.join(snaps, d), "hf cache"
    return None, None


def main() -> int:
    offline = os.environ.get("SCRIBE_OFFLINE_MODE") == "1"
    print("Malayalam Scribe Environment")
    print("-" * 40)

    print(f"Python: {sys.version.split()[0]} (experiment env: {sys.prefix})")

    packages = ["torch", "nemo_toolkit", "transformers", "sentencepiece",
                "sacremoses", "indic_nlp_library_itt", "faster_whisper", "psutil"]
    for p in packages:
        try:
            print(f"  {p}: {im.version(p)}")
        except Exception:
            print(f"  {p}: MISSING")

    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except Exception as e:
        print(f"CUDA check failed: {e}")

    nemo_path, src = _cached_nemo_path()
    print(f"ASR model: {_ok(bool(nemo_path))}" + (f" ({src}: {nemo_path})" if nemo_path else ""))

    mt_dir, mt_src = _cached_mt_dir()
    print(f"Translation model: {_ok(bool(mt_dir))}" + (f" ({mt_src})" if mt_dir else ""))

    token_set = bool(os.environ.get("HF_TOKEN"))
    need = (not nemo_path and not offline) or (not mt_dir and not offline)
    if offline:
        print("HF authentication: NOT REQUIRED (offline mode)")
    elif need:
        print(f"HF authentication: REQUIRED to download missing models (token set: {token_set})")
    else:
        print(f"HF authentication: NOT REQUIRED (models cached; token set: {token_set})")

    overall = True
    if "--live" in sys.argv and nemo_path:
        try:
            import tempfile, subprocess, pathlib
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
            from scribe_engine.stt.experimental.indic_malayalam.asr import transcribe_malayalam
            with tempfile.TemporaryDirectory() as td:
                wav = pathlib.Path(td, "t.wav")
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                                "-i", "sine=frequency=440:duration=1", str(wav)], check=True)
                r = transcribe_malayalam(str(wav), device="cpu")
            print(f"Local ASR inference: {'PASS' if r['malayalam_text'] is not None else 'FAIL'}")
        except Exception as e:
            overall = False
            print(f"Local ASR inference: FAIL ({type(e).__name__}: {str(e)[:120]})")

    if "--live" in sys.argv and mt_dir:
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
            from scribe_engine.stt.experimental.indic_malayalam.translation import translate_ml_to_en
            r = translate_ml_to_en("എനിക്ക് പനിയുണ്ട്", device="cpu")
            good = bool(r.get("english_text", "").strip())
            print(f"Local translation inference: {'PASS' if good else 'FAIL'}")
        except Exception as e:
            overall = False
            print(f"Local translation inference: FAIL ({type(e).__name__}: {str(e)[:120]})")

    ready = (nemo_path and mt_dir) if not offline else bool(nemo_path)
    print(f"Overall: {'READY' if ready and overall else 'NOT READY'}")
    return 0 if ready and overall else 1


if __name__ == "__main__":
    sys.exit(main())
