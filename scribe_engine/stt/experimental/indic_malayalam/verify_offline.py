"""Prove the complete Malayalam pipeline runs with no token and no network.

Runs: Malayalam audio -> IndicConformer -> Malayalam transcript -> IndicTrans2
-> English, with every credential stripped from the process and outbound
network access actually blocked at the socket layer.

This is deliberately stronger than setting HF_HUB_OFFLINE=1. Offline env vars
are honoured only by libraries that choose to check them; a hard socket block
proves nothing reached the network, whoever tried. Loopback stays open so the
check does not interfere with anything running locally.

Usage (from the isolated experiment interpreter):
    python -m scribe_engine.stt.experimental.indic_malayalam.verify_offline \\
        --asr-path <path to .nemo> --mt-path <snapshot dir> [--device cpu|cuda]

Exit code 0 = the full pipeline ran offline and tokenless; 1 = it did not.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

CREDENTIAL_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
)

_blocked_attempts: list[str] = []


def strip_credentials() -> list[str]:
    """Remove every HF credential from this process. Values are never printed."""
    removed = []
    for name in CREDENTIAL_VARS:
        if os.environ.pop(name, None) is not None:
            removed.append(name)
    # huggingface_hub also reads a token file; point HF at a dir that has none.
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return removed


def enable_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["SCRIBE_OFFLINE_MODE"] = "1"


def block_network() -> None:
    """Fail any outbound connection to a non-loopback address."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _is_loopback(address) -> bool:
        try:
            host = address[0] if isinstance(address, tuple) else str(address)
        except Exception:
            return False
        return host in ("127.0.0.1", "::1", "localhost")

    def guarded_connect(self, address, *a, **kw):
        if not _is_loopback(address):
            _blocked_attempts.append(str(address))
            raise OSError("NETWORK BLOCKED by verify_offline: outbound connection refused")
        return real_connect(self, address, *a, **kw)

    def guarded_connect_ex(self, address, *a, **kw):
        if not _is_loopback(address):
            _blocked_attempts.append(str(address))
            return 1
        return real_connect_ex(self, address, *a, **kw)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asr-path", required=True, help="Local .nemo checkpoint")
    ap.add_argument("--mt-path", required=True, help="Local IndicTrans2 snapshot dir")
    ap.add_argument("--audio", default=None, help="Fixture wav (defaults to ml_fever_cough)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None, help="Write the result as JSON here")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    audio = args.audio or os.path.join(here, "test_audio", "ml_fever_cough.wav")

    print("=" * 66)
    print("Offline + tokenless full-pipeline verification")
    print("=" * 66)

    # ---- 1. local artefacts exist -------------------------------------
    print("\n[1] Local model artefacts")
    asr_ok = os.path.isfile(args.asr_path)
    mt_ok = os.path.isdir(args.mt_path)
    print(f"  ASR checkpoint present   : {asr_ok}")
    if asr_ok:
        print(f"  ASR size (bytes)         : {os.path.getsize(args.asr_path)}")
    print(f"  MT snapshot dir present  : {mt_ok}")
    if mt_ok:
        names = sorted(os.listdir(args.mt_path))
        print(f"  MT files                 : {len(names)}")
        for n in names:
            print(f"    - {n}")
    print(f"  audio fixture present    : {os.path.isfile(audio)}")
    if not (asr_ok and mt_ok and os.path.isfile(audio)):
        print("\nFAIL: required local artefacts are missing.")
        return 1

    # ---- 2. strip credentials -----------------------------------------
    print("\n[2] Credentials")
    removed = strip_credentials()
    print(f"  removed from process     : {removed or '(none were set)'}")
    for name in CREDENTIAL_VARS:
        assert name not in os.environ, name
    print("  credentials in process   : NONE")

    # ---- 3/4. offline mode + hard network block ------------------------
    print("\n[3] Offline mode")
    enable_offline_env()
    print("  HF_HUB_OFFLINE           : 1")
    print("  TRANSFORMERS_OFFLINE     : 1")
    print("  SCRIBE_OFFLINE_MODE      : 1")

    print("\n[4] Network")
    block_network()
    print("  outbound sockets         : BLOCKED (loopback still permitted)")

    os.environ["MALAYALAM_ASR_MODEL_PATH"] = args.asr_path
    os.environ["INDIC_TRANS_MODEL_PATH"] = args.mt_path

    # ---- 5. run the complete pipeline ----------------------------------
    print("\n[5] Running Malayalam audio -> ASR -> MT -> English")
    sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..", "..", "..")))
    result: dict = {"device": args.device, "audio": os.path.basename(audio)}
    try:
        from scribe_engine.stt.experimental.indic_malayalam.provider import (
            MalayalamIndicProvider,
        )

        t0 = time.time()
        tr = MalayalamIndicProvider(device=args.device).transcribe(audio)
        total = round(time.time() - t0, 2)

        ml = (tr.source_transcript or "").strip()
        en = (tr.translated_transcript or "").strip()
        result.update({
            "malayalam_transcript": ml,
            "english_translation": en,
            "total_seconds": total,
            "asr_seconds": tr.provider_meta.get("asr_time_seconds"),
            "translation_seconds": tr.provider_meta.get("translation_time_seconds"),
            "asr_model": tr.provider_meta.get("asr_model"),
            "mt_model": tr.provider_meta.get("mt_model"),
        })
        print(f"  total_seconds            : {total}")
        print(f"  asr_seconds              : {result['asr_seconds']}")
        print(f"  translation_seconds      : {result['translation_seconds']}")
        print(f"\n  Malayalam : {ml[:90]}")
        print(f"  English   : {en[:90]}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        print(f"\n  blocked network attempts : {len(_blocked_attempts)}")
        for a in _blocked_attempts[:10]:
            print(f"    -> {a}")
        result["blocked_network_attempts"] = _blocked_attempts
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
        print("\nFAIL: the pipeline could not complete offline.")
        return 1

    # ---- 6. verdict -----------------------------------------------------
    print("\n[6] Verdict")
    print(f"  blocked network attempts : {len(_blocked_attempts)}")
    for a in _blocked_attempts[:10]:
        print(f"    -> {a}")
    result["blocked_network_attempts"] = _blocked_attempts
    ok = bool(result.get("malayalam_transcript")) and bool(result.get("english_translation"))
    result["offline_tokenless_pass"] = ok
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"  wrote                    : {args.out}")
    print("\n" + ("PASS: complete pipeline ran with no token and no network."
                  if ok else "FAIL: pipeline produced empty output."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
