# -*- coding: utf-8 -*-
"""Async-path real-audio verification against the LIVE production service.

Submits the real Malayalam consultation through the deployed HTTPS endpoint,
asserts the doctor-facing request returns well under the processing time,
polls the job to completion, and verifies the six concepts + validated HPI
through the asynchronous pipeline.
"""
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _base() -> str:
    """Live base URL: ISCRIBE_BASE_URL env var, else the watchdog's record.

    Quick-tunnel URLs change on every tunnel restart; the iScribe watchdog
    keeps the current one in CURRENT_TUNNEL_URL.txt.
    """
    env = os.environ.get("ISCRIBE_BASE_URL", "").strip()
    if env:
        return env.rstrip("/")
    url_file = Path(r"C:\Users\akthe\.iscribe-logs\CURRENT_TUNNEL_URL.txt")
    try:
        url = url_file.read_text(encoding="utf-8").strip()
        if url:
            return url.rstrip("/")
    except OSError:
        pass
    raise SystemExit(
        "No tunnel URL: set ISCRIBE_BASE_URL or start the watchdog tunnel.")


BASE = _base()
TOKEN_PATH = Path(r"C:\Users\akthe\OneDrive\Documents\New folder\.env")
AUDIO = Path(r"C:\Users\akthe\OneDrive\Documents\New folder\experiments"
             r"\malayalam_speech_translation\previous_consultation.m4a")


def token() -> str:
    for line in TOKEN_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ISCRIBE_ACCESS_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("token not found in .env")


def api(path: str, data=None, headers=None, files=None, method="GET"):
    hdrs = dict(headers or {})
    body = None
    if files:
        boundary = "----bnd7f3a9c2"
        name, filename, content, ctype = files
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        hdrs["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif data is not None:
        body = json.dumps(data).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=hdrs)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, json.loads(resp.read().decode())


def main() -> int:
    tok = token()
    hdr = {"X-Access-Token": tok}

    # 1. Create consultation (Malayalam)
    _, cons = api("/api/consultations",
                  {"patient_id": "ASYNC-VERIFY-1", "language": "ml"},
                  headers=hdr, method="POST")
    cid = cons["id"]

    # 2. Upload the real audio
    status, up = api(f"/api/consultations/{cid}/audio",
                     files=("file", AUDIO.name, AUDIO.read_bytes(), "audio/mp4"),
                     headers=hdr, method="POST")
    assert status == 200, up

    # 3. Submit processing — measure doctor-facing latency
    t0 = time.perf_counter()
    status, resp = api(f"/api/consultations/{cid}/process", {}, headers=hdr,
                       method="POST")
    submit_s = time.perf_counter() - t0
    assert status == 202, (status, resp)
    job_id = resp["job_id"]
    print(f"doctor-facing submission: {submit_s:.2f}s (job {job_id}, "
          f"status={resp['job_status']})")

    # 4. Poll to completion (CPU ASR takes ~1-2 min)
    deadline = time.time() + 420
    rec = None
    while time.time() < deadline:
        _, rec = api(f"/api/consultations/{cid}", headers=hdr)
        if rec["status"] in ("ready", "error"):
            break
        time.sleep(5)
    assert rec and rec["status"] == "ready", rec.get("error")

    result = rec["result"]
    v2 = result["clinical_note_v2"]
    entities = result["normalized_clinical_entities"]["entities"]
    concepts = {e["concept"]: e["status"] for e in entities}

    print("fact-graph note validator:", "PASS" if v2["validation"]["valid"] else "FAIL")
    print("concepts:", json.dumps(concepts, ensure_ascii=False))
    expected = {"fever": "RESOLVED", "fatigue": "PRESENT", "headache": "PRESENT",
                "cough": "PRESENT", "phlegm": "PRESENT",
                "difficulty_eating": "PRESENT"}
    ok = all(concepts.get(c) == s for c, s in expected.items())
    print("six-concept check:", "MATCH" if ok else "MISMATCH")
    print()
    print("=== HPI (async path) ===")
    print(v2["note"].split("2. History of Present Illness")[1].split("3.")[0].strip())

    ok = ok and v2["validation"]["valid"] and submit_s < 2.0
    print()
    print("OVERALL:", "SUCCESS" if ok else "REVIEW NEEDED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
