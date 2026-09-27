# -*- coding: utf-8 -*-
"""Full live-pipeline verification against the real Malayalam consultation.

Drives the production HTTP API exactly as the UI would:
  login -> create (language=ml) -> upload audio -> process -> poll -> report.

The access key is read from .env and used only in memory (cookie jar); it is
never printed, logged, or written to any artifact. Audio goes to the local
service only. Prints: raw transcript, speaker info, entities with
status/confidence/provenance, and the unresolved-form classification.
"""
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
BASE = "http://127.0.0.1:8123"
AUDIO = HERE.parent / "malayalam_speech_translation" / "previous_consultation.m4a"


def token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*ISCRIBE_ACCESS_TOKEN\s*=\s*(.+)", line)
        if m:
            return m.group(1).strip().strip('"')
    raise SystemExit("ISCRIBE_ACCESS_TOKEN not found in .env")


def post(opener: urllib.request.OpenerDirector, url: str, data=None,
         headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers=headers or {})
    try:
        with opener.open(req, timeout=600) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> int:
    opener = urllib.request.build_opener(_NoRedirect)  # capture the 303 itself

    form = (f"access_token={token()}").encode()
    req = urllib.request.Request(
        f"{BASE}/api/login", data=form, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with opener.open(req, timeout=60) as resp:
            code, set_cookie = resp.status, resp.headers.get("Set-Cookie", "")
    except urllib.error.HTTPError as e:
        code, set_cookie = e.code, e.headers.get("Set-Cookie", "")
    print(f"login: {code}")
    assert code in (200, 303) and set_cookie, "login failed (code printed only)"
    # Production sets the session cookie Secure (https tunnels); loopback http
    # needs it attached manually. Cookie VALUE never printed.
    session_cookie = set_cookie.split(";", 1)[0]
    auth_headers = {"Cookie": session_cookie}

    code, body = post(opener, f"{BASE}/api/consultations",
                      json.dumps({"patient_id": "ML-FIX-VERIFY",
                                  "doctor": "Dr. Demo",
                                  "language": "ml"}).encode(),
                      {"Content-Type": "application/json", **auth_headers})
    cid = json.loads(body)["id"]
    print(f"consultation {cid}: created ({code})")

    boundary = "verifyboundary8131"
    audio = AUDIO.read_bytes()
    mp = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; "
        f"filename=\"{AUDIO.name}\"\r\nContent-Type: audio/mp4\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    code, body = post(opener, f"{BASE}/api/consultations/{cid}/audio", mp,
                      {"Content-Type": f"multipart/form-data; boundary={boundary}",
                       **auth_headers})
    print(f"upload: {code}")

    t0 = time.time()
    code, body = post(opener, f"{BASE}/api/consultations/{cid}/process",
                      headers=auth_headers)
    print(f"process: {code}")

    record = {}
    for _ in range(120):
        time.sleep(2)
        req = urllib.request.Request(f"{BASE}/api/consultations/{cid}",
                                     headers=auth_headers)
        with opener.open(req, timeout=30) as r:
            record = json.loads(r.read().decode("utf-8"))
        if record.get("status") in ("ready", "failed", "error"):
            break
    status = record.get("status")
    print(f"final status: {status} after {time.time() - t0:.1f}s")
    if status != "ready":
        print(json.dumps({"error": record.get("error"),
                          "stages": record.get("stages")}, indent=2))
        return 1

    result = record.get("result") or {}
    transcript = result.get("transcript") or {}
    speakers = result.get("speakers") or {}
    normalized = result.get("normalized_clinical_entities") or {}

    out = {
        "consultation": cid,
        "status": status,
        "meta": {k: transcript.get(k) for k in
                 ("stt_provider", "language", "requested_language")},
        "raw_asr_transcript": transcript.get("text") or "",
        "asr_corrections_applied": transcript.get("asr_corrections_applied"),
        "speakers": {
            "method": speakers.get("method"),
            "roles_known": speakers.get("roles_known"),
            "confidence": speakers.get("confidence"),
            "n_turns": len(speakers.get("turns") or []),
            "first_turns": [
                {k: t.get(k) for k in ("speaker", "text")}
                for t in (speakers.get("turns") or [])[:4]
            ],
        },
        "entities": [
            {"concept": e.get("concept"), "english": e.get("english"),
             "status": e.get("status"), "temporality": e.get("temporality"),
             "confidence": e.get("confidence"),
             "surface_text": e.get("surface_text"),
             "source_clause": (e.get("source_clause") or "")[:120]}
            for e in normalized.get("entities", [])
        ],
        "present": normalized.get("present"),
        "absent": normalized.get("absent"),
        "resolved": normalized.get("resolved"),
        "questioned": normalized.get("questioned"),
        "normalized_summary": normalized.get("normalized_summary"),
        "clinical_note_symptoms": (result.get("clinical_note") or {}).get(
            "fields", {}).get("symptoms_reported"),
        "clinical_note_denies": (result.get("clinical_note") or {}).get(
            "fields", {}).get("denies"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
