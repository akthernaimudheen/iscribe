# -*- coding: utf-8 -*-
"""Dump the raw transcript + corrections from the last async verification run."""
import io
import json
import os
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _base() -> str:
    """Live base URL: env override, else the watchdog's CURRENT_TUNNEL_URL.txt."""
    env = os.environ.get("ISCRIBE_BASE_URL", "").strip()
    if env:
        return env.rstrip("/")
    from pathlib import Path
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
ENV = r"C:\Users\akthe\OneDrive\Documents\New folder\.env"

tok = [l.split("=", 1)[1].strip() for l in open(ENV, encoding="utf-8")
       if l.strip().startswith("ISCRIBE_ACCESS_TOKEN=")][0]

# List consultations, take the latest ASYNC-VERIFY-1
req = urllib.request.Request(BASE + "/api/consultations",
                             headers={"X-Access-Token": tok})
with urllib.request.urlopen(req, timeout=60) as r:
    lst = json.loads(r.read().decode())
cid = [c["id"] for c in lst if c.get("patient_id") == "ASYNC-VERIFY-1"][-1]

req = urllib.request.Request(
    BASE + f"/api/consultations/{cid}", headers={"X-Access-Token": tok})
with urllib.request.urlopen(req, timeout=60) as r:
    rec = json.loads(r.read().decode())

print("RAW TRANSCRIPT (verbatim):")
print(rec["result"]["transcript"]["text"])
print()
print("CORRECTIONS APPLIED:", json.dumps(
    rec["result"]["transcript"].get("asr_corrections_applied"), ensure_ascii=False))
