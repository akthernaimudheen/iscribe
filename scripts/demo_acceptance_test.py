"""Client-demo acceptance test against a live deployment.

Walks the exact path a clinician will: upload English consultation audio, wait
for real STT, read the transcript, check the clinical draft, edit it, export.

Asserts the things the demo would be embarrassing to fail:
  * the transcript shown is ACTUAL STT output, not the bundled demo text
  * the transcript is English, with no foreign script
  * the clinical note contains nothing that was not said
  * clinician edits reach the export
  * no filesystem paths or secrets are exposed

    python scripts/demo_acceptance_test.py --base-url https://<host>

Synthetic, non-PHI audio only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "english_consultation.wav"
TRUTH = ROOT / "tests" / "fixtures" / "english_consultation.truth.json"

# Boilerplate the old build injected. If any of it reappears in a note, the
# system is fabricating clinical content again.
FABRICATION_MARKERS = [
    "no acute distress", "acute distress",
    "basic blood test", "allergy screening",
    "mild improvement", "cooperative during examination",
    "adequate hydration", "warm fluids", "light meals",
    "5-7 days", "1-2 weeks",
    "upper respiratory condition",
    "[template]",
    "lungs: clear", "lungs clear",
]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def load_key() -> str:
    key = os.environ.get("ISCRIBE_ACCESS_TOKEN", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ISCRIBE_ACCESS_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8123")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    key = load_key()
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    print(f"\niScribe client-demo acceptance test -> {base}\n")

    if not FIXTURE.exists():
        check("English consultation fixture exists", False, str(FIXTURE))
        return summarise()

    with httpx.Client(base_url=base, timeout=180.0, follow_redirects=False) as c:
        # ---- 0. readiness -------------------------------------------------
        print("0. Deployment")
        ready = c.get("/api/ready").json()
        check("speech engine ready", ready.get("models_ready") is True,
              f"{ready.get('stt_provider')} ({ready.get('provider_selection')})")
        check("language locked to English", ready.get("language") == "en",
              str(ready.get("language")))
        check("deepgram key not exposed in API",
              "api_key" not in json.dumps(ready).lower()
              and "deepgram_api_key" not in json.dumps(ready).lower())

        # ---- 1. sign in ---------------------------------------------------
        print("\n1. Access")
        login = c.post("/api/login", data={"access_token": key})
        check("sign in", login.status_code == 303, f"HTTP {login.status_code}")
        if login.status_code != 303:
            return summarise()
        if base.startswith("http://"):
            c.headers.update({"X-Access-Token": key})

        # ---- 2. consultation + upload -------------------------------------
        print("\n2. Upload English consultation audio")
        cid = c.post("/api/consultations", json={
            "patient_id": "DEMO-ACCEPT-01", "doctor": "Dr. Demo",
        }).json()["id"]
        t0 = time.perf_counter()
        with FIXTURE.open("rb") as fh:
            up = c.post(f"/api/consultations/{cid}/audio",
                        files={"file": (FIXTURE.name, fh, "audio/wav")}, timeout=300.0)
        check("upload accepted", up.status_code == 200, f"HTTP {up.status_code}")
        check("no server path in upload response", "audio_file" not in up.json())
        print(f"       upload {round(time.perf_counter()-t0,2)}s "
              f"({FIXTURE.stat().st_size/1024:.0f} KB)")

        # ---- 3. real transcription ----------------------------------------
        print("\n3. Transcription (real STT)")
        t0 = time.perf_counter()
        started = c.post(f"/api/consultations/{cid}/process")
        check("processing started", started.status_code == 200, f"HTTP {started.status_code}")

        final, fails = None, 0
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            time.sleep(1.5)
            try:
                cur = c.get(f"/api/consultations/{cid}").json()
            except httpx.HTTPError:
                fails += 1
                if fails > 20:
                    break
                continue
            fails = 0
            if cur["status"] in ("ready", "error"):
                final = cur
                break
        proc_s = round(time.perf_counter() - t0, 2)

        if final is None or final["status"] == "error":
            check("processing completed", False,
                  (final or {}).get("error", "timed out"))
            return summarise()
        check("processing completed", True, f"{proc_s}s")

        r = final["result"]
        transcript = r["transcript"]["text"]
        meta = r["meta"]
        validation = r.get("validation", {})

        print(f"       provider={meta.get('stt_provider')} "
              f"requested={meta.get('requested_language')} "
              f"reported={meta.get('reported_language')} "
              f"validation={validation.get('severity')}")

        check("language reported as English", meta.get("reported_language") == "en",
              str(meta.get("reported_language")))
        check("language was requested explicitly (not auto-detect)",
              meta.get("requested_language") == "en")
        check("transcript validation passed", validation.get("severity") == "ok",
              f"{validation.get('severity')} {validation.get('codes')}")

        # THE critical check: is this real STT output or the bundled demo text?
        demo_text = "I'm Ramesh"
        check("transcript is REAL STT output, not the bundled demo transcript",
              demo_text.lower() not in transcript.lower() and len(transcript) > 200,
              f"{len(transcript)} chars")
        check("transcript contains no non-Latin script",
              not re.search(r"[\u0900-\u0DFF\u0600-\u06FF\u4E00-\u9FFF]", transcript))

        # Faithfulness against ground truth. Uses the same number/unit folding
        # as scripts/transcription_check.py so a provider that writes "3 days"
        # instead of "three days" is not scored as having lost the duration.
        sys.path.insert(0, str(ROOT / "scripts"))
        from transcription_check import normalise as fold

        low = transcript.lower()
        folded = " ".join(fold(transcript))

        def has(term: str) -> bool:
            return term.lower() in low or " ".join(fold(term)) in folded

        present = [t for t in truth["must_contain"] if has(t)]
        for group in truth.get("must_contain_any", []):
            if any(has(v) for v in group):
                present.append(group[0])
        total = len(truth["must_contain"]) + len(truth.get("must_contain_any", []))
        check("key clinical terms present in transcript",
              len(present) >= total - 3, f"{len(present)}/{total}")

        # ---- 4. clinical draft --------------------------------------------
        print("\n4. Clinical draft (must contain nothing that was not said)")
        note = r["clinical_note"]
        rx = r["prescription"]
        blob = (note["text"] + " " + rx["text"]).lower()
        leaked = [m for m in FABRICATION_MARKERS if m in blob]
        check("clinical note invents nothing", not leaked,
              f"fabricated: {leaked}" if leaked else "no boilerplate found")
        check("unspoken fields read 'Not mentioned'",
              note["fields"].get("physical_findings") == "Not mentioned",
              str(note["fields"].get("physical_findings")))
        check("symptoms extracted from the transcript",
              "fever" in note["fields"]["symptoms_reported"].lower()
              or "cough" in note["fields"]["symptoms_reported"].lower(),
              note["fields"]["symptoms_reported"][:60])

        # ---- 5. speaker turns ---------------------------------------------
        print("\n5. Speaker separation")
        turns = r["speakers"]["turns"]
        check("transcript is split into speaker turns", len(turns) > 3,
              f"{len(turns)} turns, method={r['speakers']['method']}")
        check("speaker-role confidence is reported honestly",
              "roles_known" in r["speakers"],
              f"roles_known={r['speakers'].get('roles_known')}")

        # ---- 6. clinician edit ---------------------------------------------
        print("\n6. Clinician review and edit")
        marker = "CLINICIAN-EDIT-VERIFIED"
        rev = c.patch(f"/api/consultations/{cid}/review",
                      json={"clinical_note_fields": {"impression": marker}})
        check("edit saved", rev.status_code == 200 and rev.json()["reviewed"] is True)

        # ---- 7. export -------------------------------------------------------
        print("\n7. Export")
        c.post(f"/api/consultations/{cid}/complete")
        exp = c.get(f"/api/consultations/{cid}/export")
        check("export downloads", exp.status_code == 200)
        check("export carries the clinician's edit", marker in exp.text)
        # Approved-only export: the raw transcript is no longer part of the
        # EHR document (the doctor already validated the note against it).
        check("export omits the raw transcript",
              transcript[:60] not in exp.text)
        check("export invents nothing",
              not [m for m in FABRICATION_MARKERS if m in exp.text.lower()])
        check("export states the language",
              "Language" in exp.text)

        # ---- 8. leakage -------------------------------------------------------
        print("\n8. No leakage")
        full = json.dumps(final)
        # Look for real absolute paths, not for the substring "audio_file",
        # which also matches the legitimate "audio_filename" (basename only).
        drive_paths = re.findall(r'"[A-Za-z]:\\\\\\\\[^"]+"', full)
        posix_paths = re.findall(r'"/(?:Users|home|var|opt)/[^"]+"', full)
        check("no filesystem paths in API response",
              not drive_paths and not posix_paths,
              f"found: {(drive_paths + posix_paths)[:2]}" if (drive_paths or posix_paths) else "")
        check("audio exposed as basename only, not a path",
              '"audio_file"' not in full
              and "\\\\" not in json.dumps(
                  (final.get("result") or {}).get("meta", {}).get("audio_filename", "")))
        check("no access key in API response", bool(key) and key not in full)
        check("no API key field in API response",
              "deepgram_api_key" not in full.lower())

    # ---- 9. logs -----------------------------------------------------------
    print("\n9. Log hygiene")
    log = ROOT / "data" / "logs" / "iscribe.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        check("no transcript content in log",
              "paracetamol" not in text.lower() and "cetirizine" not in text.lower())
        check("no patient id in log", "DEMO-ACCEPT-01" not in text)
        check("no secrets in log", (not key or key not in text)
              and "deepgram_api_key" not in text.lower())
        check("provider and language logged",
              "stt_provider=" in text and "requested_language=" in text)
    else:
        check("application log found", False, str(log))

    print("\n" + "=" * 66)
    print("ACTUAL TRANSCRIPT SHOWN TO THE CLINICIAN:")
    print("=" * 66)
    print(transcript)
    print("=" * 66)
    return summarise()


def summarise() -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{passed}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
