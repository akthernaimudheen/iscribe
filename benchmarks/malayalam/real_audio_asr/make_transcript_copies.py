# -*- coding: utf-8 -*-
"""Copy raw transcripts verbatim into .txt files (review convenience only)."""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

r = json.load(open(os.path.join(HERE, "indicconformer_ctc.json"), encoding="utf-8"))
with open(os.path.join(HERE, "transcript_indicconformer_ctc.txt"), "w",
          encoding="utf-8") as fh:
    fh.write(r["raw_transcript"] + "\n")

w = json.load(open(os.path.join(HERE, "faster_whisper.json"), encoding="utf-8"))
for entry in w["results"]:
    with open(os.path.join(HERE, f"transcript_whisper_{entry['config']}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(entry.get("raw_transcript") or f"[{entry['status']}: "
                 f"{entry.get('error', '')}]" + "\n")
print("transcript copies written")
