#!/usr/bin/env python3
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

parser = argparse.ArgumentParser(description="Build the pinned Spanish conjugation lookup.")
parser.add_argument("source", type=Path)
args = parser.parse_args()
expected = "7a165d4b44eb078d23964f3eee3813f5140c5df0ffab0143c806c562e67f8771"
if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected:
    parser.error("Source checksum differs from the pinned Spanish morphology data")
verbs = defaultdict(set)
with args.source.open(encoding="utf-8") as stream:
    for line in stream:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        lemma, form, features = parts
        tags = set(features.split(";"))
        if "V" in tags and "PRO" not in tags and lemma.isalpha() and form.isalpha():
            verbs[unicodedata.normalize("NFC", lemma).casefold()].add(unicodedata.normalize("NFC", form).casefold())
extra = Path(__file__).resolve().parent.parent / "data" / "spanish-verbs-extra.json"
for lemma, forms in json.loads(extra.read_text(encoding="utf-8")).items():
    verbs[lemma].update(forms)
payload = {lemma: sorted(forms) for lemma, forms in sorted(verbs.items())}
encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
destination = Path(__file__).resolve().parent.parent / "data" / "spanish-verbs.json.gz"
destination.write_bytes(gzip.compress(encoded, mtime=0))
print(f"Wrote {len(payload)} Spanish verb paradigms")
