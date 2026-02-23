#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove "empty" rows from a places JSONL file.

Keeps lines where:
  - places is a non-empty list

Drops lines where:
  - places is missing, not a list, or an empty list []

Also:
  - Writes dropped JSON objects to <out>.dropped.jsonl
  - Writes bad/invalid JSON lines to <out>.bad.jsonl
  - Optionally replaces the input file in-place (with a .bak backup)

JSON parsing uses Python's built-in json module. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_empty_places(obj: dict, keep_errors: bool) -> bool:
    # If you want to keep rows that had an error during extraction, set --keep_errors
    if keep_errors and isinstance(obj.get("error"), str) and obj.get("error").strip():
        return False

    places = obj.get("places", None)
    return (not isinstance(places, list)) or (len(places) == 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL file")
    ap.add_argument("--out", dest="out_path", default=None, help="Output JSONL file (default: <in>.nonempty.jsonl)")
    ap.add_argument("--keep_errors", action="store_true", default=False,
                    help="Keep rows that contain an 'error' field even if places is empty")
    ap.add_argument("--inplace", action="store_true", default=False,
                    help="Replace input file with cleaned output (creates a .bak backup)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = Path(args.out_path) if args.out_path else in_path.with_suffix(in_path.suffix + ".nonempty.jsonl")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    dropped_path = out_path.with_suffix(out_path.suffix + ".dropped.jsonl")
    bad_path = out_path.with_suffix(out_path.suffix + ".bad.jsonl")

    total = kept = dropped = bad = 0

    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, \
         tmp_path.open("w", encoding="utf-8") as fout, \
         dropped_path.open("w", encoding="utf-8") as fdropped, \
         bad_path.open("w", encoding="utf-8") as fbad:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                fbad.write(line + "\n")
                continue

            if isinstance(obj, dict) and is_empty_places(obj, keep_errors=args.keep_errors):
                dropped += 1
                fdropped.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            kept += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Move tmp -> out
    tmp_path.replace(out_path)

    # Optionally replace original input file
    if args.inplace:
        bak_path = in_path.with_suffix(in_path.suffix + ".bak")
        if bak_path.exists():
            bak_path.unlink()
        in_path.replace(bak_path)
        out_path.replace(in_path)
        print(f"[OK] In-place cleaned. Backup written to: {bak_path}")
        print(f"     Clean file is now:            {in_path}")
    else:
        print(f"[OK] Cleaned file written to:      {out_path}")

    print(f"Total JSON lines read:             {total}")
    print(f"Kept (non-empty places):           {kept}")
    print(f"Dropped (empty places):            {dropped}  -> {dropped_path}")
    print(f"Bad JSON lines:                    {bad}      -> {bad_path}")


if __name__ == "__main__":
    main()