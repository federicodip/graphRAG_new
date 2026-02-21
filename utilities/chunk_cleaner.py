#!/usr/bin/env python3
"""
Rewrite (clean) pre-chunked JSONL files in data/chunks and write cleaned copies
to a new directory (default: data/chunks_clean).

What it does:
- For each *.jsonl in input dir:
  - Reads each JSON line (chunk record)
  - Cleans the "text" field:
      * removes page markers like "#p12" / "#p12."
      * removes trailing "Source: ..." block (if present)
      * collapses whitespace
  - Writes the cleaned record as JSONL to output dir (same filename)
- Keeps the original chunk fields (articleId, chunkId, seq) unchanged.

Run:
  python rewrite_chunks.py
  python rewrite_chunks.py --in_dir data/chunks --out_dir data/chunks_clean
  python rewrite_chunks.py --in_dir data/chunks --out_dir data/chunks_clean --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


PAGE_MARK_RE = re.compile(r"#p\d+\b\.?", flags=re.IGNORECASE)
SOURCE_TRAILER_RE = re.compile(r"\s*Source:\s.*$", flags=re.DOTALL)


def clean_chunk_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PAGE_MARK_RE.sub(" ", text)          # drop "#p12"
    text = SOURCE_TRAILER_RE.sub("", text)      # drop trailing "Source: ..."
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield (line_number, obj) for each non-empty JSONL line."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/chunks", help="Input directory with *.jsonl")
    parser.add_argument("--out_dir", default="data/chunks_clean", help="Output directory for cleaned *.jsonl")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing out_dir (files will be overwritten).",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {in_dir}")

    jsonl_files = sorted(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {in_dir}")

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output dir already exists: {out_dir}. Use --overwrite or pick a new --out_dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    total_skipped = 0

    for src_path in jsonl_files:
        dst_path = out_dir / src_path.name

        written = 0
        read = 0

        with dst_path.open("w", encoding="utf-8") as out_f:
            for line_no, row in iter_jsonl(src_path):
                read += 1
                total_in += 1

                # Basic validation: keep row as-is except cleaned text
                article_id = str(row.get("articleId", "")).strip()
                chunk_id = str(row.get("chunkId", "")).strip()
                if not article_id or not chunk_id:
                    total_skipped += 1
                    continue

                raw_text = row.get("text", "")
                cleaned = clean_chunk_text(str(raw_text))

                # If cleaning nuked everything, skip it
                if not cleaned:
                    total_skipped += 1
                    continue

                row["text"] = cleaned

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                total_out += 1

        print(f"WROTE {dst_path.name}: {written}/{read} chunks")

    print("\nDone rewriting.")
    print(f"Input chunks read:     {total_in}")
    print(f"Cleaned chunks written:{total_out}")
    print(f"Skipped:              {total_skipped}")
    print(f"Output dir:           {out_dir.resolve()}")


if __name__ == "__main__":
    main()
