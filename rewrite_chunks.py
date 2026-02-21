#!/usr/bin/env python3
"""
Rewrite/clean pre-chunked JSONL files.

Input JSONL schema (per line):
  {
    "articleId": "...",
    "chunkId": "isaw-papers-1-2011:0019",
    "seq": 19,
    "text": "..."
  }

Output:
- cleaned text (remove #p12, collapse whitespace, remove trailing "Source: ...")
- hard cap per chunk by characters (default 850)
- if a chunk is split, emit multiple rows:
    chunkId: "<orig>::s00", "<orig>::s01", ...
    parentChunkId: "<orig>"
    subseq: 0, 1, ...

Run:
  python .\rewrite_chunks.py --in_dir data/chunks --out_dir data/chunks_clean --max_chars 850 --overlap 120
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PAGE_MARK_RE = re.compile(r"#p\d+\b\.?", flags=re.IGNORECASE)
SOURCE_TRAILER_RE = re.compile(r"\s*Source:\s.*$", flags=re.DOTALL)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PAGE_MARK_RE.sub(" ", text)
    text = SOURCE_TRAILER_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_by_chars(text: str, max_chars: int, overlap: int, soft_boundary: int = 80) -> List[str]:
    """
    Split text into overlapping chunks of at most max_chars.
    Tries to cut on whitespace near the boundary (within soft_boundary).
    """
    if len(text) <= max_chars:
        return [text]

    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    chunks: List[str] = []
    step = max_chars - overlap
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + max_chars)

        if end < n:
            cut = text.rfind(" ", start, end)
            if cut != -1 and cut >= end - soft_boundary and cut > start:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            if len(chunk) > max_chars:
                chunk = chunk[:max_chars].strip()
            chunks.append(chunk)

        if end >= n:
            break

        start = start + step

    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/chunks", help="Input dir containing *.jsonl")
    ap.add_argument("--out_dir", default="data/chunks_clean", help="Output dir for rewritten *.jsonl")
    ap.add_argument("--max_chars", type=int, default=850, help="Hard cap per chunk (chars)")
    ap.add_argument("--overlap", type=int, default=120, help="Overlap for split chunks (chars)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {in_dir}")

    total_in = 0
    total_out = 0
    max_seen = 0

    print(f"[config] in_dir={in_dir} out_dir={out_dir} max_chars={args.max_chars} overlap={args.overlap}")

    for jf in jsonl_files:
        out_path = out_dir / jf.name
        in_count = 0
        out_count = 0
        file_max = 0

        with out_path.open("w", encoding="utf-8", newline="\n") as w:
            for row in iter_jsonl(jf):
                in_count += 1
                total_in += 1

                article_id = str(row.get("articleId", "")).strip()
                chunk_id = str(row.get("chunkId", "")).strip()
                seq = row.get("seq", 0)
                raw = str(row.get("text", "") or "")

                if not article_id or not chunk_id or not raw.strip():
                    continue

                text = clean_text(raw)
                if not text:
                    continue

                file_max = max(file_max, len(text))
                max_seen = max(max_seen, len(text))

                parts = split_by_chars(text, max_chars=args.max_chars, overlap=args.overlap)

                if len(parts) == 1:
                    out_row = dict(row)
                    out_row["text"] = parts[0]
                    w.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    out_count += 1
                    total_out += 1
                else:
                    for j, part in enumerate(parts):
                        out_row = dict(row)
                        out_row["text"] = part
                        out_row["parentChunkId"] = chunk_id
                        out_row["subseq"] = j
                        out_row["chunkId"] = f"{chunk_id}::s{j:02d}"
                        w.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        out_count += 1
                        total_out += 1

        print(f"[file] {jf.name}: in={in_count} out={out_count} max_clean_len={file_max}")
        total_in += 0  # (kept for clarity)

    print("\nDone.")
    print(f"Total in lines:  {total_in}")
    print(f"Total out lines: {total_out}")
    print(f"Max cleaned len seen (chars): {max_seen}")


if __name__ == "__main__":
    main()
