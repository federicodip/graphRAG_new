#!/usr/bin/env python3
"""
Merge ALL chunked JSONL files in a folder into per-article JSON files,
and strip the trailing "Source: ..." metadata from each chunk before merging.

Usage (from repo root):
  python merge_chunks_folder.py
  python merge_chunks_folder.py --chunks-dir data/chunks --articles-dir data/articles
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def safe_int(x: Any, default: int = 10**18) -> int:
    try:
        return int(x)
    except Exception:
        return default


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "article"


def strip_trailing_source(text: str, tail_max_chars: int = 800) -> str:
    """
    Removes trailing metadata like:
      "... Source: Alexander Jones ... (2011)."
    Only strips if the LAST occurrence of 'source:' (case-insensitive)
    appears near the end of the chunk (within tail_max_chars),
    to avoid deleting legitimate 'source:' mentions in the body.

    Also handles typos like 'Csource:' because it searches for 'source:' anywhere.
    """
    if not text:
        return text

    # Find last occurrence of "source:" (case-insensitive)
    matches = list(re.finditer(r"(?i)source:", text))
    if not matches:
        return text

    last = matches[-1]
    # Only strip if it's likely the appended footer
    if len(text) - last.start() <= tail_max_chars:
        return text[: last.start()].rstrip()

    return text


def iter_jsonl_files(chunks_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(chunks_dir.glob(pattern))


def load_all_chunks(
    jsonl_files: List[Path],
    strip_source: bool,
    source_tail_maxchars: int,
) -> Dict[str, List[Tuple[int, str, str]]]:
    grouped: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {fp} line {lineno}: {e}") from e

                article_id = str(obj.get("articleId", "")).strip()
                if not article_id:
                    continue

                seq = safe_int(obj.get("seq"))
                chunk_id = str(obj.get("chunkId", ""))
                text = str(obj.get("text", ""))

                if strip_source:
                    text = strip_trailing_source(text, tail_max_chars=source_tail_maxchars)

                grouped[article_id].append((seq, chunk_id, text))

    return grouped


def merge_and_write(
    grouped: Dict[str, List[Tuple[int, str, str]]],
    articles_dir: Path,
    sep: str,
    header_prefix: str,
) -> int:
    articles_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for article_id, chunks in grouped.items():
        chunks.sort(key=lambda t: (t[0], t[1]))
        merged_text = sep.join(t[2] for t in chunks if t[2])

        final_text = f"{header_prefix}{article_id}\n\n{merged_text}"
        out_obj = {"articleId": article_id, "text": final_text}

        out_name = sanitize_filename(article_id) + ".json"
        out_path = articles_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    ap.add_argument("--articles-dir", type=Path, default=Path("data/articles"))
    ap.add_argument("--recursive", action="store_true", help="Search chunks-dir recursively")
    ap.add_argument("--sep", default="\n\n", help="Separator inserted between merged chunks")
    ap.add_argument("--header-prefix", default="ARTICLE_ID: ", help="Header prefix in output text")

    ap.add_argument(
        "--strip-source",
        action="store_true",
        default=True,
        help="Strip trailing 'Source: ...' footer from each chunk (default: on)",
    )
    ap.add_argument(
        "--no-strip-source",
        dest="strip_source",
        action="store_false",
        help="Disable stripping trailing 'Source: ...' footer",
    )
    ap.add_argument(
        "--source-tail-maxchars",
        type=int,
        default=500,
        help="Only strip 'Source:' if it occurs within this many chars from chunk end",
    )

    args = ap.parse_args()

    if not args.chunks_dir.exists():
        raise FileNotFoundError(f"Chunks folder not found: {args.chunks_dir}")

    jsonl_files = iter_jsonl_files(args.chunks_dir, args.recursive)
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in: {args.chunks_dir}")

    grouped = load_all_chunks(
        jsonl_files,
        strip_source=args.strip_source,
        source_tail_maxchars=args.source_tail_maxchars,
    )

    n = merge_and_write(grouped, args.articles_dir, args.sep, args.header_prefix)

    print(f"Read {len(jsonl_files)} jsonl files.")
    print(f"Wrote {n} merged article JSON files to: {args.articles_dir}")


if __name__ == "__main__":
    main()
