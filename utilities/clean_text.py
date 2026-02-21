#!/usr/bin/env python3
"""
Clean merged article JSON files (conservative):
- remove page markers like "#p12" / "#p12."
- collapse whitespace (including newlines) to single spaces

Default:
  input:  data/articles
  output: data/articles_clean

Run from repo root:
  python clean_articles.py
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any


PAGE_MARK_RE = re.compile(r"#p\d+\b\.?", flags=re.IGNORECASE)


def clean_body(text: str) -> str:
    if not text:
        return ""

    # Normalize newlines first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page markers like "#p21" or "#p21."
    text = PAGE_MARK_RE.sub(" ", text)

    # Collapse whitespace (incl. \n\n) to single spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_header(full_text: str) -> tuple[str, str]:
    """
    If the text starts with 'ARTICLE_ID: ...', preserve that first line as header.
    Otherwise return ("", full_text).
    """
    if not full_text:
        return "", ""

    if full_text.startswith("ARTICLE_ID:"):
        first_line, _, rest = full_text.partition("\n")
        rest = rest.lstrip("\n")  # handle "ARTICLE_ID: ...\n\n..."
        return first_line.strip(), rest

    return "", full_text


def process_file(in_path: Path, out_path: Path) -> None:
    obj: Dict[str, Any] = json.loads(in_path.read_text(encoding="utf-8"))

    article_id = str(obj.get("articleId", "")).strip()
    full_text = str(obj.get("text", ""))

    header, body = split_header(full_text)
    if not header and article_id:
        header = f"ARTICLE_ID: {article_id}"

    cleaned = clean_body(body)

    obj["articleId"] = article_id
    obj["text"] = (header + "\n" + cleaned).strip() if header else cleaned

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("data/articles"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/articles_clean"))
    ap.add_argument("--inplace", action="store_true", help="Overwrite files in --in-dir")
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = in_dir if args.inplace else args.out_dir

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {in_dir}")

    for fp in files:
        out_fp = out_dir / fp.name
        process_file(fp, out_fp)

    print(f"Processed {len(files)} file(s). Output: {out_dir}")


if __name__ == "__main__":
    main()
