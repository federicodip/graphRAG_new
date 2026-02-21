"""
Generate one question per chunk from JSONL chunk files.

TWEAKS APPLIED:
- Target is 6 chunks per file, BUT if a file has fewer chunks, we use all of them.
- Chunks are picked RANDOMLY by default (not the first six).
- Removes trailing "Source: ..." lines from passages to prevent bibliographic questions.
- Forces content-based questions (no author/title/journal/source/citation questions).
- Prefixes each question with: "According to {articleId}, ..."

Input:
  data/chunks/*.jsonl
  Each line must be JSON with at least:
    - articleId (str)
    - chunkId (str)
    - seq (int)
    - text (str)

Output:
  data/generated_questions.csv

Run (from repo root):
  python generate_questions.py
  python generate_questions.py --seed 42
  python generate_questions.py --n_per_file 10 --out data/q.csv --seed 123

Requires (pip):
  python-dotenv langchain-ollama
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


# --- Prompting ---

SYSTEM_INSTRUCTIONS = (
    "Write ONE question that can be answered using ONLY the passage.\n"
    "Hard rules:\n"
    "- Ask about the MAIN CONTENT, not bibliographic metadata.\n"
    "- Do NOT ask about the author, title, journal, 'Source:', citation, publication, or references.\n"
    "- The question must include at least ONE specific detail from the passage "
    "(a number, date/century, named entity, place, or technical term).\n"
    "- Output ONLY the question text (no quotes, no numbering, no extra text).\n"
)

# Keep passages compact to help the model focus.
WHITESPACE_RE = re.compile(r"\s+")
PAGE_MARK_RE = re.compile(r"#p\d+\b\.?", flags=re.IGNORECASE)
SOURCE_TAIL_RE = re.compile(r"\bSource:\s.*$", flags=re.IGNORECASE)  # remove trailing Source: ...


def clean_passage(text: str) -> str:
    """Light cleanup: remove page markers, strip trailing Source:, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PAGE_MARK_RE.sub(" ", text)
    text = SOURCE_TAIL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def ensure_question_format(q: str) -> str:
    """Make sure it looks like a clean question line."""
    q = q.strip().strip('"').strip()
    # sometimes models output multiple lines; keep the first non-empty
    if "\n" in q:
        q = next((line.strip() for line in q.splitlines() if line.strip()), q).strip()
    if not q.endswith("?"):
        q += "?"
    return q


def prefix_with_article(article_id: str, question: str) -> str:
    """Prefix question with provenance label."""
    q = question.strip()
    if not q:
        return q

    # Lowercase first char for "According to X, ..." if it begins with a letter.
    if q and q[0].isalpha():
        q = q[0].lower() + q[1:]

    return f"According to {article_id}, {q}"


# --- JSONL IO ---

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_chunks_random(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Pick up to n chunks at random.
    If fewer than n chunks exist, return all of them.
    """
    if len(rows) <= n:
        return rows
    idxs = rng.sample(range(len(rows)), n)
    idxs.sort()  # keeps output in a stable-ish order
    return [rows[i] for i in idxs]


# --- LLM call ---

def make_question(llm: ChatOllama, passage: str) -> str:
    """Ask the LLM for a single content-based question answerable by the passage."""
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"PASSAGE:\n{passage}\n\n"
        f"Question:"
    )
    out = llm.invoke(prompt).content
    return ensure_question_format(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", default="data/chunks", help="Folder containing *.jsonl chunk files")
    parser.add_argument("--n_per_file", type=int, default=6, help="Target number of chunks per jsonl file (default: 6)")
    parser.add_argument("--out", default="data/generated_questions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    load_dotenv()

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("CHAT_MODEL", "llama3.1:8b")

    llm = ChatOllama(
        model=chat_model,
        base_url=ollama_base_url,
        temperature=0,
    )

    chunks_dir = Path(args.chunks_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in: {chunks_dir}")

    rng = random.Random(args.seed)

    # CSV columns
    fieldnames = [
        "file",
        "articleId",
        "chunkId",
        "seq",
        "question",
        "chunk_text",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for jf in jsonl_files:
            rows = read_jsonl(jf)
            selected = pick_chunks_random(rows, args.n_per_file, rng)

            if len(rows) < args.n_per_file:
                print(f"ℹ️  {jf.name}: only {len(rows)} chunks available; using all of them.")
            else:
                print(f"Processing {jf.name}: using {len(selected)} random chunks (target={args.n_per_file})")

            for obj in selected:
                article_id = str(obj.get("articleId", "")).strip()
                chunk_id = str(obj.get("chunkId", "")).strip()
                seq = obj.get("seq", "")
                text = obj.get("text", "")

                passage = clean_passage(text if isinstance(text, str) else "")
                if not passage or not article_id:
                    continue

                try:
                    q = make_question(llm, passage)
                    q = prefix_with_article(article_id, q)
                except Exception as e:
                    q = f"[ERROR generating question: {e}]"

                writer.writerow(
                    {
                        "file": jf.name,
                        "articleId": article_id,
                        "chunkId": chunk_id,
                        "seq": seq,
                        "question": q,
                        "chunk_text": passage,
                    }
                )

    print(f"\nDONE Wrote: {out_path}")


if __name__ == "__main__":
    main()
