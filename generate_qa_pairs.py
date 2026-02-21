#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_qa_pairs.py

Generate citation-grounded QA pairs from local chunk JSONL files using the OpenAI Responses API.

Input:
  data/chunks_clean/*.jsonl
  each line: {"articleId": "...", "chunkId": "...", "seq": 0, "text": "..."}

Output:
  data/qa_pairs.jsonl  (one JSON object per line)

Features:
- Generates a mix of SINGLE-chunk and MULTI-chunk (2 chunks) questions.
- Enforces structured JSON output via JSON Schema.
- Validates that evidence quotes are verbatim substrings of the chunk text.
- Incremental write + resume: if output exists, it will skip already-generated items.
- Progress prints.

Install:
  pip install openai

Env:
  set OPENAI_API_KEY=...
Optional:
  set OPENAI_QA_MODEL=gpt-5.2   (default is gpt-5.2)

Examples:
  # Generate 40 for a quick test
  python generate_qa_pairs.py --chunks_dir data/chunks_clean --out data/qa_pairs_40.jsonl --n 40

  # Generate 400 with 30% multi-hop (two chunks)
  python generate_qa_pairs.py --n 400 --pair_ratio 0.30 --out data/qa_pairs_400.jsonl

Notes:
- Keep temperature low; the script validates and retries if the model output is invalid.
"""

from __future__ import annotations

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

from openai import OpenAI


# -------------------------
# IO: read chunks
# -------------------------

def iter_chunks(chunks_dir: Path) -> Iterable[Dict[str, Any]]:
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {chunks_dir}")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text", "") or "").strip()
                chunk_id = str(obj.get("chunkId", "") or "").strip()
                if not chunk_id or not text:
                    continue
                yield {
                    "articleId": str(obj.get("articleId", "") or "").strip(),
                    "chunkId": chunk_id,
                    "seq": int(obj.get("seq", 0) or 0),
                    "text": text,
                }


def load_existing_keys(out_path: Path) -> Set[str]:
    """
    Resume support: return a set of keys already generated.
    Keys are either:
      "SINGLE::<chunkId>"
      "PAIR::<chunkIdA>::<chunkIdB>"  (sorted)
    """
    if not out_path.exists():
        return set()

    keys = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            kind = obj.get("difficulty")
            evid = obj.get("evidence", []) or []
            chunk_ids = sorted({e.get("chunkId") for e in evid if isinstance(e, dict) and e.get("chunkId")})
            if not chunk_ids:
                continue
            if kind == "multi" and len(chunk_ids) >= 2:
                keys.add(f"PAIR::{chunk_ids[0]}::{chunk_ids[1]}")
            else:
                keys.add(f"SINGLE::{chunk_ids[0]}")
    return keys


# -------------------------
# Prompting + schema
# -------------------------

QA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string", "minLength": 8},
        "answer": {"type": "string", "minLength": 1},
        "difficulty": {"type": "string", "enum": ["single", "multi"]},
        "style": {"type": "string", "enum": ["factoid", "list", "compare", "definition", "when", "where", "why"]},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "chunkId": {"type": "string", "minLength": 1},
                    "quote": {"type": "string", "minLength": 1},
                },
                "required": ["chunkId", "quote"],
            },
        },
    },
    "required": ["question", "answer", "difficulty", "style", "evidence"],
}


SYSTEM_INSTRUCTIONS = """You are generating evaluation QA pairs for a retrieval-grounded QA system.

Hard constraints:
- Use ONLY the provided chunk text(s). Do not use outside knowledge.
- The answer must be fully supported by the provided text(s).
- Provide evidence quotes that are verbatim substrings of the chunk text(s).
- Each evidence quote must be <= 20 words.
- Ignore bibliographic noise (e.g., "Paris, 2006" publication location lines) unless the place is discussed as a topic.
- Avoid trivial questions like "What is mentioned?".
- Write the question so it stands alone (do NOT mention "chunk", ids, or "text above").

Return ONLY valid JSON that matches the provided schema.
"""


def build_user_prompt_single(chunk: Dict[str, Any]) -> str:
    return f"""Create ONE high-quality QA pair from this chunk.

Chunk metadata:
- chunkId: {chunk["chunkId"]}
- articleId: {chunk["articleId"]}
- seq: {chunk["seq"]}

Chunk text:
\"\"\"{chunk["text"]}\"\"\"

Requirements:
- difficulty must be "single"
- evidence must include this chunkId
- Provide 1–2 evidence quotes (<=20 words each), exact substrings from the chunk text.
"""


def build_user_prompt_pair(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> str:
    return f"""Create ONE high-quality QA pair that REQUIRES combining information from BOTH chunks.

Chunk A:
- chunkId: {chunk_a["chunkId"]}
- articleId: {chunk_a["articleId"]}
- seq: {chunk_a["seq"]}
Text A:
\"\"\"{chunk_a["text"]}\"\"\"

Chunk B:
- chunkId: {chunk_b["chunkId"]}
- articleId: {chunk_b["articleId"]}
- seq: {chunk_b["seq"]}
Text B:
\"\"\"{chunk_b["text"]}\"\"\"

Requirements:
- difficulty must be "multi"
- evidence must include BOTH chunkIds (at least one quote per chunk)
- Provide 2 evidence quotes total (<=20 words each), exact substrings from the relevant chunk text.
- The question should not be answerable from only one chunk.
"""


# -------------------------
# Validation
# -------------------------

def quote_ok(quote: str, source_text: str) -> bool:
    if not quote or len(quote.split()) > 20:
        return False
    return quote in source_text


def validate_qa_obj(obj: Dict[str, Any], chunk_map: Dict[str, str]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not a dict"
    for k in ["question", "answer", "difficulty", "style", "evidence"]:
        if k not in obj:
            return False, f"missing field: {k}"
    evid = obj.get("evidence", []) or []
    if not isinstance(evid, list) or len(evid) < 1:
        return False, "evidence empty"
    for e in evid:
        if not isinstance(e, dict):
            return False, "evidence item not dict"
        cid = e.get("chunkId")
        q = e.get("quote", "")
        if cid not in chunk_map:
            return False, f"unknown chunkId in evidence: {cid}"
        if not quote_ok(q, chunk_map[cid]):
            return False, f"bad quote (not substring or >20 words) for chunkId={cid}"
    return True, "ok"


# -------------------------
# OpenAI call with retries
# -------------------------

def call_openai_structured(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_retries: int = 4,
    sleep_base_s: float = 1.5,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qa_pair",
                        "schema": QA_SCHEMA,
                        "strict": True,
                    },
                },
            )
            # The SDK surfaces generated text at output_text in many examples
            text = getattr(resp, "output_text", None)
            if not text:
                # fallback: try to serialize response and locate output text
                text = str(resp)
            return json.loads(text)
        except Exception as e:
            last_err = e
            time.sleep(sleep_base_s * attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", default="data/chunks_clean")
    ap.add_argument("--out", default="data/qa_pairs.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--pair_ratio", type=float, default=0.30, help="fraction of multi-chunk questions (0..1)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--min_chars", type=int, default=350, help="skip chunks shorter than this")
    ap.add_argument("--max_chars", type=int, default=2200, help="truncate chunk text to this many chars for prompting")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--max_attempts_per_item", type=int, default=6)
    args = ap.parse_args()

    random.seed(args.seed)

    chunks_dir = Path(args.chunks_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = os.getenv("OPENAI_QA_MODEL", "gpt-5.2")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load and filter chunks
    chunks = [c for c in iter_chunks(chunks_dir) if len(c["text"]) >= args.min_chars]
    if not chunks:
        raise SystemExit("No chunks available after filtering (min_chars too high?)")

    # Truncate for prompt
    for c in chunks:
        if len(c["text"]) > args.max_chars:
            c["text"] = c["text"][: args.max_chars]

    # Resume
    existing = load_existing_keys(out_path)
    print(f"[INFO] chunks loaded: {len(chunks)} | existing QA items: {len(existing)}")
    print(f"[INFO] model: {model}")

    target_multi = int(round(args.n * args.pair_ratio))
    target_single = args.n - target_multi

    made_single = 0
    made_multi = 0
    written = 0
    failures = 0

    # Precompute adjacency by article for better multi-hop coherence
    by_article: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        by_article.setdefault(c["articleId"], []).append(c)
    for aid in by_article:
        by_article[aid].sort(key=lambda x: (x["seq"], x["chunkId"]))

    def pick_pair() -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        # Prefer adjacent chunks in the same article
        aid = random.choice(list(by_article.keys()))
        lst = by_article[aid]
        if len(lst) < 2:
            return None
        i = random.randrange(0, len(lst) - 1)
        return lst[i], lst[i + 1]

    def key_single(cid: str) -> str:
        return f"SINGLE::{cid}"

    def key_pair(a: str, b: str) -> str:
        x, y = sorted([a, b])
        return f"PAIR::{x}::{y}"

    with out_path.open("a", encoding="utf-8") as out_f:
        while written < args.n:
            want_multi = (made_multi < target_multi) and (random.random() < args.pair_ratio)
            if want_multi:
                pair = pick_pair()
                if not pair:
                    continue
                a, b = pair
                k = key_pair(a["chunkId"], b["chunkId"])
                if k in existing:
                    continue

                chunk_map = {a["chunkId"]: a["text"], b["chunkId"]: b["text"]}
                prompt = build_user_prompt_pair(a, b)
                kind = "multi"
            else:
                c = random.choice(chunks)
                k = key_single(c["chunkId"])
                if k in existing:
                    continue

                chunk_map = {c["chunkId"]: c["text"]}
                prompt = build_user_prompt_single(c)
                kind = "single"

            ok = False
            last_reason = ""
            for attempt in range(1, args.max_attempts_per_item + 1):
                try:
                    obj = call_openai_structured(
                        client=client,
                        model=model,
                        user_prompt=prompt,
                        temperature=args.temperature,
                    )
                    valid, reason = validate_qa_obj(obj, chunk_map)
                    if not valid:
                        last_reason = reason
                        continue

                    # Stamp a deterministic id + minimal metadata
                    obj["id"] = f"qa_{written+1:05d}"
                    obj["created_from"] = kind
                    obj["source_articles"] = sorted({c.get("articleId", "") for c in [*chunk_map.keys()]})  # not critical

                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    out_f.flush()

                    existing.add(k)
                    written += 1
                    if kind == "multi":
                        made_multi += 1
                    else:
                        made_single += 1

                    if written % max(1, args.print_every) == 0:
                        print(f"[PROGRESS] written={written}/{args.n} | single={made_single} | multi={made_multi} | failures={failures}")
                        print(f"  latest: {obj['question']}")
                    ok = True
                    break
                except Exception as e:
                    last_reason = str(e)
                    time.sleep(1.0 * attempt)

            if not ok:
                failures += 1
                if failures % 10 == 0:
                    print(f"[WARN] failures={failures}. last_reason={last_reason}")

    print("\n[OK] Done.")
    print(f"Total written: {written}")
    print(f"Singles:       {made_single}")
    print(f"Multis:        {made_multi}")
    print(f"Failures:      {failures}")
    print(f"Output:        {out_path.resolve()}")


if __name__ == "__main__":
    main()
