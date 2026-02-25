#!/usr/bin/env python3
"""
LLM-as-a-judge evaluation for your QA outputs CSV.

Input CSV columns expected (at minimum):
- style, question, gold_answer
- base_answer, rag_answer, graphrag_answer
(Optional): base_model, rag_sources, graphrag_sources, rag_hit_count, graphrag_hit_count, error

Outputs:
- <out_csv> with appended columns:
  base_score, rag_score, graphrag_score,
  base_pass, rag_pass, graphrag_pass,
  winner, judge_rationale

- <out_jsonl> with full judge payload per row (for audit/debug)

Usage:
  export OPENAI_API_KEY="..."
  python judge_eval.py --in_csv qa_model_outputs.csv --out_csv judged.csv

Notes:
- This is "reference-guided grading": judge compares to gold_answer (not to sources).
- To reduce cost, the script grades base/rag/graphrag in ONE judge call per row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # loads .env from current working dir

# -----------------------------
# Config / Prompt
# -----------------------------

JUDGE_SYSTEM = (
    "You are an impartial evaluator for question answering over scholarly texts.\n"
    "You will be given: (1) a question, (2) a gold/reference answer, and (3) three candidate answers.\n"
    "Your job is to score each candidate answer relative to the gold answer.\n\n"
    "Rules:\n"
    "- Use ONLY the gold answer as the reference for correctness.\n"
    "- Focus on key factual content and relationships. Ignore superficial paraphrase.\n"
    "- Penalize contradictions with the gold answer strongly.\n"
    "- Penalize hallucinated additions that conflict with the gold.\n"
    "- If an answer is empty, nonsense, or refuses (e.g., 'I don't know', asks for the paper), score 0.\n"
    "- For list questions: full credit requires all required items; partial credit proportional to coverage, "
    "but contradictions drop the score.\n"
    "- For compare/why/definition: must capture the core relationship/claim.\n\n"
    "Return STRICT JSON only, with the schema below."
)

# We ask the judge for strict JSON; we still parse defensively.
JUDGE_USER_TEMPLATE = """\
Evaluate the three candidate answers against the gold answer.

Question style: {style}
Question: {question}

Gold answer:
{gold}

Candidate A (base_answer):
{base}

Candidate B (rag_answer):
{rag}

Candidate C (graphrag_answer):
{graphrag}

Return JSON with EXACT keys:
{{
  "base_score": <integer 0-10>,
  "rag_score": <integer 0-10>,
  "graphrag_score": <integer 0-10>,
  "base_pass": <boolean>,
  "rag_pass": <boolean>,
  "graphrag_pass": <boolean>,
  "winner": <"base"|"rag"|"graphrag"|"tie">,
  "judge_rationale": <string, max 2 sentences>
}}

Pass criterion: score >= {threshold}.
Winner: highest score; "tie" if two or three are tied for highest.
"""

ABSTAIN_PATTERNS = [
    r"^\s*i\s+don['’]t\s+know\s*\.?\s*$",
    r"^\s*i\s+do\s+not\s+know\s*\.?\s*$",
    r"^\s*don['’]t\s+know\s*\.?\s*$",
    r"^\s*cannot\s+answer\s*\.?\s*$",
    r"^\s*no\s+idea\s*\.?\s*$",
]


# -----------------------------
# Helpers
# -----------------------------

def normalize_answer(a: Optional[str]) -> str:
    if a is None:
        return ""
    a = str(a)
    # common spreadsheet artifacts
    if a.strip().upper() in {"NA", "N/A", "NULL", "NONE"}:
        return ""
    if a.strip() == "#NAME?":
        return ""
    return a.strip()


def looks_like_abstain(a: str) -> bool:
    if not a:
        return True
    low = a.strip().lower()
    for pat in ABSTAIN_PATTERNS:
        if re.match(pat, low):
            return True
    # other common refusals
    if "i don't have access" in low or "i do not have access" in low:
        return True
    if "paste the passage" in low or "provide a link" in low or "upload" in low:
        return True
    return False


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract first JSON object from model output.
    """
    text = text.strip()
    # Fast path
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Find first {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in: {text[:200]!r}")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def _extract_text_from_part(part: Any) -> str:
    """
    Extract text from a content part that may be dict-like or object-like.
    """
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            nested = text.get("value")
            if isinstance(nested, str):
                return nested
        return ""

    # Object-like fallbacks (OpenAI SDK typed objects)
    text_attr = getattr(part, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(text_attr, dict):
        nested = text_attr.get("value")
        if isinstance(nested, str):
            return nested

    try:
        dump_fn = getattr(part, "model_dump", None)
        if callable(dump_fn):
            return _extract_text_from_part(dump_fn())
    except Exception:
        pass
    return ""


def get_chat_content_text(content: Any) -> str:
    """
    Normalize chat completion message content into plain text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _extract_text_from_part(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def get_responses_text(resp: Any) -> str:
    """
    Extract text robustly from Responses API objects.
    """
    # Fast path available in most SDK versions.
    text = normalize_answer(getattr(resp, "output_text", ""))
    if text:
        return text

    try:
        dump_fn = getattr(resp, "model_dump", None)
        if callable(dump_fn):
            data = dump_fn()
        elif isinstance(resp, dict):
            data = resp
        else:
            data = {}
    except Exception:
        data = {}

    parts = []
    for out_item in data.get("output", []) or []:
        if isinstance(out_item, dict):
            for c in out_item.get("content", []) or []:
                t = _extract_text_from_part(c)
                if t:
                    parts.append(t)
    return "\n".join(parts).strip()


def clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def compute_winner(scores: Dict[str, int]) -> str:
    max_score = max(scores.values())
    best = [k for k, v in scores.items() if v == max_score]
    if len(best) != 1:
        return "tie"
    return best[0]


def get_row_key(row: Dict[str, Any], idx: int, id_column: str) -> str:
    """
    Stable key for resume mode. Prefer explicit ID, then row_index, then fallback idx.
    """
    preferred = str(row.get(id_column, "") or "").strip()
    if preferred:
        return f"{id_column}:{preferred}"

    row_index = str(row.get("row_index", "") or "").strip()
    if row_index:
        return f"row_index:{row_index}"

    return f"idx:{idx}"


@dataclass
class JudgeResult:
    base_score: int
    rag_score: int
    graphrag_score: int
    base_pass: bool
    rag_pass: bool
    graphrag_pass: bool
    winner: str
    judge_rationale: str
    raw: Dict[str, Any]


def judge_row(
    client: OpenAI,
    judge_model: str,
    style: str,
    question: str,
    gold: str,
    base: str,
    rag: str,
    graphrag: str,
    threshold: int,
    temperature: Optional[float] = None,
    max_completion_tokens: int = 250,
    retry: int = 2,
) -> JudgeResult:
    prompt = JUDGE_USER_TEMPLATE.format(
        style=style or "unknown",
        question=question,
        gold=gold,
        base=base,
        rag=rag,
        graphrag=graphrag,
        threshold=threshold,
    )

    last_err: Optional[Exception] = None
    for attempt in range(retry + 1):
        try:
            content = ""
            api_used = ""

            # Primary path for GPT-5 family: Responses API.
            try:
                resp_req: Dict[str, Any] = {
                    "model": judge_model,
                    "input": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": prompt + "\n\nReturn ONLY a valid JSON object.",
                        },
                    ],
                    "max_output_tokens": max_completion_tokens,
                }
                if temperature is not None:
                    resp_req["temperature"] = temperature
                resp = client.responses.create(**resp_req)
                content = get_responses_text(resp)
                api_used = "responses"
            except Exception:
                content = ""

            # Fallback: chat completions in strict JSON mode.
            if not content:
                req: Dict[str, Any] = {
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": max_completion_tokens,
                    "response_format": {"type": "json_object"},
                }
                if temperature is not None:
                    req["temperature"] = temperature
                resp = client.chat.completions.create(**req)
                content = get_chat_content_text(resp.choices[0].message.content)
                api_used = "chat_json"

            # Fallback: chat completions without JSON mode, explicit JSON instruction.
            if not content:
                fallback_req: Dict[str, Any] = {
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": prompt + "\n\nReturn ONLY a valid JSON object.",
                        },
                    ],
                    "max_completion_tokens": max_completion_tokens,
                }
                if temperature is not None:
                    fallback_req["temperature"] = temperature
                resp2 = client.chat.completions.create(**fallback_req)
                content = get_chat_content_text(resp2.choices[0].message.content)
                api_used = "chat_plain"

            if not content:
                raise ValueError("Judge returned empty content across all API paths")

            data = extract_json_object(content)

            # Normalize & clamp
            base_score = clamp_int(data.get("base_score", 0), 0, 10)
            rag_score = clamp_int(data.get("rag_score", 0), 0, 10)
            graphrag_score = clamp_int(data.get("graphrag_score", 0), 0, 10)

            # If answers are obvious abstains, enforce 0 (guards judge flakiness)
            if looks_like_abstain(base):
                base_score = 0
            if looks_like_abstain(rag):
                rag_score = 0
            if looks_like_abstain(graphrag):
                graphrag_score = 0

            # Pass booleans (judge provides, but we enforce threshold deterministically)
            base_pass = base_score >= threshold
            rag_pass = rag_score >= threshold
            graphrag_pass = graphrag_score >= threshold

            scores = {"base": base_score, "rag": rag_score, "graphrag": graphrag_score}
            winner = compute_winner(scores)

            rationale = str(data.get("judge_rationale", "")).strip()
            if len(rationale) > 350:
                rationale = rationale[:350].rstrip() + "…"

            # Keep raw for audit
            raw = {
                "judge_model": judge_model,
                "api_used": api_used,
                "response_json": data,
                "response_text": content,
            }

            return JudgeResult(
                base_score=base_score,
                rag_score=rag_score,
                graphrag_score=graphrag_score,
                base_pass=base_pass,
                rag_pass=rag_pass,
                graphrag_pass=graphrag_pass,
                winner=winner,
                judge_rationale=rationale,
                raw=raw,
            )

        except Exception as e:
            last_err = e
            # tighten instruction on retry
            if attempt < retry:
                time.sleep(0.4)
                continue

    raise RuntimeError(f"Judge failed after retries: {last_err}") from last_err


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV with questions + answers.")
    ap.add_argument("--out_csv", default="judged_outputs.csv", help="Output CSV with appended judge columns.")
    ap.add_argument("--out_jsonl", default="judged_outputs.jsonl", help="Per-row judge payloads (audit log).")
    ap.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", "gpt-5-mini"))
    ap.add_argument("--threshold", type=int, default=int(os.getenv("JUDGE_THRESHOLD", "6")))
    ap.add_argument("--max_rows", type=int, default=0, help="If >0, evaluate only first N rows.")
    ap.add_argument("--id_column", default="id", help="Column used as stable resume key.")
    ap.add_argument(
        "--max_completion_tokens",
        type=int,
        default=int(os.getenv("JUDGE_MAX_COMPLETION_TOKENS", "250")),
        help="Max completion tokens for judge output.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. If omitted, API default is used.",
    )
    ap.add_argument("--resume", action="store_true", help="Resume by skipping rows already present in out_csv.")
    ap.add_argument("--save_every", type=int, default=25, help="Flush outputs every N rows.")
    ap.add_argument("--log_every", type=int, default=10, help="Progress log interval when not verbose.")
    ap.add_argument("--verbose", action="store_true", help="Print per-row progress.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls (rate-limit friendly).")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read input
    with open(args.in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    # Resume support: read out_csv to know how many already processed
    processed_keys = set()
    if args.resume and os.path.exists(args.out_csv):
        with open(args.out_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for out_idx, out_row in enumerate(r):
                processed_keys.add(get_row_key(out_row, out_idx, args.id_column))
        print(f"Resume enabled: found {len(processed_keys)} rows already in {args.out_csv}", file=sys.stderr)

    remaining_rows = 0
    for idx, row in enumerate(rows):
        key = get_row_key(row, idx, args.id_column)
        if args.resume and key in processed_keys:
            continue
        remaining_rows += 1

    print(
        f"Starting judge eval (input_rows={len(rows)}, will_judge={remaining_rows}, "
        f"resume={args.resume}, model={args.judge_model}, threshold={args.threshold})",
        file=sys.stderr,
    )

    # Prepare writer (append if resume, else overwrite)
    out_exists = os.path.exists(args.out_csv)
    mode = "a" if (args.resume and out_exists) else "w"
    jsonl_exists = os.path.exists(args.out_jsonl)
    jsonl_mode = "a" if (args.resume and jsonl_exists) else "w"

    # Determine output fieldnames: input + new columns
    new_cols = [
        "base_score", "rag_score", "graphrag_score",
        "base_pass", "rag_pass", "graphrag_pass",
        "winner", "judge_rationale", "judge_error",
    ]
    fieldnames = list(rows[0].keys()) if rows else []
    for c in new_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    n_done = 0
    n_skipped = 0
    n_errors = 0
    start_ts = time.perf_counter()
    log_every = max(1, int(args.log_every))

    # Counters
    sum_scores = {"base": 0, "rag": 0, "graphrag": 0}
    pass_counts = {"base": 0, "rag": 0, "graphrag": 0}
    win_counts = {"base": 0, "rag": 0, "graphrag": 0, "tie": 0}

    with open(args.out_csv, mode, encoding="utf-8", newline="") as out_f, \
         open(args.out_jsonl, jsonl_mode, encoding="utf-8") as jsonl_f:

        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for idx, row in enumerate(rows):
            style = (row.get("style") or "").strip()
            question = (row.get("question") or "").strip()
            gold = normalize_answer(row.get("gold_answer"))
            base = normalize_answer(row.get("base_answer"))
            rag = normalize_answer(row.get("rag_answer"))
            graphrag = normalize_answer(row.get("graphrag_answer"))

            key = get_row_key(row, idx, args.id_column)
            if args.resume and key in processed_keys:
                n_skipped += 1
                if args.verbose:
                    print(f"[row={idx} key={key}] skipped (already judged)", file=sys.stderr)
                continue

            row_start = time.perf_counter()
            if args.verbose:
                q_preview = (question[:100] + "...") if len(question) > 100 else question
                print(f"[row={idx} key={key}] judging: {q_preview}", file=sys.stderr)

            # Judge
            judge_error = ""
            try:
                jr = judge_row(
                    client=client,
                    judge_model=args.judge_model,
                    style=style,
                    question=question,
                    gold=gold,
                    base=base,
                    rag=rag,
                    graphrag=graphrag,
                    threshold=args.threshold,
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                )
            except Exception as exc:
                n_errors += 1
                judge_error = str(exc)
                jr = JudgeResult(
                    base_score=0,
                    rag_score=0,
                    graphrag_score=0,
                    base_pass=False,
                    rag_pass=False,
                    graphrag_pass=False,
                    winner="tie",
                    judge_rationale="Judge failed for this row; see judge_error.",
                    raw={"judge_model": args.judge_model, "error": judge_error},
                )

            # Update row
            row["base_score"] = str(jr.base_score)
            row["rag_score"] = str(jr.rag_score)
            row["graphrag_score"] = str(jr.graphrag_score)
            row["base_pass"] = str(jr.base_pass)
            row["rag_pass"] = str(jr.rag_pass)
            row["graphrag_pass"] = str(jr.graphrag_pass)
            row["winner"] = jr.winner
            row["judge_rationale"] = jr.judge_rationale
            row["judge_error"] = judge_error

            writer.writerow(row)
            n_done += 1

            # Audit log
            audit = {
                "row_index": idx,
                "style": style,
                "question": question,
                "gold_answer": gold,
                "base_answer": base,
                "rag_answer": rag,
                "graphrag_answer": graphrag,
                "result": {
                    "base_score": jr.base_score,
                    "rag_score": jr.rag_score,
                    "graphrag_score": jr.graphrag_score,
                    "base_pass": jr.base_pass,
                    "rag_pass": jr.rag_pass,
                    "graphrag_pass": jr.graphrag_pass,
                    "winner": jr.winner,
                    "judge_rationale": jr.judge_rationale,
                    "judge_error": judge_error,
                },
                "raw": jr.raw,
            }
            jsonl_f.write(json.dumps(audit, ensure_ascii=False) + "\n")

            # Aggregates
            sum_scores["base"] += jr.base_score
            sum_scores["rag"] += jr.rag_score
            sum_scores["graphrag"] += jr.graphrag_score
            pass_counts["base"] += int(jr.base_pass)
            pass_counts["rag"] += int(jr.rag_pass)
            pass_counts["graphrag"] += int(jr.graphrag_pass)
            win_counts[jr.winner] = win_counts.get(jr.winner, 0) + 1

            if args.sleep:
                time.sleep(args.sleep)

            if n_done % args.save_every == 0:
                out_f.flush()
                jsonl_f.flush()
                print(f"Saved {n_done} judged rows…", file=sys.stderr)

            elapsed_row = time.perf_counter() - row_start
            if args.verbose:
                status = "ok" if not judge_error else f"error={judge_error}"
                api_used = str(jr.raw.get("api_used", "-"))
                print(
                    f"[row={idx} key={key}] done in {elapsed_row:.2f}s "
                    f"(winner={jr.winner}, api={api_used}, {status})",
                    file=sys.stderr,
                )
            elif n_done % log_every == 0:
                elapsed_total = time.perf_counter() - start_ts
                print(
                    f"Progress: judged={n_done}/{remaining_rows}, skipped={n_skipped}, "
                    f"errors={n_errors}, elapsed={elapsed_total:.1f}s",
                    file=sys.stderr,
                )

    # Print summary
    if n_done > 0:
        avg_base = sum_scores["base"] / n_done
        avg_rag = sum_scores["rag"] / n_done
        avg_graph = sum_scores["graphrag"] / n_done

        print("\n--- Judge Summary ---")
        print(f"Rows judged: {n_done} (skipped: {n_skipped}, judge_errors: {n_errors})")
        print(f"Judge model: {args.judge_model} | pass threshold: {args.threshold}")
        print(f"Avg score  base: {avg_base:.2f} | pass@{args.threshold}: {pass_counts['base']/n_done:.2f}")
        print(f"Avg score   rag: {avg_rag:.2f} | pass@{args.threshold}: {pass_counts['rag']/n_done:.2f}")
        print(f"Avg score graph: {avg_graph:.2f} | pass@{args.threshold}: {pass_counts['graphrag']/n_done:.2f}")
        print(f"Wins (base/rag/graphrag/tie): "
              f"{win_counts.get('base',0)}/{win_counts.get('rag',0)}/{win_counts.get('graphrag',0)}/{win_counts.get('tie',0)}")
        print(f"\nWrote: {args.out_csv}")
        print(f"Wrote: {args.out_jsonl}")
    else:
        print("No rows judged (maybe everything was skipped due to --resume).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
