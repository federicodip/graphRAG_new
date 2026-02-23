#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean Wikidata-entity extraction JSONL.

(unchanged docstring)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_WS = re.compile(r"\s+")

def fold_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_key(s: str) -> str:
    s = (s or "").strip().replace("\u2019", "'")
    s = _PUNCT_EDGE.sub("", s)
    s = fold_ascii(s)
    s = _WS.sub(" ", s).strip()
    return s.lower()

def strip_possessive(s: str) -> str:
    s = (s or "").replace("\u2019", "'").strip()
    return re.sub(r"(?:'s)$", "", s, flags=re.IGNORECASE).strip()


def load_pleiades_name_set(pleiades_index: Path) -> Set[str]:
    if not pleiades_index.exists():
        raise FileNotFoundError(f"Pleiades index not found: {pleiades_index}")

    names: Set[str] = set()

    with pleiades_index.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            title = row.get("title") or ""
            alt = row.get("altNames") or []

            if isinstance(title, str) and title.strip():
                names.add(norm_key(title))

            if isinstance(alt, list):
                for a in alt:
                    if isinstance(a, str) and a.strip():
                        names.add(norm_key(a))

    names.discard("")
    return names


# -------------------------
# Filters
# -------------------------

CODE_RE = re.compile(r"^[A-Z0-9]{2,8}$")

BAD_DESC_SUBSTRINGS = [
    # original
    "video game",
    "radio station",
    "painting",
    "family name",
    "given name",
    "tanker",
    "genus",
    "album",
    "song",
    "single",
    "band",
    "surname",
    "non profit organization",
    

    # NEW: automotive / brand sense errors
    "automobile",
    "car",
    "marque",
    "ford",

    # NEW: naval / military hardware sense errors
    "aircraft carrier",
    "united states navy",

    # NEW: geology / rock sense errors
    "igneous rock",
    "rock",

    # NEW: modern US location leakage
    "city in",
    "county seat",
    "nebraska",
    "georgia",
]

def looks_like_code(surface: str) -> bool:
    s = (surface or "").strip()
    if not s:
        return False
    if CODE_RE.fullmatch(s) and (any(ch.isdigit() for ch in s) or s.isupper()):
        return True
    return False

def bad_domain(label: str, desc: str) -> Tuple[bool, str]:
    blob = f"{label or ''} {desc or ''}".lower()
    for sub in BAD_DESC_SUBSTRINGS:
        if sub in blob:
            return True, sub
    return False, ""

def is_place_like(surface: str, pleiades_names: Set[str]) -> bool:
    if not surface:
        return False
    k1 = norm_key(surface)
    k2 = norm_key(strip_possessive(surface))
    return (k1 in pleiades_names) or (k2 in pleiades_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/extractions_wikidata_ollama.jsonl")
    ap.add_argument("--out_clean", default="data/extractions_wikidata_ollama_clean.jsonl")
    ap.add_argument("--out_rejected", default="data/extractions_wikidata_ollama_rejected.jsonl")
    ap.add_argument("--pleiades_index", default="data/pleiades_index.jsonl")

    ap.add_argument(
        "--code_whitelist",
        default="LRA1,LRA2,LRA3,LRA4,LRA5,LRA6,LRA7,LRA8,LRA9,LRA10,LRA11,LRA12,LRA13,LRA14",
        help="Comma-separated codes you want to keep as LOCAL concepts (still removed from Wikidata entities).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_clean = Path(args.out_clean)
    out_rejected = Path(args.out_rejected)
    pleiades_index = Path(args.pleiades_index)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    whitelist_codes = {c.strip().upper() for c in (args.code_whitelist or "").split(",") if c.strip()}
    pleiades_names = load_pleiades_name_set(pleiades_index)

    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_rejected.parent.mkdir(parents=True, exist_ok=True)

    rows_in = 0
    rows_out = 0
    ents_in = 0
    ents_kept = 0
    ents_rejected = 0
    rej_place = 0
    rej_code = 0
    rej_bad_domain = 0
    rows_dropped_empty = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_clean.open("w", encoding="utf-8") as fclean, \
         out_rejected.open("w", encoding="utf-8") as frej:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            rows_in += 1
            obj = json.loads(line)

            entities = obj.get("entities") or []
            if not isinstance(entities, list):
                entities = []

            ents_in += len(entities)
            kept: List[Dict[str, Any]] = []

            for e in entities:
                if not isinstance(e, dict):
                    ents_rejected += 1
                    frej.write(json.dumps({
                        "chunkId": obj.get("chunkId"),
                        "articleId": obj.get("articleId"),
                        "seq": obj.get("seq"),
                        "surface": str(e),
                        "qid": None,
                        "label": None,
                        "description": None,
                        "type": None,
                        "reasons": ["malformed_entity_item"],
                    }, ensure_ascii=False) + "\n")
                    continue

                surface = str(e.get("surface") or "").strip()
                qid = e.get("qid")
                label = str(e.get("label") or "").strip()
                desc = str(e.get("description") or "").strip()
                etype = str(e.get("type") or "").strip()

                reasons: List[str] = []

                if is_place_like(surface, pleiades_names):
                    reasons.append("place_blocklist_pleiades")
                    rej_place += 1

                if looks_like_code(surface):
                    if surface.upper() in whitelist_codes:
                        reasons.append("code_whitelisted_keep_local_concept")
                    else:
                        reasons.append("code_drop_unwhitelisted")
                    rej_code += 1

                bad, hit = bad_domain(label, desc)
                if bad:
                    reasons.append(f"bad_domain:{hit}")
                    rej_bad_domain += 1

                if reasons:
                    ents_rejected += 1
                    frej.write(json.dumps({
                        "chunkId": obj.get("chunkId"),
                        "articleId": obj.get("articleId"),
                        "seq": obj.get("seq"),
                        "surface": surface,
                        "qid": qid,
                        "label": label,
                        "description": desc,
                        "type": etype,
                        "reasons": reasons,
                    }, ensure_ascii=False) + "\n")
                    continue

                kept.append(e)
                ents_kept += 1

            if not kept:
                rows_dropped_empty += 1
                continue
            
            obj["entities"] = kept
            fclean.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows_out += 1

    print("[OK] Cleaning finished.")
    print(f"Rows in:            {rows_in}")
    print(f"Rows out:           {rows_out}")
    print(f"Entities in:        {ents_in}")
    print(f"Entities kept:      {ents_kept}")
    print(f"Entities rejected:  {ents_rejected}")
    print("Rejection breakdown (not mutually exclusive):")
    print(f"  place_blocklist:  {rej_place}")
    print(f"  code:             {rej_code}")
    print(f"  bad_domain:       {rej_bad_domain}")
    print(f"Clean file:         {out_clean.resolve()}")
    print(f"Rejected file:      {out_rejected.resolve()}")
    print(f"Rows dropped (empty entities): {rows_dropped_empty}")


if __name__ == "__main__":
    main()