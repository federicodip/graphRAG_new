#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama place extraction over local chunk files (NO Wikidata), resumable.

Input:
  data/chunks_clean/*.jsonl
  each line: {articleId, chunkId, seq, text}

Requires:
  data/pleiades_index.jsonl
  each line: {"pleiadesId":"423025","title":"Roma","uri":"...","altNames":[...]}

Output:
  data/extractions_places_ollama.jsonl  (one line per chunk)
  data/extractions_places_ollama_sample.json (pretty sample)

Typical usage:

  # Batch 1 (first 500 chunks), fresh start:
  python ollama_extract_from_chunks.py --limit 500

  # Batch 2 (next 500 new chunks), resume safely:
  python ollama_extract_from_chunks.py --limit 500 --resume

  # Run all remaining, resume:
  python ollama_extract_from_chunks.py --limit 0 --resume

Notes:
- By default it SKIPS ambiguous surfaces (surface -> >1 pleiadesId) unless --disambiguate.
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import requests


# -------------------------
# Normalization
# -------------------------

_PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$")

def fold_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_key(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u2019", "'")
    s = _PUNCT_EDGE.sub("", s)
    s = fold_ascii(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def strip_possessive(s: str) -> str:
    s = (s or "").replace("\u2019", "'").strip()
    return re.sub(r"(?:'s)$", "", s, flags=re.IGNORECASE).strip()


# -------------------------
# JSON parsing for Ollama
# -------------------------

def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # salvage first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from model output")


# -------------------------
# Ollama client
# -------------------------

class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        return extract_json_object(content)


# -------------------------
# Prompts
# -------------------------

EXTRACTION_SYSTEM = (
    "You are an information extraction engine for historical scholarship.\n"
    "Task: extract ONLY geographic PLACE mentions from the text.\n\n"
    "STRICT RULES:\n"
    "- Every extracted surface MUST appear verbatim in the input text.\n"
    "- Do NOT invent entities.\n"
    "- Extract ONLY proper place names (cities, regions, rivers, islands, provinces, etc.).\n"
    "- Ignore generic nouns unless part of a proper name (e.g., 'the temple' is NOT a place).\n"
    "- Ignore bibliographic/publication-place noise (e.g., 'Paris, 2006', publisher locations, catalogue city lines).\n"
    "- If uncertain, omit.\n\n"
    "OUTPUT JSON ONLY.You MUST return an object with a key 'places'.\n"
    "'places' MUST be a JSON array.\n"
    "Each element of 'places' MUST be an object with keys:\n"
    "  - surface (string)\n"
    "  - confidence (number 0..1)\n"
    "Never output strings inside the places array.\n Schema:\n"
    "{ \"places\": [{\"surface\": \"...\", \"confidence\": 0.0}] }\n"
)

def extract_places(ollama: OllamaClient, chunk_text: str) -> Dict[str, Any]:
    user = f"TEXT:\n{chunk_text}\n\nReturn JSON."
    return ollama.chat_json(system=EXTRACTION_SYSTEM, user=user, temperature=0.0)


def choose_pleiades_id(
    ollama: OllamaClient,
    chunk_text: str,
    place_surface: str,
    candidates: List[Dict[str, Any]],
) -> Optional[str]:
    candidates = candidates[:12]
    cand_lines = "\n".join(
        f"- {c['pleiadesId']}: {c.get('title','')} (uri={c.get('uri','')})"
        for c in candidates
    )
    system = (
        "You are an entity linking assistant.\n"
        "Task: choose the best Pleiades place ID for a PLACE mention in context.\n"
        "Rules:\n"
        "- Choose only if clearly a geographic place mention in the argument (not bibliographic noise).\n"
        "- If none match, return null.\n"
        "- Output MUST be JSON only.\n"
        "Schema: {\"chosen_pleiadesId\": \"...\" | null}\n"
    )
    user = (
        f"PLACE SURFACE (verbatim): {place_surface!r}\n\n"
        f"CONTEXT:\n{chunk_text}\n\n"
        f"CANDIDATES:\n{cand_lines}\n\n"
        "Return JSON."
    )
    obj = ollama.chat_json(system=system, user=user, temperature=0.0)
    pid = obj.get("chosen_pleiadesId", None)
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    return None


# -------------------------
# Read chunks JSONL
# -------------------------

def iter_chunk_rows(chunks_dir: Path):
    """
    Yields (articleId, chunkId, seq, text) from *.jsonl in chunks_dir.
    """
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {chunks_dir}")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                article_id = str(obj.get("articleId", "")).strip()
                chunk_id = str(obj.get("chunkId", "")).strip()
                seq = obj.get("seq", 0)
                text = str(obj.get("text", "")).strip()
                if not chunk_id or not text:
                    continue
                yield article_id, chunk_id, int(seq) if str(seq).isdigit() else 0, text


# -------------------------
# Pleiades index (file-based)
# -------------------------

def load_pleiades_index(index_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Pleiades index not found: {index_path}\n"
            f"Expected JSONL lines like: "
            f'{{"pleiadesId":"423025","title":"Roma","uri":"...","altNames":["Rome"]}}'
        )

    name_to_pids: Dict[str, List[str]] = defaultdict(list)
    pid_info: Dict[str, Dict[str, Any]] = {}

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row.get("pleiadesId", "")).strip()
            if not pid:
                continue
            title = str(row.get("title", "")).strip()
            uri = str(row.get("uri", "")).strip()
            alt = row.get("altNames", []) or []
            if not isinstance(alt, list):
                alt = []

            pid_info[pid] = {"pleiadesId": pid, "title": title, "uri": uri, "altNames": alt}

            names = []
            if title:
                names.append(title)
            for a in alt:
                if isinstance(a, str) and a.strip():
                    names.append(a.strip())

            for nm in names:
                k = norm_key(nm)
                if not k:
                    continue
                name_to_pids[k].append(pid)

    for k, lst in list(name_to_pids.items()):
        seen = set()
        dedup = []
        for pid in lst:
            if pid in seen:
                continue
            seen.add(pid)
            dedup.append(pid)
        name_to_pids[k] = dedup

    return name_to_pids, pid_info


# -------------------------
# Resume support
# -------------------------

def load_processed_chunk_ids(out_path: Path) -> set[str]:
    """
    Reads existing output JSONL and collects chunkId values.
    Safe against partial/bad lines.
    """
    done: set[str] = set()
    if not out_path.exists():
        return done

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("chunkId")
                if isinstance(cid, str) and cid:
                    done.add(cid)
            except Exception:
                # ignore corrupted/partial lines
                continue
    return done


# -------------------------
# Guardrails (cheap filters)
# -------------------------

STOP_SURFACES = {
    "this","that","these","those","here","there","well","zone","scope","same",
    "one","two","three","four","five","may","might","must","shall","will","would","could","should",
}

CITATION_CITY_YEAR = re.compile(r"\b[A-Z][a-z]+\s*,\s*(?:1[6-9]\d{2}|20\d{2})\b")

def looks_like_bibliographic_context(text: str, surface: str) -> bool:
    idx = text.find(surface)
    if idx == -1:
        return False
    lo = max(0, idx - 60)
    hi = min(len(text), idx + len(surface) + 60)
    window = text[lo:hi]
    return bool(CITATION_CITY_YEAR.search(window))


#-------------------------
# helper to coerce model output into expected list of place dicts
#-------------------------


def coerce_places_list(raw_places: Any) -> List[Dict[str, Any]]:
    """
    Coerce model output into a list of {"surface": str, "confidence": float}.
    Handles cases where raw_places is a string, list of strings, dict, etc.
    """
    out: List[Dict[str, Any]] = []

    if raw_places is None:
        return out

    # If the model returns a single string
    if isinstance(raw_places, str):
        s = raw_places.strip()
        if s:
            out.append({"surface": s, "confidence": 0.5})
        return out

    # If the model returns a dict (rare, but seen)
    if isinstance(raw_places, dict):
        # If it already looks like a place item
        if "surface" in raw_places or "name" in raw_places or "place" in raw_places:
            s = str(raw_places.get("surface") or raw_places.get("name") or raw_places.get("place") or "").strip()
            if s:
                out.append({"surface": s, "confidence": float(raw_places.get("confidence", 0.5) or 0.5)})
        return out

    # If the model returns a list
    if isinstance(raw_places, list):
        for it in raw_places:
            if isinstance(it, dict):
                s = str(it.get("surface") or it.get("name") or it.get("place") or "").strip()
                if not s:
                    continue
                out.append({"surface": s, "confidence": float(it.get("confidence", 0.5) or 0.5)})
            elif isinstance(it, str):
                s = it.strip()
                if s:
                    out.append({"surface": s, "confidence": 0.5})
            else:
                continue
        return out

    return out



# -------------------------
# Main pipeline
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", default="data/chunks_clean")
    ap.add_argument("--pleiades_index", default="data/pleiades_index.jsonl")
    ap.add_argument("--out", default="data/extractions_places_ollama.jsonl")
    ap.add_argument("--sample_out", default="data/extractions_places_ollama_sample.json")

    ap.add_argument("--limit", type=int, default=40, help="Process N NEW chunks (0 = all remaining).")
    ap.add_argument("--start", type=int, default=0, help="Skip first N chunks in iteration order (optional).")

    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume from existing output by skipping already processed chunkIds and appending.")
    ap.add_argument("--flush_every", type=int, default=10,
                    help="Flush output every N written chunks (default 10). Lower = safer, slower.")
    ap.add_argument("--sleep_ms", type=int, default=0)

    ap.add_argument("--ollama_url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--model", default=os.getenv("CHAT_MODEL", "llama3.2:3b"))

    ap.add_argument("--skip_ambiguous", action="store_true", default=True)
    ap.add_argument("--no-skip_ambiguous", dest="skip_ambiguous", action="store_false")

    ap.add_argument("--disambiguate", action="store_true", default=False)
    ap.add_argument("--max_disamb_cands", type=int, default=10)

    ap.add_argument("--verbose", action="store_true", default=False)
    ap.add_argument("--print_every", type=int, default=25, help="Progress print frequency (default 25).")
    ap.add_argument("--max_print_places", type=int, default=12)

    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    pleiades_index = Path(args.pleiades_index)
    out_path = Path(args.out)
    sample_path = Path(args.sample_out)

    print(f"[INFO] Loading Pleiades index: {pleiades_index}")
    name_to_pids, pid_info = load_pleiades_index(pleiades_index)
    print(f"[INFO] Pleiades places: {len(pid_info)} | name keys: {len(name_to_pids)}")
    print(f"[INFO] Using Ollama model: {args.model} @ {args.ollama_url}")

    already_done = set()
    if args.resume:
        already_done = load_processed_chunk_ids(out_path)
        print(f"[INFO] Resume enabled. Already processed chunkIds: {len(already_done)}")
    else:
        # fresh start: overwrite
        if out_path.exists():
            print(f"[WARN] Overwriting existing output (use --resume to append safely): {out_path}")

    ollama = OllamaClient(base_url=args.ollama_url, model=args.model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample: List[Dict[str, Any]] = []

    seen_total = 0
    written_new = 0
    skipped_already = 0

    extracted_raw = 0
    linked_ok = 0
    skipped_ambig = 0
    skipped_no_candidate = 0
    skipped_filters = 0
    failed = 0

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as out_f:
        for article_id, chunk_id, seq, text in iter_chunk_rows(chunks_dir):
            seen_total += 1
            if args.start and seen_total <= args.start:
                continue

            if chunk_id in already_done:
                skipped_already += 1
                continue

            # Stop after N *new* chunks (unless limit=0)
            if args.limit and written_new >= args.limit:
                break

            try:
                extracted = extract_places(ollama, text)
            except Exception as e:
                failed += 1
                row = {
                    "chunkId": chunk_id,
                    "articleId": article_id,
                    "seq": seq,
                    "error": f"extraction_failed: {e}",
                    "places": [],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_new += 1
                already_done.add(chunk_id)
                continue

            places = coerce_places_list(extracted.get("places", []))
            extracted_raw += len(places)

            place_out: List[Dict[str, Any]] = []
            seen_surface = set()

            for it in places:
                surface = str(it.get("surface", "")).strip()
                if not surface or surface in seen_surface:
                    continue
                seen_surface.add(surface)

                if surface not in text:
                    skipped_filters += 1
                    continue

                k_surface = norm_key(surface)
                if k_surface in STOP_SURFACES or len(surface) < 3:
                    skipped_filters += 1
                    continue
                if looks_like_bibliographic_context(text, surface):
                    skipped_filters += 1
                    continue

                conf = float(it.get("confidence", 0.5) or 0.5)

                k1 = norm_key(surface)
                k2 = norm_key(strip_possessive(surface))

                cand_pids: List[str] = []
                for k in [k1, k2]:
                    if k:
                        cand_pids.extend(name_to_pids.get(k, []))

                dedup = []
                seen = set()
                for pid in cand_pids:
                    if pid in seen:
                        continue
                    seen.add(pid)
                    dedup.append(pid)
                cand_pids = dedup

                if not cand_pids:
                    skipped_no_candidate += 1
                    continue

                chosen_pid: Optional[str] = None
                if len(cand_pids) == 1:
                    chosen_pid = cand_pids[0]
                else:
                    if args.skip_ambiguous and not args.disambiguate:
                        skipped_ambig += 1
                        continue
                    if args.disambiguate and len(cand_pids) <= args.max_disamb_cands:
                        cands = [pid_info[pid] for pid in cand_pids if pid in pid_info]
                        chosen_pid = choose_pleiades_id(ollama, text, surface, cands)
                    else:
                        skipped_ambig += 1
                        continue

                if chosen_pid and chosen_pid in pid_info:
                    info = pid_info[chosen_pid]
                    place_out.append(
                        {
                            "surface": surface,
                            "confidence": conf,
                            "pleiadesId": chosen_pid,
                            "title": info.get("title", ""),
                            "uri": info.get("uri", ""),
                            "method": "ollama_extract+pleiades_index",
                        }
                    )
                    linked_ok += 1

            row = {
                "chunkId": chunk_id,
                "articleId": article_id,
                "seq": seq,
                "places": place_out,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_new += 1
            already_done.add(chunk_id)

            if len(sample) < 12:
                sample.append({"chunkId": chunk_id, "snippet": text[:280], "places": place_out})

            if args.verbose:
                shown = place_out[: args.max_print_places]
                if shown:
                    print(f"  chunk={chunk_id}  linked_places={len(place_out)}")
                    for p in shown:
                        print(f"    - {p['surface']}  ->  {p['pleiadesId']}  {p.get('title','')}")
                else:
                    print(f"  chunk={chunk_id}  linked_places=0")

            if written_new % max(1, args.flush_every) == 0:
                out_f.flush()

            if written_new % max(1, args.print_every) == 0:
                print(
                    f"[PROGRESS] new_written={written_new} | seen_total={seen_total} | already_done={skipped_already} "
                    f"| raw_places={extracted_raw} | linked={linked_ok} | ambig_skipped={skipped_ambig} "
                    f"| noCand={skipped_no_candidate} | filtered={skipped_filters} | failed={failed}"
                )

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        out_f.flush()

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Done.")
    print(f"Seen total chunks (incl. skipped): {seen_total}")
    print(f"Already done skipped:              {skipped_already}")
    print(f"New chunks written this run:       {written_new}")
    print(f"Raw extracted places:              {extracted_raw}")
    print(f"Linked places:                     {linked_ok}")
    print(f"Skipped ambiguous:                 {skipped_ambig}")
    print(f"Skipped no candidate:              {skipped_no_candidate}")
    print(f"Filtered (junk/etc):               {skipped_filters}")
    print(f"Failures:                          {failed}")
    print(f"Output JSONL:                      {out_path.resolve()}")
    print(f"Sample JSON:                       {sample_path.resolve()}")


if __name__ == "__main__":
    main()
