#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama Wikidata entity extraction over local chunk files.

Input:
  data/chunks_clean/*.jsonl
  each line: {articleId, chunkId, seq, text}

Output:
  data/extractions_wikidata_ollama.jsonl   (one line per chunk)
  data/extractions_wikidata_ollama_sample.json (pretty sample)

What it extracts:
  Non-place named entities suitable for Wikidata linking:
    PERSON, ORG, WORK, EVENT, CONCEPT, OTHER
  (Places should be handled by your Pleiades pipeline.)

Run (quick test):
  set OLLAMA_BASE_URL=http://localhost:11434
  set CHAT_MODEL=llama3.2:3b
  python ollama_extract_wikidata_from_chunks.py --limit 40 --verbose

Resume (append and skip already processed chunkIds):
  python ollama_extract_wikidata_from_chunks.py --limit 0 --resume

Notes:
- Links via Wikidata wbsearchentities API.
- Uses a proper User-Agent and default throttle (250ms). Tune with --wd_sleep_ms.
- If Wikidata returns 403 in your environment, try increasing --wd_sleep_ms,
  or run with --extract_only to at least get surfaces + types.
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

import requests


# -------------------------
# Normalization / JSON salvage
# -------------------------

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
# Wikidata client (search + throttling + exact-match shortcut)
# -------------------------

class WikidataClient:
    def __init__(
        self,
        sleep_ms: int = 250,
        language: str = "en",
        limit: int = 8,
        user_agent: str = "GraphRAG-WikidataLinker/1.0 (contact: you@example.org)",
        timeout_s: int = 30,
    ):
        self.url = "https://www.wikidata.org/w/api.php"
        self.sleep_ms = max(0, int(sleep_ms))
        self.language = language
        self.limit = int(limit)
        self.timeout_s = timeout_s
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
        self.session = requests.Session()

    def search(self, term: str) -> List[Dict[str, Any]]:
        t = (term or "").strip()
        if not t:
            return []
        params = {
            "action": "wbsearchentities",
            "search": t,
            "language": self.language,
            "format": "json",
            "type": "item",
            "limit": self.limit,
        }
        r = self.session.get(self.url, params=params, headers=self.headers, timeout=self.timeout_s)
        # if something is blocked, raise with info
        r.raise_for_status()
        data = r.json()
        hits = data.get("search", []) or []
        out = []
        for h in hits:
            out.append({
                "qid": h.get("id"),
                "label": (h.get("label") or "").strip(),
                "description": (h.get("description") or "").strip(),
                "url": h.get("concepturi") or (("https://www.wikidata.org/entity/" + h.get("id")) if h.get("id") else ""),
                "aliases": h.get("aliases") or [],
            })
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000.0)
        return out

    @staticmethod
    def exact_match_qid(surface: str, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Return QID if candidate label/alias matches surface exactly (case-insensitive)."""
        s = (surface or "").strip().lower()
        if not s:
            return None
        for c in candidates:
            if (c.get("label") or "").strip().lower() == s:
                return c.get("qid")
            for a in c.get("aliases") or []:
                if isinstance(a, str) and a.strip().lower() == s:
                    return c.get("qid")
        return None


# -------------------------
# Prompts
# -------------------------

# Keep it short; llama3.2:3b tends to behave better with concise constraints.
EXTRACT_SYSTEM = (
    "You are an information extraction engine.\n"
    "Task: extract named entities that should have a Wikidata item.\n"
    "IMPORTANT:\n"
    "- Extract ONLY entities that appear verbatim in the text.\n"
    "- DO NOT extract geographic places (cities/regions/rivers/countries). Places are handled elsewhere.\n"
    "- Prefer: PERSON, ORG, WORK, EVENT, CONCEPT.\n"
    "- Avoid generic terms (e.g., 'method', 'period', 'temple', 'system').\n"
    "- Avoid bibliographic noise (publisher cities, 'Paris, 2006', journal titles only if clearly WORK).\n"
    "- If uncertain, omit.\n\n"
    "Return JSON only with schema:\n"
    "{ \"entities\": [ {\"surface\":\"...\",\"type\":\"PERSON|ORG|WORK|EVENT|CONCEPT|OTHER\",\"confidence\":0.0} ] }\n"
)

DISAMBIG_SYSTEM = (
    "You are an entity linking assistant.\n"
    "Task: choose the single best Wikidata item for the mention in context.\n"
    "Rules:\n"
    "- Choose only if clearly correct for the context.\n"
    "- If none match, return null.\n"
    "- Output JSON only: {\"chosen_qid\":\"Q...\" | null}\n"
)

ALLOWED_TYPES = {"PERSON", "ORG", "WORK", "EVENT", "CONCEPT", "OTHER"}

# surfaces that are almost always junk entities in extraction output
STOP_SURFACES = {
    "this","that","these","those","here","there","same",
    "one","two","three","four","five","may","might","must","shall","will","would","could","should",
    "method","methods","system","period","terms","term","table","figure","chapter","section",
}

# quick “looks like place/citation line” patterns to exclude even if LLM emits them
CITATION_CITY_YEAR = re.compile(r"\b[A-Z][a-z]+\s*,\s*(?:1[6-9]\d{2}|20\d{2})\b")


def extract_entities(ollama: OllamaClient, chunk_text: str) -> Dict[str, Any]:
    user = f"TEXT:\n{chunk_text}\n\nReturn JSON."
    return ollama.chat_json(system=EXTRACT_SYSTEM, user=user, temperature=0.0)


def choose_wikidata_qid(
    ollama: OllamaClient,
    chunk_text: str,
    surface: str,
    etype: str,
    candidates: List[Dict[str, Any]],
) -> Optional[str]:
    # keep prompt compact
    lines = []
    for c in candidates[:10]:
        lines.append(f"- {c.get('qid')}: {c.get('label')} — {c.get('description')}")
    cand_block = "\n".join(lines)

    user = (
        f"MENTION: {surface!r}\n"
        f"TYPE_HINT: {etype}\n\n"
        f"CONTEXT:\n{chunk_text}\n\n"
        f"CANDIDATES:\n{cand_block}\n\n"
        "Return JSON."
    )
    obj = ollama.chat_json(system=DISAMBIG_SYSTEM, user=user, temperature=0.0)
    qid = obj.get("chosen_qid", None)
    if isinstance(qid, str) and qid.strip().startswith("Q"):
        return qid.strip()
    return None


# -------------------------
# Read chunks JSONL
# -------------------------

def iter_chunk_rows(chunks_dir: Path):
    """Yields (articleId, chunkId, seq, text) from *.jsonl files in chunks_dir."""
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
# Resume support
# -------------------------

def load_seen_chunk_ids(out_path: Path) -> set:
    seen = set()
    if not out_path.exists():
        return seen
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("chunkId")
                if cid:
                    seen.add(str(cid))
            except Exception:
                continue
    return seen


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", default="data/chunks_clean", help="Directory with cleaned *.jsonl chunk files")
    ap.add_argument("--out", default="data/extractions_wikidata_ollama.jsonl", help="Output JSONL (one line per chunk)")
    ap.add_argument("--sample_out", default="data/extractions_wikidata_ollama_sample.json", help="Pretty sample JSON")
    ap.add_argument("--limit", type=int, default=40, help="Process first N chunks (0 = all). Default 40.")
    ap.add_argument("--resume", action="store_true", default=False, help="Append + skip chunkIds already present in --out")
    ap.add_argument("--sleep_ms", type=int, default=0, help="Optional sleep between chunks")

    ap.add_argument("--ollama_url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--model", default=os.getenv("CHAT_MODEL", "llama3.2:3b"))

    ap.add_argument("--wd_language", default="en")
    ap.add_argument("--wd_candidates", type=int, default=8)
    ap.add_argument("--wd_sleep_ms", type=int, default=250)
    ap.add_argument("--wd_user_agent", default=os.getenv(
        "WIKIDATA_UA",
        "GraphRAG-WikidataLinker/1.0 (contact: federico@localhost)"
    ))

    ap.add_argument("--extract_only", action="store_true", default=False,
                    help="Do NOT call Wikidata; only output extracted surfaces/types (qid=null).")

    ap.add_argument("--disambiguate", action="store_true", default=True,
                    help="Use Ollama to choose among multiple Wikidata candidates (default True).")
    ap.add_argument("--no-disambiguate", dest="disambiguate", action="store_false")

    ap.add_argument("--max_entities", type=int, default=12, help="Max extracted entities per chunk to process/link.")
    ap.add_argument("--verbose", action="store_true", default=False, help="Print extracted+linked entities per chunk.")
    ap.add_argument("--print_every", type=int, default=1, help="Print progress every N chunks (default 1).")

    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_path = Path(args.out)
    sample_path = Path(args.sample_out)

    print(f"[INFO] Chunks dir: {chunks_dir}")
    print(f"[INFO] Using Ollama model: {args.model} @ {args.ollama_url}")
    print(f"[INFO] Wikidata: candidates={args.wd_candidates} lang={args.wd_language} sleep_ms={args.wd_sleep_ms} extract_only={args.extract_only}")

    ollama = OllamaClient(base_url=args.ollama_url, model=args.model)
    wd = WikidataClient(
        sleep_ms=args.wd_sleep_ms,
        language=args.wd_language,
        limit=args.wd_candidates,
        user_agent=args.wd_user_agent,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and out_path.exists()) else "w"

    seen_chunk_ids = load_seen_chunk_ids(out_path) if (args.resume and out_path.exists()) else set()
    if args.resume:
        print(f"[INFO] Resume enabled. Already have {len(seen_chunk_ids)} chunkIds in output; will skip them.")

    processed = 0
    skipped_seen = 0
    extracted_raw = 0
    linked_ok = 0
    unlinked = 0
    filtered = 0
    failed_extract = 0
    failed_wd = 0

    sample: List[Dict[str, Any]] = []

    with out_path.open(mode, encoding="utf-8") as out_f:
        for article_id, chunk_id, seq, text in iter_chunk_rows(chunks_dir):
            if args.limit and processed >= args.limit:
                break

            if args.resume and chunk_id in seen_chunk_ids:
                skipped_seen += 1
                continue

            processed += 1

            # 1) Extract entities with LLM
            try:
                extracted = extract_entities(ollama, text)
            except Exception as e:
                failed_extract += 1
                row = {
                    "chunkId": chunk_id,
                    "articleId": article_id,
                    "seq": seq,
                    "error": f"extraction_failed: {e}",
                    "entities": [],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            ents = extracted.get("entities", []) or []
            if not isinstance(ents, list):
                ents = []

            # keep limited number to avoid slowdowns on long chunks
            ents = ents[: max(0, int(args.max_entities))]
            extracted_raw += len(ents)

            out_entities: List[Dict[str, Any]] = []
            out_unlinked: List[Dict[str, Any]] = []
            seen_surface = set()

            for it in ents:
                surface = str(it.get("surface", "")).strip()
                etype = str(it.get("type", "OTHER")).strip().upper()
                conf = float(it.get("confidence", 0.5) or 0.5)

                if not surface or surface in seen_surface:
                    continue
                seen_surface.add(surface)

                # strict verbatim
                if surface not in text:
                    filtered += 1
                    continue

                # type normalization
                if etype not in ALLOWED_TYPES:
                    etype = "OTHER"

                # filters
                k = norm_key(surface)
                if not k or k in STOP_SURFACES or len(surface) < 3:
                    filtered += 1
                    continue
                if CITATION_CITY_YEAR.search(text[max(0, text.find(surface)-60): text.find(surface)+len(surface)+60]):
                    # helps block "Paris, 2006" type noise if it leaks through even though we exclude places
                    filtered += 1
                    continue

                if args.extract_only:
                    out_unlinked.append({"surface": surface, "type": etype, "confidence": conf, "reason": "extract_only"})
                    unlinked += 1
                    continue

                # 2) Wikidata search candidates
                try:
                    cands = wd.search(surface)
                except Exception as e:
                    failed_wd += 1
                    out_unlinked.append({"surface": surface, "type": etype, "confidence": conf, "reason": f"wikidata_error:{e}"})
                    unlinked += 1
                    continue

                if not cands:
                    out_unlinked.append({"surface": surface, "type": etype, "confidence": conf, "reason": "no_candidates"})
                    unlinked += 1
                    continue

                # 3) Exact-match shortcut (label/alias == surface)
                chosen_qid = wd.exact_match_qid(surface, cands)
                method = "wbsearch_exact"

                # 4) Disambiguation if needed
                if not chosen_qid and args.disambiguate:
                    chosen_qid = choose_wikidata_qid(ollama, text, surface, etype, cands)
                    method = "wbsearch+ollama_disambig" if chosen_qid else "wbsearch_unresolved"

                if not chosen_qid:
                    out_unlinked.append({"surface": surface, "type": etype, "confidence": conf, "reason": "unresolved"})
                    unlinked += 1
                    continue

                # find chosen candidate info
                chosen = next((c for c in cands if c.get("qid") == chosen_qid), None)
                out_entities.append({
                    "surface": surface,
                    "type": etype,
                    "confidence": conf,
                    "qid": chosen_qid,
                    "label": (chosen.get("label") if chosen else ""),
                    "description": (chosen.get("description") if chosen else ""),
                    "uri": (chosen.get("url") if chosen else f"https://www.wikidata.org/entity/{chosen_qid}"),
                    "method": method,
                })
                linked_ok += 1

            row = {
                "chunkId": chunk_id,
                "articleId": article_id,
                "seq": seq,
                "entities": out_entities,
                "unlinked": out_unlinked,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if len(sample) < 12:
                sample.append({
                    "chunkId": chunk_id,
                    "snippet": text[:280],
                    "entities": out_entities,
                    "unlinked": out_unlinked[:8],
                })

            # Progress printing
            if processed % max(1, args.print_every) == 0:
                print(
                    f"[PROGRESS] processed={processed} | "
                    f"raw={extracted_raw} | linked={linked_ok} | unlinked={unlinked} | "
                    f"filtered={filtered} | extract_fail={failed_extract} | wd_fail={failed_wd} | skipped_seen={skipped_seen}"
                )

            if args.verbose:
                if out_entities:
                    print(f"  chunk={chunk_id}  linked_entities={len(out_entities)}")
                    for e in out_entities[:10]:
                        print(f"    - {e['surface']} [{e['type']}] -> {e['qid']} {e.get('label','')}")
                else:
                    print(f"  chunk={chunk_id}  linked_entities=0")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Done.")
    print(f"Chunks processed:    {processed}")
    print(f"Skipped (resume):    {skipped_seen}")
    print(f"Raw extracted ents:  {extracted_raw}")
    print(f"Linked ents:         {linked_ok}")
    print(f"Unlinked ents:       {unlinked}")
    print(f"Filtered:            {filtered}")
    print(f"Extraction failures: {failed_extract}")
    print(f"Wikidata failures:   {failed_wd}")
    print(f"Output JSONL:        {out_path.resolve()}")
    print(f"Sample JSON:         {sample_path.resolve()}")


if __name__ == "__main__":
    main()
