#!/usr/bin/env python3
"""
Ingest Pleiades dump into Neo4j as (:Place).

Creates/updates:
  (:Place {pleiadesId, uri, title, altNames, reprLat, reprLon})

Supports inputs:
- directory of *.json (one record per file)
- .jsonl / .jsonl.gz (one record per line) ✅ streaming
- .json/.geojson/.json.gz/.geojson.gz     ✅ streaming via ijson (recommended)
  - including JSON-LD like:
    {"@context":..., "@graph":[ {...Place...}, {...Place...}, ... ]}

Optional conversion for big single-file dumps:
  --convert_jsonl converts input to *.jsonl.gz next to it, then ingests that.

Recommended run (no conversion needed for your file; it's already streamable):
  python ingest_pleiades.py --input data/pleiades/pleiades-places-20260216.json.gz --reset

Or with conversion:
  python ingest_pleiades.py --input data/pleiades/pleiades-places-20260216.json.gz --convert_jsonl --reset
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase

PLEIADES_URI_RE = re.compile(r"/places/(\d+)", re.IGNORECASE)

try:
    import ijson  # type: ignore
except Exception:
    ijson = None


# ---------------- I/O helpers ----------------

def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def looks_like_jsonl(path: Path, probe: int = 3) -> bool:
    """Detect NDJSON/JSONL disguised as *.json.gz."""
    got = 0
    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("{"):
                return False
            try:
                obj = json.loads(s)
            except Exception:
                return False
            if not isinstance(obj, dict):
                return False
            got += 1
            if got >= probe:
                return True
    return False


def is_place_like(rec: Dict[str, Any]) -> bool:
    """
    Your file's @graph contains Place objects (with id digits + uri /places/<id>).
    We filter to avoid accidentally ingesting nested Feature/Location-like objects.
    """
    # Common in your file
    pid = rec.get("id")
    if isinstance(pid, str) and pid.isdigit():
        return True

    uri = rec.get("uri") or rec.get("@id")
    if isinstance(uri, str) and PLEIADES_URI_RE.search(uri):
        return True

    # JSON-LD may carry @type
    t = rec.get("@type")
    if isinstance(t, str) and t.lower() == "place":
        return True

    return False


def iter_records_streaming_json(input_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream big JSON/JSON-LD via ijson.

    For your file:
      {"@graph":[ {...Place...}, {...Place...}, ... ]}
    => prefix: "@graph.item"
    """
    if ijson is None:
        raise RuntimeError("Install ijson for streaming big JSON/JSON-LD: pip install ijson")

    prefixes = [
        "@graph.item",          # ✅ your file: list of Place dicts
        "places.item",          # {"places":[...]}
        "items.item",           # {"items":[...]}
        "results.item",         # {"results":[...]}
        "item",                 # top-level array: [ {...}, ... ]
    ]

    for pref in prefixes:
        with open_text(input_path) as f:
            gen = ijson.items(f, pref)
            first = next(gen, None)
            if isinstance(first, dict):
                if is_place_like(first):
                    yield first
                for obj in gen:
                    if isinstance(obj, dict) and is_place_like(obj):
                        yield obj
                return

    raise RuntimeError(
        "Could not locate a stream of place records. "
        "Your file should work with prefix '@graph.item'."
    )


def iter_records(input_path: Path) -> Iterator[Dict[str, Any]]:
    # directory of per-place JSON files
    if input_path.is_dir():
        for p in sorted(input_path.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and is_place_like(obj):
                    yield obj
            except Exception:
                continue
        return

    suffixes = "".join(input_path.suffixes).lower()

    # jsonl (streaming)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        with open_text(input_path) as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    raise ValueError(f"Bad JSON on {input_path} line {line_no}: {e}") from e
                if isinstance(obj, dict) and is_place_like(obj):
                    yield obj
        return

    # disguised jsonl
    if looks_like_jsonl(input_path):
        with open_text(input_path) as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    raise ValueError(f"Bad JSON on {input_path} line {line_no}: {e}") from e
                if isinstance(obj, dict) and is_place_like(obj):
                    yield obj
        return

    # big JSON/JSON-LD streaming
    yield from iter_records_streaming_json(input_path)


# ---------------- Extraction logic ----------------

def extract_pleiades_id(rec: Dict[str, Any]) -> Optional[str]:
    # direct id fields
    for key in ("pleiadesId", "id", "uid", "place_id"):
        v = rec.get(key)
        if isinstance(v, (int, float)) and int(v) > 0:
            return str(int(v))
        if isinstance(v, str) and v.strip().isdigit():
            return v.strip()

    # uri-ish fields
    for key in ("uri", "url", "@id", "id_uri", "path", "link"):
        v = rec.get(key)
        if isinstance(v, str):
            m = PLEIADES_URI_RE.search(v)
            if m:
                return m.group(1)

    # properties might hold it (less likely for @graph places, but keep)
    props = rec.get("properties")
    if isinstance(props, dict):
        for key in ("id", "uid", "pleiadesId", "@id", "uri", "url", "link"):
            v = props.get(key)
            if isinstance(v, str):
                m = PLEIADES_URI_RE.search(v)
                if m:
                    return m.group(1)
            if isinstance(v, (int, float)) and int(v) > 0:
                return str(int(v))

    return None


def extract_title_and_names(rec: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    title = None
    alt: List[str] = []

    # JSON-LD place has title at top-level
    if not title:
        title = rec.get("title") or rec.get("name") or rec.get("reprName")

    names = rec.get("names")
    if isinstance(names, list):
        for n in names:
            if isinstance(n, dict):
                v = n.get("name") or n.get("title") or n.get("romanized")
                if isinstance(v, str):
                    alt.append(v)
            elif isinstance(n, str):
                alt.append(n)

    alt_names = rec.get("altNames") or rec.get("alternate_names") or rec.get("alternative_names")
    if isinstance(alt_names, list):
        for v in alt_names:
            if isinstance(v, str):
                alt.append(v)

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    alt2: List[str] = []
    seen = set()
    for s in alt:
        s2 = norm(s)
        if not s2:
            continue
        k = s2.casefold()
        if k not in seen:
            seen.add(k)
            alt2.append(s2)

    if isinstance(title, str):
        title = re.sub(r"\s+", " ", title).strip() or None

    return title, alt2


def extract_reprpoint(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # returns (lat, lon)
    for key in ("reprPoint", "representative_point", "reprpoint"):
        rp = rec.get(key)
        if isinstance(rp, (list, tuple)) and len(rp) >= 2:
            lon, lat = rp[0], rp[1]  # pleiades uses [lon, lat]
            try:
                return float(lat), float(lon)
            except Exception:
                pass

    return None, None


# ---------------- Conversion helpers ----------------

def json_default(o: Any):
    # ijson can produce Decimal values
    if isinstance(o, Decimal):
        if o == o.to_integral_value():
            return int(o)
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def make_jsonl_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".json") or name.endswith(".geojson"):
        name = name.rsplit(".", 1)[0]
    return input_path.parent / f"{name}.jsonl.gz"


def convert_to_jsonl_gz(input_path: Path, out_path: Path) -> int:
    print(f"[convert] writing streaming JSONL to: {out_path.name}")
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for rec in iter_records(input_path):
            out.write(json.dumps(rec, ensure_ascii=False, default=json_default))
            out.write("\n")
            n += 1
            if n % 50000 == 0:
                print(f"[convert] records_written={n}")
    print(f"[convert] done. records_written={n}")
    return n


# ---------------- Main ----------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pleiades dump path (file/dir; can be .gz)")
    parser.add_argument("--reset", action="store_true", help="Delete existing Place nodes first")
    parser.add_argument("--convert_jsonl", action="store_true", help="Convert input to .jsonl.gz first")
    parser.add_argument("--force_convert", action="store_true", help="Overwrite existing converted .jsonl.gz")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=5000)
    args = parser.parse_args()

    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise ValueError("NEO4J_PASSWORD is not set")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    suffixes = "".join(input_path.suffixes).lower()
    if args.convert_jsonl and not input_path.is_dir() and not (suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz")):
        out_path = make_jsonl_path(input_path)
        if out_path.exists() and not args.force_convert:
            print(f"[convert] using existing: {out_path.name}")
        else:
            n = convert_to_jsonl_gz(input_path, out_path)
            if n == 0:
                raise RuntimeError("Conversion produced 0 records. Aborting.")
        input_path = out_path
        print(f"[convert] ingesting from: {input_path.name}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    q_reset = "MATCH (p:Place) DETACH DELETE p"

    q_constraints = [
        "CREATE CONSTRAINT place_id IF NOT EXISTS FOR (p:Place) REQUIRE p.pleiadesId IS UNIQUE",
        "CREATE INDEX place_title IF NOT EXISTS FOR (p:Place) ON (p.title)",
    ]

    q_upsert = """
    UNWIND $rows AS row
    MERGE (p:Place {pleiadesId: row.pleiadesId})
    SET p.uri      = row.uri,
        p.title    = row.title,
        p.altNames = row.altNames,
        p.reprLat  = row.reprLat,
        p.reprLon  = row.reprLon
    RETURN count(*) AS n
    """

    with driver.session(database=database) as session:
        for q in q_constraints:
            session.execute_write(lambda tx, qq=q: tx.run(qq).consume())

        if args.reset:
            print("[reset] deleting Place nodes...")
            session.execute_write(lambda tx: tx.run(q_reset).consume())
            print("[reset] done")

        rows: List[Dict[str, Any]] = []
        n_ing = 0
        n_skip = 0

        for rec in iter_records(input_path):
            pid = extract_pleiades_id(rec)
            if not pid:
                n_skip += 1
                continue

            title, alt = extract_title_and_names(rec)
            lat, lon = extract_reprpoint(rec)

            rows.append(
                {
                    "pleiadesId": pid,
                    "uri": f"https://pleiades.stoa.org/places/{pid}",
                    "title": title,
                    "altNames": alt,
                    "reprLat": lat,
                    "reprLon": lon,
                }
            )

            if len(rows) >= args.batch_size:
                rec2 = session.execute_write(lambda tx: tx.run(q_upsert, rows=rows).single())
                n_ing += int(rec2["n"]) if rec2 is not None else len(rows)
                rows.clear()

                if n_ing and n_ing % args.print_every == 0:
                    print(f"[progress] places_upserted={n_ing} skipped={n_skip}")

        if rows:
            rec2 = session.execute_write(lambda tx: tx.run(q_upsert, rows=rows).single())
            n_ing += int(rec2["n"]) if rec2 is not None else len(rows)

        print(f"\nDone. Places upserted={n_ing} skipped_records={n_skip}")

    driver.close()


if __name__ == "__main__":
    main()
