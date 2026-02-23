#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest Pleiades place extraction JSONL into Neo4j (NO APOC).

Reads Neo4j connection defaults from .env:
  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

Input JSONL (one line per chunk), expected keys:
  chunkId, articleId, seq,
  places: [{
    surface, confidence, pleiadesId, title, uri, method
  }, ...]

Creates / merges:
  (:Place {pleiadesId})
  (:Chunk {chunkId})-[:MENTIONS]->(:Place)

Notes:
- Uses MERGE, safe to re-run.
- Skips rows where the Chunk node does not exist (counts them).
- Adds r.method_tag so you can delete just this import later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase


PID_RE = re.compile(r"^\d+$")
RELTYPE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


CYPHER_ENSURE_CONSTRAINTS = [
    "CREATE CONSTRAINT place_pid_unique IF NOT EXISTS FOR (p:Place) REQUIRE p.pleiadesId IS UNIQUE",
    "CREATE CONSTRAINT chunk_chunkId_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
]

CYPHER_COUNT_MISSING_CHUNKS = """
UNWIND $chunkIds AS cid
OPTIONAL MATCH (c:Chunk {chunkId: cid})
WITH cid, c
WHERE c IS NULL
RETURN count(*) AS missing
"""


def _mask_secret(s: Optional[str]) -> str:
    if not s:
        return "(missing)"
    if len(s) <= 4:
        return "***"
    return s[:2] + "***" + s[-2:]


def _normalize_pleiades_uri(pid: str, uri: str) -> str:
    uri = (uri or "").strip()
    if uri:
        # normalize http -> https just in case
        uri = uri.replace("http://pleiades.stoa.org/places/", "https://pleiades.stoa.org/places/")
        return uri
    # fallback: canonical place URI
    return f"https://pleiades.stoa.org/places/{pid}"


def build_ingest_cypher(rel_type: str) -> str:
    if not RELTYPE_RE.match(rel_type):
        raise ValueError(f"Invalid relationship type: {rel_type!r}")

    # relationship type must be literal in Cypher (not a parameter), so we inject only after validation
    return f"""
UNWIND $rows AS row
MATCH (c:Chunk {{chunkId: row.chunkId}})

MERGE (p:Place {{pleiadesId: row.pleiadesId}})
  ON CREATE SET p.createdAt = datetime()

SET p.uri = coalesce(row.uri, p.uri),
    p.title = coalesce(row.title, p.title),
    p.lastSeenAt = datetime()

MERGE (c)-[r:{rel_type}]->(p)
  ON CREATE SET r.createdAt = datetime(),
                r.method_tag = $method_tag

SET r.surface = row.surface,
    r.mentionType = "PLACE",
    r.confidence = row.confidence,
    r.method = row.method,
    r.lastSeenAt = datetime()

RETURN count(*) AS merged
"""


def iter_rows(
    jsonl_path: Path,
    max_places_per_chunk: int = 999,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {
        "lines": 0,
        "chunks_with_places": 0,
        "places_seen": 0,
        "places_kept": 0,
        "bad_pid": 0,
        "missing_surface": 0,
        "missing_title": 0,
        "missing_uri": 0,
    }

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["lines"] += 1
            obj = json.loads(line)

            chunk_id = str(obj.get("chunkId", "") or "").strip()
            if not chunk_id:
                continue

            places = obj.get("places") or []
            if not isinstance(places, list) or not places:
                continue

            stats["chunks_with_places"] += 1
            places = places[:max_places_per_chunk]

            for pl in places:
                if not isinstance(pl, dict):
                    continue
                stats["places_seen"] += 1

                pid = str(pl.get("pleiadesId", "") or "").strip()
                surface = str(pl.get("surface", "") or "").strip()
                if not pid or not PID_RE.match(pid):
                    stats["bad_pid"] += 1
                    continue
                if not surface:
                    stats["missing_surface"] += 1
                    continue

                title = str(pl.get("title", "") or "").strip()
                uri = str(pl.get("uri", "") or "").strip()
                if not title:
                    stats["missing_title"] += 1
                if not uri:
                    stats["missing_uri"] += 1

                uri = _normalize_pleiades_uri(pid, uri)

                rows.append({
                    "chunkId": chunk_id,
                    "pleiadesId": pid,
                    "surface": surface,
                    "confidence": float(pl.get("confidence", 0.5) or 0.5),
                    "title": title or None,
                    "uri": uri or None,
                    "method": str(pl.get("method", "") or "unknown"),
                })
                stats["places_kept"] += 1

    return rows, stats


def batched(lst: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--env", dest="env_path", default=".env")
    ap.add_argument("--show_env", action="store_true", default=True)
    ap.add_argument("--no_show_env", dest="show_env", action="store_false")

    ap.add_argument("--in", dest="in_path", default="data/extractions_places_ollama.jsonl")

    ap.add_argument("--uri", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--db", default=None)

    ap.add_argument("--batch", type=int, default=2000, help="rows per transaction")
    ap.add_argument("--method_tag", default="pleiades_import_v1")
    ap.add_argument("--rel_type", default="MENTIONS", help="Relationship type to use (default MENTIONS)")
    ap.add_argument("--ensure_constraints", action="store_true", default=True)
    ap.add_argument("--no_ensure_constraints", dest="ensure_constraints", action="store_false")
    ap.add_argument("--validate_only", action="store_true", default=False)

    args = ap.parse_args()

    load_dotenv(dotenv_path=args.env_path, override=False)

    neo4j_uri = args.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.user or os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = args.password or os.getenv("NEO4J_PASSWORD", "")
    neo4j_db = args.db or os.getenv("NEO4J_DATABASE", "neo4j")

    if args.show_env:
        print("[ENV] Using:")
        print(f"  env file:        {Path(args.env_path).resolve()}")
        print(f"  NEO4J_URI:       {neo4j_uri}")
        print(f"  NEO4J_USERNAME:  {neo4j_user}")
        print(f"  NEO4J_PASSWORD:  {_mask_secret(neo4j_password)}")
        print(f"  NEO4J_DATABASE:  {neo4j_db}")
        print(f"  REL_TYPE:        {args.rel_type}")
        print(f"  METHOD_TAG:      {args.method_tag}")

    if not neo4j_password:
        raise SystemExit("Missing Neo4j password: set NEO4J_PASSWORD in .env or pass --password")

    jsonl_path = Path(args.in_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input not found: {jsonl_path}")

    print(f"[INFO] Reading JSONL: {jsonl_path}")
    rows, stats = iter_rows(jsonl_path)
    print(f"[INFO] Place rows to ingest: {len(rows)}")
    print("[INFO] JSONL stats:", stats)

    if args.validate_only:
        print("[OK] validate_only enabled — not writing to Neo4j.")
        return

    cypher_ingest = build_ingest_cypher(args.rel_type)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    driver.verify_connectivity()

    with driver.session(database=neo4j_db) as session:
        if args.ensure_constraints:
            for q in CYPHER_ENSURE_CONSTRAINTS:
                session.run(q)

        # diagnostic: missing chunks
        chunk_ids = sorted({r["chunkId"] for r in rows})
        missing = session.run(CYPHER_COUNT_MISSING_CHUNKS, chunkIds=chunk_ids).single()["missing"]
        if missing:
            print(f"[WARN] Missing Chunk nodes for {missing} chunkIds (those relationships will not be created).")

        total_merged = 0
        batches = 0

        for batch_rows in batched(rows, args.batch):
            batches += 1
            res = session.run(cypher_ingest, rows=batch_rows, method_tag=args.method_tag)
            merged = res.single()["merged"]
            total_merged += merged

            if batches % 10 == 0:
                print(f"[PROGRESS] batches={batches} | merged_rows={total_merged}")

        print("\n[OK] Done.")
        print(f"Batches:          {batches}")
        print(f"Merged rows:      {total_merged}")
        print(f"Relationship tag: {args.method_tag}")
        print(f"Relationship type:{args.rel_type}")

    driver.close()


if __name__ == "__main__":
    main()