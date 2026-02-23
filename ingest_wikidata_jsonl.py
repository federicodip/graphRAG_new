#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest cleaned Wikidata extraction JSONL into Neo4j (NO APOC).

Reads Neo4j connection defaults from .env:
  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

Input JSONL (one line per chunk), expected keys:
  chunkId, articleId, seq,
  entities: [{
    surface, type, confidence, qid,
    label, description, uri, method,
    OPTIONAL: instanceOfQids (list[str]), lat (float), lon (float)
  }, ...]

Creates / merges:
  (:WikidataEntity {qid})
  (:Chunk {chunkId})-[:MENTIONS]->(:WikidataEntity)

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


QID_RE = re.compile(r"^Q\d+$")


CYPHER_ENSURE_CONSTRAINTS = [
    "CREATE CONSTRAINT wikidata_qid_unique IF NOT EXISTS FOR (w:WikidataEntity) REQUIRE w.qid IS UNIQUE",
    "CREATE CONSTRAINT chunk_chunkId_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
]

CYPHER_INGEST_BATCH = """
UNWIND $rows AS row
MATCH (c:Chunk {chunkId: row.chunkId})

MERGE (w:WikidataEntity {qid: row.qid})
  ON CREATE SET w.createdAt = datetime()

SET w.uri = coalesce(row.uri, w.uri),
    w.label = coalesce(row.label, w.label),
    w.description = coalesce(row.description, w.description),
    w.lastSeenAt = datetime()

MERGE (c)-[r:MENTIONS]->(w)
  ON CREATE SET r.createdAt = datetime()

SET r.surface = row.surface,
    r.mentionType = row.type,
    r.confidence = row.confidence,
    r.method = row.method,
    r.method_tag = $method_tag,
    r.lastSeenAt = datetime()

RETURN count(*) AS merged
"""

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


def _normalize_wd_uri(qid: str, uri: str) -> str:
    uri = (uri or "").strip()
    if not uri:
        return f"https://www.wikidata.org/entity/{qid}"
    # normalize http -> https and wikidata.org host
    uri = uri.replace("http://www.wikidata.org/entity/", "https://www.wikidata.org/entity/")
    uri = uri.replace("http://wikidata.org/entity/", "https://www.wikidata.org/entity/")
    return uri


def iter_rows(
    jsonl_path: Path,
    max_entities_per_chunk: int = 999,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Flatten chunk-level JSONL into per-entity rows.
    Returns (rows, stats).
    """
    rows: List[Dict[str, Any]] = []
    stats = {
        "lines": 0,
        "chunks_with_entities": 0,
        "entities_seen": 0,
        "entities_kept": 0,
        "bad_qid": 0,
        "missing_label": 0,
        "missing_description": 0,
        "missing_uri": 0,
        "optional_instanceOfQids": 0,
        "optional_coords": 0,
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

            entities = obj.get("entities") or []
            if not isinstance(entities, list) or not entities:
                continue
            stats["chunks_with_entities"] += 1

            entities = entities[:max_entities_per_chunk]

            for e in entities:
                if not isinstance(e, dict):
                    continue
                stats["entities_seen"] += 1

                qid = str(e.get("qid", "") or "").strip()
                surface = str(e.get("surface", "") or "").strip()
                if not qid or not surface:
                    continue

                if not QID_RE.match(qid):
                    stats["bad_qid"] += 1
                    continue

                label = str(e.get("label", "") or "").strip()
                desc = str(e.get("description", "") or "").strip()
                uri = str(e.get("uri", "") or "").strip()

                if not label:
                    stats["missing_label"] += 1
                if not desc:
                    stats["missing_description"] += 1
                if not uri:
                    stats["missing_uri"] += 1

                # Optional extras
                inst = e.get("instanceOfQids", None)
                if isinstance(inst, list) and inst:
                    stats["optional_instanceOfQids"] += 1
                else:
                    inst = None

                lat = e.get("lat", None)
                lon = e.get("lon", None)
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    stats["optional_coords"] += 1
                else:
                    lat, lon = None, None

                uri = _normalize_wd_uri(qid, uri)

                rows.append({
                    "chunkId": chunk_id,
                    "qid": qid,
                    "surface": surface,
                    "type": str(e.get("type", "") or "OTHER"),
                    "confidence": float(e.get("confidence", 0.5) or 0.5),
                    "label": label or None,
                    "description": desc or None,
                    "uri": uri or None,
                    "method": str(e.get("method", "") or "unknown"),
                    "instanceOfQids": inst,  # None if not present
                    "lat": lat,
                    "lon": lon,
                })
                stats["entities_kept"] += 1

    return rows, stats


def batched(lst: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", dest="env_path", default=".env", help="Path to .env (default ./.env)")
    ap.add_argument("--show_env", action="store_true", default=True)
    ap.add_argument("--no_show_env", dest="show_env", action="store_false")

    ap.add_argument("--in", dest="in_path", default="data/extractions_wikidata_ollama_clean.jsonl")

    # Neo4j args default from env
    ap.add_argument("--uri", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)  # not required anymore
    ap.add_argument("--db", default=None)

    ap.add_argument("--batch", type=int, default=1000, help="entities per transaction")
    ap.add_argument("--method_tag", default="wikidata_import_v1", help="tag stored on relationships for easy cleanup")
    ap.add_argument("--ensure_constraints", action="store_true", default=True)
    ap.add_argument("--no_ensure_constraints", dest="ensure_constraints", action="store_false")
    ap.add_argument("--validate_only", action="store_true", default=False, help="Read + validate JSONL, do not write to Neo4j")

    args = ap.parse_args()

    # Load .env into process env (12-factor style)
    load_dotenv(dotenv_path=args.env_path, override=False)

    # Apply defaults from env (but allow CLI override)
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

    if not neo4j_password:
        raise SystemExit("Missing Neo4j password: set NEO4J_PASSWORD in .env or pass --password")

    jsonl_path = Path(args.in_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input not found: {jsonl_path}")

    print(f"[INFO] Reading JSONL: {jsonl_path}")
    rows, stats = iter_rows(jsonl_path)
    print(f"[INFO] Entity rows to ingest: {len(rows)}")
    print("[INFO] JSONL stats:", stats)

    if args.validate_only:
        print("[OK] validate_only enabled — not writing to Neo4j.")
        return

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    # Optional connectivity check
    driver.verify_connectivity()

    with driver.session(database=neo4j_db) as session:
        if args.ensure_constraints:
            for q in CYPHER_ENSURE_CONSTRAINTS:
                session.run(q)

        # Count missing chunks (diagnostic)
        chunk_ids = sorted({r["chunkId"] for r in rows})
        missing = session.run(CYPHER_COUNT_MISSING_CHUNKS, chunkIds=chunk_ids).single()["missing"]
        if missing:
            print(f"[WARN] Missing Chunk nodes for {missing} chunkIds (those relationships will not be created).")

        total_merged = 0
        batches = 0

        for batch_rows in batched(rows, args.batch):
            batches += 1
            res = session.run(CYPHER_INGEST_BATCH, rows=batch_rows, method_tag=args.method_tag)
            merged = res.single()["merged"]
            total_merged += merged

            if batches % 10 == 0:
                print(f"[PROGRESS] batches={batches} | merged_rows={total_merged}")

        print("\n[OK] Done.")
        print(f"Batches:          {batches}")
        print(f"Merged rows:      {total_merged}")
        print(f"Relationship tag: {args.method_tag}")

    driver.close()


if __name__ == "__main__":
    main()