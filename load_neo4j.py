#!/usr/bin/env python3
"""
Load ISAW articles + chunks into Neo4j.

Creates:
  (:Article {articleId, title, year, journal, url, authors})
  (:Chunk {chunkId, articleId, seq, text, parentChunkId?, subseq?, file?})
  (:Article)-[:HAS_CHUNK]->(:Chunk)

Optional:
  (:Chunk)-[:NEXT]->(:Chunk) within each article (by seq)
  (:Chunk)-[:SUBCHUNK_OF]->(:Chunk) for subchunks (parentChunkId)

Run:
  python load_neo4j.py --reset --chunks_dir data/chunks_clean --metadata_dir data/articles_metadata
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON on {path} line {line_no}: {e}") from e


def as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    return None


def load_article_metadata(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not metadata_dir.exists():
        return meta_map

    for p in sorted(metadata_dir.glob("*.json")):
        try:
            obj = read_json(p)
            aid = str(obj.get("articleId", "")).strip()
            if not aid:
                continue
            meta_map[aid] = obj
        except Exception:
            continue
    return meta_map


def count_lines_fast(path: Path) -> int:
    # fast-ish line count without parsing JSON
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def neo4j_execute_write(session, query: str, params: Dict[str, Any]) -> int:
    def _tx(tx):
        res = tx.run(query, params)
        rec = res.single()
        return int(rec["n"]) if rec and "n" in rec else 0

    return session.execute_write(_tx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete existing Article/Chunk graph content first")
    parser.add_argument("--chunks_dir", default="data/chunks_clean", help="Directory with chunk .jsonl files")
    parser.add_argument("--metadata_dir", default="data/articles_metadata", help="Directory with article metadata .json files")
    parser.add_argument("--batch_size", type=int, default=500, help="UNWIND batch size")
    parser.add_argument("--print_every", type=int, default=2000, help="Print progress every N ingested chunks")
    parser.add_argument("--no_next", action="store_true", help="Do NOT create :NEXT relationships")
    parser.add_argument("--no_subchunk", action="store_true", help="Do NOT create :SUBCHUNK_OF relationships")
    args = parser.parse_args()

    load_dotenv()

    repo_root = Path(__file__).resolve().parent
    chunks_dir = (repo_root / args.chunks_dir).resolve() if not Path(args.chunks_dir).is_absolute() else Path(args.chunks_dir).resolve()
    metadata_dir = (repo_root / args.metadata_dir).resolve() if not Path(args.metadata_dir).is_absolute() else Path(args.metadata_dir).resolve()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        raise ValueError("NEO4J_PASSWORD is not set in your environment/.env")

    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {chunks_dir}")

    print(f"[config] chunks_dir={chunks_dir}")
    print(f"[config] metadata_dir={metadata_dir}")
    print(f"[config] neo4j uri={uri} db={database}")
    print(f"[config] batch_size={args.batch_size} print_every={args.print_every}")
    print(f"[config] create_next={not args.no_next} create_subchunk={not args.no_subchunk}")

    # Pre-scan: total chunk count + articleId set (for metadata join sanity)
    print("[scan] counting lines + collecting articleIds...")
    total_est = 0
    chunk_article_ids = set()
    for jf in jsonl_files:
        total_est += count_lines_fast(jf)
        # collect articleId cheaply (parse only a few lines)
        try:
            for i, row in enumerate(iter_jsonl(jf)):
                aid = str(row.get("articleId", "")).strip()
                if aid:
                    chunk_article_ids.add(aid)
                if i >= 50:
                    break
        except Exception:
            pass
    print(f"[scan] estimated total lines (incl blanks): {total_est}")
    print(f"[scan] distinct articleIds observed (sampled): {len(chunk_article_ids)}")

    meta_map = load_article_metadata(metadata_dir)
    meta_ids = set(meta_map.keys())
    matched = len(meta_ids.intersection(chunk_article_ids)) if chunk_article_ids else 0
    print(f"[meta] loaded {len(meta_map)} article metadata records")
    print(f"[meta] metadata articleId matches (sampled): {matched}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("[neo4j] connectivity ok")

    q_constraints = [
        "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.articleId IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
        "CREATE INDEX chunk_article IF NOT EXISTS FOR (c:Chunk) ON (c.articleId)",
        "CREATE INDEX chunk_seq IF NOT EXISTS FOR (c:Chunk) ON (c.seq)",
    ]

    q_reset = """
    MATCH (n)
    WHERE n:Article OR n:Chunk
    DETACH DELETE n
    """

    q_upsert_articles = """
    UNWIND $rows AS row
    MERGE (a:Article {articleId: row.articleId})
    SET a.title   = coalesce(row.title, a.title),
        a.year    = coalesce(row.year, a.year),
        a.journal = coalesce(row.journal, a.journal),
        a.url     = coalesce(row.url, a.url),
        a.authors = coalesce(row.authors, a.authors)
    RETURN count(*) AS n
    """

    q_upsert_chunks = """
    UNWIND $rows AS row
    MERGE (c:Chunk {chunkId: row.chunkId})
    SET c.articleId      = row.articleId,
        c.seq            = row.seq,
        c.subseq         = row.subseq,
        c.parentChunkId  = row.parentChunkId,
        c.text           = row.text,
        c.file           = row.file
    MERGE (a:Article {articleId: row.articleId})
    MERGE (a)-[:HAS_CHUNK]->(c)
    RETURN count(*) AS n
    """

    q_subchunk = """
    UNWIND $pairs AS p
    MATCH (child:Chunk {chunkId: p.child})
    MERGE (parent:Chunk {chunkId: p.parent})
    MERGE (child)-[:SUBCHUNK_OF]->(parent)
    RETURN count(*) AS n
    """

    q_next = """
    UNWIND $pairs AS p
    MATCH (a:Chunk {chunkId: p.frm})
    MATCH (b:Chunk {chunkId: p.to})
    MERGE (a)-[:NEXT]->(b)
    RETURN count(*) AS n
    """

    with driver.session(database=database) as session:
        # constraints
        for q in q_constraints:
            session.execute_write(lambda tx, qq=q: tx.run(qq).consume())
        print("[neo4j] constraints/indexes ensured")

        if args.reset:
            print("[reset] deleting existing Article/Chunk graph content...")
            session.execute_write(lambda tx: tx.run(q_reset).consume())
            print("[reset] done")

        # load article metadata first
        if meta_map:
            rows = []
            for aid, obj in meta_map.items():
                rows.append(
                    {
                        "articleId": aid,
                        "title": obj.get("title"),
                        "year": as_int(obj.get("year")),
                        "journal": obj.get("journal"),
                        "url": obj.get("url"),
                        "authors": obj.get("authors") if isinstance(obj.get("authors"), list) else None,
                    }
                )
            n = neo4j_execute_write(session, q_upsert_articles, {"rows": rows})
            print(f"[articles] upserted {n}")

        ingested = 0
        skipped = 0
        next_edges = 0
        sub_edges = 0

        for jf in jsonl_files:
            print(f"\n[file] {jf.name}")
            rows: List[Dict[str, Any]] = []
            raw_rows_for_edges: List[Dict[str, Any]] = []

            for row in iter_jsonl(jf):
                article_id = str(row.get("articleId", "")).strip()
                chunk_id = str(row.get("chunkId", "")).strip()
                text = str(row.get("text", "")).strip()

                if not article_id or not chunk_id or not text:
                    skipped += 1
                    continue

                seq = as_int(row.get("seq"))
                parent = str(row.get("parentChunkId", "")).strip() or None
                subseq = as_int(row.get("subseq")) if "subseq" in row else None

                out = {
                    "articleId": article_id,
                    "chunkId": chunk_id,
                    "seq": seq,
                    "subseq": subseq,
                    "parentChunkId": parent,
                    "text": text,
                    "file": str(jf),
                }

                rows.append(out)
                raw_rows_for_edges.append(out)

                if len(rows) >= args.batch_size:
                    n = neo4j_execute_write(session, q_upsert_chunks, {"rows": rows})
                    ingested += n
                    rows.clear()

                    if ingested and ingested % args.print_every == 0:
                        print(f"[progress] ingested={ingested} skipped={skipped}")

            # final batch
            if rows:
                n = neo4j_execute_write(session, q_upsert_chunks, {"rows": rows})
                ingested += n
                rows.clear()

            # relationships per-file (optional)
            if not args.no_subchunk:
                pairs = []
                for r in raw_rows_for_edges:
                    if r.get("parentChunkId"):
                        pairs.append({"child": r["chunkId"], "parent": r["parentChunkId"]})
                if pairs:
                    n = neo4j_execute_write(session, q_subchunk, {"pairs": pairs})
                    sub_edges += n

            if not args.no_next:
                pairs = []

                # top-level NEXT within article: parentChunkId is None
                top = [r for r in raw_rows_for_edges if not r.get("parentChunkId")]
                top_sorted = sorted(top, key=lambda r: (r["seq"] if r["seq"] is not None else 10**9, r["chunkId"]))
                for i in range(len(top_sorted) - 1):
                    pairs.append({"frm": top_sorted[i]["chunkId"], "to": top_sorted[i + 1]["chunkId"]})

                # subchunk NEXT within each parentChunkId
                subgroups: Dict[str, List[Dict[str, Any]]] = {}
                for r in raw_rows_for_edges:
                    parent = r.get("parentChunkId")
                    if parent:
                        subgroups.setdefault(parent, []).append(r)

                for parent, lst in subgroups.items():
                    lst_sorted = sorted(lst, key=lambda r: (r["subseq"] if r["subseq"] is not None else 10**9,
                                                           r["seq"] if r["seq"] is not None else 10**9,
                                                           r["chunkId"]))
                    for i in range(len(lst_sorted) - 1):
                        pairs.append({"frm": lst_sorted[i]["chunkId"], "to": lst_sorted[i + 1]["chunkId"]})

                if pairs:
                    n = neo4j_execute_write(session, q_next, {"pairs": pairs})
                    next_edges += n

            print(f"[file] done {jf.name}: ingested_so_far={ingested} skipped_so_far={skipped}")

        print("\nDone.")
        print(f"Chunks ingested: {ingested}")
        print(f"Rows skipped:   {skipped}")
        print(f"NEXT edges:     {next_edges}")
        print(f"SUBCHUNK edges: {sub_edges}")

    driver.close()


if __name__ == "__main__":
    main()
