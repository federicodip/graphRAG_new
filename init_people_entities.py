#!/usr/bin/env python3
"""
Initialize :Person and :Entity node schemas in Neo4j, and optionally seed nodes.

Fixes included:
1) Removes the Cypher sanity-query syntax error (uses MATCH/COUNT instead of list comprehension).
2) Deduplicates author names like "Christopher Blackwell" vs "Christopher W Blackwell" by
   using a stronger canonical normalization for IDs (middle initials + punctuation removed).

Creates:
  Constraints:
    - Person.personId UNIQUE
    - Entity.entityId UNIQUE
  Indexes:
    - Person.nameNorm
    - Entity.nameNorm
    - Entity.kind
  Optional full-text indexes:
    - person_name_ft on Person.name
    - entity_name_ft on Entity.name

Optional seeding:
  1) --seed_from_article_authors
     Creates Person nodes from (:Article).authors (list of strings).

  2) --persons_jsonl path/to/persons.jsonl
     Each line JSON object, e.g.:
       {"name":"Alexander Jones","aliases":["Jones, Alexander"],"source":"manual","confidence":1.0}
     Optional: {"personId":"person:..."} to force IDs.

  3) --entities_jsonl path/to/entities.jsonl
     Each line JSON object, e.g.:
       {"name":"Antikythera mechanism","kind":"artifact","aliases":[],"source":"manual","confidence":1.0}
     Optional: {"entityId":"entity:..."} to force IDs.

Run:
  python init_people_entities.py
  python init_people_entities.py --seed_from_article_authors
  python init_people_entities.py --persons_jsonl data/persons.jsonl --entities_jsonl data/entities.jsonl
  python init_people_entities.py --reset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ----------------------------
# utils
# ----------------------------

WS_RE = re.compile(r"\s+", re.UNICODE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
SLUG_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.IGNORECASE)


def norm_name(s: str) -> str:
    """Loose normalization used for display/search fields."""
    s = (s or "").strip()
    s = WS_RE.sub(" ", s)
    return s.lower()


def canonical_person_key(name: str) -> str:
    """
    Stronger normalization used to generate stable Person IDs
    (tries to collapse middle initials / punctuation differences).

    Examples:
      "Christopher W Blackwell" -> "christopher blackwell"
      "Christopher W. Blackwell" -> "christopher blackwell"
      "Christopher Blackwell" -> "christopher blackwell"
    """
    s = (name or "").strip().lower()
    s = s.replace(",", " ")
    s = SUFFIX_RE.sub(" ", s)
    s = NON_ALNUM_RE.sub(" ", s)         # drop punctuation
    s = WS_RE.sub(" ", s).strip()

    parts = s.split()
    if len(parts) >= 3:
        # remove single-letter middle initials (w, w., etc.) anywhere after first token
        parts2 = [parts[0]]
        for p in parts[1:]:
            if len(p) == 1:
                continue
            parts2.append(p)
        parts = parts2

    return " ".join(parts)


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = SLUG_RE.sub("-", s).strip("-")
    return s or "x"


def gen_person_id(name: str) -> str:
    """
    Person ID is based on canonical_person_key() to dedupe near-duplicates.
    """
    key = canonical_person_key(name)
    base = slugify(key)[:32]
    h = short_hash(key, 10)
    return f"person:{base}:{h}"


def gen_entity_id(name: str, kind: Optional[str]) -> str:
    k = (kind or "thing").strip().lower() or "thing"
    base = slugify(norm_name(name))[:32]
    h = short_hash(f"{k}::{norm_name(name)}", 10)
    return f"entity:{k}:{base}:{h}"


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


# ----------------------------
# cypher
# ----------------------------

Q_CONSTRAINTS = [
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.personId IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entityId IS UNIQUE",
]

Q_INDEXES = [
    "CREATE INDEX person_nameNorm IF NOT EXISTS FOR (p:Person) ON (p.nameNorm)",
    "CREATE INDEX entity_nameNorm IF NOT EXISTS FOR (e:Entity) ON (e.nameNorm)",
    "CREATE INDEX entity_kind IF NOT EXISTS FOR (e:Entity) ON (e.kind)",
]

Q_FT_INDEXES = [
    "CREATE FULLTEXT INDEX person_name_ft IF NOT EXISTS FOR (p:Person) ON EACH [p.name]",
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
]

Q_RESET = """
MATCH (n)
WHERE n:Person OR n:Entity
DETACH DELETE n
"""

Q_UPSERT_PERSONS = """
UNWIND $rows AS row
MERGE (p:Person {personId: row.personId})
SET p.name       = coalesce(p.name, row.name),
    p.nameNorm   = row.nameNorm,
    p.nameKey    = row.nameKey,
    p.aliases    = row.aliases,
    p.source     = row.source,
    p.confidence = row.confidence
RETURN count(*) AS n
"""

Q_UPSERT_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {entityId: row.entityId})
SET e.name       = row.name,
    e.nameNorm   = row.nameNorm,
    e.kind       = row.kind,
    e.aliases    = row.aliases,
    e.source     = row.source,
    e.confidence = row.confidence
RETURN count(*) AS n
"""

Q_AUTHORS = """
MATCH (a:Article)
WHERE a.authors IS NOT NULL
UNWIND a.authors AS name
WITH trim(name) AS name
WHERE name <> ""
RETURN DISTINCT name
ORDER BY name
"""


def execute_write_count(session, cypher: str, params: Dict[str, Any]) -> int:
    """
    Correctly reads RETURN count(*) AS n from Neo4j driver Record.
    The previous version used `"n" in rec` which is unreliable.
    """
    def _tx(tx):
        res = tx.run(cypher, params)
        rec = res.single()
        n = rec.get("n") if rec is not None else 0
        return int(n) if n is not None else 0

    return session.execute_write(_tx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete all Person/Entity nodes first")
    parser.add_argument("--no_fulltext", action="store_true", help="Do NOT create full-text indexes on names")

    parser.add_argument("--seed_from_article_authors", action="store_true", help="Create Person nodes from Article.authors")
    parser.add_argument("--persons_jsonl", default=None, help="JSONL file of persons to upsert")
    parser.add_argument("--entities_jsonl", default=None, help="JSONL file of entities to upsert")

    parser.add_argument("--batch_size", type=int, default=500, help="UNWIND batch size")
    args = parser.parse_args()

    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        raise ValueError("Missing Neo4j password. Set NEO4J_PASSWORD in your .env")

    print(f"[config] neo4j uri={uri} db={database} user={user}")
    print(f"[config] batch_size={args.batch_size}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("[neo4j] connectivity ok")

    with driver.session(database=database) as session:
        # schema
        for q in Q_CONSTRAINTS:
            session.execute_write(lambda tx, qq=q: tx.run(qq).consume())
        for q in Q_INDEXES:
            session.execute_write(lambda tx, qq=q: tx.run(qq).consume())
        if not args.no_fulltext:
            for q in Q_FT_INDEXES:
                session.execute_write(lambda tx, qq=q: tx.run(qq).consume())
        print("[neo4j] constraints/indexes ensured")

        if args.reset:
            print("[reset] deleting Person/Entity nodes...")
            session.execute_write(lambda tx: tx.run(Q_RESET).consume())
            print("[reset] done")

        # seed from authors
        if args.seed_from_article_authors:
            names = [r["name"] for r in session.run(Q_AUTHORS)]
            print(f"[seed] authors found: {len(names)}")

            rows: List[Dict[str, Any]] = []
            total = 0

            for name in names:
                pid = gen_person_id(name)
                key = canonical_person_key(name)

                rows.append({
                    "personId": pid,
                    "name": name.strip(),
                    "nameNorm": norm_name(name),
                    "nameKey": key,
                    "aliases": [],  # you can optionally append variants later
                    "source": "article-authors",
                    "confidence": 1.0,
                })

                if len(rows) >= args.batch_size:
                    n = execute_write_count(session, Q_UPSERT_PERSONS, {"rows": rows})
                    total += n
                    rows.clear()

            if rows:
                n = execute_write_count(session, Q_UPSERT_PERSONS, {"rows": rows})
                total += n

            print(f"[seed] Person upserted from authors: {total}")

        # seed from persons jsonl
        if args.persons_jsonl:
            pth = Path(args.persons_jsonl)
            if not pth.exists():
                raise FileNotFoundError(pth)
            print(f"[seed] persons_jsonl={pth}")

            rows: List[Dict[str, Any]] = []
            total = 0

            for obj in iter_jsonl(pth):
                name = str(obj.get("name", "")).strip()
                if not name:
                    continue

                # If user provided personId, respect it; otherwise generate from canonical key.
                pid = str(obj.get("personId", "")).strip() or gen_person_id(name)
                key = canonical_person_key(name)

                aliases = obj.get("aliases") if isinstance(obj.get("aliases"), list) else []
                source = str(obj.get("source", "unknown")).strip() or "unknown"
                conf = obj.get("confidence", None)
                conf = float(conf) if conf is not None else None

                rows.append({
                    "personId": pid,
                    "name": name,
                    "nameNorm": norm_name(name),
                    "nameKey": key,
                    "aliases": aliases,
                    "source": source,
                    "confidence": conf,
                })

                if len(rows) >= args.batch_size:
                    n = execute_write_count(session, Q_UPSERT_PERSONS, {"rows": rows})
                    total += n
                    rows.clear()

            if rows:
                n = execute_write_count(session, Q_UPSERT_PERSONS, {"rows": rows})
                total += n

            print(f"[seed] Person upserted from JSONL: {total}")

        # seed from entities jsonl
        if args.entities_jsonl:
            pth = Path(args.entities_jsonl)
            if not pth.exists():
                raise FileNotFoundError(pth)
            print(f"[seed] entities_jsonl={pth}")

            rows: List[Dict[str, Any]] = []
            total = 0

            for obj in iter_jsonl(pth):
                name = str(obj.get("name", "")).strip()
                if not name:
                    continue

                kind = str(obj.get("kind", "thing")).strip() or "thing"
                eid = str(obj.get("entityId", "")).strip() or gen_entity_id(name, kind)

                aliases = obj.get("aliases") if isinstance(obj.get("aliases"), list) else []
                source = str(obj.get("source", "unknown")).strip() or "unknown"
                conf = obj.get("confidence", None)
                conf = float(conf) if conf is not None else None

                rows.append({
                    "entityId": eid,
                    "name": name,
                    "nameNorm": norm_name(name),
                    "kind": kind.lower(),
                    "aliases": aliases,
                    "source": source,
                    "confidence": conf,
                })

                if len(rows) >= args.batch_size:
                    n = execute_write_count(session, Q_UPSERT_ENTITIES, {"rows": rows})
                    total += n
                    rows.clear()

            if rows:
                n = execute_write_count(session, Q_UPSERT_ENTITIES, {"rows": rows})
                total += n

            print(f"[seed] Entity upserted from JSONL: {total}")

        # quick sanity (no Cypher syntax that depends on list comprehensions)
        sanity = session.run("""
            OPTIONAL MATCH (p:Person)
            WITH count(p) AS persons
            OPTIONAL MATCH (e:Entity)
            RETURN persons, count(e) AS entities
        """).single()

        if sanity:
            print(f"[sanity] Person={sanity['persons']} Entity={sanity['entities']}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
