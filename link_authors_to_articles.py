#!/usr/bin/env python3
"""
Create/merge author Persons and connect them to Articles:

  (:Person:Author {personId, name, nameNorm, nameKey, aliases, source, confidence})
    -[:WROTE {source:'article-metadata', authorIndex:i}]->
  (:Article)

Key points (fixes your crash):
- MERGE authors by (Person {personId}) so we reuse existing Person nodes and never violate
  the Person.personId uniqueness constraint.
- nameKey/nameNorm are still stored for lookup, but are NOT used as the merge key.
- Canonicalizes author keys by removing accents/punctuation and collapsing whitespace.
- Also drops single-letter MIDDLE initials so:
    "christopher w blackwell" and "christopher blackwell" -> same personId
- Optional cleanup: if an older node exists for the "with-middle-initial" id, move its
  WROTE relationships to the canonical node and delete the old node.

Run:
  python link_authors_to_articles.py
  python link_authors_to_articles.py --reset_wrote
  python link_authors_to_articles.py --batch_size 500
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ----------------------------
# name canonicalization
# ----------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_SLUG_RE = re.compile(r"[^a-z0-9]+", flags=re.IGNORECASE)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canon_basic(s: str) -> str:
    """
    Lowercase, strip accents, remove punctuation, collapse whitespace.
    """
    s = (s or "").strip()
    if not s:
        return ""
    s = strip_accents(s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def canon_drop_middle_initials(s: str) -> str:
    """
    Like canon_basic, but removes single-letter tokens ONLY in the middle.

    Examples:
      "christopher w blackwell" -> "christopher blackwell"
      "w christopher blackwell"  -> stays "w christopher blackwell" (first token kept)
      "christopher blackwell"    -> "christopher blackwell"
    """
    base = canon_basic(s)
    toks = base.split()
    if len(toks) <= 2:
        return base

    first = toks[0]
    last = toks[-1]
    middle = toks[1:-1]

    middle2 = [t for t in middle if not (len(t) == 1)]
    out = [first] + middle2 + [last]
    return " ".join(out).strip()

def slugify_key(key: str) -> str:
    key = (key or "").strip().lower()
    key = _SLUG_RE.sub("-", key).strip("-")
    return (key or "x")[:32]

def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def person_id_from_key(name_key: str) -> str:
    """
    Matches your earlier style:
      person:<slug>:<hash10>
    where hash10 is sha1(canonical_key)[:10]
    """
    nk = (name_key or "").strip().lower()
    return f"person:{slugify_key(nk)}:{short_hash(nk, 10)}"


# ----------------------------
# cypher
# ----------------------------

Q_SCHEMA = [
    # You probably already have Person.personId unique from earlier scripts.
    # These are safe to (re)run:
    "CREATE INDEX author_nameKey IF NOT EXISTS FOR (a:Author) ON (a.nameKey)",
    "CREATE INDEX author_nameNorm IF NOT EXISTS FOR (a:Author) ON (a.nameNorm)",
    "CREATE INDEX article_articleId IF NOT EXISTS FOR (a:Article) ON (a.articleId)",
]

Q_FETCH_ARTICLES = """
MATCH (a:Article)
WHERE a.articleId IS NOT NULL AND a.authors IS NOT NULL AND size(a.authors) > 0
RETURN a.articleId AS articleId, a.authors AS authors
ORDER BY a.articleId
"""

Q_RESET_WROTE = """
MATCH (:Person:Author)-[r:WROTE]->(:Article)
WHERE coalesce(r.source,'') = 'article-metadata'
DELETE r
"""

Q_UPSERT_AND_LINK = """
UNWIND $rows AS row
MATCH (a:Article {articleId: row.articleId})

MERGE (p:Person {personId: row.personId})
SET p:Author
SET
  // keep a stable canonical key for matching/debug
  p.nameKey   = coalesce(p.nameKey, row.nameKey),
  p.nameNorm  = coalesce(p.nameNorm, row.nameNorm),
  // prefer the "most informative" display name (usually the longer one)
  p.name      = CASE
                  WHEN p.name IS NULL THEN row.displayName
                  WHEN size(toString(row.displayName)) > size(toString(p.name)) THEN row.displayName
                  ELSE p.name
                END,
  p.source    = coalesce(p.source, 'article-authors'),
  p.confidence= coalesce(p.confidence, 1.0),
  p.aliases   = coalesce(p.aliases, [])

WITH p, a, row
MERGE (p)-[r:WROTE]->(a)
  ON CREATE SET r.source = 'article-metadata',
                r.authorIndex = row.authorIndex
SET r.authorIndex = coalesce(r.authorIndex, row.authorIndex)

RETURN count(*) AS n
"""

Q_MERGE_VARIANT_INTO_CANON = """
// Move WROTE rels from old -> canon, add old.name/aliases into canon.aliases, then delete old.
// Only runs if old has at least one WROTE->Article (so we don't nuke unrelated Persons).
MATCH (old:Person {personId:$oldId})
MATCH (canon:Person {personId:$canonId})
WHERE old <> canon
  AND (old)-[:WROTE]->(:Article)

OPTIONAL MATCH (old)-[r:WROTE]->(a:Article)
FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END |
  MERGE (canon)-[r2:WROTE]->(a)
  ON CREATE SET r2.source = coalesce(r.source, 'article-metadata'),
                r2.authorIndex = r.authorIndex
  SET r2.authorIndex = coalesce(r2.authorIndex, r.authorIndex),
      r2.source      = coalesce(r2.source, r.source)
)

WITH old, canon,
     [x IN (coalesce(canon.aliases, []) +
            [coalesce(old.name,'')] +
            coalesce(old.aliases, []))
      WHERE x IS NOT NULL AND trim(toString(x)) <> ""] AS al
SET canon.aliases =
  reduce(out = [], x IN al |
    CASE WHEN x IN out THEN out ELSE out + x END
  )

// keep the longer display name after merge
SET canon.name =
  CASE
    WHEN canon.name IS NULL THEN old.name
    WHEN old.name IS NULL THEN canon.name
    WHEN size(toString(old.name)) > size(toString(canon.name)) THEN old.name
    ELSE canon.name
  END

DETACH DELETE old
RETURN 1 AS merged
"""


def execute_write_count(session, cypher: str, params: Dict[str, Any]) -> int:
    def _tx(tx):
        rec = tx.run(cypher, params).single()
        return int(rec["n"]) if rec and "n" in rec else 0
    return session.execute_write(_tx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--reset_wrote", action="store_true", help="Delete WROTE rels (source=article-metadata) before re-linking")
    ap.add_argument("--no_merge_initial_variants", action="store_true", help="Do NOT merge old 'middle-initial' nodes into canonical")
    args = ap.parse_args()

    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise ValueError("Set NEO4J_PASSWORD in your .env or env vars")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"[neo4j] connected uri={uri} db={database} user={user}")

    total_rows = 0
    total_written = 0
    batch: List[Dict[str, Any]] = []
    # mappings: (oldId -> canonId) for names where middle initials were dropped
    merge_map: List[Tuple[str, str]] = []

    with driver.session(database=database) as session:
        for q in Q_SCHEMA:
            session.execute_write(lambda tx, qq=q: tx.run(qq).consume())
        print("[neo4j] schema ensured (indexes)")

        if args.reset_wrote:
            session.execute_write(lambda tx: tx.run(Q_RESET_WROTE).consume())
            print("[reset] deleted WROTE rels (source=article-metadata)")

        articles = list(session.run(Q_FETCH_ARTICLES))
        print(f"[scan] articles with authors: {len(articles)}")

        for rec in articles:
            article_id = rec["articleId"]
            authors = rec["authors"] or []

            for idx, raw_name in enumerate(authors):
                display = str(raw_name or "").strip()
                if not display:
                    continue

                raw_key = canon_basic(display)                 # without dropping middle initials
                canon_key = canon_drop_middle_initials(display) # with middle-initial collapsing
                if not canon_key:
                    continue

                canon_pid = person_id_from_key(canon_key)

                # if we changed the key by dropping a middle initial, remember the old id too
                if raw_key and raw_key != canon_key:
                    old_pid = person_id_from_key(raw_key)
                    if old_pid != canon_pid:
                        merge_map.append((old_pid, canon_pid))

                row = {
                    "articleId": article_id,
                    "authorIndex": idx,
                    "displayName": display,
                    "nameNorm": canon_key,
                    "nameKey": canon_key,
                    "personId": canon_pid,
                }
                batch.append(row)

                if len(batch) >= args.batch_size:
                    n = execute_write_count(session, Q_UPSERT_AND_LINK, {"rows": batch})
                    total_written += n
                    total_rows += len(batch)
                    batch.clear()
                    print(f"[progress] processed_rows={total_rows} wrote_links={total_written}")

        if batch:
            n = execute_write_count(session, Q_UPSERT_AND_LINK, {"rows": batch})
            total_written += n
            total_rows += len(batch)
            batch.clear()

        # Optional: merge old "with-middle-initial" nodes into canonical ids
        merged_variants = 0
        if not args.no_merge_initial_variants and merge_map:
            # de-dupe mappings
            seen = set()
            uniq = []
            for old_id, canon_id in merge_map:
                if (old_id, canon_id) not in seen:
                    seen.add((old_id, canon_id))
                    uniq.append((old_id, canon_id))

            for old_id, canon_id in uniq:
                recm = session.execute_write(
                    lambda tx: tx.run(Q_MERGE_VARIANT_INTO_CANON, oldId=old_id, canonId=canon_id).single()
                )
                if recm:
                    merged_variants += 1

        sanity = session.run("""
            MATCH (p:Author)-[r:WROTE]->(a:Article)
            RETURN count(r) AS wrote_rels, count(DISTINCT p) AS authors, count(DISTINCT a) AS articles
        """).single()

        print("\nDone.")
        print(f"Rows processed: {total_rows}")
        print(f"WROTE merged:   {total_written}")
        print(f"Variants merged (middle initial): {merged_variants}")
        if sanity:
            print(f"Sanity: WROTE={sanity['wrote_rels']} authors={sanity['authors']} articles={sanity['articles']}")

    driver.close()


if __name__ == "__main__":
    main()
