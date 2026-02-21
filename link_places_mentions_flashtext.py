#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-precision linker: (:Chunk)-[:MENTIONS]->(:Place)

Design choices (conservative on purpose):
- All matching is CASE-SENSITIVE (so 'temple' won't match 'Temple')
- Single-token names are ONLY accepted if:
    - in a whitelist (e.g., Rome, Egypt), OR
    - preceded by a toponymic preposition (in/at/near/from/to/...)
  This removes junk like "This", "Well", "Scope", "Alone", etc.
- Optional slash splitting (A/B -> A, B) is OFF by default; when enabled,
  it only splits into parts that look like proper names and are not generic.

Writes relationship props for audit:
  r.matchedName (canonical lowercase)
  r.matchedSurface (exact surface from chunk)
  r.method, r.count, timestamps

Usage:
  pip install neo4j flashtext
  python link_places_mentions.py --limit-chunks 50
  python link_places_mentions.py
"""

import os
import re
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from neo4j import GraphDatabase
from flashtext import KeywordProcessor


# --- Place name sources (your Place nodes have title + altNames) ---
NAME_FIELDS = ["title", "altNames"]

# Minimal stopwords (used only to reject obviously meaningless candidate labels)
STOP = {"the", "and", "or", "a", "an", "of", "to", "in", "on", "at", "by", "for", "from"}

# Words that are too generic as standalone "places"
GENERIC_WORDS = {
    "temple", "church", "fort", "settlement", "region", "zone", "scope", "well",
    "river", "mountain", "valley", "plain", "harbor", "port", "market", "bridge",
    "road", "street", "gate", "tower", "palace", "city", "town", "village",
}

# Prepositions that often introduce toponyms
TOPONYM_PREP = {
    "in", "at", "near", "from", "to", "into", "onto", "around", "within",
    "across", "between", "outside", "toward", "towards", "beyond", "through",
    "over", "under",
}

# Single-token whitelist (allow even without preposition)
# Expand as needed; keep it small to preserve precision.
SINGLE_TOKEN_WHITELIST = {
    "rome", "athens", "alexandria", "jerusalem", "sparta", "carthage", "antioch",
    "egypt", "italy", "greece", "sicily", "nubia",
}

# If a single-token name is not whitelisted, require preposition context.
REQUIRE_PREP_FOR_SINGLETONS = True

# Only index labels with len >= this
MIN_LABEL_LEN = 4


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def looks_like_proper_name(token: str) -> bool:
    """
    Heuristic for slash parts: keep only if it looks like a proper name token.
    """
    if len(token) < MIN_LABEL_LEN:
        return False
    if token.lower() in STOP:
        return False
    if token.lower() in GENERIC_WORDS:
        return False
    # Must start with uppercase letter (Latin)
    return bool(re.match(r"^[A-Z][A-Za-z\-’'`]*$", token))


def tokenize_slash_parts(name: str) -> List[str]:
    """
    Optional: "A/B" -> ["A/B", "A", "B"] (filtered by looks_like_proper_name)
    """
    out = [name]
    if "/" in name:
        parts = [p.strip() for p in re.split(r"\s*/\s*", name) if p.strip()]
        for p in parts:
            if looks_like_proper_name(p):
                out.append(p)
    # de-dup preserving order
    seen = set()
    dedup = []
    for x in out:
        x_lc = x.lower()
        if x_lc in seen:
            continue
        seen.add(x_lc)
        dedup.append(x)
    return dedup


def iter_name_variants(place_props: dict, split_slash: bool) -> List[str]:
    raw: List[str] = []

    for f in NAME_FIELDS:
        v = place_props.get(f)
        if not v:
            continue
        if isinstance(v, str):
            raw.append(v)
        elif isinstance(v, list):
            raw.extend([x for x in v if isinstance(x, str)])

    cleaned: List[str] = []
    seen = set()

    for x in raw:
        x = norm_spaces(x)
        if not x:
            continue

        candidates = tokenize_slash_parts(x) if split_slash else [x]
        for cand in candidates:
            cand = norm_spaces(cand)
            if not cand:
                continue
            if len(cand) < MIN_LABEL_LEN:
                continue

            cand_lc = cand.lower()
            if cand_lc in STOP:
                continue

            # if single token and generic word, skip indexing entirely
            if " " not in cand and cand_lc in GENERIC_WORDS and cand_lc not in SINGLE_TOKEN_WHITELIST:
                continue

            if cand_lc in seen:
                continue
            seen.add(cand_lc)
            cleaned.append(cand)

    return cleaned


def is_word_char(ch: str) -> bool:
    # Treat unicode letters/digits/underscore as word chars
    return ch.isalnum() or ch == "_"


def boundary_ok(text: str, start: int, end: int) -> bool:
    """
    Ensure match is not inside a larger word.
    """
    if start > 0 and is_word_char(text[start - 1]):
        return False
    if end < len(text) and is_word_char(text[end]):
        return False
    return True


def prev_alpha_word(text: str, start: int) -> str:
    """
    Get last alphabetic token immediately before start.
    """
    left = text[:start]
    m = re.search(r"([A-Za-z]+)\s*$", left)
    return m.group(1).lower() if m else ""


def accept_match(text: str, start: int, end: int, canonical_lc: str) -> bool:
    """
    Conservative acceptance rules.
    """
    if not boundary_ok(text, start, end):
        return False

    surface = text[start:end]
    # If the surface isn't capitalized at all, reject (case-sensitive matching should already help,
    # but this blocks odd cases like ALL LOWER that sneak in via altNames).
    if not (surface[:1].isupper() or surface.isupper()):
        return False

    single_token = (" " not in canonical_lc)

    if single_token:
        if canonical_lc in SINGLE_TOKEN_WHITELIST:
            return True
        if REQUIRE_PREP_FOR_SINGLETONS:
            return prev_alpha_word(text, start) in TOPONYM_PREP
        return True

    # Multi-word: accept (already case-sensitive + boundary-checked)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    ap.add_argument("--db", default=os.getenv("NEO4J_DB", "neo4j"))

    ap.add_argument("--skip-ambiguous", action="store_true", default=True,
                    help="Skip names mapping to >1 place (default True). Use --no-skip-ambiguous to link all.")
    ap.add_argument("--no-skip-ambiguous", dest="skip_ambiguous", action="store_false")

    ap.add_argument("--limit-chunks", type=int, default=0, help="For testing: only process first N chunks.")
    ap.add_argument("--batch", type=int, default=2000, help="Write batch size (rows) to Neo4j.")

    ap.add_argument("--split-slash", action="store_true", default=False,
                    help="Split slash titles A/B into A and B (filtered). Default False (safer).")

    ap.add_argument("--method-tag", default="flashtext_precise_v1",
                    help="Stored in r.method so you can delete/re-run cleanly.")
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    # 1) Load places -> build name->pleiadesIds map + keyword matcher
    name_to_ids: Dict[str, List[str]] = defaultdict(list)
    place_count = 0

    # CASE-SENSITIVE matcher only
    kp = KeywordProcessor(case_sensitive=True)

    with driver.session(database=args.db) as session:
        q_places = """
        MATCH (p:Place)
        RETURN p.pleiadesId AS pleiadesId, p AS p
        """
        for rec in session.run(q_places):
            place_count += 1
            pid = rec["pleiadesId"]
            props = dict(rec["p"])

            for nm in iter_name_variants(props, split_slash=args.split_slash):
                canonical = nm.lower()
                name_to_ids[canonical].append(pid)

    keyword_count = 0
    ambiguous_skipped = 0

    for canonical_lc, ids in name_to_ids.items():
        if args.skip_ambiguous and len(set(ids)) > 1:
            ambiguous_skipped += 1
            continue

        # Add surface forms to the case-sensitive matcher.
        # For precision, DO NOT add lowercase forms for single tokens.
        # We add TitleCase and UPPER variants.
        surface_title = canonical_lc.title()
        surface_upper = canonical_lc.upper()

        # If the canonical contains spaces, title() can mangle some things (e.g., "al-"), but still useful.
        kp.add_keyword(surface_title, canonical_lc)
        kp.add_keyword(surface_upper, canonical_lc)

        # Also add the exact original casing if it differs (best-effort):
        # Many Pleiades titles are Title Case already; this ensures exact string is included.
        # (We can't recover the original casing here, but title() covers most.)
        keyword_count += 1

    print(f"Places loaded: {place_count}")
    print(f"Keywords prepared: {keyword_count} (ambiguous skipped: {ambiguous_skipped}, skip_ambiguous={args.skip_ambiguous})")
    print(f"Mode: split_slash={args.split_slash}, require_prep_singletons={REQUIRE_PREP_FOR_SINGLETONS}")

    # 2) Stream chunks, extract matches with spans, apply gating, write in batches
    rows = []
    chunks_seen = 0
    mentions_written = 0
    rejected = 0

    write_query = """
    UNWIND $rows AS row
    MATCH (c:Chunk {chunkId: row.chunkId})
    MATCH (p:Place {pleiadesId: row.pleiadesId})
    MERGE (c)-[r:MENTIONS]->(p)
    ON CREATE SET r.createdAt = datetime()
    SET r.matchedName = row.matchedName,
        r.matchedSurface = row.matchedSurface,
        r.method = row.method,
        r.count = row.count,
        r.lastSeenAt = datetime()
    """

    with driver.session(database=args.db) as session:
        q_chunks = "MATCH (c:Chunk) RETURN c.chunkId AS chunkId, c.text AS text ORDER BY c.chunkId"
        stream = session.run(q_chunks)

        for rec in stream:
            chunks_seen += 1
            if args.limit_chunks and chunks_seen > args.limit_chunks:
                break

            chunk_id = rec["chunkId"]
            text = rec["text"] or ""
            if not text:
                continue

            hits = kp.extract_keywords(text, span_info=True)  # list of (canonical_lc, start, end)
            if not hits:
                continue

            # Count accepted hits per canonical label
            counts = Counter()
            surface_example = {}

            for canonical_lc, start, end in hits:
                # canonical_lc was produced by the keyword processor; enforce presence in map
                if canonical_lc not in name_to_ids:
                    continue
                if not accept_match(text, start, end, canonical_lc):
                    rejected += 1
                    continue

                counts[canonical_lc] += 1
                # keep one surface example (for audit)
                if canonical_lc not in surface_example:
                    surface_example[canonical_lc] = text[start:end]

            if not counts:
                continue

            for canonical_lc, cnt in counts.items():
                ids = list(dict.fromkeys(name_to_ids.get(canonical_lc, [])))
                if not ids:
                    continue

                if args.skip_ambiguous and len(ids) != 1:
                    continue

                target_ids = ids if not args.skip_ambiguous else [ids[0]]
                for pid in target_ids:
                    rows.append({
                        "chunkId": chunk_id,
                        "pleiadesId": pid,
                        "matchedName": canonical_lc,
                        "matchedSurface": surface_example.get(canonical_lc, ""),
                        "count": int(cnt),
                        "method": args.method_tag
                    })
                    mentions_written += 1

            if len(rows) >= args.batch:
                session.run(write_query, rows=rows)
                rows.clear()

        if rows:
            session.run(write_query, rows=rows)
            rows.clear()

    driver.close()
    print(f"Chunks processed: {chunks_seen}")
    print(f"MENTIONS rows written (pre-merge): {mentions_written}")
    print(f"Rejected hits (gating/boundary/context): {rejected}")
    print("Done.")


if __name__ == "__main__":
    main()
