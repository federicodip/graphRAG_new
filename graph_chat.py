#!/usr/bin/env python3
"""
Interactive GraphRAG chat CLI with hybrid retrieval:
- vector retrieval from Chroma
- graph retrieval from Neo4j fulltext over Chunk.text
- reciprocal rank fusion (RRF) to combine both signals
- intent routes for graph aggregation/list questions

Run:
  python graph_chat.py
  python graph_chat.py --mode hybrid --vector-k 8 --graph-k 8 --final-k 8
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j import GraphDatabase


SYSTEM_RULES = (
    "You are a careful assistant.\n"
    "Use ONLY the provided context to answer.\n"
    "If the answer is not in context, say: I don't know.\n"
    "Do not include citations, source IDs, article titles, or reference phrases like 'according to'.\n"
    "Provide only the direct answer in plain prose.\n"
)


@dataclass
class Hit:
    chunk_id: str
    article_id: str
    text: str
    seq: Optional[int] = None
    places: List[str] = field(default_factory=list)
    vector_rank: Optional[int] = None
    graph_rank: Optional[int] = None
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    fused_score: float = 0.0


def clip_text(s: str, max_chars: int = 1800) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def to_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_author_count_name(question: str) -> Optional[str]:
    q = (question or "").strip()
    patterns = [
        r"^\s*how many .* written by (?P<name>.+?)\??\s*$",
        r"^\s*how many .* did (?P<name>.+?) write\??\s*$",
    ]
    for pat in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if m:
            name = normalize_space(m.group("name").strip(" .?\t\n\r"))
            if name:
                return name
    return None


def extract_top_authors_n(question: str) -> Optional[int]:
    q = (question or "").strip()
    patterns = [
        r"^\s*(?:top|show top|list top)\s*(?P<n>\d+)\s*(?:authors?|writers?).*",
        r"^\s*who are the top\s*(?P<n>\d+)\s*authors?.*",
        r"^\s*most prolific authors\s*(?:top\s*(?P<n>\d+))?.*",
    ]
    for pat in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if m:
            n_raw = m.groupdict().get("n")
            if not n_raw:
                return 5
            try:
                n = int(n_raw)
            except Exception:
                return None
            return max(1, min(n, 50))
    return None


def extract_articles_by_author_name(question: str) -> Optional[str]:
    q = (question or "").strip()
    patterns = [
        r"^\s*(?:list|show|what are)\s*(?:the\s*)?articles.*by\s+(?P<name>.+?)\??\s*$",
        r"^\s*articles.*written by\s+(?P<name>.+?)\??\s*$",
        r"^\s*which articles did\s+(?P<name>.+?)\s+write\??\s*$",
    ]
    for pat in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if m:
            name = normalize_space(m.group("name").strip(" .?\t\n\r"))
            if name:
                return name
    return None


def extract_place_name_for_mentions(question: str) -> Optional[str]:
    q = (question or "").strip()
    patterns = [
        r"^\s*(?:which|what)\s+articles.*(?:mention|mentions|mentioned)\s+(?P<place>.+?)\??\s*$",
        r"^\s*list\s+articles\s+mentioning\s+(?P<place>.+?)\??\s*$",
        r"^\s*how many\s+articles\s+(?:mention|mentions|mentioned)\s+(?P<place>.+?)\??\s*$",
    ]
    for pat in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if m:
            place = normalize_space(m.group("place").strip(" .?\t\n\r"))
            if place:
                return place
    return None


def query_best_author_match(driver, database: str, raw_name: str) -> Optional[Dict[str, str]]:
    cypher = """
    MATCH (p:Person:Author)
    WHERE toLower(coalesce(p.name, '')) CONTAINS $needle
       OR toLower(coalesce(p.nameNorm, '')) CONTAINS $needle
    WITH p,
         CASE
           WHEN toLower(coalesce(p.name, '')) = $needle THEN 0
           WHEN toLower(coalesce(p.nameNorm, '')) = $needle THEN 0
           ELSE 1
         END AS score
    ORDER BY score ASC, size(coalesce(p.name, '')) ASC
    LIMIT 1
    RETURN p.name AS author, p.personId AS personId
    """
    needle = normalize_space(raw_name).lower()
    if not needle:
        return None

    with driver.session(database=database) as session:
        rec = session.run(cypher, {"needle": needle}).single()
        if not rec or rec["author"] is None:
            return None
        return {"author": rec["author"], "personId": rec["personId"]}


def query_author_article_count(driver, database: str, raw_name: str) -> Optional[Dict[str, object]]:
    best = query_best_author_match(driver, database, raw_name)
    if best is None:
        return None

    cypher = """
    MATCH (p:Person {personId: $person_id})
    OPTIONAL MATCH (p)-[:WROTE]->(a:Article)
    RETURN p.name AS author,
           p.personId AS personId,
           count(DISTINCT a) AS articleCount,
           collect(DISTINCT a.articleId)[0..100] AS articleIds
    """
    with driver.session(database=database) as session:
        rec = session.run(cypher, {"person_id": best["personId"]}).single()
        if not rec or rec["author"] is None:
            return None
        return {
            "author": rec["author"],
            "personId": rec["personId"],
            "articleCount": int(rec["articleCount"]),
            "articleIds": [x for x in (rec["articleIds"] or []) if isinstance(x, str)],
        }


def query_top_authors(driver, database: str, n: int) -> List[Dict[str, object]]:
    cypher = """
    MATCH (p:Person:Author)-[:WROTE]->(a:Article)
    WITH p, count(DISTINCT a) AS articleCount
    ORDER BY articleCount DESC, toLower(coalesce(p.name, '')) ASC
    LIMIT $n
    RETURN p.name AS author, p.personId AS personId, articleCount
    """
    out: List[Dict[str, object]] = []
    with driver.session(database=database) as session:
        rows = session.run(cypher, {"n": max(1, min(n, 50))})
        for row in rows:
            out.append(
                {
                    "author": row["author"] or "unknown",
                    "personId": row["personId"],
                    "articleCount": int(row["articleCount"]),
                }
            )
    return out


def query_articles_by_author(driver, database: str, raw_name: str) -> Optional[Dict[str, object]]:
    best = query_best_author_match(driver, database, raw_name)
    if best is None:
        return None

    cypher = """
    MATCH (p:Person {personId: $person_id})-[:WROTE]->(a:Article)
    RETURN p.name AS author,
           p.personId AS personId,
           collect(DISTINCT a.articleId) AS articleIds
    """
    with driver.session(database=database) as session:
        rec = session.run(cypher, {"person_id": best["personId"]}).single()
        if not rec or rec["author"] is None:
            return None
        ids = sorted([x for x in (rec["articleIds"] or []) if isinstance(x, str)])
        return {
            "author": rec["author"],
            "personId": rec["personId"],
            "articleCount": len(ids),
            "articleIds": ids,
        }


def query_articles_mentioning_place(driver, database: str, raw_place: str) -> Optional[Dict[str, object]]:
    place_match = """
    MATCH (p:Place)
    WHERE toLower(coalesce(p.title, '')) CONTAINS $needle
       OR any(x IN coalesce(p.altNames, []) WHERE toLower(toString(x)) CONTAINS $needle)
    WITH p,
         CASE WHEN toLower(coalesce(p.title, '')) = $needle THEN 0 ELSE 1 END AS score
    ORDER BY score ASC, size(coalesce(p.title, '')) ASC
    LIMIT 1
    RETURN p.pleiadesId AS pleiadesId, p.title AS title
    """
    needle = normalize_space(raw_place).lower()
    if not needle:
        return None

    with driver.session(database=database) as session:
        best = session.run(place_match, {"needle": needle}).single()
        if not best or best["pleiadesId"] is None:
            return None

        cypher = """
        MATCH (p:Place {pleiadesId: $pid})
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(p)
        OPTIONAL MATCH (a:Article)-[:HAS_CHUNK]->(c)
        RETURN p.title AS place,
               p.pleiadesId AS pleiadesId,
               count(DISTINCT a) AS articleCount,
               collect(DISTINCT a.articleId)[0..100] AS articleIds
        """
        rec = session.run(cypher, {"pid": best["pleiadesId"]}).single()
        if not rec:
            return None
        ids = sorted([x for x in (rec["articleIds"] or []) if isinstance(x, str)])
        return {
            "place": rec["place"] or raw_place,
            "pleiadesId": rec["pleiadesId"],
            "articleCount": int(rec["articleCount"]),
            "articleIds": ids,
        }


def vector_hits(vector_store: Chroma, query: str, k: int) -> List[Hit]:
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    out: List[Hit] = []
    for rank, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        chunk_id = str(meta.get("chunkId") or "").strip()
        if not chunk_id:
            continue
        out.append(
            Hit(
                chunk_id=chunk_id,
                article_id=str(meta.get("articleId") or meta.get("source") or "unknown"),
                text=doc.page_content,
                seq=to_int(meta.get("chunk")),
                vector_rank=rank,
                vector_score=float(score),
            )
        )
    return out


def graph_hits(driver, database: str, index_name: str, query: str, k: int) -> List[Hit]:
    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $search_text) YIELD node, score
    WHERE node:Chunk
    OPTIONAL MATCH (node)-[:MENTIONS]->(p:Place)
    RETURN node.chunkId AS chunkId,
           node.articleId AS articleId,
           node.seq AS seq,
           node.text AS text,
           score AS graphScore,
           collect(DISTINCT p.title)[0..4] AS places
    ORDER BY score DESC
    LIMIT $k
    """
    out: List[Hit] = []
    with driver.session(database=database) as session:
        rows = session.run(cypher, {"index_name": index_name, "search_text": query, "k": k})
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row["chunkId"] or "").strip()
            if not chunk_id:
                continue
            out.append(
                Hit(
                    chunk_id=chunk_id,
                    article_id=str(row["articleId"] or "unknown"),
                    text=str(row["text"] or ""),
                    seq=to_int(row["seq"]),
                    places=[p for p in (row["places"] or []) if isinstance(p, str)],
                    graph_rank=rank,
                    graph_score=float(row["graphScore"]),
                )
            )
    return out


def fuse_hits(vector: List[Hit], graph: List[Hit], rrf_k: int = 60) -> List[Hit]:
    by_chunk: Dict[str, Hit] = {}

    for h in vector:
        by_chunk[h.chunk_id] = h

    for g in graph:
        existing = by_chunk.get(g.chunk_id)
        if existing is None:
            by_chunk[g.chunk_id] = g
            continue
        existing.graph_rank = g.graph_rank
        existing.graph_score = g.graph_score
        if not existing.text and g.text:
            existing.text = g.text
        if not existing.article_id or existing.article_id == "unknown":
            existing.article_id = g.article_id
        if existing.seq is None:
            existing.seq = g.seq
        if g.places:
            existing.places = g.places

    for h in by_chunk.values():
        score = 0.0
        if h.vector_rank is not None:
            score += 1.0 / (rrf_k + h.vector_rank)
        if h.graph_rank is not None:
            score += 1.0 / (rrf_k + h.graph_rank)
        h.fused_score = score

    return sorted(by_chunk.values(), key=lambda x: x.fused_score, reverse=True)


def format_context(hits: List[Hit], budget_chars: int) -> str:
    used = 0
    blocks = []
    for i, h in enumerate(hits, start=1):
        tag_bits = [f"{h.article_id}:{h.chunk_id}"]
        if h.vector_rank is not None:
            tag_bits.append(f"v#{h.vector_rank}")
        if h.graph_rank is not None:
            tag_bits.append(f"g#{h.graph_rank}")
        if h.places:
            tag_bits.append("places=" + ", ".join(h.places[:3]))

        body = clip_text(h.text, max_chars=1800)
        block = f"[{i}] {' | '.join(tag_bits)}\n{body}"
        if used + len(block) > budget_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def print_retrieval_table(hits: List[Hit]) -> None:
    print("\nTop fused hits:")
    for h in hits:
        signals = []
        if h.vector_rank is not None:
            signals.append(f"v#{h.vector_rank}")
        if h.graph_rank is not None:
            signals.append(f"g#{h.graph_rank}")
        sig = ",".join(signals) if signals else "-"
        places = ", ".join(h.places[:2]) if h.places else "-"
        print(
            f"- {h.article_id}:{h.chunk_id} "
            f"(seq={h.seq if h.seq is not None else '?'}, {sig}, fused={h.fused_score:.4f}, places={places})"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hybrid", "vector", "graph"], default="hybrid")
    p.add_argument("--vector-k", type=int, default=8)
    p.add_argument("--graph-k", type=int, default=8)
    p.add_argument("--final-k", type=int, default=8)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--context-budget", type=int, default=14000)
    p.add_argument("--graph-index", default="chunkText")
    p.add_argument("--temperature", type=float, default=0.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    chat_model = os.getenv("CHAT_MODEL")
    chroma_dir = os.getenv("CHROMA_DIR", "chroma")
    collection = os.getenv("CHROMA_COLLECTION", "isaw_articles")

    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL is not set")
    if not chat_model:
        raise ValueError("CHAT_MODEL is not set")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    llm = ChatOllama(model=chat_model, base_url=ollama_base_url, temperature=args.temperature)

    neo4j_driver = None
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    if args.mode in {"hybrid", "graph"}:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD is not set")
        neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        neo4j_driver.verify_connectivity()

    print("GraphRAG chat ready. Type 'exit' to quit.")
    print(f"Mode={args.mode} vector_k={args.vector_k} graph_k={args.graph_k} final_k={args.final_k}\n")

    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break

            if neo4j_driver is not None:
                author_name = extract_author_count_name(q)
                if author_name:
                    agg = query_author_article_count(neo4j_driver, neo4j_database, author_name)
                    if agg is None:
                        print(f"\nI don't know. No Author node matched '{author_name}'.")
                    else:
                        author = agg["author"]
                        count = agg["articleCount"]
                        print(f"\n{author} wrote {count} ISAW article(s).")
                        ids = agg.get("articleIds") or []
                        if ids:
                            print("Matched articleIds:")
                            for aid in ids:
                                print(f"- {aid}")
                    print("\nSources:")
                    print("- neo4j:(Person:Author)-[:WROTE]->(:Article)")
                    print("\n" + "=" * 70 + "\n")
                    continue

                top_n = extract_top_authors_n(q)
                if top_n is not None:
                    rows = query_top_authors(neo4j_driver, neo4j_database, top_n)
                    if not rows:
                        print("\nI don't know. No author/article links found.")
                    else:
                        print(f"\nTop {top_n} author(s) by article count:")
                        for i, row in enumerate(rows, start=1):
                            print(f"{i}. {row['author']} ({row['articleCount']})")
                    print("\nSources:")
                    print("- neo4j:(Person:Author)-[:WROTE]->(:Article)")
                    print("\n" + "=" * 70 + "\n")
                    continue

                by_author = extract_articles_by_author_name(q)
                if by_author:
                    rows = query_articles_by_author(neo4j_driver, neo4j_database, by_author)
                    if rows is None:
                        print(f"\nI don't know. No Author node matched '{by_author}'.")
                    else:
                        print(f"\n{rows['author']} wrote {rows['articleCount']} article(s).")
                        for aid in rows.get("articleIds") or []:
                            print(f"- {aid}")
                    print("\nSources:")
                    print("- neo4j:(Person:Author)-[:WROTE]->(:Article)")
                    print("\n" + "=" * 70 + "\n")
                    continue

                place_name = extract_place_name_for_mentions(q)
                if place_name:
                    rows = query_articles_mentioning_place(neo4j_driver, neo4j_database, place_name)
                    if rows is None:
                        print(f"\nI don't know. No Place node matched '{place_name}'.")
                    else:
                        print(f"\n{rows['articleCount']} article(s) mention {rows['place']}.")
                        for aid in rows.get("articleIds") or []:
                            print(f"- {aid}")
                    print("\nSources:")
                    print("- neo4j:(Chunk)-[:MENTIONS]->(:Place), (:Article)-[:HAS_CHUNK]->(:Chunk)")
                    print("\n" + "=" * 70 + "\n")
                    continue

            v_hits: List[Hit] = []
            g_hits: List[Hit] = []

            if args.mode in {"hybrid", "vector"}:
                v_hits = vector_hits(vector_store, q, args.vector_k)
            if args.mode in {"hybrid", "graph"}:
                g_hits = graph_hits(neo4j_driver, neo4j_database, args.graph_index, q, args.graph_k)

            if args.mode == "vector":
                fused = v_hits
            elif args.mode == "graph":
                fused = g_hits
            else:
                fused = fuse_hits(v_hits, g_hits, rrf_k=args.rrf_k)

            top = fused[: args.final_k]
            if not top:
                print("\nNo retrieval hits.\n")
                continue

            print_retrieval_table(top)
            context = format_context(top, budget_chars=args.context_budget)
            prompt = (
                f"{SYSTEM_RULES}\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {q}\n\n"
                "ANSWER:"
            )
            answer = llm.invoke(prompt).content.strip()

            print("\n" + answer)
            print("\nSources:")
            for h in top:
                print(f"- {h.article_id}:{h.chunk_id}")
            print("\n" + "=" * 70 + "\n")
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()


if __name__ == "__main__":
    main()
