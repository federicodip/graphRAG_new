#!/usr/bin/env python3
"""
Prepare Neo4j schema for GraphRAG-ISAW:

Constraints:
- Article.articleId UNIQUE
- Chunk.chunkId UNIQUE
- Place.pleiadesId UNIQUE
- WikidataEntity.qid UNIQUE

Indexes:
- fulltext index over Chunk.text for linker candidate retrieval
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


def main() -> None:
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        raise ValueError("NEO4J_PASSWORD is not set")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    statements = [
        # Uniqueness constraints
        "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.articleId IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
        "CREATE CONSTRAINT place_id IF NOT EXISTS FOR (p:Place) REQUIRE p.pleiadesId IS UNIQUE",
        "CREATE CONSTRAINT wd_qid IF NOT EXISTS FOR (w:WikidataEntity) REQUIRE w.qid IS UNIQUE",

        # Fulltext index for linker (Neo4j 5+)
        "CREATE FULLTEXT INDEX chunkText IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    ]

    with driver.session(database=database) as session:
        for s in statements:
            session.execute_write(lambda tx, q=s: tx.run(q).consume())

    driver.close()
    print("[neo4j] schema ready (constraints + fulltext index chunkText)")


if __name__ == "__main__":
    main()
