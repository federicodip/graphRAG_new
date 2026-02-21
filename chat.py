#!/usr/bin/env python3
"""
Simple CLI RAG chat:
- similarity search in Chroma
- pass retrieved chunks to Ollama chat model
- answer with Sources (articleId)

"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        chunk = d.metadata.get("chunk", "?")
        blocks.append(f"[{i}] SOURCE={src} CHUNK={chunk}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def list_source_chunks(docs: List[Document]) -> List[str]:
    seen, out = set(), []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        chunk = d.metadata.get("chunk", "?")
        key = (src, chunk)
        if key not in seen:
            seen.add(key)
            out.append(f"{src} (chunk {chunk})")
    return out


def main() -> None:
    load_dotenv()

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    chat_model = os.getenv("CHAT_MODEL")

    chroma_dir = os.getenv("CHROMA_DIR", "chroma")
    collection = os.getenv("CHROMA_COLLECTION", "isaw_articles")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    llm = ChatOllama(model=chat_model, base_url=ollama_base_url, temperature=0)

    system_rules = (
        "You are a careful assistant.\n"
        "Use ONLY the CONTEXT to answer.\n"
        "If the answer is not in the context, say: I don't know.\n"
    )

    print("Ask a question (type 'exit' to quit)\n")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        docs = vector_store.similarity_search(q, k=6)
        context = format_context(docs)

        prompt = (
            f"{system_rules}\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {q}\n\n"
            f"ANSWER:"
        )

        answer = llm.invoke(prompt).content.strip()

        print("\n" + answer)
        print("\nSources:")
        for s in list_source_chunks(docs):
            print(f"- {s}")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
