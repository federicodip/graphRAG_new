#!/usr/bin/env python3
"""
Ingest cleaned/pre-chunked JSONL files into a persistent Chroma DB.

Default expects rewritten chunks in: data/chunks_clean

Progress:
- prints per file
- prints every N chunks
- embeds/writes in batches
- if a batch fails, retries one-by-one and prints the offending chunkId + length

Run:
  python .\ingest_chunks.py --reset
  python .\ingest_chunks.py --chunks_dir data/chunks_clean --reset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_article_metadata(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not metadata_dir.exists():
        return meta_map

    for p in sorted(metadata_dir.glob("*.json")):
        try:
            obj = read_json(p)
            aid = str(obj.get("articleId", "")).strip()
            if aid:
                meta_map[aid] = obj
        except Exception:
            continue
    return meta_map


def make_doc(row: Dict[str, Any], jf: Path, meta_map: Dict[str, Dict[str, Any]]) -> Optional[Tuple[Document, str]]:
    article_id = str(row.get("articleId", "")).strip()
    chunk_id = str(row.get("chunkId", "")).strip()
    seq = row.get("seq", 0)
    text = str(row.get("text", "") or "").strip()

    if not article_id or not chunk_id or not text:
        return None

    m: Dict[str, Any] = dict(meta_map.get(article_id, {}))
    m.update(
        {
            "source": article_id,
            "articleId": article_id,
            "chunkId": chunk_id,
            "chunk": int(seq) if str(seq).isdigit() else seq,
            "file": str(jf),
        }
    )

    # carry through optional fields if present
    if "parentChunkId" in row:
        m["parentChunkId"] = row.get("parentChunkId")
    if "subseq" in row:
        m["subseq"] = row.get("subseq")

    return Document(page_content=text, metadata=m), chunk_id


def add_batch(vector_store: Chroma, docs, ids, jf_name: str) -> None:
    vector_store.add_documents(docs, ids=ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Delete and rebuild Chroma DB")
    ap.add_argument("--chunks_dir", default=None, help="Override chunks dir (default: data/chunks_clean)")
    ap.add_argument("--metadata_dir", default=None, help="Override metadata dir (default: data/articles_metadata)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--print_every", type=int, default=256)
    args = ap.parse_args()

    load_dotenv()

    repo_root = Path(__file__).resolve().parent
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else (repo_root / "data" / "chunks_clean")
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else (repo_root / "data" / "articles_metadata")

    chroma_dir = Path(os.getenv("CHROMA_DIR", "chroma"))
    collection = os.getenv("CHROMA_COLLECTION", "isaw_articles")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL is not set (e.g., mxbai-embed-large or nomic-embed-text)")

    if args.reset and chroma_dir.exists():
        print("[reset] deleting chroma")
        shutil.rmtree(chroma_dir)

    print(f"[config] chunks_dir={chunks_dir}")
    print(f"[config] metadata_dir={metadata_dir}")
    print(f"[config] chroma_dir={chroma_dir} collection={collection}")
    print(f"[config] ollama_base_url={ollama_base_url} embedding_model={embedding_model}")
    print(f"[config] batch_size={args.batch_size} print_every={args.print_every}")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)

    # Preflight call so you fail fast if Ollama/embeddings are misconfigured
    _ = embeddings.embed_query("test")

    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    meta_map = load_article_metadata(metadata_dir)
    print(f"[meta] loaded {len(meta_map)} article metadata records")

    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {chunks_dir}")

    total_ok = 0
    total_skipped = 0

    for jf in jsonl_files:
        print(f"\n[file] {jf.name}")
        batch_docs = []
        batch_ids = []
        file_ok = 0
        file_skipped = 0

        for row in iter_jsonl(jf):
            item = make_doc(row, jf, meta_map)
            if item is None:
                file_skipped += 1
                total_skipped += 1
                continue

            doc, cid = item
            batch_docs.append(doc)
            batch_ids.append(cid)

            if len(batch_docs) >= args.batch_size:
                try:
                    add_batch(vector_store, batch_docs, batch_ids, jf.name)
                except Exception as e:
                    print(f"[error] batch write failed ({jf.name}) size={len(batch_docs)}: {e}")
                    # retry one-by-one to reveal the offending chunk
                    for d, i in zip(batch_docs, batch_ids):
                        try:
                            vector_store.add_documents([d], ids=[i])
                        except Exception as ee:
                            L = len(d.page_content)
                            print(f"[bad] {jf.name} chunkId={i} chars={L} error={ee}")
                            # keep going; skip this one
                    # continue after isolating bad ones

                file_ok += len(batch_docs)
                total_ok += len(batch_docs)
                if total_ok % args.print_every == 0:
                    print(f"[progress] total_ingested={total_ok} (skipped={total_skipped})")

                batch_docs = []
                batch_ids = []

        # final batch
        if batch_docs:
            try:
                add_batch(vector_store, batch_docs, batch_ids, jf.name)
            except Exception as e:
                print(f"[error] final batch write failed ({jf.name}) size={len(batch_docs)}: {e}")
                for d, i in zip(batch_docs, batch_ids):
                    try:
                        vector_store.add_documents([d], ids=[i])
                    except Exception as ee:
                        L = len(d.page_content)
                        print(f"[bad] {jf.name} chunkId={i} chars={L} error={ee}")

            file_ok += len(batch_docs)
            total_ok += len(batch_docs)

        print(f"[file] done {jf.name}: ingested={file_ok} skipped={file_skipped}")

    try:
        vector_store.persist()
    except Exception:
        pass

    print("\nDone.")
    print(f"Chunks ingested: {total_ok}")
    print(f"Chunks skipped:  {total_skipped}")
    print(f"Chroma: {chroma_dir} | collection: {collection}")


if __name__ == "__main__":
    main()
