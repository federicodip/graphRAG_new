#!/usr/bin/env python3
"""
Batch QA runner that queries:
1) Base model (OpenAI, e.g. gpt-5-mini)
2) Vector RAG (Chroma + Ollama)
3) Hybrid GraphRAG (Chroma + Neo4j + Ollama)

Input: CSV with at least a `question` column (defaults to data/qa_pairs.csv)
Output: flat CSV with answers from each system (defaults to data/qa_model_outputs.csv)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j import GraphDatabase

from graph_chat import SYSTEM_RULES, format_context, fuse_hits, graph_hits, vector_hits


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def invoke_ollama(llm: ChatOllama, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = normalize_text(item.get("text"))
                if txt:
                    parts.append(txt)
            else:
                txt = normalize_text(item)
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()
    return normalize_text(content)


def build_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_RULES}\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def answer_with_vector_rag(
    question: str,
    vector_store: Chroma,
    llm: ChatOllama,
    vector_k: int,
    final_k: int,
    context_budget: int,
) -> Dict[str, Any]:
    hits = vector_hits(vector_store, question, vector_k)
    top = hits[:final_k]
    if not top:
        return {
            "answer": "I don't know.",
            "sources": "",
            "hit_count": 0,
        }

    context = format_context(top, budget_chars=context_budget)
    answer = invoke_ollama(llm, build_prompt(question, context))
    if not answer:
        answer = "I don't know."

    return {
        "answer": answer,
        "sources": ";".join(f"{h.article_id}:{h.chunk_id}" for h in top),
        "hit_count": len(top),
    }


def answer_with_hybrid_graphrag(
    question: str,
    vector_store: Chroma,
    llm: ChatOllama,
    neo4j_driver,
    neo4j_database: str,
    graph_index: str,
    vector_k: int,
    graph_k: int,
    final_k: int,
    rrf_k: int,
    context_budget: int,
) -> Dict[str, Any]:
    v_hits = vector_hits(vector_store, question, vector_k)
    g_hits = graph_hits(neo4j_driver, neo4j_database, graph_index, question, graph_k)
    fused = fuse_hits(v_hits, g_hits, rrf_k=rrf_k)
    top = fused[:final_k]

    if not top:
        return {
            "answer": "I don't know.",
            "sources": "",
            "hit_count": 0,
        }

    context = format_context(top, budget_chars=context_budget)
    answer = invoke_ollama(llm, build_prompt(question, context))
    if not answer:
        answer = "I don't know."

    return {
        "answer": answer,
        "sources": ";".join(f"{h.article_id}:{h.chunk_id}" for h in top),
        "hit_count": len(top),
    }


def answer_with_base_model(question: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    sys_prompt = (
        "You are a QA assistant. Be concise and precise. "
        "Return 1-2 sentences, no bullets. "
        "Use only information you are confident about; if uncertain, say 'I don't know.'"
    )
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ],
        )
        text = normalize_text(getattr(response, "output_text", ""))
        if text:
            return text
    except Exception:
        pass

    # Backward-compatible fallback for older SDK/API paths.
    chat_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    text = normalize_text(chat_resp.choices[0].message.content)
    return text if text else "I don't know."


def read_existing_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()

    seen: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_text(row.get("id")) or normalize_text(row.get("row_index"))
            if key:
                seen.add(key)
    return seen


def read_existing_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_text(row.get("id")) or normalize_text(row.get("row_index"))
            if key:
                out[key] = row
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-run QA over base model, RAG, and GraphRAG")
    p.add_argument("--input-csv", default="data/qa_pairs.csv")
    p.add_argument("--output-csv", default="data/qa_model_outputs.csv")
    p.add_argument("--limit", type=int, default=0, help="0 means all rows")
    p.add_argument("--start-row", type=int, default=0, help="0-based row offset")
    p.add_argument("--resume", action="store_true", help="Skip rows already present in output CSV")
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true", help="Print per-question progress")
    p.add_argument("--log-every", type=int, default=25, help="Progress interval when not verbose")
    p.add_argument(
        "--run-target",
        choices=["all", "base", "local"],
        default="all",
        help="all=base+rag+graphrag, base=openai only, local=rag+graphrag only",
    )

    p.add_argument("--base-model", default="gpt-5")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-rag", action="store_true")
    p.add_argument("--skip-graphrag", action="store_true")

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
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    chat_model = os.getenv("CHAT_MODEL")
    chroma_dir = os.getenv("CHROMA_DIR", "chroma")
    collection = os.getenv("CHROMA_COLLECTION", "isaw_articles")

    do_base = args.run_target in {"all", "base"} and not args.skip_base
    do_local = args.run_target in {"all", "local"}
    do_rag = do_local and not args.skip_rag
    do_graphrag = do_local and not args.skip_graphrag
    if not (do_base or do_rag or do_graphrag):
        raise ValueError("Nothing to run: check --run-target / --skip-* options")

    needs_rag = do_rag or do_graphrag
    if needs_rag and not embedding_model:
        raise ValueError("EMBEDDING_MODEL is not set")
    if needs_rag and not chat_model:
        raise ValueError("CHAT_MODEL is not set")

    vector_store: Optional[Chroma] = None
    llm: Optional[ChatOllama] = None
    if needs_rag:
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
        vector_store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        llm = ChatOllama(model=chat_model, base_url=ollama_base_url, temperature=args.temperature)

    neo4j_driver = None
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    if do_graphrag:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD is not set")
        neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        neo4j_driver.verify_connectivity()

    seen_ids = read_existing_ids(output_csv) if args.resume else set()
    existing_rows = read_existing_rows(output_csv)
    log_every = max(1, int(args.log_every))

    fieldnames = [
        "row_index",
        "id",
        "key",
        "difficulty",
        "style",
        "question",
        "gold_answer",
        "base_model",
        "base_answer",
        "rag_answer",
        "graphrag_answer",
        "rag_sources",
        "graphrag_sources",
        "rag_hit_count",
        "graphrag_hit_count",
        "error",
    ]

    if output_csv.parent:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    updated = 0
    skipped_resume = 0
    total_rows: Optional[int] = None

    if args.verbose:
        with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
            total_rows = max(0, sum(1 for _ in f) - 1)
        print(
            "Starting batch run "
            f"(input={input_csv}, output={output_csv}, start_row={args.start_row}, "
            f"limit={args.limit or 'all'}, total_rows={total_rows})"
        )

    try:
        selected_ids: List[str] = []
        input_rows_by_id: Dict[str, Dict[str, Any]] = {}

        with input_csv.open("r", encoding="utf-8-sig", newline="") as in_f:
            reader = csv.DictReader(in_f)
            for idx, row in enumerate(reader):
                if idx < args.start_row:
                    continue
                if args.limit > 0 and processed >= args.limit:
                    break

                qid = normalize_text(row.get("id")) or str(idx)
                input_rows_by_id[qid] = row
                selected_ids.append(qid)
                prev = existing_rows.get(qid, {})

                if args.resume and qid in seen_ids:
                    base_done = (not do_base) or bool(normalize_text(prev.get("base_answer")))
                    rag_done = (not do_rag) or bool(normalize_text(prev.get("rag_answer")))
                    graph_done = (not do_graphrag) or bool(normalize_text(prev.get("graphrag_answer")))
                    if base_done and rag_done and graph_done:
                        processed += 1
                        skipped_resume += 1
                        continue

                question = normalize_text(row.get("question"))
                if not question:
                    processed += 1
                    continue
                row_start = time.perf_counter()
                if args.verbose:
                    q_preview = (question[:96] + "...") if len(question) > 96 else question
                    print(f"[row={idx} id={qid}] running: {q_preview}")

                out_row: Dict[str, Any] = {
                    "row_index": idx,
                    "id": qid,
                    "key": normalize_text(row.get("key")),
                    "difficulty": normalize_text(row.get("difficulty")),
                    "style": normalize_text(row.get("style")),
                    "question": question,
                    "gold_answer": normalize_text(row.get("answer")),
                    "base_model": normalize_text(prev.get("base_model")),
                    "base_answer": normalize_text(prev.get("base_answer")),
                    "rag_answer": normalize_text(prev.get("rag_answer")),
                    "graphrag_answer": normalize_text(prev.get("graphrag_answer")),
                    "rag_sources": normalize_text(prev.get("rag_sources")),
                    "graphrag_sources": normalize_text(prev.get("graphrag_sources")),
                    "rag_hit_count": normalize_text(prev.get("rag_hit_count")) or 0,
                    "graphrag_hit_count": normalize_text(prev.get("graphrag_hit_count")) or 0,
                    "error": normalize_text(prev.get("error")),
                }

                errors: List[str] = []
                if out_row["error"]:
                    errors.append(out_row["error"])

                if do_base:
                    try:
                        out_row["base_model"] = args.base_model
                        out_row["base_answer"] = answer_with_base_model(question, args.base_model)
                    except Exception as exc:
                        errors.append(f"base={exc}")

                if do_rag:
                    try:
                        rag_result = answer_with_vector_rag(
                            question=question,
                            vector_store=vector_store,
                            llm=llm,
                            vector_k=args.vector_k,
                            final_k=args.final_k,
                            context_budget=args.context_budget,
                        )
                        out_row["rag_answer"] = rag_result["answer"]
                        out_row["rag_sources"] = rag_result["sources"]
                        out_row["rag_hit_count"] = rag_result["hit_count"]
                    except Exception as exc:
                        errors.append(f"rag={exc}")

                if do_graphrag:
                    try:
                        graph_result = answer_with_hybrid_graphrag(
                            question=question,
                            vector_store=vector_store,
                            llm=llm,
                            neo4j_driver=neo4j_driver,
                            neo4j_database=neo4j_database,
                            graph_index=args.graph_index,
                            vector_k=args.vector_k,
                            graph_k=args.graph_k,
                            final_k=args.final_k,
                            rrf_k=args.rrf_k,
                            context_budget=args.context_budget,
                        )
                        out_row["graphrag_answer"] = graph_result["answer"]
                        out_row["graphrag_sources"] = graph_result["sources"]
                        out_row["graphrag_hit_count"] = graph_result["hit_count"]
                    except Exception as exc:
                        errors.append(f"graphrag={exc}")

                out_row["error"] = " | ".join(errors)
                existing_rows[qid] = out_row

                processed += 1
                written += 1
                updated += 1
                elapsed = time.perf_counter() - row_start

                if args.verbose:
                    status = "ok" if not errors else f"errors={out_row['error']}"
                    print(f"[row={idx} id={qid}] done in {elapsed:.2f}s ({status})")
                elif written % log_every == 0:
                    print(f"Progress: wrote {written} rows (last id={qid}, {elapsed:.2f}s)")

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        with output_csv.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            # Preserve input order for rows selected now; keep prior rows if present and not selected.
            selected_set = set(selected_ids)
            for qid in selected_ids:
                row_out = existing_rows.get(qid)
                if row_out:
                    writer.writerow(row_out)
            for qid, row_out in existing_rows.items():
                if qid not in selected_set:
                    writer.writerow(row_out)

        print(
            f"Done. Updated {updated} rows, skipped_by_resume={skipped_resume}, "
            f"wrote {len(existing_rows)} total rows to {output_csv}"
        )
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()


if __name__ == "__main__":
    main()
