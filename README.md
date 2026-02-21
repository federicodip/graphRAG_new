# graphRAG_new

Local-first RAG + Neo4j “GraphRAG” scaffold for article/chunk corpora (initially built around ISAW-style article + chunk JSONL).

This repo contains:
- **Vector RAG**: Chroma (persistent) + Ollama embeddings + a simple CLI chat.
- **Graph layer** (Neo4j): `(:Article)`, `(:Chunk)`, optional `(:Place)` (Pleiades), `(:Person/:Author)`, `(:Entity)`, and utility scripts to connect them.

> Note: this repository does **not** ship the corpus. `data/` and `chroma/` are gitignored.

---

## Features

### Vector RAG (local)
- Ingest pre-chunked JSONL into a persistent **Chroma** collection using **Ollama embeddings**.
- Query via a minimal CLI (`chat.py`) that retrieves top‑k chunks and prompts an Ollama chat model.

### GraphRAG scaffold (Neo4j)
- Load `Article` + `Chunk` into Neo4j and connect:
  - `(:Article)-[:HAS_CHUNK]->(:Chunk)`
  - optional `(:Chunk)-[:NEXT]->(:Chunk)` (sequence)
  - optional `(:Chunk)-[:SUBCHUNK_OF]->(:Chunk)` (when you split chunks)
- Ingest **Pleiades places** into `(:Place)` and link:
  - `(:Chunk)-[:MENTIONS]->(:Place)` (high-precision FlashText linker)
- Create/merge author nodes and link:
  - `(:Person:Author)-[:WROTE]->(:Article)`

---

## Repository layout

```
.
├─ chat.py
├─ ingest_chunks.py
├─ rewrite_chunks.py
├─ load_neo4j.py
├─ schema_neo4j.py
├─ ingest_pleiades.py
├─ link_places_mentions_flashtext.py
├─ init_people_entities.py
├─ link_authors_to_articles.py
├─ create_qa.py
├─ generate_qa_pairs.py
├─ check_chroma_count.py
├─ data/                  # NOT COMMITTED (gitignored)
│  ├─ chunks/             # input chunks (JSONL)
│  ├─ chunks_clean/       # cleaned/split chunks (JSONL)
│  ├─ articles_metadata/  # optional article metadata (JSON per articleId)
│  ├─ pleiades/           # optional Pleiades dump(s)
│  └─ ...
└─ chroma/                # NOT COMMITTED (gitignored)
```


`.gitignore` includes: `.env`, `venv/`, `data/`, `chroma/`.

---

## Requirements

- Python 3.10+ recommended
- **Ollama** running locally (for embeddings and/or chat)
- Optional: **Neo4j** (Desktop or Server) if you want the graph layer

Install Python dependencies:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> You may need to install additional packages depending on your environment (e.g., `python-dotenv`, `langchain-chroma`, `langchain-ollama`), since `requirements.txt` is intentionally minimal.

---

## Configuration

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

Example `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
CHAT_MODEL=llama3.2:3b
EMBEDDING_MODEL=mxbai-embed-large

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=CHANGE_ME
NEO4J_DATABASE=neo4j

# OpenAI (only if you use generate_qa_pairs.py)
OPENAI_API_KEY=CHANGE_ME
```

---

## Data formats

### Chunk JSONL (input)
Each line in `data/chunks/*.jsonl` should look like:

```json
{"articleId":"isaw_paper11","chunkId":"isaw_paper11:0019","seq":19,"text":"..."}
```

### Chunk JSONL (cleaned/split output)
`rewrite_chunks.py` produces `data/chunks_clean/*.jsonl` and may emit split subchunks:

```json
{"articleId":"...","chunkId":"orig::s00","seq":19,"text":"...","parentChunkId":"orig","subseq":0}
```

### Optional article metadata JSON
If present in `data/articles_metadata/*.json`, files should include `articleId` plus any of:

```json
{"articleId":"isaw_paper11","title":"...","year":2016,"journal":"ISAW Papers","url":"...","authors":["A. Author","B. Author"]}
```

---

## Quickstart: Vector RAG (local)

### 1) Clean/split chunks
```bash
python rewrite_chunks.py --in_dir data/chunks --out_dir data/chunks_clean --max_chars 850 --overlap 120
```

### 2) Ingest into Chroma
```bash
python ingest_chunks.py --reset
# or:
python ingest_chunks.py --chunks_dir data/chunks_clean --reset
```

### 3) Sanity-check the vector store
```bash
python check_chroma_count.py
```

### 4) Chat
```bash
python chat.py
```

---

## Quickstart: Neo4j graph layer

### 0) Start Neo4j
Start Neo4j and ensure `NEO4J_*` values in `.env` are correct.

### 1) Create schema (constraints + fulltext index)
```bash
python schema_neo4j.py
```

### 2) (Optional) Create Person/Entity schema + seed nodes
```bash
python init_people_entities.py
# or seed persons from (:Article).authors after loading articles:
python init_people_entities.py --seed_from_article_authors
```

### 3) Load Articles + Chunks
```bash
python load_neo4j.py --reset --chunks_dir data/chunks_clean --metadata_dir data/articles_metadata
```

This creates:
- `(:Article {articleId, title, year, journal, url, authors})`
- `(:Chunk {chunkId, articleId, seq, text, parentChunkId?, subseq?, file?})`
- `(:Article)-[:HAS_CHUNK]->(:Chunk)`
- optional `:NEXT` and `:SUBCHUNK_OF` relationships

### 4) Ingest Pleiades places (optional)
```bash
python ingest_pleiades.py --input data/pleiades/<your-pleiades-dump>.json.gz --reset
```

This creates:
- `(:Place {pleiadesId, uri, title, altNames, reprLat, reprLon})`

### 5) Link place mentions (high-precision dictionary linker)
```bash
python link_places_mentions_flashtext.py
# optionally limit chunks:
python link_places_mentions_flashtext.py --limit-chunks 500
```

Creates:
- `(:Chunk)-[:MENTIONS]->(:Place)` with audit properties (matched surface, counts, timestamps, method).

### 6) Link authors to articles
```bash
python link_authors_to_articles.py
# optionally:
python link_authors_to_articles.py --reset_wrote
```

Creates/merges:
- `(:Person:Author {personId, name, nameNorm, nameKey, ...})-[:WROTE]->(:Article)`

---

## LLM extraction utilities (optional)

### Place extraction via Ollama (no Wikidata)
Generates a JSONL file with extracted place surfaces per chunk, using a local Pleiades index:

```bash
python ollama_extract_from_chunks.py --limit 500
python ollama_extract_from_chunks.py --limit 0 --resume
```

### Wikidata entity extraction + linking (requires internet)
Extracts non-place entities and optionally links them via Wikidata search API:

```bash
python ollama_extract_wd_from_chunks.py --limit 40 --verbose
python ollama_extract_wd_from_chunks.py --limit 0 --resume
# or extract-only (no Wikidata calls):
python ollama_extract_wd_from_chunks.py --extract_only
```

Outputs:
- `data/extractions_wikidata_ollama.jsonl`

---

## QA dataset generation (optional)

### Local “one question per chunk” (Ollama)
```bash
python create_qa.py --chunks_dir data/chunks --n_per_file 6 --out data/generated_questions.csv --seed 42
```

### Citation-grounded QA pairs (OpenAI API; requires internet)
```bash
python generate_qa_pairs.py --chunks_dir data/chunks_clean --out data/qa_pairs.jsonl --n 400 --pair_ratio 0.30
```


