# Research RAG

**Section-aware retrieval-augmented generation over research papers — fully offline.**

Drop a folder of PDFs in, ask questions in natural language, and get answers grounded
in specific sections of specific papers — with the source passage rendered back on the
original PDF page. No API keys, no cloud calls: parsing, embedding, reranking, and
generation all run locally.

```
"What datasets were used for Arabic offensive language detection?"

  → classified as a `dataset` question
  → matched to the "Data Description and Pre-processing" heading in 3 papers
  → searched only within those sections
  → reranked, then summarised per paper by a local Llama model
```

---

## Why this is not another naive RAG demo

Most RAG-over-PDF projects flatten a document into fixed-size chunks and hope cosine
similarity finds the right one. Research papers have structure worth keeping, and the
whole design here is built around exploiting it.

| Problem with naive RAG | What this project does |
| --- | --- |
| Fixed-size chunks split tables and paragraphs mid-thought | [Docling](https://github.com/DS4SD/docling) parses layout and a `HierarchicalChunker` splits on real document structure |
| Chunks lose their place in the document | Every chunk stores its `section_name`, `subsection_name`, and page numbers |
| "What was the methodology?" retrieves the abstract, which *mentions* methodology | Queries are classified to a section type, then search is **restricted** to matching sections |
| Papers number sections inconsistently (`3.1`, `A.`, `III.`) | Numbering convention is auto-detected per paper (decimal / IEEE / alphabetic) and orphaned subsections are re-parented |
| References and ethics statements pollute results | Boilerplate sections are dropped at ingest time via a blacklist plus a title-detection heuristic |
| Top-k vector hits are often loosely relevant | A cross-encoder reranks 4× the candidates down to the best `k` |
| Answers are hard to trust | Each answer links to the exact passage, rendered onto its original PDF page |

---

## Architecture

```
        PDF
         │
         ▼
   ┌──────────────┐   layout-aware parsing + hierarchical chunking
   │   Docling    │
   └──────┬───────┘
          │  chunks + headings
          ▼
   ┌──────────────────────────────────┐   3-stage section classification:
   │      Section classifier          │   1. keyword match
   │  (keywords → LLM → content)      │   2. LLM batch classify ambiguous
   └──────┬───────────────────────────┘   3. verify against first paragraph
          │  section_name / section_type / pages
          ▼
   ┌──────────────┐   BAAI/bge-small-en-v1.5, 384-dim, normalised
   │   Embedder   │
   └──────┬───────┘
          │  vectors + metadata
          ▼
   ┌──────────────────────────────────┐
   │   Weaviate  (Docker container)   │   HNSW, cosine, self-provided vectors
   └──────┬───────────────────────────┘   exact-match filters on section fields
          │
          ▼  ── query path ──────────────────────────────────────────────
   ┌──────────────┐   classify query → match to stored headings → filter
   │ Hierarchical │   → vector search within section → global fallback
   │   retrieval  │
   └──────┬───────┘
          ▼
   ┌──────────────┐   cross-encoder/ms-marco-MiniLM-L-6-v2
   │   Reranker   │
   └──────┬───────┘
          ▼
   ┌──────────────┐   per-paper grounded summary
   │    Ollama    │   (quick search, or map-reduce "deep scan")
   └──────┬───────┘
          ▼
   FastAPI + web UI, with PyMuPDF rendering the source page
```

### Two retrieval modes

**Quick search** — classify the query, filter to the matching sections, vector search,
rerank, summarise. Fast; good for pointed questions.

**Deep scan** — identify the target section, match it to every paper's real heading,
then map-reduce *every* chunk in that section: summarise each chunk, then reduce to one
answer per paper. Slower but exhaustive; good for "compare the methodology across all
papers"-style questions.

### Retrieval safety rails

Section filtering is enforced in three independent places, because an LLM classifier
alone is not trustworthy enough to silently drop content:

1. **Cross-validation at match time** — a heading the LLM matched is discarded if its
   dominant ingest-time `section_type` contradicts the query's target type.
2. **Database-level filter** — Weaviate filters on `section_name` with `FIELD`
   tokenization, so `"Data Collection"` matches that heading exactly and never a
   section merely containing the word *Data*.
3. **Post-retrieval filter in Python** — hits are re-checked against the allowed set
   regardless of what the database returned.

If a section turns out to be empty, retrieval falls back to a global search rather
than returning nothing.

---

## Quickstart

### Prerequisites

- **Python 3.11+**
- **Docker** — runs the Weaviate vector database
- **[Ollama](https://ollama.com)** — runs the local LLM

```bash
ollama pull llama3.2          # ~2 GB; any instruct model works
```

### Install

```bash
git clone https://github.com/<your-username>/research-rag.git
cd research-rag

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Start the vector database

```bash
docker compose up -d
```

Weaviate is published on **8081** (REST) and **50052** (gRPC) rather than the usual
8080/50051, so it does not collide with another Weaviate you may already be running.
Override with `WEAVIATE_PORT` / `WEAVIATE_GRPC_PORT` in a `.env` file — `docker
compose` and the app both read them.

Check it's healthy:

```bash
docker compose ps             # should show "healthy"
curl http://localhost:8081/v1/.well-known/ready
```

### Ingest papers and ask questions

```bash
mkdir -p papers && cp /path/to/*.pdf papers/

python -m research_rag.cli ingest papers/
python -m research_rag.cli list
python -m research_rag.cli sections --paper mypaper.pdf
python -m research_rag.cli search "What datasets were used?"
```

### Or use the web UI

```bash
python app.py                 # opens http://localhost:7860
```

Upload PDFs, chat over them, filter to a single paper, toggle deep scan and reranking,
and click any cited passage to see it highlighted on the original PDF page.

---

## Configuration

Every setting is an environment variable with a sensible default — see
[`.env.example`](.env.example). Copy it to `.env` to override:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEAVIATE_HOST` / `WEAVIATE_PORT` / `WEAVIATE_GRPC_PORT` | `localhost` / `8081` / `50052` | Where the database lives |
| `WEAVIATE_COLLECTION` | `ResearchChunk` | Collection name |
| `OLLAMA_MODEL` | `llama3.2:latest` | Generation + classification model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim sentence embeddings |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `TOP_K` | `5` | Chunks passed to the LLM |
| `RERANK_FETCH_MULTIPLIER` | `4` | Candidates fetched per final chunk |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `7860` | Web app bind address |

Changing `EMBEDDING_MODEL` changes the vector dimension — wipe and re-ingest
(`docker compose down -v`) when you do.

---

## Project layout

```
research_rag/
  config.py              env-driven settings, single source of truth
  ingest.py              Docling conversion, heading repair, boilerplate filtering
  section_classifier.py  3-stage heading classification + query → section matching
  embedder.py            sentence-transformers wrapper
  vector_store.py        Weaviate client: schema, writes, filtered vector search
  search.py              quick-search pipeline with the section safety rails
  map_reduce.py          deep-scan pipeline
  reranker.py            cross-encoder reranking
  pdf_viewer.py          renders a chunk back onto its PDF page
  cli.py                 ingest / search / list / sections / delete
app.py                   FastAPI server + JSON API
static/index.html        single-page web UI
scripts/
  migrate_chroma_to_weaviate.py    one-time import from a legacy ChromaDB index
docker-compose.yml       Weaviate service
```

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness plus paper/chunk counts |
| `GET` | `/api/papers` | List ingested papers |
| `POST` | `/api/papers` | Upload and ingest PDFs |
| `DELETE` | `/api/papers/{filename}` | Remove a paper and its chunks |
| `POST` | `/api/chat` | Ask a question (quick search or deep scan) |
| `POST` | `/api/render-chunk` | Render a chunk on its source PDF page |

---

## Data model

One Weaviate collection, self-provided vectors (embeddings are computed locally, so no
vectorizer module is enabled on the server):

| Property | Type | Notes |
| --- | --- | --- |
| `text` | `text` | Chunk content |
| `source_file` | `text` | `FIELD` tokenization — exact filename match |
| `source_path` | `text` | Not indexed; used for PDF rendering |
| `chunk_index` | `int` | Position in the document |
| `heading` | `text` | Full breadcrumb, e.g. `3 Methods > 3.1 Data` |
| `section_name` | `text` | `FIELD` tokenization — exact heading match |
| `subsection_name` | `text` | `FIELD` tokenization |
| `section_type` | `text` | One of `abstract`, `introduction`, `related_work`, `methodology`, `dataset`, `results`, `conclusion`, `general` |
| `page_numbers` | `text` | Comma-separated source pages |

Object UUIDs are a deterministic `uuid5` of `source_file::chunk_index`, so re-ingesting
a paper overwrites its chunks instead of duplicating them.

---

## Migrating from ChromaDB

Earlier versions of this project used an embedded ChromaDB store. To carry an existing
index over without re-running Docling and the LLM classifier on every paper:

```bash
pip install chromadb                                  # only needed for the migration
docker compose up -d
python scripts/migrate_chroma_to_weaviate.py --dry-run
python scripts/migrate_chroma_to_weaviate.py
```

It copies chunks, metadata, and the already-computed embeddings straight across, and is
safe to re-run. Once verified, `chroma_data/` can be deleted.

---

## Notes & limitations

- **Not a general PDF chatbot.** It assumes academic papers with recognisable section
  headings. Slide decks, scanned documents without OCR, and reports with no headings
  degrade to plain semantic search.
- **Section classification uses an LLM** and is therefore fallible. The cross-validation
  and post-retrieval filters above exist to contain that, not to eliminate it.
- **Ingestion is slow** — Docling layout analysis plus per-heading LLM classification
  runs roughly 1–3 minutes per paper on CPU. Retrieval afterwards is fast.
- **Anonymous access is enabled** on the Weaviate container. That is fine for a local,
  single-user setup; add authentication before exposing it on a network.
- Papers are not committed to this repository — published articles are copyrighted.
  `papers/` is gitignored; supply your own PDFs.

---

## License

MIT — see [LICENSE](LICENSE).
