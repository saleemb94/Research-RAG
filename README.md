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
| Section vocabulary is field-specific — `corpus`, `cohort`, `specimen` and `primary sources` all mean *the data* | Classification is by section **function**, with vocabulary from nine disciplines and a deliberate refusal to guess on words that flip meaning between fields |

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
than returning nothing — and the filters in step 3 are **switched off on that
fallback path**. That detail is not incidental: applying an allow-list for a section
the paper does not have deletes every fallback hit and returns an empty answer while
the store is holding perfectly good chunks. It shipped that way until the golden
question set caught it, and `tests/test_retrieval_filters.py` now pins it.

**Scope decides how sections are matched.** Inside one paper, the target is matched
onto that paper's real headings, so "methodology" finds a section actually called
"Our Approach". Across the corpus that would be wrong: heading lists get pooled, the
model picks names that exist in only one or two papers, and filtering on them makes
every other paper unreachable. So a corpus-wide query filters on `section_type`
instead, which is assigned per chunk at ingest precisely so that it generalises.

### Working across disciplines

Papers are not all from one field, so the classifier is built around section
**function** rather than any field's vocabulary.

The eight labels (`abstract`, `introduction`, `theory`, `related_work`, `methodology`,
`dataset`, `results`, `conclusion`) describe roles that recur across empirical research
regardless of subject. `theory` covers papers that are not empirical at all — theorems
and proofs in mathematics, derivations in physics, formal models in economics,
conceptual frameworks in the social sciences.

Each label carries vocabulary from several fields, because the same role is named
differently everywhere:

| Role | Computing | Medicine | Social science | Humanities |
| --- | --- | --- | --- | --- |
| `dataset` | corpus, benchmark, training set | cohort, study population, inclusion criteria | participants, respondents, sample | primary sources, archival material |
| `methodology` | proposed method, architecture | trial protocol, randomization, assay | interview protocol, coding procedure | fieldwork, ethnography |
| `related_work` | related work, prior art | systematic review, meta-analysis | literature review | historiography |

**Where it deliberately refuses to guess.** Some headings mean different things in
different fields, and a confident wrong answer from the fast keyword stage would
prevent the later stages from ever reading the section's text. Those headings are
listed in `_DEFER_TO_CONTENT` and returned as unclassified on purpose, which routes
them to the LLM stages that *can* read the content:

| Heading | In computing | In another field |
| --- | --- | --- |
| "Statistical Analysis" | reported findings | the analysis plan, inside Methods (medicine) |
| "Survey" | a literature survey | a questionnaire instrument (sociology) |
| "Materials" | — | reagents (chemistry), stimuli (psychology), or half of "Materials and Methods" |
| "Model" | an implemented system | a formal model (economics, theory) |
| "Baseline Characteristics" | a comparison result | a description of the participants (clinical trials) |

That last one used to be classified as a *result*, which is simply wrong for a clinical
paper — it describes who was enrolled.

### Choosing a model

The summarisation step is forgiving — most instruct models write something reasonable
from retrieved text. The *routing* steps are not: they decide which sections get
searched at all, so an error there silently changes the answer without looking like a
failure. `scripts/eval_models.py` scores exactly those three call sites against ground
truth spanning nine disciplines:

```bash
python scripts/eval_models.py                      # every installed model
python scripts/eval_models.py qwen3:8b --think     # measure reasoning mode
python scripts/eval_models.py -v qwen2.5:7b        # show every wrong answer
```

Measured on an RTX 5060 Laptop (8 GB), 50 graded items:

| Model | Headings | Query | Target | Overall | Time |
| --- | --- | --- | --- | --- | --- |
| **qwen3:4b-instruct** (default) | 96% | 100% | 100% | **98%** | **5.5 s** |
| qwen3:8b (reasoning off) | 92% | 100% | 100% | 96% | 9.0 s |
| qwen2.5:7b | 88% | 95% | 100% | 92% | 10.9 s |
| qwen3:8b (reasoning on) | 92% | 90% | 100% | 92% | 133.5 s |
| llama3.2:3b | 33% | 90% | 83% | 62% | 10.1 s |

Three things that surprised me and are worth knowing:

- **Bigger was not better.** `qwen3:4b-instruct` beat `qwen3:8b` on accuracy *and* was
  faster. These tasks reward instruction-following on a fixed label set, not knowledge.
- **Reasoning actively hurt.** With thinking enabled, `qwen3:8b` took 15x longer and
  scored *lower* (92% vs 96%) — it deliberates its way out of correct one-word answers,
  spending ~715 tokens where 2 would do. Reasoning is disabled by default for this
  reason; set `OLLAMA_THINK=true` to override.
- **Heading classification is where models separate.** Every model handles query
  routing well; only the good ones classify unfamiliar headings such as
  "Baseline Characteristics" or "Historiography" correctly. That step runs at ingest
  and every later retrieval inherits its mistakes.

Note that the 2507 "Instruct" (non-thinking) refresh only exists for 4B, 30B-A3B and
235B — there is no `qwen3:8b-instruct`. At 8B you get the hybrid model, so disable
reasoning explicitly. Ollama returns reasoning in a separate `thinking` field, so it
never corrupts parsed output; the cost is latency, not correctness.

All Ollama traffic goes through [`research_rag/llm.py`](research_rag/llm.py), which
applies this policy in one place.

### Measuring retrieval and generation quality

`tests/golden_qa.json` holds 43 questions over the 11 sample papers, with 111 fact
groups that a correct answer must contain. Every fact was read from the source PDF,
not from the search index, so the set also detects content lost during ingestion.

A RAG system can fail in three separate places and one end-to-end number hides which
one broke, so the scorer reports them apart:

```bash
python scripts/eval_rag.py --coverage      # is the fact in the index at all? (no LLM)
python scripts/eval_rag.py --retrieval     # asked blind, does the right paper come back?
python scripts/eval_rag.py --generation    # given the right paper, is the fact stated?
python scripts/eval_rag.py --synthesis   # corpus-wide cited answers
python scripts/eval_rag.py --all -v
```

`--coverage` is the one to run after changing anything in ingestion: it is instant,
needs no LLM, and a drop there means content was dropped or mangled before retrieval
ever got a chance.

The corpus is 17 papers — 11 supplied plus 6 fetched from arXiv with
`scripts/fetch_arxiv.py` (ids pinned in `papers/arxiv_manifest.json`). The arXiv set
deliberately spans cs.CL, q-bio.PE, stat.AP and econ.EM, so the discipline-agnostic
classifier is exercised on real epidemiology, clinical-trial and econometrics papers
rather than assumed to work.

Current scores, and what the set caught on its first run:

| | before | after |
| --- | --- | --- |
| Coverage — fact is in the index | 99% | **100%** |
| Retrieval — right paper returned | 72% | **81%** |
| Retrieval — right paper ranked first | 56% | **70%** |
| Routing — right section targeted | 46% | **78%** |
| Generation — fact stated in the answer | 59% | **79%** |
| Generation — fully correct answers | 49% | **71%** |
| Synthesis — facts stated in one cited answer | 61% | **89%** |
| Synthesis — expected papers actually cited | 50% | **71%** |

The "before" column is not a weaker model; it is the same pipeline with three silent
bugs that only graded ground truth could surface — heading matching pooled across
papers, post-retrieval filters cancelling the global fallback, and abstracts dropped
as boilerplate. Each produced a fluent, plausible answer while discarding correct
content, which is exactly the failure mode eyeballing output cannot catch.

### Running the tests

Classification is covered by tests that need neither Ollama nor Weaviate, so they run
in milliseconds:

```bash
python tests/test_section_classifier.py     # standalone
pytest tests/                               # or with pytest installed
```

They assert correct labels for headings drawn from computing, medicine, psychology,
chemistry, physics, mathematics, economics, law and history, and pin the two
distinctions the retrieval design depends on: background is not related work, and
ambiguous headings must defer rather than guess.

---

## Quickstart

### Prerequisites

- **Python 3.11+**
- **Docker** — runs the Weaviate vector database
- **[Ollama](https://ollama.com)** — runs the local LLM

```bash
ollama pull qwen3:4b-instruct   # ~2.5 GB; see "Choosing a model" below
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
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | Generation + classification model |
| `OLLAMA_THINK` | `false` | Let hybrid models reason first (slower, less accurate here) |
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
  llm.py                 single entry point for Ollama calls
  pdf_viewer.py          renders a chunk back onto its PDF page
  cli.py                 ingest / search / list / sections / delete
app.py                   FastAPI server + JSON API
static/index.html        single-page web UI
scripts/
  migrate_chroma_to_weaviate.py    one-time import from a legacy ChromaDB index
  eval_models.py                   score models on the routing tasks
  eval_rag.py                      score coverage / retrieval / generation / synthesis
  fetch_arxiv.py                   pull open-access papers by pinned arXiv id
tests/
  test_section_classifier.py       cross-discipline classification tests (no LLM needed)
  test_retrieval_filters.py        section-filter regression tests (no LLM needed)
  test_synthesis.py                citation and diversification tests (no LLM needed)
  golden_qa.json                   71 graded questions over 17 papers
docker-compose.yml       Weaviate service
```

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness plus paper/chunk counts |
| `GET` | `/api/papers` | List ingested papers |
| `POST` | `/api/papers` | Upload and ingest PDFs |
| `DELETE` | `/api/papers/{filename}` | Remove a paper and its chunks |
| `POST` | `/api/ask` | One cited answer — corpus-wide, one paper, or a scratch upload |
| `POST` | `/api/chat` | Per-paper breakdown (quick search or deep scan) |
| `GET`/`POST`/`DELETE` | `/api/scratch` | Ad-hoc uploads, isolated from the library |
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
| `section_type` | `text` | One of `abstract`, `introduction`, `theory`, `related_work`, `methodology`, `dataset`, `results`, `conclusion`, `general` |
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
