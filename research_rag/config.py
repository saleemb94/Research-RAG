"""
Central configuration.

Every value can be overridden with an environment variable, so the same code
runs unchanged locally, in Docker, or in CI. Copy `.env.example` to `.env` and
adjust — python-dotenv loads it automatically if the file exists.
"""

import os

try:  # optional dependency: the app works fine without a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 384)

# ── LLM (Ollama) ───────────────────────────────────────────────────────────
# qwen3:4b-instruct is the 2507 non-thinking release. It scored highest on the
# routing benchmark (scripts/eval_models.py) while also being the fastest and
# smallest of the models tested - see the model notes in the README.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Hybrid models (qwen3:8b, qwen3:14b, ...) reason before answering. That is slower
# and measurably worse at short classification answers, so it is off by default.
OLLAMA_THINK = _env_bool("OLLAMA_THINK", False)

# ── Vector store (Weaviate, run via docker compose) ────────────────────────
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = _env_int("WEAVIATE_PORT", 8081)
WEAVIATE_GRPC_PORT = _env_int("WEAVIATE_GRPC_PORT", 50052)
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "ResearchChunk")
# Ad-hoc uploads live in their own collection rather than behind a flag on the
# main one. Isolation is then structural: nothing you drop in to ask one
# question can leak into a corpus-wide answer, and cleanup is a drop.
WEAVIATE_SCRATCH_COLLECTION = os.getenv(
    "WEAVIATE_SCRATCH_COLLECTION", "ResearchChunkScratch"
)

# Upper bound for full-collection scans (listing sections, map-reduce fetches).
# Must stay <= the QUERY_MAXIMUM_RESULTS set on the Weaviate container.
MAX_FETCH = _env_int("MAX_FETCH", 100_000)
# Page size used to walk a collection; Weaviate caps a single response well
# below MAX_FETCH, so scans are paginated.
FETCH_PAGE_SIZE = _env_int("FETCH_PAGE_SIZE", 1_000)

# Write a one-line description of every section at ingest, and show it to the
# router. Costs a few seconds per paper and nothing at query time. Turn off to
# route on heading names alone (useful for measuring what it is worth).
USE_SECTION_SUMMARIES = _env_bool("USE_SECTION_SUMMARIES", True)

# Search the section summaries as a second retrieval channel and union the
# results with chunk search. Every fact the pipeline was missing appears in some
# summary, and the summaries are 6% the size of the chunk text, so the signal is
# there - but the end-to-end effect measured within run-to-run noise, because the
# channel selects chunks and the summary's density is lost on expansion.
# Kept on, and left switchable, so the question can be settled with repeats.
USE_SECTION_CHANNEL = _env_bool("USE_SECTION_CHANNEL", True)

# Fuse BM25 with the vector search as a third channel. Most facts the pipeline
# misses are rare literal strings, which a dense embedding smears and a keyword
# index matches exactly.
USE_HYBRID_CHANNEL = _env_bool("USE_HYBRID_CHANNEL", True)
# 1.0 is pure vector, 0.0 pure keyword. Low favours keyword; 0.5 was measured
# losing a case that pure vector found, 0.3 did not.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.3"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = _env_int("TOP_K", 5)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_FETCH_MULTIPLIER = _env_int("RERANK_FETCH_MULTIPLIER", 4)

# ── Paths ──────────────────────────────────────────────────────────────────
PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")

# ── Web app ────────────────────────────────────────────────────────────────
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = _env_int("APP_PORT", 7860)
