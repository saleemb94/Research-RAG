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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Vector store (Weaviate, run via docker compose) ────────────────────────
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = _env_int("WEAVIATE_PORT", 8081)
WEAVIATE_GRPC_PORT = _env_int("WEAVIATE_GRPC_PORT", 50052)
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "ResearchChunk")

# Upper bound for full-collection scans (listing sections, map-reduce fetches).
# Must stay <= the QUERY_MAXIMUM_RESULTS set on the Weaviate container.
MAX_FETCH = _env_int("MAX_FETCH", 100_000)
# Page size used to walk a collection; Weaviate caps a single response well
# below MAX_FETCH, so scans are paginated.
FETCH_PAGE_SIZE = _env_int("FETCH_PAGE_SIZE", 1_000)

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = _env_int("TOP_K", 5)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_FETCH_MULTIPLIER = _env_int("RERANK_FETCH_MULTIPLIER", 4)

# ── Paths ──────────────────────────────────────────────────────────────────
PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")

# ── Web app ────────────────────────────────────────────────────────────────
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = _env_int("APP_PORT", 7860)
