"""
A searchable index over the section summaries.

The summaries were already being written at ingest and used only to help the
router read a heading list. Measured against the facts the pipeline was missing,
they turn out to be the densest thing in the index: all 22 missed fact groups -
AraVec, ArCybC, OSACT5, "hallucination", Mechanical Turk, 98.83% - appear
somewhere in a summary, and the summaries together are 6% the size of the chunk
text. They are dense because the summariser was told to name concrete things:
datasets and their sizes, methods, instruments, metrics.

Searching them is therefore a second retrieval channel with a different failure
mode from chunk search. A chunk saying "we used AraVec SkipGram embeddings" is
one sentence buried in a methodology section and ranks poorly for "which
embedding models are used"; the summary of that section names AraVec directly
and ranks first.

It is a channel, not a gate. Summaries are lossy by construction, so anything
they omit must still be reachable - the results are unioned with chunk search
and the reranker decides. Hard-filtering on a derived signal has cost this
pipeline recall twice already.
"""

from __future__ import annotations

from .config import WEAVIATE_COLLECTION
from .vector_store import VectorStore

SECTION_COLLECTION = f"{WEAVIATE_COLLECTION}SectionSummary"

# Sections pulled from the summary channel per query. Small: each one expands
# into all of its chunks, so this is a section budget, not a chunk budget.
SUMMARY_SECTIONS = 6


def open_section_index(collection: str = SECTION_COLLECTION) -> VectorStore:
    """
    The section index is chunk-shaped, so it reuses VectorStore rather than
    duplicating a schema: one object per (paper, section) whose `text` is the
    summary. Everything - the vector search, the filters, the paging - already
    works on that shape.
    """
    return VectorStore(collection=collection)


def build_entries(store: VectorStore, source_filter: str | None = None) -> list[dict]:
    """Collect one (paper, section, summary) record per section from the chunk store."""
    seen: dict[tuple[str, str], dict] = {}
    for h in store.get_by_section(source_filter=source_filter):
        p = h.properties
        summary = (p.get("section_summary") or "").strip()
        if not summary:
            continue
        key = (p["source_file"], p["section_name"])
        if key in seen:
            continue
        seen[key] = {
            "source_file": p["source_file"],
            "section_name": p["section_name"],
            "summary": summary,
            "page_numbers": p.get("page_numbers", ""),
            "source_path": p.get("source_path", ""),
        }
    return list(seen.values())


def index_entries(index: VectorStore, entries: list[dict], embedder) -> int:
    """Embed each summary and store it as a searchable record."""
    if not entries:
        return 0
    vectors = embedder.embed([e["summary"] for e in entries])
    index.insert_chunks([
        {
            "properties": {
                "text": e["summary"],
                "source_file": e["source_file"],
                "source_path": e["source_path"],
                # chunk_index keeps records from one paper distinct, since the
                # store derives object ids from source_file plus chunk_index.
                "chunk_index": i,
                "heading": e["section_name"],
                "section_name": e["section_name"],
                "subsection_name": "",
                "section_type": "general",
                "page_numbers": e["page_numbers"],
                "section_summary": e["summary"],
            },
            "vector": v,
        }
        for i, (e, v) in enumerate(zip(entries, vectors))
    ])
    return len(entries)


def sections_for_query(
    index: VectorStore,
    query_vector: list[float],
    limit: int = SUMMARY_SECTIONS,
    source_filter: str | None = None,
) -> list[tuple[str, str]]:
    """Return (paper, section) pairs whose summary best matches the question."""
    hits = index.search(query_vector, limit=limit, source_filter=source_filter)
    out: list[tuple[str, str]] = []
    for h in hits:
        pair = (h.properties.get("source_file", ""), h.properties.get("section_name", ""))
        if all(pair) and pair not in out:
            out.append(pair)
    return out


def chunks_for_sections(
    store: VectorStore,
    pairs: list[tuple[str, str]],
    per_section: int = 4,
) -> list:
    """
    Pull the chunks behind the matched sections.

    Capped per section: a matched section can run to dozens of chunks, and the
    point is to give the reranker candidates it would not otherwise see, not to
    flood it with one section.
    """
    out = []
    for paper, section in pairs:
        hits = store.get_by_section_name([section], source_filter=paper)
        out.extend(hits[:per_section])
    return out
