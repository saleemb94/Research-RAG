from __future__ import annotations

from typing import TYPE_CHECKING

from .llm import generate as _llm

from .config import (
    OLLAMA_MODEL,
    RERANK_FETCH_MULTIPLIER,
    TOP_K,
    USE_SECTION_SUMMARIES,
)
from .embedder import Embedder
from .section_classifier import (
    classify_query,
    match_headings_to_target,
    prefer_exact_types,
)
from .vector_store import SearchHit, VectorStore

if TYPE_CHECKING:
    from .reranker import Reranker


def _build_prompt(query: str, source: str, context_blocks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"You are a research assistant. Using ONLY the excerpts below from '{source}', "
        f"answer the following question concisely and accurately.\n\n"
        f"Question: {query}\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Answer:"
    )


def _deduplicate(hits: list[SearchHit]) -> list[SearchHit]:
    seen, out = set(), []
    for h in hits:
        uid = (h.properties["source_file"], h.properties.get("chunk_index"))
        if uid not in seen:
            seen.add(uid)
            out.append(h)
    return out


def _hierarchical_search(
    query_vector: list[float],
    store: VectorStore,
    target_sections: list[str],
    fetch_k: int,
    top_k: int,
    source_filter: str | None = None,
    use_section_names: bool = False,
) -> tuple[list[SearchHit], bool]:
    """
    Section-filtered vector search with global fallback.

    `use_section_names=True`  → filter by section_name field (exact stored names,
                                 set when user picks from the dropdown)
    `use_section_names=False` → filter by section_type label (from classify_query)
    """
    use_sections = bool(target_sections) and target_sections != ["general"]

    if use_sections:
        kwargs = dict(
            section_names=target_sections if use_section_names else None,
            section_types=None if use_section_names else target_sections,
            source_filter=source_filter,
        )
        hits = store.search(query_vector, limit=fetch_k, **kwargs)
        if hits:
            # Return whatever is in the section - never blend in chunks from
            # other sections, even if only a few results were found.
            return hits, True
        # Section is completely empty → fall back to global so the user
        # gets something rather than a blank response.
        global_hits = store.search(
            query_vector, limit=fetch_k, source_filter=source_filter
        )
        return global_hits, False

    return store.search(
        query_vector, limit=fetch_k, source_filter=source_filter
    ), False


def search_and_summarize(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = TOP_K,
    model: str = OLLAMA_MODEL,
    reranker: "Reranker | None" = None,
    forced_sections: list[str] | None = None,
    source_filter: str | None = None,
) -> dict:
    # Determine which section(s) to target and how to filter.
    # `orig_section_type` holds the canonical label from classify_query and is
    # used later for a post-retrieval safety check regardless of which filter path
    # was taken.
    orig_section_type: str | None = None

    if forced_sections:
        target_sections = forced_sections
        use_section_names = True
    else:
        target_sections = classify_query(query, model)
        use_section_names = False

        if len(target_sections) == 1 and target_sections[0] != "general":
            orig_section_type = target_sections[0]   # e.g. "methodology"

        # Matching the target onto real stored headings only makes sense inside a
        # single paper. Across the corpus the heading lists are pooled, so the LLM
        # picks names that exist in only one or two papers ("DATA PREPROCESSING
        # AND ANNOTATION") and filtering on them makes every other paper
        # unreachable. The cross-paper equivalent of a heading is section_type,
        # which is assigned per chunk at ingest precisely so that it generalizes,
        # so an unscoped query stays on the section_type path.
        if orig_section_type and source_filter:
            stored_names = store.get_unique_section_names(source_filter)
            if stored_names:
                natural = orig_section_type.replace("_", " ")
                section_summary = store.get_section_summary(source_filter)
                matched = match_headings_to_target(
                    natural, stored_names, model, section_summary,
                    descriptions=(
                        store.get_section_descriptions(source_filter)
                        if USE_SECTION_SUMMARIES else None
                    ),
                )
                if matched:
                    # Cross-validate: keep only sections whose dominant
                    # ingest-time section_type agrees with the query target.
                    # Use "general" as the missing-key default - an unknown
                    # section gets the benefit of the doubt, but NOT a free
                    # pass disguised as the expected type.
                    type_map = store.get_section_type_map(source_filter)
                    validated = prefer_exact_types(
                        matched, type_map.get, orig_section_type
                    )
                    if validated:
                        target_sections = validated
                        use_section_names = True
                    # All matches vetoed → fall back to section_type filtering.

    query_vector = embedder.embed_one(query)
    fetch_k = top_k * RERANK_FETCH_MULTIPLIER if reranker else top_k

    hits, section_filtered = _hierarchical_search(
        query_vector, store, target_sections, fetch_k, top_k,
        source_filter=source_filter,
        use_section_names=use_section_names,
    )

    # Post-retrieval safety checks ────────────────────────────────────────────
    if hits:
        # 1. When filtering by section_name, enforce the allowed set in Python
        #    as well - guarantees no stray section bleeds in regardless of what
        #    the database returned.
        #
        #    Gated on `section_filtered`, NOT on `use_section_names`: when the
        #    requested sections turn out to be empty, _hierarchical_search falls
        #    back to a global search and reports section_filtered=False. Applying
        #    the allow-list to those fallback hits would delete every one of them
        #    and return an empty answer while the store had perfectly good chunks
        #    in hand - which is exactly what used to happen.
        if section_filtered and use_section_names and target_sections:
            allowed = set(target_sections)
            hits = [h for h in hits
                    if h.properties.get("section_name") in allowed]

        # 2. Drop chunks whose ingest-time section_type clearly contradicts the
        #    query's target type.  "general" chunks are kept (unclassified, not
        #    wrong).  Only runs when we have a single unambiguous target type.
        #
        #    Also gated on `section_filtered`, for the same reason as above: on
        #    the fallback path the requested section does not exist in this
        #    paper, so every fallback hit contradicts the target by definition
        #    and this check would empty the result set.
        if section_filtered and orig_section_type:
            hits = prefer_exact_types(
                hits,
                lambda h: h.properties.get("section_type"),
                orig_section_type,
            )

    if not hits:
        return {
            "query": query,
            "target_sections": target_sections,
            "section_filtered": False,
            "papers_found": 0,
            "results": [],
        }

    if reranker:
        hits = reranker.rerank(query, hits, top_n=top_k)
    else:
        hits = hits[:top_k]

    by_source: dict[str, list[SearchHit]] = {}
    for obj in hits:
        by_source.setdefault(obj.properties["source_file"], []).append(obj)

    results = []
    for source, objects in by_source.items():
        context_blocks = [
            f"[Section: {obj.properties.get('heading') or 'N/A'}"
            f"{' | Pages: ' + obj.properties['page_numbers'] if obj.properties.get('page_numbers') else ''}]\n"
            f"{obj.properties['text']}"
            for obj in objects
        ]
        prompt = _build_prompt(query, source, context_blocks)
        summary = _llm(prompt, model)

        results.append({
            "source": source,
            "chunks_used": len(objects),
            "reranked": reranker is not None,
            "section_filtered": section_filtered,
            "sections_used": sorted({
                obj.properties.get("section_name") or
                obj.properties.get("section_type", "general")
                for obj in objects
            }),
            "summary": summary,
            "chunks": [
                {
                    "text": obj.properties["text"],
                    "heading": obj.properties.get("heading", ""),
                    "section_name": obj.properties.get("section_name", ""),
                    "subsection_name": obj.properties.get("subsection_name", ""),
                    "section_type": obj.properties.get("section_type", "general"),
                    "page_numbers": obj.properties.get("page_numbers", ""),
                    "source_path": obj.properties.get("source_path", ""),
                    "distance": obj.distance,
                }
                for obj in objects
            ],
        })

    return {
        "query": query,
        "target_sections": target_sections,
        "section_filtered": section_filtered,
        "papers_found": len(by_source),
        "reranked": reranker is not None,
        "results": results,
    }
