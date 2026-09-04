"""
Deep scan - two-step LLM targeting + map-reduce summarization.

Step 1  identify_target_section()   - one LLM call → single target phrase
Step 2  match_headings_to_target()  - one LLM call → picks from DB-stored section names
MAP     _map_one()                  - one LLM call per chunk → short passage summary
REDUCE  _reduce()                   - one LLM call per paper → final answer
"""
from __future__ import annotations

from .llm import generate as _llm

from .config import USE_SECTION_SUMMARIES
from .section_classifier import (
    classify_heading,
    identify_target_section,
    match_headings_to_target,
    prefer_exact_types,
)
from .vector_store import SearchHit, VectorStore

_MAP_PROMPT = """\
You are a research assistant reading one passage from a paper.
Answer ONLY based on what this passage says. Ignore irrelevant content.

Question: {query}

Passage from "{source}" - section: {heading} | pages: {pages}
{text}

Write 1–3 concise bullet points about what this passage says relevant to the \
question. If nothing is relevant, reply with exactly: no relevant content\
"""

_REDUCE_PROMPT = """\
You are a research assistant. Synthesise the passage summaries below into one \
clear, comprehensive answer.

Question: {query}

Passage summaries from "{source}":
{summaries}

Write a single well-structured answer using ONLY the information above. \
Cover every relevant point found across all passages.\
"""


def _map_one(query: str, source: str, chunk: SearchHit, model: str) -> str:
    heading = chunk.properties.get("heading") or "N/A"
    pages = chunk.properties.get("page_numbers") or "N/A"
    prompt = _MAP_PROMPT.format(
        query=query, source=source,
        heading=heading, pages=pages,
        text=chunk.properties["text"],
    )
    return _llm(prompt, model)


def _reduce(query: str, source: str, summaries: list[str], model: str) -> str:
    relevant = [s for s in summaries if s.lower() != "no relevant content"]
    if not relevant:
        return "No relevant content was found in this section for the given question."
    body = "\n\n---\n\n".join(
        f"Passage {i + 1}:\n{s}" for i, s in enumerate(relevant)
    )
    prompt = _REDUCE_PROMPT.format(query=query, source=source, summaries=body)
    return _llm(prompt, model)


def deep_scan_papers(
    query: str,
    store: VectorStore,
    model: str,
    batch_size: int = 3,            # kept for API compatibility
    forced_sections: list[str] | None = None,
    source_filter: str | None = None,
) -> dict:
    """
    Two-step LLM section targeting followed by map-reduce summarization.

    Step 1 - Identify target section (one LLM call, or use UI selection).
    Step 2 - Match target against section names stored in the DB at ingest
              time (one LLM call); returns a parsed list of matching names.
    Then   - Retrieve chunks by section_name, run map → reduce.
    """

    # ── Step 1: identify the target section ──────────────────────────────
    if forced_sections:
        target_section = forced_sections[0].replace("_", " ").lower().strip()
        print(f"  [Step 1] UI override → '{target_section}'")
    else:
        target_section = identify_target_section(query, model)
        print(f"  [Step 1] LLM identified → '{target_section}'")

    if not target_section or target_section == "general":
        return {
            "query": query,
            "target_sections": [target_section or "unknown"],
            "papers_found": 0,
            "results": [],
            "error": (
                "Could not identify a specific section for this query. "
                "Please select a section from the dropdown or rephrase your question."
            ),
        }

    # ── Step 2: match against section names stored in the DB ─────────────
    # Pull the unique section names recorded at ingest time.
    stored_section_names = store.get_unique_section_names(source_filter)
    print(f"  [Step 2] Stored section names: {stored_section_names}")

    if not stored_section_names:
        return {
            "query": query,
            "target_sections": [target_section],
            "papers_found": 0,
            "results": [],
            "error": "No sections found. Try ingesting some papers first.",
        }

    section_summary = store.get_section_summary(source_filter)
    matched_names = match_headings_to_target(
        target_section, stored_section_names, model, section_summary,
        descriptions=(
            store.get_section_descriptions(source_filter)
            if USE_SECTION_SUMMARIES else None
        ),
    )
    print(f"  [Step 2] LLM matched → {matched_names}")

    # Cross-validate: drop any matched name whose dominant ingest-time
    # section_type contradicts the target (same guard as quick search).
    if matched_names:
        expected_type = classify_heading(target_section)   # e.g. "methodology"
        if expected_type and expected_type != "general":
            type_map = store.get_section_type_map(source_filter)
            matched_names = prefer_exact_types(
                matched_names, type_map.get, expected_type
            )
            print(f"  [Step 2] After cross-validation → {matched_names}")

    if not matched_names:
        return {
            "query": query,
            "target_sections": [target_section],
            "papers_found": 0,
            "results": [],
            "error": (
                f"No section matching '{target_section}' was found. "
                f"Available sections: {', '.join(stored_section_names)}. "
                "Try selecting a section from the dropdown."
            ),
        }

    # ── Retrieve chunks by matched section names ──────────────────────────
    all_chunks = store.get_by_section_name(
        section_names=matched_names,
        source_filter=source_filter,
    )
    print(f"  Retrieved {len(all_chunks)} chunk(s) from matched sections")

    if not all_chunks:
        return {
            "query": query,
            "target_sections": [target_section],
            "papers_found": 0,
            "results": [],
            "error": (
                f"Sections {matched_names} were found in the index but returned "
                "no chunks. Try re-ingesting the paper."
            ),
        }

    # Group by paper (already sorted by source_file + chunk_index)
    by_source: dict[str, list[SearchHit]] = {}
    for chunk in all_chunks:
        by_source.setdefault(chunk.properties["source_file"], []).append(chunk)

    results = []
    for source, chunks in by_source.items():
        print(f"  [Paper] {source} - {len(chunks)} chunk(s)")

        # ── MAP ───────────────────────────────────────────────────────────
        print(f"    MAP: {len(chunks)} passage(s)...")
        map_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"      passage {i}/{len(chunks)}...")
            map_summaries.append(_map_one(query, source, chunk, model))

        # ── REDUCE ────────────────────────────────────────────────────────
        print(f"    REDUCE → final answer...")
        final_summary = _reduce(query, source, map_summaries, model)

        results.append({
            "source": source,
            "chunks_used": len(chunks),
            "reranked": False,
            "section_filtered": True,
            "sections_used": matched_names,
            "summary": final_summary,
            "map_summaries": map_summaries,
            "chunks": [
                {
                    "text": c.properties["text"],
                    "heading": c.properties.get("heading", ""),
                    "section_name": c.properties.get("section_name", ""),
                    "subsection_name": c.properties.get("subsection_name", ""),
                    "section_type": c.properties.get("section_type", "general"),
                    "page_numbers": c.properties.get("page_numbers", ""),
                    "source_path": c.properties.get("source_path", ""),
                    "distance": None,
                }
                for c in chunks
            ],
        })

    if not results:
        return {
            "query": query,
            "target_sections": [target_section],
            "papers_found": 0,
            "results": [],
            "error": f"No content found for target section '{target_section}'.",
        }

    return {
        "query": query,
        "target_sections": [target_section],
        "section_filtered": True,
        "papers_found": len(results),
        "reranked": False,
        "results": results,
    }
