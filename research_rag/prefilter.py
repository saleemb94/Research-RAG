"""
Stage one of two-stage retrieval: decide which papers are worth searching.

The flat design searches every chunk in the library for every question. That is
fine at fifty papers and wrong at a thousand, for two reasons. Cost grows with
the shelf rather than with the question. And precision falls as the library
becomes topically uniform - a corpus of retrieval papers has dozens that can
plausibly answer "which datasets were used", and the top-k fills with near
misses from papers that were never the right one.

Ranking papers first fixes both. Chunk search then runs inside a small, less
confusable pool, and its cost is set by the shortlist width rather than by the
corpus.

The section-summary index does the ranking. It already exists, it is one
short summary per section rather than one vector per chunk, and measured on 54
papers against 107 graded questions it puts the right paper inside the top 20
93% of the time - against 89% for the paper cards, which is why summaries and
not cards. Recall at the widths that matter:

    top 5    69%        top 10   81%        top 20   93%

A shortlist is a recall ceiling: nothing stage two does can recover a paper
stage one dropped. Width is therefore the safety margin, and 20 of 54 papers
prunes little - the point is that 20 of 1,000 prunes almost everything while
keeping the same 93%.
"""

from __future__ import annotations

from .vector_store import VectorStore


def shortlist_papers(
    query_vector: list[float],
    section_index: VectorStore,
    limit_papers: int,
    per_paper_summaries: int = 6,
) -> list[str]:
    """
    Papers worth searching for this question, best first.

    Scored by each paper's single best-matching section summary rather than by
    an average, because a paper is worth reading if any one of its sections is
    relevant. Averaging would bury a specialized paper with one very relevant
    section under a general one that is mildly relevant throughout.
    """
    if limit_papers <= 0:
        return []
    # Enough summaries that the tail of the ranking still has papers in it:
    # one paper can easily own the first several hits.
    hits = section_index.search(
        query_vector, limit=limit_papers * per_paper_summaries
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for h in hits:
        name = h.properties.get("source_file")
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
            if len(ordered) >= limit_papers:
                break
    return ordered
