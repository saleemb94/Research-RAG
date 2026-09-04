"""
Regression tests for the section-filtering logic in search.py.

These pin three defects the golden question set exposed. All three were silent:
each produced a plausible-looking response while quietly discarding correct
results, so nothing short of graded ground truth would have caught them.

  1. Heading matching ran over headings pooled from every paper, so an unscoped
     query got filtered to section names that exist in only one or two of them
     and the rest of the corpus became unreachable.
  2. The post-retrieval "safety" filters were applied to the global fallback
     results, deleting every one of them and returning an empty answer.
  3. Front matter filed under a long paper title was dropped as boilerplate,
     discarding the abstract.

Runs without Ollama or Weaviate - the store and the LLM are stubbed.

    python tests/test_retrieval_filters.py
    pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_rag import search as S  # noqa: E402
from research_rag.ingest import _is_skippable, _looks_like_paper_title  # noqa: E402
from research_rag.vector_store import SearchHit  # noqa: E402

PAPER_A = "alpha.pdf"
PAPER_B = "beta.pdf"

# Two papers that name the same role differently - the situation heading
# matching exists to handle, and the situation that broke it across a corpus.
CHUNKS = [
    (PAPER_A, 0, "Our Data Collection Protocol", "dataset", "alpha collected 1051 videos"),
    (PAPER_A, 1, "Our Data Collection Protocol", "dataset", "alpha split 80/20"),
    (PAPER_A, 2, "Findings", "results", "alpha reached 87% accuracy"),
    (PAPER_B, 0, "Study Population", "dataset", "beta enrolled 248 participants"),
    (PAPER_B, 1, "Findings", "results", "beta reached 91% accuracy"),
]


def _hit(paper, idx, section, stype, text):
    return SearchHit(
        properties={
            "text": text, "source_file": paper, "chunk_index": idx,
            "section_name": section, "subsection_name": "",
            "section_type": stype, "heading": section,
            "page_numbers": "1", "source_path": "",
        },
        distance=0.1,
    )


class FakeStore:
    """Minimal stand-in implementing only what search_and_summarize touches."""

    def __init__(self):
        self.hits = [_hit(*c) for c in CHUNKS]

    def search(self, query_vector, limit=5, source_filter=None,
               section_types=None, section_names=None):
        out = []
        for h in self.hits:
            p = h.properties
            if source_filter and p["source_file"] != source_filter:
                continue
            if section_names and p["section_name"] not in section_names:
                continue
            if section_types and p["section_type"] not in section_types:
                continue
            out.append(h)
        return out[:limit]

    def _scoped(self, source_filter):
        return [h for h in self.hits
                if not source_filter or h.properties["source_file"] == source_filter]

    def get_unique_section_names(self, source_filter=None):
        seen = []
        for h in self._scoped(source_filter):
            n = h.properties["section_name"]
            if n not in seen:
                seen.append(n)
        return seen

    def get_section_summary(self, source_filter=None):
        return {n: [] for n in self.get_unique_section_names(source_filter)}

    def get_section_descriptions(self, source_filter=None):
        # Sections indexed before summaries existed have none; the router must
        # degrade to heading names rather than break.
        return {}

    def get_section_type_map(self, source_filter=None):
        return {h.properties["section_name"]: h.properties["section_type"]
                for h in self._scoped(source_filter)}


class FakeEmbedder:
    def embed_one(self, text):
        return [0.1] * 8


class _Stub:
    """Replace the LLM-backed helpers search.py calls."""

    def __init__(self, query_labels, matched_headings):
        self.query_labels = query_labels
        self.matched_headings = matched_headings
        self.match_calls = 0

    def __enter__(self):
        self._real = (S.classify_query, S.match_headings_to_target, S._llm)

        def match(target, headings, model, section_summary=None, descriptions=None):
            self.match_calls += 1
            return [h for h in self.matched_headings if h in headings]

        S.classify_query = lambda q, m: list(self.query_labels)
        S.match_headings_to_target = match
        S._llm = lambda prompt, model: "stub summary"
        return self

    def __exit__(self, *_):
        S.classify_query, S.match_headings_to_target, S._llm = self._real


def _run(query_labels, matched, source_filter=None, forced_sections=None):
    with _Stub(query_labels, matched) as stub:
        result = S.search_and_summarize(
            "a question", FakeEmbedder(), FakeStore(),
            top_k=5, model="stub", source_filter=source_filter,
            forced_sections=forced_sections,
        )
    return result, stub


# ── Bug 1: cross-paper heading pooling ─────────────────────────────────────

def test_unscoped_query_does_not_match_headings():
    """An unscoped query must stay on section_type, which generalizes."""
    result, stub = _run(["dataset"], ["Our Data Collection Protocol"])
    assert stub.match_calls == 0, "heading matching ran on a corpus-wide query"
    assert result["papers_found"] == 2, (
        "filtering on one paper's heading hid the other paper; "
        f"got {result['papers_found']} paper(s)"
    )


def test_scoped_query_still_matches_headings():
    """Inside one paper, heading matching is exactly the right tool."""
    result, stub = _run(["dataset"], ["Our Data Collection Protocol"],
                        source_filter=PAPER_A)
    assert stub.match_calls == 1, "heading matching should run when scoped"
    assert result["papers_found"] == 1
    assert result["results"][0]["source"] == PAPER_A


def test_unscoped_query_reaches_every_paper_for_its_type():
    result, _ = _run(["results"], [])
    assert {r["source"] for r in result["results"]} == {PAPER_A, PAPER_B}


# ── Bug 2: safety filters canceling the global fallback ───────────────────

def test_fallback_survives_the_post_retrieval_filters():
    """
    Target a section this paper does not have. _hierarchical_search falls back
    to a global search and reports section_filtered=False; the allow-list must
    not then delete every one of those fallback hits.
    """
    result, _ = _run(["dataset"], [], source_filter=PAPER_A,
                     forced_sections=["No Such Heading In This Paper"])
    assert result["section_filtered"] is False, "test premise: fallback should fire"
    assert result["papers_found"] == 1, (
        "global fallback was undone by the post-retrieval filter - the query "
        "returned nothing while the store held perfectly good chunks"
    )
    assert result["results"][0]["chunks_used"] > 0


def test_fallback_fires_for_headings_belonging_to_another_paper():
    """The exact corpus-wide failure: headings picked from a different paper."""
    result, _ = _run(["dataset"], [], source_filter=PAPER_A,
                     forced_sections=["Study Population"])  # exists only in PAPER_B
    assert result["papers_found"] == 1
    assert result["results"][0]["source"] == PAPER_A


def test_section_filter_still_enforced_when_it_succeeds():
    """The allow-list must keep working on the non-fallback path."""
    result, _ = _run(["dataset"], [], source_filter=PAPER_A,
                     forced_sections=["Our Data Collection Protocol"])
    assert result["section_filtered"] is True
    used = {c["section_name"] for c in result["results"][0]["chunks"]}
    assert used == {"Our Data Collection Protocol"}, f"stray section leaked in: {used}"


# ── Bug 3: front matter discarded with the paper title ─────────────────────

LONG_TITLE = (
    "Offensive Language Detection in Arabic Social Networks Using "
    "Evolutionary-Based Classifiers Learned From Fine-Tuned Embeddings"
)


def test_long_paper_title_is_not_treated_as_boilerplate():
    assert _looks_like_paper_title(LONG_TITLE), "test premise: this reads as a title"
    assert not _is_skippable(LONG_TITLE), (
        "front matter under a long title was dropped, discarding the abstract"
    )


def test_real_boilerplate_is_still_skipped():
    for name in ["References", "Bibliography", "Acknowledgments", "Funding",
                 "Conflicts of Interest", "Data Availability", "Appendix"]:
        assert _is_skippable(name), f"{name!r} should still be skipped"


def test_genuine_sections_are_never_skipped():
    for name in ["INTRODUCTION", "METHODOLOGY", "Study Population",
                 "Proof of Theorem 2", "EXPERIMENTS AND RESULTS"]:
        assert not _is_skippable(name), f"{name!r} must be kept"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
