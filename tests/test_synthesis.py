"""
Tests for cross-paper synthesis: citation validation and source diversification.

Both are pure functions, so these need neither Ollama nor Weaviate.

    python tests/test_synthesis.py
    pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_rag.synthesize import (  # noqa: E402
    condense_question,
    diversify_by_paper,
    validate_citations,
)
from research_rag.vector_store import SearchHit  # noqa: E402


def _hit(paper, idx):
    return SearchHit(
        properties={"source_file": paper, "chunk_index": idx, "text": f"{paper}#{idx}"},
        distance=0.1 * idx,
    )


# ── citation validation ────────────────────────────────────────────────────

def test_valid_citations_are_kept():
    text = "The corpus holds 20,000 posts [2]. SVM reached 84% [3][5]."
    cleaned, cited, dropped = validate_citations(text, 8)
    assert cited == [2, 3, 5]
    assert dropped == []
    assert cleaned == text


def test_out_of_range_citations_are_removed():
    """
    A dangling marker is worse than none: it looks like provenance and resolves
    to nothing, so the reader cannot tell a grounded claim from an invented one.
    """
    cleaned, cited, dropped = validate_citations("Claim A [2]. Claim B [9][12].", 8)
    assert cited == [2]
    assert dropped == [9, 12]
    assert "[9]" not in cleaned and "[12]" not in cleaned
    assert "[2]" in cleaned


def test_removing_a_citation_leaves_clean_punctuation():
    cleaned, _, _ = validate_citations("A fact [99]. Another [1].", 8)
    assert "  " not in cleaned
    assert cleaned.startswith("A fact.")


def test_answer_without_citations_survives():
    cleaned, cited, dropped = validate_citations("The sources do not answer this.", 8)
    assert (cited, dropped) == ([], [])
    assert cleaned == "The sources do not answer this."


def test_zero_is_not_a_valid_citation():
    _, cited, dropped = validate_citations("Claim [0].", 8)
    assert cited == [] and dropped == [0]


# ── source diversification ─────────────────────────────────────────────────

def test_diversify_spreads_across_papers():
    """
    A relevance-ranked list clusters. Without a cap, one paper that discusses a
    topic densely takes every slot and the answer reports it as the whole corpus.
    """
    hits = [_hit("a.pdf", i) for i in range(6)] + [_hit("b.pdf", i) for i in range(3)]
    picked = diversify_by_paper(hits, limit=4, per_paper=2)
    papers = [h.properties["source_file"] for h in picked]
    assert len(picked) == 4
    assert papers.count("a.pdf") == 2, f"cap not enforced: {papers}"
    assert papers.count("b.pdf") == 2


def test_diversify_keeps_the_top_hit():
    hits = [_hit("a.pdf", 0)] + [_hit("a.pdf", i) for i in range(1, 5)]
    picked = diversify_by_paper(hits, limit=3, per_paper=1)
    assert picked[0].properties["chunk_index"] == 0


def test_diversify_tops_up_rather_than_returning_too_few():
    """With one paper and a cap of 2, still return `limit` sources, not 2."""
    hits = [_hit("a.pdf", i) for i in range(6)]
    picked = diversify_by_paper(hits, limit=4, per_paper=2)
    assert len(picked) == 4, "under-filled the answer when the corpus is narrow"


def test_diversify_disabled_is_plain_truncation():
    hits = [_hit("a.pdf", i) for i in range(6)]
    assert diversify_by_paper(hits, limit=3, per_paper=0) == hits[:3]


def test_diversify_preserves_rank_order():
    hits = [_hit("a.pdf", 0), _hit("b.pdf", 1), _hit("a.pdf", 2), _hit("c.pdf", 3)]
    picked = diversify_by_paper(hits, limit=4, per_paper=1)
    assert [h.properties["chunk_index"] for h in picked] == [0, 1, 3, 2]


# ── history condensation ───────────────────────────────────────────────────

def test_condense_passes_through_without_history():
    """No history means nothing to resolve, and no LLM call worth making."""
    q = "What datasets were used?"
    assert condense_question(q, None) == q
    assert condense_question(q, []) == q
    assert condense_question(q, [{"role": "user", "content": ""}]) == q


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


# ── the per-paper cap depends on the kind of question ──────────────────────

def test_focused_questions_may_take_more_from_one_paper():
    """
    A focused question's evidence is inside one paper. "What accuracy did
    BiLSTM, CNN and GRU each reach" needs numbers the paper states across
    several passages, and the corpus-wide cap of two truncates the answer
    before retrieval is consulted.
    """
    from research_rag.synthesize import cap_for, PER_PAPER_SOURCES
    assert cap_for("focused") > PER_PAPER_SOURCES


def test_corpus_questions_keep_the_spreading_cap():
    """
    The cap exists so one densely-worded paper cannot take every slot in a
    corpus-wide answer. Raising it there would reintroduce exactly that.
    """
    from research_rag.synthesize import cap_for, PER_PAPER_SOURCES
    assert cap_for("corpus") == PER_PAPER_SOURCES
    assert cap_for("enumerate") == PER_PAPER_SOURCES


def test_the_focused_cap_still_leaves_room_for_other_papers():
    """Focused is a classification and is sometimes wrong, so it is a cap and
    not an exemption: one paper must never be able to take every slot."""
    from research_rag.synthesize import cap_for, MAX_SOURCES
    assert cap_for("focused") < MAX_SOURCES


# ── steps that decide rather than write are greedy ─────────────────────────

def test_decisions_are_greedy():
    """
    The router picks between three different pipelines. Sampling that decision
    made the same question take different paths on different runs, which is not
    variance in the answer but variance in what produced it.
    """
    from research_rag.llm import temperature_for
    assert temperature_for("routing") == 0
    assert temperature_for("extract") == 0
    assert temperature_for("classify") == 0


def test_answering_keeps_the_model_default():
    """
    Writing prose is not a decision with one right answer, so the answer step
    is left alone rather than forced to greedy on the same reasoning.
    """
    from research_rag.llm import temperature_for
    assert temperature_for("answer") is None
    assert temperature_for("anything-else") is None


def test_temperature_reaches_the_request():
    """
    A temperature that never arrives in options is the failure this would have
    had: the constant reads correctly and nothing changes at the server.
    """
    import research_rag.llm as llm
    seen = {}

    class _Resp:
        response = "ok"

    def fake_generate(model, prompt, **kwargs):
        seen.update(kwargs)
        return _Resp()

    real = llm.ollama.generate
    llm.ollama.generate = fake_generate
    try:
        llm.generate("hi", model="m", temperature=0.0, max_tokens=8)
    finally:
        llm.ollama.generate = real
    assert seen["options"]["temperature"] == 0.0
    assert seen["options"]["num_predict"] == 8


# ── the cards saying "I cannot" is not an answer ───────────────────────────

def test_nothing_found_is_not_a_truthy_answer():
    """
    NOTHING_FOUND is a non-empty string, so `if raw:` accepted it as an answer
    and returned it to the reader instead of falling through to retrieval. The
    result was the worst thing a research tool can say: that the library does
    not cover something it does. Three of forty gold-passage questions returned
    it verbatim while the answering passage sat in the index.
    """
    import re
    from research_rag.paper_cards import NOTHING_FOUND
    src = (Path(__file__).resolve().parent.parent
           / "research_rag" / "synthesize.py").read_text(encoding="utf-8")
    calls = src.count("raw, used = answer_from_cards(")
    guards = src.count("if raw and raw.strip() != NOTHING_FOUND:")
    assert calls > 0, "no card call sites found; this test needs updating"
    assert guards == calls, (
        f"{calls} card call site(s) but {guards} guarded against the sentinel"
    )
    assert NOTHING_FOUND, "the sentinel must stay non-empty for this to matter"


# ── which claim to check first ─────────────────────────────────────────────

def test_a_figure_absent_from_its_citation_scores_low():
    """The provable case: the claim states a number the cited passage does not."""
    from research_rag.confidence import score_claim
    ok = score_claim("BiLSTM reached 82.02% accuracy",
                     ["the BiLSTM model reached accuracy 82.02% on the test set"])
    bad = score_claim("BiLSTM reached 91.5% accuracy",
                      ["the BiLSTM model reached accuracy 82.02% on the test set"])
    assert ok["confidence"] > 0.8
    assert bad["confidence"] < 0.4
    assert bad["numbers_missing"] == ["91.5%"]


def test_document_pointers_are_not_claims_about_data():
    """"Figure 3" is a pointer, not an assertion, and counting it as an
    unsupported figure made every answer mentioning one look ungrounded."""
    from research_rag.confidence import score_claim
    r = score_claim("These results appear in Figure 3 and Table 2",
                    ["the results are plotted for the reader"])
    assert r["numbers"] == [], r["numbers"]


def test_claims_are_sentences_not_citation_fragments():
    """
    Splitting at citation groups stranded the text after the last marker as its
    own claim. Those fragments scored near zero and captured the flag, so it
    pointed at punctuation instead of at anything worth checking.
    """
    from research_rag.confidence import split_claims
    got = split_claims(
        "BiLSTM reached 82.02% accuracy [1][2]. These results appear in Figure 3 [1].")
    assert len(got) == 2, got
    assert got[0]["cites"] == [1, 2]
    assert all(c["text"][0].isupper() for c in got), got


def test_a_claim_of_absence_is_never_flagged():
    """
    A claim that a paper does not mention something cannot echo the passage it
    cites, so its low score carries no information. The first version flagged
    nothing else.
    """
    from research_rag.confidence import weakest
    claims = [
        {"text": "The paper does not mention Macro-F1 at all.", "cites": [1],
         "confidence": 0.0, "numbers_missing": [], "number_support": None},
        {"text": "The model reached an accuracy of 91.5 percent on the set.",
         "cites": [2], "confidence": 0.2, "numbers_missing": ["91.5"],
         "number_support": 0.0},
    ]
    w = weakest(claims)
    assert w is not None and w["cites"] == [2], w


def test_a_well_grounded_answer_is_not_flagged():
    """Flagging the least-good claim in a good answer is noise dressed as insight."""
    from research_rag.confidence import weakest
    claims = [{"text": "The model reached an accuracy of 82.02 percent here.",
               "cites": [1], "confidence": 0.9, "numbers_missing": [],
               "number_support": 1.0}]
    assert weakest(claims) is None


if __name__ == "__main__":
    raise SystemExit(main())
