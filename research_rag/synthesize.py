"""
Cross-paper synthesis: one grounded answer, with citations.

The original pipeline fans out and returns a separate summary per paper, which
suits "compare the methodology across these papers" but not "which datasets have
been used for Arabic offensive language detection" - for that the reader wants
one answer that draws on several papers and says which claim came from where.

This module produces that answer. It also costs less than the fan-out: one LLM
call regardless of how many papers matched, instead of one per paper.

Citations are validated after generation. A model that invents [9] when only
eight sources were supplied would otherwise produce an answer whose provenance
silently does not resolve, which is worse than no citation at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import (
    OLLAMA_MODEL,
    RERANK_FETCH_MULTIPLIER,
    TOP_K,
    HYBRID_ALPHA,
    USE_HYBRID_CHANNEL,
    USE_SECTION_CHANNEL,
)
from .embedder import Embedder
from .llm import generate as _llm
from .llm import model_for
from .enumerate_ import classify_question, gather, reduce_findings
from .paper_cards import answer_from_cards
from .section_index import chunks_for_sections, sections_for_query
from .search import _hierarchical_search
from .section_classifier import classify_query, prefer_exact_types
from .vector_store import SearchHit, VectorStore

# Sources shown to the model. Beyond roughly a dozen a small model starts citing
# by position rather than by content, and the prompt crowds out the answer.
MAX_SOURCES = 8

# Most chunks any single paper may contribute to one answer. Two lets a paper
# support a claim and qualify it, while leaving room for the rest of the corpus.
PER_PAPER_SOURCES = 2

# Characters of each chunk included. Enough for the claim, short enough that
# eight of them still leave the model room to reason.
_SOURCE_CHARS = 900

_ANSWER_PROMPT = """\
You are a research assistant. Answer the question using ONLY the numbered sources
below, which are excerpts from research papers.

Write ONE continuous answer that synthesises across sources. Do not write a
separate paragraph per paper and do not repeat the question.

Cite every factual claim with its source number in square brackets, like [2]. A
sentence resting on two sources ends with [1][4]. Never cite a number that does
not appear in the list. Where sources disagree, say so and cite both.

If the sources do not answer the question, say that plainly instead of guessing.

Question: {query}

Sources:
{sources}

Answer:"""

_CONDENSE_PROMPT = """\
Rewrite the follow-up question as a standalone question that can be understood
without the conversation.

Resolve pronouns and references ("it", "that paper", "the second one") using the
conversation. Keep the user's wording wherever you can. Add no new topics.
If it already stands alone, return it unchanged.

Conversation so far:
{history}

Follow-up: {question}

Standalone question:"""


@dataclass
class Source:
    """One cited excerpt, carrying everything the UI needs to show provenance."""

    number: int
    source_file: str
    section_name: str
    page_numbers: str
    source_path: str
    text: str
    distance: float | None = None

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "source": self.source_file,
            "section_name": self.section_name,
            "page_numbers": self.page_numbers,
            "source_path": self.source_path,
            "text": self.text,
            "distance": self.distance,
        }


@dataclass
class SynthesisResult:
    query: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    cited_numbers: list[int] = field(default_factory=list)
    dropped_citations: list[int] = field(default_factory=list)
    target_sections: list[str] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "cited": self.cited_numbers,
            "dropped_citations": self.dropped_citations,
            "target_sections": self.target_sections,
            "papers": self.papers,
            "papers_found": len(self.papers),
        }


def condense_question(
    question: str,
    history: list[dict] | None,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Fold a follow-up into a standalone question.

    Retrieval embeds the query literally, so "and what about its dataset?" would
    otherwise be matched as-is and retrieve nothing useful. Only the last few
    turns are used; older context rarely changes what the follow-up refers to
    and it makes the model drift.
    """
    if not history:
        return question
    recent = [h for h in history if h.get("content")][-4:]
    if not recent:
        return question

    rendered = "\n".join(
        f"{'User' if h.get('role') == 'user' else 'Assistant'}: "
        f"{' '.join(str(h['content']).split())[:400]}"
        for h in recent
    )
    try:
        standalone = _llm(
            _CONDENSE_PROMPT.format(history=rendered, question=question), model
        )
    except Exception:
        return question           # never let rewriting break the query
    standalone = " ".join(standalone.split()).strip('"')
    # A model that returns a preamble, an empty string or an essay is not to be
    # trusted over the question the user actually typed.
    if not standalone or len(standalone) > 400:
        return question
    return standalone


def diversify_by_paper(
    hits: list[SearchHit], limit: int, per_paper: int
) -> list[SearchHit]:
    """
    Spread the source slots across papers instead of letting one paper take them all.

    A relevance-ranked list clusters: for "which datasets have been used across
    these papers", the eight best chunks are often eight chunks from the two
    papers that discuss datasets most densely, and the answer then reports those
    two as though they were the whole corpus. Capping each paper and filling the
    remaining slots by rank keeps the best chunk from each paper while still
    preferring relevance.

    Rank order is preserved throughout, so the top hit is always kept.
    """
    if per_paper <= 0:
        return hits[:limit]

    taken: dict[str, int] = {}
    picked: list[SearchHit] = []
    overflow: list[SearchHit] = []
    for h in hits:
        paper = h.properties.get("source_file", "")
        if taken.get(paper, 0) < per_paper:
            taken[paper] = taken.get(paper, 0) + 1
            picked.append(h)
        else:
            overflow.append(h)
        if len(picked) == limit:
            return picked
    # Corpus smaller than the cap implies - top up by rank so we never return
    # fewer sources than we could.
    return (picked + overflow)[:limit]


def _build_sources(hits: list[SearchHit]) -> list[Source]:
    return [
        Source(
            number=i + 1,
            source_file=h.properties.get("source_file", ""),
            section_name=h.properties.get("section_name", ""),
            page_numbers=h.properties.get("page_numbers", ""),
            source_path=h.properties.get("source_path", ""),
            text=h.properties.get("text", ""),
            distance=h.distance,
        )
        for i, h in enumerate(hits)
    ]


def _render_sources(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{s.number}] {s.source_file}"
        f"{' - ' + s.section_name if s.section_name else ''}"
        f"{' (p. ' + s.page_numbers + ')' if s.page_numbers else ''}\n"
        f"{s.text[:_SOURCE_CHARS]}"
        for s in sources
    )


def validate_citations(answer: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """
    Strip citations that point at nothing, and report what was kept and removed.

    A dangling [9] against eight sources is worse than no marker: it looks like
    provenance and resolves to nothing. Returns (cleaned answer, cited, dropped).
    """
    cited: list[int] = []
    dropped: list[int] = []

    def repl(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            if n not in cited:
                cited.append(n)
            return match.group(0)
        if n not in dropped:
            dropped.append(n)
        return ""

    cleaned = re.sub(r"\[(\d+)\]", repl, answer)
    # Tidy up whitespace left where a bad marker was removed.
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, sorted(cited), sorted(dropped)


def synthesize_answer(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = TOP_K,
    model: str = OLLAMA_MODEL,
    reranker=None,
    source_filter: str | None = None,
    history: list[dict] | None = None,
    max_sources: int = MAX_SOURCES,
    section_filter: bool | None = None,
    per_paper: int = PER_PAPER_SOURCES,
    allow_enumeration: bool = True,
    cards: list | None = None,
    section_index=None,
) -> SynthesisResult:
    """
    Retrieve across the corpus (or within one paper) and write a single cited answer.

    `source_filter` set is the single-paper case: same code path, narrower scope.
    """
    standalone = condense_question(query, history, model)

    # "Which X appear across these papers" is an aggregation, not a retrieval:
    # the members of the answer sit one per paper, so any top-k over chunks has
    # a recall ceiling below the true answer. Fan out instead. Only for
    # corpus-wide questions - inside one paper there is nothing to aggregate.
    kind = (
        classify_question(standalone, model)
        if allow_enumeration and not source_filter
        else "focused"
    )

    # A question about the collection - "which of these are not about text
    # classification" - has no supporting passage anywhere, because no paper
    # states what it is not. Retrieval and fan-out both fail on it by
    # construction: every paper answers NONE, or answers a different question.
    # The cards are the only place that knowledge exists.
    if kind == "corpus" and cards:
        raw, used = answer_from_cards(standalone, cards, model)
        if raw:
            answer, cited, dropped = validate_citations(raw, len(used))
            sources = [
                Source(number=i + 1, source_file=c.source_file,
                       section_name="paper card", page_numbers="",
                       source_path="", text=c.render(i + 1))
                for i, c in enumerate(used)
            ]
            return SynthesisResult(
                query=standalone, answer=answer,
                sources=[s for s in sources if s.number in cited] or sources[:8],
                cited_numbers=cited, dropped_citations=dropped,
                target_sections=["paper-cards"],
                papers=[used[n - 1].source_file for n in cited if 1 <= n <= len(used)],
            )

    if kind == "enumerate":
        findings = gather(
            standalone, embedder, store,
            model=model, reranker=reranker, cards=cards,
        )
        if findings:
            raw = reduce_findings(standalone, findings, model)
            answer, cited, dropped = validate_citations(raw, len(findings))
            sources = [
                Source(
                    number=f.number,
                    source_file=f.paper,
                    section_name=f.hits[0].properties.get("section_name", ""),
                    page_numbers=f.hits[0].properties.get("page_numbers", ""),
                    source_path=f.hits[0].properties.get("source_path", ""),
                    text=f.hits[0].properties.get("text", ""),
                    distance=f.hits[0].distance,
                )
                for f in findings
            ]
            return SynthesisResult(
                query=standalone, answer=answer, sources=sources,
                cited_numbers=cited, dropped_citations=dropped,
                target_sections=["enumeration"],
                papers=[f.paper for f in findings],
            )
        # Few or no papers could contribute. That is the signature of a question
        # *about* the collection rather than about its contents - "which studies
        # here are not about text classification" has no supporting passage
        # anywhere, because no paper states what it is not, so every paper
        # answers NONE. The card table is the only thing that can answer it, and
        # measured alone it took that question from 0/3 facts to 3/3.
        if cards and len(findings) < 2:
            raw, used = answer_from_cards(standalone, cards, model)
            if raw:
                answer, cited, dropped = validate_citations(raw, len(used))
                sources = [
                    Source(
                        number=i + 1, source_file=c.source_file,
                        section_name="paper card", page_numbers="",
                        source_path="", text=c.render(i + 1),
                    )
                    for i, c in enumerate(used)
                ]
                return SynthesisResult(
                    query=standalone, answer=answer,
                    sources=[s for s in sources if s.number in cited] or sources[:8],
                    cited_numbers=cited, dropped_citations=dropped,
                    target_sections=["paper-cards"],
                    papers=[used[n - 1].source_file for n in cited
                            if 1 <= n <= len(used)],
                )
        # Otherwise fall through to ordinary retrieval rather than reporting an
        # empty aggregation.

    # Filter by section only inside one paper, for the same reason heading
    # matching is scoped that way. A corpus-wide question is about a topic, not
    # about a section role: "what causal inference methods are discussed here"
    # classifies as related_work, and filtering on that throws away the
    # econometrics paper the vector search had already ranked first. Measured
    # over the synthesis questions, dropping the filter took facts stated from
    # 61% to 89% and expected papers cited from 50% to 64%, turning two answers
    # that flatly denied the corpus contained the topic into correct ones.
    if section_filter is None:
        section_filter = bool(source_filter)

    target_sections = (
        classify_query(standalone, model_for("routing", model))
        if section_filter else ["general"]
    )
    orig_section_type = (
        target_sections[0]
        if len(target_sections) == 1 and target_sections[0] != "general"
        else None
    )

    query_vector = embedder.embed_one(standalone)
    fetch_k = max(max_sources, top_k) * (RERANK_FETCH_MULTIPLIER if reranker else 1)

    hits, section_filtered = _hierarchical_search(
        query_vector, store, target_sections, fetch_k, top_k,
        source_filter=source_filter, use_section_names=False,
    )
    # Same rule as quick search: the type veto applies only when the section
    # filter actually held, never to global fallback results.
    if hits and section_filtered and orig_section_type:
        hits = prefer_exact_types(
            hits, lambda h: h.properties.get("section_type"), orig_section_type
        )

    # Second retrieval channel: the section summaries, searched directly. They
    # name concrete things - datasets, models, metrics - where a chunk buries the
    # same fact in a sentence, so this surfaces candidates the chunk channel
    # ranks poorly. Unioned rather than substituted: summaries are lossy, and the
    # reranker is what decides between the two sets.
    if section_index is not None and USE_SECTION_CHANNEL:
        try:
            pairs = sections_for_query(
                section_index, query_vector, source_filter=source_filter
            )
            seen = {
                (h.properties.get("source_file"), h.properties.get("chunk_index"))
                for h in hits
            }
            for h in chunks_for_sections(store, pairs):
                key = (h.properties.get("source_file"), h.properties.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        except (OSError, RuntimeError, ValueError):
            pass        # the extra channel is an improvement, never a dependency

    # Third channel: BM25 fused with the vector. Rare proper nouns - AraVec,
    # ArCybC, OSACT5 - have almost no dense neighbourhood and are matched exactly
    # by a keyword index. Unioned like the others; the reranker decides.
    if USE_HYBRID_CHANNEL:
        try:
            seen = {
                (h.properties.get("source_file"), h.properties.get("chunk_index"))
                for h in hits
            }
            for h in store.hybrid_search(
                standalone, query_vector, limit=fetch_k,
                alpha=HYBRID_ALPHA, source_filter=source_filter,
            ):
                key = (h.properties.get("source_file"), h.properties.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        except (OSError, RuntimeError, ValueError):
            pass        # an extra channel is an improvement, never a dependency

    if not hits:
        return SynthesisResult(
            query=standalone,
            answer="Nothing in the indexed papers addresses this question.",
            target_sections=target_sections,
        )

    # Rank first, then spread across papers. Scoping to a single paper makes
    # diversification meaningless, so it is skipped there.
    if reranker:
        hits = reranker.rerank(standalone, hits, top_n=max(max_sources * 3, max_sources))
    hits = (
        hits[:max_sources] if source_filter
        else diversify_by_paper(hits, max_sources, per_paper)
    )

    sources = _build_sources(hits)
    raw = _llm(
        _ANSWER_PROMPT.format(query=standalone, sources=_render_sources(sources)),
        model_for("answer", model),
    )
    answer, cited, dropped = validate_citations(raw, len(sources))

    return SynthesisResult(
        query=standalone,
        answer=answer,
        sources=sources,
        cited_numbers=cited,
        dropped_citations=dropped,
        target_sections=target_sections,
        # Ordered by first appearance so the UI can list papers by prominence.
        papers=list(dict.fromkeys(s.source_file for s in sources)),
    )
