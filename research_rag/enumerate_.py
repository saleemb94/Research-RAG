"""
Enumeration questions: fan out over papers, then reduce.

"Which named Arabic corpora are used across these papers" cannot be answered by
top-k retrieval, and the reason is structural rather than a tuning failure. The
answer is a *set* whose members are spread one-per-paper, and the terms that
identify them - ArCybC, OSACT5, AraVec - appear in the answer, not in the
question. Dense retrieval smears rare proper nouns, and BM25 can only match
words the query already contains, so neither can surface a member it has not
been told to look for. Widening the pool does not fix it either: measured, more
sources put more of the right papers in front of the model while making it
worse at using them.

So enumeration is treated as an aggregation, not a retrieval: ask each candidate
paper the question separately, discard the papers that have nothing to say, and
reduce what survives into one cited answer. Recall is then bounded by the paper
shortlist rather than by a top-k of chunks.

This costs one LLM call per shortlisted paper. It is reserved for questions that
actually need it, which _is_enumeration decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import OLLAMA_MODEL
from .embedder import Embedder
from .llm import generate as _llm
from .vector_store import SearchHit, VectorStore

# Papers asked individually. Enough to cover a corpus-wide question without the
# latency running away; the shortlist is ranked, so the tail is rarely useful.
MAX_PAPERS = 10

# Chunks shown per paper when asking it the question.
CHUNKS_PER_PAPER = 4

# Candidates pulled before the shortlist is taken. Wide on purpose: this is the
# only stage that decides which papers get a hearing at all.
SHORTLIST_POOL = 300

# A paper's contribution is a short list of names, so the extraction call is
# capped. Without a cap the model writes prose about the items instead of
# listing them, and since generation dominates the wall clock, that alone took
# the fan-out from seconds per paper to tens of seconds.
EXTRACT_MAX_TOKENS = 160

_DETECT_PROMPT = """\
Decide whether answering this question requires gathering items from MANY papers
and combining them, or whether one focused passage would answer it.

Answer ENUMERATE if the question asks which/what things appear across a
collection - languages, datasets, models, platforms, techniques, scores - so a
complete answer is a list assembled from several papers.

Answer FOCUSED if a single passage would answer it, including questions about
one specific study, one number, or one definition.

Examples:
  "Which datasets have been used for Arabic offensive language detection?" -> ENUMERATE
  "What preprocessing steps are commonly applied to tweets?" -> ENUMERATE
  "Which transformer models have been fine-tuned across these papers?" -> ENUMERATE
  "What accuracy did the CNN-BiLSTM-GRU model achieve?" -> FOCUSED
  "What is retrieval-augmented generation?" -> FOCUSED
  "How were patients randomised in the TechStep trial?" -> FOCUSED

Question: {query}

Reply with one word, ENUMERATE or FOCUSED:"""

_EXTRACT_PROMPT = """\
Below are excerpts from ONE research paper, "{paper}".

Question: {query}

List ONLY the specific items from this paper that answer the question - names,
datasets, models, languages, figures - one short line each. Name them exactly as
the excerpts do.

Every item must satisfy the WHOLE question, not part of it. If the question names
something this paper does not do - a technique it never uses, a language it never
covers, an experiment it never ran - reply NONE, even when the paper is about a
related topic and even when it would answer some other part of the question.

If this paper contains nothing that answers the question, reply with exactly:
NONE

Excerpts:
{excerpts}

Items:"""

_REDUCE_PROMPT = """\
You are a research assistant. Below is what each paper contributes to one
question. Combine them into a single answer.

Question: {query}

Write ONE continuous answer that gathers the items across papers. Cite each item
with the source number in square brackets, like [2]. The ONLY valid source
numbers are 1 to {n_sources}; never write any other number in brackets. Group
related items rather than listing paper by paper.

Check the findings actually answer the question that was asked. If they only
address part of it, or if the question assumes something none of these papers
did, say so plainly instead of assembling an answer out of loosely related
items.

Findings:
{findings}

Answer:"""

_NONE = re.compile(r"^\s*(none|n/?a|nothing|no\b)", re.I)

# Papers cite their own references as [50], [124]. Those markers ride along in the
# excerpts, the extraction step copies them into its item list, and the reduce
# step then emits them in the answer, where they look like source numbers
# pointing at sources that do not exist. Strip them from each paper's findings so
# the only brackets the reducer ever sees are the ones assigned here.
_REF_MARKER = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")


def strip_reference_markers(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", _REF_MARKER.sub("", text)).strip()


@dataclass
class PaperFinding:
    paper: str
    number: int
    items: str
    hits: list[SearchHit]


def is_enumeration(query: str, model: str = OLLAMA_MODEL) -> bool:
    """One cheap call to decide whether the expensive path is warranted."""
    try:
        verdict = _llm(_DETECT_PROMPT.format(query=query), model)
    except Exception:
        return False        # never let the detector break a normal question
    return "enumerate" in verdict.strip().lower()[:40]


def shortlist_papers(
    query_vector: list[float],
    store: VectorStore,
    limit: int = MAX_PAPERS,
) -> list[str]:
    """
    Papers worth asking, ranked by their best-matching chunk.

    Deliberately a wide, unfiltered vector sweep: this is the recall ceiling for
    the whole answer, and a paper omitted here can never contribute.
    """
    hits = store.search(query_vector, limit=SHORTLIST_POOL)
    order: list[str] = []
    for h in hits:
        paper = h.properties.get("source_file", "")
        if paper and paper not in order:
            order.append(paper)
        if len(order) == limit:
            break
    return order


def gather(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    model: str = OLLAMA_MODEL,
    reranker=None,
    max_papers: int = MAX_PAPERS,
    chunks_per_paper: int = CHUNKS_PER_PAPER,
) -> list[PaperFinding]:
    """Ask each shortlisted paper the question; keep the ones that answer."""
    query_vector = embedder.embed_one(query)
    papers = shortlist_papers(query_vector, store, max_papers)

    findings: list[PaperFinding] = []
    for paper in papers:
        hits = store.search(
            query_vector,
            limit=chunks_per_paper * (3 if reranker else 1),
            source_filter=paper,
        )
        if reranker and hits:
            hits = reranker.rerank(query, hits, top_n=chunks_per_paper)
        hits = hits[:chunks_per_paper]
        if not hits:
            continue

        excerpts = "\n\n".join(
            f"({h.properties.get('section_name') or 'n/a'}) {h.properties['text'][:800]}"
            for h in hits
        )
        try:
            items = _llm(
                _EXTRACT_PROMPT.format(paper=paper, query=query, excerpts=excerpts),
                model,
                max_tokens=EXTRACT_MAX_TOKENS,
            )
        except Exception:
            continue
        items = items.strip()
        # A paper with nothing to contribute must drop out, or the reduce step
        # spends its attention explaining absences.
        if not items or _NONE.match(items):
            continue
        findings.append(
            PaperFinding(
                paper=paper,
                number=len(findings) + 1,
                items=strip_reference_markers(items),
                hits=hits,
            )
        )
    return findings


def reduce_findings(query: str, findings: list[PaperFinding], model: str) -> str:
    if not findings:
        return "Nothing in the indexed papers addresses this question."
    rendered = "\n\n".join(
        f"[{f.number}] {f.paper}\n{f.items[:700]}" for f in findings
    )
    return _llm(
        _REDUCE_PROMPT.format(
            query=query, findings=rendered, n_sources=len(findings)
        ),
        model,
    )
