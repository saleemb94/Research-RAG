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

from .config import HYBRID_ALPHA, OLLAMA_MODEL, USE_HYBRID_CHANNEL
from .embedder import Embedder
from .llm import generate as _llm
from .llm import model_for
from .paper_cards import PaperCard
from .vector_store import SearchHit, VectorStore

# Papers asked individually. Enough to cover a corpus-wide question without the
# latency running away; the shortlist is ranked, so the tail is rarely useful.
MAX_PAPERS = 10

# Chunks shown per paper when asking it the question.
#
# Eight rather than four because the reranker, not retrieval, was the binding
# constraint. Traced on two facts the pipeline kept missing: the paper was in the
# shortlist, the chunk naming "Mechanical Turk" and "AraVec" was in the candidate
# pool, and the cross-encoder then dropped both out of the top four. It is a
# small general-purpose model scoring passage-answers-query, and a passage that
# merely *names* the thing asked about scores below one that discusses the topic
# at length. Cutting at eight keeps them; dropping the reranker entirely does
# not, so it earns its place - it was simply cutting too deep.
CHUNKS_PER_PAPER = 8

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

Answer CORPUS if the question is about the collection itself: which papers are
or are not about something, what fields the library covers, how the papers
relate. Answering needs to know what each paper IS, and no single passage in any
paper states that.

Answer ENUMERATE if the question gathers items from many papers into a list -
languages, datasets, models, platforms, techniques, scores.

Answer FOCUSED if a single passage would answer it, including questions about
one specific study, one number, or one definition.

Examples:
  "Which studies here are not about text classification?" -> CORPUS
  "What fields do these papers cover?" -> CORPUS
  "Which of these are the theoretical papers?" -> CORPUS
  "Which datasets have been used for Arabic offensive language detection?" -> ENUMERATE
  "What preprocessing steps are commonly applied to tweets?" -> ENUMERATE
  "Which transformer models have been fine-tuned across these papers?" -> ENUMERATE
  "What accuracy did the CNN-BiLSTM-GRU model achieve?" -> FOCUSED
  "What is retrieval-augmented generation?" -> FOCUSED
  "How were patients randomised in the TechStep trial?" -> FOCUSED

Question: {query}

Reply with one word, CORPUS or ENUMERATE or FOCUSED:"""

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

Say plainly that the collection does not cover it ONLY when nothing in the
findings is relevant, or when the question assumes something none of these
papers did. A question with several parts is not such a case: answer the parts
the findings do support and note briefly what the papers do not cover. A partial
answer is more useful than a refusal.

Findings:
{findings}

Answer:"""

_SELECT_PROMPT = """\
Below is a one-line card for every paper in a library.

Question: {query}

Which papers could contribute something to this question? Judge from what each
card says the paper is about: a paper on a different subject contributes nothing,
and a paper is worth including even when the card does not list the exact item,
as long as its subject fits.

Reply with the numbers only, comma-separated, for example: 2, 5, 9
Reply "none" if no paper fits.

Cards:
{cards}

Numbers:"""

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


def classify_question(query: str, model: str = OLLAMA_MODEL) -> str:
    """
    One cheap call routing a question to the machinery that can answer it.

    Returns "corpus", "enumerate" or "focused". The three need genuinely
    different mechanisms, which is why this is a routing decision rather than a
    tuning knob: a question about the collection has no supporting passage
    anywhere and can only come from the paper cards; an enumeration needs a
    fan-out because its members sit one per paper; everything else is ordinary
    retrieval.
    """
    try:
        verdict = _llm(_DETECT_PROMPT.format(query=query),
                       model_for("routing", model), max_tokens=8)
    except (OSError, RuntimeError, ValueError):
        return "focused"    # a transport hiccup must not change the answer shape
    v = verdict.strip().lower()[:40]
    if "corpus" in v:
        return "corpus"
    return "enumerate" if "enumerate" in v else "focused"


def is_enumeration(query: str, model: str = OLLAMA_MODEL) -> bool:
    return classify_question(query, model) == "enumerate"


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


def select_by_card(
    query: str, cards: list[PaperCard], model: str = OLLAMA_MODEL
) -> list[str]:
    """
    Ask the card table which papers are worth interrogating.

    Chunk similarity picks papers whose *wording* matches the question, which is
    the wrong test when the question is about subject matter. "Which studies are
    not about text classification" retrieves text-classification papers, because
    no passage anywhere states what a paper is not. A card says what each paper
    is, so the shortlist can be drawn on subject rather than on phrasing.
    """
    if not cards:
        return []
    rendered = "\n".join(
        f"[{i + 1}] {c.source_file} - {c.discipline or 'unknown field'}: {c.summary}"
        for i, c in enumerate(cards)
    )
    try:
        raw = _llm(_SELECT_PROMPT.format(query=query, cards=rendered),
                   model_for("routing", model), max_tokens=60)
    except (OSError, RuntimeError, ValueError):
        # Only transport and model failures degrade to "no card opinion"; a
        # programming error must surface rather than look like an empty result.
        return []
    if _NONE.match(raw.strip()):
        return []
    picked = []
    for n in re.findall(r"\d+", raw):
        i = int(n) - 1
        if 0 <= i < len(cards) and cards[i].source_file not in picked:
            picked.append(cards[i].source_file)
    return picked


def gather(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    model: str = OLLAMA_MODEL,
    reranker=None,
    max_papers: int = MAX_PAPERS,
    chunks_per_paper: int = CHUNKS_PER_PAPER,
    cards: list[PaperCard] | None = None,
) -> list[PaperFinding]:
    """Ask each shortlisted paper the question; keep the ones that answer."""
    query_vector = embedder.embed_one(query)
    # Measured: unioning a card-chosen shortlist in front of the vector one made
    # things worse (facts 74% -> 64%), because card picks displaced vector picks
    # under the cap while adding nothing the extraction step could use. The cards
    # earn their place elsewhere - see the fallback in synthesize_answer.
    papers = shortlist_papers(query_vector, store, max_papers)

    findings: list[PaperFinding] = []
    for paper in papers:
        # Both channels inside the paper too, not just at corpus level. This was
        # the gap: enumeration questions are precisely the ones asking after rare
        # proper nouns, and the per-paper retrieval was pure vector. Measured on
        # "how many annotators", vector spent all four slots on Related Work
        # while hybrid went to Dataset Collection, where the answer is.
        pool = store.search(
            query_vector,
            limit=chunks_per_paper * (3 if reranker else 1),
            source_filter=paper,
        )
        if USE_HYBRID_CHANNEL:
            seen = {
                (h.properties.get("source_file"), h.properties.get("chunk_index"))
                for h in pool
            }
            try:
                for h in store.hybrid_search(
                    query, query_vector,
                    limit=chunks_per_paper * (3 if reranker else 1),
                    alpha=HYBRID_ALPHA, source_filter=paper,
                ):
                    key = (h.properties.get("source_file"),
                           h.properties.get("chunk_index"))
                    if key not in seen:
                        seen.add(key)
                        pool.append(h)
            except (OSError, RuntimeError, ValueError):
                pass        # an extra channel is an improvement, never a dependency

        if reranker and pool:
            pool = reranker.rerank(query, pool, top_n=chunks_per_paper)
        hits = pool[:chunks_per_paper]
        if not hits:
            continue

        excerpts = "\n\n".join(
            f"({h.properties.get('section_name') or 'n/a'}) {h.properties['text'][:800]}"
            for h in hits
        )
        try:
            items = _llm(
                _EXTRACT_PROMPT.format(paper=paper, query=query, excerpts=excerpts),
                model_for("extract", model),
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


NOTHING_FOUND = "Nothing in the indexed papers addresses this question."


def reduce_findings(query: str, findings: list[PaperFinding], model: str) -> str:
    if not findings:
        return NOTHING_FOUND
    rendered = "\n\n".join(
        f"[{f.number}] {f.paper}\n{f.items[:700]}" for f in findings
    )
    answer = _llm(
        _REDUCE_PROMPT.format(
            query=query, findings=rendered, n_sources=len(findings)
        ),
        model_for("answer", model),
    )
    # The reducer sometimes echoes the extraction step's sentinel and replies with
    # a bare "NONE". It is the correct verdict, but a user reading it sees a
    # stray token rather than an answer, so it is turned back into a sentence.
    if not answer.strip() or _NONE.match(answer.strip()):
        return NOTHING_FOUND
    return answer
