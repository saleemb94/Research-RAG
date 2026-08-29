"""
Paper cards: one structured record per paper, built at ingest.

Chunk retrieval answers questions whose answer sits *in* a passage. It cannot
answer questions about the corpus itself - "which studies here are not about
text classification" has no supporting passage anywhere, because no paper says
what it is not. And enumerating across papers by retrieval means one LLM call
per paper, which is correct but grows linearly with the library.

A card fixes both. Each paper gets a small structured record - what it studies,
which datasets, models, languages, platforms and headline numbers it names -
written once at ingest. Seventeen cards fit in a single prompt, so enumeration
becomes one call over a table instead of N calls over chunks, and questions
about the collection become answerable at all.

Cards are routing and enumeration metadata. Answers built from them cite the
paper, and the chunk-level path remains the way to quote a passage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from .config import (
    OLLAMA_MODEL,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
)
from .llm import generate as _llm
from .vector_store import WeaviateUnavailableError

CARD_COLLECTION = "ResearchPaperCard"

# Text handed to the card writer. The opening of a paper carries what it is
# about; the rest repeats it at greater length.
_CARD_CONTEXT_CHARS = 6000

# Cards are short by construction, so the writer is capped: without it the model
# writes prose around the JSON and the parse gets harder, not richer.
_CARD_MAX_TOKENS = 420

# Papers report what they found and what went wrong, and a corpus-level question
# often asks after exactly that - "what limitations do these papers report", "what
# weaknesses of LLMs are described". Without these two fields the card schema had
# nowhere to put such content, so those questions were routed to a mechanism
# structurally incapable of answering them: "hallucination" and "traceability"
# appeared in none of the seventeen cards while sitting in six chunks each.
# "limitations" earns its place: it is what let a question about reported
# weaknesses be answered at all, and costs 33 characters across the whole table.
# A "findings" field was tried alongside it and removed - the model wrote
# sentences rather than names, 2157 characters for 25 items, which grew the card
# table by 60% and cost more elsewhere than it gained. What a paper found is
# already in `summary`; what went wrong was the genuine gap.
_LIST_FIELDS = ("datasets", "models", "languages", "platforms", "metrics",
                "methods", "limitations")

_CARD_PROMPT = """\
Below are excerpts from the beginning of one research paper.

Return a JSON object describing it, with exactly these keys:

  "summary"    one sentence, max 30 words, on what the paper studies
  "discipline" the field, e.g. "NLP / hate speech detection", "epidemiology",
               "econometrics", "clinical trial", "human-computer interaction"
  "datasets"   named datasets or corpora it uses, e.g. ["OSACT5", "ArCybC"]
  "models"     named models or architectures, e.g. ["AraBERT", "CNN-BiLSTM-GRU"]
  "languages"  languages of the data studied, e.g. ["Arabic", "English"]
  "platforms"  where data came from, e.g. ["Twitter", "Reddit"]
  "metrics"    headline results with their numbers, e.g. ["accuracy 98.83%"]
  "methods"    key techniques, e.g. ["LIME", "genetic algorithm"]
  "limitations" weaknesses, failure modes or caveats the paper reports - its own
               or those of the technology it studies. Short phrases, not
               sentences, e.g. ["hallucination", "poor traceability"]

Use ONLY names that appear in the excerpts. Never invent a dataset or a number.
Use an empty list for anything the paper does not have - a theoretical paper has
no datasets, and that is the correct answer, not a guess.

Excerpts:
{text}

JSON:"""

_ANSWER_PROMPT = """\
Below is a card for every paper in a library, listing what each one studies and
the datasets, models, languages, platforms, metrics, methods and
limitations it reports.

Question: {query}

Answer using ONLY these cards. Cite each claim with the paper's number in square
brackets, like [2]; the only valid numbers are 1 to {n}. Write one continuous
answer, not a list of papers.

Read every line of a card, not just its field label and summary. A paper whose
field says "clinical trial" may still list the technique being asked about in its
methods, and judging it by the label alone would wrongly exclude it.

If the cards do not answer the question, or the question assumes something none
of these papers did, say so plainly instead of assembling loosely related items.

Cards:
{cards}

Answer:"""


@dataclass
class PaperCard:
    source_file: str
    summary: str = ""
    discipline: str = ""
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def as_properties(self) -> dict:
        d = {"source_file": self.source_file, "summary": self.summary,
             "discipline": self.discipline}
        for f in _LIST_FIELDS:
            d[f] = [str(x)[:120] for x in getattr(self, f) if str(x).strip()][:12]
        return d

    def render(self, number: int) -> str:
        lines = [f"[{number}] {self.source_file}"]
        if self.discipline:
            lines.append(f"    field: {self.discipline}")
        if self.summary:
            lines.append(f"    about: {self.summary}")
        for f in _LIST_FIELDS:
            vals = getattr(self, f)
            if vals:
                lines.append(f"    {f}: {', '.join(vals[:10])}")
        return "\n".join(lines)

    def embedding_text(self) -> str:
        parts = [self.summary, self.discipline]
        parts += [", ".join(getattr(self, f)) for f in _LIST_FIELDS]
        return " | ".join(p for p in parts if p)


def _parse_card(source_file: str, raw: str) -> PaperCard:
    """
    Pull a card out of whatever the model returned.

    A small model wraps JSON in prose, fences it, or trails a comment, so the
    first balanced object is extracted rather than parsing the whole reply. A
    card that fails to parse still yields the summary text, which is better than
    no card at all.
    """
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    def as_list(v) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [p.strip() for p in v.split(",") if p.strip()]
        return []

    return PaperCard(
        source_file=source_file,
        summary=str(data.get("summary") or "").strip()[:400]
        or " ".join(raw.split())[:200],
        discipline=str(data.get("discipline") or "").strip()[:120],
        **{f: as_list(data.get(f)) for f in _LIST_FIELDS},
    )


def build_card(source_file: str, chunks_text: list[str], model: str = OLLAMA_MODEL) -> PaperCard:
    text = "\n\n".join(chunks_text)[:_CARD_CONTEXT_CHARS]
    if not text.strip():
        return PaperCard(source_file=source_file)
    try:
        raw = _llm(_CARD_PROMPT.format(text=text), model, max_tokens=_CARD_MAX_TOKENS)
    except Exception as exc:                      # a card is never worth failing an ingest
        print(f"    ! card failed for {source_file}: {exc}")
        return PaperCard(source_file=source_file)
    return _parse_card(source_file, raw)


class PaperCardStore:
    """Weaviate-backed store for one card per paper."""

    def __init__(
        self,
        host: str = WEAVIATE_HOST,
        port: int = WEAVIATE_PORT,
        grpc_port: int = WEAVIATE_GRPC_PORT,
        collection: str = CARD_COLLECTION,
    ):
        self._client = None
        self._name = collection
        try:
            self._client = weaviate.connect_to_local(
                host=host, port=port, grpc_port=grpc_port
            )
        except Exception as exc:
            raise WeaviateUnavailableError(
                f"Could not connect to Weaviate at {host}:{port}.\n"
                f"Start it with:  docker compose up -d\nOriginal error: {exc}"
            ) from exc
        self._ensure()
        self._col = self._client.collections.get(collection)

    def _ensure(self):
        if self._client.collections.exists(self._name):
            col = self._client.collections.get(self._name)
            have = {p.name for p in col.config.get().properties}
            for f in _LIST_FIELDS:
                if f not in have:
                    col.config.add_property(
                        Property(name=f, data_type=DataType.TEXT_ARRAY)
                    )
            return
        self._client.collections.create(
            name=self._name,
            description="One structured card per paper, for corpus-level questions.",
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="source_file", data_type=DataType.TEXT,
                         tokenization=Tokenization.FIELD),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="discipline", data_type=DataType.TEXT),
                *[Property(name=f, data_type=DataType.TEXT_ARRAY) for f in _LIST_FIELDS],
            ],
        )

    def upsert(self, card: PaperCard, vector: list[float] | None = None):
        uid = generate_uuid5(f"card::{card.source_file}")
        # Cards are rewritten whenever a paper is re-ingested, so replace rather
        # than accumulate.
        self._col.data.delete_many(
            where=Filter.by_property("source_file").equal(card.source_file)
        )
        self._col.data.insert_many(
            [DataObject(properties=card.as_properties(),
                        vector=vector or [0.0], uuid=uid)]
        )

    def all_cards(self) -> list[PaperCard]:
        out: list[PaperCard] = []
        for obj in self._col.iterator():
            p = obj.properties
            out.append(PaperCard(
                source_file=p.get("source_file") or "",
                summary=p.get("summary") or "",
                discipline=p.get("discipline") or "",
                **{f: list(p.get(f) or []) for f in _LIST_FIELDS},
            ))
        return sorted(out, key=lambda c: c.source_file)

    def count(self) -> int:
        return self._col.aggregate.over_all(total_count=True).total_count or 0

    def delete(self, source_file: str):
        self._col.data.delete_many(
            where=Filter.by_property("source_file").equal(source_file)
        )

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def answer_from_cards(
    query: str, cards: list[PaperCard], model: str = OLLAMA_MODEL
) -> tuple[str, list[PaperCard]]:
    """Answer a corpus-level question from the card table. One LLM call, any size."""
    if not cards:
        return "", []
    rendered = "\n".join(c.render(i + 1) for i, c in enumerate(cards))
    answer = _llm(
        _ANSWER_PROMPT.format(query=query, cards=rendered, n=len(cards)), model
    )
    return answer, cards
