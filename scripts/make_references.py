"""
Write a reference answer for each gold-passage question, from its gold passages.

    python scripts/make_references.py        ->  tests/golden_references.json

What this produces is a ceiling, not a right answer. The model is handed exactly
the passages a perfect retriever would have found and asked the same question the
live system gets, so the reference is what this system would say if retrieval
never made a mistake. Comparing the live answer against it therefore isolates the
cost of imperfect retrieval from the limits of the model doing the writing: a low
score means retrieval lost something, not that the model cannot express it.

That framing is the whole point, and it is worth being precise about what it
excludes. Agreement with this reference is not correctness. If the model
misreads a passage it will misread it identically on both sides and score a
perfect match, so this metric cannot see that failure at all. Factual accuracy
is what the fact groups in golden_qa.json are for; these two measure different
things and neither substitutes for the other.

Generated once and committed, so scoring runs do not depend on regenerating them
and a change in the numbers is a change in the system rather than in the yardstick.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_rag.config import OLLAMA_MODEL          # noqa: E402
from research_rag.llm import generate as _llm         # noqa: E402
from research_rag.vector_store import VectorStore     # noqa: E402

GOLD = ROOT / "tests" / "golden_passages.json"
OUT = ROOT / "tests" / "golden_references.json"

_PROMPT = """Answer the question using only the passages below. Be specific: name
the models, datasets, numbers and findings the passages actually state. Do not add
information that is not in them, and do not mention the passages or their numbering.

Question: {question}

Passages:
{passages}

Answer:"""


def main() -> int:
    items = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    with VectorStore() as store:
        by_paper: dict[str, dict[int, str]] = {}
        for row in store._scan(None, ["source_file", "chunk_index", "text"]):
            by_paper.setdefault(row["source_file"], {})[row["chunk_index"]] = row["text"]

        out = []
        for i, it in enumerate(items, 1):
            chunks = by_paper.get(it["paper"], {})
            # Grade 2 first: a passage that fully answers should lead the prompt,
            # and a supporting one should not crowd it out.
            gold = sorted(it["gold"], key=lambda g: (-g["grade"], g["chunk_index"]))
            passages = [chunks.get(g["chunk_index"], "") for g in gold]
            passages = [p for p in passages if p]
            if not passages:
                print(f"  [{i}/{len(items)}] {it['id']}: no gold text, skipped")
                continue
            body = "\n\n".join(f"- {' '.join(p.split())}" for p in passages)
            answer = _llm(
                _PROMPT.format(question=it["question"], passages=body),
                OLLAMA_MODEL,
                temperature=0.0,       # a yardstick that moves is not a yardstick
            )
            answer = " ".join((answer or "").split())
            print(f"  [{i}/{len(items)}] {it['id']}: {len(answer)} chars")
            out.append({
                "id": it["id"],
                "question": it["question"],
                "paper": it["paper"],
                "n_gold": len(passages),
                "reference": answer,
            })

    OUT.write_text(
        json.dumps({
            "note": ("Reference answers written from the gold passages of "
                     "golden_passages.json, at temperature 0. They are the "
                     "perfect-retrieval ceiling for this model, not ground truth: "
                     "agreement with them measures what retrieval cost, not "
                     "whether the answer is factually right."),
            "items": out,
        }, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT.relative_to(ROOT)} ({len(out)} references)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
