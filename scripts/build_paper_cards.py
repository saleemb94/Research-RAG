"""
Build the two derived indexes: a card per paper, and a searchable section-summary index.

Cards are written during ingest, so this is only needed to backfill a library
that was ingested before cards existed, or to rebuild them after changing the
card prompt. It reads chunk text straight out of Weaviate, so it never re-runs
Docling: rebuilding 17 cards takes under a minute, against roughly half an hour
for a full re-ingest.

    python scripts/build_paper_cards.py            # only papers without a card
    python scripts/build_paper_cards.py --force    # rewrite every card
    python scripts/build_paper_cards.py --show     # print what is stored
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_rag.config import OLLAMA_MODEL  # noqa: E402
from research_rag.embedder import Embedder  # noqa: E402
from research_rag.paper_cards import PaperCardStore, build_card  # noqa: E402
from research_rag.section_index import (  # noqa: E402
    build_entries, index_entries, open_section_index,
)
from research_rag.vector_store import VectorStore, WeaviateUnavailableError  # noqa: E402

# Chunks fed to the card writer, in document order. The front of a paper states
# what it is about; later sections restate it at length.
LEAD_CHUNKS = 14


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Rewrite cards that already exist")
    ap.add_argument("--show", action="store_true", help="Print stored cards and exit")
    ap.add_argument("--sections-only", action="store_true",
                    help="Rebuild only the section-summary index")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    args = ap.parse_args()

    try:
        store = VectorStore()
        cards = PaperCardStore()
    except WeaviateUnavailableError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

    try:
        if args.show:
            existing = cards.all_cards()
            print(f"{len(existing)} card(s):\n")
            for i, c in enumerate(existing, 1):
                print(c.render(i), "\n")
            return

        papers = store.list_sources()
        have = {c.source_file for c in cards.all_cards()}
        todo = papers if args.force else [p for p in papers if p not in have]
        if args.sections_only:
            todo = []
        print(f"{len(papers)} paper(s) indexed, {len(have)} with cards, {len(todo)} to build")

        # The section index is rebuilt either way: it is derived from chunk text,
        # so it goes stale whenever papers change even if every card is current.
        embedder = Embedder()
        t0 = time.perf_counter()
        for i, paper in enumerate(todo, 1):
            hits = store.get_by_section(source_filter=paper)      # document order
            texts = [h.properties["text"] for h in hits[:LEAD_CHUNKS]]
            card = build_card(paper, texts, args.model)
            cards.upsert(card, embedder.embed_one(card.embedding_text() or paper))
            print(f"  [{i}/{len(todo)}] {paper[:46]:48s} "
                  f"{card.discipline[:28]:30s} "
                  f"{len(card.datasets)}d {len(card.models)}m {len(card.languages)}l")
        if todo:
            print(f"\nbuilt {len(todo)} card(s) in {time.perf_counter() - t0:.0f}s; "
                  f"{cards.count()} stored")

        # Section summaries, embedded so they can be searched directly. They are
        # 6% the size of the chunk text and name concrete things, which makes
        # them a second retrieval channel rather than only routing context.
        t0 = time.perf_counter()
        with open_section_index() as index:
            entries = build_entries(store)
            for paper in {e["source_file"] for e in entries}:
                index.delete_source(paper)          # rebuild, never accumulate
            n = index_entries(index, entries, embedder)
            print(f"indexed {n} section summar(ies) in {time.perf_counter() - t0:.0f}s")
    finally:
        store.close()
        cards.close()


if __name__ == "__main__":
    main()
