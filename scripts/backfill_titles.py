"""
Resolve real titles for papers that were ingested before titles existed.

No re-parsing: a title comes from the PDF's own metadata, the largest text on
its first page, or the model reading that page - all of which are available
straight from the file. Most papers resolve from metadata alone, which costs
nothing.

    python scripts/backfill_titles.py --dry-run    # show what it would set
    python scripts/backfill_titles.py              # write them to the cards
    python scripts/backfill_titles.py --force      # re-resolve titles already set

A paper whose title cannot be resolved keeps its filename in the interface,
which is the same behavior as before, so nothing is lost by failing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_rag import Embedder  # noqa: E402
from research_rag.config import OLLAMA_MODEL, PAPERS_DIR  # noqa: E402
from research_rag.paper_cards import PaperCard, PaperCardStore  # noqa: E402
from research_rag.titles import extract_title  # noqa: E402
from research_rag.vector_store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-resolve papers that already have a title")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    a = ap.parse_args()

    papers_dir = Path(PAPERS_DIR)
    # A card's vector has to be rewritten with it: upsert replaces the object,
    # and its default for a missing vector is a one-dimensional zero, which
    # would quietly destroy the card index that corpus questions rely on.
    embedder = Embedder() if not a.dry_run else None

    with VectorStore() as store, PaperCardStore() as cards:
        indexed = sorted(store.list_sources())
        by_file = {c.source_file: c for c in cards.all_cards()}

        print(f"{len(indexed)} paper(s) in the index, "
              f"{len(by_file)} card(s), "
              f"{sum(1 for c in by_file.values() if c.title)} already titled\n")

        how_counts: Counter = Counter()
        set_count = skipped = missing_file = unresolved = 0

        for name in indexed:
            card = by_file.get(name)
            if card and card.title and not a.force:
                skipped += 1
                continue

            path = papers_dir / name
            if not path.exists():
                # The chunks are indexed but the file is gone, so there is
                # nothing left to read a title out of.
                print(f"  --  {name[:58]:60} (no file on disk)")
                missing_file += 1
                continue

            title, how = extract_title(str(path), a.model)
            how_counts[how or "none"] += 1
            if not title:
                print(f"  --  {name[:58]:60} (unresolved, keeps its filename)")
                unresolved += 1
                continue

            print(f"  {how[:4]:4} {name[:34]:36} -> {title[:64]}")
            if not a.dry_run:
                # Keep everything else the card holds; only the title changes.
                target = card or PaperCard(source_file=name)
                target.title = title
                cards.upsert(
                    target,
                    embedder.embed_one(target.embedding_text() or name),
                )
                set_count += 1

        print()
        print(f"  resolved from : {dict(how_counts)}")
        print(f"  titles written: {set_count}"
              f"{'  (dry run - nothing written)' if a.dry_run else ''}")
        print(f"  already titled: {skipped}")
        print(f"  unresolved    : {unresolved}")
        print(f"  file missing  : {missing_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
