"""
Re-embed every chunk with its paper title and section as context.

Backfill for research_rag/context.py. The stored text is untouched - only the
vector changes - so citations, the source panel and every answer still show the
passage exactly as before.

    python scripts/reembed_with_context.py --dry-run
    python scripts/reembed_with_context.py

Cheap enough to be worth doing rather than reasoning about: embedding is 1.6%
of ingestion time, so a few thousand chunks take seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from weaviate.util import generate_uuid5  # noqa: E402

from research_rag import Embedder  # noqa: E402
from research_rag.context import context_prefix, embedding_text  # noqa: E402
from research_rag.paper_cards import PaperCardStore  # noqa: E402
from research_rag.vector_store import VectorStore  # noqa: E402

BATCH = 256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with VectorStore() as store, PaperCardStore() as cards:
        rows = store._scan(None, ["source_file", "chunk_index", "text",
                                  "section_name", "subsection_name"])
        titles = {c.source_file: c.title for c in cards.all_cards() if c.title}

    print(f"{len(rows)} chunks, {len(titles)} titled papers")
    sample = rows[0]
    print("\n  example prefix:")
    print("   ", context_prefix(titles.get(sample['source_file']),
                                sample.get('section_name'),
                                sample.get('subsection_name'))[:100])
    n_prefixed = sum(
        1 for r in rows
        if context_prefix(titles.get(r["source_file"]), r.get("section_name"),
                          r.get("subsection_name"))
    )
    print(f"\n  chunks that gain context: {n_prefixed}/{len(rows)} "
          f"({n_prefixed / len(rows):.0%})")

    if a.dry_run:
        print("\n  dry run - nothing written")
        return 0

    embedder = Embedder()
    with VectorStore() as store:
        col = store._col
        done = 0
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            texts = [
                embedding_text(r.get("text") or "",
                               titles.get(r["source_file"]),
                               r.get("section_name"), r.get("subsection_name"))
                for r in batch
            ]
            vectors = embedder.embed(texts)
            for r, v in zip(batch, vectors):
                col.data.update(
                    uuid=generate_uuid5(f"{r['source_file']}::{r['chunk_index']}"),
                    vector=v,
                )
            done += len(batch)
            print(f"  re-embedded {done}/{len(rows)}")
    print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
