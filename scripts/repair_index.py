"""
Repair parsing defects in an already-ingested index, without re-parsing.

Everything here rewrites or deletes stored objects, so it costs about a minute
rather than the half hour a full re-ingest of a large library would. The same
rules also run at ingest, so a paper added later never needs this.

    python scripts/repair_index.py --dry-run
    python scripts/repair_index.py

Three defects, each measured before it was written:

  noise chunks    Docling emits "<!-- formula-not-decoded -->" for formulas it
                  cannot parse, and those index as chunks. Add bare figure
                  captions, fragments with fewer than 15 letters, and publisher
                  furniture, and 14.8% of a 54-paper index carried no content
                  at all. Verified against the golden set: removing all of it
                  loses zero facts.

  front matter    A paper's first heading is its title, and it was being kept
                  as a section - so the abstract was never labeled `abstract`
                  and section filtering could not reach it. 17 of 54 papers.
                  The old rule guessed from length and vocabulary and was
                  defeated by titles containing ordinary section words:
                  "analysis", "approach", "method", "evaluation". Now the
                  heading is compared against the title we resolve anyway.

  caption headings  "Algorithm 1 XAI-based word replacement algorithm." became
                  a section heading owning 36 chunks, which lost their real
                  section. Captions now inherit the heading above them.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from weaviate.util import generate_uuid5  # noqa: E402

from research_rag.chunk_rules import (  # noqa: E402
    FRONT_MATTER_SECTION, is_caption_heading, noise_verdict, same_title,
)
from research_rag.paper_cards import PaperCardStore  # noqa: E402
from research_rag.vector_store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with VectorStore() as store, PaperCardStore() as cards:
        rows = store._scan(None, ["source_file", "chunk_index", "text",
                                  "section_name", "section_type", "heading"])
        titles = {c.source_file: c.title for c in cards.all_cards() if c.title}

    by_paper: dict[str, list[dict]] = {}
    for r in rows:
        by_paper.setdefault(r["source_file"], []).append(r)

    drop: list[tuple[str, int]] = []
    patch: dict[tuple[str, int], dict] = {}
    tags: Counter = Counter()

    for paper, chunks in by_paper.items():
        chunks.sort(key=lambda r: r.get("chunk_index") or 0)
        title = titles.get(paper, "")

        # Carry the last heading that was a real section, so a caption can
        # inherit it rather than becoming one.
        last_real_name = ""
        last_real_type = ""

        for r in chunks:
            idx = r.get("chunk_index") or 0
            key = (paper, idx)

            verdict = noise_verdict(r.get("text"))
            if verdict:
                drop.append(key)
                tags[verdict] += 1
                continue

            name = (r.get("section_name") or "").strip()
            new_name, new_type = None, None

            if title and same_title(name, title):
                # The paper's own title, wherever it appears.
                if name != FRONT_MATTER_SECTION:
                    new_name, new_type = FRONT_MATTER_SECTION, "abstract"
                    tags["front matter relabeled"] += 1
            elif is_caption_heading(name):
                if last_real_name:
                    new_name = last_real_name
                    new_type = last_real_type or r.get("section_type")
                    tags["caption heading reparented"] += 1
            else:
                last_real_name = name or last_real_name
                last_real_type = r.get("section_type") or last_real_type

            if new_name is not None:
                props = {"section_name": new_name}
                if new_type:
                    props["section_type"] = new_type
                patch[key] = props

    n = len(rows)
    print(f"{n} chunks across {len(by_paper)} papers\n")
    print("  planned changes:")
    for k, v in tags.most_common():
        print(f"    {k:28} {v:5}")
    print(f"    {'-> chunks deleted':28} {len(drop):5}  ({len(drop)/n:.1%})")
    print(f"    {'-> chunks relabeled':28} {len(patch):5}  ({len(patch)/n:.1%})")

    if a.dry_run:
        print("\n  dry run - nothing written")
        return 0

    with VectorStore() as store:
        col = store._col
        for paper, idx in drop:
            col.data.delete_by_id(generate_uuid5(f"{paper}::{idx}"))
        for (paper, idx), props in patch.items():
            col.data.update(uuid=generate_uuid5(f"{paper}::{idx}"),
                            properties=props)
        print(f"\n  applied. index now holds {store.count_chunks()} chunks")

    # Section summaries are derived from section names, several of which just
    # changed, so the second retrieval channel has to be rebuilt.
    print("  rebuilding the section-summary index...")
    from research_rag import Embedder
    from research_rag.section_index import (
        build_entries, index_entries, open_section_index,
    )
    embedder = Embedder()
    with VectorStore() as store, open_section_index() as index:
        entries = build_entries(store)
        for paper in {e["source_file"] for e in entries}:
            index.delete_source(paper)
        print(f"  section index: {index_entries(index, entries, embedder)} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
