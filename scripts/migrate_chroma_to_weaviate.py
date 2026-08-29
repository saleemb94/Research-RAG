"""
One-time migration: copy an existing ChromaDB index into Weaviate.

This project used a local ChromaDB persistent client before moving to Weaviate.
Re-ingesting means re-running Docling conversion plus LLM section classification
on every PDF, which is slow; this script copies the already-computed chunks,
embeddings and metadata straight across instead.

Requirements:
    pip install chromadb          # only needed for this script
    docker compose up -d          # Weaviate must be running

Usage:
    python scripts/migrate_chroma_to_weaviate.py
    python scripts/migrate_chroma_to_weaviate.py --chroma-path ./chroma_data --dry-run

Safe to re-run: chunk UUIDs in Weaviate are derived from source_file +
chunk_index, so a second run overwrites rather than duplicating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_rag.vector_store import VectorStore, WeaviateUnavailableError  # noqa: E402

DEFAULT_CHROMA_PATH = "./chroma_data"
DEFAULT_CHROMA_COLLECTION = "research_chunks"
BATCH = 500

# Properties the Weaviate schema declares. Anything else in the old Chroma
# metadata is dropped rather than silently failing the insert.
KNOWN_PROPS = {
    "text",
    "source_file",
    "source_path",
    "chunk_index",
    "heading",
    "section_name",
    "subsection_name",
    "section_type",
    "page_numbers",
}


def read_chroma(path: str, collection: str) -> list[dict]:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print(
            "chromadb is not installed. This script needs it only to read the old "
            "index:\n    pip install chromadb",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not Path(path).exists():
        print(f"No Chroma data found at {path}", file=sys.stderr)
        raise SystemExit(1)

    client = chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )
    # chromadb >= 0.6 returns collection names; older versions return objects.
    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if collection not in existing:
        print(
            f"Collection {collection!r} not found in {path}. Available: {existing}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    col = client.get_collection(collection)
    total = col.count()
    print(f"Reading {total} chunk(s) from Chroma collection {collection!r}...")

    records: list[dict] = []
    offset = 0
    while offset < total:
        page = col.get(
            limit=BATCH,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = page["ids"]
        if not ids:
            break
        for doc, meta, emb in zip(
            page["documents"], page["metadatas"], page["embeddings"]
        ):
            props = {k: v for k, v in (meta or {}).items() if k in KNOWN_PROPS}
            props["text"] = doc or ""
            props.setdefault("source_file", "")
            props["chunk_index"] = int(props.get("chunk_index") or 0)
            # Weaviate is typed: text properties must be strings, not None.
            for key in (
                "source_path",
                "heading",
                "section_name",
                "subsection_name",
                "page_numbers",
            ):
                props[key] = str(props.get(key) or "")
            props["section_type"] = str(props.get("section_type") or "general")
            # Chroma hands back numpy float32 scalars; Weaviate needs plain floats.
            records.append({"properties": props, "vector": [float(x) for x in emb]})
        offset += len(ids)
        print(f"  read {offset}/{total}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--chroma-collection", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and summarise the Chroma index without writing to Weaviate",
    )
    args = parser.parse_args()

    records = read_chroma(args.chroma_path, args.chroma_collection)
    if not records:
        print("Nothing to migrate.")
        return

    sources = sorted({r["properties"]["source_file"] for r in records})
    dim = len(records[0]["vector"])
    print(f"\n{len(records)} chunk(s) across {len(sources)} paper(s), {dim}-dim vectors:")
    for s in sources:
        n = sum(1 for r in records if r["properties"]["source_file"] == s)
        print(f"  {n:5d}  {s}")

    if args.dry_run:
        print("\nDry run - nothing written.")
        return

    try:
        with VectorStore() as store:
            print("\nWriting to Weaviate...")
            for start in range(0, len(records), BATCH):
                store.insert_chunks(records[start : start + BATCH])
                print(f"  wrote {min(start + BATCH, len(records))}/{len(records)}")
            print(f"\nDone. Weaviate now holds {store.count_chunks()} chunk(s):")
            for s in store.list_sources():
                print(f"  - {s}  ({store.count_chunks(s)} chunks)")
    except WeaviateUnavailableError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
