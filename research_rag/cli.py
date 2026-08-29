import argparse
import os
import sys

from .config import (
    OLLAMA_MODEL,
    TOP_K,
    WEAVIATE_COLLECTION,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
)
from .embedder import Embedder
from .ingest import ingest_folder, ingest_pdf
from .search import search_and_summarize
from .vector_store import VectorStore, WeaviateUnavailableError


def cmd_ingest(args, embedder, store):
    from .paper_cards import PaperCardStore
    from .section_index import build_entries, index_entries, open_section_index

    with PaperCardStore() as cards:
        if os.path.isdir(args.path):
            ingest_folder(args.path, embedder, store,
                          skip_existing=not args.force, card_store=cards)
        else:
            ingest_pdf(args.path, embedder, store,
                       skip_existing=not args.force, card_store=cards)

    # Refresh the section-summary index. It is derived from the chunks that were
    # just written, and without this step a fresh install has an empty second
    # retrieval channel and no sign that anything is missing. Rebuilding all of
    # it takes about a second, so there is no reason to do it incrementally.
    with open_section_index() as index:
        entries = build_entries(store)
        for paper in {e["source_file"] for e in entries}:
            index.delete_source(paper)
        n = index_entries(index, entries, embedder)
    print(f"  -> Section index: {n} summar(ies)")


def cmd_search(args, embedder, store):
    result = search_and_summarize(
        args.query, embedder, store, top_k=args.top_k, model=args.model
    )
    print(f"\nQuery : {result['query']}")
    print(f"Papers: {result['papers_found']}\n")
    for paper in result["results"]:
        print("=" * 70)
        print(f"Paper      : {paper['source']}")
        print(f"Chunks used: {paper['chunks_used']}")
        print(f"\nSummary:\n{paper['summary']}\n")


def cmd_list(args, embedder, store):
    sources = store.list_sources()
    if not sources:
        print("No papers ingested yet.")
        return
    print(f"Ingested papers ({len(sources)}):")
    for s in sources:
        print(f"  - {s}  ({store.count_chunks(s)} chunks)")


def cmd_delete(args, embedder, store):
    store.delete_source(args.name)
    print(f"Deleted all chunks for: {args.name}")


def cmd_sections(args, embedder, store):
    summary = store.get_section_summary(args.paper)
    if not summary:
        print("No sections found.")
        return
    type_map = store.get_section_type_map(args.paper)
    for section, subs in summary.items():
        print(f"  {section}  [{type_map.get(section, 'general')}]")
        for sub in subs:
            print(f"      - {sub}")


def main():
    parser = argparse.ArgumentParser(
        prog="research-rag",
        description="Offline research-paper RAG - Docling, Weaviate & Ollama",
    )
    parser.add_argument("--host", default=WEAVIATE_HOST, help="Weaviate host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=WEAVIATE_PORT, help="Weaviate REST port (default: %(default)s)")
    parser.add_argument("--grpc-port", type=int, default=WEAVIATE_GRPC_PORT, dest="grpc_port", help="Weaviate gRPC port (default: %(default)s)")
    parser.add_argument("--collection", default=WEAVIATE_COLLECTION, help="Collection name (default: %(default)s)")

    subs = parser.add_subparsers(dest="command", required=True)

    p_ingest = subs.add_parser("ingest", help="Ingest a PDF file or folder of PDFs")
    p_ingest.add_argument("path", help="Path to a PDF file or a folder")
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if already stored")

    p_search = subs.add_parser("search", help="Search and summarize")
    p_search.add_argument("query", help="Natural-language question")
    p_search.add_argument("--top-k", type=int, default=TOP_K, dest="top_k")
    p_search.add_argument("--model", default=OLLAMA_MODEL)

    subs.add_parser("list", help="List ingested papers")

    p_sections = subs.add_parser("sections", help="Show the section tree of the index")
    p_sections.add_argument("--paper", default=None, help="Limit to one paper (filename as stored)")

    p_delete = subs.add_parser("delete", help="Remove a paper from the store")
    p_delete.add_argument("name", help="Filename as stored (e.g. paper.pdf)")

    args = parser.parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "search": cmd_search,
        "list": cmd_list,
        "sections": cmd_sections,
        "delete": cmd_delete,
    }
    # `list`, `sections` and `delete` never embed anything, so skip loading the
    # sentence-transformer model for them (saves several seconds per call).
    needs_embedder = args.command in ("ingest", "search")

    try:
        with VectorStore(
            host=args.host,
            port=args.port,
            grpc_port=args.grpc_port,
            collection=args.collection,
        ) as store:
            embedder = Embedder() if needs_embedder else None
            handlers[args.command](args, embedder, store)
    except WeaviateUnavailableError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
