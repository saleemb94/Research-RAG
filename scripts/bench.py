"""
Benchmark the pipeline stage by stage.

The golden-set scripts answer "is the answer right". They say nothing about
where the time goes, or whether parsing produced a usable document in the first
place - and a RAG system can be perfectly accurate and unusable, or fast and
structurally broken. This measures the parts those scripts leave out.

    python scripts/bench.py --structure    # what ingestion produced (instant)
    python scripts/bench.py --retrieval    # latency per channel, p50/p95
    python scripts/bench.py --generation   # answer latency and output rate
    python scripts/bench.py --ingest 3     # parse/classify/embed/write split
    python scripts/bench.py --all

`--structure` reads the live index and costs nothing, so it is the one to run
after touching ingestion. `--ingest` writes to a throwaway collection and
deletes it afterwards, so it never disturbs the library.

What each number is for
-----------------------
structure   Parsing quality, which nothing else checks. A paper whose headings
            did not survive is invisible to section filtering, and shows up
            here as chunks typed `general` - not as a wrong answer, just a
            quietly worse one.
retrieval   Where the latency budget goes before the model is even called, and
            what each of the three channels contributes.
generation  Whether the model is the bottleneck, and by how much.
ingest      Which stage owns the minutes. "Ingestion is slow" is not
            actionable; "classification is 70% of it" is.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_rag import Embedder, Reranker, VectorStore  # noqa: E402
from research_rag.config import OLLAMA_MODEL, PAPERS_DIR, TOP_K  # noqa: E402
from research_rag.vector_store import WeaviateUnavailableError  # noqa: E402

BAR = "=" * 74

# Deliberately ordinary questions: the point is timing under normal use, not
# best-case latency on a question that happens to hit one channel.
QUERIES = [
    "What datasets were used?",
    "Which models were fine-tuned?",
    "What evaluation metrics are reported?",
    "How were the annotations collected?",
    "What limitations do the authors acknowledge?",
    "Which languages are covered?",
    "What preprocessing was applied?",
    "How was the model trained?",
]


def pct(values: list[float], p: float) -> float:
    """Percentile without numpy, on a small sample."""
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[i]


def summarise(name: str, samples: list[float], unit: str = "ms"):
    if not samples:
        print(f"  {name:26} no samples")
        return
    scale = 1000 if unit == "ms" else 1
    vals = [v * scale for v in samples]
    print(f"  {name:26} p50 {pct(vals, 50):7.1f}{unit}   "
          f"p95 {pct(vals, 95):7.1f}{unit}   "
          f"mean {statistics.mean(vals):7.1f}{unit}   n={len(vals)}")


# ── structure: what ingestion actually produced ───────────────────────────

def bench_structure():
    """
    Parsing and classification quality, read straight off the index.

    Nothing else in the repo measures this. The golden set notices when a fact
    is missing, but not when a paper's structure was lost - which degrades
    section filtering into plain semantic search without ever failing.
    """
    print(f"\n{BAR}\nSTRUCTURE - what ingestion produced\n{BAR}")
    with VectorStore() as store:
        rows = store._scan(None, ["source_file", "section_name", "section_type",
                                  "page_numbers", "text"])
    if not rows:
        print("  index is empty")
        return

    by_paper: dict[str, list[dict]] = {}
    for r in rows:
        by_paper.setdefault(r.get("source_file") or "?", []).append(r)

    types = Counter(r.get("section_type") or "?" for r in rows)
    lengths = [len(r.get("text") or "") for r in rows]
    n = len(rows)

    print(f"  papers                     : {len(by_paper)}")
    print(f"  chunks                     : {n}")
    print(f"  chunks per paper           : "
          f"median {statistics.median(len(v) for v in by_paper.values()):.0f}   "
          f"min {min(len(v) for v in by_paper.values())}   "
          f"max {max(len(v) for v in by_paper.values())}")
    print(f"  chunk length (chars)       : "
          f"median {statistics.median(lengths):.0f}   "
          f"p95 {pct([float(x) for x in lengths], 95):.0f}   "
          f"max {max(lengths)}")

    # The headline parsing metric: a chunk typed `general` is one whose heading
    # told the classifier nothing, so section filtering cannot reach it.
    typed = n - types.get("general", 0)
    print(f"\n  typed to a real section    : {typed}/{n} ({typed / n:.0%})"
          "        <- section filtering can use these")
    print(f"  fell back to `general`     : {types.get('general', 0)}/{n} "
          f"({types.get('general', 0) / n:.0%})")
    print("\n  section_type distribution:")
    for t, c in types.most_common():
        print(f"    {t:14} {c:6}  {c / n:5.1%}")

    # Structural completeness per paper, which is where a bad parse hides.
    no_abstract = [p for p, v in by_paper.items()
                   if not any((r.get("section_type") or "") == "abstract" for r in v)]
    no_pages = [p for p, v in by_paper.items()
                if not any(r.get("page_numbers") for r in v)]
    mostly_general = [p for p, v in by_paper.items()
                      if sum(1 for r in v
                             if (r.get("section_type") or "") == "general") > len(v) * 0.6]
    print(f"\n  papers with no abstract    : {len(no_abstract)}")
    print(f"  papers with no page numbers: {len(no_pages)}"
          "        <- citations cannot be rendered for these")
    print(f"  papers >60% `general`      : {len(mostly_general)}"
          "        <- parsed, but structurally flat")
    for p in mostly_general[:6]:
        share = sum(1 for r in by_paper[p]
                    if (r.get("section_type") or "") == "general") / len(by_paper[p])
        print(f"      {share:4.0%}  {p[:58]}")


# ── retrieval: where the latency goes before the model is called ──────────

def bench_retrieval(repeat: int):
    print(f"\n{BAR}\nRETRIEVAL - latency per channel\n{BAR}")
    embedder = Embedder()
    reranker = Reranker()
    fetch = TOP_K * 4

    embed_t, vector_t, hybrid_t, rerank_t = [], [], [], []
    with VectorStore() as store:
        for _ in range(repeat):
            for q in QUERIES:
                t = time.perf_counter(); vec = embedder.embed_one(q); embed_t.append(time.perf_counter() - t)

                t = time.perf_counter(); hits = store.search(vec, limit=fetch); vector_t.append(time.perf_counter() - t)

                t = time.perf_counter(); store.hybrid_search(q, vec, limit=fetch, alpha=0.3); hybrid_t.append(time.perf_counter() - t)

                if hits:
                    t = time.perf_counter(); reranker.rerank(q, hits, top_n=TOP_K); rerank_t.append(time.perf_counter() - t)

    summarise("embed the query", embed_t)
    summarise("chunk vector search", vector_t)
    summarise("BM25 hybrid search", hybrid_t)
    summarise("cross-encoder rerank", rerank_t)
    total = [sum(x) for x in zip(embed_t, vector_t, hybrid_t,
                                 rerank_t or [0.0] * len(embed_t))]
    summarise("→ retrieval, end to end", total)
    print("\n  Reranking is usually the largest share, and it is the one that")
    print("  scales with how many candidates are fetched rather than with")
    print("  corpus size - which is the knob worth turning first.")


# ── generation: is the model the bottleneck ───────────────────────────────

def bench_generation(repeat: int):
    print(f"\n{BAR}\nGENERATION - answer latency and output rate\n{BAR}")
    from fastapi.testclient import TestClient
    import app as appmod

    lat, chars, sources = [], [], []
    with TestClient(appmod.app) as client:
        for _ in range(repeat):
            for q in QUERIES[:4]:
                t = time.perf_counter()
                r = client.post("/api/ask", json={"query": q}).json()
                dt = time.perf_counter() - t
                if r.get("ok"):
                    lat.append(dt)
                    chars.append(len(r.get("answer") or ""))
                    sources.append(len(r.get("sources") or []))

    summarise("whole /api/ask call", lat, unit="s")
    if lat:
        rate = [c / t for c, t in zip(chars, lat)]
        print(f"  answer length              : median {statistics.median(chars):.0f} chars")
        print(f"  output rate                : median {statistics.median(rate):.0f} chars/s")
        print(f"  sources returned           : median {statistics.median(sources):.0f}")
        print(f"\n  Model: {OLLAMA_MODEL}. Retrieval above is milliseconds, so")
        print("  effectively all of this is generation - which is why the model")
        print("  choice, not the index, sets how fast the product feels.")


# ── ingest: which stage owns the minutes ──────────────────────────────────

BENCH_COLLECTION = "BenchThrowaway"


def bench_ingest(count: int):
    """
    Ingest a few papers into a throwaway collection, timing each stage.

    A separate collection because this writes: benchmarking must not mutate the
    library, and re-ingesting real papers would rewrite their chunks.
    """
    print(f"\n{BAR}\nINGEST - where the time goes\n{BAR}")
    from research_rag.ingest import ingest_pdf

    pdfs = sorted(Path(PAPERS_DIR).glob("*.pdf"))[:count]
    if not pdfs:
        print(f"  no PDFs in {PAPERS_DIR}")
        return

    embedder = Embedder()
    rows = []
    store = VectorStore(collection=BENCH_COLLECTION)
    try:
        for pdf in pdfs:
            timings: dict = {}
            t0 = time.perf_counter()
            # No card store: cards are measured separately and would double the
            # LLM cost of a timing run.
            n = ingest_pdf(str(pdf), embedder, store, skip_existing=False,
                           timings=timings)
            total = time.perf_counter() - t0
            pages = 0
            try:
                import pymupdf
                with pymupdf.open(str(pdf)) as d:
                    pages = d.page_count
            except Exception:
                pass
            rows.append({"paper": pdf.name, "chunks": n, "pages": pages,
                         "total": total, **timings})
    finally:
        # Drop the whole collection rather than deleting per source: it exists
        # only for this run.
        try:
            store._client.collections.delete(BENCH_COLLECTION)
        except Exception:
            pass
        store.close()

    stages = ["parse", "classify", "embed", "summarise", "write"]
    print(f"\n  {'paper':30} {'pp':>3} {'chunks':>6} {'total':>7}  " +
          "  ".join(f"{s:>9}" for s in stages))
    for r in rows:
        print(f"  {r['paper'][:30]:30} {r['pages']:>3} {r['chunks']:>6} "
              f"{r['total']:>6.1f}s  " +
              "  ".join(f"{r.get(s, 0):>8.1f}s" for s in stages))

    print()
    tot = sum(r["total"] for r in rows)
    for s in stages:
        v = sum(r.get(s, 0) for r in rows)
        print(f"  {s:12} {v:7.1f}s   {v / tot:5.1%} of ingestion")
    other = tot - sum(sum(r.get(s, 0) for r in rows) for s in stages)
    print(f"  {'other':12} {other:7.1f}s   {other / tot:5.1%}")
    pages = sum(r["pages"] for r in rows) or 1
    print(f"\n  {tot:.1f}s for {len(rows)} paper(s), {pages} page(s)"
          f"  ->  {tot / len(rows):.1f}s per paper, {tot / pages:.2f}s per page")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--ingest", type=int, metavar="N", default=0,
                    help="Ingest N papers into a throwaway collection and time it")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    if a.all:
        a.structure = a.retrieval = a.generation = True
        a.ingest = a.ingest or 2
    if not (a.structure or a.retrieval or a.generation or a.ingest):
        ap.error("choose at least one of --structure / --retrieval / "
                 "--generation / --ingest N / --all")

    try:
        if a.structure:
            bench_structure()
        if a.retrieval:
            bench_retrieval(a.repeat)
        if a.generation:
            bench_generation(a.repeat)
        if a.ingest:
            bench_ingest(a.ingest)
    except WeaviateUnavailableError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
