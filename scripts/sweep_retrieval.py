"""
Sweep retrieval settings against the golden set, without the LLM.

eval_rag.py measures the whole pipeline and takes ten minutes a run, which is
too slow to tune a single number with. This scores only what a retrieval knob
can change - is the right paper in the results, and is it first - by running
the three channels directly and reranking. Seconds per setting rather than
minutes, so a sweep is cheap enough to actually do.

    python scripts/sweep_retrieval.py --alpha 0 0.15 0.3 0.5 0.75 1
    python scripts/sweep_retrieval.py --prefilter 0 10 20 30

`--alpha` is the hybrid weight: 0 is pure BM25, 1 is pure vector.

The numbers here are not comparable to eval_rag's: this skips question
routing and answer generation, so it is an A/B instrument, not a report card.
Use it to choose a setting, then confirm with the real evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_rag import Embedder, Reranker, VectorStore  # noqa: E402
from research_rag.config import RERANK_FETCH_MULTIPLIER, TOP_K  # noqa: E402
from research_rag.prefilter import shortlist_papers  # noqa: E402
from research_rag.section_index import open_section_index  # noqa: E402

GOLDEN = ROOT / "tests" / "golden_qa.json"


def load_questions():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return [i for i in data["items"] if (i.get("mode") or "single") == "single"]


def retrieve(query, vec, store, section_index, reranker, alpha, prefilter_n,
             max_sources):
    fetch_k = max_sources * RERANK_FETCH_MULTIPLIER
    scope = None
    if prefilter_n:
        picked = shortlist_papers(vec, section_index, prefilter_n)
        if picked:
            scope = picked

    hits = list(store.search(vec, limit=fetch_k, source_filter=scope))
    seen = {(h.properties.get("source_file"), h.properties.get("chunk_index"))
            for h in hits}
    if alpha < 1.0:
        for h in store.hybrid_search(query, vec, limit=fetch_k, alpha=alpha,
                                     source_filter=scope):
            key = (h.properties.get("source_file"), h.properties.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                hits.append(h)
    if reranker and hits:
        hits = reranker.rerank(query, hits, top_n=max_sources)
    return hits[:max_sources]


def score(questions, store, section_index, embedder, reranker, alpha,
          prefilter_n, max_sources):
    found = first = 0
    t0 = time.perf_counter()
    for it in questions:
        vec = embedder.embed_one(it["question"])
        hits = retrieve(it["question"], vec, store, section_index, reranker,
                        alpha, prefilter_n, max_sources)
        papers = [h.properties.get("source_file") for h in hits]
        if it["paper"] in papers:
            found += 1
        if papers and papers[0] == it["paper"]:
            first += 1
    return found, first, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, nargs="*", default=None)
    ap.add_argument("--prefilter", type=int, nargs="*", default=None)
    ap.add_argument("--max-sources", type=int, default=10)
    a = ap.parse_args()

    alphas = a.alpha if a.alpha is not None else [0.3]
    prefilters = a.prefilter if a.prefilter is not None else [0]

    questions = load_questions()
    embedder, reranker = Embedder(), Reranker()
    n = len(questions)

    print(f"\n{n} questions, top-{a.max_sources} after reranking\n")
    print(f"  {'alpha':>6} {'prefilter':>10} {'right paper':>13} "
          f"{'ranked first':>14} {'seconds':>9}")
    with VectorStore() as store, open_section_index() as section_index:
        for pf in prefilters:
            for al in alphas:
                found, first, secs = score(questions, store, section_index,
                                           embedder, reranker, al, pf,
                                           a.max_sources)
                print(f"  {al:>6.2f} {pf if pf else '-':>10} "
                      f"{found}/{n} ({found/n:>3.0%}) "
                      f"{first}/{n} ({first/n:>3.0%}) {secs:>8.1f}s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
