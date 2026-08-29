"""
Score the pipeline against the golden question set in tests/golden_qa.json.

A RAG system can fail in three different places, and a single end-to-end number
hides which one broke. This script measures them separately:

  coverage    Is the fact even in the index? Facts were read from the source PDFs,
              so a miss here means ingestion dropped or mangled the content, and
              no amount of retrieval tuning would recover it.
  retrieval   Asked without naming a paper, does the right paper come back, and
              does the router target the right section type?
  generation  Given the right paper, does the answer actually state the fact?
              Scored against the paper the fact really lives in, so a low score
              is the LLM's summary omitting or contradicting retrieved content.

Usage:
    python scripts/eval_rag.py --coverage             # no LLM, instant
    python scripts/eval_rag.py --generation
    python scripts/eval_rag.py --retrieval
    python scripts/eval_rag.py --all -v
    python scripts/eval_rag.py --all --limit 8        # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "golden_qa.json"


def norm(text: str) -> str:
    """Lowercase, strip thousands separators, collapse whitespace, pad the ends."""
    t = (text or "").lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)   # 1,054 -> 1054
    t = re.sub(r"\s+", " ", t)
    return f" {t} "


def group_hit(group: list[str], haystack: str) -> bool:
    """A fact group is satisfied when any of its surface forms appears."""
    return any(norm(v).strip() in haystack for v in group)


def score_facts(groups: list[list[str]], text: str) -> tuple[int, list[int]]:
    hay = norm(text)
    hits = [1 if group_hit(g, hay) else 0 for g in groups]
    return sum(hits), hits


def load_items(limit: int | None) -> list[dict]:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = data["items"]
    return items[:limit] if limit else items


# ── coverage: are the facts in the index at all? ───────────────────────────

def run_coverage(items, verbose):
    from research_rag.vector_store import VectorStore

    print(f"\n{'=' * 74}\nCOVERAGE - are the golden facts present in the index?\n{'=' * 74}")
    with VectorStore() as store:
        cache: dict[str, str] = {}
        per_paper = defaultdict(lambda: [0, 0])
        total_ok = total_n = 0
        missing = []
        for it in items:
            paper = it["paper"]
            if paper not in cache:
                hits = store.get_by_section(source_filter=paper)
                cache[paper] = norm(" ".join(h.properties["text"] for h in hits))
            ok, flags = score_facts(it["key_facts"], cache[paper])
            n = len(it["key_facts"])
            total_ok += ok
            total_n += n
            per_paper[paper][0] += ok
            per_paper[paper][1] += n
            if ok < n:
                gone = [it["key_facts"][i][0] for i, f in enumerate(flags) if not f]
                missing.append((it["id"], gone))

    for paper, (ok, n) in sorted(per_paper.items()):
        flag = "  <-- gaps" if ok < n else ""
        print(f"  {ok:3d}/{n:3d}  {ok / n:5.0%}  {paper[:56]}{flag}")
    print(f"\n  TOTAL {total_ok}/{total_n} ({total_ok / total_n:.0%}) of golden facts are retrievable")
    if missing:
        print(f"\n  Facts NOT found in the index ({len(missing)} question(s)):")
        for qid, gone in missing:
            print(f"    {qid}: {gone}")
    return {"coverage": total_ok / total_n if total_n else 0.0}


# ── retrieval + generation, through the real API ───────────────────────────

def run_pipeline(items, verbose, do_retrieval, do_generation, deep=False):
    from fastapi.testclient import TestClient
    import app as appmod

    results = []
    with TestClient(appmod.app) as client:
        for i, it in enumerate(items, 1):
            row = {"id": it["id"], "paper": it["paper"]}
            print(f"  [{i}/{len(items)}] {it['id']}", flush=True)

            if do_retrieval:
                t0 = time.perf_counter()
                r = client.post("/api/chat", json={
                    "query": it["question"], "deep_scan": False,
                    "use_reranker": True, "top_k": 5,
                }).json()
                row["r_seconds"] = time.perf_counter() - t0
                papers = [p["source"] for p in r.get("papers", [])] if r.get("ok") else []
                row["found"] = it["paper"] in papers
                row["top1"] = bool(papers) and papers[0] == it["paper"]
                row["n_papers"] = len(papers)
                want = it.get("section_type")
                row["routed"] = (
                    None if not want else want in (r.get("target_sections") or [])
                )
                row["targets"] = r.get("target_sections")

            if do_generation:
                t0 = time.perf_counter()
                body = {
                    "query": it["question"],
                    "deep_scan": deep,
                    "source_filter": it["paper"],
                }
                if not deep:                      # deep scan ignores both
                    body |= {"use_reranker": True, "top_k": 5}
                r = client.post("/api/chat", json=body).json()
                row["g_seconds"] = time.perf_counter() - t0
                summary, sections = "", []
                if r.get("ok"):
                    for p in r.get("papers", []):
                        if p["source"] == it["paper"]:
                            summary, sections = p["summary"], p.get("sections_used", [])
                ok, flags = score_facts(it["key_facts"], summary)
                row["g_ok"], row["g_n"] = ok, len(it["key_facts"])
                row["sections_used"] = sections
                row["missed"] = [
                    it["key_facts"][j][0] for j, f in enumerate(flags) if not f
                ]
                row["summary"] = summary
            results.append(row)
    return results


def report(results, verbose, do_retrieval, do_generation, deep=False):
    out = {}
    if do_retrieval:
        n = len(results)
        found = sum(r["found"] for r in results)
        top1 = sum(r["top1"] for r in results)
        routed_items = [r for r in results if r["routed"] is not None]
        routed = sum(r["routed"] for r in routed_items)
        print(f"\n{'=' * 74}\nRETRIEVAL - question asked without naming a paper\n{'=' * 74}")
        print(f"  correct paper returned : {found}/{n} ({found / n:.0%})")
        print(f"  correct paper ranked 1 : {top1}/{n} ({top1 / n:.0%})")
        if routed_items:
            print(f"  section routed correctly: {routed}/{len(routed_items)} ({routed / len(routed_items):.0%})")
        out["found"] = found / n
        out["top1"] = top1 / n
        misses = [r for r in results if not r["found"]]
        if misses:
            print(f"\n  Target paper NOT retrieved ({len(misses)}):")
            for r in misses:
                print(f"    {r['id']:22s} targets={r['targets']} returned {r['n_papers']} paper(s)")

    if do_generation:
        tot_ok = sum(r["g_ok"] for r in results)
        tot_n = sum(r["g_n"] for r in results)
        full = sum(1 for r in results if r["g_ok"] == r["g_n"])
        zero = sum(1 for r in results if r["g_ok"] == 0)
        mode = "deep scan" if deep else "quick search"
        secs = sum(r.get("g_seconds", 0.0) for r in results)
        print(f"\n{'=' * 74}\nGENERATION ({mode}) - answer quality with the correct paper in scope\n{'=' * 74}")
        print(f"  wall clock             : {secs:.0f}s total, {secs / len(results):.1f}s per question")
        print(f"  facts stated in answer : {tot_ok}/{tot_n} ({tot_ok / tot_n:.0%})")
        print(f"  fully correct answers  : {full}/{len(results)} ({full / len(results):.0%})")
        print(f"  answers with no facts  : {zero}/{len(results)}")
        out["fact_recall"] = tot_ok / tot_n
        out["full"] = full / len(results)
        weak = sorted(
            (r for r in results if r["g_ok"] < r["g_n"]),
            key=lambda r: r["g_ok"] / max(r["g_n"], 1),
        )
        if weak:
            print(f"\n  Incomplete answers ({len(weak)}):")
            for r in weak:
                print(f"    {r['id']:22s} {r['g_ok']}/{r['g_n']}  missed={r['missed']}")
                if verbose:
                    print(f"      sections={r['sections_used']}")
                    print(f"      answer  ={r['summary'][:220]!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deep", action="store_true",
                    help="Answer with deep scan (map-reduce) instead of quick search")
    ap.add_argument("-o", "--out", default="eval_results.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.all:
        a.coverage = a.retrieval = a.generation = True
    if not (a.coverage or a.retrieval or a.generation):
        ap.error("choose at least one of --coverage / --retrieval / --generation / --all")

    items = load_items(a.limit)
    print(f"Golden set: {len(items)} question(s) over "
          f"{len({i['paper'] for i in items})} paper(s), "
          f"{sum(len(i['key_facts']) for i in items)} fact group(s)")

    if a.coverage:
        run_coverage(items, a.verbose)
    if a.retrieval or a.generation:
        results = run_pipeline(items, a.verbose, a.retrieval, a.generation, deep=a.deep)
        report(results, a.verbose, a.retrieval, a.generation, deep=a.deep)
        (ROOT / a.out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nPer-question detail written to eval_results.json")


if __name__ == "__main__":
    main()
