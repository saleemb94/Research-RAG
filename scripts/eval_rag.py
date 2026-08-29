"""
Score the pipeline against the golden question set in tests/golden_qa.json.

A RAG system can fail in three different places, and a single end-to-end number
hides which one broke. This script measures them separately:

  coverage    Is the fact even in the index? Facts were read from the source PDFs,
              so a miss here means ingestion dropped or mangled the content, and
              no amount of retrieval tuning would recover it.
  retrieval   A single-paper fact asked of the whole corpus: does the right
              paper come back? This is needle retrieval, not the corpus-wide
              Ask tab, which is scored by --synthesis.
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
import statistics
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


def load_items(limit: int | None, mode: str = "single") -> list[dict]:
    """`mode` is "single" (a fact inside one paper) or "synthesis" (corpus-wide)."""
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = [i for i in data["items"] if (i.get("mode") or "single") == mode]
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


# Phrases an honest answer uses when the corpus does not hold the answer. This is
# a blunt check, but the failure it looks for is not subtle: a fabricated answer
# states findings, names models and quotes numbers, and contains none of these.
_ABSTAIN = (
    "do not mention", "does not mention", "not mentioned", "no mention",
    "do not discuss", "does not discuss", "not discussed",
    "do not provide", "does not provide", "not provided",
    "do not contain", "does not contain", "not contain",
    "do not report", "does not report", "not report",
    "do not address", "does not address", "not addressed",
    "do not specify", "does not specify", "not specified",
    "no information", "not available", "none of the", "no papers",
    "not covered", "cannot be answered", "no evidence", "not present",
    "do not include", "does not include", "no such", "not applicable",
    "sources do not", "no study", "no studies", "not apply",
)


def abstained(answer: str) -> bool:
    # Import the phrase the pipeline actually emits rather than guessing at it.
    # Hand-maintaining this list has now mis-scored a correct refusal as a
    # fabrication three times, which sends you hunting a hallucination that was
    # never there.
    try:
        from research_rag.enumerate_ import NOTHING_FOUND
    except ImportError:
        NOTHING_FOUND = ""

    a = (answer or "").strip().lower()
    if NOTHING_FOUND and NOTHING_FOUND.lower() in a:
        return True
    # A bare sentinel is an abstention too, and scoring it as a fabrication sent
    # me looking for a hallucination that was not there.
    if a in ("none", "n/a", "nothing", ""):
        return True
    return any(m in a for m in _ABSTAIN)


# ── negative controls: does it invent an answer that is not there? ─────────

def run_negative(items, verbose):
    """
    Every other metric measures denying content that is present. This measures
    the opposite and, for a research tool, the more dangerous failure: stating
    findings for something the corpus does not contain. Retrieval always returns
    something, so nothing upstream prevents it.
    """
    from fastapi.testclient import TestClient
    import app as appmod

    rows = []
    with TestClient(appmod.app) as client:
        for i, it in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] {it['id']}", flush=True)
            r = client.post("/api/ask", json={"query": it["question"]}).json()
            answer = r.get("answer", "") if r.get("ok") else ""
            rows.append({"id": it["id"], "why": it.get("why", ""),
                         "abstained": abstained(answer), "answer": answer})

    ok = sum(r["abstained"] for r in rows)
    bar = "=" * 74
    print(f"\n{bar}\nNEGATIVE CONTROLS - the corpus cannot answer these\n{bar}")
    print(f"  correctly declined     : {ok}/{len(rows)} ({ok / len(rows):.0%})")
    for r in rows:
        print(f"  {'ok ' if r['abstained'] else 'INVENTED'} {r['id']:18s} {r['why']}")
        if not r["abstained"] or verbose:
            print(f"        {r['answer'][:230]!r}")
    return rows


# ── synthesis: one cited answer over the whole corpus ──────────────────────

def run_synthesis(items, verbose, client=None):
    """
    Score the corpus-wide tab: does one answer state the facts, and does it
    actually draw on the papers that hold them?

    Paper coverage is scored separately from facts because they fail apart. An
    answer can name four datasets while citing one paper, which reads as
    authoritative and silently under-reports the corpus.
    """
    from fastapi.testclient import TestClient
    import app as appmod

    # The client must be shared across repeats. Leaving the TestClient context
    # runs the app's lifespan shutdown, which closes every Weaviate client, so a
    # second run against a fresh context queries closed connections and scores
    # zero - which looks like a catastrophic regression rather than a harness bug.
    if client is None:
        with TestClient(appmod.app) as c:
            return run_synthesis(items, verbose, client=c)

    rows = []
    if True:
        for i, it in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] {it['id']}", flush=True)
            t0 = time.perf_counter()
            r = client.post("/api/ask", json={"query": it["question"]}).json()
            secs = time.perf_counter() - t0

            answer = r.get("answer", "") if r.get("ok") else ""
            ok, flags = score_facts(it["key_facts"], answer)
            cited_nums = set(r.get("cited") or [])
            cited_papers = {
                s["source"] for s in (r.get("sources") or [])
                if s["number"] in cited_nums
            }
            want = set(it.get("expected_papers") or [])
            rows.append({
                "id": it["id"], "seconds": secs,
                "g_ok": ok, "g_n": len(it["key_facts"]),
                "papers_hit": len(want & cited_papers), "papers_want": len(want),
                "cited_papers": sorted(cited_papers),
                "dropped": r.get("dropped_citations") or [],
                "missed": [it["key_facts"][j][0] for j, f in enumerate(flags) if not f],
                "answer": answer,
            })

    f_ok = sum(r["g_ok"] for r in rows); f_n = sum(r["g_n"] for r in rows)
    p_ok = sum(r["papers_hit"] for r in rows); p_n = sum(r["papers_want"] for r in rows)
    bad = sum(len(r["dropped"]) for r in rows)
    secs = sum(r["seconds"] for r in rows)
    bar = "=" * 74
    print(f"\n{bar}\nSYNTHESIS - one cited answer across the whole corpus\n{bar}")
    print(f"  facts stated in answer  : {f_ok}/{f_n} ({f_ok / f_n:.0%})")
    print(f"  expected papers cited   : {p_ok}/{p_n} ({p_ok / p_n:.0%})")
    print(f"  invalid citations emitted: {bad}")
    print(f"  wall clock              : {secs:.0f}s total, {secs / len(rows):.1f}s per question")
    for r in rows:
        mark = "ok " if r["g_ok"] == r["g_n"] and r["papers_hit"] == r["papers_want"] else "   "
        print(f"  {mark} {r['id']:20s} facts {r['g_ok']}/{r['g_n']}  "
              f"papers {r['papers_hit']}/{r['papers_want']}  missed={r['missed']}")
        if verbose:
            print(f"        cited: {[c[:34] for c in r['cited_papers']]}")
            print(f"        {r['answer'][:220]!r}")
    (ROOT / "eval_synthesis.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


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
    ap.add_argument("--synthesis", action="store_true",
                    help="Score the corpus-wide cited-answer questions")
    ap.add_argument("--negative", action="store_true",
                    help="Score the questions the corpus cannot answer")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Run N times and report mean and spread. Answers vary "
                         "between identical runs by several points, so a single "
                         "run cannot settle a small difference.")
    ap.add_argument("--deep", action="store_true",
                    help="Answer with deep scan (map-reduce) instead of quick search")
    ap.add_argument("-o", "--out", default="eval_results.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.all:
        a.coverage = a.retrieval = a.generation = a.synthesis = a.negative = True
    if not (a.coverage or a.retrieval or a.generation or a.synthesis or a.negative):
        ap.error("choose at least one of --coverage / --retrieval / "
                 "--generation / --synthesis / --all")

    items = load_items(a.limit, "single")
    syn_items = load_items(a.limit, "synthesis")
    neg_items = load_items(a.limit, "negative")
    print(f"Golden set: {len(items)} single-paper question(s) over "
          f"{len({i['paper'] for i in items})} paper(s) "
          f"+ {len(syn_items)} synthesis + {len(neg_items)} negative control(s); "
          f"{sum(len(i['key_facts']) for i in items + syn_items)} fact group(s)")

    if a.coverage:
        run_coverage(items, a.verbose)
    if a.synthesis and syn_items:
        from fastapi.testclient import TestClient
        import app as appmod
        with TestClient(appmod.app) as _client:
            runs = [run_synthesis(syn_items, a.verbose, client=_client)
                    for _ in range(max(1, a.repeat))]
        if len(runs) > 1:
            facts = [sum(r["g_ok"] for r in run) for run in runs]
            papers = [sum(r["papers_hit"] for r in run) for run in runs]
            f_n = sum(r["g_n"] for r in runs[0])
            p_n = sum(r["papers_want"] for r in runs[0])
            bar = "=" * 74
            print(f"\n{bar}\nSYNTHESIS over {len(runs)} runs\n{bar}")
            print(f"  facts  : {statistics.mean(facts):.1f}/{f_n} "
                  f"({statistics.mean(facts) / f_n:.0%})  "
                  f"range {min(facts)}-{max(facts)}  "
                  f"sd {statistics.pstdev(facts):.1f}")
            print(f"  papers : {statistics.mean(papers):.1f}/{p_n} "
                  f"({statistics.mean(papers) / p_n:.0%})  "
                  f"range {min(papers)}-{max(papers)}  "
                  f"sd {statistics.pstdev(papers):.1f}")
            # Per-question stability: a question that swings is not evidence
            # either way, and knowing which ones swing matters as much as the mean.
            print("\n  per-question spread (facts):")
            for i, it in enumerate(syn_items):
                vals = [run[i]["g_ok"] for run in runs]
                flag = "  <-- unstable" if max(vals) != min(vals) else ""
                print(f"    {it['id']:22s} {vals}  /{runs[0][i]['g_n']}{flag}")
    if a.negative and neg_items:
        run_negative(neg_items, a.verbose)
    if a.retrieval or a.generation:
        results = run_pipeline(items, a.verbose, a.retrieval, a.generation, deep=a.deep)
        report(results, a.verbose, a.retrieval, a.generation, deep=a.deep)
        (ROOT / a.out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nPer-question detail written to eval_results.json")


if __name__ == "__main__":
    main()
