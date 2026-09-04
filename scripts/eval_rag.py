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
  passages    Retrieval scored at passage granularity against hand-labeled gold
              passages: Recall@k, MRR@k, MAP@k and nDCG@k. The paper-level
              numbers cannot separate returning the paragraph that states the
              result from returning the same paper's related work.
  thematic    A discussion question over a topic rather than a paper, which is
              what a reader with a reading pile actually asks. Scored on facts,
              on how many of the topic's papers the answer draws on, and on
              whether its citations are distinct pieces of evidence.

Usage:
    python scripts/eval_rag.py --coverage             # no LLM, instant
    python scripts/eval_rag.py --generation
    python scripts/eval_rag.py --retrieval
    python scripts/eval_rag.py --passages        # Recall/MRR/MAP/nDCG @k
    python scripts/eval_rag.py --thematic -v
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
    """Lowercase, repair numbers, collapse whitespace, pad the ends."""
    t = (text or "").lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)     # 1,054 -> 1054
    # Layout parsing sometimes splits a decimal point: "29 . 6%". Scoring a
    # fact as absent for that reason measures the PDF extractor, not the
    # system, so both sides are normalized before comparison.
    t = re.sub(r"(?<=\d)\s+\.\s+(?=\d)", ".", t)
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
    """"single" (a fact in one paper), "synthesis", "thematic" or "negative"."""
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = [i for i in data["items"] if (i.get("mode") or "single") == mode]
    return items[:limit] if limit else items


def prune_expected_papers(syn_items, verbose=False) -> int:
    """
    Drop expected papers that are no longer indexed.

    The "expected papers cited" metric asks whether the answer cited the papers
    that ought to have contributed. Leaving a removed paper in that list makes
    the metric unwinnable and reads as a citation failure, when in fact the
    answer could not have cited a document that is not there. Facts are not
    attributed per paper, so a fact whose only source was removed still counts
    against the facts metric - which is why that number needs reading with the
    absent list in hand.
    """
    from research_rag.vector_store import VectorStore

    with VectorStore() as store:
        present = set(store.list_sources())
    dropped = 0
    for it in syn_items:
        want = it.get("expected_papers") or []
        keep = [p for p in want if p in present]
        if len(keep) != len(want):
            dropped += len(want) - len(keep)
            if verbose:
                for p in want:
                    if p not in present:
                        print(f"    {it['id']}: dropped absent {p}")
            it["expected_papers"] = keep
    return dropped


def split_by_presence(items) -> tuple[list[dict], list[dict], set[str]]:
    """
    Separate questions whose paper is still in the index from those whose is not.

    A library is curated over time, and scoring a question whose source document
    has been removed measures nothing: it reports 0% and reads as a collapse in
    retrieval quality when the only thing that changed is which papers are on
    the shelf. Reported as absent rather than silently averaged in.
    """
    from research_rag.vector_store import VectorStore

    with VectorStore() as store:
        present = set(store.list_sources())
    here = [i for i in items if not i.get("paper") or i["paper"] in present]
    gone = [i for i in items if i.get("paper") and i["paper"] not in present]
    return here, gone, {i["paper"] for i in gone}


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


# ── thematic: a discussion question spanning a topic, not one paper ────────

def run_thematic(items, verbose, client=None):
    """
    Score the questions a reader actually asks of a reading pile: "discuss the
    limitations of RAG", not "what F1 did paper X report".

    These are scored differently from --synthesis on purpose. A synthesis
    question has a closed answer, so every expected paper must appear. A
    discussion question does not: the corpus holds eight papers on hate speech
    and a good four-paragraph answer draws on some of them, not all eight.
    Demanding the full set would score a genuinely good answer as a failure.

    So breadth is judged against `min_papers`, and three things are measured
    that the synthesis scorer does not:

      on-topic     citations that land outside the theme entirely. An answer
                   about RAG limitations citing a hate-speech paper is padding
                   its evidence, which reads as authoritative and is not.

                   Topic boundaries in a real corpus are fuzzy, so each theme
                   carries a second list, `related_papers`: papers that satisfy
                   most of the theme's fact groups in their own text without
                   being primarily about it, which is mostly the surveys.
                   Citing one is neither a hit nor a miss. Without that split
                   the metric scores the boundary I drew rather than the
                   system, and a survey citation looks like a mistake.
      distinct     distinct passages behind the citations. Eight markers all
                   pointing at one paragraph looks like eight sources in the
                   interface and is one.
      breadth      questions meeting their min_papers bar.
    """
    from fastapi.testclient import TestClient
    import app as appmod

    if client is None:
        with TestClient(appmod.app) as c:
            return run_thematic(items, verbose, client=c)

    rows = []
    for i, it in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {it['id']}", flush=True)
        t0 = time.perf_counter()
        r = client.post("/api/ask", json={"query": it["question"]}).json()
        secs = time.perf_counter() - t0

        answer = r.get("answer", "") if r.get("ok") else ""
        ok, flags = score_facts(it["key_facts"], answer)

        cited_nums = set(r.get("cited") or [])
        cited = [s for s in (r.get("sources") or []) if s["number"] in cited_nums]
        cited_papers = {s["source"] for s in cited}
        # Two citations quoting the same paragraph are one piece of evidence.
        passages = {(s["source"], (s.get("text") or "")[:160]) for s in cited}

        want = set(it.get("expected_papers") or [])
        near = set(it.get("related_papers") or [])
        on_topic = want & cited_papers
        off_topic = cited_papers - want - near
        need = it.get("min_papers", 3)
        rows.append({
            "id": it["id"], "theme": it.get("theme", ""), "seconds": secs,
            "g_ok": ok, "g_n": len(it["key_facts"]),
            "on_topic": len(on_topic), "cited_n": len(cited_papers),
            "off_n": len(off_topic),
            "passages": len(passages), "citations": len(cited),
            "need": need, "broad": len(on_topic) >= need,
            "off_topic": sorted(off_topic),
            "cited_papers": sorted(cited_papers),
            "dropped": r.get("dropped_citations") or [],
            "missed": [it["key_facts"][j][0] for j, f in enumerate(flags) if not f],
            "answer": answer,
        })

    f_ok = sum(r["g_ok"] for r in rows); f_n = sum(r["g_n"] for r in rows)
    on = sum(r["on_topic"] for r in rows); tot = sum(r["cited_n"] for r in rows)
    off = sum(r["off_n"] for r in rows)
    broad = sum(1 for r in rows if r["broad"])
    cits = sum(r["citations"] for r in rows); dis = sum(r["passages"] for r in rows)
    bad = sum(len(r["dropped"]) for r in rows)
    secs = sum(r["seconds"] for r in rows)
    bar = "=" * 74
    print(f"\n{bar}\nTHEMATIC - discussion questions across a topic\n{bar}")
    print(f"  facts stated in answer  : {f_ok}/{f_n} ({f_ok / f_n:.0%})")
    print(f"  met breadth bar         : {broad}/{len(rows)} ({broad / len(rows):.0%})")
    print(f"  cited papers core/related/off: {on}/{tot - on - off}/{off}"
          f"  ({off / max(tot, 1):.0%} outside the theme)")
    print(f"  distinct passages cited : {dis}/{cits} ({dis / max(cits, 1):.0%})")
    print(f"  invalid citations emitted: {bad}")
    print(f"  wall clock              : {secs:.0f}s total, {secs / len(rows):.1f}s per question")
    for r in rows:
        mark = "ok " if r["broad"] and r["g_ok"] == r["g_n"] else "   "
        print(f"  {mark} {r['id']:20s} facts {r['g_ok']}/{r['g_n']}  "
              f"papers {r['on_topic']}/{r['need']}  "
              f"passages {r['passages']}/{r['citations']}  missed={r['missed']}")
        if verbose:
            if r["off_topic"]:
                print(f"        off topic: {[c[:34] for c in r['off_topic']]}")
            print(f"        cited: {[c[:34] for c in r['cited_papers']]}")
            print(f"        {r['answer'][:220]!r}")
    (ROOT / "eval_thematic.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


# ── passage-level retrieval: which passages came back, and in what order ───

GOLD_PASSAGES = ROOT / "tests" / "golden_passages.json"


def load_passage_items(limit: int | None = None) -> list[dict]:
    if not GOLD_PASSAGES.exists():
        return []
    data = json.loads(GOLD_PASSAGES.read_text(encoding="utf-8"))
    items = data["items"]
    return items[:limit] if limit else items


def _dcg(gains: list[float]) -> float:
    import math
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_ranking(rel: list[int], ideal: list[int], k: int) -> dict:
    """
    Rank-aware scores for one question.

    `rel` is the graded relevance of each returned passage in rank order, 0 for
    a passage that is not gold. `ideal` is every gold grade, best first, which
    is what a perfect ranking would have produced. Reported apart because they
    answer different questions: recall asks whether the evidence was found at
    all, MRR how soon the first correct passage appeared, MAP how well the gold
    passages are packed toward the top, and nDCG the same weighted by grade,
    which is the only one that distinguishes a passage that fully answers from
    one that merely supports.
    """
    cut = rel[:k]
    n_gold = len(ideal)
    found = sum(1 for r in cut if r > 0)

    rr = 0.0
    for i, r in enumerate(cut):
        if r > 0:
            rr = 1.0 / (i + 1)
            break

    # Average precision over the cut, divided by the gold available. Dividing
    # by min(n_gold, k) rather than n_gold keeps it from being unreachable when
    # a question has more gold passages than the system is allowed to return.
    hits = 0
    ap = 0.0
    for i, r in enumerate(cut):
        if r > 0:
            hits += 1
            ap += hits / (i + 1)
    ap /= max(1, min(n_gold, k))

    idcg = _dcg([float(g) for g in ideal[:k]])
    ndcg = _dcg([float(r) for r in cut]) / idcg if idcg else 0.0

    return {"recall": found / n_gold if n_gold else 0.0, "rr": rr,
            "ap": ap, "ndcg": ndcg, "found": found, "n_gold": n_gold}


def run_passages(items, verbose, k=10, client=None):
    """
    Score retrieval at passage granularity rather than paper granularity.

    The paper-level numbers cannot tell a run that returned the paragraph
    stating the result from one that returned the same paper's related-work
    section. On a corpus where most papers are about retrieval, that is most of
    the question. Gold passages are pinned by a verbatim anchor rather than a
    chunk id, so re-chunking the corpus does not silently invalidate the set.
    """
    from fastapi.testclient import TestClient
    import app as appmod

    if client is None:
        with TestClient(appmod.app) as c:
            return run_passages(items, verbose, k=k, client=c)

    rows = []
    for i, it in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {it['id']}", flush=True)
        t0 = time.perf_counter()
        # /api/ask returns the ranked passages a reader actually sees cited,
        # which is the unit these metrics are about. /api/chat was tried and is
        # the wrong shape: it returns k papers carrying one chunk each, so
        # there is no within-paper ranking to score.
        r = client.post("/api/ask", json={"query": it["question"], "top_k": k}).json()
        secs = time.perf_counter() - t0

        # A question the router sends to the paper cards never reaches passage
        # retrieval, so it has no ranking to score. Counting it as recall 0
        # would book a routing decision as a retrieval failure and quietly drag
        # every average down. Reported as its own number instead.
        carded = (r.get("target_sections") or []) == ["paper-cards"]

        by_anchor = {g["anchor"].lower(): g["grade"] for g in it["gold"]}
        rel = []
        for src in (r.get("sources") or [])[:k]:
            text = " ".join((src.get("text") or "").split()).lower()
            grade = 0
            if src.get("source") == it["paper"]:
                for a, g in by_anchor.items():
                    if a in text:
                        grade = max(grade, g)
            rel.append(grade)

        ideal = sorted((g["grade"] for g in it["gold"]), reverse=True)
        m = score_ranking(rel, ideal, k)
        m |= {"id": it["id"], "seconds": secs, "returned": len(rel),
              "rel": rel, "paper": it["paper"], "carded": carded}
        rows.append(m)

    carded = [r for r in rows if r["carded"]]
    rows = [r for r in rows if not r["carded"]]
    n = len(rows)
    bar = "=" * 74
    print(f"\n{bar}\nPASSAGE RETRIEVAL - is the answering passage returned, and where?\n{bar}")
    print(f"  Recall@{k}  : {statistics.mean(r['recall'] for r in rows):.3f}")
    print(f"  MRR@{k}     : {statistics.mean(r['rr'] for r in rows):.3f}")
    print(f"  MAP@{k}     : {statistics.mean(r['ap'] for r in rows):.3f}")
    print(f"  nDCG@{k}    : {statistics.mean(r['ndcg'] for r in rows):.3f}")
    hit1 = sum(1 for r in rows if r["rel"] and r["rel"][0] > 0)
    none = sum(1 for r in rows if r["found"] == 0)
    print(f"  gold passage ranked first : {hit1}/{n} ({hit1 / n:.0%})")
    print(f"  no gold passage returned  : {none}/{n} ({none / n:.0%})")
    print(f"  wall clock                : {sum(r['seconds'] for r in rows):.0f}s")
    if carded:
        print(f"  not scored, routed to cards: {len(carded)} "
              f"({[r['id'] for r in carded]})")
    if verbose or none:
        print("\n  per question (rank of gold grades in returned order):")
        for r in sorted(rows, key=lambda x: x["ndcg"]):
            print(f"    {r['id']:18s} nDCG {r['ndcg']:.2f}  recall {r['found']}/{r['n_gold']}"
                  f"  rel={r['rel']}")
    (ROOT / "eval_passages.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


# ── references: how much did imperfect retrieval cost the answer? ──────────

REFERENCES = ROOT / "tests" / "golden_references.json"


def run_references(verbose, limit=None, client=None):
    """
    Score the live answer against the answer this model gives when handed the
    gold passages outright.

    Everything else here scores by substring, which cannot credit a correct
    paraphrase and so reports a floor. This is the complement: greedy token
    matching over contextual embeddings, which credits meaning and is blind to
    wording. Neither is sufficient alone, and where they disagree is where the
    interesting cases live.

    The reference is a ceiling rather than a truth. Both sides come from the same
    model, so a misreading appears identically in both and scores a perfect
    match; this metric cannot see factual error, which is what the fact groups
    are for. What it can see is content that retrieval failed to put in front of
    the model, which is the one thing the fact groups keep attributing to
    generation.
    """
    from fastapi.testclient import TestClient
    import app as appmod
    from semantic_score import SemanticScorer, baseline_floor

    if not REFERENCES.exists():
        print("No tests/golden_references.json - run scripts/make_references.py first.")
        return []

    items = json.loads(REFERENCES.read_text(encoding="utf-8"))["items"]
    if limit:
        items = items[:limit]

    if client is None:
        with TestClient(appmod.app) as c:
            return run_references(verbose, limit=limit, client=c)

    scorer = SemanticScorer()
    rows = []
    for i, it in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {it['id']}", flush=True)
        t0 = time.perf_counter()
        r = client.post("/api/ask", json={"query": it["question"]}).json()
        secs = time.perf_counter() - t0
        answer = r.get("answer", "") if r.get("ok") else ""
        # Citation markers are not prose and match nothing in the reference.
        plain = re.sub(r"\[\d+\]", " ", answer)
        s = scorer.score(plain, it["reference"])
        rows.append({"id": it["id"], "seconds": secs, **s,
                     "answer": answer, "reference": it["reference"]})

    # An unrelated pair scores well above zero on same-domain prose, so the
    # floor is measured rather than assumed: without it a 0.85 is unreadable.
    floor = baseline_floor(scorer, [it["reference"] for it in items])
    n = len(rows)
    bar = "=" * 74
    print(f"\n{bar}\nREFERENCES - live answer against the perfect-retrieval ceiling\n{bar}")
    print(f"  F1        : {statistics.mean(r['f1'] for r in rows):.3f}")
    print(f"  precision : {statistics.mean(r['p'] for r in rows):.3f}  (content not in the reference)")
    print(f"  recall    : {statistics.mean(r['r'] for r in rows):.3f}  (reference content missing)")
    print(f"  unrelated-pair floor: {floor:.3f}   headroom above it: "
          f"{statistics.mean(r['f1'] for r in rows) - floor:+.3f}")
    worst = sorted(rows, key=lambda r: r["f1"])[:8]
    print("\n  furthest from the ceiling:")
    for r in worst:
        print(f"    {r['id']:18s} F1 {r['f1']:.3f}  P {r['p']:.3f}  R {r['r']:.3f}")
        if verbose:
            print(f"        live: {r['answer'][:150]!r}")
            print(f"        ref : {r['reference'][:150]!r}")
    (ROOT / "eval_references.json").write_text(
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
    ap.add_argument("--passages", action="store_true",
                    help="Passage-level retrieval: Recall@k, MRR, MAP, nDCG")
    ap.add_argument("--at-k", type=int, default=10,
                    help="Cutoff k for the passage-level metrics (default 10)")
    ap.add_argument("--references", action="store_true",
                    help="Semantic score against the perfect-retrieval ceiling")
    ap.add_argument("--thematic", action="store_true",
                    help="Score the topic-wide discussion questions")
    ap.add_argument("--negative", action="store_true",
                    help="Score the questions the corpus cannot answer")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-missing", action="store_true",
                    help="Score questions whose paper is no longer indexed "
                         "(they can only fail; off by default)")
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
        a.coverage = a.retrieval = a.generation = True
        a.synthesis = a.thematic = a.negative = a.passages = True
        a.references = True
    if not (a.coverage or a.retrieval or a.generation or a.synthesis
            or a.thematic or a.negative or a.passages or a.references):
        ap.error("choose at least one of --coverage / --retrieval / "
                 "--generation / --synthesis / --thematic / --passages / --all")

    items = load_items(a.limit, "single")
    items, absent, absent_papers = split_by_presence(items)
    if absent:
        print()
        print(f"NOTE: {len(absent)} question(s) skipped - "
              f"{len(absent_papers)} of their papers are no longer in the index:")
        for p in sorted(absent_papers):
            print(f"    {p}")
        print("  Re-add them, or prune the golden set. Scoring a question whose")
        print("  paper was removed measures the library, not the system.")
        if a.include_missing:
            items = items + absent
            print("  --include-missing given: scoring them anyway.")
        print()
    syn_items = load_items(a.limit, "synthesis")
    if syn_items:
        dropped = prune_expected_papers(syn_items, a.verbose)
        if dropped:
            print(f"NOTE: {dropped} expected-paper reference(s) dropped from the "
                  f"synthesis set - those papers are no longer indexed, so no "
                  f"answer could cite them.")
            print("  The facts metric still counts facts that only those papers "
                  "stated, so read it alongside the absent list above.")
            print()
    the_items = load_items(a.limit, "thematic")
    neg_items = load_items(a.limit, "negative")
    print(f"Golden set: {len(items)} single-paper question(s) over "
          f"{len({i['paper'] for i in items})} paper(s) "
          f"+ {len(syn_items)} synthesis + {len(the_items)} thematic "
          f"+ {len(neg_items)} negative control(s); "
          f"{sum(len(i['key_facts']) for i in items + syn_items + the_items)} "
          f"fact group(s)")

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
    if a.passages:
        pas_items = load_passage_items(a.limit)
        if not pas_items:
            print("No tests/golden_passages.json - skipping passage metrics.")
        else:
            run_passages(pas_items, a.verbose, k=a.at_k)

    if a.references:
        run_references(a.verbose, limit=a.limit)

    if a.thematic and the_items:
        from fastapi.testclient import TestClient
        import app as appmod
        with TestClient(appmod.app) as _client:
            runs = [run_thematic(the_items, a.verbose, client=_client)
                    for _ in range(max(1, a.repeat))]
        if len(runs) > 1:
            facts = [sum(r["g_ok"] for r in run) for run in runs]
            broad = [sum(1 for r in run if r["broad"]) for run in runs]
            f_n = sum(r["g_n"] for r in runs[0])
            bar = "=" * 74
            print(f"\n{bar}\nTHEMATIC over {len(runs)} runs\n{bar}")
            print(f"  facts  : {statistics.mean(facts):.1f}/{f_n} "
                  f"({statistics.mean(facts) / f_n:.0%})  "
                  f"range {min(facts)}-{max(facts)}  "
                  f"sd {statistics.pstdev(facts):.1f}")
            print(f"  breadth: {statistics.mean(broad):.1f}/{len(the_items)} "
                  f"range {min(broad)}-{max(broad)}")

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
