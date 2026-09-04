"""
Benchmark Ollama models on the classification tasks this pipeline actually depends on.

The generation step (writing a summary from retrieved chunks) is forgiving - almost any
instruct model produces something reasonable. The *routing* steps are not: they decide
which sections get searched at all, so an error there silently changes the answer.

Three tasks are measured, matching the three real call sites:

  headings   classify_headings_batch()   ingest time, once per paper, batched
  query      classify_query()            every quick search
  target     identify_target_section()   every deep scan

Ground truth spans nine disciplines, so a model that has only seen ML papers scores
poorly rather than looking deceptively fine.

Usage:
    python scripts/eval_models.py                        # every installed model
    python scripts/eval_models.py qwen2.5:7b llama3.2    # specific models
    python scripts/eval_models.py --think qwen3:8b       # allow reasoning output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama  # noqa: E402

from research_rag import llm as _llm_mod  # noqa: E402
from research_rag import section_classifier as sc  # noqa: E402

# ── Ground truth ───────────────────────────────────────────────────────────

# (heading, expected_label). Only headings whose label is unambiguous across
# disciplines - anything genuinely ambiguous belongs in _DEFER_TO_CONTENT, not here.
HEADINGS: list[tuple[str, str]] = [
    # computing
    ("Related Work", "related_work"),
    ("Proposed Method", "methodology"),
    ("Experimental Results", "results"),
    # medicine
    ("Trial Design", "methodology"),
    ("Inclusion and Exclusion Criteria", "dataset"),
    ("Baseline Characteristics of Enrolled Patients", "dataset"),
    ("Adverse Events", "results"),
    # psychology / social science
    ("Participants", "dataset"),
    ("Interview Protocol", "methodology"),
    ("Hypotheses Development", "theory"),
    # life sciences
    ("Reagents and Cell Lines", "dataset"),
    ("Sample Preparation", "methodology"),
    # physics / engineering
    ("Experimental Apparatus", "methodology"),
    ("Governing Equations", "theory"),
    # mathematics
    ("Proof of Theorem 2", "theory"),
    ("Preliminaries and Notation", "introduction"),
    # economics
    ("Identification Strategy", "methodology"),
    ("Descriptive Statistics", "dataset"),
    ("Robustness Checks", "results"),
    # humanities / law
    ("Historiography", "related_work"),
    ("Primary Sources", "dataset"),
    ("Fieldwork", "methodology"),
    # cross-cutting
    ("Limitations and Future Work", "conclusion"),
    ("Motivation", "introduction"),
]

# (query, expected_label). Scored as "expected label appears in the returned list",
# because classify_query is allowed to return up to two.
QUERIES: list[tuple[str, str]] = [
    ("What datasets were used?", "dataset"),
    ("How many participants were enrolled?", "dataset"),
    ("What corpus did the authors annotate?", "dataset"),
    ("What preprocessing was applied to the data?", "dataset"),
    ("How were patients randomized into groups?", "methodology"),
    ("What statistical tests were applied?", "methodology"),
    ("What instrument measured the outcome?", "methodology"),
    ("What model architecture was proposed?", "methodology"),
    ("What is the proof of the main theorem?", "theory"),
    ("What theoretical framework grounds this study?", "theory"),
    ("What assumptions does the formal model make?", "theory"),
    ("What did previous researchers already find?", "related_work"),
    ("What gaps does the literature review identify?", "related_work"),
    ("What accuracy did the system achieve?", "results"),
    ("What were the primary outcomes?", "results"),
    ("What adverse events occurred?", "results"),
    ("What are the limitations of this study?", "conclusion"),
    ("What future work do the authors suggest?", "conclusion"),
    ("Why does this problem matter?", "introduction"),
    ("What background concepts are needed to follow the paper?", "introduction"),
]

# identify_target_section returns a free phrase; it is mapped back through
# classify_heading, which is exactly what map_reduce does before cross-validating.
TARGETS: list[tuple[str, str]] = [
    ("What is the proof of the main theorem?", "theory"),
    ("Who were the study participants?", "dataset"),
    ("How were patients randomized?", "methodology"),
    ("What did prior studies report?", "related_work"),
    ("What were the primary outcomes?", "results"),
    ("What are the study's limitations?", "conclusion"),
]


class _Patched:
    """
    Force every section_classifier LLM call through one model, and optionally
    disable reasoning output. Records latency and whether the model emitted
    anything into Ollama's separate `thinking` field.
    """

    def __init__(self, model: str, think: bool | None):
        self.model, self.think = model, think
        self.calls, self.seconds, self.thinking_calls = 0, 0.0, 0
        self._real = _llm_mod.ollama.generate

    def __enter__(self):
        def patched(model=None, prompt=None, **kw):
            if self.think is not None:
                kw["think"] = self.think
            t0 = time.perf_counter()
            resp = self._real(model=self.model, prompt=prompt, **kw)
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            if getattr(resp, "thinking", None):
                self.thinking_calls += 1
            return resp

        _llm_mod.ollama.generate = patched
        return self

    def __exit__(self, *_):
        _llm_mod.ollama.generate = self._real


def _score_headings(model: str) -> tuple[int, int, list[str]]:
    names = [h for h, _ in HEADINGS]
    got = sc.classify_headings_batch(names, model)
    wrong = [
        f"{h!r}: got {got.get(h)!r}, want {want!r}"
        for h, want in HEADINGS
        if got.get(h) != want
    ]
    return len(HEADINGS) - len(wrong), len(HEADINGS), wrong


def _score_queries(model: str) -> tuple[int, int, list[str]]:
    wrong = []
    for q, want in QUERIES:
        got = sc.classify_query(q, model)
        if want not in got:
            wrong.append(f"{q!r}: got {got}, want {want!r}")
    return len(QUERIES) - len(wrong), len(QUERIES), wrong


def _score_targets(model: str) -> tuple[int, int, list[str]]:
    wrong = []
    for q, want in TARGETS:
        phrase = sc.identify_target_section(q, model)
        mapped = sc.classify_heading(phrase)
        if mapped != want:
            wrong.append(f"{q!r}: got {phrase!r} -> {mapped!r}, want {want!r}")
    return len(TARGETS) - len(wrong), len(TARGETS), wrong


TASKS = {"headings": _score_headings, "query": _score_queries, "target": _score_targets}


def evaluate(model: str, think: bool | None, verbose: bool) -> dict:
    print(f"\n{'=' * 72}\n{model}\n{'=' * 72}")
    row = {"model": model, "total_ok": 0, "total_n": 0}
    with _Patched(model, think) as p:
        for name, fn in TASKS.items():
            try:
                ok, n, wrong = fn(model)
            except Exception as exc:
                print(f"  {name:9s} ERROR: {exc}")
                row[name] = None
                continue
            row[name] = ok / n
            row["total_ok"] += ok
            row["total_n"] += n
            print(f"  {name:9s} {ok:2d}/{n:2d}  ({ok / n:5.0%})")
            if verbose and wrong:
                for w in wrong:
                    print(f"              - {w}")
    row["seconds"] = p.seconds
    row["calls"] = p.calls
    row["thinking_calls"] = p.thinking_calls
    row["accuracy"] = row["total_ok"] / row["total_n"] if row["total_n"] else 0.0
    print(
        f"  {'TOTAL':9s} {row['total_ok']:2d}/{row['total_n']:2d}  "
        f"({row['accuracy']:5.0%})   {p.seconds:6.1f}s over {p.calls} calls"
    )
    if p.thinking_calls:
        print(
            f"  note: emitted reasoning into the separate `thinking` field on "
            f"{p.thinking_calls}/{p.calls} calls"
        )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="Model tags (default: all installed)")
    ap.add_argument("--think", dest="think", action="store_true", help="Allow reasoning output")
    ap.add_argument("--no-think", dest="think", action="store_false", help="Suppress reasoning")
    ap.add_argument("-v", "--verbose", action="store_true", help="List every wrong answer")
    ap.set_defaults(think=None)
    args = ap.parse_args()

    models = args.models
    if not models:
        installed = ollama.list()
        models = [
            m.model for m in installed.models
            if "embed" not in m.model  # embedding models cannot do this
        ]
    if not models:
        print("No models found. Pull one with: ollama pull qwen3:8b", file=sys.stderr)
        raise SystemExit(1)

    rows = [evaluate(m, args.think, args.verbose) for m in models]
    rows.sort(key=lambda r: (-r["accuracy"], r["seconds"]))

    print(f"\n{'=' * 72}\nSUMMARY (ranked by accuracy, then speed)\n{'=' * 72}")
    print(f"{'model':28s} {'head':>6s} {'query':>6s} {'target':>7s} {'total':>7s} {'time':>8s}")
    for r in rows:
        def pct(v):
            return f"{v:.0%}" if v is not None else "err"
        print(
            f"{r['model'][:28]:28s} {pct(r.get('headings')):>6s} {pct(r.get('query')):>6s} "
            f"{pct(r.get('target')):>7s} {r['accuracy']:>6.0%} {r['seconds']:>7.1f}s"
        )


if __name__ == "__main__":
    main()
