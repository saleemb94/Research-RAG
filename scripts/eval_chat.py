"""
Score the two conversational tabs against tests/golden_conversations.json.

The corpus-wide set asks self-contained questions, so it never touches what these
tabs exist for: a follow-up that only means something given what came before.
"And how were they split for training?" is embedded literally by retrieval, so
without history it matches nothing useful.

Two things are scored, and they can fail independently:

  resolution  did the follow-up get rewritten into a standalone question that
              actually names what it referred to
  facts       did the answer state what it should

A question can resolve correctly and still be answered badly, and - more
misleadingly - a question can be answered well by luck while the rewrite dropped
the reference entirely, which only shows up on the turn after.

    python scripts/eval_chat.py                 # single-paper conversations
    python scripts/eval_chat.py --scratch       # upload a PDF and chat with it
    python scripts/eval_chat.py --all -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_rag import norm, score_facts  # noqa: E402

GOLDEN = ROOT / "tests" / "golden_conversations.json"


def _ask(client, question, history, **extra):
    body = {"query": question, "history": history, **extra}
    return client.post("/api/ask", json=body).json()


def run_conversation(client, conv, verbose, **extra) -> list[dict]:
    """Walk one conversation, carrying history exactly as the UI does."""
    history: list[dict] = []
    rows = []
    for i, turn in enumerate(conv["turns"], 1):
        t0 = time.perf_counter()
        r = _ask(client, turn["q"], history, **extra)
        secs = time.perf_counter() - t0

        answer = r.get("answer", "") if r.get("ok") else ""
        rewritten = r.get("query", turn["q"])
        ok, flags = score_facts(turn["key_facts"], answer)

        # Only meaningful for a turn that cannot stand alone.
        resolved = None
        if turn.get("needs_history"):
            wanted = turn.get("resolves_to") or []
            hay = norm(rewritten)
            resolved = any(norm(w).strip() in hay for w in wanted)

        rows.append({
            "conversation": conv["id"], "turn": i, "q": turn["q"],
            "rewritten": rewritten, "answer": answer, "seconds": secs,
            "ok": ok, "n": len(turn["key_facts"]), "resolved": resolved,
            "missed": [turn["key_facts"][j][0] for j, f in enumerate(flags) if not f],
        })

        # The UI stores the rewritten question, because that is what retrieval
        # actually saw; storing the raw follow-up would make the next rewrite
        # resolve against something the system never used.
        history.append({"role": "user", "content": rewritten})
        history.append({"role": "assistant", "content": answer[:600]})

        mark = "ok " if ok == len(turn["key_facts"]) else "   "
        res = "" if resolved is None else (" resolved" if resolved else " UNRESOLVED")
        print(f"    {mark}turn {i}: {ok}/{len(turn['key_facts'])}{res}  {secs:5.1f}s")
        if verbose or ok < len(turn["key_facts"]) or resolved is False:
            print(f"        asked   : {turn['q']}")
            print(f"        rewrote : {rewritten}")
            if rows[-1]["missed"]:
                print(f"        missed  : {rows[-1]['missed']}")
    return rows


def report(rows, label):
    n_turns = len(rows)
    facts_ok = sum(r["ok"] for r in rows)
    facts_n = sum(r["n"] for r in rows)
    full = sum(1 for r in rows if r["ok"] == r["n"])
    follow = [r for r in rows if r["resolved"] is not None]
    res_ok = sum(1 for r in follow if r["resolved"])
    secs = sum(r["seconds"] for r in rows)

    bar = "=" * 74
    print(f"\n{bar}\n{label}\n{bar}")
    print(f"  turns                  : {n_turns}")
    print(f"  facts stated           : {facts_ok}/{facts_n} ({facts_ok / facts_n:.0%})")
    print(f"  fully correct turns    : {full}/{n_turns} ({full / n_turns:.0%})")
    if follow:
        print(f"  follow-ups resolved    : {res_ok}/{len(follow)} ({res_ok / len(follow):.0%})")
    print(f"  wall clock             : {secs:.0f}s total, {secs / n_turns:.1f}s per turn")
    bad = [r for r in follow if not r["resolved"]]
    if bad:
        print(f"\n  follow-ups that lost the reference ({len(bad)}):")
        for r in bad:
            print(f"    {r['conversation']} turn {r['turn']}: {r['q']}")
            print(f"      became: {r['rewritten']}")


def run_single_paper(client, data, verbose):
    rows = []
    for conv in data["conversations"]:
        print(f"\n  {conv['id']}  ({conv['paper'][:46]})")
        rows += run_conversation(client, conv, verbose, source_filter=conv["paper"])
    report(rows, "SINGLE-PAPER CONVERSATIONS")
    return rows


def run_unscoped(client, data, verbose):
    """
    Conversations with no paper filter, where resolving the reference decides
    which paper is retrieved. This is where the resolution metric has teeth: a
    follow-up that keeps "the adversarial robustness one" retrieves whatever the
    words happen to match, not the paper the user meant.
    """
    rows = []
    for conv in data.get("unscoped", []):
        print(f"\n  {conv['id']}  (whole library)")
        rows += run_conversation(client, conv, verbose)
    if rows:
        report(rows, "CORPUS-WIDE CONVERSATIONS")
    return rows


def run_scratch(client, data, verbose):
    """
    Upload, converse, prove isolation, clean up.

    Isolation is the point of the separate collection, so it is asserted rather
    than assumed: a document dropped in to ask one question must never appear in
    a library-wide answer.
    """
    rows = []
    bar = "=" * 74
    if True:
        before = set(client.get("/api/papers").json()["papers"])
        for spec in data.get("scratch", []):
            src = ROOT / spec["source_pdf"]
            if not src.exists():
                print(f"  ! missing {src}")
                continue
            print(f"\n  {spec['id']}  uploading {spec['upload_as']}")
            t0 = time.perf_counter()
            with open(src, "rb") as fh:
                up = client.post(
                    "/api/scratch",
                    files={"files": (spec["upload_as"], fh, "application/pdf")},
                ).json()
            res = up["results"][0]
            print(f"    ingest: {res['status']} - {res.get('message')} "
                  f"in {res.get('seconds')}s (wall {time.perf_counter() - t0:.0f}s)")
            if res["status"] != "ok":
                continue

            conv = {"id": spec["id"], "turns": spec["turns"]}
            rows += run_conversation(client, conv, verbose,
                                     scratch=True, source_filter=spec["upload_as"])

            scratch_now = set(client.get("/api/scratch").json()["papers"])
            corpus_now = set(client.get("/api/papers").json()["papers"])
            leaked = spec["upload_as"] in corpus_now or corpus_now != before
            print(f"\n{bar}\nISOLATION\n{bar}")
            print(f"  in the scratch store   : {spec['upload_as'] in scratch_now}")
            print(f"  library unchanged      : {not leaked}"
                  f"{'   <-- LEAKED' if leaked else ''}")

            # A corpus-wide question must not see it.
            probe = client.post("/api/ask", json={
                "query": "What modalities are used for hate speech detection?"}).json()
            cited = {s["source"] for s in (probe.get("sources") or [])}
            print(f"  absent from a library-wide answer: "
                  f"{spec['upload_as'] not in cited}")

            client.delete(f"/api/scratch/{spec['upload_as']}")
            gone = spec["upload_as"] not in set(
                client.get("/api/scratch").json()["papers"])
            print(f"  removed on cleanup     : {gone}")
    if rows:
        report(rows, "AD-HOC UPLOAD CONVERSATIONS")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", action="store_true", help="Only the upload tab")
    ap.add_argument("--single", action="store_true", help="Only the single-paper tab")
    ap.add_argument("--unscoped", action="store_true",
                    help="Only the corpus-wide conversations")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.all or not (a.scratch or a.single or a.unscoped):
        a.scratch = a.single = a.unscoped = True

    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"{len(data['conversations'])} conversation(s), "
          f"{sum(len(c['turns']) for c in data['conversations'])} turns; "
          f"{len(data.get('scratch', []))} scratch scenario(s)")
    from fastapi.testclient import TestClient
    import app as appmod

    with TestClient(appmod.app) as client:
        if a.single:
            run_single_paper(client, data, a.verbose)
        if a.unscoped:
            run_unscoped(client, data, a.verbose)
        if a.scratch:
            run_scratch(client, data, a.verbose)


if __name__ == "__main__":
    main()
