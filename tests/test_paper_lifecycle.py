"""
A paper lives in three collections. Adding and removing one must keep all three
in step, and nothing else in the system enforces that.

  ResearchChunk                the chunks and their vectors
  ResearchPaperCard            one structured card, used for corpus questions
  ResearchChunkSectionSummary  section summaries, the second retrieval channel

Every one of these was wrong at some point: delete cleaned the first two and
left the third, so a deleted paper stayed citable; upload wrote none of the
third, so a paper added through the UI was invisible to that channel; and upload
wrote the PDF to disk before checking whether the name was taken, which replaced
the file while keeping the old chunks - the index describing one document while
the viewer rendered pages from another.

Uses a small generated PDF rather than a real paper so the whole lifecycle runs
in seconds. Skips cleanly if Weaviate or PyMuPDF is unavailable.

    python tests/test_paper_lifecycle.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NAME = "zz_lifecycle_probe.pdf"
REPLACEMENT = "zz_lifecycle_probe_other.pdf"

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def make_pdf(path: Path, body: str):
    """A one-page paper with a real heading, so classification has something."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "A Probe Document", fontsize=16)
    page.insert_text((72, 130), "1 Methods", fontsize=13)
    y = 160
    for line in body.split("\n"):
        page.insert_text((72, y), line, fontsize=10)
        y += 16
    doc.save(str(path))
    doc.close()


def counts(appmod, name: str) -> dict:
    """How many places know about this paper."""
    return {
        "chunks": name in appmod.store.list_sources(),
        "card": any(c.source_file == name
                    for c in appmod.card_store.all_cards()),
        "summaries": len(appmod.section_index.list_sources()),
        "in_summaries": name in appmod.section_index.list_sources(),
    }


def main() -> int:
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        print("pymupdf not installed - skipping")
        return 0

    from fastapi.testclient import TestClient
    import app as appmod

    tmp = ROOT / "tests" / "_tmp_lifecycle"
    tmp.mkdir(exist_ok=True)
    probe = tmp / NAME
    other = tmp / REPLACEMENT
    make_pdf(probe, "The probe corpus contains 4242 annotated samples.\n"
                    "Accuracy reached 71.5 percent on the held-out split.")
    make_pdf(other, "Entirely different content that must not overwrite it.")

    dest = appmod.PAPERS_DIR / NAME
    client = TestClient(appmod.app)
    try:
        # -- clean slate, in case an earlier run died mid-way ---------------
        client.delete(f"/api/papers/{NAME}")

        # -- upload -------------------------------------------------------
        with open(probe, "rb") as fh:
            r = client.post("/api/papers",
                            files={"files": (NAME, fh, "application/pdf")}).json()
        res = r["results"][0]
        check("upload succeeds", res["status"] == "ok", res.get("message", ""))

        after = counts(appmod, NAME)
        check("chunks are stored", after["chunks"])
        check("a paper card is written", after["card"])
        check("section summaries are indexed", after["in_summaries"],
              f"{after['summaries']} papers in the summary index")

        # -- re-upload under the same name must not touch the PDF on disk --
        before_hash = hashlib.sha256(dest.read_bytes()).hexdigest() \
            if dest.exists() else None
        with open(other, "rb") as fh:
            r2 = client.post("/api/papers",
                             files={"files": (NAME, fh, "application/pdf")}).json()
        msg = r2["results"][0].get("message", "")
        check("re-upload is refused, not duplicated", "already ingested" in msg, msg)
        after_hash = hashlib.sha256(dest.read_bytes()).hexdigest() \
            if dest.exists() else None
        check("the stored PDF is left untouched", before_hash == after_hash,
              "different content under a taken name must not replace the file")

        # -- delete must empty all three ----------------------------------
        # Guarantee there is something to orphan, independently of whether
        # upload indexed it. Otherwise the upload bug masks the delete bug:
        # nothing was ever added, so nothing is left behind, and the delete
        # assertion passes on code that never cleans the summaries at all.
        if not counts(appmod, NAME)["in_summaries"]:
            from research_rag.section_index import build_entries, index_entries
            index_entries(appmod.section_index,
                          build_entries(appmod.store, source_filter=NAME),
                          appmod.embedder)
        check("summaries exist before the delete",
              counts(appmod, NAME)["in_summaries"])

        client.delete(f"/api/papers/{NAME}")
        gone = counts(appmod, NAME)
        check("chunks are removed", not gone["chunks"])
        check("the paper card is removed", not gone["card"])
        check("section summaries are removed", not gone["in_summaries"],
              "a deleted paper must not stay citable")
        check("the PDF is removed from papers/", not dest.exists())
    finally:
        client.close()

    # -- the app must survive being started more than once ----------------
    # The stores were opened at import and closed on shutdown, so they could
    # only be closed once: a second client context left every request failing
    # with a closed-client error. It scored zero and read as a total
    # regression. This has cost two debugging sessions, so it is pinned.
    try:
        with TestClient(appmod.app) as c1:
            first = c1.get("/api/health").json().get("ok")
        with TestClient(appmod.app) as c2:
            second = c2.get("/api/health").json().get("ok")
        check("a second app context still serves requests",
              bool(first) and bool(second), f"first={first} second={second}")
    except Exception as exc:
        check("a second app context still serves requests", False,
              f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}")

    if True:
        client.delete(f"/api/papers/{NAME}")
        client.close()
        for f in (probe, other):
            f.unlink(missing_ok=True)
        tmp.rmdir()

    failed = sum(1 for _, ok in CHECKS if not ok)
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} check(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
