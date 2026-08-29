"""
Capture the README screenshots from the running app.

These are the first thing a reader sees, so they are generated rather than taken
by hand: a hand-cropped screenshot goes stale the moment the page changes and
nobody notices. This drives the real app against the real corpus, so a shot that
no longer matches the UI means the script failed, not that the image drifted.

    python scripts/screenshots.py

Writes docs/screenshots/*.png. Needs Weaviate up, Ollama running, and Playwright
with a browser installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_ui import Server  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
ANSWER_TIMEOUT_MS = 180_000

# A question that exercises the fan-out path and cites several papers, so the
# shot shows what the tab is actually for rather than a one-line lookup. It is
# deliberately not the placeholder text in the input box - a screenshot that
# answers its own example prompt looks staged.
ASK_Q = "Which transformer models have been fine-tuned across these papers?"

# Two turns, because the second cannot be understood alone: "that combination"
# only resolves against the first answer. A single-turn shot would show nothing
# the Ask tab does not already show.
PAPER_MATCH = "IndoBERTweet"
PAPER_TURNS = [
    "Which models are combined in this paper?",
    "What accuracy did that combination reach?",
]


def _wait_answer(page, thread: str):
    page.wait_for_selector(f"{thread} .answer", timeout=ANSWER_TIMEOUT_MS)
    page.wait_for_timeout(600)  # let the source list finish painting


def capture(page, base_url):
    OUT.mkdir(parents=True, exist_ok=True)
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.getElementById('health').textContent.includes('chunks')",
        timeout=60_000)

    # 1. a synthesised answer with citations
    page.fill("#ask-input", ASK_Q)
    page.click("#ask-form button[type=submit], #ask-form button:not([type=button])")
    _wait_answer(page, "#ask-thread")
    page.screenshot(path=str(OUT / "ask.png"))
    print("  ask.png")

    # 2. the same answer with a citation opened onto its source page
    chips = page.eval_on_selector_all("#ask-thread .cite", "c => c.length")
    if chips:
        page.click("#ask-thread .cite >> nth=0")
        page.wait_for_selector("#src-img:not(.hidden), #src-text:not(.hidden)",
                               timeout=60_000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "citation.png"))
        print("  citation.png")
        page.click("#src-close")

    # 3. a two-turn conversation scoped to one paper
    page.click('.tab-btn[data-tab="paper"]')
    value = page.eval_on_selector(
        "#paper-select",
        "(s, m) => [...s.options].find(o => o.value.includes(m))?.value",
        PAPER_MATCH)
    if not value:
        raise SystemExit(f"no paper matching {PAPER_MATCH!r} is ingested")
    page.select_option("#paper-select", value)
    for i, q in enumerate(PAPER_TURNS, 1):
        page.fill("#paper-input", q)
        page.click("#paper-form button[type=submit], "
                   "#paper-form button:not([type=button])")
        page.wait_for_function(
            "n => document.querySelectorAll('#paper-thread .answer').length >= n",
            arg=i, timeout=ANSWER_TIMEOUT_MS)
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT / "paper.png"))
    print("  paper.png")


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed")
        return 1

    with Server() as server:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900},
                                    device_scale_factor=2)
            try:
                capture(page, server.url)
            finally:
                browser.close()
    print(f"\nwrote to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
