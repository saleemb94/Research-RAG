"""
What counts as a usable chunk, and what a heading really is.

Shared by ingestion and by scripts/repair_index.py so a paper added today and a
paper repaired from last month are judged identically. Every threshold here was
measured against a 54-paper index rather than guessed, and the removal rules
were checked against the golden set: they delete 14.8% of chunks and lose zero
graded facts.
"""

from __future__ import annotations

import re

# Section name given to front matter - the title block and abstract that
# Docling files under the paper's own title.
FRONT_MATTER_SECTION = "Abstract"

# Docling's stand-in for a formula it could not parse. Indexed as-is it is a
# chunk with no content at all; one paper carried seventeen of them.
_PLACEHOLDER = re.compile(r"^\s*<!--.*-->\s*$", re.S)

# Journal furniture stamped into the page by the publisher's PDF pipeline.
_FURNITURE = re.compile(
    r"authorized licensed use|downloaded on .{0,40}from ieee xplore"
    r"|restrictions apply|^\s*©|all rights reserved|personal use is permitted"
    r"|this article has been accepted for publication",
    re.I,
)

# A caption with no figure attached. "Fig. 4. Distribution of Word Counts in
# Cyberbullying Instances" is a label for something the index does not hold.
_CAPTION = re.compile(r"^\s*(fig(ure)?\.?|table|algorithm|listing)\s*\d+\s*[.:]?\s", re.I)

# Below this a chunk cannot carry a retrievable claim: "3:", "(a) OHSUMED",
# "Mgen2, p2, pc2, pm2".
_MIN_LETTERS = 15
_MAX_CAPTION_CHARS = 200
_MAX_FURNITURE_CHARS = 300


def noise_verdict(text: str | None) -> str | None:
    """
    Why this chunk carries no retrievable content, or None if it does.

    Deliberately conservative. A short chunk can still be the answer - "The
    best accuracy was 98.83%." is thirty characters - so length alone is never
    the test; what disqualifies a chunk is having no words in it at all, or
    being a known artefact.
    """
    t = (text or "").strip()
    if not t:
        return "empty"
    if _PLACEHOLDER.match(t):
        return "placeholder"
    if _FURNITURE.search(t) and len(t) < _MAX_FURNITURE_CHARS:
        return "furniture"
    if sum(c.isalpha() for c in t) < _MIN_LETTERS:
        return "no words"
    if _CAPTION.match(t) and len(t) < _MAX_CAPTION_CHARS:
        return "caption only"
    return None


def is_caption_heading(section_name: str | None) -> bool:
    """True when a heading is a figure, table or algorithm caption.

    Docling promotes these to headings, and they then own every chunk until the
    next real heading - "Algorithm 1 XAI-based word replacement algorithm."
    owned thirty-six. Those chunks lose their actual section, so section
    filtering cannot reach them."""
    return bool(_CAPTION.match((section_name or "").strip()))


def _key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def same_title(heading: str | None, title: str | None) -> bool:
    """
    Is this heading the paper's own title?

    Compared against the title resolved at ingest rather than guessed at. The
    heuristic this replaces asked whether a heading was long and free of
    section vocabulary, and titles are routinely neither: thirteen of seventeen
    misses were titles containing "analysis", "approach", "method",
    "evaluation", "robustness" - even "notation", matched inside "Annotations".

    Compared on letters and digits alone, because PDF extraction inserts line
    breaks, hyphens and stray spacing that differ between the heading and the
    metadata title.
    """
    h, t = _key(heading), _key(title)
    if not h or not t or len(t) < 12:
        return False
    return h == t or h.startswith(t[:40]) or t.startswith(h[:40])
