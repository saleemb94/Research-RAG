"""
Work out what a paper is actually called.

Filenames are what publishers happen to export: `1-s2.0-S277244252300045X-main`,
`3628797.3628810`, `arxiv_2304.00913`. Showing those as titles makes a shelf of
real research look like a directory listing, so a title is resolved once per
paper at ingest and stored on its card.

Three sources, cheapest and most reliable first:

  1. the PDF's own metadata title, which is right when it is present and
     obviously wrong when it is not - "Microsoft Word - draft3.doc", the
     filename again, a DOI;
  2. the largest text on the first page, which is how a title is actually
     marked in a typeset paper: font size, not position;
  3. the model, given the first page's text.

Every candidate goes through the same validation, because a bad title is worse
than a filename - it is a filename that lies. Nothing here fails an ingest: if
no source yields something plausible, the caller keeps using the filename.
"""

from __future__ import annotations

import re

from .config import OLLAMA_MODEL
from .llm import generate as _llm

# A title is a sentence-ish phrase. These bounds reject running heads, author
# lines, DOIs and the odd full paragraph that leads a badly typeset paper.
MIN_CHARS = 16
MAX_CHARS = 300
MIN_WORDS = 3

_JUNK = re.compile(
    r"^(microsoft word|untitled|document\d*|draft|paper|manuscript|preprint"
    r"|\d+\s*$|doi:|http)", re.I
)
# Author lines are the most common false positive: a run of capitalised names
# separated by commas or "and", often with an affiliation trailing.
_AUTHORS = re.compile(
    r"^[A-Z][a-z]+\s+[A-Z][a-z]*\.?\s*(,|and\b|&)"
    r"|^([A-Z]{2,}\s+){1,3}(AND|and)\s+[A-Z]{2,}",
)
_ALL_CAPS_SHORT = re.compile(r"^[^a-z]{0,40}$")

_PROMPT = """Below is the first page of a research paper.

Reply with the paper's title, exactly as printed, and nothing else. No quotes,
no label, no explanation. If no title is visible, reply NONE.

FIRST PAGE:
{text}

Title:"""


def _clean(raw: str) -> str:
    t = (raw or "").strip().strip('"“”\'')
    t = re.sub(r"\s+", " ", t)                       # newlines mid-title
    t = re.sub(r"^(title|paper)\s*[:\-]\s*", "", t, flags=re.I)
    t = re.sub(r"\s*[\*†‡¹²³]+$", "", t)  # footnote marks
    return t.strip(" .,;:-–—")


def plausible(candidate: str, source_file: str = "") -> bool:
    """Would a reader accept this as the paper's title?"""
    t = _clean(candidate)
    if not (MIN_CHARS <= len(t) <= MAX_CHARS):
        return False
    if len(t.split()) < MIN_WORDS:
        return False
    if _JUNK.match(t) or _AUTHORS.match(t):
        return False
    # Deliberately no "reject titles matching the filename" rule. It looks
    # sensible and is wrong: when a filename is already the title, the typeset
    # version is still the better label - properly punctuated and capitalised -
    # and rejecting it sent perfectly good titles back to the underscored
    # filename. Identifier-style filenames are caught by the letters ratio
    # below instead, which is what that rule was really reaching for.
    # A line with no lowercase letters at all is a running head or a banner.
    if _ALL_CAPS_SHORT.match(t):
        return False
    # Must read as words, not as an identifier.
    letters = sum(c.isalpha() for c in t)
    if letters < len(t) * 0.6:
        return False
    return True


def from_metadata(doc) -> str:
    """The PDF's declared title, if it declared a usable one."""
    try:
        return _clean((doc.metadata or {}).get("title") or "")
    except Exception:
        return ""


def from_layout(doc, max_spans: int = 400) -> str:
    """
    The largest text on the first page.

    In a typeset paper the title is set larger than anything else on page one,
    which survives two-column layouts and logo banners better than "the first
    line" does. Spans at the same size are joined so a title that wraps across
    two lines comes back whole.
    """
    try:
        page = doc[0]
        spans = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if text:
                        spans.append((round(span.get("size", 0), 1),
                                      span.get("bbox", [0, 0, 0, 0])[1], text))
                    if len(spans) > max_spans:
                        raise StopIteration
    except StopIteration:
        pass
    except Exception:
        return ""
    if not spans:
        return ""

    biggest = max(s[0] for s in spans)
    # Same size, in reading order. A wrapped title is several spans.
    parts = [t for size, _, t in sorted(spans, key=lambda s: s[1])
             if size >= biggest - 0.6]
    return _clean(" ".join(parts))


def from_model(first_page_text: str, model: str = OLLAMA_MODEL) -> str:
    if not first_page_text.strip():
        return ""
    try:
        raw = _llm(_PROMPT.format(text=first_page_text[:2500]), model,
                   max_tokens=80)
    except Exception:
        return ""
    if raw.strip().upper().startswith("NONE"):
        return ""
    return _clean(raw.splitlines()[0] if raw else "")


def extract_title(pdf_path: str, model: str = OLLAMA_MODEL) -> tuple[str, str]:
    """
    Resolve one paper's title.

    Returns (title, how) where `how` is the source that produced it -
    "metadata", "layout", "model" or "" when none did. The caller keeps the
    filename in that last case.
    """
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - pymupdf is a hard dependency
        return "", ""

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return "", ""

    try:
        name = str(pdf_path).replace("\\", "/").rsplit("/", 1)[-1]

        meta = from_metadata(doc)
        if plausible(meta, name):
            return _clean(meta), "metadata"

        layout = from_layout(doc)
        if plausible(layout, name):
            return _clean(layout), "layout"

        try:
            page_text = doc[0].get_text()
        except Exception:
            page_text = ""
        guess = from_model(page_text, model)
        if plausible(guess, name):
            return _clean(guess), "model"

        return "", ""
    finally:
        doc.close()
