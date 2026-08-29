from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

_HIGHLIGHT_COLOR = (1.0, 0.85, 0.0)  # golden yellow
_PAGE_GAP = 8                          # px between stacked pages
_PAGE_GAP_COLOR = (200, 200, 200)


def _search_phrases(text: str, word_window: int = 5, max_phrases: int = 60) -> list[str]:
    """
    Generate overlapping phrases that together cover the full chunk text.
    Uses a dynamic step so that max_phrases spans the entire token range,
    giving much denser highlight coverage than a fixed step.
    """
    words = text.split()
    if len(words) <= word_window:
        return [" ".join(words)]
    total_steps = len(words) - word_window
    step = max(1, total_steps // max_phrases)
    phrases, i = [], 0
    while i <= len(words) - word_window and len(phrases) < max_phrases:
        phrases.append(" ".join(words[i : i + word_window]))
        i += step
    return phrases


def render_chunk(
    source_path: str,
    page_numbers: str,
    chunk_text: str,
    zoom: float = 1.8,
) -> Image.Image | None:
    """
    Open the PDF at source_path, render the page(s) listed in page_numbers,
    highlight text matching chunk_text, and return a PIL image.
    Returns None if the file is missing or page info is unavailable.
    """
    if not source_path:
        return None
    path = Path(source_path)
    if not path.is_file():
        return None

    try:
        pages = [int(p) for p in page_numbers.split(",") if p.strip()]
    except (ValueError, AttributeError):
        pages = []

    if not pages:
        return None

    doc = fitz.open(str(path))
    rendered: list[Image.Image] = []

    for page_no in pages:
        if page_no < 1 or page_no > len(doc):
            continue
        page = doc[page_no - 1]

        for phrase in _search_phrases(chunk_text):
            for rect in page.search_for(phrase):
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=_HIGHLIGHT_COLOR)
                annot.update()

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        rendered.append(Image.open(io.BytesIO(pix.tobytes("png"))))

    doc.close()

    if not rendered:
        return None
    if len(rendered) == 1:
        return rendered[0]

    # Stack multiple pages vertically with a thin gap
    total_h = sum(img.height for img in rendered) + _PAGE_GAP * (len(rendered) - 1)
    max_w = max(img.width for img in rendered)
    canvas = Image.new("RGB", (max_w, total_h), _PAGE_GAP_COLOR)
    y = 0
    for img in rendered:
        canvas.paste(img, (0, y))
        y += img.height + _PAGE_GAP
    return canvas
