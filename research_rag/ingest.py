from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

from .embedder import Embedder
from .llm import generate as _llm
from .section_classifier import (
    classify_heading,
    classify_headings_batch,
    verify_with_first_paragraph,
)
from .config import OLLAMA_MODEL
from .vector_store import VectorStore


# ---------------------------------------------------------------------------
# Section blacklist — chunks in these sections are never stored
# ---------------------------------------------------------------------------

_SKIP_SECTION_NAMES = {
    # Citations / bibliography
    "references", "bibliography", "works cited", "reference list",
    "literature cited", "notes", "endnotes", "footnotes",
    # Author admin
    "acknowledgments", "acknowledgements", "acknowledgment",
    "author contributions", "author contribution statement",
    "author contributions statement", "authors contributions",
    "about the authors", "author biographies", "author information",
    "credit authorship contribution statement", "orcid", "orcid ids",
    # Legal / ethics / compliance
    "conflicts of interest", "conflict of interest",
    "declaration of competing interest", "declaration of competing interests",
    "declarations", "disclosure", "disclosures", "financial disclosure",
    "ethical considerations", "ethics statement", "ethics approval",
    "ethics approval and consent to participate", "irb approval",
    "informed consent statement", "informed consent", "consent for publication",
    "safety considerations", "safety standards",
    "trial registration", "registration", "protocol registration",
    "permissions", "copyright", "provenance and peer review",
    # Funding
    "funding", "funding information", "funding statement",
    "role of the funding source", "grant support",
    # Data & supplementary
    "data availability", "data availability statement",
    "code availability", "supporting information",
    "supplementary material", "supplementary materials",
    "supplementary information", "additional file", "additional files",
    # Format markers / reference apparatus
    "acm reference format", "acmreference format",
    "appendix", "appendices", "abbreviations", "list of abbreviations",
    "nomenclature", "glossary", "keywords", "highlights",
}

# Keywords whose presence in a section_name indicates it IS a real section
# (used to rescue long headings that might look like titles but aren't).
#
# Only terms that appear in section names but almost never in paper titles.
# Generic research words like "detection", "model", "framework", "generation",
# "design" and "simulation" are excluded because they are common in titles
# ("Offensive Language Detection in Arabic Social Networks Using...",
#  "Designing a Haptic Boot for Space...").
#
# Covers section vocabulary from several disciplines, not just computing: a
# long clinical heading like "Participants and Recruitment Procedures for the
# Longitudinal Cohort" must not be mistaken for a paper title and discarded.
# The asymmetry matters — a missed rescue silently drops real content, while a
# false rescue only stores one extra heading.
_SECTION_KEYWORDS = {
    # structure, all fields
    "introduction", "related", "literature", "background", "preliminary",
    "method", "approach", "algorithm", "overview", "motivation",
    "contribution", "discussion", "conclusion", "future",
    "analysis", "experiment", "evaluation", "result", "finding",
    "limitation", "implication", "recommendation", "summary",
    # data, under whatever name the field uses
    "dataset", "data", "corpus", "sample", "specimen", "cohort",
    "participant", "population", "subject", "respondent",
    "inclusion", "exclusion", "eligibility", "recruitment", "demographic",
    # procedure
    "materials", "procedure", "protocol", "instrument", "apparatus",
    "measure", "questionnaire", "interview", "fieldwork", "setup",
    # formal / theoretical
    "theoretical", "conceptual", "theorem", "lemma", "proof", "derivation",
    "hypothes", "notation", "definition",
    # quantitative social science / economics
    "estimation", "robustness", "descriptive", "specification", "outcome",
}


def _looks_like_paper_title(section_name: str) -> bool:
    """
    Return True if the section_name is almost certainly the document title
    rather than a real section heading.
    Heuristics: very long, mixed-case, and no standard section keywords.
    """
    if len(section_name) < 55:
        return False
    # All-caps headings like "PHASE I: BASE CLASSIFIERS LEARNED FROM..." are real sections
    if section_name.isupper():
        return False
    lower = section_name.lower()
    if any(kw in lower for kw in _SECTION_KEYWORDS):
        return False
    return True


# Section name given to front matter that Docling filed under the paper title.
FRONT_MATTER_SECTION = "Abstract"


def _is_skippable(section_name: str) -> bool:
    """True for boilerplate that carries no research content."""
    if not section_name:
        return True
    # Normalise: lowercase, strip trailing punctuation
    normalised = section_name.strip().lower().rstrip(":.")
    return normalised in _SKIP_SECTION_NAMES


# ---------------------------------------------------------------------------
# Heading utilities
# ---------------------------------------------------------------------------

# Matches ONLY a complete leading numbering prefix, always followed by
# at least one whitespace character.  Never strips the first letter of a word.
#
#   (?:\d+\.?)+          →  1  /  1.  /  1.2  /  1.2.3
#   [IVXivxLCDM]+\.?     →  III  /  IV.
#   [A-Za-z][.:]         →  A.  /  A:  /  a.  /  a:   (single-letter labels)
#   [):]?                →  optional closing paren or colon  (handles "1) Heading")
#   \s+                  →  at least one space  (prevents stripping word-start chars)
_NUM_PREFIX = re.compile(
    r'^(?:'
    r'(?:\d+\.?)+|'
    r'[IVXivxLCDM]+\.?|'
    r'[A-Za-z][.:]'
    r')[):]?\s+'
)

# Used in ingest_pdf to detect orphaned subsection headings.
#
# A "top-level prefix" is one of:
#   - Pure digits:           1   12   123
#   - Roman numerals (caps): I   II   III   IV   VIII   (up to 4 chars)
#   - Single uppercase letter: A   B   C
# followed immediately by ".", ")", or whitespace.
# The ")" case handles IEEE L3 sub-subsections like "1) Word2Vec".
#
# A "dot-separated sub-level" has two components joined by a dot with no space:
#   3.1   I.A   A.1   1.2.3 (parent = "1")   II.B   C.2
#
# NOTE: IEEE-style papers use "A. Heading" (dot + space), which does NOT match
# _HEADING_SUB_NUM.  Those are handled separately via _build_parent_maps().
#
# Deliberately excludes lowercase-start words so normal prose headings
# like "Introduction" or "Background" are never misidentified.
_HEADING_TOP_NUM = re.compile(
    r'^(\d{1,3}|[IVXLCDM]{1,4}|[A-Z])[.)\s]'
)
_HEADING_SUB_NUM = re.compile(
    r'^(\d{1,3}|[IVXLCDM]{1,4}|[A-Z])\.([A-Za-z0-9])'
)

# Known valid Roman-numeral words used as section prefixes (up to XXV covers
# all realistic paper section counts).  Used to distinguish "I. Introduction"
# (Roman L1) from "A. Data" (letter L2) in IEEE-style numbering detection.
_ROMAN_WORDS: frozenset[str] = frozenset([
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII", "XXIII", "XXIV", "XXV",
])


def _clean_heading(h: str) -> str:
    """Strip leading numbering/label from a heading (e.g. '2.1 Methods' → 'Methods')."""
    h = h.strip()
    cleaned = _NUM_PREFIX.sub('', h).strip()
    return cleaned if cleaned else h   # never return empty


def _extract_doc_headers(document) -> list[str]:
    """
    Extract all section-header texts from a Docling document in document order.

    This captures headings that may not appear in any chunk's metadata — most
    importantly, IEEE Roman-numeral sections (e.g. "II. BACKGROUND") whose
    entire content lives inside lettered sub-sections (A, B, C) so the
    HierarchicalChunker never emits a chunk whose headings[0] is the Roman
    section itself.  Having this list lets Phase 3 of _build_parent_maps
    advance _cur_roman_h past those "invisible" sections.

    Returns an empty list on any failure so callers degrade gracefully.
    """
    headers: list[str] = []
    try:
        for item in getattr(document, "texts", []):
            label = getattr(item, "label", None)
            label_val = (
                getattr(label, "value", str(label)) if label is not None else ""
            )
            if label_val == "section_header":
                txt = getattr(item, "text", "").strip()
                if txt:
                    headers.append(txt)
    except Exception:
        pass
    return headers


def _build_parent_maps(
    chunks: list,
    doc_headers: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, str], bool, list[list[str]]]:
    """
    Detect the paper's section-numbering convention and build parent maps for
    orphan recovery, then return corrected heading lists for every chunk.

    Conventions handled:
      Decimal  — "3. Methodology" / "3.1 Data" / "3.1.2 Sub"
                 _HEADING_SUB_NUM matches "3.1" → parent prefix "3"
      IEEE     — "I. Introduction" / "A. Data" / "1) Word2Vec"
                 Roman numerals = L1, uppercase letters = L2, digits = L3
                 Detected when BOTH roman-numeral AND letter prefixes appear.
      Alphabetic/mixed — "A. Methods" / "A.1 Sub"
                 _HEADING_SUB_NUM matches "A.1" → parent prefix "A"
      Unnumbered — no prefix; Docling handles hierarchy natively, no fix needed.

    `doc_headers` — section-header texts from the Docling document object in
    document order (from _extract_doc_headers).  Supplying this list is the
    key to handling IEEE sections that have no direct chunk content: by
    scanning the ordered list we update _cur_roman_h even when "II. BACKGROUND"
    never appears as headings[0] in any chunk.

    Returns:
        num_prefix_map    — top-level prefix  → raw heading text
        letter_parent_map — letter prefix     → Roman-numeral heading (IEEE only)
        ieee_style        — True if IEEE convention detected
        corrected_headings — per-chunk list[str] with parent injected when needed
    """
    # ── Phase 1: collect every "apparent top-level" prefix ───────────────
    # Process doc_headers first (complete, in document order) then fall back
    # to chunk headings for anything the document-level extraction missed.
    prefix_order: list[tuple[str, str, str]] = []
    seen_h: set[str] = set()

    def _try_add(h0: str) -> None:
        if h0 in seen_h:
            return
        seen_h.add(h0)
        _m_top = _HEADING_TOP_NUM.match(h0)
        _m_sub = _HEADING_SUB_NUM.match(h0)
        if not _m_top or _m_sub:
            return
        _pfx = _m_top.group(1)
        if _pfx.upper() in _ROMAN_WORDS:
            _ptype = "roman"
        elif _pfx.isdigit():
            _ptype = "digit"
        elif len(_pfx) == 1 and _pfx.isupper():
            _ptype = "letter"
        else:
            return
        prefix_order.append((_pfx, _ptype, h0))

    for _h in (doc_headers or []):
        _try_add(_h.strip())
    for _c in chunks:
        _h_list = _c.meta.export_json_dict().get("headings", [])
        if _h_list:
            _try_add(_h_list[0].strip())

    # ── Phase 2: detect convention, build parent maps ────────────────────
    types_seen = {pt for _, pt, _ in prefix_order}
    # IEEE if the paper uses Roman-numeral L1 sections AND letter L2 sections.
    ieee_style = "roman" in types_seen and "letter" in types_seen

    num_prefix_map: dict[str, str] = {}     # top-level prefix → raw heading
    letter_parent_map: dict[str, str] = {}  # letter prefix → Roman parent heading

    if ieee_style:
        cur_roman: str | None = None
        for _pfx, _ptype, _h0 in prefix_order:
            if _ptype == "roman":
                if _pfx not in num_prefix_map:
                    num_prefix_map[_pfx] = _h0
                cur_roman = _h0
            elif _ptype == "letter" and cur_roman:
                if _pfx not in letter_parent_map:
                    letter_parent_map[_pfx] = cur_roman
            # digits in IEEE are L3; they fold into the Roman parent directly
    else:
        for _pfx, _ptype, _h0 in prefix_order:
            if _pfx not in num_prefix_map:
                num_prefix_map[_pfx] = _h0

    # ── Phase 3: produce corrected heading lists for every chunk ─────────
    # Build a position index over ordered doc_headers so we can advance a
    # monotonic pointer and update _cur_roman_h even when a Roman section
    # has no chunk of its own (key fix for IEEE "empty" parent sections).
    _ordered: list[str] = [h.strip() for h in (doc_headers or [])]
    _hdr_pos: dict[str, list[int]] = {}
    for _i, _h in enumerate(_ordered):
        _hdr_pos.setdefault(_h, []).append(_i)

    _cur_roman_h: str | None = None
    _ptr = 0  # advances monotonically through _ordered

    corrected: list[list[str]] = []
    for _c in chunks:
        _raw = [h.strip() for h in _c.meta.export_json_dict().get("headings", [])]

        # ── Update _cur_roman_h ───────────────────────────────────────────
        if _raw:
            _h0 = _raw[0]
            if _ordered:
                # Walk the ordered header list up to (and including) the
                # position of this chunk's first heading.  Any Roman headers
                # encountered along the way update _cur_roman_h — this is
                # what handles IEEE sections with no direct chunk content.
                _cands = [p for p in _hdr_pos.get(_h0, []) if p >= _ptr]
                if _cands:
                    for _idx in range(_ptr, _cands[0] + 1):
                        _dh = _ordered[_idx]
                        _m = _HEADING_TOP_NUM.match(_dh)
                        if _m and _m.group(1).upper() in _ROMAN_WORDS:
                            _cur_roman_h = _dh
                    _ptr = _cands[0] + 1
                else:
                    # Heading not in ordered list — fall back to direct check
                    _m_any = _HEADING_TOP_NUM.match(_h0)
                    if _m_any and not _HEADING_SUB_NUM.match(_h0):
                        if _m_any.group(1).upper() in _ROMAN_WORDS:
                            _cur_roman_h = _h0
            else:
                # No ordered headers available — use direct check (old path)
                _m_any = _HEADING_TOP_NUM.match(_h0)
                if _m_any and not _HEADING_SUB_NUM.match(_h0):
                    if _m_any.group(1).upper() in _ROMAN_WORDS:
                        _cur_roman_h = _h0

        # ── Orphan recovery ───────────────────────────────────────────────
        if len(_raw) == 1:
            _h0 = _raw[0]

            # Case A: dot-separated prefix "3.1 Data", "I.A Sub"
            _m_sub = _HEADING_SUB_NUM.match(_h0)
            if _m_sub:
                _parent_h = num_prefix_map.get(_m_sub.group(1))
                if _parent_h:
                    _raw = [_parent_h, _h0]

            else:
                _m_top = _HEADING_TOP_NUM.match(_h0)
                if _m_top:
                    _pfx = _m_top.group(1)

                    # Case B: IEEE letter subsection "A. Data Collection"
                    # _cur_roman_h is now position-aware via the doc_headers
                    # ordered walk, so this correctly anchors each letter sub
                    # to its actual Roman parent (not just the first-seen one).
                    if (ieee_style
                            and not _pfx.isdigit()
                            and _pfx.upper() not in _ROMAN_WORDS
                            and _cur_roman_h):
                        _raw = [_cur_roman_h, _h0]

                    # Case C: IEEE digit sub-subsection "1) Word2Vec"
                    # Fold directly into the current Roman parent (2-level
                    # model: section_name / subsection_name, not sub-sub).
                    elif ieee_style and _pfx.isdigit() and _cur_roman_h:
                        _raw = [_cur_roman_h, _h0]

        corrected.append(_raw)

    return num_prefix_map, letter_parent_map, ieee_style, corrected


def _extract_page_numbers(meta: dict) -> str:
    pages = set()
    for item in meta.get("doc_items", []):
        for prov in item.get("prov", []):
            if "page_no" in prov:
                pages.add(prov["page_no"])
    return ",".join(str(p) for p in sorted(pages))


# ---------------------------------------------------------------------------
# Three-stage section-type classification (kept for quick-search filtering)
# ---------------------------------------------------------------------------

def _build_section_type_map(
    chunks: list,
    model: str,
    chunk_headings: list[list[str]] | None = None,
) -> dict[str, str]:
    """
    Hierarchy-aware two-pass classification.

    `chunk_headings` — optional pre-corrected heading lists (one per chunk) from
    _build_parent_maps().  When supplied, these are used instead of the raw
    chunk metadata so that classification sees the already-fixed hierarchy.

    Pass 1 — classify every TOP-LEVEL heading (headings[0]) through all 3 stages.
             Subsections under an already-typed parent are never classified;
             they inherit the parent's type via the chunk loop.

    Pass 2 — only for top-level headings still 'general' after Pass 1,
             classify their direct subsections (headings[1]).  This lets an
             unrecognisably-named section ("Our System") be typed through its
             children ("Data Collection" → dataset, "Experiments" → results).
    """
    # Collect top-level headings and subsections grouped by parent.
    top_to_text: dict[str, str] = {}
    top_to_subs: dict[str, dict[str, str]] = {}   # h0 → {h1: first_text}

    for i, chunk in enumerate(chunks):
        if chunk_headings is not None:
            headings = chunk_headings[i]
        else:
            headings = [h.strip() for h in chunk.meta.export_json_dict().get("headings", [])]
        if not headings:
            continue
        h0 = headings[0]
        if h0 and h0 not in top_to_text:
            top_to_text[h0] = chunk.text
        if len(headings) >= 2 and h0:
            subs = top_to_subs.setdefault(h0, {})
            h1 = headings[1]
            if h1 and h1 not in subs:
                subs[h1] = chunk.text

    # ── Pass 1: top-level headings ────────────────────────────────────────
    type_map: dict[str, str] = {h: classify_heading(h) for h in top_to_text}
    print(f"  Stage 1 (keywords, top-level): {dict(sorted(type_map.items()))}")

    ambiguous = [h for h, t in type_map.items() if t == "general"]
    if ambiguous:
        print(f"  Stage 2 (LLM batch): {len(ambiguous)} ambiguous top-level heading(s)...")
        type_map.update(classify_headings_batch(ambiguous, model))

    still_general = [h for h, t in type_map.items() if t == "general"]
    if still_general:
        print(f"  Stage 3 (content verify): {len(still_general)} heading(s)...")
        for h in still_general:
            first_text = top_to_text.get(h, "")
            if first_text:
                verified = verify_with_first_paragraph(h, first_text, model)
                type_map[h] = verified
                print(f"    '{h}' → {verified}")

    # ── Pass 2: subsections of still-'general' parents ────────────────────
    general_parents = [h for h, t in type_map.items() if t == "general"]
    if general_parents:
        sub_list, sub_to_text = [], {}
        for h0 in general_parents:
            for h1, text in top_to_subs.get(h0, {}).items():
                if h1 not in sub_to_text:
                    sub_list.append(h1)
                    sub_to_text[h1] = text
        if sub_list:
            print(f"  Pass 2 (subsections of {len(general_parents)} untyped parent(s)): "
                  f"{len(sub_list)} subsection(s)...")
            sub_map = {h: classify_heading(h) for h in sub_list}
            sub_ambiguous = [h for h, t in sub_map.items() if t == "general"]
            if sub_ambiguous:
                sub_map.update(classify_headings_batch(sub_ambiguous, model))
            type_map.update(sub_map)

    return type_map


# ---------------------------------------------------------------------------
# Per-section summaries
# ---------------------------------------------------------------------------

# How much of a section is shown to the summariser. Sections run long; the
# opening chunks carry what the section is about, which is all routing needs.
_SUMMARY_CONTEXT_CHARS = 3000

_SECTION_SUMMARY_PROMPT = """\
Below are excerpts from the section "{section}" of a research paper.{subs}

Write ONE sentence, at most 40 words, stating what this section actually
contains. Name the concrete things it reports - datasets and their sizes,
methods, instruments, participants, metrics, findings - because this sentence
is used to decide whether the section can answer a question.

Do not editorialise, do not begin with "This section", and do not invent
anything that is not in the excerpts.

Excerpts:
{text}

One sentence:"""


def _summarise_sections(
    chunks_data: list[dict], model: str
) -> dict[str, str]:
    """
    Write a one-line description of what each section actually contains.

    A heading alone is a poor routing signal: "METHODOLOGY" does not reveal that
    the section also states the corpus size, which is where most papers put it.
    The summary makes that visible, so section selection is based on evidence
    rather than on guessing from a title.

    Summaries are routing metadata only - they never reach the answer, so a
    hallucinated one can cost precision but can never corrupt a cited result.
    """
    by_section: dict[str, list[dict]] = {}
    for item in chunks_data:
        props = item["properties"]
        by_section.setdefault(props["section_name"], []).append(props)

    summaries: dict[str, str] = {}
    for section, props in by_section.items():
        props.sort(key=lambda p: p["chunk_index"])
        text = "\n".join(p["text"] for p in props)[:_SUMMARY_CONTEXT_CHARS]
        if not text.strip():
            continue

        seen: set[str] = set()
        subs = [
            p["subsection_name"] for p in props
            if p["subsection_name"] and not (
                p["subsection_name"] in seen or seen.add(p["subsection_name"])
            )
        ]
        sub_line = (
            f"\nIts subsections are: {', '.join(subs[:8])}." if subs else ""
        )

        try:
            summary = _llm(
                _SECTION_SUMMARY_PROMPT.format(
                    section=section, subs=sub_line, text=text
                ),
                model,
            )
        except Exception as exc:      # never let summarisation break an ingest
            print(f"    ! summary failed for {section!r}: {exc}")
            continue
        summaries[section] = " ".join(summary.split())[:400]

    return summaries


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

# Layout parsing sometimes puts spaces around a decimal point, so "29.6%"
# is extracted as "29 . 6%". It is rare - 0.4% of chunks in a 54-paper
# corpus - but it lands squarely on the figures people ask about: accuracies,
# perplexities, F1 scores. An exact-match search for the real number then
# misses, and any answer quoting it looks broken.
_SPLIT_DECIMAL = re.compile(r"(?<=\d)\s+\.\s+(?=\d)")


def _repair_numbers(text: str) -> str:
    """Rejoin decimals that layout parsing split apart."""
    return _SPLIT_DECIMAL.sub(".", text or "")


class _Clock:
    """
    Optional stage timer.

    `ingest_pdf(..., timings={})` fills the dict with seconds per stage. When
    no dict is passed every call is a no-op, so the instrumentation costs
    nothing on the normal path.
    """

    __slots__ = ("into", "_open")

    def __init__(self, into: dict | None):
        self.into = into
        self._open: dict = {}

    def start(self, stage: str):
        if self.into is not None:
            self._open[stage] = time.perf_counter()

    def stop(self, stage: str):
        if self.into is not None and stage in self._open:
            self.into[stage] = self.into.get(stage, 0.0) + (
                time.perf_counter() - self._open.pop(stage)
            )


def ingest_pdf(
    pdf_path: str,
    embedder: Embedder,
    store: VectorStore,
    skip_existing: bool = True,
    model: str = OLLAMA_MODEL,
    fast: bool = False,
    card_store=None,
    timings: dict | None = None,
) -> int:
    """
    Convert, chunk, classify, embed and store one PDF.

    `fast=True` skips both LLM passes - the multi-stage section-type
    classification and the per-section summaries - and keeps only the instant
    keyword classifier. That is the difference between roughly six seconds and
    a couple of minutes, which is what makes "upload and start asking" viable.
    Section-aware routing is weaker for such a document, but a single-document
    conversation is already scoped to one paper, so it leans on plain semantic
    search anyway.
    """
    pdf_path = Path(pdf_path)
    clock = _Clock(timings)

    if store.source_exists(pdf_path.name):
        if skip_existing:
            print(f"  Skipping (already ingested): {pdf_path.name}")
            return 0
        # Re-ingesting: clear the old chunks first. Chunk boundaries can shift
        # between Docling versions, so overwriting by ID alone would strand
        # orphaned chunks from the previous run.
        print(f"  Re-ingesting: clearing existing chunks for {pdf_path.name}")
        store.delete_source(pdf_path.name)

    print(f"Converting: {pdf_path.name}")
    clock.start("parse")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    chunker = HierarchicalChunker()
    chunks = list(chunker.chunk(result.document))
    clock.stop("parse")
    print(f"  → {len(chunks)} chunks extracted")

    # ── Orphan recovery: detect convention, build parent maps, fix headings ─
    # Extract ALL section headers from the Docling document object first.
    # This gives us Roman-numeral IEEE sections (e.g. "II. BACKGROUND") that
    # have no direct chunk content because all their text is inside A/B/C
    # sub-sections — the HierarchicalChunker never emits a chunk with that
    # heading as headings[0], so without this list _cur_roman_h would stall
    # at the previous section and misattribute every subsection after it.
    doc_headers = _extract_doc_headers(result.document)
    _, _, ieee_style, corrected_headings = _build_parent_maps(chunks, doc_headers)
    if ieee_style:
        print("  → IEEE-style numbering detected (Roman L1 / letter L2)")

    # Three-stage section_type map using corrected headings. The fast path keeps
    # only stage 1, which needs no LLM at all.
    clock.start("classify")
    if fast:
        section_type_map = {
            h.strip(): classify_heading(h)
            for hs in corrected_headings for h in hs if h.strip()
        }
        print(f"  -> Fast path: keyword-only types for {len(section_type_map)} heading(s)")
    else:
        section_type_map = _build_section_type_map(chunks, model, corrected_headings)
    clock.stop("classify")

    clock.start("embed")
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed(texts)
    clock.stop("embed")
    print(f"  → Embeddings computed")

    chunks_data = []
    skipped = 0
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Use pre-corrected headings (orphan parents already injected)
        headings: list[str] = corrected_headings[i]

        # ── Structural fields (authoritative, no LLM needed) ──────────────
        section_name = _clean_heading(headings[0]) if len(headings) >= 1 else ""
        subsection_name = _clean_heading(headings[1]) if len(headings) >= 2 else ""

        # Skip boilerplate sections that add no research content
        if _is_skippable(section_name):
            skipped += 1
            continue

        # Docling files everything before the first real heading - abstract,
        # authors, index terms - under the paper title. That used to be treated
        # as boilerplate and dropped, which silently discarded the abstract of
        # every paper with a title longer than the heuristic's threshold. The
        # abstract is the densest section in a paper, so relabel this front
        # matter instead of discarding it.
        if _looks_like_paper_title(section_name):
            section_name = FRONT_MATTER_SECTION
            headings = [FRONT_MATTER_SECTION] + headings[1:]
            front_matter = True
        else:
            front_matter = False

        heading_str = " > ".join(headings) if headings else ""

        # ── section_type: top-level section wins; subsections can only refine
        # a "general" parent, never override a specific one.
        # Example: "3 Methodology → 3.1 Related Techniques" stays "methodology"
        # because the parent is already specific.  But "General → 3.1 Related Work"
        # correctly becomes "related_work" because the parent is "general".
        if front_matter:
            section_type = "abstract"
        else:
            section_type = "general"
            for h in headings:                      # coarse → fine (top → subsection)
                t = section_type_map.get(h.strip(), "general")
                if t != "general":
                    section_type = t
                    break                           # first non-general heading wins

        chunks_data.append({
            "properties": {
                "text": _repair_numbers(chunk.text),
                "source_file": pdf_path.name,
                "source_path": str(pdf_path.resolve()),
                "chunk_index": i,
                "heading": heading_str,
                "section_name": section_name,
                "subsection_name": subsection_name,
                "section_type": section_type,
                "page_numbers": _extract_page_numbers(chunk.meta.export_json_dict()),
                "section_summary": "",
            },
            "vector": vector,
        })

    # One-line description of every section, used later to route a question to
    # the right section instead of guessing from the heading alone.
    if fast:
        section_summaries = {}
    else:
        clock.start("summarise")
        section_summaries = _summarise_sections(chunks_data, model)
        clock.stop("summarise")
        for item in chunks_data:
            props = item["properties"]
            props["section_summary"] = section_summaries.get(props["section_name"], "")
        print(f"  -> Summarised {len(section_summaries)} section(s)")

    clock.start("write")
    store.insert_chunks(chunks_data)
    clock.stop("write")

    # A structured card for the paper as a whole. Chunks answer questions whose
    # answer sits in a passage; the card answers questions about the paper, which
    # no passage states. Skipped on the fast path along with the other LLM work.
    if card_store is not None and not fast:
        from .paper_cards import build_card
        from .titles import extract_title
        clock.start("card")
        lead = [c["properties"]["text"] for c in chunks_data[:14]]
        card = build_card(pdf_path.name, lead, model)
        # What the paper is called, so the interface can stop showing whatever
        # the publisher named the file. Usually free: most PDFs declare it.
        card.title, how = extract_title(str(pdf_path), model)
        card_store.upsert(card, embedder.embed_one(card.embedding_text() or pdf_path.name))
        print(f"  -> Card: {card.discipline or 'unclassified'}")
        clock.stop("card")
        if card.title:
            print(f"  -> Title ({how}): {card.title[:70]}")

    dist = Counter(c["properties"]["section_name"] for c in chunks_data)
    print(f"  ✓ Stored {len(chunks_data)} chunks  (skipped {skipped} from boilerplate sections)")
    print(f"    Section distribution: {dict(sorted(dist.items()))}")
    return len(chunks_data)


def ingest_folder(
    folder_path: str,
    embedder: Embedder,
    store: VectorStore,
    skip_existing: bool = True,
    model: str = OLLAMA_MODEL,
    card_store=None,
) -> int:
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {folder}")
        return 0
    print(f"Found {len(pdfs)} PDF(s) in {folder}\n")
    total = 0
    for pdf in pdfs:
        total += ingest_pdf(str(pdf), embedder, store, skip_existing, model,
                            card_store=card_store)
    print(f"\n✓ Done: {total} total chunks from {len(pdfs)} PDF(s)")
    return total
