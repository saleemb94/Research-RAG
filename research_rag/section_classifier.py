"""
Three-stage section classification:

Stage 1  classify_heading()             — keyword match, instant, no LLM
Stage 2  classify_headings_batch()      — LLM batch call for anything still "general"
Stage 3  verify_with_first_paragraph()  — LLM reads first paragraph to confirm/correct
"""
from __future__ import annotations

import re

import ollama

SECTION_TYPES = [
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "dataset",
    "results",
    "conclusion",
    "general",
]

# Aliases: normalise common LLM variations → canonical type
_ALIASES: dict[str, str] = {
    # related_work — prior research by others on the same problem
    "related_works": "related_work",
    "related work": "related_work",
    "related works": "related_work",
    "literature review": "related_work",
    "literature survey": "related_work",
    "literature study": "related_work",
    "prior work": "related_work",
    "prior works": "related_work",
    "previous work": "related_work",
    "previous works": "related_work",
    "survey": "related_work",
    "state of the art": "related_work",
    "existing work": "related_work",
    "existing works": "related_work",
    "existing approaches": "related_work",
    "existing methods": "related_work",
    "comparison with related": "related_work",
    # note: "background study" removed — "background" alone → introduction (below)
    # methodology
    "methodology": "methodology",
    "methods": "methodology",
    "method": "methodology",
    "approach": "methodology",
    "proposed method": "methodology",
    "proposed approach": "methodology",
    "proposed framework": "methodology",
    "proposed model": "methodology",
    "our approach": "methodology",
    "our method": "methodology",
    "our model": "methodology",
    "our framework": "methodology",
    "system design": "methodology",
    "system architecture": "methodology",
    "model architecture": "methodology",
    "model": "methodology",
    "framework": "methodology",
    "algorithm": "methodology",
    "implementation": "methodology",
    "experimental design": "methodology",
    "study design": "methodology",
    "study protocol": "methodology",
    "materials and methods": "methodology",
    "training procedure": "methodology",
    # dataset
    "data": "dataset",
    "datasets": "dataset",
    "corpus": "dataset",
    "experimental setup": "dataset",
    "data collection": "dataset",
    "data preprocessing": "dataset",
    "preprocessing": "dataset",
    "data preparation": "dataset",
    "annotation": "dataset",
    "labeling": "dataset",
    "ground truth": "dataset",
    "benchmark": "dataset",
    "training data": "dataset",
    "materials": "dataset",
    "sampling": "dataset",
    # results
    "experiment": "results",
    "experiments": "results",
    "evaluation": "results",
    "experimental results": "results",
    "findings": "results",
    "performance": "results",
    "performance evaluation": "results",
    "simulation results": "results",
    "ablation study": "results",
    "ablation": "results",
    "case study": "results",
    "user study": "results",
    "error analysis": "results",
    "qualitative analysis": "results",
    "quantitative analysis": "results",
    "comparative study": "results",
    # conclusion
    "conclusions": "conclusion",
    "discussion": "conclusion",
    "general discussion": "conclusion",
    "future work": "conclusion",
    "future directions": "conclusion",
    "future prospects": "conclusion",
    "limitations": "conclusion",
    "summary": "conclusion",
    "concluding remarks": "conclusion",
    "implications": "conclusion",
    "conclusion and future work": "conclusion",
    # introduction
    "background": "introduction",
    "motivation": "introduction",
    "overview": "introduction",
    "problem statement": "introduction",
    "context": "introduction",
    "challenge": "introduction",
    "problem formulation": "introduction",
    "research objective": "introduction",
    "contributions": "introduction",
    "preliminaries": "introduction",
    "foundations": "introduction",
    "technical background": "introduction",
    "basic concepts": "introduction",
}


def _normalize(raw: str) -> str:
    """
    Normalise raw LLM text to a valid SECTION_TYPES entry.
    Handles spaces, hyphens, underscores, common aliases, and plurals.
    Returns "" if no match found.
    """
    text = raw.strip().lower().rstrip(".")
    # Try direct alias first (handles multi-word forms like "related work")
    if text in _ALIASES:
        return _ALIASES[text]
    # Collapse whitespace/hyphens to underscore then check again
    slugged = re.sub(r"[\s\-]+", "_", text)
    if slugged in SECTION_TYPES:
        return slugged
    if slugged in _ALIASES:
        return _ALIASES[slugged]
    # Partial prefix match as last resort (e.g. "methodolog" → "methodology")
    for st in SECTION_TYPES:
        if st.startswith(slugged) or slugged.startswith(st):
            return st
    return ""


# ── Stage 1 ────────────────────────────────────────────────────────────────

# Strips ONLY a complete leading numbering prefix such as "1 ", "1. ", "1.2 ",
# "III. ", "A. " — never strips individual letters that are part of a word.
_HEADING_NUM_PREFIX = re.compile(
    r'^(?:'
    r'(?:\d+\.?)+|'          # 1  /  1.  /  1.2  /  1.2.3
    r'[IVXivxLCDM]+\.?|'    # III  /  iv.
    r'[A-Z]\.'               # A.
    r')\s+'                  # must be followed by whitespace so we don't eat word-start chars
)

_HEADING_KEYWORDS: list[tuple[str, list[str]]] = [
    # Check related_work BEFORE methodology so "Related Techniques" doesn't hit "technique"
    ("abstract",     ["abstract"]),
    ("related_work", [
        "related work", "related study", "related studies",
        "literature review", "literature survey", "literature study",
        "prior work", "previous work", "state of the art",
        "background and related", "survey of", "review of",
        "existing work", "existing approach", "existing method",
        "comparison with", "related technique", "surveys",
        "prior art", "prior research",
    ]),
    ("methodology",  [
        "method", "methodology", "approach", "algorithm",
        "framework", "proposed", "architecture",
        "procedure", "system design", "system architecture",
        "model architecture", "implementation",
        "materials and method", "our model", "our approach",
        "model design", "system overview",
        "experimental design", "study design", "study protocol",
        "task design", "survey design", "training procedure",
        "feature extraction", "preprocessing pipeline",
        "apparatus", "instrument",
    ]),
    ("dataset",      [
        "dataset", "data collection", "data description",
        "corpus", "benchmark", "experimental setup",
        "experimental data", "data set", "training data",
        "data acquisition", "annotation", "labeling",
        "ground truth", "data preprocessing", "preprocessing",
        "data preparation", "sampling", "data source",
        "training set", "test set", "validation set",
        "materials", "participants", "study population",
    ]),
    ("results",      [
        "result", "experiment", "evaluation", "performance",
        "finding", "accuracy", "ablation",
        "analysis", "empirical", "quantitative", "qualitative",
        "application", "downstream",
        "case study", "user study", "simulation",
        "comparative", "error analysis",
        "performance evaluation", "model performance",
        "baseline", "benchmark result",
    ]),
    ("introduction", [
        "introduction", "motivation", "overview",
        "problem statement", "background", "preliminar",
        "foundations", "technical background", "basic concepts",
        "prerequisite", "context", "challenge",
        "problem formulation", "research question",
        "research objective", "contribution",
    ]),
    ("conclusion",   [
        "conclusion", "concluding", "discussion",
        "future work", "future direction", "future prospect",
        "limitation", "summary", "implication",
        "recommendation", "general discussion",
    ]),
]


def classify_heading(heading: str) -> str:
    """Keyword-based heading → canonical type. Fast, no LLM."""
    if not heading:
        return "general"
    # Strip only the numeric/roman-numeral prefix, never word characters
    clean = _HEADING_NUM_PREFIX.sub("", heading.strip()).strip().lower()
    if not clean:
        clean = heading.strip().lower()
    for section_type, keywords in _HEADING_KEYWORDS:
        for kw in keywords:
            if kw in clean:
                return section_type
    return "general"


# ── Stage 2 ────────────────────────────────────────────────────────────────

_BATCH_PROMPT = """\
Classify each section heading from a research paper.

Use ONLY these exact labels (copy them exactly, with underscores):
  abstract, introduction, related_work, methodology, dataset, results, conclusion, general

Guidelines:
  abstract      = abstract, extended abstract, executive summary
  introduction  = introduction, motivation, problem statement, overview, context, challenge,
                  AND any "Background" or "Preliminaries" section that explains foundational
                  technologies or concepts (e.g. "Background", "Preliminaries", "Foundations",
                  "Technical Background", "Basic Concepts")
  related_work  = sections that survey or compare PRIOR RESEARCH by other authors
                  (e.g. "Related Work", "Literature Review", "Prior Work", "Survey of Literature",
                  "State of the Art", "Existing Approaches")
  methodology   = how the research was designed/implemented: methods, approach, proposed system,
                  framework, model, algorithm, implementation, experimental design, study design,
                  system architecture, training procedure, apparatus
  dataset       = data, corpus, benchmark, experimental setup, data collection, preprocessing,
                  annotation, ground truth, participants, study population, materials
  results       = experiments, evaluation, performance, findings, ablation, analysis,
                  case study, user study, simulation, error analysis, model performance
  conclusion    = conclusions, discussion, summary, future work, future directions, limitations,
                  implications, recommendations, concluding remarks
  general       = only if none of the above apply

Critical distinction — Background vs Related Work:
  "Background" / "Preliminaries" / "Foundations" explain underlying technologies and concepts
  needed to understand the paper  →  label as introduction
  "Related Work" / "Literature Review" / "Prior Work" discuss what previous researchers
  have done on the same problem   →  label as related_work
  These are DIFFERENT sections. Never confuse them.

Headings to classify:
{numbered_headings}

Reply with ONLY numbered lines, one per heading, in this exact format:
1. label
2. label
(Use underscores, e.g. "related_work" not "related work")"""


def classify_headings_batch(headings: list[str], model: str) -> dict[str, str]:
    """LLM batch-classifies a list of headings. Returns {heading: section_type}."""
    if not headings:
        return {}
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headings))
    prompt = _BATCH_PROMPT.format(numbered_headings=numbered)
    response = ollama.generate(model=model, prompt=prompt)
    raw = response.response.strip()

    result: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        # Match "1. related_work" or "1. related work" or "1) related_work"
        m = re.match(r"^(\d+)[.)]\s*(.+)$", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        label = _normalize(m.group(2))
        if label and 0 <= idx < len(headings):
            result[headings[idx]] = label

    # Fill any headings the LLM missed with "general"
    for h in headings:
        if h not in result:
            result[h] = "general"
    return result


# ── Stage 3 ────────────────────────────────────────────────────────────────

_VERIFY_PROMPT = """\
A research paper section is titled "{heading}".
Its opening paragraph reads:

\"\"\"{text}\"\"\"

Based on the CONTENT (not just the title), which of these labels best describes \
what this section is actually about?

  abstract, introduction, related_work, methodology, dataset, results, conclusion, general

Use underscores (e.g. "related_work" not "related work").
Reply with ONLY the single label — nothing else."""


def verify_with_first_paragraph(heading: str, first_paragraph: str, model: str) -> str:
    """Use first paragraph content to confirm or correct a section classification."""
    prompt = _VERIFY_PROMPT.format(heading=heading, text=first_paragraph[:800])
    response = ollama.generate(model=model, prompt=prompt)
    # Take the first non-empty line of the response
    for line in response.response.strip().splitlines():
        label = _normalize(line.strip())
        if label:
            return label
    return "general"


# ── Two-step heading selection ─────────────────────────────────────────────

_IDENTIFY_SECTION_PROMPT = """\
A user is asking this question about a research paper:
"{query}"

What single section of a research paper would contain the answer?
Reply with ONE short phrase — just the section name (e.g. "related work", \
"methodology", "experimental results", "dataset", "conclusion").
No explanation. No punctuation at the end. Just the section name."""


def identify_target_section(query: str, model: str) -> str:
    """
    Step 1 — Ask the LLM what section of a paper the query is about.
    Returns a single natural-language phrase, e.g. "related work".
    """
    prompt = _IDENTIFY_SECTION_PROMPT.format(query=query)
    response = ollama.generate(model=model, prompt=prompt)
    # Take the first non-empty line, lowercase, strip trailing punctuation
    for line in response.response.strip().splitlines():
        phrase = line.strip().lower().rstrip(".")
        if phrase:
            return phrase
    return "general"


_MATCH_HEADINGS_PROMPT = """\
You are matching top-level section headings from a research paper to a target section.

Target section: "{target}"

Top-level sections (numbered for selection) with their subsections shown for context:
{numbered_headings}

Rules:
- Choose only TOP-LEVEL section numbers (e.g. "3" or "3, 5").
  Selecting a top-level section automatically includes all its listed subsections —
  do NOT try to select individual subsections.
- Include a section only if it genuinely belongs to the target (or is a known alias):
    "related work"  ↔  "Literature Review", "Prior Work", "Related Studies", "Survey"
    "methodology"   ↔  "Our Approach", "Proposed Method", "System Design", "Framework"
    "introduction"  ↔  "Background", "Preliminaries", "Foundations", "Technical Background"
- If no section matches, reply with: none

Critical distinction — never confuse these two:
  "Background" / "Preliminaries" / "Foundations" explain foundational technologies and
  concepts needed to understand the paper → INTRODUCTION family, NOT related work.
  "Related Work" / "Literature Review" / "Prior Work" survey previous research by
  other authors on the same problem → RELATED WORK family, NOT background.
  A subsection named "Related Work" inside a "Background" or "Methodology" section
  does NOT make the whole parent section a "related work" section.

Reply with ONLY the top-level section numbers, comma-separated (e.g. "3, 4").
No explanation. No labels. Just the numbers (or "none")."""


def match_headings_to_target(
    target_section: str,
    headings: list[str],
    model: str,
    section_summary: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    LLM picks which top-level sections of the paper match the target.

    `section_summary` — optional {section_name: [subsection_name, ...]} mapping from
    VectorStore.get_section_summary().  When supplied, each section is shown with its
    subsections so the LLM understands the hierarchy and avoids confusing a subsection
    heading with a top-level section of the same name.
    """
    if not headings:
        return []

    lines: list[str] = []
    for i, h in enumerate(headings):
        lines.append(f"{i + 1}. {h}")
        if section_summary:
            subs = section_summary.get(h, [])
            for s in subs[:6]:
                lines.append(f"   - {s}")
            if len(subs) > 6:
                lines.append(f"   - … ({len(subs) - 6} more subsections)")

    numbered = "\n".join(lines)
    prompt = _MATCH_HEADINGS_PROMPT.format(
        target=target_section,
        numbered_headings=numbered,
    )
    response = ollama.generate(model=model, prompt=prompt)
    raw = response.response.strip().lower()

    if "none" in raw:
        return []

    matched = []
    for part in re.split(r"[,\s]+", raw):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(headings) and headings[idx] not in matched:
                matched.append(headings[idx])
    return matched


# ── Query classification ───────────────────────────────────────────────────

_QUERY_PROMPT = """\
You are helping route a research question to the right section of a paper.

Task: identify which section(s) of a research paper you would search IN to find \
the answer. Focus on WHERE the answer lives, not on what topics the question mentions.

Critical rule — "topic IN section" queries:
  If the question says "in the related work", "from the methodology", "in the \
literature review", "in the background", "in the experiments", etc., that named \
section IS the target. Return ONLY that section, not the topic alongside it.
  Examples:
    "What methods are mentioned in the related work?"  → related_work
    "What technologies are explained in the background?" → introduction
    "What datasets are used in the methodology?"       → methodology
    "What results are in the conclusion?"              → conclusion

Use ONLY these exact labels (with underscores):
  abstract, introduction, related_work, methodology, dataset, results, conclusion, general

  introduction  = introduction, motivation, problem statement, overview, AND any
                  "Background" or "Preliminaries" section that explains foundational
                  technologies or concepts (NOT prior research by others)
  related_work  = literature review, prior research by other authors, existing work
                  comparison, surveys of past work (NOT background concepts)
  methodology   = how the research was done, methods, algorithms, models, framework
  dataset       = data, corpus, benchmarks, preprocessing
  results       = experiments, evaluation, metrics, performance, findings
  conclusion    = conclusions, limitations, future directions, discussion
  general       = genuinely spans multiple sections with no named section in the query

Critical distinction — background vs related work:
  "background" / "preliminaries" / "foundations" describe technologies and concepts
  needed to understand the paper  →  introduction
  "related work" / "literature review" / "prior work" compare previous research
  by other authors on the same problem  →  related_work

Question: {query}

Reply with ONLY a comma-separated list of 1 or 2 labels. Use underscores. No explanation."""


def classify_query(query: str, model: str) -> list[str]:
    """LLM maps a user question to target section type(s)."""
    prompt = _QUERY_PROMPT.format(query=query)
    response = ollama.generate(model=model, prompt=prompt)
    raw = response.response.strip()
    valid = []
    for part in re.split(r"[,\n]", raw):
        # Strip list markers the LLM might add: "1. ", "- ", "• ", etc.
        clean = re.sub(r"^[\d.)\-•\s]+", "", part.strip())
        label = _normalize(clean)
        if label and label not in valid:
            valid.append(label)
    return valid if valid else ["general"]
