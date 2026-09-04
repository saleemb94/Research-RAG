"""
Discipline-agnostic section classification.

Three stages:

Stage 1  classify_heading()             - keyword match, instant, no LLM
Stage 2  classify_headings_batch()      - LLM batch call for anything still "general"
Stage 3  verify_with_first_paragraph()  - LLM reads first paragraph to confirm/correct

Design note - why the vocabulary is deliberately incomplete
-----------------------------------------------------------
Section *function* is near-universal across academic fields (an IMRaD-shaped paper
looks structurally similar in oncology and in machine learning), but section
*vocabulary* is not. "Corpus" means a dataset in linguistics; "cohort" means the
same thing in epidemiology; "sources" means it in history.

Some words also flip meaning between fields. "Statistical Analysis" is part of the
methods in a clinical paper but reports findings in a CS paper; "Survey" is a
literature review in computer science and a questionnaire instrument in sociology;
"Baseline Characteristics" describes participants in a trial, not a comparison
result. Guessing on those is worse than not guessing, because a confident wrong
answer at Stage 1 prevents Stages 2 and 3 from ever reading the section's content.

So Stage 1 fires only on terms that mean the same thing everywhere. Genuinely
ambiguous headings are listed in _DEFER_TO_CONTENT and deliberately returned as
"general", which routes them to the LLM stages that can read the actual text.
"""
from __future__ import annotations

import re

from .llm import generate as _llm
from .llm import temperature_for

SECTION_TYPES = [
    "abstract",
    "introduction",
    "theory",
    "related_work",
    "methodology",
    "dataset",
    "results",
    "conclusion",
    "general",
]

# The prefix fallback in _normalize() needs a floor, or a stray "t" from a chatty
# model would resolve to "theory".
_MIN_PREFIX_MATCH = 4

# Section types that routinely carry each other's content.
#
# Retrieval cross-validates an LLM's heading choice against the section's
# ingest-time type and drops the match when they disagree. Applied strictly that
# veto discards correct answers, because papers do not partition content the way
# the label set does: most describe their corpus *inside* the methods section
# (APA style literally files "Participants" under Method), and a Discussion
# routinely restates the numbers from Results.
#
# Deliberately excluded: introduction <-> related_work. Background explaining
# concepts and a survey of other people's work are genuinely different sections,
# that confusion is the most common one in practice, and the prompts and the
# retrieval design both go out of their way to keep them apart.
_COMPATIBLE_SECTION_TYPES: dict[str, frozenset[str]] = {
    "dataset": frozenset({"methodology"}),
    "methodology": frozenset({"dataset", "theory"}),
    "theory": frozenset({"methodology", "introduction"}),
    "introduction": frozenset({"theory"}),
    "results": frozenset({"conclusion"}),
    "conclusion": frozenset({"results"}),
}

# An abstract states the paper's headline dataset, method and results at once, so
# it can legitimately answer a query aimed at any of them. "general" means the
# classifier never made a call, which is not evidence of a contradiction.
_WILDCARD_SECTION_TYPES = frozenset({"general", "abstract"})


def section_type_matches(
    stored_type: str | None, target_type: str | None, strict: bool = False
) -> bool:
    """
    True when a chunk stored under `stored_type` may answer a `target_type` query.

    `strict=True` accepts only an exact label (plus the wildcards); `strict=False`
    additionally accepts the adjacent types above.

    Callers should try strict first and widen only when it yields nothing - see
    prefer_exact_types(). Widening unconditionally is actively harmful: a paper
    with a real "Autism screening data for toddlers" section then also matches
    its "Proposed methodology" section on a dataset query, and the genuine
    dataset section gets crowded out of the top k.
    """
    if not target_type or target_type == "general":
        return True
    stored = stored_type or "general"
    if stored == target_type or stored in _WILDCARD_SECTION_TYPES:
        return True
    if strict:
        return False
    return stored in _COMPATIBLE_SECTION_TYPES.get(target_type, frozenset())


def prefer_exact_types(items, type_of, target_type):
    """
    Keep the items whose section_type matches `target_type` exactly, and only
    widen to adjacent types when that leaves nothing.

    Precision where the paper labels the section the way the query asks for it,
    recall where it does not - most papers describe their corpus inside the
    methods section and have no `dataset` section at all.
    """
    if not target_type or target_type == "general":
        return list(items)
    exact = [i for i in items if section_type_matches(type_of(i), target_type, strict=True)]
    if exact:
        return exact
    return [i for i in items if section_type_matches(type_of(i), target_type)]

# Aliases: normalize common LLM variations -> canonical type.
# Grouped by discipline where a term is field-specific.
_ALIASES: dict[str, str] = {
    # ── related_work - prior research by others on the same problem ────────
    "related_works": "related_work",
    "related work": "related_work",
    "related works": "related_work",
    "related studies": "related_work",
    "related research": "related_work",
    "literature review": "related_work",
    "literature survey": "related_work",
    "literature study": "related_work",
    "review of literature": "related_work",
    "prior work": "related_work",
    "prior works": "related_work",
    "prior art": "related_work",
    "previous work": "related_work",
    "previous works": "related_work",
    "previous studies": "related_work",
    "state of the art": "related_work",
    "existing work": "related_work",
    "existing works": "related_work",
    "existing approaches": "related_work",
    "existing methods": "related_work",
    "comparison with related": "related_work",
    # humanities / social science
    "historiography": "related_work",
    "scholarly context": "related_work",
    "systematic review": "related_work",
    "scoping review": "related_work",
    "meta-analysis": "related_work",
    # note: "background study" removed - "background" alone → introduction (below)

    # ── theory - formal or conceptual development ─────────────────────────
    "theory": "theory",
    "theoretical": "theory",
    "theoretical framework": "theory",
    "theoretical background": "theory",
    "theoretical model": "theory",
    "theoretical analysis": "theory",
    "theoretical foundation": "theory",
    "theoretical foundations": "theory",
    "conceptual framework": "theory",
    "conceptual model": "theory",
    "formal model": "theory",
    "mathematical model": "theory",
    "mathematical formulation": "theory",
    "model formulation": "theory",
    "problem definition": "theory",
    "definitions": "theory",
    "theorem": "theory",
    "theorems": "theory",
    "lemma": "theory",
    "proof": "theory",
    "proofs": "theory",
    "derivation": "theory",
    "governing equations": "theory",
    "hypotheses": "theory",
    "hypothesis development": "theory",
    "hypotheses development": "theory",
    "complexity analysis": "theory",
    "convergence analysis": "theory",

    # ── methodology - how THIS study was carried out ──────────────────────
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
    "algorithm": "methodology",
    "implementation": "methodology",
    "experimental design": "methodology",
    "study design": "methodology",
    "study protocol": "methodology",
    "research design": "methodology",
    "materials and methods": "methodology",
    "training procedure": "methodology",
    # clinical / life sciences
    "trial design": "methodology",
    "trial protocol": "methodology",
    "randomization": "methodology",
    "randomisation": "methodology",
    "intervention": "methodology",
    "interventions": "methodology",
    "outcome measures": "methodology",
    "statistical analysis": "methodology",
    "statistical methods": "methodology",
    "assay": "methodology",
    "synthesis": "methodology",
    "sample preparation": "methodology",
    "surgical technique": "methodology",
    # social science / humanities
    "analytic strategy": "methodology",
    "analytical strategy": "methodology",
    "data analysis": "methodology",
    "coding procedure": "methodology",
    "fieldwork": "methodology",
    "ethnographic methods": "methodology",
    # economics
    "empirical strategy": "methodology",
    "identification strategy": "methodology",
    "estimation strategy": "methodology",
    "model specification": "methodology",
    "econometric model": "methodology",
    # physical sciences / engineering
    "experimental apparatus": "methodology",
    "apparatus": "methodology",
    "instrumentation": "methodology",
    "numerical method": "methodology",
    "numerical methods": "methodology",
    "simulation setup": "methodology",
    "boundary conditions": "methodology",

    # ── dataset - what the study was performed ON ─────────────────────────
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
    "sampling": "dataset",
    # clinical / social science
    "participants": "dataset",
    "subjects": "dataset",
    "patients": "dataset",
    "cohort": "dataset",
    "study population": "dataset",
    "study sample": "dataset",
    "inclusion criteria": "dataset",
    "exclusion criteria": "dataset",
    "eligibility criteria": "dataset",
    "baseline characteristics": "dataset",
    "recruitment": "dataset",
    # life sciences
    "specimens": "dataset",
    "reagents": "dataset",
    "cell lines": "dataset",
    # humanities / archival
    "sources": "dataset",
    "primary sources": "dataset",
    "archival sources": "dataset",
    "descriptive statistics": "dataset",
    "summary statistics": "dataset",

    # ── results - what was observed or measured ───────────────────────────
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
    "error analysis": "results",
    "qualitative analysis": "results",
    "quantitative analysis": "results",
    "comparative study": "results",
    # cross-field
    "observations": "results",
    "outcomes": "results",
    "empirical results": "results",
    "estimation results": "results",
    "robustness checks": "results",
    "sensitivity analysis": "results",
    "adverse events": "results",
    "validation": "results",

    # ── conclusion ────────────────────────────────────────────────────────
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
    "recommendations": "conclusion",
    "policy implications": "conclusion",
    "clinical implications": "conclusion",
    "practical implications": "conclusion",

    # ── introduction ──────────────────────────────────────────────────────
    "background": "introduction",
    "motivation": "introduction",
    "overview": "introduction",
    "problem statement": "introduction",
    "context": "introduction",
    "challenge": "introduction",
    "problem formulation": "introduction",
    "research objective": "introduction",
    "research objectives": "introduction",
    "research questions": "introduction",
    "aims": "introduction",
    "contributions": "introduction",
    "preliminaries": "introduction",
    "foundations": "introduction",
    "technical background": "introduction",
    "basic concepts": "introduction",
    "clinical context": "introduction",
    "notation": "introduction",
}


def _normalize(raw: str) -> str:
    """
    Normalize raw LLM text to a valid SECTION_TYPES entry.
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
    # Partial prefix match as last resort (e.g. "methodolog" -> "methodology").
    # Requires a few characters so a single stray letter cannot match a label.
    if len(slugged) >= _MIN_PREFIX_MATCH:
        for st in SECTION_TYPES:
            if st.startswith(slugged) or slugged.startswith(st):
                return st
    return ""


# ── Stage 1 ────────────────────────────────────────────────────────────────

# Strips ONLY a complete leading numbering prefix such as "1 ", "1. ", "1.2 ",
# "III. ", "A. " - never strips individual letters that are part of a word.
_HEADING_NUM_PREFIX = re.compile(
    r'^(?:'
    r'(?:\d+\.?)+|'          # 1  /  1.  /  1.2  /  1.2.3
    r'[IVXivxLCDM]+\.?|'    # III  /  iv.
    r'[A-Z]\.'               # A.
    r')\s+'                  # must be followed by whitespace so we don't eat word-start chars
)

# Headings whose meaning genuinely flips between disciplines. Stage 1 refuses to
# guess on these and returns "general", which hands them to the LLM stages that
# can read the section body. Matched against the whole cleaned heading, so
# "Experimental Setup" defers while "Experimental Setup and Apparatus" does not.
_DEFER_TO_CONTENT: frozenset[str] = frozenset({
    # analysis plan (methods) vs reported analysis (results)
    "analysis",
    "analyses",
    "data analysis",
    "statistical analysis",
    # literature survey (related work) vs questionnaire instrument (methods)
    "survey",
    "surveys",
    # reagents/stimuli (dataset) vs "Materials and Methods" (methodology)
    "materials",
    # formal model (theory) vs implemented system (methodology)
    "model",
    "models",
    "framework",
    "modeling",
    "modeling",
    # apparatus (methods) vs data + hyperparameters (dataset)
    "experimental setup",
    "setup",
    "experimental procedure",
    # research strategy (methods) vs reported case (results)
    "case study",
    "case studies",
    # instrument (methods) vs outcome (results)
    "measures",
    "measurement",
    "measurements",
    "instruments",
    "procedure",
    "procedures",
    "protocol",
    # downstream use (results) vs worked example (theory)
    "application",
    "applications",
    # discipline-dependent: a design section, or the artefact itself
    "design",
    "architecture",
})

_HEADING_KEYWORDS: list[tuple[str, list[str]]] = [
    # Order is priority. Rationale for the sequence:
    #   theory        before methodology  - "Theoretical Framework" is not "framework"
    #   related_work  before methodology  - "Related Techniques" is not a method
    #   methodology   before dataset      - "Materials and Methods" is methods
    #   methodology   before results      - a clinical "Statistical Analysis" is methods
    #   dataset       before results      - "Baseline Characteristics" is participants
    ("abstract",     ["abstract", "executive summary", "synopsis"]),
    ("theory", [
        "theory", "theoretical", "conceptual framework", "conceptual model",
        "formal model", "formal analysis", "mathematical model",
        "mathematical formulation", "model formulation",
        "theorem", "lemma", "corollary", "proposition ", "axiom",
        "proof of", "proofs", "derivation", "governing equation",
        "hypothesis development", "hypotheses development",
        "complexity analysis", "convergence analysis",
        "problem definition", "formal definition",
    ]),
    ("related_work", [
        "related work", "related study", "related studies", "related research",
        "literature review", "literature survey", "literature study",
        "prior work", "previous work", "previous studies", "state of the art",
        "background and related", "survey of", "review of",
        "existing work", "existing approach", "existing method",
        "comparison with", "related technique",
        "prior art", "prior research",
        # humanities / evidence synthesis
        "historiography", "scholarly context",
        "systematic review", "scoping review", "meta-analysis",
    ]),
    ("methodology",  [
        "method", "methodology", "approach", "algorithm",
        "proposed", "system design", "system architecture",
        "model architecture", "implementation",
        "materials and method", "our model", "our approach",
        "our framework", "our system", "our method",
        "model design", "system overview",
        "experimental design", "study design", "research design",
        "study protocol", "task design", "survey design",
        "training procedure", "feature extraction", "preprocessing pipeline",
        "apparatus", "instrumentation",
        # clinical / life sciences
        "trial design", "trial protocol", "randomiz", "randomis",
        "intervention", "outcome measure", "statistical method",
        "assay", "synthesis of", "sample preparation", "surgical technique",
        # social science / humanities
        "analytic strategy", "analytical strategy", "coding procedure",
        "fieldwork", "ethnograph", "interview protocol",
        # economics / quantitative social science
        "empirical strategy", "identification strategy", "estimation strategy",
        "model specification", "econometric",
        # physical sciences / engineering
        "numerical method", "simulation setup", "boundary condition",
        "computational detail",
    ]),
    ("dataset",      [
        "dataset", "data collection", "data description",
        "corpus", "benchmark", "experimental data", "data set",
        "training data", "data acquisition", "annotation", "labeling",
        "ground truth", "data preprocessing", "preprocessing",
        "data preparation", "sampling", "data source",
        "training set", "test set", "validation set",
        # clinical / social science
        "participant", "study population", "study sample",
        "subjects", "patient", "cohort",
        "inclusion criteria", "exclusion criteria", "eligibility criteria",
        "baseline characteristic", "recruitment", "demographics",
        # life sciences
        "specimen", "reagent", "cell line",
        # humanities / archival
        "primary source", "archival", "source material",
        "descriptive statistic", "summary statistic",
    ]),
    ("results",      [
        "result", "experiment", "evaluation", "performance",
        "finding", "accuracy", "ablation",
        "empirical", "quantitative analysis", "qualitative analysis",
        "downstream", "simulation result",
        "comparative", "error analysis",
        "performance evaluation", "model performance",
        "baseline compar", "benchmark result",
        # cross-field
        "observation", "outcome", "robustness check",
        "sensitivity analysis", "adverse event",
    ]),
    ("introduction", [
        "introduction", "motivation", "overview",
        "problem statement", "background", "preliminar",
        "foundations", "technical background", "basic concepts",
        "prerequisite", "context", "challenge",
        "problem formulation", "research question",
        "research objective", "research aim", "contribution",
        "notation",
    ]),
    ("conclusion",   [
        "conclusion", "concluding", "discussion",
        "future work", "future direction", "future prospect",
        "limitation", "summary", "implication",
        "recommendation", "general discussion",
    ]),
]


def classify_heading(heading: str) -> str:
    """
    Keyword-based heading -> canonical type. Fast, no LLM.

    Returns "general" both for headings that match nothing and for headings whose
    meaning depends on the discipline (see _DEFER_TO_CONTENT). In both cases the
    caller escalates to the LLM stages.
    """
    if not heading:
        return "general"
    # Strip only the numeric/roman-numeral prefix, never word characters
    clean = _HEADING_NUM_PREFIX.sub("", heading.strip()).strip().lower()
    if not clean:
        clean = heading.strip().lower()
    # Headings are often stored with trailing punctuation ("Methodology:")
    clean = clean.rstrip(":.").strip()

    if clean in _DEFER_TO_CONTENT:
        return "general"

    for section_type, keywords in _HEADING_KEYWORDS:
        for kw in keywords:
            if kw in clean:
                return section_type
    return "general"


# ── Stage 2 ────────────────────────────────────────────────────────────────

_LABEL_GUIDE = """\
  abstract     = the paper's standalone summary
  introduction = why the work matters and what problem it addresses; also background
                 or preliminaries that explain concepts the reader needs first
                 ("Introduction", "Motivation", "Problem Statement", "Background",
                  "Preliminaries", "Clinical Context", "Notation")
  theory       = formal or conceptual development: theorems, proofs, derivations,
                 mathematical or economic models, theoretical/conceptual frameworks,
                 hypothesis development
                 ("Theoretical Framework", "Proof of Theorem 2", "Model Derivation",
                  "Hypotheses Development", "Governing Equations")
  related_work = what OTHER researchers have already published on this problem
                 ("Related Work", "Literature Review", "Prior Art", "Historiography",
                  "Systematic Review")
  methodology  = how THIS study was actually carried out: design, procedure, protocol,
                 apparatus, instruments, analysis plan
                 ("Methods", "Materials and Methods", "Study Design", "Trial Protocol",
                  "Experimental Apparatus", "Statistical Analysis", "Fieldwork",
                  "Identification Strategy")
  dataset      = what the study was performed ON: data, corpora, samples, specimens,
                 participants, cohorts, archival sources
                 ("Dataset", "Participants", "Study Population", "Sample",
                  "Baseline Characteristics", "Primary Sources", "Reagents")
  results      = what was observed, measured or estimated
                 ("Results", "Findings", "Evaluation", "Outcomes", "Observations",
                  "Robustness Checks")
  conclusion   = interpretation and closing: discussion, limitations, implications,
                 future work, recommendations
  general      = only if none of the above genuinely apply"""

_DISTINCTIONS = """\
Distinctions that are easy to get wrong:
  Background vs Related Work - "Background" / "Preliminaries" / "Foundations" explain
    concepts needed to follow the paper -> introduction. "Related Work" / "Literature
    Review" / "Prior Work" survey what other researchers did -> related_work.
    These are DIFFERENT sections. Never confuse them.
  Theory vs Methodology - a formal model, derivation or proof -> theory. The concrete
    procedure used to run the study -> methodology.
  Methodology vs Dataset - how the study was DONE -> methodology. What it was done ON
    (people, samples, data, texts) -> dataset.
  Field-dependent words - judge from the whole heading and the paper's field:
    "Survey" = a literature survey (related_work) OR a questionnaire (methodology)
    "Analysis" = an analysis plan (methodology) OR reported findings (results)
    "Materials" = "Materials and Methods" (methodology) OR stimuli/reagents (dataset)
    "Model" = a formal model (theory) OR an implemented system (methodology)"""

_BATCH_PROMPT = """\
Classify each section heading from a research paper.

The paper may come from ANY academic discipline - computer science, medicine,
psychology, physics, chemistry, economics, law, education, history. Judge each
heading by the ROLE it plays in a paper, not by whether it uses vocabulary from
any one field.

Use ONLY these exact labels (copy them exactly, with underscores):
  abstract, introduction, theory, related_work, methodology, dataset, results, conclusion, general

Label definitions, by function:
{label_guide}

{distinctions}

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
    prompt = _BATCH_PROMPT.format(
        label_guide=_LABEL_GUIDE,
        distinctions=_DISTINCTIONS,
        numbered_headings=numbered,
    )
    raw = _llm(prompt, model, temperature=temperature_for("classify"))

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

The paper may come from any academic discipline. Based on the CONTENT (not just the
title), which of these labels best describes what this section is actually about?

  abstract, introduction, theory, related_work, methodology, dataset, results, conclusion, general

{label_guide}

Use underscores (e.g. "related_work" not "related work").
Reply with ONLY the single label - nothing else."""


def verify_with_first_paragraph(heading: str, first_paragraph: str, model: str) -> str:
    """Use first paragraph content to confirm or correct a section classification."""
    prompt = _VERIFY_PROMPT.format(
        heading=heading,
        text=first_paragraph[:800],
        label_guide=_LABEL_GUIDE,
    )
    raw = _llm(prompt, model, temperature=temperature_for("classify"))
    # Take the first non-empty line of the response
    for line in raw.splitlines():
        label = _normalize(line.strip())
        if label:
            return label
    return "general"


# ── Two-step heading selection ─────────────────────────────────────────────

_IDENTIFY_SECTION_PROMPT = """\
A user is asking this question about a research paper:
"{query}"

Which single section of the paper contains the answer? The paper may be from any
academic discipline, so choose by the section's ROLE, not by the wording of the
question.

Pick EXACTLY ONE of these phrases and reply with it verbatim:
  introduction   - why the work matters; background concepts the reader needs first
  theory         - theorems, proofs, derivations, formal or conceptual models,
                   theoretical frameworks, hypothesis development
  related work   - what OTHER researchers published on this problem
  methodology    - how THIS study was carried out: design, procedure, protocol,
                   apparatus, instruments, analysis plan, statistical methods
  dataset        - what the study was performed ON: data, corpora, samples,
                   participants, cohorts, specimens, archival sources
  results        - what was observed, measured or estimated; outcomes, findings
  conclusion     - discussion, limitations, implications, future work

Reply with ONE phrase from that list and nothing else. Do not combine two phrases.
No explanation. No punctuation."""


def identify_target_section(query: str, model: str) -> str:
    """
    Step 1 - Ask the LLM what section of a paper the query is about.
    Returns a single natural-language phrase, e.g. "related work".
    """
    prompt = _IDENTIFY_SECTION_PROMPT.format(query=query)
    raw = _llm(prompt, model, temperature=temperature_for("classify"))
    # Take the first non-empty line, lowercase, strip trailing punctuation
    for line in raw.splitlines():
        phrase = line.strip().lower().rstrip(".")
        if phrase:
            return phrase
    return "general"


_MATCH_HEADINGS_PROMPT = """\
You are matching top-level section headings from a research paper to a target section.
The paper may come from any academic discipline, so match on what a section DOES, not
on the vocabulary of any one field.

Target section: "{target}"

Top-level sections (numbered for selection) with their subsections shown for context:
{numbered_headings}

Rules:
- Where a section shows a "contains:" line, trust it over the heading: it
  says what the section actually holds, and papers routinely put content
  under a heading that does not advertise it.
- Choose only TOP-LEVEL section numbers (e.g. "3" or "3, 5").
  Selecting a top-level section automatically includes all its listed subsections -
  do NOT try to select individual subsections.
- Include a section only if it genuinely belongs to the target (or is a known alias):
    "related work"  <-> "Literature Review", "Prior Work", "Related Studies",
                        "Historiography", "Systematic Review"
    "methodology"   <-> "Our Approach", "Proposed Method", "Materials and Methods",
                        "Study Design", "Trial Protocol", "Identification Strategy"
    "dataset"       <-> "Participants", "Study Population", "Corpus", "Sample",
                        "Primary Sources", "Baseline Characteristics"
    "results"       <-> "Findings", "Evaluation", "Outcomes", "Observations"
    "theory"        <-> "Theoretical Framework", "Model Derivation", "Proofs",
                        "Hypotheses Development"
    "introduction"  <-> "Background", "Preliminaries", "Foundations", "Clinical Context"
- If no section matches, reply with: none

Critical distinction - never confuse these two:
  "Background" / "Preliminaries" / "Foundations" explain foundational concepts needed
  to understand the paper -> INTRODUCTION family, NOT related work.
  "Related Work" / "Literature Review" / "Prior Work" survey previous research by
  other authors on the same problem -> RELATED WORK family, NOT background.
  A subsection named "Related Work" inside a "Background" or "Methodology" section
  does NOT make the whole parent section a "related work" section.

Reply with ONLY the top-level section numbers, comma-separated (e.g. "3, 4").
No explanation. No labels. Just the numbers (or "none")."""


def match_headings_to_target(
    target_section: str,
    headings: list[str],
    model: str,
    section_summary: dict[str, list[str]] | None = None,
    descriptions: dict[str, str] | None = None,
) -> list[str]:
    """
    LLM picks which top-level sections of the paper match the target.

    `section_summary` - optional {section_name: [subsection_name, ...]} mapping from
    VectorStore.get_section_summary().  When supplied, each section is shown with its
    subsections so the LLM understands the hierarchy and avoids confusing a subsection
    heading with a top-level section of the same name.

    `descriptions` - optional {section_name: "what the section contains"} written at
    ingest time by _summarize_sections().  A heading alone is a weak signal: nothing
    about "METHODOLOGY" reveals that it also states the corpus size, which is where
    most papers put it.  The description makes that visible, so the choice rests on
    what a section holds rather than on what it is called.
    """
    if not headings:
        return []

    lines: list[str] = []
    for i, h in enumerate(headings):
        lines.append(f"{i + 1}. {h}")
        desc = (descriptions or {}).get(h, "").strip()
        if desc:
            lines.append(f"   contains: {desc}")
        if section_summary:
            subs = section_summary.get(h, [])
            if subs:
                shown = ", ".join(subs[:6])
                more = f", … (+{len(subs) - 6} more)" if len(subs) > 6 else ""
                lines.append(f"   subsections: {shown}{more}")

    numbered = "\n".join(lines)
    prompt = _MATCH_HEADINGS_PROMPT.format(
        target=target_section,
        numbered_headings=numbered,
    )
    raw = _llm(prompt, model, temperature=temperature_for("classify")).lower()

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
You are helping route a research question to the right section of a paper. The paper
may be from any academic discipline, so think about which section ROLE holds the
answer rather than which field the wording comes from.

Task: identify which section(s) of a research paper you would search IN to find \
the answer. Focus on WHERE the answer lives, not on what topics the question mentions.

Critical rule - "topic IN section" queries:
  If the question says "in the related work", "from the methodology", "in the \
literature review", "in the background", "in the experiments", etc., that named \
section IS the target. Return ONLY that section, not the topic alongside it.
  Examples:
    "What methods are mentioned in the related work?"  -> related_work
    "What technologies are explained in the background?" -> introduction
    "What datasets are used in the methodology?"       -> methodology
    "What results are in the conclusion?"              -> conclusion

Use ONLY these exact labels (with underscores):
  abstract, introduction, theory, related_work, methodology, dataset, results, conclusion, general

  introduction  = introduction, motivation, problem statement, overview, AND any
                  "Background" or "Preliminaries" section that explains foundational
                  concepts (NOT prior research by others)
  theory        = theorems, proofs, derivations, formal or conceptual models,
                  theoretical frameworks, hypothesis development
  related_work  = literature review, prior research by other authors, existing work
                  comparison, surveys of past work (NOT background concepts)
  methodology   = how the research was done: design, procedure, protocol, apparatus,
                  instruments, analysis plan
  dataset       = what it was done on: data, corpora, samples, participants, cohorts,
                  specimens, archival sources
  results       = experiments, evaluation, metrics, performance, findings, outcomes
  conclusion    = conclusions, limitations, future directions, discussion
  general       = genuinely spans multiple sections with no named section in the query

Critical distinction - background vs related work:
  "background" / "preliminaries" / "foundations" describe concepts needed to
  understand the paper  ->  introduction
  "related work" / "literature review" / "prior work" compare previous research
  by other authors on the same problem  ->  related_work

Question: {query}

Reply with ONLY a comma-separated list of 1 or 2 labels. Use underscores. No explanation."""


def classify_query(query: str, model: str) -> list[str]:
    """LLM maps a user question to target section type(s)."""
    prompt = _QUERY_PROMPT.format(query=query)
    raw = _llm(prompt, model, temperature=temperature_for("classify"))
    valid = []
    for part in re.split(r"[,\n]", raw):
        # Strip list markers the LLM might add: "1. ", "- ", "• ", etc.
        clean = re.sub(r"^[\d.)\-•\s]+", "", part.strip())
        label = _normalize(clean)
        if label and label not in valid:
            valid.append(label)
    return valid if valid else ["general"]
