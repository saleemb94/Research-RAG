"""
Stage-1 heading classification tests.

These cover the keyword stage only, so they need neither Ollama nor Weaviate and
run in milliseconds. Run either way:

    python tests/test_section_classifier.py     # standalone, no dependencies
    pytest tests/                               # if pytest is installed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_rag.section_classifier import (  # noqa: E402
    SECTION_TYPES,
    _normalize,
    classify_heading,
    section_type_matches,
)

# ── Headings that must classify the same way in every discipline ───────────
# (discipline, heading, expected label)
CROSS_DISCIPLINE: list[tuple[str, str, str]] = [
    # Computer science / ML — the corpus this project was originally built on
    ("cs", "1. Introduction", "introduction"),
    ("cs", "2 Related Work", "related_work"),
    ("cs", "III. LITERATURE REVIEW", "related_work"),
    ("cs", "3.1 Proposed Method", "methodology"),
    ("cs", "Our Framework", "methodology"),
    ("cs", "Feature Extraction and Feature Selection", "methodology"),
    ("cs", "Dataset", "dataset"),
    ("cs", "Data Description and Pre-processing", "dataset"),
    ("cs", "EVALUATION AND RESULTS", "results"),
    ("cs", "Ablation Study", "results"),
    ("cs", "CONCLUSION", "conclusion"),

    # Medicine / clinical trials
    ("medicine", "Materials and Methods", "methodology"),
    ("medicine", "Trial Design", "methodology"),
    ("medicine", "Randomization and Blinding", "methodology"),
    ("medicine", "Outcome Measures", "methodology"),
    ("medicine", "Study Population", "dataset"),
    ("medicine", "Inclusion Criteria", "dataset"),
    ("medicine", "Exclusion Criteria", "dataset"),
    ("medicine", "Baseline Characteristics", "dataset"),
    ("medicine", "Patient Recruitment", "dataset"),
    ("medicine", "Adverse Events", "results"),
    ("medicine", "Clinical Implications", "conclusion"),

    # Psychology / social science
    ("social", "Participants", "dataset"),
    ("social", "Study Design", "methodology"),
    ("social", "Interview Protocol", "methodology"),
    ("social", "Coding Procedure", "methodology"),
    ("social", "Analytic Strategy", "methodology"),
    ("social", "Theoretical Framework", "theory"),
    ("social", "Hypotheses Development", "theory"),
    ("social", "Demographics of Respondents", "dataset"),

    # Life sciences / chemistry
    ("life-sci", "Reagents and Cell Lines", "dataset"),
    ("life-sci", "Specimen Collection", "dataset"),
    ("life-sci", "Sample Preparation", "methodology"),
    ("life-sci", "Assay Conditions", "methodology"),

    # Physics / engineering
    ("physics", "Experimental Apparatus", "methodology"),
    ("physics", "Boundary Conditions", "methodology"),
    ("physics", "Numerical Methods", "methodology"),
    ("physics", "Governing Equations", "theory"),
    ("physics", "Observations", "results"),

    # Mathematics / theoretical CS
    ("math", "2. Preliminaries", "introduction"),
    ("math", "Proof of Theorem 2", "theory"),
    ("math", "Lemma 3 and its Corollaries", "theory"),
    ("math", "Complexity Analysis", "theory"),
    ("math", "Model Derivation", "theory"),
    ("math", "Notation", "introduction"),

    # Economics / quantitative social science
    ("economics", "Identification Strategy", "methodology"),
    ("economics", "Empirical Strategy", "methodology"),
    ("economics", "Model Specification", "methodology"),
    ("economics", "Descriptive Statistics", "dataset"),
    ("economics", "Robustness Checks", "results"),
    ("economics", "Policy Implications", "conclusion"),

    # Humanities / law
    ("humanities", "Historiography", "related_work"),
    ("humanities", "Primary Sources", "dataset"),
    ("humanities", "Archival Material", "dataset"),
    ("humanities", "Fieldwork", "methodology"),
    ("humanities", "Conceptual Framework", "theory"),

    # Evidence synthesis
    ("review", "Systematic Review of Prior Studies", "related_work"),
    ("review", "Scoping Review", "related_work"),
]

# ── Headings that mean different things in different fields ────────────────
# Stage 1 must refuse to guess and return "general" so the LLM stages, which
# can read the section body, decide instead.
MUST_DEFER: list[str] = [
    "Analysis",              # analysis plan (methods) vs reported analysis (results)
    "Statistical Analysis",  # methods in medicine, results in CS
    "Data Analysis",         # methods in qualitative research
    "Survey",                # literature survey vs questionnaire instrument
    "Materials",             # reagents/stimuli vs "Materials and Methods"
    "Model",                 # formal model (theory) vs implemented system
    "Framework",
    "Experimental Setup",    # apparatus vs data + hyperparameters
    "Case Study",            # research strategy vs reported case
    "Measures",              # instrument vs outcome
    "Procedure",
    "Protocol",
    "Applications",
    "Design",
    "Architecture",
]

# ── The background/related-work distinction the whole design hinges on ─────
BACKGROUND_VS_RELATED: list[tuple[str, str]] = [
    ("Background", "introduction"),
    ("Technical Background", "introduction"),
    ("Preliminaries", "introduction"),
    ("Foundations", "introduction"),
    ("Clinical Context", "introduction"),
    ("Related Work", "related_work"),
    ("Literature Review", "related_work"),
    ("Prior Art", "related_work"),
    ("Background and Related Work", "related_work"),
]

# ── Prefix stripping and punctuation ───────────────────────────────────────
FORMATTING: list[tuple[str, str]] = [
    ("1 Introduction", "introduction"),
    ("1. Introduction", "introduction"),
    ("1.2 Methodology", "methodology"),
    ("IV. RESULTS", "results"),
    ("A. Data Collection", "dataset"),
    ("Methodology:", "methodology"),
    ("Conclusion.", "conclusion"),
    # A leading letter must never be eaten off a real word
    ("Apparatus", "methodology"),
]

# ── _normalize: LLM output is messy ────────────────────────────────────────
NORMALIZE: list[tuple[str, str]] = [
    ("related work", "related_work"),
    ("related_work", "related_work"),
    ("Related-Work", "related_work"),
    ("methodolog", "methodology"),
    ("literature review", "related_work"),
    ("theoretical framework", "theory"),
    ("theory", "theory"),
    ("dataset.", "dataset"),
    # Too short to prefix-match: must NOT resolve to a label
    ("t", ""),
    ("g", ""),
    ("re", ""),
]


def _check(label: str, cases, fn) -> list[str]:
    failures = []
    for case in cases:
        *context, arg, expected = case if len(case) == 3 else (None, *case)
        got = fn(arg)
        if got != expected:
            tag = f"[{context[0]}] " if context and context[0] else ""
            failures.append(f"  {label}: {tag}{arg!r} -> {got!r}, expected {expected!r}")
    return failures


def test_taxonomy_is_wellformed():
    assert "theory" in SECTION_TYPES, "theory label missing from the taxonomy"
    assert "general" in SECTION_TYPES
    assert len(SECTION_TYPES) == len(set(SECTION_TYPES)), "duplicate labels"


def test_cross_discipline_headings():
    failures = _check("cross-discipline", CROSS_DISCIPLINE, classify_heading)
    assert not failures, "\n" + "\n".join(failures)


def test_ambiguous_headings_defer_to_content():
    failures = [
        f"  defer: {h!r} -> {classify_heading(h)!r}, expected 'general'"
        for h in MUST_DEFER
        if classify_heading(h) != "general"
    ]
    assert not failures, "\n" + "\n".join(failures)


def test_background_is_not_related_work():
    failures = _check("background-vs-related", BACKGROUND_VS_RELATED, classify_heading)
    assert not failures, "\n" + "\n".join(failures)


def test_heading_formatting():
    failures = _check("formatting", FORMATTING, classify_heading)
    assert not failures, "\n" + "\n".join(failures)


def test_normalize():
    failures = _check("normalize", NORMALIZE, _normalize)
    assert not failures, "\n" + "\n".join(failures)


def test_canonical_phrases_round_trip():
    """
    identify_target_section() is prompted to reply with one of these phrases, and
    map_reduce cross-validates the result by feeding it back through
    classify_heading(). Every phrase the prompt offers must therefore survive that
    round trip, or the cross-validation silently degrades to a no-op.
    """
    phrases = {
        "introduction": "introduction",
        "theory": "theory",
        "related work": "related_work",
        "methodology": "methodology",
        "dataset": "dataset",
        "results": "results",
        "conclusion": "conclusion",
    }
    failures = [
        f"  round-trip: {p!r} -> {classify_heading(p)!r}, expected {want!r}"
        for p, want in phrases.items()
        if classify_heading(p) != want
    ]
    assert not failures, "\n" + "\n".join(failures)


def test_every_label_is_reachable_from_keywords():
    """No label should be defined but unreachable through Stage 1."""
    produced = {classify_heading(h) for _, h, _ in CROSS_DISCIPLINE}
    unreachable = set(SECTION_TYPES) - produced - {"abstract", "general"}
    assert not unreachable, f"labels never produced by the test corpus: {unreachable}"


def test_section_type_compatibility():
    """
    The retrieval cross-validation is a veto on clearly wrong sections, not a
    demand for an exact label match: papers routinely describe their corpus
    inside the methods section, and a Discussion restates the results.
    """
    should_match = [
        ("methodology", "dataset", "corpus is usually described inside Methods"),
        ("dataset", "methodology", "and the relation is symmetric"),
        ("conclusion", "results", "Discussion restates the numbers"),
        ("results", "conclusion", "symmetric"),
        ("theory", "methodology", "a formal model sits next to its implementation"),
        ("abstract", "results", "an abstract states the headline result"),
        ("abstract", "dataset", "and names the dataset"),
        ("general", "dataset", "unclassified is not evidence of a contradiction"),
        ("dataset", "dataset", "identity"),
        ("results", "general", "a general query vetoes nothing"),
    ]
    should_not = [
        ("related_work", "introduction", "background is not prior work"),
        ("introduction", "related_work", "and prior work is not background"),
        ("results", "dataset", "a results table is not the dataset description"),
        ("conclusion", "methodology", "conclusions do not describe the procedure"),
        ("related_work", "results", "other people's work is not this paper's result"),
    ]
    failures = []
    for stored, target, why in should_match:
        if not section_type_matches(stored, target):
            failures.append(f"  {stored!r} should answer {target!r}: {why}")
    for stored, target, why in should_not:
        if section_type_matches(stored, target):
            failures.append(f"  {stored!r} must NOT answer {target!r}: {why}")
    assert not failures, "\n" + "\n".join(failures)


def test_background_and_related_work_never_compatible():
    """The one distinction the whole retrieval design is built to preserve."""
    assert not section_type_matches("related_work", "introduction")
    assert not section_type_matches("introduction", "related_work")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}{e}")
    total = len(tests)
    print(f"\n{total - failed}/{total} test(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
