"""
How well does each claim's cited passage actually support it?

The problem this exists for: an answer that is right 70% of the time and
confident 100% of the time is harder to use than one that is right less often
and says which parts to check. The failure is invisible at the point of reading
- a wrong claim looks exactly like a right one, carries a citation, and the
citation resolves to a real passage. The reader has no signal about where to
look, so either they verify everything, which removes the reason to use the
tool, or they verify nothing.

What is measured here is **grounding, not truth**: how much of a claim is
actually present in the passage it cites. That is a narrower thing than
correctness and it is the honest one, because it is checkable without another
model's opinion. A claim can be grounded and wrong if the paper itself is wrong,
and this will not catch that. What it catches is the failure mode that matters
for a citation tool - a claim drifting away from the passage it points at.

Two signals, deliberately both cheap and deterministic. No extra model call per
claim: a confidence score that costs a second LLM round trip would double the
latency of every answer, and a sampled model scoring its own output is not
evidence of anything.

  numbers   Every figure in the claim must appear in the cited passages. This is
            the hard signal and the one that matters most for research prose,
            where claims are largely numeric. "84.76% accuracy [3]" where source
            3 contains no 84.76 is a provable grounding failure, not a judgment
            call.
  terms     Distinctive words of the claim, present in the cited passages.
            Stopwords and words common to the whole corpus carry no evidence, so
            only longer and rarer tokens count.
"""

from __future__ import annotations

import re

# A claim is a sentence, and its citations are every marker inside it.
#
# Splitting at the citation groups instead was tried first and is wrong. One
# sentence does often carry two claims with different citations, so the finer
# unit looked better - but it strands the text after the last marker as its own
# claim, and those fragments ("These results are supported by Figure 1.") are
# not claims at all. They then scored near zero and dominated the flag, so the
# feature pointed at punctuation rather than at anything a reader should check.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_CITE_NUM = re.compile(r"\[(\d+)\]")

# Written forms are not comparable digit by digit, so they are excluded rather
# than counted as missing: "two datasets" against a passage saying "2 datasets"
# is a formatting difference, not a grounding failure.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")

# "Figure 1", "Table 4", "Section 3" are pointers into a document, not claims
# about data. Counting them as unsupported figures made every answer that
# mentioned a figure look ungrounded.
_REFERENCE_NUM = re.compile(
    r"\b(?:figure|fig\.?|table|tab\.?|section|sect\.?|eq\.?|equation|"
    r"algorithm|appendix|chapter|step|phase|rq)\s*\d+", re.I)

_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "that", "this", "these", "those", "it", "its", "is", "are", "was", "were",
    "be", "been", "as", "at", "by", "from", "which", "while", "also", "than",
    "then", "there", "their", "they", "not", "no", "however", "therefore",
    "such", "some", "other", "both", "each", "used", "using", "use", "can",
    "may", "more", "most", "one", "two", "based", "paper", "papers", "study",
    "studies", "source", "sources", "report", "reports", "reported", "states",
    "show", "shows", "shown", "found", "across", "between", "within", "about",
}


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)              # 1,054 -> 1054
    t = re.sub(r"(?<=\d)\s+\.\s+(?=\d)", ".", t)         # 29 . 6 -> 29.6
    return re.sub(r"\s+", " ", t)


def split_claims(answer: str) -> list[dict]:
    """
    Break an answer into (text, citation numbers) spans, in reading order.

    Text before any marker is returned with no citations. That is not an
    oversight: an uncited sentence in a cited answer is exactly the thing a
    reader should be warned about, and dropping it would hide it.
    """
    if not answer:
        return []
    claims: list[dict] = []
    for sentence in _SENTENCE.split(answer.strip()):
        text = sentence.strip()
        # Enough letters to be a statement rather than a stray fragment.
        if len(re.sub(r"[^a-z]", "", text.lower())) < 20:
            continue
        claims.append({
            "text": text,
            "cites": sorted({int(n) for n in _CITE_NUM.findall(text)}),
        })
    return claims


def score_claim(text: str, cited_texts: list[str]) -> dict:
    """
    Grounding for one claim against the passages it cites.

    Numbers dominate when a claim has any, because they are the part a reader
    would otherwise have to check by hand and the part a model is most likely to
    carry across from the wrong passage.
    """
    hay = _norm(" ".join(cited_texts))
    claim = _norm(text)

    # Strip citation markers and document pointers before looking for figures,
    # so "[3]" and "Figure 1" are never mistaken for data the claim asserts.
    bare = _REFERENCE_NUM.sub(" ", _CITE_NUM.sub(" ", claim))
    nums = _NUMBER.findall(bare)
    # A bare year is usually citation furniture rather than a claim about data.
    nums = [n for n in nums if not re.fullmatch(r"(19|20)\d{2}", n.rstrip("%"))]
    num_hits = [n for n in nums if n.rstrip("%") in hay]
    num_score = (len(num_hits) / len(nums)) if nums else None

    words = [w for w in re.findall(r"[a-z][a-z0-9-]{3,}", bare) if w not in _STOP]
    term_hits = [w for w in words if w in hay]
    term_score = (len(term_hits) / len(words)) if words else None

    if not cited_texts:
        conf = 0.0
    elif num_score is None and term_score is None:
        conf = 0.0
    elif num_score is None:
        conf = term_score
    elif term_score is None:
        conf = num_score
    else:
        # Weighted toward numbers, which are checkable and load-bearing; terms
        # alone are satisfied too easily by same-domain vocabulary.
        conf = 0.65 * num_score + 0.35 * term_score

    return {
        "confidence": round(conf, 3),
        "numbers": nums,
        "numbers_missing": [n for n in nums if n not in num_hits],
        "term_support": None if term_score is None else round(term_score, 3),
        "number_support": None if num_score is None else round(num_score, 3),
        "uncited": not cited_texts,
    }


def annotate(answer: str, sources: list[dict]) -> list[dict]:
    """
    Score every claim in an answer. `sources` are dicts with `number` and `text`.
    """
    by_num = {s["number"]: (s.get("text") or "") for s in sources}
    out = []
    for c in split_claims(answer):
        texts = [by_num[n] for n in c["cites"] if n in by_num]
        out.append({**c, **score_claim(c["text"], texts)})
    return out


# A claim that something is *absent* cannot echo the passage it cites, because
# the whole assertion is that the passage does not contain the thing. Its low
# score carries no information, and the first version of this flag pointed at
# nothing else: "the paper does not mention Macro-F1", "no source specifies how
# the articles are split", "the collection does not cover it". All correct, all
# scored near zero, all useless to flag.
_META = re.compile(
    r"\b(?:does not|do not|doesn't|don't|no source|none of the|not mention|"
    r"not specify|not report|not provide|not cover|not address|no relevant|"
    r"is unclear|cannot be determined|answers come from)\b", re.I)


def weakest(claims: list[dict]) -> dict | None:
    """
    The one claim a reader should check first, or nothing.

    Only one is marked, and often none. Colouring every claim trains people to
    ignore the colour, and ranking noisy scores implies a precision these signals
    do not have. One flag asks one answerable question: is this the one that is
    wrong? Opening the citation settles it in seconds.

    The flag is deliberately biased toward the provable case. A claim stating a
    figure that does not appear in the passage it cites is a grounding failure
    that can be demonstrated, not a judgement - so those are flagged first and
    ranked by how much of the claim's arithmetic is unsupported. Only when no
    claim asserts an unsupported figure does a weak wording match qualify, and
    then only well below the usual range, because term overlap alone is noisy.
    """
    scored = [c for c in claims if c["cites"] and not _META.search(c["text"])]
    if not scored:
        return None

    # 1. A figure in the claim that is not in the cited passage.
    numeric = [c for c in scored if c["numbers_missing"]]
    if numeric:
        return min(numeric, key=lambda c: (c["number_support"], c["confidence"]))

    # 2. Otherwise only a claim whose wording barely appears in its passage, at
    #    a threshold set from the measured distribution rather than picked: the
    #    lower quartile of real claims sits near 0.35, so 0.25 flags roughly the
    #    weakest sixth rather than half of them.
    worst = min(scored, key=lambda c: c["confidence"])
    return worst if worst["confidence"] < 0.25 else None
