"""
Token-level semantic similarity between an answer and a reference.

Why this exists: the golden sets score by substring, which cannot credit a
correct paraphrase. That is not a hypothetical limitation. The thematic question
*how do these papers evaluate retrieval and generation quality* scores 1 of 5
fact groups while correctly describing perplexity differences, DCG@k and the
eRAG method, purely because it never types the letters "nDCG". Substring scoring
measures vocabulary; this measures meaning, and the two disagree in exactly the
cases worth knowing about.

This is BERTScore's algorithm - greedy cosine matching between the token
embeddings of candidate and reference, precision from the candidate's side,
recall from the reference's - and deliberately not the `bert-score` package.
That package defaults to roberta-large, which is a 1.4GB download and a new
dependency for a scoring script, in a project whose point is that it runs fully
offline with no model downloads. The backbone here is the embedding model the
pipeline already loads, so this adds nothing to install.

Call it BERTScore-style rather than BERTScore. The algorithm is the same, the
backbone is smaller, and the absolute numbers are therefore not comparable with
published BERTScore figures. They are comparable with each other, which is all
a regression metric needs.
"""

from __future__ import annotations

import numpy as np


class SemanticScorer:
    def __init__(self, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            from research_rag.config import EMBEDDING_MODEL

            model = SentenceTransformer(EMBEDDING_MODEL)
        self._model = model

    def _tokens(self, text: str) -> np.ndarray:
        """
        Contextual embedding per token, L2-normalized so a dot product is cosine.

        Sentence-transformers pools by default; the pooled vector is exactly what
        this is trying not to use, because a single vector per answer cannot tell
        which parts of the reference went missing.
        """
        vecs = self._model.encode(
            [text or ""], output_value="token_embeddings", convert_to_numpy=False
        )[0]
        arr = vecs.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        # Strip the two special tokens; they match everything and inflate both sides.
        if len(arr) > 2:
            arr = arr[1:-1]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)

    def score(self, candidate: str, reference: str) -> dict:
        """
        Precision, recall and F1 over greedily matched tokens.

        Reported apart because they fail apart, the same way facts and papers do
        in the synthesis scorer. Recall falling means the answer left out part of
        what the reference said. Precision falling means it added something the
        reference does not support, which for a cited research tool is the more
        serious direction.
        """
        if not (candidate or "").strip() or not (reference or "").strip():
            return {"p": 0.0, "r": 0.0, "f1": 0.0}
        c, r = self._tokens(candidate), self._tokens(reference)
        sim = c @ r.T                      # cosine, both sides normalized
        p = float(sim.max(axis=1).mean())  # each candidate token to its best reference token
        rec = float(sim.max(axis=0).mean())
        f1 = 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)
        return {"p": p, "r": rec, "f1": f1}

    def score_many(self, pairs: list[tuple[str, str]]) -> list[dict]:
        return [self.score(c, r) for c, r in pairs]


def baseline_floor(scorer: SemanticScorer, texts: list[str], pairs: int = 40) -> float:
    """
    Mean F1 between unrelated answers, which is the number that makes the rest
    readable.

    Contextual embeddings of same-domain prose are similar to each other whatever
    they say: fifty papers about retrieval share vocabulary, structure and
    register. Without knowing what an unrelated pair scores, an F1 of 0.85 could
    be excellent or could be the floor, and the metric would be decorative.
    """
    import random

    rng = random.Random(0)
    if len(texts) < 2:
        return 0.0
    scores = []
    for _ in range(pairs):
        a, b = rng.sample(range(len(texts)), 2)
        scores.append(scorer.score(texts[a], texts[b])["f1"])
    return sum(scores) / len(scores)
