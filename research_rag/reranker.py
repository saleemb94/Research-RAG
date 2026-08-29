from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL
from .vector_store import SearchHit


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"Loading reranker model: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return hits
        pairs = [(query, hit.properties["text"]) for hit in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        return [hit for _, hit in ranked[:top_n]]
