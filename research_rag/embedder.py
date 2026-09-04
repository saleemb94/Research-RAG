from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, QUERY_INSTRUCTION


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        ).tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a passage, or anything stored in the index."""
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a search query.

        Separate from embed_one because the two are not the same operation for
        an asymmetric retrieval model. Passages are stored bare and the query
        carries the instruction the model was trained to expect; using one
        method for both silently makes the query look like a passage.
        """
        return self.embed([QUERY_INSTRUCTION + text])[0]
