"""Local BM25 retrieval with a dependency-free fallback for tests."""
from __future__ import annotations

from .embeddings import cosine_similarity, embed, tokenize


class BM25Retriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        try:
            from rank_bm25 import BM25Okapi
            self._model = BM25Okapi([tokenize(document) for document in documents])
        except ImportError:
            self._model = None
            self._embeddings = [embed(document) for document in documents]

    def scores(self, query: str) -> list[float]:
        if self._model is not None:
            return [float(score) for score in self._model.get_scores(tokenize(query))]
        query_embedding = embed(query)
        return [cosine_similarity(query_embedding, item) for item in self._embeddings]
