from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Retriever
from .embeddings import cosine_similarity, embed
from .models import LegalChunk
from .reranker import LocalCrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[LegalChunk, ...]
    scores: tuple[float, ...]
    sufficient: bool


class LegalRetriever:
    """Hybrid local retriever: dense semantic + BM25 + RRF + optional reranking."""

    def __init__(self, index_path: Path, minimum_relevance: float = 0.12, embedding_model: str = "all-MiniLM-L6-v2", reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enable_reranker: bool = False, enable_dense: bool = True) -> None:
        self.index_path = index_path
        self.minimum_relevance = minimum_relevance
        self.embedding_model = embedding_model
        self.enable_dense = enable_dense
        self.reranker = LocalCrossEncoderReranker(reranker_model, enable_reranker)
        self._dense_encoder = None
        self._dense_load_attempted = False
        self._dense_vectors: list | None = None

    def _dense_scores(self, query: str, documents: list[str]) -> list[float] | None:
        if not self.enable_dense:
            return None
        if not self._dense_load_attempted:
            self._dense_load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                self._dense_encoder = SentenceTransformer(self.embedding_model, local_files_only=True)
            except Exception:
                logger.warning("Local dense model unavailable; using BM25/term-vector fallback until it is installed locally")
        if self._dense_encoder is None:
            return None
        if self._dense_vectors is None or len(self._dense_vectors) != len(documents):
            self._dense_vectors = self._dense_encoder.encode(documents, normalize_embeddings=True)
        query_vector = self._dense_encoder.encode([query], normalize_embeddings=True)[0]
        return [float(query_vector @ vector) for vector in self._dense_vectors]

    @staticmethod
    def _rank(scores: list[float], limit: int) -> list[int]:
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]

    @staticmethod
    def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for position, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1 / (k + position)
        return fused

    def search(self, query: str, limit: int = 4) -> RetrievalResult:
        if not self.index_path.exists():
            return RetrievalResult((), (), False)
        records = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not records:
            return RetrievalResult((), (), False)
        documents = [item["chunk"]["text"] for item in records]
        lexical_scores = BM25Retriever(documents).scores(query)
        term_scores = [cosine_similarity(embed(query), item["embedding"]) for item in records]
        dense_scores = self._dense_scores(query, documents)
        rankings = [self._rank(lexical_scores, 50), self._rank(term_scores, 50)]
        if dense_scores is not None:
            rankings.append(self._rank(dense_scores, 50))
        fused = self._rrf(rankings)
        candidate_indices = [index for index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:50]]
        candidates = [(fused[index], documents[index]) for index in candidate_indices]
        reranked = self.reranker.rerank(query, candidates)
        index_by_text = {text: index for index, text in enumerate(documents)}
        ranked = []
        for fused_score, text in reranked[:limit]:
            index = index_by_text[text]
            evidence_score = max(term_scores[index], dense_scores[index] if dense_scores is not None else 0.0)
            ranked.append((evidence_score, fused_score, LegalChunk.from_dict(records[index]["chunk"])))
        accepted = [(score, chunk) for score, _, chunk in ranked if score >= self.minimum_relevance]
        return RetrievalResult(
            tuple(chunk for _, chunk in accepted),
            tuple(round(score, 4) for score, _ in accepted),
            bool(accepted),
        )
