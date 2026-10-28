from __future__ import annotations

import json
import logging
import re
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
    has_relevant_chunks: bool

    @property
    def sufficient(self) -> bool:
        """Compatibility alias; this is a relevance cutoff, not legal sufficiency."""
        return self.has_relevant_chunks


@dataclass(frozen=True)
class RetrievalDebugChunk:
    chunk: LegalChunk
    relevance_score: float
    bm25_score: float
    term_score: float
    dense_score: float | None
    rrf_score: float
    ranking_score: float
    passed_minimum_relevance: bool
    exact_section_match: bool = False


@dataclass(frozen=True)
class RetrievalDebugResult:
    query: str
    minimum_relevance: float
    index_available: bool
    dense_available: bool
    reranker_active: bool
    result: RetrievalResult
    retrieved: tuple[RetrievalDebugChunk, ...]


class LegalRetriever:
    """Hybrid local retriever: dense semantic + BM25 + RRF + optional reranking."""

    SECTION_QUERY = re.compile(r"\b(?:sections?|secs?\.?)\s*(?:no\.?\s*)?(\d+[a-z]?)\b", re.IGNORECASE)
    EXACT_SECTION_SCORE = 1.0

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
        return self._search(query, limit)[0]

    def debug_search(self, query: str, limit: int = 4) -> RetrievalDebugResult:
        """Expose local ranking signals for offline evaluation, never the API response."""
        return self._search(query, limit)[1]

    def _search(self, query: str, limit: int) -> tuple[RetrievalResult, RetrievalDebugResult]:
        if not self.index_path.exists():
            result = RetrievalResult((), (), False)
            return result, RetrievalDebugResult(query, self.minimum_relevance, False, False, False, result, ())
        records = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not records:
            result = RetrievalResult((), (), False)
            return result, RetrievalDebugResult(query, self.minimum_relevance, True, False, False, result, ())
        documents = [item["chunk"]["text"] for item in records]
        lexical_scores = BM25Retriever(documents).scores(query)
        term_scores = [cosine_similarity(embed(query), item["embedding"]) for item in records]
        dense_scores = self._dense_scores(query, documents)
        requested_sections = {match.group(1).casefold() for match in self.SECTION_QUERY.finditer(query)}
        exact_section_indices = {
            index for index, item in enumerate(records)
            if requested_sections
            and str(item["chunk"].get("section_number") or "").casefold() in requested_sections
        }
        if exact_section_indices:
            source_hints = {index: self._source_hint(query, records[index]["chunk"]) for index in exact_section_indices}
            best_source_hint = max(source_hints.values(), default=0.0)
            if best_source_hint > 0:
                exact_section_indices = {
                    index for index, score in source_hints.items() if score == best_source_hint
                }
        rankings = [self._rank(lexical_scores, 50), self._rank(term_scores, 50)]
        if dense_scores is not None:
            rankings.append(self._rank(dense_scores, 50))
        fused = self._rrf(rankings)
        normal_candidates = [index for index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:50]]
        if exact_section_indices:
            # A literal statutory section request is an identifier lookup. Keep
            # matching metadata candidates even when prose similarity ranked
            # them below the semantic top 50.
            exact_candidates = sorted(
                exact_section_indices,
                key=lambda index: (
                    self._source_hint(query, records[index]["chunk"]),
                    max(term_scores[index], dense_scores[index] if dense_scores is not None else 0.0),
                    fused.get(index, 0.0),
                ),
                reverse=True,
            )
            # Exact section lookups should not be diluted with semantically
            # similar but different provisions; if there is no exact metadata
            # match, normal hybrid retrieval below remains the fallback.
            candidate_indices = exact_candidates
        else:
            candidate_indices = normal_candidates
        # Preserve the record index through reranking. Chunk text is not unique,
        # so mapping a reranked text back to an index can select the wrong source.
        candidates = [(fused.get(index, 0.0), index, documents[index]) for index in candidate_indices]
        reranked = self.reranker.rerank(query, candidates)
        if exact_section_indices:
            reranked = sorted(
                reranked,
                key=lambda item: (
                    item[1] in exact_section_indices,
                    self._source_hint(query, records[item[1]]["chunk"]),
                    item[0],
                ),
                reverse=True,
            )
        reranker_active = self.reranker.enabled and self.reranker._model is not None
        ranked: list[RetrievalDebugChunk] = []
        for reranker_score, index, _ in reranked[:limit]:
            exact_section_match = index in exact_section_indices
            evidence_score = max(
                term_scores[index],
                dense_scores[index] if dense_scores is not None else 0.0,
                self.EXACT_SECTION_SCORE if exact_section_match else 0.0,
            )
            ranked.append(RetrievalDebugChunk(
                chunk=LegalChunk.from_dict(records[index]["chunk"]),
                relevance_score=evidence_score,
                bm25_score=lexical_scores[index],
                term_score=term_scores[index],
                dense_score=dense_scores[index] if dense_scores is not None else None,
                rrf_score=fused.get(index, 0.0),
                ranking_score=reranker_score,
                passed_minimum_relevance=evidence_score >= self.minimum_relevance,
                exact_section_match=exact_section_match,
            ))
        accepted = [item for item in ranked if item.passed_minimum_relevance]
        result = RetrievalResult(
            tuple(item.chunk for item in accepted),
            tuple(round(item.relevance_score, 4) for item in accepted),
            bool(accepted),
        )
        return result, RetrievalDebugResult(
            query=query,
            minimum_relevance=self.minimum_relevance,
            index_available=True,
            dense_available=dense_scores is not None,
            reranker_active=reranker_active,
            result=result,
            retrieved=tuple(ranked),
        )

    @staticmethod
    def _source_hint(query: str, chunk: dict) -> float:
        """Prefer a requested Act/document name among same-numbered sections."""
        query_words = set(re.findall(r"[a-z0-9]+", query.casefold()))
        source_words = set(re.findall(
            r"[a-z0-9]+",
            f"{chunk.get('act_name') or ''} {chunk.get('document_name') or ''}".casefold(),
        ))
        ignored = {"the", "and", "of", "in", "act", "2023", "law"}
        meaningful = source_words - ignored
        return len(meaningful & query_words) / len(meaningful) if meaningful else 0.0
