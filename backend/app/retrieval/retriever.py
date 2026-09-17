from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .embeddings import cosine_similarity, embed
from .models import LegalChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[LegalChunk, ...]
    scores: tuple[float, ...]
    sufficient: bool


class LegalRetriever:
    def __init__(self, index_path: Path, minimum_relevance: float = 0.12) -> None:
        self.index_path = index_path
        self.minimum_relevance = minimum_relevance

    def search(self, query: str, limit: int = 4) -> RetrievalResult:
        if not self.index_path.exists():
            return RetrievalResult((), (), False)
        records = json.loads(self.index_path.read_text(encoding="utf-8"))
        query_embedding = embed(query)
        ranked = sorted(
            ((cosine_similarity(query_embedding, item["embedding"]), LegalChunk.from_dict(item["chunk"])) for item in records),
            key=lambda item: item[0], reverse=True,
        )[:limit]
        accepted = [(score, chunk) for score, chunk in ranked if score >= self.minimum_relevance]
        return RetrievalResult(
            tuple(chunk for _, chunk in accepted),
            tuple(round(score, 4) for score, _ in accepted),
            bool(accepted),
        )
