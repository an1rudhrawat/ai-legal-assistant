"""Optional fully local cross-encoder reranking."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker:
    def __init__(self, model_name: str, enabled: bool = False) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._load_attempted = False

    def rerank(self, query: str, candidates: list[tuple[float, int, str]]) -> list[tuple[float, int, str]]:
        """Rerank candidates while retaining their index in the local index.

        The text is not a stable identifier: two sections from different sources
        can legitimately contain the same wording.
        """
        if not self.enabled or not candidates:
            return candidates
        if not self._load_attempted:
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
                logger.warning("Local reranker unavailable; using rank-fusion order")
        if self._model is None:
            return candidates
        scores = self._model.predict([(query, text) for _, _, text in candidates])
        return sorted(
            ((float(score), index, text) for score, (_, index, text) in zip(scores, candidates)),
            key=lambda candidate: candidate[0],
            reverse=True,
        )
