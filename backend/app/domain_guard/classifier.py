"""Local semantic domain classification for first-responder requests."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

DomainLabel = Literal["legal", "non_legal", "ambiguous"]


@dataclass(frozen=True)
class DomainClassification:
    label: DomainLabel
    confidence: float
    legal_probability: float
    reason: str


class DomainClassifier(Protocol):
    def classify(self, text: str) -> DomainClassification: ...


class LocalSemanticDomainClassifier:
    """SentenceTransformer + LogisticRegression classifier loaded from disk.

    Model loading is lazy so importing the FastAPI application never downloads a
    model or retrains it. If the local model has not been trained yet, requests
    are conservatively marked ambiguous and continue to clarification/retrieval.
    """

    def __init__(self, model_path: Path, embedding_model: str, legal_threshold: float = 0.75, non_legal_threshold: float = 0.25) -> None:
        if not 0 <= non_legal_threshold < legal_threshold <= 1:
            raise ValueError("domain classification thresholds must satisfy 0 <= non-legal < legal <= 1")
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.legal_threshold = legal_threshold
        self.non_legal_threshold = non_legal_threshold
        self._classifier = None
        self._encoder = None
        self._load_attempted = False

    def _load(self) -> bool:
        if self._load_attempted:
            return self._classifier is not None and self._encoder is not None
        self._load_attempted = True
        classifier_file = self.model_path / "classifier.joblib"
        if not classifier_file.exists():
            logger.warning("Local domain classifier is unavailable; run backend/scripts/train_domain_classifier.py")
            return False
        try:
            import joblib
            from sentence_transformers import SentenceTransformer

            self._classifier = joblib.load(classifier_file)
            self._encoder = SentenceTransformer(self.embedding_model, local_files_only=True)
            return True
        except Exception:
            logger.exception("Could not load the local semantic domain classifier")
            self._classifier = self._encoder = None
            return False

    def classify(self, text: str) -> DomainClassification:
        if not self._load():
            return DomainClassification("ambiguous", 0.0, 0.5, "local semantic model unavailable")
        vector = self._encoder.encode([text], normalize_embeddings=True)
        probabilities = self._classifier.predict_proba(vector)[0]
        classes = list(self._classifier.classes_)
        legal_probability = float(probabilities[classes.index("legal")])
        if legal_probability >= self.legal_threshold:
            return DomainClassification("legal", legal_probability, legal_probability, "semantic legal-situation classification")
        if legal_probability <= self.non_legal_threshold:
            return DomainClassification("non_legal", 1 - legal_probability, legal_probability, "semantic non-legal classification")
        return DomainClassification("ambiguous", max(legal_probability, 1 - legal_probability), legal_probability, "semantic confidence between thresholds")
