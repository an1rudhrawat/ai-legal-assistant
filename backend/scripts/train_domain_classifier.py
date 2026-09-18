"""Train and evaluate the local semantic legal-situation classifier."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_split(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["text"] for row in rows], [row["label"] for row in rows]


def report(name: str, labels: list[str], predictions: list[str]) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=["legal", "non_legal"], average="binary", pos_label="legal", zero_division=0)
    print(f"{name}: accuracy={accuracy_score(labels, predictions):.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print("confusion matrix [legal, non_legal]:")
    print(confusion_matrix(labels, predictions, labels=["legal", "non_legal"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/domain_classifier")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "data/domain_classifier/model")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--allow-download", action="store_true", help="Download the embedding model when it is not already cached locally")
    args = parser.parse_args()
    train_text, train_labels = load_split(args.data_dir / "train.csv")
    validation_text, validation_labels = load_split(args.data_dir / "validation.csv")
    test_text, test_labels = load_split(args.data_dir / "test.csv")
    encoder = SentenceTransformer(args.embedding_model, local_files_only=not args.allow_download)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    classifier.fit(encoder.encode(train_text, normalize_embeddings=True), train_labels)
    report("validation", validation_labels, classifier.predict(encoder.encode(validation_text, normalize_embeddings=True)))
    report("test", test_labels, classifier.predict(encoder.encode(test_text, normalize_embeddings=True)))
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model_dir / "classifier.joblib")
    print(f"Saved classifier to {args.model_dir / 'classifier.joblib'}")


if __name__ == "__main__":
    main()
