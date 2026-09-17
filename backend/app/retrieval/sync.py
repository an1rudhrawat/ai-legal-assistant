from __future__ import annotations

import argparse
from pathlib import Path

from .corpus import sync_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description="Incrementally sync raw legal sources into the local RAG index.")
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/vector_store/legal_index.json")
    args = parser.parse_args()
    result = sync_corpus(args.raw, args.processed, args.output)
    print(f"Corpus synced: {result.added} added, {result.updated} updated, {result.unchanged} unchanged; {result.indexed_chunks} chunks indexed.")


if __name__ == "__main__":
    main()
