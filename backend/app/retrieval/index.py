from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import embed
from .models import LegalChunk


def build_index(chunks_path: Path, index_path: Path) -> int:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return build_index_from_chunks(chunks, index_path)


def build_index_from_chunks(chunks: list[dict], index_path: Path) -> int:
    if not chunks:
        raise ValueError("No chunks found; ingestion did not produce an indexable document")
    records = [{"chunk": LegalChunk.from_dict(chunk).to_dict(), "embedding": embed(chunk["text"])} for chunk in chunks]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(records), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local legal retrieval index.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/legal_chunks.json"))
    parser.add_argument("--output", type=Path, default=Path("data/vector_store/legal_index.json"))
    args = parser.parse_args()
    print(f"Indexed {build_index(args.chunks, args.output)} legal chunks at {args.output}")


if __name__ == "__main__":
    main()
