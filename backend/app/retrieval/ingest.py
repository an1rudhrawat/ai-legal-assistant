from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .chunker import chunk_document


REQUIRED_METADATA = {"document_id", "document_name", "act_name", "document_type", "source_url", "issuing_authority"}


def read_source(path: Path) -> tuple[str, dict[str, str | None]]:
    if path.suffix.lower() == ".json":
        source = json.loads(path.read_text(encoding="utf-8"))
        return str(source.pop("text", "")), source
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return text, {}
    return path.read_text(encoding="utf-8"), {}


def inferred_metadata(source_path: Path) -> dict[str, str | None]:
    """Provide transparent fallback provenance for a single-file upload.

    Local files cannot establish an official URL, issuer, or version. Those values
    are intentionally labelled as unverified rather than guessed.
    """
    title = re.sub(r"[_-]+", " ", source_path.stem).strip() or "Uploaded legal document"
    document_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "uploaded-document"
    return {
        "document_id": document_id,
        "document_name": title,
        "act_name": title,
        "document_type": "uploaded legal document (provenance unverified)",
        "source_url": f"Local upload: {source_path.name}",
        "issuing_authority": "Not supplied; verify before relying on this source",
        "document_version": None,
        "effective_date": None,
        "publication_date": None,
    }


def ingest_file(source_path: Path, metadata_path: Path | None, output_path: Path) -> int:
    text, embedded_metadata = read_source(source_path)
    metadata = {**inferred_metadata(source_path), **embedded_metadata}
    if metadata_path:
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    missing = sorted(key for key in REQUIRED_METADATA if not metadata.get(key))
    if missing:
        raise ValueError(f"Missing required provenance fields: {', '.join(missing)}")
    chunks = chunk_document(text, metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([chunk.to_dict() for chunk in chunks], indent=2), encoding="utf-8")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest an authoritative Indian legal source.")
    parser.add_argument("source", type=Path, help="Text, PDF, or JSON source file")
    parser.add_argument("--metadata", type=Path, help="JSON provenance metadata for text/PDF files")
    parser.add_argument("--output", type=Path, default=Path("data/processed/legal_chunks.json"))
    args = parser.parse_args()
    print(f"Wrote {ingest_file(args.source, args.metadata, args.output)} legal chunks to {args.output}")


if __name__ == "__main__":
    main()
