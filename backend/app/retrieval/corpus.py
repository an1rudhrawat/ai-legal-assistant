"""Incremental, file-based corpus synchronisation for the local RAG index."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .index import build_index_from_chunks
from .ingest import ingest_file


MANIFEST_NAME = ".rag_manifest.json"
SOURCE_SUFFIXES = {".json", ".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class CorpusSyncResult:
    added: int
    updated: int
    unchanged: int
    indexed_chunks: int


def _metadata_path(source: Path) -> Path | None:
    for suffix in (".metadata.json", ".meta.json"):
        candidate = source.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _is_metadata_file(path: Path) -> bool:
    return path.name.endswith((".metadata.json", ".meta.json"))


def _sources(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not _is_metadata_file(path)
        # The repository keeps upload instructions in this directory; they are
        # not source material and must never enter the legal evidence index.
        and path.name.lower() != "readme.md"
    )


def _digest(source: Path, metadata: Path | None) -> str:
    hasher = hashlib.sha256()
    hasher.update(source.read_bytes())
    if metadata:
        hasher.update(b"\0metadata\0")
        hasher.update(metadata.read_bytes())
    return hasher.hexdigest()


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Corpus manifest is invalid")
    return value


def sync_corpus(raw_dir: Path, processed_dir: Path, index_path: Path) -> CorpusSyncResult:
    """Chunk only added/changed files, then index all active persisted chunks.

    The manifest key is the path relative to ``raw_dir``. A content hash includes
    the companion metadata file, so editing either deliberately reprocesses only
    that source. Deleted sources are excluded from the rebuilt index.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = processed_dir / "documents"
    documents_dir.mkdir(exist_ok=True)
    manifest_path = processed_dir / MANIFEST_NAME
    previous = _load_manifest(manifest_path)
    current: dict[str, dict[str, str]] = {}
    added = updated = unchanged = 0

    for source in _sources(raw_dir):
        relative = source.relative_to(raw_dir).as_posix()
        metadata = _metadata_path(source)
        digest = _digest(source, metadata)
        output_name = hashlib.sha256(relative.encode("utf-8")).hexdigest() + ".json"
        output = documents_dir / output_name
        old = previous.get(relative)
        current[relative] = {"digest": digest, "chunks_file": output_name}
        if old and old.get("digest") == digest and output.exists():
            unchanged += 1
            continue
        ingest_file(source, metadata, output)
        if old:
            updated += 1
        else:
            added += 1

    all_chunks: list[dict] = []
    for relative in sorted(current):
        chunks_file = documents_dir / current[relative]["chunks_file"]
        all_chunks.extend(json.loads(chunks_file.read_text(encoding="utf-8")))

    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate document_id values create duplicate chunk IDs; give each source a unique document_id")
    corpus_changed = current != previous
    if all_chunks and (corpus_changed or not index_path.exists()):
        build_index_from_chunks(all_chunks, index_path)
    elif not all_chunks and corpus_changed and index_path.exists():
        index_path.unlink()

    manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    return CorpusSyncResult(added, updated, unchanged, len(all_chunks))
