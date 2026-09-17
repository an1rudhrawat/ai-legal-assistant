from __future__ import annotations

import re
from datetime import UTC, datetime

from .models import LegalChunk

SECTION_PATTERN = re.compile(
    r"(?ms)^\s*(\d+[A-Za-z]?)\.\s+([^\n]+)\n(.*?)(?=^\s*\d+[A-Za-z]?\.\s+|\Z)" 
)


def chunk_document(text: str, metadata: dict[str, str | None]) -> list[LegalChunk]:
    """Keep Act sections intact instead of splitting on an arbitrary character count."""
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        raise ValueError("Document text is empty")

    sections = list(SECTION_PATTERN.finditer(cleaned))
    parts = [(match.group(1), match.group(2).strip(), match.group(0).strip()) for match in sections]
    if not parts:
        parts = [(None, None, cleaned)]

    ingested_at = datetime.now(UTC).isoformat()
    chunks: list[LegalChunk] = []
    for index, (section, title, content) in enumerate(parts, start=1):
        chunks.append(LegalChunk(
            chunk_id=f"{metadata['document_id']}-chunk-{index}",
            document_id=metadata["document_id"] or "unknown-document",
            document_name=metadata["document_name"] or "Unnamed document",
            act_name=metadata["act_name"] or metadata["document_name"] or "Unnamed Act",
            document_type=metadata["document_type"] or "legislation",
            section_number=section or metadata.get("section_number"),
            section_title=title or metadata.get("section_title"),
            chapter=metadata.get("chapter"),
            subsection=metadata.get("subsection"),
            source_url=metadata["source_url"] or "",
            issuing_authority=metadata["issuing_authority"] or "",
            document_version=metadata.get("document_version"),
            effective_date=metadata.get("effective_date"),
            publication_date=metadata.get("publication_date"),
            ingested_at=ingested_at,
            text=content,
        ))
    return chunks
