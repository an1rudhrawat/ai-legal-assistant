from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    document_id: str
    document_name: str
    act_name: str
    document_type: str
    section_number: str | None
    section_title: str | None
    chapter: str | None
    subsection: str | None
    source_url: str
    issuing_authority: str
    document_version: str | None
    effective_date: str | None
    publication_date: str | None
    ingested_at: str
    text: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> "LegalChunk":
        return cls(**data)
