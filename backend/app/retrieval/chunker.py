from __future__ import annotations

import re
from datetime import UTC, datetime

from .models import LegalChunk

# Restrict indentation to spaces/tabs: ``\s*`` can cross newlines and turn a
# numbered paragraph several lines later into a false section heading.
SECTION_LINE = re.compile(r"^[ \t]*(\d{1,3}[A-Za-z]?)\.[ \t]*([^\r\n]+)[ \t]*$", re.MULTILINE)
ENACTMENT_MARKER = re.compile(r"(?im)^[ \t]*BE[ \t]+it[ \t]+enacted[ \t]+by\b")
PDF_DASH_MARKERS = "\u2013\u2014\ufffd"


def _section_body(text: str) -> str:
    """Drop a statute's cover, table of contents and legislative preamble."""
    match = ENACTMENT_MARKER.search(text)
    return text[match.start():] if match else text


def _section_candidates(text: str) -> list[re.Match[str]]:
    candidates: list[re.Match[str]] = []
    previous: tuple[int, str] | None = None
    has_enactment = bool(ENACTMENT_MARKER.search(text))
    for match in SECTION_LINE.finditer(text):
        raw_number = match.group(1)
        number_match = re.fullmatch(r"(\d+)([A-Za-z]?)", raw_number)
        if not number_match:
            continue
        key = (int(number_match.group(1)), number_match.group(2).lower())
        if previous is None:
            # An enacted statute's operative body should begin with section 1.
            if has_enactment and key[0] != 1:
                continue
        elif key <= previous:
            # Ignore page footnotes and repeated TOC-like numbers in body text.
            continue
        candidates.append(match)
        previous = key
    return candidates


def _heading_title(line: str, continuation: str = "") -> str:
    """Extract a wrapped heading and strip the PDF heading/body separator."""
    title = line.lstrip(f" \t\u00a0{PDF_DASH_MARKERS}")

    def body_separator(value: str) -> re.Match[str] | None:
        # The PDF text layer uses en/em dashes after heading punctuation, and
        # sometimes puts the separator on a wrapped continuation line.
        return re.search(rf"(?<=[.!?])[ \t\u00a0]*[{PDF_DASH_MARKERS}]+", value)

    separator = body_separator(title)
    if separator:
        title = title[:separator.start()]
    else:
        for next_line in continuation.splitlines()[:4]:
            stripped = next_line.strip()
            if not stripped or re.fullmatch(r"\d+", stripped):
                continue
            if SECTION_LINE.match(next_line) or re.match(r"^\(\s*1\s*\)", stripped):
                break
            title = f"{title} {stripped}"
            separator = body_separator(title)
            if separator:
                title = title[:separator.start()]
                break
            if re.search(r"[ \t]+\(1\)(?:[ \t]|$)", title):
                title = re.split(r"[ \t]+\(1\)", title, maxsplit=1)[0]
                break
            if re.search(r"[.!?][ \t]*$", title):
                break
    title = re.sub(r"\s+", " ", title).strip(f" \t\u00a0{PDF_DASH_MARKERS}")
    title = re.sub(r"\s+([,.;:])", r"\1", title)
    return title


def chunk_document(text: str, metadata: dict[str, str | None]) -> list[LegalChunk]:
    """Chunk operative numbered sections, excluding an Act's contents pages."""
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        raise ValueError("Document text is empty")

    operative = _section_body(cleaned)
    sections = _section_candidates(operative)
    parts: list[tuple[str | None, str | None, str]] = []
    if sections:
        for index, match in enumerate(sections):
            end = sections[index + 1].start() if index + 1 < len(sections) else len(operative)
            title = _heading_title(match.group(2), operative[match.end():end])
            parts.append((match.group(1), title or None, operative[match.start():end].strip()))
    else:
        # Preserve a source that has no parseable numbered sections as one
        # searchable chunk rather than dropping its contents.
        parts = [(None, None, operative)]

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
