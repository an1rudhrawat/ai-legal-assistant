import json

from app.retrieval.corpus import sync_corpus


def write_source(path, document_id: str, text: str) -> None:
    path.write_text(json.dumps({
        "document_id": document_id,
        "document_name": document_id,
        "act_name": document_id,
        "document_type": "legislation",
        "source_url": "https://www.indiacode.nic.in/",
        "issuing_authority": "Government of India",
        "text": text,
    }), encoding="utf-8")


def test_sync_preserves_unchanged_chunks_and_only_chunks_new_sources(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    index = tmp_path / "index.json"
    raw.mkdir()
    first = raw / "first.json"
    write_source(first, "first-act", "1. First section\nThe first authoritative rule.")

    initial = sync_corpus(raw, processed, index)
    first_chunk_file = next((processed / "documents").glob("*.json"))
    original_chunks = first_chunk_file.read_bytes()
    assert (initial.added, initial.updated, initial.unchanged, initial.indexed_chunks) == (1, 0, 0, 1)

    repeat = sync_corpus(raw, processed, index)
    assert (repeat.added, repeat.updated, repeat.unchanged) == (0, 0, 1)
    assert first_chunk_file.read_bytes() == original_chunks

    write_source(raw / "second.json", "second-act", "1. Second section\nThe second authoritative rule.")
    later = sync_corpus(raw, processed, index)
    assert (later.added, later.updated, later.unchanged, later.indexed_chunks) == (1, 0, 1, 2)
    assert first_chunk_file.read_bytes() == original_chunks
    assert len(json.loads(index.read_text(encoding="utf-8"))) == 2


def test_sync_accepts_a_single_pdf_or_text_upload_without_companion_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "Bharatiya-Nyaya-Sanhita-2023.txt").write_text("1. Short title\nThe Act applies in India.", encoding="utf-8")

    result = sync_corpus(raw, tmp_path / "processed", tmp_path / "index.json")

    assert result.indexed_chunks == 1
    chunk = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))[0]["chunk"]
    assert chunk["document_id"] == "bharatiya-nyaya-sanhita-2023"
    assert chunk["source_url"] == "Local upload: Bharatiya-Nyaya-Sanhita-2023.txt"


def test_sync_ignores_the_raw_directory_readme(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "README.md").write_text("Instructions for uploading source material.", encoding="utf-8")
    write_source(raw / "act.json", "test-act", "1. Short title\nAn authoritative rule.")

    result = sync_corpus(raw, tmp_path / "processed", tmp_path / "index.json")

    assert result.added == 1
    assert result.indexed_chunks == 1
