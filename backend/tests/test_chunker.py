from app.retrieval.chunker import chunk_document


METADATA = {
    "document_id": "sample-act",
    "document_name": "Sample Act",
    "act_name": "Sample Act",
    "document_type": "legislation",
    "source_url": "local",
    "issuing_authority": "not supplied",
}


def test_act_contents_and_numbered_preamble_are_not_indexed_as_sections():
    source = """ARRANGEMENT OF SECTIONS
1. Short title, commencement and application.
2. Definitions.
3. General explanations.
The government reviewed the law:
3. First, it consulted the public.
4. Next, it prepared a bill.
BE it enacted by Parliament as follows:
CHAPTER I
1. Short title, commencement and application. ��(1) This Act may be called the Sample Act.
1. 1st day of July, as appointed by notification.
2. Definitions. ��In this Act, these words have the following meanings.
3. General explanations. ��Unless the context requires otherwise.
"""

    chunks = chunk_document(source, METADATA)

    assert [chunk.section_number for chunk in chunks] == ["1", "2", "3"]
    assert [chunk.section_title for chunk in chunks] == [
        "Short title, commencement and application.",
        "Definitions.",
        "General explanations.",
    ]
    assert "reviewed the law" not in chunks[0].text
    assert "1st day of July" in chunks[0].text


def test_non_section_source_is_kept_as_one_chunk():
    chunks = chunk_document("A short official guidance document without numbered sections.", METADATA)

    assert len(chunks) == 1
    assert chunks[0].section_number is None
    assert "without numbered sections" in chunks[0].text


def test_chunk_ids_are_repeatable_and_unique_within_a_document():
    text = "BE it enacted by Parliament as follows:\n1. First. Text.\n2. Second. Text."

    first = chunk_document(text, METADATA)
    second = chunk_document(text, METADATA)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
