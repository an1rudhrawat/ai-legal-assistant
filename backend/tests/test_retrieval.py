import json

from app.retrieval.embeddings import embed
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import LegalRetriever


def make_index(tmp_path):
    chunks = [
        {"chunk_id": "fir-1", "document_id": "bnss", "document_name": "BNSS", "act_name": "BNSS", "document_type": "legislation", "section_number": "173", "section_title": "Information in cognizable cases", "chapter": "XIII", "subsection": None, "source_url": "https://www.indiacode.nic.in/", "issuing_authority": "Government of India", "document_version": "2023", "effective_date": None, "publication_date": None, "ingested_at": "2026-01-01T00:00:00+00:00", "text": "Police receive information relating to a cognizable offence and register a first information report."},
        {"chunk_id": "tenant-1", "document_id": "tenant", "document_name": "Official tenancy guidance", "act_name": "Tenancy", "document_type": "official guidance", "section_number": None, "section_title": "Eviction", "chapter": None, "subsection": None, "source_url": "https://example.gov.in/", "issuing_authority": "State Government", "document_version": "2023", "effective_date": None, "publication_date": None, "ingested_at": "2026-01-01T00:00:00+00:00", "text": "A landlord tenant eviction dispute may require notice and a legal process."},
    ]
    path = tmp_path / "index.json"
    path.write_text(json.dumps([{"chunk": item, "embedding": embed(item["text"])} for item in chunks]), encoding="utf-8")
    return path


def test_situation_query_retrieves_relevant_material_and_provenance(tmp_path):
    result = LegalRetriever(make_index(tmp_path), enable_dense=False).search("The police will not register my FIR")
    assert result.sufficient
    assert result.chunks[0].section_number == "173"
    assert result.chunks[0].issuing_authority == "Government of India"
    assert result.scores[0] > 0


def test_terminology_query_retrieves_material(tmp_path):
    result = LegalRetriever(make_index(tmp_path), enable_dense=False).search("What is a first information report?")
    assert result.sufficient
    assert result.chunks[0].chunk_id == "fir-1"


def test_explicit_section_query_prioritises_exact_section_metadata(tmp_path):
    index = make_index(tmp_path)
    records = json.loads(index.read_text(encoding="utf-8"))
    unrelated_bns_section = dict(records[0]["chunk"])
    unrelated_bns_section.update({
        "chunk_id": "bns-section-173",
        "document_id": "bns",
        "document_name": "Bharatiya Nyaya Sanhita, 2023",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "173",
        "section_title": "An unrelated provision",
        "text": "This provision concerns an unrelated criminal offence.",
    })
    records.append({"chunk": unrelated_bns_section, "embedding": embed(unrelated_bns_section["text"])})
    for number in range(60):
        distractor = dict(records[0]["chunk"])
        distractor.update({
            "chunk_id": f"distractor-{number}",
            "document_id": "other-act",
            "document_name": "Other Act",
            "act_name": "Other Act",
            "section_number": str(number + 1),
            "section_title": "Unrelated procedural provision",
            "text": f"A wholly unrelated rule about topic {number}.",
        })
        records.append({"chunk": distractor, "embedding": embed(distractor["text"])})
    index.write_text(json.dumps(records), encoding="utf-8")

    debug = LegalRetriever(index, enable_dense=False).debug_search(
        "What is the heading of Bharatiya Nyaya Sanhita section 173?", limit=2
    )

    assert debug.result.chunks[0].chunk_id == "bns-section-173"
    assert len(debug.result.chunks) == 1
    assert all(chunk.chunk.section_number == "173" for chunk in debug.retrieved)
    assert debug.retrieved[0].exact_section_match
    assert debug.retrieved[0].relevance_score == 1.0


def test_irrelevant_query_has_insufficient_evidence(tmp_path):
    result = LegalRetriever(make_index(tmp_path), minimum_relevance=0.2, enable_dense=False).search("pasta recipe ingredients")
    assert not result.sufficient
    assert result.chunks == ()


def test_relevance_threshold_is_applied(tmp_path):
    index = make_index(tmp_path)
    assert LegalRetriever(index, minimum_relevance=0.01, enable_dense=False).search("police FIR").sufficient
    assert not LegalRetriever(index, minimum_relevance=0.99, enable_dense=False).search("police FIR").sufficient


def test_bm25_scores_exact_statutory_terms():
    scores = BM25Retriever(["police receive a first information report", "landlord tenant eviction", "court procedure notice"]).scores("first information report")
    assert scores[0] > scores[1]


def test_rrf_promotes_documents_found_by_multiple_retrievers():
    fused = LegalRetriever._rrf([[0, 1, 2], [1, 0, 3]])
    assert fused[0] > fused[2]
    assert fused[1] > fused[3]


def test_reranking_preserves_the_correct_chunk_when_text_is_identical(tmp_path):
    index = make_index(tmp_path)
    records = json.loads(index.read_text(encoding="utf-8"))
    duplicate = dict(records[0]["chunk"])
    duplicate["chunk_id"] = "fir-duplicate"
    duplicate["document_id"] = "different-document"
    duplicate["document_name"] = "Different official source"
    records.append({"chunk": duplicate, "embedding": records[0]["embedding"]})
    index.write_text(json.dumps(records), encoding="utf-8")

    retriever = LegalRetriever(index, enable_dense=False)
    # Model output ordering is represented by the reranker result; it must retain
    # the original record index rather than recover it from duplicate text.
    def select_duplicate(_query, candidates):
        duplicate_candidate = next(candidate for candidate in candidates if candidate[1] == 2)
        return [duplicate_candidate, *(candidate for candidate in candidates if candidate[1] != 2)]

    retriever.reranker.rerank = select_duplicate

    result = retriever.search("police first information report", limit=1)

    assert result.sufficient
    assert result.chunks[0].chunk_id == "fir-duplicate"
