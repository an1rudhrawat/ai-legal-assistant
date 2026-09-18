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
