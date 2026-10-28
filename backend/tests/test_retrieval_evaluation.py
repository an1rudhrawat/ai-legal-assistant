import json
from pathlib import Path

from app.retrieval.embeddings import embed
from app.retrieval.evaluation import EvaluationCase, evaluate_cases, load_evaluation_cases
from app.retrieval.retriever import LegalRetriever


def make_index(tmp_path):
    chunk = {
        "chunk_id": "fir-1", "document_id": "bnss", "document_name": "BNSS", "act_name": "BNSS",
        "document_type": "legislation", "section_number": "173", "section_title": "Information",
        "chapter": None, "subsection": None, "source_url": "https://www.indiacode.nic.in/",
        "issuing_authority": "Government of India", "document_version": "2023", "effective_date": None,
        "publication_date": None, "ingested_at": "2026-01-01T00:00:00+00:00",
        "text": "Police receive information relating to a cognizable offence and register a first information report.",
    }
    index = tmp_path / "index.json"
    index.write_text(json.dumps([{"chunk": chunk, "embedding": embed(chunk["text"])}]), encoding="utf-8")
    return index


def test_debug_search_exposes_local_signals_without_changing_result(tmp_path):
    retriever = LegalRetriever(make_index(tmp_path), enable_dense=False)

    debug = retriever.debug_search("police first information report")

    assert debug.result.has_relevant_chunks
    assert debug.result.sufficient  # Compatibility alias is preserved.
    assert not debug.dense_available
    assert debug.retrieved[0].term_score > 0
    assert debug.retrieved[0].passed_minimum_relevance


def test_missing_index_is_reported_as_no_relevant_chunks(tmp_path):
    debug = LegalRetriever(tmp_path / "missing.json", enable_dense=False).debug_search("police complaint")

    assert not debug.index_available
    assert not debug.result.has_relevant_chunks
    assert debug.retrieved == ()


def test_minilm_and_reranker_fallbacks_leave_local_retrieval_usable(tmp_path):
    retriever = LegalRetriever(make_index(tmp_path), enable_dense=True, enable_reranker=True)
    retriever._dense_load_attempted = True  # Simulate an unavailable local MiniLM model.
    retriever.reranker._load_attempted = True  # Simulate an unavailable local CrossEncoder.

    debug = retriever.debug_search("police first information report")

    assert debug.result.has_relevant_chunks
    assert not debug.dense_available
    assert not debug.reranker_active


def test_evaluation_metrics_distinguish_known_evidence_from_insufficient_corpus(tmp_path):
    retriever = LegalRetriever(make_index(tmp_path), enable_dense=False)
    cases = (
        EvaluationCase("known", "What is a first information report?", "FIR", "BNSS", ("173",), ("FIR",), True, True, "criminal", "present"),
        EvaluationCase("unsupported", "Explain judicial review", "constitutional", None, (), (), False, True, "constitutional", "absent"),
    )

    report = evaluate_cases(retriever, cases)

    assert report["summary"]["test_cases"] == 2
    assert report["summary"]["insufficient_evidence"] == 1
    assert report["metrics"]["1"]["recall"] == 1.0
    assert report["metrics"]["1"]["precision"] == 1.0
    assert report["metrics"]["1"]["hit"] == 1.0
    assert not report["cases"][1]["matched_expected_evidence"]


def test_corpus_aware_dataset_has_required_supported_and_unsupported_cases():
    dataset = Path(__file__).resolve().parents[1] / "evaluation" / "retrieval_cases.json"
    cases = load_evaluation_cases(dataset)

    assert len(cases) >= 12
    assert any(case.id == "threatening-message" and case.expected_sections == ("351",) for case in cases)
    assert any(case.id == "non-legal-recipe" and not case.in_scope for case in cases)
    assert any(not case.expected_sufficient for case in cases)
