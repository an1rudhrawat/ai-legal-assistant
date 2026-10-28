"""Offline, corpus-aware evaluation helpers for the local legal retriever."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.issue_classifier import LegalIssueClassifier
from app.situation import SituationUnderstander

from .query_expansion import expand_query
from .retriever import LegalRetriever, RetrievalDebugResult


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    user_query: str
    expected_topic: str
    expected_document: str | None
    expected_sections: tuple[str, ...]
    relevant_concepts: tuple[str, ...]
    expected_sufficient: bool
    in_scope: bool
    category: str
    corpus_note: str

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.expected_document or self.expected_sections)


@dataclass(frozen=True)
class EvaluatedCase:
    case: EvaluationCase
    expanded_query: str | None
    debug: RetrievalDebugResult | None
    matched: bool
    expectation_met: bool


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return tuple(EvaluationCase(
        id=record["id"],
        user_query=record["user_query"],
        expected_topic=record["expected_topic"],
        expected_document=record.get("expected_document"),
        expected_sections=tuple(record.get("expected_sections", [])),
        relevant_concepts=tuple(record.get("relevant_concepts", [])),
        expected_sufficient=bool(record["expected_sufficient"]),
        in_scope=bool(record["in_scope"]),
        category=record["category"],
        corpus_note=record["corpus_note"],
    ) for record in records)


def build_expanded_query(message: str) -> str:
    """Use the same local situation and issue hints as LegalAssistantService."""
    situation = SituationUnderstander().understand(message)
    assessment = LegalIssueClassifier().assess(message)
    return " ".join((expand_query(message, situation), *assessment.categories))


def _matches_expected(case: EvaluationCase, document_name: str, section_number: str | None) -> bool:
    if not case.has_ground_truth:
        return False
    document_matches = not case.expected_document or document_name.casefold() == case.expected_document.casefold()
    section_matches = not case.expected_sections or section_number in case.expected_sections
    return document_matches and section_matches


def evaluate_cases(retriever: LegalRetriever, cases: tuple[EvaluationCase, ...], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Evaluate retrieval, not legal correctness or the LLM's final answer.

    Metrics exclude cases without known corpus evidence. Those cases are still
    reported separately to show whether the retriever properly returns no
    relevant chunks for an unsupported scenario.
    """
    maximum_k = max(ks)
    evaluated: list[EvaluatedCase] = []
    for case in cases:
        if not case.in_scope:
            evaluated.append(EvaluatedCase(case, None, None, False, True))
            continue
        expanded_query = build_expanded_query(case.user_query)
        debug = retriever.debug_search(expanded_query, limit=maximum_k)
        matched = any(_matches_expected(case, chunk.chunk.document_name, chunk.chunk.section_number) for chunk in debug.retrieved if chunk.passed_minimum_relevance)
        expectation_met = matched if case.expected_sufficient else not debug.result.has_relevant_chunks
        evaluated.append(EvaluatedCase(case, expanded_query, debug, matched, expectation_met))

    grounded = [item for item in evaluated if item.case.has_ground_truth and item.case.in_scope]
    metrics: dict[str, dict[str, float]] = {}
    for k in ks:
        recalls: list[float] = []
        precisions: list[float] = []
        hits: list[float] = []
        for item in grounded:
            retrieved = [chunk for chunk in item.debug.retrieved[:k] if chunk.passed_minimum_relevance] if item.debug else []
            relevant = [chunk for chunk in retrieved if _matches_expected(item.case, chunk.chunk.document_name, chunk.chunk.section_number)]
            expected_count = len(item.case.expected_sections) or 1
            recalls.append(min(len(relevant) / expected_count, 1.0))
            precisions.append(len(relevant) / len(retrieved) if retrieved else 0.0)
            hits.append(1.0 if relevant else 0.0)
        metrics[str(k)] = {
            "recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
            "hit": round(sum(hits) / len(hits), 4) if hits else 0.0,
        }

    attempted = [item for item in evaluated if item.debug]
    scores = [score for item in attempted for score in item.debug.result.scores]
    failures = [item for item in attempted if not item.expectation_met]
    failed_by_category: dict[str, int] = {}
    for item in failures:
        failed_by_category[item.case.category] = failed_by_category.get(item.case.category, 0) + 1
    return {
        "summary": {
            "test_cases": len(cases),
            "retrieval_attempted": len(attempted),
            "successful_retrievals": sum(item.debug.result.has_relevant_chunks for item in attempted),
            "failed_retrievals": len(failures),
            "insufficient_evidence": sum(not item.debug.result.has_relevant_chunks for item in attempted),
            "average_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "failures_by_category": failed_by_category,
        },
        "metrics": metrics,
        "cases": [
            {
                "id": item.case.id,
                "category": item.case.category,
                "expanded_query": item.expanded_query,
                "matched_expected_evidence": item.matched,
                "expectation_met": item.expectation_met,
                "corpus_note": item.case.corpus_note,
                "retrieved": [
                    {
                        "document": chunk.chunk.document_name,
                        "section": chunk.chunk.section_number,
                        "score": round(chunk.relevance_score, 4),
                    }
                    for chunk in item.debug.retrieved if chunk.passed_minimum_relevance
                ] if item.debug else [],
            }
            for item in evaluated
        ],
    }


def format_debug(result: RetrievalDebugResult) -> str:
    """Human-readable local debug output for manual retrieval inspection."""
    lines = [
        f"Query: {result.query}",
        f"Index available: {result.index_available}; dense available: {result.dense_available}; reranker active: {result.reranker_active}",
        f"Minimum relevance: {result.minimum_relevance}",
    ]
    for position, item in enumerate(result.retrieved, start=1):
        lines.extend((
            f"Retrieved #{position}: {item.chunk.document_name}, section {item.chunk.section_number or 'not specified'}",
            f"  relevance={item.relevance_score:.4f}; bm25={item.bm25_score:.4f}; term={item.term_score:.4f}; "
            f"dense={'unavailable' if item.dense_score is None else f'{item.dense_score:.4f}'}; "
            f"rrf={item.rrf_score:.4f}; ranking={item.ranking_score:.4f}; accepted={item.passed_minimum_relevance}",
        ))
    return "\n".join(lines)
