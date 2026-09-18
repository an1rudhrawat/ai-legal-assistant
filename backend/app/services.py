from __future__ import annotations

import logging
import re
from collections import defaultdict
from threading import Lock

from app.core.constants import DISCLAIMER, INSUFFICIENT_EVIDENCE_MESSAGE
from app.domain_guard import ResponseScopeValidator
from app.issue_classifier import IssueAssessment, LegalIssueClassifier
from app.llm.base import LLMProvider
from app.prompts import SYSTEM_PROMPT, build_grounded_message
from app.retrieval import LegalRetriever
from app.retrieval.query_expansion import expand_query
from app.schemas import Citation
from app.situation import Situation, SituationUnderstander

logger = logging.getLogger(__name__)


class SessionRequestLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def allow(self, session_id: str) -> bool:
        with self._lock:
            if self._counts[session_id] >= self.limit:
                return False
            self._counts[session_id] += 1
            return True


def ensure_disclaimer(text: str) -> str:
    cleaned = text.strip()
    if DISCLAIMER.lower() in cleaned.lower():
        return cleaned
    return f"{cleaned}\n\n{DISCLAIMER}"


def asks_for_exact_citation(message: str) -> bool:
    return bool(re.search(r"\b(exact|specific)\b.{0,45}\b(section|statute|citation|case law)\b|\bsection number\b", message.lower()))


def has_citation_hedge(text: str) -> bool:
    return bool(re.search(r"\b(cannot (?:confirm|verify)|not certain|uncertain|may|please verify|consult (?:the )?official)\b", text.lower()))


def citations_from_chunks(chunks: tuple) -> list[Citation]:
    # The API exposes human-readable source fields, never internal chunk identifiers.
    seen: set[tuple[str, str | None]] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        key = (chunk.document_name, chunk.section_number)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source_name=chunk.document_name,
                section=chunk.section_number,
                source_url=chunk.source_url,
                document_version=chunk.document_version,
            ))
    return citations


def validate_model_citations(text: str, chunks: tuple) -> bool:
    cited_ids = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", text))
    available_ids = {chunk.chunk_id for chunk in chunks}
    return cited_ids.issubset(available_ids)


class GroundedAnswer:
    def __init__(self, response: str, citations: list[Citation], assessment: IssueAssessment, sufficient: bool, evidence_sufficient: bool | None = None) -> None:
        self.response = response
        self.citations = citations
        self.assessment = assessment
        self.sufficient = sufficient
        # A validator can reject a generated answer after evidence was found.
        # Keep that distinct from a retrieval miss for ambiguity handling.
        self.evidence_sufficient = sufficient if evidence_sufficient is None else evidence_sufficient


class LegalAssistantService:
    def __init__(self, provider: LLMProvider, retriever: LegalRetriever, validator: ResponseScopeValidator | None = None, classifier: LegalIssueClassifier | None = None, understander: SituationUnderstander | None = None) -> None:
        self.provider = provider
        self.retriever = retriever
        self.validator = validator or ResponseScopeValidator()
        self.classifier = classifier or LegalIssueClassifier()
        self.understander = understander or SituationUnderstander()

    async def answer(self, message: str, situation: Situation | None = None) -> GroundedAnswer:
        assessment = self.classifier.assess(message)
        situation = situation or self.understander.understand(message)
        expanded_query = " ".join((expand_query(message, situation), *assessment.categories))
        result = self.retriever.search(expanded_query)
        if not result.sufficient:
            return GroundedAnswer(ensure_disclaimer(INSUFFICIENT_EVIDENCE_MESSAGE), [], assessment, False, False)

        context = "\n\n".join(
            f"Source ID: [{chunk.chunk_id}]\nSource: {chunk.document_name}\n"
            f"Section: {chunk.section_number or 'not specified'}\n{chunk.text}"
            for chunk in result.chunks
        )
        text = await self.provider.generate(system_prompt=SYSTEM_PROMPT, user_message=build_grounded_message(message, context, situation))
        if not validate_model_citations(text, result.chunks):
            logger.warning("model supplied a citation not present in retrieved evidence")
            return GroundedAnswer(ensure_disclaimer(INSUFFICIENT_EVIDENCE_MESSAGE), [], assessment, False, True)
        valid, reason = self.validator.validate(text)
        if not valid:
            logger.warning("response scope validator triggered: %s", reason)
            from app.core.constants import RESPONSE_SCOPE_FALLBACK
            return GroundedAnswer(ensure_disclaimer(RESPONSE_SCOPE_FALLBACK), [], assessment, False, True)
        return GroundedAnswer(ensure_disclaimer(text), citations_from_chunks(result.chunks), assessment, True)
