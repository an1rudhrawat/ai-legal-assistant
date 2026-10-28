from __future__ import annotations

import logging
import re
from collections import defaultdict
from threading import Lock
from typing import Sequence

from app.core.constants import AMBIGUOUS_SITUATION_MESSAGE, DISCLAIMER, INSUFFICIENT_EVIDENCE_MESSAGE
from app.domain_guard import ResponseScopeValidator
from app.issue_classifier import IssueAssessment, LegalIssueClassifier
from app.llm.base import LLMProvider, ProviderError
from app.prompts import SYSTEM_PROMPT, build_grounded_message
from app.retrieval import LegalRetriever
from app.retrieval.query_expansion import expand_query
from app.schemas import Citation
from app.situation import Situation, SituationUnderstander

logger = logging.getLogger(__name__)
SECTION_REFERENCE_RE = re.compile(
    r"\b(?:section|sec\.?|s\.)[\s:*_`#-]*(?:no\.?\s*)?(\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)

SUMMARY_SYSTEM_PROMPT = """Create a concise continuity summary for a legal-information chat.
Use only facts and questions explicitly stated by the user. Treat all user statements as unverified,
preserve uncertainty and qualifiers, and retain important parties, dates, amounts, deadlines,
safety concerns, and unresolved questions. Do not add legal conclusions or repeat previous assistant
legal claims as established facts. The summary is context for follow-up interpretation only, never
legal evidence. Keep it under 180 words; do not omit critical details to meet the limit."""


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


def retrieval_query_with_previous_reply(
    query: str,
    history: Sequence[dict[str, str]],
    *,
    max_reply_chars: int = 1200,
) -> str:
    """Use the latest assistant reply to resolve follow-up references for search.

    Prior replies are fallible conversation context, never legal evidence. The
    retriever uses this text only to find current corpus chunks; generation is
    still grounded in those chunks.
    """
    refers_to_context = re.search(
        r"\b(?:this|that|it|they|them|those|these|above|previous|earlier|same)\b"
        r"|\bwhat\s+(?:if|about)\b|\bhow\s+about\b|\bhow\s+so\b"
        r"|\bcan\s+you\s+(?:explain|elaborate|clarify|expand)\s+"
        r"(?:on\s+)?(?:that|this|it|further|more)\b",
        query,
        re.IGNORECASE,
    )
    if not refers_to_context:
        return query
    previous_reply = next(
        (item["content"] for item in reversed(history) if item.get("role") == "assistant"),
        "",
    )
    if not previous_reply:
        return query
    previous_reply = previous_reply.replace(DISCLAIMER, "")
    previous_reply = re.sub(r"\[[A-Za-z0-9_-]+\]", "", previous_reply)
    previous_reply = re.sub(r"\s+", " ", previous_reply).strip()[:max_reply_chars]
    if not previous_reply:
        return query
    return f"{query}\nPrevious assistant reply (unverified retrieval context): {previous_reply}"


FACT_DEPENDENT_QUERY = re.compile(
    r"(?:\b(?:can i|could i|am i|will i|does this|would this)\b.{0,100}"
    r"\b(?:claim|invoke|rely on|qualify|be liable|be guilty|be punished|"
    r"private defen[cs]e|self[ -]?defen[cs]e|illegal|an offence|an offense|arrest|detain)\b"
    r"|\bcan\s+(?:the\s+)?(?:police|they)\b.{0,80}\b(?:arrest|detain)\b"
    r"|\b(?:was|is)\s+(?:what they did|that|this)\b.{0,80}"
    r"\b(?:an? offence|an? offense|illegal|criminal)\b)",
    re.IGNORECASE,
)


def needs_situation_clarification(message: str, situation: Situation) -> bool:
    """Do not turn a fact-free liability question into a legal conclusion."""
    generic_police_event = "police or enforcement interaction" in situation.events
    bare_arrest_question = re.search(r"\bcan\s+(?:the\s+)?police\b.{0,80}\b(?:arrest|detain)\b", message, re.IGNORECASE)
    no_described_event = not situation.events or (generic_police_event and bare_arrest_question)
    return no_described_event and bool(FACT_DEPENDENT_QUERY.search(message))


def citations_from_chunks(chunks: tuple, text: str) -> list[Citation]:
    cited_ids = set(re.findall(
        r"\[\s*(?:source(?:\s+identifier)?\s*:\s*)?([A-Za-z0-9_-]+)\s*\]",
        text,
        re.IGNORECASE,
    ))
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        if chunk.chunk_id not in cited_ids:
            continue
        key = chunk.chunk_id
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source_id=chunk.chunk_id,
                source_name=chunk.document_name,
                section=chunk.section_number,
                source_url=chunk.source_url,
                document_version=chunk.document_version,
            ))
    return citations


def repair_exact_heading_citation(
    message: str,
    text: str,
    chunks: tuple,
) -> tuple[str, object] | None:
    """Supply a canonical ID only for an exact, metadata-backed heading answer.

    This does not accept a model-supplied unknown ID. It discards the model
    citation and emits a citation from a unique retrieved chunk only when the
    entire cleaned answer is exactly that chunk's section title.
    """
    if not re.search(r"\b(?:heading|title)\b", message, re.IGNORECASE):
        return None
    if re.search(r"\b(?:explain|describe|why|conditions?|punishment|penalty|exceptions?)\b", message, re.IGNORECASE):
        return None

    body = text.replace(DISCLAIMER, "")
    body = re.sub(r"(?im)^\s*(?:source(?:\s+identifier)?|citation)\s*:.*$", "", body)
    body = re.sub(r"\[[^\]]*\]", "", body)
    body = re.sub(r"[*_`#]", "", body)
    body = re.sub(r"(?im)^\s*(?:heading(?:\s+of\s+[^:\n]+)?|section\s+heading|title)\s*:\s*", "", body)
    body = re.sub(r"(?im)^\s*section\s*:?\s*\d+[A-Za-z]?\s*[:—-]?\s*", "", body)
    body = re.sub(r"(?i)^\s*the\s+heading\s+of\s+[^\n:]{0,120}\s+is\s*:\s*", "", body)
    answer_key = " ".join(re.findall(r"[a-z0-9]+", body.casefold()))
    if not answer_key:
        return None

    matches = [
        chunk for chunk in chunks
        if chunk.section_title
        and " ".join(re.findall(r"[a-z0-9]+", chunk.section_title.casefold())) == answer_key
        and answer_key in " ".join(re.findall(r"[a-z0-9]+", chunk.text.casefold()))
    ]
    if len(matches) != 1:
        return None
    chunk = matches[0]
    include_number = bool(re.search(r"\bwhich\s+section\b|\bwhat\s+section\b", message, re.IGNORECASE))
    citation = f"[{chunk.chunk_id}]"
    response = (
        f"Section {chunk.section_number}: {chunk.section_title} {citation}"
        if include_number else f"Section heading: {chunk.section_title} {citation}"
    )
    return response, chunk


def validate_model_citations(text: str, chunks: tuple) -> bool:
    cited_ids = set(re.findall(
        r"\[\s*(?:source(?:\s+identifier)?\s*:\s*)?([A-Za-z0-9_-]+)\s*\]",
        text,
        re.IGNORECASE,
    ))
    available_ids = {chunk.chunk_id for chunk in chunks}
    if not cited_ids or not cited_ids.issubset(available_ids):
        return False
    # Source IDs are not enough when a model states a section number in prose.
    # Accept only section references present in the retrieved evidence.
    referenced_sections = set(SECTION_REFERENCE_RE.findall(text))
    available_sections = {chunk.section_number for chunk in chunks if chunk.section_number}
    for chunk in chunks:
        available_sections.update(SECTION_REFERENCE_RE.findall(chunk.text))
    return referenced_sections.issubset(available_sections)


def validate_model_claims(text: str, chunks: tuple = ()) -> bool:
    """Reject narrow deterministic risks; this is not a semantic legal verifier."""
    exhaustive_absence = re.compile(
        r"\b(?:no such|no)\s+(?:(?:supreme|high)\s+court\s+)?"
        r"(?:case|law|statute|remedy|right|deadline)\b.{0,100}\bexists?\b",
        re.IGNORECASE | re.DOTALL,
    )
    if exhaustive_absence.search(text):
        return False

    # A concrete duration or monetary amount stated as a punishment is easy to
    # compare against supplied text. This deliberately narrow check avoids
    # treating a retrieved section number as evidence for an unstated penalty.
    punishment_amount = re.compile(
        r"\b(?:imprisonment|jail|fine|penalty|punishment|punished)\b"
        r"[^.!?\n]{0,120}?\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred|thousand)\s+"
        r"(?:years?|months?|weeks?|days?|rupees?|lakhs?|crores?)\b",
        re.IGNORECASE,
    )
    evidence = " ".join(chunk.text.casefold() for chunk in chunks)
    for match in punishment_amount.finditer(text):
        amount = re.search(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred|thousand)\s+"
            r"(?:years?|months?|weeks?|days?|rupees?|lakhs?|crores?)\b",
            match.group(0),
            re.IGNORECASE,
        )
        if amount and amount.group(0).casefold() not in evidence:
            return False
    return True


class GroundedAnswer:
    def __init__(self, response: str, citations: list[Citation], assessment: IssueAssessment, sufficient: bool, evidence_sufficient: bool | None = None, citation_repaired: bool = False) -> None:
        self.response = response
        self.citations = citations
        self.assessment = assessment
        self.sufficient = sufficient
        # A validator can reject a generated answer after evidence was found.
        # Keep that distinct from a retrieval miss for ambiguity handling.
        self.evidence_sufficient = sufficient if evidence_sufficient is None else evidence_sufficient
        self.citation_repaired = citation_repaired


class LegalAssistantService:
    def __init__(self, provider: LLMProvider, retriever: LegalRetriever, validator: ResponseScopeValidator | None = None, classifier: LegalIssueClassifier | None = None, understander: SituationUnderstander | None = None) -> None:
        self.provider = provider
        self.retriever = retriever
        self.validator = validator or ResponseScopeValidator()
        self.classifier = classifier or LegalIssueClassifier()
        self.understander = understander or SituationUnderstander()

    async def summarize_history(
        self,
        existing_summary: str,
        rolled_messages: Sequence[dict],
    ) -> str:
        user_statements = "\n".join(
            f"- {item['content']}" for item in rolled_messages if item["role"] == "user"
        )
        if not user_statements:
            return existing_summary
        summary_input = (
            f"Existing unverified continuity summary:\n{existing_summary or '(none)'}"
            f"\n\nAdditional user messages, verbatim and unverified:\n{user_statements}"
        )
        summary = await self.provider.generate(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_message=summary_input,
        )
        if not summary.strip() or len(summary) > 2400 or len(summary.split()) > 180:
            raise ProviderError("The conversation summary was empty or exceeded its safe size limit")
        return summary.strip()

    async def answer(
        self,
        message: str,
        situation: Situation | None = None,
        *,
        retrieval_query: str | None = None,
        history: Sequence[dict[str, str]] = (),
        summary: str = "",
    ) -> GroundedAnswer:
        retrieval_query = retrieval_query or message
        assessment = self.classifier.assess(retrieval_query)
        situation = situation or self.understander.understand(retrieval_query)
        if needs_situation_clarification(message, situation):
            return GroundedAnswer(
                ensure_disclaimer(AMBIGUOUS_SITUATION_MESSAGE),
                [],
                assessment,
                False,
                False,
            )
        search_query = retrieval_query_with_previous_reply(retrieval_query, history)
        expanded_query = " ".join((expand_query(search_query, situation), *assessment.categories))
        result = self.retriever.search(expanded_query)
        if not result.has_relevant_chunks:
            return GroundedAnswer(ensure_disclaimer(INSUFFICIENT_EVIDENCE_MESSAGE), [], assessment, False, False)

        context = "\n\n".join(
            f"Source ID: [{chunk.chunk_id}]\nSource: {chunk.document_name}\n"
            f"Act: {chunk.act_name}\n"
            f"Section: {chunk.section_number or 'not specified'}\n"
            f"Section title: {chunk.section_title or 'not specified'}\n"
            f"Document type: {chunk.document_type}\n"
            f"Issuing authority: {chunk.issuing_authority or 'not supplied'}\n"
            f"Document version: {chunk.document_version or 'not supplied'}\n"
            f"Effective date: {chunk.effective_date or 'not supplied'}\n"
            f"Publication date: {chunk.publication_date or 'not supplied'}\n"
            f"Source URL (provenance only; do not reproduce): {chunk.source_url or 'not supplied'}\n"
            f"Legal text:\n{chunk.text}"
            for chunk in result.chunks
        )
        continuity: list[dict[str, str]] = []
        if summary:
            continuity.append({
                "role": "user",
                "content": (
                    "Continuity summary for context only; user-reported and unverified, "
                    "not legal evidence:\n" + summary
                ),
            })
        continuity.extend(history)
        text = await self.provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_message=build_grounded_message(
                message,
                context,
                situation,
                tuple(chunk.chunk_id for chunk in result.chunks),
            ),
            history=continuity,
        )
        citation_repaired = False
        if not validate_model_citations(text, result.chunks):
            repaired = repair_exact_heading_citation(message, text, result.chunks)
            if repaired is None:
                logger.warning("model supplied a citation not present in retrieved evidence")
                return GroundedAnswer(ensure_disclaimer(INSUFFICIENT_EVIDENCE_MESSAGE), [], assessment, False, True)
            text, repaired_chunk = repaired
            if not validate_model_citations(text, (repaired_chunk,)):
                logger.error("canonical heading citation repair failed its citation validator")
                return GroundedAnswer(ensure_disclaimer(INSUFFICIENT_EVIDENCE_MESSAGE), [], assessment, False, True)
            citation_repaired = True
            logger.warning("replaced model citation with verified chunk ID after exact section-title match: %s", repaired_chunk.chunk_id)
        if not validate_model_claims(text, result.chunks):
            logger.warning("model made an exhaustive legal absence claim that top-k evidence cannot establish")
            return GroundedAnswer(ensure_disclaimer(INSUFFICIENT_EVIDENCE_MESSAGE), [], assessment, False, True)
        valid, reason = self.validator.validate(text)
        if not valid:
            logger.warning("response scope validator triggered: %s", reason)
            from app.core.constants import RESPONSE_SCOPE_FALLBACK
            return GroundedAnswer(ensure_disclaimer(RESPONSE_SCOPE_FALLBACK), [], assessment, False, True)
        return GroundedAnswer(ensure_disclaimer(text), citations_from_chunks(result.chunks, text), assessment, True, citation_repaired=citation_repaired)
