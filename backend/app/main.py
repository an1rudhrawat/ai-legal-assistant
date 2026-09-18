from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.core.constants import AMBIGUOUS_SITUATION_MESSAGE, EMERGENCY_MESSAGE, OUT_OF_SCOPE_MESSAGE, RATE_LIMIT_MESSAGE, SESSION_LIMIT_MESSAGE
from app.domain_guard import LocalSemanticDomainClassifier, ObviousOutOfScopeGate
from app.llm import ProviderError, ProviderRateLimitError, create_provider
from app.retrieval import LegalRetriever, sync_corpus
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services import LegalAssistantService, SessionRequestLimiter, ensure_disclaimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Legal First Responder", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.state.out_of_scope_gate = ObviousOutOfScopeGate()
    app.state.domain_classifier = LocalSemanticDomainClassifier(
        settings.domain_classifier_model_path,
        settings.embedding_model,
        settings.domain_legal_threshold,
        settings.domain_non_legal_threshold,
    )
    app.state.service = LegalAssistantService(
        create_provider(settings),
        LegalRetriever(settings.retrieval_index_path, settings.retrieval_minimum_relevance, settings.embedding_model, settings.reranker_model, settings.enable_reranker, settings.enable_dense_retrieval),
    )
    app.state.limiter = SessionRequestLimiter(settings.session_request_limit)

    @app.on_event("startup")
    async def sync_retrieval_corpus() -> None:
        result = refresh_retrieval_corpus(settings)
        logger.info("RAG corpus synced: %s added, %s updated, %s unchanged, %s chunks", result.added, result.updated, result.unchanged, result.indexed_chunks)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        classification = app.state.domain_classifier.classify(request.message)
        if app.state.out_of_scope_gate.rejects(request.message) or classification.label == "non_legal":
            logger.info("domain classifier rejected query: %s", classification.reason)
            return ChatResponse(response=ensure_disclaimer(OUT_OF_SCOPE_MESSAGE), refused=True, domain_label="non_legal")
        if not app.state.limiter.allow(request.session_id):
            return ChatResponse(response=ensure_disclaimer(SESSION_LIMIT_MESSAGE), refused=True, rate_limited=True)
        situation = app.state.service.understander.understand(request.message)
        assessment = app.state.service.classifier.assess(request.message)
        if assessment.emergency or situation.urgent:
            return ChatResponse(
                response=ensure_disclaimer(EMERGENCY_MESSAGE),
                urgency="emergency",
                issue_categories=list(assessment.categories),
                domain_label=classification.label,
            )
        # A semantically uncertain request with no extracted event must not be
        # allowed to retrieve a coincidental statute and invent a legal context.
        if classification.label == "ambiguous" and not situation.events:
            return ChatResponse(response=ensure_disclaimer(AMBIGUOUS_SITUATION_MESSAGE), domain_label="ambiguous")
        refresh_retrieval_corpus(settings)
        try:
            answer = await app.state.service.answer(request.message, situation)
            if classification.label == "ambiguous" and not answer.evidence_sufficient:
                return ChatResponse(response=ensure_disclaimer(AMBIGUOUS_SITUATION_MESSAGE), domain_label="ambiguous")
            return ChatResponse(
                response=answer.response,
                urgency="urgent" if answer.assessment.urgent else None,
                issue_categories=list(answer.assessment.categories),
                citations=answer.citations,
                domain_label=classification.label,
            )
        except ProviderRateLimitError:
            logger.warning("LLM provider rate limit encountered")
            return ChatResponse(response=ensure_disclaimer(RATE_LIMIT_MESSAGE), rate_limited=True)
        except ProviderError:
            logger.exception("LLM provider failed")
            return ChatResponse(response=ensure_disclaimer("The assistant is unavailable right now. Please try again later."), rate_limited=False)

    return app


def refresh_retrieval_corpus(settings: Settings):
    """Pick up added or changed files without requiring an API restart."""
    return sync_corpus(settings.retrieval_raw_path, settings.retrieval_processed_path, settings.retrieval_index_path)


app = create_app()
