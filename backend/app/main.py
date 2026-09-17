from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.core.constants import EMERGENCY_MESSAGE, OUT_OF_SCOPE_MESSAGE, RATE_LIMIT_MESSAGE, SESSION_LIMIT_MESSAGE
from app.domain_guard import LegalDomainGuard
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
    app.state.guard = LegalDomainGuard()
    app.state.service = LegalAssistantService(
        create_provider(settings),
        LegalRetriever(settings.retrieval_index_path, settings.retrieval_minimum_relevance),
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
        decision = app.state.guard.evaluate(request.message)
        if not decision.allowed:
            logger.info("legal domain guard rejected query: %s", decision.reason)
            return ChatResponse(response=ensure_disclaimer(OUT_OF_SCOPE_MESSAGE), refused=True)
        if not app.state.limiter.allow(request.session_id):
            return ChatResponse(response=ensure_disclaimer(SESSION_LIMIT_MESSAGE), refused=True, rate_limited=True)
        assessment = app.state.service.classifier.assess(request.message)
        if assessment.emergency:
            return ChatResponse(
                response=ensure_disclaimer(EMERGENCY_MESSAGE),
                urgency="emergency",
                issue_categories=list(assessment.categories),
            )
        refresh_retrieval_corpus(settings)
        try:
            answer = await app.state.service.answer(request.message)
            return ChatResponse(
                response=answer.response,
                urgency="urgent" if answer.assessment.urgent else None,
                issue_categories=list(answer.assessment.categories),
                citations=answer.citations,
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
