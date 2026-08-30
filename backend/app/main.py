from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.core.constants import OUT_OF_SCOPE_MESSAGE, RATE_LIMIT_MESSAGE, SESSION_LIMIT_MESSAGE
from app.domain_guard import LegalDomainGuard
from app.llm import ProviderError, ProviderRateLimitError, create_provider
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services import LegalAssistantService, SessionRequestLimiter, ensure_disclaimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Indian Legal Terminology Assistant", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.state.guard = LegalDomainGuard()
    app.state.service = LegalAssistantService(create_provider(settings))
    app.state.limiter = SessionRequestLimiter(settings.session_request_limit)

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
        try:
            answer = await app.state.service.answer(request.message)
            return ChatResponse(response=answer)
        except ProviderRateLimitError:
            logger.warning("LLM provider rate limit encountered")
            return ChatResponse(response=ensure_disclaimer(RATE_LIMIT_MESSAGE), rate_limited=True)
        except ProviderError:
            logger.exception("LLM provider failed")
            return ChatResponse(response=ensure_disclaimer("The assistant is unavailable right now. Please try again later."), rate_limited=False)

    return app


app = create_app()
