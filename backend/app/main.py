from __future__ import annotations

import logging
import sqlite3
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.conversations import ConversationNotFound, ConversationStore, RequestIdConflict
from app.core.config import Settings
from app.core.constants import AMBIGUOUS_SITUATION_MESSAGE, EMERGENCY_MESSAGE, OUT_OF_SCOPE_MESSAGE, RATE_LIMIT_MESSAGE, SESSION_LIMIT_MESSAGE
from app.domain_guard import LocalSemanticDomainClassifier, ObviousOutOfScopeGate
from app.llm import ProviderError, ProviderRateLimitError, create_provider
from app.retrieval import LegalRetriever, sync_corpus
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreated,
    ConversationHistory,
    ConversationMessage,
    HealthResponse,
)
from app.services import LegalAssistantService, SessionRequestLimiter, ensure_disclaimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not 2 <= settings.conversation_history_limit <= 10:
        raise ValueError("conversation_history_limit must be between 2 and 10")
    app = FastAPI(title="Legal First Responder", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["DELETE", "GET", "POST"], allow_headers=["Content-Type"])
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
    app.state.conversations = ConversationStore(settings.conversation_database_path)
    app.state.conversation_history_limit = settings.conversation_history_limit

    @app.on_event("startup")
    async def sync_retrieval_corpus() -> None:
        result = refresh_retrieval_corpus(settings)
        logger.info("RAG corpus synced: %s added, %s updated, %s unchanged, %s chunks", result.added, result.updated, result.unchanged, result.indexed_chunks)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/api/conversations", response_model=ConversationCreated, status_code=201)
    async def create_conversation() -> ConversationCreated:
        try:
            return ConversationCreated(**app.state.conversations.create())
        except sqlite3.Error as exc:
            logger.exception("conversation database failed to create a conversation")
            raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationHistory)
    async def get_conversation(conversation_id: UUID) -> ConversationHistory:
        try:
            context = app.state.conversations.context(str(conversation_id))
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except sqlite3.Error as exc:
            logger.exception("conversation database failed to load a conversation")
            raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=[
                ConversationMessage(
                    id=message["id"],
                    role=message["role"],
                    content=message["content"],
                    created_at=message["created_at"],
                )
                for message in context["messages"]
            ],
        )

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: UUID) -> Response:
        try:
            app.state.conversations.delete(str(conversation_id))
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except sqlite3.Error as exc:
            logger.exception("conversation database failed to delete a conversation")
            raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc
        return Response(status_code=204)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            conversation_id = (
                str(request.conversation_id)
                if request.conversation_id
                else app.state.conversations.create()["conversation_id"]
            )
            cached = app.state.conversations.cached_response(
                conversation_id,
                str(request.request_id),
                request.message,
            )
            if cached:
                return ChatResponse.model_validate(cached)
            context = app.state.conversations.context(conversation_id)
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except RequestIdConflict as exc:
            raise HTTPException(status_code=409, detail="request_id was already used for another message.") from exc
        except sqlite3.Error as exc:
            logger.exception("conversation database failed to load chat context")
            raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc

        def complete(response: ChatResponse) -> ChatResponse:
            response.conversation_id = UUID(conversation_id)
            try:
                saved = app.state.conversations.save_turn(
                    conversation_id,
                    str(request.request_id),
                    request.message,
                    response.model_dump(mode="json"),
                )
            except ConversationNotFound as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc
            except RequestIdConflict as exc:
                raise HTTPException(status_code=409, detail="request_id was already used for another message.") from exc
            except sqlite3.Error as exc:
                logger.exception("conversation database failed to save a chat turn")
                raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc
            return ChatResponse.model_validate(saved)

        if app.state.out_of_scope_gate.rejects(request.message):
            return complete(ChatResponse(
                response=ensure_disclaimer(OUT_OF_SCOPE_MESSAGE),
                refused=True,
                domain_label="non_legal",
                conversation_id=UUID(conversation_id),
            ))
        if not app.state.limiter.allow(request.session_id):
            return complete(ChatResponse(
                response=ensure_disclaimer(SESSION_LIMIT_MESSAGE),
                refused=True,
                rate_limited=True,
                conversation_id=UUID(conversation_id),
            ))

        try:
            messages = context["messages"]
            history_limit = app.state.conversation_history_limit
            rolled_count = max(0, len(messages) - history_limit)
            rolled_count -= rolled_count % 2
            summary = context["summary"]
            summary_through = context["summary_through_sequence"]
            if rolled_count > summary_through:
                rolled = [
                    message
                    for message in messages
                    if summary_through < message["sequence"] <= rolled_count
                ]
                summary = await app.state.service.summarize_history(summary, rolled)
                try:
                    saved_summary = app.state.conversations.update_summary(
                        conversation_id,
                        summary_through,
                        summary,
                        rolled_count,
                    )
                except ConversationNotFound as exc:
                    raise HTTPException(status_code=404, detail="Conversation not found.") from exc
                except sqlite3.Error as exc:
                    logger.exception("conversation database failed to save its summary")
                    raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc
                if not saved_summary:
                    raise HTTPException(
                        status_code=409,
                        detail="Conversation changed concurrently; retry the request.",
                    )
                context = app.state.conversations.context(conversation_id)
                messages = context["messages"]
                summary = context["summary"]

            recent_messages = messages[-history_limit:]
            prior_user_messages = [
                item["content"] for item in recent_messages if item["role"] == "user"
            ][-2:]
            query_parts = []
            if summary:
                query_parts.append(f"Prior user-reported context (unverified): {summary}")
            if prior_user_messages:
                query_parts.extend(prior_user_messages)
            query_parts.append(request.message)
            retrieval_query = " ".join(query_parts) if len(query_parts) > 1 else request.message

            classification = app.state.domain_classifier.classify(retrieval_query)
            if classification.label == "non_legal":
                logger.info("domain classifier rejected query: %s", classification.reason)
                return complete(ChatResponse(
                    response=ensure_disclaimer(OUT_OF_SCOPE_MESSAGE),
                    refused=True,
                    domain_label="non_legal",
                    conversation_id=UUID(conversation_id),
                ))
            situation = app.state.service.understander.understand(retrieval_query)
            assessment = app.state.service.classifier.assess(retrieval_query)
            if assessment.emergency or situation.urgent:
                return complete(ChatResponse(
                    response=ensure_disclaimer(EMERGENCY_MESSAGE),
                    urgency="emergency",
                    issue_categories=list(assessment.categories),
                    domain_label=classification.label,
                    conversation_id=UUID(conversation_id),
                ))
            # A semantically uncertain request with no extracted event must not be
            # allowed to retrieve a coincidental statute and invent a legal context.
            if classification.label == "ambiguous" and not situation.events:
                return complete(ChatResponse(
                    response=ensure_disclaimer(AMBIGUOUS_SITUATION_MESSAGE),
                    domain_label="ambiguous",
                    conversation_id=UUID(conversation_id),
                ))
            refresh_retrieval_corpus(settings)
            answer = await app.state.service.answer(
                request.message,
                situation,
                retrieval_query=retrieval_query,
                history=[
                    {"role": item["role"], "content": item["content"]}
                    for item in recent_messages
                ],
                summary=summary,
            )
            if classification.label == "ambiguous" and not answer.evidence_sufficient:
                return complete(ChatResponse(
                    response=ensure_disclaimer(AMBIGUOUS_SITUATION_MESSAGE),
                    domain_label="ambiguous",
                    conversation_id=UUID(conversation_id),
                ))
            return complete(ChatResponse(
                response=answer.response,
                urgency="urgent" if answer.assessment.urgent else None,
                issue_categories=list(answer.assessment.categories),
                citations=answer.citations,
                domain_label=classification.label,
                conversation_id=UUID(conversation_id),
            ))
        except ProviderRateLimitError:
            logger.warning("LLM provider rate limit encountered")
            return ChatResponse(
                response=ensure_disclaimer(RATE_LIMIT_MESSAGE),
                rate_limited=True,
                conversation_id=UUID(conversation_id),
            )
        except ProviderError:
            logger.exception("LLM provider failed")
            return ChatResponse(
                response=ensure_disclaimer("The assistant is unavailable right now. Please try again later."),
                conversation_id=UUID(conversation_id),
            )
        except sqlite3.Error as exc:
            logger.exception("conversation database failed during chat")
            raise HTTPException(status_code=503, detail="Conversation storage is unavailable.") from exc

    return app


def refresh_retrieval_corpus(settings: Settings):
    """Pick up added or changed files without requiring an API restart."""
    return sync_corpus(settings.retrieval_raw_path, settings.retrieval_processed_path, settings.retrieval_index_path)


app = create_app()
