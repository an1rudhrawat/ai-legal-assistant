from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    session_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    conversation_id: UUID | None = None
    request_id: UUID = Field(default_factory=uuid4)


class Citation(BaseModel):
    source_id: str | None = None
    source_name: str
    section: str | None = None
    source_url: str
    document_version: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: UUID
    refused: bool = False
    rate_limited: bool = False
    urgency: str | None = None
    issue_categories: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    domain_label: str | None = None


class ConversationCreated(BaseModel):
    conversation_id: UUID
    created_at: str


class ConversationMessage(BaseModel):
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ConversationHistory(BaseModel):
    conversation_id: UUID
    messages: list[ConversationMessage]


class HealthResponse(BaseModel):
    status: str = "ok"
