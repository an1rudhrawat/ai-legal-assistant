from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    session_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class Citation(BaseModel):
    source_name: str
    section: str | None = None
    source_url: str
    document_version: str | None = None


class ChatResponse(BaseModel):
    response: str
    refused: bool = False
    rate_limited: bool = False
    urgency: str | None = None
    issue_categories: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    domain_label: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
