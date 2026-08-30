from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    session_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    response: str
    refused: bool = False
    rate_limited: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
