from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    session_request_limit: int = 20
    legal_guard_strict: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        configured_origins = tuple(
            item.strip()
            for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if item.strip()
        )
        local_origins = ("http://localhost:5173", "http://127.0.0.1:5173")
        origins = tuple(dict.fromkeys((*configured_origins, *local_origins)))
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "groq").lower(),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            cors_origins=origins,
            session_request_limit=int(os.getenv("SESSION_REQUEST_LIMIT", "20")),
            legal_guard_strict=os.getenv("LEGAL_GUARD_STRICT", "true").lower() == "true",
        )
