from app.core.config import Settings

from .base import LLMProvider
from .groq import GroqProvider


def create_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "groq":
        return GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            base_url=settings.groq_base_url,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
