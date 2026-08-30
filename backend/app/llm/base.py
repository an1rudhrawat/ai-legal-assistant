from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, *, system_prompt: str, user_message: str) -> str:
        """Return one assistant response or raise a ProviderError."""
