from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ProviderError(Exception):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        history: Sequence[ChatMessage] = (),
    ) -> str:
        """Return one assistant response or raise a ProviderError."""
