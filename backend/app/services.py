from __future__ import annotations

import logging
import re
from collections import defaultdict
from threading import Lock

from app.core.constants import CITATION_UNCERTAINTY_MESSAGE, DISCLAIMER
from app.domain_guard import ResponseScopeValidator
from app.llm.base import LLMProvider
from app.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class SessionRequestLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def allow(self, session_id: str) -> bool:
        with self._lock:
            if self._counts[session_id] >= self.limit:
                return False
            self._counts[session_id] += 1
            return True


def ensure_disclaimer(text: str) -> str:
    cleaned = text.strip()
    if DISCLAIMER.lower() in cleaned.lower():
        return cleaned
    return f"{cleaned}\n\n{DISCLAIMER}"


def asks_for_exact_citation(message: str) -> bool:
    return bool(re.search(r"\b(exact|specific)\b.{0,45}\b(section|statute|citation|case law)\b|\bsection number\b", message.lower()))


def has_citation_hedge(text: str) -> bool:
    return bool(re.search(r"\b(cannot (?:confirm|verify)|not certain|uncertain|may|please verify|consult (?:the )?official)\b", text.lower()))


class LegalAssistantService:
    def __init__(self, provider: LLMProvider, validator: ResponseScopeValidator | None = None) -> None:
        self.provider = provider
        self.validator = validator or ResponseScopeValidator()

    async def answer(self, message: str) -> str:
        text = await self.provider.generate(system_prompt=SYSTEM_PROMPT, user_message=message)
        if asks_for_exact_citation(message) and not has_citation_hedge(text):
            logger.warning("response citation certainty policy triggered")
            return ensure_disclaimer(CITATION_UNCERTAINTY_MESSAGE)
        valid, reason = self.validator.validate(text)
        if not valid:
            logger.warning("response scope validator triggered: %s", reason)
            from app.core.constants import RESPONSE_SCOPE_FALLBACK
            return ensure_disclaimer(RESPONSE_SCOPE_FALLBACK)
        return ensure_disclaimer(text)
