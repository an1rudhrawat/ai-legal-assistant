from __future__ import annotations

import httpx

from .base import LLMProvider, ProviderError, ProviderRateLimitError


class GroqProvider(LLMProvider):
    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    async def generate(self, *, system_prompt: str, user_message: str) -> str:
        if not self.api_key:
            raise ProviderError("GROQ_API_KEY is not configured")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ProviderError("LLM provider request failed") from exc
        if response.status_code == 429:
            raise ProviderRateLimitError("provider rate limit")
        if response.is_error:
            raise ProviderError(f"LLM provider error: {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("LLM provider returned an invalid response") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("LLM provider returned an empty response")
        return content.strip()
