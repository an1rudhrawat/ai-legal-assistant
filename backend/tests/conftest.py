import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.main import create_app


class FakeProvider(LLMProvider):
    def __init__(self, response: str = "An FIR is a police record of information about a cognizable offence.") -> None:
        self.response = response
        self.calls = 0

    async def generate(self, *, system_prompt: str, user_message: str) -> str:
        self.calls += 1
        return self.response


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def client(provider: FakeProvider) -> TestClient:
    app = create_app(Settings(groq_api_key="test", session_request_limit=3))
    app.state.service.provider = provider
    return TestClient(app)
