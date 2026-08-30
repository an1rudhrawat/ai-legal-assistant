from app.core.constants import (
    CITATION_UNCERTAINTY_MESSAGE,
    DISCLAIMER,
    OUT_OF_SCOPE_MESSAGE,
    RATE_LIMIT_MESSAGE,
    RESPONSE_SCOPE_FALLBACK,
)
from app.llm.base import ProviderRateLimitError


def payload(message: str, session_id: str = "test-session-123") -> dict:
    return {"message": message, "session_id": session_id}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_returns_legal_response_and_disclaimer(client):
    response = client.post("/api/chat", json=payload("What is an FIR?"))
    body = response.json()
    assert response.status_code == 200
    assert body["refused"] is False
    assert "FIR" in body["response"]
    assert DISCLAIMER in body["response"]


def test_rejected_request_never_invokes_provider(client, provider):
    response = client.post("/api/chat", json=payload("Write Python code", "rejected-session"))
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert OUT_OF_SCOPE_MESSAGE in response.json()["response"]
    assert provider.calls == 0


def test_validator_replaces_invalid_provider_output(client, provider):
    provider.response = "```python\nprint('legal')\n```"
    response = client.post("/api/chat", json=payload("What is bail?", "scope-session"))
    assert RESPONSE_SCOPE_FALLBACK in response.json()["response"]
    assert "print('legal')" not in response.json()["response"]


def test_rate_limit_provider_error_is_graceful(client, provider):
    async def rate_limited(**kwargs):
        raise ProviderRateLimitError("429")
    provider.generate = rate_limited
    response = client.post("/api/chat", json=payload("What is bail?", "provider-rate-session"))
    assert response.status_code == 200
    assert response.json()["rate_limited"] is True
    assert RATE_LIMIT_MESSAGE in response.json()["response"]


def test_session_limiter_prevents_extra_provider_calls(client, provider):
    session = "limited-session"
    for _ in range(3):
        assert client.post("/api/chat", json=payload("What is bail?", session)).status_code == 200
    response = client.post("/api/chat", json=payload("What is bail?", session))
    assert response.json()["rate_limited"] is True
    assert provider.calls == 3


def test_exact_citation_request_is_hedged_not_confident(client, provider):
    provider.response = "Section 154 applies to this situation."
    response = client.post("/api/chat", json=payload("Cite the exact section number for an FIR", "citation-session"))
    assert CITATION_UNCERTAINTY_MESSAGE in response.json()["response"]
    assert "cannot confidently verify" in response.json()["response"].lower()
