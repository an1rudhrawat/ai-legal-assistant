from app.core.constants import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    DISCLAIMER,
    OUT_OF_SCOPE_MESSAGE,
    RATE_LIMIT_MESSAGE,
    RESPONSE_SCOPE_FALLBACK,
)
from app.llm.base import ProviderRateLimitError
from app.domain_guard.classifier import DomainClassification


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


def test_situation_based_request_reaches_legal_pipeline(client, provider):
    response = client.post("/api/chat", json=payload("The police won't register my complaint.", "situation-session"))
    assert response.status_code == 200
    assert response.json()["refused"] is False
    assert provider.calls == 1


def test_ambiguous_fact_free_request_asks_for_context(client, provider):
    client.app.state.domain_classifier.classify = lambda _: DomainClassification("ambiguous", 0.5, 0.5, "test")
    response = client.post("/api/chat", json=payload("What should I do?", "ambiguous-session"))
    assert "Please describe what happened" in response.json()["response"]
    assert provider.calls == 0


def test_validator_replaces_invalid_provider_output(client, provider):
    provider.response = "```python\nprint('legal')\n```"
    response = client.post("/api/chat", json=payload("What is bail?", "scope-session"))
    assert RESPONSE_SCOPE_FALLBACK in response.json()["response"]
    assert "print('legal')" not in response.json()["response"]


def test_validator_keeps_ordinary_fir_explanation(client, provider):
    provider.response = "Let me explain: an FIR is information recorded by police about a cognizable offence."
    response = client.post("/api/chat", json=payload("What is an FIR?", "fir-explanation-session"))
    assert "Let me explain" in response.json()["response"]


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


def test_exact_citation_request_returns_only_retrieved_source_metadata(client, provider):
    provider.response = "Section 154 applies to this situation."
    response = client.post("/api/chat", json=payload("Cite the exact section number for an FIR", "citation-session"))
    body = response.json()
    assert body["citations"][0]["section"] == "173"
    assert body["citations"][0]["source_name"] == "Bharatiya Nagarik Suraksha Sanhita, 2023"


def test_fabricated_model_citation_is_rejected(client, provider):
    provider.response = "Section 999 applies. [invented-section]"
    response = client.post("/api/chat", json=payload("What is an FIR?", "fabricated-citation-session"))
    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []


def test_emergency_message_does_not_call_provider(client, provider):
    response = client.post("/api/chat", json=payload("I am in immediate danger and someone is threatening to kill me", "emergency-session"))
    assert response.json()["urgency"] == "emergency"
    assert provider.calls == 0
