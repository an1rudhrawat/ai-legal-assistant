from app.core.constants import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    DISCLAIMER,
    OUT_OF_SCOPE_MESSAGE,
    RATE_LIMIT_MESSAGE,
    RESPONSE_SCOPE_FALLBACK,
)
from app.llm.base import ProviderRateLimitError
from app.domain_guard.classifier import DomainClassification
from app.services import repair_exact_heading_citation, validate_model_citations, validate_model_claims
from types import SimpleNamespace


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


def test_retrieved_chunks_are_passed_to_the_llm_context(client, provider):
    provider.response = "An FIR records information about a cognizable offence. [bnss-fir-1]"

    response = client.post("/api/chat", json=payload("The police will not register my FIR.", "grounding-session"))

    assert response.status_code == 200
    assert provider.calls == 1
    assert "Retrieved legal material and provenance:" in provider.last_user_message
    assert "Source ID: [bnss-fir-1]" in provider.last_user_message
    assert "Allowed citation IDs (copy one exactly; do not rewrite):\n- [bnss-fir-1]" in provider.last_user_message
    assert "Act: BNSS" in provider.last_user_message
    assert "Section title: Information in cognizable cases" in provider.last_user_message
    assert "Issuing authority: Government of India" in provider.last_user_message
    assert "Source URL (provenance only; do not reproduce): https://www.indiacode.nic.in/" in provider.last_user_message
    assert "Information relating to a cognizable offence" in provider.last_user_message
    assert '"reported_actors": ["user", "police"]' in provider.last_user_message
    assert "Structured retrieval hints (not legal findings):" in provider.last_user_message
    assert "Do not invent a contact, portal, emergency number" in provider.last_system_prompt
    assert "Answer only what the user asked" in provider.last_system_prompt
    assert "brevity must not remove a qualification" in provider.last_system_prompt
    assert response.json()["citations"][0]["section"] == "173"
    assert response.json()["citations"][0]["source_id"] == "bnss-fir-1"


def test_insufficient_retrieval_returns_safe_fallback_without_llm(client, provider):
    response = client.post("/api/chat", json=payload("Explain judicial review", "no-evidence-session"))

    assert response.status_code == 200
    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []
    assert provider.calls == 0


def test_ambiguous_fact_free_request_asks_for_context(client, provider):
    client.app.state.domain_classifier.classify = lambda _: DomainClassification("ambiguous", 0.5, 0.5, "test")
    response = client.post("/api/chat", json=payload("What should I do?", "ambiguous-session"))
    assert "Please describe what happened" in response.json()["response"]
    assert provider.calls == 0


def test_fact_free_private_defence_claim_is_clarified_before_llm(client, provider):
    response = client.post("/api/chat", json=payload("Can I claim private defence here?", "private-defence-session"))

    assert response.status_code == 200
    assert "Please describe what happened" in response.json()["response"]
    assert response.json()["citations"] == []
    assert provider.calls == 0


def test_fact_free_private_defence_question_is_clarified_even_if_domain_classifier_says_legal(client, provider):
    provider.response = "Yes, you can claim private defence here. [bnss-fir-1]"

    response = client.post("/api/chat", json=payload("Can I claim private defence here?", "private-defence-legal-classifier-session"))

    assert "Please describe what happened" in response.json()["response"]
    assert provider.calls == 0


def test_fact_free_offence_question_is_clarified_even_if_domain_classifier_says_legal(client, provider):
    provider.response = "The described conduct is an offence. [bnss-fir-1]"

    response = client.post("/api/chat", json=payload("Was what they did an offence under this Act?", "fact-free-offence-session"))

    assert "Please describe what happened" in response.json()["response"]
    assert response.json()["citations"] == []
    assert provider.calls == 0


def test_fact_free_punishment_question_asks_for_offence_or_provision(client, provider):
    response = client.post("/api/chat", json=payload("Can I be punished under this provision?", "fact-free-punishment-session"))

    assert "name the offence or legal provision" in response.json()["response"]
    assert response.json()["citations"] == []
    assert provider.calls == 0


def test_fact_free_arrest_question_asks_for_context(client, provider):
    response = client.post("/api/chat", json=payload("Can police arrest me for this?", "fact-free-arrest-session"))

    assert "Please describe what happened" in response.json()["response"]
    assert provider.calls == 0


def test_validator_replaces_invalid_provider_output(client, provider):
    provider.response = "```python\nprint('legal')\n``` [bnss-fir-1]"
    response = client.post("/api/chat", json=payload("What is an FIR?", "scope-session"))
    assert RESPONSE_SCOPE_FALLBACK in response.json()["response"]
    assert "print('legal')" not in response.json()["response"]


def test_validator_keeps_ordinary_fir_explanation(client, provider):
    provider.response = "Let me explain: an FIR is information recorded by police about a cognizable offence. [bnss-fir-1]"
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
    provider.response = "Section 173 is in the retrieved material. [bnss-fir-1]"
    response = client.post("/api/chat", json=payload("Cite the exact section number for an FIR", "citation-session"))
    body = response.json()
    assert body["citations"][0]["section"] == "173"
    assert body["citations"][0]["source_name"] == "Bharatiya Nagarik Suraksha Sanhita, 2023"


def test_fabricated_model_citation_is_rejected(client, provider):
    provider.response = "Section 999 applies. [invented-section]"
    response = client.post("/api/chat", json=payload("What is an FIR?", "fabricated-citation-session"))
    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []


def test_unsupported_plain_text_section_reference_is_rejected(client, provider):
    provider.response = "Section 154 applies to this situation. [bnss-fir-1]"

    response = client.post("/api/chat", json=payload("What is an FIR?", "invented-section-session"))

    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []


def test_exhaustive_case_absence_claim_is_rejected(client, provider):
    provider.response = "No such Supreme Court case exists about that issue. [bnss-fir-1]"

    response = client.post("/api/chat", json=payload("Name a case about an FIR", "unsupported-absence-session"))

    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []


def test_citation_validator_accepts_cross_referenced_section_in_retrieved_text():
    chunk = SimpleNamespace(
        chunk_id="bns-35",
        section_number="35",
        text="Right of private defence. The restrictions are contained in section 37.",
    )

    assert validate_model_citations(
        "**Section:** 35 is relevant, subject to section: 37. [Source: bns-35]",
        (chunk,),
    )


def test_citation_validator_rejects_unknown_or_inconsistent_source_id():
    chunk = SimpleNamespace(chunk_id="bns-section-14", section_number="14", text="Section 14. Rule text.")

    assert not validate_model_citations("Section 14 applies. [bns-section-1]", (chunk,))
    assert not validate_model_citations("Section 14 applies. [unknown-source]", (chunk,))


def test_citation_validator_rejects_missing_source_id():
    chunk = SimpleNamespace(chunk_id="bnss-fir-1", section_number="173", text="Information in cognizable cases.")

    assert not validate_model_citations("An FIR records a cognizable offence.", (chunk,))


def test_section_14_canonical_chunk_id_is_accepted():
    chunk = SimpleNamespace(
        chunk_id="bharatiya-nyaya-sanhita-2023-chunk-14",
        section_number="14",
        text="14. Act done by a person bound by law.",
    )

    assert validate_model_citations(
        "Section 14 is about an act done by a person bound by law. [bharatiya-nyaya-sanhita-2023-chunk-14]",
        (chunk,),
    )


def test_invalid_or_missing_heading_id_is_replaced_only_for_exact_retrieved_title():
    chunk = SimpleNamespace(
        chunk_id="bharatiya-nyaya-sanhita-2023-chunk-61",
        section_number="61",
        section_title="Criminal conspiracy.",
        text="61. Criminal conspiracy. Any person who...",
    )

    repaired = repair_exact_heading_citation(
        "What topic did I ask about? Give the section heading only.",
        "**Criminal conspiracy**\n\n[bharatiya-nyaya-sanh-2023-chunk-61]",
        (chunk,),
    )

    assert repaired is not None
    assert repaired[0] == "Section heading: Criminal conspiracy. [bharatiya-nyaya-sanhita-2023-chunk-61]"
    assert validate_model_citations(repaired[0], (chunk,))


def test_heading_citation_is_not_repaired_when_model_adds_extra_legal_claims():
    chunk = SimpleNamespace(
        chunk_id="bns-section-35",
        section_number="35",
        section_title="Right of private defence of body and of property.",
        text="35. Right of private defence of body and of property .�Every person has a right.",
    )

    repaired = repair_exact_heading_citation(
        "Which section was that, and what is its heading?",
        "Section: 35\nHeading: Right of private defence of body and of property. This means you can use force.",
        (chunk,),
    )

    assert repaired is None

def test_unsupported_specific_penalty_is_rejected_but_evidence_supported_duration_is_allowed():
    chunk = SimpleNamespace(chunk_id="bns-42", section_number="42", text="An offence is punishable with imprisonment for a term which may extend to one year.")
    other_chunk = SimpleNamespace(chunk_id="bnss-173", section_number="173", text="Information relating to a cognizable offence may be given to police.")

    assert validate_model_claims("The punishment is imprisonment for two years.", (chunk,)) is False
    assert validate_model_claims("The punishment is imprisonment for one year.", (chunk,)) is True
    assert validate_model_claims("The punishment is imprisonment for one year.", (other_chunk,)) is False


def test_api_rejects_penalty_amount_not_present_in_retrieved_text(client, provider):
    provider.response = "The punishment is imprisonment for one year. [bnss-fir-1]"

    response = client.post("/api/chat", json=payload("What is the punishment under BNSS section 173?", "unsupported-penalty-session"))

    assert provider.calls == 1
    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []


def test_missing_model_citation_returns_evidence_fallback(client, provider):
    provider.response = "An FIR records information about a cognizable offence."

    response = client.post("/api/chat", json=payload("What is an FIR?", "missing-citation-session"))

    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.json()["response"]
    assert response.json()["citations"] == []


def test_emergency_message_does_not_call_provider(client, provider):
    response = client.post("/api/chat", json=payload("I am in immediate danger and someone is threatening to kill me", "emergency-session"))
    assert response.json()["urgency"] == "emergency"
    assert provider.calls == 0
