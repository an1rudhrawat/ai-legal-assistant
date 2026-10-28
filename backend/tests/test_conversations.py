import sqlite3
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.conversations import ConversationStore
from app.llm.base import ProviderError
from app.llm.groq import GroqProvider
from app.services import retrieval_query_with_previous_reply


def create_conversation(client: TestClient) -> str:
    response = client.post("/api/conversations")
    assert response.status_code == 201
    conversation_id = response.json()["conversation_id"]
    UUID(conversation_id)
    return conversation_id


def chat(client: TestClient, message: str, conversation_id: str, request_id: str | None = None):
    return client.post(
        "/api/chat",
        json={
            "message": message,
            "session_id": "conversation-test",
            "conversation_id": conversation_id,
            "request_id": request_id or str(uuid4()),
        },
    )


def test_follow_up_receives_prior_messages_and_keeps_latest_question_verbatim(client, provider):
    conversation_id = create_conversation(client)
    first = chat(client, "The police will not register my FIR.", conversation_id)
    assert first.status_code == 200

    second_question = "What if they refuse?"
    second = chat(client, second_question, conversation_id)

    assert second.status_code == 200
    assert [item["role"] for item in provider.last_history] == ["user", "assistant"]
    assert provider.last_history[0]["content"] == "The police will not register my FIR."
    assert f"User question (verbatim):\n{second_question}" in provider.last_user_message


def test_retrieval_uses_previous_reply_only_for_contextual_follow_ups():
    history = [
        {"role": "assistant", "content": "Section 173 covers information in cognizable cases."}
    ]

    assert "Section 173" in retrieval_query_with_previous_reply("Can you explain that?", history)
    assert retrieval_query_with_previous_reply("What is bail?", history) == "What is bail?"


def test_conversation_histories_are_isolated(client, provider):
    first_id = create_conversation(client)
    second_id = create_conversation(client)
    assert chat(client, "What is an FIR?", first_id).status_code == 200

    assert chat(client, "What is bail?", second_id).status_code == 200

    assert provider.last_history == []


def test_conversation_persists_across_api_requests_and_store_instances(client, provider):
    conversation_id = create_conversation(client)
    assert chat(client, "What is an FIR?", conversation_id).status_code == 200

    history = client.get(f"/api/conversations/{conversation_id}")
    assert history.status_code == 200
    assert [message["role"] for message in history.json()["messages"]] == ["user", "assistant"]

    reopened_store = ConversationStore(client.app.state.conversations.database_path)
    persisted = reopened_store.context(conversation_id)
    assert persisted["messages"][0]["content"] == "What is an FIR?"


def test_new_conversation_starts_without_previous_context(client, provider):
    first_id = create_conversation(client)
    second_id = create_conversation(client)
    assert chat(client, "What is an FIR?", first_id).status_code == 200

    response = chat(client, "What is bail?", second_id)

    assert response.status_code == 200
    assert response.json()["conversation_id"] == second_id
    assert provider.last_history == []


def test_contextual_retrieval_query_is_separate_from_original_question(
    client, provider, monkeypatch
):
    conversation_id = create_conversation(client)
    provider.response = "Section 173 concerns information in cognizable cases. [bnss-fir-1]"
    assert chat(client, "The police will not register my FIR.", conversation_id).status_code == 200
    retrieved_queries = []
    original_search = client.app.state.service.retriever.search

    def capture_query(query, *args, **kwargs):
        retrieved_queries.append(query)
        return original_search(query, *args, **kwargs)

    monkeypatch.setattr(client.app.state.service.retriever, "search", capture_query)
    question = "What if they refuse?"
    response = chat(client, question, conversation_id)

    assert response.status_code == 200
    assert "The police will not register my FIR." in retrieved_queries[0]
    assert question in retrieved_queries[0]
    assert "Previous assistant reply (unverified retrieval context)" in retrieved_queries[0]
    assert "Section 173 concerns information in cognizable cases" in retrieved_queries[0]
    assert f"User question (verbatim):\n{question}" in provider.last_user_message


def test_previous_assistant_reply_is_retrieval_context_not_an_unbounded_memory_dump(
    client, provider, monkeypatch
):
    conversation_id = create_conversation(client)
    provider.response = "Section 173 concerns information in cognizable cases. [bnss-fir-1]"
    assert chat(client, "What section covers an FIR?", conversation_id).status_code == 200

    queries = []
    retriever = client.app.state.service.retriever
    original_search = retriever.search
    monkeypatch.setattr(retriever, "search", lambda query, *args, **kwargs: (queries.append(query), original_search(query, *args, **kwargs))[1])
    provider.response = "Section 173 concerns information in cognizable cases. [bnss-fir-1]"
    assert chat(client, "Can you explain that?", conversation_id).status_code == 200

    assert "Section 173 concerns information in cognizable cases" in queries[0]
    assert "[bnss-fir-1]" not in queries[0]
    assert "This is general legal information" not in queries[0]


def test_long_conversation_is_summarized_and_recent_history_is_bounded(client, provider):
    client.app.state.conversation_history_limit = 2
    conversation_id = create_conversation(client)
    assert chat(client, "What is an FIR?", conversation_id).status_code == 200
    assert chat(client, "What is bail?", conversation_id).status_code == 200

    response = chat(client, "What is an FIR?", conversation_id)

    assert response.status_code == 200
    assert provider.calls == 4
    assert provider.last_history[0]["content"].startswith("Continuity summary for context only")
    assert len(provider.last_history) == 3


def test_retries_with_same_request_id_return_saved_result_without_duplicate_messages(
    client, provider
):
    conversation_id = create_conversation(client)
    request_id = str(uuid4())
    first = chat(client, "What is an FIR?", conversation_id, request_id)
    calls_after_first = provider.calls

    retry = chat(client, "What is an FIR?", conversation_id, request_id)

    assert retry.status_code == 200
    assert retry.json() == first.json()
    assert provider.calls == calls_after_first
    assert len(client.get(f"/api/conversations/{conversation_id}").json()["messages"]) == 2
    changed_payload = chat(client, "What is bail?", conversation_id, request_id)
    assert changed_payload.status_code == 409


def test_groq_failure_does_not_persist_an_incomplete_turn(client, provider):
    conversation_id = create_conversation(client)

    async def fail(**kwargs):
        raise ProviderError("temporary provider failure")

    provider.generate = fail
    response = chat(client, "What is an FIR?", conversation_id)

    assert response.status_code == 200
    assert "unavailable right now" in response.json()["response"]
    assert client.get(f"/api/conversations/{conversation_id}").json()["messages"] == []


def test_database_save_failure_is_reported_as_service_unavailable(client, monkeypatch):
    conversation_id = create_conversation(client)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(client.app.state.conversations, "save_turn", fail)
    response = chat(client, "What is an FIR?", conversation_id)

    assert response.status_code == 503
    assert "Conversation storage is unavailable" in response.json()["detail"]


def test_delete_removes_conversation_and_messages(client):
    conversation_id = create_conversation(client)
    assert chat(client, "What is an FIR?", conversation_id).status_code == 200

    deleted = client.delete(f"/api/conversations/{conversation_id}")

    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    assert chat(client, "What is an FIR?", conversation_id).status_code == 404


def test_invalid_and_missing_conversation_ids_are_handled(client):
    invalid = client.get("/api/conversations/not-a-uuid")
    assert invalid.status_code == 422

    unknown_id = str(uuid4())
    missing = chat(client, "What is an FIR?", unknown_id)
    assert missing.status_code == 404

    created_implicitly = client.post(
        "/api/chat",
        json={"message": "What is an FIR?", "session_id": "conversation-test"},
    )
    assert created_implicitly.status_code == 200
    UUID(created_implicitly.json()["conversation_id"])


@pytest.mark.asyncio
async def test_groq_messages_keep_history_roles_and_order(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        is_error = False

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "A grounded answer."}}]}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            captured["messages"] = json["messages"]
            return FakeResponse()

    monkeypatch.setattr("app.llm.groq.httpx.AsyncClient", FakeAsyncClient)
    provider = GroqProvider(api_key="test", model="test-model", base_url="https://example.invalid")

    response = await provider.generate(
        system_prompt="Safety instructions",
        history=[
            {"role": "user", "content": "My landlord locked me out."},
            {"role": "assistant", "content": "What belongings are missing?"},
        ],
        user_message="What if he refuses?",
    )

    assert response == "A grounded answer."
    assert [(item["role"], item["content"]) for item in captured["messages"]] == [
        ("system", "Safety instructions"),
        ("user", "My landlord locked me out."),
        ("assistant", "What belongings are missing?"),
        ("user", "What if he refuses?"),
    ]


@pytest.mark.parametrize("message", ["What is an FIR?", "What is bail?"])
def test_empty_history_works(client, provider, message):
    conversation_id = create_conversation(client)

    response = chat(client, message, conversation_id)

    assert response.status_code == 200
    assert provider.last_history == []
