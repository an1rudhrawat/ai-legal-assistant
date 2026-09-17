import pytest
from app.retrieval.embeddings import embed
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
def retrieval_index(tmp_path):
    chunks = [
        {"chunk_id": "bnss-fir-1", "document_id": "bnss-2023", "document_name": "Bharatiya Nagarik Suraksha Sanhita, 2023", "act_name": "BNSS", "document_type": "legislation", "section_number": "173", "section_title": "Information in cognizable cases", "chapter": "XIII", "subsection": None, "source_url": "https://www.indiacode.nic.in/", "issuing_authority": "Government of India", "document_version": "2023", "effective_date": None, "publication_date": None, "ingested_at": "2026-01-01T00:00:00+00:00", "text": "Information relating to a cognizable offence may be given to an officer in charge of a police station."},
        {"chunk_id": "notice-1", "document_id": "notice-guide", "document_name": "Official court procedure guidance", "act_name": "Court procedure", "document_type": "official guidance", "section_number": None, "section_title": "Summons", "chapter": None, "subsection": None, "source_url": "https://www.indiacode.nic.in/", "issuing_authority": "Government of India", "document_version": "2023", "effective_date": None, "publication_date": None, "ingested_at": "2026-01-01T00:00:00+00:00", "text": "A court summons informs a person of a court proceeding and should be read carefully."},
        {"chunk_id": "bail-1", "document_id": "bail-guide", "document_name": "Official bail procedure guidance", "act_name": "Criminal procedure", "document_type": "official guidance", "section_number": None, "section_title": "Bail", "chapter": None, "subsection": None, "source_url": "https://www.indiacode.nic.in/", "issuing_authority": "Government of India", "document_version": "2023", "effective_date": None, "publication_date": None, "ingested_at": "2026-01-01T00:00:00+00:00", "text": "Bail is a legal process concerning release from custody in a criminal case."},
    ]
    index = tmp_path / "legal_index.json"
    import json
    index.write_text(json.dumps([{"chunk": chunk, "embedding": embed(chunk["text"])} for chunk in chunks]), encoding="utf-8")
    return index


@pytest.fixture
def client(provider: FakeProvider, retrieval_index) -> TestClient:
    app = create_app(Settings(groq_api_key="test", session_request_limit=3, retrieval_index_path=retrieval_index))
    app.state.service.provider = provider
    return TestClient(app)
