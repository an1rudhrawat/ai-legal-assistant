from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    session_request_limit: int = 20
    legal_guard_strict: bool = True
    retrieval_index_path: Path = Path("data/vector_store/legal_index.json")
    retrieval_raw_path: Path = Path("data/raw")
    retrieval_processed_path: Path = Path("data/processed")
    retrieval_minimum_relevance: float = 0.12
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_dense_retrieval: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = False
    domain_classifier_model_path: Path = Path("data/domain_classifier/model")
    domain_legal_threshold: float = 0.70
    domain_non_legal_threshold: float = 0.35

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        configured_origins = tuple(
            item.strip()
            for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if item.strip()
        )
        local_origins = ("http://localhost:5173", "http://127.0.0.1:5173")
        origins = tuple(dict.fromkeys((*configured_origins, *local_origins)))
        project_root = Path(__file__).resolve().parents[3]
        index_setting = Path(os.getenv("RETRIEVAL_INDEX_PATH", "data/vector_store/legal_index.json"))
        index_path = index_setting if index_setting.is_absolute() else project_root / index_setting
        raw_setting = Path(os.getenv("RETRIEVAL_RAW_PATH", "data/raw"))
        processed_setting = Path(os.getenv("RETRIEVAL_PROCESSED_PATH", "data/processed"))
        classifier_setting = Path(os.getenv("DOMAIN_CLASSIFIER_MODEL_PATH", "data/domain_classifier/model"))
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "groq").lower(),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            cors_origins=origins,
            session_request_limit=int(os.getenv("SESSION_REQUEST_LIMIT", "20")),
            legal_guard_strict=os.getenv("LEGAL_GUARD_STRICT", "true").lower() == "true",
            retrieval_index_path=index_path,
            retrieval_raw_path=raw_setting if raw_setting.is_absolute() else project_root / raw_setting,
            retrieval_processed_path=processed_setting if processed_setting.is_absolute() else project_root / processed_setting,
            retrieval_minimum_relevance=float(os.getenv("RETRIEVAL_MINIMUM_RELEVANCE", "0.12")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            enable_dense_retrieval=os.getenv("ENABLE_DENSE_RETRIEVAL", "true").lower() == "true",
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            enable_reranker=os.getenv("ENABLE_RERANKER", "false").lower() == "true",
            domain_classifier_model_path=classifier_setting if classifier_setting.is_absolute() else project_root / classifier_setting,
            domain_legal_threshold=float(os.getenv("DOMAIN_LEGAL_THRESHOLD", "0.70")),
            domain_non_legal_threshold=float(os.getenv("DOMAIN_NON_LEGAL_THRESHOLD", "0.35")),
        )
