"""Run the deterministic local retrieval evaluation without calling the LLM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.retrieval.evaluation import evaluate_cases, format_debug, load_evaluation_cases
from app.retrieval.retriever import LegalRetriever


PROJECT_ROOT = BACKEND_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local legal retriever against corpus-aware cases.")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "backend/evaluation/retrieval_cases.json")
    parser.add_argument("--debug-case", help="Print local retrieval signals for one case ID.")
    args = parser.parse_args()
    settings = Settings.from_env()
    retriever = LegalRetriever(
        settings.retrieval_index_path,
        settings.retrieval_minimum_relevance,
        settings.embedding_model,
        settings.reranker_model,
        settings.enable_reranker,
        settings.enable_dense_retrieval,
    )
    cases = load_evaluation_cases(args.cases)
    report = evaluate_cases(retriever, cases)
    print(json.dumps(report, indent=2))
    if args.debug_case:
        case = next((item for item in cases if item.id == args.debug_case), None)
        if case is None:
            raise SystemExit(f"Unknown evaluation case: {args.debug_case}")
        from app.retrieval.evaluation import build_expanded_query
        print("\n" + format_debug(retriever.debug_search(build_expanded_query(case.user_query), limit=5)))


if __name__ == "__main__":
    main()
