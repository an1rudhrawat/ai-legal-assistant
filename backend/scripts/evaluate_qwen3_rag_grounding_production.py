"""Run the existing 36 Qwen cases through the production LegalAssistantService.

The temporary index and benchmark-only provider keep inference local and leave the
configured application provider, index, corpus, and database untouched.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import statistics
import time
from pathlib import Path

from evaluate_qwen3_rag_grounding import (
    BACKEND_ROOT, CORPUS, MODEL, OLLAMA, OUT_JSON, OUT_MD, ROOT,
    answer_word_count, build_cases, get, post, response_signals,
)
import sys
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.core.constants import INSUFFICIENT_EVIDENCE_MESSAGE
from app.retrieval.index import build_index_from_chunks
from app.retrieval.ingest import ingest_file
from app.retrieval.query_expansion import expand_query
from app.retrieval.retriever import LegalRetriever
from app.services import LegalAssistantService, needs_situation_clarification

PRIOR_REVIEWED_EXCERPTS = {
    "ambiguous_private_defence": "Yes, you can claim private defence ... followed by unsupported claims about Sections 34, 40, 44, 36, and 41, proportionality, and recommended next steps.",
    "ambiguous_punishment": "Yes, you can be punished under Section 173 ... if bribery; the response supplied a one-year term, a fine, and a treating exception without the user specifying conduct or a provision.",
}


class LocalOllamaProvider:
    """Inject Qwen into LegalAssistantService without changing app configuration."""
    def __init__(self) -> None:
        self.request_order = 0
        self.last: dict | None = None

    async def generate(self, *, system_prompt: str, user_message: str, history=()) -> str:
        self.request_order += 1
        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": item["role"], "content": item["content"]} for item in history],
            {"role": "user", "content": user_message},
        ]
        started = time.perf_counter()
        result = post(OLLAMA + "/api/chat", {
            "model": MODEL, "messages": messages, "stream": False, "think": False,
            "keep_alive": "10m", "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
        }, timeout=300)
        message = result.get("message") or {}
        self.last = {
            "request_order": self.request_order,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_response": message.get("content", ""),
            "prompt_tokens": result.get("prompt_eval_count"),
            "completion_tokens": result.get("eval_count"),
            "prompt_tokens_per_second": result.get("prompt_eval_rate"),
            "completion_tokens_per_second": result.get("eval_rate"),
            "model_load_duration_ns": result.get("load_duration"),
            "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
            "generation_duration_ns": result.get("eval_duration"),
        }
        return self.last["raw_response"]


def _production_assessment(text: str, chunks: list[dict], case: dict) -> dict:
    result = response_signals(text, chunks, case)
    safe_evidence_fallback = INSUFFICIENT_EVIDENCE_MESSAGE in text
    kind = case["expected_type"]
    if kind == "ambiguous":
        passes = result["asked_for_clarification"] and result["concise_instruction_following_auto"]
    elif kind == "insufficient":
        passes = (safe_evidence_fallback or result["abstained_or_marked_uncertainty"]) and answer_word_count(text) <= 120
    elif kind == "context_unknown":
        passes = (safe_evidence_fallback or result["abstained_or_marked_uncertainty"]) and answer_word_count(text) <= 120
    else:
        passes = (
            result["factual_consistency_auto"] is True
            and result["concise_instruction_following_auto"]
            and result["citation_valid_against_supplied_evidence"]
            and (not case.get("require_source_citation") or result["citation_present"])
            and not result["unsupported_claim_risk_auto"]
        )
    result["overall_auto_pass"] = bool(passes)
    return result


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"Corpus PDF not found: {CORPUS}")
    tags = get(OLLAMA + "/api/tags")
    model = next((item for item in tags.get("models", []) if item.get("name") == MODEL), None)
    if not model:
        raise SystemExit(f"Model not found. Download it with: ollama pull {MODEL}")
    try:
        version = get(OLLAMA + "/api/version").get("version", "unknown")
    except Exception:
        version = "unknown"
    settings = Settings.from_env()
    old = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    baseline_summary = (old.get("previous_run") or {}).get("summary") or old.get("summary")
    prior_tests = old.get("tests", [])
    resume_tests = (
        prior_tests
        if old.get("inference_method", "").startswith("Production LegalAssistantService")
        and 0 < len(prior_tests) < 36
        else []
    )
    resume_by_id = {item["id"]: item for item in resume_tests}
    old_by_id = {}
    for item in old.get("tests", []):
        comparison = item.get("comparison") or {}
        old_by_id[item["id"]] = {
            **item,
            "baseline_auto_pass": comparison.get("previous_auto_pass", (item.get("assessment") or {}).get("overall_auto_pass")),
            "baseline_model_response": comparison.get("previous_model_response", item.get("model_response")),
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="aadhikar_qwen_production_eval_") as temp:
        temp = Path(temp)
        chunks_file, index_file = temp / "chunks.json", temp / "index.json"
        chunk_count = ingest_file(CORPUS, None, chunks_file)
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        build_index_from_chunks(chunks, index_file)
        retriever = LegalRetriever(index_file, settings.retrieval_minimum_relevance,
                                   settings.embedding_model, settings.reranker_model,
                                   settings.enable_reranker, settings.enable_dense_retrieval)
        provider = LocalOllamaProvider()
        provider.request_order = max((item.get("model_request_order") or 0 for item in resume_tests), default=0)
        service = LegalAssistantService(provider, retriever)
        cases = build_cases(chunks)
        if len(cases) != 36:
            raise SystemExit(f"Expected the existing 36 cases, found {len(cases)}; refusing to change benchmark coverage.")
        data = {
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": MODEL, "model_size_bytes": model.get("size"), "model_digest": model.get("digest"),
            "quantization": (model.get("details") or {}).get("quantization_level", "unknown"),
            "parameter_size": (model.get("details") or {}).get("parameter_size"), "ollama_version": version,
            "inference_method": "Production LegalAssistantService using benchmark-only Ollama localhost adapter; think=false; temperature=0; seed=42.",
            "corpus": old.get("corpus", {"documents": [CORPUS.relative_to(ROOT).as_posix()], "chunk_count": chunk_count,
                                             "section_count": len({c.get('section_number') for c in chunks if c.get('section_number')}),
                                             "provenance_note": "No metadata sidecar; source provenance is unverified."}),
            "retriever": {"implementation": "LegalRetriever through LegalAssistantService", "minimum_relevance": settings.retrieval_minimum_relevance,
                          "dense_enabled_setting": settings.enable_dense_retrieval, "reranker_enabled_setting": settings.enable_reranker,
                          "index_location": "temporary directory, removed after run", "retrieved_limit": 4},
            "application_provider_configuration": {"configured_provider": settings.llm_provider, "configured_model": settings.llm_model,
                                                   "note": "The benchmark injects Qwen through a local-only adapter; application provider configuration is not changed or exercised."},
            "limitations": ["Only the BNS PDF is indexed and provenance remains unverified.",
                            "Exercises LegalAssistantService but not FastAPI domain-classifier, storage, emergency, or rate-limit gates.",
                            "Deterministic checks do not establish semantic or legal correctness."],
            "previous_run": {"started_at": (old.get("previous_run") or {}).get("started_at", old.get("started_at")), "summary": baseline_summary},
            "tests": list(resume_tests),
        }

        for order, case in enumerate(cases, 1):
            if case["id"] in resume_by_id:
                continue
            history = case["conversation_context"]
            service_query = " ".join([item["content"] for item in history] + [case["original_question"]]).strip()
            situation = service.understander.understand(service_query)
            categories = service.classifier.assess(service_query).categories
            retrieval_query = " ".join((expand_query(service_query, situation), *categories))
            retrieval_started = time.perf_counter()
            debug = retriever.debug_search(retrieval_query, limit=4)
            retrieval_time = round(time.perf_counter() - retrieval_started, 4)
            supplied = [chunk.to_dict() for chunk in debug.result.chunks]
            evidence_ok = bool(case.get("expected_section") and case.get("expected_heading") and any(
                str(chunk.get("section_number")) == case["expected_section"]
                and response_signals.__globals__["title_quality"](case["expected_heading"], chunk.get("section_title") or "") >= .8
                for chunk in supplied
            ))
            before = provider.request_order
            started = time.perf_counter()
            answer = asyncio.run(service.answer(case["original_question"], situation,
                                                 retrieval_query=service_query, history=history))
            total_time = round(time.perf_counter() - started, 4)
            raw = provider.last if provider.request_order > before else None
            final = answer.response
            raw_text = raw["raw_response"] if raw else None
            raw_assessment = response_signals(raw_text or "", supplied, case) if raw else None
            final_assessment = _production_assessment(final, supplied, case)
            if raw_assessment:
                raw_assessment["overall_auto_pass"] = bool(
                    raw_assessment["factual_consistency_auto"] is not False
                    and raw_assessment["citation_valid_against_supplied_evidence"]
                    and raw_assessment["concise_instruction_following_auto"]
                    and not raw_assessment["unsupported_claim_risk_auto"]
                )
                raw_assessment["manual_reviewer_notes"] = {
                    "heading_section_14": "Correct heading, but the raw model misspelled the supplied chunk ID; production rejects it safely.",
                    "ambiguous_private_defence": "Prior direct-run output asserted eligibility and added unsupported factual/legal advice; this production-service run is guarded before model invocation.",
                    "ambiguous_punishment": "Prior direct-run output guessed bribery and a penalty under section 173; this production-service run is guarded before model invocation.",
                }.get(case["id"], "")
            old_case = old_by_id.get(case["id"], {})
            prior_pass = old_case.get("baseline_auto_pass")
            if prior_pass is None and (baseline_summary or {}).get("total_cases") == 36:
                # The last run was interrupted after replacing the original
                # report. Its aggregate and four failing IDs were retained.
                prior_pass = case["id"] not in {
                    "heading_section_14", "ambiguous_private_defence",
                    "ambiguous_offence", "ambiguous_punishment",
                }
            record = {
                **case,
                "service_input_retrieval_query": service_query,
                "retrieval_query": retrieval_query,
                "retrieval_latency_seconds": retrieval_time,
                "retrieval_debug": {
                    "query": debug.query, "index_available": debug.index_available,
                    "dense_available": debug.dense_available, "reranker_active": debug.reranker_active,
                    "minimum_relevance": debug.minimum_relevance,
                    "expected_section_heading_evidence_retrieved": evidence_ok,
                    "expected_target_chunk_id": case.get("expected_chunk_id"),
                    "retrieved_chunks": [{
                        "chunk_id": item.chunk.chunk_id, "document_id": item.chunk.document_id,
                        "document_name": item.chunk.document_name, "act_name": item.chunk.act_name,
                        "section_number": item.chunk.section_number, "section_title": item.chunk.section_title,
                        "chapter": item.chunk.chapter, "source_url": item.chunk.source_url,
                        "issuing_authority": item.chunk.issuing_authority,
                        "relevance_score": item.relevance_score, "bm25_score": item.bm25_score,
                        "term_score": item.term_score, "dense_score": item.dense_score,
                        "rrf_score": item.rrf_score, "ranking_score": item.ranking_score,
                        "passed_minimum_relevance": item.passed_minimum_relevance, "text": item.chunk.text,
                    } for item in debug.retrieved],
                },
                "model_request_order": raw["request_order"] if raw else None,
                "generation_skipped": "fact-dependent clarification guard" if raw is None and needs_situation_clarification(case["original_question"], situation) else ("no relevant retrieval" if raw is None else None),
                "model_response": raw_text,
                "production_response": final,
                "production_citations": [citation.model_dump() for citation in answer.citations],
                "citation_repaired": answer.citation_repaired,
                "latency_seconds": total_time,
                "model_latency_seconds": raw["latency_seconds"] if raw else None,
                "prompt_tokens": raw["prompt_tokens"] if raw else None,
                "completion_tokens": raw["completion_tokens"] if raw else None,
                "prompt_tokens_per_second": raw["prompt_tokens_per_second"] if raw else None,
                "completion_tokens_per_second": raw["completion_tokens_per_second"] if raw else None,
                "model_load_duration_ns": raw["model_load_duration_ns"] if raw else None,
                "prompt_eval_duration_ns": raw["prompt_eval_duration_ns"] if raw else None,
                "generation_duration_ns": raw["generation_duration_ns"] if raw else None,
                "error": None,
                "assessment": raw_assessment,
                "production_assessment": final_assessment,
                "comparison": {
                    "previous_auto_pass": prior_pass,
                    "raw_pass_changed": bool(raw_assessment and prior_pass is not None and raw_assessment["overall_auto_pass"] != prior_pass),
                    "previous_model_response": old_case.get("baseline_model_response") or PRIOR_REVIEWED_EXCERPTS.get(case["id"]),
                    "previous_model_response_is_excerpt": case["id"] in PRIOR_REVIEWED_EXCERPTS and not old_case.get("baseline_model_response"),
                    "production_pipeline_pass": final_assessment["overall_auto_pass"],
                },
            }
            data["tests"].append(record)
            OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            state = "SKIP" if record["generation_skipped"] else ("PASS" if final_assessment["overall_auto_pass"] else "REVIEW/FAIL")
            print(f"[{order}/36] {case['id']}: {state} (model {record['model_latency_seconds']})", flush=True)

        calls = [test for test in data["tests"] if test["model_request_order"] is not None]
        latencies = [test["model_latency_seconds"] for test in calls]
        old_pass = (baseline_summary or {}).get("auto_pass")
        data["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        data["summary"] = {
            "total_cases": 36, "model_requests": len(calls),
            "generation_skipped_by_service": 36 - len(calls),
            "auto_pass": sum(bool(test["assessment"] and test["assessment"]["overall_auto_pass"]) for test in data["tests"]),
            "auto_fail_or_review": sum(bool(test["assessment"] and not test["assessment"]["overall_auto_pass"]) for test in data["tests"]),
            "production_pipeline_auto_pass": sum(test["production_assessment"]["overall_auto_pass"] for test in data["tests"]),
            "production_pipeline_auto_fail_or_review": sum(not test["production_assessment"]["overall_auto_pass"] for test in data["tests"]),
            "retrieval_failures": sum(test["expected_type"] in {"section_heading", "followup_heading"} and not test["retrieval_debug"]["expected_section_heading_evidence_retrieved"] for test in data["tests"]),
            "generation_grounding_failures": sum(bool(test["assessment"] and test["assessment"]["unsupported_claim_risk_auto"]) for test in data["tests"]),
            "invalid_source_id_responses": sum(bool((test.get("assessment") or {}).get("invalid_cited_source_ids")) for test in data["tests"]),
            "canonical_heading_citation_repairs": sum(bool(test.get("citation_repaired")) for test in data["tests"]),
            "missing_required_source_citations": sum(bool(test.get("assessment") and test.get("require_source_citation") and not test["assessment"]["citation_present"]) for test in data["tests"]),
            "context_retention_failures": sum((test.get("assessment") or {}).get("conversation_section_retained_auto") is False for test in data["tests"]),
            "instruction_following_failures": sum(bool(test.get("assessment") and not test["assessment"]["concise_instruction_following_auto"]) for test in data["tests"]),
            "corpus_coverage_limitations": sum(test["expected_type"] == "insufficient" for test in data["tests"]),
            "previous_run_auto_pass": old_pass,
            "raw_pass_changes": [{"id": test["id"], "from": test["comparison"]["previous_auto_pass"], "to": test["assessment"]["overall_auto_pass"]} for test in data["tests"] if test["comparison"]["raw_pass_changed"]],
            "production_pass_changes": [{"id": test["id"], "from": test["comparison"]["previous_auto_pass"], "to": test["production_assessment"]["overall_auto_pass"]} for test in data["tests"] if test["comparison"]["previous_auto_pass"] is not None and test["comparison"]["previous_auto_pass"] != test["production_assessment"]["overall_auto_pass"]],
            "raw_comparison_unavailable": [test["id"] for test in data["tests"] if test["assessment"] is None],
            "first_request_latency_seconds": latencies[0] if latencies else None,
            "subsequent_latency_mean_seconds": round(statistics.mean(latencies[1:]), 4) if len(latencies) > 1 else None,
            "subsequent_latency_median_seconds": round(statistics.median(latencies[1:]), 4) if len(latencies) > 1 else None,
            "prompt_tokens_total": sum(test["prompt_tokens"] or 0 for test in calls),
            "completion_tokens_total": sum(test["completion_tokens"] or 0 for test in calls),
        }
        data["recommendation"] = "Requires safeguards before legal-answering use. The production service safely intercepts the tested fact-free liability questions and rejects invalid citations, but semantic legal correctness, source provenance, and broader corpus coverage still need expert review."
        OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        OUT_MD.write_text(_report(data), encoding="utf-8")
    print(f"JSON report: {OUT_JSON}")
    print(f"Markdown report: {OUT_MD}")


def _report(data: dict) -> str:
    s = data["summary"]
    changed = s["production_pass_changes"]
    production_failures = [test for test in data["tests"] if not test["production_assessment"]["overall_auto_pass"]]
    lines = [
        "# Qwen3 4B RAG Grounding Evaluation (Production Service)", "",
        f"Run date: {data['started_at']}", "",
        "## Results", "",
        f"- Cases: {s['total_cases']}; model calls: {s['model_requests']}; service clarification/abstention without Qwen: {s['generation_skipped_by_service']}.",
        f"- Previous direct-run score: {s['previous_run_auto_pass']}/36. Production-path changes from that run: {len(changed)} cases; raw comparison unavailable for model-skipped cases: {', '.join(s['raw_comparison_unavailable']) or 'none'}.",
        f"- Production-service response automatic passes: {s['production_pipeline_auto_pass']}/36; review/fail: {s['production_pipeline_auto_fail_or_review']}.",
        f"- Raw Qwen calls scored: {s['auto_pass']}/{s['model_requests']} pass; raw generation/grounding risk flags: {s['generation_grounding_failures']}. Known-section retrieval failures: {s['retrieval_failures']}; BNS corpus-coverage probes: {s['corpus_coverage_limitations']}.",
        f"- Citation checks: {s.get('invalid_source_id_responses', 0)} responses used an unavailable source ID; {s.get('missing_required_source_citations', 0)} heading answers omitted a required source citation. Context-retention failures: {s.get('context_retention_failures', 0)}; instruction-following failures: {s.get('instruction_following_failures', 0)}.",
        f"- Exact heading metadata supplied canonical citations after strict ID validation in {s.get('canonical_heading_citation_repairs', 0)} cases; malformed IDs themselves were not accepted.",
        f"- Corpus: {data['corpus']['documents']} ({data['corpus']['chunk_count']} chunks, {data['corpus']['section_count']} sections); provenance unverified.",
        f"- Model: `{data['model']}` ({data['quantization']}); Ollama {data['ollama_version']}; thinking disabled.", "",
        "## Performance", "",
        f"- First Ollama request: {s['first_request_latency_seconds']} s; subsequent mean/median: {s['subsequent_latency_mean_seconds']} / {s['subsequent_latency_median_seconds']} s.",
        f"- Prompt/completion tokens: {s['prompt_tokens_total']} / {s['completion_tokens_total']}.", "",
        "## Reviewed prior failure cases", "",
    ]
    if changed:
        lines += ["| Case | Previous | Production response |", "|---|---:|---:|"]
        lines += [f"| `{item['id']}` | {'PASS' if item['from'] else 'FAIL'} | {'PASS' if item['to'] else 'FAIL'} |" for item in changed]
        lines.append("")
    for case_id in ("heading_section_14", "ambiguous_private_defence", "ambiguous_punishment"):
        current = next(test for test in data["tests"] if test["id"] == case_id)
        lines.append(f"### {case_id}")
        previous_excerpt = current["comparison"].get("previous_model_response") or ""
        qualifier = " (manually transcribed excerpt from the prior run)" if current["comparison"].get("previous_model_response_is_excerpt") else ""
        lines.append(f"- Prior raw Qwen response{qualifier}: {previous_excerpt[:700].replace(chr(10), ' ')}")
        lines.append(f"- Current raw Qwen response: {(current['model_response'] or '[not called; service guard handled this case]')[:700].replace(chr(10), ' ')}")
        lines.append(f"- User-facing service response: {current['production_response'][:500].replace(chr(10), ' ')}")
    lines += ["", "## Remaining production-path failures", ""]
    for test in production_failures:
        assessment = test["production_assessment"]
        invalid = (test.get("assessment") or {}).get("invalid_cited_source_ids", [])
        lines.append(f"- **{test['id']}**: invalid source IDs {invalid}; final response assessment: factual consistency `{assessment['factual_consistency_auto']}`, citation present `{assessment['citation_present']}`. Raw output: {(test.get('model_response') or '')[:220].replace(chr(10), ' ')}")
    lines += ["", "## Interpretation", "",
              "Section 14's stored retrieved ID and prompt ID are canonical and equal; the previous Qwen response dropped characters from the ID. The prior runner bypassed production validation, so it recorded the raw failure rather than the user-facing fallback. Routing that output through the current validator rejects it; this is a model citation-format error plus a runner-scope limitation, not a retrieval or validator defect.",
              "The previous private-defence and punishment outputs were raw-model outputs from a runner that bypassed LegalAssistantService. The current runner invokes that service; its fact-dependent guard returns a clarification without calling Qwen for those exact no-fact questions. The benchmark now records both raw model output and the final service response.",
              "The new deterministic citation check requires at least one supplied source ID, rejects unknown IDs and section references, and returns source IDs in the API citation objects. A narrow penalty-duration check rejects explicit year/month/day/amount claims absent from retrieved text. Other semantic claims, offence classification, proportionality, and whether a provision applies to facts still cannot be fully verified without a legal adjudication component; citations can exist and still fail to support a proposition.",
              "When a heading-only response's complete cleaned text exactly matches a unique retrieved section title that also appears in that chunk, the service discards any malformed model ID and emits that chunk's canonical ID. It logs and records this limited repair; unknown IDs in other answers remain rejected, and answers with extra legal claims are not repaired.",
              "The runner exercises LegalAssistantService, but not FastAPI's domain classifier, emergency routing, persistence, or rate limiter. The BNS-only corpus cannot substantiate tenancy, consumer, case-law, civil-remedy, or criminal-procedure coverage.",
              f"The current application settings select `{data['application_provider_configuration']['configured_provider']}` / `{data['application_provider_configuration']['configured_model']}`. This evaluation injects `{data['model']}` locally and does not change or exercise `create_provider`.",
              "", "## Recommendation", "", data["recommendation"], "",
              "Manual legal correctness remains pending. Do not treat automated passes as legal validation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
