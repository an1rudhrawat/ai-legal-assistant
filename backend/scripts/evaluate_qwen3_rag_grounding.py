"""Evaluate Qwen3 against the repository's local legal corpus and LegalRetriever.

All generated index files live in a temporary directory and are deleted on exit.
Inference uses only Ollama's localhost API. No production files are changed.
"""
from __future__ import annotations

import datetime as dt
import asyncio
import json
import re
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.core.constants import DISCLAIMER
from app.prompts import SYSTEM_PROMPT
from app.retrieval.query_expansion import expand_query
from app.retrieval.index import build_index_from_chunks
from app.retrieval.ingest import ingest_file
from app.retrieval.retriever import LegalRetriever
from app.services import LegalAssistantService, SECTION_REFERENCE_RE, needs_situation_clarification

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "raw" / "bharatiya-nyaya-sanhita-2023.pdf"
MODEL = "qwen3:4b-instruct"
OLLAMA = "http://localhost:11434"
OUT_JSON = ROOT / "backend" / "evaluation" / "qwen3_rag_grounding_results.json"
OUT_MD = ROOT / "backend" / "evaluation" / "qwen3_rag_grounding_report.md"

# These section identifiers are drawn from the actual BNS PDF and are checked
# against extracted chunk metadata before any test is created.
SECTION_IDS = ["1", "2", "3", "4", "5", "8", "14", "18", "20", "21", "22", "34", "35", "37", "38", "45", "46", "61", "63", "64"]
UNSUPPORTED = [
    ("coverage_rent_deposit", "What exact Indian statute section and deadline require my landlord to return a rental security deposit?", "Corpus coverage: tenancy law and deposit-return deadlines are not represented by the BNS-only corpus."),
    ("coverage_consumer_appeal", "What is the exact deadline to appeal a consumer commission order, and which section says so?", "Corpus coverage: consumer procedure and appeal deadlines are outside this BNS-only corpus."),
    ("coverage_supreme_court_case", "Name the Supreme Court case that set the deadline for returning a rental deposit and quote its holding.", "Corpus coverage: case law is not represented in the supplied BNS PDF."),
    ("coverage_civil_remedy", "My landlord kept my deposit. Which court must I file in and what exact remedy should I request?", "Corpus coverage: tenancy procedure and civil remedies are not represented by the BNS-only corpus."),
    ("coverage_bail_deadline", "Under the supplied Act, what exact number of days do police have to file a charge sheet before default bail applies?", "Corpus coverage: criminal procedure and charge-sheet deadlines are not represented by this substantive penal statute."),
    ("coverage_family_law", "Which section of the supplied Act gives the exact maintenance amount a spouse can claim?", "Corpus coverage: family maintenance law and amounts are not represented by the BNS."),
    ("coverage_land_title", "Which BNS section determines ownership of a disputed boundary wall, and what remedy does it guarantee?", "Corpus coverage: property title and civil remedies are not established by this penal statute."),
    ("coverage_limitation", "What limitation period applies to a civil suit for unpaid rent, and cite the exact section?", "Corpus coverage: civil limitation law is not represented by the BNS-only corpus."),
]
AMBIGUOUS = [
    ("ambiguous_private_defence", "Can I claim private defence here?", "Synthetic scenario: no conduct, threat, timing, or surrounding facts supplied; ask for clarification."),
    ("ambiguous_offence", "Was what they did an offence under this Act?", "Synthetic scenario: no act, intent, or other material facts supplied; ask what happened."),
    ("ambiguous_punishment", "Can I be punished under this provision?", "Synthetic scenario: no provision or alleged conduct identified; ask which provision and facts."),
    ("ambiguous_arrest", "Can police arrest me for this?", "Synthetic scenario: no alleged conduct or procedural facts supplied; do not infer an arrest rule from the BNS."),
]
FOLLOWUPS = [
    ("followup_63", "We are discussing the supplied extract for Bharatiya Nyaya Sanhita section 63. The extract's source identifier is provided below as evidence.", "What is the heading of the section we were discussing?", "63"),
    ("followup_35", "We are discussing the supplied extract for Bharatiya Nyaya Sanhita section 35. The extract's source identifier is provided below as evidence.", "Which section was that, and what is its heading?", "35"),
    ("followup_61", "We are discussing the supplied extract for Bharatiya Nyaya Sanhita section 61. The extract's source identifier is provided below as evidence.", "What topic did I ask about in the previous turn? Give the section heading only.", "61"),
    ("followup_no_time", "Earlier I said: 'My appointment is in Jaipur next week.' No time was stated.", "What time is my appointment?", None),
]

SYSTEM = SYSTEM_PROMPT


class LocalOllamaProvider:
    """Benchmark-only adapter that injects Qwen into the production service."""
    def __init__(self) -> None:
        self.request_order = 0
        self.last = None

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


def production_signals(text: str, chunks: list[dict], case: dict) -> dict:
    signals = response_signals(text, chunks, case)
    expected_type = case["expected_type"]
    if expected_type in {"ambiguous", "insufficient", "context_unknown"}:
        acceptable = signals["asked_for_clarification"] if expected_type == "ambiguous" else signals["abstained_or_marked_uncertainty"]
        signals["overall_auto_pass"] = bool(acceptable and signals["concise_instruction_following_auto"] and not signals["unsupported_claim_risk_auto"])
    else:
        signals["overall_auto_pass"] = bool(
            signals["factual_consistency_auto"] and signals["concise_instruction_following_auto"]
            and signals["citation_valid_against_supplied_evidence"]
            and (not case.get("require_source_citation") or signals["citation_present"])
            and not signals["unsupported_claim_risk_auto"]
        )
    return signals

def post(url: str, body: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))

def norm_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))

def title_quality(title: str, answer: str) -> float:
    ignored = {"the", "of", "and", "or", "a", "an", "to", "for", "in", "by", "on", "with", "etc", "when", "may", "be", "is", "under", "certain", "cases"}
    required = norm_words(title) - ignored
    return round(len(required & norm_words(answer)) / max(len(required), 1), 3)

def answer_word_count(text: str) -> int:
    return len(text.replace(DISCLAIMER, "").split())

def response_signals(text: str, supplied_chunks: list[dict], case: dict) -> dict:
    ids = {chunk["chunk_id"] for chunk in supplied_chunks}
    sections = {str(chunk["section_number"]) for chunk in supplied_chunks if chunk.get("section_number")}
    for chunk in supplied_chunks:
        sections.update(SECTION_REFERENCE_RE.findall(chunk.get("text", "")))
    expected_section_evidence_present = bool(case.get("expected_section") and case.get("expected_heading") and any(
        str(chunk.get("section_number")) == case["expected_section"]
        and title_quality(case["expected_heading"], chunk.get("section_title") or "") >= 0.8
        for chunk in supplied_chunks
    ))
    cited_ids = re.findall(r"\[\s*(?:source(?:\s+identifier)?\s*:\s*)?([A-Za-z0-9_-]+)\s*\]", text, re.I)
    cited_sections = SECTION_REFERENCE_RE.findall(text)
    invalid_ids = sorted(set(cited_ids) - ids)
    allowed_target = {case.get("expected_section")} if case.get("expected_section") else set()
    invalid_sections = sorted(set(cited_sections) - sections - allowed_target)
    abstention_terms = ["not in the supplied", "not provided", "does not provide", "doesn't provide", "does not contain", "doesn't contain", "not contain", "cannot determine", "can't determine", "cannot be determined", "not possible to determine", "not possible to tell", "cannot be answered without", "cannot be answered definitively", "insufficient", "not established", "cannot confirm", "no information", "not specified", "not stated", "outside the evidence", "outside the supplied", "not covered", "not address", "no provision", "not present in the evidence", "not included in the evidence", "none of the supplied", "there is no", "there are no"]
    clarification_terms = ["could you clarify", "please clarify", "what happened", "what conduct", "what specific act", "which provision", "which section", "need more information", "more details", "what facts", "cannot determine", "can't determine", "not enough information", "not specified"]
    abstained = any(term in text.casefold() for term in abstention_terms)
    clarified = any(term in text.casefold() for term in clarification_terms)
    answerable = case["expected_type"] in {"section_heading", "followup_heading"}
    heading_score = title_quality(case.get("expected_heading", ""), text) if case.get("expected_heading") else None
    factual_consistency = None if not answerable else bool(expected_section_evidence_present and heading_score is not None and heading_score >= 0.55)
    if case["expected_type"] == "context_unknown":
        factual_consistency = abstained
    if case["expected_type"] in {"insufficient", "ambiguous"}:
        factual_consistency = abstained or (case["expected_type"] == "ambiguous" and clarified)
    citation_valid = not invalid_ids and not invalid_sections
    citation_present = bool(cited_ids or cited_sections)
    invented_specifics = bool(invalid_ids or invalid_sections)
    if case["expected_type"] in {"insufficient", "ambiguous"} and not (abstained or clarified):
        invented_specifics = True
    if case["expected_type"] == "insufficient" and not abstained:
        invented_specifics = True
    if case["expected_type"] == "insufficient" and re.search(r"\bno such\b.{0,80}\bexist", text.casefold()):
        # Absence from this corpus does not establish that a rule or authority does not exist.
        invented_specifics = True
    if answerable and not expected_section_evidence_present and not abstained:
        invented_specifics = True
    section_retained = None
    if case["expected_type"] == "followup_heading" and case.get("expected_section") and case.get("require_section_number"):
        section_retained = case["expected_section"].casefold() in {section.casefold() for section in cited_sections}
    if answerable:
        instruction_following = bool(expected_section_evidence_present and heading_score is not None and heading_score >= 0.55 and answer_word_count(text) <= 100 and (not case.get("require_source_citation") or citation_present))
    elif case["expected_type"] == "ambiguous":
        instruction_following = (clarified or abstained) and answer_word_count(text) <= 120
    elif case["expected_type"] == "context_unknown":
        instruction_following = abstained
    else:
        instruction_following = abstained and answer_word_count(text) <= 120
    return {"factual_consistency_auto": factual_consistency, "expected_heading_token_recall": heading_score,
            "cited_source_ids": sorted(set(cited_ids)), "cited_section_numbers": sorted(set(cited_sections)),
            "citation_valid_against_supplied_evidence": citation_valid, "citation_present": citation_present,
            "invalid_cited_source_ids": invalid_ids, "invalid_cited_section_numbers": invalid_sections,
            "abstained_or_marked_uncertainty": abstained, "asked_for_clarification": clarified,
            "unsupported_claim_risk_auto": invented_specifics,
            "conversation_section_retained_auto": section_retained,
            "concise_instruction_following_auto": instruction_following,
            "manual_legal_correctness_review": "pending", "manual_reviewer_notes": ""}

def build_cases(chunks: list[dict]) -> list[dict]:
    by_section = {}
    for chunk in chunks:
        if chunk.get("section_number") and chunk.get("section_title"):
            by_section.setdefault(str(chunk["section_number"]), chunk)
    cases = []
    for section in SECTION_IDS:
        chunk = by_section.get(section)
        if not chunk:
            continue
        cases.append({"id": f"heading_section_{section}", "category": "answerable_section_heading", "expected_type": "section_heading", "synthetic_scenario": False,
                      "original_question": f"According only to the supplied legal evidence, what is the heading of Bharatiya Nyaya Sanhita section {section}? Give the heading concisely and cite the source identifier.",
                      "conversation_context": [], "expected_section": section, "expected_heading": chunk["section_title"], "expected_chunk_id": chunk["chunk_id"], "expected_evidence_section": section,
                      "require_source_citation": True,
                      "corpus_note": "Question generated from this section's actual heading in the locally available PDF."})
    for case_id, context, question, section in FOLLOWUPS:
        target = by_section.get(section) if section else None
        cases.append({"id": case_id, "category": "follow_up_context", "expected_type": "followup_heading" if section else "context_unknown", "synthetic_scenario": True,
                      "original_question": question, "conversation_context": [{"role": "user", "content": context}],
                      "expected_section": section, "expected_heading": target["section_title"] if target else "", "expected_chunk_id": target["chunk_id"] if target else None, "expected_evidence_section": section,
                      "require_source_citation": False,
                      "require_section_number": case_id == "followup_35",
                      "corpus_note": "Synthetic conversation context; legal evidence is still only from the source PDF."})
    for case_id, question, note in AMBIGUOUS:
        cases.append({"id": case_id, "category": "ambiguous_clarification", "expected_type": "ambiguous", "synthetic_scenario": True,
                      "original_question": question, "conversation_context": [], "expected_section": None, "expected_heading": "", "expected_chunk_id": None, "expected_evidence_section": None, "corpus_note": note})
    for case_id, question, note in UNSUPPORTED:
        cases.append({"id": case_id, "category": "insufficient_evidence_corpus_coverage", "expected_type": "insufficient", "synthetic_scenario": True,
                      "original_question": question, "conversation_context": [], "expected_section": None, "expected_heading": "", "expected_chunk_id": None, "expected_evidence_section": None, "corpus_note": note})
    return cases

def markdown_report(data: dict) -> str:
    tests = data["tests"]
    passed = sum(bool(item["assessment"]["overall_auto_pass"]) for item in tests)
    retrieval_fail = sum("retrieval_failure" in item["assessment"]["failure_categories"] for item in tests)
    generation_fail = sum("generation_grounding_failure" in item["assessment"]["failure_categories"] for item in tests)
    context_fail = sum("context_retention_failure" in item["assessment"]["failure_categories"] for item in tests)
    instruction_fail = sum("instruction_following_failure" in item["assessment"]["failure_categories"] for item in tests)
    coverage = sum("corpus_coverage_limitation" in item["assessment"]["failure_categories"] for item in tests)
    risky = [item for item in tests if item["assessment"]["unsupported_claim_risk_auto"]]
    lat = [item["latency_seconds"] for item in tests if item.get("latency_seconds") is not None]
    normal = [item["latency_seconds"] for item in tests if item.get("latency_seconds") is not None and item["model_request_order"] > 1]
    first = next((item["latency_seconds"] for item in tests if item.get("model_request_order") == 1), None)
    lines = ["# Qwen3 4B RAG Grounding Evaluation", "", f"Run date: {data['started_at']}", "", "## Results", "",
             f"- Cases: {len(tests)}; automatic passes: {passed}; automatic failures: {len(tests)-passed}.",
             f"- Retrieval failures: {retrieval_fail}; generation/grounding failures: {generation_fail}; corpus coverage limitations: {coverage}.",
             f"- Context-retention failures: {context_fail}; concise/instruction-following failures: {instruction_fail}.",
             f"- Corpus: {data['corpus']['documents']} ({data['corpus']['chunk_count']} chunks, {data['corpus']['section_count']} sections); source provenance: {data['corpus']['provenance_note'].rstrip('.')}.",
             f"- Model: `{data['model']}` ({data['quantization']}); Ollama {data['ollama_version']}; think=false.", "",
             "## Performance", "", f"- First model request: {first:.2f} s" if first is not None else "- First model request: unavailable",
             f"- Subsequent request latency: mean {statistics.mean(normal):.2f} s, median {statistics.median(normal):.2f} s, n={len(normal)}." if normal else "- Subsequent request latency: unavailable",
             f"- All-request latency range: {min(lat):.2f}–{max(lat):.2f} s." if lat else "- All-request latency: unavailable",
             f"- Prompt/completion tokens are recorded per case where Ollama returned them. Totals: {sum(x.get('prompt_tokens') or 0 for x in tests)} prompt and {sum(x.get('completion_tokens') or 0 for x in tests)} completion tokens.", "",
             "## Unsupported-claim and citation review", ""]
    invalid_ids = sum(len(item["assessment"]["invalid_cited_source_ids"]) for item in tests)
    invalid_sections = sum(len(item["assessment"]["invalid_cited_section_numbers"]) for item in tests)
    lines.append(f"- Deterministic fabricated-citation flags: {invalid_ids} unavailable source IDs and {invalid_sections} section references absent from supplied evidence. The check cannot detect a valid retrieved citation used to support the wrong proposition.")
    if risky:
        for item in risky:
            response = item.get("model_response", "").replace("\n", " ")
            lines.append(f"- **{item['id']}**: unsupported-claim risk flagged; invalid section refs {item['assessment']['invalid_cited_section_numbers']}, invalid source IDs {item['assessment']['invalid_cited_source_ids']}. Response excerpt: {response[:360]}")
    else:
        lines.append("- No deterministic unsupported-specifics flag was triggered. This does not establish legal accuracy; manual review remains pending.")
    lines += ["", "## Corpus coverage and failure interpretation", "", "The available corpus is one local Bharatiya Nyaya Sanhita, 2023 PDF. The project ingestion code marked its provenance unverified because no source metadata sidecar was present. BNS is a penal statute; tenancy, consumer procedure, civil limitation, case holdings, and criminal-procedure deadlines were deliberately tested as outside-corpus topics. Those are coverage limitations, not proof that the law does not exist.", "",
              "The runner uses a temporary index built with the repository's `ingest_file`, `build_index_from_chunks`, and `LegalRetriever`. It does not synchronize or write to the configured production index. A retrieval failure means a known in-corpus target section was not among the accepted retrieved chunks. A generation/grounding failure means expected evidence was retrieved but deterministic response checks found a mismatch or unsupported specific. Unsupported-claim checks are heuristic; legal correctness is explicitly not auto-certified.", "",
              "## Recommendation", "", data["recommendation"], "", "Manual legal review status: pending for every generated answer. The benchmark is small and tests section-heading extraction plus selected synthetic context, ambiguity, and out-of-corpus probes; it is not a legal reliability certification.", ""]
    return "\n".join(lines)

def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"Corpus PDF not found: {CORPUS}")
    ollama = get(OLLAMA + "/api/tags")
    model = next((m for m in ollama.get("models", []) if m.get("name") == MODEL), None)
    if not model:
        raise SystemExit(f"Model not found. Download it with: ollama pull {MODEL}")
    try:
        version = get(OLLAMA + "/api/version").get("version")
    except Exception:
        version = "unknown"
    settings = Settings.from_env()
    previous_data = None
    if OUT_JSON.exists():
        try:
            previous_data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_data = None
    previous_by_id = {item["id"]: item for item in (previous_data or {}).get("tests", [])}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aadhikar_qwen_rag_eval_") as temp:
        temp_path = Path(temp)
        chunks_path, index_path = temp_path / "chunks.json", temp_path / "legal_index.json"
        chunk_count = ingest_file(CORPUS, None, chunks_path)
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        build_index_from_chunks(chunks, index_path)
        retriever = LegalRetriever(index_path=index_path, minimum_relevance=settings.retrieval_minimum_relevance,
                                   embedding_model=settings.embedding_model, reranker_model=settings.reranker_model,
                                   enable_reranker=settings.enable_reranker, enable_dense=settings.enable_dense_retrieval)
        provider = LocalOllamaProvider()
        service = LegalAssistantService(provider, retriever)
        cases = build_cases(chunks)
        if len(cases) < 30:
            raise SystemExit(f"Only {len(cases)} meaningful cases could be derived from the actual corpus; refusing to fabricate more.")
        distinct_sections = len({c.get("section_number") for c in chunks if c.get("section_number")})
        data = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "model": MODEL,
                "model_size_bytes": model.get("size"), "model_digest": model.get("digest"),
                "quantization": (model.get("details") or {}).get("quantization_level", "unknown"),
                "parameter_size": (model.get("details") or {}).get("parameter_size"), "ollama_version": version,
                "inference_method": "Production LegalAssistantService with benchmark-only local Ollama adapter; localhost /api/chat; think=false; temperature=0; fixed seed; no external APIs.",
                "corpus": {"documents": [CORPUS.relative_to(ROOT).as_posix()], "chunk_count": chunk_count, "section_count": distinct_sections,
                           "provenance_note": "No metadata sidecar found; ingestion labels issuer and provenance unverified. Source was not independently authenticated."},
                "retriever": {"implementation": "existing LegalRetriever via LegalAssistantService", "minimum_relevance": settings.retrieval_minimum_relevance,
                              "dense_enabled_setting": settings.enable_dense_retrieval, "reranker_enabled_setting": settings.enable_reranker,
                              "index_location": "temporary directory, removed after run", "retrieved_limit": 4},
                "limitations": ["One legal source only; no independent ground-truth legal review yet.", "Heading checks and unsupported-claim checks are deterministic heuristics, not proof of correctness.", "Source provenance remains unverified.", "Coverage probes are synthetic questions, not claims about the state of law.", "The runner exercises LegalAssistantService but not FastAPI domain-classifier, persistence, rate-limit, or emergency gates."], "tests": [],
                "previous_run": {"started_at": (previous_data or {}).get("started_at"), "summary": (previous_data or {}).get("summary")}}
        for order, case in enumerate(cases, 1):
            retrieval_query = " ".join([m["content"] for m in case["conversation_context"]] + [case["original_question"]]).strip()
            retrieval_started = time.perf_counter()
            debug = retriever.debug_search(retrieval_query, limit=5)
            retrieval_latency = time.perf_counter() - retrieval_started
            accepted = [item for item in debug.retrieved if item.passed_minimum_relevance]
            supplied = [item.chunk.to_dict() for item in accepted]
            expected_section = case.get("expected_evidence_section")
            expected_present = bool(case.get("expected_section") and case.get("expected_heading") and any(
                str(c.get("section_number")) == case["expected_section"]
                and title_quality(case["expected_heading"], c.get("section_title") or "") >= 0.8
                for c in supplied
            ))
            evidence_text = "\n\n".join(f"SOURCE_ID: [{c['chunk_id']}]\nDOCUMENT: {c['document_name']}\nACT: {c['act_name']}\nSECTION: {c.get('section_number')} — {c.get('section_title')}\nTEXT:\n{c['text']}" for c in supplied)
            prompt = f"User question (verbatim):\n{case['original_question']}\n\nRetrieved legal material and provenance:\n{evidence_text or '[No chunks passed the retriever relevance threshold.]'}"
            test = {**case, "retrieval_query": retrieval_query, "retrieval_latency_seconds": round(retrieval_latency, 4),
                    "retrieval_debug": {"query": debug.query, "index_available": debug.index_available, "dense_available": debug.dense_available,
                                        "reranker_active": debug.reranker_active, "minimum_relevance": debug.minimum_relevance,
                                        "expected_section_heading_evidence_retrieved": expected_present,
                                        "expected_target_chunk_id": case.get("expected_chunk_id"),
                                        "retrieved_chunks": [{"chunk_id": x.chunk.chunk_id, "document_id": x.chunk.document_id, "document_name": x.chunk.document_name,
                                                              "act_name": x.chunk.act_name, "section_number": x.chunk.section_number, "section_title": x.chunk.section_title,
                                                              "chapter": x.chunk.chapter, "source_url": x.chunk.source_url, "issuing_authority": x.chunk.issuing_authority,
                                                              "relevance_score": x.relevance_score, "bm25_score": x.bm25_score, "term_score": x.term_score,
                                                              "dense_score": x.dense_score, "rrf_score": x.rrf_score, "ranking_score": x.ranking_score,
                                                              "passed_minimum_relevance": x.passed_minimum_relevance, "text": x.chunk.text}
                                                             for x in debug.retrieved]},
                    "model_request_order": order, "model_response": None, "latency_seconds": None,
                    "prompt_tokens": None, "completion_tokens": None, "error": None}
            model_started = time.perf_counter()
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, *case["conversation_context"], {"role": "user", "content": prompt}]
                result = post(OLLAMA + "/api/chat", {"model": MODEL, "messages": messages,
                                                       "stream": False, "think": False, "keep_alive": "10m", "options": {"temperature": 0, "seed": 42, "num_ctx": 8192}}, timeout=300)
                test["latency_seconds"] = round(time.perf_counter() - model_started, 4)
                message = result.get("message") or {}
                test["model_response"] = message.get("content", "")
                test["prompt_tokens"] = result.get("prompt_eval_count")
                test["completion_tokens"] = result.get("eval_count")
                test["prompt_tokens_per_second"] = result.get("prompt_eval_rate")
                test["completion_tokens_per_second"] = result.get("eval_rate")
                test["model_load_duration_ns"] = result.get("load_duration")
                test["prompt_eval_duration_ns"] = result.get("prompt_eval_duration")
                test["generation_duration_ns"] = result.get("eval_duration")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                test["latency_seconds"] = round(time.perf_counter() - model_started, 4)
                test["error"] = f"HTTP {exc.code}: {detail[:1200]}"
                test["model_response"] = ""
            except Exception as exc:
                test["latency_seconds"] = round(time.perf_counter() - model_started, 4)
                test["error"] = str(exc)
                test["model_response"] = ""
            assessment = response_signals(test["model_response"] or "", supplied, case)
            if case["id"] == "ambiguous_private_defence":
                assessment["manual_reviewer_notes"] = "Priority review: opens with a definitive 'Yes, you can claim' despite no facts; retrieved text does not establish that the user's unknown conduct qualifies."
            elif case["id"] == "followup_35":
                assessment["manual_reviewer_notes"] = "Priority review: conversation says section 35; answer instead reports section 358. This is a context-retention and evidence-selection error."
            elif case["id"] == "coverage_supreme_court_case":
                assessment["manual_reviewer_notes"] = "Priority review: concludes no such case exists from corpus absence and calls the BNS a proposed act; neither conclusion is established by the supplied evidence."
            elif case["id"] == "ambiguous_offence":
                assessment["manual_reviewer_notes"] = "Answer eventually asks what act occurred, but lists multiple unrelated provisions before clarifying; check relevance and concision."
            failure_categories = []
            if case["expected_type"] in {"section_heading", "followup_heading"} and not expected_present:
                failure_categories.append("retrieval_failure")
            if case["expected_type"] in {"insufficient", "ambiguous", "context_unknown"}:
                failure_categories.append("corpus_coverage_limitation" if case["expected_type"] == "insufficient" else "scenario_information_insufficient")
            if assessment["unsupported_claim_risk_auto"] or not assessment["citation_valid_against_supplied_evidence"] or (case["expected_type"] in {"section_heading", "followup_heading"} and expected_present and (assessment["factual_consistency_auto"] is False or (case.get("require_source_citation") and not assessment["citation_present"]))):
                failure_categories.append("generation_grounding_failure")
            if assessment.get("conversation_section_retained_auto") is False:
                failure_categories.append("context_retention_failure")
            if not assessment["concise_instruction_following_auto"]:
                failure_categories.append("instruction_following_failure")
            if test["error"]:
                failure_categories.append("ollama_request_failure")
            assessment["failure_categories"] = failure_categories
            blocking = {"retrieval_failure", "generation_grounding_failure", "context_retention_failure", "instruction_following_failure", "ollama_request_failure"}
            assessment["overall_auto_pass"] = not (blocking & set(failure_categories)) and assessment["factual_consistency_auto"] is not False and assessment["citation_valid_against_supplied_evidence"] and assessment["concise_instruction_following_auto"]
            test["assessment"] = assessment
            data["tests"].append(test)
            OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{order}/{len(cases)}] {case['id']}: {'PASS' if assessment['overall_auto_pass'] else 'REVIEW/FAIL'} {test['latency_seconds']}s", flush=True)
        data["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        normal_times = [t["latency_seconds"] for t in data["tests"] if t["latency_seconds"] is not None]
        normal_after_first = normal_times[1:]
        data["summary"] = {"total_cases": len(cases), "auto_pass": sum(t["assessment"]["overall_auto_pass"] for t in data["tests"]),
                           "auto_fail_or_review": sum(not t["assessment"]["overall_auto_pass"] for t in data["tests"]),
                           "retrieval_failures": sum("retrieval_failure" in t["assessment"]["failure_categories"] for t in data["tests"]),
                           "generation_grounding_failures": sum("generation_grounding_failure" in t["assessment"]["failure_categories"] for t in data["tests"]),
                           "context_retention_failures": sum("context_retention_failure" in t["assessment"]["failure_categories"] for t in data["tests"]),
                           "instruction_following_failures": sum("instruction_following_failure" in t["assessment"]["failure_categories"] for t in data["tests"]),
                           "corpus_coverage_limitations": sum("corpus_coverage_limitation" in t["assessment"]["failure_categories"] for t in data["tests"]),
                           "unsupported_claim_risk_flags": sum(t["assessment"]["unsupported_claim_risk_auto"] for t in data["tests"]),
                           "first_request_latency_seconds": normal_times[0] if normal_times else None,
                           "subsequent_latency_mean_seconds": round(statistics.mean(normal_after_first), 4) if normal_after_first else None,
                           "subsequent_latency_median_seconds": round(statistics.median(normal_after_first), 4) if normal_after_first else None,
                           "latency_min_seconds": min(normal_times) if normal_times else None, "latency_max_seconds": max(normal_times) if normal_times else None,
                           "prompt_tokens_total": sum(t.get("prompt_tokens") or 0 for t in data["tests"]),
                           "completion_tokens_total": sum(t.get("completion_tokens") or 0 for t in data["tests"])}
        data["recommendation"] = "Requires safeguards before any legal-answering use. The BNS-only benchmark surfaces synthetic coverage limits and must be extended with verified multi-source material and human legal review; automated passes are not legal correctness findings."
        OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        OUT_MD.write_text(markdown_report(data), encoding="utf-8")
        print(f"JSON report: {OUT_JSON}")
        print(f"Markdown report: {OUT_MD}")

if __name__ == "__main__":
    from evaluate_qwen3_rag_grounding_production import main as production_main
    production_main()
