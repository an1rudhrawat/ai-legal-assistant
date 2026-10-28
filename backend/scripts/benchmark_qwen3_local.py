"""Standalone local Ollama benchmark for Qwen3; uses only Python's standard library."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen3:4b-instruct"
BASE_URL = "http://localhost:11434"
CASES = [
    {"id": "logic_1", "category": "multi_step_logic", "messages": [{"role": "user", "content": "Asha is older than Bala. Bala is older than Chitra. Deepa is younger than Asha but older than Bala. List all four from oldest to youngest."}], "expected": "Asha, Deepa, Bala, Chitra", "checks": ["asha", "deepa", "bala", "chitra"]},
    {"id": "logic_2", "category": "multi_step_logic", "messages": [{"role": "user", "content": "A train leaves at 09:15 and travels 120 km at a constant 60 km/h. It stops for 20 minutes, then travels another 30 km at 60 km/h. At what time does it arrive? Show the time calculation briefly."}], "expected": "12:05 (2 hours + 20 minutes + 30 minutes)", "checks": ["12:05"]},
    {"id": "context_followup", "category": "conversation_context", "messages": [{"role": "user", "content": "My hearing is on 18 March 2027 at the District Court in Pune."}, {"role": "assistant", "content": "Understood. Your hearing is on 18 March 2027 at the District Court in Pune."}, {"role": "user", "content": "Which court and city did I mention, and on what date? Answer in one sentence."}], "expected": "District Court in Pune, 18 March 2027", "checks": ["district court", "pune", "18 march 2027"]},
    {"id": "ambiguity", "category": "clarification", "messages": [{"role": "user", "content": "Can I file it tomorrow?"}], "expected": "Ask what 'it' is and relevant jurisdiction/deadline details before answering.", "checks": ["what", "it"], "any_checks": ["file", "document", "case", "application", "jurisdiction", "deadline", "which"]},
    {"id": "evidence_legal", "category": "supplied_evidence", "messages": [{"role": "system", "content": "Answer only from the supplied fictional sample evidence. Do not present it as real law. If the evidence does not answer something, say so."}, {"role": "user", "content": "Sample evidence (fictional): Rule 7 says a written application must be submitted to the Clerk within 14 calendar days after receiving notice. If the notice is received on 1 June, what is the deadline under this rule?"}], "expected": "15 June if counting starts the day after receipt; acknowledge the sample says 14 days and counting convention is not explicitly defined.", "checks": ["14", "june"], "any_checks": ["count", "calendar", "15"]},
    {"id": "insufficient_evidence", "category": "uncertainty", "messages": [{"role": "system", "content": "Use only this supplied evidence. Do not fill gaps with outside knowledge."}, {"role": "user", "content": "Evidence: The agreement was signed on 4 April. Does it require a written notice before termination?"}], "expected": "State evidence is insufficient to determine the notice requirement.", "any_checks": ["not", "insufficient", "doesn't say", "does not say", "cannot determine", "can't determine", "unknown", "not provided", "need more"]},
    {"id": "instruction_concise", "category": "instruction_following", "messages": [{"role": "user", "content": "In exactly three bullet points, explain what documents a person might gather before asking a lawyer about a rental deposit dispute. Keep each bullet under 10 words; give general preparation only, no legal conclusion."}], "expected": "Exactly 3 bullets, each under 10 words, preparatory documents only.", "checks": ["- "], "special": "three_bullets"},
    {"id": "legal_uncertainty", "category": "legal_uncertainty", "messages": [{"role": "user", "content": "I live in India. My landlord kept my deposit. Can you tell me exactly which law guarantees I will get it back? If you need facts or documents, say so."}], "expected": "Avoid guaranteeing outcome or inventing a specific law; ask for state, agreement and facts and suggest qualified help/source verification.", "any_checks": ["state", "agreement", "depend", "document", "cannot", "can't", "need", "not guarantee", "not enough", "lawyer", "legal advice"]},
    {"id": "simple_arithmetic", "category": "reasoning_correctness", "messages": [{"role": "user", "content": "A filing fee is ₹250 per form. I submit 3 forms and receive a ₹100 waiver. What is the amount due? Reply with the calculation and total."}], "expected": "3 × ₹250 − ₹100 = ₹650", "checks": ["650"]},
    {"id": "context_not_invent", "category": "context_retention", "messages": [{"role": "user", "content": "My appointment is in Jaipur next week."}, {"role": "assistant", "content": "You have an appointment in Jaipur next week."}, {"role": "user", "content": "What time is it?"}], "expected": "Say no time was mentioned and ask the user to provide it.", "any_checks": ["not mention", "didn't mention", "no time", "what time", "don't know", "not specified", "not provided", "tell me", "provide"]},
]
THINK_CASES = [
    {"id": "think_logic", "category": "thinking_enabled_reasoning", "messages": [{"role": "user", "content": "Three boxes are labeled Apples, Oranges, and Mixed. Every label is wrong. You may draw one fruit from one box. Which box should you draw from to identify all boxes, and why? Give a short answer."}], "expected": "Draw from the box labeled Mixed; since label is wrong it contains only one fruit type; use that to infer remaining labels.", "checks": ["mixed"], "any_checks": ["one fruit", "one type", "only apples", "only oranges"]},
    {"id": "think_evidence", "category": "thinking_enabled_evidence", "messages": [{"role": "system", "content": "Use only the supplied fictional evidence. Be explicit about missing information."}, {"role": "user", "content": "Evidence: A notice was delivered on 2 August. A rule allows an appeal within 10 days. The evidence does not define whether the delivery day counts. What is the deadline?"}], "expected": "Identify ambiguity: 10 August if delivery day counts, 12 August if next day starts count; request applicable counting rule.", "checks": ["10 august"], "any_checks": ["12 august", "depends", "ambiguous", "count", "not specify", "not defined"]},
]

def get_json(url: str, timeout: int = 5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def post_json(url: str, body: dict, timeout: int = 300):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def evaluate(case: dict, response: str) -> dict:
    lowered = response.lower()
    checks = case.get("checks", [])
    passed = all(item.lower() in lowered for item in checks)
    any_checks = case.get("any_checks", [])
    if any_checks:
        passed = passed and any(item.lower() in lowered for item in any_checks)
    if case.get("special") == "three_bullets":
        bullets = [line for line in response.splitlines() if line.strip().startswith(("-", "•", "*"))]
        passed = len(bullets) == 3 and all(len(line.lstrip("-*• ").split()) < 10 for line in bullets)
    if case["id"] == "legal_uncertainty":
        # A merely cautious opening does not offset fabricated legal authorities later in the answer.
        risky_claims = ["section 10", "section 11", "section 12", "30 days", "s. k. singh", "guarantees that your deposit", "civil suit", "rent control authority"]
        passed = passed and not any(claim in lowered for claim in risky_claims)
    if case["id"] == "think_evidence":
        # Here identifying the missing counting convention is the correct result; do not reward guessed dates.
        passed = any(term in lowered for term in ["cannot be determined", "not explicitly defined", "unknown due", "cannot determine"])
    return {"pass": bool(passed), "assessment": "Pass: deterministic checks met." if passed else "Fail: one or more deterministic checks not met; review response manually."}

def hardware_info() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "processor_env": __import__("os").environ.get("PROCESSOR_IDENTIFIER"), "logical_processors_env": __import__("os").environ.get("NUMBER_OF_PROCESSORS"), "note": "Memory and GPU metadata were unavailable through this environment's system inventory permissions."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "evaluation" / "qwen3_local_benchmark_results.json")
    args = parser.parse_args()
    try:
        tags = get_json(args.url.rstrip("/") + "/api/tags")
    except Exception as exc:
        raise SystemExit(f"Ollama API unavailable at {args.url}: {exc}")
    match = next((model for model in tags.get("models", []) if model.get("name") == MODEL), None)
    if not match:
        raise SystemExit(f"{MODEL} not found. Download with: ollama pull {MODEL}")
    info = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "model": MODEL,
        "model_size_bytes": match.get("size"), "digest": match.get("digest"),
        "quantization": (match.get("details") or {}).get("quantization_level"),
        "parameter_size": (match.get("details") or {}).get("parameter_size"),
        "ollama_version": None, "hardware": hardware_info(),
        "method": "Ollama local /api/chat; think:false for latency cases and think:true for separate reasoning cases; no external LLM API.",
        "limitations": ["Small hand-built sample; not a legal reliability evaluation.", "String-based automatic checks are approximate and require human review.", "Latency depends on local hardware, Ollama settings, thermal state, and concurrent load.", "Thinking text is not evidence of correctness; answers are assessed against expected outcomes."],
        "tests": [],
    }
    try:
        version = get_json(args.url.rstrip("/") + "/api/version")
        info["ollama_version"] = version.get("version")
    except Exception:
        pass
    for think, cases in ((False, CASES), (True, THINK_CASES)):
        for index, case in enumerate(cases):
            payload = {"model": MODEL, "messages": case["messages"], "stream": False, "think": think,
                       "keep_alive": "10m", "options": {"temperature": 0, "seed": 42}}
            started = time.perf_counter()
            try:
                result = post_json(args.url.rstrip("/") + "/api/chat", payload, 300)
                elapsed = time.perf_counter() - started
                message = result.get("message") or {}
                answer = message.get("content", "")
                thinking = message.get("thinking")
                test = {"id": case["id"], "category": case["category"], "thinking_enabled": think,
                        "request_order_in_mode": index + 1, "first_request_in_mode": index == 0,
                        "input_messages": case["messages"], "expected_outcome": case["expected"],
                        "response": answer, "thinking_output": thinking if think else None,
                        "latency_seconds": round(elapsed, 3), "prompt_tokens": result.get("prompt_eval_count"),
                        "completion_tokens": result.get("eval_count"),
                        "prompt_tokens_per_second": result.get("prompt_eval_rate"),
                        "completion_tokens_per_second": result.get("eval_rate"),
                        "load_duration_ns": result.get("load_duration"),
                        "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
                        "eval_duration_ns": result.get("eval_duration"), **evaluate(case, answer)}
            except Exception as exc:
                test = {"id": case["id"], "category": case["category"], "thinking_enabled": think,
                        "request_order_in_mode": index + 1, "first_request_in_mode": index == 0,
                        "input_messages": case["messages"], "expected_outcome": case["expected"],
                        "latency_seconds": round(time.perf_counter() - started, 3), "error": str(exc),
                        "pass": False, "assessment": "Request failed."}
            info["tests"].append(test)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{test['id']}: {'PASS' if test['pass'] else 'FAIL'} ({test['latency_seconds']}s)")
    normal = [t for t in info["tests"] if not t["thinking_enabled"] and "latency_seconds" in t]
    later = [t["latency_seconds"] for t in normal[1:]]
    info["summary"] = {"normal_tests": len(CASES), "normal_passes": sum(t["pass"] for t in normal),
                       "thinking_tests": len(THINK_CASES), "thinking_passes": sum(t["pass"] for t in info["tests"] if t["thinking_enabled"]),
                       "first_normal_request_latency_seconds": normal[0]["latency_seconds"] if normal else None,
                       "subsequent_normal_latency_mean_seconds": round(statistics.mean(later), 3) if later else None,
                       "subsequent_normal_latency_median_seconds": round(statistics.median(later), 3) if later else None}
    info["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    info["overall_assessment"] = "Results are descriptive only. Review each response and assess integration fit separately; this sample cannot establish legal reliability."
    args.output.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
