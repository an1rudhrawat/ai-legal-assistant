# Qwen3 4B RAG Grounding Evaluation (Production Service)

Run date: 2026-10-25T12:12:10.171930+00:00

## Results

- Cases: 36; model calls: 24; service clarification/abstention without Qwen: 12.
- Previous direct-run score: 32/36. Production-path changes from that run: 4 cases; raw comparison unavailable for model-skipped cases: followup_no_time, ambiguous_private_defence, ambiguous_offence, ambiguous_punishment, ambiguous_arrest, coverage_rent_deposit, coverage_consumer_appeal, coverage_supreme_court_case, coverage_civil_remedy, coverage_bail_deadline, coverage_land_title, coverage_limitation.
- Production-service response automatic passes: 36/36; review/fail: 0.
- Raw Qwen calls scored: 23/24 pass; raw generation/grounding risk flags: 1. Known-section retrieval failures: 0; BNS corpus-coverage probes: 8.
- Citation checks: 1 responses used an unavailable source ID; 0 heading answers omitted a required source citation. Context-retention failures: 0; instruction-following failures: 0.
- Exact heading metadata supplied canonical citations after strict ID validation in 3 cases; malformed IDs themselves were not accepted.
- Corpus: ['data/raw/bharatiya-nyaya-sanhita-2023.pdf'] (358 chunks, 358 sections); provenance unverified.
- Model: `qwen3:4b-instruct` (Q4_K_M); Ollama 0.40.11; thinking disabled.

## Performance

- First Ollama request: 3.2372 s; subsequent mean/median: 3.0201 / 3.0126 s.
- Prompt/completion tokens: 35942 / 1284.

## Reviewed prior failure cases

| Case | Previous | Production response |
|---|---:|---:|
| `heading_section_14` | FAIL | PASS |
| `ambiguous_private_defence` | FAIL | PASS |
| `ambiguous_offence` | FAIL | PASS |
| `ambiguous_punishment` | FAIL | PASS |

### heading_section_14
- Prior raw Qwen response: The heading of Bharatiya Nyaya Sanhita section 14 is:   **Act done by a person bound, or by mistake of fact believing himself bound, by law.**  Source identifier: [bharatiya-nyaya-sanh-2023-chunk-14]
- Current raw Qwen response: **Act done by a person bound, or by mistake of fact believing himself bound, by law**  [bharatiya-nyaya-sanhita-2023-chunk-14]  This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.
- User-facing service response: **Act done by a person bound, or by mistake of fact believing himself bound, by law**  [bharatiya-nyaya-sanhita-2023-chunk-14]  This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.
### ambiguous_private_defence
- Prior raw Qwen response: Yes, you can claim private defence ... followed by unsupported claims about Sections 34, 40, 44, 36, and 41, proportionality, and recommended next steps.
- Current raw Qwen response: [not called; service guard handled this case]
- User-facing service response: Please describe what happened, who was involved, and what action you are currently being asked to take. If you are asking about punishment, name the offence or legal provision you mean. I can then check whether relevant legal information is available.  This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.
### ambiguous_punishment
- Prior raw Qwen response: Yes, you can be punished under Section 173 ... if bribery; the response supplied a one-year term, a fine, and a treating exception without the user specifying conduct or a provision.
- Current raw Qwen response: [not called; service guard handled this case]
- User-facing service response: Please describe what happened, who was involved, and what action you are currently being asked to take. If you are asking about punishment, name the offence or legal provision you mean. I can then check whether relevant legal information is available.  This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.

## Remaining production-path failures


## Interpretation

Section 14's stored retrieved ID and prompt ID are canonical and equal; the previous Qwen response dropped characters from the ID. The prior runner bypassed production validation, so it recorded the raw failure rather than the user-facing fallback. Routing that output through the current validator rejects it; this is a model citation-format error plus a runner-scope limitation, not a retrieval or validator defect.
The previous private-defence and punishment outputs were raw-model outputs from a runner that bypassed LegalAssistantService. The current runner invokes that service; its fact-dependent guard returns a clarification without calling Qwen for those exact no-fact questions. The benchmark now records both raw model output and the final service response.
The new deterministic citation check requires at least one supplied source ID, rejects unknown IDs and section references, and returns source IDs in the API citation objects. A narrow penalty-duration check rejects explicit year/month/day/amount claims absent from retrieved text. Other semantic claims, offence classification, proportionality, and whether a provision applies to facts still cannot be fully verified without a legal adjudication component; citations can exist and still fail to support a proposition.
When a heading-only response's complete cleaned text exactly matches a unique retrieved section title that also appears in that chunk, the service discards any malformed model ID and emits that chunk's canonical ID. It logs and records this limited repair; unknown IDs in other answers remain rejected, and answers with extra legal claims are not repaired.
The runner exercises LegalAssistantService, but not FastAPI's domain classifier, emergency routing, persistence, or rate limiter. The BNS-only corpus cannot substantiate tenancy, consumer, case-law, civil-remedy, or criminal-procedure coverage.
The current application settings select `groq` / `openai/gpt-oss-20b`. This evaluation injects `qwen3:4b-instruct` locally and does not change or exercise `create_provider`.

## Recommendation

Requires safeguards before legal-answering use. The production service safely intercepts the tested fact-free liability questions and rejects invalid citations, but semantic legal correctness, source provenance, and broader corpus coverage still need expert review.

Manual legal correctness remains pending. Do not treat automated passes as legal validation.
