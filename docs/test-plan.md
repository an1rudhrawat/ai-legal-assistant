# Test plan

Backend tests use a fake LLM provider; no test sends an external LLM request or consumes quota.

- Legal terminology and concept requests are allowed.
- Non-legal, ambiguous, mixed-topic, and jailbreak-style inputs are rejected.
- Rejected inputs do not call the provider.
- Output validation rejects code, off-scope text, medical/financial advice, and foreign law presented as controlling.
- API responses conform to the chat and health contracts.
- A mocked provider 429 produces the fixed busy message.
- The in-memory per-session cap prevents extra provider calls.
- An exact-section request with a confident mock citation is replaced by a hedged response instead of presenting an unverified section as fact.

Frontend Vitest tests cover the persistent disclaimer, typed fallback when browser recognition is unavailable, successful response rendering, and legal-refusal rendering. Speech browser APIs are mocked or intentionally unavailable in these tests.
