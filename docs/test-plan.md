# Test plan

## What is automated

Backend tests use a fake `LLMProvider`, local fixture indexes, and temporary SQLite databases. They do not send requests to Groq or Ollama and do not consume API quota.

- **Request and domain guards:** legal, non-legal, ambiguous, mixed-topic, and instruction-override inputs; rejected requests avoid provider calls where expected.
- **Situation and issue hints:** extraction of neutral user-reported facts, urgency indicators, and retrieval concepts without deciding an offence.
- **Retrieval and corpus:** chunking, provenance, section matching, local ranker behavior, corpus sync, and offline corpus-aware retrieval metrics.
- **Grounding and citations:** retrieved evidence reaches the provider prompt; valid, missing, unknown, and inconsistent source IDs; unsupported section and punishment details; insufficient evidence responses.
- **Conversations and API:** conversation isolation and persistence, message ordering, contextual follow-ups, summaries, request idempotency, deletion, API schema, and database/provider failure paths.
- **Response scope:** selected off-topic output and unsupported patterns are replaced by a fixed response.
- **Frontend:** typed-input fallback, chat turn rendering and scrolling, citation presentation, formatting, disclaimer visibility, submit behavior, and mocked browser speech APIs.

## Commands

From `backend`:

```powershell
python -m pytest
```

From `frontend`:

```powershell
npm test
npm run build
```

## What tests do not establish

- A fake provider does not measure actual Groq latency, rate limits, token use, or answer quality.
- Fixture documents do not verify retrieval quality over every file in `data/raw`.
- Rule-based validators do not establish semantic support or legal correctness for every claim.
- Passing tests do not show that the corpus covers a question or that a legal rule applies to a user's facts.
- Browser mocks do not verify microphone behavior or speech synthesis on real browsers and devices.
- Qwen evaluation reports are separate experiments and are not part of the application unit-test suite.

Use corpus-aware retrieval evaluation and manual review alongside automated tests. For higher-stakes answer-quality work, review actual outputs and evidence with a qualified legal reviewer.
