# Architecture

## Legal First Responder retrieval flow

```text
Input → scope guard → urgency assessment → issue classification → legal retrieval
→ grounded LLM → citation validation → response validation → disclaimer → user
```

`app.retrieval` ingests authoritative documents into section-preserving chunks and stores complete provenance. It then builds a local term-vector index. The retriever returns chunks, relevance scores, and provenance; a result below the configurable threshold prevents an LLM call. `app.issue_classifier` supplies extensible routing hints, not legal determinations.

The API returns user-facing citations (`source_name`, section, URL, version), without exposing chunk IDs. Browser speech synthesis receives only the answer text. Immediate-danger wording produces a short safety-first response before retrieval.

The product is a voice-enabled Indian legal first responder. It accepts both legal terminology and ordinary-language descriptions of plausible legal situations. The guard is deliberately broader than retrieval: it admits a possible legal situation without classifying an offence, while retrieval determines whether authoritative material is available. The browser captures speech when supported, submits only the final text transcript, and speaks the server response using browser synthesis.

```text
Microphone → browser SpeechRecognition → transcript → POST /api/chat
  → deterministic request guard → provider-neutral LLM adapter → deterministic output validator
  → enforced disclaimer → browser UI → browser SpeechSynthesis
```

The FastAPI backend is the trust boundary. It owns the legal-domain guard, response validator, session request counter, LLM API key, prompt, and provider adapter. The React client never receives an LLM secret and offers typed input if speech recognition is unavailable.

`LLMProvider` is the stable integration interface. `GroqProvider` is its first implementation. Future RAG can be added as a retrieval service between the request guard and provider without changing the voice layer or `/api/chat` contract. Phase 1 deliberately contains no retrieval, vector database, persistent database, authentication, or audio storage.
