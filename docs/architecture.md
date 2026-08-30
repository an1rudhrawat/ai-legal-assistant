# Architecture

Phase 1 is a voice-enabled assistant for Indian legal terminology and general legal concepts. The browser captures speech when supported, submits only the final text transcript, and speaks the server response using browser synthesis.

```text
Microphone → browser SpeechRecognition → transcript → POST /api/chat
  → deterministic request guard → provider-neutral LLM adapter → deterministic output validator
  → enforced disclaimer → browser UI → browser SpeechSynthesis
```

The FastAPI backend is the trust boundary. It owns the legal-domain guard, response validator, session request counter, LLM API key, prompt, and provider adapter. The React client never receives an LLM secret and offers typed input if speech recognition is unavailable.

`LLMProvider` is the stable integration interface. `GroqProvider` is its first implementation. Future RAG can be added as a retrieval service between the request guard and provider without changing the voice layer or `/api/chat` contract. Phase 1 deliberately contains no retrieval, vector database, persistent database, authentication, or audio storage.
