# Project guide

This guide explains how to install, run, configure, and maintain Aadhikar AI Legal Assistant. It describes the current implementation; it is not a promise that every subject of Indian law is covered.

## 1. What the project does

Aadhikar accepts typed questions and, when supported by the browser, speech input. The backend checks the request, searches a local index of legal sources, and asks Groq to draft a response from the retrieved evidence. It validates citations and response scope, adds a disclaimer, saves the conversation, and returns the result to the browser.

The system has two very different model paths:

- **Application answers:** use the Groq API. `LLM_PROVIDER=groq` is currently the only provider implemented by the application.
- **Qwen experiments:** the standalone scripts under `backend/scripts` call a local Ollama server for benchmarking. They do not switch or configure the application provider.

The retriever and the domain classifier are local. Legal source coverage depends on the files in `data/raw`; the system should abstain when useful evidence is unavailable. Retrieval scores indicate text relevance, not legal sufficiency or correctness.

## 2. Requirements

- Windows PowerShell instructions are shown below; the Python and npm commands also work on other platforms with the appropriate virtual-environment activation command.
- Python 3.11 or newer.
- Node.js 20 or newer and npm.
- A Groq API key for generated application answers.
- Network access for installing packages and, if needed, downloading the local `all-MiniLM-L6-v2` model during classifier training.
- Optional: Ollama and the benchmark model for Qwen evaluation only.

## 3. Install and start

### Backend

From the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env` and set `GROQ_API_KEY`. The `.env` file is local configuration and must not be committed. If PowerShell blocks activation, use the Python executable directly instead, for example `\.venv\Scripts\python.exe -m pytest`.

Train the local semantic domain classifier if `data/domain_classifier/model/classifier.joblib` is missing:

```powershell
python scripts\train_domain_classifier.py --allow-download
```

The embedding model is downloaded only when `--allow-download` is supplied and it is not already cached. The app itself loads local models only; it does not download them while serving requests.

Build or refresh the legal index, then start the API:

```powershell
python -m app.retrieval.sync
uvicorn app.main:app --reload
```

The API listens on `http://localhost:8000`. Its interactive OpenAPI page is `http://localhost:8000/docs`; `GET /health` is a simple liveness response and does not check Groq, model, or index availability.

### Frontend

In a second terminal, from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`. Browser microphone support varies; if it is unavailable, type into the composer. The app sends the final text transcript to the backend, not an audio file.

To point the frontend at a different API address, copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL`. Restart Vite after changing frontend environment variables.

## 4. Configuration

Backend settings are read from `backend/.env`. Relative data paths are resolved from the repository root, not the current terminal directory.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | Provider selector. Only `groq` is implemented. |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model identifier sent to Groq. |
| `GROQ_API_KEY` | empty | Required to generate answers and conversation summaries. |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq-compatible API base URL. |
| `CORS_ORIGINS` | local Vite origins | Comma-separated browser origins allowed by the API. Localhost origins are also added automatically. |
| `SESSION_REQUEST_LIMIT` | `20` | Allowed chat requests per session ID, held in backend process memory. |
| `RETRIEVAL_INDEX_PATH` | `data/vector_store/legal_index.json` | Local JSON search index. |
| `RETRIEVAL_RAW_PATH` | `data/raw` | Input legal sources. |
| `RETRIEVAL_PROCESSED_PATH` | `data/processed` | Persisted chunks and sync manifest. |
| `RETRIEVAL_MINIMUM_RELEVANCE` | `0.45` | Minimum retrieval relevance score for a chunk to be passed to generation. This is not a legal-confidence score. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer used by dense retrieval and the domain classifier. |
| `ENABLE_DENSE_RETRIEVAL` | `true` | Enables local dense retrieval when the model is available on disk. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder name used when reranking is enabled. |
| `ENABLE_RERANKER` | `false` | Enables optional local cross-encoder reranking. |
| `DOMAIN_CLASSIFIER_MODEL_PATH` | `data/domain_classifier/model` | Folder containing the trained classifier. |
| `DOMAIN_LEGAL_THRESHOLD` | `0.70` | Legal probability at or above which the classifier labels a request legal. |
| `DOMAIN_NON_LEGAL_THRESHOLD` | `0.35` | Legal probability at or below which it labels a request non-legal. Values between thresholds are ambiguous. |
| `CONVERSATION_DATABASE_PATH` | `data/conversations.sqlite3` | Local SQLite conversation database. |
| `CONVERSATION_HISTORY_LIMIT` | `6` | Number of prior messages kept in recent model context; valid values are 2–10. |

`LEGAL_GUARD_STRICT` appears in the sample environment file but is not currently used by the runtime pipeline. Changing it does not change guard behavior.

The frontend has one setting, `VITE_API_BASE_URL`, which defaults to `http://localhost:8000`.

## 5. Legal corpus and retrieval

### Add a source

Put authoritative legal material in `data/raw`. Supported source formats are PDF, TXT, Markdown, and JSON. Prefer official legislation, gazette material, government procedure, and court documents. Do not add blogs or unsourced summaries.

For a PDF named `example-act.pdf`, an optional companion file named `example-act.metadata.json` can provide citation provenance:

```json
{
  "document_id": "example-act-2024",
  "document_name": "Example Act, 2024",
  "act_name": "Example Act",
  "document_type": "legislation",
  "source_url": "https://official.example.gov/act.pdf",
  "issuing_authority": "Issuing authority",
  "document_version": "2024 text",
  "effective_date": null,
  "publication_date": null
}
```

The required metadata fields are `document_id`, `document_name`, `act_name`, `document_type`, `source_url`, and `issuing_authority`. Each `document_id` must be unique. If metadata is omitted, the importer labels provenance as unverified; a filename alone is not evidence that a document is current or official. A JSON source may include its text and metadata in the same file.

In the current workspace, `data/raw` contains a Bharatiya Nyaya Sanhita (BNS) PDF. It does not include every law needed for broad Indian legal questions; in particular, state tenancy rules and separate criminal-procedure or anti-corruption materials must be added if those topics are to be covered.

### Build and inspect the index

The application synchronizes `data/raw` at startup and before normal chat processing. It preserves chunks for unchanged files, reprocesses changed files, and builds `data/vector_store/legal_index.json` from active chunks. Run the same process manually from `backend`:

```powershell
python -m app.retrieval.sync
```

The pipeline is local: it extracts text, makes section-aware chunks, computes term vectors, and writes a JSON index. Search combines BM25, term-vector similarity, and (when locally available) dense MiniLM similarity using reciprocal-rank fusion. A local cross-encoder can rerank results when `ENABLE_RERANKER=true` and its model is cached. The default index is a JSON file; the project does not use a hosted vector database.

If dense retrieval or reranking models are unavailable locally, the retriever logs a warning and falls back to its other local ranking signals. A query that retrieves no chunk above the configured relevance threshold returns the insufficient-evidence response without calling Groq.

### Evaluate retrieval

The corpus-aware test cases are in `backend/evaluation/retrieval_cases.json`. From `backend`, run:

```powershell
python scripts\evaluate_retrieval.py
python scripts\evaluate_retrieval.py --debug-case <case-id>
```

This script is deterministic/local and prints a JSON report to the terminal. It evaluates whether expected documents/sections were retrieved; it does not evaluate generated legal answers.

## 6. Conversation and API behavior

The browser stores its active conversation ID in session storage. The backend stores each completed user/assistant turn and its response payload in SQLite. By default, the latest six prior messages are included in model context. Older turns are rolled into a short continuity summary generated through the configured provider. Summaries contain user-reported context only and are not legal evidence.

For a referential follow-up, the latest assistant reply may also be added to the retrieval query as a bounded, explicitly unverified search hint. Current legal claims must still be supported by chunks retrieved for that request. The original current question is passed separately and unchanged to the answer prompt.

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Basic API liveness response. |
| `POST /api/conversations` | Create an empty conversation; returns its UUID and creation time. |
| `GET /api/conversations/{conversation_id}` | Read saved messages for a conversation. |
| `DELETE /api/conversations/{conversation_id}` | Delete the conversation and its messages. |
| `POST /api/chat` | Submit a question; creates a conversation when no ID is supplied. |

Example chat request:

```json
{
  "message": "What does the supplied source say about this issue?",
  "session_id": "browser-session-identifier",
  "conversation_id": "00000000-0000-4000-8000-000000000000",
  "request_id": "00000000-0000-4000-8000-000000000001"
}
```

`message` is limited to 3,000 characters. `session_id` must be 8–128 letters, numbers, underscores, or hyphens. `conversation_id` may be omitted. `request_id` is optional; reusing it with the same message returns the saved result, while reusing it with a different message returns a conflict.

Chat responses include the answer text, conversation ID, refusal/rate-limit flags, optional urgency and issue-category labels, and citations. Citation objects contain a source ID, source name, section, URL, and document version where available. The frontend displays human-readable citation details and does not show the internal source ID.

## 7. Safety, privacy, and current limits

- The assistant is not a lawyer and does not determine guilt, liability, a defence, or whether a provision applies to a user's situation.
- The only legal evidence for generated legal claims is the material retrieved for that request. An answer can be refused because the corpus is missing a relevant law even when the law exists elsewhere.
- Citation checks verify that cited IDs came from retrieved chunks and reject some unsupported section and penalty references. They do **not** prove that every sentence is legally correct or semantically supported.
- The local domain classifier routes requests; it is not a legal classifier. If model files are missing, requests are treated conservatively as ambiguous.
- Conversation contents are stored locally, but recent messages, a continuity summary when generated, the current question, and retrieved legal context are sent to Groq for inference. Do not submit sensitive case details unless this data flow is acceptable.
- The app has no sign-in, user accounts, or ownership checks for conversation IDs. A UUID is not authentication. This prototype is not ready for deployment with private user conversations.
- The request counter is held in process memory; it is not a distributed or per-account quota system.
- Speech recognition is browser-provided and may depend on browser-specific services. The app does not store an audio recording.
- The legal corpus is incomplete and source provenance must be maintained. A single statute cannot answer questions governed by missing central or state laws.

## 8. Tests and model evaluation

Run backend and frontend checks from their respective directories:

```powershell
cd backend
python -m pytest

cd ..\frontend
npm test
npm run build
```

Backend tests use a fake provider and local fixture indexes; they do not call Groq or Ollama. Frontend tests run in Vitest/jsdom and do not exercise real browser microphone or speech services. See [the test plan](test-plan.md).

Optional Qwen scripts use local Ollama and are separate from application serving:

- `backend/scripts/benchmark_qwen3_local.py` tests general response behavior and writes JSON to `backend/evaluation/qwen3_local_benchmark_results.json` by default. `--url` and `--output` can override its Ollama URL and output path.
- `backend/scripts/evaluate_qwen3_rag_grounding.py` runs a standalone local RAG-grounding evaluation and writes JSON and Markdown reports under `backend/evaluation`.
- `backend/scripts/evaluate_qwen3_rag_grounding_production.py` exercises the existing `LegalAssistantService` with a temporary local Qwen provider for comparison; it does not edit production configuration or switch the application provider.

The evaluations are diagnostic samples, not proof of legal reliability. Read the report's corpus limits and manual-review notes; never treat an automated pass or a valid citation ID as a legal correctness guarantee.

## 9. Useful directories

```text
backend/app/                 FastAPI routes, prompts, services, providers, guards, and RAG
backend/tests/               Backend unit and API tests
backend/scripts/             Retrieval/classifier utilities and local Qwen evaluation scripts
backend/evaluation/           Retrieval cases and saved model-evaluation reports
frontend/src/                React application, API client, speech hooks, formatting, and tests
data/raw/                    Source legal documents and optional provenance metadata
data/processed/              Generated persisted chunks and corpus-sync manifest
data/vector_store/            Generated local JSON retrieval index
data/domain_classifier/       Versioned classifier seed splits and local model files
data/conversations.sqlite3    Local chat history (created at runtime, not committed)
docs/                        Architecture, safety, tests, classifier evaluation, this guide
```

For a high-level request flow, see [architecture](architecture.md). For permitted use and safeguards, see [scope and safety](scope-and-safety.md).

## 10. Troubleshooting

**The answer says it does not have enough authoritative evidence.**
Check that the topic is covered by current files in `data/raw`, run `python -m app.retrieval.sync` from `backend`, and inspect retrieval with `python scripts\evaluate_retrieval.py`. Do not lower the threshold just to force unrelated chunks into the prompt.

**The backend says the provider is unavailable or rate limited.**
Check `GROQ_API_KEY`, `LLM_PROVIDER=groq`, the configured model name, network access, and the Groq account's current limits. Ollama being installed does not make it the production provider.

**Many requests ask for clarification.**
Check for `data/domain_classifier/model/classifier.joblib`. If it is missing, train the classifier using the command above. Also confirm that the sentence-transformer model is cached locally. An unavailable classifier intentionally routes requests to the ambiguous path.

**The frontend cannot reach the API.**
Confirm that Uvicorn is running on the configured API address and that the frontend address is listed in `CORS_ORIGINS`. If the API address differs from localhost port 8000, set `VITE_API_BASE_URL` and restart Vite.

**Microphone input is unavailable.**
Use typed chat. SpeechRecognition support differs by browser; the backend does not require a microphone.

**A Qwen benchmark cannot connect.**
Start Ollama, verify that the selected model is installed locally, and use the script's `--url` option if the Ollama API is not at its default local address. This affects the evaluation script only, not application answers.
