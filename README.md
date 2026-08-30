# AI Legal Assistant — Phase 1

A final-year project prototype for Indian citizens: a voice-enabled assistant for **Indian legal terminology and general legal concepts**. It is not a substitute for a lawyer and does not claim to understand all Indian law.

## Architecture

`Browser microphone → SpeechRecognition → server legal guard → LLM → response scope validator → SpeechSynthesis`

The React frontend uses browser speech where available and always provides typed input. The FastAPI backend keeps the LLM key private, runs deterministic request and response safety checks, enforces a disclaimer, and exposes `POST /api/chat` plus `GET /health`. There is no RAG, database, authentication, or audio storage in Phase 1. See [architecture](docs/architecture.md) and [scope and safety](docs/scope-and-safety.md).

## Setup

Requirements: Python 3.11+ and Node.js 20+ (for the frontend). Browser speech recognition is not universal; unsupported browsers can use the typed input fallback.

Copy `backend/.env.example` to `backend/.env` and set a free-tier provider key. The default adapter expects `GROQ_API_KEY`. Never add this key to the frontend or commit `.env` files.

## Run the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in `frontend/.env` only if the backend is not at `http://localhost:8000`.

## Run tests

```powershell
cd backend
pytest

cd ..\frontend
npm test
```

Tests mock the LLM provider and do not use API quota. The suite includes legal-only guard cases, adversarial attempts, response validation, rate-limit handling, citation hedging, typed input fallback, refusal rendering, and the disclaimer banner. See [test plan](docs/test-plan.md).
