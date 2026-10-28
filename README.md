# Aadhikar AI Legal Assistant

Aadhikar is a prototype that helps people ask ordinary-language questions about Indian legal information. It combines a React chat interface, a FastAPI backend, a local legal-document search index, and a Groq-hosted language model. Browser speech features are optional; typed chat is always available.

It is an information and triage prototype, not a lawyer. It can answer only when its local corpus contains useful material, and it does not establish that a law applies to a user's facts.

## Start the project

Requirements: Python 3.11 or newer and Node.js 20 or newer. The backend needs a Groq API key for generated answers.

1. In PowerShell, prepare the backend:

   ```powershell
   cd backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

2. Edit `backend/.env` and set `GROQ_API_KEY`. The production provider is Groq; Ollama/Qwen scripts are for evaluation only.

3. Build the local index from the documents currently in `data/raw`:

   ```powershell
   python -m app.retrieval.sync
   ```

4. Train the local legal-domain classifier if its model files are not present. The first run may download the embedding model:

   ```powershell
   python scripts\train_domain_classifier.py --allow-download
   ```

5. Start the backend from the `backend` directory:

   ```powershell
   uvicorn app.main:app --reload
   ```

6. In a second terminal, start the frontend:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

Open the local address printed by Vite (usually `http://localhost:5173`). The backend API documentation is at `http://localhost:8000/docs`.

## How it works

```text
Browser chat or speech transcript → FastAPI safety checks → local legal retrieval
→ Groq answer grounded in retrieved text → citation and response checks → browser
```

The app stores conversation turns in local SQLite and uses recent context to interpret follow-up questions. Legal documents, embeddings, and retrieval run locally. The current corpus is intentionally limited; when relevant evidence is missing, the assistant abstains. Read [the project guide](docs/project-guide.md) for setup, configuration, APIs, corpus management, evaluation, and troubleshooting.

## Documentation

- [Project guide](docs/project-guide.md): setup, configuration, API, corpus, testing, evaluation, and troubleshooting.
- [Architecture](docs/architecture.md): request flow and component responsibilities.
- [Scope and safety](docs/scope-and-safety.md): intended use and important limitations.
- [Test plan](docs/test-plan.md): automated coverage and what it does not verify.
- [Domain-classifier evaluation](docs/domain-classifier-evaluation.md): local classifier design and recorded evaluation limits.
- [Corpus source instructions](data/raw/README.md): accepted source files and provenance metadata.

## Run the automated checks

```powershell
cd backend
python -m pytest

cd ..\frontend
npm test
npm run build
```

The automated tests use mocked providers. They do not call Groq or Ollama and do not establish legal correctness.
