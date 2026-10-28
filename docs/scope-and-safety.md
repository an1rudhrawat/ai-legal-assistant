# Scope and safety

## Intended use

Aadhikar is a prototype for general Indian legal information and first-responder triage. It can explain material found in its configured corpus and help a user identify what information may be relevant. It is not a lawyer, legal representative, court, or emergency service. It does not provide personalised legal representation or establish a person's legal rights, liability, guilt, or eligibility for a defence.

## Evidence and uncertainty

- The configured legal corpus is the source of legal facts, section references, procedures, penalties, deadlines, and citations used in generated answers.
- A question outside the corpus can receive an insufficient-evidence response even when an answer exists in law. Retrieval failure and corpus coverage are limitations, not proof that no law applies.
- User statements, prior assistant replies, and generated continuity summaries are context, not legal evidence. Follow-up queries may use prior text to find current source chunks, but new legal claims still need current retrieved evidence.
- The assistant should preserve uncertainty, ask for a critical missing fact, or abstain when evidence is insufficient. It should not fill gaps from model memory.
- A retrieved citation or a valid source ID does not establish that a claim is legally correct or that the provision applies to the user's facts.

## Safeguards in the current implementation

1. A deterministic request gate blocks obvious off-topic or instruction-override content.
2. A local semantic classifier routes requests as legal, non-legal, or ambiguous. It does not decide whether an event is an offence. If its model is unavailable, the route becomes ambiguous.
3. Urgency and situation components provide routing hints. A narrow immediate-danger match returns safety-first guidance before retrieval or LLM generation; it does not contact emergency services or provide a universal emergency number.
4. The retriever filters chunks below a configurable relevance cutoff. No accepted chunk means a fixed insufficient-evidence response without an LLM call.
5. Prompts require answers to stay grounded in supplied evidence and to cite allowed source IDs.
6. Deterministic citation and response-scope checks reject unknown IDs and selected unsupported patterns. They are safeguards, not complete legal or semantic verification.
7. The API appends the legal-information disclaimer to returned text. The frontend hides a duplicate disclaimer in the message body while showing a persistent disclaimer near the composer.

## Privacy and security limits

- Conversations are stored in local SQLite. The backend sends the current question, bounded recent history, any generated continuity summary, and retrieved legal text to Groq for answer generation. The same configured provider may be called to summarize old user turns.
- The app has no user accounts, authentication, or per-user conversation authorization. A conversation UUID is not proof of ownership. Do not expose the API to the public internet or use it for sensitive production conversations without adding suitable access controls.
- The session request counter is held in backend process memory and resets when the process restarts. It is not a billing or account-level limit.
- Browser speech recognition is browser-controlled and may rely on browser-specific services. The app sends recognized text to the backend and does not store an audio recording.
- API secrets belong only in `backend/.env`; never place Groq credentials in frontend configuration or commit local `.env` files.

## Known limits

- Corpus coverage is determined by the documents installed in `data/raw`; missing central or state law cannot be recovered by lowering a relevance threshold.
- Citation and output validators use deterministic checks. They cannot verify every legal proposition, detect every hallucination, check the live status of every amendment, or determine how a court would apply law to disputed facts.
- Domain-classifier seed data and evaluation samples are small. They are not representative of all Indian languages, regions, legal topics, or user phrasing.
- Tests use fake providers and fixture documents. They do not validate live Groq behavior, end-to-end retrieval over every production source, legal correctness, or browser speech behavior.
- The Qwen/Ollama benchmarks are experiments. They do not make Qwen the production model and their pass rates are not a legal-reliability claim.

For a specific matter, users should consult a licensed advocate or the appropriate official authority. The response-level disclaimer is: “This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.”
