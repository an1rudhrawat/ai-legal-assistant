# Scope and safety

## First-responder safeguards

The product is a legal information and triage assistant, not a lawyer or legal representative. It supports situation-based questions only when authoritative retrieved material is sufficient. Legal facts, provisions, procedures, deadlines, and citations must come from retrieved context; an LLM citation not matching a retrieved chunk is rejected.

Immediate-danger wording is handled before retrieval with a concise safety-first prompt to contact appropriate local emergency, police, or medical services. The application does not invent emergency numbers, determine a user's legal position from incomplete facts, or call the LLM when available authoritative evidence is insufficient.

The assistant supports Indian legal terminology, general legal concepts, and plausible first-responder situations described in ordinary language. The domain guard decides only whether a situation may need legal information; it does not decide that an offence occurred or determine a user's legal position. It does not claim to understand all Indian law and does not provide personalised legal advice.

The server evaluates every input before an LLM call. A local semantic classifier distinguishes legal, non-legal, and ambiguous situations using a locally stored model; it does not decide that an offence occurred. A small deterministic gate rejects only obvious unrelated or bypass content. Ambiguous requests are asked for facts when retrieval cannot establish relevant authoritative evidence. Explicit non-legal, mixed-topic, and jailbreak/role-play requests receive a fixed refusal and do not consume an LLM request.

The LLM system prompt reinforces this boundary and requires a confident refusal when it cannot verify statute sections, amendments, or case citations. The application independently appends the disclaimer to all returned text; it never trusts the model alone to preserve it.

After generation, a pure rule-based validator blocks responses containing code/programming syntax, medical or financial advice, foreign jurisdiction law presented as controlling, or no legal-domain vocabulary. The rejected model text is never sent to the browser. Validator events are logged without storing message contents or other PII.

For a specific matter, users must consult a licensed advocate. The displayed and response-level disclaimer is: “This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.”
