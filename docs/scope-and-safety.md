# Scope and safety

The assistant is limited to **Indian legal terminology and general legal concepts**. It does not claim to understand all Indian law and does not provide personalised legal advice.

The server evaluates every input before an LLM call. The deterministic guard combines curated Indian legal phrases, legal terms, question-intent patterns, explicit non-legal patterns, and a conservative score threshold. Non-legal, ambiguous, mixed-topic, and jailbreak/role-play requests receive a fixed refusal and do not consume an LLM request.

The LLM system prompt reinforces this boundary and requires a confident refusal when it cannot verify statute sections, amendments, or case citations. The application independently appends the disclaimer to all returned text; it never trusts the model alone to preserve it.

After generation, a pure rule-based validator blocks responses containing code/programming syntax, medical or financial advice, foreign jurisdiction law presented as controlling, or no legal-domain vocabulary. The rejected model text is never sent to the browser. Validator events are logged without storing message contents or other PII.

For a specific matter, users must consult a licensed advocate. The displayed and response-level disclaimer is: “This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.”
