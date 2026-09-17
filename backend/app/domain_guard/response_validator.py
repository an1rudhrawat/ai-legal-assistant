from __future__ import annotations

import re

from .guard import LegalDomainGuard


class ResponseScopeValidator:
    """Rule-based output drift detection. It is intentionally not an accuracy verifier."""

    DISALLOWED = (
        r"```",
        # Match recognisable code syntax, not ordinary prose such as
        # "Let me explain" or "the function of a court".
        r"\b(?:import\s+[A-Za-z_]\w*|from\s+\S+\s+import\s+|def\s+\w+\s*\(|function\s+\w+\s*\(|(?:const|let|var)\s+\w+\s*=|select\s+.+\s+from|console\.log\s*\()",
        r"\b(under (?:us|u\.s\.|uk|united kingdom|california|texas|australian) law|california penal code|us constitution)\b",
        r"\b(diagnos[ei]|prescrib[ei]|dosage|take \d+ ?mg|medical advice)\b",
        r"\b(buy|sell|invest in|stock tip|financial advice|portfolio allocation)\b",
    )

    def __init__(self, guard: LegalDomainGuard | None = None) -> None:
        self.guard = guard or LegalDomainGuard()

    def validate(self, text: str) -> tuple[bool, str]:
        normalised = text.lower()
        if any(re.search(pattern, normalised, re.IGNORECASE | re.DOTALL) for pattern in self.DISALLOWED):
            return False, "disallowed output pattern"
        legal_signal = any(self.guard._contains_term(normalised, term) for term in self.guard.LEGAL_TERMS)
        legal_signal = legal_signal or any(phrase in normalised for phrase in self.guard.LEGAL_PHRASES)
        if not legal_signal:
            return False, "no legal-domain vocabulary"
        return True, "within deterministic scope"
