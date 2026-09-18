"""Small hard gate for content that is obviously outside the product's scope."""
from __future__ import annotations

import re


class ObviousOutOfScopeGate:
    """Reject only explicit bypasses and unmistakably unrelated requests.

    It deliberately does not try to recognise legal subject matter. That is the
    local semantic classifier's job.
    """

    PATTERNS = (
        r"\b(ignore (?:all |previous |your )?(?:rules|instructions)|jailbreak|unrestricted assistant)\b",
        r"\b(write|debug|fix|generate)\b.{0,40}\b(code|python|javascript|sql|html|css|program)\b",
        r"\b(recipe|cook|ingredient)\b", r"\b(cricket|football|movie|song|celebrity)\b",
        r"\b(weather|temperature|forecast)\b", r"\b(quantum mechanics|calculus|algebra)\b",
    )

    def rejects(self, text: str) -> bool:
        normalised = re.sub(r"\s+", " ", text.lower()).strip()
        return any(re.search(pattern, normalised) for pattern in self.PATTERNS)
