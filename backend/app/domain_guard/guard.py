from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    score: int
    reason: str


class LegalDomainGuard:
    """Conservative deterministic classifier for the deliberately narrow Phase 1 scope."""

    LEGAL_PHRASES = {
        "first information report": 5, "anticipatory bail": 5, "writ petition": 5,
        "police complaint": 4, "legal notice": 4, "court summons": 4,
        "consumer complaint": 4, "maintenance claim": 4, "right to information": 4,
        "fundamental rights": 4, "criminal procedure": 4, "civil procedure": 4,
        "property dispute": 4, "domestic violence": 4, "cyber crime": 4,
        "cybercrime": 4, "labour law": 4, "labor law": 4,
    }
    LEGAL_TERMS = {
        "fir": 4, "bail": 4, "arrest": 4, "affidavit": 4, "summons": 4,
        "warrant": 4, "offence": 3, "offense": 3, "petition": 3, "plaint": 3,
        "appeal": 3, "tribunal": 3, "litigation": 3, "advocate": 3, "lawyer": 3,
        "court": 3, "judge": 3, "magistrate": 4, "police": 3, "constitution": 4,
        "ipc": 4, "crpc": 4, "bns": 4, "bnss": 4, "bsa": 4, "act": 2,
        "section": 2, "statute": 3, "legal": 2, "law": 2, "rights": 2,
        "contract": 3, "agreement": 3, "tenant": 3, "landlord": 3, "eviction": 4,
        "divorce": 4, "custody": 3, "maintenance": 4, "inheritance": 3,
        "will": 3, "property": 2, "cheque": 2, "fraud": 3, "theft": 3,
        "harassment": 3, "complaint": 2, "complainant": 3, "accused": 3,
        "evidence": 3, "remand": 4, "charge sheet": 4, "chargesheet": 4,
        "india": 1, "indian": 1,
    }
    LEGAL_INTENTS = (
        r"\bwhat (is|does)\b.{0,40}\b(mean|means)\b",
        r"\bhow (do|can|does|to)\b.{0,60}\b(file|apply|appeal|complain|seek)\b",
        r"\b(can|may|do) i\b.{0,60}\b(right|file|claim|sue|complain)\b",
        r"\bprocedure\b", r"\bpenalty\b", r"\bpunishment\b",
    )
    NON_LEGAL_PATTERNS = (
        r"\b(write|debug|fix|generate)\b.{0,40}\b(code|python|javascript|sql|html|css|program)\b",
        r"\b(recipe|cook|ingredient|calories|diet)\b", r"\b(cricket|football|movie|song|celebrity)\b",
        r"\b(weather|temperature|forecast)\b", r"\b(stock|crypto|bitcoin|investment|portfolio)\b",
        r"\b(diagnose|dosage|prescribe|symptom|medicine)\b", r"\b(joke|poem|story)\b",
        r"\b(ignore (all |previous |your )?(rules|instructions)|jailbreak|unrestricted assistant)\b",
    )

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("-", " ")).strip()

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text))

    def evaluate(self, message: str) -> GuardDecision:
        text = self._normalise(message)
        non_legal = [pattern for pattern in self.NON_LEGAL_PATTERNS if re.search(pattern, text)]
        if non_legal:
            return GuardDecision(False, 0, "explicit non-legal or bypass pattern")

        score = sum(weight for phrase, weight in self.LEGAL_PHRASES.items() if phrase in text)
        score += sum(weight for term, weight in self.LEGAL_TERMS.items() if self._contains_term(text, term))
        has_intent = any(re.search(pattern, text) for pattern in self.LEGAL_INTENTS)
        if score >= 3 and (has_intent or len(text.split()) <= 18):
            return GuardDecision(True, score, "legal terminology or concept")
        return GuardDecision(False, score, "insufficient legal context")
