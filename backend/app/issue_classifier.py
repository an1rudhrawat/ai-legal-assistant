from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IssueAssessment:
    categories: tuple[str, ...]
    urgent: bool
    emergency: bool


class LegalIssueClassifier:
    """Lightweight routing hints; it does not make legal findings."""
    CATEGORY_PATTERNS = {
        "police_fir": r"\b(fir|police|complaint|station|arrest(?:ed)?|detain(?:ed)?)\b",
        "cybercrime": r"(?=.*\b(online|cyber|internet|social media|account|digital)\b)(?=.*\b(threat(?:ening)?|harass(?:ment)?|fraud|blackmail)\b)|\b(cybercrime|cyber crime)\b",
        "landlord_tenant": r"\b(landlord|tenant|rent|evict|eviction)\b",
        "court_process": r"\b(summons|court notice|hearing|warrant)\b",
        "legal_notice": r"\blegal notice\b",
        "documents_identity": r"\b(document|certificate|passport|id|identity)\b",
        "harassment_threats": r"\b(threat|threatening|harass|violence|abuse)\b",
    }
    EMERGENCY_PATTERN = r"\b(immediate danger|ongoing attack|being attacked|attack in progress|threat(?:en)?(?:ing)? (?:to )?(?:kill|life)|ongoing violence|need emergency|child in danger)\b"
    URGENT_PATTERN = r"\b(arrested|detained|evict(?:ed|ion)? tomorrow|destroy(?:ing|ed) evidence|deadline today|summons)\b"

    def assess(self, message: str) -> IssueAssessment:
        text = message.lower()
        categories = tuple(name for name, pattern in self.CATEGORY_PATTERNS.items() if re.search(pattern, text))
        emergency = bool(re.search(self.EMERGENCY_PATTERN, text))
        return IssueAssessment(categories, emergency or bool(re.search(self.URGENT_PATTERN, text)), emergency)
