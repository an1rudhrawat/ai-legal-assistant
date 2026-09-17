from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    score: int
    reason: str


class LegalDomainGuard:
    """Identify plausible Indian legal questions without making legal findings.

    A first responder must accept a concrete account of a possible dispute or
    harmful act even when the person does not know its legal name.  This guard
    is intentionally a relevance screen, not an offence classifier.
    """

    LEGAL_PHRASES = {
        "first information report": 5, "anticipatory bail": 5, "writ petition": 5,
        "police complaint": 4, "legal notice": 4, "court summons": 4,
        "taken my documents": 4, "threatening me online": 5, "received a summons": 5,
        "immediate danger": 5, "ongoing violence": 5, "threat to my life": 5,
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
        "threat": 3, "threatening": 3, "detained": 4, "notice": 3,
        "documents": 2, "passport": 3, "blackmail": 4,
        "india": 1, "indian": 1,
    }
    LEGAL_INTENTS = (
        r"\bwhat (is|does)\b.{0,40}\b(mean|means)\b",
        r"\bhow (do|can|does|to)\b.{0,60}\b(file|apply|appeal|complain|seek)\b",
        r"\b(can|may|do) i\b.{0,60}\b(right|file|claim|sue|complain)\b",
        r"\bprocedure\b", r"\bpenalty\b", r"\bpunishment\b",
    )
    # These patterns identify a person's interaction with a legal institution.
    # An actor name by itself is not enough: the pattern requires a procedural
    # event so ordinary mentions of (for example) a police officer do not pass.
    INSTITUTIONAL_SITUATION_PATTERNS = (
        r"\bpolice\b.{0,80}\b(stop(?:ped)?|ask(?:ed|ing)?|refus(?:e|ed|ing)|register|sign|search(?:ed|ing)?|detain(?:ed|ing)?|seiz(?:e|ed|ing)|pay|receipt)\b",
        r"\b(stop(?:ped)?|ask(?:ed|ing)?|refus(?:e|ed|ing)|register|sign|search(?:ed|ing)?|detain(?:ed|ing)?|seiz(?:e|ed|ing))\b.{0,80}\bpolice\b",
        r"\b(?:court|judge|magistrate)\b.{0,80}\b(notice|letter|appear|hearing|summons|warrant)\b",
        r"\b(?:notice|letter|summons)\b.{0,80}\b(?:court|judge|magistrate)\b",
        r"\blandlord\b.{0,80}\b(change(?:d)? the locks|lock(?:ed)? out|evict(?:ed|ion)?|remove(?:d)? me)\b",
        r"\bemployer\b.{0,80}\b(?:has not|hasn't|have not|haven't|won't|refus(?:es|ed) to)\b.{0,40}\b(?:pay|paid|salary|wages?)\b",
    )
    # These are descriptions of a concrete event, not a list of legal labels.
    # They only establish that the person may need legal information.
    CONCRETE_SITUATION_PATTERNS = (
        r"\b(?:i|someone|(?:my )?(?:neighbou?r|friend|partner|employer|landlord)|he|she|they)\b.{0,40}\bbroke into\b.{0,60}\b(?:house|home|flat|property)\b",
        r"\b(?:someone|(?:my )?neighbou?r|he|she|they)\b.{0,30}\b(?:entered|came into)\b.{0,40}\b(?:my )?(?:house|home|flat|property)\b.{0,50}\bwithout permission\b",
        r"\b(?:someone|(?:my )?(?:neighbou?r|friend)|he|she|they)\b.{0,30}\b(?:took|stole|kept)\b.{0,30}\b(?:my|our)\b",
        r"\b(?:won't|will not|refuses? to|refused to)\s+return\b",
        r"\b(?:someone|(?:my )?(?:neighbou?r|friend)|he|she|they|i)\b.{0,30}\b(?:hit|assaulted|hurt|damaged|destroyed)\b.{0,50}\b(?:me|my|our|someone|property)\b",
        r"\b(?:someone|(?:my )?(?:neighbou?r|friend)|he|she|they)\b.{0,30}\btook\b.{0,30}\b(?:my|our)\s+(?:documents?|passport|id(?:entity)?|certificate)\b",
        r"\b(?:someone|(?:my )?(?:neighbou?r|friend)|he|she|they)\b.{0,30}\b(?:threaten(?:ed|ing)?|blackmail(?:ed|ing)?)\b",
        r"\b(?:i|someone)\b.{0,30}\b(?:damaged|broke|destroyed)\b.{0,50}\bproperty\b",
    )
    RIGHTS_OR_PROCEDURE_PATTERNS = (
        r"\bwhat (?:can|should) i do\b", r"\bwhat are my rights\b",
        r"\bcan (?:they|i) do this\b", r"\bam i allowed to\b",
        r"\bwhat happens next\b", r"\bwhere can i complain\b",
        r"\bwho should i contact\b", r"\bwhat should i do now\b",
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

        explicit_score = sum(weight for phrase, weight in self.LEGAL_PHRASES.items() if phrase in text)
        explicit_score += sum(weight for term, weight in self.LEGAL_TERMS.items() if self._contains_term(text, term))
        has_intent = any(re.search(pattern, text) for pattern in self.LEGAL_INTENTS)
        institutional = any(re.search(pattern, text) for pattern in self.INSTITUTIONAL_SITUATION_PATTERNS)
        concrete = any(re.search(pattern, text) for pattern in self.CONCRETE_SITUATION_PATTERNS)
        rights_or_procedure = any(re.search(pattern, text) for pattern in self.RIGHTS_OR_PROCEDURE_PATTERNS)

        if explicit_score >= 3:
            return GuardDecision(True, explicit_score, "explicit legal terminology or concept")
        if institutional or concrete:
            score = 3 + int(rights_or_procedure)
            return GuardDecision(True, score, "plausible legal situation")
        return GuardDecision(False, explicit_score, "insufficient legal context")
