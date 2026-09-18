"""Deterministic, non-adjudicative extraction for retrieval query expansion."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Situation:
    intent: str
    events: tuple[str, ...]
    actors: tuple[str, ...]
    potential_legal_concepts: tuple[str, ...]
    jurisdiction: str = "India"
    urgent: bool = False
    needs_retrieval: bool = True


class SituationUnderstander:
    """Extract retrieval hints, never a legal conclusion or offence finding."""

    EVENT_RULES = (
        (r"\b(?:attacked|hit|hurt|injured|fight|jaw)\b", "physical altercation", ("bodily injury", "private defence")),
        (r"\b(?:broke into|entered .*without permission|trespass)\b", "entry onto property", ("criminal trespass", "house trespass", "possession")),
        (r"\b(?:took|stole|won't return|refuses to return).{0,60}\b(?:phone|property|documents?|passport|id)\b", "property or document retention", ("property possession", "dishonest taking")),
        (r"\b(?:threat|blackmail|harass)\b", "threat or harassment", ("criminal intimidation", "harassment")),
        (r"\b(?:landlord|tenant|locks|evict)\b", "housing dispute", ("tenancy", "eviction", "possession")),
        (r"\b(?:employer|salary|wage|paid me)\b", "employment payment dispute", ("wages", "employment dispute")),
        (r"\b(?:police|challan|receipt|detain|seize|search)\b", "police or enforcement interaction", ("police procedure", "complaint procedure")),
        (r"\b(?:fraud|scam|online|whatsapp|account)\b", "digital or financial dispute", ("cybercrime", "fraud")),
        (r"\b(?:accident|vehicle|driving|traffic)\b", "road or traffic incident", ("traffic procedure", "accident liability")),
    )
    URGENT = re.compile(r"\b(immediate danger|ongoing violence|ongoing attack|being attacked|attack in progress|threat(?:ening)? to (?:kill|my life))\b", re.I)

    def understand(self, text: str) -> Situation:
        normalised = " ".join(text.lower().split())
        events: list[str] = []
        concepts: list[str] = []
        for pattern, event, rule_concepts in self.EVENT_RULES:
            if re.search(pattern, normalised):
                events.append(event)
                concepts.extend(rule_concepts)
        actors = tuple(actor for actor in ("user", "neighbour", "police", "landlord", "employer", "other person") if actor in normalised or (actor == "user" and re.search(r"\b(i|my|me)\b", normalised)))
        intent = "liability_assessment" if re.search(r"\b(am i liable|can i|what can i do|what should i do)\b", normalised) else "general_information"
        return Situation(intent, tuple(dict.fromkeys(events)), actors, tuple(dict.fromkeys(concepts)), urgent=bool(self.URGENT.search(normalised)))
