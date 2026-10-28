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
    requested_objective: str | None = None
    current_situation: str | None = None
    factual_entities: tuple[str, ...] = ()
    jurisdiction: str = "India"
    urgent: bool = False
    needs_retrieval: bool = True

    def to_prompt_context(self) -> dict[str, object]:
        """Return explicit retrieval hints without turning reports into findings."""
        return {
            "requested_objective": self.requested_objective or "unknown",
            "reported_events": list(self.events) or ["unknown"],
            "reported_actors": list(self.actors) or ["unknown"],
            "current_reported_situation": self.current_situation or "unknown",
            "factual_entities": list(self.factual_entities) or ["unknown"],
            "immediate_risk_indicator": "reported immediate-risk indicator" if self.urgent else "not identified from the message",
            "tentative_retrieval_concepts": list(self.potential_legal_concepts) or ["unknown"],
            "jurisdiction_hint": self.jurisdiction,
        }


class SituationUnderstander:
    """Extract retrieval hints, never a legal conclusion or offence finding."""

    EVENT_RULES = (
        (r"\b(?:damage(?:d|s|ing)?|vandal(?:i[sz](?:e|ed|ing|ation))?|destroy(?:ed|s|ing)?)\b.{0,80}\b(?:car|vehicle|property|wall|door|window|object)\b", "reported property damage", ("mischief", "property damage")),
        (r"\b(?:police|officer)\b.{0,80}\b(?:money|payment|bribe|unofficial)\b", "reported request or issue involving payment to police", ("police procedure", "official receipt", "complaint procedure")),
        (r"\b(?:court )?summons\b", "reported court summons", ("court procedure", "summons procedure")),
        (r"\b(?:attacked|hit|hurt|injured|fight|jaw)\b", "physical altercation", ("bodily injury", "private defence")),
        (r"\b(?:broke into|entered .*without permission|trespass)\b", "entry onto property", ("criminal trespass", "house trespass", "possession")),
        (r"\b(?:took|stole|won't return|refuses to return).{0,60}\b(?:phone|property|documents?|passport|id)\b", "property or document retention", ("property possession", "dishonest taking")),
        (r"\b(?:threat(?:en(?:ed|ing)?|s)?|blackmail|harass)\b", "threat or harassment", ("criminal intimidation", "harassment")),
        (r"\b(?:landlord|tenant|locks|evict)\b", "housing dispute", ("tenancy", "eviction", "possession")),
        (r"\b(?:employer|salary|wage|paid me)\b", "employment payment dispute", ("wages", "employment dispute")),
        (r"\b(?:police|challan|receipt|detain|seize|search)\b", "police or enforcement interaction", ("police procedure", "complaint procedure")),
        (r"\b(?:fraud|scam|online|whatsapp|account)\b", "digital or financial dispute", ("cybercrime", "fraud")),
        (r"\b(?:accident|vehicle|driving|traffic)\b", "road or traffic incident", ("traffic procedure", "accident liability")),
    )
    URGENT = re.compile(r"\b(immediate danger|ongoing violence|ongoing attack|being attacked|attack in progress|threat(?:ening)? to (?:kill|my life))\b", re.I)
    ENTITY_RULES = (
        (r"\b(?:money|payment|bribe|receipt)\b", "payment or receipt"),
        (r"\b(?:court )?summons\b", "court summons"),
        (r"\b(?:phone|documents?|passport|(?:identity )?id|certificate)\b", "document or personal property"),
        (r"\b(?:house|home|flat|property|locks?)\b", "property or residence"),
        (r"\b(?:whatsapp|online|account|social media)\b", "online account or communication"),
    )

    def understand(self, text: str) -> Situation:
        normalised = " ".join(text.lower().split())
        events: list[str] = []
        concepts: list[str] = []
        for pattern, event, rule_concepts in self.EVENT_RULES:
            if re.search(pattern, normalised):
                events.append(event)
                concepts.extend(rule_concepts)
        actors = tuple(actor for actor in ("user", "neighbour", "police", "officer", "court", "landlord", "employer", "other person") if actor in normalised or (actor == "user" and re.search(r"\b(i|my|me)\b", normalised)))
        entities = tuple(label for pattern, label in self.ENTITY_RULES if re.search(pattern, normalised))
        intent = "liability_assessment" if re.search(r"\b(am i liable|can i|what can i do|what should i do)\b", normalised) else "general_information"
        requested_objective = self._requested_objective(normalised)
        current_situation = f"reported {events[0]}" if events else None
        return Situation(
            intent,
            tuple(dict.fromkeys(events)),
            actors,
            tuple(dict.fromkeys(concepts)),
            requested_objective=requested_objective,
            current_situation=current_situation,
            factual_entities=tuple(dict.fromkeys(entities)),
            urgent=bool(self.URGENT.search(normalised)),
        )

    @staticmethod
    def _requested_objective(text: str) -> str | None:
        if re.search(r"\b(?:what should|what can|how do|how can|steps?|help me)\b", text):
            return "practical next steps"
        if re.search(r"\b(?:rights?|allowed|can they)\b", text):
            return "rights or protections information"
        if re.search(r"\b(?:summons|notice|procedure|hearing|court)\b", text):
            return "procedure information"
        if re.search(r"\b(?:what is|explain|meaning of)\b", text):
            return "legal information"
        return None
