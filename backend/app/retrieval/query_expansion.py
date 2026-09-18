"""Use non-adjudicative situation hints to make statute retrieval more robust."""
from __future__ import annotations

from app.situation import Situation


def expand_query(user_text: str, situation: Situation) -> str:
    """Return original language plus tentative retrieval concepts.

    The concepts guide search only; they are not shown as conclusions and do
    not assert that a described event satisfies an offence.
    """
    concepts = " ".join(situation.potential_legal_concepts)
    return " ".join(part for part in (user_text, concepts) if part).strip()
