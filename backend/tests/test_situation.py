from app.retrieval.query_expansion import expand_query
from app.situation import SituationUnderstander


def test_physical_altercation_extracts_retrieval_hints_without_finding_an_offence():
    situation = SituationUnderstander().understand("I broke my neighbour's jaw because he attacked me. Am I liable?")
    assert "physical altercation" in situation.events
    assert "private defence" in situation.potential_legal_concepts
    assert "bodily injury" in expand_query("I broke my neighbour's jaw", situation)


def test_ambiguous_question_does_not_invent_events():
    situation = SituationUnderstander().understand("What should I do?")
    assert situation.events == ()


def test_past_attack_is_not_marked_urgent_without_immediate_context():
    assert not SituationUnderstander().understand("My neighbour attacked me yesterday.").urgent
