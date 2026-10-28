from app.retrieval.query_expansion import expand_query
from app.situation import SituationUnderstander


def test_physical_altercation_extracts_retrieval_hints_without_finding_an_offence():
    situation = SituationUnderstander().understand("I broke my neighbour's jaw because he attacked me. Am I liable?")
    assert "physical altercation" in situation.events
    assert "private defence" in situation.potential_legal_concepts
    assert "bodily injury" in expand_query("I broke my neighbour's jaw", situation)


def test_property_damage_expands_to_the_bns_topic_without_finding_an_offence():
    question = "My neighbour deliberately damaged my car. What legal information is available?"
    situation = SituationUnderstander().understand(question)

    assert "reported property damage" in situation.events
    expanded = expand_query(question, situation)
    assert "mischief" in expanded
    assert "property damage" in expanded


def test_ambiguous_question_does_not_invent_events():
    situation = SituationUnderstander().understand("What should I do?")
    assert situation.events == ()


def test_past_attack_is_not_marked_urgent_without_immediate_context():
    assert not SituationUnderstander().understand("My neighbour attacked me yesterday.").urgent


def test_police_payment_report_extracts_neutral_facts_and_retrieval_hints():
    situation = SituationUnderstander().understand("A police officer is demanding unofficial payment. What should I do?")

    assert "police" in situation.actors
    assert "officer" in situation.actors
    assert "reported request or issue involving payment to police" in situation.events
    assert "payment or receipt" in situation.factual_entities
    assert "official receipt" in situation.potential_legal_concepts
    assert situation.requested_objective == "practical next steps"
    assert "bribery" not in situation.potential_legal_concepts
    expanded = expand_query("A police officer is demanding unofficial payment.", situation)
    assert expanded.startswith("A police officer is demanding unofficial payment.")
    assert "official receipt" in expanded


def test_summons_and_threat_reports_remain_non_adjudicative():
    summons = SituationUnderstander().understand("I received a court summons. What should I do?")
    threat = SituationUnderstander().understand("Someone is threatening to kill me.")

    assert "reported court summons" in summons.events
    assert "court summons" in summons.factual_entities
    assert summons.requested_objective == "practical next steps"
    assert "threat or harassment" in threat.events
    assert threat.urgent
    assert threat.to_prompt_context()["immediate_risk_indicator"] == "reported immediate-risk indicator"


def test_missing_information_is_explicitly_unknown_in_prompt_context():
    situation = SituationUnderstander().understand("Please help")

    assert situation.to_prompt_context()["factual_entities"] == ["unknown"]
    assert situation.to_prompt_context()["requested_objective"] == "unknown"
