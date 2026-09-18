import pytest

from app.issue_classifier import LegalIssueClassifier


@pytest.mark.parametrize(("message", "category"), [
    ("The police will not register my FIR", "police_fir"),
    ("I have been arrested", "police_fir"),
    ("I received a legal notice", "legal_notice"),
    ("My landlord will evict me tomorrow", "landlord_tenant"),
    ("Someone is threatening me online", "cybercrime"),
    ("I received a court summons", "court_process"),
])
def test_first_responder_scenarios_are_classified(message, category):
    assert category in LegalIssueClassifier().assess(message).categories


def test_immediate_danger_is_an_emergency():
    assessment = LegalIssueClassifier().assess("There is ongoing violence and I am in immediate danger")
    assert assessment.emergency
    assert assessment.urgent


def test_past_altercation_is_not_automatically_an_emergency():
    assert not LegalIssueClassifier().assess("My neighbour attacked me yesterday and I hit him.").emergency
