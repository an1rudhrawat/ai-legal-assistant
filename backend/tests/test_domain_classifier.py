from app.domain_guard.classifier import LocalSemanticDomainClassifier


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        return texts


class FakeModel:
    classes_ = ["legal", "non_legal"]

    def __init__(self, legal_probability):
        self.legal_probability = legal_probability

    def predict_proba(self, vectors):
        return [[self.legal_probability, 1 - self.legal_probability] for _ in vectors]


def classifier_with_probability(probability):
    classifier = LocalSemanticDomainClassifier.__new__(LocalSemanticDomainClassifier)
    classifier.legal_threshold = 0.75
    classifier.non_legal_threshold = 0.25
    classifier._encoder = FakeEncoder()
    classifier._classifier = FakeModel(probability)
    classifier._load = lambda: True
    return classifier


def test_high_legal_probability_is_legal():
    result = classifier_with_probability(0.9).classify("ordinary-language situation")
    assert result.label == "legal"
    assert result.confidence == 0.9


def test_low_legal_probability_is_non_legal():
    assert classifier_with_probability(0.1).classify("recipe").label == "non_legal"


def test_midrange_probability_is_ambiguous():
    assert classifier_with_probability(0.5).classify("What should I do?").label == "ambiguous"


def test_regression_examples_are_classified_by_the_semantic_classifier_contract():
    legal = classifier_with_probability(0.9)
    non_legal = classifier_with_probability(0.1)
    for text in (
        "I broke my neighbour's jaw because he attacked me. Am I liable?",
        "I broke into my neighbour's house to take my phone.",
        "My landlord changed the locks.",
        "My employer has not paid me.",
        "The police took my phone.",
        "Someone is threatening me.",
        "What does section 35 of BNS say?",
    ):
        assert legal.classify(text).label == "legal"
    for text in ("Explain Python decorators.", "Who won the cricket match?", "How do I cook pasta?"):
        assert non_legal.classify(text).label == "non_legal"
