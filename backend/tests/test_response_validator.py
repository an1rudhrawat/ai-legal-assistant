from app.domain_guard import ResponseScopeValidator


def test_accepts_legal_output():
    assert ResponseScopeValidator().validate("An FIR is a record made by police about a cognizable offence.")[0]


def test_accepts_ordinary_explanatory_legal_prose():
    response = "Let me explain: an FIR is information recorded by police about a cognizable offence."
    assert ResponseScopeValidator().validate(response)[0]


def test_rejects_code_output():
    assert not ResponseScopeValidator().validate("For legal help run ```python\nprint('hello')\n```")[0]


def test_rejects_unfenced_code_output():
    assert not ResponseScopeValidator().validate("const complaint = 'FIR';")[0]


def test_rejects_foreign_law_as_controlling():
    assert not ResponseScopeValidator().validate("Under US law, this is decided by the California Penal Code.")[0]


def test_rejects_medical_and_off_scope_output():
    assert not ResponseScopeValidator().validate("Take 20 mg of this medicine as medical advice.")[0]


def test_rejects_output_without_legal_vocabulary():
    assert not ResponseScopeValidator().validate("Here is a poem about the rain.")[0]
