from hoppath.tokenize import tokenize


def test_lowercases_and_splits() -> None:
    assert tokenize("Anna reports to Kowalski.") == ["anna", "reports", "to", "kowalski"]


def test_keeps_identifiers_and_numbers() -> None:
    assert tokenize("room_204 costs 12 GB") == ["room_204", "costs", "12", "gb"]


def test_unicode_words() -> None:
    assert tokenize("Kowalskiego biuro") == ["kowalskiego", "biuro"]


def test_empty() -> None:
    assert tokenize("") == []
    assert tokenize("  ...  ") == []
