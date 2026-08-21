from hoppath.normalize import TrivialNormalizer

norm = TrivialNormalizer()


def test_casefold_and_punctuation() -> None:
    assert norm.canonical("Kowalski") == "kowalski"
    assert norm.canonical("U.S.A") == "u s a"


def test_possessive() -> None:
    assert norm.canonical("Anna's") == "anna"
    assert norm.canonical("Anna’s") == "anna"


def test_naive_plural() -> None:
    assert norm.canonical("reports") == "report"
    # documented exceptions: short words and ss/us/is endings untouched
    assert norm.canonical("gas") == "gas"
    assert norm.canonical("boss") == "boss"
    assert norm.canonical("status") == "status"
    assert norm.canonical("analysis") == "analysis"
    # documented over-strip: looks plural, isn't
    assert norm.canonical("kansas") == "kansa"


def test_whitespace_collapse() -> None:
    assert norm.canonical("  New   York  ") == "new york"


def test_idempotent() -> None:
    for surface in ("Anna's", "New  York", "reports", "U.S.A"):
        once = norm.canonical(surface)
        assert norm.canonical(once) == once


def test_empty_and_symbol_only() -> None:
    assert norm.canonical("") == ""
    assert norm.canonical("!!!") == ""
