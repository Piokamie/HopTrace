import pytest

from hoppath.config import ExtractorConfig
from hoppath.mentions import Mention, MentionExtractor

extractor = MentionExtractor()


def entities(text: str) -> set[str]:
    return {m.entity for m in extractor.extract(text)}


def surfaces(text: str) -> set[str]:
    return {m.surface for m in extractor.extract(text)}


def test_multiword_capitalized_span() -> None:
    assert "anna kowalska" in entities("Yesterday Anna Kowalska joined the platform team.")


def test_sentence_initial_single_word_needs_recurrence() -> None:
    # "Yesterday" opens the sentence and never recurs capitalized: dropped.
    assert "yesterday" not in entities("Yesterday the meeting happened.")
    # "Anna" opens a sentence but recurs non-initially: kept.
    found = entities("Anna left early. We asked Anna about it.")
    assert "anna" in found


def test_function_words_never_single_mentions() -> None:
    assert "the" not in entities("The meeting. The room. The plan.")


def test_quoted_terms() -> None:
    assert "zephyr" in entities('The project is called "Zephyr" internally.')
    assert "beam width" in entities("Set the `beam width` parameter.")


def test_corpus_caps_rescue_sentence_initial() -> None:
    # Alone, sentence-initial "Kowalski" is untrusted and dropped ...
    assert "kowalski" not in entities("Kowalski sits in Room 204.")
    # ... but corpus-wide evidence of non-initial capitalization keeps it.
    mentions = extractor.extract("Kowalski sits in Room 204.", corpus_caps={"kowalski"})
    assert "kowalski" in {m.entity for m in mentions}


def test_code_identifiers() -> None:
    text = "Call hop_expand or RetrievalConfig or hoppath.store.open here."
    found = entities(text)
    assert "hop_expand" in found
    assert "retrievalconfig" in found
    assert "hoppath store open" in found


def test_numbers_with_units() -> None:
    found = surfaces("The server has 12 GB memory; the office is Room 204; code 4B.")
    assert "12 GB" in found
    assert "Room 204" in found
    assert "4B" in found


def test_plain_number_words_not_extracted() -> None:
    # "2 of" and similar must not become entities via the spaced-unit rule
    assert not any("of" in s for s in surfaces("We saw 2 of them and 3 to go."))


def test_overlap_prefers_longer_span() -> None:
    text = "Yesterday Anna Kowalska spoke."
    mentions = extractor.extract(text)
    anna = [m for m in mentions if "anna" in m.entity]
    assert anna == [Mention(entity="anna kowalska", surface="Anna Kowalska", start=10, end=23)]


def test_untrusted_sentence_opener_dropped_from_run() -> None:
    # "Yesterday" heads the run but is not a trusted capitalized word
    assert "yesterday anna kowalska" not in entities("Yesterday Anna Kowalska spoke.")


def test_spans_point_into_text() -> None:
    text = 'Kowalski sits in Room 204 near the "Atlas" board.'
    for m in extractor.extract(text):
        assert text[m.start : m.end] == m.surface


def test_deterministic() -> None:
    text = 'Anna reports to Kowalski. See `beam_width` and "Atlas" in Room 204.'
    assert extractor.extract(text) == extractor.extract(text)


def test_ner_flag_without_spacy_model() -> None:
    spacy = pytest.importorskip("spacy")
    cfg = ExtractorConfig(ner=True, spacy_model="nonexistent_model_xyz")
    with pytest.raises((OSError, IOError)):
        MentionExtractor(cfg)
    del spacy
