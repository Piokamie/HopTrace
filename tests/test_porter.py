import pytest

from hoppath.porter import stem
from hoppath.tokenize import analyze, tokenize

# Classic pairs from Porter's paper and the reference implementation's
# vocabulary. If any of these drift, the BM25 baseline gate is at risk.
KNOWN = [
    ("caresses", "caress"),
    ("ponies", "poni"),
    ("ties", "ti"),
    ("caress", "caress"),
    ("cats", "cat"),
    ("feed", "feed"),
    ("agreed", "agre"),
    ("plastered", "plaster"),
    ("bled", "bled"),
    ("motoring", "motor"),
    ("sing", "sing"),
    ("conflated", "conflat"),
    ("troubled", "troubl"),
    ("sized", "size"),
    ("hopping", "hop"),
    ("tanned", "tan"),
    ("falling", "fall"),
    ("hissing", "hiss"),
    ("fizzed", "fizz"),
    ("failing", "fail"),
    ("filing", "file"),
    ("happy", "happi"),
    ("sky", "sky"),
    ("relational", "relat"),
    ("conditional", "condit"),
    ("rational", "ration"),
    ("valenci", "valenc"),
    ("hesitanci", "hesit"),
    ("digitizer", "digit"),
    ("conformabli", "conform"),
    ("radicalli", "radic"),
    ("differentli", "differ"),
    ("vileli", "vile"),
    ("analogousli", "analog"),
    ("vietnamization", "vietnam"),
    ("predication", "predic"),
    ("operator", "oper"),
    ("feudalism", "feudal"),
    ("decisiveness", "decis"),
    ("hopefulness", "hope"),
    ("callousness", "callous"),
    ("formaliti", "formal"),
    ("sensitiviti", "sensit"),
    ("sensibiliti", "sensibl"),
    ("triplicate", "triplic"),
    ("formative", "form"),
    ("formalize", "formal"),
    ("electriciti", "electr"),
    ("electrical", "electr"),
    ("hopeful", "hope"),
    ("goodness", "good"),
    ("revival", "reviv"),
    ("allowance", "allow"),
    ("inference", "infer"),
    ("airliner", "airlin"),
    ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"),
    ("defensible", "defens"),
    ("irritant", "irrit"),
    ("replacement", "replac"),
    ("adjustment", "adjust"),
    ("dependent", "depend"),
    ("adoption", "adopt"),
    ("homologou", "homolog"),
    ("communism", "commun"),
    ("activate", "activ"),
    ("angulariti", "angular"),
    ("homologous", "homolog"),
    ("effective", "effect"),
    ("bowdlerize", "bowdler"),
    ("probate", "probat"),
    ("rate", "rate"),
    ("cease", "ceas"),
    ("controll", "control"),
    ("roll", "roll"),
]


@pytest.mark.parametrize(("word", "expected"), KNOWN)
def test_known_stems(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_short_words_untouched() -> None:
    assert stem("as") == "as"
    assert stem("be") == "be"


def test_analyze_english_removes_stopwords_and_stems() -> None:
    terms = analyze("The runners were running to the stations", "english")
    assert terms == ["runner", "were", "run", "station"]


def test_analyze_simple_is_tokenize() -> None:
    text = "The Runners were running!"
    assert analyze(text, "simple") == tokenize(text)


def test_analyze_unknown_analyzer() -> None:
    with pytest.raises(ValueError, match="unknown analyzer"):
        analyze("x", "german")
