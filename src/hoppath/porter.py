"""Porter stemmer (Porter, 1980) — the algorithm Lucene's English analyzer
applies, implemented dependency-free so the BM25 floor can reproduce
published Lucene/Elasticsearch baselines.

Reference: M.F. Porter, "An algorithm for suffix stripping", Program 14(3).
"""

from __future__ import annotations

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """m in [C](VC)^m[V]: the number of vowel-consonant sequences."""
    m = 0
    i = 0
    n = len(stem)
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    if len(word) < 3:
        return False
    return (
        _is_consonant(word, len(word) - 3)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
)

_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def stem(word: str) -> str:
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _step2(word)
    word = _step3(word)
    word = _step4(word)
    word = _step5(word)
    return word


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            return word[:-1]
        return word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if not _has_vowel(stem_part):
                return word
            word = stem_part
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if _ends_double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if _measure(word) == 1 and _ends_cvc(word):
                return word + "e"
            return word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step2(word: str) -> str:
    for suffix, replacement in _STEP2:
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if _measure(stem_part) > 0:
                return stem_part + replacement
            return word
    return word


def _step3(word: str) -> str:
    for suffix, replacement in _STEP3:
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if _measure(stem_part) > 0:
                return stem_part + replacement
            return word
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if suffix == "ion" and not stem_part.endswith(("s", "t")):
                return word
            if _measure(stem_part) > 1:
                return stem_part
            return word
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem_part = word[:-1]
        m = _measure(stem_part)
        if m > 1 or (m == 1 and not _ends_cvc(stem_part)):
            word = stem_part
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word
