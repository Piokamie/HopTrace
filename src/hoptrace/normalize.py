"""Alias normalization: maps mention surface forms to canonical entity keys.

`Normalizer` is the pluggable seam where a v2 lemmatizer (spaCy, Morfeusz)
plugs in. The trivial implementation is deliberately weak on inflected
languages — see DESIGN.md, known limits.
"""

from __future__ import annotations

import re
from typing import Protocol


class Normalizer(Protocol):
    def canonical(self, surface: str) -> str: ...


_POSSESSIVE_RE = re.compile(r"['’]s\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


class TrivialNormalizer:
    """Case, punctuation, possessive, and naive plural normalization.

    The plural rule is a documented approximation: it strips a trailing
    's' from words of length >= 4 unless they end in 'ss', 'us' or 'is',
    which over-strips words like 'kansas'. Inflection is untouched.
    """

    def canonical(self, surface: str) -> str:
        s = surface.casefold().strip()
        s = _POSSESSIVE_RE.sub("", s)
        s = _PUNCT_RE.sub(" ", s)
        s = _WS_RE.sub(" ", s).strip()
        if not s:
            return ""
        return " ".join(self._singular(word) for word in s.split(" "))

    @staticmethod
    def _singular(word: str) -> str:
        if len(word) >= 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
