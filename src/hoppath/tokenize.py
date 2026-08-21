"""Shared word tokenizer for chunk sizing, BM25 postings, and queries; all
three must agree or BM25 statistics drift.

Analyzers: ``simple`` (lowercase tokens) and ``english`` (Lucene-style
stopwords + Porter stemming, needed to reproduce published BM25 baselines;
the ingest default). The ingest analyzer is recorded in corpus meta and
read back at query time.
"""

from __future__ import annotations

import re

from hoppath.porter import stem

_WORD_RE = re.compile(r"\w+")

#: Lucene EnglishAnalyzer default stopword set.
LUCENE_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    )
)

ANALYZERS = ("simple", "english")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (unicode word chars, underscores included)."""
    return [m.group(0).casefold() for m in _WORD_RE.finditer(text)]


def analyze(text: str, analyzer: str = "simple") -> list[str]:
    """Index/query terms under the named analyzer."""
    tokens = tokenize(text)
    if analyzer == "simple":
        return tokens
    if analyzer == "english":
        return [stem(t) for t in tokens if t not in LUCENE_STOPWORDS]
    raise ValueError(f"unknown analyzer {analyzer!r}; expected one of {ANALYZERS}")
