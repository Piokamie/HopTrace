"""Shared word tokenizer used by chunk sizing, BM25 postings, and queries.

One tokenizer for everything: chunk token counts, index-time postings, and
query-time terms must agree or BM25 statistics silently drift. Two
analyzers: ``simple`` (plain lowercase tokens) and ``english``
(Lucene-style stopword removal + Porter stemming — required to reproduce
the published BM25 baselines, and the ingest default). The analyzer used
at ingest is recorded in corpus meta and read back at query time.
"""

from __future__ import annotations

import re

from hoptrace.porter import stem

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
    """Index/query terms under the named analyzer.

    ``simple`` is plain :func:`tokenize`; ``english`` mimics Lucene's
    EnglishAnalyzer (stopword removal + Porter stemming), which the
    published BM25 baselines HopTrace must reproduce were built with.
    """
    tokens = tokenize(text)
    if analyzer == "simple":
        return tokens
    if analyzer == "english":
        return [stem(t) for t in tokens if t not in LUCENE_STOPWORDS]
    raise ValueError(f"unknown analyzer {analyzer!r}; expected one of {ANALYZERS}")
