"""Rule-based mention extraction. Deterministic, versioned, no LLM.

Passes: capitalized spans, quoted terms, code identifiers, numbers with
units. An optional spaCy NER pass adds candidates behind the ``ner`` config
flag. Overlapping candidates are resolved longest-first. Bump
``EXTRACTOR_VERSION`` on any change to the rules — it is stored per corpus
and gates the regression bracket.

Sentence-initial capitalization is ambiguous ("Yesterday" vs. "Anna"), so
a capitalized word in sentence-initial position is only trusted when the
same word also appears capitalized in a non-initial position — in the same
text or anywhere in the corpus (callers pass corpus-wide evidence via
``corpus_caps``; see ``collect_non_initial_caps``).
"""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from hoptrace.config import ExtractorConfig
from hoptrace.normalize import Normalizer, TrivialNormalizer

if TYPE_CHECKING:
    from spacy.language import Language

EXTRACTOR_VERSION = "1"

_TOKEN_RE = re.compile(r"[\w'’-]+")
_QUOTED_RES = (
    re.compile(r'"([^"\n]{2,80})"'),
    re.compile(r"(?<!\w)'([^'\n]{2,80})'(?!\w)"),
    re.compile(r"`([^`\n]{1,80})`"),
    # typographic quotes, common in word-processor and CMS text
    re.compile(r"“([^“”\n]{2,80})”"),
    re.compile(r"‘([^‘’\n]{2,80})’"),
)
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b|\b(?:[A-Z][a-z0-9]+){2,}\b")
_DOTTED_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
#: Attached unit ("4B", "10km") — up to three trailing letters.
_NUM_UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?[A-Za-z]{1,3}\b")
#: Spaced unit: uppercase-initial ("12 GB") or a known lowercase unit.
_NUM_SPACED_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s(?:[A-Z][A-Za-z]{0,3}|kg|km|cm|mm|ms|hz|mhz|ghz|kb|mb|gb|tb)\b"
)
#: Label-number pairs ("Room 204", "Gate 4B").
_LABEL_NUM_RE = re.compile(r"\b[A-Z][a-z]+ \d+[A-Za-z]?\b")

_SPACY_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP"}
)

#: Function words never admitted as single-token capitalized-span mentions.
_SINGLE_TOKEN_STOP = frozenset(
    "the a an and or but if it he she we they you this that these those "  # noqa: SIM905
    "his her its their my your our as at by for from in of on to with "
    "who what when where why how which whose whom is are was were does do did".split()
)


class _TrustAllCaps:
    """Sentinel trust set for query-time extraction: user-authored queries
    capitalize deliberately, so the prose-oriented sentence-initial
    distrust rule must not eat a query-leading entity ("Kowalski office
    location?"). Junk admissions are filtered downstream by index
    resolution, and failures surface in ``unresolved_mentions``."""

    def __contains__(self, item: object) -> bool:
        return True


#: Pass as ``corpus_caps`` when extracting from queries rather than prose.
QUERY_TRUST: AbstractSet[str] = _TrustAllCaps()  # type: ignore[assignment]

_MAX_CAP_RUN = 4


@dataclass(frozen=True)
class Mention:
    entity: str
    surface: str
    start: int
    end: int


def collect_non_initial_caps(text: str) -> frozenset[str]:
    """Casefolded words seen capitalized in non-sentence-initial position.

    Ingest runs this over every chunk first, then feeds the union back into
    ``extract`` so sentence-initial names are recognized corpus-wide.
    """
    tokens = list(_TOKEN_RE.finditer(text))
    return frozenset(
        tokens[i].group(0).casefold()
        for i in range(len(tokens))
        if tokens[i].group(0)[0].isupper() and not _is_sentence_initial(text, tokens, i)
    )


class MentionExtractor:
    def __init__(
        self,
        cfg: ExtractorConfig | None = None,
        normalizer: Normalizer | None = None,
    ) -> None:
        self._cfg = cfg or ExtractorConfig()
        self._normalizer = normalizer or TrivialNormalizer()
        self._nlp: Language | None = None
        if self._cfg.ner:
            self._nlp = _load_spacy(self._cfg.spacy_model)

    def extract(self, text: str, corpus_caps: AbstractSet[str] = frozenset()) -> list[Mention]:
        candidates: list[tuple[int, int]] = []
        candidates.extend(_capitalized_spans(text, corpus_caps))
        for quoted in _QUOTED_RES:
            candidates.extend(m.span(1) for m in quoted.finditer(text))
        for pattern in (
            _SNAKE_RE,
            _CAMEL_RE,
            _DOTTED_RE,
            _NUM_SPACED_UNIT_RE,
            _NUM_UNIT_RE,
            _LABEL_NUM_RE,
        ):
            candidates.extend(m.span() for m in pattern.finditer(text))
        if self._nlp is not None:
            doc = self._nlp(text)
            candidates.extend(
                (ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in _SPACY_LABELS
            )

        mentions: list[Mention] = []
        for start, end in _resolve_overlaps(candidates):
            surface = text[start:end]
            entity = self._normalizer.canonical(surface)
            if len(entity) < 2:
                continue
            mentions.append(Mention(entity=entity, surface=surface, start=start, end=end))
        return mentions


def _load_spacy(model: str) -> Language:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the NER pass requires spaCy: install with `pip install 'hoptrace[ner]'` "
            f"and download the model with `python -m spacy download {model}`"
        ) from exc
    nlp: Any = spacy.load(model, disable=["lemmatizer", "tagger"])
    return cast("Language", nlp)


def _capitalized_spans(text: str, corpus_caps: AbstractSet[str]) -> list[tuple[int, int]]:
    """Runs of 1-4 adjacent capitalized tokens.

    Sentence-initial handling: a single capitalized sentence opener is kept
    only if trusted (seen capitalized non-initially in this text or in
    ``corpus_caps``); an untrusted sentence opener heading a longer run is
    dropped from the run ("Yesterday Anna Kowalska" yields "Anna Kowalska").
    """
    tokens = list(_TOKEN_RE.finditer(text))
    capitalized = [t.group(0)[0].isupper() for t in tokens]
    initial = [_is_sentence_initial(text, tokens, i) for i in range(len(tokens))]

    # corpus_caps can hold millions of entries and this runs once per
    # chunk — membership-test both sets, never merge/copy.
    local_caps = {
        tokens[i].group(0).casefold()
        for i in range(len(tokens))
        if capitalized[i] and not initial[i]
    }

    def trusted(word: str) -> bool:
        return word in local_caps or word in corpus_caps

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        if not capitalized[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(tokens)
            and capitalized[j + 1]
            and j + 1 - i < _MAX_CAP_RUN
            and text[tokens[j].end() : tokens[j + 1].start()].isspace()
        ):
            j += 1
        start_idx = i
        if initial[i] and not trusted(tokens[i].group(0).casefold()) and j > i:
            start_idx = i + 1
        if start_idx == j:
            word = tokens[start_idx].group(0).casefold()
            if word not in _SINGLE_TOKEN_STOP and (not initial[start_idx] or trusted(word)):
                spans.append((tokens[start_idx].start(), tokens[start_idx].end()))
        else:
            spans.append((tokens[start_idx].start(), tokens[j].end()))
        i = j + 1
    return spans


def _is_sentence_initial(text: str, tokens: list[re.Match[str]], index: int) -> bool:
    if index == 0:
        return True
    between = text[tokens[index - 1].end() : tokens[index].start()]
    return any(c in ".!?\n" for c in between)


def _resolve_overlaps(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep non-overlapping spans, preferring longer ones; output sorted."""
    kept: list[tuple[int, int]] = []
    for start, end in sorted(set(candidates), key=lambda s: (s[0] - s[1], s[0])):
        if all(end <= k_start or start >= k_end for k_start, k_end in kept):
            kept.append((start, end))
    return sorted(kept)
