"""Per-dataset parsing into a unified question shape; missing keys raise
``SchemaError`` naming the file."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paragraph:
    #: Stable key within the question (title for HotpotQA-format, idx-title
    #: for MuSiQue where titles repeat).
    key: str
    title: str
    text: str
    is_gold: bool


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    text: str
    answer: str
    #: e.g. "bridge", "comparison", "2hop"; comparison/yes-no handling in
    #: the diagnostics keys off this.
    qtype: str
    paragraphs: tuple[Paragraph, ...]

    @property
    def gold_keys(self) -> frozenset[str]:
        return frozenset(p.key for p in self.paragraphs if p.is_gold)


class SchemaError(RuntimeError):
    pass


def _require(mapping: dict[str, object], key: str, source: Path, context: str) -> object:
    if key not in mapping:
        raise SchemaError(f"{source}: missing key {key!r} in {context}")
    return mapping[key]


def iter_beir_corpus(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yields (doc_id, title, text) from a BEIR corpus.jsonl."""
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_id" not in row or "text" not in row:
                raise SchemaError(f"{path}: line {line_no} lacks _id/text")
            yield str(row["_id"]), str(row.get("title") or ""), str(row["text"])


def load_beir_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_id" not in row or "text" not in row:
                raise SchemaError(f"{path}: line {line_no} lacks _id/text")
            queries[str(row["_id"])] = str(row["text"])
    return queries


def load_beir_qrels(path: Path) -> dict[str, set[str]]:
    """TSV with a header row: query-id, corpus-id, score."""
    qrels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        header = fh.readline()
        if "query" not in header:
            raise SchemaError(f"{path}: unexpected qrels header {header!r}")
        for line_no, line in enumerate(fh, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise SchemaError(f"{path}: line {line_no} has {len(parts)} fields, expected 3")
            qid, doc_id, score = parts
            try:
                relevance = int(score)
            except ValueError as exc:
                raise SchemaError(
                    f"{path}: line {line_no} has non-integer score {score!r}"
                ) from exc
            if relevance > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def load_musique(path: Path) -> list[EvalQuestion]:
    """MuSiQue-Ans JSONL: id, question, answer, paragraphs[{idx, title,
    paragraph_text, is_supporting}]. qtype is the hop count from the id
    prefix ("2hop", "3hop", "4hop")."""
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [_musique_question(row, path, f"question #{i}") for i, row in enumerate(rows)]


def load_musique_json(path: Path) -> list[EvalQuestion]:
    """Same schema as ``load_musique``, as a JSON array (HippoRAG's packaging)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SchemaError(f"{path}: expected a top-level list")
    return [_musique_question(row, path, f"question #{i}") for i, row in enumerate(data)]


def load_corpus_entries(path: Path) -> list[tuple[str, str]]:
    """A JSON array of {title, text}: an explicit retrieval corpus."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SchemaError(f"{path}: expected a top-level list")
    entries: list[tuple[str, str]] = []
    for i, row in enumerate(data):
        context = f"corpus entry #{i}"
        entries.append(
            (
                str(_require(row, "title", path, context)),
                str(_require(row, "text", path, context)),
            )
        )
    return entries


def _musique_question(row: dict[str, object], path: Path, context: str) -> EvalQuestion:
    qid = str(_require(row, "id", path, context))
    raw_paragraphs = _require(row, "paragraphs", path, context)
    if not isinstance(raw_paragraphs, list):
        raise SchemaError(f"{path}: {context} has non-list paragraphs")
    paragraphs = tuple(
        Paragraph(
            key=f"{_require(p, 'idx', path, context)}-{_require(p, 'title', path, context)}",
            title=str(_require(p, "title", path, context)),
            text=str(_require(p, "paragraph_text", path, context)),
            is_gold=bool(_require(p, "is_supporting", path, context)),
        )
        for p in raw_paragraphs
    )
    return EvalQuestion(
        qid=qid,
        text=str(_require(row, "question", path, context)),
        answer=str(_require(row, "answer", path, context)),
        qtype=qid.split("_", 1)[0].replace("__", ""),
        paragraphs=paragraphs,
    )


def load_hotpot_format(path: Path) -> list[EvalQuestion]:
    """HotpotQA-format JSON (also 2WikiMultihopQA): a list of {_id,
    question, answer, type, context: [[title, [sentences]]],
    supporting_facts: [[title, sent_no]]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SchemaError(f"{path}: expected a top-level list")
    questions: list[EvalQuestion] = []
    for i, row in enumerate(data):
        context_label = f"question #{i}"
        qid = str(_require(row, "_id", path, context_label))
        supporting = _require(row, "supporting_facts", path, context_label)
        context_field = _require(row, "context", path, context_label)
        if not isinstance(supporting, list) or not isinstance(context_field, list):
            raise SchemaError(f"{path}: {context_label} has malformed context/supporting_facts")
        gold_titles = {str(fact[0]) for fact in supporting}
        for entry in context_field:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise SchemaError(f"{path}: {context_label} has malformed context entry {entry!r}")
        paragraphs = tuple(
            Paragraph(
                key=str(title),
                title=str(title),
                text=" ".join(str(s) for s in sentences),
                is_gold=str(title) in gold_titles,
            )
            for title, sentences in context_field
        )
        questions.append(
            EvalQuestion(
                qid=qid,
                text=str(_require(row, "question", path, context_label)),
                answer=str(_require(row, "answer", path, context_label)),
                qtype=str(row.get("type") or "unknown"),
                paragraphs=paragraphs,
            )
        )
    return questions
