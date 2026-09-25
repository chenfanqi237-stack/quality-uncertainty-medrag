"""Strict, dependency-light JSONL loaders for questions and retrieved evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import EvidenceType, MedicalQuestion, RetrievedEvidence, Stance


def _records(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            yield line_number, value


def _parse_options(raw: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(raw, dict) and len(raw) >= 2:
        return tuple(str(key) for key in raw), tuple(str(value) for value in raw.values())
    if isinstance(raw, list) and len(raw) >= 2 and all(isinstance(item, str) for item in raw):
        labels = tuple(chr(ord("A") + index) for index in range(len(raw)))
        return labels, tuple(raw)
    raise ValueError("options must be a mapping or a list with at least two strings")


def _answer_index(answer: object, labels: tuple[str, ...], options: tuple[str, ...]) -> int:
    if isinstance(answer, int) and not isinstance(answer, bool):
        index = answer
    elif isinstance(answer, str) and answer in labels:
        index = labels.index(answer)
    elif isinstance(answer, str) and answer in options:
        index = options.index(answer)
    else:
        raise ValueError("answer must be an option index, label, or exact option text")
    if not 0 <= index < len(options):
        raise ValueError("answer index is out of range")
    return index


def load_medqa_questions(path: str | Path) -> list[MedicalQuestion]:
    questions: list[MedicalQuestion] = []
    seen: set[str] = set()
    for line_number, raw in _records(path):
        try:
            question_id = str(raw.get("id", raw.get("question_id", ""))).strip()
            if not question_id or question_id in seen:
                raise ValueError("question id is empty or duplicated")
            question_text = str(raw["question"]).strip()
            if not question_text:
                raise ValueError("question text is empty")
            labels, options = _parse_options(raw["options"])
            index = _answer_index(raw["answer"], labels, options)
            question = MedicalQuestion(
                question_id=question_id,
                question=question_text,
                option_labels=labels,
                options=options,
                answer_index=index,
                relevant_doc_ids=tuple(str(item) for item in raw.get("relevant_doc_ids", [])),
                metadata=dict(raw.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid question record at {path}:{line_number}: {exc}") from exc
        seen.add(question_id)
        questions.append(question)
    if not questions:
        raise ValueError(f"No question records found in {path}")
    return questions


def load_retrieved_evidence(path: str | Path) -> dict[str, list[RetrievedEvidence]]:
    grouped: dict[str, list[RetrievedEvidence]] = defaultdict(list)
    seen_doc_ids: set[str] = set()
    for line_number, raw in _records(path):
        try:
            doc_id = str(raw["doc_id"]).strip()
            if not doc_id or doc_id in seen_doc_ids:
                raise ValueError("doc_id is empty or duplicated")
            evidence = RetrievedEvidence(
                schema_version=str(raw["schema_version"]),
                question_id=str(raw["question_id"]).strip(),
                doc_id=doc_id,
                rank=int(raw["rank"]),
                text=str(raw["text"]).strip(),
                source=str(raw["source"]).strip(),
                evidence_type=EvidenceType(raw["evidence_type"]),
                retrieval_score=float(raw["retrieval_score"]),
                annotated_stances={
                    str(label): Stance(value)
                    for label, value in dict(raw.get("annotated_stances", {})).items()
                },
                metadata=dict(raw.get("metadata", {})),
            )
            if evidence.schema_version != "2.0":
                raise ValueError("unsupported schema_version")
            if evidence.rank < 1 or not evidence.question_id or not evidence.text or not evidence.source:
                raise ValueError("rank and required strings are invalid")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid evidence record at {path}:{line_number}: {exc}") from exc
        seen_doc_ids.add(doc_id)
        grouped[evidence.question_id].append(evidence)
    for records in grouped.values():
        records.sort(key=lambda item: item.rank)
        ranks = [item.rank for item in records]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Evidence ranks must be unique within each question")
    if not grouped:
        raise ValueError(f"No evidence records found in {path}")
    return dict(grouped)
