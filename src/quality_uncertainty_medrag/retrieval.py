"""Retriever backed by precomputed JSONL evidence."""

from __future__ import annotations

from typing import Mapping, Sequence

from .models import MedicalQuestion, RetrievedEvidence


class PrecomputedRetriever:
    def __init__(self, evidence_by_question: Mapping[str, Sequence[RetrievedEvidence]]) -> None:
        self._evidence = {key: tuple(value) for key, value in evidence_by_question.items()}

    def retrieve(self, question: MedicalQuestion, *, top_k: int) -> tuple[RetrievedEvidence, ...]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        return self._evidence.get(question.question_id, ())[:top_k]

