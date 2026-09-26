"""Separate evaluation utilities for retrieval ranking and question answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .models import MedicalQuestion, PipelineResult, QuestionPrediction


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """Retrieval metrics over questions with relevant-document annotations."""

    question_count: int
    claim_count: int
    retrieval_evaluated_question_count: int
    retrieval_mrr_at_k: float | None
    retrieval_hit_at_k: float | None
    retrieval_k: int

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def exact_match_accuracy(references: Sequence[str], predictions: Sequence[str]) -> float:
    if not references or len(references) != len(predictions):
        raise ValueError("References and predictions must have the same nonzero length")
    matches = sum(
        reference.strip().casefold() == prediction.strip().casefold()
        for reference, prediction in zip(references, predictions)
    )
    return matches / len(references)


def question_prediction_accuracy(
    questions: Sequence[MedicalQuestion], predictions: Sequence[QuestionPrediction]
) -> float:
    if not questions:
        raise ValueError("questions must not be empty")
    prediction_by_id: Mapping[str, QuestionPrediction] = {
        prediction.question_id: prediction for prediction in predictions
    }
    if len(prediction_by_id) != len(predictions) or set(prediction_by_id) != {
        question.question_id for question in questions
    }:
        raise ValueError("Predictions must contain exactly one prediction per question")
    for question in questions:
        prediction = prediction_by_id[question.question_id]
        if set(prediction.option_scores) != set(question.option_labels):
            raise ValueError("Each prediction must score every option in its question exactly once")
    correct = sum(
        prediction_by_id[question.question_id].predicted_option_label == question.answer_label
        for question in questions
    )
    return correct / len(questions)


def evaluate_retrieval(
    questions: Sequence[MedicalQuestion], results: Sequence[PipelineResult], *, k: int
) -> RetrievalEvaluationSummary:
    """Score retrieval once per annotated question, independently of answers.

    Candidate claims share a retrieval pool, so its ordering is read from the
    first claim. Questions without gold document IDs are not evaluated. If no
    questions have annotations, the metrics are unavailable and returned as None.
    """

    if k < 1 or not questions:
        raise ValueError("k must be positive and questions must not be empty")
    by_key: Mapping[tuple[str, str], PipelineResult] = {
        (item.question.question_id, item.claim.option_label): item for item in results
    }
    expected_keys = {
        (question.question_id, claim.option_label)
        for question in questions
        for claim in question.candidate_claims
    }
    if len(by_key) != len(results) or set(by_key) != expected_keys:
        raise ValueError("Results must contain exactly one result per candidate claim")

    reciprocal_rank_total = 0.0
    hit_total = 0
    evaluated_question_count = 0
    for question in questions:
        relevant = set(question.relevant_doc_ids)
        if not relevant:
            continue
        evaluated_question_count += 1
        first_claim = question.candidate_claims[0]
        claim_result = by_key[(question.question_id, first_claim.option_label)]
        ranked_ids = [item.evidence.doc_id for item in claim_result.evidence]
        first = next(
            (rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in relevant), None
        )
        if first is not None:
            reciprocal_rank_total += 1 / first
            hit_total += first <= k

    return RetrievalEvaluationSummary(
        question_count=len(questions),
        claim_count=len(results),
        retrieval_evaluated_question_count=evaluated_question_count,
        retrieval_mrr_at_k=(
            reciprocal_rank_total / evaluated_question_count
            if evaluated_question_count else None
        ),
        retrieval_hit_at_k=(
            hit_total / evaluated_question_count if evaluated_question_count else None
        ),
        retrieval_k=k,
    )
