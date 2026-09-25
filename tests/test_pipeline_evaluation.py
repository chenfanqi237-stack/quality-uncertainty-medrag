import pytest

from quality_uncertainty_medrag.aggregation import QualityWeightedVoteAggregator
from quality_uncertainty_medrag.evaluation import (
    evaluate_pipeline,
    exact_match_accuracy,
    question_prediction_accuracy,
)
from quality_uncertainty_medrag.loaders import (
    load_medqa_questions,
    load_retrieved_evidence,
)
from quality_uncertainty_medrag.models import EvidenceType, QuestionPrediction, Stance
from quality_uncertainty_medrag.pipeline import BaselinePipeline
from quality_uncertainty_medrag.quality import ConfiguredEvidenceTypeScorer
from quality_uncertainty_medrag.retrieval import PrecomputedRetriever
from quality_uncertainty_medrag.stance import AnnotatedStanceClassifier


def make_pipeline():
    scores = {
        EvidenceType.META_ANALYSIS.value: 1.0,
        EvidenceType.SYSTEMATIC_REVIEW.value: 0.9,
        EvidenceType.EVIDENCE_BASED_GUIDELINE.value: 0.8,
        EvidenceType.RANDOMIZED_CONTROLLED_TRIAL.value: 0.7,
        EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL.value: 0.6,
        EvidenceType.COHORT_STUDY.value: 0.5,
        EvidenceType.CASE_SERIES_OR_STUDY.value: 0.4,
        EvidenceType.INDIVIDUAL_CASE_REPORT.value: 0.3,
        EvidenceType.EXPERT_OPINION.value: 0.2,
        EvidenceType.OTHER.value: 0.1,
    }
    return BaselinePipeline(
        retriever=PrecomputedRetriever(load_retrieved_evidence("data/synthetic_evidence.jsonl")),
        quality_scorer=ConfiguredEvidenceTypeScorer(scores, "test"),
        stance_classifier=AnnotatedStanceClassifier(),
        aggregator=QualityWeightedVoteAggregator(),
        top_k=3,
    )


def test_end_to_end_synthetic_baseline_is_claim_level():
    questions = load_medqa_questions("data/synthetic_medqa.jsonl")
    pipeline = make_pipeline()
    results = [
        pipeline.run(question, claim)
        for question in questions
        for claim in question.candidate_claims
    ]

    summary = evaluate_pipeline(questions, results, k=3)

    assert summary.question_count == 3
    assert summary.claim_count == 12
    assert summary.mean_reciprocal_rank == 0.5
    assert summary.hit_at_k == 2 / 3

    decisions = {
        (result.claim.question_id, result.claim.option_label): result.aggregation.decision
        for result in results
    }
    assert decisions[("q1", "A")] is Stance.CONTRADICT
    assert decisions[("q1", "B")] is Stance.SUPPORT
    assert decisions[("q2", "C")] is Stance.IRRELEVANT


def test_question_prediction_accuracy_uses_final_mcq_predictions():
    questions = load_medqa_questions("data/synthetic_medqa.jsonl")
    predictions = [
        QuestionPrediction("q1", "B", {"A": 0.2, "B": 0.8, "C": 0.0, "D": 0.0}),
        QuestionPrediction("q2", "A", {"A": 0.7, "B": 0.2, "C": 0.1, "D": 0.0}),
        QuestionPrediction("q3", "C", {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1}),
    ]

    assert question_prediction_accuracy(questions, predictions) == 1.0


def test_question_prediction_accuracy_requires_every_option_score():
    questions = load_medqa_questions("data/synthetic_medqa.jsonl")
    predictions = [
        QuestionPrediction("q1", "B", {"A": 0.2, "B": 0.8}),
        QuestionPrediction("q2", "A", {"A": 0.7, "B": 0.2, "C": 0.1, "D": 0.0}),
        QuestionPrediction("q3", "C", {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1}),
    ]

    with pytest.raises(ValueError, match="score every option"):
        question_prediction_accuracy(questions, predictions)


def test_exact_match_accuracy():
    assert exact_match_accuracy(["A", "Beta"], ["a", " beta "]) == 1.0
