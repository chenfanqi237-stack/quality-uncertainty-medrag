import json
from dataclasses import replace

import pytest

from quality_uncertainty_medrag.aggregation import QualityWeightedVoteAggregator
from quality_uncertainty_medrag.evaluation import (
    evaluate_retrieval,
    exact_match_accuracy,
    question_prediction_accuracy,
)
from quality_uncertainty_medrag.loaders import (
    load_medqa_questions,
    load_retrieved_evidence,
)
from quality_uncertainty_medrag.models import (
    AggregationDecision,
    EvidenceType,
    QuestionPrediction,
)
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
        result
        for question in questions
        for result in pipeline.run_question(question)
    ]

    summary = evaluate_retrieval(questions, results, k=3)

    assert summary.question_count == 3
    assert summary.claim_count == 12
    assert summary.retrieval_evaluated_question_count == 2
    assert summary.retrieval_mrr_at_k == 0.75
    assert summary.retrieval_hit_at_k == 1.0
    assert summary.retrieval_k == 3

    decisions = {
        (result.claim.question_id, result.claim.option_label): result.aggregation.decision
        for result in results
    }
    assert decisions[("q1", "A")] is AggregationDecision.CONTRADICT
    assert decisions[("q1", "B")] is AggregationDecision.SUPPORT
    assert decisions[("q2", "C")] is AggregationDecision.ABSTAIN


def test_unannotated_questions_do_not_reduce_retrieval_metrics():
    questions = load_medqa_questions("data/synthetic_medqa.jsonl")
    pipeline = make_pipeline()
    results = [result for question in questions for result in pipeline.run_question(question)]
    annotated_questions = [question for question in questions if question.relevant_doc_ids]
    annotated_ids = {question.question_id for question in annotated_questions}
    annotated_results = [
        result for result in results if result.question.question_id in annotated_ids
    ]

    full_summary = evaluate_retrieval(questions, results, k=1)
    annotated_summary = evaluate_retrieval(annotated_questions, annotated_results, k=1)

    assert full_summary.retrieval_evaluated_question_count == 2
    assert full_summary.retrieval_mrr_at_k == annotated_summary.retrieval_mrr_at_k == 0.75
    assert full_summary.retrieval_hit_at_k == annotated_summary.retrieval_hit_at_k == 0.5


def test_all_empty_relevance_annotations_have_null_retrieval_metrics():
    questions = [
        replace(question, relevant_doc_ids=())
        for question in load_medqa_questions("data/synthetic_medqa.jsonl")
    ]
    pipeline = make_pipeline()
    results = [result for question in questions for result in pipeline.run_question(question)]

    summary = evaluate_retrieval(questions, results, k=3)

    assert summary.question_count == 3
    assert summary.claim_count == 12
    assert summary.retrieval_evaluated_question_count == 0
    assert summary.retrieval_mrr_at_k is None
    assert summary.retrieval_hit_at_k is None
    assert summary.retrieval_k == 3
    serialized = json.loads(json.dumps(summary.to_dict()))
    assert serialized["retrieval_mrr_at_k"] is None
    assert serialized["retrieval_hit_at_k"] is None
    assert "mean_reciprocal_rank" not in serialized
    assert "hit_at_k" not in serialized
    assert "k" not in serialized


@pytest.mark.parametrize(
    ("relevant_doc_ids", "k", "expected_mrr", "expected_hit"),
    [
        (("q1-guideline",), 1, 1.0, 1.0),
        (("q1-rct",), 1, 0.5, 0.0),
        (("q1-rct",), 2, 0.5, 1.0),
        (("q1-opinion",), 2, 1 / 3, 0.0),
        (("q1-opinion", "q1-rct"), 2, 0.5, 1.0),
        (("not-retrieved",), 3, 0.0, 0.0),
    ],
)
def test_retrieval_metrics_use_gold_document_ids(
    relevant_doc_ids, k, expected_mrr, expected_hit
):
    question = replace(
        load_medqa_questions("data/synthetic_medqa.jsonl")[0],
        relevant_doc_ids=relevant_doc_ids,
    )
    results = make_pipeline().run_question(question)

    summary = evaluate_retrieval([question], results, k=k)

    assert summary.retrieval_evaluated_question_count == 1
    assert summary.retrieval_mrr_at_k == pytest.approx(expected_mrr)
    assert summary.retrieval_hit_at_k == expected_hit
    assert summary.retrieval_k == k


def test_annotated_retrieval_miss_stays_in_metric_denominator():
    questions = load_medqa_questions("data/synthetic_medqa.jsonl")
    questions[2] = replace(questions[2], relevant_doc_ids=("not-retrieved",))
    pipeline = make_pipeline()
    results = [result for question in questions for result in pipeline.run_question(question)]

    summary = evaluate_retrieval(questions, results, k=3)

    assert summary.retrieval_evaluated_question_count == 3
    assert summary.retrieval_mrr_at_k == 0.5
    assert summary.retrieval_hit_at_k == 2 / 3


@pytest.mark.parametrize("invalid_results", ["missing", "duplicate"])
def test_retrieval_requires_every_candidate_result_even_without_annotations(invalid_results):
    question = replace(
        load_medqa_questions("data/synthetic_medqa.jsonl")[0], relevant_doc_ids=()
    )
    results = list(make_pipeline().run_question(question))
    if invalid_results == "missing":
        results.pop()
    else:
        results.append(results[0])

    with pytest.raises(ValueError, match="exactly one result per candidate claim"):
        evaluate_retrieval([question], results, k=3)


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
