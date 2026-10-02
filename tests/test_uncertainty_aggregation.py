"""Synthetic tests for the probability-based experimental aggregator."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, asdict

import pytest

from quality_uncertainty_medrag.aggregation import (
    QualityUncertaintyWeightedAggregator,
    QualityWeightedVoteAggregator,
)
from quality_uncertainty_medrag.models import (
    AggregationDecision,
    AggregationResult,
    EvidenceType,
    MedicalQuestion,
    QualityScore,
    RetrievedEvidence,
    ScoredEvidence,
    Stance,
    StancePrediction,
)
from quality_uncertainty_medrag.pipeline import BaselinePipeline
from quality_uncertainty_medrag.uncertainty_aggregation import (
    QualityUncertaintyAggregationResult,
)


@pytest.fixture
def question() -> MedicalQuestion:
    return MedicalQuestion(
        question_id="synthetic-uncertainty",
        question="Synthetic question for a controlled aggregation experiment.",
        option_labels=("A", "B"),
        options=("Synthetic claim A", "Synthetic claim B"),
        answer_index=0,
    )


def scored(
    question: MedicalQuestion,
    doc_id: str,
    quality: float,
    probabilities: tuple[float, float, float],
    prediction_type: type[StancePrediction] = StancePrediction,
) -> ScoredEvidence:
    return ScoredEvidence(
        evidence=RetrievedEvidence(
            schema_version="2.0",
            question_id=question.question_id,
            doc_id=doc_id,
            rank=1,
            text="Explicitly synthetic evidence.",
            source="synthetic",
            evidence_type=EvidenceType.OTHER,
            retrieval_score=0.0,
        ),
        quality=QualityScore(value=quality, scorer_name="synthetic-fixed-quality"),
        stance=prediction_type(
            probabilities=dict(zip(Stance, probabilities)),
            classifier_name="synthetic-fixed-probabilities",
        ),
    )


def aggregate(question: MedicalQuestion, *items: ScoredEvidence):
    return QualityUncertaintyWeightedAggregator().aggregate(
        question, question.candidate_claims[0], items
    )


def expected_entropy(probabilities: tuple[float, float, float]) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0.0) / math.log(3.0)


def test_equal_quality_confident_evidence_has_more_effective_weight(question):
    confident = scored(question, "confident", 0.7, (0.95, 0.03, 0.02))
    uncertain = scored(question, "uncertain", 0.7, (0.4, 0.3, 0.3))
    result = aggregate(question, confident, uncertain)

    first, second = result.per_evidence
    assert first.quality == second.quality == 0.7
    assert first.normalized_entropy < second.normalized_entropy
    assert first.effective_weight > second.effective_weight
    assert result.decision is AggregationDecision.SUPPORT


def test_uncertain_contradiction_downweighted_despite_higher_quality(question):
    support = scored(question, "support", 0.3, (1.0, 0.0, 0.0))
    contradict = scored(question, "contradict", 0.9, (0.32, 0.36, 0.32))
    result = aggregate(question, support, contradict)
    baseline = QualityWeightedVoteAggregator().aggregate(
        question, question.candidate_claims[0], (support, contradict)
    )

    assert baseline.decision is AggregationDecision.CONTRADICT
    assert result.decision is AggregationDecision.SUPPORT
    assert result.per_evidence[0].effective_weight > result.per_evidence[1].effective_weight
    assert result.aggregate_score > 0.0


@pytest.mark.parametrize(
    "support_quality,contradict_quality,decision",
    [
        (0.9, 0.1, AggregationDecision.SUPPORT),
        (0.1, 0.9, AggregationDecision.CONTRADICT),
        (0.5, 0.5, AggregationDecision.ABSTAIN),
    ],
)
def test_one_hot_reduces_to_quality_weighted_directional_evidence(
    question, support_quality, contradict_quality, decision
):
    items = (
        scored(question, "support", support_quality, (1.0, 0.0, 0.0)),
        scored(question, "contradict", contradict_quality, (0.0, 1.0, 0.0)),
    )
    result = aggregate(question, *items)
    baseline = QualityWeightedVoteAggregator().aggregate(
        question, question.candidate_claims[0], items
    )

    assert result.decision is baseline.decision is decision
    assert result.aggregate_score == pytest.approx(
        (support_quality - contradict_quality) / (support_quality + contradict_quality)
    )
    assert result.total_effective_weight == pytest.approx(support_quality + contradict_quality)
    assert [entry.normalized_entropy for entry in result.per_evidence] == [0.0, 0.0]
    assert [entry.effective_weight for entry in result.per_evidence] == [
        support_quality,
        contradict_quality,
    ]


def test_one_hot_irrelevant_keeps_formula_denominator_and_zero_direction(question):
    result = aggregate(
        question,
        scored(question, "support", 0.9, (1.0, 0.0, 0.0)),
        scored(question, "contradict", 0.1, (0.0, 1.0, 0.0)),
        scored(question, "irrelevant", 0.5, (0.0, 0.0, 1.0)),
    )

    assert result.total_effective_weight == pytest.approx(1.5)
    assert result.aggregate_score == pytest.approx(0.8 / 1.5)
    assert result.per_evidence[-1].directional_score == 0.0
    assert result.per_evidence[-1].normalized_entropy == 0.0
    assert result.per_evidence[-1].effective_weight == 0.5


def test_uniform_probabilities_have_exact_maximum_uncertainty_and_zero_weight(question):
    uniform = (1.0 / 3.0,) * 3
    result = aggregate(question, scored(question, "uniform", 1.0, uniform))

    assert result.per_evidence[0].normalized_entropy == 1.0
    assert result.per_evidence[0].effective_weight == 0.0
    assert result.total_effective_weight == 0.0
    assert result.aggregate_score is None
    assert result.decision is AggregationDecision.ABSTAIN


def test_uniform_evidence_does_not_change_confident_evidence_result(question):
    support = scored(question, "support", 0.2, (1.0, 0.0, 0.0))
    uniform = scored(question, "uniform", 1.0, (1.0 / 3.0,) * 3)
    without = aggregate(question, support)
    with_uniform = aggregate(question, support, uniform)

    assert with_uniform.aggregate_score == without.aggregate_score == 1.0
    assert with_uniform.total_effective_weight == without.total_effective_weight == 0.2
    assert with_uniform.decision is without.decision is AggregationDecision.SUPPORT


def test_approximately_normalized_uniform_distribution_keeps_entropy_bounded(question):
    result = aggregate(question, scored(question, "approximately-uniform", 0.8, (0.3333335,) * 3))
    diagnostic = result.per_evidence[0]

    assert diagnostic.normalized_entropy == 1.0
    assert diagnostic.effective_weight == 0.0
    assert result.total_effective_weight == 0.0
    assert result.aggregate_score is None
    assert result.decision is AggregationDecision.ABSTAIN


def test_empty_evidence_abstains_without_undefined_numeric_score(question):
    result = aggregate(question)

    assert result.evidence_count == 0
    assert result.per_evidence == ()
    assert result.total_effective_weight == 0.0
    assert result.aggregate_score is None
    assert result.decision is AggregationDecision.ABSTAIN


def test_zero_quality_has_zero_total_effective_weight(question):
    result = aggregate(
        question,
        scored(question, "support", 0.0, (1.0, 0.0, 0.0)),
        scored(question, "contradict", 0.0, (0.05, 0.9, 0.05)),
    )

    assert result.total_effective_weight == 0.0
    assert result.aggregate_score is None
    assert result.decision is AggregationDecision.ABSTAIN


@pytest.mark.parametrize(
    "probabilities,decision",
    [
        ((0.4500000000002, 0.45, 0.0999999999998), AggregationDecision.SUPPORT),
        ((0.45, 0.4500000000002, 0.0999999999998), AggregationDecision.CONTRADICT),
    ],
)
def test_unresolved_hard_label_keeps_small_probability_direction(
    question, probabilities, decision
):
    item = scored(question, "near-tie", 0.8, probabilities)
    assert item.stance.label is None
    result = aggregate(question, item)

    assert result.total_effective_weight > 0.0
    assert result.aggregate_score == pytest.approx(
        probabilities[0] - probabilities[1], rel=1e-12, abs=0.0
    )
    assert result.decision is decision


def test_equal_nonuniform_direction_abstains_with_positive_effective_weight(question):
    result = aggregate(question, scored(question, "tie", 0.8, (0.45, 0.45, 0.1)))

    assert result.total_effective_weight > 0.0
    assert result.aggregate_score == 0.0
    assert result.decision is AggregationDecision.ABSTAIN


def test_soft_irrelevant_argmax_can_still_contribute_direction(question):
    item = scored(question, "soft-irrelevant", 0.7, (0.3, 0.1, 0.6))
    assert item.stance.label is Stance.IRRELEVANT
    result = aggregate(question, item)

    assert result.per_evidence[0].directional_score == pytest.approx(0.2)
    assert result.aggregate_score == pytest.approx(0.2)
    assert result.decision is AggregationDecision.SUPPORT


class DistributionOnlyStancePrediction(StancePrediction):
    @property
    def label(self):
        raise AssertionError("The experimental method must not read the hard label")

    @property
    def confidence(self):
        raise AssertionError("The experimental method must not read confidence")

    @property
    def top_probability(self):
        raise AssertionError("The experimental method must use the full distribution")

    @property
    def is_tied(self):
        raise AssertionError("Hard-label tie semantics must not determine the score")


def test_aggregation_reads_full_distribution_without_hard_label_or_confidence(question):
    result = aggregate(
        question,
        scored(question, "distribution", 0.6, (0.7, 0.2, 0.1), DistributionOnlyStancePrediction),
    )

    assert result.decision is AggregationDecision.SUPPORT
    assert result.aggregate_score == pytest.approx(0.5)


def test_diagnostics_match_independent_entropy_and_weight_calculation(question):
    probabilities = [(0.8, 0.15, 0.05), (0.1, 0.7, 0.2)]
    qualities = [0.4, 0.9]
    result = aggregate(
        question,
        *(scored(question, f"doc-{i}", q, p) for i, (q, p) in enumerate(zip(qualities, probabilities))),
    )
    expected_weights = []
    expected_contributions = []
    for index, (entry, q, p) in enumerate(zip(result.per_evidence, qualities, probabilities)):
        entropy = expected_entropy(p)
        weight = q * (1.0 - entropy)
        direction = p[0] - p[1]
        assert entry.doc_id == f"doc-{index}"
        assert entry.quality == q
        assert entry.directional_score == pytest.approx(direction)
        assert entry.normalized_entropy == pytest.approx(entropy)
        assert entry.effective_weight == pytest.approx(weight)
        expected_weights.append(weight)
        expected_contributions.append(weight * direction)

    assert result.total_effective_weight == pytest.approx(sum(expected_weights))
    assert result.aggregate_score == pytest.approx(sum(expected_contributions) / sum(expected_weights))


def test_tiny_nonzero_probabilities_produce_finite_entropy_and_score(question):
    result = aggregate(
        question,
        scored(question, "tiny", 0.9, (1.0 - 2e-15, 1e-15, 1e-15)),
    )
    diagnostic = result.per_evidence[0]

    assert 0.0 < diagnostic.normalized_entropy < 1.0
    assert math.isfinite(diagnostic.normalized_entropy)
    assert math.isfinite(diagnostic.effective_weight)
    assert math.isfinite(result.aggregate_score)
    assert result.decision is AggregationDecision.SUPPORT


def test_tiny_quality_does_not_underflow_nonzero_direction_to_abstention(question):
    probabilities = (0.5000000000001, 0.4999999999999, 0.0)
    result = aggregate(question, scored(question, "tiny-quality", 1e-320, probabilities))

    assert result.total_effective_weight > 0.0
    assert result.aggregate_score == probabilities[0] - probabilities[1]
    assert result.decision is AggregationDecision.SUPPORT


def test_opposite_equal_distribution_contributions_cancel_exactly(question):
    result = aggregate(
        question,
        scored(question, "support", 0.6, (0.8, 0.1, 0.1)),
        scored(question, "contradict", 0.6, (0.1, 0.8, 0.1)),
    )

    assert result.total_effective_weight > 0.0
    assert result.aggregate_score == 0.0
    assert result.decision is AggregationDecision.ABSTAIN


def test_result_and_diagnostics_are_immutable_and_json_serializable(question):
    result = aggregate(question, scored(question, "support", 0.8, (0.8, 0.1, 0.1)))
    assert isinstance(result, AggregationResult)
    assert isinstance(result, QualityUncertaintyAggregationResult)
    payload = json.loads(json.dumps(asdict(result), allow_nan=False))

    assert payload["decision"] == "SUPPORT"
    assert payload["per_evidence"][0]["doc_id"] == "support"
    assert payload["per_evidence"][0]["quality"] == 0.8
    assert payload["aggregate_score"] == pytest.approx(result.aggregate_score)
    with pytest.raises(FrozenInstanceError):
        result.aggregate_score = 0.0
    with pytest.raises(FrozenInstanceError):
        result.per_evidence[0].effective_weight = 0.0


def test_zero_weight_result_serializes_score_as_json_null(question):
    result = aggregate(question, scored(question, "zero", 0.0, (1.0, 0.0, 0.0)))
    payload = json.loads(json.dumps(asdict(result), allow_nan=False))

    assert payload["aggregate_score"] is None
    assert payload["decision"] == "ABSTAIN"


def test_generation_free_aggregation_is_deterministic(question):
    items = (
        scored(question, "first", 0.7, (0.75, 0.2, 0.05)),
        scored(question, "second", 0.5, (0.1, 0.6, 0.3)),
    )
    assert aggregate(question, *items) == aggregate(question, *items)


def test_existing_pipeline_accepts_experimental_aggregator(question):
    item = scored(question, "pipeline-doc", 0.6, (0.8, 0.15, 0.05))

    class Retriever:
        def retrieve(self, question, *, top_k):
            return (item.evidence,)

    class QualityScorer:
        def score(self, question, evidence):
            return item.quality

    class Classifier:
        def classify(self, question, claim, evidence):
            return item.stance

    pipeline = BaselinePipeline(
        retriever=Retriever(),
        quality_scorer=QualityScorer(),
        stance_classifier=Classifier(),
        aggregator=QualityUncertaintyWeightedAggregator(),
        top_k=1,
    )
    results = pipeline.run_question(question)

    assert len(results) == len(question.candidate_claims)
    assert all(isinstance(result.aggregation, QualityUncertaintyAggregationResult) for result in results)
    assert all(result.aggregation.decision is AggregationDecision.SUPPORT for result in results)
    assert all(result.aggregation.aggregate_score == pytest.approx(0.65) for result in results)
