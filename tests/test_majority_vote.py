import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from quality_uncertainty_medrag.aggregation import (
    MajorityVoteAggregationResult,
    MajorityVoteAggregator,
    QualityWeightedVoteAggregator,
)
from quality_uncertainty_medrag.loaders import load_medqa_questions, load_retrieved_evidence
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
from quality_uncertainty_medrag.quality import ConfiguredEvidenceTypeScorer
from quality_uncertainty_medrag.retrieval import PrecomputedRetriever
from quality_uncertainty_medrag.stance import AnnotatedStanceClassifier


QUESTION = MedicalQuestion("q", "Which option?", ("A", "B"), ("one", "two"), 0)
CLAIM = QUESTION.candidate_claims[0]


def scored(doc_id, label, quality=0.5, probabilities=None):
    if probabilities is None:
        probabilities = {stance: float(stance is label) for stance in Stance}
    return ScoredEvidence(
        evidence=RetrievedEvidence(
            schema_version="2.0",
            question_id=QUESTION.question_id,
            doc_id=doc_id,
            rank=1,
            text="Synthetic evidence",
            source="test",
            evidence_type=EvidenceType.OTHER,
            retrieval_score=0.0,
        ),
        quality=QualityScore(quality, "test", components={"hierarchy": quality}),
        stance=StancePrediction(probabilities, "test"),
    )


def aggregate(items):
    return MajorityVoteAggregator().aggregate(QUESTION, CLAIM, items)


@pytest.mark.parametrize(
    ("labels", "decision", "support_votes", "contradict_votes"),
    [
        (
            (Stance.SUPPORT, Stance.CONTRADICT, Stance.SUPPORT),
            AggregationDecision.SUPPORT,
            2,
            1,
        ),
        (
            (Stance.CONTRADICT, Stance.SUPPORT, Stance.CONTRADICT),
            AggregationDecision.CONTRADICT,
            1,
            2,
        ),
        ((Stance.SUPPORT, Stance.CONTRADICT), AggregationDecision.ABSTAIN, 1, 1),
    ],
)
def test_majority_vote_compares_directional_vote_counts(
    labels, decision, support_votes, contradict_votes
):
    result = aggregate([scored(str(index), label) for index, label in enumerate(labels)])

    assert result.claim == CLAIM
    assert result.decision is decision
    assert result.support_vote_count == support_votes
    assert result.contradict_vote_count == contradict_votes
    assert result.support_weight == float(support_votes)
    assert result.contradict_weight == float(contradict_votes)
    assert result.irrelevant_count == 0
    assert result.unresolved_count == 0
    assert result.evidence_count == len(labels)


def test_all_semantically_irrelevant_evidence_abstains():
    result = aggregate([scored("a", Stance.IRRELEVANT), scored("b", Stance.IRRELEVANT)])

    assert result.decision is AggregationDecision.ABSTAIN
    assert result.support_vote_count == result.contradict_vote_count == 0
    assert result.irrelevant_count == 2
    assert result.unresolved_count == 0
    assert result.evidence_count == 2


def test_empty_evidence_abstains():
    result = aggregate([])

    assert result.decision is AggregationDecision.ABSTAIN
    assert result.support_vote_count == result.contradict_vote_count == 0
    assert result.irrelevant_count == result.unresolved_count == 0
    assert result.evidence_count == 0


@pytest.mark.parametrize(
    "probabilities",
    [
        {Stance.SUPPORT: 0.45, Stance.CONTRADICT: 0.45, Stance.IRRELEVANT: 0.10},
        {Stance.SUPPORT: 0.5, Stance.CONTRADICT: 0.0, Stance.IRRELEVANT: 0.5},
        {stance: 1 / 3 for stance in Stance},
    ],
)
def test_unresolved_prediction_is_counted_separately_from_irrelevant(probabilities):
    unresolved = scored("tie", None, quality=1.0, probabilities=probabilities)
    assert unresolved.stance.label is None

    result = aggregate([unresolved, scored("unrelated", Stance.IRRELEVANT)])

    assert result.decision is AggregationDecision.ABSTAIN
    assert result.support_vote_count == result.contradict_vote_count == 0
    assert result.irrelevant_count == 1
    assert result.unresolved_count == 1
    assert result.evidence_count == 2


def test_unresolved_predictions_cannot_overturn_directional_majority():
    probabilities = {Stance.SUPPORT: 0.5, Stance.CONTRADICT: 0.5, Stance.IRRELEVANT: 0.0}
    result = aggregate(
        [
            scored("support", Stance.SUPPORT),
            scored("tie-a", None, probabilities=probabilities),
            scored("tie-b", None, probabilities=probabilities),
        ]
    )

    assert result.decision is AggregationDecision.SUPPORT
    assert result.support_vote_count == 1
    assert result.contradict_vote_count == 0
    assert result.unresolved_count == 2


def test_quality_values_have_no_effect_on_majority_vote():
    low_quality_support = [
        scored("support-a", Stance.SUPPORT, quality=0.0),
        scored("support-b", Stance.SUPPORT, quality=0.0),
        scored("contradict", Stance.CONTRADICT, quality=1.0),
    ]
    high_quality_support = [
        scored("support-a", Stance.SUPPORT, quality=1.0),
        scored("support-b", Stance.SUPPORT, quality=1.0),
        scored("contradict", Stance.CONTRADICT, quality=0.0),
    ]

    assert aggregate(low_quality_support) == aggregate(high_quality_support)
    assert aggregate(low_quality_support).decision is AggregationDecision.SUPPORT
    assert (
        QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, low_quality_support).decision
        is AggregationDecision.CONTRADICT
    )


def test_probability_magnitudes_have_no_effect_when_hard_labels_are_same():
    weak_support = {Stance.SUPPORT: 0.34, Stance.CONTRADICT: 0.33, Stance.IRRELEVANT: 0.33}
    items = [
        scored("a", Stance.SUPPORT, probabilities=weak_support),
        scored("b", Stance.SUPPORT, probabilities=weak_support),
        scored("c", Stance.CONTRADICT),
    ]
    one_hot_items = [
        scored("a", Stance.SUPPORT),
        scored("b", Stance.SUPPORT),
        scored("c", Stance.CONTRADICT),
    ]

    assert aggregate(items) == aggregate(one_hot_items)
    assert aggregate(items).decision is AggregationDecision.SUPPORT


def test_aggregator_only_accesses_hard_labels():
    class LabelOnlyPrediction:
        def __init__(self, label):
            self.label = label

        def __getattr__(self, name):
            raise AssertionError(f"Aggregation accessed stance {name!r}")

    class LabelOnlyEvidence:
        def __init__(self, label):
            self.stance = LabelOnlyPrediction(label)

        def __getattr__(self, name):
            raise AssertionError(f"Aggregation accessed evidence {name!r}")

    result = aggregate(
        [
            LabelOnlyEvidence(Stance.SUPPORT),
            LabelOnlyEvidence(Stance.SUPPORT),
            LabelOnlyEvidence(Stance.CONTRADICT),
            LabelOnlyEvidence(Stance.IRRELEVANT),
            LabelOnlyEvidence(None),
        ]
    )

    assert result.decision is AggregationDecision.SUPPORT
    assert result.support_vote_count == 2
    assert result.contradict_vote_count == 1
    assert result.irrelevant_count == result.unresolved_count == 1
    assert result.evidence_count == 5
    assert (
        result.support_vote_count
        + result.contradict_vote_count
        + result.irrelevant_count
        + result.unresolved_count
        == result.evidence_count
    )


def test_majority_vote_diagnostics_are_immutable_and_json_serializable():
    tied = {Stance.SUPPORT: 0.5, Stance.CONTRADICT: 0.5, Stance.IRRELEVANT: 0.0}
    result = aggregate(
        [
            scored("support", Stance.SUPPORT),
            scored("contradict", Stance.CONTRADICT),
            scored("irrelevant", Stance.IRRELEVANT),
            scored("unresolved", None, probabilities=tied),
        ]
    )

    assert isinstance(result, AggregationResult)
    assert isinstance(result, MajorityVoteAggregationResult)
    payload = json.loads(json.dumps(asdict(result)))
    assert payload["claim"]["option_label"] == "A"
    assert payload["decision"] == "ABSTAIN"
    assert payload["support_vote_count"] == 1
    assert payload["contradict_vote_count"] == 1
    assert payload["irrelevant_count"] == 1
    assert payload["unresolved_count"] == 1
    assert payload["evidence_count"] == 4
    with pytest.raises(FrozenInstanceError):
        result.support_vote_count = 99


def test_majority_vote_is_interchangeable_in_existing_synthetic_pipeline():
    pipeline = BaselinePipeline(
        retriever=PrecomputedRetriever(load_retrieved_evidence("data/synthetic_evidence.jsonl")),
        quality_scorer=ConfiguredEvidenceTypeScorer(
            {evidence_type.value: 0.0 for evidence_type in EvidenceType}, "all-zero"
        ),
        stance_classifier=AnnotatedStanceClassifier(),
        aggregator=MajorityVoteAggregator(),
        top_k=3,
    )
    results = [
        result
        for question in load_medqa_questions("data/synthetic_medqa.jsonl")
        for result in pipeline.run_question(question)
    ]
    results_by_claim = {
        (result.claim.question_id, result.claim.option_label): result.aggregation
        for result in results
    }

    assert len(results) == 12
    assert all(isinstance(result.aggregation, MajorityVoteAggregationResult) for result in results)
    assert results_by_claim[("q1", "A")].decision is AggregationDecision.CONTRADICT
    assert results_by_claim[("q1", "B")].decision is AggregationDecision.SUPPORT
    assert results_by_claim[("q2", "A")].decision is AggregationDecision.SUPPORT
    assert results_by_claim[("q2", "C")].decision is AggregationDecision.ABSTAIN
    assert results_by_claim[("q2", "C")].irrelevant_count == 3
    assert results_by_claim[("q3", "C")].decision is AggregationDecision.ABSTAIN
    assert results_by_claim[("q3", "C")].irrelevant_count == 2
