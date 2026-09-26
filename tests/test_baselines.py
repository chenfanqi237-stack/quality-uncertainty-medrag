import json
import math
import pickle
from dataclasses import asdict

import pytest

from quality_uncertainty_medrag.aggregation import QualityWeightedVoteAggregator
from quality_uncertainty_medrag.models import (
    AggregationDecision,
    CandidateClaim,
    EvidenceType,
    MedicalQuestion,
    QualityScore,
    QuestionPrediction,
    RetrievedEvidence,
    ScoredEvidence,
    Stance,
    StancePrediction,
)
from quality_uncertainty_medrag.quality import ConfiguredEvidenceTypeScorer
from quality_uncertainty_medrag.stance import AnnotatedStanceClassifier


QUESTION = MedicalQuestion(
    question_id="q",
    question="Question?",
    option_labels=("A", "B"),
    options=("one", "two"),
    answer_index=0,
)
CLAIM = QUESTION.candidate_claims[0]


def evidence(doc_id, evidence_type, stances):
    return RetrievedEvidence(
        schema_version="2.0",
        question_id="q",
        doc_id=doc_id,
        rank=1,
        text="text",
        source="source",
        evidence_type=evidence_type,
        retrieval_score=0.9,
        annotated_stances=stances,
    )


def prediction(label, probabilities=None):
    if probabilities is None:
        probabilities = {stance: float(stance is label) for stance in Stance}
    return StancePrediction(probabilities=probabilities, classifier_name="test")


def scored(doc_id, quality, stance):
    item = evidence(doc_id, EvidenceType.OTHER, {"A": stance})
    return ScoredEvidence(
        evidence=item,
        quality=QualityScore(
            value=quality,
            scorer_name="test",
            components={"hierarchy": quality},
        ),
        stance=prediction(stance),
    )


def test_configured_quality_scorer_exposes_hierarchy_component():
    scores = {item.value: 0.5 for item in EvidenceType}
    scores[EvidenceType.META_ANALYSIS.value] = 1.0
    scorer = ConfiguredEvidenceTypeScorer(scores, "configured")
    result = scorer.score(
        QUESTION, evidence("d", EvidenceType.META_ANALYSIS, {"A": Stance.SUPPORT})
    )
    assert result.value == 1.0
    assert result.components == {"hierarchy": 1.0}


def test_quality_score_rejects_nonfinite_component():
    with pytest.raises(ValueError, match="finite"):
        QualityScore(0.5, "test", components={"recency": math.nan})


def test_quality_scorer_requires_all_types():
    with pytest.raises(ValueError, match="Missing"):
        ConfiguredEvidenceTypeScorer({"other": 0.1}, "incomplete")


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({Stance.SUPPORT: 1.0, Stance.CONTRADICT: 0.0}, "exactly"),
        (
            {Stance.SUPPORT: 1.1, Stance.CONTRADICT: 0.0, Stance.IRRELEVANT: -0.1},
            r"in \[0, 1\]",
        ),
        (
            {Stance.SUPPORT: 0.5, Stance.CONTRADICT: 0.4, Stance.IRRELEVANT: 0.0},
            "sum",
        ),
    ],
)
def test_stance_prediction_validates_probability_distribution(probabilities, message):
    with pytest.raises(ValueError, match=message):
        prediction(Stance.SUPPORT, probabilities)


def test_stance_prediction_preserves_distribution_and_derives_label():
    probabilities = {
        Stance.SUPPORT: 0.7,
        Stance.CONTRADICT: 0.2,
        Stance.IRRELEVANT: 0.1,
    }
    result = prediction(Stance.SUPPORT, probabilities)
    assert dict(result.probabilities) == probabilities
    assert result.label is Stance.SUPPORT
    assert result.confidence == 0.7
    assert result.top_probability == 0.7
    assert result.is_tied is False


def test_stance_prediction_normalizes_serialized_string_keys():
    result = StancePrediction(
        probabilities={"SUPPORT": 0.7, "CONTRADICT": 0.2, "IRRELEVANT": 0.1},
        classifier_name="test",
    )
    assert all(isinstance(stance, Stance) for stance in result.probabilities)
    assert {stance.value: value for stance, value in result.probabilities.items()} == {
        "SUPPORT": 0.7,
        "CONTRADICT": 0.2,
        "IRRELEVANT": 0.1,
    }


def test_stance_prediction_accepts_small_rounding_error_in_probability_sum():
    result = StancePrediction(
        probabilities={
            Stance.SUPPORT: 0.6,
            Stance.CONTRADICT: 0.3,
            Stance.IRRELEVANT: 0.1000005,
        },
        classifier_name="test",
    )
    assert result.label is Stance.SUPPORT


@pytest.mark.parametrize(
    "probabilities",
    [
        {Stance.SUPPORT: 0.5, Stance.CONTRADICT: 0.5, Stance.IRRELEVANT: 0.0},
        {Stance.SUPPORT: 0.4, Stance.CONTRADICT: 0.2, Stance.IRRELEVANT: 0.4},
        {
            Stance.SUPPORT: 1 / 3,
            Stance.CONTRADICT: 1 / 3,
            Stance.IRRELEVANT: 1 / 3,
        },
    ],
)
def test_stance_prediction_returns_none_when_highest_probability_is_tied(probabilities):
    result = StancePrediction(probabilities=probabilities, classifier_name="test")
    assert result.label is None
    assert result.confidence == max(probabilities.values())
    assert result.top_probability == max(probabilities.values())
    assert result.is_tied is True


def test_directional_tie_confidence_reports_top_probability():
    result = StancePrediction(
        probabilities={
            Stance.SUPPORT: 0.45,
            Stance.CONTRADICT: 0.45,
            Stance.IRRELEVANT: 0.10,
        },
        classifier_name="test",
    )

    assert result.label is None
    assert result.is_tied is True
    assert result.confidence == 0.45
    assert result.top_probability == 0.45


def test_stance_prediction_preserves_unique_irrelevant_label():
    result = StancePrediction(
        probabilities={
            Stance.SUPPORT: 0.1,
            Stance.CONTRADICT: 0.2,
            Stance.IRRELEVANT: 0.7,
        },
        classifier_name="test",
    )

    assert result.label is Stance.IRRELEVANT
    assert result.is_tied is False


def test_model_mappings_support_generic_dataclass_json_serialization():
    quality = QualityScore(0.8, "test", components={"hierarchy": 0.8})
    stance = StancePrediction(
        probabilities={
            Stance.SUPPORT: 0.7,
            Stance.CONTRADICT: 0.2,
            Stance.IRRELEVANT: 0.1,
        },
        classifier_name="test",
    )
    question_prediction = QuestionPrediction("q", "B", {"A": 0.2, "B": 0.8})

    quality_payload = json.loads(json.dumps(asdict(quality)))
    stance_payload = json.loads(json.dumps(asdict(stance)))
    question_prediction_payload = json.loads(json.dumps(asdict(question_prediction)))

    assert quality_payload["components"] == {"hierarchy": 0.8}
    assert stance_payload["probabilities"]["SUPPORT"] == 0.7
    assert question_prediction_payload["option_scores"] == {"A": 0.2, "B": 0.8}
    assert pickle.loads(pickle.dumps(quality)) == quality
    assert pickle.loads(pickle.dumps(stance)) == stance
    assert pickle.loads(pickle.dumps(question_prediction)) == question_prediction


def test_model_mappings_are_defensively_copied_from_inputs():
    components = {"hierarchy": 0.8}
    probabilities = {
        Stance.SUPPORT: 0.7,
        Stance.CONTRADICT: 0.2,
        Stance.IRRELEVANT: 0.1,
    }
    option_scores = {"A": 0.2, "B": 0.8}
    quality = QualityScore(0.8, "test", components=components)
    stance = StancePrediction(probabilities=probabilities, classifier_name="test")
    question_prediction = QuestionPrediction("q", "B", option_scores)

    components["hierarchy"] = 0.1
    probabilities[Stance.SUPPORT] = 0.1
    option_scores["B"] = 0.1

    assert quality.components["hierarchy"] == 0.8
    assert stance.probabilities[Stance.SUPPORT] == 0.7
    assert question_prediction.option_scores["B"] == 0.8


def test_validated_model_mappings_cannot_be_mutated_after_construction():
    quality = QualityScore(0.8, "test", components={"hierarchy": 0.8})
    stance = StancePrediction(
        probabilities={
            Stance.SUPPORT: 0.7,
            Stance.CONTRADICT: 0.2,
            Stance.IRRELEVANT: 0.1,
        },
        classifier_name="test",
    )
    question_prediction = QuestionPrediction("q", "B", {"A": 0.2, "B": 0.8})

    with pytest.raises(TypeError, match="immutable"):
        quality.components["hierarchy"] = math.nan
    with pytest.raises(TypeError, match="immutable"):
        stance.probabilities.clear()
    with pytest.raises(TypeError, match="immutable"):
        question_prediction.option_scores.pop("B")


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (
            [scored("a", 0.8, Stance.SUPPORT), scored("b", 0.3, Stance.CONTRADICT)],
            AggregationDecision.SUPPORT,
        ),
        (
            [scored("a", 0.2, Stance.SUPPORT), scored("b", 0.7, Stance.CONTRADICT)],
            AggregationDecision.CONTRADICT,
        ),
    ],
)
def test_quality_weighted_vote_remains_claim_level(items, expected):
    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, items)
    assert result.claim == CLAIM
    assert result.decision is expected


def test_quality_weighted_vote_equal_directional_weights_abstains():
    items = [scored("a", 0.5, Stance.SUPPORT), scored("b", 0.5, Stance.CONTRADICT)]
    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, items)
    assert result.decision is AggregationDecision.ABSTAIN
    assert json.loads(json.dumps(asdict(result)))["decision"] == "ABSTAIN"


def test_quality_weighted_vote_without_directional_evidence_abstains():
    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, [])
    assert result.decision is AggregationDecision.ABSTAIN
    assert result.support_weight == 0.0
    assert result.contradict_weight == 0.0
    assert result.irrelevant_count == 0
    assert result.evidence_count == 0


def test_semantic_irrelevant_evidence_is_distinct_from_aggregate_abstain():
    item = scored("irrelevant", 0.9, Stance.IRRELEVANT)
    assert item.stance.label is Stance.IRRELEVANT

    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, [item])

    assert result.decision is AggregationDecision.ABSTAIN
    assert result.irrelevant_count == 1
    assert result.support_weight == 0.0
    assert result.contradict_weight == 0.0


def test_quality_weighted_vote_treats_rounding_level_weight_difference_as_tie():
    items = [
        scored("support-a", 0.1, Stance.SUPPORT),
        scored("support-b", 0.2, Stance.SUPPORT),
        scored("contradict", 0.3, Stance.CONTRADICT),
    ]
    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, items)
    assert result.decision is AggregationDecision.ABSTAIN


def test_quality_weighted_vote_explicitly_ignores_unresolved_stance_tie():
    item = ScoredEvidence(
        evidence=evidence("tie", EvidenceType.OTHER, {}),
        quality=QualityScore(0.9, "test", components={"hierarchy": 0.9}),
        stance=StancePrediction(
            probabilities={
                Stance.SUPPORT: 0.5,
                Stance.CONTRADICT: 0.5,
                Stance.IRRELEVANT: 0.0,
            },
            classifier_name="test",
        ),
    )
    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, [item])
    assert result.decision is AggregationDecision.ABSTAIN
    assert result.support_weight == 0.0
    assert result.contradict_weight == 0.0
    assert result.irrelevant_count == 0
    assert result.evidence_count == 1


def test_annotated_classifier_returns_complete_one_hot_probabilities():
    item = evidence("d", EvidenceType.OTHER, {"A": Stance.CONTRADICT})
    result = AnnotatedStanceClassifier().classify(QUESTION, CLAIM, item)
    assert result.label is Stance.CONTRADICT
    assert result.probabilities == {
        Stance.SUPPORT: 0.0,
        Stance.CONTRADICT: 1.0,
        Stance.IRRELEVANT: 0.0,
    }


def test_annotated_classifier_requires_claim_specific_annotation():
    item = evidence("d", EvidenceType.OTHER, {"B": Stance.SUPPORT})
    with pytest.raises(ValueError, match="option A"):
        AnnotatedStanceClassifier().classify(QUESTION, CLAIM, item)


def test_candidate_claim_validation():
    with pytest.raises(ValueError, match="nonnegative"):
        CandidateClaim("q", -1, "A", "answer")


def test_medical_question_rejects_empty_option_text():
    with pytest.raises(ValueError, match="Option text"):
        MedicalQuestion("q", "Question?", ("A", "B"), ("one", " "), 0)


def test_medical_question_metadata_is_deeply_copied_frozen_and_serializable():
    metadata = {"source": {"tags": ["synthetic"]}}
    question = MedicalQuestion(
        "q",
        "Question?",
        ("A", "B"),
        ("one", "two"),
        0,
        metadata=metadata,
    )

    metadata["source"]["tags"].append("mutated")
    metadata["source"]["added"] = True

    assert question.metadata["source"]["tags"] == ("synthetic",)
    assert "added" not in question.metadata["source"]
    with pytest.raises(TypeError, match="immutable"):
        question.metadata["source"] = {}
    with pytest.raises(TypeError, match="immutable"):
        question.metadata["source"]["added"] = True
    payload = json.loads(json.dumps(asdict(question)))
    assert payload["metadata"] == {"source": {"tags": ["synthetic"]}}
    restored = pickle.loads(pickle.dumps(question))
    assert restored == question
    with pytest.raises(TypeError, match="immutable"):
        restored.metadata["source"]["added"] = True


@pytest.mark.parametrize(
    "metadata",
    [
        {"unsupported": {"set"}},
        {"unsupported": bytearray(b"mutable")},
        {"unsupported": math.nan},
        {1: "non-string key"},
    ],
)
def test_medical_question_metadata_rejects_non_json_values(metadata):
    with pytest.raises(TypeError, match="Metadata"):
        MedicalQuestion("q", "Question?", ("A", "B"), ("one", "two"), 0, metadata=metadata)


def test_retrieved_evidence_annotations_and_metadata_are_copied_frozen_and_serializable():
    annotated_stances = {"A": "SUPPORT"}
    metadata = {"details": {"years": [2020]}}
    item = RetrievedEvidence(
        schema_version="2.0",
        question_id="q",
        doc_id="d",
        rank=1,
        text="text",
        source="source",
        evidence_type=EvidenceType.OTHER,
        retrieval_score=0.9,
        annotated_stances=annotated_stances,
        metadata=metadata,
    )

    annotated_stances["A"] = "CONTRADICT"
    metadata["details"]["years"].append(2021)

    assert item.annotated_stances["A"] is Stance.SUPPORT
    assert item.metadata["details"]["years"] == (2020,)
    with pytest.raises(TypeError, match="immutable"):
        item.annotated_stances["A"] = Stance.CONTRADICT
    with pytest.raises(TypeError, match="immutable"):
        item.metadata["details"] = {}
    with pytest.raises(TypeError, match="immutable"):
        item.metadata["details"]["years"] = (2020, 2021)
    payload = json.loads(json.dumps(asdict(item)))
    assert payload["annotated_stances"] == {"A": "SUPPORT"}
    assert payload["metadata"] == {"details": {"years": [2020]}}
    restored = pickle.loads(pickle.dumps(item))
    assert restored == item
    with pytest.raises(TypeError, match="immutable"):
        restored.annotated_stances["A"] = Stance.CONTRADICT


def test_retrieved_evidence_rejects_invalid_annotated_stance():
    with pytest.raises(ValueError, match="valid stance"):
        evidence("d", EvidenceType.OTHER, {"A": "MAYBE"})


def test_question_prediction_preserves_option_scores():
    result = QuestionPrediction(
        question_id="q",
        predicted_option_label="B",
        option_scores={"A": -0.2, "B": 0.8},
    )
    assert result.predicted_option_label == "B"
    assert result.option_scores == {"A": -0.2, "B": 0.8}


def test_question_prediction_requires_predicted_option_in_scores():
    with pytest.raises(ValueError, match="scored options"):
        QuestionPrediction("q", "C", {"A": 0.1, "B": 0.2})


def test_question_prediction_requires_multiple_option_scores():
    with pytest.raises(ValueError, match="option scores"):
        QuestionPrediction("q", "A", {"A": 1.0})
