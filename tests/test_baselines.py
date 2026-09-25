import math

import pytest

from quality_uncertainty_medrag.aggregation import QualityWeightedVoteAggregator
from quality_uncertainty_medrag.models import (
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
    ("items", "expected"),
    [
        ([scored("a", 0.8, Stance.SUPPORT), scored("b", 0.3, Stance.CONTRADICT)], Stance.SUPPORT),
        ([scored("a", 0.2, Stance.SUPPORT), scored("b", 0.7, Stance.CONTRADICT)], Stance.CONTRADICT),
        ([scored("a", 0.5, Stance.SUPPORT), scored("b", 0.5, Stance.CONTRADICT)], Stance.IRRELEVANT),
        ([scored("a", 0.9, Stance.IRRELEVANT)], Stance.IRRELEVANT),
    ],
)
def test_quality_weighted_vote_remains_claim_level(items, expected):
    result = QualityWeightedVoteAggregator().aggregate(QUESTION, CLAIM, items)
    assert result.claim == CLAIM
    assert result.decision is expected


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
