import pytest

from quality_uncertainty_medrag.aggregation import QualityWeightedVoteAggregator
from quality_uncertainty_medrag.models import (
    CandidateClaim,
    EvidenceType,
    MedicalQuestion,
    QualityScore,
    RetrievedEvidence,
    Stance,
    StancePrediction,
)
from quality_uncertainty_medrag.pipeline import BaselinePipeline


class CountingRetriever:
    def __init__(self, evidence):
        self.evidence = tuple(evidence)
        self.calls = 0

    def retrieve(self, question, *, top_k):
        self.calls += 1
        return self.evidence[:top_k]


class CountingQualityScorer:
    def __init__(self):
        self.calls = []

    def score(self, question, evidence):
        self.calls.append(evidence.doc_id)
        return QualityScore(
            value=0.5,
            scorer_name="counting-quality",
            components={"hierarchy": 0.5},
        )


class CountingStanceClassifier:
    def __init__(self):
        self.calls = []

    def classify(self, question, claim, evidence):
        self.calls.append((claim.option_label, evidence.doc_id))
        return StancePrediction(
            probabilities={
                Stance.SUPPORT: 1.0,
                Stance.CONTRADICT: 0.0,
                Stance.IRRELEVANT: 0.0,
            },
            classifier_name="counting-stance",
        )


class CountingAggregator:
    def __init__(self):
        self.calls = []
        self._delegate = QualityWeightedVoteAggregator()

    def aggregate(self, question, claim, evidence):
        self.calls.append(claim.option_label)
        return self._delegate.aggregate(question, claim, evidence)


def make_question():
    return MedicalQuestion(
        question_id="q",
        question="Which option?",
        option_labels=("A", "B", "C", "D"),
        options=("one", "two", "three", "four"),
        answer_index=0,
    )


def make_evidence(doc_id, rank):
    return RetrievedEvidence(
        schema_version="2.0",
        question_id="q",
        doc_id=doc_id,
        rank=rank,
        text=f"Evidence {doc_id}",
        source="synthetic",
        evidence_type=EvidenceType.OTHER,
        retrieval_score=1.0,
    )


def make_counting_pipeline():
    retrieved = (
        make_evidence("doc-second", 2),
        make_evidence("doc-first", 1),
        make_evidence("doc-third", 3),
    )
    retriever = CountingRetriever(retrieved)
    quality_scorer = CountingQualityScorer()
    stance_classifier = CountingStanceClassifier()
    aggregator = CountingAggregator()
    pipeline = BaselinePipeline(
        retriever=retriever,
        quality_scorer=quality_scorer,
        stance_classifier=stance_classifier,
        aggregator=aggregator,
        top_k=3,
    )
    return pipeline, retriever, quality_scorer, stance_classifier, aggregator


def test_run_question_reuses_retrieval_and_quality_for_all_four_claims():
    question = make_question()
    pipeline, retriever, quality_scorer, stance_classifier, aggregator = (
        make_counting_pipeline()
    )

    results = pipeline.run_question(question)

    expected_doc_ids = ("doc-second", "doc-first", "doc-third")
    assert len(results) == 4
    assert tuple(result.claim.option_label for result in results) == ("A", "B", "C", "D")
    assert retriever.calls == 1
    assert quality_scorer.calls == list(expected_doc_ids)
    assert stance_classifier.calls == [
        (option_label, doc_id)
        for option_label in ("A", "B", "C", "D")
        for doc_id in expected_doc_ids
    ]
    assert aggregator.calls == ["A", "B", "C", "D"]
    assert all(
        tuple(item.evidence.doc_id for item in result.evidence) == expected_doc_ids
        for result in results
    )
    for evidence_index in range(len(expected_doc_ids)):
        assert len({id(result.evidence[evidence_index].evidence) for result in results}) == 1
        assert len({id(result.evidence[evidence_index].quality) for result in results}) == 1


@pytest.mark.parametrize(
    ("claim", "message"),
    [
        (CandidateClaim("other", 0, "A", "one"), "does not belong"),
        (CandidateClaim("q", 99, "A", "one"), "does not identify"),
        (CandidateClaim("q", 0, "A", "different"), "does not match"),
    ],
)
def test_run_preserves_claim_validation_before_retrieval(claim, message):
    pipeline, retriever, _, _, _ = make_counting_pipeline()

    with pytest.raises(ValueError, match=message):
        pipeline.run(make_question(), claim)

    assert retriever.calls == 0


def test_run_remains_a_single_claim_compatibility_api():
    question = make_question()
    pipeline, retriever, quality_scorer, stance_classifier, aggregator = (
        make_counting_pipeline()
    )

    result = pipeline.run(question, question.candidate_claims[0])

    expected_doc_ids = ["doc-second", "doc-first", "doc-third"]
    assert result.claim.option_label == "A"
    assert retriever.calls == 1
    assert quality_scorer.calls == expected_doc_ids
    assert stance_classifier.calls == [("A", doc_id) for doc_id in expected_doc_ids]
    assert aggregator.calls == ["A"]
