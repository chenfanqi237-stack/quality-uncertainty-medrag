"""Composition root for the lightweight baseline."""

from __future__ import annotations

from .interfaces import EvidenceAggregator, EvidenceQualityScorer, Retriever, StanceClassifier
from .models import (
    CandidateClaim,
    MedicalQuestion,
    PipelineResult,
    QualityScore,
    RetrievedEvidence,
    ScoredEvidence,
)


_QualityScoredEvidence = tuple[RetrievedEvidence, QualityScore]


class BaselinePipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        quality_scorer: EvidenceQualityScorer,
        stance_classifier: StanceClassifier,
        aggregator: EvidenceAggregator,
        top_k: int,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._retriever = retriever
        self._quality_scorer = quality_scorer
        self._stance_classifier = stance_classifier
        self._aggregator = aggregator
        self._top_k = top_k

    @staticmethod
    def _validate_claim(question: MedicalQuestion, claim: CandidateClaim) -> None:
        if claim.question_id != question.question_id:
            raise ValueError("Candidate claim does not belong to the supplied question")
        try:
            expected_claim = question.candidate_claims[claim.option_index]
        except IndexError as exc:
            raise ValueError("Candidate claim does not identify a supplied answer option") from exc
        if claim != expected_claim:
            raise ValueError("Candidate claim does not match the supplied answer option")

    def _retrieve_and_score(
        self, question: MedicalQuestion
    ) -> tuple[_QualityScoredEvidence, ...]:
        retrieved = tuple(self._retriever.retrieve(question, top_k=self._top_k))
        return tuple(
            (item, self._quality_scorer.score(question, item)) for item in retrieved
        )

    def _run_claim(
        self,
        question: MedicalQuestion,
        claim: CandidateClaim,
        evidence_pool: tuple[_QualityScoredEvidence, ...],
    ) -> PipelineResult:
        scored = tuple(
            ScoredEvidence(
                evidence=item,
                quality=quality,
                stance=self._stance_classifier.classify(question, claim, item),
            )
            for item, quality in evidence_pool
        )
        aggregation = self._aggregator.aggregate(question, claim, scored)
        return PipelineResult(question=question, claim=claim, evidence=scored, aggregation=aggregation)

    def run(self, question: MedicalQuestion, claim: CandidateClaim) -> PipelineResult:
        """Run one claim independently for compatibility with existing callers."""

        self._validate_claim(question, claim)
        evidence_pool = self._retrieve_and_score(question)
        return self._run_claim(question, claim, evidence_pool)

    def run_question(self, question: MedicalQuestion) -> tuple[PipelineResult, ...]:
        """Run every candidate claim using one retrieved and quality-scored pool."""

        claims = question.candidate_claims
        for claim in claims:
            self._validate_claim(question, claim)
        evidence_pool = self._retrieve_and_score(question)
        return tuple(self._run_claim(question, claim, evidence_pool) for claim in claims)
