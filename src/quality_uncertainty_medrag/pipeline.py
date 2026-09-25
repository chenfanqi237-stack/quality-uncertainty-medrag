"""Composition root for the lightweight baseline."""

from __future__ import annotations

from .interfaces import EvidenceAggregator, EvidenceQualityScorer, Retriever, StanceClassifier
from .models import CandidateClaim, MedicalQuestion, PipelineResult, ScoredEvidence


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

    def run(self, question: MedicalQuestion, claim: CandidateClaim) -> PipelineResult:
        if claim.question_id != question.question_id:
            raise ValueError("Candidate claim does not belong to the supplied question")
        try:
            expected_claim = question.candidate_claims[claim.option_index]
        except IndexError as exc:
            raise ValueError("Candidate claim does not identify a supplied answer option") from exc
        if claim != expected_claim:
            raise ValueError("Candidate claim does not match the supplied answer option")
        retrieved = self._retriever.retrieve(question, top_k=self._top_k)
        scored = tuple(
            ScoredEvidence(
                evidence=item,
                quality=self._quality_scorer.score(question, item),
                stance=self._stance_classifier.classify(question, claim, item),
            )
            for item in retrieved
        )
        aggregation = self._aggregator.aggregate(question, claim, scored)
        return PipelineResult(question=question, claim=claim, evidence=scored, aggregation=aggregation)
