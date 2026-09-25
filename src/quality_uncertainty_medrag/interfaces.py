"""Interchangeable boundaries for retrieval, models, scoring, and aggregation."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .models import (
    AggregationResult,
    CandidateClaim,
    MedicalQuestion,
    PipelineResult,
    QualityScore,
    QuestionPrediction,
    RetrievedEvidence,
    ScoredEvidence,
    StancePrediction,
)


class TextGenerationBackend(Protocol):
    """Boundary for a future local Qwen model or hosted API client."""

    def generate(self, prompt: str, *, generation_config: Mapping[str, object] | None = None) -> str: ...


class Retriever(Protocol):
    def retrieve(self, question: MedicalQuestion, *, top_k: int) -> Sequence[RetrievedEvidence]: ...


class EvidenceQualityScorer(Protocol):
    def score(self, question: MedicalQuestion, evidence: RetrievedEvidence) -> QualityScore: ...


class StanceClassifier(Protocol):
    def classify(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: RetrievedEvidence
    ) -> StancePrediction: ...


class EvidenceAggregator(Protocol):
    def aggregate(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: Sequence[ScoredEvidence]
    ) -> AggregationResult: ...


class AnswerGenerator(Protocol):
    """Reserved boundary for a future evidence-conditioned answer model."""

    def generate_answer(
        self, question: MedicalQuestion, claim_results: Sequence[PipelineResult]
    ) -> QuestionPrediction: ...
