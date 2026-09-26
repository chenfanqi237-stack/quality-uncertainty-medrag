"""Baselines for studying conflicting evidence in medical RAG."""

from .models import (
    AggregationDecision,
    CandidateClaim,
    EvidenceType,
    MedicalQuestion,
    QuestionPrediction,
    RetrievedEvidence,
    Stance,
)

__all__ = [
    "AggregationDecision",
    "CandidateClaim",
    "EvidenceType",
    "MedicalQuestion",
    "QuestionPrediction",
    "RetrievedEvidence",
    "Stance",
]
