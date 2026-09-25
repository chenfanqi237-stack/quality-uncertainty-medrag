"""Baselines for studying conflicting evidence in medical RAG."""

from .models import (
    CandidateClaim,
    EvidenceType,
    MedicalQuestion,
    QuestionPrediction,
    RetrievedEvidence,
    Stance,
)

__all__ = [
    "CandidateClaim",
    "EvidenceType",
    "MedicalQuestion",
    "QuestionPrediction",
    "RetrievedEvidence",
    "Stance",
]
