"""Transparent metadata-only quality baseline."""

from __future__ import annotations

from typing import Mapping

from .models import EvidenceType, MedicalQuestion, QualityScore, RetrievedEvidence


class ConfiguredEvidenceTypeScorer:
    """Map supplied evidence-type metadata to a configured score in [0, 1]."""

    def __init__(self, scores: Mapping[str, float], scorer_name: str) -> None:
        self._scores = {EvidenceType(key): float(value) for key, value in scores.items()}
        missing = set(EvidenceType) - set(self._scores)
        if missing:
            raise ValueError(f"Missing evidence type scores: {sorted(item.value for item in missing)}")
        for value in self._scores.values():
            if not 0.0 <= value <= 1.0:
                raise ValueError("Configured quality scores must be in [0, 1]")
        self._name = scorer_name

    def score(self, question: MedicalQuestion, evidence: RetrievedEvidence) -> QualityScore:
        value = self._scores[evidence.evidence_type]
        return QualityScore(
            value=value,
            scorer_name=self._name,
            components={"hierarchy": value},
            rationale=f"Configured score for evidence_type={evidence.evidence_type.value}",
        )
