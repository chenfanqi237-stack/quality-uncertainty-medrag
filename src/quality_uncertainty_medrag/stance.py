"""Fixture classifier for tests; real classifiers implement the same interface."""

from __future__ import annotations

from .models import CandidateClaim, MedicalQuestion, RetrievedEvidence, Stance, StancePrediction


class AnnotatedStanceClassifier:
    """Read a supplied annotation without calling a model.

    This is only for smoke tests and baseline plumbing. Production inputs can
    omit annotations when a Qwen-backed or API-backed classifier is added.
    """

    name = "annotated-fixture-baseline"

    def classify(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: RetrievedEvidence
    ) -> StancePrediction:
        if claim.option_label not in evidence.annotated_stances:
            raise ValueError(
                f"Evidence {evidence.doc_id} has no annotated stance for option {claim.option_label}"
            )
        label = evidence.annotated_stances[claim.option_label]
        return StancePrediction(
            probabilities={stance: float(stance is label) for stance in Stance},
            classifier_name=self.name,
            rationale="Copied from the optional annotated_stances fixture field",
        )
