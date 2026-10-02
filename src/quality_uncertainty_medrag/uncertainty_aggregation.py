"""Experimental quality/entropy-weighted aggregation for synthetic research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import (
    PROBABILITY_TIE_ABS_TOLERANCE,
    AggregationDecision,
    AggregationResult,
    CandidateClaim,
    MedicalQuestion,
    ScoredEvidence,
    Stance,
)


_LOG_THREE = math.log(3.0)
_UNIFORM_PROBABILITY = 1.0 / 3.0


@dataclass(frozen=True)
class EvidenceUncertaintyDiagnostic:
    """Copied scalar diagnostics for one input evidence item, in input order."""

    doc_id: str
    quality: float
    directional_score: float
    normalized_entropy: float
    effective_weight: float


@dataclass(frozen=True)
class QualityUncertaintyAggregationResult(AggregationResult):
    """Claim-level experimental result compatible with EvidenceAggregator.

    aggregate_score is None when total_effective_weight is zero. The inherited
    support_weight and contradict_weight store summed effective probability
    mass, sum(w * p_support) and sum(w * p_contradict), respectively. Their
    separately rounded difference is not used to decide the score's sign.
    """

    aggregate_score: float | None
    total_effective_weight: float
    per_evidence: tuple[EvidenceUncertaintyDiagnostic, ...]
    unresolved_count: int


def _normalized_entropy(probabilities: Mapping[Stance, float]) -> float:
    values = tuple(probabilities[stance] for stance in Stance)
    # Enforce the exact mathematical identity, without a near-uniform tolerance.
    if values == (_UNIFORM_PROBABILITY,) * 3:
        return 1.0
    entropy = math.fsum(-value * math.log(value) for value in values if value > 0.0)
    # The model accepts probability sums approximately equal to one. Protect
    # the theoretical bounds against those small deviations and roundoff,
    # without renormalizing or changing the input directional probabilities.
    return min(1.0, max(0.0, entropy / _LOG_THREE))


class QualityUncertaintyWeightedAggregator:
    """Prototype of exactly the specified probability/quality/entropy method.

    Every item contributes w=q*(1-H/log(3)) and direction p_support-p_contradict.
    Effective weights include IRRELEVANT probability distributions, even when
    their direction is zero. Hard labels never filter or weight evidence.
    Diagnostic irrelevant/unresolved counts follow existing stance tie
    semantics; that tolerance has no role in the aggregate decision.
    """

    def aggregate(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: Sequence[ScoredEvidence]
    ) -> QualityUncertaintyAggregationResult:
        diagnostics = []
        support_values = []
        contradict_values = []
        irrelevant_count = 0
        unresolved_count = 0
        for item in evidence:
            probabilities = item.stance.probabilities
            quality = item.quality.value
            direction = probabilities[Stance.SUPPORT] - probabilities[Stance.CONTRADICT]
            uncertainty = _normalized_entropy(probabilities)
            weight = quality * (1.0 - uncertainty)
            diagnostics.append(EvidenceUncertaintyDiagnostic(
                doc_id=item.evidence.doc_id, quality=quality, directional_score=direction,
                normalized_entropy=uncertainty, effective_weight=weight,
            ))
            support_values.append(weight * probabilities[Stance.SUPPORT])
            contradict_values.append(weight * probabilities[Stance.CONTRADICT])

            # Keep inherited count diagnostics consistent with StancePrediction
            # without accessing its hard-label properties or using them in S.
            highest = max(probabilities.values())
            winners = tuple(
                stance for stance in Stance
                if math.isclose(
                    probabilities[stance], highest, rel_tol=0.0,
                    abs_tol=PROBABILITY_TIE_ABS_TOLERANCE,
                )
            )
            if len(winners) > 1:
                unresolved_count += 1
            elif winners[0] is Stance.IRRELEVANT:
                irrelevant_count += 1

        total_weight = math.fsum(item.effective_weight for item in diagnostics)
        if total_weight == 0.0:
            score = None
            decision = AggregationDecision.ABSTAIN
        else:
            # Algebraically sum(w*s)/sum(w), with scaling before multiplication
            # to avoid prematurely underflowing tiny w*s products to zero.
            score = math.fsum(
                (item.effective_weight / total_weight) * item.directional_score
                for item in diagnostics
            )
            if score > 0.0:
                decision = AggregationDecision.SUPPORT
            elif score < 0.0:
                decision = AggregationDecision.CONTRADICT
            else:
                decision = AggregationDecision.ABSTAIN
        return QualityUncertaintyAggregationResult(
            claim=claim, decision=decision,
            support_weight=math.fsum(support_values), contradict_weight=math.fsum(contradict_values),
            irrelevant_count=irrelevant_count, evidence_count=len(evidence),
            aggregate_score=score, total_effective_weight=total_weight,
            per_evidence=tuple(diagnostics), unresolved_count=unresolved_count,
        )
