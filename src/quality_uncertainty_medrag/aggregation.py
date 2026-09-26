"""Simple quality-weighted evidence vote baseline."""

from __future__ import annotations

import math
from typing import Sequence

from .models import (
    AggregationDecision,
    AggregationResult,
    CandidateClaim,
    MedicalQuestion,
    ScoredEvidence,
    Stance,
)


WEIGHT_TIE_ABS_TOLERANCE = 1e-12


class QualityWeightedVoteAggregator:
    """Compare summed quality for supporting and contradicting evidence.

    IRRELEVANT evidence and unresolved stance ties contribute no weight. A tie
    or no directional evidence returns AggregationDecision.ABSTAIN. This is a
    transparent comparison baseline, not an uncertainty-aware method.
    """

    def aggregate(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: Sequence[ScoredEvidence]
    ) -> AggregationResult:
        support_values: list[float] = []
        contradict_values: list[float] = []
        irrelevant = 0
        for item in evidence:
            label = item.stance.label
            if label is None:
                continue
            if label is Stance.SUPPORT:
                support_values.append(item.quality.value)
            elif label is Stance.CONTRADICT:
                contradict_values.append(item.quality.value)
            elif label is Stance.IRRELEVANT:
                irrelevant += 1

        support = math.fsum(support_values)
        contradict = math.fsum(contradict_values)
        if math.isclose(
            support,
            contradict,
            rel_tol=0.0,
            abs_tol=WEIGHT_TIE_ABS_TOLERANCE,
        ):
            decision = AggregationDecision.ABSTAIN
        elif support > contradict:
            decision = AggregationDecision.SUPPORT
        else:
            decision = AggregationDecision.CONTRADICT
        return AggregationResult(
            claim=claim,
            decision=decision,
            support_weight=support,
            contradict_weight=contradict,
            irrelevant_count=irrelevant,
            evidence_count=len(evidence),
        )
