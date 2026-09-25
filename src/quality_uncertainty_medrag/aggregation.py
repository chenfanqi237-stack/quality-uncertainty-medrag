"""Simple quality-weighted evidence vote baseline."""

from __future__ import annotations

from typing import Sequence

from .models import AggregationResult, CandidateClaim, MedicalQuestion, ScoredEvidence, Stance


class QualityWeightedVoteAggregator:
    """Compare summed quality for supporting and contradicting evidence.

    IRRELEVANT evidence contributes no weight. A tie or no directional
    evidence returns IRRELEVANT as the baseline's abstention state. This is a
    transparent comparison baseline, not an uncertainty-aware method.
    """

    def aggregate(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: Sequence[ScoredEvidence]
    ) -> AggregationResult:
        support = sum(item.quality.value for item in evidence if item.stance.label is Stance.SUPPORT)
        contradict = sum(
            item.quality.value for item in evidence if item.stance.label is Stance.CONTRADICT
        )
        irrelevant = sum(item.stance.label is Stance.IRRELEVANT for item in evidence)
        if support > contradict:
            decision = Stance.SUPPORT
        elif contradict > support:
            decision = Stance.CONTRADICT
        else:
            decision = Stance.IRRELEVANT
        return AggregationResult(
            claim=claim,
            decision=decision,
            support_weight=support,
            contradict_weight=contradict,
            irrelevant_count=irrelevant,
            evidence_count=len(evidence),
        )
