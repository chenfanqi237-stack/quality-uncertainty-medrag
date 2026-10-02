"""Plain majority and quality-weighted hard-vote evidence baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .models import (
    AggregationDecision,
    AggregationResult,
    CandidateClaim,
    MedicalQuestion,
    ScoredEvidence,
    Stance,
)
from .uncertainty_aggregation import QualityUncertaintyWeightedAggregator


WEIGHT_TIE_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MajorityVoteAggregationResult(AggregationResult):
    """Claim-level majority result with explicit vote diagnostics.

    The inherited support_weight and contradict_weight fields hold unit-vote
    totals for compatibility with AggregationResult. The count fields expose
    their meaning directly and preserve unresolved predictions separately.
    """

    support_vote_count: int
    contradict_vote_count: int
    unresolved_count: int


class MajorityVoteAggregator:
    """Count hard SUPPORT and CONTRADICT labels, with one vote per item.

    Only the existing stance label is read. Semantic IRRELEVANT labels and
    unresolved labels (None) receive no directional vote. Equal counts,
    including an empty directional pool, return AggregationDecision.ABSTAIN.
    """

    def aggregate(
        self, question: MedicalQuestion, claim: CandidateClaim, evidence: Sequence[ScoredEvidence]
    ) -> MajorityVoteAggregationResult:
        support = 0
        contradict = 0
        irrelevant = 0
        unresolved = 0
        for item in evidence:
            label = item.stance.label
            if label is Stance.SUPPORT:
                support += 1
            elif label is Stance.CONTRADICT:
                contradict += 1
            elif label is Stance.IRRELEVANT:
                irrelevant += 1
            elif label is None:
                unresolved += 1

        if support > contradict:
            decision = AggregationDecision.SUPPORT
        elif contradict > support:
            decision = AggregationDecision.CONTRADICT
        else:
            decision = AggregationDecision.ABSTAIN
        return MajorityVoteAggregationResult(
            claim=claim,
            decision=decision,
            support_weight=float(support),
            contradict_weight=float(contradict),
            irrelevant_count=irrelevant,
            evidence_count=len(evidence),
            support_vote_count=support,
            contradict_vote_count=contradict,
            unresolved_count=unresolved,
        )


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
