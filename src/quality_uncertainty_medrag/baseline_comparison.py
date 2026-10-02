"""Controlled synthetic comparison of the two existing hard-vote baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .aggregation import MajorityVoteAggregator, QualityWeightedVoteAggregator
from .models import (
    AggregationDecision,
    EvidenceType,
    MedicalQuestion,
    QualityScore,
    RetrievedEvidence,
    ScoredEvidence,
    Stance,
    StancePrediction,
)


# ID, description, SUPPORT qualities, CONTRADICT qualities, IRRELEVANT
# qualities, expected majority decision, expected quality-weighted decision.
_SCENARIOS = (
    (
        "scenario_1", "Three low-quality SUPPORT items versus one high-quality CONTRADICT item",
        (0.1, 0.1, 0.1), (0.9,), (),
        AggregationDecision.SUPPORT, AggregationDecision.CONTRADICT,
    ),
    (
        "scenario_2", "One high-quality SUPPORT item versus three low-quality CONTRADICT items",
        (0.9,), (0.1, 0.1, 0.1), (),
        AggregationDecision.CONTRADICT, AggregationDecision.SUPPORT,
    ),
    (
        "scenario_3", "Equal directional counts and equal total quality",
        (0.5, 0.5), (0.5, 0.5), (),
        AggregationDecision.ABSTAIN, AggregationDecision.ABSTAIN,
    ),
    (
        "scenario_4", "All evidence is semantically IRRELEVANT",
        (), (), (0.1, 0.5, 0.9),
        AggregationDecision.ABSTAIN, AggregationDecision.ABSTAIN,
    ),
    (
        "scenario_5a", "Fixed two-to-one SUPPORT majority with higher CONTRADICT quality total",
        (0.1, 0.1), (0.9,), (),
        AggregationDecision.SUPPORT, AggregationDecision.CONTRADICT,
    ),
    (
        "scenario_5b", "Same items and stances as 5a, with higher SUPPORT quality total",
        (0.45, 0.45), (0.1,), (),
        AggregationDecision.SUPPORT, AggregationDecision.SUPPORT,
    ),
)

_CSV_FIELDS = (
    "scenario_id", "support_count", "contradict_count",
    "support_quality_sum", "contradict_quality_sum",
    "majority_vote_decision", "quality_weighted_vote_decision",
    "evidence_count", "irrelevant_count", "unresolved_count",
)


def _make_evidence(
    scenario_id: str,
    question: MedicalQuestion,
    support_qualities: tuple[float, ...],
    contradict_qualities: tuple[float, ...],
    irrelevant_qualities: tuple[float, ...],
) -> tuple[ScoredEvidence, ...]:
    # Pair 5a/5b deliberately keeps IDs, labels and ordering identical.
    family_id = "scenario_5" if scenario_id in {"scenario_5a", "scenario_5b"} else scenario_id
    inputs = [
        (label, quality)
        for label, qualities in (
            (Stance.SUPPORT, support_qualities),
            (Stance.CONTRADICT, contradict_qualities),
            (Stance.IRRELEVANT, irrelevant_qualities),
        )
        for quality in qualities
    ]
    return tuple(
        ScoredEvidence(
            evidence=RetrievedEvidence(
                schema_version="2.0", question_id=question.question_id,
                doc_id=f"{family_id}-doc-{rank:02d}", rank=rank,
                text=f"Controlled synthetic fixture with hard stance {label.value}.",
                source="controlled-synthetic-baseline-comparison",
                evidence_type=EvidenceType.OTHER, retrieval_score=0.0,
                metadata={"synthetic": True, "scenario_family": family_id},
            ),
            quality=QualityScore(
                value=quality, scorer_name="controlled-synthetic-input",
                components={"synthetic_input": quality},
                rationale="Quality assigned explicitly for this controlled experiment.",
            ),
            stance=StancePrediction(
                probabilities={stance: float(stance is label) for stance in Stance},
                classifier_name="controlled-synthetic-hard-label",
            ),
        )
        for rank, (label, quality) in enumerate(inputs, start=1)
    )


def run_comparison() -> dict[str, Any]:
    """Run five fixed scenarios (six rows) without randomness or model calls."""

    question = MedicalQuestion(
        question_id="controlled-synthetic-question",
        question="Does this synthetic evidence pool support the fixture claim?",
        option_labels=("A", "B"), options=("Fixture claim", "Alternative fixture claim"),
        # Required fixture field; neither aggregator uses it or selects an answer.
        answer_index=0, metadata={"synthetic": True},
    )
    claim = question.candidate_claims[0]
    majority = MajorityVoteAggregator()
    weighted = QualityWeightedVoteAggregator()
    rows = []
    for scenario_id, description, supports, contradicts, irrelevant, expected_majority, expected_weighted in _SCENARIOS:
        evidence = _make_evidence(scenario_id, question, supports, contradicts, irrelevant)
        majority_result = majority.aggregate(question, claim, evidence)
        weighted_result = weighted.aggregate(question, claim, evidence)
        if (majority_result.decision, weighted_result.decision) != (expected_majority, expected_weighted):
            raise RuntimeError(f"Unexpected baseline decisions in {scenario_id}")
        rows.append({
            "scenario_id": scenario_id, "description": description,
            "support_count": majority_result.support_vote_count,
            "contradict_count": majority_result.contradict_vote_count,
            "support_quality_sum": weighted_result.support_weight,
            "contradict_quality_sum": weighted_result.contradict_weight,
            "majority_vote_decision": majority_result.decision.value,
            "quality_weighted_vote_decision": weighted_result.decision.value,
            "evidence_count": majority_result.evidence_count,
            "irrelevant_count": majority_result.irrelevant_count,
            "unresolved_count": majority_result.unresolved_count,
            "input_evidence": [
                {"doc_id": item.evidence.doc_id, "stance": item.stance.label.value, "quality": item.quality.value}
                for item in evidence
            ],
        })
    return {
        "experiment": "controlled-synthetic-baseline-comparison",
        "quality_source": "assigned synthetic inputs",
        "randomness": "none",
        "scenario_group_count": 5,
        "comparison_row_count": len(rows),
        "all_expected_decisions_match": True,
        "scenarios": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_comparison"))
    args = parser.parse_args(argv)
    report = run_comparison()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["scenarios"])
    for row in report["scenarios"]:
        print(
            f"{row['scenario_id']}: support={row['support_count']} contradict={row['contradict_count']} "
            f"support_quality={row['support_quality_sum']:.6g} contradict_quality={row['contradict_quality_sum']:.6g} "
            f"majority={row['majority_vote_decision']} weighted={row['quality_weighted_vote_decision']}"
        )
    print(f"All expected decisions match. Reports saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
