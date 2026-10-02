"""Deterministic directional conflict grid for the existing vote baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


_COUNT_PAIRS = ((9, 1), (7, 3), (5, 5), (3, 7), (1, 9))
# Condition name, uniform SUPPORT quality, uniform CONTRADICT quality.
_QUALITY_CONDITIONS = (
    ("equal", 0.5, 0.5),
    ("support_small_gap", 0.55, 0.45),
    ("contradict_small_gap", 0.45, 0.55),
    ("support_large_gap", 0.9, 0.1),
    ("contradict_large_gap", 0.1, 0.9),
    ("support_extreme_gap", 0.95, 0.05),
    ("contradict_extreme_gap", 0.05, 0.95),
)
_CSV_FIELDS = (
    "scenario_id", "support_count", "contradict_count",
    "support_quality_mean", "contradict_quality_mean",
    "support_quality_sum", "contradict_quality_sum", "conflict_ratio",
    "quality_condition", "quality_gap_direction", "quality_gap_magnitude",
    "majority_vote_decision", "quality_weighted_vote_decision", "baselines_disagree",
    "evidence_count", "irrelevant_count", "unresolved_count",
)


def _quality_direction(support_quality: float, contradict_quality: float) -> str:
    if support_quality > contradict_quality:
        return "support_higher"
    if contradict_quality > support_quality:
        return "contradict_higher"
    return "equal"


def _make_evidence(
    scenario_id: str, question: MedicalQuestion, support_count: int, contradict_count: int,
    support_quality: float, contradict_quality: float,
) -> tuple[ScoredEvidence, ...]:
    inputs = (
        [(Stance.SUPPORT, support_quality)] * support_count
        + [(Stance.CONTRADICT, contradict_quality)] * contradict_count
    )
    return tuple(
        ScoredEvidence(
            evidence=RetrievedEvidence(
                schema_version="2.0", question_id=question.question_id,
                doc_id=f"{scenario_id}-doc-{rank:02d}", rank=rank,
                text=f"Directional synthetic stress fixture: {label.value}.",
                source="controlled-conflict-stress-test", evidence_type=EvidenceType.OTHER,
                retrieval_score=0.0, metadata={"synthetic": True, "scenario_id": scenario_id},
            ),
            quality=QualityScore(
                value=quality, scorer_name="controlled-synthetic-input",
                components={"synthetic_input": quality},
                rationale="Uniform quality explicitly assigned to this directional group.",
            ),
            stance=StancePrediction(
                probabilities={stance: float(stance is label) for stance in Stance},
                classifier_name="controlled-synthetic-hard-label",
            ),
        )
        for rank, (label, quality) in enumerate(inputs, start=1)
    )


def generate_scenarios() -> list[dict[str, Any]]:
    """Evaluate the fixed 5-by-7 grid using both unmodified aggregators.

    conflict_ratio is a SUPPORT:CONTRADICT count composition, e.g. "9:1".
    Means are the assigned uniform group qualities. Sums and decisions come
    directly from the existing aggregators. No randomness or external data
    is used, so repeated generation produces identical records.
    """

    question = MedicalQuestion(
        question_id="synthetic-conflict-stress-question",
        question="Does this controlled synthetic pool support the fixture claim?",
        option_labels=("A", "B"), options=("Fixture claim", "Alternative fixture claim"),
        # Schema-required synthetic placeholder; no gold answers are read.
        answer_index=0, metadata={"synthetic": True},
    )
    claim = question.candidate_claims[0]
    majority = MajorityVoteAggregator()
    weighted = QualityWeightedVoteAggregator()
    rows = []
    for support_count, contradict_count in _COUNT_PAIRS:
        for condition, support_quality, contradict_quality in _QUALITY_CONDITIONS:
            scenario_id = f"stress-s{support_count:02d}-c{contradict_count:02d}-{condition}"
            evidence = _make_evidence(
                scenario_id, question, support_count, contradict_count,
                support_quality, contradict_quality,
            )
            majority_result = majority.aggregate(question, claim, evidence)
            weighted_result = weighted.aggregate(question, claim, evidence)
            rows.append({
                "scenario_id": scenario_id,
                "support_count": majority_result.support_vote_count,
                "contradict_count": majority_result.contradict_vote_count,
                "support_quality_mean": support_quality,
                "contradict_quality_mean": contradict_quality,
                "support_quality_sum": weighted_result.support_weight,
                "contradict_quality_sum": weighted_result.contradict_weight,
                "conflict_ratio": f"{support_count}:{contradict_count}",
                "quality_condition": condition,
                "quality_gap_direction": _quality_direction(support_quality, contradict_quality),
                "quality_gap_magnitude": abs(support_quality - contradict_quality),
                "majority_vote_decision": majority_result.decision.value,
                "quality_weighted_vote_decision": weighted_result.decision.value,
                "baselines_disagree": majority_result.decision != weighted_result.decision,
                "evidence_count": majority_result.evidence_count,
                "irrelevant_count": majority_result.irrelevant_count,
                "unresolved_count": majority_result.unresolved_count,
            })
    return rows


def _summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    disagreements = sum(row["baselines_disagree"] for row in rows)
    reversals = sum(
        row["baselines_disagree"]
        and row["majority_vote_decision"] != AggregationDecision.ABSTAIN.value
        and row["quality_weighted_vote_decision"] != AggregationDecision.ABSTAIN.value
        for row in rows
    )
    decision_counts = {}
    for baseline, field in (
        ("majority_vote", "majority_vote_decision"),
        ("quality_weighted_vote", "quality_weighted_vote_decision"),
    ):
        counts = Counter(row[field] for row in rows)
        decision_counts[baseline] = {decision.value: counts[decision.value] for decision in AggregationDecision}
    return {
        "total_scenarios": total,
        "disagreement_count": disagreements,
        "disagreement_percentage": 100.0 * disagreements / total if total else None,
        "directional_reversal_count": reversals,
        "abstention_difference_count": disagreements - reversals,
        "decision_counts": decision_counts,
    }


def summarize_scenarios(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize baseline differences, including directional/ABSTAIN differences.

    These are descriptive comparisons, with no claim truth or accuracy target.
    Empty input has no disagreement percentage rather than a misleading zero.
    """

    by_ratio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ratio[row["conflict_ratio"]].append(row)
        by_direction[row["quality_gap_direction"]].append(row)
    return {
        "experiment": "controlled-directional-conflict-stress-test",
        "randomness": "none",
        "quality_source": "assigned uniform synthetic inputs",
        "conflict_ratio_definition": "SUPPORT count:CONTRADICT count",
        "disagreement_definition": "Any different aggregate decisions, including ABSTAIN versus a direction",
        **_summarize_group(rows),
        "by_conflict_ratio": {key: _summarize_group(group) for key, group in by_ratio.items()},
        "by_quality_gap_direction": {key: _summarize_group(group) for key, group in by_direction.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--include-uncertainty", action="store_true",
        help="Compare all three aggregators across fixed stance-probability conditions.",
    )
    args = parser.parse_args(argv)
    if args.include_uncertainty:
        from .uncertainty_stress_test import main as uncertainty_main

        return uncertainty_main([
            "--output-dir", str(args.output_dir or Path("outputs/uncertainty_stress_test")),
        ])
    if args.output_dir is None:
        args.output_dir = Path("outputs/conflict_stress_test")
    rows = generate_scenarios()
    summary = summarize_scenarios(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scenarios.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        f"scenarios={summary['total_scenarios']} disagreements={summary['disagreement_count']} "
        f"disagreement_percentage={summary['disagreement_percentage']:.2f}% "
        f"directional_reversals={summary['directional_reversal_count']} "
        f"abstention_differences={summary['abstention_difference_count']}"
    )
    print(f"Reports saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
