"""Deterministic conflict/quality/confidence comparisons of three aggregators.

These synthetic inputs measure behavioral sensitivity. They have no independent
claim-truth annotations and do not measure correctness or comparative accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .aggregation import MajorityVoteAggregator, QualityWeightedVoteAggregator
from .conflict_stress_test import _make_evidence, generate_scenarios as generate_conflict_scenarios
from .models import AggregationDecision, MedicalQuestion, Stance, StancePrediction
from .uncertainty_aggregation import QualityUncertaintyWeightedAggregator


# SUPPORT-oriented triples; CONTRADICT inputs mirror the first two entries.
# Insertion order determines the stable ordering of generated confidence pairs.
CONFIDENCE_TEMPLATES = {
    "CONFIDENT": (0.90, 0.05, 0.05),
    "MODERATE": (0.70, 0.20, 0.10),
    "UNCERTAIN": (0.45, 0.40, 0.15),
}
_CONFIDENCE_RANK = {"CONFIDENT": 2, "MODERATE": 1, "UNCERTAIN": 0}
_CSV_FIELDS = (
    "scenario_id", "support_count", "contradict_count",
    "support_quality_mean", "contradict_quality_mean",
    "support_quality_sum", "contradict_quality_sum", "conflict_ratio",
    "quality_condition", "quality_gap_direction", "quality_gap_magnitude",
    "support_confidence_condition", "contradict_confidence_condition",
    "confidence_asymmetry", "majority_confidence_relation",
    "quality_confidence_relation", "scenario_families",
    "majority_vote_decision", "quality_weighted_vote_decision",
    "quality_uncertainty_weighted_decision", "quality_uncertainty_aggregate_score",
    "total_effective_weight", "majority_vs_quality_weighted_disagree",
    "quality_weighted_vs_quality_uncertainty_disagree", "all_three_agree",
    "evidence_count", "irrelevant_count", "unresolved_count",
)
_COMPARISON_FLAGS = (
    ("majority_vs_quality_weighted", "majority_vs_quality_weighted_disagree"),
    ("quality_weighted_vs_quality_uncertainty", "quality_weighted_vs_quality_uncertainty_disagree"),
)
_DECISION_FIELDS = (
    ("majority_vote", "majority_vote_decision"),
    ("quality_weighted_vote", "quality_weighted_vote_decision"),
    ("quality_uncertainty_weighted", "quality_uncertainty_weighted_decision"),
)


def _confidence_relations(
    base: dict[str, Any], support_condition: str, contradict_condition: str,
) -> dict[str, Any]:
    support_rank = _CONFIDENCE_RANK[support_condition]
    contradict_rank = _CONFIDENCE_RANK[contradict_condition]
    if support_rank == contradict_rank:
        asymmetry = "equal_confidence"
    elif support_rank > contradict_rank:
        asymmetry = "support_more_confident"
    else:
        asymmetry = "contradict_more_confident"

    support_majority = base["support_count"] > base["contradict_count"]
    contradict_majority = base["contradict_count"] > base["support_count"]
    if not support_majority and not contradict_majority:
        majority_relation = "no_majority"
    elif support_rank == contradict_rank:
        majority_relation = "equal_confidence"
    elif support_majority == (support_rank > contradict_rank):
        majority_relation = "majority_more_confident"
    else:
        majority_relation = "minority_more_confident"

    support_higher_quality = base["support_quality_mean"] > base["contradict_quality_mean"]
    contradict_higher_quality = base["contradict_quality_mean"] > base["support_quality_mean"]
    if not support_higher_quality and not contradict_higher_quality:
        quality_relation = "no_quality_gap"
    elif support_rank == contradict_rank:
        quality_relation = "equal_confidence"
    elif support_higher_quality == (support_rank > contradict_rank):
        quality_relation = "higher_quality_more_confident"
    else:
        quality_relation = "lower_quality_more_confident"

    families = []
    if support_condition == contradict_condition:
        families.append("equal_confidence")
    extreme_pair = {support_condition, contradict_condition} == {"CONFIDENT", "UNCERTAIN"}
    if extreme_pair and majority_relation == "minority_more_confident":
        families.append("majority_uncertain_minority_confident")
    if extreme_pair and majority_relation == "majority_more_confident":
        families.append("majority_confident_minority_uncertain")
    if extreme_pair and quality_relation == "lower_quality_more_confident":
        families.append("higher_quality_uncertain_lower_quality_confident")
    if extreme_pair and quality_relation == "higher_quality_more_confident":
        families.append("higher_quality_confident_lower_quality_uncertain")
    return {
        "confidence_asymmetry": asymmetry,
        "majority_confidence_relation": majority_relation,
        "quality_confidence_relation": quality_relation,
        "scenario_families": tuple(families),
    }


def generate_scenarios() -> list[dict[str, Any]]:
    """Expand the existing 35-row grid by every ordered confidence-template pair.

    Each of the 315 pools is passed unchanged to all three existing aggregators.
    SUPPORT and CONTRADICT are input directions, not truth labels. The two hard-vote
    baselines retain their hard-label inputs across all confidence conditions.
    """

    question = MedicalQuestion(
        question_id="synthetic-uncertainty-stress-question",
        question="Does this controlled synthetic pool support the fixture claim?",
        option_labels=("A", "B"), options=("Fixture claim", "Alternative fixture claim"),
        # Required synthetic schema placeholder; never used to compute outcomes.
        answer_index=0, metadata={"synthetic": True},
    )
    claim = question.candidate_claims[0]
    majority = MajorityVoteAggregator()
    quality_weighted = QualityWeightedVoteAggregator()
    quality_uncertainty = QualityUncertaintyWeightedAggregator()
    rows = []
    for base in generate_conflict_scenarios():
        for support_condition, support_probabilities in CONFIDENCE_TEMPLATES.items():
            for contradict_condition, contradict_template in CONFIDENCE_TEMPLATES.items():
                scenario_id = (
                    f"{base['scenario_id']}-s-{support_condition.lower()}"
                    f"-c-{contradict_condition.lower()}"
                )
                one_hot_pool = _make_evidence(
                    scenario_id, question, base["support_count"], base["contradict_count"],
                    base["support_quality_mean"], base["contradict_quality_mean"],
                )
                contradict_probabilities = (
                    contradict_template[1], contradict_template[0], contradict_template[2],
                )
                pool = tuple(
                    replace(item, stance=StancePrediction(
                        probabilities=dict(zip(
                            (Stance.SUPPORT, Stance.CONTRADICT, Stance.IRRELEVANT),
                            support_probabilities if index < base["support_count"]
                            else contradict_probabilities,
                        )),
                        classifier_name="controlled-synthetic-probability-template",
                    ))
                    for index, item in enumerate(one_hot_pool)
                )
                majority_result = majority.aggregate(question, claim, pool)
                quality_result = quality_weighted.aggregate(question, claim, pool)
                uncertainty_result = quality_uncertainty.aggregate(question, claim, pool)
                rows.append({
                    "scenario_id": scenario_id,
                    "support_count": majority_result.support_vote_count,
                    "contradict_count": majority_result.contradict_vote_count,
                    "support_quality_mean": base["support_quality_mean"],
                    "contradict_quality_mean": base["contradict_quality_mean"],
                    "support_quality_sum": quality_result.support_weight,
                    "contradict_quality_sum": quality_result.contradict_weight,
                    "conflict_ratio": base["conflict_ratio"],
                    "quality_condition": base["quality_condition"],
                    "quality_gap_direction": base["quality_gap_direction"],
                    "quality_gap_magnitude": base["quality_gap_magnitude"],
                    "support_confidence_condition": support_condition,
                    "contradict_confidence_condition": contradict_condition,
                    **_confidence_relations(base, support_condition, contradict_condition),
                    "majority_vote_decision": majority_result.decision.value,
                    "quality_weighted_vote_decision": quality_result.decision.value,
                    "quality_uncertainty_weighted_decision": uncertainty_result.decision.value,
                    "quality_uncertainty_aggregate_score": uncertainty_result.aggregate_score,
                    "total_effective_weight": uncertainty_result.total_effective_weight,
                    "majority_vs_quality_weighted_disagree": majority_result.decision != quality_result.decision,
                    "quality_weighted_vs_quality_uncertainty_disagree": quality_result.decision != uncertainty_result.decision,
                    "all_three_agree": majority_result.decision == quality_result.decision == uncertainty_result.decision,
                    "evidence_count": majority_result.evidence_count,
                    "irrelevant_count": majority_result.irrelevant_count,
                    "unresolved_count": majority_result.unresolved_count,
                })
    return rows


def _summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    summary: dict[str, Any] = {"total_scenarios": total}
    for name, flag in _COMPARISON_FLAGS:
        count = sum(row[flag] for row in rows)
        summary[f"{name}_disagreement_count"] = count
        summary[f"{name}_disagreement_percentage"] = 100.0 * count / total if total else None
    agreement_count = sum(row["all_three_agree"] for row in rows)
    summary["all_three_agree_count"] = agreement_count
    summary["all_three_agree_percentage"] = 100.0 * agreement_count / total if total else None
    decision_counts = {}
    for name, field in _DECISION_FIELDS:
        counts = Counter(row[field] for row in rows)
        decision_counts[name] = {
            decision.value: counts[decision.value] for decision in AggregationDecision
        }
    summary["decision_counts"] = decision_counts
    return summary


def summarize_scenarios(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize decision differences without inferring correctness or accuracy.

    Scenario-family groups overlap. The other groupings each partition the input;
    empty input has undefined percentages represented as JSON null.
    """

    fields = (
        "conflict_ratio", "quality_gap_direction", "confidence_asymmetry",
        "majority_confidence_relation", "quality_confidence_relation",
    )
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: defaultdict(list) for field in fields
    }
    confidence_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for field in fields:
            groups[field][row[field]].append(row)
        pair = f"{row['support_confidence_condition']}:{row['contradict_confidence_condition']}"
        confidence_pairs[pair].append(row)
        for family in row["scenario_families"]:
            families[family].append(row)
    return {
        "experiment": "controlled-conflict-quality-confidence-stress-test",
        "interpretation": "Behavioral sensitivity only; no independent ground-truth correctness labels or accuracy comparison.",
        "randomness": "none",
        "quality_source": "Existing conflict stress-test assigned uniform synthetic inputs",
        "conflict_ratio_definition": "SUPPORT count:CONTRADICT count",
        "confidence_pair_definition": "SUPPORT condition:CONTRADICT condition",
        "confidence_templates": {
            name: dict(zip(("SUPPORT", "CONTRADICT", "IRRELEVANT"), values))
            for name, values in CONFIDENCE_TEMPLATES.items()
        },
        "contradict_template_definition": "Mirror SUPPORT and CONTRADICT probabilities; preserve IRRELEVANT probability",
        "grid_definition": "Existing 5 conflict ratios x 7 quality conditions x 3 SUPPORT confidence conditions x 3 CONTRADICT confidence conditions = 315 scenarios",
        "ordering": "Existing conflict grid order, then SUPPORT and CONTRADICT conditions in CONFIDENT, MODERATE, UNCERTAIN order",
        "disagreement_definition": "Any different aggregate decisions, including ABSTAIN versus a direction",
        "numerical_semantics": "Scores and decisions are returned unchanged by the existing aggregators. Quality-weighted hard vote uses its existing absolute tie tolerance of 1e-12; uncertainty-weighted vote uses exact score signs. Floating-point residuals near mathematically balanced inputs can therefore produce directional uncertainty decisions and are not rounded or adjusted.",
        "scenario_family_definition": "Overlapping tags; asymmetric named families require CONFIDENT versus UNCERTAIN, with a strict count majority or strict quality gap as appropriate",
        **_summarize_group(rows),
        **{
            f"by_{field}": {name: _summarize_group(group) for name, group in mapping.items()}
            for field, mapping in groups.items()
        },
        "by_confidence_pair": {
            name: _summarize_group(group) for name, group in confidence_pairs.items()
        },
        "by_scenario_family": {
            name: _summarize_group(group) for name, group in families.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/uncertainty_stress_test"))
    args = parser.parse_args(argv)
    rows = generate_scenarios()
    summary = summarize_scenarios(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scenarios.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({
            **row, "scenario_families": "|".join(row["scenario_families"]),
        } for row in rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        f"scenarios={summary['total_scenarios']} "
        f"majority_vs_quality_weighted_disagreements={summary['majority_vs_quality_weighted_disagreement_count']} "
        f"quality_weighted_vs_quality_uncertainty_disagreements={summary['quality_weighted_vs_quality_uncertainty_disagreement_count']} "
        f"all_three_agree={summary['all_three_agree_count']}"
    )
    print("Behavioral sensitivity only; no independent correctness labels or accuracy comparison.")
    print(f"Reports saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
