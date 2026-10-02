"""Controlled sensitivity checks without independent correctness labels."""

import csv
import itertools
import json
import math

import pytest

from quality_uncertainty_medrag.conflict_stress_test import (
    generate_scenarios as generate_hard_vote_scenarios,
    main as baseline_main,
)
from quality_uncertainty_medrag.uncertainty_stress_test import (
    CONFIDENCE_TEMPLATES,
    generate_scenarios,
    main,
    summarize_scenarios,
)


TEMPLATES = {
    "CONFIDENT": (0.90, 0.05, 0.05),
    "MODERATE": (0.70, 0.20, 0.10),
    "UNCERTAIN": (0.45, 0.40, 0.15),
}
CONFIDENCE_ORDER = {"UNCERTAIN": 0, "MODERATE": 1, "CONFIDENT": 2}
DECISION_FIELDS = (
    "majority_vote_decision",
    "quality_weighted_vote_decision",
    "quality_uncertainty_weighted_decision",
)
SUMMARY_FLAGS = (
    ("majority_vs_quality_weighted_disagree", "majority_vs_quality_weighted_disagreement"),
    ("quality_weighted_vs_quality_uncertainty_disagree", "quality_weighted_vs_quality_uncertainty_disagreement"),
    ("all_three_agree", "all_three_agree"),
)


@pytest.fixture(scope="module")
def scenarios():
    return generate_scenarios()


def find_scenario(rows, ratio, quality, support_confidence, contradict_confidence):
    return next(
        row for row in rows
        if (
            row["conflict_ratio"], row["quality_condition"],
            row["support_confidence_condition"], row["contradict_confidence_condition"],
        ) == (ratio, quality, support_confidence, contradict_confidence)
    )


def decision_from_score(score):
    if score is None or score == 0.0:
        return "ABSTAIN"
    return "SUPPORT" if score > 0.0 else "CONTRADICT"


def test_generation_covers_existing_grid_and_every_confidence_pair(scenarios):
    assert generate_scenarios() == scenarios
    assert len(scenarios) == 315
    assert len({row["scenario_id"] for row in scenarios}) == 315
    assert dict(CONFIDENCE_TEMPLATES) == TEMPLATES
    original_grid = {
        (row["conflict_ratio"], row["quality_condition"])
        for row in generate_hard_vote_scenarios()
    }
    observed = {
        (
            row["conflict_ratio"], row["quality_condition"],
            row["support_confidence_condition"], row["contradict_confidence_condition"],
        ) for row in scenarios
    }
    assert observed == {
        (ratio, quality, support, contradict)
        for ratio, quality in original_grid
        for support, contradict in itertools.product(TEMPLATES, repeat=2)
    }


def test_hard_vote_baselines_match_existing_results_for_every_confidence_pair(scenarios):
    original = {
        (row["conflict_ratio"], row["quality_condition"]): row
        for row in generate_hard_vote_scenarios()
    }
    for row in scenarios:
        baseline = original[row["conflict_ratio"], row["quality_condition"]]
        for field in (
            "support_count", "contradict_count", "support_quality_mean",
            "contradict_quality_mean", "support_quality_sum", "contradict_quality_sum",
            "quality_gap_direction", "majority_vote_decision", "quality_weighted_vote_decision",
        ):
            assert row[field] == baseline[field]
        assert row["majority_vs_quality_weighted_disagree"] is baseline["baselines_disagree"]


def test_confidence_asymmetry_is_classified_without_a_support_orientation_bias(scenarios):
    for row in scenarios:
        support = CONFIDENCE_ORDER[row["support_confidence_condition"]]
        contradict = CONFIDENCE_ORDER[row["contradict_confidence_condition"]]
        if support == contradict:
            expected = "equal_confidence"
        elif support > contradict:
            expected = "support_more_confident"
        else:
            expected = "contradict_more_confident"
        assert row["confidence_asymmetry"] == expected
        if row["support_count"] == row["contradict_count"]:
            assert row["majority_confidence_relation"] == "no_majority"
        elif support == contradict:
            assert row["majority_confidence_relation"] == "equal_confidence"
        else:
            majority_is_support = row["support_count"] > row["contradict_count"]
            majority_more_confident = (support > contradict) == majority_is_support
            assert row["majority_confidence_relation"] == (
                "majority_more_confident" if majority_more_confident else "minority_more_confident"
            )
        if row["support_quality_mean"] == row["contradict_quality_mean"]:
            assert row["quality_confidence_relation"] == "no_quality_gap"
        elif support == contradict:
            assert row["quality_confidence_relation"] == "equal_confidence"
        else:
            higher_quality_is_support = row["support_quality_mean"] > row["contradict_quality_mean"]
            higher_quality_more_confident = (support > contradict) == higher_quality_is_support
            assert row["quality_confidence_relation"] == (
                "higher_quality_more_confident" if higher_quality_more_confident
                else "lower_quality_more_confident"
            )


def test_requested_scenario_families_cover_both_directional_orientations(scenarios):
    expected_family_counts = {
        "equal_confidence": 105,
        "majority_uncertain_minority_confident": 28,
        "majority_confident_minority_uncertain": 28,
        "higher_quality_uncertain_lower_quality_confident": 30,
        "higher_quality_confident_lower_quality_uncertain": 30,
    }
    assert set().union(*(set(row["scenario_families"]) for row in scenarios)) == set(expected_family_counts)
    for family, expected_count in expected_family_counts.items():
        selected = [row for row in scenarios if family in row["scenario_families"]]
        assert len(selected) == expected_count
        if family == "equal_confidence":
            assert all(row["support_confidence_condition"] == row["contradict_confidence_condition"] for row in selected)
            continue
        assert {row["support_confidence_condition"] for row in selected} == {"CONFIDENT", "UNCERTAIN"}
        assert {row["contradict_confidence_condition"] for row in selected} == {"CONFIDENT", "UNCERTAIN"}
        if family.startswith("majority_"):
            assert all(row["support_count"] != row["contradict_count"] for row in selected)
            expected = "minority_more_confident" if family == "majority_uncertain_minority_confident" else "majority_more_confident"
            assert all(row["majority_confidence_relation"] == expected for row in selected)
        else:
            assert all(row["support_quality_mean"] != row["contradict_quality_mean"] for row in selected)
            expected = "lower_quality_more_confident" if family == "higher_quality_uncertain_lower_quality_confident" else "higher_quality_more_confident"
            assert all(row["quality_confidence_relation"] == expected for row in selected)


@pytest.mark.parametrize(
    "ratio,support_confidence,contradict_confidence,majority,experimental",
    [
        ("9:1", "UNCERTAIN", "CONFIDENT", "SUPPORT", "CONTRADICT"),
        ("1:9", "CONFIDENT", "UNCERTAIN", "CONTRADICT", "SUPPORT"),
        ("9:1", "CONFIDENT", "UNCERTAIN", "SUPPORT", "SUPPORT"),
        ("1:9", "UNCERTAIN", "CONFIDENT", "CONTRADICT", "CONTRADICT"),
    ],
)
def test_majority_and_minority_confidence_sensitivity(
    scenarios, ratio, support_confidence, contradict_confidence, majority, experimental,
):
    row = find_scenario(scenarios, ratio, "equal", support_confidence, contradict_confidence)
    assert row["majority_vote_decision"] == majority
    assert row["quality_weighted_vote_decision"] == majority
    assert row["quality_uncertainty_weighted_decision"] == experimental


@pytest.mark.parametrize(
    "quality,support_confidence,contradict_confidence,weighted,experimental",
    [
        ("support_large_gap", "UNCERTAIN", "CONFIDENT", "SUPPORT", "CONTRADICT"),
        ("contradict_large_gap", "CONFIDENT", "UNCERTAIN", "CONTRADICT", "SUPPORT"),
        ("support_large_gap", "CONFIDENT", "UNCERTAIN", "SUPPORT", "SUPPORT"),
        ("contradict_large_gap", "UNCERTAIN", "CONFIDENT", "CONTRADICT", "CONTRADICT"),
    ],
)
def test_quality_confidence_alignment_and_opposition(
    scenarios, quality, support_confidence, contradict_confidence, weighted, experimental,
):
    row = find_scenario(scenarios, "5:5", quality, support_confidence, contradict_confidence)
    assert row["majority_vote_decision"] == "ABSTAIN"
    assert row["quality_weighted_vote_decision"] == weighted
    assert row["quality_uncertainty_weighted_decision"] == experimental


@pytest.mark.parametrize("confidence", TEMPLATES)
def test_symmetric_equal_confidence_evidence_abstains(scenarios, confidence):
    row = find_scenario(scenarios, "5:5", "equal", confidence, confidence)
    assert [row[field] for field in DECISION_FIELDS] == ["ABSTAIN"] * 3
    assert row["quality_uncertainty_aggregate_score"] == 0.0
    assert row["total_effective_weight"] > 0.0
    assert row["all_three_agree"] is True


def test_experimental_score_matches_independent_probability_formula(scenarios):
    for row in scenarios:
        support_values = TEMPLATES[row["support_confidence_condition"]]
        contradict_values = TEMPLATES[row["contradict_confidence_condition"]]
        support_entropy = -math.fsum(p * math.log(p) for p in support_values) / math.log(3)
        contradict_entropy = -math.fsum(p * math.log(p) for p in contradict_values) / math.log(3)
        support_weight = row["support_quality_mean"] * (1.0 - support_entropy)
        contradict_weight = row["contradict_quality_mean"] * (1.0 - contradict_entropy)
        weights = [support_weight] * row["support_count"] + [contradict_weight] * row["contradict_count"]
        directions = [support_values[0] - support_values[1]] * row["support_count"]
        directions += [contradict_values[1] - contradict_values[0]] * row["contradict_count"]
        denominator = math.fsum(weights)
        expected_score = math.fsum((weight / denominator) * direction for weight, direction in zip(weights, directions))
        assert row["total_effective_weight"] == pytest.approx(denominator, abs=1e-14)
        assert row["quality_uncertainty_aggregate_score"] == pytest.approx(expected_score, abs=1e-14)
        # Match exact sign of the recorded score, including floating point
        # residuals: this experiment must not invent an abstention tolerance.
        assert row["quality_uncertainty_weighted_decision"] == decision_from_score(
            row["quality_uncertainty_aggregate_score"]
        )


def test_comparison_flags_describe_decisions_and_do_not_encode_correctness(scenarios):
    for row in scenarios:
        majority, quality, uncertainty = (row[field] for field in DECISION_FIELDS)
        assert row["majority_vs_quality_weighted_disagree"] is (majority != quality)
        assert row["quality_weighted_vs_quality_uncertainty_disagree"] is (quality != uncertainty)
        assert row["all_three_agree"] is (majority == quality == uncertainty)
        assert not any("accuracy" in field or "correct" in field or "gold" in field for field in row)


def assert_summary_counts(summary, rows):
    assert summary["total_scenarios"] == len(rows)
    for flag, prefix in SUMMARY_FLAGS:
        count = sum(row[flag] for row in rows)
        assert summary[f"{prefix}_count"] == count
        if rows:
            assert summary[f"{prefix}_percentage"] == pytest.approx(100.0 * count / len(rows))
        else:
            assert summary[f"{prefix}_percentage"] is None
    for aggregator, field in zip(
        ("majority_vote", "quality_weighted_vote", "quality_uncertainty_weighted"), DECISION_FIELDS,
    ):
        expected = {decision: sum(row[field] == decision for row in rows) for decision in ("SUPPORT", "CONTRADICT", "ABSTAIN")}
        assert summary["decision_counts"][aggregator] == expected


def test_summary_groups_and_pairwise_counts_are_consistent(scenarios):
    summary = summarize_scenarios(scenarios)
    assert_summary_counts(summary, scenarios)
    for group_field, row_field in (
        ("by_conflict_ratio", "conflict_ratio"),
        ("by_quality_gap_direction", "quality_gap_direction"),
        ("by_confidence_asymmetry", "confidence_asymmetry"),
    ):
        groups = summary[group_field]
        assert set(groups) == {row[row_field] for row in scenarios}
        assert sum(group["total_scenarios"] for group in groups.values()) == 315
        for key, group in groups.items():
            selected = [row for row in scenarios if row[row_field] == key]
            assert_summary_counts(group, selected)
    assert {key: group["total_scenarios"] for key, group in summary["by_conflict_ratio"].items()} == {
        ratio: 63 for ratio in ("9:1", "7:3", "5:5", "3:7", "1:9")
    }
    assert {key: group["total_scenarios"] for key, group in summary["by_confidence_asymmetry"].items()} == {
        "equal_confidence": 105, "support_more_confident": 105, "contradict_more_confident": 105,
    }
    def all_keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from all_keys(nested)
    assert not any("accuracy" in key or "correct" in key for key in all_keys(summary))
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_empty_summary_reports_no_misleading_percentage():
    summary = summarize_scenarios([])
    assert_summary_counts(summary, [])
    for group in ("by_conflict_ratio", "by_quality_gap_direction", "by_confidence_asymmetry"):
        assert summary[group] == {}


def test_cli_outputs_are_complete_and_byte_reproducible(tmp_path, scenarios):
    output_dir = tmp_path / "uncertainty sensitivity outputs"
    arguments = ["--output-dir", str(output_dir)]
    assert main(arguments) == 0
    csv_path, json_path = output_dir / "scenarios.csv", output_dir / "summary.json"
    first_csv, first_json = csv_path.read_bytes(), json_path.read_bytes()
    assert json.loads(first_json) == summarize_scenarios(scenarios)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 315
    text_fields = (
        "scenario_id", "conflict_ratio", "quality_condition", "quality_gap_direction",
        "support_confidence_condition", "contradict_confidence_condition", "confidence_asymmetry",
        "majority_confidence_relation", "quality_confidence_relation", *DECISION_FIELDS,
    )
    float_fields = (
        "support_quality_mean", "contradict_quality_mean", "support_quality_sum",
        "contradict_quality_sum", "quality_uncertainty_aggregate_score", "total_effective_weight",
    )
    for csv_row, scenario in zip(csv_rows, scenarios):
        for field in text_fields:
            assert csv_row[field] == scenario[field]
        for field in ("support_count", "contradict_count"):
            assert int(csv_row[field]) == scenario[field]
        for field in float_fields:
            # CSV must preserve enough precision to recover the actual score,
            # including tiny signed residuals that affect exact-sign decisions.
            assert float(csv_row[field]) == scenario[field]
        for field, _ in SUMMARY_FLAGS:
            assert csv_row[field].casefold() == str(scenario[field]).casefold()
    assert main(arguments) == 0
    assert csv_path.read_bytes() == first_csv
    assert json_path.read_bytes() == first_json


def test_existing_entry_point_opt_in_writes_the_same_new_reports(tmp_path):
    direct = tmp_path / "direct"
    opt_in = tmp_path / "opt-in"
    assert main(["--output-dir", str(direct)]) == 0
    assert baseline_main(["--include-uncertainty", "--output-dir", str(opt_in)]) == 0
    for filename in ("scenarios.csv", "summary.json"):
        assert (direct / filename).read_bytes() == (opt_in / filename).read_bytes()


def test_existing_entry_point_preserves_separate_default_output_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert baseline_main(["--include-uncertainty"]) == 0
    uncertainty_dir = tmp_path / "outputs" / "uncertainty_stress_test"
    baseline_dir = tmp_path / "outputs" / "conflict_stress_test"
    assert not baseline_dir.exists()
    with (uncertainty_dir / "scenarios.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 315
    assert json.loads((uncertainty_dir / "summary.json").read_text(encoding="utf-8"))["total_scenarios"] == 315
    saved_new_reports = {
        filename: (uncertainty_dir / filename).read_bytes()
        for filename in ("scenarios.csv", "summary.json")
    }
    assert baseline_main([]) == 0
    with (baseline_dir / "scenarios.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 35
    assert json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))["total_scenarios"] == 35
    for filename, saved_bytes in saved_new_reports.items():
        assert (uncertainty_dir / filename).read_bytes() == saved_bytes
