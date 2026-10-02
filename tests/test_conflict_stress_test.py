"""Behavioral checks for deterministic directional conflict stress tests."""

import csv
import json

import pytest

from quality_uncertainty_medrag.conflict_stress_test import (
    generate_scenarios,
    main,
    summarize_scenarios,
)


@pytest.fixture
def scenarios():
    return generate_scenarios()


def find_scenario(rows, ratio, condition):
    return next(
        row
        for row in rows
        if row["conflict_ratio"] == ratio and row["quality_condition"] == condition
    )


def test_generation_is_deterministic_and_covers_the_controlled_grid(scenarios):
    assert generate_scenarios() == scenarios
    assert len(scenarios) == 35
    assert len({row["scenario_id"] for row in scenarios}) == 35
    assert {
        (row["conflict_ratio"], row["quality_condition"]) for row in scenarios
    } == {
        (ratio, condition)
        for ratio in ("9:1", "7:3", "5:5", "3:7", "1:9")
        for condition in (
            "equal",
            "support_small_gap",
            "contradict_small_gap",
            "support_large_gap",
            "contradict_large_gap",
            "support_extreme_gap",
            "contradict_extreme_gap",
        )
    }
    assert json.loads(json.dumps(scenarios)) == scenarios


def test_record_counts_and_quality_statistics_describe_directional_inputs(scenarios):
    conditions = {
        "equal": (0.5, 0.5, "equal"),
        "support_small_gap": (0.55, 0.45, "support_higher"),
        "contradict_small_gap": (0.45, 0.55, "contradict_higher"),
        "support_large_gap": (0.9, 0.1, "support_higher"),
        "contradict_large_gap": (0.1, 0.9, "contradict_higher"),
        "support_extreme_gap": (0.95, 0.05, "support_higher"),
        "contradict_extreme_gap": (0.05, 0.95, "contradict_higher"),
    }
    for row in scenarios:
        support, contradict = map(int, row["conflict_ratio"].split(":"))
        assert row["support_count"] == support
        assert row["contradict_count"] == contradict
        assert support + contradict == 10
        support_quality, contradict_quality, direction = conditions[
            row["quality_condition"]
        ]
        assert row["support_quality_mean"] == pytest.approx(support_quality)
        assert row["contradict_quality_mean"] == pytest.approx(contradict_quality)
        assert row["quality_gap_direction"] == direction
        assert row["support_quality_sum"] == pytest.approx(
            support * row["support_quality_mean"]
        )
        assert row["contradict_quality_sum"] == pytest.approx(
            contradict * row["contradict_quality_mean"]
        )
        assert row["baselines_disagree"] is (
            row["majority_vote_decision"] != row["quality_weighted_vote_decision"]
        )


@pytest.mark.parametrize(
    "ratio,condition,majority,weighted",
    [
        ("9:1", "equal", "SUPPORT", "SUPPORT"),
        ("1:9", "equal", "CONTRADICT", "CONTRADICT"),
        ("5:5", "equal", "ABSTAIN", "ABSTAIN"),
        ("9:1", "contradict_large_gap", "SUPPORT", "ABSTAIN"),
        ("1:9", "support_large_gap", "CONTRADICT", "ABSTAIN"),
        ("9:1", "contradict_extreme_gap", "SUPPORT", "CONTRADICT"),
        ("1:9", "support_extreme_gap", "CONTRADICT", "SUPPORT"),
        ("7:3", "contradict_large_gap", "SUPPORT", "CONTRADICT"),
        ("3:7", "support_large_gap", "CONTRADICT", "SUPPORT"),
        ("7:3", "contradict_small_gap", "SUPPORT", "SUPPORT"),
        ("3:7", "support_small_gap", "CONTRADICT", "CONTRADICT"),
        ("5:5", "support_small_gap", "ABSTAIN", "SUPPORT"),
        ("5:5", "contradict_small_gap", "ABSTAIN", "CONTRADICT"),
    ],
)
def test_expected_agreement_reversal_and_tie_edges(
    scenarios, ratio, condition, majority, weighted
):
    row = find_scenario(scenarios, ratio, condition)
    assert row["majority_vote_decision"] == majority
    assert row["quality_weighted_vote_decision"] == weighted


def test_equal_quality_preserves_hard_majority_and_quality_changes_do_not_change_it(
    scenarios,
):
    for ratio in ("9:1", "7:3", "5:5", "3:7", "1:9"):
        rows = [row for row in scenarios if row["conflict_ratio"] == ratio]
        assert len({row["majority_vote_decision"] for row in rows}) == 1
        equal = find_scenario(rows, ratio, "equal")
        assert equal["majority_vote_decision"] == equal["quality_weighted_vote_decision"]
        assert equal["baselines_disagree"] is False


def test_summary_has_correct_disagreement_totals_and_consistent_groups(scenarios):
    summary = summarize_scenarios(scenarios)
    assert summary["total_scenarios"] == 35
    assert summary["disagreement_count"] == 14
    assert summary["disagreement_percentage"] == pytest.approx(40.0)
    expected_ratios = {"9:1": 2, "7:3": 2, "5:5": 6, "3:7": 2, "1:9": 2}
    expected_directions = {"equal": (5, 0), "support_higher": (15, 7), "contradict_higher": (15, 7)}
    for ratio, disagreements in expected_ratios.items():
        group = summary["by_conflict_ratio"][ratio]
        assert group["total_scenarios"] == 7
        assert group["disagreement_count"] == disagreements
        assert group["disagreement_percentage"] == pytest.approx(disagreements / 7 * 100)
    for direction, (total, disagreements) in expected_directions.items():
        group = summary["by_quality_gap_direction"][direction]
        assert group["total_scenarios"] == total
        assert group["disagreement_count"] == disagreements
        assert group["disagreement_percentage"] == pytest.approx(disagreements / total * 100)
    for groups in (
        summary["by_conflict_ratio"],
        summary["by_quality_gap_direction"],
    ):
        assert sum(group["total_scenarios"] for group in groups.values()) == 35
        assert sum(group["disagreement_count"] for group in groups.values()) == 14
        for group in groups.values():
            for baseline in ("majority_vote", "quality_weighted_vote"):
                counts = group["decision_counts"][baseline]
                assert set(counts) == {"SUPPORT", "CONTRADICT", "ABSTAIN"}
                assert sum(counts.values()) == group["total_scenarios"]


def test_cli_saves_consistent_and_byte_reproducible_reports(tmp_path, scenarios):
    output_dir = tmp_path / "stress test outputs"
    arguments = ["--output-dir", str(output_dir)]
    assert main(arguments) == 0
    csv_path = output_dir / "scenarios.csv"
    summary_path = output_dir / "summary.json"
    first_csv = csv_path.read_bytes()
    first_json = summary_path.read_bytes()
    assert json.loads(first_json) == summarize_scenarios(scenarios)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(scenarios)
    for csv_row, scenario in zip(rows, scenarios):
        for field in (
            "scenario_id",
            "conflict_ratio",
            "quality_condition",
            "quality_gap_direction",
            "majority_vote_decision",
            "quality_weighted_vote_decision",
        ):
            assert csv_row[field] == scenario[field]
        for field in ("support_count", "contradict_count"):
            assert int(csv_row[field]) == scenario[field]
        for field in (
            "support_quality_mean",
            "contradict_quality_mean",
            "support_quality_sum",
            "contradict_quality_sum",
        ):
            assert float(csv_row[field]) == pytest.approx(scenario[field])
        assert csv_row["baselines_disagree"].casefold() == str(
            scenario["baselines_disagree"]
        ).casefold()
    assert main(arguments) == 0
    assert csv_path.read_bytes() == first_csv
    assert summary_path.read_bytes() == first_json
