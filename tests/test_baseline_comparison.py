"""Behavioral checks for the controlled, model-free comparison experiment."""

import csv
import json

import pytest

from quality_uncertainty_medrag.baseline_comparison import main, run_comparison


@pytest.fixture
def report():
    return run_comparison()


@pytest.mark.parametrize(
    "scenario_id,support_count,contradict_count,support_sum,contradict_sum,majority,weighted",
    [
        ("scenario_1", 3, 1, 0.3, 0.9, "SUPPORT", "CONTRADICT"),
        ("scenario_2", 1, 3, 0.9, 0.3, "CONTRADICT", "SUPPORT"),
        ("scenario_3", 2, 2, 1.0, 1.0, "ABSTAIN", "ABSTAIN"),
        ("scenario_4", 0, 0, 0.0, 0.0, "ABSTAIN", "ABSTAIN"),
        ("scenario_5a", 2, 1, 0.2, 0.9, "SUPPORT", "CONTRADICT"),
        ("scenario_5b", 2, 1, 0.9, 0.1, "SUPPORT", "SUPPORT"),
    ],
)
def test_controlled_scenarios_show_expected_baseline_behavior(
    report,
    scenario_id,
    support_count,
    contradict_count,
    support_sum,
    contradict_sum,
    majority,
    weighted,
):
    row = next(row for row in report["scenarios"] if row["scenario_id"] == scenario_id)

    assert row["support_count"] == support_count
    assert row["contradict_count"] == contradict_count
    assert row["support_quality_sum"] == pytest.approx(support_sum)
    assert row["contradict_quality_sum"] == pytest.approx(contradict_sum)
    assert row["majority_vote_decision"] == majority
    assert row["quality_weighted_vote_decision"] == weighted
    assert row["unresolved_count"] == 0
    assert row["evidence_count"] == len(row["input_evidence"])
    if scenario_id == "scenario_4":
        assert row["irrelevant_count"] == row["evidence_count"] > 0
    else:
        assert row["irrelevant_count"] == 0
        assert row["evidence_count"] == support_count + contradict_count


def test_quality_distribution_comparison_changes_only_quality_inputs(report):
    rows = {row["scenario_id"]: row for row in report["scenarios"]}
    before = rows["scenario_5a"]
    after = rows["scenario_5b"]

    assert len(before["input_evidence"]) == len(after["input_evidence"]) == 3
    for first, second in zip(before["input_evidence"], after["input_evidence"]):
        assert {key: value for key, value in first.items() if key != "quality"} == {
            key: value for key, value in second.items() if key != "quality"
        }
        assert first["quality"] != second["quality"]
    assert before["support_count"] == after["support_count"] == 2
    assert before["contradict_count"] == after["contradict_count"] == 1
    assert before["majority_vote_decision"] == after["majority_vote_decision"]
    assert before["quality_weighted_vote_decision"] != after["quality_weighted_vote_decision"]


def test_comparison_report_is_deterministic_and_explicitly_synthetic(report):
    assert run_comparison() == report
    assert report["experiment"] == "controlled-synthetic-baseline-comparison"
    assert report["quality_source"] == "assigned synthetic inputs"
    assert [row["scenario_id"] for row in report["scenarios"]] == [
        "scenario_1",
        "scenario_2",
        "scenario_3",
        "scenario_4",
        "scenario_5a",
        "scenario_5b",
    ]
    assert json.loads(json.dumps(report)) == report


def test_cli_reports_are_consistent_and_byte_reproducible(tmp_path):
    output_dir = tmp_path / "comparison outputs"
    arguments = ["--output-dir", str(output_dir)]
    assert main(arguments) == 0

    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    report = json.loads(first_json)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == len(report["scenarios"]) == 6
    assert report == run_comparison()
    for csv_row, json_row in zip(csv_rows, report["scenarios"]):
        for field in (
            "scenario_id",
            "majority_vote_decision",
            "quality_weighted_vote_decision",
        ):
            assert csv_row[field] == json_row[field]
        for field in (
            "support_count",
            "contradict_count",
            "evidence_count",
            "irrelevant_count",
            "unresolved_count",
        ):
            assert int(csv_row[field]) == json_row[field]
        for field in ("support_quality_sum", "contradict_quality_sum"):
            assert float(csv_row[field]) == pytest.approx(json_row[field])

    assert main(arguments) == 0
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
