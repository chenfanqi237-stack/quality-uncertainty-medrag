import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag import stance_uncertainty_evaluation as evaluation


def inventory_item(key, seed, status="VALID", stance="SUPPORT"):
    return {
        "row": {
            "question_id": key[0],
            "candidate_option_id": key[1],
            "evidence_doc_id": key[2],
        },
        "seed": seed,
        "status": status,
        "entry": {"stance": stance} if status == "VALID" else None,
    }


def pair_inventory(key, statuses=None, labels=None):
    statuses = statuses or ["VALID"] * 10
    labels = labels or ["SUPPORT"] * 10
    return [
        inventory_item(key, seed, status, label)
        for seed, status, label in zip(evaluation.SEEDS, statuses, labels)
    ]


def evaluated_pair(index, uncertainty, error=False, reference="SUPPORT"):
    return {
        "stable_pair_identity": f"q{index}|A|d{index}",
        "pair_id": f"P{index:03d}",
        "reference_stance": reference,
        "frozen_qwen_hard_stance": "CONTRADICT" if error else reference,
        "error": error,
        "evaluation_included": True,
        "u_3": uncertainty,
        "u_directional": uncertainty,
        "relevance": 1.0,
    }


def test_completeness_counts_real_identities_and_missing_failed_seeds():
    complete = pair_inventory(("q1", "A", "d1"), labels=["SUPPORT"] * 5 + ["CONTRADICT"] * 3 + ["IRRELEVANT"] * 2)
    statuses = ["VALID"] * 8 + ["FAILED", "MISSING"]
    incomplete = pair_inventory(("q2", "B", "d2"), statuses=statuses)
    rows, groups = evaluation.sample_completeness_rows(complete + incomplete)
    assert len(groups) == 2
    assert rows[0]["completion_status"] == "COMPLETE"
    assert (rows[0]["observed_support_count"], rows[0]["observed_contradict_count"],
            rows[0]["observed_irrelevant_count"]) == (5, 3, 2)
    assert rows[1]["completion_status"] == "INCOMPLETE"
    assert rows[1]["failed_seeds"] == "109" and rows[1]["missing_seeds"] == "110"


def test_duplicate_pair_seed_and_out_of_range_seed_are_rejected():
    items = pair_inventory(("q1", "A", "d1"))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.group_inventory(items + [items[0]])
    items[0] = {**items[0], "seed": 999}
    with pytest.raises(ValueError, match="101-110"):
        evaluation.group_inventory(items)


def test_pair_uncertainty_uses_complete_pairs_only_and_handles_all_irrelevant():
    complete_key = ("q1", "A", "d1")
    incomplete_key = ("q2", "B", "d2")
    items = pair_inventory(complete_key, labels=["IRRELEVANT"] * 10)
    items += pair_inventory(incomplete_key, statuses=["VALID"] * 9 + ["MISSING"])
    completeness, groups = evaluation.sample_completeness_rows(items)
    joined = {complete_key: {"pair_id": "P1", "batch": "x", "reference_stance": "IRRELEVANT",
                             "frozen_qwen_hard_stance": "IRRELEVANT"}}
    rows = evaluation.pair_uncertainty_rows(completeness, groups, joined)
    assert len(rows) == 1
    assert rows[0]["p_irrelevant"] == 1.0 and rows[0]["u_3"] == 0.0
    assert rows[0]["u_directional"] == 1.0 and rows[0]["directional_score"] == 0.0


def test_error_detection_score_direction_ties_and_one_class_cases():
    perfect = [evaluated_pair(0, 0.1, False), evaluated_pair(1, 0.2, False),
               evaluated_pair(2, 0.8, True), evaluated_pair(3, 0.9, True)]
    result = evaluation.error_detection_metrics(perfect, "u_3")
    assert result["auroc"]["value"] == 1.0
    assert result["average_precision"]["value"] == 1.0
    assert result["error_prevalence_baseline"]["value"] == 0.5
    tied = evaluation.error_detection_metrics(
        [evaluated_pair(0, 0.5, False), evaluated_pair(1, 0.5, True)], "u_3"
    )
    assert tied["auroc"]["value"] == tied["average_precision"]["value"] == 0.5
    no_errors = evaluation.error_detection_metrics(
        [evaluated_pair(0, 0.1, False), evaluated_pair(1, 0.2, False)], "u_3"
    )
    assert no_errors["auroc"]["status"] == "NA"
    assert no_errors["average_precision"]["status"] == "NA"
    all_errors = evaluation.error_detection_metrics(
        [evaluated_pair(0, 0.1, True), evaluated_pair(1, 0.2, True)], "u_3"
    )
    assert all_errors["auroc"]["status"] == "NA"
    assert all_errors["average_precision"]["value"] == 1.0


def test_selective_accuracy_prespecified_coverage_and_tie_breaking():
    rows = [evaluated_pair(i, 0.5 if i < 2 else float(i), error=(i == 1)) for i in range(5)]
    result = evaluation.risk_coverage_rows(rows, fields=("u_3",), coverages=(0.25, 0.5, 0.75, 1.0))
    assert [row["retained_pair_count"] for row in result] == [2, 3, 4, 5]
    assert result[0]["selective_accuracy"] == 0.5
    assert result[-1]["selective_risk"] == pytest.approx(0.2)


def test_partial_marking_missing_label_exclusion_and_strict_final_gate():
    rows = [evaluated_pair(0, 0.1, False)]
    excluded = {**evaluated_pair(1, 0.9, True), "evaluation_included": False,
                "evaluation_exclusion_reason": "missing reference label"}
    status = {"expected": 600, "valid": 20, "failed": 0, "missing": 580}
    payload = evaluation.evaluation_payload(rows + [excluded], evaluation.PARTIAL_DIAGNOSTIC,
                                            status, complete_pairs=2, incomplete_pairs=58)
    assert payload["claim_status"] == "EXPLORATORY / INCOMPLETE / NOT FOR FINAL PAPER CLAIMS"
    assert payload["completed_pairs_with_labels"] == 1
    assert payload["completed_pairs_excluded_for_missing_labels_or_join"] == 1
    with pytest.raises(evaluation.FinalEvaluationGateError, match="600"):
        evaluation.enforce_mode_gate(evaluation.FINAL_EVALUATION, status, 2, 58)
    final = {"expected": 600, "valid": 600, "failed": 0, "missing": 0}
    evaluation.enforce_mode_gate(evaluation.FINAL_EVALUATION, final, 60, 0)


def test_reference_isolation_audit(tmp_path):
    output = tmp_path / evaluation.OUTPUT_REL
    output.mkdir(parents=True)
    audit = {
        "allowed_input_fields": list(evaluation.INPUT_FIELDS),
        "prompt_fields": list(evaluation.PROMPT_FIELDS),
        "reference_opened_during_sampling": False,
        "hard_prediction_file_opened_during_sampling": False,
        "model_payload_contains_reference": False,
        "medqa_gold_or_other_options_passed": False,
        "dev_31_50_accessed": False,
    }
    (output / "leakage_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    assert evaluation.validate_reference_isolation(tmp_path)["passed"] is True
    audit["reference_opened_during_sampling"] = True
    (output / "leakage_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="reference_opened"):
        evaluation.validate_reference_isolation(tmp_path)


def test_aggregation_registry_uses_the_three_existing_methods():
    methods = evaluation.aggregation_methods()
    assert [method.name for method in methods] == [
        "unweighted_hard_stance_majority",
        "quality_weighted_hard_stance",
        "quality_uncertainty_weighted_soft_stance",
    ]
    row = {"question_id": "q", "candidate_option_id": "A", "evidence_type": "systematic_review"}
    assert evaluation.aggregation_readiness([row])["ready_for_comparable_experiment"] is True


def test_artifact_writes_are_deterministic_and_never_overwrite(tmp_path):
    completeness, _ = evaluation.sample_completeness_rows(pair_inventory(("q", "A", "d")))
    pairs = [evaluated_pair(0, 0.2, False)]
    status = {"expected": 600, "valid": 10, "failed": 0, "missing": 590}
    payload = evaluation.evaluation_payload(pairs, evaluation.PARTIAL_DIAGNOSTIC, status, 1, 59)
    manifest = {
        "checkpoint": {"sha256": "abc"},
        "aggregation_preparation": {"precise_gap": "missing quality metadata"},
    }
    first, second = tmp_path / "first", tmp_path / "second"
    evaluation.write_artifacts(first, completeness, pairs, payload,
                               payload["selective_accuracy"], manifest)
    evaluation.write_artifacts(second, completeness, pairs, payload,
                               payload["selective_accuracy"], manifest)
    for name in evaluation.ARTIFACT_NAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.write_artifacts(first, completeness, pairs, payload,
                                   payload["selective_accuracy"], manifest)
