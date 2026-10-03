import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag import topk_aggregation_development as topk


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    return topk.prepare_inputs(ROOT)


def test_real_topk_coverage_and_shared_evidence(prepared):
    expected = {
        3: {"evidence": 65, "pairs": 325, "hard": 28, "uncertainty": 22,
            "missing_hard": 297, "missing_uncertainty": 303, "calls": 3030,
            "zero_weight_evidence": 56, "zero_weight_pairs": 280, "missing_abstract": 1,
            "all_zero_weight_questions": 15},
        5: {"evidence": 107, "pairs": 535, "hard": 38, "uncertainty": 30,
            "missing_hard": 497, "missing_uncertainty": 505, "calls": 5050,
            "zero_weight_evidence": 93, "zero_weight_pairs": 465, "missing_abstract": 2,
            "all_zero_weight_questions": 13},
    }
    for k, values in expected.items():
        records, selected, coverage = topk.build_topk_records(prepared, k)
        assert len(selected) == values["evidence"]
        assert len(records) == values["pairs"]
        assert coverage["questions_with_nonempty_evidence"] == 22
        assert coverage["questions_with_zero_evidence"] == 8
        assert coverage["structurally_complete_five_option_questions"] == 22
        assert coverage["existing_hard_stance_matches"] == values["hard"]
        assert coverage["existing_uncertainty_matches"] == values["uncertainty"]
        assert coverage["missing_deterministic_hard_calls"] == values["missing_hard"]
        assert coverage["missing_uncertainty_pairs"] == values["missing_uncertainty"]
        assert coverage["missing_10_seed_stochastic_calls"] == values["calls"]
        assert coverage["selected_evidence_with_zero_quality_weight"] == values["zero_weight_evidence"]
        assert coverage["evidence_option_pairs_with_zero_quality_weight"] == values["zero_weight_pairs"]
        assert coverage["selected_evidence_with_missing_abstract"] == values["missing_abstract"]
        assert coverage["all_selected_quality_weights_zero_questions"] == values["all_zero_weight_questions"]
        assert coverage["fully_inference_complete_five_option_questions"] == 0
        by_question_option = {}
        for row in records:
            by_question_option.setdefault((row["question_id"], row["candidate_option_id"]), []).append(
                row["evidence_doc_id"]
            )
        for question in prepared.questions:
            groups = [by_question_option.get((question["question_id"], option), [])
                      for option in question["options"]]
            assert all(group == groups[0] for group in groups)


def test_canonical_rows_exclude_gold_and_preserve_missingness(prepared):
    records, _, _ = topk.build_topk_records(prepared, 3)
    forbidden = {"answer", "answer_idx", "answer_index", "reference_stance", "error"}
    assert all(not forbidden.intersection(row) for row in records)
    missing = next(row for row in records if row["stochastic_uncertainty_status"] == "MISSING")
    assert missing["uncertainty_pair_id"] is None
    assert missing["p_support"] is None and missing["u_3"] is None
    assert missing["pmid"] == missing["evidence_doc_id"]


def test_generation_free_interface_diagnostic_uses_equal_evidence(prepared):
    result = topk.run_interface_diagnostic(prepared)
    assert result["evidence_option_rows"] == 60
    assert result["question_option_groups"] == 50
    assert result["evidence_per_option_distribution"] == {1: 40, 2: 10}
    assert result["identical_ordered_evidence_for_all_methods"] is True
    assert result["identical_quality_for_all_methods"] is True
    assert result["reference_labels_used"] is False
    assert result["gold_answers_used"] is False
    assert result["zero_evidence_behavior"] == {
        "method_a": "ABSTAIN", "method_b": "ABSTAIN",
        "method_c": "ABSTAIN", "method_c_score": None,
    }
    assert result["all_zero_quality_groups"] > 0


def test_output_manifest_is_reproducible_and_refuses_overwrite(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    topk.build_outputs(ROOT, first)
    topk.build_outputs(ROOT, second)
    for name in topk.OUTPUT_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gold_answer_isolation"]["gold_fields_written_to_canonical_join"] is False
    assert manifest["new_inference_run"] is False
    with pytest.raises(FileExistsError):
        topk.build_outputs(ROOT, first)
