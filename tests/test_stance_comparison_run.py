"""Integrity checks for the development-only stance comparison runner."""
import csv
import json
import math

import pytest

from quality_uncertainty_medrag import stance_comparison_run as comparison


def make_reference(path):
    fields = ["batch", "pair_id", "question_id", "question_text", "candidate_option_id",
              "candidate_option_text", "pmid", "evidence_title", "evidence_abstract", "human_stance"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(60):
            writer.writerow({"batch": "structural" if i < 30 else "directional", "pair_id": f"pair-{i}",
                             "question_id": f"medqa-us-dev-{1+i//2:06d}", "question_text": "What clinical condition?",
                             "candidate_option_id": "A" if i % 2 == 0 else "B", "candidate_option_text": "Candidate only",
                             "pmid": str(10_000 + i), "evidence_title": "Evidence", "evidence_abstract": "Abstract",
                             "human_stance": "SUPPORT" if i < 10 else "CONTRADICT" if i < 15 else "IRRELEVANT"})


def test_unlabeled_projection_excludes_reference_and_other_options(tmp_path):
    source = tmp_path / "reference.csv"
    make_reference(source)
    rows = comparison.read_unlabeled_inputs(source)
    assert len(rows) == 60
    assert set(rows[0]) == set(comparison.KEY_FIELDS) | set(comparison.TEXT_FIELDS)
    assert all("human_stance" not in row and "answer" not in row and "options" not in row for row in rows)
    assert rows[0]["candidate_option_text"] == "Candidate only"


def test_projection_rejects_future_development_questions(tmp_path):
    source = tmp_path / "reference.csv"
    make_reference(source)
    text = source.read_text(encoding="utf-8").replace("medqa-us-dev-000001", "medqa-us-dev-000031", 1)
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="dev 1-30"):
        comparison.read_unlabeled_inputs(source)


def test_cache_exact_text_and_frozen_prompt_required():
    row = {"question_id": "medqa-us-dev-000001", "candidate_option_id": "A", "evidence_doc_id": "12345",
           "question_stem": "Question", "candidate_option_text": "Candidate", "evidence_title": "Title",
           "evidence_abstract": "Abstract"}
    from quality_uncertainty_medrag import llm_stance as qwen
    entry = {"schema_version": 1, **{field: row[field] for field in comparison.KEY_FIELDS},
             "input_sha256": comparison.input_hash(row), "classifier_version": qwen.CLASSIFIER_VERSION,
             "prompt_template_sha256": comparison.sha(qwen.STANCE_PROMPT),
             "prompt_sha256": comparison.sha(qwen.build_stance_prompt(**{f: row[f] for f in comparison.TEXT_FIELDS})),
             "model_metadata": vars(comparison.EXPECTED_QWEN),
             "probabilities": {"SUPPORT": 0.7, "CONTRADICT": 0.1, "IRRELEVANT": 0.2},
             "argmax_label": "SUPPORT", "generation_attempts": 1,
             "attempts": [{"index": 1, "status": "success", "timestamp": "2026-09-30T00:00:00Z"}]}
    context_fields = ("schema_version", "question_id", "candidate_option_id", "evidence_doc_id",
                      "classifier_version", "prompt_template_sha256", "prompt_sha256", "input_sha256",
                      "model_metadata")
    entry["cache_key"] = comparison.sha(comparison.canonical({name: entry[name] for name in context_fields}))
    assert comparison._cache_match(entry, "qwen_v1", row)
    assert not comparison._cache_match(entry, "qwen_v1", {**row, "candidate_option_text": "Other candidate"})
    assert not comparison._cache_match({**entry, "prompt_template_sha256": "bad"}, "qwen_v1", row)


def test_macro_metrics_include_all_three_classes_and_exact_nll():
    rows = [
        {"reference_stance": "SUPPORT", "predicted_stance": "SUPPORT", "p_support": 1.0,
         "p_contradict": 0.0, "p_irrelevant": 0.0},
        {"reference_stance": "CONTRADICT", "predicted_stance": "IRRELEVANT", "p_support": 0.0,
         "p_contradict": 0.0, "p_irrelevant": 1.0},
        {"reference_stance": "IRRELEVANT", "predicted_stance": "IRRELEVANT", "p_support": 0.0,
         "p_contradict": 0.0, "p_irrelevant": 1.0},
    ]
    result = comparison._metric(rows)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class"]["CONTRADICT"]["f1"] == 0
    assert result["macro_f1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert result["multiclass_nll"] == "Infinity"
    assert result["nll_zero_reference_score_count"] == 1
    assert result["exact_one_hot_count"] == 3


def test_invalid_scores_and_ties_do_not_silently_enter_metrics():
    base = {"p_support": 0.45, "p_contradict": 0.45, "p_irrelevant": 0.10,
            "argmax_label": None}
    with pytest.raises(ValueError, match="tie"):
        comparison.validate_scores(base)
    with pytest.raises(ValueError, match="sum"):
        comparison.validate_scores({**base, "p_irrelevant": 0.2})
