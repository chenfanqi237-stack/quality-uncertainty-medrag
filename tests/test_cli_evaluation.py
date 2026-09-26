import json
import logging
from pathlib import Path

import pytest
import yaml

from quality_uncertainty_medrag.cli import main


@pytest.mark.parametrize(
    ("include_annotations", "evaluated_count", "expected_mrr", "expected_hit"),
    [(True, 2, 0.75, 1.0), (False, 0, None, None)],
)
def test_cli_reports_retrieval_metrics_separately(
    tmp_path, caplog, include_annotations, evaluated_count, expected_mrr, expected_hit
):
    rows = [
        json.loads(line)
        for line in Path("data/synthetic_medqa.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if not include_annotations:
        for row in rows:
            row.pop("relevant_doc_ids", None)
    question_path = tmp_path / "questions.jsonl"
    question_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "outputs"
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    config["paths"] = {
        "questions": str(question_path),
        "evidence": str(Path("data/synthetic_evidence.jsonl").resolve()),
        "output_dir": str(output_dir),
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    try:
        assert main(["--config", str(config_path)]) == 0
    finally:
        logger = logging.getLogger("quality_uncertainty_medrag")
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {
        "question_count": 3,
        "claim_count": 12,
        "retrieval_evaluated_question_count": evaluated_count,
        "retrieval_mrr_at_k": expected_mrr,
        "retrieval_hit_at_k": expected_hit,
        "retrieval_k": 3,
    }
    assert len((output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 12
    assert f"retrieval_evaluated_question_count={evaluated_count}" in caplog.text
    assert f"retrieval_mrr_at_k={expected_mrr}" in caplog.text
    assert f"retrieval_hit_at_k={expected_hit}" in caplog.text
    assert "retrieval_k=3" in caplog.text
