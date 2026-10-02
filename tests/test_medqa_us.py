"""MedQA-USMLE conversion tests using only small synthetic upstream records."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag.loaders import load_medqa_questions
from quality_uncertainty_medrag.medqa_us import (
    adapt_medqa_us_record,
    inspect_medqa_us,
    load_medqa_us_dev,
    main,
    write_medqa_us_subset,
)


def upstream_record(*, answer_label: str = "B") -> dict:
    options = {"C": "Third option", "A": "First option", "B": "Second option"}
    return {
        "question": "Which synthetic option is correct?",
        "options": options,
        "answer": options[answer_label],
        "answer_idx": answer_label,
        "meta_info": "step1",
        "extra": {"tags": ["synthetic"], "original_id": 17},
    }


def write_source(path: Path, records: list[object]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("answer_label,answer_index", [("C", 0), ("A", 1), ("B", 2)])
def test_adapter_maps_answer_labels_to_the_original_option_index(answer_label, answer_index):
    raw = upstream_record(answer_label=answer_label)

    question = adapt_medqa_us_record(raw, line_number=7)

    assert question.question_id == "medqa-us-dev-000007"
    assert question.answer_index == answer_index
    assert question.answer_label == answer_label
    assert question.answer_text == raw["answer"]
    assert question.relevant_doc_ids == ()


def test_adapter_preserves_option_order_text_and_original_metadata():
    raw = upstream_record()
    raw["question"] = "  Preserve the original question exactly.  "
    raw["options"]["C"] = "  Third option with spaces  "
    original = copy.deepcopy(raw)

    question = adapt_medqa_us_record(raw, line_number=2)

    assert raw == original
    assert question.question == original["question"]
    assert question.option_labels == ("C", "A", "B")
    assert question.options == tuple(original["options"].values())
    assert tuple(claim.option_label for claim in question.candidate_claims) == ("C", "A", "B")
    assert tuple(claim.option_text for claim in question.candidate_claims) == question.options
    assert question.metadata["dataset"] == "MedQA-USMLE"
    assert question.metadata["split"] == "dev"
    assert question.metadata["source_line"] == 2
    assert question.metadata["upstream"]["answer"] == original["answer"]
    assert question.metadata["upstream"]["answer_idx"] == original["answer_idx"]
    assert question.metadata["upstream"]["meta_info"] == "step1"
    assert question.metadata["upstream"]["extra"]["tags"] == ("synthetic",)
    assert question.metadata["upstream"]["extra"]["original_id"] == 17
    assert "question" not in question.metadata["upstream"]
    assert "options" not in question.metadata["upstream"]
    raw["extra"]["tags"].append("later change")
    assert question.metadata["upstream"]["extra"]["tags"] == ("synthetic",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", None),
        ("question", 42),
        ("question", "  "),
        ("options", None),
        ("options", ["first", "second"]),
        ("options", {"A": "only one option"}),
        ("options", {"": "first", "B": "second"}),
        ("options", {1: "first", "B": "second"}),
        ("options", {"A": None, "B": "second"}),
        ("options", {"A": "first", "B": 42}),
        ("options", {"A": "first", "B": "  "}),
        ("answer_idx", None),
        ("answer_idx", 1),
        ("answer_idx", "D"),
        ("answer", None),
        ("answer", 1),
        ("answer", "incorrect answer text"),
    ],
)
def test_adapter_rejects_malformed_required_fields(field, value):
    raw = upstream_record()
    raw[field] = value

    with pytest.raises(ValueError):
        adapt_medqa_us_record(raw, line_number=1)


@pytest.mark.parametrize("field", ["question", "options", "answer", "answer_idx"])
def test_adapter_rejects_missing_required_fields(field):
    raw = upstream_record()
    del raw[field]

    with pytest.raises(ValueError):
        adapt_medqa_us_record(raw, line_number=1)


@pytest.mark.parametrize("raw", [None, [], "question", 42])
def test_adapter_rejects_nonobject_records(raw):
    with pytest.raises(ValueError):
        adapt_medqa_us_record(raw, line_number=1)


def test_first_n_subset_is_deterministic_and_uses_physical_line_ids(tmp_path):
    source = tmp_path / "dev.jsonl"
    records = [upstream_record(answer_label=label) for label in ("C", "A", "B")]
    source.write_text(
        "\n" + json.dumps(records[0]) + "\n\n"
        + json.dumps(records[1]) + "\n" + json.dumps(records[2]) + "\n",
        encoding="utf-8",
    )

    first = load_medqa_us_dev(source, limit=2)
    second = load_medqa_us_dev(source, limit=2)

    assert first == second
    assert first.source_record_count == 2
    assert first.malformed_records == ()
    assert tuple(question.question_id for question in first.questions) == (
        "medqa-us-dev-000002", "medqa-us-dev-000004"
    )
    assert tuple(question.answer_label for question in first.questions) == ("C", "A")
    assert len(load_medqa_us_dev(source).questions) == 3

    output_one = tmp_path / "first.jsonl"
    output_two = tmp_path / "second.jsonl"
    write_medqa_us_subset(first.questions, output_one)
    write_medqa_us_subset(second.questions, output_two)
    assert output_one.read_bytes() == output_two.read_bytes()


@pytest.mark.parametrize("malformed_line", ["not JSON", "null", "[]", '{"question": null}'])
def test_strict_loader_reports_malformed_source_path_and_line(tmp_path, malformed_line):
    source = tmp_path / "dev.jsonl"
    source.write_text(json.dumps(upstream_record()) + "\n" + malformed_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_medqa_us_dev(source)

    assert str(source) in str(error.value)
    assert ":2" in str(error.value)


def test_opt_in_skipping_counts_malformed_records_without_backfilling_first_n(tmp_path):
    source = tmp_path / "dev.jsonl"
    source.write_text(
        json.dumps(upstream_record(answer_label="C")) + "\n\nnot JSON\n"
        + json.dumps(upstream_record(answer_label="A")) + "\n"
        + json.dumps(upstream_record(answer_label="B")) + "\n",
        encoding="utf-8",
    )

    result = load_medqa_us_dev(source, limit=3, skip_malformed=True)

    assert result.source_record_count == 3
    assert len(result.questions) == 2
    assert tuple(question.answer_label for question in result.questions) == ("C", "A")
    assert len(result.malformed_records) == 1
    assert result.malformed_records[0].line_number == 3
    assert result.malformed_records[0].message


def test_export_roundtrips_through_existing_question_loader(tmp_path):
    source = write_source(tmp_path / "upstream.jsonl", [upstream_record()])
    result = load_medqa_us_dev(source)
    output = tmp_path / "processed" / "subset.jsonl"

    write_medqa_us_subset(result.questions, output)

    assert tuple(load_medqa_questions(output)) == result.questions
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert tuple(exported["options"]) == ("C", "A", "B")
    assert exported["answer"] == "B"
    assert not exported.get("relevant_doc_ids")
    assert "evidence" not in exported
    assert "annotated_stances" not in exported


def test_inspection_reports_option_distribution_claims_and_skipped_records(tmp_path):
    first = upstream_record(answer_label="A")
    second = upstream_record(answer_label="C")
    second["options"]["D"] = "Fourth option"
    source = write_source(tmp_path / "dev.jsonl", [first, None, second])
    result = load_medqa_us_dev(source, skip_malformed=True)

    report = inspect_medqa_us(result, sample_size=2)

    assert report["question_count"] == 2
    assert report["option_count_distribution"] == {"3": 1, "4": 1}
    assert report["candidate_claim_count"] == 7
    assert report["source_record_count"] == 3
    assert report["malformed_record_count"] == 1
    assert report["skipped_record_count"] == 1
    assert report["samples"] == [
        {"question_id": "medqa-us-dev-000001", "gold_answer_label": "A"},
        {"question_id": "medqa-us-dev-000003", "gold_answer_label": "C"},
    ]
    assert report["malformed_records"][0]["line_number"] == 2
    assert report["malformed_records"][0]["message"]
    assert json.loads(json.dumps(report)) == report


def test_cli_inspects_without_writing_and_exports_a_requested_subset(tmp_path, capsys):
    source = write_source(tmp_path / "dev.jsonl", [upstream_record(), upstream_record(answer_label="A")])
    files_before = set(tmp_path.iterdir())

    main(["--source", str(source), "--limit", "1"])

    inspected = json.loads(capsys.readouterr().out)
    assert inspected["question_count"] == 1
    assert set(tmp_path.iterdir()) == files_before
    output = tmp_path / "subset.jsonl"

    main(["--source", str(source), "--limit", "1", "--output", str(output)])

    exported_report = json.loads(capsys.readouterr().out)
    assert exported_report["question_count"] == 1
    assert len(load_medqa_questions(output)) == 1


def test_cli_strict_mode_does_not_export_a_partially_valid_subset(tmp_path, capsys):
    source = write_source(tmp_path / "dev.jsonl", [upstream_record(), None])
    output = tmp_path / "subset.jsonl"

    with pytest.raises(SystemExit) as error:
        main(["--source", str(source), "--output", str(output)])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert str(source) in stderr
    assert ":2" in stderr
    assert not output.exists()


def test_cli_refuses_to_overwrite_upstream_source(tmp_path, capsys):
    source = write_source(tmp_path / "dev.jsonl", [upstream_record()])
    original_bytes = source.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["--source", str(source), "--output", str(source)])

    assert error.value.code == 2
    assert capsys.readouterr().err
    assert source.read_bytes() == original_bytes
