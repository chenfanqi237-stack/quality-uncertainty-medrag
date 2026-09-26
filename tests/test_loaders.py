import json

import pytest

from quality_uncertainty_medrag.loaders import load_medqa_questions, load_retrieved_evidence
from quality_uncertainty_medrag.models import EvidenceType, Stance


def write_records(tmp_path, name, *records):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def evidence_record(**overrides):
    return {
        "schema_version": "2.0",
        "question_id": "q",
        "doc_id": "d",
        "rank": 1,
        "text": "text",
        "source": "source",
        "evidence_type": "other",
        "retrieval_score": 0.1,
        **overrides,
    }


def question_record(**overrides):
    return {
        "id": "q",
        "question": "Question?",
        "options": {"A": "one", "B": "two"},
        "answer": "A",
        **overrides,
    }


INVALID_TEXT_VALUES = [None, 1, True, [], {}, "", " \t "]


def test_medqa_loader_accepts_option_mapping_and_derives_candidate_claims():
    question = load_medqa_questions("data/synthetic_medqa.jsonl")[0]
    assert question.option_labels == ("A", "B", "C", "D")
    assert question.answer_label == "B"
    assert question.answer_text == "Intervention Beta"
    assert [claim.option_label for claim in question.candidate_claims] == ["A", "B", "C", "D"]
    assert question.candidate_claims[1].option_index == 1
    assert question.candidate_claims[1].option_text == "Intervention Beta"
    assert not hasattr(question, "claim")
    assert not hasattr(question, "expected_stance")


def test_medqa_loader_accepts_option_list_and_index(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "x", "question": "Q?", "options": ["one", "two"], "answer": 1})
        + "\n",
        encoding="utf-8",
    )
    question = load_medqa_questions(path)[0]
    assert question.answer_label == "B"
    assert question.candidate_claims[0].question_id == "x"


def test_medqa_loader_accepts_question_id_alias(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "x",
                "question": "Q?",
                "options": {"A": "one", "B": "two"},
                "answer": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_medqa_questions(path)[0].question_id == "x"


def test_evidence_loader_groups_sorts_and_loads_per_option_stances():
    grouped = load_retrieved_evidence("data/synthetic_evidence.jsonl")
    first = grouped["q1"][0]
    assert first.doc_id == "q1-guideline"
    assert first.schema_version == "2.0"
    assert first.evidence_type is EvidenceType.EVIDENCE_BASED_GUIDELINE
    assert first.annotated_stances["A"] is Stance.CONTRADICT
    assert first.annotated_stances["B"] is Stance.SUPPORT


def test_evidence_loader_rejects_duplicate_question_document_pair_with_new_rank(tmp_path):
    path = write_records(
        tmp_path,
        "evidence.jsonl",
        evidence_record(rank=1),
        evidence_record(rank=2),
    )
    with pytest.raises(ValueError, match="duplicated"):
        load_retrieved_evidence(path)


def test_evidence_loader_allows_shared_document_and_rank_across_questions(tmp_path):
    path = write_records(
        tmp_path,
        "evidence.jsonl",
        evidence_record(question_id="q1"),
        evidence_record(question_id="q2"),
    )
    grouped = load_retrieved_evidence(path)
    assert set(grouped) == {"q1", "q2"}
    assert grouped["q1"][0].doc_id == grouped["q2"][0].doc_id == "d"
    assert grouped["q1"][0].rank == grouped["q2"][0].rank == 1


def test_evidence_loader_rejects_duplicate_rank_with_distinct_documents(tmp_path):
    path = write_records(
        tmp_path,
        "evidence.jsonl",
        evidence_record(doc_id="d1"),
        evidence_record(doc_id="d2"),
    )
    with pytest.raises(ValueError, match="ranks.*unique"):
        load_retrieved_evidence(path)


@pytest.mark.parametrize("field", ["id", "question_id", "question"])
@pytest.mark.parametrize("invalid", INVALID_TEXT_VALUES)
def test_question_loader_rejects_invalid_required_text(tmp_path, field, invalid):
    record = question_record()
    if field == "question_id":
        record.pop("id")
    record[field] = invalid
    path = write_records(tmp_path, "questions.jsonl", record)
    with pytest.raises(ValueError, match="Invalid question record.*:1:"):
        load_medqa_questions(path)


@pytest.mark.parametrize("field", ["question_id", "doc_id", "text", "source"])
@pytest.mark.parametrize("invalid", INVALID_TEXT_VALUES)
def test_evidence_loader_rejects_invalid_required_text(tmp_path, field, invalid):
    record = evidence_record(**{field: invalid})
    path = write_records(tmp_path, "evidence.jsonl", record)
    with pytest.raises(ValueError, match="Invalid evidence record.*:1:"):
        load_retrieved_evidence(path)


@pytest.mark.parametrize("representation", ["mapping", "list"])
@pytest.mark.parametrize("invalid", INVALID_TEXT_VALUES)
def test_question_loader_rejects_invalid_option_text(tmp_path, representation, invalid):
    options = {"A": invalid, "B": "two"} if representation == "mapping" else [invalid, "two"]
    record = question_record(options=options, answer=1)
    path = write_records(tmp_path, "questions.jsonl", record)
    with pytest.raises(ValueError, match="Invalid question record.*:1:"):
        load_medqa_questions(path)


@pytest.mark.parametrize("label", ["", " \t "])
def test_question_loader_rejects_empty_option_label(tmp_path, label):
    record = question_record(options={label: "one", "B": "two"}, answer="B")
    path = write_records(tmp_path, "questions.jsonl", record)
    with pytest.raises(ValueError, match="Invalid question record.*:1:"):
        load_medqa_questions(path)


@pytest.mark.parametrize("label", ["", " \t "])
def test_evidence_loader_rejects_empty_annotated_stance_label(tmp_path, label):
    record = evidence_record(annotated_stances={label: "SUPPORT"})
    path = write_records(tmp_path, "evidence.jsonl", record)
    with pytest.raises(ValueError, match="Invalid evidence record.*:1:"):
        load_retrieved_evidence(path)


@pytest.mark.parametrize(
    "invalid", [None, True, "0.5", [], {}, float("nan"), float("inf"), -float("inf")]
)
def test_evidence_loader_rejects_non_numeric_or_non_finite_score(tmp_path, invalid):
    path = write_records(tmp_path, "evidence.jsonl", evidence_record(retrieval_score=invalid))
    with pytest.raises(ValueError, match="Invalid evidence record.*:1:"):
        load_retrieved_evidence(path)


@pytest.mark.parametrize("score", [-12, -0.5, 0, 7.25])
def test_evidence_loader_accepts_finite_scores_without_unit_interval_bound(tmp_path, score):
    path = write_records(tmp_path, "evidence.jsonl", evidence_record(retrieval_score=score))
    loaded = load_retrieved_evidence(path)["q"][0]
    assert loaded.retrieval_score == score
    assert isinstance(loaded.retrieval_score, float)


def test_loaders_normalize_required_text_and_preserve_option_content(tmp_path):
    questions_path = write_records(
        tmp_path,
        "questions.jsonl",
        question_record(
            id=" q ",
            question=" Question? ",
            options={" A ": " one ", "B": "two"},
            answer=" A ",
        ),
    )
    question = load_medqa_questions(questions_path)[0]
    assert question.question_id == "q"
    assert question.question == "Question?"
    assert question.option_labels == (" A ", "B")
    assert question.options == (" one ", "two")

    evidence_path = write_records(
        tmp_path,
        "evidence.jsonl",
        evidence_record(question_id=" q ", doc_id=" d ", text=" text ", source=" source "),
    )
    evidence = load_retrieved_evidence(evidence_path)["q"][0]
    assert (evidence.question_id, evidence.doc_id, evidence.text, evidence.source) == (
        "q", "d", "text", "source"
    )


def test_question_schema_removes_legacy_single_claim_fields():
    schema = json.loads(open("schemas/medqa_question.schema.json", encoding="utf-8").read())
    assert "claim" not in schema["properties"]
    assert "expected_stance" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_new_schemas_describe_candidate_question_and_claim_outputs():
    candidate = json.loads(open("schemas/candidate_claim.schema.json", encoding="utf-8").read())
    question_prediction = json.loads(
        open("schemas/question_prediction.schema.json", encoding="utf-8").read()
    )
    claim_prediction = json.loads(
        open("schemas/claim_prediction.schema.json", encoding="utf-8").read()
    )

    assert set(candidate["required"]) == {
        "question_id",
        "option_index",
        "option_label",
        "option_text",
    }
    assert question_prediction["properties"]["option_scores"]["minProperties"] == 2
    assert claim_prediction["properties"]["decision"]["enum"] == [
        "SUPPORT",
        "CONTRADICT",
        "ABSTAIN",
    ]
    stance = claim_prediction["properties"]["evidence"]["items"]["properties"]["stance"]
    assert set(stance["required"]) == {"SUPPORT", "CONTRADICT", "IRRELEVANT"}
    assert "ABSTAIN" not in stance["properties"]
