import json

import pytest

from quality_uncertainty_medrag.loaders import load_medqa_questions, load_retrieved_evidence
from quality_uncertainty_medrag.models import EvidenceType, Stance


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


def test_evidence_loader_rejects_duplicate_doc_ids(tmp_path):
    row = {
        "schema_version": "2.0",
        "question_id": "q",
        "doc_id": "d",
        "rank": 1,
        "text": "text",
        "source": "source",
        "evidence_type": "other",
        "retrieval_score": 0.1,
    }
    path = tmp_path / "evidence.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        load_retrieved_evidence(path)


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
    stance = claim_prediction["properties"]["evidence"]["items"]["properties"]["stance"]
    assert set(stance["required"]) == {"SUPPORT", "CONTRADICT", "IRRELEVANT"}
