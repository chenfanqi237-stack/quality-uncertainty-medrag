import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from quality_uncertainty_medrag import llm_stance
from quality_uncertainty_medrag.llm_stance import (
    LLMStanceClassifier, StanceCacheError, StanceGenerationError, StanceModelConfig,
    StanceOutputError, build_stance_prompt, parse_stance_output,
)
from quality_uncertainty_medrag.models import Stance

VALID = '{"SUPPORT":0.7,"CONTRADICT":0.1,"IRRELEVANT":0.2}'
INPUT = dict(question_id="q1", candidate_option_id="A", evidence_doc_id="123",
             question_stem="Which mechanism explains the finding?", candidate_option_text="Mechanism X",
             evidence_title="A mechanism study", evidence_abstract="The study describes mechanism X.")


class Backend:
    def __init__(self, outputs=None):
        self.outputs = list(outputs if outputs is not None else [VALID])
        self.calls = []

    def generate(self, prompt, *, generation_config=None):
        self.calls.append((prompt, dict(generation_config)))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def classifier(tmp_path, backend=None, **kwargs):
    backend = backend or Backend()
    return LLMStanceClassifier(backend, cache_dir=tmp_path, **kwargs), backend


def test_full_distribution_label_and_serialization(tmp_path):
    model, backend = classifier(tmp_path)
    result = model.classify_texts(**INPUT)
    assert result.probabilities == {Stance.SUPPORT: 0.7, Stance.CONTRADICT: 0.1, Stance.IRRELEVANT: 0.2}
    assert result.label is Stance.SUPPORT
    assert json.loads(json.dumps(asdict(result)))["probabilities"]["SUPPORT"] == 0.7
    record = model.last_record
    assert record["argmax_label"] == "SUPPORT" and record["generation_attempts"] == 1
    assert record["cache_status"] == "MISS" and record["generation_retries"] == 0
    assert backend.calls[0][1] == {"temperature": 0, "seed": 42}
    payload = json.loads(backend.calls[0][0].removeprefix(llm_stance.STANCE_PROMPT))
    assert set(payload) == set(llm_stance.MODEL_INPUT_FIELDS)
    assert "q1" not in payload and "123" not in payload


@pytest.mark.parametrize("raw", [
    None, "", "[]", "null", "true", '"text"', "SUPPORT",
    '```json\n' + VALID + '\n```', "Explanation: " + VALID, VALID + " trailing",
    '{"SUPPORT":1,"CONTRADICT":0}',
    '{"SUPPORT":1,"CONTRADICT":0,"IRRELEVANT":0,"reason":"x"}',
    '{"SUPPORT":0.5,"SUPPORT":0.7,"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"support":0.7,"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":"0.7","CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":true,"CONTRADICT":0,"IRRELEVANT":0}',
    '{"SUPPORT":null,"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":{},"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":NaN,"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":Infinity,"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":1e400,"CONTRADICT":0.1,"IRRELEVANT":0.2}',
    '{"SUPPORT":-0.1,"CONTRADICT":0.9,"IRRELEVANT":0.2}',
    '{"SUPPORT":1.1,"CONTRADICT":0,"IRRELEVANT":0}',
    '{"SUPPORT":0.2,"CONTRADICT":0.2,"IRRELEVANT":0.2}',
])
def test_invalid_outputs_are_rejected(raw):
    with pytest.raises(StanceOutputError):
        parse_stance_output(raw)


def test_ties_remain_unresolved_and_are_not_irrelevance(tmp_path):
    model, _ = classifier(tmp_path, Backend(['{"SUPPORT":0.45,"CONTRADICT":0.45,"IRRELEVANT":0.1}']))
    result = model.classify_texts(**INPUT)
    assert result.label is None and result.is_tied
    assert result.confidence == 0.45
    assert model.last_record["argmax_label"] is None and model.last_record["is_tied"] is True


def test_sum_tolerance_preserves_original_values():
    result = parse_stance_output('{"SUPPORT":0.7,"CONTRADICT":0.1,"IRRELEVANT":0.2000001}')
    assert result.probabilities[Stance.IRRELEVANT] == 0.2000001


def test_identical_retry_and_sanitized_audit(tmp_path):
    model, backend = classifier(tmp_path, Backend([RuntimeError("SECRET REASONING"), "<think>SECRET</think>", VALID]))
    result = model.classify_texts(**INPUT)
    assert result.label is Stance.SUPPORT and len(backend.calls) == 3
    assert backend.calls[0] == backend.calls[1] == backend.calls[2]
    assert model.last_record["generation_attempts"] == 3 and model.last_record["generation_retries"] == 2
    saved = [p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json")]
    assert saved and all("SECRET" not in text and "<think>" not in text for text in saved)
    attempts = model.last_attempt_record["attempts"]
    assert [x["status"] for x in attempts] == ["failed", "failed", "success"]
    assert [x["failure_stage"] for x in attempts[:2]] == ["generation", "output_validation"]


def test_exhaustion_records_three_failures_and_no_result_cache(tmp_path):
    model, backend = classifier(tmp_path, Backend(["invalid"] * 4))
    with pytest.raises(StanceGenerationError):
        model.classify_texts(**INPUT)
    assert len(backend.calls) == 3 and model.last_record is None
    assert model.last_attempt_record["retry_count"] == 2
    assert not list(tmp_path.glob("*.json"))
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1


def test_cache_hit_has_zero_calls_preserves_exact_distribution(tmp_path):
    first, _ = classifier(tmp_path)
    one = first.classify_texts(**INPUT)
    second, backend = classifier(tmp_path, Backend([]))
    two = second.classify_texts(**INPUT)
    assert two == one and backend.calls == []
    assert second.last_record["cache_status"] == "HIT"
    assert second.last_record["generation_attempts"] == 0
    assert second.last_record["cached_generation_attempts"] == 1


@pytest.mark.parametrize("field,value", [
    ("question_id", "q2"), ("candidate_option_id", "B"), ("evidence_doc_id", "999"),
    ("candidate_option_text", "different candidate"), ("question_stem", "different stem"),
    ("evidence_title", "new title"), ("evidence_abstract", "changed abstract"),
])
def test_changed_inputs_use_distinct_cache_keys(tmp_path, field, value):
    model, backend = classifier(tmp_path, Backend([VALID, VALID]))
    model.classify_texts(**INPUT)
    key = model.last_record["cache_key"]
    model.classify_texts(**{**INPUT, field: value})
    assert model.last_record["cache_key"] != key and len(backend.calls) == 2


@pytest.mark.parametrize("config", [
    replace(StanceModelConfig(), model_digest="a" * 64),
    replace(StanceModelConfig(), backend_id="another-backend"),
    replace(StanceModelConfig(), model="another-model"),
    replace(StanceModelConfig(), ollama_version="0.34.5"),
    replace(StanceModelConfig(), thinking=False),
    replace(StanceModelConfig(), seed=43),
    replace(StanceModelConfig(), temperature=0.1),
])
def test_model_identity_and_settings_are_in_cache_key(tmp_path, config):
    first, _ = classifier(tmp_path)
    first.classify_texts(**INPUT)
    other, backend = classifier(tmp_path, model_config=config)
    other.classify_texts(**INPUT)
    assert first.last_record["cache_key"] != other.last_record["cache_key"] and len(backend.calls) == 1


def test_prompt_and_version_are_in_cache_key(tmp_path, monkeypatch):
    first, _ = classifier(tmp_path)
    first.classify_texts(**INPUT)
    monkeypatch.setattr(llm_stance, "STANCE_PROMPT", llm_stance.STANCE_PROMPT + "New version.\n")
    monkeypatch.setattr(llm_stance, "CLASSIFIER_VERSION", "v2")
    other, _ = classifier(tmp_path)
    other.classify_texts(**INPUT)
    assert first.last_record["cache_key"] != other.last_record["cache_key"]


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(probabilities={"SUPPORT":0.2,"CONTRADICT":0.2,"IRRELEVANT":0.2}),
    lambda x: x.update(argmax_label="CONTRADICT"),
    lambda x: x.update(is_tied=True), lambda x: x.update(question_id="other"),
    lambda x: x.update(generation_attempts=True), lambda x: x.update(attempts=[]),
    lambda x: x["model_metadata"].update(thinking=1),
])
def test_corrupt_cache_fails_without_regeneration(tmp_path, mutation):
    model, backend = classifier(tmp_path)
    model.classify_texts(**INPUT)
    path = next(tmp_path.glob("*.json"))
    entry = json.loads(path.read_text(encoding="utf-8"))
    mutation(entry)
    path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(StanceCacheError):
        model.classify_texts(**INPUT)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', "not json", '{"x":NaN}'])
def test_unreadable_cache_fails_without_calls(tmp_path, raw):
    model, backend = classifier(tmp_path)
    model.classify_texts(**INPUT)
    next(tmp_path.glob("*.json")).write_text(raw, encoding="utf-8")
    with pytest.raises(StanceCacheError):
        model.classify_texts(**INPUT)
    assert len(backend.calls) == 1


def test_interface_never_reads_gold_other_options_or_evidence_annotations(tmp_path):
    class Question:
        question_id = "q1"
        question = INPUT["question_stem"]
        def __getattr__(self, name):
            raise AssertionError("Forbidden question field " + name)
    class Evidence:
        question_id = "q1"
        doc_id = "123"
        metadata = {"title": INPUT["evidence_title"], "abstract": INPUT["evidence_abstract"],
                    "upstream": {"answer": "GOLD_SENTINEL"}}
        def __getattr__(self, name):
            raise AssertionError("Forbidden evidence field " + name)
    claim = SimpleNamespace(question_id="q1", option_label="A", option_text=INPUT["candidate_option_text"])
    model, backend = classifier(tmp_path)
    model.classify(Question(), claim, Evidence())
    assert "GOLD_SENTINEL" not in backend.calls[0][0]
    assert model.last_record["candidate_option_id"] == "A"


def test_mismatched_interface_ids_fail_before_model(tmp_path):
    model, backend = classifier(tmp_path)
    with pytest.raises(ValueError):
        model.classify(SimpleNamespace(question_id="q1"), SimpleNamespace(question_id="q2"), SimpleNamespace(question_id="q1"))
    assert not backend.calls


@pytest.mark.parametrize("field,value", [("question_stem",None),("candidate_option_text",1),
    ("evidence_title", ""),("evidence_abstract",None),("question_id", ""),("candidate_option_id",True)])
def test_invalid_inputs_make_no_calls(tmp_path, field, value):
    model, backend = classifier(tmp_path)
    with pytest.raises(ValueError):
        model.classify_texts(**{**INPUT, field:value})
    assert not backend.calls


def test_empty_abstract_is_preserved_not_fabricated(tmp_path):
    model, backend = classifier(tmp_path)
    model.classify_texts(**{**INPUT, "evidence_abstract":""})
    assert json.loads(backend.calls[0][0].removeprefix(llm_stance.STANCE_PROMPT))["evidence_abstract"] == ""


def test_backend_recorded_settings_must_match(tmp_path):
    backend = Backend()
    backend.think = False
    with pytest.raises(ValueError):
        classifier(tmp_path, backend)
    assert backend.calls == []


@pytest.mark.parametrize("retries", [True,-1,3,1.2])
def test_retry_limit_is_bounded(tmp_path,retries):
    with pytest.raises(ValueError):
        classifier(tmp_path,max_retries=retries)


def test_storage_failure_is_not_retried_as_model_failure(tmp_path, monkeypatch):
    model, backend = classifier(tmp_path)
    def fail(*args, **kwargs):
        raise StanceCacheError("storage failed")
    monkeypatch.setattr(model,"_atomic",fail)
    with pytest.raises(StanceCacheError):
        model.classify_texts(**INPUT)
    assert len(backend.calls) == 1
