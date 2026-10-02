"""Offline development-only checks; no real MedQA records or live services."""
import hashlib
import json

import pytest

from quality_uncertainty_medrag.clinical_query import ClinicalQueryError, ClinicalQueryReformulator
from quality_uncertainty_medrag.clinical_query_relaxation import CoreClinicalQueryReformulator
from quality_uncertainty_medrag.clinical_query_minimal import MinimalClinicalQueryReformulator
from quality_uncertainty_medrag.query_cache import CachedClinicalQueryReformulator, QueryCacheError
from quality_uncertainty_medrag.query_relaxation import QueryRelaxationRetriever, GENERATION_SETTINGS


class SequenceBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompt, *, generation_config=None):
        self.calls.append((prompt, dict(generation_config or {})))
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result


def cached(path, backend, factory=ClinicalQueryReformulator, **kwargs):
    return CachedClinicalQueryReformulator(
        backend, backend_id="ollama/qwen3:8b", cache_dir=path,
        generation_settings=GENERATION_SETTINGS, model_digest="a" * 64,
        reformulator_factory=factory, **kwargs)


def reformulate(wrapper):
    return wrapper.reformulate(question_id="dev-synthetic", question_text="Development stem.")


@pytest.mark.parametrize("outputs", [
    [TimeoutError("SECRET_PROVIDER_CONTENT"), "Named condition"],
    ["invalid\noutput", "Named condition"],
    [RuntimeError("SECRET_PROVIDER_CONTENT"), "", "Named condition"],
])
def test_identical_prompt_retry_audit_and_normal_cache_reuse(tmp_path, outputs):
    backend = SequenceBackend(outputs)
    wrapper = cached(tmp_path, backend)
    assert reformulate(wrapper) == "Named condition"
    diagnostic = wrapper.last_record["query_generation"]
    assert diagnostic["retry_count"] == len(outputs) - 1
    assert diagnostic["attempt_count"] == len(outputs)
    assert [a["status"] for a in diagnostic["attempts"]] == ["failed"] * (len(outputs)-1) + ["success"]
    assert all(call == backend.calls[0] for call in backend.calls)
    assert json.loads(backend.calls[0][0].split("QUESTION INPUT (JSON):\n")[1]) == {
        "question_id": "dev-synthetic", "question_text": "Development stem."}
    audit = json.loads((tmp_path / diagnostic["attempt_log"]).read_text())
    assert audit == wrapper.last_attempt_record
    assert "SECRET_PROVIDER_CONTENT" not in json.dumps(audit)
    assert "Development stem." not in json.dumps(audit)
    saved = next(tmp_path.glob("*.json")).read_bytes()
    assert reformulate(wrapper) == "Named condition"
    assert len(backend.calls) == len(outputs)
    assert wrapper.last_record["cache_hit"] is True
    assert wrapper.last_record["query_generation"]["retry_count"] == 0
    assert next(tmp_path.glob("*.json")).read_bytes() == saved
    assert json.loads((tmp_path / diagnostic["attempt_log"]).read_text()) == audit


def test_exhaustion_preserves_every_failure_without_caching_invalid_output(tmp_path):
    backend = SequenceBackend([TimeoutError("secret"), "", "Query: invalid"])
    wrapper = cached(tmp_path, backend)
    with pytest.raises(ClinicalQueryError):
        reformulate(wrapper)
    assert len(backend.calls) == 3 and wrapper.last_record is None
    record = wrapper.last_attempt_record
    assert record["attempt_count"] == 3 and record["retry_count"] == 2
    assert all(a["status"] == "failed" for a in record["attempts"])
    assert record["attempts"][0]["failure_stage"] == "generation"
    assert record["attempts"][1]["failure_stage"] == "output_validation"
    assert list(tmp_path.glob("*.json")) == []
    assert json.loads(next((tmp_path / "attempts").glob("*.json")).read_text()) == record


@pytest.mark.parametrize("bad", [-1, 3, True, 1.5, None])
def test_retry_limit_cannot_exceed_two(tmp_path, bad):
    with pytest.raises(ValueError, match="max_generation_retries"):
        cached(tmp_path, SequenceBackend([]), max_generation_retries=bad)


def test_storage_failure_is_not_a_generation_retry(tmp_path, monkeypatch):
    wrapper = cached(tmp_path, SequenceBackend(["Named condition"]))
    def fail(*args):
        raise QueryCacheError("Storage unavailable")
    monkeypatch.setattr(wrapper, "_write_once", fail)
    with pytest.raises(QueryCacheError):
        reformulate(wrapper)
    assert len(wrapper._backend.calls) == 1


@pytest.mark.parametrize("factory,expected,extra", [
    (ClinicalQueryReformulator, "539e5bf21d20c37d011459bd1998077d4e421201794bf6ee5be56eafdabe94cf", {}),
    (CoreClinicalQueryReformulator, "276bb36bf6c240eed660886dca8deac18daf35a9631d0e733fff40bff1e3c67c",
     {"primary_query": "Named condition secondary clue"}),
])
def test_frozen_primary_and_first_fallback_prompts_are_byte_identical(factory, expected, extra):
    backend = SequenceBackend(["Named condition"])
    factory(backend, backend_id="mock").reformulate(
        question_id="dev-synthetic", question_text="Development stem.", **extra)
    assert hashlib.sha256(backend.calls[0][0].encode()).hexdigest() == expected


def test_pre_revision_primary_cache_key_and_string_are_unchanged(tmp_path):
    backend = SequenceBackend(["Named condition"])
    wrapper = cached(tmp_path, backend)
    reformulate(wrapper)
    context = {"schema_version": 1, "question_id": "dev-synthetic",
               "prompt_sha256": "539e5bf21d20c37d011459bd1998077d4e421201794bf6ee5be56eafdabe94cf",
               "backend_id": "ollama/qwen3:8b", "reformulator_id": "clinical-query-reformulator-v1",
               "model_digest": "a" * 64, "generation_settings": GENERATION_SETTINGS}
    key = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert wrapper.last_record["cache_key"] == key
    unused = SequenceBackend([])
    again = cached(tmp_path, unused, max_generation_retries=0)
    assert reformulate(again) == "Named condition"
    assert unused.calls == []


class Client:
    def __init__(self, counts):
        self.counts = counts
        self.searches = []
        self.fetches = []

    def search(self, query, *, top_k):
        self.searches.append((query, top_k))
        count = self.counts[query]
        return {"count": str(count), "idlist": ["123"] if count else []}

    def fetch(self, pmids):
        self.fetches.append(list(pmids))
        return {p: {"pmid": p, "title": "Mock article", "abstract": "Mock abstract",
                    "journal": "Mock", "publication_date": "2024",
                    "publication_types": ["Journal Article"], "pubmed_metadata": {}} for p in pmids}


@pytest.mark.parametrize("primary_count,core_count,minimal_count,expected", [
    (5, 0, 0, "primary"), (20, 0, 0, "primary"),
    (0, 5, 0, "fallback"), (4, 25, 0, "fallback"),
    (0, 0, 0, "minimal_fallback"), (0, 4, 3, "minimal_fallback"),
    (4, 0, 100, "minimal_fallback"),
])
def test_cascade_boundaries_final_selection_and_exact_cached_queries(tmp_path, primary_count, core_count, minimal_count, expected):
    core_backend = SequenceBackend(["Named condition modifier"])
    minimal_backend = SequenceBackend(["Named condition"])
    core = cached(tmp_path / "core", core_backend, CoreClinicalQueryReformulator)
    minimal = cached(tmp_path / "minimal", minimal_backend, MinimalClinicalQueryReformulator)
    client = Client({"Exact primary": primary_count, "Named condition modifier": core_count,
                     "Named condition": minimal_count})
    stage = QueryRelaxationRetriever(client, fallback_reformulator=core, minimal_reformulator=minimal)
    args = dict(question_id="dev-synthetic", question_text="Development stem.", primary_query="Exact primary", top_k=15)
    first = stage.retrieve(**args)
    assert first.report["query_stage_used"] == expected
    minimal_triggered = primary_count < 5 and core_count < 5
    assert first.report["minimal_fallback_triggered"] is minimal_triggered
    assert len(client.searches) == 1 + int(primary_count < 5) + int(minimal_triggered)
    assert all(k == 15 for _, k in client.searches)
    assert len(core_backend.calls) == int(primary_count < 5)
    assert len(minimal_backend.calls) == int(minimal_triggered)
    if minimal_triggered:
        prompt, config = minimal_backend.calls[0]
        assert json.loads(prompt.split("QUESTION INPUT (JSON):\n")[1]) == {
            **{k: args[k] for k in ("question_id", "question_text", "primary_query")},
            "first_fallback_query": "Named condition modifier"}
        assert config == {"temperature": 0.0, "seed": 42}
        assert "1-3" in prompt and "SINGLE" in prompt and "ONE necessary qualifier" in prompt
        assert first.report["fallback_pubmed_match_count"] == core_count
        assert first.report["pubmed_total_match_count"] == minimal_count
        if first.evidence:
            assert first.evidence[0].metadata["minimal_fallback_query"] == "Named condition"
    second = stage.retrieve(**args)
    assert second.report["query_used"] == first.report["query_used"]
    assert len(core_backend.calls) == int(primary_count < 5)
    assert len(minimal_backend.calls) == int(minimal_triggered)
    if minimal_triggered:
        assert second.report["minimal_fallback_query_cache_hit"] is True


@pytest.mark.parametrize("extra", ["options", "answer", "answer_idx", "upstream_answer", "metadata", "gold_labels"])
def test_minimal_api_rejects_options_and_gold(extra):
    backend = SequenceBackend([])
    with pytest.raises(TypeError):
        MinimalClinicalQueryReformulator(backend, backend_id="mock").reformulate(
            question_id="dev-synthetic", question_text="Development stem.", primary_query="Primary",
            first_fallback_query="Core", **{extra: "SECRET_GOLD"})
    assert backend.calls == []


def test_minimal_contexts_have_distinct_keys_and_invalid_reasoning_is_retried(tmp_path):
    backend = SequenceBackend(["<think>SECRET_REASONING</think> query", "Named condition", "Other condition"])
    wrapper = cached(tmp_path, backend, MinimalClinicalQueryReformulator)
    args = dict(question_id="dev-synthetic", question_text="Development stem.", primary_query="Named condition clues")
    assert wrapper.reformulate(**args, first_fallback_query="Named condition modifier") == "Named condition"
    key = wrapper.last_record["cache_key"]
    assert wrapper.last_record["query_generation"]["retry_count"] == 1
    assert wrapper.reformulate(**args, first_fallback_query="Other condition modifier") == "Other condition"
    assert wrapper.last_record["cache_key"] != key
    assert "SECRET_REASONING" not in "\n".join(p.read_text() for p in tmp_path.rglob("*.json"))
