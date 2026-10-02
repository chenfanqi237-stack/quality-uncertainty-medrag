"""Exact-query caching with mocked model responses only."""

from __future__ import annotations

import hashlib
import json

import pytest

from quality_uncertainty_medrag.clinical_query import ClinicalQueryError
from quality_uncertainty_medrag.query_cache import (
    CachedClinicalQueryReformulator,
    QueryCacheError,
)


class RecordingBackend:
    def __init__(self, output="Gonococcal arthritis antibiotic treatment"):
        self.output = output
        self.calls = []

    def generate(self, prompt, *, generation_config=None):
        self.calls.append((prompt, dict(generation_config or {})))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def cached(tmp_path, backend, **kwargs):
    defaults = {
        "backend_id": "ollama/mock",
        "cache_dir": tmp_path,
        "generation_settings": {"temperature": 0.0, "seed": 42, "think": True},
        "model_digest": "model-digest-a",
    }
    defaults.update(kwargs)
    return CachedClinicalQueryReformulator(backend, **defaults)


def run(reformulator, *, question_id="q-1", question_text="Septic arthritis."):
    return reformulator.reformulate(question_id=question_id, question_text=question_text)


def test_cache_hit_reuses_exact_saved_query_and_never_calls_backend_again(tmp_path):
    backend = RecordingBackend('  "Gonococcal  arthritis\tantibiotic treatment"  ')
    reformulator = cached(tmp_path, backend)

    query = run(reformulator)
    first_record = reformulator.last_record
    backend.output = RuntimeError("No new generation should occur")
    assert run(reformulator) == query == "Gonococcal arthritis antibiotic treatment"
    assert len(backend.calls) == 1
    assert first_record["cache_hit"] is False
    assert reformulator.last_record["cache_hit"] is True
    assert first_record["created_at"] == reformulator.last_record["created_at"]


def test_disk_roundtrip_in_fresh_instance_preserves_case_punctuation_and_fields(tmp_path):
    output = '"diabetes mellitus"[Title/Abstract] AND "renal disease"'
    backend = RecordingBackend(output)
    first = cached(tmp_path, backend)
    assert run(first) == output
    unused = RecordingBackend(RuntimeError("Must not be reached"))
    second = cached(tmp_path, unused)

    assert run(second) == output
    assert unused.calls == []
    assert second.last_record["cache_hit"] is True
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "changed",
    [
        {"question_id": "q-2"},
        {"question_text": "Another clinical stem."},
        {"backend_id": "ollama/another-model"},
        {"model_digest": "model-digest-b"},
        {"model_digest": None},
        {"generation_settings": {"temperature": 0.0, "seed": 43, "think": True}},
        {"generation_settings": {"temperature": 0.0, "seed": 42, "think": False}},
    ],
)
def test_changed_request_context_gets_separate_entry(tmp_path, changed):
    backend = RecordingBackend()
    first = cached(tmp_path, backend)
    run(first)
    first_key = first.last_record["cache_key"]
    arguments = {key: value for key, value in changed.items() if key.startswith("question_")}
    settings = {key: value for key, value in changed.items() if key not in arguments}
    second = cached(tmp_path, backend, **settings)

    run(second, **arguments)

    assert len(backend.calls) == 2
    assert second.last_record["cache_key"] != first_key
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_key_uses_actual_prompt_and_only_identifier_and_stem_reach_model(tmp_path):
    backend = RecordingBackend()
    reformulator = cached(tmp_path, backend)
    stem = 'A distinctive clue with "quotes".\nAnother finding.'

    run(reformulator, question_text=stem)

    prompt, config = backend.calls[0]
    assert json.loads(prompt.split("QUESTION INPUT (JSON):\n", 1)[1]) == {
        "question_id": "q-1", "question_text": stem,
    }
    assert config == {"temperature": 0.0}
    record = reformulator.last_record
    assert record["prompt_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert "question_text" not in saved and "prompt" not in saved
    assert "options" not in saved and "answer" not in saved and "reasoning" not in saved
    assert stem not in json.dumps(saved)


def test_changed_prompt_hash_misses_without_changing_production_prompt(tmp_path, monkeypatch):
    backend = RecordingBackend()
    reformulator = cached(tmp_path, backend)
    run(reformulator)
    first_key = reformulator.last_record["cache_key"]
    from quality_uncertainty_medrag import query_cache

    original = query_cache.ClinicalQueryReformulator.reformulate

    def altered(self, *, question_id, question_text):
        # Mock a future prompt change only at the test's backend boundary.
        actual_backend = self._backend

        class PrefixBackend:
            def generate(self, prompt, *, generation_config=None):
                return actual_backend.generate("mock prompt revision\n" + prompt,
                                               generation_config=generation_config)

        self._backend = PrefixBackend()
        try:
            return original(self, question_id=question_id, question_text=question_text)
        finally:
            self._backend = actual_backend

    monkeypatch.setattr(query_cache.ClinicalQueryReformulator, "reformulate", altered)
    run(reformulator)

    assert len(backend.calls) == 2
    assert reformulator.last_record["cache_key"] != first_key


def test_generation_settings_are_copied_and_reformulator_config_takes_precedence(tmp_path):
    backend = RecordingBackend()
    settings = {"temperature": 0.9, "seed": 42, "think": True, "extra": {"n": [1]}}
    reformulator = cached(tmp_path, backend, generation_settings=settings)
    settings["seed"] = 999
    settings["extra"]["n"].append(2)

    run(reformulator)

    effective = reformulator.last_record["generation_settings"]
    assert effective == {"temperature": 0.0, "seed": 42, "think": True, "extra": {"n": [1]}}
    reformulator.last_record["generation_settings"]["seed"] = 999
    assert run(reformulator) == backend.output
    assert reformulator.last_record["generation_settings"]["seed"] == 42


def test_order_of_generation_settings_does_not_change_key(tmp_path):
    backend = RecordingBackend()
    first = cached(tmp_path, backend, generation_settings={"think": True, "seed": 42})
    second = cached(tmp_path, backend, generation_settings={"seed": 42, "think": True})
    run(first)
    run(second)
    assert len(backend.calls) == 1
    assert first.last_record["cache_key"] == second.last_record["cache_key"]


@pytest.mark.parametrize("invalid", [None, "", "Query: arthritis", "arthritis\nexplanation"])
def test_invalid_model_output_is_not_cached(tmp_path, invalid):
    backend = RecordingBackend(invalid)
    reformulator = cached(tmp_path, backend)
    with pytest.raises(ClinicalQueryError):
        run(reformulator)
    assert reformulator.last_record is None
    assert list(tmp_path.glob("*.json")) == []
    backend.output = "arthritis treatment"
    assert run(reformulator) == "arthritis treatment"
    assert len(backend.calls) == 4  # Three invalid attempts, then a valid call.


def test_backend_failure_is_not_cached(tmp_path):
    backend = RecordingBackend(RuntimeError("Mock unavailable"))
    reformulator = cached(tmp_path, backend)
    with pytest.raises(ClinicalQueryError):
        run(reformulator)
    assert list(tmp_path.glob("*.json")) == []
    assert reformulator.last_record is None


@pytest.mark.parametrize(
    "corruption",
    [
        "not json",
        "[]",
        {"backend_id": "wrong-model"},
        {"model_digest": "different-digest"},
        {"generated_query": " arthritis treatment "},
        {"generated_query": "Query: arthritis treatment"},
        {"created_at": None},
        {"extra_field": "unexpected"},
    ],
)
def test_corrupt_cache_fails_without_regeneration(tmp_path, corruption):
    backend = RecordingBackend()
    reformulator = cached(tmp_path, backend)
    run(reformulator)
    path = next(tmp_path.glob("*.json"))
    entry = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(corruption, str):
        text = corruption
    else:
        entry.update(corruption)
        text = json.dumps(entry)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(QueryCacheError):
        run(reformulator)
    assert len(backend.calls) == 1
    assert reformulator.last_record is None
    assert path.read_text(encoding="utf-8") == text


def test_duplicate_json_fields_fail_without_regeneration(tmp_path):
    backend = RecordingBackend()
    reformulator = cached(tmp_path, backend)
    run(reformulator)
    path = next(tmp_path.glob("*.json"))
    text = path.read_text(encoding="utf-8").strip()
    path.write_text(text[:-1] + ',"generated_query":"different output"}', encoding="utf-8")
    with pytest.raises(QueryCacheError):
        run(reformulator)
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "settings",
    [None, [], {"": 1}, {1: 1}, {"temperature": float("nan")}, {"value": object()}],
)
def test_invalid_cache_settings_fail_before_model_generation(tmp_path, settings):
    backend = RecordingBackend()
    with pytest.raises(ValueError):
        cached(tmp_path, backend, generation_settings=settings)
    assert backend.calls == []


def test_invalid_input_fails_before_generation_or_cache_writes(tmp_path):
    backend = RecordingBackend()
    reformulator = cached(tmp_path, backend)
    with pytest.raises(ClinicalQueryError):
        run(reformulator, question_text=None)
    assert backend.calls == []
    assert reformulator.last_record is None
    assert list(tmp_path.glob("*.json")) == []


def test_first_writer_entry_is_never_overwritten(tmp_path, monkeypatch):
    backend = RecordingBackend("First exact query")
    reformulator = cached(tmp_path, backend)
    run(reformulator)
    path = next(tmp_path.glob("*.json"))
    first_bytes = path.read_bytes()
    from quality_uncertainty_medrag import query_cache

    # Simulate another writer publishing the complete entry after the miss
    # check and before this writer's atomic link.
    real_exists = query_cache.Path.exists
    state = {"checked": False}

    def race_exists(self):
        if self == path and not state["checked"]:
            state["checked"] = True
            return False
        return real_exists(self)

    monkeypatch.setattr(query_cache.Path, "exists", race_exists)
    backend.output = "Later different query"
    assert run(reformulator) == "First exact query"
    assert path.read_bytes() == first_bytes
    assert len(backend.calls) == 2
    assert list(tmp_path.glob("*.tmp")) == []
