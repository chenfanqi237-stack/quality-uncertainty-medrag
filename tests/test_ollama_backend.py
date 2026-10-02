"""Ollama transport and clinical-query integration contracts, entirely offline."""

from __future__ import annotations

import copy
import io
import json
import socket
from types import MappingProxyType
from urllib.error import HTTPError, URLError

import pytest

from quality_uncertainty_medrag import ollama_backend
from quality_uncertainty_medrag.clinical_query import (
    ClinicalQueryError,
    ClinicalQueryReformulator,
)
from quality_uncertainty_medrag.ollama_backend import (
    OllamaBackendError,
    OllamaTextGenerationBackend,
)


SECRET_PROMPT = "private-question-stem-do-not-include-in-error"
SECRET_PROVIDER_TEXT = "private-provider-body-do-not-include-in-error"


class FakeResponse(io.BytesIO):
    """Small urllib-compatible response; no socket is ever opened."""

    status = 200

    def getcode(self):
        return self.status


@pytest.fixture(autouse=True)
def forbid_unmocked_transport(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Every Ollama test must explicitly mock urlopen.")

    monkeypatch.setattr(ollama_backend, "urlopen", unexpected_request)


def install_response(monkeypatch, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return install_bytes(monkeypatch, body)


def install_bytes(monkeypatch, body):
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append((request, timeout))
        return FakeResponse(body)

    monkeypatch.setattr(ollama_backend, "urlopen", fake_urlopen)
    return calls


def request_payload(calls):
    assert len(calls) == 1
    request, _ = calls[0]
    assert isinstance(request, ollama_backend.Request)
    return json.loads(request.data.decode("utf-8"))


def question_records(prompt):
    """Inspect the existing reformulator's data without fixing its prompt text."""
    decoder = json.JSONDecoder()
    records = []
    for index, character in enumerate(prompt):
        if character != "{":
            continue
        try:
            record, _ = decoder.raw_decode(prompt[index:])
        except ValueError:
            continue
        if isinstance(record, dict) and "question_id" in record:
            records.append(record)
    return records


def assert_safe_error(error):
    assert isinstance(error, RuntimeError)
    message = str(error)
    assert message
    assert SECRET_PROMPT not in message
    assert SECRET_PROVIDER_TEXT not in message


def test_default_request_and_raw_response_are_preserved(monkeypatch):
    raw = ' \n  "关节炎   antibiotic treatment"  \n '
    calls = install_response(monkeypatch, {"response": raw, "done": True})
    backend = OllamaTextGenerationBackend()

    assert backend.backend_id == "ollama/qwen3:8b"
    assert backend.generate(SECRET_PROMPT) == raw
    assert request_payload(calls) == {
        "model": "qwen3:8b",
        "prompt": SECRET_PROMPT,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "seed": 42},
    }
    request, timeout = calls[0]
    assert request.full_url == "http://localhost:11434/api/generate"
    assert request.get_method() == "POST"
    headers = {name.casefold(): value for name, value in request.header_items()}
    assert headers["content-type"].split(";", 1)[0] == "application/json"
    assert "authorization" not in headers
    assert "api_key" not in request_payload(calls)
    assert timeout == 180.0


def test_custom_connection_and_generation_options_do_not_mutate_input(monkeypatch):
    calls = install_response(monkeypatch, {"response": "clinical query", "done": True})
    config = {"temperature": 0.25, "top_p": 0.9, "num_predict": 64, "stop": ["END"]}
    original = copy.deepcopy(config)
    backend = OllamaTextGenerationBackend(
        base_url="http://127.0.0.1:12345///", model="qwen3:4b", timeout=12.5, seed=7
    )

    assert backend.generate("query input", generation_config=MappingProxyType(config)) == "clinical query"
    assert config == original
    assert backend.backend_id == "ollama/qwen3:4b"
    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:12345/api/generate"
    assert timeout == 12.5
    payload = request_payload(calls)
    assert payload["model"] == "qwen3:4b"
    assert payload["options"] == dict(original, seed=7)
    assert payload["stream"] is False
    assert payload["think"] is False


def test_explicit_zero_temperature_and_seed_override_survive_merge(monkeypatch):
    calls = install_response(monkeypatch, {"response": "clinical query", "done": True})
    config = {"temperature": 0, "seed": 0}
    backend = OllamaTextGenerationBackend(seed=99)

    backend.generate("query input", generation_config=config)

    assert request_payload(calls)["options"] == {"temperature": 0, "seed": 0}
    assert config == {"temperature": 0, "seed": 0}


def test_per_call_options_do_not_change_following_defaults(monkeypatch):
    calls = install_response(monkeypatch, {"response": "clinical query", "done": True})
    backend = OllamaTextGenerationBackend()

    backend.generate("first", generation_config={"temperature": 0.5, "seed": 11})
    backend.generate("second")

    assert len(calls) == 2
    first = json.loads(calls[0][0].data)
    second = json.loads(calls[1][0].data)
    assert first["options"] == {"temperature": 0.5, "seed": 11}
    assert second["options"] == {"temperature": 0.0, "seed": 42}


def test_thinking_is_not_returned_as_generated_text(monkeypatch):
    install_response(monkeypatch, {
        "thinking": SECRET_PROVIDER_TEXT,
        "response": "gonococcal arthritis antibiotic treatment",
        "done": True,
    })

    assert OllamaTextGenerationBackend().generate("stem") == "gonococcal arthritis antibiotic treatment"


def test_enabled_thinking_changes_only_the_wire_thinking_flag(monkeypatch):
    calls = install_response(monkeypatch, {
        "thinking": SECRET_PROVIDER_TEXT, "response": "clinical query", "done": True,
    })
    backend = OllamaTextGenerationBackend(think=True)
    assert backend.generate("stem") == "clinical query"
    assert backend.think is True
    payload = request_payload(calls)
    assert payload == {
        "model": "qwen3:8b", "prompt": "stem", "stream": False, "think": True,
        "options": {"temperature": 0.0, "seed": 42},
    }


@pytest.mark.parametrize("think", [None, 0, 1, "true", "false", [], {}])
def test_invalid_thinking_mode_is_rejected_before_transport(think):
    with pytest.raises(ValueError):
        OllamaTextGenerationBackend(think=think)


def test_existing_reformulator_receives_only_identifier_and_stem(monkeypatch):
    calls = install_response(monkeypatch, {
        "response": ' \n  "gonococcal   arthritis\tantibiotic treatment"  \n ',
        "thinking": SECRET_PROVIDER_TEXT,
        "done": True,
    })
    backend = OllamaTextGenerationBackend()
    reformulator = ClinicalQueryReformulator(backend, backend_id=backend.backend_id)
    stem = 'Fever and dysuria. She describes "migratory arthritis".\nSymptoms recur.'

    query = reformulator.reformulate(question_id="q-clinical", question_text=stem)

    assert query == "gonococcal arthritis antibiotic treatment"
    payload = request_payload(calls)
    assert question_records(payload["prompt"]) == [
        {"question_id": "q-clinical", "question_text": stem}
    ]
    assert payload["options"] == {"temperature": 0.0, "seed": 42}
    assert reformulator.backend_id == "ollama/qwen3:8b"
    assert reformulator.reformulator_id == "clinical-query-reformulator-v1"


@pytest.mark.parametrize("prompt", [None, 42, False, [], {}, "", " \t\n "])
def test_invalid_prompt_is_rejected_before_transport(prompt):
    with pytest.raises(ValueError):
        OllamaTextGenerationBackend().generate(prompt)


@pytest.mark.parametrize("config", [
    [], "temperature=0", 42, False,
    {1: 0}, {"stop": {"END"}}, {"temperature": float("nan")},
    {"top_p": float("inf")}, {"temperature": object()},
])
def test_invalid_config_is_rejected_before_transport(config):
    with pytest.raises(ValueError):
        OllamaTextGenerationBackend().generate(SECRET_PROMPT, generation_config=config)


@pytest.mark.parametrize("kwargs", [
    {"base_url": ""}, {"base_url": None}, {"base_url": 42},
    {"base_url": "ftp://localhost:11434"},
    {"model": ""}, {"model": " \t "}, {"model": None}, {"model": 42},
    {"timeout": 0}, {"timeout": -1}, {"timeout": float("nan")},
    {"timeout": float("inf")}, {"timeout": True},
    {"seed": None}, {"seed": "42"}, {"seed": True}, {"seed": 1.5},
])
def test_invalid_constructor_configuration_is_rejected_before_transport(kwargs):
    with pytest.raises(ValueError):
        OllamaTextGenerationBackend(**kwargs)


@pytest.mark.parametrize("body", [
    SECRET_PROVIDER_TEXT.encode(), b"{", b"\xff",
    b"null", b"[]", b'"response string"', b"42",
])
def test_invalid_json_or_top_level_response_fails_safely(monkeypatch, body):
    install_bytes(monkeypatch, body)

    with pytest.raises(OllamaBackendError) as error:
        OllamaTextGenerationBackend().generate(SECRET_PROMPT)

    assert_safe_error(error.value)


@pytest.mark.parametrize("payload", [
    {"done": True},
    {"response": None, "done": True},
    {"response": 42, "done": True},
    {"response": {}, "done": True},
    {"response": [], "done": True},
    {"response": "", "done": True},
    {"response": " \t\n ", "done": True},
    {"response": SECRET_PROVIDER_TEXT},
    {"response": SECRET_PROVIDER_TEXT, "done": False},
    {"response": SECRET_PROVIDER_TEXT, "done": None},
    {"response": SECRET_PROVIDER_TEXT, "done": "true"},
    {"response": SECRET_PROVIDER_TEXT, "done": 1},
    {"response": "clinical query", "done": True, "error": SECRET_PROVIDER_TEXT},
])
def test_missing_invalid_or_incomplete_output_fails_safely(monkeypatch, payload):
    install_response(monkeypatch, payload)

    with pytest.raises(OllamaBackendError) as error:
        OllamaTextGenerationBackend().generate(SECRET_PROMPT)

    assert_safe_error(error.value)


@pytest.mark.parametrize("failure", [
    URLError(SECRET_PROVIDER_TEXT),
    ConnectionRefusedError(SECRET_PROVIDER_TEXT),
    socket.timeout(SECRET_PROVIDER_TEXT),
    TimeoutError(SECRET_PROVIDER_TEXT),
    HTTPError(
        "http://localhost:11434/api/generate", 500, SECRET_PROVIDER_TEXT,
        {}, io.BytesIO(SECRET_PROVIDER_TEXT.encode()),
    ),
])
def test_connection_http_and_timeout_errors_are_wrapped_safely(monkeypatch, failure):
    def failing_urlopen(request, *, timeout):
        raise failure

    monkeypatch.setattr(ollama_backend, "urlopen", failing_urlopen)

    with pytest.raises(OllamaBackendError) as error:
        OllamaTextGenerationBackend().generate(SECRET_PROMPT)

    assert_safe_error(error.value)


def test_transport_failure_propagates_through_existing_reformulator(monkeypatch):
    install_response(monkeypatch, {"error": SECRET_PROVIDER_TEXT})
    backend = OllamaTextGenerationBackend()
    reformulator = ClinicalQueryReformulator(backend, backend_id=backend.backend_id)

    with pytest.raises(ClinicalQueryError) as error:
        reformulator.reformulate(question_id="q-private", question_text=SECRET_PROMPT)

    assert_safe_error(error.value.__cause__)
    assert SECRET_PROMPT not in str(error.value)
    assert SECRET_PROVIDER_TEXT not in str(error.value)
