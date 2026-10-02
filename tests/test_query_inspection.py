"""Stem-only inspection, durable query reuse and safe Ollama transport, offline."""

from __future__ import annotations

import io
import json

import pytest

from quality_uncertainty_medrag import ollama_backend, query_inspection
from quality_uncertainty_medrag.clinical_query import ClinicalQueryError


BASE_URL = "http://localhost:11434"
MODEL = "qwen3:8b"
DIGEST = "mock-qwen-model-digest"
REASONING_SECRET = "THINKING_SECRET_MUST_NEVER_BE_RETAINED"
FORBIDDEN_INPUTS = (
    "OPTIONS_SECRET_NOT_FOR_QUERY",
    "ANSWER_SECRET_NOT_FOR_QUERY",
    "ANSWER_IDX_SECRET_NOT_FOR_QUERY",
    "UPSTREAM_GOLD_SECRET_NOT_FOR_QUERY",
)


@pytest.fixture(autouse=True)
def no_unmocked_network(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Inspection tests must mock every network request")

    monkeypatch.setattr(query_inspection, "urlopen", unexpected)
    monkeypatch.setattr(ollama_backend, "urlopen", unexpected)


def questions(count=3):
    return [
        {
            "id": f"question-{index}",
            "question": f"Distinct clinical stem {index} with discriminative findings.",
            "options": {"A": FORBIDDEN_INPUTS[0]},
            "answer": FORBIDDEN_INPUTS[1],
            "answer_idx": FORBIDDEN_INPUTS[2],
            "metadata": {"upstream": {"answer": FORBIDDEN_INPUTS[3]}},
        }
        for index in range(1, count + 1)
    ]


def write_questions(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def arguments(tmp_path, count=3):
    return {
        "input_path": tmp_path / "questions.jsonl",
        "output_path": tmp_path / "inspection.json",
        "cache_dir": tmp_path / "cache",
        "limit": count,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def install_ollama(monkeypatch, queries, *, on_generation=None, digest=DIGEST):
    metadata_calls = []
    generation_calls = []

    def metadata(url, *, timeout):
        metadata_calls.append(url)
        if url == BASE_URL + "/api/tags":
            body = {"models": [{"name": MODEL, "model": MODEL, "digest": digest}]}
        elif url == BASE_URL + "/api/version":
            body = {"version": "mock-ollama-version"}
        else:
            pytest.fail("Only local Ollama model metadata may be requested")
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    def generate(request, *, timeout):
        assert request.full_url == BASE_URL + "/api/generate"
        assert request.get_method() == "POST"
        body = json.loads(request.data.decode("utf-8"))
        data = json.loads(body["prompt"].split("QUESTION INPUT (JSON):\n", 1)[1])
        generation_calls.append((body, data))
        if on_generation is not None:
            on_generation(generation_calls)
        query = queries[data["question_id"]]
        payload = {
            "model": MODEL, "done": True, "response": query,
            "thinking": REASONING_SECRET,
        }
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(query_inspection, "urlopen", metadata)
    monkeypatch.setattr(ollama_backend, "urlopen", generate)
    return metadata_calls, generation_calls


def test_inspection_generates_once_per_stem_and_persists_exact_queries_immediately(
    tmp_path, monkeypatch, capsys,
):
    params = arguments(tmp_path)
    records = questions()
    write_questions(params["input_path"], records)
    expected = {
        "question-1": '"clinical phrase"[Title/Abstract] AND antibiotic treatment',
        "question-2": "Cyclic vomiting syndrome children recurrent episodes",
        "question-3": "major depressive disorder insomnia treatment",
    }

    def verify_completed_queries_are_already_durable(calls):
        completed = len(calls) - 1
        entries = [read_json(path) for path in params["cache_dir"].glob("*.json")]
        assert len(entries) == completed
        assert {entry["question_id"]: entry["generated_query"] for entry in entries} == {
            record["id"]: expected[record["id"]] for record in records[:completed]
        }
        if completed:
            report = read_json(params["output_path"])
            assert report["completed_question_count"] == completed
            assert [row["generated_query"] for row in report["results"]] == [
                expected[record["id"]] for record in records[:completed]
            ]

    metadata_calls, generation_calls = install_ollama(
        monkeypatch, expected, on_generation=verify_completed_queries_are_already_durable,
    )
    report = query_inspection.run_inspection(**params)

    assert len(generation_calls) == len(records)
    assert all(url.startswith(BASE_URL + "/api/") for url in metadata_calls)
    for (body, data), record in zip(generation_calls, records):
        assert data == {"question_id": record["id"], "question_text": record["question"]}
        assert set(data) == {"question_id", "question_text"}
        assert all(secret not in body["prompt"] for secret in FORBIDDEN_INPUTS)
        assert body["model"] == MODEL
        assert body["stream"] is False
        assert body["think"] is True
        assert body["options"] == {"temperature": 0.0, "seed": 42}

    assert report["status"] == "complete"
    assert report["completed_question_count"] == len(records)
    assert report["model_calls_this_run"] == len(records)
    assert report["cache_hits_this_run"] == 0
    assert report["pubmed_called"] is False
    assert report["reasoning_retained"] is False
    assert report["model_digest"] == DIGEST
    assert report["ollama_version"] == "mock-ollama-version"
    assert read_json(params["output_path"]) == report
    assert [row["question_id"] for row in report["results"]] == [r["id"] for r in records]
    for row in report["results"]:
        assert row["generated_query"] == expected[row["question_id"]]
        assert row["backend_id"] == "ollama/qwen3:8b"
        assert row["model"] == MODEL
        assert row["thinking_mode"] == "ON"
        assert row["temperature"] == 0.0
        assert row["seed"] == 42
        assert row["model_digest"] == DIGEST
        assert row["generation_parameters"] == {
            "temperature": 0.0, "seed": 42, "think": True, "stream": False,
        }
        assert row["model_calls_this_run"] == 1
        assert row["cache_hit"] is False
        assert len(row["prompt_sha256"]) == len(row["cache_key"]) == 64

    printed = capsys.readouterr()
    assert printed.err == ""
    assert printed.out.strip().splitlines() == [
        "question_id\tgenerated_query",
        *(record["id"] + "\t" + expected[record["id"]] for record in records),
    ]
    saved_texts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.json")
        if path != params["input_path"]
    )
    assert REASONING_SECRET not in saved_texts + printed.out + printed.err
    assert all(secret not in saved_texts for secret in FORBIDDEN_INPUTS)
    assert all(record["question"] not in saved_texts for record in records)


def test_same_context_inspection_reuses_exact_queries_without_model_calls(tmp_path, monkeypatch):
    params = arguments(tmp_path, count=2)
    write_questions(params["input_path"], questions(2))
    expected = {"question-1": "Specific Clinical Concept treatment", "question-2": "different clinical mechanism"}
    _, calls = install_ollama(monkeypatch, expected)
    first = query_inspection.run_inspection(**params)
    output_bytes = params["output_path"].read_bytes()
    cache_bytes = {path.name: path.read_bytes() for path in params["cache_dir"].glob("*.json")}
    first_queries = [row["generated_query"] for row in first["results"]]

    def must_not_generate(request, *, timeout):
        pytest.fail("A saved query must never be regenerated for the same context")

    monkeypatch.setattr(ollama_backend, "urlopen", must_not_generate)
    second = query_inspection.run_inspection(**params)

    assert len(calls) == 2
    assert [row["generated_query"] for row in second["results"]] == first_queries
    assert params["output_path"].read_bytes() == output_bytes
    assert {path.name: path.read_bytes() for path in params["cache_dir"].glob("*.json")} == cache_bytes


def test_saved_report_reuses_exact_queries_even_if_cache_files_are_missing(tmp_path, monkeypatch):
    params = arguments(tmp_path, count=2)
    write_questions(params["input_path"], questions(2))
    expected = {"question-1": "Specific Clinical Concept treatment", "question-2": "different clinical mechanism"}
    install_ollama(monkeypatch, expected)
    first = query_inspection.run_inspection(**params)
    first_bytes = params["output_path"].read_bytes()
    for path in params["cache_dir"].glob("*.json"):
        path.unlink()

    def must_not_generate(request, *, timeout):
        pytest.fail("A saved inspection query must be reused even if its cache entry is missing")

    monkeypatch.setattr(ollama_backend, "urlopen", must_not_generate)
    reused = query_inspection.run_inspection(**params)

    assert reused["results"] == first["results"]
    assert params["output_path"].read_bytes() == first_bytes


@pytest.mark.parametrize("corruption", ["invalid_query", "wrong_digest", "duplicate_row", "reasoning_field"])
def test_invalid_saved_report_fails_before_regeneration_and_preserves_file(tmp_path, monkeypatch, corruption):
    params = arguments(tmp_path, count=1)
    write_questions(params["input_path"], questions(1))
    install_ollama(monkeypatch, {"question-1": "specific clinical concept"})
    report = query_inspection.run_inspection(**params)
    if corruption == "invalid_query":
        report["results"][0]["generated_query"] = "Query: unsafe wrapper"
    elif corruption == "wrong_digest":
        report["results"][0]["model_digest"] = "different-digest"
    elif corruption == "duplicate_row":
        report["results"].append(dict(report["results"][0]))
    else:
        report["results"][0]["thinking"] = REASONING_SECRET
    params["output_path"].write_text(json.dumps(report), encoding="utf-8")
    invalid_bytes = params["output_path"].read_bytes()

    def must_not_generate(request, *, timeout):
        pytest.fail("Malformed saved results must fail instead of regeneration")

    monkeypatch.setattr(ollama_backend, "urlopen", must_not_generate)
    with pytest.raises(ValueError):
        query_inspection.run_inspection(**params)
    assert params["output_path"].read_bytes() == invalid_bytes


@pytest.mark.parametrize("change", ["seed", "thinking", "stem", "digest"])
def test_changed_inspection_context_does_not_overwrite_or_generate(tmp_path, monkeypatch, change):
    params = arguments(tmp_path, count=1)
    records = questions(1)
    write_questions(params["input_path"], records)
    _, calls = install_ollama(monkeypatch, {"question-1": "clinical condition treatment"})
    query_inspection.run_inspection(**params)
    original_output = params["output_path"].read_bytes()
    cache_bytes = {path.name: path.read_bytes() for path in params["cache_dir"].glob("*.json")}
    if change == "seed":
        params["seed"] = 43
    elif change == "thinking":
        params["think"] = False
    elif change == "stem":
        records[0]["question"] = "Changed clinically meaningful stem."
        write_questions(params["input_path"], records)
    else:
        install_ollama(monkeypatch, {}, digest="different-model-digest")

    def must_not_generate(request, *, timeout):
        pytest.fail("A conflicting inspection path must fail before generation")

    monkeypatch.setattr(ollama_backend, "urlopen", must_not_generate)
    with pytest.raises(ValueError):
        query_inspection.run_inspection(**params)

    assert len(calls) == 1
    assert params["output_path"].read_bytes() == original_output
    assert {path.name: path.read_bytes() for path in params["cache_dir"].glob("*.json")} == cache_bytes


@pytest.mark.parametrize(
    "records",
    [
        [{"id": None, "question": "Valid stem"}],
        [{"id": 7, "question": "Valid stem"}],
        [{"id": " ", "question": "Valid stem"}],
        [{"id": "valid", "question": None}],
        [{"id": "valid", "question": ["not", "text"]}],
        [{"id": "valid", "question": "\t"}],
        [{"id": "valid"}],
        [None],
        [],
    ],
)
def test_malformed_inputs_fail_before_any_network_or_output_writes(tmp_path, records):
    params = arguments(tmp_path, count=1)
    write_questions(params["input_path"], records)
    with pytest.raises((ValueError, KeyError, TypeError)):
        query_inspection.run_inspection(**params)
    assert not params["output_path"].exists()
    assert not params["cache_dir"].exists()


def test_duplicate_identifiers_fail_before_any_network(tmp_path):
    params = arguments(tmp_path, count=2)
    records = questions(2)
    records[1]["id"] = records[0]["id"]
    write_questions(params["input_path"], records)
    with pytest.raises(ValueError):
        query_inspection.run_inspection(**params)
    assert not params["output_path"].exists()
    assert not params["cache_dir"].exists()


def test_invalid_generation_exhausts_retries_and_preserves_completed_query(tmp_path, monkeypatch):
    params = arguments(tmp_path, count=2)
    write_questions(params["input_path"], questions(2))
    expected = {"question-1": "Specific clinical concept treatment", "question-2": "Query: invalid wrapper"}
    _, calls = install_ollama(monkeypatch, expected)

    with pytest.raises(ClinicalQueryError):
        query_inspection.run_inspection(**params)

    assert len(calls) == 4
    report = read_json(params["output_path"])
    assert report["completed_question_count"] == 1
    assert [row["generated_query"] for row in report["results"]] == [expected["question-1"]]
    entries = [read_json(path) for path in params["cache_dir"].glob("*.json")]
    assert len(entries) == 1
    assert entries[0]["question_id"] == "question-1"
    assert entries[0]["generated_query"] == expected["question-1"]
    assert REASONING_SECRET not in params["output_path"].read_text(encoding="utf-8")


def test_partial_inspection_resumes_only_missing_questions_and_preserves_completed_row(tmp_path, monkeypatch):
    params = arguments(tmp_path, count=3)
    write_questions(params["input_path"], questions(3))
    _, initial_calls = install_ollama(
        monkeypatch,
        {"question-1": "Specific Clinical Concept treatment", "question-2": "Query: invalid wrapper"},
    )
    with pytest.raises(ClinicalQueryError):
        query_inspection.run_inspection(**params)
    previous = read_json(params["output_path"])
    completed_row = previous["results"][0]
    _, resumed_calls = install_ollama(
        monkeypatch,
        {"question-2": "different clinical mechanism", "question-3": "another clinical treatment"},
    )

    resumed = query_inspection.run_inspection(**params)

    assert [data["question_id"] for _, data in initial_calls] == [
        "question-1", "question-2", "question-2", "question-2",
    ]
    assert [data["question_id"] for _, data in resumed_calls] == ["question-2", "question-3"]
    assert resumed["results"][0] == completed_row
    assert [row["question_id"] for row in resumed["results"]] == ["question-1", "question-2", "question-3"]
    assert resumed["status"] == "complete"
    assert resumed["completed_question_count"] == 3
    assert len(list(params["cache_dir"].glob("*.json"))) == 3


def test_reasoning_tags_in_final_response_fail_without_saving_the_response(tmp_path, monkeypatch, capsys):
    params = arguments(tmp_path, count=1)
    write_questions(params["input_path"], questions(1))
    _, calls = install_ollama(
        monkeypatch, {"question-1": "<think>" + REASONING_SECRET + "</think>query"},
    )
    with pytest.raises(ClinicalQueryError):
        query_inspection.run_inspection(**params)
    assert len(calls) == 3
    assert list(params["cache_dir"].glob("*.json")) == []
    all_saved = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json"))
    output = capsys.readouterr()
    assert REASONING_SECRET not in all_saved + output.out + output.err


def test_inspection_accepts_recovered_query_and_reuses_saved_retry_diagnostics(tmp_path, monkeypatch):
    params = arguments(tmp_path, count=1)
    write_questions(params["input_path"], questions(1))

    def two_failures(calls):
        if len(calls) <= 2:
            raise TimeoutError("Temporary generation failure")

    _, calls = install_ollama(
        monkeypatch, {"question-1": "named clinical concept"}, on_generation=two_failures,
    )
    report = query_inspection.run_inspection(**params)
    row = report["results"][0]
    assert row["generated_query"] == "named clinical concept"
    assert row["model_calls_this_run"] == 3
    assert row["query_generation"]["retry_count"] == 2
    assert [attempt["status"] for attempt in row["query_generation"]["attempts"]] == [
        "failed", "failed", "success",
    ]
    assert calls[0] == calls[1] == calls[2]
    output_bytes = params["output_path"].read_bytes()
    assert query_inspection.run_inspection(**params) == report
    assert len(calls) == 3
    assert params["output_path"].read_bytes() == output_bytes


def test_legacy_inspection_report_remains_reusable(tmp_path, monkeypatch):
    params = arguments(tmp_path, count=1)
    write_questions(params["input_path"], questions(1))
    _, calls = install_ollama(monkeypatch, {"question-1": "named clinical concept"})
    report = query_inspection.run_inspection(**params)
    report["results"][0].pop("query_generation")
    query_inspection._save(params["output_path"], report)
    assert query_inspection.run_inspection(**params) == report
    assert len(calls) == 1
