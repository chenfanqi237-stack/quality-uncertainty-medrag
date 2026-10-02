"""Clinical query reformulation contracts, using only mocked generation."""

from __future__ import annotations

import json

import pytest

from quality_uncertainty_medrag.clinical_query import (
    ClinicalQueryError,
    ClinicalQueryReformulator,
    normalize_query_output,
)


class RecordingBackend:
    def __init__(self, output="gonococcal arthritis antibiotic treatment"):
        self.output = output
        self.calls = []

    def generate(self, prompt, *, generation_config=None):
        self.calls.append((prompt, generation_config))
        return self.output


def prompt_records(prompt):
    """Inspect JSON input without coupling tests to surrounding instruction text."""
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


def test_reformulator_prompt_contains_only_question_identifier_and_stem():
    backend = RecordingBackend()
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock-clinical-model")
    stem = "Fever, dysuria, and migratory septic arthritis with a non-maltose-fermenting organism."

    assert reformulator.reformulate(question_id="q-clinical", question_text=stem) == backend.output

    prompt, config = backend.calls[0]
    assert prompt_records(prompt) == [{"question_id": "q-clinical", "question_text": stem}]
    assert config["temperature"] == 0
    assert reformulator.backend_id == "mock-clinical-model"
    assert reformulator.reformulator_id == "clinical-query-reformulator-v1"


def test_json_input_preserves_stem_quotes_and_line_breaks():
    backend = RecordingBackend()
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock")
    stem = 'She describes "episodic vomiting".\nNo symptoms between episodes.'

    reformulator.reformulate(question_id="q-quoted", question_text=stem)

    assert prompt_records(backend.calls[0][0]) == [
        {"question_id": "q-quoted", "question_text": stem}
    ]


def test_generation_configuration_is_fresh_and_normalization_is_deterministic():
    backend = RecordingBackend('  "gonococcal   arthritis\tantibiotic treatment"  ')
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock")

    first = reformulator.reformulate(question_id="q-1", question_text="Septic arthritis.")
    backend.calls[0][1]["temperature"] = 1
    second = reformulator.reformulate(question_id="q-1", question_text="Septic arthritis.")

    assert first == second == "gonococcal arthritis antibiotic treatment"
    assert backend.calls[0][0] == backend.calls[1][0]
    assert backend.calls[0][1] is not backend.calls[1][1]
    assert backend.calls[1][1]["temperature"] == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  recurrent\t episodic  vomiting  ", "recurrent episodic vomiting"),
        ('"cyclic vomiting syndrome children"', "cyclic vomiting syndrome children"),
        ("'major depressive disorder insomnia'", "major depressive disorder insomnia"),
        ('"major depressive disorder"[Title/Abstract] AND insomnia',
         '"major depressive disorder"[Title/Abstract] AND insomnia'),
        ('"diabetes mellitus" AND "renal disease"',
         '"diabetes mellitus" AND "renal disease"'),
    ],
)
def test_valid_output_normalization_preserves_pubmed_phrases_and_fields(raw, expected):
    assert normalize_query_output(raw) == expected
    assert normalize_query_output(normalize_query_output(raw)) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None, 42, False, [], {}, "", " \t\n ", '""', "''",
        "arthritis\ntreatment", "arthritis\rtreatment",
        '```\narthritis treatment\n```', '{"query": "arthritis treatment"}',
        '["arthritis treatment"]', "- arthritis treatment", "1. arthritis treatment",
        "Query: arthritis treatment", "Explanation: arthritis treatment",
        "Query = arthritis treatment", "This query targets arthritis treatment",
    ],
)
def test_invalid_output_is_rejected_without_a_query(raw):
    with pytest.raises(ClinicalQueryError):
        normalize_query_output(raw)


@pytest.mark.parametrize("raw", [None, "", "Query: arthritis treatment"])
def test_reformulator_rejects_invalid_backend_output(raw):
    backend = RecordingBackend(raw)
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock")

    with pytest.raises(ClinicalQueryError):
        reformulator.reformulate(question_id="q-1", question_text="Septic arthritis.")

    assert len(backend.calls) == 1


def test_backend_failure_is_reported_without_manufacturing_a_query():
    class FailingBackend:
        def generate(self, prompt, *, generation_config=None):
            raise RuntimeError("mock provider unavailable")

    reformulator = ClinicalQueryReformulator(FailingBackend(), backend_id="mock")

    with pytest.raises(ClinicalQueryError):
        reformulator.reformulate(question_id="q-1", question_text="Septic arthritis.")


@pytest.mark.parametrize(
    ("question_id", "question_text"),
    [(None, "Valid stem"), ("", "Valid stem"), ("q", None), ("q", 42), ("q", " ")],
)
def test_invalid_inputs_fail_before_generation(question_id, question_text):
    backend = RecordingBackend()
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock")

    with pytest.raises(ClinicalQueryError):
        reformulator.reformulate(question_id=question_id, question_text=question_text)

    assert backend.calls == []
