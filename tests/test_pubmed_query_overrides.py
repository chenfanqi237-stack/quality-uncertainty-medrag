"""Manual PubMed query overrides, validated without live network access."""

from __future__ import annotations

import json

import pytest

from quality_uncertainty_medrag import pubmed
from quality_uncertainty_medrag.pubmed_query_overrides import (
    load_query_overrides,
    validate_query_overrides,
)


MANUAL_QUERIES = {
    "medqa-us-dev-000001": "gonococcal arthritis Neisseria gonorrhoeae antibiotic treatment",
    "medqa-us-dev-000002": "cyclic vomiting syndrome children recurrent episodic vomiting",
    "medqa-us-dev-000003": "major depressive disorder insomnia appetite loss treatment",
}


class QuestionTextOnly:
    """Raise if retrieval tries to read options, metadata, or gold labels."""

    def __init__(self, question_id, question="Diabetes renal complications"):
        self.question_id = question_id
        self.question = question

    def __getattr__(self, name):
        raise AssertionError(f"Retrieval must not access question field {name!r}")


class FakeClient:
    def __init__(self):
        self.search_calls = []
        self.fetch_calls = []

    def search(self, query, *, top_k):
        self.search_calls.append((query, top_k))
        return {
            "idlist": [str(pmid) for pmid in range(101, 107)],
            "count": "123",
            "querytranslation": "PubMed automatic translation",
        }

    def fetch(self, pmids):
        self.fetch_calls.append(tuple(pmids))
        return {
            pmid: {
                "pmid": pmid,
                "title": f"Article {pmid}",
                "abstract": f"Biomedical abstract {pmid}",
                "journal": "Example journal",
                "publication_date": "2020",
                "publication_types": ["Journal Article"],
                "pubmed_metadata": {"pmid": pmid},
            }
            for pmid in pmids
        }


def forbidden_builder(_):
    raise AssertionError("The automatic builder must not run for overridden questions")


def test_override_preserves_exact_query_in_search_metadata_and_report():
    query = "  gonococcal arthritis AND treatment  "
    client = FakeClient()
    retriever = pubmed.PubMedRetriever(
        client, query_builder=forbidden_builder, query_overrides={"q-1": query}
    )

    evidence = retriever.retrieve(QuestionTextOnly("q-1", "Unrecognized narrative"), top_k=15)

    assert client.search_calls == [(query, 15)]
    assert all(item.metadata["query"] == query for item in evidence)
    assert all(item.metadata["query_source"] == "manual_override" for item in evidence)
    assert all(item.metadata["query_builder"] == "manual-query-override-v1" for item in evidence)
    assert retriever.reports[0]["query"] == query
    assert retriever.reports[0]["query_source"] == "manual_override"
    assert all(not item.annotated_stances for item in evidence)


def test_partial_overrides_fall_back_to_existing_builder_for_other_questions():
    client = FakeClient()
    builder_calls = []

    def builder(text):
        builder_calls.append(text)
        return pubmed.build_pubmed_query(text)

    retriever = pubmed.PubMedRetriever(
        client, query_builder=builder, query_overrides={"q-manual": "exact diagnostic query"}
    )
    retriever.retrieve(QuestionTextOnly("q-manual"), top_k=15)
    fallback = QuestionTextOnly("q-automatic", "Diabetes renal complications")
    evidence = retriever.retrieve(fallback, top_k=15)

    assert builder_calls == [fallback.question]
    expected = pubmed.build_pubmed_query(fallback.question)
    assert client.search_calls == [("exact diagnostic query", 15), (expected, 15)]
    assert all(item.metadata["query"] == expected for item in evidence)
    assert all(item.metadata["query_source"] == "automatic" for item in evidence)
    assert all(item.metadata["query_builder"] == pubmed.QUERY_VERSION for item in evidence)
    assert retriever.reports[1]["query"] == expected
    assert retriever.reports[1]["query_source"] == "automatic"


def test_retriever_defensively_copies_the_override_mapping():
    overrides = {"q-1": "original diagnostic query"}
    client = FakeClient()
    retriever = pubmed.PubMedRetriever(client, query_builder=forbidden_builder, query_overrides=overrides)
    overrides["q-1"] = "modified by caller"
    overrides.clear()

    retriever.retrieve(QuestionTextOnly("q-1"), top_k=15)

    assert client.search_calls == [("original diagnostic query", 15)]


def test_validation_copies_mapping_without_normalizing_valid_strings():
    original = {" q-1 ": "  diagnostic query  "}
    validated = validate_query_overrides(original)

    assert validated == original
    assert validated is not original
    original.clear()
    assert validated == {" q-1 ": "  diagnostic query  "}


@pytest.mark.parametrize(
    "value", [[], None, {"": "query"}, {"   ": "query"}, {1: "query"},
              {"q": None}, {"q": 42}, {"q": ""}, {"q": "\t\n"}]
)
def test_invalid_override_mappings_are_rejected(value):
    with pytest.raises(ValueError):
        validate_query_overrides(value)


def test_empty_override_mapping_is_valid():
    assert validate_query_overrides({}) == {}


def test_load_override_json_preserves_exact_queries(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(MANUAL_QUERIES), encoding="utf-8")

    assert load_query_overrides(path) == MANUAL_QUERIES


@pytest.mark.parametrize(
    "content", ["not JSON", "[]", "null", '{"q": null}', '{"q": 7}', '{"q": " "}']
)
def test_invalid_json_override_files_are_rejected(tmp_path, content):
    path = tmp_path / "overrides.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_query_overrides(path)


def test_duplicate_question_ids_in_override_json_are_rejected(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text('{"q-1": "first", "q-1": "second"}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_query_overrides(path)


def test_cli_manual_experiment_retrieves_only_first_three_and_prints_exact_report(tmp_path, monkeypatch, capsys):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps(MANUAL_QUERIES), encoding="utf-8")
    question_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "evidence.jsonl"
    report_path = tmp_path / "report.json"
    questions = [QuestionTextOnly(question_id, "Unrecognized narrative") for question_id in MANUAL_QUERIES]
    questions.append(QuestionTextOnly("medqa-us-dev-000004"))
    client = FakeClient()
    monkeypatch.setattr(pubmed, "load_medqa_questions", lambda _: questions)
    monkeypatch.setattr(pubmed, "PubMedClient", lambda *args, **kwargs: client)

    assert pubmed.main([
        "--questions", str(question_path), "--query-overrides", str(overrides_path),
        "--limit", "3", "--top-k", "15", "--output", str(output_path),
        "--report", str(report_path), "--cache-dir", str(tmp_path / "cache"),
    ]) == 0

    assert client.search_calls == [(query, 15) for query in MANUAL_QUERIES.values()]
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed == saved
    assert len(printed) == 3
    for item, (question_id, query) in zip(printed, MANUAL_QUERIES.items()):
        assert item["question_id"] == question_id
        assert item["query"] == query
        assert item["pubmed_total_match_count"] == "123"
        assert item["top_5"] == [
            {"pmid": str(pmid), "title": f"Article {pmid}"} for pmid in range(101, 106)
        ]
        assert item["query_source"] == "manual_override"
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 18
    assert {row["question_id"] for row in rows} == set(MANUAL_QUERIES)
    assert all(row["metadata"]["query"] == MANUAL_QUERIES[row["question_id"]] for row in rows)
    assert all("annotated_stances" not in row for row in rows)


@pytest.mark.parametrize("content", ["not JSON", "[]", "null", '{"q": null}'])
def test_cli_invalid_override_file_fails_before_client_construction(tmp_path, monkeypatch, content):
    path = tmp_path / "overrides.json"
    path.write_text(content, encoding="utf-8")

    def forbidden_client(*args, **kwargs):
        raise AssertionError("Invalid override file must fail before any request client is created")

    monkeypatch.setattr(pubmed, "PubMedClient", forbidden_client)
    monkeypatch.setattr(pubmed, "load_medqa_questions", lambda _: [QuestionTextOnly("q-1")])
    with pytest.raises(SystemExit) as error:
        pubmed.main([
            "--questions", str(tmp_path / "questions.jsonl"), "--query-overrides", str(path),
            "--output", str(tmp_path / "evidence.jsonl"), "--report", str(tmp_path / "report.json"),
        ])

    assert error.value.code == 2
    assert not (tmp_path / "evidence.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("collision_flag", ["--output", "--report"])
def test_cli_prevents_outputs_from_overwriting_override_file(tmp_path, monkeypatch, collision_flag):
    path = tmp_path / "overrides.json"
    original = json.dumps(MANUAL_QUERIES)
    path.write_text(original, encoding="utf-8")

    def forbidden_client(*args, **kwargs):
        raise AssertionError("Path collision must fail before client construction")

    monkeypatch.setattr(pubmed, "PubMedClient", forbidden_client)
    with pytest.raises(SystemExit) as error:
        pubmed.main([
            "--questions", str(tmp_path / "questions.jsonl"), "--query-overrides", str(path),
            "--output", str(tmp_path / "evidence.jsonl"), "--report", str(tmp_path / "report.json"),
            collision_flag, str(path),
        ])

    assert error.value.code == 2
    assert path.read_text(encoding="utf-8") == original
