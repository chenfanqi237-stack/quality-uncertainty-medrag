"""Optional clinical-query retrieval mode with no network or live model calls."""

from __future__ import annotations

import json

import pytest

from quality_uncertainty_medrag import pubmed
from quality_uncertainty_medrag.clinical_query import ClinicalQueryError, ClinicalQueryReformulator
from quality_uncertainty_medrag.models import MedicalQuestion


class QuestionStemOnly:
    def __init__(self, question_id="q-1", question="Diabetes renal complications"):
        self.question_id = question_id
        self.question = question

    def __getattr__(self, field):
        raise AssertionError(f"Query reformulation must not read {field!r}")


class FakeClient:
    def __init__(self, pmids=("101", "102")):
        self.pmids = pmids
        self.search_calls = []
        self.fetch_calls = []

    def search(self, query, *, top_k):
        self.search_calls.append((query, top_k))
        return {"idlist": list(self.pmids), "count": "123", "querytranslation": query}

    def fetch(self, pmids):
        self.fetch_calls.append(tuple(pmids))
        return {
            pmid: {
                "pmid": pmid, "title": f"Article {pmid}", "abstract": "Clinical abstract",
                "journal": "Mock journal", "publication_date": "2020",
                "publication_types": ["Journal Article"], "pubmed_metadata": {"pmid": pmid},
            }
            for pmid in pmids
        }


class SpyReformulator:
    reformulator_id = "clinical-query-reformulator-v1"
    backend_id = "mock-query-backend"

    def __init__(self, query="diabetes kidney disease clinical management"):
        self.query = query
        self.calls = []

    def reformulate(self, **kwargs):
        self.calls.append(kwargs)
        return self.query


class RecordingBackend:
    def __init__(self, output="diabetes kidney disease clinical management"):
        self.output = output
        self.calls = []

    def generate(self, prompt, *, generation_config=None):
        self.calls.append((prompt, generation_config))
        return self.output


def forbid_keyword_builder(_):
    raise AssertionError("Keyword builder must not run in configured LLM mode")


def cli_args(tmp_path, mode=None):
    args = [
        "--questions", str(tmp_path / "questions.jsonl"), "--limit", "1", "--top-k", "15",
        "--output", str(tmp_path / "evidence.jsonl"), "--report", str(tmp_path / "report.json"),
        "--cache-dir", str(tmp_path / "cache"),
    ]
    if mode is not None:
        args.extend(["--query-mode", mode])
    return args


def test_llm_retrieval_passes_exactly_identifier_and_stem_to_reformulator():
    client = FakeClient()
    reformulator = SpyReformulator()
    question = QuestionStemOnly()
    retriever = pubmed.PubMedRetriever(
        client, query_builder=forbid_keyword_builder, query_mode="llm", query_reformulator=reformulator
    )

    evidence = retriever.retrieve(question, top_k=15)

    assert reformulator.calls == [{"question_id": question.question_id, "question_text": question.question}]
    assert client.search_calls == [(reformulator.query, 15)]
    assert [item.doc_id for item in evidence] == ["101", "102"]
    assert [item.rank for item in evidence] == [1, 2]
    assert all(not item.annotated_stances for item in evidence)
    for record in [item.metadata for item in evidence] + retriever.reports:
        assert record["query"] == reformulator.query
        assert record["query_source"] == "llm_reformulation"
        assert record["query_mode"] == "llm"
        assert record["requested_query_mode"] == "llm"
        assert record["query_reformulator_id"] == reformulator.reformulator_id
        assert record["query_backend_id"] == reformulator.backend_id
    assert all(item.metadata["query_builder"] == reformulator.reformulator_id for item in evidence)


def test_medical_question_gold_options_and_upstream_answer_never_enter_model_prompt():
    sentinels = ["SECRET_OPTION_A", "SECRET_OPTION_B", "SECRET_GOLD", "SECRET_UPSTREAM_ANSWER"]
    question = MedicalQuestion(
        question_id="q-no-leak", question="Diabetes renal complications",
        option_labels=("A", "SECRET_GOLD"), options=tuple(sentinels[:2]), answer_index=1,
        metadata={"upstream": {"answer": sentinels[3], "answer_idx": sentinels[2]}},
    )
    backend = RecordingBackend()
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock")
    client = FakeClient()

    pubmed.PubMedRetriever(client, query_mode="llm", query_reformulator=reformulator).retrieve(
        question, top_k=15
    )

    assert len(backend.calls) == 1
    prompt = backend.calls[0][0]
    assert question.question_id in prompt
    assert question.question in prompt
    assert all(marker not in prompt for marker in sentinels)


def test_zero_results_still_preserve_llm_provenance():
    client = FakeClient(pmids=())
    reformulator = SpyReformulator()
    retriever = pubmed.PubMedRetriever(client, query_mode="llm", query_reformulator=reformulator)

    assert retriever.retrieve(QuestionStemOnly(), top_k=15) == ()

    report = retriever.reports[0]
    assert report["query"] == reformulator.query
    assert report["query_mode"] == "llm"
    assert report["query_reformulator_id"] == reformulator.reformulator_id
    assert report["query_backend_id"] == reformulator.backend_id
    assert report["retrieved_article_count"] == 0


@pytest.mark.parametrize("mode", ["unknown", "LLM", ""])
def test_unknown_query_mode_is_rejected_without_requests(mode):
    client = FakeClient()

    with pytest.raises(ValueError):
        pubmed.PubMedRetriever(client, query_mode=mode)

    assert client.search_calls == client.fetch_calls == []


def test_llm_mode_requires_a_configured_reformulator_before_requests():
    client = FakeClient()

    with pytest.raises(ValueError):
        pubmed.PubMedRetriever(client, query_mode="llm")

    assert client.search_calls == client.fetch_calls == []


@pytest.mark.parametrize("output", [None, "", "Query: diabetes renal complications"])
def test_invalid_llm_output_fails_before_pubmed_search(output):
    client = FakeClient()
    backend = RecordingBackend(output)
    reformulator = ClinicalQueryReformulator(backend, backend_id="mock")
    retriever = pubmed.PubMedRetriever(client, query_mode="llm", query_reformulator=reformulator)

    with pytest.raises(ClinicalQueryError):
        retriever.retrieve(QuestionStemOnly(), top_k=15)

    assert client.search_calls == client.fetch_calls == []
    assert retriever.reports == []


def test_manual_override_has_priority_and_records_actual_manual_mode():
    client = FakeClient()
    reformulator = SpyReformulator()
    exact_query = "  exact manual diagnostic query  "
    retriever = pubmed.PubMedRetriever(
        client, query_builder=forbid_keyword_builder, query_mode="llm", query_reformulator=reformulator,
        query_overrides={"q-manual": exact_query},
    )

    evidence = retriever.retrieve(QuestionStemOnly("q-manual"), top_k=15)

    assert reformulator.calls == []
    assert client.search_calls == [(exact_query, 15)]
    for record in [item.metadata for item in evidence] + retriever.reports:
        assert record["query"] == exact_query
        assert record["query_source"] == "manual_override"
        assert record["query_mode"] == "manual"
        assert record["requested_query_mode"] == "llm"
        assert record["query_reformulator_id"] is None
        assert record["query_backend_id"] is None


def test_partial_manual_override_uses_llm_only_for_non_overridden_questions():
    client = FakeClient()
    reformulator = SpyReformulator()
    retriever = pubmed.PubMedRetriever(
        client, query_mode="llm", query_reformulator=reformulator,
        query_overrides={"q-manual": "manual diagnostic query"},
    )

    retriever.retrieve(QuestionStemOnly("q-manual"), top_k=15)
    retriever.retrieve(QuestionStemOnly("q-llm", "Episodic vomiting"), top_k=15)

    assert reformulator.calls == [{"question_id": "q-llm", "question_text": "Episodic vomiting"}]
    assert client.search_calls == [("manual diagnostic query", 15), (reformulator.query, 15)]


def test_default_and_explicit_keyword_modes_preserve_existing_queries_and_order():
    question = QuestionStemOnly()
    clients = [FakeClient(), FakeClient()]
    reformulator = SpyReformulator()
    retrievers = [
        pubmed.PubMedRetriever(clients[0]),
        pubmed.PubMedRetriever(clients[1], query_mode="keyword", query_reformulator=reformulator),
    ]
    expected = pubmed.build_pubmed_query(question.question)

    results = [retriever.retrieve(question, top_k=15) for retriever in retrievers]

    assert results[0] == results[1]
    assert reformulator.calls == []
    assert [client.search_calls for client in clients] == [[(expected, 15)], [(expected, 15)]]
    assert retrievers[0].reports == retrievers[1].reports
    for record in [item.metadata for item in results[0]] + retrievers[0].reports:
        assert record["query"] == expected
        assert record["query_mode"] == "keyword"
        assert record["query_source"] == "automatic"
        assert record["query_reformulator_id"] is None
        assert record["query_backend_id"] is None


def test_cli_injected_llm_backend_serializes_query_provenance(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    reformulator = SpyReformulator()
    monkeypatch.setattr(pubmed, "load_medqa_questions", lambda _: [QuestionStemOnly()])
    monkeypatch.setattr(pubmed, "PubMedClient", lambda *args, **kwargs: client)

    assert pubmed.main(cli_args(tmp_path, "llm"), query_reformulator=reformulator) == 0

    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == saved
    assert saved[0]["query_mode"] == "llm"
    assert saved[0]["query_backend_id"] == reformulator.backend_id
    rows = [json.loads(line) for line in (tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["metadata"]["query"] == reformulator.query for row in rows)
    assert all(row["metadata"]["query_reformulator_id"] == reformulator.reformulator_id for row in rows)


def test_cli_without_llm_configuration_fails_before_client_construction(tmp_path, monkeypatch):
    def forbidden_client(*args, **kwargs):
        raise AssertionError("Unconfigured LLM mode must fail before client construction")

    monkeypatch.setattr(pubmed, "PubMedClient", forbidden_client)

    with pytest.raises(SystemExit) as error:
        pubmed.main(cli_args(tmp_path, "llm"))

    assert error.value.code == 2
    assert not (tmp_path / "evidence.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


def test_cli_invalid_model_output_writes_no_retrieval_files(tmp_path, monkeypatch):
    client = FakeClient()
    reformulator = ClinicalQueryReformulator(RecordingBackend(""), backend_id="mock")
    monkeypatch.setattr(pubmed, "load_medqa_questions", lambda _: [QuestionStemOnly()])
    monkeypatch.setattr(pubmed, "PubMedClient", lambda *args, **kwargs: client)

    with pytest.raises(SystemExit) as error:
        pubmed.main(cli_args(tmp_path, "llm"), query_reformulator=reformulator)

    assert error.value.code == 2
    assert client.search_calls == client.fetch_calls == []
    assert not (tmp_path / "evidence.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


def test_cli_default_mode_remains_keyword_when_reformulator_is_available(tmp_path, monkeypatch):
    client = FakeClient()
    reformulator = SpyReformulator()
    question = QuestionStemOnly()
    monkeypatch.setattr(pubmed, "load_medqa_questions", lambda _: [question])
    monkeypatch.setattr(pubmed, "PubMedClient", lambda *args, **kwargs: client)

    assert pubmed.main(cli_args(tmp_path), query_reformulator=reformulator) == 0

    assert reformulator.calls == []
    assert client.search_calls == [(pubmed.build_pubmed_query(question.question), 15)]
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved[0]["query_mode"] == "keyword"
