"""Mocked threshold, isolation, cache and retrieval provenance checks."""

import hashlib
import io
import json

import pytest

from quality_uncertainty_medrag.clinical_query import ClinicalQueryError
from quality_uncertainty_medrag.clinical_query_relaxation import CoreClinicalQueryReformulator
from quality_uncertainty_medrag.pubmed import evidence_to_json_record
from quality_uncertainty_medrag.query_cache import CachedClinicalQueryReformulator, QueryCacheError
from quality_uncertainty_medrag.query_relaxation import (
    GENERATION_SETTINGS, VALIDATION_QUESTION_IDS, QueryRelaxationRetriever,
    _validation_inputs, main,
)


class Backend:
    def __init__(self, output="Core disease treatment", events=None):
        self.output = output
        self.calls = []
        self.events = events if events is not None else []

    def generate(self, prompt, *, generation_config=None):
        self.calls.append((prompt, generation_config))
        self.events.append(("model",))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class Client:
    def __init__(self, primary_count=0, fallback_count=20, events=None):
        self.primary_count = primary_count
        self.fallback_count = fallback_count
        self.events = events if events is not None else []
        self.request_count = 0

    def search(self, query, *, top_k):
        self.events.append(("search", query, top_k))
        self.request_count += 1
        count = self.primary_count if query == "Exact PRIMARY" else self.fallback_count
        return {"count": str(count), "idlist": ["12345"] if count else [], "querytranslation": query}

    def fetch(self, pmids):
        self.events.append(("fetch", tuple(pmids)))
        return {pmid: {
            "pmid": pmid, "title": "Realistic mocked title", "abstract": "Mock abstract",
            "journal": "Mock journal", "publication_date": "2024",
            "publication_types": ["Journal Article"], "pubmed_metadata": {},
        } for pmid in pmids}


def cached(tmp_path, backend, *, core=True):
    kwargs = {"reformulator_factory": CoreClinicalQueryReformulator} if core else {}
    return CachedClinicalQueryReformulator(
        backend, backend_id="ollama/qwen3:8b", cache_dir=tmp_path,
        generation_settings=GENERATION_SETTINGS, model_digest="digest-a", **kwargs,
    )


def retrieve(stage, **kwargs):
    args = dict(question_id="q1", question_text="Clinical clues only.", primary_query="Exact PRIMARY", top_k=15)
    args.update(kwargs)
    return stage.retrieve(**args)


def test_core_prompt_receives_only_id_and_stem_and_fixed_generation_settings():
    backend = Backend()
    core = CoreClinicalQueryReformulator(backend, backend_id="ollama/qwen3:8b")
    assert core.reformulate(question_id="q1", question_text="Only the clinical stem.") == backend.output
    prompt, config = backend.calls[0]
    assert json.loads(prompt.split("QUESTION INPUT (JSON):\n", 1)[1]) == {
        "question_id": "q1", "question_text": "Only the clinical stem.",
    }
    assert "2-5" in prompt and "nonessential" in prompt
    assert config == {"temperature": 0.0, "seed": 42}
    assert core.reformulator_id != "clinical-query-reformulator-v1"


def test_core_prompt_preserves_primary_concept_and_receives_only_allowed_fields():
    backend = Backend()
    primary = "gonococcal arthritis non-maltose fermenting no polysaccharide capsule"
    core = CoreClinicalQueryReformulator(backend, backend_id="ollama/qwen3:8b")
    core.reformulate(question_id="q1", question_text="The clinical stem.", primary_query=primary)
    prompt, config = backend.calls[0]
    assert json.loads(prompt.split("QUESTION INPUT (JSON):\n", 1)[1]) == {
        "question_id": "q1", "question_text": "The clinical stem.", "primary_query": primary,
    }
    assert "REMOVE SECONDARY RESTRICTIONS" in prompt
    assert "Preserve the likely disease, syndrome, organism or named mechanism" in prompt
    assert "gonococcal arthritis Neisseria gonorrhoeae" in prompt
    assert "cyclic vomiting syndrome children" in prompt
    assert config == {"temperature": 0.0, "seed": 42}


@pytest.mark.parametrize("primary", ["", " ", 123, True])
def test_invalid_primary_context_fails_before_generation(primary):
    backend = Backend()
    with pytest.raises(ClinicalQueryError, match="primary query"):
        CoreClinicalQueryReformulator(backend, backend_id="mock").reformulate(
            question_id="q1", question_text="Stem", primary_query=primary,
        )
    assert backend.calls == []


def test_primary_query_is_part_of_fallback_cache_identity(tmp_path):
    backend = Backend()
    reformulator = cached(tmp_path, backend)
    reformulator.reformulate(question_id="q1", question_text="Stem", primary_query="Core disease with modifier")
    first_key = reformulator.last_record["cache_key"]
    reformulator.reformulate(question_id="q1", question_text="Stem", primary_query="Different named mechanism")
    assert reformulator.last_record["cache_key"] != first_key
    assert len(backend.calls) == 2
    reformulator.reformulate(question_id="q1", question_text="Stem", primary_query="Core disease with modifier")
    assert reformulator.last_record["cache_hit"] is True
    assert reformulator.last_record["cache_key"] == first_key
    assert len(backend.calls) == 2


def test_retrieval_forwards_exact_primary_query_without_other_question_fields(tmp_path):
    backend = Backend()
    stage = QueryRelaxationRetriever(Client(), fallback_reformulator=cached(tmp_path, backend))
    retrieve(stage)
    prompt, _ = backend.calls[0]
    assert json.loads(prompt.split("QUESTION INPUT (JSON):\n", 1)[1]) == {
        "question_id": "q1", "question_text": "Clinical clues only.", "primary_query": "Exact PRIMARY",
    }


@pytest.mark.parametrize("extra", ["options", "answer", "answer_idx", "metadata", "upstream_answer", "gold_labels"])
def test_core_api_does_not_accept_options_or_gold_fields(extra):
    backend = Backend()
    core = CoreClinicalQueryReformulator(backend, backend_id="mock")
    with pytest.raises(TypeError):
        core.reformulate(question_id="q1", question_text="Stem", primary_query="Exact PRIMARY",
                         **{extra: "must not be exposed"})
    assert backend.calls == []


@pytest.mark.parametrize("count, triggered", [(0, True), (4, True), (5, False), (50, False)])
def test_primary_search_precedes_one_fallback_only_below_five(tmp_path, count, triggered):
    events = []
    backend = Backend(events=events)
    client = Client(count, events=events)
    stage = QueryRelaxationRetriever(client, fallback_reformulator=cached(tmp_path, backend))
    result = retrieve(stage)
    assert events[0] == ("search", "Exact PRIMARY", 15)
    assert len(backend.calls) == int(triggered)
    assert result.report["fallback_triggered"] is triggered
    assert result.report["primary_pubmed_match_count"] == count
    if triggered:
        assert events[1:3] == [("model",), ("search", "Core disease treatment", 15)]
        assert result.report["fallback_pubmed_match_count"] == 20
        assert result.report["query_stage_used"] == "fallback"
    else:
        assert result.report["fallback_query"] is None
        assert result.report["fallback_pubmed_match_count"] is None
        assert result.report["fallback_query_cache_hit"] is None
        assert result.report["query_used"] == "Exact PRIMARY"
        assert sum(event[0] == "search" for event in events) == 1


def test_fallback_cache_survives_new_instance_and_retains_exact_query(tmp_path):
    first_backend = Backend('  "Core  Disease\tTreatment"  ')
    first = QueryRelaxationRetriever(Client(), fallback_reformulator=cached(tmp_path, first_backend))
    original = retrieve(first)
    unused = Backend(RuntimeError("Model must not be called"))
    second = QueryRelaxationRetriever(Client(), fallback_reformulator=cached(tmp_path, unused))
    repeated = retrieve(second)
    assert original.report["fallback_query"] == repeated.report["fallback_query"] == "Core Disease Treatment"
    assert original.report["fallback_query_cache_hit"] is False
    assert repeated.report["fallback_query_cache_hit"] is True
    assert repeated.report["fallback_cache_key"] == original.report["fallback_cache_key"]
    assert len(first_backend.calls) == 1 and unused.calls == []


def test_primary_and_fallback_have_separate_cache_contexts(tmp_path):
    backend = Backend()
    primary = cached(tmp_path, backend, core=False)
    fallback = cached(tmp_path, backend)
    primary.reformulate(question_id="q1", question_text="Stem")
    fallback.reformulate(question_id="q1", question_text="Stem")
    assert primary.last_record["cache_key"] != fallback.last_record["cache_key"]
    assert primary.reformulator_id == "clinical-query-reformulator-v1"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_both_queries_counts_and_query_used_survive_evidence_json_serialization(tmp_path):
    result = retrieve(QueryRelaxationRetriever(Client(2, 30), fallback_reformulator=cached(tmp_path, Backend())))
    record = json.loads(json.dumps(evidence_to_json_record(result.evidence[0])))
    metadata = record["metadata"]
    assert metadata["primary_query"] == "Exact PRIMARY"
    assert metadata["fallback_query"] == metadata["query_used"] == metadata["query"] == "Core disease treatment"
    assert metadata["primary_pubmed_match_count"] == 2
    assert metadata["fallback_pubmed_match_count"] == 30
    assert metadata["query_backend_id"] == "ollama/qwen3:8b"
    assert metadata["query_reformulator_id"] == "core-clinical-query-reformulator-v2"
    assert "annotated_stances" not in record
    assert record["rank"] == 1
    assert result.report["top_5"] == [{"pmid": "12345", "title": "Realistic mocked title"}]


def test_zero_fallback_matches_end_after_minimal_stage(tmp_path):
    backend = Backend()
    client = Client(0, 0)
    result = retrieve(QueryRelaxationRetriever(client, fallback_reformulator=cached(tmp_path, backend)))
    assert result.evidence == ()
    assert result.report["fallback_pubmed_match_count"] == 0
    assert result.report["query_stage_used"] == "minimal_fallback"
    assert result.report["minimal_fallback_pubmed_match_count"] == 0
    assert len(backend.calls) == 2
    assert len([event for event in client.events if event[0] == "search"]) == 3


@pytest.mark.parametrize("output", ["", "Query: disease treatment", "disease\nexplanation", "<think>reason</think> disease"])
def test_invalid_output_exhausts_retries_without_fallback_search(tmp_path, output):
    backend = Backend(output)
    client = Client()
    with pytest.raises(ClinicalQueryError):
        retrieve(QueryRelaxationRetriever(client, fallback_reformulator=cached(tmp_path, backend)))
    assert len(backend.calls) == 3
    assert client.events == [("search", "Exact PRIMARY", 15)]
    assert list(tmp_path.glob("*.json")) == []


def test_corrupt_fallback_cache_fails_without_regeneration(tmp_path):
    backend = Backend()
    stage = QueryRelaxationRetriever(Client(), fallback_reformulator=cached(tmp_path, backend))
    retrieve(stage)
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["generated_query"] = "Query: malformed output"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(QueryCacheError):
        retrieve(stage)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("count", [None, "NaN", "-1", "4.5", True, -2])
def test_invalid_primary_match_count_does_not_trigger_model(tmp_path, count):
    backend = Backend()
    client = Client()
    client.search = lambda query, top_k: {"count": count, "idlist": []}
    with pytest.raises(ValueError, match="match count"):
        retrieve(QueryRelaxationRetriever(client, fallback_reformulator=cached(tmp_path, backend)))
    assert backend.calls == []


def fixture_inputs(tmp_path, question_ids=("q1",)):
    records = [{"id": qid, "question": "Clinical clues only.", "options": {"A": "SECRET_OPTION"},
                "answer": "SECRET_ANSWER", "metadata": {"upstream": {"answer": "SECRET_GOLD"}}}
               for qid in question_ids]
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    artifact = {
        "status": "complete", "model": "qwen3:8b", "backend_id": "ollama/qwen3:8b",
        "model_digest": "digest-a",
        "inspection_identity": {"generation_parameters": GENERATION_SETTINGS,
            "question_stem_sha256": {qid: hashlib.sha256(b"Clinical clues only.").hexdigest() for qid in question_ids}},
        "results": [{"question_id": qid, "generated_query": "Exact PRIMARY"} for qid in question_ids],
    }
    primary = tmp_path / "primary.json"
    primary.write_text(json.dumps(artifact), encoding="utf-8")
    return questions, primary


def test_input_selection_is_exact_and_does_not_expose_labels(tmp_path):
    questions, primary = fixture_inputs(tmp_path, VALIDATION_QUESTION_IDS + ("unselected",))
    artifact, inputs = _validation_inputs(questions, primary, VALIDATION_QUESTION_IDS)
    assert [row[0] for row in inputs] == list(VALIDATION_QUESTION_IDS)
    assert all(row[1:] == ("Clinical clues only.", "Exact PRIMARY") for row in inputs)
    backend = Backend()
    CoreClinicalQueryReformulator(backend, backend_id="mock").reformulate(
        question_id=inputs[0][0], question_text=inputs[0][1])
    assert "SECRET_" not in backend.calls[0][0]


def test_changed_stem_is_rejected_before_backend_construction(tmp_path):
    questions, primary = fixture_inputs(tmp_path)
    questions.write_text(json.dumps({"id": "q1", "question": "Different stem"}) + "\n")
    with pytest.raises(ValueError, match="differs"):
        _validation_inputs(questions, primary, ("q1",))


def test_cli_mocked_run_then_cache_reuse_without_live_service(tmp_path, monkeypatch):
    from quality_uncertainty_medrag import query_relaxation as module
    questions, primary = fixture_inputs(tmp_path)
    calls = []

    class FakeOllama:
        def __init__(self, **settings):
            assert settings["think"] is True and settings["seed"] == 42
            self.backend_id = "ollama/qwen3:8b"
            self.base_url = settings["base_url"]
            self.model = settings["model"]

        def generate(self, prompt, *, generation_config):
            calls.append((prompt, generation_config))
            assert "SECRET_" not in prompt
            return "Core disease treatment"

    monkeypatch.setattr(module, "OllamaTextGenerationBackend", FakeOllama)
    monkeypatch.setattr(module, "PubMedClient", lambda *args, **kwargs: Client())
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: io.BytesIO(
        json.dumps({"models": [{"name": "qwen3:8b", "digest": "digest-a"}]}).encode()))
    cache = tmp_path / "cache"
    args = ["--questions", str(questions), "--primary-queries", str(primary),
            "--question-ids", "q1", "--cache-dir", str(cache)]
    assert main(args + ["--output-dir", str(tmp_path / "first")]) == 0
    first = json.loads((tmp_path / "first/medqa_dev_1.report.json").read_text(encoding="utf-8"))
    assert first["model_calls_this_run"] == 1
    assert first["primary_queries_regenerated"] == 0
    assert len(calls) == 1
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: pytest.fail("Cache hit must not require Ollama"))
    assert main(args + ["--output-dir", str(tmp_path / "second")]) == 0
    second = json.loads((tmp_path / "second/medqa_dev_1.report.json").read_text(encoding="utf-8"))
    assert second["model_calls_this_run"] == 0
    assert second["questions"][0]["fallback_query_cache_hit"] is True
    assert len(calls) == 1
    with pytest.raises(SystemExit):
        main(args + ["--output-dir", str(tmp_path / "first")])
