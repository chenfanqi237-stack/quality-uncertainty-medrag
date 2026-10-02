"""No live Ollama, PubMed, Kaggle or GPU is needed for infrastructure checks."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quality_uncertainty_medrag.cloud_runtime import (
    CloudConfig, HELDOUT_IDS, compare_digest, heldout_stems, relative_path,
    save_json, sha256, verify_freeze, project_root,
)
from quality_uncertainty_medrag.heldout_retrieval import IdentityGuard, run

DIGEST = "a" * 64


def records():
    return [{"id": f"medqa-us-dev-{i:06d}", "question": f"Clinical stem {i}.",
             "options": {"A": "SECRET_OPTION"}, "answer": "SECRET_GOLD",
             "answer_idx": "SECRET_INDEX", "metadata": {"upstream": {"answer": "SECRET_UPSTREAM"}}}
            for i in range(1, 31)]  # Synthetic development-only fixtures.


def fixture(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0.1'\n")
    path = tmp_path / "data/questions.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(row) for row in reversed(records())) + "\n")
    save_json(tmp_path / "identity.json", {"model_digest": DIGEST})
    save_json(tmp_path / "freeze.json", {"files": {"data/questions.jsonl": sha256(path)}})
    config = CloudConfig(tmp_path, path, tmp_path / "identity.json", tmp_path / "freeze.json",
                         tmp_path / "outputs/heldout_retrieval/dev_11_30", tmp_path / "pubmed-cache",
                         "http://localhost:11434", 100)
    return config


class Backend:
    def __init__(self, fail_id=None):
        self.calls = []
        self.fail_id = fail_id

    def generate(self, prompt, *, generation_config=None):
        assert "SECRET_" not in prompt
        data = json.loads(prompt.split("QUESTION INPUT (JSON):\n", 1)[1])
        assert set(data) in ({"question_id", "question_text"},
                             {"question_id", "question_text", "primary_query"},
                             {"question_id", "question_text", "primary_query", "first_fallback_query"})
        assert data["question_id"] in HELDOUT_IDS
        assert generation_config["temperature"] == 0
        if "primary_query" in data:
            assert generation_config["seed"] == 42
        self.calls.append(data)
        if data["question_id"] == self.fail_id:
            return "invalid\nmultiple lines"
        return "Core disease" if "primary_query" in data else "Primary disease restriction"


class Client:
    request_count = 0

    def search(self, query, *, top_k):
        assert top_k == 15
        return {"count": "25" if query == "Core disease" else "0", "idlist": ["12345"]}

    def fetch(self, pmids):
        return {"12345": {"pmid": "12345", "title": "Mock article", "abstract": "Mock abstract",
                          "journal": "Mock journal", "publication_date": "2024",
                          "publication_types": ["Journal Article"], "pubmed_metadata": {}}}


def identity(digest=DIGEST):
    return {"model_digest": digest, "ollama_version": "0.34.4"}


@pytest.mark.parametrize("value", ["/tmp/data", "C:/Download/data", "C:\\Download\\data", "../data", "data/../escape", ""])
def test_nonportable_or_escaping_paths_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        relative_path(tmp_path, value)


def test_project_relative_paths_and_root_env_work(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setenv("MEDRAG_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    assert relative_path(tmp_path, "data/questions.jsonl") == tmp_path / "data/questions.jsonl"


def test_cloud_config_is_portable_and_rejects_research_changes(tmp_path, monkeypatch):
    fixture(tmp_path)
    source = Path(__file__).parents[1] / "cloud/kaggle_config.json"
    settings = json.loads(source.read_text())
    save_json(tmp_path / "cloud/kaggle_config.json", settings)
    monkeypatch.setenv("MEDRAG_OLLAMA_URL", "http://localhost:9000")
    config = CloudConfig.load(tmp_path)
    assert config.base_url == "http://localhost:9000"
    assert config.questions == tmp_path / "data/processed/medqa_us_dev_50.jsonl"
    for key, value in (("temperature", 1), ("thinking", False), ("seed", 43),
                       ("top_k", 10), ("start_question", 10), ("fallback_below_matches", 6),
                       ("model", "another-model")):
        changed = {**settings, key: value}
        save_json(tmp_path / "cloud/kaggle_config.json", changed)
        with pytest.raises(ValueError, match="Frozen run setting"):
            CloudConfig.load(tmp_path)


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1"])
def test_bad_transport_timeout_rejected(tmp_path, monkeypatch, timeout):
    fixture(tmp_path)
    source = Path(__file__).parents[1] / "cloud/kaggle_config.json"
    save_json(tmp_path / "cloud/kaggle_config.json", json.loads(source.read_text()))
    monkeypatch.setenv("MEDRAG_OLLAMA_TIMEOUT", timeout)
    with pytest.raises(ValueError):
        CloudConfig.load(tmp_path)


def test_full_digest_comparison():
    assert compare_digest(DIGEST, "sha256:" + DIGEST.upper())["digest_matches_local"]
    comparison = compare_digest(DIGEST, "b" * 64)
    assert not comparison["digest_matches_local"] and comparison["outputs_isolated_by_digest"]
    with pytest.raises(ValueError):
        compare_digest("short", DIGEST)


def test_frozen_file_changes_stop_execution(tmp_path):
    config = fixture(tmp_path)
    verify_freeze(tmp_path, config.freeze)
    config.questions.write_text("changed")
    with pytest.raises(ValueError, match="Frozen research"):
        verify_freeze(tmp_path, config.freeze)


def test_select_exact_heldout_ids_without_options_or_gold(tmp_path):
    config = fixture(tmp_path)
    selected = heldout_stems(config.questions)
    assert [qid for qid, _ in selected] == list(HELDOUT_IDS)
    assert len(selected) == 20 and all("SECRET" not in text for _, text in selected)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "null"])
def test_incomplete_or_malformed_range_rejected(tmp_path, mode):
    data = records()
    if mode == "missing":
        del data[10]
    elif mode == "duplicate":
        data.append(data[10])
    else:
        data[10]["question"] = None
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in data))
    with pytest.raises(ValueError):
        heldout_stems(path)


def test_heldout_runner_payload_isolation_cache_reuse_and_no_overwrite(tmp_path):
    config = fixture(tmp_path)
    first = Backend()
    directory = run(config, identity(), first, Client(), run_id="first")
    assert len(first.calls) == 40  # Primary + fallback for each mock question.
    assert all(data.get("primary_query") == "Primary disease restriction"
               for data in first.calls if "primary_query" in data)
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["total_saved_articles"] == 20
    assert not report["stance_classification_run"] and not report["aggregation_run"]
    second = Backend()
    repeated = run(config, identity(), second, Client(), run_id="second")
    assert second.calls == []
    rows = json.loads((repeated / "report.json").read_text())["questions"]
    assert all(r["primary_query_cache_hit"] and r["fallback_query_cache_hit"] for r in rows)
    assert all(r["final_query_used"] == "Core disease" for r in rows)
    with pytest.raises(FileExistsError):
        run(config, identity(), second, Client(), run_id="first")
    run(config, identity(), second, Client(), run_id="first", resume=True)
    assert second.calls == []


def test_failures_are_not_repaired_on_resume_or_leaked_between_questions(tmp_path):
    config = fixture(tmp_path)
    first = Backend(fail_id=HELDOUT_IDS[1])
    directory = run(config, identity(), first, Client(), run_id="failure")
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "complete_with_failures"
    assert report["failed_question_count"] == 1
    failed = report["questions"][1]
    assert failed["status"] == "failed" and failed["fallback_query"] is None
    assert failed["primary_query"] is None and failed["primary_pubmed_match_count"] is None
    assert failed["query_generation_retry_count"] == 2
    assert len(failed["query_generation"]["primary"]["attempts"]) == 3
    later = Backend()
    run(config, identity(), later, Client(), run_id="failure", resume=True)
    assert later.calls == []


def test_different_digest_separates_outputs_and_caches(tmp_path, capsys):
    config = fixture(tmp_path)
    directory = run(config, identity("b" * 64), Backend(), Client(), run_id="different")
    assert "b" * 16 in directory.parts
    assert "WARNING: MODEL DIGEST DIFFERS" in capsys.readouterr().out
    report = json.loads((directory / "report.json").read_text())
    assert not report["digest_matches_local"]
    assert report["status"] == "complete" and report["actual_model_digest"] == "b" * 64


def test_fallback_failure_preserves_primary_count_without_previous_fallback_leak(tmp_path):
    config = fixture(tmp_path)
    class FallbackFailure(Backend):
        def generate(self, prompt, *, generation_config=None):
            result = super().generate(prompt, generation_config=generation_config)
            data = self.calls[-1]
            if data["question_id"] == HELDOUT_IDS[1] and "primary_query" in data:
                return "invalid\noutput"
            return result
    directory = run(config, identity(), FallbackFailure(), Client(), run_id="fallback-failure")
    report = json.loads((directory / "report.json").read_text())
    failed = report["questions"][1]
    assert report["failed_question_count"] == 1
    assert failed["primary_query"] == "Primary disease restriction"
    assert failed["primary_pubmed_match_count"] == 0 and failed["fallback_triggered"] is True
    assert failed["fallback_query"] is None and failed["final_query_used"] is None


def test_digest_prefix_collision_is_rejected_before_inference(tmp_path):
    config = fixture(tmp_path)
    scoped = config.output / DIGEST[:16]
    save_json(scoped / "model_identity.json", {"model_digest": DIGEST[:16] + "b" * 48})
    backend = Backend()
    with pytest.raises(ValueError, match="prefix collision"):
        run(config, identity(), backend, Client(), run_id="collision")
    assert backend.calls == []


def test_identity_change_stops_model_before_generation():
    backend = Backend()
    guard = IdentityGuard(backend, DIGEST, lambda: identity("b" * 64))
    with pytest.raises(ValueError, match="changed during run"):
        guard.generate("unused")
    assert backend.calls == []


def test_development_runner_recovered_generation_failure_is_audited(tmp_path):
    config = fixture(tmp_path)
    class TransientFailure(Backend):
        failures_left = 2
        prompts = None
        def generate(self, prompt, *, generation_config=None):
            if self.prompts is None:
                self.prompts = []
            data = json.loads(prompt.split("QUESTION INPUT (JSON):\n")[1])
            if data["question_id"] == HELDOUT_IDS[0] and "primary_query" not in data:
                self.prompts.append(prompt)
                if self.failures_left:
                    self.failures_left -= 1
                    raise TimeoutError("SECRET_PROVIDER_CONTENT")
            return super().generate(prompt, generation_config=generation_config)
    backend = TransientFailure()
    directory = run(config, identity(), backend, Client(), run_id="recovered")
    report = json.loads((directory / "report.json").read_text())
    assert report["dataset_role"] == "development"
    assert report["model_calls_total"] == 42
    assert report["failed_question_count"] == 0
    row = report["questions"][0]
    assert row["query_generation_retry_count"] == 2
    assert [a["status"] for a in row["query_generation"]["primary"]["attempts"]] == ["failed", "failed", "success"]
    assert len(set(backend.prompts)) == 1
    assert "SECRET_PROVIDER_CONTENT" not in json.dumps(report)


@pytest.mark.parametrize("minimal_failure", [False, True])
def test_development_runner_minimal_selection_and_failure_provenance(tmp_path, minimal_failure):
    config = fixture(tmp_path)
    class MinimalBackend(Backend):
        def generate(self, prompt, *, generation_config=None):
            data = json.loads(prompt.split("QUESTION INPUT (JSON):\n")[1])
            result = super().generate(prompt, generation_config=generation_config)
            if "first_fallback_query" in data:
                if minimal_failure and data["question_id"] == HELDOUT_IDS[1]:
                    return "invalid\nquery"
                return "Minimal condition"
            return result
    class LowClient(Client):
        def search(self, query, *, top_k):
            assert top_k == 15
            return {"count": "3" if query == "Minimal condition" else "0", "idlist": ["12345"]}
    directory = run(config, identity(), MinimalBackend(), LowClient(), run_id="minimal")
    report = json.loads((directory / "report.json").read_text())
    row = report["questions"][0]
    assert row["query_stage_used"] == "minimal_fallback"
    assert row["minimal_fallback_triggered"] is True
    assert row["fallback_pubmed_match_count"] == 0
    assert row["minimal_fallback_pubmed_match_count"] == 3
    assert row["final_query_used"] == "Minimal condition"
    if minimal_failure:
        failed = report["questions"][1]
        assert failed["status"] == "failed"
        assert failed["primary_pubmed_match_count"] == failed["fallback_pubmed_match_count"] == 0
        assert failed["minimal_fallback_query"] is None
        assert failed["minimal_fallback_triggered"] is True
        assert failed["query_generation_retry_count"] == 2
        assert report["questions"][2]["query_generation_retry_count"] == 0


def test_notebook_cells_are_plain_python_and_have_no_saved_output():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/kaggle_qwen3_medrag.ipynb").read_text(encoding="utf-8"))
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            text = "".join(cell["source"])
            ast.parse(text)
            code.append(text)
    text = "\n".join(code)
    assert "nvidia-smi" in text and "OLLAMA_VERSION" in text and "size_vram" in text
    assert "heldout_retrieval" in text and "warn_digest" in text


def notebook_prerequisite_function(available, *, uid=0, install_succeeds=True):
    notebook = json.loads((Path(__file__).parents[1] /
        "notebooks/kaggle_qwen3_medrag.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code"
                and "def ensure_ollama_prerequisites" in "".join(c["source"]))
    text = "".join(cell["source"])
    assert text.index("ensure_ollama_prerequisites()\n") < text.index(
        "subprocess.run(['sh',str(installer)]")
    function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                    and n.name == "ensure_ollama_prerequisites")
    tools = set(available)
    calls = []

    def fake_run(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["env"]["DEBIAN_FRONTEND"] == "noninteractive"
        calls.append(command)
        if "install" in command and install_succeeds:
            tools.update(command[command.index("--no-install-recommends") + 1:])

    namespace = {
        "shutil": SimpleNamespace(which=lambda name:
            "/usr/bin/" + name if name in tools else None),
        "os": SimpleNamespace(geteuid=lambda: uid, environ={}),
        "subprocess": SimpleNamespace(run=fake_run),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 "prerequisite-test", "exec"), namespace)
    return namespace["ensure_ollama_prerequisites"], calls


def test_notebook_installs_missing_zstd_before_ollama():
    ensure, calls = notebook_prerequisite_function({"curl", "tar", "apt-get"})
    ensure()
    assert calls == [
        ["/usr/bin/apt-get", "update", "-qq"],
        ["/usr/bin/apt-get", "install", "-y", "--no-install-recommends", "zstd"],
    ]


def test_notebook_skips_install_when_prerequisites_exist():
    ensure, calls = notebook_prerequisite_function({"curl", "tar", "zstd"})
    ensure()
    assert calls == []


def test_notebook_uses_noninteractive_sudo_when_needed():
    ensure, calls = notebook_prerequisite_function(
        {"curl", "tar", "apt-get", "sudo"}, uid=1000)
    ensure()
    assert len(calls) == 2
    assert all(command[:2] == ["/usr/bin/sudo", "-n"] for command in calls)


def test_notebook_fails_clearly_without_package_manager():
    ensure, calls = notebook_prerequisite_function({"curl", "tar"})
    with pytest.raises(RuntimeError, match="apt-get"):
        ensure()
    assert calls == []


def test_notebook_verifies_zstd_after_installation():
    ensure, _ = notebook_prerequisite_function(
        {"curl", "tar", "apt-get"}, install_succeeds=False)
    with pytest.raises(RuntimeError, match="did not provide: zstd"):
        ensure()
