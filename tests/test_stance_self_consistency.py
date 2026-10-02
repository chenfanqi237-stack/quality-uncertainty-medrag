import csv
import hashlib
import json
import math
import ast
import sys
import zipfile
from pathlib import Path

import pytest

from quality_uncertainty_medrag import stance_self_consistency as sc


class MockBackend:
    def __init__(self, responses=None):
        self.responses = iter(responses or [])
        self.calls = []

    def generate(self, prompt, *, generation_config):
        self.calls.append((prompt, dict(generation_config)))
        return next(self.responses)


def pair(index=1):
    return {"question_id": f"medqa-us-dev-{index:06d}", "candidate_option_id": "B",
            "evidence_doc_id": f"{index}", "question_stem": "What treats condition X?",
            "candidate_option_text": "Treatment A", "evidence_title": "Treatment A for condition X",
            "evidence_abstract": "Treatment A is discussed."}


def test_empirical_frequencies_and_three_class_entropy():
    result = sc.uncertainty(["SUPPORT"] * 5 + ["CONTRADICT"] * 3 + ["IRRELEVANT"] * 2)
    assert (result["n_support"], result["n_contradict"], result["n_irrelevant"]) == (5, 3, 2)
    assert (result["p_support"], result["p_contradict"], result["p_irrelevant"]) == (.5, .3, .2)
    expected = -sum(x * math.log(x) for x in (.5, .3, .2)) / math.log(3)
    assert result["u_3"] == pytest.approx(expected)
    assert result["relevance"] == pytest.approx(.8)


def test_directional_normalization_and_binary_entropy():
    result = sc.uncertainty(["SUPPORT"] * 6 + ["CONTRADICT"] * 2 + ["IRRELEVANT"] * 2)
    assert result["directional_score"] == pytest.approx(.5)
    assert result["u_directional"] == pytest.approx(sc.entropy((.75, .25), classes=2))
    assert sc.entropy((1, 0, 0), classes=3) == 0


def test_zero_directional_mass_convention():
    result = sc.uncertainty(["IRRELEVANT"] * 10)
    assert result["u_directional"] == 1.0
    assert result["directional_score"] == 0.0
    assert result["relevance"] == 0.0
    assert result["u_3"] == 0.0


def test_cache_identity_includes_seed_temperature_prompt_and_inputs(monkeypatch):
    a = sc.sample_context(pair(), 101)[2]
    assert a != sc.sample_context(pair(), 102)[2]
    monkeypatch.setattr(sc, "TEMPERATURE", .8)
    assert a != sc.sample_context(pair(), 101)[2]
    monkeypatch.setattr(sc, "TEMPERATURE", .7)
    modified = pair()
    modified["candidate_option_text"] = "Treatment B"
    assert a != sc.sample_context(modified, 101)[2]


def test_identical_retries_and_exact_cache_reuse(tmp_path):
    backend = MockBackend(["bad response", '{"SUPPORT":0.4,"CONTRADICT":0.4,"IRRELEVANT":0.2}',
                           '{"SUPPORT":1,"CONTRADICT":0,"IRRELEVANT":0}'])
    first = sc.sample_one(pair(), 101, backend, tmp_path)
    second = sc.sample_one(pair(), 101, backend, tmp_path)
    assert first["stance"] == second["stance"] == "SUPPORT"
    assert first["cache_status"] == "MISS" and second["cache_status"] == "HIT"
    assert first["model_calls"] == 3 and second["model_calls"] == 0
    assert first["retries_this_run"] == 2
    assert len(backend.calls) == 3
    assert all(config == {"temperature": .7, "seed": 101} for _, config in backend.calls)
    assert len({prompt for prompt, _ in backend.calls}) == 1
    assert list(tmp_path.glob("*.json")) and list((tmp_path / "attempts").glob("*.json"))


def test_model_payload_isolation(tmp_path):
    row = pair()
    row["question_stem"] = "Which therapy is used?"
    row["candidate_option_text"] = "Aspirin"
    backend = MockBackend(['{"SUPPORT":0,"CONTRADICT":0,"IRRELEVANT":1}'])
    sc.sample_one(row, 101, backend, tmp_path)
    prompt = backend.calls[0][0]
    payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
    assert set(payload) == set(sc.PROMPT_FIELDS)
    assert payload["candidate_option_text"] == "Aspirin"
    assert "answer_idx" not in prompt and "final_stance" not in prompt
    assert "other_candidate_options" not in prompt
    with pytest.raises(ValueError):
        sc.sample_context({**row, "human_stance": "SUPPORT"}, 101)


def test_selective_prediction_and_tie_breaking():
    rows = []
    for index in range(60):
        correct = index < 42
        rows.append({"pair_id": f"P{index:03d}", "reference_stance": "SUPPORT",
                     "frozen_qwen_hard_stance": "SUPPORT" if correct else "IRRELEVANT",
                     "hard_prediction_correct": correct, "u_3": index / 60})
    full = sc.selective(rows, "u_3", 1.0)
    low = sc.selective(rows, "u_3", .7)
    assert full["retained_pair_count"] == 60 and full["accuracy"] == pytest.approx(.7)
    assert low["retained_pair_count"] == 42 and low["accuracy"] == 1.0
    assert low["macro_f1"] == pytest.approx(1 / 3)


def test_auroc_auprc_ties_and_undefined():
    assert sc.auroc([.9, .8, .2, .1], [True, True, False, False]) == 1
    assert sc.auprc([.9, .8, .2, .1], [True, True, False, False]) == 1
    assert sc.auroc([.5, .5], [True, False]) == .5
    assert sc.auprc([.5, .5], [True, False]) == .5
    assert sc.auroc([.1, .2], [True, True]) is None
    assert sc.auprc([.1, .2], [False, False]) is None


def test_read_unlabeled_rejects_reference_field_and_duplicate(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    rows = [pair(i + 1) for i in range(60)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(sc, "INPUT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(sc.read_unlabeled(path)) == 60
    rows[0]["final_stance"] = "SUPPORT"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(sc, "INPUT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="allowlist"):
        sc.read_unlabeled(path)


def test_end_to_end_mock_sampling_then_reference_join(tmp_path, monkeypatch):
    # Archive I/O is tested separately; this test covers inference/cache/evaluation flow.
    monkeypatch.setattr(sc, "export_checkpoint", lambda *args, **kwargs: None)
    inputs = [pair(i + 1) for i in range(60)]
    input_path = tmp_path / sc.INPUT_REL
    input_path.parent.mkdir(parents=True)
    input_path.write_text("\n".join(json.dumps(r) for r in inputs) + "\n", encoding="utf-8")
    monkeypatch.setattr(sc, "INPUT_SHA256", hashlib.sha256(input_path.read_bytes()).hexdigest())
    backend = MockBackend(['{"SUPPORT":0,"CONTRADICT":0,"IRRELEVANT":1}'] * 600)
    audit = sc.collect(tmp_path, backend=backend, verify_identity=False)
    assert audit["actual_model_calls"] == 600 and audit["cache_hits"] == 0
    assert len(backend.calls) == 600
    assert all(set(json.loads(p.split("Input JSON:\n", 1)[1])) == set(sc.PROMPT_FIELDS)
               for p, _ in backend.calls)
    cached = sc.collect(tmp_path, backend=backend, verify_identity=False)
    assert cached["cache_hits"] == 600 and cached["new_model_calls_this_invocation"] == 0
    assert cached["actual_model_calls"] == 600 and len(backend.calls) == 600

    reference_path = tmp_path / sc.REFERENCE_REL
    reference_path.parent.mkdir(parents=True)
    ref_columns = ["pair_id", "batch", "question_id", "candidate_option_id", "pmid", "final_stance"]
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ref_columns)
        writer.writeheader()
        for i, row in enumerate(inputs):
            writer.writerow({"pair_id": f"P{i:03d}", "batch": "structural" if i < 30 else "directional",
                             "question_id": row["question_id"], "candidate_option_id": row["candidate_option_id"],
                             "pmid": row["evidence_doc_id"],
                             "final_stance": "SUPPORT" if i < 8 else "CONTRADICT" if i < 15 else "IRRELEVANT"})
    monkeypatch.setattr(sc, "REFERENCE_SHA256", hashlib.sha256(reference_path.read_bytes()).hexdigest())
    hard_path = tmp_path / sc.HARD_REL
    hard_path.parent.mkdir(parents=True, exist_ok=True)
    with hard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "question_id", "candidate_option_id", "pmid",
                                                     "reference_stance", "qwen_v1_prediction"])
        writer.writeheader()
        for i, row in enumerate(inputs):
            writer.writerow({"pair_id": f"P{i:03d}", "question_id": row["question_id"],
                             "candidate_option_id": row["candidate_option_id"], "pmid": row["evidence_doc_id"],
                             "reference_stance": "SUPPORT" if i < 8 else "CONTRADICT" if i < 15 else "IRRELEVANT",
                             "qwen_v1_prediction": "IRRELEVANT"})
    metrics = sc.evaluate(tmp_path)
    assert metrics["n"] == 60 and metrics["hard_correct"] == 45
    assert metrics["by_reference_class"]["CONTRADICT"]["count"] == 7
    assert metrics["error_detection"]["u_3"]["auroc"] == .5
    assert (tmp_path / sc.OUTPUT_REL / "uncertainty_report.md").exists()


GOOD = '{"SUPPORT":1,"CONTRADICT":0,"IRRELEVANT":0}'


@pytest.fixture
def resume_root(tmp_path_factory, monkeypatch):
    # Short paths also exercise Windows without exceeding legacy path limits.
    root = tmp_path_factory.mktemp("sc")
    rows = []
    for i in range(60):
        row = pair(i % 30 + 1)
        row["candidate_option_id"] = "A" if i < 30 else "B"
        row["candidate_option_text"] = f"Synthetic candidate {i}"
        rows.append(row)
    path = root / sc.INPUT_REL
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(sc, "INPUT_SHA256", sc.sha_bytes(path.read_bytes()))
    return root, rows


def forbid_live_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No live backend or network access allowed")
    monkeypatch.setattr(sc, "verify_ollama_identity", forbidden)
    monkeypatch.setattr(sc, "OllamaTextGenerationBackend", forbidden)
    return forbidden


def test_exhausted_sample_continues_batch_and_periodic_exports(resume_root, monkeypatch):
    root, rows = resume_root
    class OneFailure:
        def __init__(self):
            self.calls = []
        def generate(self, prompt, *, generation_config):
            self.calls.append((prompt, dict(generation_config)))
            if generation_config["seed"] == 102 and 'Synthetic candidate 0"' in prompt:
                raise TimeoutError("PRIVATE RAW RESPONSE MUST NOT BE SAVED")
            return GOOD
    backend = OneFailure()
    events = []
    export = sc.export_checkpoint
    def observe_export(*args, **kwargs):
        state = sc.validate_restored_checkpoint(root)
        events.append((state["valid"], state["failed"]))
        # Check actual failure/final archives, without flooding the Windows ZIP scanner.
        if state["valid"] in (1, 599, 600):
            return export(*args, **kwargs)
    monkeypatch.setattr(sc, "export_checkpoint", observe_export)
    audit = sc.collect(root, backend=backend, verify_identity=False)
    assert (audit["expected"], audit["valid"], audit["failed"], audit["missing"]) == (600, 599, 1, 0)
    assert audit["actual_model_calls"] == 602 and audit["retries"] == 2
    assert len(backend.calls) == 602  # Later samples were processed normally.
    assert events == [(1, 1)] + [(i, 1) for i in range(25, 600, 25)] + [(599, 1)]
    key = sc.sample_context(rows[0], 102)[2]
    cache = root / sc.CACHE_REL
    assert not (cache / f"{key}.json").exists()
    failed = json.loads((cache / "failures" / key / "0001.json").read_text())
    assert failed["status"] == "FAILED" and "stance" not in failed
    assert failed["attempt_count"] == 3 and failed["seed"] == 102 and failed["temperature"] == .7
    assert failed["error_class"] == "TimeoutError" and failed["error_stage"] == "generation"
    assert failed["timestamp"] and failed["pair_identity"] == sc.sample_context(rows[0], 102)[0]["ids"]
    assert "PRIVATE RAW" not in (cache / "attempts" / f"{key}.json").read_text()
    saved = {p.name: p.read_bytes() for p in cache.glob("*.json")}
    assert len(saved) == 599
    partial = [json.loads(line) for line in (root / sc.OUTPUT_REL / "samples.jsonl").read_text().splitlines()]
    assert len(partial) == 599 and all(r["cache_key"] != key for r in partial)
    with pytest.raises(ValueError, match="600"):
        sc.evaluate(root)
    forbid_live_backend(monkeypatch)
    resumed = sc.collect(root)
    assert resumed["cache_hits"] == 599 and resumed["failed_skipped"] == 1
    assert resumed["new_model_calls_this_invocation"] == 0
    assert saved == {p.name: p.read_bytes() for p in cache.glob("*.json")}
    retry = MockBackend([GOOD])
    recovered = sc.collect(root, backend=retry, verify_identity=False, retry_failed=True)
    assert recovered["valid"] == 600 and recovered["failed"] == 0
    assert recovered["cache_hits"] == 599 and recovered["new_model_calls_this_invocation"] == 1
    assert len(retry.calls) == 1
    assert all((cache / name).read_bytes() == data for name, data in saved.items())
    assert recovered["actual_model_calls"] == 603 and recovered["retries"] == 2


def test_checkpoint_roundtrip_audits_and_explicit_retry(resume_root, tmp_path_factory, monkeypatch):
    root, rows = resume_root
    cache = root / sc.CACHE_REL
    first = sc.sample_one(rows[0], 101, MockBackend([GOOD]), cache)
    failed = sc.sample_one(rows[0], 102, MockBackend(["invalid"] * 3), cache)
    assert failed["status"] == "FAILED" and "stance" not in failed
    out = root / sc.OUTPUT_REL
    (out / "partial_output.csv").write_text("partial\n", encoding="utf-8")
    sc.write_json(out / "progress.json", {"valid": 1, "failed": 1, "missing": 598})
    archive = sc.export_checkpoint(root)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        assert (sc.CACHE_REL / f"{first['cache_key']}.json").as_posix() in names
        assert (sc.CACHE_REL / "attempts" / f"{failed['cache_key']}.json").as_posix() in names
        assert (sc.CACHE_REL / "failures" / failed["cache_key"] / "0001.json").as_posix() in names
        for name in ("progress.json", "status.json", "run_manifest.json", "partial_output.csv"):
            assert (sc.OUTPUT_REL / name).as_posix() in names
        assert sc.INPUT_REL.as_posix() in names
        assert not any(name.endswith(".zip") or name.endswith(".tmp") for name in names)
    restored = tmp_path_factory.mktemp("rs")
    state = sc.restore_checkpoint(archive, restored)
    assert (state["valid"], state["failed"], state["missing"]) == (1, 1, 598)
    restored_cache = restored / sc.CACHE_REL
    original = (cache / f"{first['cache_key']}.json").read_bytes()
    attempt_history = (restored_cache / "attempts" / f"{failed['cache_key']}.json").read_bytes()
    failure_history = (restored_cache / "failures" / failed["cache_key"] / "0001.json").read_bytes()
    no_calls = MockBackend()
    assert sc.sample_one(rows[0], 101, no_calls, restored_cache)["cache_status"] == "HIT"
    assert sc.sample_one(rows[0], 102, no_calls, restored_cache)["cache_status"] == "FAILED_SKIPPED"
    assert no_calls.calls == []
    retry = MockBackend([GOOD])
    assert sc.sample_one(rows[0], 101, retry, restored_cache, retry_failed=True)["cache_status"] == "HIT"
    recovered = sc.sample_one(rows[0], 102, retry, restored_cache, retry_failed=True)
    assert recovered["status"] == "VALID" and len(retry.calls) == 1
    assert (restored_cache / f"{first['cache_key']}.json").read_bytes() == original
    assert (restored_cache / "attempts" / "history" / failed["cache_key"] / "0001.json").read_bytes() == attempt_history
    assert (restored_cache / "failures" / failed["cache_key"] / "0001.json").read_bytes() == failure_history
    assert sc.validate_restored_checkpoint(restored)["failed"] == 0
    assert sc.scan_samples(restored)[1]["total_model_calls"] == 4
    updated = sc.export_checkpoint(restored)
    with zipfile.ZipFile(updated) as z:
        assert any("attempts/history/" in name for name in z.namelist())


def test_legacy_exhausted_audit_is_skipped_and_migrated(resume_root):
    root, rows = resume_root
    context, _, key = sc.sample_context(rows[0], 101)
    cache = root / sc.CACHE_REL
    sc.write_json(cache / "attempts" / f"{key}.json", {"context": context, "status": "INCOMPLETE",
                  "attempts": [{"index": i, "status": "failed", "error_type": "StanceOutputError"}
                               for i in range(1, 4)]})
    assert sc.validate_restored_checkpoint(root)["failed"] == 1
    mock = MockBackend()
    result = sc.sample_one(rows[0], 101, mock, cache)
    assert result["status"] == "FAILED" and mock.calls == []
    assert (cache / "failures" / key / "0001.json").exists()


def test_partial_attempts_resume_with_remaining_budget(resume_root):
    root, rows = resume_root
    context, _, key = sc.sample_context(rows[0], 101)
    cache = root / sc.CACHE_REL
    sc.write_json(cache / "attempts" / f"{key}.json", {"context": context, "status": "INCOMPLETE",
                  "attempts": [{"index": 1, "status": "failed", "error_type": "TimeoutError"}]})
    backend = MockBackend(["bad", "bad"])
    result = sc.sample_one(rows[0], 101, backend, cache)
    assert result["status"] == "FAILED" and len(backend.calls) == 2
    second = MockBackend(["bad"] * 3)
    sc.sample_one(rows[0], 101, second, cache, retry_failed=True)
    sc.sample_one(rows[0], 101, second, cache)
    assert len(second.calls) == 3
    assert len(list((cache / "failures" / key).glob("*.json"))) == 2


def test_status_cli_is_read_only_and_makes_zero_model_calls(resume_root, monkeypatch, capsys):
    root, rows = resume_root
    sc.sample_one(rows[0], 101, MockBackend([GOOD]), root / sc.CACHE_REL)
    forbid_live_backend(monkeypatch)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(sys, "argv", ["probe", "status", "--root", str(root)])
    sc.main()
    state = json.loads(capsys.readouterr().out)
    assert (state["expected"], state["valid"], state["failed"], state["missing"]) == (600, 1, 0, 599)
    assert state["completion_percentage"] == pytest.approx(100 / 600)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_atomic_checkpoint_preserves_previous_zip_on_replace_error(resume_root, monkeypatch):
    root, rows = resume_root
    archive = sc.export_checkpoint(root)
    original = archive.read_bytes()
    replace = sc.atomic_replace
    def fail_zip(temporary, path):
        if path.suffix == ".zip":
            raise PermissionError("simulated locked ZIP")
        replace(temporary, path)
    monkeypatch.setattr(sc, "atomic_replace", fail_zip)
    with pytest.raises(PermissionError):
        sc.export_checkpoint(root)
    assert archive.read_bytes() == original
    assert not list(archive.parent.glob("*.tmp"))


def test_atomic_replace_retries_transient_lock(tmp_path, monkeypatch):
    source, target = tmp_path / "new.tmp", tmp_path / "old.zip"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    replace = Path.replace
    calls = []
    def transient(path, destination):
        calls.append(path)
        if len(calls) < 3:
            assert target.read_bytes() == b"old"
            raise PermissionError("simulated scanner lock")
        return replace(path, destination)
    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr(sc.time, "sleep", lambda _: None)
    sc.atomic_replace(source, target)
    assert len(calls) == 3 and target.read_bytes() == b"new"


def test_retry_failed_cli_is_explicit(monkeypatch, tmp_path, capsys):
    calls = []
    def mocked_collect(root, **kwargs):
        calls.append(kwargs)
        return {"valid": 600}
    monkeypatch.setattr(sc, "collect", mocked_collect)
    monkeypatch.setattr(sys, "argv", ["probe", "sample", "--root", str(tmp_path)])
    sc.main()
    monkeypatch.setattr(sys, "argv", ["probe", "sample", "--root", str(tmp_path), "--retry-failed"])
    sc.main()
    assert [item["retry_failed"] for item in calls] == [False, True]


def test_restore_rejects_conflicting_files_and_tampering(resume_root, tmp_path_factory):
    root, rows = resume_root
    archive = sc.export_checkpoint(root)
    target = tmp_path_factory.mktemp("cf")
    path = target / sc.INPUT_REL
    path.parent.mkdir(parents=True)
    path.write_text("unrelated", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        sc.restore_checkpoint(archive, target)
    assert path.read_text() == "unrelated"
    assert not (target / sc.OUTPUT_REL).exists()
    corrupt = root / "corrupt.zip"
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(corrupt, "w") as dest:
        for name in src.namelist():
            dest.writestr(name, b"tampered" if name == sc.INPUT_REL.as_posix() else src.read(name))
    with pytest.raises(ValueError, match="SHA256"):
        sc.restore_checkpoint(corrupt, tmp_path_factory.mktemp("bad"))


def test_frozen_settings_and_cloud_export_wrapper():
    assert sc.MODEL == "qwen3:8b" and sc.OLLAMA_VERSION == "0.34.4"
    assert sc.DIGEST == "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
    assert sc.CLASSIFIER_VERSION == "llm-medical-stance-v1"
    assert sc.TEMPERATURE == .7 and sc.SEEDS == tuple(range(101, 111))
    assert sc.frozen_settings()["thinking"] is True
    assert sc.frozen_settings()["frozen_hard_settings"] == {"thinking": True, "temperature": 0, "seed": 42}
    assert sc.CACHE_REL == sc.OUTPUT_REL / "cache"
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/kaggle_stance_self_consistency_10.ipynb").read_text(encoding="utf-8"))
    source = "".join(nb["cells"][2]["source"])
    tree = ast.parse(source)
    block = next(node for node in tree.body if isinstance(node, ast.Try))
    setup_tree = ast.parse("".join(nb["cells"][0]["source"]))
    assert any(isinstance(node, ast.ImportFrom)
               and any(name.name == "export_checkpoint" for name in node.names)
               for node in ast.walk(setup_tree))  # Reject stale runtime bundles before collection.
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "export_checkpoint" for stmt in block.finalbody for node in ast.walk(stmt))
    run = next(node for node in ast.walk(block) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "run")
    assert next(kw.value.value for kw in run.keywords if kw.arg == "check") is False
    generator = (root / "cloud/prepare_self_consistency_kaggle.py").read_text(encoding="utf-8")
    generated_source = generator.split("SAMPLING = r'''", 1)[1].split("'''", 1)[0]
    assert source.strip() == generated_source.strip()


def test_bounded_batch_processes_exactly_25_new_units_and_resumes(resume_root, monkeypatch):
    root, rows = resume_root
    first = MockBackend([GOOD] * 25)
    initial = sc.collect(root, backend=first, verify_identity=False, max_new_samples=25)
    assert (initial["valid"], initial["failed"], initial["missing"]) == (25, 0, 575)
    assert initial["new_units_this_invocation"] == initial["new_valid_this_invocation"] == 25
    assert initial["new_failed_this_invocation"] == 0 and len(first.calls) == 25
    assert initial["max_new_samples"] == 25
    first_key = sc.sample_context(rows[0], 101)[2]
    first_path = root / sc.CACHE_REL / f"{first_key}.json"
    frozen_bytes = first_path.read_bytes()
    assert (root / sc.OUTPUT_REL / sc.CHECKPOINT_NAME).is_file()

    # Cache reuse is independent of replacing a ZIP still held by a Windows scanner.
    exports = []
    monkeypatch.setattr(sc, "export_checkpoint", lambda *args, **kwargs: exports.append(sc.validate_restored_checkpoint(root)))
    second = MockBackend([GOOD] * 25)
    resumed = sc.collect(root, backend=second, verify_identity=False, max_new_samples=25)
    assert (resumed["valid"], resumed["failed"], resumed["missing"]) == (50, 0, 550)
    assert resumed["cache_hits"] == 25 and resumed["new_units_this_invocation"] == 25
    assert len(second.calls) == 25 and first_path.read_bytes() == frozen_bytes
    assert sc.validate_restored_checkpoint(root)["valid"] == 50
    assert exports[-1]["valid"] == 50


def test_failed_unit_counts_toward_bounded_limit_and_checkpoint(resume_root, monkeypatch):
    root, rows = resume_root
    class OneExhausted:
        def __init__(self):
            self.calls = []
        def generate(self, prompt, *, generation_config):
            self.calls.append((prompt, dict(generation_config)))
            if generation_config["seed"] == 102 and 'Synthetic candidate 0"' in prompt:
                raise TimeoutError("private response is never saved")
            return GOOD
    backend = OneExhausted()
    result = sc.collect(root, backend=backend, verify_identity=False, max_new_samples=25)
    assert (result["valid"], result["failed"], result["missing"]) == (24, 1, 575)
    assert (result["new_units_this_invocation"], result["new_valid_this_invocation"],
            result["new_failed_this_invocation"]) == (25, 24, 1)
    assert len(backend.calls) == 27  # Three identical attempts for one failed unit.
    failure_key = sc.sample_context(rows[0], 102)[2]
    checkpoint = root / sc.OUTPUT_REL / sc.CHECKPOINT_NAME
    with zipfile.ZipFile(checkpoint) as archive:
        names = set(archive.namelist())
        assert sc.INPUT_REL.as_posix() in names
        assert (sc.CACHE_REL / "failures" / failure_key / "0001.json").as_posix() in names
        assert (sc.CACHE_REL / "attempts" / f"{failure_key}.json").as_posix() in names
        assert all((sc.OUTPUT_REL / name).as_posix() in names
                   for name in ("progress.json", "status.json", "run_manifest.json", "samples.jsonl"))
    monkeypatch.setattr(sc, "export_checkpoint", lambda *args, **kwargs: None)
    second = MockBackend([GOOD] * 25)
    resumed = sc.collect(root, backend=second, verify_identity=False, max_new_samples=25)
    assert (resumed["valid"], resumed["failed"], resumed["missing"]) == (49, 1, 550)
    assert resumed["new_units_this_invocation"] == len(second.calls) == 25
    assert resumed["failed_skipped"] == 1


def test_unexpected_interrupt_exports_resumable_checkpoint(resume_root):
    root, rows = resume_root
    class InterruptAfterOne:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, *, generation_config):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt("synthetic interruption")
            return GOOD
    backend = InterruptAfterOne()
    with pytest.raises(KeyboardInterrupt):
        sc.collect(root, backend=backend, verify_identity=False, max_new_samples=25)
    assert backend.calls == 2
    state = sc.validate_restored_checkpoint(root)
    assert (state["valid"], state["failed"], state["missing"]) == (1, 0, 599)
    checkpoint = root / sc.OUTPUT_REL / sc.CHECKPOINT_NAME
    with zipfile.ZipFile(checkpoint) as archive:
        assert archive.testzip() is None
        assert (sc.CACHE_REL / (sc.sample_context(rows[0], 101)[2] + ".json")).as_posix() in archive.namelist()
        assert (sc.OUTPUT_REL / "unexpected_error_audit.json").as_posix() in archive.namelist()
    assert sc.restore_checkpoint(checkpoint, root) == state


def test_retry_failed_is_explicit_and_bounded_to_one_unit(resume_root, monkeypatch):
    root, rows = resume_root
    failed = sc.collect(root, backend=MockBackend(["invalid"] * 3),
                        verify_identity=False, max_new_samples=1)
    assert (failed["valid"], failed["failed"], failed["missing"]) == (0, 1, 599)
    assert failed["new_units_this_invocation"] == 1
    first_key = sc.sample_context(rows[0], 101)[2]
    original_failure = (root / sc.CACHE_REL / "failures" / first_key / "0001.json").read_bytes()
    monkeypatch.setattr(sc, "export_checkpoint", lambda *args, **kwargs: None)
    retry = MockBackend([GOOD])
    recovered = sc.collect(root, backend=retry, verify_identity=False,
                           retry_failed=True, max_new_samples=1)
    assert (recovered["valid"], recovered["failed"], recovered["missing"]) == (1, 0, 599)
    assert recovered["new_units_this_invocation"] == len(retry.calls) == 1
    assert (root / sc.CACHE_REL / "failures" / first_key / "0001.json").read_bytes() == original_failure
    assert (root / sc.CACHE_REL / "attempts" / "history" / first_key / "0001.json").is_file()


def test_incompatible_checkpoint_identity_is_rejected(resume_root, tmp_path_factory):
    root, _ = resume_root
    archive = sc.export_checkpoint(root)
    incompatible = root / "incompatible.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(incompatible, "w") as dest:
        for name in source.namelist():
            data = source.read(name)
            if name == "checkpoint_manifest.json":
                manifest = json.loads(data)
                manifest["settings"]["digest"] = "different-model-digest"
                data = json.dumps(manifest).encode()
            dest.writestr(name, data)
    with pytest.raises(ValueError, match="settings differ"):
        sc.restore_checkpoint(incompatible, tmp_path_factory.mktemp("incompatible"))


def test_bounded_cli_forwarding_and_validation(monkeypatch, tmp_path, capsys):
    calls = []
    original_collect = sc.collect
    monkeypatch.setattr(sc, "collect", lambda root, **kwargs: calls.append(kwargs) or {"valid": 25})
    monkeypatch.setattr(sys, "argv", ["probe", "sample", "--root", str(tmp_path),
                                       "--max-new-samples", "25", "--retry-failed"])
    sc.main()
    assert calls[0]["max_new_samples"] == 25 and calls[0]["retry_failed"] is True
    monkeypatch.setattr(sys, "argv", ["probe", "status", "--root", str(tmp_path),
                                       "--max-new-samples", "25"])
    with pytest.raises(SystemExit):
        sc.main()
    monkeypatch.setattr(sc, "collect", original_collect)
    with pytest.raises(ValueError, match="positive integer"):
        sc.collect(tmp_path, max_new_samples=0)
