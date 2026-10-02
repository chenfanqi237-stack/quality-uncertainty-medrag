"""Offline checks for the Colab adapter; never contact Drive or Ollama."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag import stance_self_consistency as sc


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "cloud/prepare_self_consistency_colab.py"
NOTEBOOK = ROOT / "notebooks/colab_stance_self_consistency_10.ipynb"


def load_builder():
    spec = importlib.util.spec_from_file_location("colab_builder_offline", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_requires_exact_200_sample_checkpoint_and_matching_runtime(monkeypatch):
    builder = load_builder()
    def no_backend(*args, **kwargs):
        raise AssertionError("Offline Colab inspection must not contact a model")
    monkeypatch.setattr(sc, "OllamaTextGenerationBackend", no_backend)
    monkeypatch.setattr(sc, "verify_ollama_identity", no_backend)
    checks = builder.inspect_inputs()
    assert checks["model_digest"] == sc.DIGEST
    assert checks["ollama_version"] == sc.OLLAMA_VERSION
    assert checks["checkpoint_cache_directory"] == sc.CACHE_REL.as_posix()
    assert checks["checkpoint_sha256"] == builder.sha256(builder.CHECKPOINT)
    assert checks["runtime_sha256"] == builder.sha256(builder.RUNTIME)

    monkeypatch.setattr(builder, "CHECKPOINT", ROOT / "cloud/checkpoint/stance_self_consistency_10_checkpoint.zip")
    with pytest.raises(ValueError, match="200-sample state"):
        builder.inspect_inputs()


def test_checkpoint_selection_requires_200_and_prefers_highest_progress():
    builder = load_builder()
    records = [
        {"path": "valid200.zip", "status": {"valid": 200, "failed": 0},
         "timestamp": "2026-10-01T00:00:00Z", "mtime_ns": 1},
        {"path": "valid225-a.zip", "status": {"valid": 225, "failed": 0},
         "timestamp": "2026-10-02T00:00:00Z", "mtime_ns": 2},
        {"path": "valid225-b.zip", "status": {"valid": 225, "failed": 1},
         "timestamp": "2026-10-03T00:00:00Z", "mtime_ns": 3},
    ]
    assert builder.select_latest_compatible(records)["path"] == "valid225-b.zip"
    with pytest.raises(ValueError, match="at least 200"):
        builder.select_latest_compatible([
            {"path": "valid199.zip", "status": {"valid": 199, "failed": 0}}
        ])


def test_colab_notebook_discovers_resume_and_persists_every_25_sample_chunk():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 3
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in notebook["cells"])
    setup, gpu, batch = ("".join(cell["source"]) for cell in notebook["cells"])
    for source in (setup, gpu, batch):
        ast.parse(source)
    builder = load_builder()
    assert builder.sha256(builder.RUNTIME) in setup
    assert builder.sha256(builder.CHECKPOINT) in setup
    assert "drive.mount('/content/drive')" in setup
    assert "DRIVE_DIRECTORY = Path('/content/drive/MyDrive/medrag_self_consistency')" in setup
    assert "DRIVE_DIRECTORY.rglob('*checkpoint*.zip')" in setup
    assert "restore_checkpoint(LOCAL_CHECKPOINT, PROJECT_DIR)" in setup
    assert "MINIMUM_VALID_SAMPLES = 200" in setup
    assert "selected = max(compatible" in setup
    assert "frozen_settings()" in setup
    assert "Initial zero-inference status:" in setup
    assert "ABORT BEFORE INFERENCE" in setup
    assert "batch02_checkpoint.zip" not in setup
    assert "size_vram" in gpu and "nvidia-smi" in gpu
    assert "EXPECTED_MODEL_DIGEST" in gpu and "EXPECTED_OLLAMA_VERSION" in gpu
    assert "'prompt': ''" in gpu  # Model preload has no stance input.
    assert "while True" not in gpu
    assert "ollama', 'pull', 'qwen3:8b'" in gpu
    assert "'--max-new-samples', '25'" in batch
    assert batch.count("subprocess.run(batch_command") == 1
    assert "while True" in batch
    assert "export_checkpoint(PROJECT_DIR)" in batch
    assert "restore_checkpoint(native, Path(temporary_root))" in batch
    assert "file_sha256(versioned) != native_sha256" in batch
    assert "checkpoint_valid" in batch and "versioned.exists()" in batch
    assert "published_state, last_persisted_path = publish_checkpoint" in batch
    assert "new_units < 0 or new_units > 25" in batch
    assert "NO_ELIGIBLE_SAMPLES_REMAIN" in batch
    assert "Chunk verified in Drive; continuing to the next bounded chunk." in batch
    assert "--retry-failed" not in batch
    assert "ollama', 'pull'" not in batch
    assert batch.index("subprocess.run(batch_command") < batch.index(
        "published_state, last_persisted_path = publish_checkpoint")
    assert "evaluate" not in setup + gpu + batch


def test_runtime_bundle_is_reused_without_rebuild():
    builder = load_builder()
    assert builder.sha256(builder.RUNTIME) == (
        "9f8a0070118a846d543891931d63c3195fffe529ddf1073e826342b87315a70d"
    )
