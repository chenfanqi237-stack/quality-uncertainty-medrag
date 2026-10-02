"""Offline integrity checks for the manual Kaggle self-consistency package."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

from quality_uncertainty_medrag import stance_self_consistency as sc


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "cloud/self_consistency_10_runtime.zip"
NOTEBOOK = ROOT / "notebooks/kaggle_stance_self_consistency_10.ipynb"


def test_runtime_bundle_contains_only_exact_current_code_and_unlabeled_inputs(tmp_path):
    with zipfile.ZipFile(BUNDLE) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        names = set(archive.namelist())
        assert names == set(manifest["files"]) | {"manifest.json"}
        assert len(names) == 8  # Six source modules, one input, one manifest.
        assert manifest["reference_labels_included"] is False
        assert manifest["medqa_gold_included"] is False
        assert manifest["model_digest"] == sc.DIGEST
        assert manifest["prompt_sha256"] == sc.sha_text(sc.STANCE_PROMPT)
        assert manifest["unlabeled_input_sha256"] == sc.INPUT_SHA256
        assert manifest["reference_sha256_for_later_evaluation"] == sc.REFERENCE_SHA256
        assert manifest["checkpoint_cache_directory"] == sc.CACHE_REL.as_posix()
        assert manifest["checkpoint_archive_name"] == sc.CHECKPOINT_NAME
        assert manifest["temperature"] == sc.TEMPERATURE == .7
        assert manifest["thinking"] is True and manifest["seeds"] == list(sc.SEEDS)
        assert manifest["pairs"] == 60 and manifest["samples_expected"] == 600
        assert manifest["default_max_new_samples"] == 25

        for name, expected_hash in manifest["files"].items():
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == expected_hash
            assert data == (ROOT / name).read_bytes()
            assert "stance_reference_60_adjudicated.csv" not in name
            assert "predictions_60.csv" not in name

        records = [json.loads(line) for line in archive.read(sc.INPUT_REL.as_posix()).decode().splitlines()]
        assert len(records) == len({sc.pair_key(row) for row in records}) == 60
        assert all(set(row) == set(sc.INPUT_FIELDS) for row in records)
        assert all(1 <= int(row["question_id"].rsplit("-", 1)[1]) <= 30 for row in records)
        assert not any({"answer", "answer_idx", "final_stance", "human_stance", "other_options"} & set(row)
                       for row in records)

        fresh_input = tmp_path / sc.INPUT_REL
        fresh_input.parent.mkdir(parents=True)
        fresh_input.write_bytes(archive.read(sc.INPUT_REL.as_posix()))
        assert sc.validate_restored_checkpoint(tmp_path) == {
            "expected": 600, "valid": 0, "failed": 0, "missing": 600,
            "completion_percentage": 0.0,
        }


def test_notebook_has_restore_status_sampling_and_final_native_export():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 3
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in notebook["cells"])
    setup, gpu, sampling = ("".join(cell["source"]) for cell in notebook["cells"])
    for source in (setup, gpu, sampling):
        ast.parse(source)
    assert "restore_checkpoint(checkpoints[0], R)" in setup
    assert "Initial status (zero model calls)" in setup
    assert "self_consistency_10_runtime.zip" in setup
    assert "hashlib.sha256(data).hexdigest() != digest" in setup
    assert "subprocess.run(['nvidia-smi'], check=True)" in gpu
    assert "OLLAMA_VERSION='0.34.4'" in gpu
    assert "models[0]['digest'] != manifest['model_digest']" in gpu

    tree = ast.parse(sampling)
    block = next(node for node in tree.body if isinstance(node, ast.Try))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "export_checkpoint"
               for stmt in block.finalbody for node in ast.walk(stmt))
    assert "shutil.copyfile(native, temporary)" in sampling
    assert "os.replace(temporary, published)" in sampling
    assert "RETRY_FAILED = False" in sampling
    assert "'status', '--root', str(R)" in sampling
    assert "'sample', '--root', str(R)" in sampling
    assert "'sample', '--root', str(R), '--max-new-samples', '25'" in sampling
    assert sampling.count("completed = subprocess.run(command") == 1
    assert "restore_checkpoint(native, Path(temporary_root))" in sampling
    assert "hashlib.sha256(published.read_bytes()).hexdigest()" in sampling
    assert "Download the checkpoint ZIP to your local computer and verify the downloaded file BEFORE starting another batch." in sampling
    assert sampling.index("'status', '--root', str(R)") < sampling.index("completed = subprocess.run(command")
    assert sampling.count("'status', '--root', str(R)") == 2
    assert "check=False" in sampling
    assert "evaluate" not in sampling
