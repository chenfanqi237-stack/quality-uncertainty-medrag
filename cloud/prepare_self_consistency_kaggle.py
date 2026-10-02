"""Offline builder for the frozen, unlabeled Kaggle sampling runtime."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "cloud/self_consistency_10_runtime.zip"
NOTEBOOK = ROOT / "notebooks/kaggle_stance_self_consistency_10.ipynb"
PACKAGE = Path("src/quality_uncertainty_medrag")
INPUT = Path("outputs/stance_model_comparison/reference_60_adjudicated/unlabeled_inputs_60.jsonl")
FILES = tuple(PACKAGE / name for name in (
    "__init__.py", "models.py", "interfaces.py", "llm_stance.py",
    "ollama_backend.py", "stance_self_consistency.py")) + (INPUT,)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.strip().splitlines(keepends=True)}


def build_manifest() -> tuple[dict, dict[str, bytes]]:
    sys.path.insert(0, str(ROOT / "src"))
    from quality_uncertainty_medrag import stance_self_consistency as sc
    frozen = (sc.MODEL, sc.DIGEST, sc.OLLAMA_VERSION, sc.CLASSIFIER_VERSION,
              sc.TEMPERATURE, sc.SEEDS, sc.REFERENCE_SHA256, sc.INPUT_REL,
              sc.CACHE_REL, sc.CHECKPOINT_INTERVAL)
    expected = ("qwen3:8b", "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
                "0.34.4", "llm-medical-stance-v1", .7, tuple(range(101, 111)),
                "ae5d55ef412fa5a33997108fdebd36b13fa017953ca5281a36a3deeb5806065d",
                INPUT, sc.OUTPUT_REL / "cache", 25)
    if frozen != expected:
        raise ValueError("Frozen research settings or checkpoint implementation differ")
    rows = sc.read_unlabeled(ROOT / INPUT)
    if len(rows) != 60 or any(not 1 <= int(r["question_id"].rsplit("-", 1)[1]) <= 30 for r in rows):
        raise ValueError("Expected exactly 60 unlabeled dev 1–30 pairs")
    payload = {p.as_posix(): (ROOT / p).read_bytes() for p in FILES}
    manifest = {
        "bundle_version": 2, "purpose": "Unlabeled stochastic stance sampling only",
        "reference_labels_included": False, "medqa_gold_included": False,
        "model": sc.MODEL, "model_digest": sc.DIGEST, "ollama_version": sc.OLLAMA_VERSION,
        "prompt_version": sc.CLASSIFIER_VERSION, "prompt_sha256": sc.sha_text(sc.STANCE_PROMPT),
        "thinking": True, "temperature": sc.TEMPERATURE, "seeds": list(sc.SEEDS),
        "pairs": 60, "samples_expected": 600, "unlabeled_input_sha256": sc.INPUT_SHA256,
        "default_max_new_samples": 25,
        "reference_sha256_for_later_evaluation": sc.REFERENCE_SHA256,
        "checkpoint_cache_directory": sc.CACHE_REL.as_posix(),
        "checkpoint_archive_name": sc.CHECKPOINT_NAME,
        "files": {name: sha256(data) for name, data in payload.items()},
    }
    return manifest, payload


def write_bundle(manifest: dict, payload: dict[str, bytes]) -> None:
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE.with_name(BUNDLE.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
            info = zipfile.ZipInfo("manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(payload) | {"manifest.json"}:
                raise ValueError("Runtime bundle inventory or CRC differs")
            if any(sha256(archive.read(name)) != digest for name, digest in manifest["files"].items()):
                raise ValueError("Runtime bundle SHA256 mismatch")
        os.replace(temporary, BUNDLE)
    finally:
        temporary.unlink(missing_ok=True)


SETUP = r'''
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, subprocess, sys, uuid, zipfile
W = Path('/kaggle/working')
R = W / 'quality-uncertainty-medrag'
bundles = sorted(Path('/kaggle/input').rglob('self_consistency_10_runtime.zip'))
if len(bundles) > 1:
    raise RuntimeError('Multiple runtime bundles supplied')
if bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        if archive.testzip() is not None:
            raise ValueError('Runtime ZIP CRC check failed')
        manifest = json.loads(archive.read('manifest.json'))
        source_names = set(archive.namelist())
        file_bytes = {name: archive.read(name) for name in manifest['files']}
    bundle_source = str(bundles[0])
else:
    # Some Kaggle uploads expose the ZIP's contents as an expanded dataset.
    matches = []
    for path in Path('/kaggle/input').rglob('manifest.json'):
        candidate = json.loads(path.read_text(encoding='utf-8'))
        if candidate.get('bundle_version') == 2 and candidate.get('purpose') == 'Unlabeled stochastic stance sampling only':
            matches.append((path, candidate))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one uploaded runtime ZIP or expanded runtime')
    manifest_path, manifest = matches[0]
    source_names = set(manifest['files']) | {'manifest.json'}
    file_bytes = {name: (manifest_path.parent / name).read_bytes() for name in manifest['files']}
    bundle_source = str(manifest_path.parent)
def verify_and_install_runtime():
    if not (manifest['bundle_version'] == 2 and manifest['reference_labels_included'] is False
            and manifest['medqa_gold_included'] is False and manifest['pairs'] == 60
            and manifest['samples_expected'] == 600 and manifest['default_max_new_samples'] == 25
            and manifest['model'] == 'qwen3:8b'
            and manifest['ollama_version'] == '0.34.4'
            and manifest['model_digest'] == '500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
            and manifest['prompt_version'] == 'llm-medical-stance-v1'
            and manifest['temperature'] == 0.7 and manifest['thinking'] is True
            and manifest['seeds'] == list(range(101, 111))):
        raise ValueError('Frozen runtime manifest differs')
    if source_names != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Runtime ZIP file inventory differs')
    for name, digest in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
            raise ValueError('Unsafe runtime ZIP path')
        data = file_bytes[name]
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Runtime ZIP SHA256 mismatch: ' + name)
        target = R.joinpath(*relative.parts)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Existing runtime file differs: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
verify_and_install_runtime()
sys.path.insert(0, str(R / 'src'))
from quality_uncertainty_medrag.stance_self_consistency import (
    CACHE_REL, CHECKPOINT_NAME, OUTPUT_REL, export_checkpoint,
    restore_checkpoint, validate_restored_checkpoint)
if CACHE_REL.as_posix() != manifest['checkpoint_cache_directory'] or CHECKPOINT_NAME != manifest['checkpoint_archive_name']:
    raise ValueError('Bundled checkpoint path differs')
# Add one previous checkpoint ZIP as an input to resume automatically.
checkpoints = sorted(Path('/kaggle/input').rglob(CHECKPOINT_NAME))
if len(checkpoints) > 1:
    raise RuntimeError('Multiple checkpoints supplied; keep only the intended checkpoint input')
if checkpoints:
    print('Restored checkpoint:', checkpoints[0], restore_checkpoint(checkpoints[0], R))
print('Verified unlabeled runtime bundle:', bundle_source)
print('Initial status (zero model calls):', json.dumps(validate_restored_checkpoint(R), sort_keys=True))
'''


OLLAMA = r'''
from urllib.request import urlopen
import json, os, shutil, subprocess, time
from pathlib import Path
W = Path('/kaggle/working')
model_ready = False
model_error = None
try:
    subprocess.run(['nvidia-smi'], check=True)
    missing = [name for name in ('curl', 'tar', 'zstd') if shutil.which(name) is None]
    if missing:
        prefix = [] if os.geteuid() == 0 else ['sudo', '-n']
        env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
        subprocess.run(prefix + ['apt-get', 'update', '-qq'], env=env, check=True)
        subprocess.run(prefix + ['apt-get', 'install', '-y', '--no-install-recommends'] + missing,
                       env=env, check=True)
    if shutil.which('ollama') is None:
        installer = W / 'ollama-install.sh'
        subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '-o', str(installer)], check=True)
        subprocess.run(['sh', str(installer)], env=dict(os.environ, OLLAMA_VERSION='0.34.4'), check=True)
    serve_env = dict(os.environ, OLLAMA_HOST='127.0.0.1:11434',
                     OLLAMA_MODELS=str(W / 'ollama-models'), OLLAMA_DEBUG='false')
    def api(path):
        with urlopen('http://localhost:11434' + path, timeout=15) as response:
            return json.load(response)
    try:
        version = api('/api/version')['version']
    except Exception:
        server_log = (W / 'ollama-server.log').open('ab')
        server = subprocess.Popen(['ollama', 'serve'], env=serve_env,
                                  stdout=server_log, stderr=subprocess.STDOUT)
        for _ in range(90):
            try:
                version = api('/api/version')['version']
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError('Ollama server exited; inspect ollama-server.log')
                time.sleep(1)
        else:
            raise RuntimeError('Ollama did not start')
    if version != '0.34.4':
        raise RuntimeError('Ollama version differs from frozen 0.34.4')
    subprocess.run(['ollama', 'pull', 'qwen3:8b'], env=serve_env, check=True)
    models = [item for item in api('/api/tags')['models'] if item['name'] == 'qwen3:8b']
    if len(models) != 1 or models[0]['digest'] != manifest['model_digest']:
        raise RuntimeError('Frozen qwen3:8b digest mismatch')
    model_ready = True
    print('Verified Ollama version/digest:', version, models[0]['digest'])
except Exception as exc:
    model_error = type(exc).__name__ + ': ' + str(exc)
    print('Runtime/model setup failed; sampling skipped:', model_error)
'''


SAMPLING = r'''
import hashlib, json, os, shutil, subprocess, sys, tempfile, uuid, zipfile
from pathlib import Path
R = Path('/kaggle/working/quality-uncertainty-medrag')
W = Path('/kaggle/working')
native = R / OUTPUT_REL / CHECKPOINT_NAME
published = W / CHECKPOINT_NAME
run_env = dict(os.environ, PYTHONPATH=str(R / 'src'), OLLAMA_HOST='127.0.0.1:11434',
               OLLAMA_MODELS=str(W / 'ollama-models'))
RETRY_FAILED = False  # Change explicitly only when previously failed samples should be retried.
command = [sys.executable, '-u', '-m', 'quality_uncertainty_medrag.stance_self_consistency',
           'sample', '--root', str(R), '--max-new-samples', '25']
if RETRY_FAILED:
    command.append('--retry-failed')
try:
    print('Status before sampling (zero model calls):')
    before = subprocess.run([sys.executable, '-m', 'quality_uncertainty_medrag.stance_self_consistency',
                             'status', '--root', str(R)], cwd=R, env=run_env, check=False)
    if before.returncode != 0:
        print('Status failed; sampling skipped; checkpoint export will still be attempted')
    elif model_ready:
        print('Sampling command:', ' '.join(command))
        completed = subprocess.run(command, cwd=R, env=run_env, check=False)
        if completed.returncode != 0:
            print('Sampling returned non-zero:', completed.returncode,
                  'See error output above; preserving latest checkpoint')
    else:
        print('Sampling skipped because GPU/Ollama setup failed:', model_error)
    print('Status after sampling (zero model calls):')
    subprocess.run([sys.executable, '-m', 'quality_uncertainty_medrag.stance_self_consistency',
                    'status', '--root', str(R)], cwd=R, env=run_env, check=False)
except BaseException as exc:
    print('Sampling interrupted or failed:', type(exc).__name__, str(exc))
finally:
    # Use the project's atomic exporter; retain the previous ZIP if refresh fails.
    try:
        export_checkpoint(R)
    except Exception as exc:
        print('Final checkpoint refresh failed; retaining prior native checkpoint:',
              type(exc).__name__, str(exc))
    if native.is_file():
        with zipfile.ZipFile(native) as archive:
            if archive.testzip() is not None or 'checkpoint_manifest.json' not in archive.namelist():
                raise ValueError('Native checkpoint ZIP failed integrity check')
        # restore_checkpoint verifies every manifest SHA256, frozen settings,
        # exact cache identities and all expected progress counts in isolation.
        with tempfile.TemporaryDirectory(prefix='stance-checkpoint-verify-') as temporary_root:
            verified_status = restore_checkpoint(native, Path(temporary_root))
        temporary = published.with_name(published.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            shutil.copyfile(native, temporary)
            os.replace(temporary, published)
        finally:
            temporary.unlink(missing_ok=True)
        archive_hash = hashlib.sha256(published.read_bytes()).hexdigest()
        print('Checkpoint available for download:', published.resolve(),
              'size_bytes=', published.stat().st_size, 'SHA256=', archive_hash,
              'verified_status=', verified_status)
        print('Download the checkpoint ZIP to your local computer and verify the downloaded file BEFORE starting another batch.')
    else:
        print('No native checkpoint exists to publish; inspect setup/status errors above')
        print('Do not start another batch until a valid checkpoint ZIP has been downloaded and verified.')
'''


def write_notebook() -> None:
    notebook = {"cells": [code(SETUP), code(OLLAMA), code(SAMPLING)],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    manifest, payload = build_manifest()
    write_bundle(manifest, payload)
    write_notebook()
    print(BUNDLE, BUNDLE.stat().st_size, sha256(BUNDLE.read_bytes()), NOTEBOOK)


if __name__ == "__main__":
    main()
