"""Prepare the offline Colab adapter for resumable checkpointed sampling.

This builder only reads the local runtime bundle and checkpoint. It never
contacts Colab, Google Drive, Ollama, or any model service.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "cloud/self_consistency_10_runtime.zip"
CHECKPOINT = ROOT / "cloud/checkpoint/checkpoint_valid200_failed000_20261002T113900Z_a33d9adb.zip"
NOTEBOOK = ROOT / "notebooks/colab_stance_self_consistency_10.ipynb"
REQUIRED_STATUS = {"expected": 600, "valid": 200, "failed": 0, "missing": 400}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_rank(record: dict) -> tuple:
    """Prefer more valid samples, then more processed units, then recency."""
    status = record["status"]
    return (
        status["valid"],
        status["valid"] + status["failed"],
        record.get("timestamp", ""),
        record.get("mtime_ns", 0),
        str(record.get("path", "")),
    )


def select_latest_compatible(records: list[dict], *, minimum_valid: int = 200) -> dict:
    eligible = [record for record in records if record["status"]["valid"] >= minimum_valid]
    if not eligible:
        raise ValueError(f"No compatible checkpoint has at least {minimum_valid} valid samples")
    return max(eligible, key=checkpoint_rank)


def inspect_inputs() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from quality_uncertainty_medrag import stance_self_consistency as sc

    with zipfile.ZipFile(RUNTIME) as archive:
        if archive.testzip() is not None:
            raise ValueError("Runtime ZIP CRC check failed")
        manifest = json.loads(archive.read("manifest.json"))
        if (manifest.get("model_digest") != sc.DIGEST
                or manifest.get("prompt_version") != sc.CLASSIFIER_VERSION
                or manifest.get("unlabeled_input_sha256") != sc.INPUT_SHA256
                or manifest.get("reference_labels_included") is not False
                or manifest.get("medqa_gold_included") is not False
                or manifest.get("default_max_new_samples") != 25
                or set(archive.namelist()) != set(manifest["files"]) | {"manifest.json"}):
            raise ValueError("Runtime manifest differs from frozen experiment")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Runtime file SHA256 mismatch: {name}")
            if archive.read(name) != (ROOT / name).read_bytes():
                raise ValueError(f"Runtime file differs from current source: {name}")

    with zipfile.ZipFile(CHECKPOINT) as archive:
        if archive.testzip() is not None:
            raise ValueError("Checkpoint ZIP CRC check failed")
        checkpoint_manifest = json.loads(archive.read("checkpoint_manifest.json"))
        if checkpoint_manifest.get("settings") != sc.frozen_settings():
            raise ValueError("Checkpoint uses incompatible frozen settings")
        if checkpoint_manifest.get("status") != {
                **REQUIRED_STATUS, "completion_percentage": 100 / 3}:
            raise ValueError("Checkpoint manifest is not the verified 200-sample state")
    with tempfile.TemporaryDirectory(prefix="medrag-colab-preflight-") as temporary_root:
        actual = sc.restore_checkpoint(CHECKPOINT, Path(temporary_root))
    if any(actual[key] != value for key, value in REQUIRED_STATUS.items()):
        raise ValueError("Checkpoint cache identities do not yield 200 valid samples")

    return {"runtime_sha256": sha256(RUNTIME), "checkpoint_sha256": sha256(CHECKPOINT),
            "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
            "model_digest": sc.DIGEST, "ollama_version": sc.OLLAMA_VERSION,
            "prompt_version": sc.CLASSIFIER_VERSION,
            "checkpoint_cache_directory": sc.CACHE_REL.as_posix()}


def code(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip().splitlines(keepends=True)}


LEGACY_SETUP_PRE_200 = r'''
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path, PurePosixPath
import hashlib, json, shutil, sys, zipfile

# Upload these two local files into this Google Drive directory before Run All:
# cloud/self_consistency_10_runtime.zip and cloud/checkpoint/batch02_checkpoint.zip
DRIVE_DIRECTORY = Path('/content/drive/MyDrive/medrag_self_consistency')  # Configurable.
RUNTIME_FILE = DRIVE_DIRECTORY / 'self_consistency_10_runtime.zip'
INPUT_CHECKPOINT = DRIVE_DIRECTORY / 'batch02_checkpoint.zip'
PROJECT_DIR = Path('/content/quality-uncertainty-medrag')
LOCAL_CHECKPOINT = Path('/content/medrag_input_checkpoint_valid100.zip')
EXPECTED_RUNTIME_SHA256 = '<<RUNTIME_SHA256>>'
EXPECTED_CHECKPOINT_SHA256 = '<<CHECKPOINT_SHA256>>'
EXPECTED_MODEL_DIGEST = '<<MODEL_DIGEST>>'
EXPECTED_OLLAMA_VERSION = '<<OLLAMA_VERSION>>'
EXPECTED_INITIAL = {'expected': 600, 'valid': 100, 'failed': 0, 'missing': 500}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

for path, digest in ((RUNTIME_FILE, EXPECTED_RUNTIME_SHA256),
                     (INPUT_CHECKPOINT, EXPECTED_CHECKPOINT_SHA256)):
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError('Required Drive input missing or SHA256 mismatch: ' + str(path))

with zipfile.ZipFile(RUNTIME_FILE) as archive:
    if archive.testzip() is not None:
        raise ValueError('Runtime ZIP CRC check failed')
    runtime_manifest = json.loads(archive.read('manifest.json'))
    if not (runtime_manifest['reference_labels_included'] is False
            and runtime_manifest['medqa_gold_included'] is False
            and runtime_manifest['pairs'] == 60
            and runtime_manifest['samples_expected'] == 600
            and runtime_manifest['default_max_new_samples'] == 25
            and runtime_manifest['model_digest'] == EXPECTED_MODEL_DIGEST
            and runtime_manifest['ollama_version'] == EXPECTED_OLLAMA_VERSION
            and runtime_manifest['prompt_version'] == 'llm-medical-stance-v1'
            and runtime_manifest['temperature'] == 0.7
            and runtime_manifest['thinking'] is True
            and runtime_manifest['seeds'] == list(range(101, 111))):
        raise ValueError('Runtime settings differ from frozen experiment')
    if set(archive.namelist()) != set(runtime_manifest['files']) | {'manifest.json'}:
        raise ValueError('Runtime file inventory differs')
    for name, expected_hash in runtime_manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
            raise ValueError('Unsafe runtime file path')
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError('Runtime file SHA256 mismatch: ' + name)
        target = PROJECT_DIR.joinpath(*relative.parts)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Existing writable source differs: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)

sys.path.insert(0, str(PROJECT_DIR / 'src'))
from quality_uncertainty_medrag.stance_self_consistency import (
    CACHE_REL, CHECKPOINT_NAME, OUTPUT_REL, DIGEST, OLLAMA_VERSION,
    INPUT_REL, read_unlabeled, restore_checkpoint, validate_restored_checkpoint,
    export_checkpoint)
if DIGEST != EXPECTED_MODEL_DIGEST or OLLAMA_VERSION != EXPECTED_OLLAMA_VERSION:
    raise ValueError('Loaded source has different frozen model identity')
if CACHE_REL.as_posix() != runtime_manifest['checkpoint_cache_directory']:
    raise ValueError('Cache directory differs from runtime manifest')
assert len(read_unlabeled(PROJECT_DIR / INPUT_REL)) == 60

# A previous Colab output with any additional completed units must be selected
# explicitly; never silently restart from the older Kaggle V3 checkpoint.
for previous in DRIVE_DIRECTORY.glob('checkpoint_valid*.zip'):
    with zipfile.ZipFile(previous) as archive:
        prior = json.loads(archive.read('checkpoint_manifest.json'))
    prior_state = prior.get('status', {})
    if (prior.get('settings', {}).get('digest') == EXPECTED_MODEL_DIGEST
            and prior_state.get('valid', 0) + prior_state.get('failed', 0) > 100):
        raise RuntimeError('Drive already contains a newer checkpoint: ' + str(previous))

shutil.copyfile(INPUT_CHECKPOINT, LOCAL_CHECKPOINT)
if file_sha256(LOCAL_CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError('Local checkpoint copy changed during transfer')
with zipfile.ZipFile(LOCAL_CHECKPOINT) as archive:
    if archive.testzip() is not None:
        raise ValueError('Checkpoint ZIP CRC check failed')
restored = restore_checkpoint(LOCAL_CHECKPOINT, PROJECT_DIR)
initial = validate_restored_checkpoint(PROJECT_DIR)
if restored != initial or any(initial[key] != value for key, value in EXPECTED_INITIAL.items()):
    raise RuntimeError('ABORT BEFORE INFERENCE: restored status is not 600/100/0/500')
print('Initial zero-inference status:', json.dumps(initial, sort_keys=True))
print('Original Drive checkpoint preserved:', INPUT_CHECKPOINT)
'''


SETUP = r'''
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path, PurePosixPath
import hashlib, json, shutil, sys, tempfile, zipfile

DRIVE_DIRECTORY = Path('/content/drive/MyDrive/medrag_self_consistency')  # Configurable.
RUNTIME_FILE = DRIVE_DIRECTORY / 'self_consistency_10_runtime.zip'
PROJECT_DIR = Path('/content/quality-uncertainty-medrag')
LOCAL_CHECKPOINT = Path('/content/medrag_selected_checkpoint.zip')
EXPECTED_RUNTIME_SHA256 = '<<RUNTIME_SHA256>>'
KNOWN_BASELINE_CHECKPOINT_SHA256 = '<<CHECKPOINT_SHA256>>'
EXPECTED_MODEL_DIGEST = '<<MODEL_DIGEST>>'
EXPECTED_OLLAMA_VERSION = '<<OLLAMA_VERSION>>'
EXPECTED_SAMPLES = 600
MINIMUM_VALID_SAMPLES = 200

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def state_counts(status):
    return {name: int(status[name]) for name in ('expected', 'valid', 'failed', 'missing')}

if not RUNTIME_FILE.is_file() or file_sha256(RUNTIME_FILE) != EXPECTED_RUNTIME_SHA256:
    raise ValueError('Required runtime ZIP is missing or has the wrong SHA256: ' + str(RUNTIME_FILE))
with zipfile.ZipFile(RUNTIME_FILE) as archive:
    if archive.testzip() is not None:
        raise ValueError('Runtime ZIP CRC check failed')
    runtime_manifest = json.loads(archive.read('manifest.json'))
    if not (runtime_manifest['reference_labels_included'] is False
            and runtime_manifest['medqa_gold_included'] is False
            and runtime_manifest['pairs'] == 60
            and runtime_manifest['samples_expected'] == EXPECTED_SAMPLES
            and runtime_manifest['default_max_new_samples'] == 25
            and runtime_manifest['model_digest'] == EXPECTED_MODEL_DIGEST
            and runtime_manifest['ollama_version'] == EXPECTED_OLLAMA_VERSION
            and runtime_manifest['prompt_version'] == 'llm-medical-stance-v1'
            and runtime_manifest['temperature'] == 0.7
            and runtime_manifest['thinking'] is True
            and runtime_manifest['seeds'] == list(range(101, 111))):
        raise ValueError('Runtime settings differ from frozen experiment')
    if set(archive.namelist()) != set(runtime_manifest['files']) | {'manifest.json'}:
        raise ValueError('Runtime file inventory differs')
    for name, expected_hash in runtime_manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
            raise ValueError('Unsafe runtime file path')
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError('Runtime file SHA256 mismatch: ' + name)
        target = PROJECT_DIR.joinpath(*relative.parts)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Existing writable source differs: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)

sys.path.insert(0, str(PROJECT_DIR / 'src'))
from quality_uncertainty_medrag.stance_self_consistency import (
    CACHE_REL, CHECKPOINT_NAME, OUTPUT_REL, DIGEST, OLLAMA_VERSION,
    INPUT_REL, export_checkpoint, frozen_settings, read_unlabeled,
    restore_checkpoint, validate_restored_checkpoint)
if DIGEST != EXPECTED_MODEL_DIGEST or OLLAMA_VERSION != EXPECTED_OLLAMA_VERSION:
    raise ValueError('Loaded source has different frozen model identity')
if CACHE_REL.as_posix() != runtime_manifest['checkpoint_cache_directory']:
    raise ValueError('Cache directory differs from runtime manifest')
assert len(read_unlabeled(PROJECT_DIR / INPUT_REL)) == 60

def inspect_drive_checkpoint(path):
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None or 'checkpoint_manifest.json' not in archive.namelist():
                raise ValueError('ZIP integrity/manifest failure')
            manifest = json.loads(archive.read('checkpoint_manifest.json'))
        if manifest.get('settings') != frozen_settings():
            raise ValueError('frozen settings mismatch')
        declared = state_counts(manifest.get('status', {}))
        if (declared['expected'] != EXPECTED_SAMPLES
                or declared['valid'] + declared['failed'] + declared['missing'] != EXPECTED_SAMPLES):
            raise ValueError('invalid status totals')
        with tempfile.TemporaryDirectory(prefix='medrag-drive-checkpoint-') as temporary_root:
            restored = restore_checkpoint(path, Path(temporary_root))
        if state_counts(restored) != declared:
            raise ValueError('manifest/cache status mismatch')
        return {'path': path, 'status': declared, 'timestamp': manifest.get('timestamp', ''),
                'mtime_ns': path.stat().st_mtime_ns, 'sha256': file_sha256(path)}
    except Exception as exc:
        print('Rejected checkpoint candidate:', path.name, type(exc).__name__, str(exc))
        return None

candidate_paths = sorted(set(DRIVE_DIRECTORY.rglob('*checkpoint*.zip')))
validated = [record for record in (inspect_drive_checkpoint(path) for path in candidate_paths)
             if record is not None]
compatible = [record for record in validated
              if record['status']['valid'] >= MINIMUM_VALID_SAMPLES]
if not compatible:
    raise RuntimeError('ABORT BEFORE INFERENCE: no compatible Drive checkpoint has at least 200 valid samples')
selected = max(compatible, key=lambda record: (
    record['status']['valid'], record['status']['valid'] + record['status']['failed'],
    record['timestamp'], record['mtime_ns'], str(record['path'])))
shutil.copyfile(selected['path'], LOCAL_CHECKPOINT)
if file_sha256(LOCAL_CHECKPOINT) != selected['sha256']:
    raise ValueError('Selected checkpoint changed during local transfer')
restored = restore_checkpoint(LOCAL_CHECKPOINT, PROJECT_DIR)
initial = validate_restored_checkpoint(PROJECT_DIR)
if (state_counts(restored) != state_counts(initial)
        or initial['expected'] != EXPECTED_SAMPLES
        or initial['valid'] < MINIMUM_VALID_SAMPLES):
    raise RuntimeError('ABORT BEFORE INFERENCE: restored checkpoint is below confirmed progress')
print('Selected latest compatible Drive checkpoint:', selected['path'])
print('Selected checkpoint SHA256:', selected['sha256'])
print('Known 200-sample baseline SHA256:', KNOWN_BASELINE_CHECKPOINT_SHA256)
print('Initial zero-inference status:', json.dumps(state_counts(initial), sort_keys=True))
'''


MODEL_SETUP = r'''
import json, os, shutil, subprocess, time
from urllib.request import Request, urlopen
from pathlib import Path

model_ready = False
model_error = None
try:
    if initial['valid'] >= EXPECTED_SAMPLES:
        raise StopIteration('sampling already complete')
    subprocess.run(['nvidia-smi'], check=True)
    missing = [name for name in ('curl', 'tar', 'zstd') if shutil.which(name) is None]
    if missing:
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        subprocess.run(['apt-get', 'install', '-y', '--no-install-recommends'] + missing, check=True)
    if shutil.which('ollama') is None:
        installer = Path('/content/ollama-install.sh')
        subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '-o', str(installer)], check=True)
        subprocess.run(['sh', str(installer)],
                       env=dict(os.environ, OLLAMA_VERSION=EXPECTED_OLLAMA_VERSION), check=True)
    serve_env = dict(os.environ, OLLAMA_HOST='127.0.0.1:11434',
                     OLLAMA_MODELS='/content/ollama-models', OLLAMA_DEBUG='false')
    def api(path, payload=None, timeout=15):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request('http://127.0.0.1:11434' + path, data=data,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    try:
        version = api('/api/version')['version']
    except Exception:
        server_log = Path('/content/ollama-server.log').open('ab')
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
            raise RuntimeError('Ollama server did not start')
    if version != EXPECTED_OLLAMA_VERSION:
        raise RuntimeError('Ollama version mismatch')
    subprocess.run(['ollama', 'pull', 'qwen3:8b'], env=serve_env, check=True)
    models = [m for m in api('/api/tags')['models'] if m['name'] == 'qwen3:8b']
    if len(models) != 1 or models[0]['digest'] != EXPECTED_MODEL_DIGEST:
        raise RuntimeError('Frozen model digest mismatch')
    # Empty prompt preloads the model without creating a stance judgment.
    api('/api/generate', {'model': 'qwen3:8b', 'prompt': '',
                          'stream': False, 'keep_alive': '5m'}, timeout=600)
    loaded = [m for m in api('/api/ps')['models'] if m['name'] == 'qwen3:8b']
    if len(loaded) != 1 or loaded[0].get('size_vram', 0) <= 0:
        raise RuntimeError('qwen3:8b was not verified in GPU VRAM; sampling is blocked')
    subprocess.run(['ollama', 'ps'], env=serve_env, check=True)
    model_ready = True
    print('Verified GPU model VRAM bytes:', loaded[0]['size_vram'])
    print('Verified model digest:', models[0]['digest'])
except StopIteration:
    print('Sampling is already complete; GPU/model setup is skipped.')
except Exception as exc:
    model_error = type(exc).__name__ + ': ' + str(exc)
    print('GPU/Ollama setup failed; bounded sampling will be skipped:', model_error)
'''


LEGACY_SINGLE_BATCH = r'''
import hashlib, json, os, shutil, subprocess, sys, tempfile, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path

native = PROJECT_DIR / OUTPUT_REL / CHECKPOINT_NAME
run_env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR / 'src'),
               OLLAMA_HOST='127.0.0.1:11434', OLLAMA_MODELS='/content/ollama-models')
batch_command = [sys.executable, '-u', '-m', 'quality_uncertainty_medrag.stance_self_consistency',
                 'sample', '--root', str(PROJECT_DIR), '--max-new-samples', '25']
batch_error = None
try:
    if not model_ready:
        print('No sampling: GPU/model verification failed:', model_error)
    else:
        before = validate_restored_checkpoint(PROJECT_DIR)
        if any(before[key] != value for key, value in EXPECTED_INITIAL.items()):
            raise RuntimeError('Initial 600/100/0/500 checkpoint state changed; sampling blocked')
        print('Running exactly ONE bounded batch:', ' '.join(batch_command))
        completed = subprocess.run(batch_command, cwd=PROJECT_DIR, env=run_env, check=False)
        if completed.returncode != 0:
            batch_error = 'Sampling process returned non-zero: ' + str(completed.returncode)
            print(batch_error, '; exporting all valid cached progress')
except BaseException as exc:
    batch_error = type(exc).__name__ + ': ' + str(exc)
    print('Bounded batch interrupted:', batch_error)
finally:
    try:
        export_checkpoint(PROJECT_DIR)
    except Exception as exc:
        print('Final refresh failed; trying the prior atomic checkpoint:',
              type(exc).__name__, str(exc))
    if not native.is_file():
        raise RuntimeError('No valid project-native checkpoint exists; do not end the session')
    with zipfile.ZipFile(native) as archive:
        if archive.testzip() is not None or 'checkpoint_manifest.json' not in archive.namelist():
            raise ValueError('Project-native checkpoint ZIP integrity failed')
    with tempfile.TemporaryDirectory(prefix='medrag-colab-verify-') as temporary_root:
        verified = restore_checkpoint(native, Path(temporary_root))
    live = validate_restored_checkpoint(PROJECT_DIR)
    if verified != live:
        raise RuntimeError('Checkpoint is behind the local cache; keep this Colab session open and repair export')
    native_sha256 = file_sha256(native)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    versioned = DRIVE_DIRECTORY / (
        f"checkpoint_valid{verified['valid']:03d}_failed{verified['failed']:03d}_"
        f"{timestamp}_{uuid.uuid4().hex[:8]}.zip")
    if versioned.exists():
        raise FileExistsError('Refusing to overwrite an existing Drive checkpoint')
    temporary_drive = DRIVE_DIRECTORY / ('.' + versioned.name + '.tmp')
    try:
        shutil.copyfile(native, temporary_drive)
        if file_sha256(temporary_drive) != native_sha256:
            raise ValueError('Drive transfer SHA256 mismatch before publication')
        temporary_drive.rename(versioned)
    finally:
        temporary_drive.unlink(missing_ok=True)
    if file_sha256(versioned) != native_sha256:
        raise ValueError('Drive checkpoint SHA256 mismatch after publication')
    with zipfile.ZipFile(versioned) as archive:
        if archive.testzip() is not None:
            raise ValueError('Drive checkpoint ZIP CRC check failed')
    print('Final status:', json.dumps(verified, sort_keys=True))
    print('Drive checkpoint:', versioned, 'bytes=', versioned.stat().st_size,
          'SHA256=', native_sha256)
    print('Confirm this ZIP is visible in Google Drive and download/verify it before closing Colab.')
    print('STOP: this notebook executes only one 25-unit bounded batch.')
'''


BATCH = r'''
import json, os, shutil, subprocess, sys, tempfile, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path

native = PROJECT_DIR / OUTPUT_REL / CHECKPOINT_NAME
run_env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR / 'src'),
               OLLAMA_HOST='127.0.0.1:11434', OLLAMA_MODELS='/content/ollama-models')
batch_command = [sys.executable, '-u', '-m', 'quality_uncertainty_medrag.stance_self_consistency',
                 'sample', '--root', str(PROJECT_DIR), '--max-new-samples', '25']
published = []
last_persisted_counts = state_counts(initial)
last_persisted_path = selected['path']

def publish_checkpoint(reason, chunk_index):
    export_checkpoint(PROJECT_DIR)
    if not native.is_file():
        raise RuntimeError('Project-native checkpoint was not created')
    with zipfile.ZipFile(native) as archive:
        if archive.testzip() is not None or 'checkpoint_manifest.json' not in archive.namelist():
            raise ValueError('Project-native checkpoint ZIP integrity failed')
    with tempfile.TemporaryDirectory(prefix='medrag-native-verify-') as temporary_root:
        verified = restore_checkpoint(native, Path(temporary_root))
    live = validate_restored_checkpoint(PROJECT_DIR)
    if state_counts(verified) != state_counts(live):
        raise RuntimeError('Project-native checkpoint is behind the live cache')

    native_sha256 = file_sha256(native)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    versioned = DRIVE_DIRECTORY / (
        f"checkpoint_valid{live['valid']:03d}_failed{live['failed']:03d}_"
        f"chunk{chunk_index:03d}_{timestamp}_{uuid.uuid4().hex[:8]}.zip")
    if versioned.exists():
        raise FileExistsError('Refusing to overwrite an existing Drive checkpoint')
    temporary_drive = DRIVE_DIRECTORY / ('.' + versioned.name + '.tmp')
    try:
        shutil.copyfile(native, temporary_drive)
        if file_sha256(temporary_drive) != native_sha256:
            raise ValueError('Drive transfer SHA256 mismatch before publication')
        temporary_drive.rename(versioned)
    finally:
        temporary_drive.unlink(missing_ok=True)
    if file_sha256(versioned) != native_sha256:
        raise ValueError('Drive checkpoint SHA256 mismatch after publication')
    with zipfile.ZipFile(versioned) as archive:
        if archive.testzip() is not None:
            raise ValueError('Drive checkpoint ZIP CRC check failed')
    with tempfile.TemporaryDirectory(prefix='medrag-drive-verify-') as temporary_root:
        drive_verified = restore_checkpoint(versioned, Path(temporary_root))
    if state_counts(drive_verified) != state_counts(live):
        raise RuntimeError('Published Drive checkpoint does not reproduce live status')
    print('Verified Drive checkpoint:', versioned.resolve())
    print('Checkpoint reason:', reason, 'bytes=', versioned.stat().st_size,
          'SHA256=', native_sha256)
    print('Persisted status:', json.dumps(state_counts(live), sort_keys=True))
    published.append({'path': str(versioned), 'sha256': native_sha256,
                      'status': state_counts(live), 'reason': reason})
    return live, versioned

chunk_index = 0
terminal_reason = None
try:
    if initial['valid'] >= EXPECTED_SAMPLES:
        terminal_reason = 'COMPLETE_FROM_RESTORED_CHECKPOINT'
    elif not model_ready:
        terminal_reason = 'MODEL_SETUP_FAILED: ' + str(model_error)
    else:
        while True:
            before = validate_restored_checkpoint(PROJECT_DIR)
            before_counts = state_counts(before)
            if before['valid'] >= EXPECTED_SAMPLES:
                terminal_reason = 'COMPLETE'
                break
            if before['missing'] == 0:
                terminal_reason = ('NO_ELIGIBLE_SAMPLES_REMAIN; valid=' + str(before['valid'])
                                   + ' failed=' + str(before['failed']))
                break
            chunk_index += 1
            before_units = before['valid'] + before['failed']
            print('Starting bounded chunk', chunk_index, json.dumps(before_counts, sort_keys=True))
            print('Sampling command:', ' '.join(batch_command))
            process_error = None
            try:
                completed = subprocess.run(batch_command, cwd=PROJECT_DIR, env=run_env, check=False)
                if completed.returncode != 0:
                    process_error = 'sampling process returned ' + str(completed.returncode)
            except BaseException as exc:
                process_error = type(exc).__name__ + ': ' + str(exc)

            # The next chunk is unreachable until Drive publication verifies successfully.
            published_state, last_persisted_path = publish_checkpoint(
                'chunk_' + str(chunk_index), chunk_index)
            last_persisted_counts = state_counts(published_state)
            after_units = published_state['valid'] + published_state['failed']
            new_units = after_units - before_units
            if new_units < 0 or new_units > 25:
                terminal_reason = 'INVALID_BOUNDED_PROGRESS: ' + str(new_units)
                break
            if process_error is not None:
                terminal_reason = 'SAMPLING_ERROR_AFTER_CHECKPOINT: ' + process_error
                break
            if published_state['valid'] >= EXPECTED_SAMPLES:
                terminal_reason = 'COMPLETE'
                break
            if new_units == 0:
                terminal_reason = 'NO_PROGRESS; no eligible uncached sample was processed'
                break
            if published_state['missing'] == 0:
                terminal_reason = ('NO_ELIGIBLE_SAMPLES_REMAIN; valid='
                                   + str(published_state['valid']) + ' failed='
                                   + str(published_state['failed']))
                break
            print('Chunk verified in Drive; continuing to the next bounded chunk.')
except BaseException as exc:
    terminal_reason = 'UNEXPECTED_ERROR: ' + type(exc).__name__ + ': ' + str(exc)
    print(terminal_reason)
    try:
        live_counts = state_counts(validate_restored_checkpoint(PROJECT_DIR))
        if live_counts != last_persisted_counts:
            preserved, last_persisted_path = publish_checkpoint('unexpected_error', chunk_index)
            last_persisted_counts = state_counts(preserved)
        else:
            print('Latest live state was already preserved at:', last_persisted_path)
    except Exception as preserve_error:
        print('Checkpoint preservation attempt failed:', type(preserve_error).__name__, str(preserve_error))

final_status = validate_restored_checkpoint(PROJECT_DIR)
print('Terminal reason:', terminal_reason)
print('Final status:', json.dumps(state_counts(final_status), sort_keys=True))
print('Last verified Drive checkpoint:', last_persisted_path)
print('Verified Drive checkpoints created this run:', len(published))
if final_status['valid'] < EXPECTED_SAMPLES and final_status['missing'] == 0:
    print('Stopped below 600 valid samples because no eligible missing samples remain; failed samples were not retried.')
print('STOP: no further sampling command will be started by this notebook.')
'''


def main() -> None:
    checks = inspect_inputs()
    replacements = {
        "<<RUNTIME_SHA256>>": checks["runtime_sha256"],
        "<<CHECKPOINT_SHA256>>": checks["checkpoint_sha256"],
        "<<MODEL_DIGEST>>": checks["model_digest"],
        "<<OLLAMA_VERSION>>": checks["ollama_version"],
    }
    setup = SETUP
    for original, replacement in replacements.items():
        setup = setup.replace(original, replacement)
    notebook = {
        "cells": [code(setup, "mount-discover-and-restore"),
                  code(MODEL_SETUP, "verify-gpu-and-model-once"),
                  code(BATCH, "sequential-bounded-chunks")],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(NOTEBOOK), "runtime_sha256": checks["runtime_sha256"],
                      "checkpoint_sha256": checks["checkpoint_sha256"],
                      "checkpoint_valid": 200,
                      "chunk_size": 25}, indent=2))


if __name__ == "__main__":
    main()
