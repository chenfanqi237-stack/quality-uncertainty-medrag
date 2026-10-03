"""Question-27-only runtime, durable checkpoints and evaluator. No default inference."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cloud"))
from quality_uncertainty_medrag import stance_self_consistency as sc
from quality_uncertainty_medrag.pubmed_evidence_type import evidence_type_from_publication_types
from question27_adapter import OPTIONS, RULES, evaluate

QID = "medqa-us-dev-000027"
PILOT = Path("outputs/aggregation_pilot/medqa_question27_k3_v1")
ORIGINAL_NAME = "checkpoint_valid600_20261002T152105Z_553a09b9.zip"
ORIGINAL_SHA = "356cdb4f30de17952e22176bc6127b41384bae3a795baf8c3724612162ee4f34"
ASSETS = ("pilot_input.jsonl", "input_manifest.json", "sampling_manifest.json", "frozen_quality_weights.json")
FIELDS = {"schema_version", "question_id", "candidate_option_id", "evidence_doc_id", "question_stem",
          "candidate_option_text", "evidence_title", "evidence_abstract", "pmid", "retrieval_rank",
          "publication_types", "evidence_type", "quality_weight", "partition"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flush_storage(handle):
    handle.flush()
    try:
        os.fsync(handle.fileno())
        return True
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise
        print("Storage does not implement fsync; ZIP/size/SHA256 readback verification is still required")
        return False


def projection(row):
    return {k: row[k] for k in sc.INPUT_FIELDS}


def load_inputs(root=ROOT):
    root = Path(root)
    directory = root / PILOT
    manifest, sampling = read_json(directory / "input_manifest.json"), read_json(directory / "sampling_manifest.json")
    weights = read_json(directory / "frozen_quality_weights.json")
    for name, expected in manifest["asset_sha256"].items():
        if name not in ASSETS or digest(directory / name) != expected:
            raise ValueError("Pilot asset hash mismatch: " + name)
    if set(manifest["asset_sha256"]) != set(ASSETS) - {"input_manifest.json"}:
        raise ValueError("Incomplete pilot asset inventory")
    if sampling["frozen_sampling_settings"] != sc.frozen_settings() or sampling["answer_rules"] != RULES:
        raise ValueError("Frozen inference or answer rules differ")
    signature = sc.sha_text(sc.canonical({k: v for k, v in sampling.items() if k != "pilot_signature"}))
    if signature != sampling["pilot_signature"]:
        raise ValueError("Pilot signature differs")
    for name, expected in sampling["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen/runtime source hash differs: " + name)
    with (directory / "pilot_input.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != 15 or len({sc.pair_key(r) for r in rows}) != 15:
        raise ValueError("STOP: expected exactly 15 unique question-27 pairs")
    vectors = []
    evidence_fields = ("evidence_doc_id", "pmid", "retrieval_rank", "evidence_title", "evidence_abstract", "publication_types", "evidence_type", "quality_weight")
    stems = set()
    for option in OPTIONS:
        selected = [r for r in rows if r["candidate_option_id"] == option]
        if len(selected) != 3 or [r["retrieval_rank"] for r in selected] != [1, 2, 3] or len({r["candidate_option_text"] for r in selected}) != 1:
            raise ValueError("STOP: options/evidence order differs")
        vector = []
        for r in selected:
            if set(r) != FIELDS or r["question_id"] != QID or r["partition"] != "medqa_us_dev_1_30" or r["pmid"] != r["evidence_doc_id"]:
                raise ValueError("Pilot input schema/identity differs")
            if evidence_type_from_publication_types(r["publication_types"]).value != r["evidence_type"] or r["quality_weight"] != weights[r["evidence_type"]]:
                raise ValueError("Frozen study type/quality differs")
            stems.add(r["question_stem"])
            vector.append(sc.canonical({k: r[k] for k in evidence_fields}))
        if sum(r["quality_weight"] > 0 for r in selected) != 2:
            raise ValueError("STOP: expected two positive-quality articles")
        vectors.append(vector)
    if len(stems) != 1 or any(v != vectors[0] for v in vectors) or set(r["candidate_option_id"] for r in rows) != set(OPTIONS):
        raise ValueError("STOP: five options do not share identical evidence")
    if ["|".join(sc.pair_key(r)) for r in rows] != sampling["pair_identities"] or weights["other"] != 0:
        raise ValueError("Frozen pair list or OTHER mapping differs")
    return rows, sampling


def inspect_original(path, rows):
    """Validate all 600 cache contexts, prompts, seeds and hashes in a temporary tree."""
    path = Path(path)
    if digest(path) != ORIGINAL_SHA:
        raise ValueError("Immutable original archive SHA256 differs")
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Original archive CRC failure")
        original_manifest = json.loads(z.read("checkpoint_manifest.json"))
    if original_manifest["settings"] != sc.frozen_settings() or original_manifest["status"] != {"expected": 600, "valid": 600, "failed": 0, "missing": 0, "completion_percentage": 100.0}:
        raise ValueError("Original archive is not the frozen complete 600-sample experiment")
    reused = []
    with tempfile.TemporaryDirectory(prefix="q27-original-audit-") as temp:
        sc.restore_checkpoint(path, Path(temp))
        inventory = sc.scan_samples(Path(temp))
        required = {sc.pair_key(r): projection(r) for r in rows}
        for item in inventory:
            key = sc.pair_key(item["row"])
            if key in required:
                expected, _, cache_key = sc.sample_context(required[key], item["seed"])
                if item["context"] != expected or item["cache_key"] != cache_key:
                    raise ValueError("Exact pair exists but frozen text/prompt identity differs")
                reused.append(item)
    return reused, {"sha256": ORIGINAL_SHA, "status": original_manifest["status"], "all_cache_contexts_validated": 600,
                    "settings_validated": True, "reusable_samples": len(reused)}


def new_meta(signature):
    return {"state_version": 1, "pilot_signature": signature, "revision": 0, "active_intents": {},
            "retry_authorizations": [], "reuse_provenance": None, "model_identity": None}


def expected_inventory(rows):
    out = []
    for row in rows:
        for seed in sc.SEEDS:
            context, _, key = sc.sample_context(projection(row), seed)
            out.append({"row": row, "seed": seed, "context": context, "cache_key": key})
    if len(out) != 150 or len({i["cache_key"] for i in out}) != 150:
        raise ValueError("Expected 150 unique pair-seed contexts")
    return out


def scan_state(root, state_dir):
    rows, sampling = load_inputs(root)
    state_dir = Path(state_dir)
    meta = read_json(state_dir / "pilot_state.json")
    if set(meta) != {"state_version", "pilot_signature", "revision", "active_intents", "retry_authorizations", "reuse_provenance", "model_identity"}:
        raise ValueError("Sampling-state schema differs")
    if meta["state_version"] != 1 or meta["pilot_signature"] != sampling["pilot_signature"] or type(meta["revision"]) is not int or meta["revision"] < 0:
        raise ValueError("Sampling state belongs to a different pilot")
    inventory = expected_inventory(rows)
    allowed = {i["cache_key"] for i in inventory}
    if set(meta["active_intents"]) - allowed:
        raise ValueError("Unknown in-flight sample")
    contexts = {i["cache_key"]: i["context"] for i in inventory}
    for key, intent in meta["active_intents"].items():
        if set(intent) != {"ids", "seed", "timestamp"} or intent["ids"] != contexts[key]["ids"] or intent["seed"] != contexts[key]["seed"] or not isinstance(intent["timestamp"], str):
            raise ValueError("In-flight intent exact identity differs")
    identity = meta["model_identity"]
    if identity is not None:
        required_identity = {"model": sc.MODEL, "digest": sc.DIGEST, "ollama_version": sc.OLLAMA_VERSION,
                             "verified_live_identity": True, "backend": "unchanged OllamaTextGenerationBackend",
                             "thinking": True, "temperature": sc.TEMPERATURE}
        if any(identity.get(k) != v for k, v in required_identity.items()) or type(identity.get("size_vram")) is not int or identity["size_vram"] <= 0:
            raise ValueError("Stored live model/config/GPU identity differs from frozen inference")
    cache = state_dir / "cache"
    for directory in (cache, cache / "attempts"):
        if any(p.stem not in allowed for p in directory.glob("*.json")):
            raise ValueError("Unexpected sample/attempt identity")
    for directory in (cache / "failures", cache / "attempts" / "history"):
        if any(p.name not in allowed or not p.is_dir() for p in directory.glob("*")):
            raise ValueError("Unexpected failure/history identity")
    for item in inventory:
        key, context = item["cache_key"], item["context"]
        path = cache / (key + ".json")
        entry = sc.read_cached_sample(path, context) if path.exists() else None
        audit_path = cache / "attempts" / (key + ".json")
        audit = sc.read_attempt_audit(audit_path, context) if audit_path.exists() else None
        failures = sc.failure_records(cache, key, context)
        cycles = {}
        for p in (cache / "attempts" / "history" / key).glob("*.json"):
            historical = sc.read_attempt_audit(p, context)
            if p.name != f"{historical['cycle']:04d}.json":
                raise ValueError("Invalid historical attempt cycle")
            cycles[historical["cycle"]] = len(historical["attempts"])
        if entry and audit and audit["status"] == "COMPLETE" and audit["attempts"] != entry["attempts"]:
            raise ValueError("Cache/completed attempt disagreement")
        if entry and audit and audit["status"] != "COMPLETE" and audit["attempts"] != entry["attempts"][:len(audit["attempts"])]:
            raise ValueError("Cache/interrupted attempt prefix disagreement")
        if not entry and audit and audit["status"] == "COMPLETE":
            raise ValueError("Complete audit without cache")
        state = "VALID" if entry else "INTERRUPTED" if key in meta["active_intents"] else "FAILED" if (audit and audit["status"] == "FAILED") or failures else "MISSING"
        if audit:
            cycles[audit["cycle"]] = len(audit["attempts"])
        for failure in failures:
            cycles[failure["cycle"]] = max(cycles.get(failure["cycle"], 0), 3)
        if entry:
            cycle = audit["cycle"] if audit else max(cycles, default=0) + 1
            cycles[cycle] = entry["generation_attempts"]
        item.update(status=state, entry=entry, recorded_generation_attempts=sum(cycles.values()))
    counts = Counter(i["status"] for i in inventory)
    status = {"expected": 150, "valid": counts["VALID"], "failed": counts["FAILED"],
              "interrupted": counts["INTERRUPTED"], "missing": counts["MISSING"],
              "complete_samples": counts["VALID"] == 150 and counts["FAILED"] == 0 and counts["INTERRUPTED"] == 0,
              "recorded_generation_attempts": sum(i["recorded_generation_attempts"] for i in inventory),
              "explicit_interrupted_retry_authorizations": sum(r["prior_status"] == "INTERRUPTED" for r in meta["retry_authorizations"])}
    return rows, sampling, meta, inventory, status


def save_meta(state_dir, meta):
    meta["revision"] += 1
    sc.write_json(Path(state_dir) / "pilot_state.json", meta)


def settle_completed_intents(root, state_dir):
    """A validated cache is a known result even if post-call bookkeeping died."""
    _, _, meta, inventory, _ = scan_state(root, state_dir)
    settled = {i["cache_key"] for i in inventory if i["status"] == "VALID"} & set(meta["active_intents"])
    if settled:
        for key in settled:
            meta["active_intents"].pop(key)
        save_meta(state_dir, meta)
        print("Recovered", len(settled), "exact valid cached result(s); cleared stale intent(s) with zero new model calls")


def archive_payload(path, manifest_name):
    """CRC, inventory, traversal and every member hash checked before extraction."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or manifest_name not in names or z.testzip() is not None:
            raise ValueError("ZIP CRC/duplicate/manifest failure")
        manifest = json.loads(z.read(manifest_name))
        if set(names) != set(manifest["files"]) | {manifest_name}:
            raise ValueError("ZIP inventory differs from manifest")
        payload = {}
        for name, wanted in manifest["files"].items():
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name or p.as_posix() != name:
                raise ValueError("Unsafe ZIP member")
            data = z.read(name)
            if sc.sha_bytes(data) != wanted:
                raise ValueError("ZIP member SHA256 mismatch")
            payload[name] = data
    return manifest, payload


def inspect_checkpoint(path, root, expected_sha=None):
    if expected_sha is not None and digest(path) != expected_sha:
        raise ValueError("Pilot checkpoint external SHA256 mismatch")
    manifest, payload = archive_payload(path, "checkpoint_manifest.json")
    rows, sampling = load_inputs(root)
    if manifest["checkpoint_version"] != "question27-k3-v1" or manifest["pilot_signature"] != sampling["pilot_signature"] or manifest["settings"] != sc.frozen_settings():
        raise ValueError("Pilot checkpoint/config incompatible")
    permitted_assets = {(PILOT / name).as_posix() for name in ASSETS}
    allowed_keys = {i["cache_key"] for i in expected_inventory(rows)}
    if not permitted_assets <= set(payload) or "state/pilot_state.json" not in payload:
        raise ValueError("Checkpoint misses pilot inputs/state")
    for name, data in payload.items():
        if name in permitted_assets:
            if data != (Path(root) / name).read_bytes():
                raise ValueError("Checkpoint pilot input bytes differ")
        elif not (name == "state/pilot_state.json" or name.startswith("state/cache/") and name.endswith(".json")):
            raise ValueError("Unrelated checkpoint payload")
        elif name.startswith("state/cache/"):
            parts = name.removeprefix("state/cache/").split("/")
            direct = len(parts) == 1 and parts[0].removesuffix(".json") in allowed_keys
            attempt = len(parts) == 2 and parts[0] == "attempts" and parts[1].removesuffix(".json") in allowed_keys
            failure = len(parts) == 3 and parts[0] == "failures" and parts[1] in allowed_keys and parts[2].removesuffix(".json").isdigit()
            history = len(parts) == 4 and parts[:2] == ["attempts", "history"] and parts[2] in allowed_keys and parts[3].removesuffix(".json").isdigit()
            if not (direct or attempt or failure or history):
                raise ValueError("Unrelated/nonnative cache path in pilot checkpoint")
    with tempfile.TemporaryDirectory(prefix="q27-checkpoint-verify-") as temp:
        stage = Path(temp)
        for name, data in payload.items():
            if name.startswith("state/"):
                target = stage / name.removeprefix("state/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        _, _, meta, _, status = scan_state(root, stage)
    if status != manifest["status"] or meta["revision"] != manifest["revision"]:
        raise ValueError("Checkpoint manifest/state count disagreement")
    if manifest["pair_identities"] != sampling["pair_identities"] or manifest["source_sha256"] != sampling["source_sha256"]:
        raise ValueError("Checkpoint exact identities/source provenance differ")
    if manifest.get("final") and (not status["complete_samples"] or meta["active_intents"] or not (meta.get("model_identity") or {}).get("verified_live_identity")):
        raise ValueError("Final checkpoint does not satisfy completion conditions")
    return manifest, payload


def export_backup(root, state_dir, local_dir, drive_dir, *, final=False):
    """Unique local ZIP -> verified Drive copy -> hash/receipt; fail closed."""
    rows, sampling, meta, _, status = scan_state(root, state_dir)
    if final and (not status["complete_samples"] or meta["active_intents"] or not (meta.get("model_identity") or {}).get("verified_live_identity")):
        raise RuntimeError("Cannot export final: incomplete samples, active intent or unverified live identity")
    local_dir, drive_dir = Path(local_dir), Path(drive_dir)
    for p in (local_dir, drive_dir):
        p.mkdir(parents=True, exist_ok=True)
    suffix = sc.timestamp().replace(":", "").replace("-", "").replace(".", "") + "_" + uuid.uuid4().hex[:8]
    name = f"q27_{'final' if final else 'checkpoint'}_valid{status['valid']:03d}_r{meta['revision']:06d}_{suffix}.zip"
    local = local_dir / name
    payload = {(PILOT / n).as_posix(): (Path(root) / PILOT / n).read_bytes() for n in ASSETS}
    state_dir = Path(state_dir).resolve()
    for p in [state_dir / "pilot_state.json", *sorted((state_dir / "cache").rglob("*.json"))]:
        if p.is_symlink() or not p.resolve().is_relative_to(state_dir):
            raise ValueError("Sampling state path escaped")
        payload["state/" + p.relative_to(state_dir).as_posix()] = p.read_bytes()
    manifest = {"checkpoint_version": "question27-k3-v1", "timestamp": sc.timestamp(), "pilot_signature": sampling["pilot_signature"],
                "settings": sc.frozen_settings(), "status": status, "revision": meta["revision"], "final": final,
                "pair_identities": sampling["pair_identities"], "source_sha256": sampling["source_sha256"],
                "files": {n: sc.sha_bytes(data) for n, data in payload.items()}}
    temp = local.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "x", zipfile.ZIP_DEFLATED) as z:
        for n, data in payload.items():
            z.writestr(n, data)
        z.writestr("checkpoint_manifest.json", sc.canonical(manifest) + "\n")
    inspect_checkpoint(temp, root)
    os.replace(temp, local)
    wanted, size = digest(local), local.stat().st_size
    destination = drive_dir / name
    if destination.exists() or destination.name == ORIGINAL_NAME:
        raise FileExistsError("Never overwrite a research checkpoint")
    transferring = drive_dir / (name + "." + uuid.uuid4().hex + ".uploading")
    with local.open("rb") as src, transferring.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        fsync_supported = flush_storage(dst)
    if transferring.stat().st_size != size or digest(transferring) != wanted:
        raise IOError("Drive transfer size/SHA256 mismatch")
    inspect_checkpoint(transferring, root, wanted)
    os.replace(transferring, destination)
    if destination.stat().st_size != size or digest(destination) != wanted:
        raise IOError("Drive final copy changed")
    inspect_checkpoint(destination, root, wanted)
    with destination.with_suffix(".zip.sha256").open("x", encoding="ascii") as f:
        f.write(wanted + "  " + name + "\n")
        flush_storage(f)
    receipt = {"path": str(destination), "local_path": str(local), "sha256": wanted, "size_bytes": size,
               "revision": meta["revision"], "status": status, "final": final, "backup_verified": True,
               "fsync_supported": fsync_supported}
    sc.write_json(destination.with_suffix(".zip.receipt.json"), receipt)
    print("VERIFIED DRIVE BACKUP:", destination.name, "valid", status["valid"], "SHA256", wanted, flush=True)
    return receipt


def restore_state(path, root, state_dir, expected_sha):
    manifest, payload = inspect_checkpoint(path, root, expected_sha)
    state_dir = Path(state_dir)
    if (state_dir / "pilot_state.json").exists():
        _, _, local_meta, _, _ = scan_state(root, state_dir)
        if local_meta["revision"] > manifest["revision"]:
            return  # Retain local progress when a prior Drive copy failed.
    for name, data in payload.items():
        if not name.startswith("state/"):
            continue
        target = state_dir / name.removeprefix("state/")
        if target.parent == state_dir / "cache" and target.exists() and target.read_bytes() != data:
            raise ValueError("Immutable valid sample conflicts during restoration")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data:
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(data)
            sc.atomic_replace(temporary, target)
    scan_state(root, state_dir)


def prepare_state(root, state_dir, drive_base, original_path=None):
    rows, sampling = load_inputs(root)
    state_dir, drive_base = Path(state_dir), Path(drive_base)
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in ("runtime", "checkpoints", "final"):
        (drive_base / name).mkdir(parents=True, exist_ok=True)
    candidates, rejected = [], []
    for directory in (drive_base / "checkpoints", drive_base / "final"):
        for p in directory.glob("q27_*.zip"):
            try:
                sidecar = p.with_suffix(".zip.sha256")
                wanted = sidecar.read_text(encoding="ascii").split()[0]
                m, _ = inspect_checkpoint(p, root, wanted)
                candidates.append((m["revision"], m["timestamp"], p, wanted))
            except Exception as exc:
                rejected.append(p.name)
                print("Rejected unverified/incompatible checkpoint:", p.name, type(exc).__name__)
    if candidates:
        revision, _, path, wanted = max(candidates, key=lambda x: (x[0], x[1], str(x[2])))
        restore_state(path, root, state_dir, wanted)
        print("Restored most recent verified Drive checkpoint:", path, "revision", revision)
    elif rejected:
        raise RuntimeError("Drive checkpoints exist but none validates; do not silently restart")
    elif not (state_dir / "pilot_state.json").exists():
        sc.write_json(state_dir / "pilot_state.json", new_meta(sampling["pilot_signature"]))
    settle_completed_intents(root, state_dir)
    _, _, meta, _, _ = scan_state(root, state_dir)
    if meta["reuse_provenance"] is None:
        original = Path(original_path) if original_path else None
        if original and original.is_file():
            reused, audit = inspect_original(original, rows)
            for item in reused:
                expected, _, key = sc.sample_context(projection(item["row"]), item["seed"])
                if item["context"] != expected:
                    raise ValueError("Reusable context changed")
                target = state_dir / "cache" / (key + ".json")
                target.parent.mkdir(parents=True, exist_ok=True)
                data = (sc.canonical(item["entry"]) + "\n").encode("utf-8")
                if target.exists() and target.read_bytes() != data:
                    raise ValueError("Reuse cache conflicts")
                if not target.exists():
                    target.write_bytes(data)
            meta["reuse_provenance"] = audit
        else:
            if sampling["expected_reusable_samples"] != 0:
                raise RuntimeError("Original archive unavailable: validated reusable samples cannot be recovered")
            meta["reuse_provenance"] = {"archive_available_in_cloud": False, "reusable_samples": 0,
                "reason": "Local complete-archive audit found zero question27 identities; missing original does not change workload", "expected_original_sha256": ORIGINAL_SHA}
            print("Original archive unavailable: zero Q27 samples can be reused (also zero in local verified archive).")
        if meta["reuse_provenance"]["reusable_samples"] != sampling["expected_reusable_samples"]:
            raise ValueError("Reuse inventory differs from prepared experiment")
        save_meta(state_dir, meta)
    receipt = export_backup(root, state_dir, state_dir.parent / "exports", drive_base / "checkpoints")
    report_status(root, state_dir)
    return receipt


def report_status(root, state_dir):
    _, _, _, inventory, status = scan_state(root, state_dir)
    print(json.dumps(status, indent=2))
    for item in inventory:
        if item["status"] in ("FAILED", "INTERRUPTED"):
            print("RETRY REQUIRES EXPLICIT AUTHORIZATION:", item["status"], item["row"]["candidate_option_id"], item["row"]["pmid"], item["seed"], item["cache_key"])
    return status


def setup_ollama(root, state_dir):
    """Called by the user in Colab only; frozen version and digest fail closed."""
    if not sys.platform.startswith("linux") or not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Model setup is allowed only in a manually started mounted Colab runtime")
    _, _, meta, _, status = scan_state(root, state_dir)
    if status["complete_samples"]:
        print("All samples already complete: model setup skipped")
        return
    subprocess.run(["nvidia-smi"], check=True)
    missing = [n for n in ("curl", "tar", "zstd") if shutil.which(n) is None]
    if missing:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", *missing], check=True)
    if shutil.which("ollama") is None:
        installer = Path("/content/q27-ollama-install.sh")
        subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh", "-o", str(installer)], check=True)
        subprocess.run(["sh", str(installer)], env=dict(os.environ, OLLAMA_VERSION=sc.OLLAMA_VERSION), check=True)
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_MODELS="/content/q27-ollama-models")
    def api(path, payload=None, timeout=15):
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request("http://127.0.0.1:11434" + path, data=data, headers={"Content-Type": "application/json"}), timeout=timeout) as response:
            return json.load(response)
    try:
        version = api("/api/version")["version"]
    except Exception:
        log = Path("/content/q27-ollama-server.log").open("ab")
        server = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(60):
            try:
                version = api("/api/version")["version"]
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError("Ollama server exited; inspect q27-ollama-server.log")
                time.sleep(1)
        else:
            raise RuntimeError("Ollama startup timeout")
    if version != sc.OLLAMA_VERSION:
        raise RuntimeError("STOP: expected frozen Ollama " + sc.OLLAMA_VERSION + "; no version substitution")
    subprocess.run(["ollama", "pull", sc.MODEL], env=env, check=True)
    identity = sc.verify_ollama_identity("http://127.0.0.1:11434")
    api("/api/generate", {"model": sc.MODEL, "prompt": "", "stream": False, "keep_alive": "5m"}, timeout=600)
    loaded = [m for m in api("/api/ps")["models"] if m.get("name") == sc.MODEL]
    if len(loaded) != 1 or loaded[0].get("size_vram", 0) <= 0:
        raise RuntimeError("STOP: frozen Qwen model not verified in GPU VRAM")
    meta["model_identity"] = {**identity, "verified_live_identity": True, "size_vram": loaded[0]["size_vram"],
                               "backend": "unchanged OllamaTextGenerationBackend", "thinking": True, "temperature": sc.TEMPERATURE}
    save_meta(state_dir, meta)
    print("Frozen live model identity verified:", json.dumps(meta["model_identity"]))


def run_missing(root, state_dir, drive_base, *, retry_keys=(), acknowledge_stale_lock=False, max_new=150, test_backend=None):
    """Skip all valid seeds. Persist intent before, and result after, every unit."""
    if type(max_new) is not int or not 1 <= max_new <= 150:
        raise ValueError("max_new must be 1..150")
    settle_completed_intents(root, state_dir)
    rows, sampling, meta, inventory, initial_status = scan_state(root, state_dir)
    if initial_status["complete_samples"]:
        print("All 150 valid samples already exist; zero new model calls")
        return initial_status
    if test_backend is None:
        if not sys.platform.startswith("linux") or not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Local inference is prohibited; execute this function manually in Colab")
        sc.verify_ollama_identity("http://127.0.0.1:11434")
    state_dir, drive_base = Path(state_dir), Path(drive_base)
    by_key = {i["cache_key"]: i for i in inventory}
    retries = set(retry_keys)
    if len(retries) != len(tuple(retry_keys)) or any(k not in by_key or by_key[k]["status"] not in ("FAILED", "INTERRUPTED") for k in retries):
        raise ValueError("Targeted retry keys must be unique failed/interrupted identities only")
    unresolved = [i for i in inventory if i["status"] == "INTERRUPTED" and i["cache_key"] not in retries]
    if unresolved:
        report_status(root, state_dir)
        raise RuntimeError("Interrupted call completion is unknown: explicit targeted retry authorization required; never automatically duplicate it")
    lock = drive_base / "pilot_run.lock"
    if lock.exists():
        if not acknowledge_stale_lock:
            raise RuntimeError("Another/stale pilot_run.lock exists. Stop other runtimes; then explicitly acknowledge stale lock")
        os.replace(lock, drive_base / "checkpoints" / ("stale_lock_" + uuid.uuid4().hex + ".json"))
    token = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as f:
        f.write(sc.canonical({"token": token, "timestamp": sc.timestamp(), "pilot_signature": sampling["pilot_signature"]}))
    completed = 0
    try:
        backend = test_backend or sc.OllamaTextGenerationBackend(base_url="http://127.0.0.1:11434", model=sc.MODEL, think=True, timeout=600, seed=sc.SEEDS[0])
        if test_backend is None and not (meta.get("model_identity") or {}).get("verified_live_identity"):
            raise RuntimeError("Run frozen model setup/verification before sampling")
        for item in inventory:
            key = item["cache_key"]
            if retries and key not in retries:
                continue  # Explicit retry requests never call unrelated missing units.
            if item["status"] == "VALID" or item["status"] == "FAILED" and key not in retries:
                continue
            if completed >= max_new:
                break
            if key in retries:
                meta["retry_authorizations"].append({"cache_key": key, "prior_status": item["status"], "timestamp": sc.timestamp(),
                    "warning": "An interrupted in-flight call may already have run; retry is explicit, not an exactly-once guarantee"})
            meta["active_intents"][key] = {"ids": item["context"]["ids"], "seed": item["seed"], "timestamp": sc.timestamp()}
            save_meta(state_dir, meta)
            export_backup(root, state_dir, state_dir.parent / "exports", drive_base / "checkpoints")
            # Reuse the original frozen bounded-retry sampler, parser and backend.
            result = sc.sample_one(projection(item["row"]), item["seed"], backend, state_dir / "cache", retry_failed=key in retries)
            meta["active_intents"].pop(key)
            save_meta(state_dir, meta)
            export_backup(root, state_dir, state_dir.parent / "exports", drive_base / "checkpoints")
            completed += 1
            print("Sample unit:", item["row"]["candidate_option_id"], item["row"]["pmid"], item["seed"], result["status"], "actual calls", result["model_calls"], flush=True)
        if retries:
            print("Targeted retries only finished. Clear RETRY_KEYS=[] before continuing other missing seeds.")
        return report_status(root, state_dir)
    except BaseException:
        # Intent remains if the call's result is unknown. Never auto-clear it.
        try:
            export_backup(root, state_dir, state_dir.parent / "exports", drive_base / "checkpoints")
        except Exception as backup_error:
            print("Backup failed; STOP. Local caches retained; prior verified Drive ZIP remains:", type(backup_error).__name__, flush=True)
        raise
    finally:
        if lock.exists() and read_json(lock).get("token") == token:
            lock.unlink()


def finish(root, state_dir, drive_base):
    status = report_status(root, state_dir)
    if not status["complete_samples"]:
        raise RuntimeError("INCOMPLETE: do not mark pilot complete")
    receipt = export_backup(root, state_dir, Path(state_dir).parent / "exports", Path(drive_base) / "final", final=True)
    if not receipt["backup_verified"] or receipt["status"]["valid"] != 150:
        raise RuntimeError("Final Drive backup not verified")
    print("PILOT COMPLETE: 15 pairs x 10 seeds; 150 valid; zero missing/failed/interrupted; final backup verified")
    return receipt


def evaluate_checkpoint(root, checkpoint, expected_sha, gold_source, output_dir):
    """Blind predictions are written before opening the evaluator-only gold source."""
    manifest, payload = inspect_checkpoint(checkpoint, root, expected_sha)
    if not manifest["final"] or not manifest["status"]["complete_samples"]:
        raise ValueError("Evaluator requires a complete final checkpoint")
    rows, sampling = load_inputs(root)
    with tempfile.TemporaryDirectory(prefix="q27-local-evaluation-") as temp:
        restore_state(checkpoint, root, Path(temp), expected_sha)
        _, _, _, inventory, _ = scan_state(root, Path(temp))
        samples = {(sc.pair_key(i["row"]), i["seed"]): i["entry"]["stance"] for i in inventory}
    result = evaluate(rows, samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind = output_dir / "blind_predictions.json"
    if any(output_dir.iterdir()):
        raise FileExistsError("Choose a new evaluation directory; never overwrite research outputs")
    blind.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    blind_hash = digest(blind)
    # Gold is accessed only now; no gold-derived input or rule enters inference.
    with Path(gold_source).open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    matches = [q for q in questions if q["id"] == QID]
    if len(matches) != 1 or matches[0]["question"] != rows[0]["question_stem"] or any(matches[0]["options"][o] != next(r["candidate_option_text"] for r in rows if r["candidate_option_id"] == o) for o in OPTIONS):
        raise ValueError("Evaluator gold source question/option identity differs; blind output preserved")
    gold = matches[0]["answer"]
    if gold not in OPTIONS:
        raise ValueError("Invalid evaluator-only gold label")
    correctness = {c: {m: {"selected_answer": v["answer"]["selected_answer"], "status": v["answer"]["status"],
                    "correct": None if v["answer"]["status"] != "ANSWERED" else v["answer"]["selected_answer"] == gold}
                    for m, v in data["methods"].items()} for c, data in result["conditions"].items()}
    evaluated = {"question_id": QID, "gold_answer": gold, "gold_joined_after_blind_prediction": True,
                 "blind_prediction_sha256": blind_hash, "checkpoint_sha256": expected_sha,
                 "evaluator_only_gold_source_sha256": digest(gold_source), "correctness": correctness,
                 "warning": "Single-question technical result; no medical QA accuracy improvement or superiority inference."}
    (output_dir / "evaluation.json").write_text(json.dumps(evaluated, indent=2) + "\n", encoding="utf-8")
    lines = ["# Question 27 technical pilot", "", "One question only; no general accuracy or reliability conclusion.", "",
             "| Quality | Method | A | B | C | D | E | Selection | Status |", "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for c, data in result["conditions"].items():
        for m, value in data["methods"].items():
            scores = ["undefined" if value["option_scores"][o]["score"] is None else format(value["option_scores"][o]["score"], ".17g") for o in OPTIONS]
            lines.append("| " + c + " | " + m + " | " + " | ".join(scores) + " | " + str(value["answer"]["selected_answer"] or "ABSTAIN") + " | " + value["answer"]["status"] + " |")
    lines += ["", "Rankings, native claim decisions, undefined scores, modal ties, B-C/C-D and quality-control effects: blind_predictions.json.",
              "Evaluator-only gold and single-question correctness: evaluation.json.", "", "Checkpoint SHA256: " + expected_sha, "Blind prediction SHA256: " + blind_hash, ""]
    (output_dir / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Evaluation saved:", output_dir)
    print("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--sha256", required=True)
    ev.add_argument("--gold-source", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        rows, sampling = load_inputs()
        print(json.dumps({"question_id": QID, "pairs": len(rows), "required_seeds": list(sc.SEEDS),
                          "required_samples": 150, "expected_reusable_samples": sampling["expected_reusable_samples"],
                          "missing_sample_calls_before_retries": 150 - sampling["expected_reusable_samples"], "frozen_configuration_validated": True}, indent=2))
    else:
        evaluate_checkpoint(ROOT, args.checkpoint, args.sha256, args.gold_source, args.output_dir)


if __name__ == "__main__":
    main()
