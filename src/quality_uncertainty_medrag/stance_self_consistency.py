"""Separate, resumable self-consistency probe for the frozen Qwen stance v1.

Sampling reads only the existing unlabeled input projection. Evaluation is a
separate command that opens the reference labels only after 600 valid samples
exist. This module never changes the frozen classifier or its prompt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from quality_uncertainty_medrag.llm_stance import (
    CLASSIFIER_VERSION, STANCE_PROMPT, StanceOutputError,
    build_stance_prompt, parse_stance_output,
)
from quality_uncertainty_medrag.ollama_backend import OllamaTextGenerationBackend

SEEDS = tuple(range(101, 111))
TEMPERATURE = 0.7
MODEL = "qwen3:8b"
DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
OLLAMA_VERSION = "0.34.4"
REFERENCE_SHA256 = "ae5d55ef412fa5a33997108fdebd36b13fa017953ca5281a36a3deeb5806065d"
INPUT_SHA256 = "3c8260a38c4df1640ab9d654b05e2ff917b744ac613a559fa66446fc856d67b0"
SAMPLE_VERSION = "qwen-v1-self-consistency-10-v1"
LABELS = ("SUPPORT", "CONTRADICT", "IRRELEVANT")
INPUT_FIELDS = ("question_id", "candidate_option_id", "evidence_doc_id", "question_stem",
                "candidate_option_text", "evidence_title", "evidence_abstract")
PROMPT_FIELDS = ("question_stem", "candidate_option_text", "evidence_title", "evidence_abstract")
REFERENCE_REL = Path("outputs/stance_annotation/dev_1_30/stance_reference_60_adjudicated.csv")
INPUT_REL = Path("outputs/stance_model_comparison/reference_60_adjudicated/unlabeled_inputs_60.jsonl")
HARD_REL = Path("outputs/stance_model_comparison/reference_60_adjudicated/predictions_60.csv")
OUTPUT_REL = Path("outputs/stance_uncertainty/self_consistency_10/reference_60")
CACHE_REL = OUTPUT_REL / "cache"
CHECKPOINT_NAME = "stance_self_consistency_10_checkpoint.zip"
CHECKPOINT_INTERVAL = 25


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_text(value):
    return sha_bytes(value.encode("utf-8"))


def atomic_replace(temporary, path):
    # Windows indexers/scanners can briefly hold the previous archive open.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def pair_key(row):
    return (row["question_id"], row["candidate_option_id"], row["evidence_doc_id"])


def read_unlabeled(path):
    if sha_bytes(path.read_bytes()) != INPUT_SHA256:
        raise ValueError("Frozen unlabeled input projection SHA256 differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 60 or len({pair_key(row) for row in rows}) != 60:
        raise ValueError("Expected 60 unique unlabeled pairs")
    for row in rows:
        if set(row) != set(INPUT_FIELDS):
            raise ValueError("Unlabeled input fields differ from allowlist")
        if any(not isinstance(row[k], str) for k in INPUT_FIELDS):
            raise ValueError("Unlabeled input fields must be strings")
        if any(not row[k].strip() for k in INPUT_FIELDS if k != "evidence_abstract"):
            raise ValueError("Required unlabeled text is empty")
    return rows


def sample_context(row, seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Seed is outside the frozen list")
    if set(row) != set(INPUT_FIELDS):
        raise ValueError("Only allowed unlabeled input fields may be passed")
    prompt = build_stance_prompt(**{k: row[k] for k in PROMPT_FIELDS})
    context = {
        "sample_version": SAMPLE_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "prompt_template_sha256": sha_text(STANCE_PROMPT),
        "prompt_sha256": sha_text(prompt),
        "ids": {k: row[k] for k in INPUT_FIELDS[:3]},
        "input_sha256": {k: sha_text(row[k]) for k in PROMPT_FIELDS},
        "model": MODEL, "model_digest": DIGEST, "ollama_version": OLLAMA_VERSION,
        "thinking": True, "temperature": TEMPERATURE, "seed": seed,
    }
    return context, prompt, sha_text(canonical(context))


def read_cached_sample(path, expected_context):
    entry = json.loads(path.read_text(encoding="utf-8"))
    if set(entry) != {"context", "stance", "generation_attempts", "attempts"}:
        raise ValueError("Stochastic cache schema differs")
    if entry["context"] != expected_context or entry["stance"] not in LABELS:
        raise ValueError("Stochastic cache identity or label differs")
    count, attempts = entry["generation_attempts"], entry["attempts"]
    if type(count) is not int or not 1 <= count <= 3 or not isinstance(attempts, list) or len(attempts) != count:
        raise ValueError("Stochastic cache attempt audit invalid")
    if any(item.get("index") != i or item.get("status") != ("success" if i == count else "failed")
           for i, item in enumerate(attempts, 1)):
        raise ValueError("Stochastic cache attempt sequence invalid")
    return entry


def read_attempt_audit(path, context):
    audit = json.loads(path.read_text(encoding="utf-8"))
    attempts = audit.get("attempts")
    cycle = audit.get("cycle", 1)  # Existing pre-checkpoint audits are cycle 1.
    if (audit.get("context") != context or type(cycle) is not int or cycle < 1
            or audit.get("status") not in ("INCOMPLETE", "FAILED", "COMPLETE")
            or not isinstance(attempts, list) or len(attempts) > 3):
        raise ValueError("Invalid stochastic attempt audit")
    complete = audit["status"] == "COMPLETE"
    if (complete and not attempts) or (audit["status"] == "FAILED" and len(attempts) != 3):
        raise ValueError("Invalid stochastic attempt count")
    for i, item in enumerate(attempts, 1):
        expected = "success" if complete and i == len(attempts) else "failed"
        if item.get("index") != i or item.get("status") != expected:
            raise ValueError("Invalid stochastic attempt sequence")
    return {**audit, "cycle": cycle}


def failure_records(cache_dir, cache_key, context):
    records = []
    for path in sorted((cache_dir / "failures" / cache_key).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("context") != context or record.get("cache_key") != cache_key
                or record.get("status") != "FAILED" or record.get("attempt_count") != 3
                or record.get("pair_identity") != context["ids"]
                or record.get("seed") != context["seed"]
                or record.get("temperature") != context["temperature"]
                or type(record.get("cycle")) is not int or record["cycle"] < 1
                or path.name != f"{record['cycle']:04d}.json"
                or not isinstance(record.get("timestamp"), str)
                or "stance" in record):
            raise ValueError("Invalid failed-sample audit")
        records.append(record)
    return records


def record_failure(cache_dir, cache_key, context, cycle, attempts):
    path = cache_dir / "failures" / cache_key / f"{cycle:04d}.json"
    if path.exists():
        failure_records(cache_dir, cache_key, context)
        return False
    last = attempts[-1]
    record = {"cache_key": cache_key, "context": context, "pair_identity": context["ids"],
              "seed": context["seed"], "temperature": context["temperature"],
              "cycle": cycle, "attempt_count": len(attempts), "status": "FAILED",
              "error_class": re.sub(r"[^A-Za-z0-9_]", "", last.get("error_type", "UnknownError"))[:80],
              "error_stage": last.get("stage") if last.get("stage") in ("generation", "parsing")
                             else "legacy_unknown", "timestamp": timestamp()}
    write_json(path, record)
    return True


def sample_one(row, seed, backend, cache_dir, *, retry_failed=False):
    """Use the unchanged v1 prompt/parser; return its unique argmax as one sample.

    A valid but tied v1 distribution has no hard stance and is retried with
    identical settings. Raw response and model thinking are never stored.
    """
    cache_dir = Path(cache_dir)
    context, prompt, cache_key = sample_context(row, seed)
    cache_path = cache_dir / (cache_key + ".json")
    if cache_path.exists():
        entry = read_cached_sample(cache_path, context)
        return {"status": "VALID", "cache_key": cache_key, "cache_status": "HIT", "model_calls": 0,
                "retries_this_run": 0, **entry}
    audit_path = cache_dir / "attempts" / (cache_key + ".json")
    attempts = []
    history = failure_records(cache_dir, cache_key, context)
    cycle = max((r["cycle"] for r in history), default=1)
    recorded = False
    prior = None
    if audit_path.exists():
        prior = read_attempt_audit(audit_path, context)
        if prior["status"] == "COMPLETE":
            raise ValueError("Existing attempt audit is inconsistent with missing cache")
        attempts, cycle = prior["attempts"], prior["cycle"]
    exhausted = len(attempts) == 3 or (prior is None and bool(history))
    if len(attempts) == 3:
        recorded = record_failure(cache_dir, cache_key, context, cycle, attempts)
    if exhausted:
        if not retry_failed:
            return {"status": "FAILED", "cache_status": "FAILED_SKIPPED", "cache_key": cache_key,
                    "context": context, "model_calls": 0, "retries_this_run": 0,
                    "failure_recorded": recorded}
        if prior is not None:
            # Preserve old audits byte-for-byte before beginning an explicitly requested cycle.
            archived = cache_dir / "attempts" / "history" / cache_key / f"{cycle:04d}.json"
            archived.parent.mkdir(parents=True, exist_ok=True)
            data = audit_path.read_bytes()
            if archived.exists() and archived.read_bytes() != data:
                raise ValueError("Attempt history conflict")
            if not archived.exists():
                with archived.open("xb") as handle:
                    handle.write(data)
        cycle = max([cycle] + [r["cycle"] for r in history]) + 1
        attempts = []
        write_json(audit_path, {"context": context, "cycle": cycle,
                               "attempts": attempts, "status": "INCOMPLETE"})
    prior_count = len(attempts)
    for index in range(prior_count + 1, 4):
        stage = "generation"
        try:
            raw = backend.generate(prompt, generation_config={"temperature": TEMPERATURE, "seed": seed})
            stage = "parsing"
            prediction = parse_stance_output(raw)
            if prediction.label is None:
                raise StanceOutputError("Tied distribution has no final hard stance")
        except Exception as exc:
            attempts.append({"index": index, "status": "failed", "stage": stage,
                             "timestamp": timestamp(),
                             "error_type": re.sub(r"[^A-Za-z0-9_]", "", type(exc).__name__)[:80]})
            write_json(audit_path, {"context": context, "cycle": cycle, "attempts": attempts,
                                   "status": "FAILED" if index == 3 else "INCOMPLETE"})
            if index == 3:
                record_failure(cache_dir, cache_key, context, cycle, attempts)
                return {"status": "FAILED", "cache_status": "MISS", "cache_key": cache_key,
                        "context": context, "model_calls": index - prior_count,
                        "retries_this_run": index - max(prior_count, 1), "failure_recorded": True}
        else:
            attempts.append({"index": index, "status": "success"})
            entry = {"context": context, "stance": prediction.label.value,
                     "generation_attempts": index, "attempts": attempts}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("x", encoding="utf-8") as handle:
                handle.write(canonical(entry) + "\n")
            write_json(audit_path, {"context": context, "cycle": cycle, "attempts": attempts, "status": "COMPLETE"})
            return {"status": "VALID", "cache_key": cache_key, "cache_status": "MISS", "model_calls": index - prior_count,
                    "retries_this_run": index - max(prior_count, 1), **entry}
    raise AssertionError("Unreachable")


def verify_ollama_identity(base_url):
    with urlopen(base_url.rstrip("/") + "/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    with urlopen(base_url.rstrip("/") + "/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    matches = [item for item in models if item.get("name") == MODEL]
    digest = matches[0].get("digest") if len(matches) == 1 else None
    if version != OLLAMA_VERSION or digest != DIGEST:
        raise RuntimeError("Ollama version/model digest differs from frozen identity")
    return {"model": MODEL, "digest": digest, "ollama_version": version}


def frozen_settings():
    return {"sample_version": SAMPLE_VERSION, "classifier_version": CLASSIFIER_VERSION,
            "model": MODEL, "digest": DIGEST, "ollama_version": OLLAMA_VERSION,
            "thinking": True, "temperature": TEMPERATURE, "seeds": list(SEEDS),
            "sample_count_per_pair": 10, "prompt_template_sha256": sha_text(STANCE_PROMPT),
            "unlabeled_input_sha256": INPUT_SHA256, "reference_sha256": REFERENCE_SHA256,
            "frozen_hard_settings": {"thinking": True, "temperature": 0, "seed": 42}}


def validate_run_manifest(root):
    path = Path(root) / OUTPUT_REL / "run_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key, value in frozen_settings().items():
            if key in saved and saved[key] != value:
                raise ValueError(f"Run manifest setting differs: {key}")
        return saved
    return {}


def scan_samples(root):
    """Read and validate the one existing cache; never contact a backend or write files."""
    root = Path(root)
    rows = read_unlabeled(root / INPUT_REL)
    validate_run_manifest(root)
    cache_dir = root / CACHE_REL
    inventory = []
    expected_keys = set()
    for row in rows:
        for seed in SEEDS:
            context, _, key = sample_context(row, seed)
            expected_keys.add(key)
            cache = cache_dir / f"{key}.json"
            entry = read_cached_sample(cache, context) if cache.exists() else None
            failures = failure_records(cache_dir, key, context)
            cycles = {}
            for path in sorted((cache_dir / "attempts" / "history" / key).glob("*.json")):
                audit = read_attempt_audit(path, context)
                if path.name != f"{audit['cycle']:04d}.json" or audit["cycle"] in cycles:
                    raise ValueError("Duplicate or invalid attempt history cycle")
                cycles[audit["cycle"]] = audit
            current_path = cache_dir / "attempts" / f"{key}.json"
            current = read_attempt_audit(current_path, context) if current_path.exists() else None
            if current is not None:
                cycle = current["cycle"]
                if cycle in cycles and cycles[cycle] != current:
                    raise ValueError("Conflicting current and historical attempts")
                cycles[cycle] = current
            if entry is None and current is not None and current["status"] == "COMPLETE":
                raise ValueError("Complete attempt audit has no valid cache entry")
            exhausted = current is not None and len(current["attempts"]) == 3
            state = "VALID" if entry else "FAILED" if exhausted or (current is None and failures) else "MISSING"
            lengths = {cycle: len(a["attempts"]) for cycle, a in cycles.items()}
            for failure in failures:
                lengths[failure["cycle"]] = max(lengths.get(failure["cycle"], 0), 3)
            if entry:
                cycle = current["cycle"] if current else max(lengths, default=0) + 1
                if current and current["status"] == "COMPLETE" and current["attempts"] != entry["attempts"]:
                    raise ValueError("Cache and completed attempt audit disagree")
                if current and current["status"] != "COMPLETE" and current["attempts"] != entry["attempts"][:len(current["attempts"])]:
                    raise ValueError("Cache and interrupted attempt audit disagree")
                lengths[cycle] = entry["generation_attempts"]
            inventory.append({"row": row, "seed": seed, "context": context, "cache_key": key,
                              "status": state, "entry": entry,
                              "total_model_calls": sum(lengths.values()),
                              "total_retries": sum(max(0, count - 1) for count in lengths.values())})
    if len(inventory) != 600 or len(expected_keys) != 600:
        raise ValueError("Expected 600 unique stochastic cache identities")
    for directory in (cache_dir, cache_dir / "attempts"):
        if any(path.stem not in expected_keys for path in directory.glob("*.json")):
            raise ValueError("Unexpected stochastic cache or attempt identity")
    for directory in (cache_dir / "failures", cache_dir / "attempts" / "history"):
        if any(path.name not in expected_keys or not path.is_dir() for path in directory.glob("*")):
            raise ValueError("Unexpected stochastic audit history identity")
    partial = root / OUTPUT_REL / "samples.jsonl"
    if partial.exists():
        valid = {item["cache_key"]: item for item in inventory if item["status"] == "VALID"}
        seen = set()
        for line in partial.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            key = sample.get("cache_key")
            item = valid.get(key)
            if (item is None or key in seen or sample.get("stance") != item["entry"]["stance"]
                    or sample.get("seed") != item["seed"] or sample.get("ids") != item["context"]["ids"]):
                raise ValueError("Partial samples disagree with exact valid cache")
            seen.add(key)
    return inventory


def summarize_status(inventory):
    counts = Counter(item["status"] for item in inventory)
    return {"expected": 600, "valid": counts["VALID"], "failed": counts["FAILED"],
            "missing": counts["MISSING"], "completion_percentage": 100 * counts["VALID"] / 600}


def validate_restored_checkpoint(root):
    """Also serves as read-only status; incomplete checkpoints are valid resume states."""
    return summarize_status(scan_samples(root))


def ensure_run_manifest(root):
    saved = validate_run_manifest(root)
    write_json(Path(root) / OUTPUT_REL / "run_manifest.json",
               {**saved, **frozen_settings(), "cache_directory": CACHE_REL.as_posix(),
                "checkpoint_interval_new_valid": CHECKPOINT_INTERVAL})


def export_checkpoint(root, archive_path=None):
    """Atomically export caches, all audits, partial outputs and unlabeled resume inputs."""
    root = Path(root)
    state = validate_restored_checkpoint(root)
    ensure_run_manifest(root)
    out = root / OUTPUT_REL
    write_json(out / "status.json", state)
    archive_path = Path(archive_path) if archive_path else out / CHECKPOINT_NAME
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {INPUT_REL.as_posix(): (root / INPUT_REL).read_bytes()}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix not in (".tmp", ".zip"):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Checkpoint source escaped project root")
            payload[path.relative_to(root).as_posix()] = path.read_bytes()
    manifest = {"checkpoint_version": 1, "timestamp": timestamp(), "settings": frozen_settings(),
                "cache_directory": CACHE_REL.as_posix(), "status": state,
                "files": {name: sha_bytes(data) for name, data in payload.items()}}
    temporary = archive_path.with_name(archive_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in payload.items():
                archive.writestr(name, data)
            archive.writestr("checkpoint_manifest.json", canonical(manifest) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Checkpoint ZIP integrity failed")
            recorded = json.loads(archive.read("checkpoint_manifest.json"))
            if recorded != manifest or set(archive.namelist()) != set(manifest["files"]) | {"checkpoint_manifest.json"}:
                raise ValueError("Checkpoint manifest differs from exported inventory")
            if any(sha_bytes(archive.read(name)) != digest for name, digest in recorded["files"].items()):
                raise ValueError("Checkpoint manifest SHA256 validation failed")
        atomic_replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return archive_path


def restore_checkpoint(archive_path, root):
    """Validate before extraction, preserve existing identical files and reject conflicts."""
    root = Path(root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "checkpoint_manifest.json" not in names:
            raise ValueError("Checkpoint has duplicate entries or no manifest")
        manifest = json.loads(archive.read("checkpoint_manifest.json"))
        if (manifest.get("checkpoint_version") != 1 or manifest.get("settings") != frozen_settings()
                or manifest.get("cache_directory") != CACHE_REL.as_posix()):
            raise ValueError("Checkpoint settings differ")
        files = manifest["files"]
        if set(names) != set(files) | {"checkpoint_manifest.json"}:
            raise ValueError("Checkpoint file inventory differs")
        payload = {}
        for name, digest in files.items():
            relative = Path(name)
            if ("\\" in name or relative.is_absolute() or ".." in relative.parts
                    or relative.as_posix() != name
                    or not (relative == INPUT_REL or OUTPUT_REL in relative.parents)
                    or relative.suffix in (".zip", ".tmp")):
                raise ValueError("Unsafe checkpoint path")
            data = archive.read(name)
            if sha_bytes(data) != digest:
                raise ValueError("Checkpoint SHA256 mismatch")
            payload[relative] = data
    with tempfile.TemporaryDirectory(prefix="sc-") as temp:
        staged = Path(temp)
        for relative, data in payload.items():
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        state = validate_restored_checkpoint(staged)
        if state != manifest["status"]:
            raise ValueError("Checkpoint status disagrees with cache")
        # Preflight every destination before copying anything.
        for relative, data in payload.items():
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError("Restore target escaped project root")
            if target.exists() and (not target.is_file() or target.read_bytes() != data):
                raise ValueError(f"Refusing to overwrite different existing file: {relative}")
        for relative in payload:
            target = root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged / relative, target)
    return validate_restored_checkpoint(root)


def write_partial_samples(out, inventory):
    temporary = out / ("samples.jsonl." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for item in inventory:
                if item["status"] == "VALID":
                    handle.write(canonical({"ids": item["context"]["ids"], "seed": item["seed"],
                                            "stance": item["entry"]["stance"],
                                            "cache_key": item["cache_key"]}) + "\n")
        atomic_replace(temporary, out / "samples.jsonl")
    finally:
        temporary.unlink(missing_ok=True)


def collect(root, *, backend=None, base_url="http://localhost:11434", verify_identity=True,
            retry_failed=False, max_new_samples=None):
    if max_new_samples is not None and (type(max_new_samples) is not int or max_new_samples < 1):
        raise ValueError("max_new_samples must be a positive integer")
    root = Path(root)
    inventory = scan_samples(root)  # No reference labels or hard predictions are read.
    out, cache_dir = root / OUTPUT_REL, root / CACHE_REL
    ensure_run_manifest(root)
    identity = {"model": MODEL, "digest": DIGEST, "ollama_version": OLLAMA_VERSION,
                "verification": "cache only; no backend contacted"}
    backend_ready = False
    counts = Counter()
    write_json(out / "leakage_audit.json", {"allowed_input_fields": list(INPUT_FIELDS),
        "prompt_fields": list(PROMPT_FIELDS), "reference_opened_during_sampling": False,
        "hard_prediction_file_opened_during_sampling": False,
        "model_payload_contains_reference": False, "medqa_gold_or_other_options_passed": False,
        "dev_31_50_accessed": False})

    def save_progress():
        state = summarize_status(inventory)
        write_json(out / "progress.json", {**state, "completed_samples": state["valid"],
                                           "updated_at": timestamp(),
                                           "max_new_samples": max_new_samples, **counts})

    def checkpoint():
        write_partial_samples(out, inventory)
        export_checkpoint(root)

    save_progress()
    try:
        for item in inventory:
            eligible = item["status"] == "MISSING" or (retry_failed and item["status"] == "FAILED")
            if eligible and max_new_samples is not None and counts["new_units"] >= max_new_samples:
                break
            if eligible and not backend_ready:
                identity = verify_ollama_identity(base_url) if verify_identity else {
                    "model": MODEL, "digest": DIGEST, "ollama_version": OLLAMA_VERSION,
                    "verification": "injected backend for tests"}
                backend = backend or OllamaTextGenerationBackend(base_url=base_url, model=MODEL, think=True,
                                                                timeout=600, seed=SEEDS[0])
                backend_ready = True
            result = sample_one(item["row"], item["seed"], backend, cache_dir, retry_failed=retry_failed)
            item["status"] = result["status"]
            if result["status"] == "VALID":
                item["entry"] = {key: result[key] for key in ("context", "stance", "generation_attempts", "attempts")}
            counts["cache_hits" if result["cache_status"] == "HIT" else "failed_skipped"
                   if result["cache_status"] == "FAILED_SKIPPED" else "cache_misses"] += 1
            counts["model_calls"] += result["model_calls"]
            counts["retries"] += result["retries_this_run"]
            new_unit = eligible and result["cache_status"] == "MISS"
            new_valid = new_unit and result["status"] == "VALID"
            counts["new_units"] += int(new_unit)
            counts["new_valid"] += int(new_valid)
            counts["new_failed"] += int(new_unit and result["status"] == "FAILED")
            save_progress()
            if result.get("failure_recorded") or (new_valid and counts["new_valid"] % CHECKPOINT_INTERVAL == 0):
                checkpoint()
    except BaseException as exc:
        # A fault outside the bounded sample retry path must still preserve all
        # valid caches written before it. The prior atomic ZIP remains if this
        # refresh cannot be completed.
        try:
            observed = scan_samples(root)
            write_partial_samples(out, observed)
            write_json(out / "unexpected_error_audit.json", {
                "error_class": type(exc).__name__, "timestamp": timestamp(),
                "new_units_this_invocation": counts["new_units"],
                "max_new_samples": max_new_samples,
            })
            export_checkpoint(root)
        except Exception as checkpoint_error:
            print(f"Checkpoint refresh after unexpected error failed: {type(checkpoint_error).__name__}")
        print(f"Collection interrupted by {type(exc).__name__}; any valid cache entries remain on disk")
        raise
    write_partial_samples(out, inventory)
    checked = scan_samples(root)
    state = summarize_status(checked)
    total_calls = sum(item["total_model_calls"] for item in checked)
    audit = {**state, "expected_samples": 600, "completed_samples": state["valid"],
             "valid_cached_samples": state["valid"], "failed_samples": state["failed"],
             "still_missing_samples": state["missing"], "cache_hits": counts["cache_hits"],
             "cache_misses": counts["cache_misses"], "failed_skipped": counts["failed_skipped"],
             "new_model_calls_this_invocation": counts["model_calls"], "actual_model_calls": total_calls,
             "new_units_this_invocation": counts["new_units"],
             "new_valid_this_invocation": counts["new_valid"],
             "new_failed_this_invocation": counts["new_failed"],
             "max_new_samples": max_new_samples,
             "retries": sum(item["total_retries"] for item in checked),
             "new_retries_this_invocation": counts["retries"], "failures": state["failed"],
             "identity": identity, "sample_version": SAMPLE_VERSION, "prompt_version": CLASSIFIER_VERSION,
             "prompt_sha256": sha_text(STANCE_PROMPT), "temperature": TEMPERATURE,
             "seeds": list(SEEDS), "thinking": True, "retry_failed_requested": retry_failed}
    write_json(out / "cache_audit.json", audit)
    export_checkpoint(root)
    return audit


def frequencies(labels):
    if len(labels) != 10 or any(label not in LABELS for label in labels):
        raise ValueError("Expected exactly ten valid stance labels")
    counts = Counter(labels)
    return {label: counts[label] / 10 for label in LABELS}


def entropy(values, *, classes):
    values = tuple(values)
    if len(values) != classes or any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise ValueError("Invalid distribution")
    if not math.isclose(math.fsum(values), 1, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Distribution must sum to one")
    return -math.fsum(x * math.log(x) for x in values if x > 0) / math.log(classes)


def uncertainty(labels):
    p = frequencies(labels)
    directional_mass = p["SUPPORT"] + p["CONTRADICT"]
    if directional_mass:
        ps, pc = p["SUPPORT"] / directional_mass, p["CONTRADICT"] / directional_mass
        u_dir = entropy((ps, pc), classes=2)
        directional_score = ps - pc
    else:
        # Convention: no directional samples means maximally unresolved direction.
        u_dir, directional_score = 1.0, 0.0
    return {"n_support": round(p["SUPPORT"] * 10), "n_contradict": round(p["CONTRADICT"] * 10),
            "n_irrelevant": round(p["IRRELEVANT"] * 10),
            "p_support": p["SUPPORT"], "p_contradict": p["CONTRADICT"],
            "p_irrelevant": p["IRRELEVANT"], "u_3": entropy(tuple(p.values()), classes=3),
            "relevance": 1 - p["IRRELEVANT"], "u_directional": u_dir,
            "directional_score": directional_score}


def classification(rows):
    """Fixed-class macro-F1; absent class precision/recall/F1 is defined as 0."""
    if not rows:
        return {"accuracy": None, "macro_f1": None}
    correct = sum(row["hard_prediction_correct"] for row in rows)
    scores = []
    for label in LABELS:
        tp = sum(r["reference_stance"] == label and r["frozen_qwen_hard_stance"] == label for r in rows)
        fp = sum(r["reference_stance"] != label and r["frozen_qwen_hard_stance"] == label for r in rows)
        fn = sum(r["reference_stance"] == label and r["frozen_qwen_hard_stance"] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": correct / len(rows), "macro_f1": statistics.mean(scores)}


def auroc(scores, errors):
    positive = [x for x, y in zip(scores, errors) if y]
    negative = [x for x, y in zip(scores, errors) if not y]
    if not positive or not negative:
        return None
    wins = math.fsum((x > y) + 0.5 * (x == y) for x in positive for y in negative)
    return wins / (len(positive) * len(negative))


def auprc(scores, errors):
    """Stepwise PR area (average precision), with all score ties grouped."""
    total_positive = sum(errors)
    if not total_positive:
        return None
    groups = {}
    for score, error in zip(scores, errors):
        groups.setdefault(score, []).append(error)
    tp = fp = 0
    area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        new_tp = sum(group)
        tp += new_tp
        fp += len(group) - new_tp
        area += (new_tp / total_positive) * (tp / (tp + fp))
    return area


def selective(rows, field, coverage):
    if len(rows) != 60 or coverage not in (1.0, 0.9, 0.8, 0.7):
        raise ValueError("Expected 60 rows and a prespecified coverage")
    retain = round(len(rows) * coverage)
    # Highest uncertainty is removed. Exact ties break by stable pair_id.
    kept = sorted(rows, key=lambda r: (r[field], r["pair_id"]))[:retain]
    return {"uncertainty": field, "coverage": coverage, "retained_pair_count": retain,
            **classification(kept)}


def evaluate(root):
    root = Path(root)
    out = root / OUTPUT_REL
    sample_path = out / "samples.jsonl"
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) != 600:
        raise ValueError("Evaluation requires all 600 samples")
    groups = {}
    for sample in samples:
        key = tuple(sample["ids"][field] for field in INPUT_FIELDS[:3])
        if sample["seed"] not in SEEDS or sample["stance"] not in LABELS:
            raise ValueError("Invalid stochastic sample")
        groups.setdefault(key, {})
        if sample["seed"] in groups[key]:
            raise ValueError("Duplicate pair/seed sample")
        groups[key][sample["seed"]] = sample["stance"]
    if len(groups) != 60 or any(set(items) != set(SEEDS) for items in groups.values()):
        raise ValueError("Missing stochastic samples")
    unlabeled = read_unlabeled(root / INPUT_REL)
    if set(groups) != {pair_key(row) for row in unlabeled}:
        raise ValueError("Stochastic samples differ from frozen unlabeled pairs")
    by_key = {pair_key(row): row for row in unlabeled}
    for sample in samples:
        key = tuple(sample["ids"][field] for field in INPUT_FIELDS[:3])
        context, _, cache_key = sample_context(by_key[key], sample["seed"])
        if sample["cache_key"] != cache_key:
            raise ValueError("Stochastic sample cache key differs")
        entry = read_cached_sample(out / "cache" / (cache_key + ".json"), context)
        if entry["stance"] != sample["stance"]:
            raise ValueError("Stochastic sample label differs from exact cache")
    # Only after every unlabeled sample/cache identity passes do labels open.
    reference_path = root / REFERENCE_REL
    if sha_bytes(reference_path.read_bytes()) != REFERENCE_SHA256:
        raise ValueError("Frozen reference SHA256 differs")
    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    with (root / HARD_REL).open(encoding="utf-8", newline="") as handle:
        hard = list(csv.DictReader(handle))
    if len(reference) != 60 or len(hard) != 60:
        raise ValueError("Reference or hard-prediction count differs")
    hard_by_id = {r["pair_id"]: r for r in hard}
    if len(hard_by_id) != 60:
        raise ValueError("Duplicate frozen hard prediction IDs")
    rows = []
    for ref in reference:
        key = (ref["question_id"], ref["candidate_option_id"], ref["pmid"])
        saved = hard_by_id[ref["pair_id"]]
        if key not in groups or (saved["question_id"], saved["candidate_option_id"], saved["pmid"]) != key:
            raise ValueError("Reference, hard prediction, and samples do not align")
        if saved["reference_stance"] != ref["final_stance"] or saved["qwen_v1_prediction"] not in LABELS:
            raise ValueError("Frozen hard prediction or reference differs")
        stats = uncertainty([groups[key][seed] for seed in SEEDS])
        rows.append({"pair_id": ref["pair_id"], "batch": ref["batch"], "question_id": key[0],
                     "candidate_option_id": key[1], "pmid": key[2],
                     "reference_stance": ref["final_stance"],
                     "frozen_qwen_hard_stance": saved["qwen_v1_prediction"],
                     "hard_prediction_correct": saved["qwen_v1_prediction"] == ref["final_stance"],
                     **stats})
    if len({r["pair_id"] for r in rows}) != 60:
        raise ValueError("Duplicate reference IDs")
    fields = list(rows[0])
    with (out / "uncertainty_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    errors = [not r["hard_prediction_correct"] for r in rows]
    def describe(items, field):
        values = [r[field] for r in items]
        return {"mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None}
    correct = [r for r in rows if r["hard_prediction_correct"]]
    incorrect = [r for r in rows if not r["hard_prediction_correct"]]
    selective_rows = [selective(rows, field, coverage) for field in ("u_3", "u_directional")
                      for coverage in (1.0, 0.9, 0.8, 0.7)]
    with (out / "selective_prediction.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selective_rows[0]))
        writer.writeheader()
        writer.writerows(selective_rows)
    metrics = {"n": 60, "hard_correct": len(correct), "hard_incorrect": len(incorrect),
        "uncertainty_by_hard_correctness": {
            group: {field: describe(items, field) for field in ("u_3", "u_directional", "relevance")}
            for group, items in (("correct", correct), ("incorrect", incorrect))},
        "error_detection": {field: {"auroc": auroc([r[field] for r in rows], errors),
                                     "auprc_average_precision": auprc([r[field] for r in rows], errors)}
                            for field in ("u_3", "u_directional")},
        "selective_prediction": selective_rows,
        "by_reference_class": {label: {"count": sum(r["reference_stance"] == label for r in rows),
            "hard_accuracy": classification([r for r in rows if r["reference_stance"] == label])["accuracy"],
            "mean_u_3": describe([r for r in rows if r["reference_stance"] == label], "u_3")["mean"],
            "mean_u_directional": describe([r for r in rows if r["reference_stance"] == label], "u_directional")["mean"],
            "mean_relevance": describe([r for r in rows if r["reference_stance"] == label], "relevance")["mean"]}
            for label in LABELS},
        "metric_definitions": {"auprc": "stepwise precision-recall area (average precision), ties grouped",
            "selective": "Retain lowest uncertainty; equal scores tie-break by pair_id; macro-F1 averages all 3 fixed classes with absent-class F1=0",
            "zero_directional_mass": "u_directional=1.0 and directional_score=0.0",
            "scores": "Empirical self-consistency frequencies, not calibrated probabilities"}}
    write_json(out / "uncertainty_metrics.json", metrics)
    audit = json.loads((out / "cache_audit.json").read_text(encoding="utf-8"))
    manifest = {"sample_version": SAMPLE_VERSION, "classifier_version": CLASSIFIER_VERSION,
        "model": MODEL, "digest": DIGEST, "ollama_version": OLLAMA_VERSION, "thinking": True,
        "temperature": TEMPERATURE, "seeds": list(SEEDS), "sample_count_per_pair": 10,
        "frozen_hard_settings": {"thinking": True, "temperature": 0, "seed": 42},
        "prompt_template_sha256": sha_text(STANCE_PROMPT),
        "reference_sha256": REFERENCE_SHA256, "unlabeled_input_sha256": INPUT_SHA256,
        "hard_prediction_sha256": sha_bytes((root / HARD_REL).read_bytes()),
        "cache_hits": audit["cache_hits"], "actual_model_calls": audit["actual_model_calls"],
        "retries": audit["retries"], "failures": audit["failures"],
        "no_production_code_or_aggregation_changes": True, "dev_31_50_accessed": False}
    write_json(out / "run_manifest.json", manifest)
    lines = ["# Frozen Qwen-v1 stance self-consistency (development 60)", "",
        "The temperature-0 Qwen-v1 argmax remains the official hard stance. Ten separate temperature-0.7 samples use the unchanged v1 prompt with fixed seeds 101–110. Their relative counts are empirical self-consistency frequencies, not calibrated probabilities. No aggregation uses these values.", "",
        "When all ten samples are IRRELEVANT, directional mass is zero; by convention u_directional=1.0 and directional_score=0.0.", "",
        f"Samples: 600; cache hits this run: {audit['cache_hits']}; actual model calls: {audit['actual_model_calls']}; retries: {audit['retries']}; failures: {audit['failures']}.",
        f"Frozen hard predictions: {len(correct)} correct, {len(incorrect)} incorrect on this development reference.", "",
        "## Correct versus incorrect hard predictions", "",
        "| Group | Mean u3 | Median u3 | Mean directional u | Median directional u | Mean relevance |",
        "|---|---:|---:|---:|---:|---:|"]
    for group in ("correct", "incorrect"):
        item = metrics["uncertainty_by_hard_correctness"][group]
        lines.append(f"| {group} | {item['u_3']['mean']:.4f} | {item['u_3']['median']:.4f} | {item['u_directional']['mean']:.4f} | {item['u_directional']['median']:.4f} | {item['relevance']['mean']:.4f} |")
    lines += ["", "## Error detection", "", "| Score | AUROC | AUPRC (average precision) |", "|---|---:|---:|"]
    for field, item in metrics["error_detection"].items():
        lines.append(f"| {field} | {item['auroc']:.4f} | {item['auprc_average_precision']:.4f} |")
    lines += ["", "## Selective prediction", "", "| Score | Coverage | Retained | Accuracy | Macro-F1 |", "|---|---:|---:|---:|---:|"]
    for item in selective_rows:
        lines.append(f"| {item['uncertainty']} | {item['coverage']:.0%} | {item['retained_pair_count']} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} |")
    lines += ["", "## Reference class", "", "| Class | N | Frozen hard accuracy | Mean u3 | Mean directional u | Mean relevance |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, item in metrics["by_reference_class"].items():
        lines.append(f"| {label} | {item['count']} | {item['hard_accuracy']:.4f} | {item['mean_u_3']:.4f} | {item['mean_u_directional']:.4f} | {item['mean_relevance']:.4f} |")
    lines += ["", "The CONTRADICT class contains only seven cases. These descriptive development-set results do not establish clinical performance or statistical significance.",
              "Reference labels were opened only after all 600 unlabeled stochastic samples existed. No MedQA gold answer, answer_idx, other candidate option, or dev 31–50 question entered sampling.", ""]
    (out / "uncertainty_report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sample", "evaluate", "status", "export", "restore", "validate-checkpoint"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Explicitly allow one new bounded cycle for each exhausted sample")
    parser.add_argument("--max-new-samples", type=int,
                        help="Process at most this many new pair/seed units (valid or permanently failed)")
    parser.add_argument("--archive", type=Path, help="Checkpoint ZIP path for export or restore")
    args = parser.parse_args()
    if args.retry_failed and args.phase != "sample":
        parser.error("--retry-failed is only valid for sample")
    if args.max_new_samples is not None and args.phase != "sample":
        parser.error("--max-new-samples is only valid for sample")
    if args.phase == "sample":
        result = collect(args.root, base_url=args.base_url, retry_failed=args.retry_failed,
                         max_new_samples=args.max_new_samples)
    elif args.phase == "evaluate":
        result = evaluate(args.root)
    elif args.phase == "export":
        result = {"checkpoint": str(export_checkpoint(args.root, args.archive))}
    elif args.phase == "restore":
        if args.archive is None:
            parser.error("restore requires --archive")
        result = restore_checkpoint(args.archive, args.root)
    else:
        result = validate_restored_checkpoint(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
