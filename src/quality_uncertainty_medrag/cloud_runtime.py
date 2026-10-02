"""Execution paths, identity checks and frozen-input guards; no research logic."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.request import urlopen

from .query_relaxation import GENERATION_SETTINGS, RELAX_BELOW_MATCH_COUNT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_root(root: str | Path | None = None) -> Path:
    supplied = root or os.environ.get("MEDRAG_PROJECT_ROOT")
    if supplied:
        result = Path(supplied).expanduser().resolve()
    else:
        result = Path(__file__).resolve().parents[2]
    if not (result / "pyproject.toml").is_file():
        raise ValueError("Project root must contain pyproject.toml; set MEDRAG_PROJECT_ROOT")
    return result


def relative_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Use nonempty project-relative paths with forward slashes")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError("Cloud paths must stay inside the project")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("Cloud path escapes project root")
    return resolved


def canonical_digest(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Missing model digest")
    result = value.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[a-f0-9]{64}", result):
        raise ValueError("Expected a full SHA256 model digest")
    return result


def compare_digest(local: str, cloud: str) -> dict:
    local, cloud = canonical_digest(local), canonical_digest(cloud)
    return {"local_model_digest": local, "actual_model_digest": cloud,
            "digest_matches_local": local == cloud, "outputs_isolated_by_digest": True}


def warn_digest(comparison: dict) -> None:
    if not comparison["digest_matches_local"]:
        print("\n" + "!" * 78)
        print("WARNING: MODEL DIGEST DIFFERS FROM LOCAL. THIS IS A SEPARATE CLOUD RUN.")
        print("Local : " + comparison["local_model_digest"])
        print("Cloud : " + comparison["actual_model_digest"])
        print("Development outputs and queries will NOT be overwritten or mixed.")
        print("!" * 78 + "\n")


def ollama_identity(base_url: str, model: str = "qwen3:8b") -> dict:
    with urlopen(base_url.rstrip("/") + "/api/version", timeout=15) as response:
        version = json.load(response)["version"]
    with urlopen(base_url.rstrip("/") + "/api/tags", timeout=15) as response:
        models = json.load(response)["models"]
    selected = next((m for m in models if m.get("name") == model or m.get("model") == model), None)
    if selected is None:
        raise ValueError("Required model is not installed: " + model)
    return {"model_name": model, "backend_id": "ollama/" + model,
            "model_digest": canonical_digest(selected["digest"]), "ollama_version": version,
            "model_details": selected.get("details", {})}


def verify_freeze(root: Path, manifest: Path) -> dict:
    hashes = read_json(manifest)["files"]
    changed = [name for name, digest in hashes.items()
               if not relative_path(root, name).is_file() or sha256(relative_path(root, name)) != digest]
    if changed:
        raise ValueError("Frozen research/input files changed: " + ", ".join(changed))
    return hashes


@dataclass(frozen=True)
class CloudConfig:
    root: Path
    questions: Path
    local_identity: Path
    freeze: Path
    output: Path
    pubmed_cache: Path
    base_url: str
    timeout: float
    top_k: int = 15

    @classmethod
    def load(cls, root: str | Path | None = None, config: Path | None = None):
        root = project_root(root)
        values = read_json(config or root / "cloud/kaggle_config.json")
        expected = {"model": "qwen3:8b", "thinking": True, "temperature": 0,
                    "seed": 42, "start_question": 11, "end_question": 30,
                    "top_k": 15, "fallback_below_matches": RELAX_BELOW_MATCH_COUNT}
        for key, value in expected.items():
            if type(values.get(key)) is not type(value) or values[key] != value:
                raise ValueError("Frozen run setting changed: " + key)
        paths = values["paths"]
        timeout = float(os.environ.get("MEDRAG_OLLAMA_TIMEOUT", "1800"))
        if not 0 < timeout < float("inf"):
            raise ValueError("MEDRAG_OLLAMA_TIMEOUT must be finite and positive")
        return cls(root, *(relative_path(root, paths[name]) for name in
                   ("questions", "local_identity", "freeze", "output", "pubmed_cache")),
                   os.environ.get("MEDRAG_OLLAMA_URL", "http://localhost:11434"), timeout)


HELDOUT_IDS = tuple(f"medqa-us-dev-{i:06d}" for i in range(11, 31))


def heldout_stems(path: Path) -> list[tuple[str, str]]:
    """Read only ID/stem; never construct a record containing options or gold."""
    selected = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = record.get("id")
            if qid not in HELDOUT_IDS:
                continue
            text = record.get("question")
            if qid in selected or not isinstance(text, str) or not text.strip():
                raise ValueError("Duplicate/malformed held-out stem: " + qid)
            selected[qid] = text
    if set(selected) != set(HELDOUT_IDS):
        raise ValueError("All questions 11-30 must be present exactly once")
    return [(qid, selected[qid]) for qid in HELDOUT_IDS]
