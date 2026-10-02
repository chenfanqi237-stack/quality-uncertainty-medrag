"""Explicit, credential-free Kaggle packaging; no model/network calls."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from .cloud_runtime import project_root, read_json, relative_path, save_json, sha256


def input_selection(root: Path) -> dict[str, str]:
    files = {}
    def add(name, purpose):
        path = relative_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(name)
        files[name] = purpose

    for folder, purpose, pattern in (
        ("src", "Current project source, including unchanged research logic", "*.py"),
        ("tests", "Offline verification and smoke tests", "*.py"),
        ("schemas", "Existing data/output schema definitions", "*.json"),
        ("configs", "Existing reproducibility configuration", "*.yaml"),
    ):
        for path in sorted((root / folder).rglob(pattern)):
            if "__pycache__" not in path.parts:
                add(path.relative_to(root).as_posix(), purpose)
    explicit = {
        "pyproject.toml": "Editable installation and dependencies",
        "README.md": "Package metadata and project documentation",
        "REFERENCE.md": "Reference provenance; reference repository itself is excluded",
        "data/synthetic_medqa.jsonl": "Existing offline baseline smoke fixture",
        "data/synthetic_evidence.jsonl": "Existing offline baseline smoke fixture",
        "data/processed/medqa_us_dev_50.jsonl": "Unchanged processed MedQA input; runner selects stems 11-30 only",
        "outputs/cloud_migration/local_model_identity.json": "Recorded local Ollama/model identity",
        "outputs/query_reformulation/medqa_dev_10_thinking_on.json": "Frozen development primary reformulation outputs",
        "data/retrieved/pubmed_medqa_us_dev_10_llm.jsonl": "Current development PubMed evidence snapshot",
        "data/retrieved/pubmed_medqa_us_dev_10_llm.report.json": "Current development retrieval/query provenance",
        "outputs/query_relaxation/prompt_v2_final/medqa_dev_4.jsonl": "Final frozen four-question fallback evidence snapshot",
        "outputs/query_relaxation/prompt_v2_final/medqa_dev_4.report.json": "Final fallback cache keys, queries and retrieval provenance",
        "cloud/kaggle_config.json": "Portable frozen execution configuration",
        "cloud/research_freeze.json": "Research/source/input integrity hashes",
        "cloud/README.md": "Execution documentation",
        "cloud/WHAT_I_NEED_TO_CLICK.md": "Unavoidable UI actions",
        "cloud/kernel-metadata.json": "GPU/private/Internet kernel template",
        "notebooks/kaggle_qwen3_medrag.ipynb": "Executable Kaggle notebook",
    }
    for name, purpose in explicit.items():
        add(name, purpose)
    primary_artifact = read_json(root / "outputs/query_reformulation/medqa_dev_10_thinking_on.json")
    for row in primary_artifact["results"]:
        key = row["cache_key"]
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid primary cache filename")
        add(f"data/cache/clinical_queries/{key}.json", "Exact development primary query cache; no regeneration")
    fallback_report = read_json(root / "outputs/query_relaxation/prompt_v2_final/medqa_dev_4.report.json")
    for row in fallback_report["questions"]:
        key = row["fallback_cache_key"]
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid fallback cache filename")
        add(f"outputs/query_relaxation/cache/{key}.json", "Exact current frozen fallback cache (older prompt variants excluded)")
    primary_report = read_json(root / "data/retrieved/pubmed_medqa_us_dev_10_llm.report.json")
    queries = {row["query"] for row in primary_report["questions"]}
    queries.update(row["query_used"] for row in fallback_report["questions"])
    pmids = set()
    for path in sorted((root / "data/cache/pubmed/searches").glob("*.json")):
        cached = read_json(path)
        if cached.get("request", {}).get("term") in queries and cached["request"].get("retmax") == "15":
            add(path.relative_to(root).as_posix(), "PubMed search snapshot for frozen development queries")
            pmids.update(cached["result"]["idlist"])
    for pmid in sorted(pmids):
        for category in ("records", "unavailable"):
            path = root / "data/cache/pubmed" / category / (pmid + ".json")
            if path.is_file():
                add(path.relative_to(root).as_posix(), "Shared per-PMID metadata/abstract or unsupported-record cache")
    return files


def build(root: Path) -> dict:
    files = input_selection(root)
    try:
        raw = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", "-z"], cwd=root)
        tracked = set(raw.decode("utf-8").split("\0"))
    except (OSError, subprocess.CalledProcessError):
        tracked = set()  # A unpacked cloud copy need not contain Git internals.
    supplement = root / "cloud/kaggle_input"
    supplement.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, purpose in sorted(files.items()):
        source = relative_path(root, name)
        entry = {"path": name, "purpose": purpose, "size_bytes": source.stat().st_size,
                 "sha256": sha256(source), "tracked_in_git": name in tracked,
                 "archive_path": "quality-uncertainty-medrag/" + name,
                 "supplement_path": None if name in tracked else "quality-uncertainty-medrag/" + name}
        if name not in tracked:
            target = supplement / entry["supplement_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != entry["sha256"]:
                shutil.copyfile(source, target)
        entries.append(entry)
    # Remove only obsolete files previously produced in this managed supplement.
    old_manifest = root / "cloud/kaggle_input_manifest.json"
    if old_manifest.is_file():
        previous = read_json(old_manifest)
        retained = {e["supplement_path"] for e in entries if e["supplement_path"]}
        for entry in previous.get("files", []):
            name = entry.get("supplement_path")
            if name and name not in retained:
                relative_path(supplement, name).unlink(missing_ok=True)
    manifest = {"schema_version": 1, "archive_prefix": "quality-uncertainty-medrag",
                "file_count": len(entries), "total_uncompressed_bytes": sum(e["size_bytes"] for e in entries),
                "generated_metadata": [{"path": "cloud/kaggle_input_manifest.json",
                    "purpose": "Self-describing manifest; self-hashing is deliberately excluded"}],
                "excluded": [".git", "virtual environments", "pytest caches", "Python bytecode",
                             "credentials", "Ollama models", "historical debug outputs", "Med-RR-reference"],
                "heldout_executed": False, "files": entries}
    save_json(old_manifest, manifest)
    metadata_copy = supplement / "quality-uncertainty-medrag/cloud/kaggle_input_manifest.json"
    metadata_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(old_manifest, metadata_copy)
    archive = root / "cloud/kaggle_upload.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for entry in entries:
            bundle.write(relative_path(root, entry["path"]), entry["archive_path"])
        bundle.write(old_manifest, "quality-uncertainty-medrag/cloud/kaggle_input_manifest.json")
    temporary.replace(archive)
    return {"files": len(entries), "supplement_files": sum(not e["tracked_in_git"] for e in entries),
            "uncompressed_bytes": manifest["total_uncompressed_bytes"], "zip_bytes": archive.stat().st_size,
            "archive": str(archive)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(project_root(args.project_root)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
