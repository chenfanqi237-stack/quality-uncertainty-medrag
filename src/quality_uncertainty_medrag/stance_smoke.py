"""Small development-only stance inspection; never retrieves or aggregates evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from urllib.request import Request, urlopen

from . import ollama_backend
from .cloud_runtime import ollama_identity, project_root, save_json, verify_freeze
from .llm_stance import (LLMStanceClassifier, MODEL_INPUT_FIELDS, STANCE_PROMPT,
                         StanceCacheError, StanceModelConfig, build_stance_prompt)
from .ollama_backend import OllamaTextGenerationBackend

DEVELOPMENT_IDS = tuple(f"medqa-us-dev-{i:06d}" for i in (11, 12, 13))
RECORD_FIELDS = set(MODEL_INPUT_FIELDS) | {"question_id", "candidate_option_id", "evidence_doc_id"}


def _development_ids(ids):
    if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        raise ValueError("Smoke inspection requires 2-3 distinct development questions")
    if any(not isinstance(qid, str) or not re.fullmatch(r"medqa-us-dev-\d{6}", qid) or not 1 <= int(qid.rsplit("-", 1)[1]) <= 30 for qid in ids):
        raise ValueError("Only exposed development questions 1-30 are allowed")


def prepare_development_inputs(questions_path, evidence_path, *, question_ids=DEVELOPMENT_IDS):
    """First two options, first ranked saved article with an abstract; no gold access."""
    _development_ids(question_ids)
    selected = {}
    with Path(questions_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            qid = raw.get("id")
            if qid not in question_ids:
                continue
            if qid in selected or not isinstance(raw["options"], dict):
                raise ValueError("Duplicate question or unsupported option representation")
            selected[qid] = (raw["question"], list(islice(raw["options"].items(), 2)))
            # Never read answer, answer_idx, metadata, or remaining option values.
    articles = {}
    with Path(evidence_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            qid = raw.get("question_id")
            if qid not in question_ids:
                continue
            metadata = raw["metadata"]
            title, abstract = metadata.get("title"), metadata.get("abstract")
            if isinstance(title, str) and title.strip() and isinstance(abstract, str) and abstract.strip():
                if qid not in articles or raw["rank"] < articles[qid][0]:
                    articles[qid] = (raw["rank"], raw["doc_id"], title, abstract)
    if set(selected) != set(question_ids) or set(articles) != set(question_ids):
        raise ValueError("Selected development questions need saved articles with abstracts")
    rows = []
    for qid in question_ids:
        stem, options = selected[qid]
        if len(options) != 2:
            raise ValueError("Expected two development candidates")
        _, pmid, title, abstract = articles[qid]
        for label, option in options:
            rows.append({"question_id": qid, "candidate_option_id": label, "evidence_doc_id": pmid,
                         "question_stem": stem, "candidate_option_text": option,
                         "evidence_title": title, "evidence_abstract": abstract})
    validate_smoke_inputs(rows)
    return rows


def validate_smoke_inputs(rows):
    if not isinstance(rows, list) or not 2 <= len(rows) <= 6:
        raise ValueError("Expected 2-6 small smoke judgments")
    seen, options, ids = set(), {}, []
    for row in rows:
        if not isinstance(row, dict) or set(row) != RECORD_FIELDS:
            raise ValueError("Smoke inputs must contain only identifiers and four text fields")
        for key, value in row.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Smoke inputs require real nonempty strings")
        qid = row["question_id"]
        if qid not in ids:
            ids.append(qid)
        key = (qid, row["candidate_option_id"], row["evidence_doc_id"])
        if key in seen:
            raise ValueError("Duplicate stance judgment")
        seen.add(key)
        options.setdefault(qid, set()).add(row["candidate_option_id"])
        build_stance_prompt(**{name: row[name] for name in MODEL_INPUT_FIELDS})
    _development_ids(ids)
    if any(len(labels) > 2 for labels in options.values()):
        raise ValueError("At most two candidates per development question")
    return ids


def _required_identity(identity):
    config = StanceModelConfig()
    if any(identity.get(key) != value for key, value in {
        "model_name": config.model, "backend_id": config.backend_id,
        "model_digest": config.model_digest, "ollama_version": config.ollama_version}.items()):
        raise ValueError("Runtime identity differs from pinned stance model; refusing substitution")


def gpu_preflight(base_url):
    identity = ollama_identity(base_url)
    _required_identity(identity)
    gpu = subprocess.check_output(["nvidia-smi"], text=True)
    config = StanceModelConfig()
    # Ollama's documented no-prompt request only loads weights, generating no query/judgment.
    body = {"model": config.model, "stream": False, "think": config.thinking,
            "options": {"temperature": config.temperature, "seed": config.seed}}
    request = Request(base_url.rstrip("/") + "/api/generate", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=300) as response:
        loaded = json.load(response)
    if loaded.get("done") is not True or loaded.get("response"):
        raise ValueError("Model-load request unexpectedly generated text")
    with urlopen(base_url.rstrip("/") + "/api/ps", timeout=15) as response:
        models = json.load(response)["models"]
    active = next(item for item in models if item.get("name") == config.model)
    if active["digest"] != config.model_digest or active.get("size_vram", 0) <= 0:
        raise ValueError("Pinned model is not loaded on GPU")
    return {"identity": identity, "gpu_info": gpu, "ollama_ps": active, "model_load_requests": 1}


@contextmanager
def _isolated_requests(backend, row, audit):
    original = ollama_backend.urlopen
    expected_prompt = build_stance_prompt(**{key: row[key] for key in MODEL_INPUT_FIELDS})
    config = StanceModelConfig()

    def guarded(request, *, timeout):
        body = json.loads(request.data.decode("utf-8"))
        if request.full_url != backend.base_url + "/api/generate" or request.get_method() != "POST":
            raise ValueError("Unexpected stance endpoint")
        if set(body) != {"model", "prompt", "stream", "think", "options"} or body["prompt"] != expected_prompt:
            raise ValueError("Unexpected stance model payload")
        if body["model"] != config.model or body["think"] is not True or body["stream"] is not False or body["options"] != {"temperature": 0.0, "seed": 42}:
            raise ValueError("Stance generation settings changed")
        actual = ollama_identity(backend.base_url)
        _required_identity(actual)
        audit.append({"question_id": row["question_id"], "candidate_option_id": row["candidate_option_id"],
                      "evidence_doc_id": row["evidence_doc_id"], "input_fields": list(MODEL_INPUT_FIELDS),
                      "prompt_sha256": hashlib.sha256(expected_prompt.encode()).hexdigest(),
                      "model_metadata": asdict(config), "timestamp": datetime.now(timezone.utc).isoformat()})
        return original(request, timeout=timeout)
    ollama_backend.urlopen = guarded
    try:
        yield
    finally:
        ollama_backend.urlopen = original


def run_smoke(*, root, inputs_path, output_dir, cache_dir, base_url="http://localhost:11434", timeout=1800):
    root, inputs_path, output_dir = Path(root), Path(inputs_path), Path(output_dir)
    with inputs_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = validate_smoke_inputs(rows)
    frozen = verify_freeze(root, root / "cloud/research_freeze.json")
    if output_dir.exists():
        raise ValueError("Refusing to overwrite an existing stance inspection")
    runtime = gpu_preflight(base_url)
    backend = OllamaTextGenerationBackend(base_url=base_url, model="qwen3:8b", timeout=timeout, seed=42, think=True)
    classifier = LLMStanceClassifier(backend, cache_dir=cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "classifier_prompt.txt").write_text(STANCE_PROMPT, encoding="utf-8")
    audit, results = [], []
    report = {"status": "running", "question_ids": ids, "runtime": runtime,
              "platform": platform.platform(), "input_file_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
              "frozen_files_sha256": frozen, "model_metadata": asdict(StanceModelConfig()), "results": results}
    for row in rows:
        try:
            with _isolated_requests(backend, row, audit):
                classifier.classify_texts(**row)
            results.append({**classifier.last_record, "status": "SUCCESS"})
        except Exception as exc:
            attempts = classifier.last_attempt_record or {}
            results.append({"question_id": row["question_id"], "candidate_option_id": row["candidate_option_id"],
                            "evidence_doc_id": row["evidence_doc_id"], "status": "FAILURE", "error_type": type(exc).__name__,
                            "p_support": None, "p_contradict": None, "p_irrelevant": None, "argmax_label": None,
                            "model_metadata": asdict(StanceModelConfig()),
                            "cache_status": "ERROR" if isinstance(exc, StanceCacheError) else "MISS",
                            "generation_attempts": attempts.get("generation_attempts", 0),
                            "generation_retries": attempts.get("retry_count", 0)})
        save_json(output_dir / "report.json", report)
        save_json(output_dir / "request_audit.json", audit)
    verify_freeze(root, root / "cloud/research_freeze.json")
    report["status"] = "complete" if all(row["status"] == "SUCCESS" for row in results) else "complete_with_failures"
    report["summary"] = {"questions": len(ids), "judgments": len(rows), "model_calls": len(audit),
        "successes": sum(x["status"] == "SUCCESS" for x in results),
        "failures": sum(x["status"] == "FAILURE" for x in results),
        "generation_attempts": sum(x["generation_attempts"] for x in results),
        "generation_retries": sum(x["generation_retries"] for x in results),
        "cache_hits": sum(x.get("cache_status") == "HIT" for x in results),
        "model_load_requests": runtime["model_load_requests"], "pubmed_requests": 0,
        "options_or_gold_in_model_payloads": False, "retrieval_or_aggregation_changes": False,
        "final_questions_31_50_used": False}
    save_json(output_dir / "report.json", report)
    with (output_dir / "stances.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            if result["status"] == "SUCCESS":
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="GPU stance smoke inspection on 2-3 exposed development questions only")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--inputs", type=Path, default=Path("data/stance_smoke/dev_11_13_inputs.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/stance/v1"))
    parser.add_argument("--base-url", default=os.environ.get("MEDRAG_OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("MEDRAG_OLLAMA_TIMEOUT", "1800")))
    args = parser.parse_args(argv)
    root = project_root(args.project_root)
    output = args.output_dir or Path("outputs/stance_smoke/dev_11_13") / datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%SZ")
    resolved = lambda path: path if path.is_absolute() else root / path
    report = run_smoke(root=root, inputs_path=resolved(args.inputs), output_dir=resolved(output),
                       cache_dir=resolved(args.cache_dir), base_url=args.base_url, timeout=args.timeout)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for row in report["results"]:
        print(json.dumps({key: row.get(key) for key in ("question_id", "candidate_option_id", "evidence_doc_id",
            "p_support", "p_contradict", "p_irrelevant", "argmax_label", "generation_attempts", "cache_status", "status")}, ensure_ascii=False))
    print("Outputs:", resolved(output))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
