"""Generate and persist stem-only clinical queries without contacting PubMed."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from urllib.request import urlopen

from . import clinical_query, ollama_backend
from .ollama_backend import OllamaTextGenerationBackend
from .query_cache import CachedClinicalQueryReformulator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata(base_url: str, endpoint: str) -> dict:
    with urlopen(base_url + endpoint, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid Ollama metadata")
    return payload


def _digest(payload: dict, model: str) -> str | None:
    for entry in payload.get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            value = entry.get("digest")
            return value if isinstance(value, str) and value else None
    return None


def _inputs(path: Path, limit: int) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in islice((line for line in handle if line.strip()), limit):
            raw = json.loads(line)
            # Deliberately access only these two fields, never labels/options/metadata.
            qid, stem = raw["id"], raw["question"]
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("Invalid question identifier")
            if not isinstance(stem, str) or not stem.strip():
                raise ValueError("Invalid question stem")
            records.append({"question_id": qid, "question_text": stem})
    if len(records) != limit or len({r["question_id"] for r in records}) != limit:
        raise ValueError("Expected the requested number of distinct questions")
    return records


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _saved_rows(previous: dict, identity: dict) -> dict[str, dict]:
    """Validate saved outputs before reuse, without generating replacements."""
    rows = previous.get("results")
    if not isinstance(rows, list) or previous.get("status") not in {"in_progress", "complete"}:
        raise ValueError("Invalid existing inspection report")
    if not isinstance(previous.get("created_at"), str) or not previous["created_at"]:
        raise ValueError("Missing saved report creation time")
    if previous.get("completed_question_count") != len(rows):
        raise ValueError("Invalid existing completion count")
    if [row.get("question_id") for row in rows if isinstance(row, dict)] != identity["question_ids"][:len(rows)]:
        raise ValueError("Invalid existing question order")
    if previous["status"] == "complete" and len(rows) != len(identity["question_ids"]):
        raise ValueError("Incomplete saved report")
    expected_fields = {
        "question_id", "generated_query", "backend_id", "model", "thinking_mode",
        "temperature", "seed", "model_digest", "generation_parameters", "prompt_sha256",
        "cache_key", "cache_hit", "query_created_at", "model_calls_this_run",
    }
    settings = identity["generation_parameters"]
    saved = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) not in (
            expected_fields, expected_fields | {"query_generation"},
        ):
            raise ValueError("Invalid saved query record")
        query = row["generated_query"]
        if clinical_query.normalize_query_output(query) != query:
            raise ValueError("Saved query must preserve the exact canonical output")
        if "<think" in query.lower() or "</think" in query.lower():
            raise ValueError("Reasoning tags in saved final text")
        if any(row[key] != identity[key] for key in ("backend_id", "model", "model_digest")):
            raise ValueError("Saved query model context differs")
        if (
            row["generation_parameters"] != settings or row["temperature"] != 0
            or row["seed"] != settings["seed"]
            or row["thinking_mode"] != ("ON" if settings["think"] else "OFF")
            or not isinstance(row["cache_hit"], bool)
            or type(row["model_calls_this_run"]) is not int
            or not (
                row["model_calls_this_run"] == 0 if row["cache_hit"]
                else 1 <= row["model_calls_this_run"] <= 3
            )
            or not isinstance(row["query_created_at"], str) or not row["query_created_at"]
        ):
            raise ValueError("Saved query settings or provenance differ")
        if "query_generation" in row:
            attempts = row["query_generation"]
            if (
                not isinstance(attempts, dict)
                or attempts.get("attempt_count") != row["model_calls_this_run"]
                or attempts.get("retry_count") != max(0, row["model_calls_this_run"] - 1)
            ):
                raise ValueError("Invalid saved query-generation diagnostics")
        for key in ("prompt_sha256", "cache_key"):
            value = row[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid saved query hash")
        saved[row["question_id"]] = row
    return saved


@contextmanager
def _isolated_requests(backend: OllamaTextGenerationBackend, context: dict):
    """Audit the actual request and discard the separate reasoning channel."""
    transport = ollama_backend.urlopen

    def guarded(request, *, timeout):
        if request.full_url != backend.base_url + "/api/generate" or request.get_method() != "POST":
            raise ValueError("Unexpected generation endpoint")
        body = json.loads(request.data.decode("utf-8"))
        if set(body) != {"model", "prompt", "stream", "think", "options"}:
            raise ValueError("Unexpected generation fields")
        if (
            body["model"] != backend.model or body["stream"] is not False
            or body["think"] is not backend.think
            or body["options"] != {"temperature": 0.0, "seed": backend.seed}
        ):
            raise ValueError("Unexpected generation settings")
        actual_input = json.loads(body["prompt"].split("QUESTION INPUT (JSON):\n", 1)[1])
        if actual_input != context["input"] or set(actual_input) != {"question_id", "question_text"}:
            raise ValueError("Generation input must contain only identifier and stem")
        context["model_calls"] += 1
        with transport(request, timeout=timeout) as response:
            server = json.loads(response.read().decode("utf-8"))
        if not isinstance(server, dict):
            raise ValueError("Invalid generation response")
        # Never access, print, or save server['thinking'].
        final = {key: server[key] for key in ("response", "done") if key in server}
        if "error" in server:
            final["error"] = True
        if server.get("model") != backend.model:
            raise ValueError("Unexpected served model")
        text = final.get("response")
        if isinstance(text, str) and ("<think" in text.lower() or "</think" in text.lower()):
            raise ValueError("Reasoning tags in final text")
        return io.BytesIO(json.dumps(final, ensure_ascii=False).encode("utf-8"))

    ollama_backend.urlopen = guarded
    try:
        yield
    finally:
        ollama_backend.urlopen = transport


def run_inspection(
    *, input_path: Path, output_path: Path, cache_dir: Path, limit: int = 10,
    base_url: str = "http://localhost:11434", model: str = "qwen3:8b",
    think: bool = True, seed: int = 42, timeout: float = 1800.0,
) -> dict:
    """Persist completed queries with bounded identical generation retries."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    inputs = _inputs(input_path, limit)
    backend = OllamaTextGenerationBackend(
        base_url=base_url, model=model, seed=seed, think=think, timeout=timeout,
    )
    model_digest = _digest(_metadata(backend.base_url, "/api/tags"), backend.model)
    version = _metadata(backend.base_url, "/api/version").get("version")
    source_sha = hashlib.sha256(Path(clinical_query.__file__).read_bytes()).hexdigest()
    settings = {"temperature": 0.0, "seed": seed, "think": think, "stream": False}
    identity = {
        "question_ids": [item["question_id"] for item in inputs],
        "question_stem_sha256": {
            item["question_id"]: hashlib.sha256(item["question_text"].encode("utf-8")).hexdigest()
            for item in inputs
        },
        "backend_id": backend.backend_id, "model": backend.model,
        "model_digest": model_digest, "generation_parameters": settings,
        "reformulator_source_sha256": source_sha,
    }
    previous = None
    saved_rows = {}
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or previous.get("inspection_identity") != identity:
            raise ValueError("Existing output belongs to another inspection; choose a new path")
        saved_rows = _saved_rows(previous, identity)
        if previous["status"] == "complete":
            print("question_id\tgenerated_query", flush=True)
            for row in previous["results"]:
                print(row["question_id"] + "\t" + row["generated_query"], flush=True)
            return previous
    reformulator = CachedClinicalQueryReformulator(
        backend, backend_id=backend.backend_id, cache_dir=cache_dir,
        generation_settings=settings, model_digest=model_digest,
    )
    report = {
        "inspection_identity": identity, "created_at": previous["created_at"] if previous else _now(),
        "input_file": str(input_path), "cache_dir": str(cache_dir),
        "backend_id": backend.backend_id, "model": backend.model,
        "thinking_mode": "ON" if think else "OFF", "temperature": 0.0, "seed": seed,
        "model_digest": model_digest, "ollama_version": version,
        "request_timeout_seconds": timeout, "pubmed_called": False,
        "reasoning_retained": False, "status": "in_progress", "results": [],
    }
    context = {"model_calls": 0}
    print("question_id\tgenerated_query", flush=True)
    with _isolated_requests(backend, context):
        for item in inputs:
            if item["question_id"] in saved_rows:
                row = saved_rows[item["question_id"]]
                report["results"].append(row)
                print(row["question_id"] + "\t" + row["generated_query"], flush=True)
                continue
            context["input"] = item
            calls_before = context["model_calls"]
            if _digest(_metadata(backend.base_url, "/api/tags"), backend.model) != model_digest:
                raise ValueError("Model digest changed before generation")
            query = reformulator.reformulate(**item)
            cached = reformulator.last_record
            after = _digest(_metadata(backend.base_url, "/api/tags"), backend.model)
            if after != model_digest:
                raise ValueError("Model digest changed during inspection")
            row = {
                "question_id": item["question_id"], "generated_query": query,
                "backend_id": backend.backend_id, "model": backend.model,
                "thinking_mode": report["thinking_mode"], "temperature": 0.0, "seed": seed,
                "model_digest": model_digest, "generation_parameters": settings.copy(),
                "prompt_sha256": cached["prompt_sha256"], "cache_key": cached["cache_key"],
                "cache_hit": cached["cache_hit"], "query_created_at": cached["created_at"],
                "model_calls_this_run": context["model_calls"] - calls_before,
                "query_generation": cached["query_generation"],
            }
            if row["model_calls_this_run"] != cached["query_generation"]["attempt_count"]:
                raise ValueError("Generation request count differs from the recorded attempts")
            report["results"].append(row)
            report["completed_question_count"] = len(report["results"])
            report["model_calls_this_run"] = context["model_calls"]
            report["cache_hits_this_run"] = sum(row["cache_hit"] for row in report["results"])
            _save(output_path, report)
            print(item["question_id"] + "\t" + query, flush=True)
    if hashlib.sha256(Path(clinical_query.__file__).read_bytes()).hexdigest() != source_sha:
        raise ValueError("Reformulation source changed during inspection")
    report.update({
        "status": "complete", "completed_at": _now(), "completed_question_count": len(inputs),
        "model_calls_this_run": context["model_calls"],
        "reused_saved_results_this_run": len(saved_rows),
        "cache_hits_this_run": sum(row["cache_hit"] for row in report["results"] if row["question_id"] not in saved_rows),
    })
    _save(output_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/medqa_us_dev_50.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/query_reformulation/medqa_dev_10_thinking_on.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/clinical_queries"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    try:
        run_inspection(
            input_path=args.questions, output_path=args.output, cache_dir=args.cache_dir,
            limit=args.limit, base_url=args.base_url, model=args.model,
            think=args.thinking == "on", seed=args.seed, timeout=args.timeout,
        )
    except Exception as exc:
        # Do not print prompt bodies, raw server text, exception messages or traces.
        print("Query inspection stopped: " + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
