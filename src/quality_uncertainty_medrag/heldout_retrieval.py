"""Development-only dev 11-30 retrieval execution, with honest checkpoints.

The historical module/output names remain for compatibility. Questions 1-30
are development data; this module does not select the untouched final set.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from .clinical_query_relaxation import CoreClinicalQueryReformulator
from .clinical_query_minimal import MinimalClinicalQueryReformulator
from .cloud_runtime import (CloudConfig, GENERATION_SETTINGS, HELDOUT_IDS, compare_digest,
                            heldout_stems, ollama_identity, read_json, save_json,
                            sha256, verify_freeze, warn_digest)
from .ollama_backend import OllamaTextGenerationBackend
from .pubmed import PubMedClient, evidence_to_json_record
from .query_cache import CachedClinicalQueryReformulator
from .query_relaxation import QueryRelaxationRetriever, RELAX_BELOW_MATCH_COUNT, _match_count


class SearchAudit:
    """Observe unchanged E-utilities calls, including counts before a failure."""
    def __init__(self, client):
        self.client = client
        self.searches = []

    def search(self, query, *, top_k):
        before = getattr(self.client, "request_count", None)
        result = self.client.search(query, top_k=top_k)
        after = getattr(self.client, "request_count", None)
        self.searches.append({"query": query, "count": _match_count(result),
                              "cache_hit": before == after if before is not None else None})
        return result

    def fetch(self, pmids):
        return self.client.fetch(pmids)


class IdentityGuard:
    def __init__(self, backend, digest, identity_reader):
        self.backend, self.digest, self.identity_reader = backend, digest, identity_reader
        self.calls = 0

    def generate(self, prompt, *, generation_config=None):
        if self.identity_reader()["model_digest"] != self.digest:
            raise ValueError("Model identity changed during run; refusing mixed outputs")
        self.calls += 1
        return self.backend.generate(prompt, generation_config=generation_config)


def run(config: CloudConfig, identity: dict, backend, client, *, run_id: str,
        identity_reader=None, resume: bool = False) -> Path:
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("run_id must contain only letters, numbers, hyphens and underscores")
    frozen = verify_freeze(config.root, config.freeze)
    stems = heldout_stems(config.questions)
    local = read_json(config.local_identity)
    comparison = compare_digest(local["model_digest"], identity["model_digest"])
    warn_digest(comparison)
    digest = comparison["actual_model_digest"]
    # Short folders avoid Windows MAX_PATH; full identity guards prefix collisions.
    base = config.output / digest[:16]
    query_cache = config.root / "data/cache/heldout_queries" / digest[:16]
    for scoped in (base, query_cache):
        scope_file = scoped / "model_identity.json"
        scope = {"model_digest": digest, "generation_settings": GENERATION_SETTINGS}
        if scope_file.exists():
            if read_json(scope_file) != scope:
                raise ValueError("Digest prefix collision or incompatible scoped settings")
        else:
            save_json(scope_file, scope)
    directory = base / "runs" / run_id
    report_path = directory / "report.json"
    guard = IdentityGuard(backend, digest, identity_reader or (lambda: identity))
    common = dict(backend_id="ollama/qwen3:8b", generation_settings=GENERATION_SETTINGS, model_digest=digest)
    primary = CachedClinicalQueryReformulator(guard, cache_dir=query_cache / "primary", **common)
    fallback = CachedClinicalQueryReformulator(guard, cache_dir=query_cache / "fallback",
                                             reformulator_factory=CoreClinicalQueryReformulator, **common)
    minimal = CachedClinicalQueryReformulator(guard, cache_dir=query_cache / "minimal",
                                             reformulator_factory=MinimalClinicalQueryReformulator, **common)
    audit = SearchAudit(client)
    stage = QueryRelaxationRetriever(audit, fallback_reformulator=fallback, minimal_reformulator=minimal)
    identity_fields = {"question_ids": list(HELDOUT_IDS), "generation_settings": GENERATION_SETTINGS,
                       "model": "qwen3:8b", "backend_id": "ollama/qwen3:8b", "top_k": config.top_k,
                       "research_files_sha256": frozen, "questions_sha256": sha256(config.questions),
                       "ollama_version": identity["ollama_version"],
                       "query_cache_relative_path": query_cache.relative_to(config.root).as_posix(), **comparison}
    identity_fields.update(dataset_role="development", retrieval_cascade_revision=2,
                           max_query_generation_retries=2)
    if report_path.exists():
        if not resume:
            raise FileExistsError("Run exists; use --resume, never overwrite it")
        report = read_json(report_path)
        if any(report.get(k) != v for k, v in identity_fields.items()):
            raise ValueError("Resume metadata differs from frozen run identity")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        report = {**identity_fields, "run_id": run_id, "status": "in_progress", "questions": [],
                  "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "platform": platform.platform(), "model_calls_total": 0,
                  "stance_classification_run": False, "quality_scoring_run": False,
                  "aggregation_run": False, "gold_fields_accessed": False}

    def checkpoint():
        report["model_calls_total"] += guard.calls
        guard.calls = 0
        save_json(report_path, report)

    checkpoint()
    completed = {row["question_id"] for row in report["questions"]}
    for qid, text in stems:
        if qid in completed:
            continue  # Includes failed questions: no silent repair/regeneration.
        fallback.last_record = None
        fallback.last_attempt_record = None
        minimal.last_record = None
        minimal.last_attempt_record = None
        stage.last_report = {}
        audit.searches = []
        row = {"question_id": qid, "model": "qwen3:8b", "backend_id": "ollama/qwen3:8b",
               "model_digest": digest, "thinking": True, "temperature": 0.0, "seed": 42,
               "primary_query": None, "primary_pubmed_match_count": None,
               "fallback_triggered": None, "fallback_query": None, "fallback_pubmed_match_count": None,
               "minimal_fallback_triggered": None, "minimal_fallback_query": None,
               "minimal_fallback_pubmed_match_count": None, "minimal_fallback_query_cache_hit": None,
               "final_query_used": None, "saved_article_count": 0,
               "top_5_pmids": [], "top_5_titles": [],
               "primary_query_cache_hit": None, "fallback_query_cache_hit": None}
        try:
            query = primary.reformulate(question_id=qid, question_text=text)
            row.update(primary_query=query, primary_query_cache_hit=primary.last_record["cache_hit"],
                       primary_cache_record=primary.last_record)
            # Persist the primary before PubMed: interruption never loses its exact query.
            save_json(directory / "primary_queries" / (qid + ".json"), primary.last_record)
            result = stage.retrieve(question_id=qid, question_text=text, primary_query=query, top_k=config.top_k)
            row.update(result.report)
            row.update(status="ok", final_query_used=result.report["query_used"],
                       top_5_pmids=[a["pmid"] for a in result.report["top_5"]],
                       top_5_titles=[a["title"] for a in result.report["top_5"]])
            evidence_path = directory / "evidence" / (qid + ".jsonl")
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = evidence_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for item in result.evidence:
                    handle.write(json.dumps(evidence_to_json_record(item), ensure_ascii=False, allow_nan=False) + "\n")
            tmp.replace(evidence_path)
        except Exception as exc:
            # Bounded retries occur inside the cache; never repair queries here.
            row.update(status="failed", failure_type=type(exc).__name__)
            row.update(stage.last_report)
            if fallback.last_record is not None:
                row.update(fallback_query=fallback.last_record["generated_query"],
                           fallback_query_cache_hit=fallback.last_record["cache_hit"])
            if minimal.last_record is not None:
                row.update(minimal_fallback_query=minimal.last_record["generated_query"],
                           minimal_fallback_query_cache_hit=minimal.last_record["cache_hit"])
            print(qid + ": FAILED (" + type(exc).__name__ + "); recorded without repair")
        row["pubmed_search_audit"] = list(audit.searches)
        if audit.searches:
            row["primary_pubmed_match_count"] = audit.searches[0]["count"]
            row["fallback_triggered"] = audit.searches[0]["count"] < RELAX_BELOW_MATCH_COUNT
        if len(audit.searches) > 1:
            row["fallback_pubmed_match_count"] = audit.searches[1]["count"]
            row["minimal_fallback_triggered"] = audit.searches[1]["count"] < RELAX_BELOW_MATCH_COUNT
        if len(audit.searches) > 2:
            row["minimal_fallback_pubmed_match_count"] = audit.searches[2]["count"]
        row["query_generation"] = {
            name: reformulator.last_record["query_generation"] if reformulator.last_record else reformulator.last_attempt_record
            for name, reformulator in (("primary", primary), ("fallback", fallback), ("minimal_fallback", minimal))
        }
        row["query_generation_retry_count"] = sum(
            record["retry_count"] for record in row["query_generation"].values() if record)
        report["questions"].append(row)
        checkpoint()
        print(json.dumps({k: row[k] for k in ("question_id", "status", "primary_query",
              "primary_pubmed_match_count", "fallback_query", "fallback_pubmed_match_count",
              "saved_article_count", "top_5_pmids", "top_5_titles")}, ensure_ascii=False), flush=True)
    report.update(status="complete_with_failures" if any(r["status"] == "failed" for r in report["questions"]) else "complete",
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  failed_question_count=sum(r["status"] == "failed" for r in report["questions"]),
                  total_saved_articles=sum(r["saved_article_count"] for r in report["questions"]))
    combined = directory / "evidence.jsonl"
    tmp = combined.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for qid in HELDOUT_IDS:
            part = directory / "evidence" / (qid + ".jsonl")
            if part.exists():
                handle.write(part.read_text(encoding="utf-8"))
    tmp.replace(combined)
    checkpoint()
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("cloud-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Offline validation, no model or PubMed requests")
    args = parser.parse_args(argv)
    config = CloudConfig.load(args.project_root)
    verify_freeze(config.root, config.freeze)
    stems = heldout_stems(config.questions)
    if args.check_only:
        print(json.dumps({"question_ids": [q for q, _ in stems], "count": len(stems),
                          "research_freeze_verified": True, "model_calls": 0, "pubmed_calls": 0}))
        return 0
    identity = ollama_identity(config.base_url)
    backend = OllamaTextGenerationBackend(base_url=config.base_url, model="qwen3:8b",
                                         think=True, seed=42, timeout=config.timeout)
    client = PubMedClient(config.pubmed_cache, email=os.environ.get("NCBI_EMAIL"), api_key=os.environ.get("NCBI_API_KEY"))
    result = run(config, identity, backend, client, run_id=args.run_id, resume=args.resume,
                 identity_reader=lambda: ollama_identity(config.base_url))
    print("Saved: " + str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
