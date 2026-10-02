"""Cached primary/core/minimal query cascade around the existing PubMed retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

from .clinical_query_relaxation import CoreClinicalQueryReformulator
from .clinical_query_minimal import MinimalClinicalQueryReformulator
from .models import RetrievedEvidence
from .ollama_backend import OllamaTextGenerationBackend
from .pubmed import PubMedClient, PubMedRetriever, evidence_to_json_record
from .query_cache import CachedClinicalQueryReformulator


VALIDATION_QUESTION_IDS = (
    "medqa-us-dev-000001", "medqa-us-dev-000002",
    "medqa-us-dev-000008", "medqa-us-dev-000009",
)
RELAX_BELOW_MATCH_COUNT = 5
GENERATION_SETTINGS = {"temperature": 0.0, "seed": 42, "think": True, "stream": False}


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _match_count(search: dict) -> int:
    value = search.get("count")
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    if type(value) is int and value >= 0:
        return value
    raise ValueError("PubMed returned an invalid total match count")


class _ExactQuery:
    def __init__(self, query: str, backend_id: str, reformulator_id: str):
        self.query = query
        self.backend_id = backend_id
        self.reformulator_id = reformulator_id

    def reformulate(self, *, question_id: str, question_text: str) -> str:
        return self.query


class _SearchSnapshot:
    """Use the just-obtained primary response without another search call."""

    def __init__(self, client, primary_query: str, primary_search: dict, top_k: int):
        self.client = client
        self.primary_query = primary_query
        self.primary_search = primary_search
        self.top_k = top_k

    def search(self, query: str, *, top_k: int) -> dict:
        if query == self.primary_query and top_k == self.top_k:
            return self.primary_search
        return self.client.search(query, top_k=top_k)

    def fetch(self, pmids):
        return self.client.fetch(pmids)


@dataclass(frozen=True)
class QueryRelaxationResult:
    evidence: tuple[RetrievedEvidence, ...]
    report: dict


class QueryRelaxationRetriever:
    """Use primary, core, then minimal queries, advancing only below five matches."""

    def __init__(self, client, *, fallback_reformulator: CachedClinicalQueryReformulator,
                 minimal_reformulator: CachedClinicalQueryReformulator | None = None,
                 primary_backend_id: str = "ollama/qwen3:8b"):
        self.client = client
        self.fallback_reformulator = fallback_reformulator
        self.minimal_reformulator = minimal_reformulator if minimal_reformulator is not None else (
            fallback_reformulator.with_reformulator(MinimalClinicalQueryReformulator))
        self.primary_backend_id = primary_backend_id
        self.last_report = {}

    def retrieve(self, *, question_id: str, question_text: str, primary_query: str,
                 top_k: int = 15) -> QueryRelaxationResult:
        _nonempty(question_id, "question_id")
        _nonempty(question_text, "question_text")
        _nonempty(primary_query, "primary_query")
        if type(top_k) is not int or not 1 <= top_k <= 10000:
            raise ValueError("top_k must be an integer in [1,10000]")
        self.last_report = {}
        # Reset per-invocation diagnostics before any search can fail.
        for reformulator in (self.fallback_reformulator, self.minimal_reformulator):
            reformulator.last_record = None
            reformulator.last_attempt_record = None
        primary_search = self.client.search(primary_query, top_k=top_k)
        primary_count = _match_count(primary_search)
        triggered = primary_count < RELAX_BELOW_MATCH_COUNT
        fallback_query = None
        cache_record = None
        fallback_count = None
        minimal_query, minimal_count, minimal_record = None, None, None
        selected_query, selected_search = primary_query, primary_search
        selected_backend, selected_id = self.primary_backend_id, "clinical-query-reformulator-v1"
        stage_used = "primary"
        self.last_report.update(primary_query=primary_query, primary_pubmed_match_count=primary_count,
                                fallback_triggered=triggered, minimal_fallback_triggered=False)
        if triggered:
            # Only the stem, ID and existing primary query cross this boundary.
            fallback_query = self.fallback_reformulator.reformulate(
                question_id=question_id, question_text=question_text, primary_query=primary_query,
            )
            _nonempty(fallback_query, "fallback_query")
            cache_record = self.fallback_reformulator.last_record
            if not isinstance(cache_record, dict) or cache_record.get("generated_query") != fallback_query:
                raise ValueError("Fallback reformulator must provide the exact persisted cache record")
            selected_search = self.client.search(fallback_query, top_k=top_k)
            fallback_count = _match_count(selected_search)
            selected_query = fallback_query
            selected_backend, selected_id = self.fallback_reformulator.backend_id, self.fallback_reformulator.reformulator_id
            stage_used = "fallback"
            self.last_report.update(fallback_query=fallback_query, fallback_pubmed_match_count=fallback_count,
                                    fallback_query_cache_hit=cache_record["cache_hit"])
            if fallback_count < RELAX_BELOW_MATCH_COUNT:
                self.last_report["minimal_fallback_triggered"] = True
                minimal_query = self.minimal_reformulator.reformulate(
                    question_id=question_id, question_text=question_text,
                    primary_query=primary_query, first_fallback_query=fallback_query,
                )
                _nonempty(minimal_query, "minimal_fallback_query")
                minimal_record = self.minimal_reformulator.last_record
                if not isinstance(minimal_record, dict) or minimal_record.get("generated_query") != minimal_query:
                    raise ValueError("Minimal reformulator must provide the exact persisted cache record")
                self.last_report.update(minimal_fallback_query=minimal_query,
                                        minimal_fallback_query_cache_hit=minimal_record["cache_hit"])
                selected_search = self.client.search(minimal_query, top_k=top_k)
                minimal_count = _match_count(selected_search)
                selected_query = minimal_query
                selected_backend, selected_id = self.minimal_reformulator.backend_id, self.minimal_reformulator.reformulator_id
                stage_used = "minimal_fallback"
                self.last_report["minimal_fallback_pubmed_match_count"] = minimal_count
        selected = _ExactQuery(selected_query, selected_backend, selected_id)
        retriever = PubMedRetriever(
            _SearchSnapshot(self.client, selected_query, selected_search, top_k),
            query_mode="llm", query_reformulator=selected,
        )
        evidence = retriever.retrieve(
            SimpleNamespace(question_id=question_id, question=question_text), top_k=top_k,
        )
        report = dict(retriever.reports[-1])
        selected_count = _match_count({"count": report["pubmed_total_match_count"]})
        provenance = {
            "primary_query": primary_query, "fallback_query": fallback_query,
            "query_used": selected.query,
            "query_stage_used": stage_used,
            "primary_pubmed_match_count": primary_count,
            "fallback_pubmed_match_count": fallback_count,
            "fallback_triggered": triggered,
            "relax_below_match_count": RELAX_BELOW_MATCH_COUNT,
            "fallback_query_cache_hit": cache_record["cache_hit"] if triggered else None,
            "fallback_cache_key": cache_record["cache_key"] if triggered else None,
            "minimal_fallback_triggered": triggered and fallback_count < RELAX_BELOW_MATCH_COUNT,
            "minimal_fallback_query": minimal_query,
            "minimal_fallback_pubmed_match_count": minimal_count,
            "minimal_fallback_query_cache_hit": minimal_record["cache_hit"] if minimal_record else None,
            "minimal_fallback_cache_key": minimal_record["cache_key"] if minimal_record else None,
        }
        report.update(provenance)
        report["pubmed_total_match_count"] = selected_count
        report["saved_article_count"] = len(evidence)
        report["fallback_cache_record"] = cache_record
        report["minimal_fallback_cache_record"] = minimal_record
        report["query_generation"] = {
            "fallback": cache_record["query_generation"] if cache_record else None,
            "minimal_fallback": minimal_record["query_generation"] if minimal_record else None,
        }
        self.last_report = dict(report)
        return QueryRelaxationResult(
            tuple(replace(item, metadata={**item.metadata, **provenance}) for item in evidence),
            report,
        )


def _validation_inputs(questions_path: Path, primary_path: Path, question_ids):
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("Select distinct question IDs")
    artifact = json.loads(primary_path.read_text(encoding="utf-8"))
    identity = artifact["inspection_identity"]
    if (
        artifact.get("status") != "complete" or artifact.get("model") != "qwen3:8b"
        or identity["generation_parameters"] != GENERATION_SETTINGS
        or artifact.get("backend_id") != "ollama/qwen3:8b"
    ):
        raise ValueError("Expected complete cached qwen3:8b thinking-ON primary queries")
    rows = artifact["results"]
    primary = {}
    for row in rows:
        qid = _nonempty(row["question_id"], "question_id")
        if qid in primary:
            raise ValueError("Duplicate cached primary question ID")
        primary[qid] = _nonempty(row["generated_query"], "generated_query")
    stems = {}
    with questions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            qid = raw["id"]  # No options, answers, indices or metadata are accessed.
            if qid not in question_ids:
                continue
            if qid in stems:
                raise ValueError("Duplicate selected question ID")
            stem = _nonempty(raw["question"], "question_text")
            if hashlib.sha256(stem.encode("utf-8")).hexdigest() != identity["question_stem_sha256"][qid]:
                raise ValueError("Question stem differs from the cached primary-query input")
            stems[qid] = stem
    if set(stems) != set(question_ids) or not set(question_ids) <= set(primary):
        raise ValueError("Selected question is missing its stem or cached primary query")
    return artifact, [(qid, stems[qid], primary[qid]) for qid in question_ids]


class _VerifiedBackend:
    """Check the installed digest before each generation attempt."""

    def __init__(self, backend: OllamaTextGenerationBackend, expected_digest: str):
        self.backend = backend
        self.expected_digest = _nonempty(expected_digest, "model_digest")
        self.calls = 0

    def generate(self, prompt: str, *, generation_config=None) -> str:
        with urlopen(self.backend.base_url + "/api/tags", timeout=15) as response:
            models = json.loads(response.read().decode("utf-8"))["models"]
        digest = next((model.get("digest") for model in models
                       if model.get("name") == self.backend.model or model.get("model") == self.backend.model), None)
        if digest != self.expected_digest:
            raise ValueError("Installed Ollama model digest differs from cached primary-query model")
        self.calls += 1
        return self.backend.generate(prompt, generation_config=generation_config)


def _save(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate cached PubMed core-query relaxation")
    parser.add_argument("--questions", type=Path, default=Path("data/processed/medqa_us_dev_50.jsonl"))
    parser.add_argument("--primary-queries", type=Path, default=Path("outputs/query_reformulation/medqa_dev_10_thinking_on.json"))
    parser.add_argument("--question-ids", nargs="+", default=VALIDATION_QUESTION_IDS)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/query_relaxation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/query_relaxation/cache"))
    parser.add_argument("--pubmed-cache-dir", type=Path, default=Path("data/cache/pubmed"))
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 10000:
        parser.error("--top-k must be in [1,10000]")
    paths = {suffix: args.output_dir / (f"medqa_dev_{len(args.question_ids)}" + suffix)
             for suffix in (".jsonl", ".report.json", ".report.md")}
    if any(path.exists() for path in paths.values()):
        parser.error("Validation outputs already exist; use a separate --output-dir to reuse the query cache")
    artifact, inputs = _validation_inputs(args.questions, args.primary_queries, args.question_ids)
    ollama = OllamaTextGenerationBackend(
        base_url=args.base_url, model="qwen3:8b", think=True, seed=42, timeout=args.timeout,
    )
    backend = _VerifiedBackend(ollama, artifact["model_digest"])
    fallback = CachedClinicalQueryReformulator(
        backend, backend_id=ollama.backend_id, cache_dir=args.cache_dir,
        model_digest=artifact["model_digest"], generation_settings=GENERATION_SETTINGS,
        reformulator_factory=CoreClinicalQueryReformulator,
    )
    client = PubMedClient(
        args.pubmed_cache_dir, email=os.environ.get("NCBI_EMAIL"), api_key=os.environ.get("NCBI_API_KEY"),
    )
    stage = QueryRelaxationRetriever(client, fallback_reformulator=fallback)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "in_progress", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_ids": list(args.question_ids), "top_k": args.top_k,
        "primary_queries_file": str(args.primary_queries),
        "primary_queries_file_sha256": hashlib.sha256(args.primary_queries.read_bytes()).hexdigest(),
        "fallback_cache_dir": str(args.cache_dir), "backend_id": ollama.backend_id,
        "model_digest": artifact["model_digest"], "generation_parameters": GENERATION_SETTINGS,
        "model_calls_this_run": 0, "pubmed_http_requests_this_run": 0,
        "primary_queries_regenerated": 0, "completed_question_count": 0,
        "total_saved_articles": 0, "questions": [],
    }
    records = []

    def checkpoint():
        report["model_calls_this_run"] = backend.calls
        report["pubmed_http_requests_this_run"] = client.request_count
        temporary = paths[".jsonl"].with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(paths[".jsonl"])
        _save(paths[".report.json"], report)

    for qid, stem, primary in inputs:
        print(f"{qid}: checking cached primary query first...", flush=True)
        calls_before = backend.calls
        try:
            result = stage.retrieve(question_id=qid, question_text=stem, primary_query=primary, top_k=args.top_k)
        except Exception as exc:
            report.update(status="failed", failed_question_id=qid,
                          error={"type": type(exc).__name__},
                          failed_question_report={**stage.last_report, "query_generation": {
                              "fallback": stage.fallback_reformulator.last_attempt_record,
                              "minimal_fallback": stage.minimal_reformulator.last_attempt_record}})
            checkpoint()
            raise
        row = result.report
        row["primary_query_cache_hit"] = True
        row["model_calls_this_question"] = backend.calls - calls_before
        report["questions"].append(row)
        records.extend(evidence_to_json_record(item) for item in result.evidence)
        report["completed_question_count"] += 1
        report["total_saved_articles"] += len(result.evidence)
        checkpoint()
        print(f"  PRIMARY ({row['primary_pubmed_match_count']}): {row['primary_query']}", flush=True)
        print(f"  FALLBACK ({row['fallback_pubmed_match_count']}): {row['fallback_query']}", flush=True)
        print(f"  MINIMAL ({row['minimal_fallback_pubmed_match_count']}): {row['minimal_fallback_query']}", flush=True)
        print(f"  query_used={row['query_stage_used']}; saved={row['saved_article_count']}; "
              f"fallback_cache_hit={row['fallback_query_cache_hit']}", flush=True)
        for article in row["top_5"]:
            print(f"  {article['pmid']} | {article['title']}", flush=True)
    report.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint()
    lines = ["# PubMed query-relaxation validation", "",
             "Primary and first fallback prompts are unchanged. Advance to core, then minimal only while matches remain below 5.", "",
             "Match counts describe retrieval volume; no relevance or correctness claims are made.", "",
             f"Model: {ollama.backend_id}; thinking ON; temperature 0; seed 42; digest {artifact['model_digest']}.", "",
             f"Saved articles: {len(records)}; model calls this run: {backend.calls}."]
    for row in report["questions"]:
        lines.extend(["", f"## {row['question_id']}", "",
                      f"Primary ({row['primary_pubmed_match_count']} matches): `{row['primary_query']}`", "",
                      f"Fallback ({row['fallback_pubmed_match_count']} matches): `{row['fallback_query']}`", "",
                      f"Minimal ({row['minimal_fallback_pubmed_match_count']} matches): `{row['minimal_fallback_query']}`", "",
                      f"Used: {row['query_stage_used']}; saved: {row['saved_article_count']}; fallback cache hit: {row['fallback_query_cache_hit']}.", "",
                      "### Top 5 saved articles", ""])
        lines.extend(f"- PMID {article['pmid']}: {article['title']}" for article in row["top_5"])
        if not row["top_5"]:
            lines.append("No saved articles.")
        lines.extend(["", "Missing abstracts: " + (", ".join(row["missing_abstract_pmids"]) or "none"),
                      "", "Missing records: " + (", ".join(row["missing_pmids"]) or "none")])
    paths[".report.md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
