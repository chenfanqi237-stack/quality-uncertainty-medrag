"""Cached PubMed E-utilities retrieval, independent of stance and answer models."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .clinical_query import ClinicalQueryReformulator
from .loaders import load_medqa_questions
from .models import MedicalQuestion, RetrievedEvidence
from .pubmed_query import build_pubmed_query
from .pubmed_query_overrides import load_query_overrides, validate_query_overrides
from .pubmed_records import evidence_type_from_publication_types, parse_pubmed_xml


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
QUERY_VERSION = "question-clinical-concepts-v2"
ARTICLE_CACHE_VERSION = 2


class PubMedError(RuntimeError):
    """A request or response failed; no replacement evidence is manufactured."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _valid_search_result(value: object) -> bool:
    return (
        isinstance(value, dict) and "error" not in value
        and isinstance(value.get("idlist"), list)
        and all(isinstance(pmid, str) and re.fullmatch(r"[0-9]+", pmid) for pmid in value["idlist"])
    )


def _valid_article_record(value: object, pmid: str) -> bool:
    return (
        isinstance(value, dict) and value.get("pmid") == pmid
        and all(isinstance(value.get(field), str) for field in ("title", "abstract", "journal"))
        and "publication_date" in value
        and (value["publication_date"] is None or isinstance(value["publication_date"], str))
        and isinstance(value.get("publication_types"), list)
        and all(isinstance(kind, str) for kind in value["publication_types"])
        and isinstance(value.get("pubmed_metadata"), dict)
    )


class PubMedClient:
    """A synchronous, paced client with query snapshots and per-PMID caching."""

    def __init__(
        self, cache_dir: str | Path, *, email: str | None = None, api_key: str | None = None,
        request_interval: float = 0.4, max_retries: int = 3, timeout: float = 30,
        opener: Callable[..., Any] | None = None, sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not math.isfinite(request_interval) or request_interval < 1 / 3:
            raise ValueError("request_interval must be at least 1/3 second")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.cache_dir = Path(cache_dir)
        self.email = email
        self.api_key = api_key
        self.request_interval = request_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._opener = opener if opener is not None else urlopen
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        self._last_request: float | None = None
        self.request_count = 0

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        delay = min(2.0 ** attempt, 30.0)
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    date = parsedate_to_datetime(retry_after)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    seconds = (date - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    seconds = 0.0
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        return delay

    def _request(self, endpoint: str, params: dict[str, str]) -> bytes:
        parameters = {**params, "tool": "quality-uncertainty-medrag"}
        if self.email:
            parameters["email"] = self.email
        if self.api_key:
            parameters["api_key"] = self.api_key
        request = Request(
            EUTILS_BASE + endpoint, data=urlencode(parameters).encode("ascii"),
            headers={"User-Agent": "quality-uncertainty-medrag/0.1", "Content-Type": "application/x-www-form-urlencoded"},
        )
        for attempt in range(self.max_retries + 1):
            now = self._clock()
            if self._last_request is not None:
                remaining = self.request_interval - (now - self._last_request)
                if remaining > 0:
                    self._sleep(remaining)
            self._last_request = self._clock()
            self.request_count += 1
            retry_after = None
            status = "network failure"
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = response.read()
                if b"api rate limit exceeded" not in body.lower():
                    return body
                status = "API rate limit exceeded"
            except HTTPError as exc:
                status = f"HTTP {exc.code}"
                if exc.code != 429 and not 500 <= exc.code < 600:
                    raise PubMedError(f"{endpoint} failed: {status}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            except (URLError, TimeoutError, OSError):
                pass
            if attempt == self.max_retries:
                raise PubMedError(f"{endpoint} failed after {attempt + 1} attempts: {status}")
            delay = self._retry_delay(attempt, retry_after)
            logging.getLogger(__name__).warning("%s: %s; retrying in %.1fs", endpoint, status, delay)
            self._sleep(delay)
        raise PubMedError(f"{endpoint} failed")

    def search(self, query: str, *, top_k: int) -> dict[str, Any]:
        _positive_integer(top_k, "top_k")
        if top_k > 10000:
            raise ValueError("PubMed ESearch supports at most 10000 results")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        identity = {"db": "pubmed", "term": query, "retmax": str(top_k), "sort": "relevance", "retmode": "json"}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        path = self.cache_dir / "searches" / f"{digest}.json"
        cached = _read_json(path)
        if isinstance(cached, dict) and _valid_search_result(cached.get("result")):
            result = cached["result"]
        else:
            body = self._request("esearch.fcgi", identity)
            try:
                payload = json.loads(body)
                result = payload["esearchresult"]
                if not _valid_search_result(result):
                    raise ValueError("invalid idlist")
                if "error" in result or "error" in payload:
                    raise ValueError("API error")
            except (ValueError, TypeError, KeyError) as exc:
                raise PubMedError("ESearch returned an invalid response") from exc
            _write_json(path, {
                "request": identity, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "result": result,
            })
        return result

    def fetch(self, pmids: Sequence[str]) -> dict[str, dict[str, Any]]:
        ordered_pmids = list(dict.fromkeys(pmids))
        if any(not isinstance(pmid, str) or not re.fullmatch(r"[0-9]+", pmid) for pmid in ordered_pmids):
            raise ValueError("PMIDs must be digit strings")
        records: dict[str, dict[str, Any]] = {}
        missing = []
        for pmid in ordered_pmids:
            cached = _read_json(self.cache_dir / "records" / f"{pmid}.json")
            if _valid_article_record(cached, pmid):
                if cached.get("parser_version") != ARTICLE_CACHE_VERSION:
                    try:
                        upgraded = parse_pubmed_xml(cached["pubmed_metadata"]["raw_xml"])[pmid]
                    except (KeyError, ValueError, TypeError):
                        missing.append(pmid)
                        continue
                    if "fetched_at_utc" in cached["pubmed_metadata"]:
                        upgraded["pubmed_metadata"]["fetched_at_utc"] = cached["pubmed_metadata"]["fetched_at_utc"]
                    upgraded["parser_version"] = ARTICLE_CACHE_VERSION
                    _write_json(self.cache_dir / "records" / f"{pmid}.json", upgraded)
                    cached = upgraded
                records[pmid] = cached
            else:
                unavailable = _read_json(self.cache_dir / "unavailable" / f"{pmid}.json")
                if (
                    isinstance(unavailable, dict) and unavailable.get("pmid") == pmid
                    and unavailable.get("parser_version") == ARTICLE_CACHE_VERSION
                ):
                    continue
                missing.append(pmid)
        for start in range(0, len(missing), 200):
            batch = missing[start:start + 200]
            body = self._request("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
            try:
                fetched = parse_pubmed_xml(body)
            except (ValueError, TypeError) as exc:
                raise PubMedError("EFetch returned an invalid response") from exc
            for pmid in batch:
                if pmid in fetched:
                    record = fetched[pmid]
                    record["parser_version"] = ARTICLE_CACHE_VERSION
                    record["pubmed_metadata"]["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
                    _write_json(self.cache_dir / "records" / f"{pmid}.json", record)
                    records[pmid] = record
                else:
                    _write_json(self.cache_dir / "unavailable" / f"{pmid}.json", {
                        "pmid": pmid, "parser_version": ARTICLE_CACHE_VERSION,
                        "reason": "Successful EFetch did not provide a supported PubmedArticle record",
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "response_xml": body.decode("utf-8"),
                    })
        return records


class PubMedRetriever:
    def __init__(
        self, client: PubMedClient, query_builder: Callable[[str], str] = build_pubmed_query, *,
        query_overrides: Mapping[str, str] | None = None,
        query_mode: str = "keyword", query_reformulator: ClinicalQueryReformulator | None = None,
    ) -> None:
        if query_mode not in ("keyword", "llm"):
            raise ValueError("query_mode must be keyword or llm")
        if query_mode == "llm" and query_reformulator is None:
            raise ValueError("LLM query mode requires a configured ClinicalQueryReformulator")
        self.client = client
        self.query_builder = query_builder
        self.query_overrides = validate_query_overrides(query_overrides) if query_overrides is not None else {}
        self.query_mode = query_mode
        self.query_reformulator = query_reformulator
        self.reports: list[dict[str, Any]] = []

    def retrieve(self, question: MedicalQuestion, *, top_k: int) -> tuple[RetrievedEvidence, ...]:
        _positive_integer(top_k, "top_k")
        reformulator_id = None
        backend_id = None
        if question.question_id in self.query_overrides:
            query = self.query_overrides[question.question_id]
            query_source = "manual_override"
            query_builder_version = "manual-query-override-v1"
            used_query_mode = "manual"
        elif self.query_mode == "llm":
            reformulator = self.query_reformulator
            if reformulator is None:
                raise ValueError("LLM query mode requires a configured ClinicalQueryReformulator")
            query = reformulator.reformulate(
                question_id=question.question_id, question_text=question.question,
            )
            query_source = "llm_reformulation"
            reformulator_id = reformulator.reformulator_id
            backend_id = reformulator.backend_id
            query_builder_version = reformulator_id
            used_query_mode = "llm"
        else:
            query = self.query_builder(question.question)
            query_source = "automatic"
            query_builder_version = QUERY_VERSION
            used_query_mode = "keyword"
        query_provenance = {
            "query_mode": used_query_mode, "requested_query_mode": self.query_mode,
            "query_reformulator_id": reformulator_id, "query_backend_id": backend_id,
        }
        search = self.client.search(query, top_k=top_k)
        pmids = list(dict.fromkeys(search["idlist"]))[:top_k]
        records = self.client.fetch(pmids)
        evidence = []
        missing_pmids = []
        for rank, pmid in enumerate(pmids, start=1):
            record = records.get(pmid)
            if record is None or not record["title"].strip():
                missing_pmids.append(pmid)
                continue
            evidence.append(RetrievedEvidence(
                schema_version="2.0", question_id=question.question_id, doc_id=pmid, rank=rank,
                text=record["title"] + ("\n\n" + record["abstract"] if record["abstract"] else ""),
                source="PubMed",
                evidence_type=evidence_type_from_publication_types(record["publication_types"]),
                retrieval_score=0.0,
                metadata={
                    "title": record["title"], "abstract": record["abstract"], "journal": record["journal"],
                    "publication_date": record["publication_date"], "publication_types": record["publication_types"],
                    "pubmed_metadata": record["pubmed_metadata"], "query": query,
                    "query_translation": search.get("querytranslation", query),
                    "query_builder": query_builder_version, "query_source": query_source,
                    **query_provenance,
                    "search_warnings": search.get("warninglist", {}),
                    "sort": "relevance", "retrieval_score_source": "not_provided_by_pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
            ))
        self.reports.append({
            "question_id": question.question_id, "query": query, "query_source": query_source,
            **query_provenance,
            "retrieved_article_count": len(evidence), "pmids": [item.doc_id for item in evidence],
            "top_5": [{"pmid": item.doc_id, "title": item.metadata["title"]} for item in evidence[:5]],
            "missing_pmids": missing_pmids,
            "missing_abstract_pmids": [item.doc_id for item in evidence if not item.metadata["abstract"]],
            "pubmed_total_match_count": search.get("count"),
            "search_warnings": search.get("warninglist", {}),
        })
        return tuple(evidence)


def evidence_to_json_record(evidence: RetrievedEvidence) -> dict[str, Any]:
    """Serialize retrieval metadata without adding fixture stance annotations."""

    return {
        "schema_version": evidence.schema_version, "question_id": evidence.question_id,
        "doc_id": evidence.doc_id, "rank": evidence.rank, "text": evidence.text,
        "source": evidence.source, "evidence_type": evidence.evidence_type.value,
        "retrieval_score": evidence.retrieval_score, "metadata": evidence.metadata,
    }


def main(
    argv: Sequence[str] | None = None, *, query_reformulator: ClinicalQueryReformulator | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Retrieve PubMed abstracts for a small MedQA dev subset")
    parser.add_argument("--questions", type=Path, default=Path("data/processed/medqa_us_dev_50.jsonl"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("data/retrieved/pubmed_medqa_us_dev_3.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/retrieved/pubmed_medqa_us_dev_3.report.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/pubmed"))
    parser.add_argument("--query-overrides", type=Path, help="Optional JSON question-ID to exact PubMed-query mapping")
    parser.add_argument(
        "--query-mode", choices=("keyword", "llm"), default="keyword",
        help="Use the existing keyword builder or an explicitly injected clinical reformulator",
    )
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    args = parser.parse_args(argv)
    if len({args.questions.resolve(), args.output.resolve(), args.report.resolve()}) != 3:
        parser.error("questions, output, and report paths must differ")
    if args.query_overrides is not None and args.query_overrides.resolve() in {args.output.resolve(), args.report.resolve()}:
        parser.error("query overrides path must differ from output and report paths")
    if args.query_mode == "llm" and query_reformulator is None:
        parser.error(
            "LLM query mode requires a configured ClinicalQueryReformulator. "
            "Inject it with main(..., query_reformulator=...) or use --query-mode keyword."
        )
    try:
        _positive_integer(args.limit, "limit")
        _positive_integer(args.top_k, "top_k")
        query_overrides = load_query_overrides(args.query_overrides) if args.query_overrides is not None else {}
        questions = load_medqa_questions(args.questions)[:args.limit]
        client = PubMedClient(args.cache_dir, email=args.email, api_key=os.environ.get("NCBI_API_KEY"))
        retriever = PubMedRetriever(
            client, query_overrides=query_overrides, query_mode=args.query_mode,
            query_reformulator=query_reformulator,
        )
        evidence = [item for question in questions for item in retriever.retrieve(question, top_k=args.top_k)]
    except (OSError, ValueError, PubMedError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in evidence:
            handle.write(json.dumps(evidence_to_json_record(item), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    _write_json(args.report, retriever.reports)
    print(json.dumps(retriever.reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
