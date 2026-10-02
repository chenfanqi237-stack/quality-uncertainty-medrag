"""Persist exact clinical queries by model, prompt and generation settings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from .clinical_query import ClinicalQueryError, ClinicalQueryReformulator, normalize_query_output
from .interfaces import TextGenerationBackend


class QueryCacheError(ClinicalQueryError):
    """A query cache entry cannot be safely reused or persisted."""


def _json_copy(value: object) -> object:
    """Copy JSON settings without retaining caller-owned mutable containers."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("generation_settings must contain finite JSON values") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QueryCacheError("The query cache entry contains duplicate JSON fields.")
        result[key] = value
    return result


class _CachingBackend:
    """Capture the unchanged reformulator prompt at its generation boundary."""

    def __init__(self, owner: CachedClinicalQueryReformulator, question_id: str) -> None:
        self.owner = owner
        self.question_id = question_id
        self.record: dict[str, object] | None = None

    def generate(self, prompt: str, *, generation_config=None) -> str:
        effective_settings = dict(self.owner._generation_settings)
        effective_settings.update(generation_config or {})
        effective_settings = _json_copy(effective_settings)
        context = {
            "schema_version": 1,
            "question_id": self.question_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "backend_id": self.owner.backend_id,
            "reformulator_id": self.owner.reformulator_id,
            "model_digest": self.owner._model_digest,
            "generation_settings": effective_settings,
        }
        cache_key = hashlib.sha256(_canonical_json(context).encode("utf-8")).hexdigest()
        context["cache_key"] = cache_key
        path = self.owner._cache_dir / f"{cache_key}.json"
        if path.exists():
            entry = self.owner._read_entry(path, context)
            self.record = dict(entry, cache_hit=True)
            return entry["generated_query"]

        # Defaults describe the selected backend; the existing reformulator's
        # per-call config is forwarded unchanged, rather than inventing options.
        attempts = []
        audit_path = self.owner._cache_dir / "attempts" / f"{cache_key}-{uuid.uuid4().hex}.json"
        for attempt in range(1, self.owner._max_generation_retries + 2):
            stage = "generation"
            try:
                raw = self.owner._backend.generate(
                    prompt, generation_config=_json_copy(generation_config),
                )
                stage = "output_validation"
                query = self.owner._normalize_output(raw)
            except Exception as exc:
                # Exception messages and raw responses can contain reasoning or
                # provider payloads. Persist classes/stages only, never content.
                chain, cause = [], exc
                while cause is not None and len(chain) < 8:
                    chain.append(type(cause).__name__)
                    cause = cause.__cause__
                attempts.append({"attempt_index": attempt, "status": "failed",
                                 "failure_stage": stage, "error_types": chain,
                                 "timestamp": datetime.now(timezone.utc).isoformat()})
                self.owner._save_attempts(audit_path, context, attempts)
                if attempt > self.owner._max_generation_retries:
                    raise
            else:
                attempts.append({"attempt_index": attempt, "status": "success",
                                 "timestamp": datetime.now(timezone.utc).isoformat()})
                self.owner._save_attempts(audit_path, context, attempts)
                break
        entry = dict(
            context, generated_query=query,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        winner = self.owner._write_once(path, entry, context)
        self.record = dict(winner, cache_hit=False)
        return winner["generated_query"]


class CachedClinicalQueryReformulator:
    """Compose the existing reformulator with an exact-query disk cache.

    ``generation_settings`` records the caller-selected backend defaults,
    including settings such as seed and thinking mode. The actual per-call
    reformulator config takes precedence in the cache key. The backend must be
    configured with those defaults; this wrapper does not change backend behavior.

    Each success is stored immediately. Hits reuse the exact saved string and
    never call the model. Corrupt entries fail rather than trigger regeneration.
    Generation/invalid-output failures allow at most two additional identical
    attempts. Failure classes/stages and retry counts are persisted separately
    under attempts/, without changing existing cache keys or entry schemas.
    An optional reformulator factory reuses these storage guarantees for a
    separate prompt; the default remains the original clinical reformulator.
    The key contains the full actual prompt hash, so a changed stem, prompt,
    model/digest or settings produces a separate entry. Stored entries contain
    no prompt, stem, options, gold answer or reasoning.
    """

    def __init__(
        self,
        backend: TextGenerationBackend,
        *,
        backend_id: str,
        cache_dir: str | Path,
        generation_settings: Mapping[str, object],
        model_digest: str | None = None,
        reformulator_factory: Callable[..., ClinicalQueryReformulator] = ClinicalQueryReformulator,
        max_generation_retries: int = 2,
    ) -> None:
        identity = reformulator_factory(backend, backend_id=backend_id)
        if type(max_generation_retries) is not int or not 0 <= max_generation_retries <= 2:
            raise ValueError("max_generation_retries must be an integer in [0,2]")
        if not isinstance(generation_settings, Mapping) or any(
            not isinstance(key, str) or not key.strip() for key in generation_settings
        ):
            raise ValueError("generation_settings must be a mapping with nonempty string keys")
        if model_digest is not None and (
            not isinstance(model_digest, str) or not model_digest.strip()
        ):
            raise ValueError("model_digest must be a nonempty string or None")
        self._backend = backend
        self._reformulator_factory = reformulator_factory
        self._normalize_output = getattr(identity, "normalize_output", normalize_query_output)
        self._backend_id = identity.backend_id
        self._reformulator_id = identity.reformulator_id
        self._cache_dir = Path(cache_dir)
        self._generation_settings = _json_copy(dict(generation_settings))
        self._model_digest = model_digest
        self._max_generation_retries = max_generation_retries
        self.last_record: dict[str, object] | None = None
        self.last_attempt_record: dict[str, object] | None = None

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def reformulator_id(self) -> str:
        return self._reformulator_id

    def reformulate(self, *, question_id: str, question_text: str,
                    primary_query: str | None = None,
                    first_fallback_query: str | None = None) -> str:
        self.last_record = None
        self.last_attempt_record = None
        adapter = _CachingBackend(self, question_id)
        reformulator = self._reformulator_factory(adapter, backend_id=self.backend_id)
        inputs = {"question_id": question_id, "question_text": question_text}
        if primary_query is not None:
            inputs["primary_query"] = primary_query
        if first_fallback_query is not None:
            inputs["first_fallback_query"] = first_fallback_query
        try:
            query = reformulator.reformulate(**inputs)
        except ClinicalQueryError as exc:
            # Keep storage failures distinguishable despite the reformulator's
            # deliberate backend exception boundary.
            if isinstance(exc.__cause__, QueryCacheError):
                raise exc.__cause__ from None
            raise
        self.last_record = _json_copy(adapter.record)
        self.last_record["query_generation"] = _json_copy(self.last_attempt_record) if self.last_attempt_record else {
            "attempt_count": 0, "retry_count": 0, "attempts": [], "cache_hit": True,
        }
        return query

    def with_reformulator(self, factory):
        """Share backend/settings/storage, keeping a separate prompt cache key."""
        return type(self)(
            self._backend, backend_id=self.backend_id, cache_dir=self._cache_dir,
            generation_settings=self._generation_settings, model_digest=self._model_digest,
            reformulator_factory=factory, max_generation_retries=self._max_generation_retries,
        )

    def _save_attempts(self, path: Path, context: dict, attempts: list) -> None:
        record = {**context, "attempt_count": len(attempts),
                  "retry_count": len(attempts) - 1, "max_retries": self._max_generation_retries,
                  "attempts": _json_copy(attempts),
                  "attempt_log": path.relative_to(self._cache_dir).as_posix()}
        self.last_attempt_record = _json_copy(record)
        temporary_path = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(_canonical_json(record) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except OSError as exc:
            raise QueryCacheError("Query-generation attempt diagnostics could not be saved.") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _read_entry(self, path: Path, context: dict[str, object]) -> dict[str, object]:
        try:
            entry = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise QueryCacheError("The saved query cache entry is unreadable or invalid JSON.") from exc
        if not isinstance(entry, dict) or set(entry) != set(context) | {
            "generated_query", "created_at",
        }:
            raise QueryCacheError("The saved query cache entry has an invalid schema.")
        if any(entry.get(key) != value for key, value in context.items()):
            raise QueryCacheError("The saved query cache entry does not match its request context.")
        if not isinstance(entry["created_at"], str) or not entry["created_at"]:
            raise QueryCacheError("The saved query cache entry is missing its creation timestamp.")
        try:
            canonical_query = self._normalize_output(entry["generated_query"])
        except ClinicalQueryError as exc:
            raise QueryCacheError("The saved query cache entry has an invalid query.") from exc
        if canonical_query != entry["generated_query"]:
            raise QueryCacheError("The saved query is not the exact canonical reformulator output.")
        return entry

    def _write_once(
        self, path: Path, entry: dict[str, object], context: dict[str, object],
    ) -> dict[str, object]:
        temporary_path = None
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._cache_dir,
                prefix=".query-", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(_canonical_json(entry) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                # Linking a complete temporary entry is atomic and cannot
                # overwrite an entry produced by another caller.
                os.link(temporary_path, path)
            except FileExistsError:
                return self._read_entry(path, context)
            return entry
        except OSError as exc:
            raise QueryCacheError("The generated query could not be saved in the query cache.") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
