"""Load explicit manual queries for diagnostic PubMed retrieval experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def validate_query_overrides(overrides: Mapping[str, str]) -> dict[str, str]:
    """Copy valid question-to-query mappings without normalizing their strings."""

    if not isinstance(overrides, Mapping):
        raise ValueError("query overrides must be a JSON object mapping question IDs to queries")
    result: dict[str, str] = {}
    for question_id, query in overrides.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("query override question IDs must be nonempty strings")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query override for {question_id!r} must be a nonempty string")
        result[question_id] = query
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate query override key: {key!r}")
        result[key] = value
    return result


def load_query_overrides(path: str | Path) -> dict[str, str]:
    """Read a UTF-8 JSON object, rejecting ambiguous duplicate question IDs."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return validate_query_overrides(payload)
