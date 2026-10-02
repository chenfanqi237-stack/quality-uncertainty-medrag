"""Adapt and inspect upstream MedQA-USMLE dev questions without running RAG."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .loaders import _parse_options, _require_nonempty_string
from .models import MedicalQuestion


DEFAULT_SOURCE = Path(
    r"D:\Download\Med-RR-reference\benchmark\MedQA\data_clean\questions\US\dev.jsonl"
)


@dataclass(frozen=True)
class MalformedRecord:
    line_number: int
    message: str


@dataclass(frozen=True)
class MedQAUSLoadResult:
    questions: tuple[MedicalQuestion, ...]
    malformed_records: tuple[MalformedRecord, ...]
    source_record_count: int


def adapt_medqa_us_record(raw: object, *, line_number: int) -> MedicalQuestion:
    """Preserve one upstream record, using its physical source line as its ID."""

    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1:
        raise ValueError("line_number must be a positive integer")
    if not isinstance(raw, dict):
        raise ValueError("MedQA-USMLE record must be an object")
    try:
        question_text = _require_nonempty_string(raw["question"], "question")
        if not isinstance(raw["options"], dict):
            raise ValueError("MedQA-USMLE options must be a mapping")
        labels, options = _parse_options(raw["options"])
        answer_label = _require_nonempty_string(raw["answer_idx"], "answer_idx")
        if answer_label not in labels:
            raise ValueError("answer_idx must identify an option label")
        answer_index = labels.index(answer_label)
        answer_text = _require_nonempty_string(raw["answer"], "answer")
        if answer_text != options[answer_index]:
            raise ValueError("answer and answer_idx identify different option texts")
        return MedicalQuestion(
            question_id=f"medqa-us-dev-{line_number:06d}",
            question=question_text,
            option_labels=labels,
            options=options,
            answer_index=answer_index,
            metadata={
                "dataset": "MedQA-USMLE",
                "split": "dev",
                "source_line": line_number,
                "upstream": {
                    key: value for key, value in raw.items() if key not in {"question", "options"}
                },
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed MedQA-USMLE question: {exc}") from exc


def load_medqa_us_dev(
    path: str | Path = DEFAULT_SOURCE,
    *,
    limit: int | None = None,
    skip_malformed: bool = False,
) -> MedQAUSLoadResult:
    """Read the first N nonblank source records in file order, without shuffling.

    Malformed rows consume the same first-N window. When explicitly skipped,
    they are reported and are never replaced with later source records.
    """

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")
    source = Path(path)
    questions: list[MedicalQuestion] = []
    malformed: list[MalformedRecord] = []
    source_record_count = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and source_record_count >= limit:
                break
            source_record_count += 1
            try:
                raw = json.loads(line)
                question = adapt_medqa_us_record(raw, line_number=line_number)
            except (TypeError, ValueError) as exc:
                message = f"Invalid MedQA-USMLE record at {source}:{line_number}: {exc}"
                if not skip_malformed:
                    raise ValueError(message) from exc
                malformed.append(MalformedRecord(line_number, message))
                continue
            questions.append(question)
    return MedQAUSLoadResult(tuple(questions), tuple(malformed), source_record_count)


def write_medqa_us_subset(questions: Sequence[MedicalQuestion], path: str | Path) -> None:
    """Write the existing question schema, preserving option insertion order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            row = {
                "id": question.question_id,
                "question": question.question,
                "options": dict(zip(question.option_labels, question.options)),
                "answer": question.answer_label,
                "metadata": question.metadata,
            }
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def inspect_medqa_us(result: MedQAUSLoadResult, *, sample_size: int = 3) -> dict[str, Any]:
    """Report data validity and MCQ structure, without computing predictions."""

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size must be a nonnegative integer")
    counts = Counter(len(question.options) for question in result.questions)
    return {
        "question_count": len(result.questions),
        "option_count_distribution": {str(size): counts[size] for size in sorted(counts)},
        "candidate_claim_count": sum(len(question.candidate_claims) for question in result.questions),
        "source_record_count": result.source_record_count,
        "malformed_record_count": len(result.malformed_records),
        "skipped_record_count": len(result.malformed_records),
        "samples": [
            {"question_id": question.question_id, "gold_answer_label": question.answer_label}
            for question in result.questions[:sample_size]
        ],
        "malformed_records": [
            {"line_number": item.line_number, "message": item.message}
            for item in result.malformed_records
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or inspect upstream MedQA-USMLE dev data")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--limit", type=int, default=50, help="First N nonblank source records (default: 50)")
    parser.add_argument("--output", type=Path, help="Optional processed JSONL output; omitted for inspection")
    parser.add_argument(
        "--skip-malformed", action="store_true",
        help="Report and skip malformed records within the selected window, without backfilling",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() == args.source.resolve():
        parser.error("output must differ from the upstream source")
    try:
        result = load_medqa_us_dev(args.source, limit=args.limit, skip_malformed=args.skip_malformed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        write_medqa_us_subset(result.questions, args.output)
    print(json.dumps(inspect_medqa_us(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
