"""Generation-free uncertainty evaluation for frozen stance samples.

This module restores a project-native checkpoint into a temporary directory,
validates it with :mod:`stance_self_consistency`, and only then opens the
AI-assisted adjudicated reference labels for evaluation.  It never invokes a
model backend and never writes to the restored checkpoint state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .aggregation import MajorityVoteAggregator, QualityWeightedVoteAggregator
from .models import CandidateClaim, MedicalQuestion, ScoredEvidence
from .stance_self_consistency import (
    CACHE_REL,
    HARD_REL,
    INPUT_FIELDS,
    LABELS,
    OUTPUT_REL,
    PROMPT_FIELDS,
    REFERENCE_REL,
    REFERENCE_SHA256,
    SEEDS,
    auprc,
    auroc,
    frozen_settings,
    pair_key,
    restore_checkpoint,
    scan_samples,
    sha_bytes,
    summarize_status,
    uncertainty,
)
from .uncertainty_aggregation import QualityUncertaintyWeightedAggregator


PARTIAL_DIAGNOSTIC = "PARTIAL_DIAGNOSTIC"
FINAL_EVALUATION = "FINAL_EVALUATION"
MODES = (PARTIAL_DIAGNOSTIC, FINAL_EVALUATION)
UNCERTAINTY_FIELDS = ("u_3", "u_directional")
COVERAGES = (0.25, 0.50, 0.75, 1.00)
ARTIFACT_NAMES = (
    "sample_completeness.csv",
    "pair_uncertainty.csv",
    "uncertainty_evaluation.json",
    "risk_coverage.csv",
    "analysis_manifest.json",
    "evaluation_summary.md",
)
REFERENCE_PROVENANCE = (
    "AI-assisted adjudicated stance reference set; not independent clinician annotation"
)


class FinalEvaluationGateError(ValueError):
    """The frozen 600-sample completeness requirements were not met."""


@dataclass(frozen=True)
class AggregationMethod:
    name: str
    description: str
    aggregator: object


def aggregation_methods() -> tuple[AggregationMethod, ...]:
    """Return the three frozen methods behind one EvidenceAggregator interface."""
    return (
        AggregationMethod(
            "unweighted_hard_stance_majority",
            "One directional vote per hard stance; IRRELEVANT contributes no direction.",
            MajorityVoteAggregator(),
        ),
        AggregationMethod(
            "quality_weighted_hard_stance",
            "Sum frozen evidence-quality scores by hard directional stance.",
            QualityWeightedVoteAggregator(),
        ),
        AggregationMethod(
            "quality_uncertainty_weighted_soft_stance",
            "Use frozen q*(1-H3) weights with soft SUPPORT-minus-CONTRADICT direction.",
            QualityUncertaintyWeightedAggregator(),
        ),
    )


def aggregate_with_shared_evidence(
    question: MedicalQuestion,
    claim: CandidateClaim,
    evidence: Sequence[ScoredEvidence],
) -> dict[str, object]:
    """Evaluate all methods on the exact same ordered evidence objects."""
    shared = tuple(evidence)
    return {
        method.name: method.aggregator.aggregate(question, claim, shared)
        for method in aggregation_methods()
    }


def stable_pair_identity(key: Sequence[str]) -> str:
    return "|".join(key)


def _seed_text(seeds: Iterable[int]) -> str:
    return ";".join(str(seed) for seed in sorted(seeds))


def group_inventory(inventory: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, str], dict[int, Mapping[str, object]]]:
    """Group validated cache inventory and reject duplicate pair/seed identities."""
    groups: dict[tuple[str, str, str], dict[int, Mapping[str, object]]] = {}
    for item in inventory:
        key = pair_key(item["row"])
        seed = item["seed"]
        if seed not in SEEDS:
            raise ValueError("Seed is outside 101-110")
        group = groups.setdefault(key, {})
        if seed in group:
            raise ValueError("Duplicate pair/seed identity")
        group[seed] = item
    if any(set(group) != set(SEEDS) for group in groups.values()):
        raise ValueError("Inventory does not contain every expected pair/seed identity")
    return groups


def sample_completeness_rows(
    inventory: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], dict[int, Mapping[str, object]]]]:
    groups = group_inventory(inventory)
    rows: list[dict[str, object]] = []
    for key in sorted(groups):
        group = groups[key]
        valid = [seed for seed in SEEDS if group[seed]["status"] == "VALID"]
        failed = [seed for seed in SEEDS if group[seed]["status"] == "FAILED"]
        missing = [seed for seed in SEEDS if group[seed]["status"] == "MISSING"]
        unknown = [seed for seed in SEEDS if group[seed]["status"] not in ("VALID", "FAILED", "MISSING")]
        if unknown:
            raise ValueError("Unknown sample completion status")
        counts = Counter(
            group[seed]["entry"]["stance"] for seed in valid
        )
        if any(label not in LABELS for label in counts):
            raise ValueError("Invalid observed stance label")
        complete = len(valid) == len(SEEDS) and not failed and not missing
        rows.append({
            "stable_pair_identity": stable_pair_identity(key),
            "question_id": key[0],
            "candidate_option_id": key[1],
            "evidence_doc_id": key[2],
            "expected_seeds": _seed_text(SEEDS),
            "valid_seeds": _seed_text(valid),
            "missing_seeds": _seed_text(missing),
            "failed_seeds": _seed_text(failed),
            "valid_seed_count": len(valid),
            "missing_seed_count": len(missing),
            "failed_seed_count": len(failed),
            "observed_support_count": counts["SUPPORT"],
            "observed_contradict_count": counts["CONTRADICT"],
            "observed_irrelevant_count": counts["IRRELEVANT"],
            "completion_status": "COMPLETE" if complete else "INCOMPLETE",
        })
    return rows, groups


def validate_reference_isolation(restored_root: Path) -> dict[str, object]:
    """Verify that inference used only the frozen unlabeled/prompt allowlists."""
    audit_path = restored_root / OUTPUT_REL / "leakage_audit.json"
    checks = {
        "unlabeled_input_fields_match_allowlist": True,
        "prompt_fields_match_allowlist": True,
        "reference_opened_during_sampling": False,
        "hard_prediction_file_opened_during_sampling": False,
        "model_payload_contains_reference": False,
        "medqa_gold_or_other_options_passed": False,
        "dev_31_50_accessed": False,
    }
    source = "validated input and prompt field allowlists"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("allowed_input_fields") != list(INPUT_FIELDS):
            raise ValueError("Leakage audit input allowlist differs")
        if audit.get("prompt_fields") != list(PROMPT_FIELDS):
            raise ValueError("Leakage audit prompt allowlist differs")
        for name in tuple(checks)[2:]:
            if audit.get(name) is not False:
                raise ValueError(f"Reference-isolation check failed: {name}")
        source = "checkpoint leakage_audit.json plus validated input/prompt allowlists"
    return {"passed": True, "source": source, "checks": checks}


def _read_reference_join(project_root: Path) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, object]]:
    reference_path = project_root / REFERENCE_REL
    if sha_bytes(reference_path.read_bytes()) != REFERENCE_SHA256:
        raise ValueError("Frozen reference SHA256 differs")
    with reference_path.open(encoding="utf-8", newline="") as handle:
        references = list(csv.DictReader(handle))
    with (project_root / HARD_REL).open(encoding="utf-8", newline="") as handle:
        hard_rows = list(csv.DictReader(handle))
    if len(references) != 60 or len(hard_rows) != 60:
        raise ValueError("Reference or frozen hard-prediction count differs")
    hard_by_id = {row["pair_id"]: row for row in hard_rows}
    if len(hard_by_id) != len(hard_rows):
        raise ValueError("Duplicate frozen hard-prediction pair_id")

    joined: dict[tuple[str, str, str], dict[str, str]] = {}
    missing_reference_labels = 0
    missing_hard_labels = 0
    for reference in references:
        pair_id = reference["pair_id"]
        hard = hard_by_id.get(pair_id)
        if hard is None:
            raise ValueError("Reference pair has no frozen hard prediction")
        key = (
            reference["question_id"],
            reference["candidate_option_id"],
            reference["pmid"],
        )
        hard_key = (hard["question_id"], hard["candidate_option_id"], hard["pmid"])
        if hard_key != key or key in joined:
            raise ValueError("Reference/hard identity mismatch or duplicate pair identity")
        reference_label = reference.get("final_stance", "").strip()
        hard_label = hard.get("qwen_v1_prediction", "").strip()
        if reference_label and reference_label not in LABELS:
            raise ValueError("Invalid reference stance")
        if hard_label and hard_label not in LABELS:
            raise ValueError("Invalid frozen hard stance")
        if hard.get("reference_stance", "").strip() not in ("", reference_label):
            raise ValueError("Frozen hard table embeds a different reference stance")
        missing_reference_labels += not bool(reference_label)
        missing_hard_labels += not bool(hard_label)
        joined[key] = {
            "pair_id": pair_id,
            "batch": reference.get("batch", ""),
            "reference_stance": reference_label,
            "frozen_qwen_hard_stance": hard_label,
        }
    return joined, {
        "reference_rows": len(references),
        "hard_prediction_rows": len(hard_rows),
        "missing_reference_labels": missing_reference_labels,
        "missing_hard_labels": missing_hard_labels,
    }


def pair_uncertainty_rows(
    completeness: Sequence[Mapping[str, object]],
    groups: Mapping[tuple[str, str, str], Mapping[int, Mapping[str, object]]],
    reference_join: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in completeness:
        if item["completion_status"] != "COMPLETE":
            continue
        key = (item["question_id"], item["candidate_option_id"], item["evidence_doc_id"])
        joined = reference_join.get(key)
        labels = [groups[key][seed]["entry"]["stance"] for seed in SEEDS]
        stats = uncertainty(labels)
        reasons = []
        if joined is None:
            reasons.append("missing reference/hard-prediction identity")
            joined = {"pair_id": "", "batch": "", "reference_stance": "", "frozen_qwen_hard_stance": ""}
        if not joined["reference_stance"]:
            reasons.append("missing reference label")
        if not joined["frozen_qwen_hard_stance"]:
            reasons.append("missing frozen hard label")
        included = not reasons
        correct = (
            joined["reference_stance"] == joined["frozen_qwen_hard_stance"]
            if included else None
        )
        rows.append({
            "stable_pair_identity": item["stable_pair_identity"],
            "pair_id": joined["pair_id"],
            "batch": joined["batch"],
            "question_id": key[0],
            "candidate_option_id": key[1],
            "evidence_doc_id": key[2],
            "reference_stance": joined["reference_stance"],
            "frozen_qwen_hard_stance": joined["frozen_qwen_hard_stance"],
            "error": (not correct) if correct is not None else "",
            "evaluation_included": included,
            "evaluation_exclusion_reason": "; ".join(reasons),
            **stats,
        })
    return rows


def describe(values: Sequence[float]) -> dict[str, object]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None,
                "standard_deviation": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def metric(value: float | None, reason: str | None = None) -> dict[str, object]:
    return {"value": value, "status": "OK" if value is not None else "NA", "reason": reason}


def error_detection_metrics(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    scores = [float(row[field]) for row in rows]
    errors = [bool(row["error"]) for row in rows]
    positives = sum(errors)
    negatives = len(errors) - positives
    auc_reason = None
    if not positives:
        auc_reason = "AUROC requires at least one error (positive class)."
    elif not negatives:
        auc_reason = "AUROC requires at least one correct prediction (negative class)."
    ap_reason = "Average precision requires at least one error (positive class)." if not positives else None
    return {
        "score": field,
        "score_direction": "higher uncertainty predicts error=1",
        "positive_class": "error=1 (frozen hard stance differs from adjudicated reference stance)",
        "evaluated_pairs": len(rows),
        "error_count": positives,
        "correct_count": negatives,
        "error_prevalence_baseline": metric(positives / len(rows) if rows else None,
                                             None if rows else "No evaluable completed pairs."),
        "auroc": metric(auroc(scores, errors), auc_reason if rows else "No evaluable completed pairs."),
        "average_precision": metric(auprc(scores, errors), ap_reason if rows else "No evaluable completed pairs."),
    }


def risk_coverage_rows(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] = UNCERTAINTY_FIELDS,
    coverages: Sequence[float] = COVERAGES,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for field in fields:
        ordered = sorted(rows, key=lambda row: (float(row[field]), row["pair_id"], row["stable_pair_identity"]))
        for requested in coverages:
            if not 0 < requested <= 1:
                raise ValueError("Coverage must be in (0, 1]")
            retained = math.ceil(len(ordered) * requested) if ordered else 0
            kept = ordered[:retained]
            errors = sum(bool(row["error"]) for row in kept)
            accuracy = (retained - errors) / retained if retained else None
            output.append({
                "uncertainty_score": field,
                "requested_coverage": requested,
                "realized_coverage": retained / len(ordered) if ordered else None,
                "total_evaluable_pairs": len(ordered),
                "retained_pair_count": retained,
                "correct_count": retained - errors,
                "error_count": errors,
                "selective_accuracy": accuracy,
                "selective_risk": 1 - accuracy if accuracy is not None else None,
                "maximum_retained_uncertainty": max((float(row[field]) for row in kept), default=None),
                "tie_policy": "lower uncertainty retained first; exact ties break by stable pair identity",
            })
    return output


def evaluation_payload(
    pair_rows: Sequence[Mapping[str, object]],
    mode: str,
    status: Mapping[str, object],
    complete_pairs: int,
    incomplete_pairs: int,
) -> dict[str, object]:
    evaluable = [row for row in pair_rows if row["evaluation_included"]]
    correct = [row for row in evaluable if not row["error"]]
    incorrect = [row for row in evaluable if row["error"]]
    per_class = {}
    for label in LABELS:
        subset = [row for row in evaluable if row["reference_stance"] == label]
        per_class[label] = {
            "count": len(subset),
            "error_count": sum(bool(row["error"]) for row in subset),
            "hard_accuracy": metric(
                sum(not row["error"] for row in subset) / len(subset) if subset else None,
                None if subset else "Reference class absent from evaluable completed pairs.",
            ),
            "uncertainty": {
                field: describe([float(row[field]) for row in subset]) for field in UNCERTAINTY_FIELDS
            },
            "relevance": describe([float(row["relevance"]) for row in subset]),
        }
    risk_rows = risk_coverage_rows(evaluable)
    return {
        "mode": mode,
        "claim_status": (
            "EXPLORATORY / INCOMPLETE / NOT FOR FINAL PAPER CLAIMS"
            if mode == PARTIAL_DIAGNOSTIC
            else "FINAL EVALUATION: completeness gate passed"
        ),
        "sample_status": dict(status),
        "complete_pairs": complete_pairs,
        "incomplete_pairs": incomplete_pairs,
        "completed_pairs_with_labels": len(evaluable),
        "completed_pairs_excluded_for_missing_labels_or_join": len(pair_rows) - len(evaluable),
        "class_distribution": {label: sum(row["reference_stance"] == label for row in evaluable) for label in LABELS},
        "error_distribution": {"error": len(incorrect), "correct": len(correct)},
        "error_detection": {
            field: error_detection_metrics(evaluable, field) for field in UNCERTAINTY_FIELDS
        },
        "uncertainty_descriptive": {
            group_name: {
                field: describe([float(row[field]) for row in group_rows])
                for field in (*UNCERTAINTY_FIELDS, "relevance")
            }
            for group_name, group_rows in (
                ("all", evaluable), ("correct", correct), ("incorrect", incorrect)
            )
        },
        "per_reference_class": per_class,
        "selective_accuracy": risk_rows,
        "metric_definitions": {
            "positive_class": "error=1 when frozen Qwen-v1 hard stance != adjudicated reference stance",
            "score_direction": "higher uncertainty predicts error",
            "auroc": "Pairwise ranking probability; exact score ties receive half credit.",
            "average_precision": "Stepwise average precision with identical uncertainty scores grouped.",
            "error_prevalence_baseline": "fraction of evaluable completed pairs with error=1",
            "selective_accuracy": "accuracy after retaining the lowest-uncertainty pairs at prespecified coverage",
            "coverage_rounding": "retained count is ceil(n * requested coverage)",
            "empirical_probabilities": "ten-seed relative frequencies; not calibrated probabilities",
            "u_3": "normalized three-class entropy: -sum(p_i log p_i)/log(3)",
            "relevance": "1 - p_irrelevant",
            "u_directional": "binary entropy after conditioning on SUPPORT or CONTRADICT; 1 when directional mass is zero",
            "zero_directional_contribution": "directional_score=0 when all ten samples are IRRELEVANT",
        },
        "limitations": (
            [
                "Only fully sampled pairs are evaluated.",
                "Results do not generalize to all 60 pairs and are not for final paper claims.",
                "No thresholds or model settings are selected from this partial subset.",
            ] if mode == PARTIAL_DIAGNOSTIC else []
        ),
    }


def enforce_mode_gate(mode: str, status: Mapping[str, object], complete_pairs: int, incomplete_pairs: int) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if mode == FINAL_EVALUATION:
        required = {"expected": 600, "valid": 600, "failed": 0, "missing": 0}
        observed = {key: status.get(key) for key in required}
        if observed != required or complete_pairs != 60 or incomplete_pairs != 0:
            raise FinalEvaluationGateError(
                "FINAL_EVALUATION requires expected=600, valid=600, failed=0, missing=0 "
                "and exactly 60 complete pairs with 10 valid seeds each; "
                f"observed {observed}, complete_pairs={complete_pairs}, incomplete_pairs={incomplete_pairs}."
            )


def aggregation_readiness(unlabeled_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    group_counts = Counter((row["question_id"], row["candidate_option_id"]) for row in unlabeled_rows)
    missing_quality = any(
        "evidence_type" not in row and "quality_score" not in row
        for row in unlabeled_rows
    )
    missing = ["evidence_type_or_frozen_quality_score"] if missing_quality else []
    return {
        "methods": [
            {"name": method.name, "description": method.description}
            for method in aggregation_methods()
        ],
        "shared_interface": "aggregate_with_shared_evidence(question, claim, evidence)",
        "same_input_evidence_enforced": True,
        "ready_for_comparable_experiment": not missing,
        "question_option_group_count": len(group_counts),
        "evidence_per_question_option": dict(Counter(group_counts.values())),
        "missing_required_quality_metadata": missing,
        "precise_gap": (
            "The frozen 60-pair input projection has neither evidence_type metadata (needed by the "
            "existing frozen quality scorer) nor a frozen quality_score. "
            "It is a stance-reference subset (50 question-option groups), not a declared complete "
            "retrieval-evidence set for every answer option. Quality-weighted aggregation cannot be "
            "compared without the frozen evidence grouping and quality metadata."
            if missing else None
        ),
        "end_to_end_medqa_answer_evaluation_run": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_identity(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "project_version": "0.1.0"}


def _source_hashes(project_root: Path) -> dict[str, str]:
    relatives = (
        Path("src/quality_uncertainty_medrag/stance_uncertainty_evaluation.py"),
        Path("src/quality_uncertainty_medrag/stance_self_consistency.py"),
        Path("src/quality_uncertainty_medrag/aggregation.py"),
        Path("src/quality_uncertainty_medrag/uncertainty_aggregation.py"),
        Path("src/quality_uncertainty_medrag/quality.py"),
    )
    return {path.as_posix(): sha_bytes((project_root / path).read_bytes()) for path in relatives}


def _format_metric(value: Mapping[str, object]) -> str:
    return "NA" if value["value"] is None else f"{value['value']:.4f}"


def summary_markdown(payload: Mapping[str, object], manifest: Mapping[str, object]) -> str:
    lines = [
        "# Stance uncertainty evaluation",
        "",
        f"**{payload['claim_status']}**",
        "",
        f"Mode: `{payload['mode']}`  ",
        f"Checkpoint SHA256: `{manifest['checkpoint']['sha256']}`  ",
        f"Reference provenance: {REFERENCE_PROVENANCE}.",
        "",
        "The ten-seed values are empirical self-consistency frequencies, not calibrated probabilities. "
        "Higher uncertainty is scored as more likely to be a frozen hard-stance error.",
        "",
        "## Completeness",
        "",
        f"- Valid samples: {payload['sample_status']['valid']} / {payload['sample_status']['expected']}",
        f"- Failed samples: {payload['sample_status']['failed']}",
        f"- Missing samples: {payload['sample_status']['missing']}",
        f"- Complete pairs: {payload['complete_pairs']}",
        f"- Incomplete pairs: {payload['incomplete_pairs']}",
        f"- Evaluable completed pairs: {payload['completed_pairs_with_labels']}",
        "",
        "## Error detection",
        "",
        "| Score | AUROC | Average precision | Error prevalence |",
        "|---|---:|---:|---:|",
    ]
    for field, values in payload["error_detection"].items():
        lines.append(
            f"| {field} | {_format_metric(values['auroc'])} | "
            f"{_format_metric(values['average_precision'])} | "
            f"{_format_metric(values['error_prevalence_baseline'])} |"
        )
        for name in ("auroc", "average_precision"):
            if values[name]["status"] == "NA":
                lines.append(f"\n{name} for {field}: NA — {values[name]['reason']}")
    lines += [
        "",
        "## Reference class distribution",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for label in LABELS:
        lines.append(f"| {label} | {payload['class_distribution'][label]} |")
    if payload["limitations"]:
        lines += ["", "## Limitations", ""] + [f"- {item}" for item in payload["limitations"]]
    gap = manifest["aggregation_preparation"]["precise_gap"]
    lines += ["", "## Aggregation experiment preparation", ""]
    lines.append(gap or "Frozen grouping and quality metadata are available for a comparable experiment.")
    lines += [
        "",
        "No new inference, threshold optimization, aggregation comparison, or end-to-end MedQA "
        "answer evaluation was performed.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    output_dir: Path,
    completeness: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    payload: Mapping[str, object],
    risk_rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "sample_completeness.csv", completeness, list(completeness[0]))
    pair_fields = [
        "stable_pair_identity", "pair_id", "batch", "question_id", "candidate_option_id",
        "evidence_doc_id", "reference_stance", "frozen_qwen_hard_stance", "error",
        "evaluation_included", "evaluation_exclusion_reason", "n_support", "n_contradict",
        "n_irrelevant", "p_support", "p_contradict", "p_irrelevant", "u_3", "relevance",
        "u_directional", "directional_score",
    ]
    _write_csv(output_dir / "pair_uncertainty.csv", pair_rows, pair_fields)
    risk_fields = [
        "uncertainty_score", "requested_coverage", "realized_coverage",
        "total_evaluable_pairs", "retained_pair_count", "correct_count", "error_count",
        "selective_accuracy", "selective_risk", "maximum_retained_uncertainty", "tie_policy",
    ]
    _write_csv(output_dir / "risk_coverage.csv", risk_rows, risk_fields)
    _write_json(output_dir / "uncertainty_evaluation.json", payload)
    _write_json(output_dir / "analysis_manifest.json", manifest)
    (output_dir / "evaluation_summary.md").write_text(
        summary_markdown(payload, manifest), encoding="utf-8"
    )


def evaluate_checkpoint(
    project_root: Path,
    checkpoint: Path,
    output_dir: Path,
    mode: str,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Validate, evaluate, and write a new immutable-by-convention result directory."""
    project_root = Path(project_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_checkpoint_sha256 and checkpoint_sha != expected_checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA256 differs from the expected digest")

    with zipfile.ZipFile(checkpoint) as archive:
        checkpoint_manifest = json.loads(archive.read("checkpoint_manifest.json"))

    with tempfile.TemporaryDirectory(prefix="quality-medrag-uncertainty-eval-") as temporary:
        restored_root = Path(temporary)
        restored_status = restore_checkpoint(checkpoint, restored_root)
        inventory = scan_samples(restored_root)
        status = summarize_status(inventory)
        if restored_status != status:
            raise ValueError("Restored checkpoint status changed during validation")
        isolation = validate_reference_isolation(restored_root)
        completeness, groups = sample_completeness_rows(inventory)
        complete_pairs = sum(row["completion_status"] == "COMPLETE" for row in completeness)
        incomplete_pairs = len(completeness) - complete_pairs
        enforce_mode_gate(mode, status, complete_pairs, incomplete_pairs)

        # Reference labels are opened only after all sample/cache identities and
        # the requested partial/final completeness gate have been validated.
        reference_join, reference_status = _read_reference_join(project_root)
        pair_rows = pair_uncertainty_rows(completeness, groups, reference_join)
        payload = evaluation_payload(pair_rows, mode, status, complete_pairs, incomplete_pairs)
        risk_rows = payload["selective_accuracy"]
        unlabeled_rows = [inventory[index * len(SEEDS)]["row"] for index in range(len(completeness))]
        manifest = {
            "analysis_timestamp": checkpoint_manifest.get("timestamp"),
            "timestamp_source": "source checkpoint manifest (used for deterministic reruns)",
            "mode": mode,
            "partial_final_flag": "partial" if mode == PARTIAL_DIAGNOSTIC else "final",
            "checkpoint": {
                "filename": checkpoint.name,
                "sha256": checkpoint_sha,
                "manifest_status": checkpoint_manifest.get("status"),
                "validated_status": status,
            },
            "frozen_configuration": frozen_settings(),
            "reference": {
                "path": REFERENCE_REL.as_posix(),
                "sha256": REFERENCE_SHA256,
                "hard_prediction_path": HARD_REL.as_posix(),
                "hard_prediction_sha256": sha_bytes((project_root / HARD_REL).read_bytes()),
                "provenance": REFERENCE_PROVENANCE,
                **reference_status,
            },
            "reference_label_isolation": isolation,
            "evaluated_pair_count": payload["completed_pairs_with_labels"],
            "complete_pair_count": complete_pairs,
            "incomplete_pair_count": incomplete_pairs,
            "metric_definitions": payload["metric_definitions"],
            "aggregation_preparation": aggregation_readiness(unlabeled_rows),
            "code": {**_git_identity(project_root), "source_sha256": _source_hashes(project_root)},
            "research_integrity": {
                "new_model_inference_run": False,
                "frozen_hard_stance_modified": False,
                "reference_labels_modified": False,
                "stochastic_samples_modified": False,
                "aggregation_weights_or_formulas_modified": False,
                "dev_31_50_accessed": False,
                "thresholds_optimized": False,
            },
            "artifacts": list(ARTIFACT_NAMES),
        }
        write_artifacts(output_dir, completeness, pair_rows, payload, risk_rows, manifest)
    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha,
        "valid_samples": status["valid"],
        "complete_pairs": complete_pairs,
        "incomplete_pairs": incomplete_pairs,
        "evaluated_pairs": payload["completed_pairs_with_labels"],
        "claim_status": payload["claim_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.project_root,
        args.checkpoint,
        args.output_dir,
        args.mode,
        args.expected_checkpoint_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
