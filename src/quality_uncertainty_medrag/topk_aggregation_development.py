"""Deterministic, generation-free Top-K aggregation development inputs.

The builder reads the frozen MedQA dev 1-30 questions, existing question-level
PubMed retrieval, Qwen-v1 hard caches, and final self-consistency results.  It
never retrieves evidence or invokes a model.  Missing predictions remain
explicitly missing and block aggregation rather than receiving defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .aggregation import MajorityVoteAggregator, QualityWeightedVoteAggregator
from .models import (
    CandidateClaim,
    EvidenceType,
    QualityScore,
    RetrievedEvidence,
    ScoredEvidence,
    Stance,
    StancePrediction,
)
from .pubmed_evidence_type import evidence_type_from_publication_types
from .stance_self_consistency import (
    CLASSIFIER_VERSION,
    DIGEST,
    LABELS,
    MODEL,
    OLLAMA_VERSION,
)
from .uncertainty_aggregation import QualityUncertaintyWeightedAggregator


JOIN_SCHEMA_VERSION = "topk-aggregation-development-v1"
PARTITION = "medqa_us_dev_1_30"
TOP_K_VALUES = (3, 5)
TOP_K_RULE = (
    "For each question, stable-sort frozen retrieved records by (rank, doc_id) "
    "and retain the first K available records; reuse that identical ordered "
    "evidence set for all five candidate options."
)
QUESTION_REL = Path("data/processed/medqa_us_dev_50.jsonl")
RETRIEVAL_RELS = (
    Path("data/retrieved/pubmed_medqa_us_dev_10_llm.jsonl"),
    Path(
        "outputs/heldout_retrieval/dev_11_30/500a1f067a9f7826/runs/"
        "kaggle-20260928T142621Z/evidence.jsonl"
    ),
)
QUALITY_CONFIG_REL = Path("configs/baseline.yaml")
HARD_CACHE_REL = Path("data/cache/stance/v1")
HARD_PREDICTIONS_REL = Path(
    "outputs/stance_model_comparison/reference_60_adjudicated/predictions_60.csv"
)
UNCERTAINTY_REL = Path(
    "outputs/stance_uncertainty/evaluation/"
    "checkpoint_valid600_20261002T152105Z_553a09b9_final_evaluation_v1/"
    "pair_uncertainty.csv"
)
SOURCE_CODE_RELS = (
    Path("src/quality_uncertainty_medrag/topk_aggregation_development.py"),
    Path("src/quality_uncertainty_medrag/aggregation.py"),
    Path("src/quality_uncertainty_medrag/uncertainty_aggregation.py"),
    Path("src/quality_uncertainty_medrag/pubmed_evidence_type.py"),
    Path("src/quality_uncertainty_medrag/quality.py"),
)
OUTPUT_FILES = (
    "canonical_topk3.jsonl",
    "canonical_topk5.jsonl",
    "selected_evidence_topk3.csv",
    "selected_evidence_topk5.csv",
    "coverage.json",
    "interface_diagnostic_60.json",
    "manifest.json",
    "summary.md",
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_pair_identity(question_id: str, option_id: str, evidence_id: str) -> str:
    return f"{question_id}|{option_id}|{evidence_id}"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid frozen JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_questions(project_root: Path) -> list[dict[str, object]]:
    """Load only inference-safe question fields; do not retain the gold answer."""
    raw = read_jsonl(project_root / QUESTION_REL)
    rows = []
    for index, item in enumerate(raw[:30], 1):
        expected_id = f"medqa-us-dev-{index:06d}"
        options = item.get("options")
        if item.get("id") != expected_id or not isinstance(options, dict):
            raise ValueError("Frozen dev 1-30 question identity differs")
        if tuple(options) != ("A", "B", "C", "D", "E"):
            raise ValueError("Expected exactly the frozen A-E candidate options")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("Question text is missing")
        if any(not isinstance(text, str) or not text.strip() for text in options.values()):
            raise ValueError("Candidate option text is missing")
        rows.append({
            "question_id": item["id"],
            "question_stem": item["question"],
            "options": dict(options),
        })
    if len(rows) != 30:
        raise ValueError("Expected frozen MedQA dev 1-30")
    return rows


def load_retrieval(project_root: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen = set()
    ranks: dict[str, set[int]] = defaultdict(set)
    for relative in RETRIEVAL_RELS:
        for row in read_jsonl(project_root / relative):
            question_id = row.get("question_id")
            doc_id = row.get("doc_id")
            rank = row.get("rank")
            metadata = row.get("metadata")
            if (
                not isinstance(question_id, str)
                or not question_id.startswith("medqa-us-dev-")
                or not 1 <= int(question_id.rsplit("-", 1)[1]) <= 30
                or not isinstance(doc_id, str)
                or not doc_id.isdigit()
                or type(rank) is not int
                or rank < 1
                or not isinstance(metadata, dict)
            ):
                raise ValueError("Invalid frozen retrieval identity")
            identity = (question_id, doc_id)
            if identity in seen or rank in ranks[question_id]:
                raise ValueError("Duplicate evidence identity or retrieval rank")
            title = metadata.get("title")
            abstract = metadata.get("abstract")
            publication_types = metadata.get("publication_types")
            if (
                not isinstance(title, str)
                or not title.strip()
                or not isinstance(abstract, str)
                or not isinstance(publication_types, list)
                or not publication_types
                or any(not isinstance(value, str) for value in publication_types)
            ):
                raise ValueError("Frozen evidence metadata is incomplete or malformed")
            mapped = evidence_type_from_publication_types(publication_types).value
            if row.get("evidence_type") != mapped:
                raise ValueError("Frozen evidence_type differs from publication-type rule")
            seen.add(identity)
            ranks[question_id].add(rank)
            grouped[question_id].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["rank"], row["doc_id"]))
    if sum(map(len, grouped.values())) != 305:
        raise ValueError("Expected 305 frozen dev 1-30 evidence records")
    return dict(grouped)


def load_quality_weights(project_root: Path) -> tuple[dict[str, float], str]:
    config = yaml.safe_load((project_root / QUALITY_CONFIG_REL).read_text(encoding="utf-8"))
    values = config["quality"]["evidence_type_scores"]
    weights = {str(key): float(value) for key, value in values.items()}
    if set(weights) != {item.value for item in EvidenceType}:
        raise ValueError("Frozen quality mapping does not cover every evidence type")
    if any(not 0 <= value <= 1 for value in weights.values()):
        raise ValueError("Frozen quality weight is outside [0, 1]")
    return weights, str(config["quality"]["scorer_name"])


def load_hard_stances(project_root: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    result: dict[tuple[str, str, str], dict[str, object]] = {}
    for path in sorted((project_root / HARD_CACHE_REL).glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        metadata = item.get("model_metadata", {})
        if (
            item.get("classifier_version") != CLASSIFIER_VERSION
            or metadata.get("model") != MODEL
            or metadata.get("model_digest") != DIGEST
            or metadata.get("ollama_version") != OLLAMA_VERSION
            or metadata.get("temperature") != 0.0
            or metadata.get("seed") != 42
            or metadata.get("thinking") is not True
            or item.get("argmax_label") not in LABELS
            or path.stem != item.get("cache_key")
        ):
            raise ValueError("Hard-stance cache differs from frozen Qwen-v1 identity")
        key = (
            item["question_id"], item["candidate_option_id"], item["evidence_doc_id"]
        )
        if key in result:
            raise ValueError("Duplicate hard-stance identity")
        result[key] = {
            "label": item["argmax_label"],
            "cache_key": item["cache_key"],
            "source": path.relative_to(project_root).as_posix(),
        }
    with (project_root / HARD_PREDICTIONS_REL).open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != 60:
        raise ValueError("Expected 60 frozen comparison predictions")
    for row in predictions:
        key = (row["question_id"], row["candidate_option_id"], row["pmid"])
        label = row["qwen_v1_prediction"]
        if label not in LABELS:
            raise ValueError("Invalid frozen hard stance")
        if key in result and result[key]["label"] != label:
            raise ValueError("Hard cache conflicts with frozen prediction table")
        result.setdefault(key, {
            "label": label,
            "cache_key": None,
            "source": HARD_PREDICTIONS_REL.as_posix(),
        })
    return result


def load_uncertainty(project_root: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    with (project_root / UNCERTAINTY_REL).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise ValueError("Expected 60 final uncertainty rows")
    result = {}
    numeric_fields = (
        "p_support", "p_contradict", "p_irrelevant", "u_3", "u_directional",
        "relevance", "directional_score",
    )
    for row in rows:
        key = (row["question_id"], row["candidate_option_id"], row["evidence_doc_id"])
        if key in result or row.get("evaluation_included") != "True":
            raise ValueError("Duplicate or excluded uncertainty identity")
        values = {field: float(row[field]) for field in numeric_fields}
        if abs(sum(values[field] for field in ("p_support", "p_contradict", "p_irrelevant")) - 1) > 1e-9:
            raise ValueError("Empirical stance frequencies do not sum to one")
        if sum(int(row[field]) for field in ("n_support", "n_contradict", "n_irrelevant")) != 10:
            raise ValueError("Uncertainty row does not contain ten valid samples")
        result[key] = {
            "pair_id": row["pair_id"],
            "source": UNCERTAINTY_REL.as_posix(),
            **values,
        }
    return result


@dataclass(frozen=True)
class PreparedInputs:
    questions: tuple[dict[str, object], ...]
    retrieval: Mapping[str, tuple[dict[str, object], ...]]
    quality_weights: Mapping[str, float]
    quality_scorer_name: str
    hard: Mapping[tuple[str, str, str], Mapping[str, object]]
    uncertainty: Mapping[tuple[str, str, str], Mapping[str, object]]


def prepare_inputs(project_root: Path) -> PreparedInputs:
    project_root = Path(project_root).resolve()
    weights, scorer_name = load_quality_weights(project_root)
    return PreparedInputs(
        questions=tuple(load_questions(project_root)),
        retrieval={key: tuple(value) for key, value in load_retrieval(project_root).items()},
        quality_weights=weights,
        quality_scorer_name=scorer_name,
        hard=load_hard_stances(project_root),
        uncertainty=load_uncertainty(project_root),
    )


def build_topk_records(inputs: PreparedInputs, k: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if k not in TOP_K_VALUES:
        raise ValueError("Only proposed K=3 and K=5 are supported")
    records = []
    selected_rows = []
    selected_by_question: dict[str, tuple[dict[str, object], ...]] = {}
    for question in inputs.questions:
        question_id = question["question_id"]
        selected = tuple(inputs.retrieval.get(question_id, ()))[:k]
        selected_by_question[question_id] = selected
        for evidence in selected:
            selected_rows.append({
                "question_id": question_id,
                "evidence_identity": f"{question_id}|{evidence['doc_id']}",
                "evidence_doc_id": evidence["doc_id"],
                "pmid": evidence["doc_id"],
                "retrieval_rank": evidence["rank"],
                "evidence_type": evidence["evidence_type"],
                "quality_weight": inputs.quality_weights[evidence["evidence_type"]],
            })
        for option_id, option_text in question["options"].items():
            for evidence in selected:
                metadata = evidence["metadata"]
                key = (question_id, option_id, evidence["doc_id"])
                identity = stable_pair_identity(*key)
                hard = inputs.hard.get(key)
                uncertainty = inputs.uncertainty.get(key)
                records.append({
                    "schema_version": JOIN_SCHEMA_VERSION,
                    "partition": PARTITION,
                    "top_k": k,
                    "pair_identity": identity,
                    "question_id": question_id,
                    "question_stem": question["question_stem"],
                    "candidate_option_id": option_id,
                    "candidate_option_text": option_text,
                    "evidence_doc_id": evidence["doc_id"],
                    "pmid": evidence["doc_id"],
                    "retrieval_rank": evidence["rank"],
                    "evidence_title": metadata["title"],
                    "evidence_abstract": metadata["abstract"],
                    "publication_types": list(metadata["publication_types"]),
                    "evidence_type": evidence["evidence_type"],
                    "quality_weight": inputs.quality_weights[evidence["evidence_type"]],
                    "hard_stance_identity": identity,
                    "hard_stance_status": "AVAILABLE" if hard else "MISSING",
                    "hard_stance": hard["label"] if hard else None,
                    "hard_stance_cache_key": hard["cache_key"] if hard else None,
                    "hard_stance_source": hard["source"] if hard else None,
                    "stochastic_uncertainty_identity": identity,
                    "stochastic_uncertainty_status": "AVAILABLE" if uncertainty else "MISSING",
                    "uncertainty_pair_id": uncertainty["pair_id"] if uncertainty else None,
                    "uncertainty_source": uncertainty["source"] if uncertainty else None,
                    "p_support": uncertainty["p_support"] if uncertainty else None,
                    "p_contradict": uncertainty["p_contradict"] if uncertainty else None,
                    "p_irrelevant": uncertainty["p_irrelevant"] if uncertainty else None,
                    "u_3": uncertainty["u_3"] if uncertainty else None,
                    "u_directional": uncertainty["u_directional"] if uncertainty else None,
                    "relevance_probability": uncertainty["relevance"] if uncertainty else None,
                    "directional_score": uncertainty["directional_score"] if uncertainty else None,
                })

    if len(records) != len({row["pair_identity"] for row in records}):
        raise ValueError("Canonical join produced duplicate pair identities")
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[(row["question_id"], row["candidate_option_id"])].append(row)
    nonempty_questions = [qid for qid, evidence in selected_by_question.items() if evidence]
    zero_questions = [qid for qid, evidence in selected_by_question.items() if not evidence]
    fewer = [qid for qid, evidence in selected_by_question.items() if 0 < len(evidence) < k]
    all_zero_weight_questions = [
        qid for qid, evidence in selected_by_question.items()
        if evidence and all(
            inputs.quality_weights[item["evidence_type"]] == 0 for item in evidence
        )
    ]
    evidence_count_distribution = dict(sorted(Counter(
        len(evidence) for evidence in selected_by_question.values()
    ).items()))
    hard_matches = sum(row["hard_stance_status"] == "AVAILABLE" for row in records)
    uncertainty_matches = sum(
        row["stochastic_uncertainty_status"] == "AVAILABLE" for row in records
    )
    hard_complete_groups = sum(
        all(row["hard_stance_status"] == "AVAILABLE" for row in rows)
        for rows in groups.values()
    )
    uncertainty_complete_groups = sum(
        all(row["stochastic_uncertainty_status"] == "AVAILABLE" for row in rows)
        for rows in groups.values()
    )
    fully_ready_questions = []
    for question in inputs.questions:
        qid = question["question_id"]
        if not selected_by_question[qid]:
            continue
        option_groups = [groups[(qid, option)] for option in question["options"]]
        if all(
            all(
                row["hard_stance_status"] == "AVAILABLE"
                and row["stochastic_uncertainty_status"] == "AVAILABLE"
                for row in rows
            )
            for rows in option_groups
        ):
            fully_ready_questions.append(qid)
    coverage = {
        "top_k": k,
        "setting_status": "PROPOSED / NOT PREVIOUSLY VALIDATED",
        "selection_rule": TOP_K_RULE,
        "questions_total": len(inputs.questions),
        "questions_with_nonempty_evidence": len(nonempty_questions),
        "questions_with_zero_evidence": len(zero_questions),
        "zero_evidence_question_ids": zero_questions,
        "questions_with_fewer_than_k_nonzero_evidence": len(fewer),
        "fewer_than_k_question_ids": fewer,
        "evidence_count_per_question_distribution": evidence_count_distribution,
        "structurally_complete_five_option_questions": len(nonempty_questions),
        "structurally_complete_five_option_question_ids": nonempty_questions,
        "selected_question_ids": [row["question_id"] for row in inputs.questions],
        "selected_evidence_records": len(selected_rows),
        "total_evidence_option_pairs": len(records),
        "selected_evidence_with_missing_pmid": 0,
        "selected_evidence_with_missing_title": 0,
        "selected_evidence_with_missing_publication_types": 0,
        "selected_evidence_with_zero_quality_weight": sum(
            row["quality_weight"] == 0 for row in selected_rows
        ),
        "evidence_option_pairs_with_zero_quality_weight": sum(
            row["quality_weight"] == 0 for row in records
        ),
        "all_selected_quality_weights_zero_questions": len(all_zero_weight_questions),
        "all_selected_quality_weights_zero_question_ids": all_zero_weight_questions,
        "existing_hard_stance_matches": hard_matches,
        "missing_deterministic_hard_calls": len(records) - hard_matches,
        "existing_uncertainty_matches": uncertainty_matches,
        "missing_uncertainty_pairs": len(records) - uncertainty_matches,
        "missing_10_seed_stochastic_calls": 10 * (len(records) - uncertainty_matches),
        "hard_complete_option_groups": hard_complete_groups,
        "uncertainty_complete_option_groups": uncertainty_complete_groups,
        "fully_inference_complete_five_option_questions": len(fully_ready_questions),
        "fully_inference_complete_question_ids": fully_ready_questions,
        "selected_evidence_with_missing_abstract": sum(
            not evidence["metadata"]["abstract"]
            for selected in selected_by_question.values() for evidence in selected
        ),
        "aggregation_policy": {
            "missing_hard_stance": "BLOCK methods A and B; no default label",
            "missing_stochastic_frequencies": "BLOCK method C; no default distribution or uncertainty",
            "zero_evidence": "existing aggregators abstain, but no answer-option tie policy exists",
        },
    }
    return records, selected_rows, coverage


@dataclass(frozen=True)
class _QuestionContext:
    """Gold-free context accepted by the current aggregators, which do not read it."""
    question_id: str
    question: str
    option_labels: tuple[str, ...]
    options: tuple[str, ...]


def _retrieved_evidence(row: Mapping[str, object]) -> RetrievedEvidence:
    return RetrievedEvidence(
        schema_version=str(row["schema_version"]),
        question_id=row["question_id"],
        doc_id=row["doc_id"],
        rank=row["rank"],
        text=row["text"],
        source=row["source"],
        evidence_type=EvidenceType(row["evidence_type"]),
        retrieval_score=float(row["retrieval_score"]),
        metadata=row["metadata"],
    )


def _hard_prediction(label: str) -> StancePrediction:
    return StancePrediction(
        probabilities={stance: float(stance.value == label) for stance in Stance},
        classifier_name="frozen-qwen-v1-hard-one-hot-adapter",
    )


def _soft_prediction(item: Mapping[str, object]) -> StancePrediction:
    return StancePrediction(
        probabilities={
            Stance.SUPPORT: item["p_support"],
            Stance.CONTRADICT: item["p_contradict"],
            Stance.IRRELEVANT: item["p_irrelevant"],
        },
        classifier_name="qwen-v1-10-seed-empirical-frequency-adapter",
    )


def run_interface_diagnostic(inputs: PreparedInputs) -> dict[str, object]:
    retrieval_by_key = {
        (row["question_id"], row["doc_id"]): row
        for rows in inputs.retrieval.values() for row in rows
    }
    questions = {row["question_id"]: row for row in inputs.questions}
    groups: dict[tuple[str, str], list[tuple[tuple[str, str, str], Mapping[str, object]]]] = defaultdict(list)
    for key, uncertainty in inputs.uncertainty.items():
        if key[0] in questions and (key[0], key[2]) in retrieval_by_key and key in inputs.hard:
            groups[key[:2]].append((key, uncertainty))
    if len(groups) != 50 or sum(map(len, groups.values())) != 60:
        raise ValueError("Expected the frozen 60-row common diagnostic subset")

    majority = MajorityVoteAggregator()
    quality_weighted = QualityWeightedVoteAggregator()
    soft = QualityUncertaintyWeightedAggregator()
    output_groups = []
    hard_soft_argmax_disagreements = 0
    all_zero_groups = 0
    for (question_id, option_id), items in sorted(groups.items()):
        question = questions[question_id]
        option_index = tuple(question["options"]).index(option_id)
        claim = CandidateClaim(
            question_id=question_id,
            option_index=option_index,
            option_label=option_id,
            option_text=question["options"][option_id],
        )
        context = _QuestionContext(
            question_id=question_id,
            question=question["question_stem"],
            option_labels=tuple(question["options"]),
            options=tuple(question["options"].values()),
        )
        hard_inputs = []
        soft_inputs = []
        evidence_ids = []
        quality_values = []
        for key, uncertainty in sorted(
            items,
            key=lambda item: (
                retrieval_by_key[(item[0][0], item[0][2])]["rank"], item[0][2]
            ),
        ):
            raw_evidence = retrieval_by_key[(key[0], key[2])]
            evidence = _retrieved_evidence(raw_evidence)
            quality = QualityScore(
                value=inputs.quality_weights[evidence.evidence_type.value],
                scorer_name=inputs.quality_scorer_name,
                components={"hierarchy": inputs.quality_weights[evidence.evidence_type.value]},
            )
            hard_prediction = _hard_prediction(inputs.hard[key]["label"])
            soft_prediction = _soft_prediction(uncertainty)
            hard_soft_argmax_disagreements += hard_prediction.label != soft_prediction.label
            hard_inputs.append(ScoredEvidence(evidence, quality, hard_prediction))
            soft_inputs.append(ScoredEvidence(evidence, quality, soft_prediction))
            evidence_ids.append(evidence.doc_id)
            quality_values.append(quality.value)
        if [item.evidence for item in hard_inputs] != [item.evidence for item in soft_inputs]:
            raise ValueError("Aggregators did not receive identical evidence identities")
        if [item.quality for item in hard_inputs] != [item.quality for item in soft_inputs]:
            raise ValueError("Aggregators did not receive identical quality inputs")
        result_a = majority.aggregate(context, claim, tuple(hard_inputs))
        result_b = quality_weighted.aggregate(context, claim, tuple(hard_inputs))
        result_c = soft.aggregate(context, claim, tuple(soft_inputs))
        all_zero = all(value == 0 for value in quality_values)
        all_zero_groups += all_zero
        if all_zero and (
            result_b.decision.value != "ABSTAIN"
            or result_c.decision.value != "ABSTAIN"
            or result_c.aggregate_score is not None
        ):
            raise ValueError("Zero-quality edge behavior differs from frozen formulas")
        output_groups.append({
            "question_id": question_id,
            "candidate_option_id": option_id,
            "evidence_doc_ids": evidence_ids,
            "evidence_count": len(evidence_ids),
            "all_quality_weights_zero": all_zero,
            "method_a_decision": result_a.decision.value,
            "method_b_decision": result_b.decision.value,
            "method_c_decision": result_c.decision.value,
            "method_c_score": result_c.aggregate_score,
        })

    empty_claim = CandidateClaim("empty", 0, "A", "placeholder-not-model-input")
    empty_context = _QuestionContext("empty", "placeholder-not-model-input", ("A", "B"), ("a", "b"))
    empty_a = majority.aggregate(empty_context, empty_claim, ())
    empty_b = quality_weighted.aggregate(empty_context, empty_claim, ())
    empty_c = soft.aggregate(empty_context, empty_claim, ())
    complete_questions = []
    for question_id, question in questions.items():
        if all((question_id, option_id) in groups for option_id in question["options"]):
            complete_questions.append(question_id)
    return {
        "status": "GENERATION-FREE INTERFACE CHECK ONLY / NOT ANSWER-SELECTION PERFORMANCE",
        "evidence_option_rows": 60,
        "question_option_groups": 50,
        "questions": len({key[0] for key in groups}),
        "complete_five_option_questions": complete_questions,
        "evidence_per_option_distribution": dict(sorted(Counter(map(len, groups.values())).items())),
        "identical_ordered_evidence_for_all_methods": True,
        "identical_quality_for_all_methods": True,
        "stance_adapters": {
            "methods_a_b": "one-hot adapter of the frozen Qwen-v1 hard stance",
            "method_c": "ten-seed empirical stance frequencies; not calibrated probabilities",
        },
        "hard_vs_soft_argmax_disagreements": hard_soft_argmax_disagreements,
        "all_zero_quality_groups": all_zero_groups,
        "all_zero_quality_behavior": "A ignores quality; B abstains; C abstains with aggregate_score=null",
        "zero_evidence_behavior": {
            "method_a": empty_a.decision.value,
            "method_b": empty_b.decision.value,
            "method_c": empty_c.decision.value,
            "method_c_score": empty_c.aggregate_score,
        },
        "missing_input_behavior": {
            "hard_stance": "builder blocks A/B; no arbitrary label",
            "stochastic_frequencies": "builder blocks C; no arbitrary distribution",
        },
        "answer_option_tie_blocker": (
            "The frozen aggregators produce claim-level evidence decisions/weights but no "
            "question-level option-score tie resolver. A final-answer scoring and tie policy "
            "must be specified before MedQA accuracy evaluation."
        ),
        "reference_labels_used": False,
        "gold_answers_used": False,
        "groups": output_groups,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_selected(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "question_id", "evidence_identity", "evidence_doc_id", "pmid",
        "retrieval_rank", "evidence_type", "quality_weight",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_manifest(project_root: Path) -> dict[str, object]:
    regular = (
        QUESTION_REL, *RETRIEVAL_RELS, QUALITY_CONFIG_REL,
        HARD_PREDICTIONS_REL, UNCERTAINTY_REL, *SOURCE_CODE_RELS,
    )
    cache_files = sorted((project_root / HARD_CACHE_REL).glob("*.json"))
    return {
        "files": {
            relative.as_posix(): sha256(project_root / relative) for relative in regular
        },
        "hard_cache_directory": HARD_CACHE_REL.as_posix(),
        "hard_cache_file_count": len(cache_files),
        "hard_cache_files": {
            path.name: sha256(path) for path in cache_files
        },
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summary_markdown(coverage: Mapping[str, Mapping[str, object]], diagnostic: Mapping[str, object]) -> str:
    lines = [
        "# Top-K evidence aggregation development preparation",
        "",
        "**PROPOSED DEVELOPMENT SETTINGS — K=3 and K=5 have not been validated or tuned.**",
        "",
        "Frozen MedQA dev 1–30 only. No inference, retrieval, reference-label tuning, or answer evaluation was run.",
        "",
        "## Coverage",
        "",
        "| K | Nonempty questions | Zero-evidence questions | Selected evidence | Pair rows | Existing hard | Missing hard | Existing uncertainty | Missing uncertainty | 10-seed calls |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("3", "5"):
        row = coverage[key]
        lines.append(
            f"| {key} | {row['questions_with_nonempty_evidence']} | "
            f"{row['questions_with_zero_evidence']} | {row['selected_evidence_records']} | "
            f"{row['total_evidence_option_pairs']} | {row['existing_hard_stance_matches']} | "
            f"{row['missing_deterministic_hard_calls']} | {row['existing_uncertainty_matches']} | "
            f"{row['missing_uncertainty_pairs']} | {row['missing_10_seed_stochastic_calls']} |"
        )
    lines += [
        "",
        "Every nonempty question is structurally expanded to all five options using the same ordered evidence. No K setting currently has a five-option question with complete hard and uncertainty coverage.",
        "",
        "## Missingness and zero-quality checks",
        "",
        (
            f"K=3 contains {coverage['3']['selected_evidence_with_zero_quality_weight']}/"
            f"{coverage['3']['selected_evidence_records']} selected evidence records with zero "
            f"quality weight, affecting {coverage['3']['all_selected_quality_weights_zero_questions']} "
            "questions whose complete selected evidence set has weight zero."
        ),
        (
            f"K=5 contains {coverage['5']['selected_evidence_with_zero_quality_weight']}/"
            f"{coverage['5']['selected_evidence_records']} selected evidence records with zero "
            f"quality weight, affecting {coverage['5']['all_selected_quality_weights_zero_questions']} "
            "questions whose complete selected evidence set has weight zero."
        ),
        (
            f"The frozen records have no missing PMID, title, or publication-type values. "
            f"They retain {coverage['3']['selected_evidence_with_missing_abstract']} original empty "
            f"abstract at K=3 and {coverage['5']['selected_evidence_with_missing_abstract']} at K=5; "
            "no abstract was invented."
        ),
        "",
        "## Existing 60-row interface diagnostic",
        "",
        f"The diagnostic contains {diagnostic['question_option_groups']} option groups and {diagnostic['evidence_option_rows']} evidence-option rows. Ordered evidence identities and quality inputs were identical across all methods. It is not a full-pool answer-selection evaluation.",
        (
            f"Only questions {', '.join(diagnostic['complete_five_option_questions'])} contain all five "
            f"options in this restricted subset; {diagnostic['all_zero_quality_groups']} of its option "
            "groups have all-zero quality weights."
        ),
        "",
        diagnostic["answer_option_tie_blocker"],
        "",
    ]
    return "\n".join(lines)


def build_outputs(project_root: Path, output_dir: Path) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite derived output directory: {output_dir}")
    inputs = prepare_inputs(project_root)
    built = {}
    for k in TOP_K_VALUES:
        records, selected, coverage = build_topk_records(inputs, k)
        built[k] = {"records": records, "selected": selected, "coverage": coverage}
    diagnostic = run_interface_diagnostic(inputs)
    output_dir.mkdir(parents=True, exist_ok=False)
    for k in TOP_K_VALUES:
        _write_jsonl(output_dir / f"canonical_topk{k}.jsonl", built[k]["records"])
        _write_selected(output_dir / f"selected_evidence_topk{k}.csv", built[k]["selected"])
    coverage = {str(k): built[k]["coverage"] for k in TOP_K_VALUES}
    _write_json(output_dir / "coverage.json", coverage)
    _write_json(output_dir / "interface_diagnostic_60.json", diagnostic)
    (output_dir / "summary.md").write_text(summary_markdown(coverage, diagnostic), encoding="utf-8")
    derived_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    manifest = {
        "join_schema_version": JOIN_SCHEMA_VERSION,
        "partition": PARTITION,
        "top_k_values": list(TOP_K_VALUES),
        "top_k_selection_rule": TOP_K_RULE,
        "settings_status": "PROPOSED / NOT PREVIOUSLY VALIDATED",
        "input": input_manifest(project_root),
        "git_commit": _git_commit(project_root),
        "selected_question_ids": [row["question_id"] for row in inputs.questions],
        "selected_evidence_identities": {
            str(k): [row["evidence_identity"] for row in built[k]["selected"]]
            for k in TOP_K_VALUES
        },
        "coverage": coverage,
        "proposed_inference_workload": {
            str(k): {
                "missing_deterministic_hard_calls": built[k]["coverage"]["missing_deterministic_hard_calls"],
                "missing_uncertainty_pairs": built[k]["coverage"]["missing_uncertainty_pairs"],
                "missing_10_seed_stochastic_calls": built[k]["coverage"]["missing_10_seed_stochastic_calls"],
            }
            for k in TOP_K_VALUES
        },
        "gold_answer_isolation": {
            "gold_fields_written_to_canonical_join": False,
            "gold_answers_used_by_interface_diagnostic": False,
            "reference_labels_used_by_interface_diagnostic": False,
        },
        "missing_value_policy": "Explicit null/MISSING; never impute stance, uncertainty, abstract, or evidence.",
        "original_inputs_modified": False,
        "new_inference_run": False,
        "derived_output_sha256": derived_hashes,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "output_dir": str(output_dir),
        "join_schema_version": JOIN_SCHEMA_VERSION,
        "coverage": coverage,
        "interface_diagnostic": {
            key: diagnostic[key]
            for key in (
                "evidence_option_rows", "question_option_groups", "questions",
                "complete_five_option_questions", "all_zero_quality_groups",
                "hard_vs_soft_argmax_disagreements",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_outputs(args.project_root, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
