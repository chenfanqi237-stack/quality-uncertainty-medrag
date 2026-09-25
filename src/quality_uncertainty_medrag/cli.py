"""Run the synthetic baseline from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import yaml

from .aggregation import QualityWeightedVoteAggregator
from .evaluation import evaluate_pipeline
from .loaders import load_medqa_questions, load_retrieved_evidence
from .pipeline import BaselinePipeline
from .quality import ConfiguredEvidenceTypeScorer
from .reproducibility import seed_everything
from .retrieval import PrecomputedRetriever
from .stance import AnnotatedStanceClassifier


def _resolve(config_dir: Path, value: str) -> Path:
    return (config_dir / value).resolve()


def _logger(output_dir: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("quality_uncertainty_medrag")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the conflicting-evidence RAG baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")

    config_dir = config_path.parent
    question_path = _resolve(config_dir, config["paths"]["questions"])
    evidence_path = _resolve(config_dir, config["paths"]["evidence"])
    output_dir = _resolve(config_dir, config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_dir, config["logging"]["level"])
    seed = int(config["seed"])
    seed_everything(seed)
    top_k = int(config["run"]["top_k"])

    questions = load_medqa_questions(question_path)
    evidence = load_retrieved_evidence(evidence_path)
    unknown_question_ids = set(evidence) - {item.question_id for item in questions}
    if unknown_question_ids:
        raise ValueError(f"Evidence refers to unknown questions: {sorted(unknown_question_ids)}")

    scorer = ConfiguredEvidenceTypeScorer(
        config["quality"]["evidence_type_scores"], config["quality"]["scorer_name"]
    )
    pipeline = BaselinePipeline(
        retriever=PrecomputedRetriever(evidence),
        quality_scorer=scorer,
        stance_classifier=AnnotatedStanceClassifier(),
        aggregator=QualityWeightedVoteAggregator(),
        top_k=top_k,
    )
    results = [
        pipeline.run(question, claim)
        for question in questions
        for claim in question.candidate_claims
    ]
    summary = evaluate_pipeline(questions, results, k=top_k)

    effective_config = {
        **config,
        "paths": {
            "questions": str(question_path),
            "evidence": str(evidence_path),
            "output_dir": str(output_dir),
        },
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, sort_keys=False)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            row = {
                "question_id": result.question.question_id,
                "option_index": result.claim.option_index,
                "option_label": result.claim.option_label,
                "option_text": result.claim.option_text,
                "decision": result.aggregation.decision.value,
                "support_weight": result.aggregation.support_weight,
                "contradict_weight": result.aggregation.contradict_weight,
                "irrelevant_count": result.aggregation.irrelevant_count,
                "evidence_count": result.aggregation.evidence_count,
                "evidence": [
                    {
                        "doc_id": item.evidence.doc_id,
                        "rank": item.evidence.rank,
                        "quality": {
                            "value": item.quality.value,
                            "components": dict(item.quality.components),
                        },
                        "stance": {
                            stance.value: probability
                            for stance, probability in item.stance.probabilities.items()
                        },
                    }
                    for item in result.evidence
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)
        handle.write("\n")

    logger.info("questions=%d seed=%d top_k=%d", len(questions), seed, top_k)
    logger.info(
        "claims=%d mrr=%.3f hit@%d=%.3f",
        summary.claim_count,
        summary.mean_reciprocal_rank,
        top_k,
        summary.hit_at_k,
    )
    logger.info("wrote outputs to %s", output_dir)
    return 0
