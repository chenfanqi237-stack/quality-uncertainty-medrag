"""Length-only control of the exact completed 18-pair ModernBERT probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import nli_stance as nli
from . import stance_nli_probe as original

CONTROL_VERSION = "modernbert-nli-supported-context-v1"
CACHE_NAMESPACE = "nli_truncation_control_v1"
PREVIOUS_RUN_RELATIVE = (
    "outputs/stance_nli_probe/dev_1_30/modernbert_mednli_diagnostic_18/"
    "nli-20260929T144606484011Z"
)
OUTPUT_RELATIVE = "outputs/stance_nli_probe/dev_1_30/modernbert_mednli_truncation_control_18"


def _finite_limit(value):
    # Huge Hugging Face sentinel values do not describe supported context.
    return type(value) is int and 0 < value < 1_000_000


def inspect_supported_context(tokenizer, config):
    token_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = getattr(config, "max_position_embeddings", None)
    if not _finite_limit(model_limit):
        raise nli.NLIProbeError("CONTEXT_LIMIT_FAILURE", "A finite model-supported maximum is required")
    effective = min(model_limit, token_limit) if _finite_limit(token_limit) else model_limit
    if effective <= tokenizer.num_special_tokens_to_add(pair=True):
        raise nli.NLIProbeError("CONTEXT_LIMIT_FAILURE", "Supported length leaves no room for pair text")
    native = getattr(tokenizer, "backend_tokenizer", None)
    return {
        "tokenizer_model_max_length": token_limit,
        "model_max_position_embeddings": model_limit,
        "tokenizer_limit_ignored_as_sentinel_or_unavailable": not _finite_limit(token_limit),
        "effective_max_length": effective,
        "selection_rule": "Largest finite length within both tokenizer and model limits; no training/probe cap",
        "loaded_native_truncation_before_control": getattr(native, "truncation", None),
        "loaded_native_padding_before_control": getattr(native, "padding", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "control_tokenizer_settings": {"add_special_tokens": True, "truncation": "longest_first",
                                       "max_length": effective, "padding": False},
    }


def tokenize_supported_pair(tokenizer, *, premise, hypothesis, context):
    limit = context["effective_max_length"]
    uncut = tokenizer(premise, hypothesis, add_special_tokens=True, truncation=False, padding=False)
    ids = uncut.get("input_ids")
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise nli.NLIProbeError("TOKENIZATION_FAILURE", "Expected one untruncated pair sequence")
    original_length = len(ids)
    encoded = tokenizer(premise, hypothesis, add_special_tokens=True, truncation="longest_first",
                        max_length=limit, padding=False, return_tensors="pt")
    shape = tuple(encoded["input_ids"].shape)
    if len(shape) != 2 or shape[0] != 1 or shape[1] != min(original_length, limit) or shape[1] <= 0:
        raise nli.NLIProbeError("TOKENIZATION_FAILURE", "Final length does not match explicit supported-context tokenization")
    return encoded, {
        "original_token_length": original_length, "final_token_length": shape[1],
        "effective_max_length": limit, "truncation_occurred": shape[1] < original_length,
        "tokens_removed": original_length - shape[1],
        "length_limit_sources": {"tokenizer_model_max_length": context["tokenizer_model_max_length"],
                                 "model_max_position_embeddings": context["model_max_position_embeddings"]},
        "truncation_strategy": "longest_first", "add_special_tokens": True, "padding": False,
        "token_lengths_include_pair_special_tokens": True,
    }


class SupportedContextNLIBackend:
    """Reuse the loaded checkpoint, label mapping and math; change only token length."""
    def __init__(self, backend):
        self.model, self.tokenizer = backend.model, backend.tokenizer
        self.torch, self.device = backend.torch, backend.device
        self.label_mapping = backend.label_mapping
        self.context = inspect_supported_context(self.tokenizer, self.model.config)
        self.identity = copy.deepcopy(backend.identity)
        self.identity.update(probe_version=CONTROL_VERSION, experiment_version=CONTROL_VERSION,
                             max_length=self.context["effective_max_length"],
                             supported_context_inspection=self.context)
        self.identity["tokenizer_settings"] = {
            "use_fast": True, **self.context["control_tokenizer_settings"],
            "length_limit_sources": {"tokenizer_model_max_length": self.context["tokenizer_model_max_length"],
                                     "model_max_position_embeddings": self.context["model_max_position_embeddings"]},
        }

    def infer(self, *, premise, hypothesis):
        encoded, truncation = tokenize_supported_pair(self.tokenizer, premise=premise,
                                                      hypothesis=hypothesis, context=self.context)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
            if tuple(logits.shape) != (1, 3):
                raise nli.NLIProbeError("INVALID_MODEL_OUTPUT", "Model must return one three-class logits vector")
            scores = self.torch.softmax(logits.to(dtype=self.torch.float64), dim=-1)
        raw = logits.detach().cpu().tolist()[0]
        normalized = scores.detach().cpu().tolist()[0]
        return {**nli.score_result(raw, normalized, self.label_mapping), "truncation": truncation}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def load_previous_run(source):
    """Use saved isolated inputs only; never read the MedQA source dataset."""
    source = Path(source)
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ("inputs.jsonl", "pairs.jsonl", "results.jsonl", "model_metadata.json")}
    if hashes["inputs.jsonl"] != original.DIAGNOSTIC_INPUT_SHA256:
        raise ValueError("Inputs differ from the original 18 diagnostic pairs")
    rows, pairs, previous = (_lines(source / name) for name in ("inputs.jsonl", "pairs.jsonl", "results.jsonl"))
    identity = _load(source / "model_metadata.json")
    for name in ("model_revision", "tokenizer_revision", "resolved_model_revision", "resolved_tokenizer_revision"):
        if identity[name] != nli.MODEL_REVISION:
            raise ValueError("Previous run must use the exact pinned model/tokenizer revision")
    if identity["model_id"] != nli.MODEL_ID or identity["tokenizer_settings"]["max_length"] != 256:
        raise ValueError("Expected the completed 256-token ModernBERT control")
    if identity["hypothesis_version"] != nli.HYPOTHESIS_VERSION or identity["premise_version"] != nli.PREMISE_VERSION:
        raise ValueError("Previous input construction version differs")
    for collection in (rows, pairs, previous):
        if len(collection) != 18 or tuple(original.key(row) for row in collection) != original.DIAGNOSTIC_KEYS:
            raise ValueError("Keys/order differ from the fixed 18 development pairs")
    for row, saved_pair, old in zip(rows, pairs, previous):
        if set(row) != set(nli.INPUT_FIELDS) | set(nli.ID_FIELDS):
            raise ValueError("Only the four allowed source texts and judgment IDs are permitted")
        pair = nli.build_pair(**{field: row[field] for field in nli.INPUT_FIELDS})
        if {name: saved_pair[name] for name in pair} != pair or old["pair"] != pair:
            raise ValueError("Saved premise/hypothesis text differs; no edits allowed")
        if old["status"] != "SUCCESS" or old["model_metadata"] != identity:
            raise ValueError("Expected 18 completed outputs from one verified model identity")
        if old["truncation"]["final_token_length"] != 256:
            raise ValueError("Expected the original 256-token final length for each pair")
        nli.score_result(old["raw_logits"], old["softmax_scores_in_model_label_order"], identity["verified_label_mapping"])
        if old["input_sha256"] != {field: nli.text_hash(row[field]) for field in nli.INPUT_FIELDS}:
            raise ValueError("Previous input hashes differ")
    return rows, pairs, previous, identity, hashes


def preservation_snapshot(root):
    frozen = original.protected_files(root)
    for relative in ("data/cache/stance/nli_v0", "outputs/stance_nli_probe"):
        for path in (root / relative).rglob("*"):
            if path.is_file():
                frozen[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frozen


def summarize_comparison(previous, results):
    if len(previous) != len(results):
        raise ValueError("Comparison row counts differ")
    def summary(rows):
        adapted = [{**r, "hard_label_agreement": False} for r in rows]
        stats = original.summarize(adapted)
        valid = [r for r in rows if r["status"] == "SUCCESS"]
        entropies = [r["normalized_entropy"] for r in valid]
        return {name: stats[name] for name in ("valid_nli_outputs", "execution_failures", "argmax_counts",
            "unique_rounded_score_triplet_count", "triplet_rounding_decimal_places", "exact_one_hot_count",
            "entropy_buckets", "entropy_bucket_boundaries", "contradiction_predictions",
            "truncated_input_count", "truncated_input_percentage")} | {
                "mean_entropy": statistics.mean(entropies) if entropies else None,
                "median_entropy": statistics.median(entropies) if entropies else None,
            }
    valid_pairs = [(old, new) for old, new in zip(previous, results) if new["status"] == "SUCCESS"]
    return {"judgments": len(results), "before": summary(previous), "after": summary(results),
            "argmax_labels_changed": sum(old["argmax_label"] != new["argmax_label"] for old, new in valid_pairs),
            "label_change_evaluated_pair_count": len(valid_pairs),
            "actual_model_calls": sum(r["model_calls"] for r in results),
            "cache_hits": sum(r["cache_status"] == "HIT" for r in results),
            "scores_are_calibrated_probabilities": False, "independent_gold_stance_annotation_available": False,
            "accuracy_evaluated": False, "superiority_claimed": False,
            "gold_or_other_options_exposed": False, "dev_31_50_used": False,
            "retrieval_run": False, "quality_scoring_run": False, "aggregation_run": False,
            "answer_prediction_run": False}


def comparison_markdown(results, summary, context):
    lines = ["# ModernBERT truncation control", "",
             "Only maximum input length changed. Scores are not calibrated probabilities; there is no independent stance gold annotation.", "",
             "Model: `" + nli.MODEL_ID + "`; revision: `" + nli.MODEL_REVISION + "`.", "",
             "## Supported context", "", "```json", json.dumps(context, indent=2), "```", "",
             "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "",
             "## Same 18 pairs", "", "Triplet order: SUPPORT, CONTRADICT, IRRELEVANT.", "",
             "| question_id | option | PMID | previous triplet | new triplet | previous argmax | new argmax | previous entropy | new entropy | previous truncated | new truncated | original / previous final / new final | removed |",
             "|---|---|---|---|---|---|---|---:|---:|---|---|---|---:|"]
    def triplet(scores):
        return "(" + ", ".join(f"{scores[label]:.6f}" for label in nli.STANCE_ORDER) + ")"
    for result in results:
        old = result["previous_256"]
        valid = result["status"] == "SUCCESS"
        trunc = result.get("truncation")
        values = [result["question_id"], result["candidate_option_id"], result["evidence_doc_id"],
                  triplet(old["nli_label_scores"]), triplet(result["nli_label_scores"]) if valid else "N/A",
                  old["argmax_label"], result.get("argmax_label", "N/A"), f'{old["normalized_entropy"]:.6f}',
                  f'{result["normalized_entropy"]:.6f}' if valid else "N/A", old["truncation"]["truncation_occurred"],
                  trunc["truncation_occurred"] if valid else "N/A",
                  f'{old["truncation"]["original_token_length"]}/256/{trunc["final_token_length"]}' if valid else "N/A",
                  trunc["tokens_removed"] if valid else "N/A"]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def run_control(*, root, previous_run=None, output_dir=None, backend_loader=nli.load_transformers_backend):
    root = Path(root).resolve()
    source = Path(previous_run).resolve() if previous_run else root / PREVIOUS_RUN_RELATIVE
    rows, pairs, previous, old_identity, source_hashes = load_previous_run(source)
    base = (root / OUTPUT_RELATIVE).resolve()
    output = Path(output_dir).resolve() if output_dir else base / datetime.now(timezone.utc).strftime("control-%Y%m%dT%H%M%S%fZ")
    if base not in output.parents or output.exists():
        raise ValueError("Use a new truncation-control output directory; existing outputs cannot be overwritten")
    frozen = preservation_snapshot(root)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs.jsonl", "pairs.jsonl"):
        (output / name).write_bytes((source / name).read_bytes())
    original.save_json(output / "previous_results.json", previous)
    report = {"status": "PREPARED", "experiment_version": CONTROL_VERSION,
              "previous_run": str(source), "previous_file_sha256": source_hashes,
              "protected_existing_files_sha256": frozen, "model_calls": 0, "results": []}
    original.save_json(output / "report.json", report)
    backend = SupportedContextNLIBackend(backend_loader(spec=nli.NLIModelSpec(),
        hf_cache_dir=root / "data/cache/huggingface/nli_probe", device="cpu", local_files_only=True))
    if backend.label_mapping != old_identity["verified_label_mapping"]:
        raise nli.NLIProbeError("LABEL_MAPPING_FAILURE", "Loaded labels differ from the previous run")
    for field in ("snapshot_file_sha256", "runtime_versions", "model_dtype", "logit_softmax_dtype", "seed", "device"):
        if backend.identity[field] != old_identity[field]:
            raise nli.NLIProbeError("MODEL_IDENTITY_FAILURE", "Only length may change; runtime/model identity differs")
    if backend.identity["model_revision"] != nli.MODEL_REVISION or backend.identity["resolved_model_revision"] != nli.MODEL_REVISION:
        raise nli.NLIProbeError("MODEL_IDENTITY_FAILURE", "Pinned model revision differs")
    original.save_json(output / "supported_context.json", backend.context)
    original.save_json(output / "model_metadata.json", backend.identity)
    print("Supported context:", nli.canonical(backend.context), flush=True)
    print("Verified labels:", nli.canonical(backend.label_mapping), flush=True)
    print("Actual pinned revision:", backend.identity["resolved_model_revision"], flush=True)
    report.update(status="RUNNING", model_metadata=backend.identity, supported_context=backend.context)
    original.save_json(output / "report.json", report)
    classifier = nli.NLIStanceClassifier(backend, cache_dir=root / "data/cache/stance" / CACHE_NAMESPACE)
    results, audit = [], []
    for row, saved_pair, old in zip(rows, pairs, previous):
        result = classifier.classify_texts(**row)
        if result["pair"] != {name: saved_pair[name] for name in ("premise", "hypothesis")}:
            raise ValueError("Exact pair text changed")
        if result["status"] == "SUCCESS" and result["truncation"]["original_token_length"] != old["truncation"]["original_token_length"]:
            raise nli.NLIProbeError("TOKENIZATION_FAILURE", "Untruncated token length changed between runs")
        result.update(experiment_version=CONTROL_VERSION, previous_256=old,
                      previous_final_token_length=old["truncation"]["final_token_length"],
                      new_maximum_length=backend.context["effective_max_length"])
        if result["status"] == "SUCCESS":
            ordered = result["softmax_scores_in_model_label_order"]
            result["softmax_scores_by_nli_label"] = {
                backend.label_mapping[str(index)]["nli_label"]: value for index, value in enumerate(ordered)}
        results.append(result)
        audit.append({**{name: row[name] for name in nli.ID_FIELDS}, "model_input_fields": ["premise", "hypothesis"],
                      "permitted_source_fields": list(nli.INPUT_FIELDS), "pair_sha256": result["pair_sha256"],
                      "input_sha256": result["input_sha256"], "cache_key": result["cache_key"],
                      "cache_status": result["cache_status"], "model_calls": result["model_calls"],
                      "gold_or_other_options_exposed": False})
        report.update(results=results, model_calls=sum(r["model_calls"] for r in results))
        original.save_json(output / "report.json", report)
        original.save_jsonl(output / "results.jsonl", results)
        original.save_json(output / "request_audit.json", audit)
    original.verify_protected(root, frozen)
    summary = summarize_comparison(previous, results)
    summary.update(effective_max_length=backend.context["effective_max_length"],
                   previous_max_length=256, model_id=nli.MODEL_ID, model_revision=nli.MODEL_REVISION,
                   old_outputs_and_caches_byte_identical=True, premise_hypothesis_bytes_unchanged=True)
    report.update(status="COMPLETE" if summary["after"]["valid_nli_outputs"] == 18 else "COMPLETE_WITH_FAILURES",
                  summary=summary, protected_existing_files_byte_identical=True)
    original.save_json(output / "report.json", report)
    original.save_json(output / "summary.json", summary)
    (output / "comparison.md").write_text(comparison_markdown(results, summary, backend.context), encoding="utf-8")
    return report, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("MEDRAG_PROJECT_ROOT", ".")))
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    report, output = run_control(root=args.project_root, previous_run=args.previous_run, output_dir=args.output_dir)
    print((output / "comparison.md").read_text(encoding="utf-8"))
    print("Execution status:", report["status"])
    print("Outputs:", output)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
