"""Exactly the existing 18 development judgments; no resampling or retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .nli_stance import (HYPOTHESIS_VERSION, ID_FIELDS, INPUT_FIELDS, MODEL_ID, MODEL_REVISION,
                         NLIModelSpec, NLIProbeError, NLIStanceClassifier, PREMISE_VERSION,
                         PROBE_VERSION, STANCE_ORDER, build_pair, canonical,
                         load_transformers_backend, normalized_entropy, text_hash)

DIAGNOSTIC_INPUT_SHA256 = "7764a28f92861b49f1883e8bc8effc700b82d79393173070d5d56955f8dd682b"
SOURCE_RELATIVE = "outputs/stance_probability_inspection/dev_1_30/v1_diagnostic_18/kaggle-probability-20260929T064805Z"
DIAGNOSTIC_KEYS = (
    ("medqa-us-dev-000004", "A", "40894990"),
    ("medqa-us-dev-000002", "A", "33224097"),
    ("medqa-us-dev-000008", "C", "38986844"),
    ("medqa-us-dev-000009", "A", "25542071"),
    ("medqa-us-dev-000010", "A", "3547382"),
    ("medqa-us-dev-000002", "A", "29556858"),
    ("medqa-us-dev-000002", "C", "40752916"),
    ("medqa-us-dev-000002", "E", "36988834"),
    ("medqa-us-dev-000004", "B", "39996184"),
    ("medqa-us-dev-000008", "D", "39816678"),
    ("medqa-us-dev-000009", "D", "33305487"),
    ("medqa-us-dev-000010", "B", "6837450"),
    ("medqa-us-dev-000010", "A", "30676481"),
    ("medqa-us-dev-000002", "B", "33224097"),
    ("medqa-us-dev-000004", "C", "35695404"),
    ("medqa-us-dev-000008", "B", "12190212"),
    ("medqa-us-dev-000009", "A", "33305487"),
    ("medqa-us-dev-000010", "D", "15517514"),
)


def key(row):
    return tuple(row[name] for name in ID_FIELDS)


def save_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def save_jsonl(path, rows):
    Path(path).write_bytes("".join(canonical(row) + "\n" for row in rows).encode("utf-8"))


def load_diagnostic(source):
    """No loader for MedQA source data exists in this probe."""
    source = Path(source)
    data = (source / "inputs.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != DIAGNOSTIC_INPUT_SHA256:
        raise ValueError("Existing diagnostic input bytes differ; resampling is prohibited")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    old_data = (source / "stances.jsonl").read_bytes()
    old = [json.loads(line) for line in old_data.decode("utf-8").splitlines() if line]
    if len(rows) != 18 or tuple(key(row) for row in rows) != DIAGNOSTIC_KEYS:
        raise ValueError("Expected exactly the fixed eighteen development pairs")
    if len(old) != 18 or tuple(key(row) for row in old) != DIAGNOSTIC_KEYS:
        raise ValueError("Qwen v1 reference keys/order differ from the original diagnostic")
    references = []
    for row, previous in zip(rows, old):
        if set(row) != set(ID_FIELDS) | set(INPUT_FIELDS):
            raise ValueError("Unexpected field: gold or other options cannot enter the probe")
        build_pair(**{field: row[field] for field in INPUT_FIELDS})
        if previous["classifier_version"] != "llm-medical-stance-v1" or previous["status"] != "SUCCESS":
            raise ValueError("Expected completed v1 outputs")
        if any(previous["input_sha256"][field] != text_hash(row[field]) for field in INPUT_FIELDS):
            raise ValueError("V1 reference does not describe the exact same four texts")
        probabilities = [previous[name] for name in ("p_support", "p_contradict", "p_irrelevant")]
        # V1 validates sums within 1e-6. This saved diagnostic sums exactly to 1.
        entropy = normalized_entropy(probabilities)
        references.append({**{field: previous[field] for field in ID_FIELDS},
                           "classifier_version": previous["classifier_version"],
                           "triplet": probabilities, "argmax_label": previous["argmax_label"],
                           "normalized_entropy": entropy, "cache_key": previous["cache_key"]})
    return rows, references, {"input_sha256": hashlib.sha256(data).hexdigest(),
                              "v1_output_sha256": hashlib.sha256(old_data).hexdigest()}


def protected_files(root):
    paths = [root / "pyproject.toml", root / "README.md", root / "REFERENCE.md"]
    for rel in ("src", "configs", "schemas", "data/cache/stance/v1", "data/cache/stance/logprob_v0",
                "outputs/stance_smoke", "outputs/stance_probability_inspection", "outputs/stance_logprob_probe"):
        directory = root / rel
        paths.extend(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def verify_protected(root, frozen):
    if any(not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != sha for name, sha in frozen.items()):
        raise RuntimeError("Protected existing research or Qwen artifacts changed")


def summarize(results):
    valid = [r for r in results if r["status"] == "SUCCESS"]
    counts = Counter(r["argmax_label"] for r in valid)
    triplets = Counter(tuple(round(r["nli_label_scores"][label], 6) for label in STANCE_ORDER) for r in valid)
    entropies = [r["normalized_entropy"] for r in valid]
    bins = Counter("<0.05" if h < .05 else "0.05-0.25" if h < .25 else "0.25-0.50" if h <= .50 else ">0.50" for h in entropies)
    truncated = sum(r["truncation"]["truncation_occurred"] for r in valid)
    return {"judgments": len(results), "valid_nli_outputs": len(valid), "execution_failures": len(results) - len(valid),
            "actual_model_calls": sum(r["model_calls"] for r in results),
            "cache_hits": sum(r["cache_status"] == "HIT" for r in results),
            "argmax_counts": {**{label: counts[label] for label in STANCE_ORDER}, "UNRESOLVED": counts[None]},
            "unique_rounded_score_triplet_count": len(triplets), "triplet_rounding_decimal_places": 6,
            "unique_rounded_score_triplets": [{"scores": list(t), "count": n} for t, n in sorted(triplets.items())],
            "exact_one_hot_count": sum(sum(p == 1 for p in r["nli_label_scores"].values()) == 1 and sum(p == 0 for p in r["nli_label_scores"].values()) == 2 for r in valid),
            "entropy_buckets": {b: bins[b] for b in ("<0.05", "0.05-0.25", "0.25-0.50", ">0.50")},
            "entropy_bucket_boundaries": "[0,0.05), [0.05,0.25), [0.25,0.50], (0.50,1]; unrounded entropy",
            "normalized_entropy_values": entropies,
            "qwen_v1_hard_label_agreement_count": sum(r["hard_label_agreement"] is True for r in valid),
            "contradiction_predictions": counts["CONTRADICT"],
            "truncated_input_count": truncated, "truncation_evaluated_pair_count": len(valid),
            "truncated_input_percentage": 100 * truncated / len(valid) if valid else None,
            "scores_are_calibrated_probabilities": False, "independent_gold_stance_annotation_available": False,
            "accuracy_evaluated": False, "superiority_claimed": False,
            "pubmed_requests": 0, "qwen_model_calls": 0, "gold_or_other_options_exposed": False,
            "dev_31_50_used": False, "quality_scoring_run": False, "aggregation_run": False,
            "answer_prediction_run": False}


def comparison_markdown(rows, summary, *, label_mapping=None):
    lines = ["# Independent medical/scientific NLI stance feasibility probe", "",
             "NLI label scores are softmax scores, not calibrated probabilities. There are no independent gold stance annotations, accuracy estimates or superiority claims.", "",
             "Hypothesis construction: `" + HYPOTHESIS_VERSION + "` (one deterministic template for all pairs).",
             "Requested pinned revision: `" + MODEL_REVISION + "`.", "",
             "Verified downloaded label mapping: " + (canonical(label_mapping) if label_mapping else "NOT YET VERIFIED; model not loaded"), "",
             "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "",
             "## All 18 original diagnostic pairs", "",
             "Triplet order: SUPPORT, CONTRADICT, IRRELEVANT. Raw logits retain the actual model's label-index order.", "",
             "| question_id | option | PMID | Qwen-v1 triplet | Qwen-v1 argmax | NLI raw logits | NLI label scores | NLI argmax | entropy | agrees | truncated | original/final tokens |",
             "|---|---|---|---|---|---|---|---|---:|---|---|---|"]
    for r in rows:
        old = r["qwen_v1_reference"]
        scores = r.get("nli_label_scores")
        score_text = str(tuple(round(scores[label], 6) for label in STANCE_ORDER)) if scores else "N/A"
        truncation = r.get("truncation")
        lengths = str(truncation["original_token_length"]) + "/" + str(truncation["final_token_length"]) if truncation else "N/A"
        vals = [r["question_id"], r["candidate_option_id"], r["evidence_doc_id"], tuple(old["triplet"]), old["argmax_label"],
                r.get("raw_logits", "N/A"), score_text, r.get("argmax_label", "N/A"),
                f'{r["normalized_entropy"]:.4f}' if scores else "N/A", r.get("hard_label_agreement", "N/A"),
                truncation["truncation_occurred"] if truncation else "N/A", lengths]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines) + "\n"


def run_probe(*, root, source=None, output_dir=None, backend_loader=load_transformers_backend,
              prepare_only=False, device="cpu", local_files_only=False, use_existing_hf_login=False):
    root = Path(root).resolve()
    source = Path(source).resolve() if source else root / SOURCE_RELATIVE
    rows, references, source_hashes = load_diagnostic(source)
    base = root / "outputs/stance_nli_probe/dev_1_30/modernbert_mednli_diagnostic_18"
    output = Path(output_dir).resolve() if output_dir else base / datetime.now(timezone.utc).strftime("nli-%Y%m%dT%H%M%S%fZ")
    if base.resolve() not in output.parents:
        raise ValueError("Outputs must be in a new diagnostic_18 run directory")
    if output.exists():
        raise ValueError("Refusing to overwrite any existing output directory")
    frozen = protected_files(root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs.jsonl").write_bytes((source / "inputs.jsonl").read_bytes())
    pairs = []
    for row, old in zip(rows, references):
        pair = build_pair(**{field: row[field] for field in INPUT_FIELDS})
        pairs.append({**{field: row[field] for field in ID_FIELDS}, **pair, "hypothesis_version": HYPOTHESIS_VERSION,
                      "premise_version": PREMISE_VERSION, "qwen_v1_reference": old,
                      "input_sha256": {field: text_hash(row[field]) for field in INPUT_FIELDS},
                      "pair_sha256": {field: text_hash(text) for field, text in pair.items()}})
    save_jsonl(output / "pairs.jsonl", pairs)
    save_json(output / "qwen_v1_reference.json", references)
    report = {"status": "PREPARED_NOT_EXECUTED", "probe_version": PROBE_VERSION,
              "requested_model_id": MODEL_ID, "requested_model_revision": MODEL_REVISION,
              "requested_tokenizer_revision": MODEL_REVISION,
              "actual_model_revision_used": None, "verified_label_mapping": None,
              "hypothesis_version": HYPOTHESIS_VERSION, "premise_version": PREMISE_VERSION,
              "source_directory": str(source), **source_hashes,
              "protected_existing_files_sha256": frozen, "judgments_prepared": 18,
              "model_calls": 0, "device_requested": device, "results": [],
              "hub_authentication_mode": "existing_login" if use_existing_hf_login else "anonymous"}
    save_json(output / "report.json", report)
    if prepare_only:
        verify_protected(root, frozen)
        summary = {"status": "PREPARED_NOT_EXECUTED", "judgments_prepared": 18, "actual_model_calls": 0,
                   "actual_model_revision_used": None, "label_mapping_verified": False,
                   "nli_score_entropy_agreement_truncation_statistics": None,
                   "gold_or_other_options_exposed": False, "dev_31_50_used": False,
                   "protected_existing_files_byte_identical": True}
        save_json(output / "summary.json", summary)
        (output / "comparison.md").write_bytes(comparison_markdown(pairs, summary).encode("utf-8"))
        return report, output
    try:
        backend = backend_loader(spec=NLIModelSpec(), hf_cache_dir=root / "data/cache/huggingface/nli_probe",
                                 device=device, local_files_only=local_files_only,
                                 use_existing_hf_login=use_existing_hf_login)
    except NLIProbeError as exc:
        verify_protected(root, frozen)
        report.update(status=exc.status, error=str(exc), protected_existing_files_byte_identical=True)
        save_json(output / "report.json", report)
        summary = {"status": exc.status, "judgments_prepared": 18, "actual_model_calls": 0,
                   "actual_model_revision_used": None, "label_mapping_verified": False,
                   "nli_score_entropy_agreement_truncation_statistics": None,
                   "gold_or_other_options_exposed": False, "dev_31_50_used": False,
                   "protected_existing_files_byte_identical": True, "error": str(exc)}
        save_json(output / "summary.json", summary)
        (output / "comparison.md").write_bytes(comparison_markdown(pairs, summary).encode("utf-8"))
        return report, output
    report.update(status="RUNNING", model_metadata=backend.identity,
                  actual_model_revision_used=backend.identity["resolved_model_revision"],
                  verified_label_mapping=backend.label_mapping)
    save_json(output / "model_metadata.json", backend.identity)
    print("Verified model.config.id2label:", canonical(backend.identity["raw_id2label"]), flush=True)
    print("Verified model.config.label2id:", canonical(backend.identity["raw_label2id"]), flush=True)
    print("Verified semantic stance mapping:", canonical(backend.label_mapping), flush=True)
    print("Parameter count:", backend.identity["parameter_count"], flush=True)
    classifier = NLIStanceClassifier(backend, cache_dir=root / "data/cache/stance/nli_v0")
    results, audits = [], []
    for row, old in zip(rows, references):
        result = classifier.classify_texts(**row)
        result.update(qwen_v1_reference=old, candidate_option_text=row["candidate_option_text"],
                      evidence_title=row["evidence_title"],
                      hard_label_agreement=(result["argmax_label"] == old["argmax_label"]) if result["status"] == "SUCCESS" else None)
        results.append(result)
        audits.append({**{field: row[field] for field in ID_FIELDS}, "model_input_fields": ["premise", "hypothesis"],
                       "permitted_source_fields": list(INPUT_FIELDS), "input_sha256": result["input_sha256"],
                       "pair_sha256": result["pair_sha256"], "cache_status": result["cache_status"],
                       "model_calls": result["model_calls"], "gold_or_other_options_exposed": False})
        report.update(results=results, model_calls=sum(r["model_calls"] for r in results))
        save_json(output / "report.json", report)
        save_jsonl(output / "results.jsonl", results)
        save_json(output / "request_audit.json", audits)
    verify_protected(root, frozen)
    summary = summarize(results)
    summary.update(actual_model_revision_used=backend.identity["resolved_model_revision"],
                   model_id=MODEL_ID, protected_existing_files_byte_identical=True)
    report.update(status="COMPLETE" if summary["valid_nli_outputs"] == 18 else "COMPLETE_WITH_EXPLICIT_FAILURES",
                  summary=summary, protected_existing_files_byte_identical=True)
    save_json(output / "report.json", report)
    save_json(output / "summary.json", summary)
    (output / "comparison.md").write_bytes(comparison_markdown(results, summary, label_mapping=backend.label_mapping).encode("utf-8"))
    return report, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("MEDRAG_PROJECT_ROOT", ".")))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-existing-hf-login", action="store_true",
                        help="Use an already configured Hub login; never create or save credentials")
    args = parser.parse_args(argv)
    report, output = run_probe(root=args.project_root, source=args.source_run, output_dir=args.output_dir,
                               prepare_only=args.prepare_only, device=args.device, local_files_only=args.local_files_only,
                               use_existing_hf_login=args.use_existing_hf_login)
    print((output / "comparison.md").read_text(encoding="utf-8"))
    print("Execution status:", report["status"])
    print("Outputs:", output)
    return 0 if report["status"] in ("COMPLETE", "PREPARED_NOT_EXECUTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
