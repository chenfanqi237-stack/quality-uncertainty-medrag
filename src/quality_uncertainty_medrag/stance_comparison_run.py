"""Frozen four-method stance comparison on the development reference 60.

Inference never reads the reference label. Evaluation is a separate command
that joins labels only after every unlabeled prediction is complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from quality_uncertainty_medrag import llm_stance as qwen
from quality_uncertainty_medrag import nli_stance as nli
from quality_uncertainty_medrag import nli_truncation_control as control
from quality_uncertainty_medrag import sentence_nli as sentence
from quality_uncertainty_medrag.ollama_backend import OllamaTextGenerationBackend

METHODS = ("qwen_v1", "nli_256", "nli_8192", "nli_sentence")
LABELS = ("SUPPORT", "CONTRADICT", "IRRELEVANT")
CSV_TO_INPUT = {
    "question_id": "question_id",
    "candidate_option_id": "candidate_option_id",
    "pmid": "evidence_doc_id",
    "question_text": "question_stem",
    "candidate_option_text": "candidate_option_text",
    "evidence_title": "evidence_title",
    "evidence_abstract": "evidence_abstract",
}
KEY_FIELDS = ("question_id", "candidate_option_id", "evidence_doc_id")
TEXT_FIELDS = tuple(nli.INPUT_FIELDS)
EXPECTED_QWEN = qwen.StanceModelConfig()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def input_hash(row):
    return {field: sha(row[field]) for field in TEXT_FIELDS}


def read_unlabeled_inputs(reference):
    """Project source columns immediately; no label survives this boundary."""
    projected = []
    with Path(reference).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(CSV_TO_INPUT) | {"batch", "pair_id", "human_stance"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError("Reference schema differs from fixed 60-pair format")
        for source in reader:
            row = {destination: source[origin] for origin, destination in CSV_TO_INPUT.items()}
            if set(row) != set(KEY_FIELDS) | set(TEXT_FIELDS):
                raise AssertionError("Model input projection changed")
            if any(not isinstance(value, str) or not value.strip()
                   for field, value in row.items() if field != "evidence_abstract"):
                raise ValueError("Missing required inference input")
            if not isinstance(row["evidence_abstract"], str):
                raise ValueError("Abstract must be a string")
            projected.append(row)
    if len(projected) != 60 or len({key(row) for row in projected}) != 60:
        raise ValueError("Expected 60 unique inference triples")
    if any(not row["question_id"].startswith("medqa-us-dev-") or
           not 1 <= int(row["question_id"].rsplit("-", 1)[1]) <= 30 for row in projected):
        raise ValueError("Input outside exposed dev 1-30")
    return projected


def validate_scores(row):
    values = [row[f"p_{name.lower()}"] for name in LABELS]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid three-class score")
    if not math.isclose(math.fsum(values), 1, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Three-class scores do not sum to one")
    maximum = max(values)
    tied = [name for name, value in zip(LABELS, values) if value == maximum]
    expected = tied[0] if len(tied) == 1 else None
    if row["argmax_label"] != expected:
        raise ValueError("Argmax differs from three-class scores")
    if expected is None:
        raise ValueError("Unresolved hard-label tie; no silent tie break in evaluation")


def _cache_match(entry, method, row):
    ids = key(row)
    hashes = input_hash(row)
    if method == "qwen_v1":
        if tuple(entry.get(field) for field in KEY_FIELDS) != ids:
            return False
        if entry.get("input_sha256") != hashes or entry.get("classifier_version") != qwen.CLASSIFIER_VERSION:
            return False
        if entry.get("prompt_template_sha256") != sha(qwen.STANCE_PROMPT):
            return False
        if entry.get("prompt_sha256") != sha(qwen.build_stance_prompt(**{f: row[f] for f in TEXT_FIELDS})):
            return False
        if entry.get("model_metadata") != vars(EXPECTED_QWEN):
            return False
        expected_context = {name: entry[name] for name in (
            "schema_version", "question_id", "candidate_option_id", "evidence_doc_id",
            "classifier_version", "prompt_template_sha256", "prompt_sha256", "input_sha256",
            "model_metadata")}
        if expected_context["schema_version"] != 1 or entry.get("cache_key") != sha(canonical(expected_context)):
            raise ValueError("Original Qwen cache context/hash mismatch")
        attempts = entry.get("attempts")
        if (type(entry.get("generation_attempts")) is not int or
                not 1 <= entry["generation_attempts"] <= 3 or
                not isinstance(attempts, list) or len(attempts) != entry["generation_attempts"] or
                attempts[-1].get("status") != "success"):
            raise ValueError("Original Qwen attempt audit is incomplete")
        probs = entry.get("probabilities", {})
        result = {"p_support": probs.get("SUPPORT"), "p_contradict": probs.get("CONTRADICT"),
                  "p_irrelevant": probs.get("IRRELEVANT"), "argmax_label": entry.get("argmax_label")}
        validate_scores(result)
        return True
    context, result = entry.get("context", {}), entry.get("result", {})
    if tuple(context.get(field) for field in KEY_FIELDS) != ids or context.get("input_sha256") != hashes:
        return False
    if entry.get("integrity_sha256") != sha(canonical({k: v for k, v in entry.items() if k != "integrity_sha256"})):
        raise ValueError("Original NLI cache integrity mismatch")
    identity = context.get("backend_identity", {})
    if identity.get("model_id") != nli.MODEL_ID or identity.get("model_revision") != nli.MODEL_REVISION:
        return False
    if identity.get("tokenizer_revision") != nli.MODEL_REVISION or identity.get("resolved_model_revision") != nli.MODEL_REVISION:
        return False
    if identity.get("hypothesis_version") != nli.HYPOTHESIS_VERSION or identity.get("premise_version") != nli.PREMISE_VERSION:
        return False
    if identity.get("max_length") != (256 if method == "nli_256" else 8192):
        return False
    if method == "nli_sentence":
        if context.get("sentence_split_version") != sentence.SPLITTER_VERSION or context.get("sentence_selection_version") != sentence.SELECTION_VERSION:
            return False
        if context.get("effective_max_length") != 8192 or context.get("hypothesis_sha256") != sha(nli.build_hypothesis(question_stem=row["question_stem"], candidate_option_text=row["candidate_option_text"])):
            return False
        if context.get("sentences") != [{"sentence_index": s["sentence_index"], "source": s["source"], "text_sha256": sha(s["text"])} for s in sentence.evidence_sentences(title=row["evidence_title"], abstract=row["evidence_abstract"])]:
            return False
        if result.get("selected_sentence") not in [s["text"] for s in sentence.evidence_sentences(title=row["evidence_title"], abstract=row["evidence_abstract"])]:
            return False
        chosen = sentence.select_sentence(result["sentences"])
        if (chosen["sentence_index"] != result.get("selected_sentence_index") or
                chosen["nli_label_scores"] != result.get("nli_label_scores") or
                chosen["argmax_label"] != result.get("argmax_label")):
            raise ValueError("Cached sentence selection differs from frozen rule")
    else:
        if context.get("pair_sha256") != {k: sha(v) for k, v in nli.build_pair(**{f: row[f] for f in TEXT_FIELDS}).items()}:
            return False
        if method == "nli_8192" and identity.get("probe_version") != control.CONTROL_VERSION:
            return False
        if method == "nli_256" and identity.get("probe_version") != nli.PROBE_VERSION:
            return False
    if result.get("status") != "SUCCESS" or result.get("model_metadata") != identity:
        return False
    nli.score_result(result["raw_logits"], result["softmax_scores_in_model_label_order"], identity["verified_label_mapping"])
    validate_scores(result)
    return True


def _cache_paths(root, method):
    relative = {
        "qwen_v1": "data/cache/stance/v1",
        "nli_256": "data/cache/stance/nli_v0",
        "nli_8192": "data/cache/stance/nli_truncation_control_v1",
        "nli_sentence": "data/cache/stance/nli_sentence_v1/evidence",
    }[method]
    return sorted((root / relative).glob("*.json"))


def reusable_cache(root, method, row):
    for path in _cache_paths(root, method):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if _cache_match(entry, method, row):
            result = entry if method == "qwen_v1" else entry["result"]
            probs = result["probabilities"] if method == "qwen_v1" else result
            return {"p_support": probs["SUPPORT"] if method == "qwen_v1" else probs["p_support"],
                    "p_contradict": probs["CONTRADICT"] if method == "qwen_v1" else probs["p_contradict"],
                    "p_irrelevant": probs["IRRELEVANT"] if method == "qwen_v1" else probs["p_irrelevant"],
                    "argmax_label": result["argmax_label"],
                    "selected_sentence": result.get("selected_sentence") if method == "nli_sentence" else None,
                    "cache_status": "EXISTING_CACHE", "cache_path": str(path), "model_calls": 0,
                    "cache_key": entry["cache_key"]}
    return None


def verify_ollama():
    with urlopen("http://localhost:11434/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    with urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    matches = [model for model in models if model["name"] == EXPECTED_QWEN.model]
    if version != EXPECTED_QWEN.ollama_version or len(matches) != 1 or matches[0]["digest"] != EXPECTED_QWEN.model_digest:
        raise RuntimeError("Local Ollama/model identity does not match frozen configuration")
    return {"ollama_version": version, "model": matches[0]["name"], "digest": matches[0]["digest"]}


def _nli_backends(root):
    backend = nli.load_transformers_backend(spec=nli.NLIModelSpec(), hf_cache_dir=root / "data/cache/huggingface/nli_probe", device="cpu", local_files_only=True)
    supported = control.SupportedContextNLIBackend(backend)
    if supported.context["effective_max_length"] != 8192:
        raise RuntimeError("Supported ModernBERT length no longer equals frozen 8192")
    return backend, supported


def _new_classifier(root, method):
    if method == "qwen_v1":
        verify_ollama()
        backend = OllamaTextGenerationBackend(model="qwen3:8b", think=True, seed=42, timeout=300)
        return qwen.LLMStanceClassifier(backend, cache_dir=root / "data/cache/stance/v1", model_config=EXPECTED_QWEN)
    base, supported = _nli_backends(root)
    if method == "nli_256":
        return nli.NLIStanceClassifier(base, cache_dir=root / "data/cache/stance/nli_v0")
    if method == "nli_8192":
        return nli.NLIStanceClassifier(supported, cache_dir=root / "data/cache/stance/nli_truncation_control_v1")
    return sentence.SentenceNLIClassifier(supported, cache_dir=root / "data/cache/stance/nli_sentence_v1")


def _output(root):
    return root / "outputs/stance_model_comparison/reference_60"


def infer_method(root, method):
    if method not in METHODS:
        raise ValueError("Unknown frozen stance method")
    root = Path(root).resolve()
    output = _output(root)
    output.mkdir(parents=True, exist_ok=True)
    source = root / "outputs/stance_annotation/dev_1_30/stance_reference_60.csv"
    rows = read_unlabeled_inputs(source)
    output_path = output / f"unlabeled_{method}.jsonl"
    existing = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            k = tuple(record["ids"][f] for f in KEY_FIELDS)
            if k in existing:
                raise ValueError("Duplicate checkpoint prediction")
            existing[k] = record
    classifier = None
    counts = Counter()
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            k = key(row)
            if k in existing:
                record = existing[k]
                if record["input_sha256"] != input_hash(row) or record["method"] != method:
                    raise ValueError("Checkpoint input/method mismatch")
                validate_scores(record)
                counts["checkpoint_reuse"] += 1
                continue
            cached = reusable_cache(root, method, row)
            if cached is not None:
                prediction = cached
                counts["existing_cache_hits"] += 1
            else:
                if classifier is None:
                    classifier = _new_classifier(root, method)
                inference = classifier.classify_texts(**row)
                if method == "qwen_v1":
                    record = classifier.last_record
                    prediction = {field: record[field] for field in ("p_support", "p_contradict", "p_irrelevant", "argmax_label", "cache_status", "cache_key")}
                    prediction.update(selected_sentence=None, cache_path=str(classifier.cache_dir / (record["cache_key"] + ".json")), model_calls=int(record["cache_status"] == "MISS"))
                else:
                    if inference["status"] != "SUCCESS":
                        raise RuntimeError(f"{method} inference failure on isolated row {index}: {inference['status']}")
                    prediction = {field: inference[field] for field in ("p_support", "p_contradict", "p_irrelevant", "argmax_label", "cache_status", "cache_key")}
                    prediction.update(selected_sentence=inference.get("selected_sentence"), cache_path=None,
                                      model_calls=inference["model_calls"])
                counts["new_model_calls"] += prediction["model_calls"]
            record = {"method": method, "ids": {field: row[field] for field in KEY_FIELDS},
                      "input_sha256": input_hash(row), **prediction}
            validate_scores(record)
            handle.write(canonical(record) + "\n")
            handle.flush()
            print(f"{method} {index}/60 {record['cache_status']} calls={record['model_calls']}", flush=True)
    completed = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(completed) != 60 or {tuple(item["ids"][f] for f in KEY_FIELDS) for item in completed} != {key(row) for row in rows}:
        raise ValueError("Unlabeled method checkpoint is not the exact 60-pair set")
    audit = {"method": method, "reference_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "model_input_fields": list(TEXT_FIELDS), "identifier_fields": list(KEY_FIELDS),
             "excluded_reference_column": "human_stance", "dev_31_50_accessed": False,
             "label_join_performed_during_inference": False, "row_count": len(rows), **counts,
             "total_model_calls_across_checkpoints": sum(item["model_calls"] for item in completed),
             "total_existing_cache_hits": sum(item["cache_status"] == "EXISTING_CACHE" for item in completed),
             "total_classifier_cache_hits": sum(item["cache_status"] == "HIT" for item in completed)}
    if method == "qwen_v1":
        audit["model_identity"] = vars(EXPECTED_QWEN)
    else:
        audit["model_identity"] = {"model_id": nli.MODEL_ID, "model_revision": nli.MODEL_REVISION,
                                   "hypothesis_version": nli.HYPOTHESIS_VERSION,
                                   "max_length": 256 if method == "nli_256" else 8192,
                                   "sentence_selection_version": sentence.SELECTION_VERSION if method == "nli_sentence" else None}
    (output / f"reuse_audit_{method}.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _metric(rows):
    n = len(rows)
    confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    distributions = Counter()
    entropies = []
    triplets = set()
    onehot = 0
    brier = []
    nll = []
    for row in rows:
        truth, pred = row["reference_stance"], row["predicted_stance"]
        confusion[truth][pred] += 1
        distributions[pred] += 1
        probs = [row[f"p_{name.lower()}"] for name in LABELS]
        entropies.append(-math.fsum(p * math.log(p) for p in probs if p > 0) / math.log(3))
        triplets.add(tuple(probs))
        onehot += sorted(probs) == [0, 0, 1]
        brier.append(math.fsum((p - int(name == truth)) ** 2 for p, name in zip(probs, LABELS)))
        nll.append(-math.log(probs[LABELS.index(truth)]) if probs[LABELS.index(truth)] > 0 else math.inf)
    per_class = {}
    for name in LABELS:
        tp = confusion[name][name]
        predicted = sum(confusion[truth][name] for truth in LABELS)
        actual = sum(confusion[name].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return {"n": n, "accuracy": sum(confusion[name][name] for name in LABELS) / n,
            "macro_precision": statistics.mean(x["precision"] for x in per_class.values()),
            "macro_recall": statistics.mean(x["recall"] for x in per_class.values()),
            "macro_f1": statistics.mean(x["f1"] for x in per_class.values()),
            "per_class": per_class, "confusion_matrix": confusion,
            "predicted_distribution": {name: distributions[name] for name in LABELS},
            "unique_score_triplets": len(triplets), "exact_one_hot_count": onehot,
            "mean_normalized_entropy": statistics.mean(entropies),
            "median_normalized_entropy": statistics.median(entropies),
            "multiclass_brier": statistics.mean(brier),
            "multiclass_nll": statistics.mean(nll) if all(math.isfinite(x) for x in nll) else "Infinity",
            "nll_zero_reference_score_count": sum(math.isinf(x) for x in nll)}


def evaluate(root):
    root = Path(root).resolve()
    output = _output(root)
    source = root / "outputs/stance_annotation/dev_1_30/stance_reference_60.csv"
    unlabeled = read_unlabeled_inputs(source)
    predictions = {}
    for method in METHODS:
        path = output / f"unlabeled_{method}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mapping = {tuple(record["ids"][f] for f in KEY_FIELDS): record for record in records}
        if len(records) != 60 or len(mapping) != 60:
            raise ValueError(f"{method} predictions incomplete or duplicate")
        for row in unlabeled:
            record = mapping.get(key(row))
            if record is None or record["input_sha256"] != input_hash(row) or record["method"] != method:
                raise ValueError("Prediction IDs/input hashes mismatch")
            validate_scores(record)
        predictions[method] = mapping
    # The reference labels are read only after all 240 predictions are validated.
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reference = list(csv.DictReader(handle))
    if len(reference) != 60 or Counter(row["human_stance"] for row in reference) != Counter({"SUPPORT": 10, "CONTRADICT": 5, "IRRELEVANT": 45}):
        raise ValueError("Fixed reference labels/distribution differ")
    if Counter(row["batch"] for row in reference) != Counter({"structural": 30, "directional": 30}):
        raise ValueError("Fixed batch sizes differ")
    columns = ["batch", "pair_id", "question_id", "option", "pmid", "reference_stance"]
    for method in METHODS:
        columns.extend([method + "_prediction", method + "_p_support", method + "_p_contradict",
                        method + "_p_irrelevant", method + "_correct"])
    columns.append("nli_sentence_selected_sentence")
    joined = []
    for source_row in reference:
        k = (source_row["question_id"], source_row["candidate_option_id"], source_row["pmid"])
        row = {"batch": source_row["batch"], "pair_id": source_row["pair_id"], "question_id": k[0],
               "option": k[1], "pmid": k[2], "reference_stance": source_row["human_stance"]}
        for method in METHODS:
            pred = predictions[method][k]
            row[method + "_prediction"] = pred["argmax_label"]
            for name in LABELS:
                row[method + "_p_" + name.lower()] = pred["p_" + name.lower()]
            row[method + "_correct"] = pred["argmax_label"] == source_row["human_stance"]
        row["nli_sentence_selected_sentence"] = predictions["nli_sentence"][k]["selected_sentence"]
        joined.append(row)
    metrics = {batch: {method: _metric([{"reference_stance": row["reference_stance"],
        "predicted_stance": row[method + "_prediction"],
        **{"p_" + name.lower(): row[method + "_p_" + name.lower()] for name in LABELS}}
        for row in joined if batch == "combined" or row["batch"] == batch]) for method in METHODS}
        for batch in ("combined", "structural", "directional")}
    all_right = sum(all(row[m + "_correct"] for m in METHODS) for row in joined)
    all_wrong = sum(all(not row[m + "_correct"] for m in METHODS) for row in joined)
    disagreement = sum(len({row[m + "_prediction"] for m in METHODS}) > 1 for row in joined)
    contradiction_patterns = Counter(
        (row["reference_stance"], method, row[method + "_prediction"])
        for row in joined for method in METHODS
        if row["reference_stance"] == "CONTRADICT" or row[method + "_prediction"] == "CONTRADICT")
    summary = {"reference_type": "development reference, not independently adjudicated clinical gold standard",
               "stance_order": list(LABELS), "method_order": list(METHODS), "metrics": metrics,
               "pair_summary": {"all_four_correct": all_right, "all_four_incorrect": all_wrong,
                   "method_disagreement_pairs": disagreement,
                   "contradiction_confusion_patterns": [
                       {"reference": t, "method": m, "prediction": p, "count": c}
                       for (t, m, p), c in sorted(contradiction_patterns.items())]},
               "soft_score_note": "NLI label scores and Qwen-reported distributions are not established calibrated probabilities. Brier/NLL are development diagnostics only. Zero assigned score for the reference class yields mathematical infinite NLL; no clipping was applied.",
               "macro_note": "Macro precision, recall, and F1 average all three fixed stance classes; undefined class precision/recall is 0.",
               "reference_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "predictions_60.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(joined)
    for batch, methods in metrics.items():
        for method, stats in methods.items():
            path = output / f"confusion_{batch}_{method}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["reference\\prediction", *LABELS])
                for truth in LABELS:
                    writer.writerow([truth, *[stats["confusion_matrix"][truth][p] for p in LABELS]])
    cache_audits = {method: json.loads((output / f"reuse_audit_{method}.json").read_text(encoding="utf-8")) for method in METHODS}
    gpu_sync_path = output / "gpu_sync_audit.json"
    if gpu_sync_path.exists():
        gpu_sync = json.loads(gpu_sync_path.read_text(encoding="utf-8"))
        if gpu_sync.get("status") != "VERIFIED_AND_SYNCHRONIZED" or gpu_sync.get("judgments") != 60:
            raise ValueError("Qwen GPU synchronization audit is incomplete")
        cache_audits["qwen_v1"]["gpu_execution_audit"] = gpu_sync
    (output / "cache_reuse_audit.json").write_text(json.dumps(cache_audits, indent=2) + "\n", encoding="utf-8")
    leakage = {"inference_input_fields": list(TEXT_FIELDS), "inference_identifier_fields": list(KEY_FIELDS),
               "projected_input_row_count": 60, "reference_column_removed_before_inference": "human_stance",
               "reference_labels_joined_after_all_predictions_complete": True,
               "label_column_in_model_input": False, "other_options_in_model_input": False,
               "static_qwen_prompt_defines_class_names": True,
               "dev_31_50_accessed": False,
               "input_hashes_verified_for_all_240_predictions": True}
    (output / "model_input_leakage_audit.json").write_text(json.dumps(leakage, indent=2) + "\n", encoding="utf-8")
    lines = ["# Development stance comparison: fixed reference 60", "",
             "The 60 reference labels support development model selection. They are not an independently adjudicated clinical gold standard.", "",
             "Class order in confusion files: SUPPORT, CONTRADICT, IRRELEVANT. Macro metrics use all three classes.",
             "Scores are not established calibrated probabilities; Brier and NLL are descriptive development diagnostics.", ""]
    if gpu_sync_path.exists():
        lines += [
            f"Qwen execution audit: {gpu_sync['actual_total_qwen_model_calls']} actual model calls "
            f"({gpu_sync['local_cpu_completed_before_handoff']} local CPU and {gpu_sync['gpu_new_model_calls']} Kaggle GPU). "
            f"{gpu_sync['gpu_duplicate_calls_due_to_omitted_exact_local_cache']} GPU calls repeated valid local cache entries "
            "omitted from the upload bundle; the original exact cache predictions were retained for evaluation. "
            f"{gpu_sync['gpu_duplicate_score_differences']} repeated output(s) had a different score triplet. "
            "See `gpu_sync_audit.json` and `gpu_raw_unlabeled_qwen_v1.jsonl`.", ""]
    for batch in ("combined", "structural", "directional"):
        lines += [f"## {batch} ({metrics[batch]['qwen_v1']['n']} pairs)", "",
                  "| Method | Accuracy | Macro F1 | Macro precision | Macro recall | SUPPORT F1 | CONTRADICT F1 | IRRELEVANT F1 | Predicted S/C/I | Brier | NLL |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for method in METHODS:
            item = metrics[batch][method]
            per = item["per_class"]
            dist = item["predicted_distribution"]
            lines.append(f"| {method} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | {item['macro_precision']:.4f} | {item['macro_recall']:.4f} | {per['SUPPORT']['f1']:.4f} | {per['CONTRADICT']['f1']:.4f} | {per['IRRELEVANT']['f1']:.4f} | {dist['SUPPORT']}/{dist['CONTRADICT']}/{dist['IRRELEVANT']} | {item['multiclass_brier']:.4f} | {item['multiclass_nll']} |")
        lines += [""]
    lines += ["## Soft-score diagnostics (combined 60)", "",
              "| Method | Unique exact S/C/I triplets | Exact one-hot | Mean normalized entropy | Median normalized entropy |",
              "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        item = metrics["combined"][method]
        lines.append(f"| {method} | {item['unique_score_triplets']} | {item['exact_one_hot_count']} | {item['mean_normalized_entropy']:.4f} | {item['median_normalized_entropy']:.4f} |")
    lines += [""]
    lines += ["## Pair agreement", "", f"All four correct: {all_right}. All four incorrect: {all_wrong}. At least two methods disagree: {disagreement}.", "",
              "## CONTRADICT confusion patterns", "",
              "| Reference | Method | Prediction | Count |", "|---|---|---|---:|"]
    for item in summary["pair_summary"]["contradiction_confusion_patterns"]:
        lines.append(f"| {item['reference']} | {item['method']} | {item['prediction']} | {item['count']} |")
    lines += ["", "See `predictions_60.csv` for pair results, `confusion_*.csv` for matrices, and the audit JSON files for cache and input isolation.", ""]
    (output / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=(*METHODS, "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root) if args.method == "evaluate" else infer_method(args.root, args.method)
    print(json.dumps(result if args.method != "evaluate" else result["pair_summary"], indent=2))


if __name__ == "__main__":
    main()
