"""Exactly six exposed examples; no MedQA source loading, retrieval or scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .llm_stance import MODEL_INPUT_FIELDS, StanceModelConfig
from .llm_stance_logprob import (CLASSIFIER_VERSION, LOGPROB_PROMPT, LogprobStanceClassifier,
                                NativeOllamaLogprobBackend, canonical, normalized_entropy,
                                text_hash, validate_texts)

SELECTED_KEYS = (
    ("medqa-us-dev-000004", "A", "40894990"),
    ("medqa-us-dev-000002", "C", "40752916"),
    ("medqa-us-dev-000008", "C", "38986844"),
    ("medqa-us-dev-000010", "A", "30676481"),
    ("medqa-us-dev-000004", "B", "39996184"),
    ("medqa-us-dev-000004", "C", "35695404"),
)
ID_FIELDS = ("question_id", "candidate_option_id", "evidence_doc_id")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def key(row):
    return tuple(row[field] for field in ID_FIELDS)


def prepare_selection(inputs_path, v1_results_path, directory):
    """Read only the previous 18-row diagnostic; never read MedQA gold/source data."""
    inputs = {key(row): row for row in load_jsonl(inputs_path)}
    previous = {key(row): row for row in load_jsonl(v1_results_path)}
    selected, references = [], []
    for judgment in SELECTED_KEYS:
        row, old = inputs[judgment], previous[judgment]
        if set(row) != set(MODEL_INPUT_FIELDS) | set(ID_FIELDS):
            raise ValueError("Unexpected input fields in existing diagnostic")
        validate_texts({field: row[field] for field in MODEL_INPUT_FIELDS})
        if old["status"] != "SUCCESS" or old["classifier_version"] != "llm-medical-stance-v1":
            raise ValueError("Expected completed frozen v1 reference")
        if old["model_metadata"] != asdict(StanceModelConfig()):
            raise ValueError("Reference model settings do not match the probe")
        if any(text_hash(row[field]) != old["input_sha256"][field] for field in MODEL_INPUT_FIELDS):
            raise ValueError("Reference and probe texts differ")
        selected.append(row)
        references.append({field: old[field] for field in (*ID_FIELDS, "p_support", "p_contradict",
            "p_irrelevant", "argmax_label", "normalized_entropy", "cache_key", "model_metadata",
            "prompt_template_sha256", "classifier_version")})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = "".join(canonical(row) + "\n" for row in selected)
    (directory / "inputs.jsonl").write_bytes(data.encode("utf-8"))
    save_json(directory / "v1_reference.json", references)
    selection = {"selected_keys": [list(x) for x in SELECTED_KEYS], "judgments": 6,
                 "selection_basis": "Existing v1 SUPPORT/IRRELEVANT outcomes including all three non-one-hot SUPPORT examples; not ground truth",
                 "input_sha256": text_hash(data), "model_metadata": asdict(StanceModelConfig()),
                 "gold_or_other_options_in_model_input": False, "dev_31_50_used": False}
    save_json(directory / "selection.json", selection)
    return selection


def validate_selection(rows, references):
    if len(rows) != 6 or tuple(key(row) for row in rows) != SELECTED_KEYS:
        raise ValueError("Only the six fixed, already exposed judgments are allowed")
    if len(references) != 6 or tuple(key(row) for row in references) != SELECTED_KEYS:
        raise ValueError("V1 reference keys differ from fixed selection")
    for row in rows:
        if set(row) != set(MODEL_INPUT_FIELDS) | set(ID_FIELDS):
            raise ValueError("Unexpected model input or metadata field")
        validate_texts({field: row[field] for field in MODEL_INPUT_FIELDS})


def summarize(results):
    successful = [r for r in results if r["status"] == "SUCCESS"]
    triplets = Counter(tuple(r["normalized_label_likelihoods"][label] for label in ("SUPPORT", "CONTRADICT", "IRRELEVANT")) for r in successful)
    entropies = [r["normalized_entropy"] for r in successful]
    labels = Counter(r["argmax_label"] for r in successful)
    statuses = Counter(r["status"] for r in results)
    bins = Counter("<0.05" if h < .05 else "0.05-0.25" if h < .25 else "0.25-0.50" if h <= .5 else ">0.50" for h in entropies)
    return {"judgments": len(results), "successful_logprob_results": len(successful),
            "actual_model_calls": sum(r["generation_attempts"] for r in results), "retries": 0,
            "cache_hits": sum(r["cache_status"] == "HIT" for r in results),
            "status_counts": dict(statuses), "LOGPROB_INCOMPLETE": statuses["LOGPROB_INCOMPLETE"],
            "final_token_alignment_failures": statuses["FINAL_TOKEN_ALIGNMENT_FAILURE"],
            "unique_normalized_likelihood_triplet_count": len(triplets),
            "unique_normalized_likelihood_triplets": [{"SUPPORT": t[0], "CONTRADICT": t[1], "IRRELEVANT": t[2], "count": n} for t, n in sorted(triplets.items())],
            "exact_one_hot_count": sum(t.count(1.) == 1 and t.count(0.) == 2 for t in triplets.elements()),
            "argmax_counts": {**{label: labels[label] for label in ("SUPPORT", "CONTRADICT", "IRRELEVANT")}, "UNRESOLVED": labels[None]},
            "normalized_entropy_values": entropies,
            "normalized_entropy_bins": {b: bins[b] for b in ("<0.05", "0.05-0.25", "0.25-0.50", ">0.50")},
            "all_three_tokens_present_count": sum(r.get("all_label_tokens_present") is True for r in results),
            "hard_argmax_agreement_count": sum(r.get("hard_argmax_agrees") is True for r in successful),
            "technical_feasibility": "DEMONSTRATED_FOR_ALL_SIX" if len(successful) == 6 else "NOT_DEMONSTRATED_FOR_ALL_SIX",
            "values_are_calibrated_probabilities": False, "accuracy_or_superiority_claimed": False,
            "pubmed_requests": 0, "gold_or_other_options_in_model_input": False, "dev_31_50_used": False,
            "v1_regenerated": False, "retrieval_quality_aggregation_or_uncertainty_modified": False}


def markdown_report(results, summary):
    lines = ["# Experimental stance logprob-v0 feasibility probe", "",
             "Likelihood triplets are normalized over the three label tokens only; they are not calibrated probabilities. No correctness or superiority claim is made.", "",
             "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "",
             "## Side-by-side (triplet order: SUPPORT, CONTRADICT, IRRELEVANT)", "",
             "| question_id | option | PMID | v1 triplet | v1 entropy | v1 argmax | v0 normalized label likelihoods | v0 entropy | v0 argmax | agrees | all labels | status |",
             "|---|---|---|---|---:|---|---|---:|---|---|---|---|"]
    for r in results:
        old = r["v1_reference"]
        triplet = r.get("normalized_label_likelihoods")
        likelihoods = str(tuple(triplet[label] for label in ("SUPPORT", "CONTRADICT", "IRRELEVANT"))) if triplet else "N/A"
        vals = [r["question_id"], r["candidate_option_id"], r["evidence_doc_id"],
                str((old["p_support"], old["p_contradict"], old["p_irrelevant"])), f'{old["normalized_entropy"]:.4f}',
                old["argmax_label"], likelihoods, f'{r["normalized_entropy"]:.6f}' if triplet else "N/A",
                r.get("argmax_label"), r["hard_argmax_agrees"], r.get("all_label_tokens_present"), r["status"]]
        lines.append("| " + " | ".join(str(x) for x in vals) + " |")
    lines.extend(["", "## Raw class-token log probabilities", "",
                  "| question_id | option | PMID | lp1 | lp2 | lp3 | final code | alignment |", "|---|---|---|---:|---:|---:|---|---|"])
    for r in results:
        lp = r.get("raw_label_logprobs", {})
        vals = [r["question_id"], r["candidate_option_id"], r["evidence_doc_id"], *[lp.get(c, "MISSING") for c in "123"],
                r.get("final_class_code", "N/A"), r.get("alignment", {}).get("strategy", "FAILED")]
        lines.append("| " + " | ".join(str(x) for x in vals) + " |")
    return "\n".join(lines) + "\n"


def run_probe(*, root, selection_dir, output_dir, base_url="http://localhost:11434"):
    # This helper checks runtime identity/GPU and loads weights without text generation.
    from .stance_smoke import gpu_preflight, _required_identity
    from .cloud_runtime import ollama_identity, verify_freeze
    root, selection_dir, output_dir = Path(root), Path(selection_dir), Path(output_dir)
    rows = load_jsonl(selection_dir / "inputs.jsonl")
    references = json.loads((selection_dir / "v1_reference.json").read_text(encoding="utf-8"))
    selection = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    validate_selection(rows, references)
    if hashlib.sha256((selection_dir / "inputs.jsonl").read_bytes()).hexdigest() != selection["input_sha256"]:
        raise ValueError("Selected input bytes differ from frozen selection")
    if output_dir.exists():
        raise ValueError("Refusing existing run output directory")
    frozen = verify_freeze(root, root / "cloud/research_freeze.json")
    # Also protect every existing v1 source/output/cache file in the cloud project.
    protected = {}
    for rel in ("src/quality_uncertainty_medrag/llm_stance.py", "data/cache/stance/v1", "outputs/stance_smoke", "outputs/stance_probability_inspection"):
        source = root / rel
        paths = [source] if source.is_file() else source.rglob("*")
        protected.update({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
    runtime = gpu_preflight(base_url)
    backend = NativeOllamaLogprobBackend(base_url=base_url)
    classifier = LogprobStanceClassifier(backend, cache_dir=root / "data/cache/stance/logprob_v0")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "classifier_prompt.txt").write_text(LOGPROB_PROMPT, encoding="utf-8")
    for filename in ("inputs.jsonl", "v1_reference.json", "selection.json"):
        (output_dir / filename).write_bytes((selection_dir / filename).read_bytes())
    results, audits = [], []
    report = {"status": "RUNNING", "classifier_version": CLASSIFIER_VERSION, "runtime": runtime,
              "model_metadata": asdict(StanceModelConfig()), "input_sha256": selection["input_sha256"],
              "prompt_sha256": text_hash(LOGPROB_PROMPT), "frozen_research_files": frozen,
              "protected_v1_files": protected, "results": results}
    for row, old in zip(rows, references):
        _required_identity(ollama_identity(base_url))
        result = classifier.classify_texts(**row)
        result.update(candidate_option_text=row["candidate_option_text"], evidence_title=row["evidence_title"],
                      v1_reference=old, hard_argmax_agrees=(result["argmax_label"] == old["argmax_label"]) if result["status"] == "SUCCESS" else None)
        results.append(result)
        if result["cache_status"] == "MISS":
            audits.append({**{field: row[field] for field in ID_FIELDS}, **backend.last_request_audit})
        save_json(output_dir / "report.json", report)
        save_json(output_dir / "request_audit.json", audits)
        (output_dir / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results), encoding="utf-8")
    verify_freeze(root, root / "cloud/research_freeze.json")
    if any(hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected for rel, expected in protected.items()):
        raise ValueError("Protected v1 files changed")
    summary = summarize(results)
    summary.update(model_load_requests=runtime["model_load_requests"], protected_v1_files_byte_identical=True,
                   full_model_digest=StanceModelConfig().model_digest)
    report.update(status="COMPLETE" if summary["successful_logprob_results"] == 6 else "COMPLETE_WITH_EXPLICIT_FAILURES", summary=summary)
    save_json(output_dir / "report.json", report)
    save_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(markdown_report(results, summary), encoding="utf-8")
    archive = root.parent / (output_dir.name + "-results.zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file(): handle.write(path, path.relative_to(root).as_posix())
        for row in results:
            path = root / "data/cache/stance/logprob_v0" / (row["cache_key"] + ".json")
            handle.write(path, path.relative_to(root).as_posix())
    return report, archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("MEDRAG_PROJECT_ROOT", ".")))
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report, archive = run_probe(root=args.project_root, selection_dir=args.selection_dir, output_dir=args.output_dir)
    print(markdown_report(report["results"], report["summary"]))
    print("Archive:", archive)


if __name__ == "__main__":
    main()
