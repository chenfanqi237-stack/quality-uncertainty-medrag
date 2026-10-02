"""Sentence-level diagnostic on the exact existing 18 development pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import nli_stance as nli
from . import nli_truncation_control as control
from . import stance_nli_probe as original
from .sentence_nli import SentenceNLIClassifier, CACHE_NAMESPACE, VERSION

FULL_CONTEXT_RUN = control.OUTPUT_RELATIVE + "/control-20260929T151119787692Z"
OUTPUT_RELATIVE = "outputs/stance_nli_probe/dev_1_30/modernbert_mednli_sentence_diagnostic_18"


def _summary(rows):
    valid = [row for row in rows if row["status"] == "SUCCESS"]
    counts=Counter(row["argmax_label"] for row in valid)
    entropies=[row["normalized_entropy"] for row in valid]
    buckets=Counter("<0.05" if h<.05 else "0.05-0.25" if h<.25 else "0.25-0.50" if h<=.50 else ">0.50" for h in entropies)
    triplets={tuple(round(row["nli_label_scores"][name],6) for name in nli.STANCE_ORDER) for row in valid}
    return {"valid_outputs":len(valid),"failures":len(rows)-len(valid),
        "argmax_counts":{**{name:counts[name] for name in nli.STANCE_ORDER},"UNRESOLVED":counts[None]},
        "contradiction_predictions":counts["CONTRADICT"],"unique_score_triplets_rounded_6":len(triplets),
        "exact_one_hot_count":sum(sorted(row["nli_label_scores"].values())==[0,0,1] for row in valid),
        "entropy_buckets":{name:buckets[name] for name in ("<0.05","0.05-0.25","0.25-0.50",">0.50")},
        "entropy_bucket_boundaries":"[0,0.05), [0.05,0.25), [0.25,0.50], (0.50,1]",
        "mean_entropy":statistics.mean(entropies) if entropies else None,
        "median_entropy":statistics.median(entropies) if entropies else None,
        "min_entropy":min(entropies) if entropies else None,"max_entropy":max(entropies) if entropies else None}


def load_controls(root):
    source=root/control.PREVIOUS_RUN_RELATIVE
    rows,pairs,old,old_identity,hashes=control.load_previous_run(source)
    full_dir=root/FULL_CONTEXT_RUN
    full=control._lines(full_dir/"results.jsonl")
    full_identity=control._load(full_dir/"model_metadata.json")
    if len(full)!=18 or tuple(original.key(r) for r in full)!=original.DIAGNOSTIC_KEYS:
        raise ValueError("Full-context control keys/order differ")
    if full_identity["tokenizer_settings"]["max_length"]!=8192:
        raise ValueError("Expected the completed supported-context control")
    for previous,current in zip(old,full):
        if current["status"]!="SUCCESS" or current["pair"]!=previous["pair"] or current["input_sha256"]!=previous["input_sha256"]:
            raise ValueError("Control pair texts/hashes differ")
        if current["model_metadata"]!=full_identity:
            raise ValueError("Full-context model identity differs between rows")
        nli.score_result(current["raw_logits"],current["softmax_scores_in_model_label_order"],old_identity["verified_label_mapping"])
    for name in ("model_revision","resolved_model_revision","tokenizer_revision","resolved_tokenizer_revision","verified_label_mapping","snapshot_file_sha256","runtime_versions","hypothesis_version"):
        if old_identity[name]!=full_identity[name]:
            raise ValueError("Control model/tokenizer/construction identity differs")
    hashes.update({"full_context_"+name:hashlib.sha256((full_dir/name).read_bytes()).hexdigest()
                   for name in ("inputs.jsonl","pairs.jsonl","results.jsonl","model_metadata.json")})
    if (full_dir/"inputs.jsonl").read_bytes()!=(source/"inputs.jsonl").read_bytes() or (full_dir/"pairs.jsonl").read_bytes()!=(source/"pairs.jsonl").read_bytes():
        raise ValueError("Both controls must contain the exact same input and pair bytes")
    return rows,pairs,old,full,full_identity,hashes


def comparison_markdown(rows,summary):
    def scores(row):
        return "("+", ".join(f'{row["nli_label_scores"][name]:.6f}' for name in nli.STANCE_ORDER)+")" if row["status"]=="SUCCESS" else "N/A"
    lines=["# Sentence-level ModernBERT development diagnostic","",
        "Triplet order: SUPPORT, CONTRADICT, IRRELEVANT. No independent gold stance labels; no accuracy or superiority claim.","",
        "## Summary","","```json",json.dumps(summary,indent=2),"```","",
        "## Exact same 18 pairs","",
        "| question_id | option | PMID | 256-token S/C/I | 8192 full-abstract S/C/I | sentence S/C/I | argmax 256 / full / sentence | entropy 256 / full / sentence | selected source/index | selected sentence |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        a,b=row["previous_256"],row["previous_8192"]
        labels=" / ".join(str(r.get("argmax_label","N/A")) for r in (a,b,row))
        entropy=" / ".join(f'{r["normalized_entropy"]:.6f}' if r["status"]=="SUCCESS" else "N/A" for r in (a,b,row))
        selected=row.get("selected_sentence","N/A").replace("|","\\|").replace("\n"," ").replace("\r"," ")
        values=[row["question_id"],row["candidate_option_id"],row["evidence_doc_id"],scores(a),scores(b),scores(row),labels,entropy,
            f'{row.get("selected_source","N/A")}/{row.get("selected_sentence_index","N/A")}',selected]
        lines.append("| "+" | ".join(str(v) for v in values)+" |")
    return "\n".join(lines)+"\n"


def run_probe(*,root,backend_loader=nli.load_transformers_backend):
    root=Path(root).resolve()
    rows,pairs,old,full,full_identity,hashes=load_controls(root)
    frozen=control.preservation_snapshot(root)
    output=root/OUTPUT_RELATIVE/datetime.now(timezone.utc).strftime("sentence-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False)
    source=root/control.PREVIOUS_RUN_RELATIVE
    for name in ("inputs.jsonl","pairs.jsonl"):
        (output/name).write_bytes((source/name).read_bytes())
    report={"status":"PREPARED","classifier_version":VERSION,"control_source_sha256":hashes,
        "protected_existing_files_sha256":frozen,"model_calls":0,"results":[]}
    original.save_json(output/"report.json",report)
    backend=control.SupportedContextNLIBackend(backend_loader(spec=nli.NLIModelSpec(),
        hf_cache_dir=root/"data/cache/huggingface/nli_probe",device="cpu",local_files_only=True))
    for name in ("model_revision","resolved_model_revision","tokenizer_revision","resolved_tokenizer_revision",
                 "verified_label_mapping","snapshot_file_sha256","runtime_versions","hypothesis_version","premise_version","seed","model_dtype","logit_softmax_dtype","device"):
        if backend.identity[name]!=full_identity[name]:
            raise nli.NLIProbeError("MODEL_IDENTITY_FAILURE","Frozen ModernBERT identity/settings differ")
    if backend.identity["tokenizer_settings"]!=full_identity["tokenizer_settings"]:
        raise nli.NLIProbeError("CONTEXT_LIMIT_FAILURE","Use the exact supported-context tokenizer settings")
    classifier=SentenceNLIClassifier(backend,cache_dir=root/"data/cache/stance"/CACHE_NAMESPACE)
    original.save_json(output/"model_metadata.json",classifier.backend.identity)
    print("Verified model revision:",backend.identity["resolved_model_revision"],flush=True)
    print("Actual supported maximum:",backend.context["effective_max_length"],flush=True)
    print("Verified labels:",nli.canonical(backend.label_mapping),flush=True)
    results,audits=[],[]
    for row,a,b in zip(rows,old,full):
        result=classifier.classify_texts(**row)
        result.update(previous_256=a,previous_8192=b)
        results.append(result)
        audits.append({**{name:row[name] for name in nli.ID_FIELDS},"permitted_source_fields":list(nli.INPUT_FIELDS),
            "hypothesis_sha256":nli.text_hash(result["hypothesis"]),"sentence_count":result["sentence_count"],
            "model_calls":result["model_calls"],"cache_status":result["cache_status"],
            "gold_or_other_options_exposed":False,"model_input_fields":["premise","hypothesis"],
            "sentence_judgments":[{"sentence_index":s["sentence_index"],"source":s["source"],
                "premise_sha256":nli.text_hash(s["text"]),"cache_key":s["cache_key"],"cache_status":s["cache_status"],
                "model_calls":s["model_calls"],"hypothesis_sha256":nli.text_hash(s["pair"]["hypothesis"])} for s in result["sentences"]]})
        report.update(status="RUNNING",results=results,model_calls=sum(r["model_calls"] for r in results))
        original.save_json(output/"report.json",report)
        original.save_jsonl(output/"results.jsonl",results)
        original.save_json(output/"request_audit.json",audits)
    original.verify_protected(root,frozen)
    valid=[r for r in results if r["status"]=="SUCCESS"]
    sources=Counter(r["selected_source"] for r in valid)
    summary={"judgments":18,"full_abstract_256":_summary(old),"full_abstract_8192":_summary(full),
        "sentence_level":_summary(results),"selected_sentence_sources":{"title":sources["title"],"abstract":sources["abstract"]},
        "total_sentence_judgments":sum(r["sentence_count"] for r in results),
        "actual_model_calls":sum(r["model_calls"] for r in results),
        "sentence_pair_truncation_count":sum(s["truncation"]["truncation_occurred"] for r in results for s in r["sentences"] if s["status"]=="SUCCESS"),
        "effective_max_length":backend.context["effective_max_length"],"model_id":nli.MODEL_ID,"model_revision":nli.MODEL_REVISION,
        "hypothesis_version":nli.HYPOTHESIS_VERSION,"sentence_selection":"Highest max(SUPPORT, CONTRADICT); earliest index for an exact tie; preserve full selected distribution",
        "old_outputs_and_caches_byte_identical":True,"dev_31_50_used":False,"gold_or_other_options_exposed":False,
        "accuracy_evaluated":False,"superiority_claimed":False,"quality_scoring_run":False,"aggregation_run":False,"answer_prediction_run":False}
    report.update(status="COMPLETE" if len(valid)==18 else "COMPLETE_WITH_FAILURES",summary=summary,
                  protected_existing_files_byte_identical=True)
    original.save_json(output/"report.json",report)
    original.save_json(output/"summary.json",summary)
    (output/"comparison.md").write_text(comparison_markdown(results,summary),encoding="utf-8")
    return report,output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    args=parser.parse_args(argv)
    report,output=run_probe(root=args.project_root)
    print(json.dumps(report["summary"],indent=2))
    print("Execution status:",report["status"])
    print("Outputs:",output)
    return 0 if report["status"]=="COMPLETE" else 2


if __name__=="__main__":
    raise SystemExit(main())
