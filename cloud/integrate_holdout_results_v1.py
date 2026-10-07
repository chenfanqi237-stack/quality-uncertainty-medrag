"""CPU-only post-holdout verification/import. No generation or scientific changes.

The completed private transfer is authoritative; existing frozen modules are
used only to reprove stochastic identities and reproduce blind scores. Gold is
opened only after that comparison succeeds. New outputs never replace sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import NormalDist

sys.dont_write_bytecode = True
VERSION = "post-holdout-generation-free-integration-v1"
TRANSFER_SHA = "eea8c0f8df00147f34da1344f9971e95d40c7e4dd422edee5e3cc4eb56660f75"
BLIND_SHA = "25ef5e04db22d400c325f5ecf98376665709b7b21b7f5e7da5e1187416f506f0"
VANILLA_SHA = "52eada0e590fa229978399ec836ab3c48207a24218a7954f2ec40df34cea9e95"
GOLD_SHA = "b6bbb9dc80b07040627930d83acce39251a3348151a603ba1ceb6d03e038b9ce"
DERIVED_SHA = "eae8db4df48f3422d6bff7cc4a47f54feedaea5fb41b423ab8a97e64005ca9f1"
ADAPTER_SHA = "6a836ccc38dac26bc6764e1bfa4b36619b3b4f0e1c0694c71370e76a4af3e0bc"
OP = "operation_record/stochastic_completion_amended_v1/"
GP = "operation_record/holdout_gold_reader_compat_v1/"
PRED = OP + "freeze/primary_abcd/holdout_blind_predictions.json"
QIDS = tuple(f"medqa-us-dev-{i:06d}" for i in range(31, 51))
BASE = Path("outputs/holdout_evaluation/dev31_50_holdout_v1")
CONDITIONS = ("original_quality", "quality_disabled")
METHODS = [("A", "quality_invariant")] + [(m, c) for c in CONDITIONS for m in "BCD"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def new_file(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)


def jread(z, name):
    return json.loads(z.read(name))


def safe_names(names):
    require(len(names) == len(set(names)), "Duplicate ZIP member")
    require(len(names) == len({n.casefold() for n in names}), "Case-colliding ZIP members")
    for n in names:
        p = PurePosixPath(n)
        require(not p.is_absolute() and ".." not in p.parts and "\\" not in n
                and not re.match(r"^[A-Za-z]:", n), "Unsafe member path")


def verify_zip(z, manifest_name, *, nested=False):
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    safe_names(names)
    require(z.testzip() is None, "ZIP CRC failure")
    m = jread(z, manifest_name)
    require(set(names) == set(m["files"]) | {manifest_name}, "ZIP manifest inventory differs")
    for n, wanted in m["files"].items():
        require(digest(z.read(n)) == wanted, "ZIP member hash differs: " + n)
    return m


def close_numbers(actual, expected, location=""):
    """Strict semantic comparison, with floating-point tolerance only."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), "Expected object: " + location)
        for k, v in expected.items():
            require(k in actual, "Missing field: " + location + k)
            close_numbers(actual[k], v, location + k + ".")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "List differs: " + location)
        for a, b in zip(actual, expected):
            close_numbers(a, b, location)
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and abs(actual - expected) <= 1e-12,
                "Numerical disagreement: " + location)
    else:
        require(actual == expected, "Value differs: " + location)


def utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def chronology(z):
    verified = jread(z, OP + "dual_freeze_verification_pass.json")
    first = jread(z, OP + "first_gold_access.json")
    require(verified["gold_accessed"] is False and verified["primary_blind_freeze"] == "VERIFIED"
            and verified["vanilla_incomplete_state_freeze"] == "VERIFIED", "Dual freeze not verified")
    require(first["verified_dual_freeze_receipt_sha256"] == digest(z.read(OP + "dual_freeze_verification_pass.json")), "Gold audit receipt hash differs")
    for r in (verified, first):
        require(r["primary_blind_sha256"] == BLIND_SHA and r["vanilla_incomplete_freeze_sha256"] == VANILLA_SHA, "Audit freeze identity differs")
    require(utc(verified["verified_utc"]) < utc(first["first_access_utc"]), "Premature recorded gold access")
    complete = jread(z, OP + "evaluation_complete.json")
    stop = jread(z, GP + "STOP_compat_evaluate_20261007T120637Z.json")
    require(complete["evaluation_sha256"] == digest(z.read(GP + "evaluation_v1/evaluation.json")), "Core evaluation receipt hash differs")
    require(utc(complete["completed_utc"]) < utc(stop["created_utc"]), "Auxiliary STOP ordering differs")
    return {"freeze_verified_utc": verified["verified_utc"], "first_gold_access_utc": first["first_access_utc"],
            "recorded_order_confirmed": True, "core_evaluation_completed_utc": complete["completed_utc"],
            "auxiliary_receipt_failed_utc": stop["created_utc"], "auxiliary_failure_after_core_completion": True,
            "scope": "Timestamp/hash-linked recorded execution; not a claim about unlogged external access"}


def verify_transfer(path, sidecar, root):
    require(file_sha(path) == TRANSFER_SHA, "Transfer SHA256 differs")
    if sidecar is not None:
        fields = Path(sidecar).read_text(encoding="utf-8-sig").split()
        require(fields and fields[0].lower() == TRANSFER_SHA, "Transfer sidecar differs")
    with zipfile.ZipFile(path) as z:
        m = verify_zip(z, "transfer_verification_manifest.json")
        require(digest(z.read(PRED)) == BLIND_SHA, "Primary freeze differs")
        require(digest(z.read(OP + "freeze/vanilla_incomplete_state_v1.zip")) == VANILLA_SHA, "Vanilla incomplete freeze differs")
        require(digest(z.read("holdout_gold_reader_compat_v1.py")) == ADAPTER_SHA, "Gold adapter differs")
        require(digest(z.read("evaluation_inputs/dev31_50_canonical_gold_v1.jsonl")) == DERIVED_SHA, "Derived evaluator input differs")
        runtime = jread(z, "Runtime_v1/runtime_manifest.json")
        compared = 0
        for n, wanted in runtime["files"].items():
            require(digest(z.read("Runtime_v1/" + n)) == wanted, "Frozen packaged runtime differs: " + n)
            local = Path(root) / n
            if local.is_file():
                require(file_sha(local) == wanted, "Frozen project file differs: " + n)
                compared += 1
            elif n.startswith(("cloud/", "src/", "docs/", "configs/", "outputs/")):
                raise ValueError("Frozen project reference missing: " + n)
        order = chronology(z)
        blind = jread(z, PRED)
        require(blind["question_ids"] == list(QIDS) and blind["gold_accessed"] is False
                and blind["question_count"] == 20 and blind["required_samples"] == 2750, "Blind scope differs")
    return {"sha256": TRANSFER_SHA, "bytes": Path(path).stat().st_size, "verified_member_hashes": len(m["files"]),
            "crc": "PASS", "frozen_runtime_files": len(runtime["files"]), "project_hashes_compared": compared,
            "chronology": order}, m


def verify_native_transport(data, signature):
    """Verify all native archives and fold the existing linear journal in memory."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        transport = verify_zip(z, "transport_manifest.json")
        require(transport["signature"] == signature and transport["complete_native_parent_history"] is True, "Native transport differs")
        journals = sorted(n for n in transport["files"] if n.startswith("native/journal/") and n.endswith(".zip"))
        folded, previous = {}, None
        for revision, name in enumerate(journals):
            blob = z.read(name)
            with zipfile.ZipFile(io.BytesIO(blob)) as member:
                m = verify_zip(member, "checkpoint_manifest.json")
                require(m["revision"] == revision and m["pilot_signature"] == signature, "Native revision/identity differs")
                require(m["kind"] == ("base" if revision == 0 else "delta") and m["parent_sha256"] == previous, "Native linear parent differs")
                for n in m["files"]:
                    b = member.read(n)
                    if n.startswith("cache/") and n.count("/") == 1 and n in folded:
                        require(folded[n] == b, "Previously valid observation changed in journal")
                    folded[n] = b
            previous = digest(blob)
        finals = [n for n in transport["files"] if n.startswith("native/final/") and n.endswith(".zip")]
        require(len(finals) == 1, "Native final inventory differs")
        with zipfile.ZipFile(io.BytesIO(z.read(finals[0]))) as final:
            fm = verify_zip(final, "checkpoint_manifest.json")
            require(fm["revision"] == len(journals) - 1 and set(folded) == set(fm["files"]), "Folded journal/final inventory differs")
            for n in fm["files"]:
                if n != "meta.json":
                    require(folded[n] == final.read(n), "Folded journal/final observation differs")
            require(json.loads(folded["meta.json"]) == json.loads(final.read("meta.json")), "Final metadata differs from native journal")
    return {"native_archives_verified": len(journals) + 1, "linear_journal_revisions": len(journals),
            "revision": fm["revision"], "head_sha256": previous, "in_memory_history_fold_matches_final": True,
            "full_native_scan_at_each_historical_revision": False}


def import_readable(path, sidecar, root, m):
    root = Path(root)
    target = root / "cloud/checkpoint/dev31_50_holdout/dev31_50_holdout_results_transfer_v1.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(file_sha(target) == TRANSFER_SHA, "Existing canonical return differs; do not overwrite")
    else:
        shutil.copyfile(path, target)
    require(file_sha(target) == TRANSFER_SHA, "Copied transfer hash differs")
    side_bytes = Path(sidecar).read_bytes() if sidecar else (TRANSFER_SHA + "  " + target.name + "\n").encode()
    side_target = Path(str(target) + ".sha256")
    if side_target.exists():
        require(side_target.read_bytes() == side_bytes, "Existing canonical sidecar differs")
    else:
        new_file(side_target, side_bytes)
    output = root / BASE / "completed_gpu_run_20261007"
    imported = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            keep = (n.startswith((OP, GP, "evaluation_inputs/")) and not (n.startswith(OP + "preserved/") or "/native_exports/" in n))
            keep |= n.startswith("operation_record/") and n.count("/") == 1
            keep |= n.count("/") == 0 and n.endswith((".md", ".json", ".py"))
            keep |= n == "docs/dev31_50_holdout_vanilla_incomplete_amendment_v1.md"
            keep |= n == m["native_archives"]["stochastic_final"]["member"] or n.startswith(m["native_archives"]["stochastic_final"]["member"] + ".")
            keep |= n == m["native_archives"]["vanilla_incomplete"]["member"] or n.startswith(m["native_archives"]["vanilla_incomplete"]["member"] + ".")
            if keep:
                data = z.read(n)
                if (output / n).exists():
                    require((output / n).read_bytes() == data, "Existing readable import differs; do not overwrite")
                else:
                    new_file(output / n, data)
                require(file_sha(output / n) == digest(data), "Readable import differs")
                imported[n] = {"sha256": digest(data), "bytes": len(data)}
    return target, output, imported


def cp_interval(k, n, tail=.025):
    """Exact binomial-tail inversion, independently coded; no SciPy dependency."""
    require(type(k) is int and type(n) is int and 0 <= k <= n, "Invalid binomial counts")
    if n == 0:
        return None
    def cdf(p, upper):
        return sum(math.comb(n, j) * p ** j * (1-p) ** (n-j) for j in range(upper + 1))
    def quantile(upper, target):
        lo, hi = 0., 1.
        for _ in range(100):
            mid = (lo + hi) / 2
            if cdf(mid, upper) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return [0. if k == 0 else quantile(k-1, 1-tail), 1. if k == n else quantile(k, tail)]


def paired_intervals(both, soft_only, hard_only, neither, answered_soft_only, answered_hard_only):
    n = both + soft_only + hard_only + neither
    ls, us = cp_interval(answered_soft_only, n, .0125)
    lh, uh = cp_interval(answered_hard_only, n, .0125)
    z = NormalDist().inv_cdf(.975)
    def wilson(p):
        den = 1 + z*z/n
        center = (p + z*z/(2*n))/den
        half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
        return center-half, center+half
    ps, ph = (both + soft_only)/n, (both + hard_only)/n
    sl, su = wilson(ps); hl, hu = wilson(ph)
    numerator = both*neither - soft_only*hard_only
    denominator = math.sqrt((both+soft_only)*(hard_only+neither)*(both+hard_only)*(soft_only+neither))
    phi = (max(numerator-n/2, 0) if numerator > 0 else numerator)/denominator if denominator else 0.
    delta = ps-ph
    lower = delta-math.sqrt(max(0., (ps-sl)**2+(hu-ph)**2-2*phi*(ps-sl)*(hu-ph)))
    upper = delta+math.sqrt(max(0., (su-ps)**2+(ph-hl)**2-2*phi*(su-ps)*(ph-hl)))
    return {"coverage": {"difference": (answered_soft_only-answered_hard_only)/n,
            "interval95": [ls-uh, us-lh], "method": "Frozen conservative marginal-bound working-model interval: two 97.5% exact CP discordance marginals, Bonferroni bounds; not exact paired CI"},
            "accuracy": {"difference": (soft_only-hard_only)/n, "interval95": [lower, upper],
            "phi_cc": phi, "method": "Supplementary pre-specified Newcombe method 10 paired approximate interval, positive-phi continuity correction"}}


def method_at(result, method, condition):
    return result["conditions"]["original_quality" if condition == "quality_invariant" else condition]["methods"][method]


def method_metrics(predictions, gold, method, condition):
    require(set(predictions) == set(gold) == set(QIDS), "N=20 scope differs")
    answered = correct = undefined = 0
    reasons, option_reasons = Counter(), Counter()
    for q in QIDS:
        require(predictions[q]["status"] == "COMPLETE", "Incomplete predictions block evaluation")
        m = method_at(predictions[q], method, condition); a = m["answer"]
        require(a["status"] in ("ANSWERED", "ABSTAIN"), "Incomplete is not abstention")
        require(set(m["option_scores"]) == set("ABCDE"), "Five option scores required")
        if a["status"] == "ANSWERED":
            require(a["selected_answer"] in "ABCDE", "Invalid answer")
            answered += 1; correct += a["selected_answer"] == gold[q]
        else:
            require(a["selected_answer"] is None, "Abstention selected an answer")
            reasons[a["reason"]] += 1
        for opt in m["option_scores"].values():
            undefined += opt["score"] is None
            require(opt["score"] is None or math.isfinite(opt["score"]), "Nonfinite score")
            option_reasons[opt["reason"]] += 1
    wrong = answered-correct
    return {"N": 20, "answered": answered, "correct": correct, "wrong_selected": wrong, "abstained": 20-answered,
            "accuracy": correct/20, "coverage": answered/20,
            "answered_accuracy": correct/answered if answered else None,
            "selective_risk": wrong/answered if answered else None,
            "accuracy_ci95": cp_interval(correct,20), "coverage_ci95": cp_interval(answered,20),
            "answered_accuracy_ci95": cp_interval(correct,answered), "selective_risk_ci95": cp_interval(wrong,answered),
            "undefined_scores": undefined, "abstention_reasons": dict(reasons), "option_diagnostic_reasons": dict(option_reasons)}


def transitions(predictions, gold, left, right, condition):
    rows = []
    for q in QIDS:
        a = method_at(predictions[q],left,condition)["answer"]
        b = method_at(predictions[q],right,condition)["answer"]
        av, bv = a["selected_answer"], b["selected_answer"]
        transition = ("both_abstain" if av is None and bv is None else "abstain_to_answer" if av is None
                      else "answer_to_abstain" if bv is None else "same_answer" if av == bv else "answer_change")
        rows.append({"question_id": q, "quality_condition": condition, "source_method": left, "destination_method": right,
                     "source_answer": av or "ABSTAIN", "destination_answer": bv or "ABSTAIN", "transition": transition,
                     "source_correct_if_answered": av == gold[q] if av else None,
                     "destination_correct_if_answered": bv == gold[q] if bv else None})
    return rows


def contrast(predictions, left, right, source_condition, dest_condition=None):
    counts = Counter()
    for q in QIDS:
        a = method_at(predictions[q],left,source_condition)
        b = method_at(predictions[q],right,dest_condition or source_condition)
        counts["final_answer_or_status_changes"] += (a["answer"]["status"],a["answer"]["selected_answer"]) != (b["answer"]["status"],b["answer"]["selected_answer"])
        counts["ranking_changes"] += [r["option"] for r in a["answer"]["rankings"]] != [r["option"] for r in b["answer"]["rankings"]]
        for o in "ABCDE":
            aa,bb = a["option_scores"][o], b["option_scores"][o]
            counts["native_claim_decision_changes"] += aa["native_claim_decision"] != bb["native_claim_decision"]
            counts["undefinedness_changes"] += (aa["score"] is None) != (bb["score"] is None)
            if aa["score"] is not None and bb["score"] is not None:
                counts["jointly_defined_score_denominator"] += 1
                counts["score_changes"] += abs(aa["score"]-bb["score"]) > 1e-12
    return {k: counts[k] for k in ("final_answer_or_status_changes","ranking_changes","native_claim_decision_changes","undefinedness_changes","jointly_defined_score_denominator","score_changes")}


def read_gold_after_reproof(root, z, questions, reproof_passed):
    require(reproof_passed, "Gold forbidden before independent blind reproof")
    import holdout_common_v1 as h
    source = Path(root)/"data/processed/medqa_us_dev_50.jsonl"
    require(file_sha(source) == GOLD_SHA, "Canonical gold source SHA differs")
    lines = []
    for line in source.read_bytes().splitlines(keepends=True):
        if not line.strip():
            continue
        qid = json.loads(h.spans(line.decode("utf-8"), {"id"})["id"])
        if qid in QIDS:
            lines.append(line)
    raw = b"".join(lines)
    require(digest(raw) == DERIVED_SHA and raw == z.read("evaluation_inputs/dev31_50_canonical_gold_v1.jsonl"), "Derived source bytes differ")
    records = [json.loads(line) for line in lines]
    require([r["id"] for r in records] == list(QIDS), "Gold scope/order differs")
    qmap = {q["question_id"]: q for q in questions}
    gold = {}
    for r in records:
        require(set(r) == {"id","question","options","answer","metadata"}, "Gold source schema differs")
        require(r["answer"] in "ABCDE" and len(r["answer"]) == 1 and set(r["options"]) == set("ABCDE"), "Invalid canonical gold")
        q = qmap[r["id"]]
        require(r["question"] == q["question_stem"] and r["options"] == q["options"], "Gold/inference projection differs")
        gold[r["id"]] = r["answer"]
    return gold


def export_csv(path, rows):
    require(bool(rows), "Empty CSV requires explicit schema")
    columns = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    raw = stream.getvalue().encode("utf-8")
    new_file(path, raw)
    restored = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    require(len(restored) == len(rows), "CSV round-trip row count differs")
    for a,b in zip(restored, rows):
        require(a == {k: "" if v is None else str(v) for k,v in b.items()}, "CSV round-trip content differs")


def inventory_files(root, paths):
    root = Path(root)
    return {Path(p).relative_to(root).as_posix(): {"sha256": file_sha(p), "bytes": Path(p).stat().st_size} for p in sorted(set(Path(p) for p in paths))}


def freeze_backup(root):
    """Private immutable snapshot; no upload, manuscript edit or inference."""
    root = Path(root).resolve()
    output = root/BASE/"canonical_analysis_v1"
    analysis = json.loads((output/"holdout_analysis_manifest.json").read_bytes())
    require(analysis["verification"]["all_expected_gpu_metrics_match"] is True, "Canonical verification required")
    sources = [root/"cloud/checkpoint/dev31_50_holdout/dev31_50_holdout_results_transfer_v1.zip",
               root/"cloud/checkpoint/dev31_50_holdout/dev31_50_holdout_results_transfer_v1.zip.sha256",
               root/"docs/dev31_50_internal_holdout_protocol_v1.md", root/BASE/"pre_execution_manifest.json",
               root/"docs/research_post_holdout_freeze_v1.md", root/"docs/dev30_disagreement_review_plan_v1.md",
               root/"cloud/integrate_holdout_results_v1.py", root/"tests/test_holdout_results_integration.py",
               root/"data/processed/medqa_us_dev_50.jsonl"]
    prior_freeze = root/"outputs/research_freeze/post_dev30_pre_review_v1/freeze_manifest.json"
    if prior_freeze.is_file():
        sources.append(prior_freeze)
    sources += [p for p in output.rglob("*") if p.is_file()]
    # Current manuscript is copied byte-for-byte as an archival snapshot only.
    manuscript = root/"paper/style_pass_20261007"
    sources += [p for p in manuscript.iterdir() if p.is_file() and p.name in
                ("main.tex","Hard_versus_Soft_Stance_Aggregation_style_20261007_final.pdf")] if manuscript.exists() else []
    review = root/"outputs/aggregation_development/dev30_natural_qa_comparison_v1/blind_review_missing_v1"
    sources += [p for p in review.rglob("*") if p.is_file()]
    dev = root/"outputs/aggregation_development/dev30_natural_qa_comparison_v1"
    sources += [dev/n for n in ("comparison_manifest.json","method_metrics.csv","paired_transitions.csv","MANIFEST_SHA256.txt")]
    for p in sources:
        require(p.is_file(), "Freeze source missing: " + p.name)
    records = inventory_files(root,sources)
    for name,record in records.items():
        classification = ("manuscript" if name.startswith("paper/") else "review" if "blind_review" in name or "disagreement_review" in name
                          else "source" if name.startswith(("cloud/","tests/","docs/","data/")) and not name.endswith((".zip",".sha256")) else "results")
        public = name in ("cloud/integrate_holdout_results_v1.py","tests/test_holdout_results_integration.py","docs/research_post_holdout_freeze_v1.md")
        record.update({"classification":classification,"experiment_role": "unaltered manuscript snapshot" if classification=="manuscript" else "post-holdout reproducibility record",
                       "visibility":"public-safe candidate" if public else "private","public_safe":public})
    freeze_dir=root/"outputs/research_freeze/post_holdout_pre_review_v1"
    freeze={"version":"post-holdout-pre-independent-review-v1","created_utc":datetime.now(timezone.utc).isoformat(),
            "status":"VERIFIED; PAPER UPDATE DEFERRED","review_status":"Independent 90-pair Dev30 review PENDING",
            "manuscript_edited":False,"model_calls_added":0,"cohorts_not_pooled":True,
            "transfer_sha256":TRANSFER_SHA,"primary_blind_sha256":BLIND_SHA,"vanilla_incomplete_freeze_sha256":VANILLA_SHA,"files":records}
    new_file(freeze_dir/"freeze_manifest.json",encoded(freeze))
    manifest_path=freeze_dir/"freeze_manifest.json"
    backup=root/"cloud/checkpoint/post_holdout_freeze/post_holdout_pre_review_backup_v1.zip"
    require(not backup.exists(), "NEW backup required; do not overwrite")
    backup.parent.mkdir(parents=True,exist_ok=True)
    # The original transfer contains the full runtime, readable GPU provenance,
    # scientific inputs and native parent chain. Do not duplicate historical ZIPs.
    selected=sorted(set(sources+[manifest_path]))
    ledger=inventory_files(root,selected)
    with zipfile.ZipFile(backup,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in selected:
            name=p.relative_to(root).as_posix()
            z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix==".zip" else zipfile.ZIP_DEFLATED)
        z.writestr("BACKUP_MANIFEST_SHA256.json",encoded({"version":VERSION,"files":{n:r["sha256"] for n,r in ledger.items()}}))
    restore=root/".tmp"/("holdout_backup_restore_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    require(not restore.exists(), "NEW restore verification directory required")
    with zipfile.ZipFile(backup) as z:
        verify_zip(z,"BACKUP_MANIFEST_SHA256.json")
        for n in ledger:
            with z.open(n) as incoming:
                target=restore/n
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming,outgoing,1024*1024)
            require(file_sha(target)==ledger[n]["sha256"],"Extracted backup restore hash differs")
    backup_sha=file_sha(backup)
    new_file(str(backup)+".sha256",(backup_sha+"  "+backup.name+"\n").encode())
    receipt={"backup_path":backup.relative_to(root).as_posix(),"sha256":backup_sha,"bytes":backup.stat().st_size,
             "crc":"PASS","internal_manifest":"PASS","all_extracted_member_hashes":"PASS","members":len(ledger)+1,
             "freeze_manifest_sha256":file_sha(manifest_path),"restore_directory":restore.relative_to(root).as_posix(),
             "off_machine_backup_performed":False,"manuscript_modified":False}
    new_file(freeze_dir/"backup_verification.json",encoded(receipt))
    for name,record in records.items():
        require(file_sha(root/name)==record["sha256"],"Frozen original changed during backup")
    return receipt


def run(root, transfer, sidecar):
    root = Path(root).resolve()
    sys.path.insert(0, str(root/"cloud"))
    import holdout_common_v1 as h
    import holdout_stance_v1 as stance
    import holdout_evaluation_v1 as frozen_evaluation
    from dev30_disagreement_review_v1 import verify_results
    readonly = {p: file_sha(p) for p in [root/"README.md", root/BASE/"pre_execution_manifest.json", root/"docs/dev31_50_internal_holdout_protocol_v1.md"]}
    verification, metadata = verify_transfer(transfer,sidecar,root)
    print("TRANSFER_AND_FROZEN_HASHES_PASS", flush=True)
    questions, rows, master = h.load(root)
    with zipfile.ZipFile(transfer) as z:
        lineage = verify_native_transport(z.read(metadata["native_archives"]["stochastic"]["member"]), master["signature"])
        print("COMPLETE_LINEAR_JOURNAL_VERIFIED", flush=True)
    target, imported, imported_files = import_readable(transfer,sidecar,root,metadata)
    final = imported/metadata["native_archives"]["stochastic_final"]["member"]
    p = stance.engine(root)
    native, payload = p.inspect_final(final,root,metadata["native_archives"]["stochastic_final"]["sha256"])
    require(native["status"] == {"complete_samples": True, "expected":2750,"valid":2750,"missing":0,"failed":0,"interrupted":0,"recorded_generation_attempts":2750}, "Final stochastic status differs")
    blind = json.loads((imported/PRED).read_bytes())
    predicted = frozen_evaluation.checkpoint_predictions(root,final,metadata["native_archives"]["stochastic_final"]["sha256"])
    require(predicted == blind["predictions"], "Independent blind prediction reproduction differs")
    print("2750_IDENTITIES_AND_ALL_BLIND_SCORES_REPRODUCED_NO_GOLD", flush=True)
    with zipfile.ZipFile(transfer) as z:
        gold = read_gold_after_reproof(root,z,questions,True)
        gpu = jread(z,GP+"evaluation_v1/evaluation.json")
    metrics = {f"{m}:{c}":method_metrics(predicted,gold,m,c) for m,c in METHODS}
    for key,value in metrics.items():
        close_numbers(gpu["metrics"][key],value,key+".")
    bc = transitions(predicted,gold,"B","C","quality_disabled")
    tc = dict(Counter(r["transition"] for r in bc))
    close_numbers(tc,{"same_answer":8,"answer_to_abstain":1,"abstain_to_answer":5,"both_abstain":6})
    cc = Counter((r["source_correct_if_answered"] is True,r["destination_correct_if_answered"] is True) for r in bc)
    ints = paired_intervals(cc[True,True],cc[False,True],cc[True,False],cc[False,False],tc["abstain_to_answer"],tc["answer_to_abstain"])
    close_numbers(ints["coverage"]["interval95"],gpu["primary"]["conservative_marginal_bound_interval95"])
    close_numbers(ints["accuracy"]["interval95"],gpu["primary"]["supplementary_newcombe10"]["interval95"])
    entropy = {c:contrast(predicted,"C","D",c) for c in CONDITIONS}
    quality = {m:contrast(predicted,m,m,"original_quality","quality_disabled") for m in "ABCD"}
    for c,value in entropy.items():
        close_numbers(gpu["entropy"][c],value)
    for m,value in quality.items():
        close_numbers(gpu["quality_gating"]["effects"][m],value)
    positive = {r["question_id"] for r in rows if r["quality_weight"] > 0}
    positive_docs = {(r["question_id"],r["evidence_doc_id"]) for r in rows if r["quality_weight"] > 0}
    require(positive == {"medqa-us-dev-000036","medqa-us-dev-000046"} and len(positive_docs)==2,"Quality inventory differs")
    dev_verification = verify_results(root)
    dev = json.loads((root/"outputs/aggregation_development/dev30_natural_qa_comparison_v1/comparison_results.json").read_bytes())
    print("ALL_METRICS_TRANSITIONS_INTERVALS_AND_SENSITIVITIES_MATCH", flush=True)
    output = root/BASE/"canonical_analysis_v1"
    require(not output.exists(), "NEW canonical analysis directory required")
    metrics_rows = []
    for key,m in metrics.items():
        method,condition=key.split(":")
        row = {"method":method,"quality_condition":condition}
        row.update({k:v for k,v in m.items() if not isinstance(v,(list,dict))})
        for metric in ("accuracy","coverage","answered_accuracy","selective_risk"):
            row[metric+"_ci95_lower"],row[metric+"_ci95_upper"] = m[metric+"_ci95"] or (None,None)
        metrics_rows.append(row)
    per = []
    for q in QIDS:
        for method,condition in METHODS:
            m=method_at(predicted[q],method,condition); a=m["answer"]
            per.append({"question_id":q,"method":method,"quality_condition":condition,"status":a["status"],
                        "selected_answer":a["selected_answer"],"correct_if_answered":a["selected_answer"]==gold[q] if a["selected_answer"] else None,
                        "abstention_reason":a["reason"] if a["status"]=="ABSTAIN" else None,
                        "undefined_option_count":len(a["undefined_options"])})
    cd=[]
    for condition in CONDITIONS:
        cd.extend(transitions(predicted,gold,"C","D",condition))
    breakdown=[{"method":key.split(":")[0],"quality_condition":key.split(":")[1],"reason":r,"count":n,"N":20} for key,m in metrics.items() for r,n in sorted(m["abstention_reasons"].items())]
    cohorts=[{"cohort":"development","N":30,"B_correct":7,"B_answered":13,"C_correct":11,"C_answered":19,"B_accuracy":7/30,"C_accuracy":11/30,"B_coverage":13/30,"C_coverage":19/30,"coverage_difference":6/30,"accuracy_difference":4/30,"soft_only_answers":6,"hard_only_answers":0,"soft_only_correct":4,"soft_only_wrong":2},
             {"cohort":"internal_holdout","N":20,"B_correct":4,"B_answered":9,"C_correct":7,"C_answered":13,"B_accuracy":4/20,"C_accuracy":7/20,"B_coverage":9/20,"C_coverage":13/20,"coverage_difference":4/20,"accuracy_difference":3/20,"soft_only_answers":5,"hard_only_answers":1,"soft_only_correct":3,"soft_only_wrong":2}]
    tables={"holdout_method_metrics.csv":metrics_rows,"holdout_per_question_predictions.csv":per,
            "holdout_BC_transitions.csv":bc,"holdout_CD_entropy_transitions.csv":cd,
            "holdout_abstention_breakdown.csv":breakdown,"dev30_vs_holdout_comparison.csv":cohorts}
    for name,table in tables.items():
        export_csv(output/name,table)
    interval_report={"N":20,"primary":ints["coverage"],"supplementary_accuracy":ints["accuracy"],
                     "paired_status_counts":tc,"paired_correctness_cells":{"both_correct":cc[True,True],"soft_only_correct":cc[False,True],"hard_only_correct":cc[True,False],"neither_correct":cc[False,False]},
                     "individual_method_intervals":{k:{x:v for x,v in m.items() if x.endswith("_ci95")} for k,m in metrics.items()},
                     "no_significance_test":True,"no_pooled_primary_estimate":True}
    new_file(output/"holdout_intervals.json",encoded(interval_report))
    new_file(output/"holdout_sensitivity_analysis.json",encoded({"entropy":entropy,"quality":quality,"positive_quality_questions":sorted(positive),"positive_quality_documents":len(positive_docs)}))
    changed=[r for r in cd if r["transition"] not in ("same_answer","both_abstain")]
    details={}
    for r in changed:
        q=r["question_id"];details[q]={"C":method_at(predicted[q],"C",r["quality_condition"]),"D":method_at(predicted[q],"D",r["quality_condition"])}
    new_file(output/"holdout_entropy_changed_case.json",encoded(details))
    summary=["# Internal holdout: independently verified, generation-free analysis", "", "N=20 (Q31-Q50), separate from development; no pooled primary estimate. Manuscript editing is deferred at the user's request.", "", "## Verified metrics", "", "| Method | Quality | Correct/N | Answered/N | Accuracy | Coverage | Answered accuracy | Selective risk |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_rows:
        summary.append(f"| {row['method']} | {row['quality_condition']} | {row['correct']}/20 | {row['answered']}/20 | {row['accuracy']:.2%} | {row['coverage']:.2%} | {row['answered_accuracy']:.2%} | {row['selective_risk']:.2%} |")
    summary += ["", "## Primary matched unit-weight hard-to-frequency comparison", "", "Eight identical answers, no answer replacement, one answer-to-abstention (previously wrong), five abstention-to-answer (three correct, two wrong), six joint abstentions.", "", f"Coverage difference +20 percentage points; conservative marginal-bound interval [{ints['coverage']['interval95'][0]*100:.8f}, {ints['coverage']['interval95'][1]*100:.8f}] pp. Accuracy difference +15 pp; pre-specified supplementary Newcombe method 10 approximate interval [{ints['accuracy']['interval95'][0]*100:.8f}, {ints['accuracy']['interval95'][1]*100:.8f}] pp. Both span zero; no significance/superiority claim.", "", "## Entropy and quality", "", "Original-quality entropy: 0/10 jointly defined scores, zero native/ranking/final changes. Unit weights: 36/95 scores changed, zero native claim changes, two ranking changes and one final decision change."]
    for r in changed:
        summary.append(f"Changed entropy case {r['question_id']}: {r['source_answer']} -> {r['destination_answer']}; destination correct among answered: {r['destination_correct_if_answered']}.")
    summary += ["", "Two positive-quality documents occur in Q36/Q46. Original B/C/D each answer one question correctly, with 90/100 undefined option scores; unit methods have 5/100 undefined scores (Q31). OTHER=0 indicates unclassified design, not clinically poor or irrelevant evidence.", "", "## Interpretation and status", "", "The net +20 pp coverage effect reproduced descriptively across the separate development and internal holdout cohorts. Added soft answers include both correct and incorrect selections. Neither statistical superiority nor clinical validity is established. Entropy was decision-inert on development but changed one holdout decision.", "", "Vanilla holdout v1 is incomplete: four VALID, Q36 FAILED after three strict-format attempts, fourteen MISSING; Q31 structural ABSTAIN. No partial Vanilla performance is reported and no textual final answer is salvaged.", "", "Independent 90-pair development disagreement review remains pending; no annotations generated. Runtime/recovery/cadence/Windows/controller/schema amendments remain private provenance, not methodological contributions. Gold-schema compatibility changes only evaluator reading of the stable A-E answer field. The post-evaluation auxiliary receipt failed after the verified core outputs completed; its STOP logs are preserved.", "", "## Integrity", "", "Transfer CRC and all member hashes passed; frozen runtime matches local scientific references; full native linear journal folded in memory to the final payload; native final validation and reproduction of every blind score passed before local gold reading. Recorded dual freeze verification preceded first recorded gold access. No inference, retrieval, sampling, re-freeze, tuning, manuscript edit, commit or push occurred in this integration."]
    new_file(output/"holdout_analysis_summary.md",("\n".join(summary)+"\n").encode())
    verification.update({"native_lineage":lineage,"final_status":native["status"],"primary_blind_sha256":BLIND_SHA,"vanilla_incomplete_freeze_sha256":VANILLA_SHA,
                         "native_final_sha256":metadata["native_archives"]["stochastic_final"]["sha256"],"gold_source_sha256":GOLD_SHA,
                         "derived_gold_sha256":DERIVED_SHA,"gold_adapter_sha256":ADAPTER_SHA,"gold_schema_fix":"evaluator/data-schema compatibility only",
                         "all_blind_predictions_independently_reproduced":True,"all_expected_gpu_metrics_match":True,
                         "manuscript_modified":False,"independent_review":"PENDING 90 pairs","new_model_calls":0,
                         "csv_rows":{n:len(t) for n,t in tables.items()},"source_transfer_byte_identical":file_sha(target)==TRANSFER_SHA})
    new_file(output/"imported_artifact_manifest.json",encoded({"source_sha256":TRANSFER_SHA,"files":imported_files}))
    manifest={"version":VERSION,"created_utc":datetime.now(timezone.utc).isoformat(),"private":True,"verification":verification,
              "analysis_code_sha256":file_sha(__file__),"frozen_code_hashes":master["source_sha256"],
              "files":inventory_files(root,output.rglob("*.*")),"paper_editing_deferred":True}
    new_file(output/"holdout_analysis_manifest.json",encoded(manifest))
    for path,wanted in readonly.items():
        require(file_sha(path)==wanted,"Protected original file changed")
    with zipfile.ZipFile(transfer) as z:
        require(digest(z.read(PRED))==BLIND_SHA and digest(z.read(OP+"freeze/vanilla_incomplete_state_v1.zip"))==VANILLA_SHA,"Frozen predictions changed")
    require(file_sha(transfer)==TRANSFER_SHA,"Original transfer changed")
    print(json.dumps({"verification":"PASS","output":str(output),"changed_entropy_cases":changed,"BC_discordant_cases":[r for r in bc if r['transition'] in ('abstain_to_answer','answer_to_abstain')],"manifest_sha256":file_sha(output/'holdout_analysis_manifest.json')},indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--transfer",required=True,type=Path)
    parser.add_argument("--sidecar",type=Path)
    args=parser.parse_args()
    run(args.root,args.transfer,args.sidecar)
