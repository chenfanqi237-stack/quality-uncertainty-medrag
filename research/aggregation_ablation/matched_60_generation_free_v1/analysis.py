"""Offline matched-input ablation. Imports no retrieval or inference backend."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from quality_uncertainty_medrag.aggregation import MajorityVoteAggregator, QualityWeightedVoteAggregator
from quality_uncertainty_medrag.models import (
    CandidateClaim, EvidenceType, QualityScore, RetrievedEvidence, ScoredEvidence,
    Stance, StancePrediction, PROBABILITY_TIE_ABS_TOLERANCE,
)
from quality_uncertainty_medrag.pubmed_evidence_type import evidence_type_from_publication_types
from quality_uncertainty_medrag.uncertainty_aggregation import (
    QualityUncertaintyWeightedAggregator, _normalized_entropy,
)

TOL = 1e-12
LABELS = tuple(s.value for s in Stance)
SEEDS = tuple(range(101, 111))
EVAL = Path("outputs/stance_uncertainty/evaluation/checkpoint_valid600_20261002T152105Z_553a09b9_final_evaluation_v1")
UNC = EVAL / "pair_uncertainty.csv"
UNLABELED = Path("outputs/stance_model_comparison/reference_60_adjudicated/unlabeled_inputs_60.jsonl")
TOPK_MANIFEST = Path("outputs/aggregation_development/topk_dev_1_30_k3_k5_v1/manifest.json")
RETRIEVAL = (
    Path("data/retrieved/pubmed_medqa_us_dev_10_llm.jsonl"),
    Path("outputs/heldout_retrieval/dev_11_30/500a1f067a9f7826/runs/kaggle-20260928T142621Z/evidence.jsonl"),
)
CONFIG = Path("configs/baseline.yaml")
CHECKPOINT = Path("cloud/checkpoint/checkpoint_valid600_20261002T152105Z_553a09b9.zip")
SAMPLE_MEMBER = "outputs/stance_uncertainty/self_consistency_10/reference_60/samples.jsonl"
SOURCE_CODE = tuple(Path("src/quality_uncertainty_medrag") / f for f in (
    "__init__.py", "models.py", "aggregation.py", "uncertainty_aggregation.py", "pubmed_evidence_type.py",
    "stance_self_consistency.py",
))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    # Iterate physical lines; str.splitlines() would also split Unicode separators in abstracts.
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(row):
    return tuple(row[f] for f in ("question_id", "candidate_option_id", "evidence_doc_id"))


def probabilities(counts):
    total = sum(counts.values())
    if not total:
        raise ValueError("Missing stochastic counts; no frequency default is allowed")
    return {s: counts[s.value] / total for s in Stance}


def validate_frequency(row):
    counts = {label: int(row["n_" + label.lower()]) for label in LABELS}
    if any(n < 0 for n in counts.values()) or sum(counts.values()) != 10:
        raise ValueError("Each pair must have exactly ten nonnegative sample counts")
    p = probabilities(counts)
    values = [float(row["p_" + s.value.lower()]) for s in Stance]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Invalid empirical frequency")
    if any(abs(p[s] - x) > TOL for s, x in zip(Stance, values)):
        raise ValueError("Counts and empirical frequencies disagree")
    if abs(_normalized_entropy(p) - float(row["u_3"])) > TOL:
        raise ValueError("Frozen entropy disagrees with sample frequencies")
    # The archived field is conditional on directional mass. It is validated
    # against its frozen definition, but NOT used as this ablation's pS-pC.
    mass = p[Stance.SUPPORT] + p[Stance.CONTRADICT]
    conditional_direction = (p[Stance.SUPPORT] / mass - p[Stance.CONTRADICT] / mass) if mass else 0.0
    if abs(conditional_direction - float(row["directional_score"])) > TOL:
        raise ValueError("Frozen conditional directional score disagrees with frequencies")
    return counts, p


def load_inputs(root=ROOT):
    paths = (UNC, UNLABELED, CONFIG, *RETRIEVAL, EVAL / "analysis_manifest.json", TOPK_MANIFEST, CHECKPOINT, *SOURCE_CODE)
    hashes = {p.as_posix(): sha(root / p) for p in paths}
    topk = json.loads((root / TOPK_MANIFEST).read_text(encoding="utf-8"))
    previous = topk["input"]["files"]
    for name, digest in hashes.items():
        if name in previous and previous[name] != digest:
            raise ValueError("Source differs from frozen Top-K manifest: " + name)
    final = json.loads((root / EVAL / "analysis_manifest.json").read_text(encoding="utf-8"))
    if hashes[CHECKPOINT.as_posix()] != final["checkpoint"]["sha256"]:
        raise ValueError("Final checkpoint hash mismatch")
    if hashes[UNLABELED.as_posix()] != final["frozen_configuration"]["unlabeled_input_sha256"]:
        raise ValueError("Frozen unlabeled identity projection hash mismatch")
    if final["mode"] != "FINAL_EVALUATION" or final["complete_pair_count"] != 60:
        raise ValueError("Expected the completed final evaluation")
    if tuple(final["frozen_configuration"]["seeds"]) != SEEDS:
        raise ValueError("Unexpected frozen seed list")
    # Read only the explicit numeric mapping in the frozen simple configuration.
    # This is intentionally not a general YAML parser or a new quality mapping.
    config_text = (root / CONFIG).read_text(encoding="utf-8")
    block = config_text.split("  evidence_type_scores:\n", 1)[1].split("\n\n", 1)[0]
    entries = [re.fullmatch(r"    ([a-z_]+): ([0-9.]+)", line) for line in block.splitlines()]
    if not entries or any(entry is None for entry in entries):
        raise ValueError("Unexpected frozen quality configuration layout")
    weights = {entry[1]: float(entry[2]) for entry in entries}
    if len(weights) != len(entries) or set(weights) != {e.value for e in EvidenceType} or weights["other"] != 0:
        raise ValueError("Incomplete, duplicate or altered frozen quality categories")
    if any(not math.isfinite(q) or not 0 <= q <= 1 for q in weights.values()):
        raise ValueError("Invalid frozen quality weight")
    documents = {}
    for path in RETRIEVAL:
        for raw in read_jsonl(root / path):
            key = (raw["question_id"], raw["doc_id"])
            if key in documents:
                raise ValueError("Duplicate question-document retrieval identity")
            types = raw["metadata"]["publication_types"]
            if evidence_type_from_publication_types(types).value != raw["evidence_type"]:
                raise ValueError("Publication types disagree with frozen mapping")
            documents[key] = raw
    if len(documents) != 305:
        raise ValueError("Unexpected development retrieval coverage")
    unlabeled = {}
    stems, options = {}, {}
    for row in read_jsonl(root / UNLABELED):
        allowed = {"question_id", "candidate_option_id", "evidence_doc_id", "question_stem", "candidate_option_text", "evidence_title", "evidence_abstract"}
        if set(row) != allowed or identity(row) in unlabeled:
            raise ValueError("Unlabeled input schema/identity mismatch")
        key = identity(row)
        if not re.fullmatch(r"medqa-us-dev-\d{6}", key[0]) or not 1 <= int(key[0][-6:]) <= 30 or key[1] not in "ABCDE":
            raise ValueError("Question/option is outside frozen development identities")
        if not key[2].isdigit() or not all(isinstance(v, str) for v in row.values()):
            raise ValueError("Malformed unlabeled identity or text")
        if stems.setdefault(key[0], row["question_stem"]) != row["question_stem"] or options.setdefault(key[:2], row["candidate_option_text"]) != row["candidate_option_text"]:
            raise ValueError("Question or option text is inconsistent")
        unlabeled[key] = row
    with (root / UNC).open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    pairs = {}
    pair_ids = set()
    for row in raw_rows:
        key = identity(row)
        if key in pairs or row["pair_id"] in pair_ids or row["stable_pair_identity"] != "|".join(key):
            raise ValueError("Duplicate or inconsistent uncertainty identity")
        if row["evaluation_included"] != "True" or key not in unlabeled:
            raise ValueError("Missing eligible exact uncertainty-input identity")
        counts, p = validate_frequency(row)
        raw = documents.get((key[0], key[2]))
        if raw is None or str(raw["metadata"].get("pmid", raw["doc_id"])) != key[2]:
            raise ValueError("Missing exact PubMed identity")
        if any(unlabeled[key]["evidence_" + field] != raw["metadata"][field] for field in ("title", "abstract")):
            raise ValueError("Frozen evidence text differs from unlabeled projection")
        pairs[key] = {"key": key, "pair_id": row["pair_id"], "counts": counts, "p": p,
                      "raw": raw, "input": unlabeled[key], "quality": weights[raw["evidence_type"]]}
        pair_ids.add(row["pair_id"])
    if len(pairs) != 60 or set(pairs) != set(unlabeled):
        raise ValueError("Expected 60 complete exact input/frequency pairs")
    with zipfile.ZipFile(root / CHECKPOINT) as archive:
        sample_bytes = archive.read(SAMPLE_MEMBER)
    samples = defaultdict(dict)
    for line in sample_bytes.decode("utf-8").splitlines():
        sample = json.loads(line)
        key, seed = identity(sample["ids"]), sample["seed"]
        if key not in pairs or type(seed) is not int or seed not in SEEDS or seed in samples[key] or sample["stance"] not in LABELS:
            raise ValueError("Invalid/duplicate archived pair-seed sample")
        samples[key][seed] = sample["stance"]
    if set(samples) != set(pairs) or any(set(v) != set(SEEDS) for v in samples.values()):
        raise ValueError("Incomplete archived samples")
    for key, values in samples.items():
        if Counter(values.values()) != Counter({k: v for k, v in pairs[key]["counts"].items() if v}):
            raise ValueError("Archived samples disagree with final frequency counts")
    groups = defaultdict(list)
    for key, pair in pairs.items():
        groups[key[:2]].append(pair)
    for group in groups.values():
        group.sort(key=lambda x: (x["raw"]["rank"], x["key"][2]))
    if len(groups) != 50 or Counter(map(len, groups.values())) != {1: 40, 2: 10}:
        raise ValueError("Expected 50 groups: 40 single-evidence and 10 multi-evidence")
    return dict(groups), dict(samples), hashes, hashlib.sha256(sample_bytes).hexdigest()


def signed_decision(score, tolerance=0):
    if score is None or abs(score) <= tolerance:
        return "ABSTAIN"
    return "SUPPORT" if score > 0 else "CONTRADICT"


def evaluate(group, condition, p_override=None):
    if condition not in ("original_quality", "quality_disabled"):
        raise ValueError("Unknown fixed analysis condition")
    shared = []
    for pair in group:
        raw = pair["raw"]
        evidence = RetrievedEvidence(
            schema_version=raw["schema_version"], question_id=raw["question_id"], doc_id=raw["doc_id"],
            rank=raw["rank"], text=raw["text"], source=raw["source"],
            evidence_type=EvidenceType(raw["evidence_type"]), retrieval_score=raw["retrieval_score"],
            metadata={k: raw["metadata"][k] for k in ("title", "abstract", "publication_types")},
        )
        quality = pair["quality"] if condition == "original_quality" else 1.0
        p = pair["p"] if p_override is None else p_override[pair["key"]]
        shared.append(ScoredEvidence(evidence, QualityScore(quality, condition), StancePrediction(p, "frozen-ten-seed-empirical-frequencies")))
    shared = tuple(shared)  # Identical objects and order supplied to A, B, C and D.
    claim = CandidateClaim(group[0]["key"][0], "ABCDE".index(group[0]["key"][1]), group[0]["key"][1], group[0]["input"]["candidate_option_text"])
    context = SimpleNamespace(question_id=claim.question_id, question=group[0]["input"]["question_stem"])
    a = MajorityVoteAggregator().aggregate(context, claim, shared)
    b = QualityWeightedVoteAggregator().aggregate(context, claim, shared)
    d = QualityUncertaintyWeightedAggregator().aggregate(context, claim, shared)
    qsum = math.fsum(x.quality.value for x in shared)
    cscore = None if qsum == 0 else math.fsum((x.quality.value / qsum) * (x.stance.probabilities[Stance.SUPPORT] - x.stance.probabilities[Stance.CONTRADICT]) for x in shared)
    result = {
        "A": {"score": (a.support_weight - a.contradict_weight) / len(shared), "denominator": len(shared), "decision": a.decision.value},
        "B": {"score": None if qsum == 0 else (b.support_weight - b.contradict_weight) / qsum, "denominator": qsum, "decision": b.decision.value},
        "C": {"score": cscore, "denominator": qsum, "decision": signed_decision(cscore)},
        "D": {"score": d.aggregate_score, "denominator": d.total_effective_weight, "decision": d.decision.value},
    }
    for name, out in result.items():
        out["tolerance_decision"] = signed_decision(out["score"], TOL)
        out["score_defined"] = out["score"] is not None
        out["within_zero_tolerance"] = out["score"] is not None and abs(out["score"]) <= TOL
        reason = ""
        if out["decision"] == "ABSTAIN":
            if name != "A" and qsum == 0:
                reason = "ALL_ANALYSIS_QUALITY_WEIGHTS_ZERO"
            elif out["denominator"] == 0:
                reason = "ZERO_ENTROPY_EFFECTIVE_WEIGHT"
            elif name in ("A", "B"):
                hard_mass = (a.support_weight + a.contradict_weight) if name == "A" else (b.support_weight + b.contradict_weight)
                reason = "BALANCED_HARD_VOTES_OR_WEIGHT_TIE" if hard_mass else "NO_DIRECTIONAL_HARD_VOTES"
                if not hard_mass and any(x.stance.label is None and (name == "A" or x.quality.value > 0) for x in shared):
                    reason += "_WITH_UNRESOLVED_ARGMAX"
            else:
                reason = "ZERO_SOFT_DIRECTION"
        out["abstention_reason"] = reason
    labels = [x.stance.label.value if x.stance.label is not None else None for x in shared]
    return result, labels


def compare(left, right):
    defined = left["score"] is not None and right["score"] is not None
    return {
        "both_scores_defined": defined,
        "score_delta": right["score"] - left["score"] if defined else None,
        "score_changed": defined and abs(right["score"] - left["score"]) > TOL,
        "definedness_changed": left["score_defined"] != right["score_defined"],
        "native_decision_changed": left["decision"] != right["decision"],
        "tolerance_decision_changed": left["tolerance_decision"] != right["tolerance_decision"],
    }


def analyze(groups, samples):
    rows = []
    cancellation_errors = []
    for key, group in sorted(groups.items()):
        original_positive = sum(pair["quality"] > 0 for pair in group)
        for condition in ("original_quality", "quality_disabled"):
            positive = original_positive if condition == "original_quality" else len(group)
            full, labels = evaluate(group, condition)
            if positive == 1 and full["D"]["denominator"] > 0:
                error = abs(full["C"]["score"] - full["D"]["score"])
                if error > TOL:
                    raise ValueError("Single-contributing-evidence entropy cancellation failed")
                cancellation_errors.append(error)
            stability = {m: {"native_changes": 0, "tolerance_changes": 0, "definedness_changes": 0,
                             "changed_seeds": [], "max_abs_score_change": None} for m in "ABCD"}
            for seed in SEEDS:
                reduced = {}
                for pair in group:
                    counts = pair["counts"].copy()
                    counts[samples[pair["key"]][seed]] -= 1
                    reduced[pair["key"]] = probabilities(counts)
                replicate, _ = evaluate(group, condition, reduced)
                for method in "ABCD":
                    change = compare(full[method], replicate[method])
                    st = stability[method]
                    st["native_changes"] += change["native_decision_changed"]
                    st["tolerance_changes"] += change["tolerance_decision_changed"]
                    st["definedness_changes"] += change["definedness_changed"]
                    if change["native_decision_changed"]:
                        st["changed_seeds"].append(seed)
                    if change["both_scores_defined"]:
                        delta = abs(change["score_delta"])
                        st["max_abs_score_change"] = max(st["max_abs_score_change"] or 0, delta)
            details = []
            for pair, label in zip(group, labels):
                raw = pair["raw"]
                details.append({
                    "pair_id": pair["pair_id"], "identity": "|".join(pair["key"]), "pmid": pair["key"][2],
                    "retrieval_rank": raw["rank"], "publication_types": raw["metadata"]["publication_types"],
                    "evidence_type": raw["evidence_type"], "original_quality_weight": pair["quality"],
                    "analysis_weight": pair["quality"] if condition == "original_quality" else 1.0,
                    "counts": pair["counts"], "frequencies": {s.value: pair["p"][s] for s in Stance},
                    "soft_stance": pair["p"][Stance.SUPPORT] - pair["p"][Stance.CONTRADICT],
                    "entropy": _normalized_entropy(pair["p"]), "matched_hard": label,
                })
            rows.append({"question_id": key[0], "candidate_option_id": key[1], "condition": condition,
                         "evidence_count": len(group), "original_positive_weight_count": original_positive,
                         "analysis_positive_weight_count": positive, "matched_hard_labels": labels,
                         "evidence_details": details, "methods": full,
                         "B_vs_C": compare(full["B"], full["C"]), "C_vs_D": compare(full["C"], full["D"]),
                         "leave_one_seed_out": stability})
    metrics = {}
    for condition in ("original_quality", "quality_disabled"):
        selected = [r for r in rows if r["condition"] == condition]
        subsets = {"all": selected, "single_evidence": [r for r in selected if r["evidence_count"] == 1],
                   "multi_evidence": [r for r in selected if r["evidence_count"] > 1],
                   "all_analysis_weights_zero": [r for r in selected if r["analysis_positive_weight_count"] == 0],
                   "original_all_zero_stratum": [r for r in selected if r["original_positive_weight_count"] == 0],
                   "two_or_more_positive_weights": [r for r in selected if r["analysis_positive_weight_count"] >= 2]}
        metrics[condition] = {}
        for name, subset in subsets.items():
            comparisons = {}
            for contrast in ("B_vs_C", "C_vs_D"):
                comparisons[contrast] = {field: sum(r[contrast][field] for r in subset) for field in (
                    "both_scores_defined", "score_changed", "definedness_changed", "native_decision_changed", "tolerance_decision_changed")}
                comparisons[contrast]["max_abs_score_delta"] = max((abs(r[contrast]["score_delta"]) for r in subset if r[contrast]["both_scores_defined"]), default=None)
            metrics[condition][name] = {"groups": len(subset), "comparisons": comparisons,
                "decisions": {m: dict(Counter(r["methods"][m]["decision"] for r in subset)) for m in "ABCD"},
                "undefined_scores": {m: sum(not r["methods"][m]["score_defined"] for r in subset) for m in "ABCD"}}
    cancellation = {"eligible_group_conditions": len(cancellation_errors), "max_abs_error": max(cancellation_errors, default=None)}
    return rows, metrics, cancellation


def csv_text(rows):
    flat = []
    for row in rows:
        out = {k: row[k] for k in ("question_id", "candidate_option_id", "condition", "evidence_count", "original_positive_weight_count", "analysis_positive_weight_count")}
        for k in ("matched_hard_labels", "evidence_details", "leave_one_seed_out"):
            out[k] = canonical(row[k])
        for method, values in row["methods"].items():
            out.update({method + "_" + k: value for k, value in values.items()})
        for contrast in ("B_vs_C", "C_vs_D"):
            out.update({contrast + "_" + k: value for k, value in row[contrast].items()})
        flat.append(out)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(flat[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    return buffer.getvalue()


def summary_text(rows, metrics, cancellation, groups):
    lines = ["# Matched-input 60-pair generation-free aggregation ablation", "",
        "This is a claim-level feasibility diagnostic, not MedQA answer accuracy, clinical reliability, or verified conflict-resolution performance.", "",
        "## Validation", "", "- 60 unique complete pairs; 50 option groups: 40 with one item and 10 with two.",
        "- All 60 identities, titles and abstracts match the frozen unlabeled projection and original development PubMed records.",
        "- All 600 archived samples are present exactly once for seeds 101-110; their class counts match the final CSV.",
        "- Publication-type categories and configured weights are unchanged. All 50 groups are evaluated in both conditions (100 CSV rows).",
        "- No reference stance labels, deterministic hard predictions, or MedQA gold answers enter aggregation or setting selection.",
        "- No inference, retrieval, GPU, new sampling or Git command was run. Frozen source hashes are checked before and after analysis.", "",
        "## Fixed methods and numerical conventions", "",
        "Matched hard is the frozen StancePrediction unique argmax; tied argmax remains unresolved (null), not a forced class. A/B/C/D receive the identical frequency-bearing evidence objects.",
        "A score=(support votes-contradict votes)/number of all items. B score=(support quality-contradict quality)/sum of all analysis quality. These are reporting normalizations, not changes to the frozen hard-vote decisions.",
        "C=sum(q*(pS-pC))/sum(q); D is the frozen sum(q*(1-H/log(3))*(pS-pC))/sum(q*(1-H/log(3))). IRRELEVANT and unresolved modal labels do not filter soft inputs.",
        "The source CSV directional_score is (pS-pC)/(pS+pC), or zero without directional samples. It is validated but not substituted for the requested unconditional soft_stance=pS-pC.",
        "Original quality retains OTHER=0. Quality disabled uses unit analysis weights for EVERY item; original stored metadata and study types are not changed.",
        "Absolute tolerance is 1e-12 with zero relative tolerance. Score changes use this tolerance. A/B/D decisions retain native frozen rules; C uses exact sign like frozen D. A separate tolerance_decision is a diagnostic only, not a replacement algorithm. Undefined scores are empty CSV fields with score_defined=False, never zero.", "",
        "## Exact comparisons", "",
        "| Condition / stratum | Groups | B-C score-comparable | B-C score changes | B-C native decision changes | B-C tolerance decision changes | C-D score-comparable | C-D score changes | C-D native decision changes | C-D tolerance decision changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, subsets in metrics.items():
        for stratum, data in subsets.items():
            b, d = data["comparisons"]["B_vs_C"], data["comparisons"]["C_vs_D"]
            values = [data["groups"], b["both_scores_defined"], b["score_changed"], b["native_decision_changed"], b["tolerance_decision_changed"], d["both_scores_defined"], d["score_changed"], d["native_decision_changed"], d["tolerance_decision_changed"]]
            lines.append("| " + condition + " / " + stratum + " | " + " | ".join(map(str, values)) + " |")
    lines += ["", "Strata overlap; they must not be added. The original-all-zero stratum remains the same 41 groups in both conditions; the analysis-all-zero stratum has zero groups when quality is disabled.", "",
        "## Decisions and undefined denominators", "",
        "| Condition | Method | SUPPORT | CONTRADICT | ABSTAIN | Undefined score |", "|---|---|---:|---:|---:|---:|"]
    for condition in metrics:
        data = metrics[condition]["all"]
        for m in "ABCD":
            counts = data["decisions"][m]
            lines.append(f"| {condition} | {m} | {counts.get('SUPPORT', 0)} | {counts.get('CONTRADICT', 0)} | {counts.get('ABSTAIN', 0)} | {data['undefined_scores'][m]} |")
    lines += ["", "## Observed changed groups", "",
        "All score or native-decision changes are listed below; exact evidence identities and distributions are in the CSV. Scores use 17 significant digits so near-zero residuals remain visible.", "",
        "| Condition | Question-option | n | B score / decision | C score / decision | D score / decision |", "|---|---|---:|---|---|---|"]
    for row in rows:
        if not any(row[x]["score_changed"] or row[x]["native_decision_changed"] or row[x]["definedness_changed"] for x in ("B_vs_C", "C_vs_D")):
            continue
        display = []
        for m in "BCD":
            out = row["methods"][m]
            score = "undefined" if out["score"] is None else format(out["score"], ".17g")
            display.append(score + " / " + out["decision"])
        lines.append("| " + row["condition"] + " | " + str(int(row["question_id"][-6:])) + "-" + row["candidate_option_id"] + " | " + str(row["evidence_count"]) + " | " + " | ".join(display) + " |")
    tied_pairs = sum(StancePrediction(pair["p"], "matched").label is None for group in groups.values() for pair in group)
    lines += ["", "## Single-contributing-item cancellation", "",
        "If exactly one item j has q_j>0 and q_j*(1-u_j)>0, C=(q_j*s_j)/q_j=s_j and D=(q_j*(1-u_j)*s_j)/(q_j*(1-u_j))=s_j. Zero-quality items contribute to neither numerator nor denominator.",
        f"Numerical check: {cancellation['eligible_group_conditions']} eligible group-condition records; maximum absolute C-D error={cancellation['max_abs_error']}. All are within 1e-12.",
        "If the effective denominator vanishes, D is undefined/ABSTAIN; cancellation cannot be asserted for an undefined ratio. Focused tests cover this boundary separately.",
        f"Original-quality groups have zero positive items in 41 cases and one in nine cases; NONE has two. The ten multi-evidence groups therefore cannot test relative entropy weighting under original quality. Baseline tied modal argmax pairs: {tied_pairs}/60.", "",
        "## Leave-one-seed-out stability", "",
        "For each saved seed, remove that same seed across every item, recompute nine-sample frequencies, and compare with the full ten-sample result. This is deterministic sensitivity analysis of existing draws, not new sampling, independent replications, or clinical study sampling variance.", "",
        "| Condition | Method | Groups with native change | Changed group-seed decisions / 500 | Groups with tolerance change | Changed tolerance decisions / 500 | Undefined-status changes / 500 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for condition in metrics:
        selected = [r for r in rows if r["condition"] == condition]
        for m in "ABCD":
            sts = [r["leave_one_seed_out"][m] for r in selected]
            vals = [sum(st["native_changes"] > 0 for st in sts), sum(st["native_changes"] for st in sts), sum(st["tolerance_changes"] > 0 for st in sts), sum(st["tolerance_changes"] for st in sts), sum(st["definedness_changes"] for st in sts)]
            lines.append("| " + condition + " | " + m + " | " + " | ".join(map(str, vals)) + " |")
    disabled = metrics["quality_disabled"]["all"]["comparisons"]
    bc, cd = disabled["B_vs_C"], disabled["C_vs_D"]
    lines += ["", "## Interpretation and pilot decision", "",
        "Original quality gating eliminates every opportunity for relative multi-item entropy weighting in this subset. A zero change there is structural, not evidence that entropy weighting succeeds or fails clinically.",
        f"With quality disabled, hard-to-soft changes {bc['score_changed']}/50 scores and {bc['native_decision_changed']}/50 native decisions ({bc['tolerance_decision_changed']}/50 tolerance decisions). Entropy changes {cd['score_changed']}/50 scores and {cd['native_decision_changed']}/50 native decisions ({cd['tolerance_decision_changed']}/50 tolerance decisions).",
        "Entropy is a function of the existing frequencies, not additional information. A score change need not change a decision; a near-zero native sign difference need not be substantive.",
        ("The observed tolerance-resolved decision changes justify, at most, a small pre-specified end-to-end FEASIBILITY pilot with complete five-option groups and paired evidence. They do not justify a large inference run or a superiority claim." if bc['tolerance_decision_changed'] or cd['tolerance_decision_changed'] else "No tolerance-resolved decision effect is observed; these results do not justify new inference solely to demonstrate an aggregation decision benefit."),
        "The 60 rows cover only restricted option-level evidence subsets. Even the two questions containing all five options here do not establish complete original retrieval coverage. No gold/reference accuracy, clinical significance, improved reliability, or real conflict-resolution effect is estimated.",
        "Source identities, input/code SHA256 hashes, archive member hash, and before/after preservation checks are recorded in manifest.json. The scientific-data guidance keeps undefined values and original quality separate from analysis controls.", ""]
    return "\n".join(lines)


def build(root=ROOT, check=False):
    groups, samples, hashes, member_hash = load_inputs(root)
    rows, metrics, cancellation = analyze(groups, samples)
    contents = {"ablation_results.csv": csv_text(rows), "ablation_summary.md": summary_text(rows, metrics, cancellation, groups)}
    for name, digest in hashes.items():
        if sha(root / name) != digest:
            raise ValueError("Frozen input changed during analysis: " + name)
    manifest = {
        "schema_version": "matched-60-generation-free-ablation-v1", "project_root": str(root.resolve()),
        "input_sha256": hashes, "archive_sample_member": SAMPLE_MEMBER, "archive_sample_member_sha256": member_hash,
        "analysis_code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("test_analysis.py"))},
        "validation": {"complete_pairs": 60, "option_groups": 50, "evidence_count_distribution": {"1": 40, "2": 10},
                       "samples": 600, "seeds": list(SEEDS), "csv_group_condition_rows": 100,
                       "exact_identity_and_text_join": True, "source_hashes_unchanged": True,
                       "matched_argmax_tied_pairs": sum(StancePrediction(p['p'], 'matched').label is None for g in groups.values() for p in g)},
        "methods": {"A": "Frozen unit hard votes; unique argmax of the same empirical frequencies; reporting margin/n",
                    "B": "Frozen quality hard votes; same frequencies; reporting margin/sum(all analysis q)",
                    "C": "Entropy-off deterministic ablation: sum(q*(pS-pC))/sum(q), exact sign",
                    "D": "Unmodified frozen QualityUncertaintyWeightedAggregator"},
        "conditions": {"original_quality": "Frozen configured weights; OTHER=0", "quality_disabled": "Unit analysis weights for every item; not study-quality reclassification"},
        "absolute_tolerance": TOL, "relative_tolerance": 0.0,
        "native_rules_preserved": True, "tolerance_decision_is_diagnostic_only": True,
        "source_directional_score": "Conditional S/C direction; validated but not used as soft_stance",
        "ablation_soft_stance": "p_support - p_contradict; computed from the same archived empirical frequencies",
        "original_quality_metadata_modified": False, "reference_labels_used": False, "medqa_gold_source_opened": False,
        "frozen_deterministic_hard_predictions_used": False, "question_level_answer_accuracy_evaluated": False,
        "new_model_calls": 0, "retrieval_calls": 0, "new_stochastic_samples": 0, "gpu_used": False,
        "stability": "Leave the same saved seed out across all items; 10 deterministic replicates per group-condition",
        "single_positive_cancellation": cancellation, "comparison_counts": metrics,
        "derived_sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in contents.items()},
        "reproduction": "python -B analysis.py; verify existing outputs with python -B analysis.py --check; focused tests: python -B -m unittest discover -s <this directory> -p test_analysis.py -v",
    }
    contents["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out = Path(__file__).parent
    if check:
        for name, text in contents.items():
            if (out / name).read_bytes() != text.encode("utf-8"):
                raise ValueError("Saved derived output differs from deterministic reproduction: " + name)
    else:
        if any((out / name).exists() for name in contents):
            raise FileExistsError("Refusing to overwrite existing derived outputs; use --check")
        for name, text in contents.items():
            with (out / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(check=args.check)
    print(json.dumps({c: result[c]["all"] for c in result}, indent=2))
