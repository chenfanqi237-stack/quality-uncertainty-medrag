"""Isolated matched-input adapter; does not change frozen aggregation modules."""
from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

from quality_uncertainty_medrag.aggregation import MajorityVoteAggregator, QualityWeightedVoteAggregator
from quality_uncertainty_medrag.models import CandidateClaim, EvidenceType, QualityScore, RetrievedEvidence, ScoredEvidence, Stance, StancePrediction
from quality_uncertainty_medrag.uncertainty_aggregation import QualityUncertaintyWeightedAggregator, _normalized_entropy

TOL = 1e-12
OPTIONS = tuple("ABCDE")
RULES = {
    "version": "question27-answer-adapter-v1",
    "A_reporting_score": "(SUPPORT votes-CONTRADICT votes)/all evidence count",
    "B_reporting_score": "(SUPPORT quality-CONTRADICT quality)/sum(all analysis quality)",
    "C_score": "sum(q*(pS-pC))/sum(q)",
    "D_score": "frozen QualityUncertaintyWeightedAggregator",
    "eligible": "defined finite strictly positive score AND native claim decision SUPPORT",
    "selection": "unique highest eligible score; highest positive ties within absolute 1e-12 abstain",
    "positive_threshold": 0.0,
    "tie_absolute_tolerance": TOL,
    "tie_relative_tolerance": 0.0,
    "modal_ties": "no hard vote; not missing; full vector retained by C/D; other evidence may make option eligible",
    "missing": "any required pair-seed missing/failed/in-flight makes entire question INCOMPLETE",
    "empty_evidence": "ABSTAIN",
    "zero_denominator": "undefined score; ineligible",
    "gold": "evaluator only, joined after immutable blind predictions",
}


def direction(score):
    return "ABSTAIN" if score is None or score == 0 else "SUPPORT" if score > 0 else "CONTRADICT"


def select_answer(option_results, *, complete=True, evidence_count=3):
    if set(option_results) != set(OPTIONS):
        raise ValueError("Answer selection requires all five options")
    if not complete:
        return {"status": "INCOMPLETE", "selected_answer": None, "reason": "MISSING_REQUIRED_SAMPLES", "rankings": []}
    rankings = sorted((label for label in OPTIONS if option_results[label]["score"] is not None),
                      key=lambda label: (-option_results[label]["score"], label))
    base = {"rankings": [{"option": label, "score": option_results[label]["score"],
                           "eligible": option_results[label]["eligible"]} for label in rankings],
            "undefined_options": [label for label in OPTIONS if option_results[label]["score"] is None]}
    eligible = [label for label in rankings if option_results[label]["eligible"]]
    if evidence_count == 0 or not eligible:
        return {**base, "status": "ABSTAIN", "selected_answer": None,
                "reason": "EMPTY_EVIDENCE_POOL" if evidence_count == 0 else "NO_POSITIVE_ELIGIBLE_SCORE"}
    best = option_results[eligible[0]]["score"]
    tied = [label for label in eligible if math.isclose(option_results[label]["score"], best, rel_tol=0, abs_tol=TOL)]
    if len(tied) != 1:
        return {**base, "status": "ABSTAIN", "selected_answer": None, "reason": "TIED_HIGHEST_POSITIVE_SCORES", "tied_options": tied}
    return {**base, "status": "ANSWERED", "selected_answer": tied[0], "reason": "UNIQUE_HIGHEST_POSITIVE_ELIGIBLE_SCORE"}


def aggregate_option(rows, counts, condition):
    if condition not in ("original_quality", "quality_disabled"):
        raise ValueError("Unknown analysis condition")
    if not rows:
        return {m: {"score": None, "native_claim_decision": "ABSTAIN", "eligible": False,
                    "denominator": 0, "reason": "EMPTY_EVIDENCE_POOL"} for m in "ABCD"}, []
    shared, details = [], []
    for row in rows:
        key = (row["question_id"], row["candidate_option_id"], row["evidence_doc_id"])
        ns = counts[key]
        if sum(ns.values()) != 10 or set(ns) != {s.value for s in Stance} or any(type(n) is not int or n < 0 for n in ns.values()):
            raise ValueError("Missing/invalid ten-sample counts, not a modal tie")
        p = {s: ns[s.value] / 10 for s in Stance}
        prediction = StancePrediction(p, "frozen-ten-seed-empirical-frequencies")
        q = row["quality_weight"] if condition == "original_quality" else 1.0
        evidence = RetrievedEvidence("question27-pilot-input-v1", row["question_id"], row["evidence_doc_id"],
            row["retrieval_rank"], row["evidence_title"] + "\n\n" + row["evidence_abstract"], "pubmed",
            EvidenceType(row["evidence_type"]), 0.0, metadata={k: row[k] for k in ("publication_types", "pmid")})
        shared.append(ScoredEvidence(evidence, QualityScore(q, condition), prediction))
        details.append({"identity": "|".join(key), "pmid": row["pmid"], "retrieval_rank": row["retrieval_rank"],
            "counts": ns, "empirical_frequencies": {s.value: p[s] for s in Stance},
            "matched_hard": prediction.label.value if prediction.label else None,
            "modal_tie": prediction.label is None, "soft_stance": p[Stance.SUPPORT] - p[Stance.CONTRADICT],
            "u3": _normalized_entropy(p), "original_quality_weight": row["quality_weight"], "analysis_weight": q})
    shared = tuple(shared)
    claim = CandidateClaim(rows[0]["question_id"], OPTIONS.index(rows[0]["candidate_option_id"]), rows[0]["candidate_option_id"], rows[0]["candidate_option_text"])
    # Frozen aggregators never read the gold-bearing MedicalQuestion fields.
    question = SimpleNamespace(question_id=claim.question_id, question=rows[0]["question_stem"])
    a = MajorityVoteAggregator().aggregate(question, claim, shared)
    b = QualityWeightedVoteAggregator().aggregate(question, claim, shared)
    d = QualityUncertaintyWeightedAggregator().aggregate(question, claim, shared)
    qsum = math.fsum(item.quality.value for item in shared)
    c = None if qsum == 0 else math.fsum((item.quality.value / qsum) * (item.stance.probabilities[Stance.SUPPORT] - item.stance.probabilities[Stance.CONTRADICT]) for item in shared)
    values = {"A": ((a.support_weight - a.contradict_weight) / len(shared), len(shared), a.decision.value),
              "B": (None if qsum == 0 else (b.support_weight - b.contradict_weight) / qsum, qsum, b.decision.value),
              "C": (c, qsum, direction(c)), "D": (d.aggregate_score, d.total_effective_weight, d.decision.value)}
    results = {}
    for method, (score, denominator, decision) in values.items():
        results[method] = {"score": score, "denominator": denominator, "native_claim_decision": decision,
            "eligible": score is not None and math.isfinite(score) and score > 0 and decision == "SUPPORT",
            "within_zero_tolerance": score is not None and abs(score) <= TOL,
            "reason": "ZERO_DENOMINATOR" if score is None else "NATIVE_CLAIM_ABSTENTION" if decision == "ABSTAIN" else "NONPOSITIVE_SCORE" if score <= 0 else "POSITIVE_SUPPORT"}
    return results, details


def evaluate(rows, samples, seeds=tuple(range(101, 111))):
    expected = {((r["question_id"], r["candidate_option_id"], r["evidence_doc_id"]), seed) for r in rows for seed in seeds}
    if set(samples) - expected:
        raise ValueError("Unexpected sample identity")
    missing = sorted(expected - set(samples))
    if missing:
        return {"status": "INCOMPLETE", "missing_sample_count": len(missing),
                "missing_samples": [{"ids": list(key), "seed": seed} for key, seed in missing], "predictions": None}
    counts = {}
    for row in rows:
        key = (row["question_id"], row["candidate_option_id"], row["evidence_doc_id"])
        labels = [samples[key, seed] for seed in seeds]
        if any(label not in {s.value for s in Stance} for label in labels):
            raise ValueError("Invalid sample stance")
        n = Counter(labels)
        counts[key] = {s.value: n[s.value] for s in Stance}
    conditions = {}
    for condition in ("original_quality", "quality_disabled"):
        methods = {m: {} for m in "ABCD"}
        details = {}
        for option in OPTIONS:
            option_rows = [r for r in rows if r["candidate_option_id"] == option]
            results, details[option] = aggregate_option(option_rows, counts, condition)
            for method in methods:
                methods[method][option] = results[method]
        conditions[condition] = {"methods": {m: {"option_scores": values,
                "answer": select_answer(values, evidence_count=len(rows) // 5)} for m, values in methods.items()}, "evidence_diagnostics": details}
    def contrast(left, right):
        comparable = left["score"] is not None and right["score"] is not None
        return {"score_delta": right["score"] - left["score"] if comparable else None,
                "score_changed": comparable and abs(right["score"] - left["score"]) > TOL,
                "undefinedness_changed": (left["score"] is None) != (right["score"] is None),
                "native_claim_decision_changed": left["native_claim_decision"] != right["native_claim_decision"]}
    effects = {}
    for condition, data in conditions.items():
        effects[condition] = {name: {option: contrast(data["methods"][left]["option_scores"][option], data["methods"][right]["option_scores"][option]) for option in OPTIONS} for name, left, right in (("B_vs_C", "B", "C"), ("C_vs_D", "C", "D"))}
        effects[condition]["selected_answer_differences"] = {name: any(data["methods"][left]["answer"][k] != data["methods"][right]["answer"][k] for k in ("status", "selected_answer")) for name, left, right in (("B_vs_C", "B", "C"), ("C_vs_D", "C", "D"))}
        effects[condition]["ranking_differences"] = {name: [r["option"] for r in data["methods"][left]["answer"]["rankings"]] != [r["option"] for r in data["methods"][right]["answer"]["rankings"]] for name, left, right in (("B_vs_C", "B", "C"), ("C_vs_D", "C", "D"))}
    effects["quality_enabled_vs_disabled"] = {m: {option: contrast(conditions["original_quality"]["methods"][m]["option_scores"][option], conditions["quality_disabled"]["methods"][m]["option_scores"][option]) for option in OPTIONS} for m in "ABCD"}
    effects["quality_enabled_vs_disabled_answer_differences"] = {m: any(conditions["original_quality"]["methods"][m]["answer"][k] != conditions["quality_disabled"]["methods"][m]["answer"][k] for k in ("status", "selected_answer")) for m in "ABCD"}
    return {"status": "COMPLETE", "question_id": rows[0]["question_id"] if rows else None, "sample_count": len(samples), "conditions": conditions, "effects": effects, "answer_rules": RULES,
            "interpretation": "One technical question; not evidence of medical QA improvement. Frequencies are not calibrated probabilities."}
