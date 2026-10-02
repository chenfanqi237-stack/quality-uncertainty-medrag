import json
from dataclasses import asdict

import pytest

from quality_uncertainty_medrag import stance_logprob_probe as runner
from quality_uncertainty_medrag.llm_stance import StanceModelConfig, STANCE_PROMPT
from quality_uncertainty_medrag.llm_stance_logprob import text_hash, normalized_entropy, canonical


def selection_data():
    inputs, references = [], []
    for i, (qid, option, pmid) in enumerate(runner.SELECTED_KEYS):
        row = dict(question_id=qid, candidate_option_id=option, evidence_doc_id=pmid,
                   question_stem="Stem only", candidate_option_text="One candidate", evidence_title="Title", evidence_abstract="Abstract")
        old = {**{f: row[f] for f in runner.ID_FIELDS}, "status": "SUCCESS",
               "classifier_version": "llm-medical-stance-v1", "model_metadata": asdict(StanceModelConfig()),
               "p_support": .9, "p_contradict": 0., "p_irrelevant": .1,
               "argmax_label": "SUPPORT", "normalized_entropy": normalized_entropy(dict(SUPPORT=.9, CONTRADICT=0., IRRELEVANT=.1)),
               "input_sha256": {f: text_hash(row[f]) for f in runner.MODEL_INPUT_FIELDS},
               "cache_key": str(i) * 64, "prompt_template_sha256": text_hash(STANCE_PROMPT)}
        inputs.append(row); references.append(old)
    return inputs, references


def test_fixed_selection_exactly_six_dev_only_and_text_identical(tmp_path):
    inputs, old = selection_data()
    source, previous = tmp_path / "inputs.jsonl", tmp_path / "stances.jsonl"
    source.write_text("".join(canonical(r)+"\n" for r in inputs))
    previous.write_text("".join(canonical(r)+"\n" for r in old))
    before = source.read_bytes(), previous.read_bytes()
    out = tmp_path / "selection"
    meta = runner.prepare_selection(source, previous, out)
    selected = runner.load_jsonl(out / "inputs.jsonl")
    references = json.loads((out / "v1_reference.json").read_text())
    runner.validate_selection(selected, references)
    assert selected == inputs
    assert meta["judgments"] == 6 and meta["dev_31_50_used"] is False
    assert all(1 <= int(r["question_id"].rsplit("-", 1)[1]) <= 30 for r in selected)
    assert before == (source.read_bytes(), previous.read_bytes())


@pytest.mark.parametrize("field", ["answer", "answer_idx", "options", "gold_label", "metadata"])
def test_selection_rejects_extra_fields(field):
    rows, refs = selection_data(); rows[0][field] = "FORBIDDEN"
    with pytest.raises(ValueError, match="Unexpected"):
        runner.validate_selection(rows, refs)


def test_selection_rejects_final_validation_and_other_keys():
    rows, refs = selection_data(); rows[0]["question_id"] = "medqa-us-dev-000031"
    with pytest.raises(ValueError, match="six fixed"):
        runner.validate_selection(rows, refs)


def test_summary_failure_denominators_and_likelihood_semantics():
    base = dict(cache_status="MISS", generation_attempts=1)
    good = {**base, "status": "SUCCESS", "normalized_label_likelihoods": dict(SUPPORT=.5, CONTRADICT=.25, IRRELEVANT=.25),
            "normalized_entropy": normalized_entropy(dict(SUPPORT=.5, CONTRADICT=.25, IRRELEVANT=.25)),
            "argmax_label": "SUPPORT", "all_label_tokens_present": True, "hard_argmax_agrees": True}
    results = [good, {**base,"status":"LOGPROB_INCOMPLETE"}, {**base,"status":"FINAL_TOKEN_ALIGNMENT_FAILURE"}]
    summary = runner.summarize(results)
    assert summary["actual_model_calls"] == 3 and summary["successful_logprob_results"] == 1
    assert summary["LOGPROB_INCOMPLETE"] == summary["final_token_alignment_failures"] == 1
    assert summary["exact_one_hot_count"] == 0
    assert summary["argmax_counts"]["SUPPORT"] == 1
    assert summary["technical_feasibility"] == "NOT_DEMONSTRATED_FOR_ALL_SIX"
    assert summary["values_are_calibrated_probabilities"] is False


def test_existing_v1_prompt_sha_stays_frozen():
    assert text_hash(STANCE_PROMPT) == "c9db305b4c2a7ec23700851c332f8109c9d2e3b62efd83e9eb1c50520670886a"
